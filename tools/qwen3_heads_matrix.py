#!/usr/bin/env python3
"""Plan the standalone fixed-head matrix for the Qwen3 three-seed campaign.

The head stage is separate from training: for every eligible training
checkpoint the matrix extracts hidden features from the fold's ``best_model``
and fits the fixed classifiers (Logistic Regression and XGBoost, no Optuna).
This tool builds that job graph and fails closed on anything that would make a
head number untraceable:

* a parent checkpoint is resolved either from an explicit parent map (one
  attempt/checkpoint per ``(route, parent_training_seed, fold)`` key) or by
  scanning candidate run roots for fold directories whose recorded resolved
  config matches the current cell config under the same reduction the readiness
  inventory uses, and whose recorded training seed equals the requested seed.
  Run names are not evidence; two *eligible* matching runs are an ambiguity and
  are refused;
* the inventory's narrowly justified declaration-only ``resources.train_nodes``
  rule is reused: a historical run that recorded the same world size and
  accumulation as the current config is accepted even though the older
  ``run_config.yaml`` predates the declaration. Any other resource or
  scientific difference is refused;
* a candidate parent must also carry successful attempt evidence: a lifecycle
  state in ``COMPLETED_ON_MN5`` or later and clean terminal job events. FAILED,
  CANCELLED, SUPERSEDED and REJECTED attempts are never selected; they are
  recorded as excluded attempts. A failed-only key is reported as
  ``blocked_failed_parent`` instead of ``waiting_for_checkpoint``;
* the extraction GPU shape is read from the parent's recorded
  ``resources.eval_gpus_per_node`` (1 for Qwen3.8 text, 4 for the Qwen3-Omni
  sharded route) and never hardcoded;
* the parent's recorded prompt hash and translation-notice version must match
  the cell config, so an English cell can never be bound to a native checkpoint
  and a native cell to an English one;
* the adapter files must exist locally (or on the planning host) and are hashed
  into the plan;
* a cell/seed/fold without a matching parent is recorded as
  ``waiting_for_checkpoint`` instead of being planned silently;
* features and head fits are written under an explicit lane-owned
  ``--cache-root`` when one is given, so reused parent run roots are never
  written into.

``--check`` re-validates a plan; ``--emit`` writes the machine-readable job
manifest. Nothing is submitted by this tool.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import sha256_file  # noqa: E402
from tools.qwen3_multiseed_inventory import (  # noqa: E402
    diff_paths,
    is_shape_path,
    prompt_sha256,
    reduce_config,
    split_fingerprint,
)
from tools.qwen3_multiseed_plan import (  # noqa: E402
    PLANNED_SEEDS,
    SPLIT_SEED,
    build_selection_map,
)

SCHEMA_VERSION = "audiollm.qwen3_heads_matrix.v1"
PARENT_MAP_SCHEMA = "audiollm.qwen3_heads_parent_map.v1"
HEAD_VARIANTS = ("logreg_raw", "xgb_raw")
HEAD_SEED = 1337
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SUCCESS_STATES = frozenset(
    {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"}
)
NEGATIVE_STATES = frozenset({"FAILED", "CANCELLED", "SUPERSEDED", "REJECTED"})
TERMINAL_FAILURES = frozenset({"FAILED", "CANCELLED", "TIMEOUT"})
DECLARATION_ONLY_SHAPE_DIFFS = ("resources.train_nodes",)
PARENT_STATUSES = ("resolved", "waiting_for_checkpoint", "blocked_failed_parent")


class HeadsMatrixError(RuntimeError):
    """Raised when the head matrix cannot be planned at all."""


def _fold_dirs(run_root: Path) -> list[Path]:
    """Fold directories under one cell's run root (``<run_root>/<run>/fold_<n>``)."""

    if not run_root.is_dir():
        return []
    return sorted(path for path in run_root.glob("*/fold_*") if path.is_dir())


def _recorded_config(fold_dir: Path) -> dict[str, Any] | None:
    path = fold_dir / "run_config.yaml"
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def attempt_eligibility(fold_dir: Path) -> dict[str, Any]:
    """Successful-attempt evidence for one candidate parent.

    The authoritative success evidence is the terminal job history: at least
    one COMPLETED ``train`` event and no FAILED/CANCELLED/TIMEOUT events. The
    lifecycle state must not be an explicit negative (FAILED, CANCELLED,
    SUPERSEDED, REJECTED); a missing status sidecar is not success. A state
    that is still RUNNING/SUBMITTED while the recorded jobs are clean is a
    lagging lifecycle pointer and is accepted with ``state_lagging=True``
    recorded, because the remote sidecars of historical campaigns were not
    advanced after local collection.
    """

    status = _load_json(fold_dir / "status.json") or {}
    state = status.get("state")
    events: list[dict[str, Any]] = []
    jobs_path = fold_dir / "jobs.jsonl"
    if jobs_path.is_file():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    completed = [event for event in events if event.get("event_type") == "COMPLETED"]
    failures = [
        event for event in events if event.get("event_type") in TERMINAL_FAILURES
    ]
    train_keys = {str(event.get("job_key")) for event in events if event.get("job_key")}
    train_completed = any(
        event.get("job_key") == "train" and event.get("event_type") == "COMPLETED"
        for event in events
    )
    reasons: list[str] = []
    if state in NEGATIVE_STATES:
        reasons.append(f"state={state}")
    elif not state:
        reasons.append("no status sidecar")
    if not completed:
        reasons.append("no COMPLETED job event")
    elif "train" in train_keys and not train_completed:
        reasons.append("no COMPLETED train event")
    if failures:
        reasons.append(
            "failed terminal events: "
            + ",".join(sorted({str(event.get("event_type")) for event in failures}))
        )
    metadata = _load_json(fold_dir / "metadata.json") or {}
    return {
        "state": state,
        "eligible": not reasons,
        "state_lagging": bool(not reasons and state not in SUCCESS_STATES),
        "completed_events": len(completed),
        "failed_events": len(failures),
        "reasons": reasons,
        "supersedes_attempt_id": metadata.get("supersedes_attempt_id"),
        "metadata_attempt_id": metadata.get("attempt_id"),
    }


def _declaration_only_shape_difference(
    shape_differences: list[str],
    recorded_config: dict[str, Any],
    payload: dict[str, Any],
    cell_config: dict[str, Any],
) -> bool:
    """The inventory's declaration-only ``resources.train_nodes`` rule.

    Older runs could not record ``resources.train_nodes`` because the lane
    shape lived in the submission. Accept exactly that case when the recorded
    world size and accumulation agree with the current config; anything else is
    a real resource difference.
    """

    if shape_differences != list(DECLARATION_ONLY_SHAPE_DIFFS):
        return False
    recorded_world = int((payload.get("training_strategy") or {}).get("world_size") or 0)
    declared_world = int((cell_config.get("resources") or {}).get("train_nodes", 1) or 1) * 4
    recorded_accum = int(
        (recorded_config.get("training") or {}).get("gradient_accumulation_steps") or 0
    )
    declared_accum = int(
        (cell_config.get("training") or {}).get("gradient_accumulation_steps") or 0
    )
    return bool(
        recorded_world
        and declared_world
        and recorded_world == declared_world
        and recorded_accum
        and recorded_accum == declared_accum
    )


def _evaluate_fold_as_parent(
    fold_dir: Path,
    *,
    cell_config: dict[str, Any],
    seed: int,
    notice_version: str | None,
) -> dict[str, Any]:
    """Validate one fold directory as a parent candidate without raising.

    Returns ``{"ok": bool, "reason": str | None, "record": dict | None}`` so the
    automatic path can skip mismatches while the explicit-map path can fail
    closed with the exact reason.
    """

    payload = _recorded_config(fold_dir)
    if payload is None:
        return {"ok": False, "reason": "run_config.yaml missing or unreadable", "record": None}
    recorded = payload.get("config")
    if not isinstance(recorded, dict):
        return {"ok": False, "reason": "run_config.yaml has no resolved config block", "record": None}
    # The requested training seed is the only approved top-level difference
    # between the cell config and a recorded attempt of another seed. Every
    # other scientific field must match; the recorded seed is validated
    # separately below and the split seed stays fixed at SPLIT_SEED.
    expected = copy.deepcopy(cell_config)
    expected["seed"] = int(seed)
    differences = diff_paths(reduce_config(expected) or {}, reduce_config(recorded) or {})
    scientific = [path for path in differences if not is_shape_path(path)]
    if scientific:
        return {
            "ok": False,
            "classification": "not_candidate",
            "reason": "recorded config differs from the cell config: " + ", ".join(scientific),
            "record": None,
        }
    shape = [path for path in differences if is_shape_path(path)]
    declaration_only = False
    if shape:
        if not _declaration_only_shape_difference(shape, recorded, payload, cell_config):
            return {
                "ok": False,
                "classification": "not_candidate",
                "reason": "recorded resource shape differs from the cell config: "
                + ", ".join(shape),
                "record": None,
            }
        declaration_only = True
    recorded_seed = int(recorded.get("seed", -1))
    if recorded_seed != int(seed):
        return {
            "ok": False,
            "classification": "not_candidate",
            "reason": f"recorded training seed {recorded_seed} != requested {seed}",
            "record": None,
        }
    recorded_prompt = payload.get("prompt_context") or {}
    prompt_recorded = recorded_prompt.get("system_prompt_sha256")
    prompt_current = prompt_sha256(cell_config)
    if prompt_recorded != prompt_current:
        return {
            "ok": False,
            "classification": "fatal",
            "fatal": True,
            "reason": "recorded prompt hash differs from the cell config",
            "record": None,
        }
    notice_recorded = recorded_prompt.get("translation_notice_version")
    if str(notice_recorded or "") != str(notice_version or ""):
        return {
            "ok": False,
            "classification": "fatal",
            "fatal": True,
            "reason": (
                "recorded translation notice version "
                f"{notice_recorded!r} does not match the cell's {notice_version!r}"
            ),
            "record": None,
        }
    split_seed = int((recorded.get("split") or {}).get("seed", -1))
    if split_seed != SPLIT_SEED:
        return {
            "ok": False,
            "classification": "fatal",
            "fatal": True,
            "reason": f"recorded split seed {split_seed} != {SPLIT_SEED}",
            "record": None,
        }
    checkpoint = fold_dir / "best_model"
    missing = [name for name in ADAPTER_FILES if not (checkpoint / name).is_file()]
    if missing:
        return {
            "ok": False,
            "classification": "not_candidate",
            "reason": f"{checkpoint} is missing {missing}",
            "record": None,
        }
    eligibility = attempt_eligibility(fold_dir)
    if not eligibility["eligible"]:
        state = eligibility["state"]
        if state in NEGATIVE_STATES:
            classification = "terminal_failed"
        elif state in {"PLANNED", "DEPLOYED", "SUBMITTED", "RUNNING"}:
            # A live attempt without completed evidence yet: wait, do not block.
            classification = "nonterminal"
        else:
            # Missing sidecar or contradictory evidence: fail closed.
            classification = "invalid"
        return {
            "ok": False,
            "classification": classification,
            "reason": "attempt is not eligible: " + "; ".join(eligibility["reasons"]),
            "record": None,
        }
    metadata = _load_json(fold_dir / "metadata.json") or {}
    attempt_id = (payload.get("tracking") or {}).get("attempt_id") or metadata.get("attempt_id")
    if not attempt_id:
        return {
            "ok": False,
            "classification": "invalid",
            "reason": "recorded attempt id is missing",
            "record": None,
        }
    if metadata.get("attempt_id") and str(metadata.get("attempt_id")) != str(attempt_id):
        return {
            "ok": False,
            "classification": "invalid",
            "reason": "run_config tracking attempt id disagrees with metadata.json",
            "record": None,
        }
    return {
        "ok": True,
        "reason": None,
        "record": {
            "status": "resolved",
            "fold_dir": str(fold_dir),
            "run_name": fold_dir.parent.name,
            "attempt_id": str(attempt_id),
            "parent_training_seed": int(seed),
            "head_seed": HEAD_SEED,
            "checkpoint_dir": str(checkpoint),
            "checkpoint_adapter_config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "checkpoint_adapter_model_sha256": sha256_file(
                checkpoint / "adapter_model.safetensors"
            ),
            "model_backend": recorded.get("model_backend"),
            "model_revision": recorded.get("model_revision"),
            "prompt_sha256": prompt_recorded,
            "translation_notice_version": notice_recorded,
            "split_seed": split_seed,
            "split_fingerprint": split_fingerprint(fold_dir / "logs" / "split_used.json"),
            "manifest_hash_recorded": payload.get("manifest_hash"),
            "world_size_recorded": (payload.get("training_strategy") or {}).get("world_size"),
            "declaration_only_shape_difference": declaration_only,
            "state": eligibility["state"],
            "state_lagging": eligibility["state_lagging"],
            "jobs_clean": bool(
                eligibility["completed_events"] and not eligibility["failed_events"]
            ),
            "supersedes_attempt_id": eligibility["supersedes_attempt_id"],
            "extract_gpus": int(
                (recorded.get("resources") or {}).get("eval_gpus_per_node", 1) or 1
            ),
            "seed": int(seed),
            "fold": int(fold_dir.name.split("_", 1)[1]),
        },
    }


def _candidate_run_roots(
    cell_config: dict[str, Any],
    route: dict[str, Any],
    scan_roots: list[Path],
    campaign_root: Path | None,
) -> list[Path]:
    """Candidate run roots for one route.

    ``--scan-root`` only prefixes the config-declared relative root, so it can
    find historical campaign outputs but never a new managed campaign root. An
    explicit ``--campaign-root`` adds ``<campaign_root>/<modality>/<dataset>``
    for the route's own modality and dataset.
    """

    run_root_template = str((cell_config.get("output_dirs") or {}).get("run_root") or "")
    relative = run_root_template.split("${PROJECT_ROOT}/", 1)[-1].lstrip("/")
    roots: list[Path] = []
    if relative:
        roots.append(PROJECT_ROOT / relative)
        roots.extend(scan_root / relative for scan_root in scan_roots)
    if campaign_root is not None:
        modality = route.get("modality")
        dataset = route.get("dataset")
        if modality and dataset:
            roots.append(Path(campaign_root) / str(modality) / str(dataset))
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _cache_dir(
    parent: dict[str, Any], route: dict[str, Any], cache_root: Path | None
) -> Path:
    """Feature cache directory for one head job.

    With an explicit lane-owned ``--cache-root`` the cache never touches the
    reused parent's run root: ``<cache_root>/<dataset>/<modality>/<run>_fold_N``.
    """

    run_name = str(parent["run_name"])
    fold = int(parent["fold"])
    seed = int(parent["parent_training_seed"])
    if cache_root is not None:
        return (
            Path(cache_root)
            / str(route["dataset"])
            / str(route["modality"])
            / f"{run_name}_fold_{fold}_pseed{seed}_hseed{HEAD_SEED}"
        )
    return (
        Path(parent["checkpoint_dir"]).parents[2]
        / "hidden_features"
        / str(route["dataset"])
        / f"{run_name}_fold_{fold}_seed{seed}"
    )


def resolve_parent(
    *,
    cell: dict[str, Any],
    cell_config: dict[str, Any],
    run_root: Path,
    fold: int,
    seed: int,
    notice_version: str | None,
    explicit: dict[str, Any] | None = None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Find the one eligible training attempt for a cell/seed/fold key."""

    if explicit is not None:
        mapped_dir = Path(str(explicit["fold_dir"]))
        result = _evaluate_fold_as_parent(
            mapped_dir, cell_config=cell_config, seed=seed, notice_version=notice_version
        )
        if not result["ok"]:
            raise HeadsMatrixError(
                f"explicit parent for {cell['config']} seed {seed} fold {fold} failed "
                f"validation: {result['reason']}"
            )
        parent = dict(result["record"])
        mapped_attempt = explicit.get("attempt_id")
        if mapped_attempt and str(mapped_attempt) != parent["attempt_id"]:
            raise HeadsMatrixError(
                f"explicit parent attempt id {mapped_attempt!r} does not match the recorded "
                f"attempt id {parent['attempt_id']!r} for {cell['config']} seed {seed} fold {fold}"
            )
        map_state = explicit.get("state")
        if map_state and str(map_state) not in SUCCESS_STATES:
            raise HeadsMatrixError(
                f"explicit parent map records non-success state {map_state!r} for "
                f"{cell['config']} seed {seed} fold {fold}"
            )
        parent["selection"] = "explicit_parent_map"
        parent["selection_reason"] = str(explicit.get("selection_reason") or "explicit parent map")
        parent["excluded_attempts"] = list(explicit.get("excluded_duplicates") or [])
        return parent

    matches: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for fold_dir in _fold_dirs(run_root):
        if fold_dir.name != f"fold_{fold}":
            continue
        result = _evaluate_fold_as_parent(
            fold_dir, cell_config=cell_config, seed=seed, notice_version=notice_version
        )
        if result["ok"]:
            matches.append(result["record"])
            continue
        if result.get("fatal"):
            raise HeadsMatrixError(
                f"parent candidate {fold_dir.parent.name} fold {fold} contradicts the cell "
                f"config: {result['reason']}"
            )
        payload = _recorded_config(fold_dir) or {}
        excluded.append(
            {
                "run_name": fold_dir.parent.name,
                "fold_dir": str(fold_dir),
                "attempt_id": (payload.get("tracking") or {}).get("attempt_id"),
                "state": (attempt_eligibility(fold_dir) or {}).get("state"),
                "reason": result["reason"],
                "classification": result.get("classification", "invalid"),
            }
        )
    if not matches:
        blocked = [
            item
            for item in excluded
            if item.get("classification") in ("terminal_failed", "invalid")
        ]
        if blocked:
            return {
                "status": "blocked_failed_parent",
                "reason": (
                    f"no eligible {seed}-seed parent for {cell['config']} fold {fold}; "
                    "failed or invalid candidates require an explicit decision"
                ),
                "excluded_attempts": excluded,
            }
        return {
            "status": "waiting_for_checkpoint",
            "reason": f"no eligible {seed}-seed run for {cell['config']} fold {fold} under {run_root}",
            "excluded_attempts": excluded,
        }
    # A retry recorded through metadata.supersedes_attempt_id replaces the
    # superseded attempt even when the older attempt's lifecycle state is stale.
    superseded_by = {
        str(match["supersedes_attempt_id"]): match["attempt_id"]
        for match in matches
        if match.get("supersedes_attempt_id")
    }
    if superseded_by:
        kept: list[dict[str, Any]] = []
        for match in matches:
            superseder = superseded_by.get(match["attempt_id"])
            if superseder:
                excluded.append(
                    {
                        "run_name": match["run_name"],
                        "fold_dir": match["fold_dir"],
                        "attempt_id": match["attempt_id"],
                        "state": match["state"],
                        "reason": f"superseded by attempt {superseder}",
                        "classification": "terminal_failed",
                    }
                )
            else:
                kept.append(match)
        matches = kept
    if not matches:
        return {
            "status": "blocked_failed_parent",
            "reason": f"every matching parent for {cell['config']} fold {fold} is superseded",
            "excluded_attempts": excluded,
        }
    if len(matches) > 1:
        raise HeadsMatrixError(
            f"ambiguous parent for {cell['config']} seed {seed} fold {fold}: "
            f"{[match['run_name'] for match in matches]}"
        )
    parent = matches[0]
    parent["selection"] = "unique_automatic"
    parent["selection_reason"] = "exactly one eligible matching parent"
    parent["excluded_attempts"] = excluded
    return parent


def _load_parent_map(
    parent_map_path: Path | None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    if parent_map_path is None:
        return {}
    payload = _load_json(Path(parent_map_path))
    if payload is None:
        raise HeadsMatrixError(f"parent map is not readable JSON: {parent_map_path}")
    if payload.get("schema_version") != PARENT_MAP_SCHEMA:
        raise HeadsMatrixError(
            f"parent map schema must be {PARENT_MAP_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            raise HeadsMatrixError("parent map entries must be objects")
        for field in ("route_id", "config", "fold_dir"):
            if not entry.get(field):
                raise HeadsMatrixError(f"parent map entry is missing {field}: {entry}")
        key = (str(entry["route_id"]), int(entry["parent_training_seed"]), int(entry["fold"]))
        if key in entries:
            raise HeadsMatrixError(f"duplicate parent map key: {key}")
        entries[key] = entry
    return entries


def build_matrix(
    *,
    seeds: list[int],
    scan_roots: list[Path],
    campaign_root: Path | None = None,
    parent_map_path: Path | None = None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    selection = build_selection_map()
    routes = selection["routes"]
    parent_map = _load_parent_map(parent_map_path)
    matrix: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "planned_seeds": seeds,
        "split_seed": SPLIT_SEED,
        "head_seed": HEAD_SEED,
        "head_variants": list(HEAD_VARIANTS),
        "scan_roots": [str(root) for root in scan_roots],
        "campaign_root": str(campaign_root) if campaign_root is not None else None,
        "cache_root": str(cache_root) if cache_root is not None else None,
        "parent_map": str(parent_map_path) if parent_map_path is not None else None,
        "routes": [],
    }
    requested_keys: set[tuple[str, int, int]] = set()
    for route in routes:
        for seed in seeds:
            for fold in route["folds"]:
                requested_keys.add((str(route["route_id"]), int(seed), int(fold)))
    unknown = sorted(set(parent_map) - requested_keys)
    if unknown:
        raise HeadsMatrixError(f"parent map keys are outside the requested matrix: {unknown}")
    for route in routes:
        cell_config = yaml.safe_load(
            (PROJECT_ROOT / route["config"]).read_text(encoding="utf-8")
        )
        candidate_roots = _candidate_run_roots(cell_config, route, scan_roots, campaign_root)
        notice_version = (cell_config.get("prompt") or {}).get("translation_notice_version")
        route_record: dict[str, Any] = {
            "route_id": route["route_id"],
            "config": route["config"],
            "dataset": route["dataset"],
            "modality": route["modality"],
            "language": route["language"],
            "backend": route["model_backend"],
            "translation_notice_version": notice_version,
            "candidate_run_roots": [str(root) for root in candidate_roots],
            "jobs": [],
        }
        for seed in seeds:
            for fold in route["folds"]:
                explicit = parent_map.get((str(route["route_id"]), int(seed), int(fold)))
                if explicit is not None and str(explicit["config"]) != str(route["config"]):
                    raise HeadsMatrixError(
                        f"parent map entry for {route['route_id']} seed {seed} fold {fold} "
                        f"declares config {explicit['config']!r}, expected {route['config']!r}"
                    )
                if explicit is not None:
                    parent = resolve_parent(
                        cell=route,
                        cell_config=cell_config,
                        run_root=Path(str(explicit["fold_dir"])).parent,
                        fold=fold,
                        seed=seed,
                        notice_version=notice_version,
                        explicit=explicit,
                        cache_root=cache_root,
                    )
                else:
                    parent = {"status": "waiting_for_checkpoint", "reason": "no candidate run root holds this cell"}
                    for run_root in candidate_roots:
                        attempt = resolve_parent(
                            cell=route,
                            cell_config=cell_config,
                            run_root=run_root,
                            fold=fold,
                            seed=seed,
                            notice_version=notice_version,
                            cache_root=cache_root,
                        )
                        if attempt["status"] == "resolved":
                            parent = attempt
                            break
                        if attempt["status"] == "blocked_failed_parent":
                            parent = attempt
                job: dict[str, Any] = {
                    "route_id": route["route_id"],
                    "config": route["config"],
                    "seed": seed,
                    "fold": fold,
                    "parent_status": parent["status"],
                }
                if parent["status"] == "resolved":
                    cache_dir = _cache_dir(parent, route, cache_root)
                    job.update(
                        {
                            "parent": parent,
                            "extract": {
                                "job_kind": "hidden_extract",
                                "gpus": parent["extract_gpus"],
                                "checkpoint_dir": parent["checkpoint_dir"],
                                "cache_dir": str(cache_dir),
                                "depends_on": None,
                            },
                            "heads": {
                                "job_kind": "hidden_classifier",
                                "gpus": 0,
                                "variants": list(HEAD_VARIANTS),
                                "seed": HEAD_SEED,
                                "cache_dir": str(cache_dir),
                                "depends_on": "extract",
                            },
                        }
                    )
                else:
                    job["reason"] = parent.get("reason")
                    if parent.get("excluded_attempts"):
                        job["excluded_attempts"] = parent["excluded_attempts"]
                route_record["jobs"].append(job)
        matrix["routes"].append(route_record)
    matrix["summary"] = {
        "routes": len(matrix["routes"]),
        "jobs": sum(len(route["jobs"]) for route in matrix["routes"]),
        "resolved": sum(
            1
            for route in matrix["routes"]
            for job in route["jobs"]
            if job["parent_status"] == "resolved"
        ),
        "waiting_for_checkpoint": sum(
            1
            for route in matrix["routes"]
            for job in route["jobs"]
            if job["parent_status"] == "waiting_for_checkpoint"
        ),
        "blocked_failed_parent": sum(
            1
            for route in matrix["routes"]
            for job in route["jobs"]
            if job["parent_status"] == "blocked_failed_parent"
        ),
    }
    return matrix


def check_matrix(matrix: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if matrix.get("schema_version") != SCHEMA_VERSION:
        failures.append("unexpected schema version")
    if tuple(matrix.get("head_variants") or ()) != HEAD_VARIANTS:
        failures.append("the fixed head variants changed")
    if int(matrix.get("head_seed", -1)) != HEAD_SEED:
        failures.append("the fixed classifier seed changed")
    if len(matrix.get("routes") or []) != 23:
        failures.append("the head matrix must cover the 23 standalone routes")
    for route in matrix.get("routes") or []:
        english = route["language"] == "english"
        for job in route["jobs"]:
            status = job.get("parent_status")
            if status not in PARENT_STATUSES:
                failures.append(f"{route['route_id']}: unknown parent status {status!r}")
                continue
            if status == "blocked_failed_parent":
                failures.append(
                    f"{route['route_id']} seed {job['seed']} fold {job['fold']}: "
                    "failed/superseded parents require an explicit decision"
                )
                continue
            if status != "resolved":
                continue
            parent = job["parent"]
            if not parent.get("attempt_id"):
                failures.append(f"{route['route_id']}: resolved parent without an attempt id")
            if int(parent.get("parent_training_seed", -1)) != int(job["seed"]):
                failures.append(
                    f"{route['route_id']}: parent training seed "
                    f"{parent.get('parent_training_seed')} != job seed {job['seed']}"
                )
            if int(parent.get("head_seed", -1)) != HEAD_SEED:
                failures.append(f"{route['route_id']}: resolved parent without head seed {HEAD_SEED}")
            if parent.get("selection") not in {"unique_automatic", "explicit_parent_map"}:
                failures.append(f"{route['route_id']}: resolved parent without a selection mode")
            if not parent.get("prompt_sha256"):
                failures.append(f"{route['route_id']}: resolved parent without a prompt hash")
            recorded_notice = parent.get("translation_notice_version")
            if english and not recorded_notice:
                failures.append(
                    f"{route['route_id']}: English route bound to a checkpoint without a notice version"
                )
            if not english and recorded_notice:
                failures.append(
                    f"{route['route_id']}: native route bound to a checkpoint with a notice version"
                )
            expected_gpus = 4 if route["backend"] == "qwen3omni" else 1
            if int(job["extract"]["gpus"]) != expected_gpus:
                failures.append(
                    f"{route['route_id']}: extraction shape {job['extract']['gpus']} does not match "
                    f"the recorded evaluation shape {expected_gpus}"
                )
            if job["heads"]["depends_on"] != "extract":
                failures.append(f"{route['route_id']}: head job without the extract dependency")
            if int(job["heads"].get("seed", -1)) != HEAD_SEED:
                failures.append(f"{route['route_id']}: head job without classifier seed {HEAD_SEED}")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        default=[],
        help="checkout whose output_model may hold the cell run roots (repeatable)",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help="managed campaign output root; adds <root>/<modality>/<dataset> per route",
    )
    parser.add_argument(
        "--parent-map",
        type=Path,
        default=None,
        help="explicit per-key parent attempt/checkpoint mapping JSON",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="lane-owned feature cache root for new head jobs",
    )
    parser.add_argument("--emit", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan-file", type=Path, default=None, help="re-check this plan instead")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan_file is not None:
        matrix = json.loads(args.plan_file.read_text(encoding="utf-8"))
    else:
        seeds = args.seed or list(PLANNED_SEEDS)
        try:
            matrix = build_matrix(
                seeds=seeds,
                scan_roots=list(args.scan_root),
                campaign_root=args.campaign_root,
                parent_map_path=args.parent_map,
                cache_root=args.cache_root,
            )
        except (HeadsMatrixError, KeyError, TypeError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        if args.emit is not None:
            args.emit.parent.mkdir(parents=True, exist_ok=True)
            args.emit.write_text(
                json.dumps(matrix, indent=2, sort_keys=False) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.emit}")
    failures = check_matrix(matrix)
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    print(
        "head matrix: "
        f"{matrix['summary']['routes']} routes, {matrix['summary']['jobs']} cell/seed/fold jobs, "
        f"{matrix['summary']['resolved']} resolved, "
        f"{matrix['summary']['waiting_for_checkpoint']} waiting for checkpoint, "
        f"{matrix['summary']['blocked_failed_parent']} blocked on failed parents"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
