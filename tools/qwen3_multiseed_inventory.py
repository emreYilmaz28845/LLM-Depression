#!/usr/bin/env python3
"""Classify the existing seed-1337 native Qwen3 standalone evidence.

Phase A of the Qwen3 multiseed matrix-readiness plan needs one decision per
matrix cell and fold: is the existing seed-1337 run reusable as the seed-1337
arm of a three-seed campaign, incomplete, incompatible, or missing? Run names
are not evidence. This tool decides from the recorded provenance:

* the resolved ``config`` block inside ``run_config.yaml`` reduced to its
  scientific content (path-bearing keys are removed at every level, so a
  different deployment or dataset path is not mistaken for a different
  cohort);
* the recorded prompt hash against the current ``resolve_system_prompt``;
* the training seed and the fixed split seed;
* the evaluation contract (likelihood decision rule and evaluation view);
* the model identity (backend, base-model basename, revision);
* the modern lifecycle sidecars (``status.json``, ``jobs.jsonl``,
  ``artifacts.json``, ``evaluations.json``) through their real event schema
  (``event_type`` SUBMITTED/STARTED/COMPLETED/FAILED/CANCELLED);
* the ``best_model`` directory and whether the adapter weights are present
  locally;
* the local compact evidence, including a strict-metric recomputation from the
  stored subject-level predictions that must reproduce the stored headline.

Verdicts, strongest first: ``reusable`` (every check passes), ``reusable_pending_shape_decision``
(the recorded science matches; only parallel-shape fields differ, for example a
2-node accumulation-16 submission of the same effective global batch 128),
``incomplete`` (the cell matches but lifecycle or local evidence is missing),
``incompatible`` (the recorded science differs from the current config),
``missing`` (no run at all). A run that carries an exclusion marker
(label-vocabulary, geriatri four-source) is recorded as an excluded candidate
and never becomes a reuse decision.

Two limits are recorded instead of hidden. Local manifest and split file
hashes are not comparable with a run recorded under another build root, because
manifest rows embed absolute paths; the tool therefore records the run's
recorded hashes and its ``logs/split_used.json`` membership fingerprint, and
compares the declared split contract through the reduced config. Adapter
weights that were deliberately not synced stay an evidence gap, not a verdict
downgrade: they are fetched separately when a stage needs them.

The tool is read-only. It writes one JSON inventory and prints a summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prompt_context import resolve_system_prompt  # noqa: E402
from src.experiment_tracking import discovery  # noqa: E402
from src.experiment_tracking.canonical import canonical_sha256, sha256_file  # noqa: E402
from src.experiment_tracking.validate import (  # noqa: E402
    ValidationError,
    recompute_strict_headline,
)
from src.utils import (  # noqa: E402
    load_yaml_with_overrides,
    resolve_input_modality,
    resolve_model_backend,
)

SCHEMA_VERSION = "audiollm.qwen3_multiseed_inventory.v1"
QWEN3_BACKENDS = frozenset({"qwen38", "qwen3omni"})
EXPECTED_SEED = 1337
EXPECTED_PREDICTION_MODE = "likelihood"
EXPECTED_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
REUSABLE_STATES = frozenset({"LOCALLY_VALIDATED", "REPORTABLE"})
TERMINAL_EVENT_TYPES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
FAILED_EVENT_TYPES = frozenset({"FAILED", "CANCELLED", "TIMEOUT"})
DEFAULT_MATRIX = PROJECT_ROOT / "configs/experiments/harmonized/standalone_matrix.yaml"
DEFAULT_EXCLUSION_MARKERS = ("four_source", "q3dlv", "label_vocab")
_FOLD_DIR_PATTERN = re.compile(r"^fold_[0-9]+$")

# Keys removed before comparing recorded and current configs. They carry local
# or deployment-specific paths, and a path difference is not a cohort change.
_PATH_KEYS = frozenset(
    {
        "output_dirs",
        "quarantine_path",
        "dataset_root",
        "label_root",
        "metadata_csv",
        "transcript_file",
        "full_transcript_path",
        "segment_transcript_path",
        "manifest_path",
        "metadata_path",
        "model_name_or_path",
    }
)
_PATH_KEY_SUFFIXES = ("_path", "_dir", "_root", "_file", "_csv")

# Parallel-shape fields. A difference here changes how the same recipe was
# distributed across ranks, not what was trained; it is surfaced for an
# explicit decision instead of being silently treated as compatible.
SHAPE_DIFF_PATHS = frozenset(
    {
        "training.gradient_accumulation_steps",
        "training.per_device_train_batch_size",
        "training.per_device_eval_batch_size",
        "training.strategy",
        "training.activation_offload",
    }
)

VERDICT_RANK = {
    "reusable": 4,
    "reusable_pending_shape_decision": 3,
    "incomplete": 2,
    "incompatible": 1,
    "missing": 0,
}


class InventoryError(RuntimeError):
    """Raised when the inventory cannot be built at all."""


def reduce_config(value: Any, *, key: str | None = None) -> Any:
    """Return the scientific content of a resolved config."""
    if key is not None and (key in _PATH_KEYS or key.endswith(_PATH_KEY_SUFFIXES)):
        return None
    if isinstance(value, dict):
        reduced: dict[str, Any] = {}
        for child_key, child in value.items():
            item = reduce_config(child, key=str(child_key))
            if item is not None:
                reduced[str(child_key)] = item
        return reduced
    if isinstance(value, list):
        return [reduce_config(item) for item in value]
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for child_key, child in value.items():
            path = f"{prefix}.{child_key}" if prefix else str(child_key)
            flat.update(flatten(child, path))
        return flat
    return {prefix: value}


def diff_paths(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    left_flat = flatten(left)
    right_flat = flatten(right)
    return sorted(
        path
        for path in set(left_flat) | set(right_flat)
        if left_flat.get(path, "<missing>") != right_flat.get(path, "<missing>")
    )


def is_shape_path(path: str) -> bool:
    return path in SHAPE_DIFF_PATHS or path.startswith("resources.")


def prompt_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(resolve_system_prompt(config).encode("utf-8")).hexdigest()


def split_fingerprint(split_used_path: Path) -> dict[str, Any] | None:
    """Deterministic subject-membership fingerprint of the split actually used."""
    if not split_used_path.is_file():
        return None
    try:
        payload = json.loads(split_used_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    normalized = {
        key: sorted(str(item) for item in payload.get(key) or [])
        for key in (
            "train_subject_ids",
            "selection_subject_ids",
            "final_eval_subject_ids",
            "train_inner_subject_ids",
            "val_inner_subject_ids",
        )
    }
    normalized["split_names"] = payload.get("split_names")
    return {"sha256": canonical_sha256(normalized), "counts": {key: len(value) for key, value in normalized.items() if isinstance(value, list)}}


def load_matrix_cells(matrix_path: Path) -> list[dict[str, Any]]:
    if not matrix_path.is_file():
        raise InventoryError(f"missing matrix: {matrix_path}")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    cells: list[dict[str, Any]] = []
    for entry in matrix.get("experiments") or []:
        config_rel = str(entry["config"])
        config_path = PROJECT_ROOT / config_rel
        if not config_path.is_file():
            raise InventoryError(f"matrix cell config is missing: {config_rel}")
        config = load_yaml_with_overrides(config_path, [])
        backend = resolve_model_backend(config)
        if backend not in QWEN3_BACKENDS:
            continue
        modality = resolve_input_modality(config)
        cells.append(
            {
                "cell_id": f"{config.get('dataset')}_{modality}_{backend}",
                "matrix": str(matrix_path.relative_to(PROJECT_ROOT)),
                "config_path": config_rel,
                "config_sha256": sha256_file(config_path),
                "dataset": str(config.get("dataset")),
                "modality": modality,
                "backend": backend,
                "recipe_id": str(config.get("recipe_id")),
                "folds": [int(fold) for fold in entry.get("folds") or [0]],
                "separate_eval": bool(entry.get("separate_eval")),
            }
        )
    return cells


def scan_candidates(scan_root: Path, exclusion_markers: tuple[str, ...]) -> list[dict[str, Any]]:
    """Cheap scan: only Qwen3-backend run directories are parsed."""
    if not scan_root.is_dir():
        raise InventoryError(f"scan root is not a directory: {scan_root}")
    candidates: list[dict[str, Any]] = []
    for run_config_path in sorted(scan_root.rglob("run_config.yaml")):
        fold_dir = run_config_path.parent
        if _FOLD_DIR_PATTERN.fullmatch(fold_dir.name) is None:
            continue
        relative = fold_dir.relative_to(scan_root)
        if len(relative.parts) < 4:
            continue
        try:
            text = run_config_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "model_backend: qwen38" not in text and "model_backend: qwen3omni" not in text:
            continue
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            continue
        config = payload["config"]
        backend = resolve_model_backend(config)
        if backend not in QWEN3_BACKENDS:
            continue
        lowered = str(relative).lower()
        markers = [marker for marker in exclusion_markers if marker in lowered]
        candidates.append(
            {
                "scan_root": str(scan_root),
                "fold_dir": str(fold_dir),
                "relative": str(relative),
                "fold": int(fold_dir.name.split("_", 1)[1]),
                "run_name": relative.parts[-2],
                "path_modality": relative.parts[-4],
                "path_dataset": relative.parts[-3],
                "backend": backend,
                "payload": payload,
                "config": config,
                "exclusion_markers": markers,
            }
        )
    return candidates


def _terminal_summary(fold_dir: Path) -> dict[str, Any]:
    jobs_path = fold_dir / "jobs.jsonl"
    events: list[dict[str, Any]] = []
    if jobs_path.is_file():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if str(event.get("event_type")) in TERMINAL_EVENT_TYPES:
                events.append(
                    {
                        "job_key": event.get("job_key") or event.get("job_name"),
                        "event_type": event.get("event_type"),
                        "exit_code": event.get("exit_code"),
                        "slurm_job_id": event.get("slurm_job_id"),
                    }
                )
    completed = [event for event in events if event["event_type"] == "COMPLETED"]
    failed = [event for event in events if event["event_type"] in FAILED_EVENT_TYPES]
    return {
        "terminal_events": events,
        "completed_events": len(completed),
        "failed_events": len(failed),
        "jobs_clean": bool(completed) and not failed,
    }


def evaluate_candidate(
    candidate: dict[str, Any], cell: dict[str, Any], current_config: dict[str, Any]
) -> dict[str, Any]:
    config = candidate["config"]
    payload = candidate["payload"]
    fold_dir = Path(candidate["fold_dir"])
    checks: dict[str, Any] = {}

    differences = diff_paths(reduce_config(current_config) or {}, reduce_config(config) or {})
    scientific_differences = [path for path in differences if not is_shape_path(path)]
    shape_differences = [path for path in differences if is_shape_path(path)]
    checks["config_differences"] = differences
    checks["scientific_differences"] = scientific_differences
    checks["shape_differences"] = shape_differences
    checks["config_matches"] = not differences
    checks["science_matches"] = not scientific_differences

    recorded_prompt = payload.get("prompt_context") or {}
    current_prompt_hash = prompt_sha256(current_config)
    current_prompt = current_config.get("prompt") or {}
    checks["prompt_sha256_recorded"] = recorded_prompt.get("system_prompt_sha256")
    checks["prompt_sha256_current"] = current_prompt_hash
    checks["prompt_matches"] = bool(
        recorded_prompt.get("system_prompt_sha256")
        and recorded_prompt.get("system_prompt_sha256") == current_prompt_hash
        and str(recorded_prompt.get("version")) == str(current_prompt.get("version"))
        and str(recorded_prompt.get("dataset_context")) == str(current_prompt.get("dataset_context"))
        and str(recorded_prompt.get("question_context_version"))
        == str(current_prompt.get("question_context_version") or "legacy_v1")
    )

    checks["modality_recorded"] = payload.get("input_modality")
    checks["modality_matches"] = payload.get("input_modality") == cell["modality"]
    checks["model_backend"] = candidate["backend"]
    checks["model_basename_recorded"] = Path(str(config.get("model_name_or_path") or "")).name
    checks["model_basename_current"] = Path(str(current_config.get("model_name_or_path") or "")).name
    checks["model_revision_recorded"] = config.get("model_revision")
    checks["model_revision_current"] = current_config.get("model_revision")
    checks["model_identity_matches"] = (
        checks["model_basename_recorded"] == checks["model_basename_current"]
        and checks["model_revision_recorded"] == checks["model_revision_current"]
    )

    checks["seed"] = config.get("seed")
    checks["split_seed"] = (config.get("split") or {}).get("seed")
    checks["seed_matches"] = checks["seed"] == EXPECTED_SEED and checks["split_seed"] == EXPECTED_SEED

    evaluation = config.get("evaluation") or {}
    checks["sample_prediction_mode"] = evaluation.get("sample_prediction_mode")
    checks["evaluation_view"] = evaluation.get("evaluation_view")
    checks["aggregation_level"] = evaluation.get("aggregation_level")
    checks["subject_score_aggregation"] = evaluation.get("subject_score_aggregation")
    checks["hierarchical_score_aggregation"] = evaluation.get("hierarchical_score_aggregation")
    checks["evaluation_contract_matches"] = (
        evaluation.get("sample_prediction_mode") == EXPECTED_PREDICTION_MODE
        and evaluation.get("evaluation_view") == EXPECTED_EVALUATION_VIEW
    )

    status_path = fold_dir / "status.json"
    state = None
    if status_path.is_file():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
        except (OSError, ValueError):
            state = None
    checks["state"] = state
    checks["state_reusable"] = state in REUSABLE_STATES

    jobs = _terminal_summary(fold_dir)
    checks["jobs"] = jobs
    checks["jobs_clean"] = jobs["jobs_clean"]

    checkpoint = fold_dir / "best_model"
    checks["best_model_present"] = checkpoint.is_dir()
    checks["best_model_adapter_files"] = all(
        (checkpoint / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")
    )

    try:
        discovered = discovery.discover_run_at(fold_dir)
    except (ValueError, OSError) as error:  # pragma: no cover - defensive
        discovered = None
        checks["discovery_error"] = str(error)
    evidence: dict[str, Any] = {"subject_predictions": None, "metrics": None, "recomputed": None}
    if discovered is not None:
        for artifact in discovered.artifacts:
            if artifact.kind == "subject_predictions" and evidence["subject_predictions"] is None:
                evidence["subject_predictions"] = artifact.relative_path
            if (
                artifact.kind == "metrics"
                and isinstance(artifact.json_content, dict)
                and "binary_strict_macro_f1" in artifact.json_content
                and evidence["metrics"] is None
            ):
                evidence["metrics"] = {
                    "relative_path": artifact.relative_path,
                    "binary_strict_macro_f1": artifact.json_content.get("binary_strict_macro_f1"),
                    "binary_strict_positive_f1": artifact.json_content.get("binary_strict_positive_f1"),
                    "binary_strict_uar": artifact.json_content.get("binary_strict_uar"),
                }
    checks["local_evidence_present"] = bool(evidence["subject_predictions"] and evidence["metrics"])
    if evidence["subject_predictions"]:
        try:
            recomputed = recompute_strict_headline(fold_dir / evidence["subject_predictions"])
        except (ValidationError, OSError, KeyError, ValueError) as error:
            checks["recomputation_error"] = str(error)
        else:
            evidence["recomputed"] = recomputed
            recorded = evidence["metrics"] or {}
            checks["recomputation_matches"] = bool(recorded) and all(
                abs(float(recomputed[key]) - float(recorded[key])) <= 1e-6
                for key in (
                    "binary_strict_macro_f1",
                    "binary_strict_positive_f1",
                    "binary_strict_uar",
                )
                if recorded.get(key) is not None
            )
    checks["evidence"] = evidence

    checks["manifest_hash_recorded"] = payload.get("manifest_hash")
    checks["manifest_path_recorded"] = payload.get("manifest_path")
    checks["split_metadata_hash_recorded"] = payload.get("split_metadata_hash")
    checks["split_fingerprint"] = split_fingerprint(fold_dir / "logs" / "split_used.json")
    training_strategy = payload.get("training_strategy") or {}
    checks["world_size_recorded"] = training_strategy.get("world_size")
    checks["effective_global_batch_recorded"] = training_strategy.get("effective_global_batch_size")

    blocking: list[str] = []
    if not checks["science_matches"]:
        blocking.append("science")
    if not checks["prompt_matches"]:
        blocking.append("prompt")
    if not checks["model_identity_matches"]:
        blocking.append("model_identity")
    if not checks["seed_matches"]:
        blocking.append("seed")
    if not checks["evaluation_contract_matches"]:
        blocking.append("evaluation_contract")
    if not checks["modality_matches"]:
        blocking.append("modality")

    missing: list[str] = []
    if not checks["state_reusable"]:
        missing.append(f"state={state}")
    if not checks["jobs_clean"]:
        missing.append("terminal_jobs")
    if not checks["best_model_present"]:
        missing.append("best_model")
    if not checks["local_evidence_present"]:
        missing.append("local_evidence")
    elif checks.get("recomputation_matches") is not True:
        missing.append("recomputation")

    evidence_gaps: list[str] = []
    if checks["best_model_present"] and not checks["best_model_adapter_files"]:
        evidence_gaps.append("adapter_weights_not_local")

    if blocking:
        verdict = "incompatible"
    elif shape_differences:
        verdict = "reusable_pending_shape_decision"
    elif missing:
        verdict = "incomplete"
    else:
        verdict = "reusable"

    return {
        "scan_root": candidate["scan_root"],
        "fold_dir": candidate["fold_dir"],
        "fold": candidate["fold"],
        "run_name": candidate["run_name"],
        "path_modality": candidate["path_modality"],
        "path_dataset": candidate["path_dataset"],
        "backend": candidate["backend"],
        "exclusion_markers": candidate["exclusion_markers"],
        "attempt_id": (payload.get("tracking") or {}).get("attempt_id"),
        "run_config_file_sha256": sha256_file(fold_dir / "run_config.yaml"),
        "config_overrides": payload.get("config_overrides"),
        "verdict": verdict,
        "blocking_checks": blocking,
        "missing_checks": missing,
        "evidence_gaps": evidence_gaps,
        "checks": checks,
    }


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    matrices = args.matrix or [DEFAULT_MATRIX]
    cells: list[dict[str, Any]] = []
    for matrix in matrices:
        cells.extend(load_matrix_cells(Path(matrix)))

    exclusion_markers = tuple(args.exclude_marker)
    scan_roots = [Path(root) for root in args.scan_root]
    candidates: list[dict[str, Any]] = []
    for scan_root in scan_roots:
        candidates.extend(scan_candidates(scan_root, exclusion_markers))

    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (candidate["path_dataset"], candidate["path_modality"], candidate["fold"])
        by_cell.setdefault(key, []).append(candidate)

    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(PROJECT_ROOT),
        "scan_roots": [str(root) for root in scan_roots],
        "matrices": [str(Path(matrix).relative_to(PROJECT_ROOT)) for matrix in matrices],
        "expected_seed": EXPECTED_SEED,
        "expected_sample_prediction_mode": EXPECTED_PREDICTION_MODE,
        "expected_evaluation_view": EXPECTED_EVALUATION_VIEW,
        "exclusion_markers": list(exclusion_markers),
        "shape_diff_paths": sorted(SHAPE_DIFF_PATHS),
        "cells": [],
        "summary": {},
    }

    counts = {name: 0 for name in VERDICT_RANK}
    for cell in cells:
        current_config = load_yaml_with_overrides(PROJECT_ROOT / cell["config_path"], [])
        fold_records: list[dict[str, Any]] = []
        cell_manifest_hashes: dict[str, set[str]] = {}
        for fold in cell["folds"]:
            candidates_for_fold = by_cell.get((cell["dataset"], cell["modality"], fold), [])
            evaluated = [
                evaluate_candidate(candidate, cell, current_config)
                for candidate in candidates_for_fold
            ]
            admissible = [item for item in evaluated if not item["exclusion_markers"]]
            excluded = [item for item in evaluated if item["exclusion_markers"]]
            admissible.sort(
                key=lambda item: (VERDICT_RANK[item["verdict"]], item["run_name"]), reverse=True
            )
            if admissible:
                best = admissible[0]
                manifest_hash = best["checks"]["manifest_hash_recorded"]
                if manifest_hash:
                    cell_manifest_hashes.setdefault(str(manifest_hash), set()).add(best["run_name"])
                fold_records.append(
                    {
                        "fold": fold,
                        "verdict": best["verdict"],
                        "selected_run": best["run_name"],
                        "selected_scan_root": best["scan_root"],
                        "selected_attempt_id": best["attempt_id"],
                        "candidates": admissible,
                        "excluded_candidates": excluded,
                    }
                )
            else:
                fold_records.append(
                    {
                        "fold": fold,
                        "verdict": "missing",
                        "selected_run": None,
                        "candidates": [],
                        "excluded_candidates": excluded,
                    }
                )
        cell_verdict_rank = max(VERDICT_RANK[record["verdict"]] for record in fold_records)
        cell_verdict = {value: key for key, value in VERDICT_RANK.items()}[cell_verdict_rank]
        counts[cell_verdict] += 1
        inventory["cells"].append(
            {
                **cell,
                "cell_verdict": cell_verdict,
                "manifest_hash_consensus": (
                    {"distinct_hashes": len(cell_manifest_hashes), "hashes": {key: sorted(value) for key, value in cell_manifest_hashes.items()}}
                    if cell_manifest_hashes
                    else None
                ),
                "folds": fold_records,
            }
        )

    inventory["summary"] = {
        "cells": len(cells),
        "cell_verdicts": counts,
        "candidate_runs_scanned": len(candidates),
        "folds": sum(len(cell["folds"]) for cell in inventory["cells"]),
    }
    return inventory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        type=Path,
        default=None,
        help="matrix YAML (repeatable; default: the native standalone matrix)",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        required=True,
        help="root to scan for run directories (repeatable)",
    )
    parser.add_argument(
        "--exclude-marker",
        action="append",
        default=list(DEFAULT_EXCLUSION_MARKERS),
        help="run-path markers excluded from reuse decisions (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inventory = build_inventory(args)
    except InventoryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for cell in inventory["cells"]:
        folds = ", ".join(f"{record['fold']}:{record['verdict']}" for record in cell["folds"])
        print(f"  {cell['cell_id']:<40} {cell['cell_verdict']:<32} folds[{folds}]")
    print("summary:", json.dumps(inventory["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
