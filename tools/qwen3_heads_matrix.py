#!/usr/bin/env python3
"""Plan the standalone fixed-head matrix for the Qwen3 three-seed campaign.

The head stage is separate from training: for every eligible training
checkpoint the matrix extracts hidden features from the fold's ``best_model``
and fits the fixed classifiers (Logistic Regression and XGBoost, no Optuna).
This tool builds that job graph and fails closed on anything that would make a
head number untraceable:

* a parent checkpoint is resolved by *scanning* the cell's run root for fold
  directories whose recorded resolved config matches the current cell config
  under the same reduction the readiness inventory uses, and whose recorded
  training seed equals the requested seed. Run names are not evidence; two
  matching runs are an ambiguity and are refused;
* the extraction GPU shape is read from the parent's recorded
  ``resources.eval_gpus_per_node`` (1 for Qwen3.8 text, 4 for the Qwen3-Omni
  sharded route) and never hardcoded;
* the parent's recorded prompt hash and translation-notice version must match
  the cell config, so an English cell can never be bound to a native checkpoint
  and a native cell to an English one;
* the adapter files must exist locally and are hashed into the plan;
* a cell/seed/fold without a matching parent is recorded as
  ``waiting_for_checkpoint`` instead of being planned silently.

``--check`` re-validates a plan; ``--emit`` writes the machine-readable job
manifest. Nothing is submitted by this tool.
"""

from __future__ import annotations

import argparse
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
    prompt_sha256,
    reduce_config,
)
from tools.qwen3_multiseed_plan import (  # noqa: E402
    PLANNED_SEEDS,
    SPLIT_SEED,
    build_selection_map,
)

SCHEMA_VERSION = "audiollm.qwen3_heads_matrix.v1"
HEAD_VARIANTS = ("logreg_raw", "xgb_raw")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


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


def resolve_parent(
    *,
    cell: dict[str, Any],
    cell_config: dict[str, Any],
    run_root: Path,
    fold: int,
    seed: int,
    notice_version: str | None,
) -> dict[str, Any]:
    """Find the one matching training attempt for a cell/seed/fold."""
    matches: list[dict[str, Any]] = []
    for fold_dir in _fold_dirs(run_root):
        if fold_dir.name != f"fold_{fold}":
            continue
        payload = _recorded_config(fold_dir)
        if payload is None:
            continue
        recorded = payload.get("config")
        if not isinstance(recorded, dict):
            continue
        if diff_paths(reduce_config(cell_config) or {}, reduce_config(recorded) or {}):
            continue
        if int(recorded.get("seed", -1)) != int(seed):
            continue
        recorded_prompt = payload.get("prompt_context") or {}
        prompt_recorded = recorded_prompt.get("system_prompt_sha256")
        prompt_current = prompt_sha256(cell_config)
        notice_recorded = recorded_prompt.get("translation_notice_version")
        matches.append(
            {
                "fold_dir": str(fold_dir),
                "run_name": fold_dir.parent.name,
                "attempt_id": (payload.get("tracking") or {}).get("attempt_id"),
                "seed": int(recorded.get("seed", -1)),
                "split_seed": int((recorded.get("split") or {}).get("seed", -1)),
                "prompt_sha256_recorded": prompt_recorded,
                "prompt_matches": prompt_recorded == prompt_current,
                "translation_notice_version_recorded": notice_recorded,
                "notice_matches": str(notice_recorded or "") == str(notice_version or ""),
                "eval_gpus_per_node": int(
                    (recorded.get("resources") or {}).get("eval_gpus_per_node", 1)
                ),
                "model_revision": recorded.get("model_revision"),
                "model_backend": recorded.get("model_backend"),
            }
        )
    if not matches:
        return {
            "status": "waiting_for_checkpoint",
            "reason": (
                f"no {seed}-seed run for {cell['config']} fold {fold} under {run_root}"
            ),
        }
    if len(matches) > 1:
        raise HeadsMatrixError(
            f"ambiguous parent for {cell['config']} seed {seed} fold {fold}: "
            f"{[match['run_name'] for match in matches]}"
        )
    match = matches[0]
    fold_dir = Path(match["fold_dir"])
    checkpoint = fold_dir / "best_model"
    missing = [name for name in ADAPTER_FILES if not (checkpoint / name).is_file()]
    if missing:
        return {
            "status": "waiting_for_checkpoint",
            "reason": f"{checkpoint} is missing {missing}",
            "candidate": match,
        }
    if not match["prompt_matches"]:
        raise HeadsMatrixError(
            f"parent {match['run_name']} fold {fold} prompt hash differs from the cell config"
        )
    if not match["notice_matches"]:
        raise HeadsMatrixError(
            "parent "
            f"{match['run_name']} fold {fold} translation notice version "
            f"{match['translation_notice_version_recorded']!r} does not match the cell's "
            f"{notice_version!r}: an English cell must not bind a native checkpoint"
        )
    if int(match["split_seed"]) != SPLIT_SEED:
        raise HeadsMatrixError(
            f"parent {match['run_name']} fold {fold} split seed is {match['split_seed']}, "
            f"expected {SPLIT_SEED}"
        )
    return {
        "status": "resolved",
        "run_name": match["run_name"],
        "attempt_id": match["attempt_id"],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_adapter_config_sha256": sha256_file(checkpoint / "adapter_config.json"),
        "checkpoint_adapter_model_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "model_backend": match["model_backend"],
        "model_revision": match["model_revision"],
        "prompt_sha256": match["prompt_sha256_recorded"],
        "translation_notice_version": match["translation_notice_version_recorded"],
        "extract_gpus": int(match["eval_gpus_per_node"]),
        "seed": seed,
        "fold": fold,
    }


def build_matrix(*, seeds: list[int], scan_roots: list[Path]) -> dict[str, Any]:
    selection = build_selection_map()
    routes = selection["routes"]
    matrix: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "planned_seeds": seeds,
        "split_seed": SPLIT_SEED,
        "head_variants": list(HEAD_VARIANTS),
        "scan_roots": [str(root) for root in scan_roots],
        "routes": [],
    }
    for route in routes:
        cell_config = yaml.safe_load(
            (PROJECT_ROOT / route["config"]).read_text(encoding="utf-8")
        )
        run_root_template = str(cell_config["output_dirs"]["run_root"])
        relative = run_root_template.split("${PROJECT_ROOT}/", 1)[-1].lstrip("/")
        candidate_roots = [PROJECT_ROOT / relative] + [
            scan_root / relative for scan_root in scan_roots
        ]
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
                parent = {"status": "waiting_for_checkpoint", "reason": "no candidate run root holds this cell"}
                for run_root in candidate_roots:
                    attempt = resolve_parent(
                        cell=route,
                        cell_config=cell_config,
                        run_root=run_root,
                        fold=fold,
                        seed=seed,
                        notice_version=notice_version,
                    )
                    if attempt["status"] == "resolved":
                        parent = attempt
                        break
                    if "candidate" in attempt:
                        parent = attempt
                job: dict[str, Any] = {
                    "route_id": route["route_id"],
                    "config": route["config"],
                    "seed": seed,
                    "fold": fold,
                    "parent_status": parent["status"],
                }
                if parent["status"] == "resolved":
                    cache_dir = (
                        Path(parent["checkpoint_dir"]).parents[2]
                        / "hidden_features"
                        / route["dataset"]
                        / f"{parent['run_name']}_fold_{fold}_seed{seed}"
                    )
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
                                "cache_dir": str(cache_dir),
                                "depends_on": "extract",
                            },
                        }
                    )
                else:
                    job["reason"] = parent.get("reason")
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
    }
    return matrix


def check_matrix(matrix: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if matrix.get("schema_version") != SCHEMA_VERSION:
        failures.append("unexpected schema version")
    if tuple(matrix.get("head_variants") or ()) != HEAD_VARIANTS:
        failures.append("the fixed head variants changed")
    if len(matrix.get("routes") or []) != 23:
        failures.append("the head matrix must cover the 23 standalone routes")
    for route in matrix.get("routes") or []:
        english = route["language"] == "english"
        for job in route["jobs"]:
            if job["parent_status"] != "resolved":
                continue
            parent = job["parent"]
            if not parent.get("attempt_id"):
                failures.append(f"{route['route_id']}: resolved parent without an attempt id")
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
            matrix = build_matrix(seeds=seeds, scan_roots=list(args.scan_root))
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
        f"{matrix['summary']['waiting_for_checkpoint']} waiting for checkpoint"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
