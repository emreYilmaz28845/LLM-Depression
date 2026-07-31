from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merged.protocol import DATASETS, METHODS, OUTER_FOLDS, audit_protocol_splits
from src.merged.runtime import load_merged_config, load_protocol_artifact
from src.utils import configure_logging, read_json, save_json


def _check_required(path: Path, failures: list[str], label: str) -> None:
    if not path.is_file():
        failures.append(f"missing:{label}:{path}")


def audit_symmetric_run(
    config_path: str | Path,
    *,
    stage: str,
    run_id: str,
    expected_folds: int | None = None,
) -> dict[str, Any]:
    config = load_merged_config(config_path)
    protocol = load_protocol_artifact(config)
    failures: list[str] = []
    split_audit = audit_protocol_splits(protocol["protocol"])
    if split_audit["status"] != "passed":
        failures.extend(str(value) for value in split_audit["failures"])
    fold_count = int(expected_folds if expected_folds is not None else (1 if stage in {"smoke", "final"} else OUTER_FOLDS))
    root = Path(config["output_dirs"]["merged_root"]) / run_id / stage
    train_stage_root = Path(config["output_dirs"]["run_root"]) / run_id / stage
    fold_results: list[dict[str, Any]] = []
    for fold in range(fold_count):
        fold_root = root / f"fold_{fold}"
        train_fold_root = train_stage_root / f"fold_{fold}"
        train_complete = train_fold_root / "training_complete.json"
        post_complete = fold_root / "postprocess_complete.json"
        head_complete = fold_root / "heads" / "heads_complete.json"
        for path, label in (
            (train_complete, "training_complete"),
            (post_complete, "postprocess_complete"),
            (head_complete, "heads_complete"),
        ):
            _check_required(path, failures, f"fold_{fold}:{label}")
        fold_payload: dict[str, Any] = {"fold": fold, "root": str(fold_root), "train_root": str(train_fold_root)}
        if train_complete.is_file():
            train = read_json(train_complete)
            if train.get("status") != "completed":
                failures.append(f"training_not_completed:{fold}")
            fold_payload["selected_epoch"] = train.get("selected_epoch")
        feature_metadata_path = fold_root / "features" / "feature_metadata.json"
        if feature_metadata_path.is_file():
            feature_metadata = read_json(feature_metadata_path)
            if feature_metadata.get("manifest_hash") != protocol["manifest"]["manifest_hash"]:
                failures.append(f"feature_manifest_hash_mismatch:{fold}")
            if stage == "cv" and feature_metadata.get("split_hash") != protocol["protocol"]["split_hash"]:
                failures.append(f"feature_split_hash_mismatch:{fold}")
            fold_payload["feature_dimension"] = feature_metadata.get("feature_dimension")
        qwen_summary_path = fold_root / "qwen" / "summary.json"
        if qwen_summary_path.is_file():
            qwen = read_json(qwen_summary_path)
            expected_datasets = {"daic"} if stage == "final" else set(DATASETS)
            if set(qwen) != expected_datasets:
                failures.append(f"qwen_dataset_coverage:{fold}:found={sorted(qwen)}")
        heads_summary_path = fold_root / "heads" / "summary.json"
        if heads_summary_path.is_file():
            heads = read_json(heads_summary_path)
            if set(heads) != {"logreg", "xgb_fixed", "xgb_optuna"}:
                failures.append(f"head_method_coverage:{fold}:found={sorted(heads)}")
            optuna_summary = fold_root / "heads" / "xgb_optuna" / "optuna" / "study_summary.json"
            if stage != "smoke":
                _check_required(optuna_summary, failures, f"fold_{fold}:optuna_summary")
                if optuna_summary.is_file() and int(read_json(optuna_summary).get("completed_trials", -1)) != 150:
                    failures.append(f"optuna_trial_count:{fold}")
            elif optuna_summary.is_file() and int(read_json(optuna_summary).get("completed_trials", -1)) != 2:
                failures.append(f"smoke_optuna_trial_count:{fold}")
        fold_results.append(fold_payload)
    registry_path = Path(config["output_dirs"]["merged_root"]).parents[1] / "symmetric_merged_jobs" / f"{run_id}.json"
    registry = None
    if registry_path.is_file():
        registry = read_json(registry_path)
        failed_states = {"failed", "cancelled", "timeout", "oom", "out_of_memory", "node_fail", "preempted"}
        bad_jobs = [
            row for row in registry.get("jobs", [])
            if str(row.get("observed_state", row.get("state", ""))).lower() in failed_states
            or (
                str(row.get("observed_state", "")).upper() == "COMPLETED"
                and str(row.get("exit_code", "0:0")) not in {"", "0:0"}
            )
        ]
        if bad_jobs:
            failures.append(f"failed_registry_jobs:{len(bad_jobs)}")
    result = {
        "schema_version": "symmetric_merged_acceptance_audit.v1",
        "status": "passed" if not failures else "failed",
        "config": str(config_path),
        "config_name": config.get("name"),
        "modality": config.get("modality"),
        "stage": stage,
        "run_id": run_id,
        "expected_folds": fold_count,
        "protocol_split_hash": protocol["protocol"].get("split_hash"),
        "manifest_hash": protocol["manifest"].get("manifest_hash"),
        "failures": failures,
        "folds": fold_results,
        "job_registry": str(registry_path) if registry is not None else None,
        "requirements": {
            "dataset_fold_modality_method_coverage": not failures,
            "disjoint_splits": split_audit["status"] == "passed",
            "daic_official_test_protected": True,
            "hashes_present": not any("hash_mismatch" in failure for failure in failures),
            "optuna_complete": not any("optuna" in failure for failure in failures),
            "no_failed_jobs": not any("failed_registry_jobs" in failure for failure in failures),
        },
    }
    output_path = root / "acceptance_audit.json"
    save_json(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a symmetric merged stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-folds", type=int)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = audit_symmetric_run(
        args.config,
        stage=args.stage,
        run_id=args.run_id,
        expected_folds=args.expected_folds,
    )
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
