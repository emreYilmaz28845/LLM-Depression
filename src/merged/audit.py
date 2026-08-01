from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merged.protocol import (
    DATASETS,
    METHODS,
    OUTER_FOLDS,
    audit_protocol_splits,
    canonical_sha256,
)
from src.merged.runtime import load_merged_config, load_protocol_artifact
from src.utils import configure_logging, read_json, read_jsonl, save_json


def _check_required(path: Path, failures: list[str], label: str) -> None:
    if not path.is_file():
        failures.append(f"missing:{label}:{path}")


def _check_present(path: Path, failures: list[str], label: str) -> None:
    if not path.exists():
        failures.append(f"missing:{label}:{path}")


def _feature_subjects(
    rows_path: Path,
    metadata: dict[str, Any],
    partition: str,
    failures: list[str],
    fold: int,
) -> tuple[set[str], set[str]]:
    """Return feature subjects/sample IDs while checking row-level provenance."""

    if not rows_path.is_file():
        return set(), set()
    rows = read_jsonl(rows_path)
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    subject_ids = {str(row.get("subject_id", "")) for row in rows}
    if any(not value or value == "None" for value in sample_ids + list(subject_ids)):
        failures.append(f"feature_identity_missing:{fold}:{partition}")
    if len(sample_ids) != len(set(sample_ids)):
        failures.append(f"feature_sample_duplicates:{fold}:{partition}")
    if any("::" not in value for value in subject_ids):
        failures.append(f"feature_subject_namespace:{fold}:{partition}")
    expected_hash = (metadata.get("row_hashes") or {}).get(partition)
    if expected_hash and canonical_sha256(rows) != expected_hash:
        failures.append(f"feature_row_hash_mismatch:{fold}:{partition}")
    summary = (metadata.get("partitions") or {}).get(partition) or {}
    if int(summary.get("row_count", len(rows))) != len(rows):
        failures.append(f"feature_row_count_mismatch:{fold}:{partition}")
    if int(summary.get("subject_count", len(subject_ids))) != len(subject_ids):
        failures.append(f"feature_subject_count_mismatch:{fold}:{partition}")
    return subject_ids, set(sample_ids)


def _audit_provenance(
    path: Path,
    failures: list[str],
    label: str,
    expected_worker: str,
) -> None:
    if not path.is_file():
        return
    payload = read_json(path)
    if not str(payload.get("source_commit", "")).strip():
        failures.append(f"provenance_source_commit_missing:{label}")
    if payload.get("worker") != expected_worker:
        failures.append(f"provenance_worker_mismatch:{label}")
    scheduler = payload.get("scheduler") or {}
    if not str(scheduler.get("SLURM_JOB_ID", "")).strip():
        failures.append(f"provenance_slurm_job_missing:{label}")


def _audit_head_inner_folds(
    path: Path,
    rows_path: Path,
    *,
    expected_subjects: set[str],
    failures: list[str],
    fold: int,
) -> None:
    """Validate grouped head-tuning assignments against outer-train rows."""

    if not path.is_file() or not rows_path.is_file():
        return
    payload = read_json(path)
    assignments = payload.get("folds") or []
    if int(payload.get("inner_folds", -1)) != 3 or len(assignments) != 3:
        failures.append(f"head_inner_fold_count:{fold}")
        return
    hash_payload = {key: value for key, value in payload.items() if key != "assignments_hash"}
    if payload.get("assignments_hash") != canonical_sha256(hash_payload):
        failures.append(f"head_inner_fold_hash_mismatch:{fold}")
    rows = read_jsonl(rows_path)
    all_indices = set(range(len(rows)))
    validation_indices: list[int] = []
    validation_subjects: list[str] = []
    for inner_fold in assignments:
        train_indices = [int(value) for value in inner_fold.get("train_row_indices", [])]
        inner_validation_indices = [int(value) for value in inner_fold.get("validation_row_indices", [])]
        if set(train_indices) & set(inner_validation_indices):
            failures.append(f"head_inner_row_overlap:{fold}:{inner_fold.get('fold')}")
        if not set(train_indices) <= all_indices or not set(inner_validation_indices) <= all_indices:
            failures.append(f"head_inner_row_out_of_range:{fold}:{inner_fold.get('fold')}")
        row_train_subjects = {str(rows[index]["subject_id"]) for index in train_indices if index in all_indices}
        row_validation_subjects = {
            str(rows[index]["subject_id"])
            for index in inner_validation_indices
            if index in all_indices
        }
        if row_train_subjects != set(str(value) for value in inner_fold.get("train_subject_ids", [])):
            failures.append(f"head_inner_train_subject_mismatch:{fold}:{inner_fold.get('fold')}")
        if row_validation_subjects != set(str(value) for value in inner_fold.get("validation_subject_ids", [])):
            failures.append(f"head_inner_validation_subject_mismatch:{fold}:{inner_fold.get('fold')}")
        if row_train_subjects & row_validation_subjects:
            failures.append(f"head_inner_subject_overlap:{fold}:{inner_fold.get('fold')}")
        if row_train_subjects | row_validation_subjects != expected_subjects:
            failures.append(f"head_inner_outer_pool_mismatch:{fold}:{inner_fold.get('fold')}")
        validation_indices.extend(inner_validation_indices)
        validation_subjects.extend(sorted(row_validation_subjects))
    if sorted(validation_indices) != sorted(all_indices) or len(validation_indices) != len(set(validation_indices)):
        failures.append(f"head_inner_validation_row_coverage:{fold}")
    if len(validation_subjects) != len(set(validation_subjects)):
        failures.append(f"head_inner_validation_subject_coverage:{fold}")


def _audit_training_artifacts(
    train_root: Path,
    *,
    stage: str,
    failures: list[str],
    fold: int,
) -> None:
    """Check persisted weighting, schedule, and selection invariants."""

    weighting_path = train_root / "logs" / "weighting_audit.json"
    if weighting_path.is_file():
        weighting = read_json(weighting_path)
        for key in ("equal_dataset_totals", "natural_class_prevalence_preserved", "no_sampling", "no_duplication"):
            if weighting.get(key) is not True:
                failures.append(f"weighting_invariant_failed:{fold}:{key}")
        if not math.isclose(float(weighting.get("mean_loss_weight", 0.0)), 1.0, rel_tol=0.0, abs_tol=1e-8):
            failures.append(f"weighting_mean_not_normalized:{fold}")
        if len(weighting.get("datasets", [])) != len(DATASETS):
            failures.append(f"weighting_dataset_coverage:{fold}")

    schedule_path = train_root / "logs" / "schedule_audit.json"
    schedule_payload = read_json(schedule_path) if schedule_path.is_file() else {}
    epochs = schedule_payload.get("epochs") or []
    if not epochs:
        failures.append(f"schedule_epochs_missing:{fold}")
    for epoch in epochs:
        example_count = int(epoch.get("example_count", -1))
        occurrences = {int(index): int(count) for index, count in (epoch.get("sample_occurrence_counts") or {}).items()}
        if example_count < 1 or set(occurrences) != set(range(example_count)) or set(occurrences.values()) != {1}:
            failures.append(f"schedule_one_time_coverage:{fold}:{epoch.get('epoch')}")
        blocks = epoch.get("blocks") or []
        flattened = [int(index) for block in blocks for index in block.get("example_indices", [])]
        if sorted(flattened) != list(range(example_count)):
            failures.append(f"schedule_block_coverage:{fold}:{epoch.get('epoch')}")

    history_path = train_root / "logs" / "training_history.json"
    if not history_path.is_file():
        return
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        failures.append(f"training_history_empty:{fold}")
        return
    if stage != "final":
        for row in history:
            metrics = row.get("component_selection_metrics") or {}
            if set(metrics) != set(DATASETS):
                failures.append(f"selection_dataset_coverage:{fold}:{row.get('epoch')}")
                continue
            values = [float(metrics[dataset]["macro_f1"]) for dataset in DATASETS]
            expected = sum(values) / len(values)
            if not math.isclose(float(row.get("mean_dataset_macro_f1", float("nan"))), expected, rel_tol=0.0, abs_tol=1e-10):
                failures.append(f"selection_mean_mismatch:{fold}:{row.get('epoch')}")


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
    split_audit = audit_protocol_splits(
        protocol["protocol"], require_daic_official_test_count=True
    )
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
        for path, label in (
            (train_fold_root / "training_identity.json", "training_identity"),
            (train_fold_root / "resolved_merged_config.json", "training_resolved_config"),
            (train_fold_root / "slurm_provenance.json", "train_provenance"),
            (train_fold_root / "logs" / "composition.json", "composition"),
            (train_fold_root / "logs" / "weighting_audit.json", "weighting_audit"),
            (train_fold_root / "logs" / "schedule_audit.json", "schedule_audit"),
            (train_fold_root / "logs" / "training_history.json", "training_history"),
            (train_fold_root / "logs" / "selected_checkpoint.json", "selected_checkpoint"),
            (train_fold_root / "best_model", "best_model"),
            (fold_root / "postprocess_identity.json", "postprocess_identity"),
            (fold_root / "resolved_merged_config.json", "postprocess_resolved_config"),
            (fold_root / "slurm_provenance.json", "postprocess_provenance"),
            (fold_root / "features" / "outer_train.npz", "outer_train_features"),
            (fold_root / "features" / "outer_train_rows.jsonl", "outer_train_feature_rows"),
            (fold_root / "features" / "outer_holdout.npz", "outer_holdout_features"),
            (fold_root / "features" / "outer_holdout_rows.jsonl", "outer_holdout_feature_rows"),
            (fold_root / "features" / "feature_metadata.json", "feature_metadata"),
            (fold_root / "qwen" / "summary.json", "qwen_summary"),
            (fold_root / "heads" / "summary.json", "heads_summary"),
            (fold_root / "heads" / "resolved_merged_config.json", "heads_resolved_config"),
            (fold_root / "heads" / "inner_folds.json", "head_inner_folds"),
            (fold_root / "heads" / "slurm_provenance.json", "head_provenance"),
        ):
            if label == "best_model":
                _check_present(path, failures, f"fold_{fold}:{label}")
            else:
                _check_required(path, failures, f"fold_{fold}:{label}")
        for path, label, worker in (
            (train_fold_root / "slurm_provenance.json", "train", "src.merged.train"),
            (fold_root / "slurm_provenance.json", "postprocess", "src.merged.postprocess"),
            (fold_root / "heads" / "slurm_provenance.json", "heads", "src.merged.heads"),
        ):
            _audit_provenance(path, failures, f"fold_{fold}:{label}", worker)
        _audit_training_artifacts(
            train_fold_root,
            stage=stage,
            failures=failures,
            fold=fold,
        )
        fold_payload: dict[str, Any] = {"fold": fold, "root": str(fold_root), "train_root": str(train_fold_root)}
        if train_complete.is_file():
            train = read_json(train_complete)
            if train.get("status") != "completed":
                failures.append(f"training_not_completed:{fold}")
            fold_payload["selected_epoch"] = train.get("selected_epoch")
            identity = read_json(train_fold_root / "training_identity.json") if (train_fold_root / "training_identity.json").is_file() else {}
            if (
                identity.get("config_name") != config.get("name")
                or identity.get("stage") != stage
                or int(identity.get("fold", -1)) != fold
                or identity.get("run_id") != run_id
                or identity.get("manifest_hash") != protocol["manifest"].get("manifest_hash")
            ):
                failures.append(f"training_identity_mismatch:{fold}")
            if int(train.get("selected_epoch", 0)) < 1 or int(train.get("selected_epoch", 0)) > 20:
                failures.append(f"selected_epoch_out_of_range:{fold}")
            if identity.get("protocol_split_hash") != protocol["protocol"].get("split_hash"):
                failures.append(f"training_split_hash_mismatch:{fold}")
        if post_complete.is_file() and read_json(post_complete).get("status") != "completed":
            failures.append(f"postprocess_not_completed:{fold}")
        post_identity_path = fold_root / "postprocess_identity.json"
        if post_identity_path.is_file():
            post_identity = read_json(post_identity_path)
            if (
                post_identity.get("config_name") != config.get("name")
                or post_identity.get("modality") != config.get("modality")
                or post_identity.get("stage") != stage
                or int(post_identity.get("fold", -1)) != fold
                or post_identity.get("run_id") != run_id
                or post_identity.get("manifest_hash") != protocol["manifest"].get("manifest_hash")
            ):
                failures.append(f"postprocess_identity_mismatch:{fold}")
        if head_complete.is_file() and read_json(head_complete).get("status") != "completed":
            failures.append(f"heads_not_completed:{fold}")
        head_identity_path = fold_root / "heads" / "heads_identity.json"
        if head_identity_path.is_file():
            head_identity = read_json(head_identity_path)
            if (
                head_identity.get("stage") != stage
                or int(head_identity.get("fold", -1)) != fold
                or head_identity.get("run_id") != run_id
            ):
                failures.append(f"heads_identity_mismatch:{fold}")
        feature_metadata_path = fold_root / "features" / "feature_metadata.json"
        feature_subjects: dict[str, set[str]] = {}
        feature_samples: dict[str, set[str]] = {}
        if feature_metadata_path.is_file():
            feature_metadata = read_json(feature_metadata_path)
            if feature_metadata.get("manifest_hash") != protocol["manifest"]["manifest_hash"]:
                failures.append(f"feature_manifest_hash_mismatch:{fold}")
            if stage == "cv" and feature_metadata.get("split_hash") != protocol["protocol"]["split_hash"]:
                failures.append(f"feature_split_hash_mismatch:{fold}")
            expected_fold_hash = protocol["protocol"].get("folds", {}).get(str(fold), {}).get("fold_hash")
            if stage != "final" and feature_metadata.get("fold_hash") != expected_fold_hash:
                failures.append(f"feature_fold_hash_mismatch:{fold}")
            if feature_metadata.get("stage") != stage or int(feature_metadata.get("fold", -1)) != fold:
                failures.append(f"feature_identity_mismatch:{fold}")
            if feature_metadata.get("modality") != config.get("modality"):
                failures.append(f"feature_modality_mismatch:{fold}")
            if int(feature_metadata.get("feature_dimension", 0)) <= 0:
                failures.append(f"feature_dimension_invalid:{fold}")
            if feature_metadata.get("gold_label_protection", {}).get("labels_passed_to_model") is not False:
                failures.append(f"feature_label_protection_failed:{fold}")
            for partition in ("outer_train", "outer_holdout"):
                subjects, samples = _feature_subjects(
                    fold_root / "features" / f"{partition}_rows.jsonl",
                    feature_metadata,
                    partition,
                    failures,
                    fold,
                )
                feature_subjects[partition] = subjects
                feature_samples[partition] = samples
            train_subjects = feature_subjects.get("outer_train", set())
            holdout_subjects = feature_subjects.get("outer_holdout", set())
            if train_subjects & holdout_subjects:
                failures.append(f"feature_train_holdout_overlap:{fold}")
            daic_official = set(
                protocol["protocol"]["components"]["daic"].get(
                    "official_test_subject_ids", []
                )
            )
            if (train_subjects | holdout_subjects) & daic_official:
                failures.append(f"official_test_in_features:{fold}")
            observed_datasets = {
                value.split("::", 1)[0]
                for value in train_subjects | holdout_subjects
                if "::" in value
            }
            # Final features contain all non-test training datasets plus the
            # untouched DAIC official-test holdout, so their combined dataset
            # coverage is still the complete five-dataset protocol.
            expected_feature_datasets = set(DATASETS)
            if observed_datasets != expected_feature_datasets:
                failures.append(
                    f"feature_dataset_coverage:{fold}:found={sorted(observed_datasets)}"
                )
            if stage == "cv":
                expected_fold = protocol["protocol"]["folds"].get(str(fold), {})
                if train_subjects != set(expected_fold.get("outer_train_subject_ids", [])):
                    failures.append(f"feature_train_subject_mismatch:{fold}")
                if holdout_subjects != set(expected_fold.get("outer_holdout_subject_ids", [])):
                    failures.append(f"feature_holdout_subject_mismatch:{fold}")
            elif stage == "final":
                expected_train = set(protocol["final_partitions"].get("train_subject_ids", []))
                expected_test = set(protocol["final_partitions"].get("daic_official_test_subject_ids", []))
                if train_subjects != expected_train:
                    failures.append(f"final_feature_train_subject_mismatch:{fold}")
                if holdout_subjects != expected_test:
                    failures.append(f"final_feature_holdout_subject_mismatch:{fold}")
            fold_payload["feature_subject_counts"] = {
                partition: len(values) for partition, values in feature_subjects.items()
            }
            fold_payload["feature_dimension"] = feature_metadata.get("feature_dimension")
        qwen_summary_path = fold_root / "qwen" / "summary.json"
        if qwen_summary_path.is_file():
            qwen = read_json(qwen_summary_path)
            expected_datasets = {"daic"} if stage == "final" else set(DATASETS)
            if set(qwen) != expected_datasets:
                failures.append(f"qwen_dataset_coverage:{fold}:found={sorted(qwen)}")
            for dataset in sorted(expected_datasets):
                item = qwen.get(dataset, {})
                if int(item.get("sample_count", 0)) <= 0 or int(item.get("subject_count", 0)) <= 0:
                    failures.append(f"qwen_empty_predictions:{fold}:{dataset}")
                prediction_path = Path(item.get("output_dir", "")) / "predictions_subject_level.csv"
                _check_required(prediction_path, failures, f"fold_{fold}:qwen_predictions:{dataset}")
        heads_summary_path = fold_root / "heads" / "summary.json"
        if heads_summary_path.is_file():
            heads = read_json(heads_summary_path)
            if set(heads) != {"logreg", "xgb_fixed", "xgb_optuna"}:
                failures.append(f"head_method_coverage:{fold}:found={sorted(heads)}")
            for method in ("logreg", "xgb_fixed", "xgb_optuna"):
                method_dir = fold_root / "heads" / method
                for filename in ("classifier_metadata.json", "classifier.joblib", "metrics_by_dataset.json", "predictions_subject_level.jsonl"):
                    _check_required(method_dir / filename, failures, f"fold_{fold}:{method}:{filename}")
                metadata_path = method_dir / "classifier_metadata.json"
                if metadata_path.is_file() and feature_subjects:
                    classifier_metadata = read_json(metadata_path)
                    if set(classifier_metadata.get("training_subject_ids", [])) != feature_subjects.get("outer_train", set()):
                        failures.append(f"head_train_subject_mismatch:{fold}:{method}")
                    if set(classifier_metadata.get("holdout_subject_ids", [])) != feature_subjects.get("outer_holdout", set()):
                        failures.append(f"head_holdout_subject_mismatch:{fold}:{method}")
                    if int(classifier_metadata.get("input_dimension", -1)) != int(fold_payload.get("feature_dimension") or -1):
                        failures.append(f"head_feature_dimension_mismatch:{fold}:{method}")
            _audit_head_inner_folds(
                fold_root / "heads" / "inner_folds.json",
                fold_root / "features" / "outer_train_rows.jsonl",
                expected_subjects=feature_subjects.get("outer_train", set()),
                failures=failures,
                fold=fold,
            )
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
    if not registry_path.is_file():
        failures.append(f"missing:job_registry:{registry_path}")
    else:
        registry = read_json(registry_path)
        if not str(registry.get("source_commit", "")).strip():
            failures.append("job_registry_source_commit_missing")
        if not str(registry.get("plan_hash", "")).strip():
            failures.append("job_registry_plan_hash_missing")
        matching_jobs = [
            row for row in registry.get("jobs", [])
            if str(row.get("modality")) == str(config.get("modality"))
            and str(row.get("stage")) == stage
        ]
        expected_jobs = fold_count * 3
        if len(matching_jobs) != expected_jobs:
            failures.append(f"job_registry_coverage:found={len(matching_jobs)}:expected={expected_jobs}")
        failed_states = {"failed", "cancelled", "timeout", "oom", "out_of_memory", "node_fail", "preempted"}
        bad_jobs = []
        incomplete_jobs = []
        for row in matching_jobs:
            observed = str(row.get("observed_state", "")).upper().split(None, 1)[0]
            if observed in failed_states or (observed == "COMPLETED" and str(row.get("exit_code", "")) not in {"", "0:0"}):
                bad_jobs.append(row)
            if observed != "COMPLETED" or str(row.get("exit_code", "")) != "0:0":
                incomplete_jobs.append(row)
        if bad_jobs:
            failures.append(f"failed_registry_jobs:{len(bad_jobs)}")
        if incomplete_jobs:
            failures.append(f"incomplete_registry_jobs:{len(incomplete_jobs)}")
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
