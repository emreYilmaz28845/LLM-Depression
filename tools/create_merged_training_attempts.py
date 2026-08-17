#!/usr/bin/env python3
"""Create modern attempt sidecars for symmetric-merged training folds.

The merged training workers predate the per-fold tracking sidecars. This tool
builds the standard sidecar set (metadata/status/jobs/artifacts/evaluations)
for every merged CV/final training fold of the Qwen and Gemma harmonized
runs, from the training evidence already on disk (training_identity.json,
resolved_merged_config.json, training_complete.json, slurm_provenance.json,
selected_checkpoint.json, the selection evaluation CSVs). The Optuna-100
merged studies then reference these attempts as their parents.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import canonical_sha256, read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.experiment_tracking.identity import evaluation_id, new_attempt_id  # noqa: E402
from src.experiment_tracking.canonical import format_utc_timestamp  # noqa: E402
from src.experiment_tracking.lifecycle import StatusRecord, append_job_event, new_job_event, write_status  # noqa: E402

RUNS = {
    "qwen": {
        "root": "output_model/symmetric_merged/harmonized_v1",
        "run": "harmonized_v1_prod_20260809T171705Z_d1e8130b",
        "prefix": "harmonized_v1_merged",
        "group": "harmonized-v1-merged",
    },
    "gemma4": {
        "root": "output_model/symmetric_merged/gemma4/harmonized_v1",
        "run": "gemma4_merged_v1_prod_20260816T0000Z_d4ff33e",
        "prefix": "gemma4_harmonized_v1_merged",
        "group": "gemma4-harmonized-v1-merged",
    },
}
MODALITIES = ("audio_only", "audio_text", "text_only")
METRIC_NAMES = ("macro_f1", "positive_f1", "accuracy", "precision", "recall")
INVALID_FINAL_TF_WARNING = (
    "invalidated by merged-final evidence correction: this record referenced "
    "Logistic Regression artifacts while declaring original_teacher_forced"
)


def _commit_for(fold_dir: Path) -> str:
    proven = read_json(fold_dir / "slurm_provenance.json")
    commit = str(proven.get("source_commit") or "")
    if len(commit) == 40:
        return commit
    return (PROJECT_ROOT / ".provenance/git_commit.txt").read_text().strip()


def _selection_predictions(fold_dir: Path, selected_epoch: int) -> Path | None:
    """Concatenate the per-dataset selection predictions into one CSV."""
    selection_root = fold_dir / "logs/selection"
    if not (selection_root / f"epoch_{selected_epoch}").is_dir():
        return None
    parts: list[tuple[str, dict]] = []
    fieldnames: set[str] = {"dataset"}
    for dataset_dir in sorted((selection_root / f"epoch_{selected_epoch}").iterdir()):
        csv_path = dataset_dir / "predictions_subject_level.csv"
        if not csv_path.is_file():
            continue
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        if not rows:
            continue
        fieldnames.update(rows[0].keys())
        for row in rows:
            row["dataset"] = dataset_dir.name
            parts.append((dataset_dir.name, row))
    if not parts:
        return None
    out = fold_dir / "logs/selection/combined_subject_predictions.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(fieldnames))
        writer.writeheader()
        for _, row in parts:
            writer.writerow(row)
    return out


def _selection_metrics(fold_dir: Path, selected_epoch: int) -> dict:
    history = read_json(fold_dir / "logs/training_history.json")
    entry = next((item for item in history if int(item.get("epoch")) == selected_epoch), None)
    if entry is None:
        return {}
    components = entry.get("component_selection_metrics") or {}
    if not components:
        return {}
    def mean(key: str) -> float:
        values = [float(c[key]) for c in components.values() if c.get(key) is not None]
        return sum(values) / len(values) if values else 0.0
    support = sum(int(c.get("support") or len(c.get("subject_ids") or [])) for c in components.values())
    return {"macro_f1": mean("macro_f1"), "positive_f1": mean("positive_f1"),
            "accuracy": mean("accuracy"), "precision": mean("precision"),
            "recall": mean("recall"), "support": support}


def _copy_immutable(source: Path, destination: Path) -> None:
    """Copy an evidence file without replacing different existing content."""
    if not source.is_file():
        raise FileNotFoundError(f"required merged postprocess evidence is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise ValueError(f"refusing to overwrite mismatched evidence: {destination}")
        return
    shutil.copy2(source, destination)


def _final_teacher_forced_evidence(
    fold_dir: Path, *, backend: str
) -> tuple[dict, Path, Path]:
    """Return the real final DAIC teacher-forced metrics and predictions.

    Final teacher-forced evaluation is produced by ``src.merged.postprocess``.
    It is separate from the hidden-state Logistic Regression head.
    """
    output_root = PROJECT_ROOT / "outputs" / fold_dir.relative_to(PROJECT_ROOT / "output_model")
    backend_dir = "gemma4" if backend == "gemma4" else "qwen"
    evaluation_root = output_root / backend_dir / "daic"
    metrics_path = evaluation_root / "metrics_original_teacher_forced.json"
    predictions_path = evaluation_root / "predictions_subject_level.csv"
    metrics_payload = read_json(metrics_path)
    if metrics_payload.get("prediction_backend") != "original_teacher_forced":
        raise ValueError(f"wrong merged-final prediction backend in {metrics_path}")
    if metrics_payload.get("evaluation_view") != "harmonized_all_windows_full_coverage":
        raise ValueError(f"wrong merged-final evaluation view in {metrics_path}")
    if metrics_payload.get("aggregation_level") != "subject":
        raise ValueError(f"wrong merged-final aggregation in {metrics_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"required merged postprocess predictions are missing: {predictions_path}")
    return metrics_payload, metrics_path, predictions_path


def _headline_metrics(payload: dict) -> dict:
    support = int(payload.get("num_subjects") or 0)
    return {
        "macro_f1": payload.get("binary_strict_macro_f1", payload.get("macro_f1")),
        "positive_f1": payload.get("binary_strict_positive_f1", payload.get("positive_f1")),
        "accuracy": payload.get("binary_strict_accuracy", payload.get("accuracy")),
        "precision": payload.get("binary_strict_precision", payload.get("precision")),
        "recall": payload.get("binary_strict_recall", payload.get("recall")),
        "support": support,
    }


def _evaluation_record(
    *, attempt_id: str, fold: int, metrics: dict, metrics_relative: str,
    predictions_relative: str, locally_verified: bool,
) -> dict:
    eval_id = evaluation_id(
        attempt_id=attempt_id, fold=fold, dataset="daic", split_name="test",
        split_protocol="daic_official_train_fit_locked_test_evaluation",
        checkpoint_role="best_model", checkpoint_path="best_model",
        backend="original_teacher_forced",
        evaluation_view="harmonized_all_windows_full_coverage",
        aggregation="subject_level", metric_namespace="headline/binary_strict",
        metrics_artifact_sha256=canonical_sha256(metrics),
    )
    return {
        "evaluation_id": eval_id,
        "dataset": "daic",
        "split_name": "test",
        "split_protocol": "daic_official_train_fit_locked_test_evaluation",
        "checkpoint_role": "best_model",
        "checkpoint_path": "best_model",
        "backend": "original_teacher_forced",
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "subject_level",
        "metric_namespace": "headline/binary_strict",
        "metrics_artifact_path": metrics_relative,
        "predictions_artifact_path": predictions_relative,
        "metrics": [
            {"name": name, "value": metrics.get(name), "support": metrics.get("support")}
            for name in METRIC_NAMES
        ],
        "locally_verified": locally_verified,
        "reportable": locally_verified,
        "warnings": [],
    }


def build_attempt(fold_dir: Path, *, backend: str, modality: str, stage: str, fold: int) -> Path:
    meta = read_json(fold_dir / "training_identity.json")
    config = read_json(fold_dir / "resolved_merged_config.json")
    complete = read_json(fold_dir / "training_complete.json")
    selected = read_json(fold_dir / "logs/selected_checkpoint.json")
    commit = _commit_for(fold_dir)
    run = RUNS[backend]
    logical = f"{run['prefix']}_{modality}_{stage}_seed1337"
    attempt_id = new_attempt_id(logical, commit)
    fold_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": "audiollm.metadata.v1",
        "group_id": f"{run['group']}-{run['run']}",
        "logical_run_name": logical,
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": 1337,
        "created_at_utc": format_utc_timestamp(utc_now()),
        "source": {
            "git_commit": commit,
            "git_branch": "main",
            "git_dirty": False,
            "deployed_source_sha256": None,
        },
        "research": {"github_issue": 60, "github_pr": 52},
        "hashes": {
            "resolved_config_sha256": meta.get("merged_config_sha256"),
            "manifest_sha256": meta.get("manifest_hash"),
            "split_sha256": meta.get("protocol_split_hash"),
        },
        "paths": {"run_config": "resolved_merged_config.json", "best_model": "best_model", "local_evidence_root": None},
        "parent": None,
        "wandb": {"project": "audiollm-depression", "entity": None, "run_id": None, "url": None, "sync_status": "NOT_EXPORTED"},
    }
    write_json_atomic(fold_dir / "metadata.json", metadata)

    record = StatusRecord(attempt_id=attempt_id, fold=fold)
    for to_state, reason in (
        ("DEPLOYED", "merged training fold deployed"),
        ("SUBMITTED", "merged training submitted to Slurm"),
        ("RUNNING", "merged training started"),
        ("COMPLETED_ON_MN5", "merged training completed"),
    ):
        record.transition(to_state, reason=reason)
    write_status(fold_dir / "status.json", record)

    slurm = read_json(fold_dir / "slurm_provenance.json")
    job_id = str(slurm.get("scheduler", {}).get("SLURM_JOB_ID") or "")
    events = [
        new_job_event(job_key="train", job_type="train", event_type="SUBMITTED",
                      attempt_id=attempt_id, fold=fold, slurm_job_id=job_id,
                      status="PENDING", reason="merged training submitted"),
        new_job_event(job_key="train", job_type="train", event_type="STARTED",
                      attempt_id=attempt_id, fold=fold, slurm_job_id=job_id,
                      status="RUNNING", reason="merged training started"),
        new_job_event(job_key="train", job_type="train", event_type="COMPLETED",
                      attempt_id=attempt_id, fold=fold, slurm_job_id=job_id,
                      status="COMPLETED", reason="merged training completed"),
    ]
    (fold_dir / "jobs.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n", encoding="utf-8")

    artifacts = [
        {"artifact_id": "art-" + canonical_sha256({"p": "resolved_merged_config.json", "a": attempt_id})[:24],
         "artifact_type": "run_config", "role": "resolved_merged_config", "path": "resolved_merged_config.json",
         "sha256": sha256_file(fold_dir / "resolved_merged_config.json"), "size_bytes": (fold_dir / "resolved_merged_config.json").stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
        {"artifact_id": "art-" + canonical_sha256({"p": "training_identity.json", "a": attempt_id})[:24],
         "artifact_type": "report", "role": "training_identity", "path": "training_identity.json",
         "sha256": sha256_file(fold_dir / "training_identity.json"), "size_bytes": (fold_dir / "training_identity.json").stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
        {"artifact_id": "art-" + canonical_sha256({"p": "training_complete.json", "a": attempt_id})[:24],
         "artifact_type": "summary", "role": "training_complete", "path": "training_complete.json",
         "sha256": sha256_file(fold_dir / "training_complete.json"), "size_bytes": (fold_dir / "training_complete.json").stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
        {"artifact_id": "art-" + canonical_sha256({"p": "slurm_provenance.json", "a": attempt_id})[:24],
         "artifact_type": "audit", "role": "slurm_provenance", "path": "slurm_provenance.json",
         "sha256": sha256_file(fold_dir / "slurm_provenance.json"), "size_bytes": (fold_dir / "slurm_provenance.json").stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
        {"artifact_id": "art-" + canonical_sha256({"p": "logs/selected_checkpoint.json", "a": attempt_id})[:24],
         "artifact_type": "summary", "role": "selected_checkpoint", "path": "logs/selected_checkpoint.json",
         "sha256": sha256_file(fold_dir / "logs/selected_checkpoint.json"), "size_bytes": (fold_dir / "logs/selected_checkpoint.json").stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
        {"artifact_id": "art-" + canonical_sha256({"p": "best_model", "a": attempt_id})[:24],
         "artifact_type": "checkpoint", "role": "best_model", "path": "best_model",
         "size_bytes": None, "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
    ]
    write_json_atomic(fold_dir / "artifacts.json", {
        "schema_version": "audiollm.artifacts.v1", "attempt_id": attempt_id, "fold": fold, "artifacts": artifacts,
    })

    selected_epoch = int(selected.get("selected_epoch") or 0)
    if stage == "final":
        dataset, split_name, split_protocol = "daic", "test", "daic_official_train_fit_locked_test_evaluation"
    else:
        dataset, split_name, split_protocol = "merged", "outer_holdout", "symmetric_merged_cv_outer_holdout"
    metrics = _selection_metrics(fold_dir, selected_epoch)
    predictions_path = _selection_predictions(fold_dir, selected_epoch)
    if stage == "final":
        tf_payload, source_metrics, source_predictions = _final_teacher_forced_evidence(
            fold_dir, backend=backend
        )
        metrics = _headline_metrics(tf_payload)
        metrics_relative = "logs/postprocess/final_daic_metrics_original_teacher_forced.json"
        predictions_relative = "logs/postprocess/final_daic_predictions_subject_level.csv"
        _copy_immutable(source_metrics, fold_dir / metrics_relative)
        _copy_immutable(source_predictions, fold_dir / predictions_relative)
        predictions_path = fold_dir / predictions_relative
        record = _evaluation_record(
            attempt_id=attempt_id,
            fold=fold,
            metrics=metrics,
            metrics_relative=metrics_relative,
            predictions_relative=predictions_relative,
            locally_verified=False,
        )
    else:
        metrics_relative = "logs/selection/combined_selection_metrics.json"
        predictions_relative = "logs/selection/combined_subject_predictions.csv"
        eval_id = evaluation_id(
            attempt_id=attempt_id, fold=fold, dataset=dataset, split_name=split_name,
            split_protocol=split_protocol, checkpoint_role="best_model", checkpoint_path="best_model",
            backend="original_teacher_forced", evaluation_view="harmonized_all_windows_full_coverage",
            aggregation="subject_level", metric_namespace="headline/binary_strict",
            metrics_artifact_sha256=canonical_sha256(metrics),
        )
        record = {
            "evaluation_id": eval_id,
            "dataset": dataset,
            "split_name": split_name,
            "split_protocol": split_protocol,
            "checkpoint_role": "best_model",
            "checkpoint_path": "best_model",
            "backend": "original_teacher_forced",
            "evaluation_view": "harmonized_all_windows_full_coverage",
            "aggregation": "subject_level",
            "metric_namespace": "headline/binary_strict",
            "metrics_artifact_path": metrics_relative,
            "predictions_artifact_path": predictions_relative,
            "metrics": [{"name": name, "value": metrics.get(name), "support": metrics.get("support")} for name in METRIC_NAMES],
            "locally_verified": False,
            "reportable": False,
            "warnings": [],
        }
        metrics_path = fold_dir / "logs/selection/combined_selection_metrics.json"
        write_json_atomic(metrics_path, metrics)
    combined_artifacts = [
        {"artifact_id": "art-" + canonical_sha256({"p": metrics_relative, "a": attempt_id})[:24],
         "artifact_type": "metrics", "role": "selection_metrics", "path": metrics_relative,
         "sha256": sha256_file(fold_dir / metrics_relative), "size_bytes": (fold_dir / metrics_relative).stat().st_size,
         "exists_on_mn5": True, "exists_locally": True, "locally_verified": False},
    ]
    if predictions_path is not None:
        combined_artifacts.append(
            {"artifact_id": "art-" + canonical_sha256({"p": predictions_relative, "a": attempt_id})[:24],
             "artifact_type": "predictions", "role": "selection_predictions", "path": predictions_relative,
             "sha256": sha256_file(predictions_path), "size_bytes": predictions_path.stat().st_size,
             "exists_on_mn5": True, "exists_locally": True, "locally_verified": False}
        )
    artifacts.extend(combined_artifacts)
    write_json_atomic(fold_dir / "artifacts.json", {
        "schema_version": "audiollm.artifacts.v1", "attempt_id": attempt_id, "fold": fold, "artifacts": artifacts,
    })
    write_json_atomic(fold_dir / "evaluations.json", {
        "schema_version": "audiollm.evaluations.v1", "attempt_id": attempt_id, "fold": fold,
        "evaluations": [record] if predictions_path is not None else [],
    })
    return fold_dir


def repair_final_teacher_forced_evaluation(
    fold_dir: Path, *, backend: str, modality: str, fold: int = 0
) -> dict:
    """Invalidate the historical LR-as-TF record and append the real TF one.

    The original metric and prediction files remain untouched. The correction
    copies the already-produced postprocess evidence under new paths and keeps
    an audit record beside the tracking sidecars.
    """
    evaluations_path = fold_dir / "evaluations.json"
    artifacts_path = fold_dir / "artifacts.json"
    evaluations = read_json(evaluations_path)
    artifacts = read_json(artifacts_path)
    attempt_id = str(evaluations["attempt_id"])
    if int(evaluations.get("fold")) != fold or int(artifacts.get("fold")) != fold:
        raise ValueError(f"fold identity mismatch in {fold_dir}")
    if artifacts.get("attempt_id") != attempt_id:
        raise ValueError(f"attempt identity mismatch in {fold_dir}")

    old_metrics_relative = "logs/selection/final_daic_metrics_by_dataset.json"
    old_predictions_relative = "logs/selection/final_daic_predictions.csv"
    bad_records = [
        record for record in evaluations.get("evaluations", [])
        if record.get("backend") == "original_teacher_forced"
        and record.get("metrics_artifact_path") == old_metrics_relative
        and record.get("predictions_artifact_path") == old_predictions_relative
    ]
    if len(bad_records) != 1:
        raise ValueError(
            f"expected exactly one historical LR-as-TF record in {fold_dir}; found {len(bad_records)}"
        )

    output_root = PROJECT_ROOT / "outputs" / fold_dir.relative_to(PROJECT_ROOT / "output_model")
    logreg_root = output_root / "heads/logreg"
    if sha256_file(fold_dir / old_metrics_relative) != sha256_file(logreg_root / "metrics_by_dataset.json"):
        raise ValueError(f"historical metrics are not the expected Logistic Regression copy: {fold_dir}")
    if sha256_file(fold_dir / old_predictions_relative) != sha256_file(logreg_root / "predictions_subject_level.csv"):
        raise ValueError(f"historical predictions are not the expected Logistic Regression copy: {fold_dir}")

    tf_payload, source_metrics, source_predictions = _final_teacher_forced_evidence(
        fold_dir, backend=backend
    )
    metrics = _headline_metrics(tf_payload)
    metrics_relative = "logs/postprocess/final_daic_metrics_original_teacher_forced.json"
    predictions_relative = "logs/postprocess/final_daic_predictions_subject_level.csv"
    destination_metrics = fold_dir / metrics_relative
    destination_predictions = fold_dir / predictions_relative
    _copy_immutable(source_metrics, destination_metrics)
    _copy_immutable(source_predictions, destination_predictions)

    corrected = _evaluation_record(
        attempt_id=attempt_id,
        fold=fold,
        metrics=metrics,
        metrics_relative=metrics_relative,
        predictions_relative=predictions_relative,
        locally_verified=True,
    )
    existing_corrected = [
        record for record in evaluations.get("evaluations", [])
        if record.get("evaluation_id") == corrected["evaluation_id"]
    ]
    if existing_corrected and existing_corrected != [corrected]:
        raise ValueError(f"mismatched corrected evaluation already exists in {fold_dir}")

    bad_record = bad_records[0]
    bad_record["locally_verified"] = False
    bad_record["reportable"] = False
    warnings = list(bad_record.get("warnings") or [])
    if INVALID_FINAL_TF_WARNING not in warnings:
        warnings.append(INVALID_FINAL_TF_WARNING)
    bad_record["warnings"] = warnings
    if not existing_corrected:
        evaluations["evaluations"].append(corrected)

    artifact_by_path = {item["path"]: item for item in artifacts.get("artifacts", [])}
    for relative, path, role in (
        (metrics_relative, destination_metrics, "teacher_forced_metrics"),
        (predictions_relative, destination_predictions, "teacher_forced_predictions"),
    ):
        expected = {
            "artifact_id": "art-" + canonical_sha256({"p": relative, "a": attempt_id})[:24],
            "artifact_type": "metrics" if role.endswith("metrics") else "predictions",
            "role": role,
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "exists_on_mn5": False,
            "exists_locally": True,
            "locally_verified": True,
        }
        existing = artifact_by_path.get(relative)
        if existing is not None and existing != expected:
            raise ValueError(f"mismatched corrected artifact record already exists: {relative}")
        if existing is None:
            artifacts["artifacts"].append(expected)

    postprocess_slurm = read_json(output_root / "slurm_provenance.json")
    audit_relative = "logs/postprocess/final_teacher_forced_evidence_correction.json"
    audit_path = fold_dir / audit_relative
    audit = {
        "schema_version": "symmetric_merged_final_tf_correction.v1",
        "attempt_id": attempt_id,
        "fold": fold,
        "backend": backend,
        "modality": modality,
        "reason": INVALID_FINAL_TF_WARNING,
        "invalidated_evaluation_id": bad_record["evaluation_id"],
        "corrected_evaluation_id": corrected["evaluation_id"],
        "source_metrics_path": str(source_metrics.relative_to(PROJECT_ROOT)),
        "source_metrics_sha256": sha256_file(source_metrics),
        "source_predictions_path": str(source_predictions.relative_to(PROJECT_ROOT)),
        "source_predictions_sha256": sha256_file(source_predictions),
        "postprocess_slurm_job_id": str(postprocess_slurm.get("scheduler", {}).get("SLURM_JOB_ID") or ""),
    }
    if audit_path.exists() and read_json(audit_path) != audit:
        raise ValueError(f"refusing to overwrite mismatched correction audit: {audit_path}")
    if not audit_path.exists():
        write_json_atomic(audit_path, audit)
    audit_artifact = {
        "artifact_id": "art-" + canonical_sha256({"p": audit_relative, "a": attempt_id})[:24],
        "artifact_type": "audit",
        "role": "teacher_forced_evidence_correction",
        "path": audit_relative,
        "sha256": sha256_file(audit_path),
        "size_bytes": audit_path.stat().st_size,
        "exists_on_mn5": False,
        "exists_locally": True,
        "locally_verified": True,
    }
    existing_audit = artifact_by_path.get(audit_relative)
    if existing_audit is not None and existing_audit != audit_artifact:
        raise ValueError(f"mismatched correction audit artifact already exists in {fold_dir}")
    if existing_audit is None:
        artifacts["artifacts"].append(audit_artifact)

    write_json_atomic(artifacts_path, artifacts)
    write_json_atomic(evaluations_path, evaluations)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="qwen,gemma4")
    parser.add_argument("--stages", default="cv,final")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--repair-final-evaluations",
        action="store_true",
        help="non-destructively invalidate historical LR-as-TF records and append real postprocess TF evidence",
    )
    args = parser.parse_args()
    backends = args.backends.split(",")
    stages = args.stages.split(",")
    folds = [int(x) for x in args.folds.split(",")]
    created = 0
    for backend in backends:
        run = RUNS[backend]
        root = PROJECT_ROOT / run["root"]
        for modality in MODALITIES:
            for stage in stages:
                fold_list = folds if stage == "cv" else [0]
                for fold in fold_list:
                    fold_dir = root / modality / run["run"] / stage / f"fold_{fold}"
                    if not (fold_dir / "training_complete.json").is_file():
                        print(f"SKIP missing training: {fold_dir}", file=sys.stderr)
                        continue
                    if args.repair_final_evaluations:
                        if stage != "final":
                            continue
                        repair_final_teacher_forced_evaluation(
                            fold_dir, backend=backend, modality=modality, fold=fold
                        )
                    else:
                        build_attempt(fold_dir, backend=backend, modality=modality, stage=stage, fold=fold)
                    created += 1
    action = "final teacher-forced evaluations repaired" if args.repair_final_evaluations else "merged training attempts created"
    print(f"{action}: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
