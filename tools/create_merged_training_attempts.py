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
import sys
from datetime import datetime, timezone
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
        heads_root = PROJECT_ROOT / "outputs" / fold_dir.relative_to(PROJECT_ROOT / "output_model") / "heads"
        metrics_by_dataset = read_json(heads_root / "logreg/metrics_by_dataset.json")
        daic_metrics = metrics_by_dataset.get("daic") or {}
        metrics = {
            "macro_f1": daic_metrics.get("macro_f1"),
            "positive_f1": daic_metrics.get("positive_f1"),
            "accuracy": daic_metrics.get("accuracy"),
            "precision": daic_metrics.get("precision") or daic_metrics.get("macro_precision"),
            "recall": daic_metrics.get("recall") or daic_metrics.get("macro_recall"),
            "support": int(daic_metrics.get("support_negative", 0) + daic_metrics.get("support_positive", 0)),
        }
        import shutil
        shutil.copy2(heads_root / "logreg/predictions_subject_level.csv", fold_dir / "logs/selection/final_daic_predictions.csv")
        shutil.copy2(heads_root / "logreg/metrics_by_dataset.json", fold_dir / "logs/selection/final_daic_metrics_by_dataset.json")
        predictions_path = fold_dir / "logs/selection/final_daic_predictions.csv"
    eval_id = evaluation_id(
        attempt_id=attempt_id, fold=fold, dataset=dataset, split_name=split_name,
        split_protocol=split_protocol, checkpoint_role="best_model", checkpoint_path="best_model",
        backend="original_teacher_forced", evaluation_view="harmonized_all_windows_full_coverage",
        aggregation="subject_level", metric_namespace="headline/binary_strict",
        metrics_artifact_sha256=canonical_sha256(metrics),
    )
    if stage == "final":
        metrics_relative = "logs/selection/final_daic_metrics_by_dataset.json"
        predictions_relative = "logs/selection/final_daic_predictions.csv"
    else:
        metrics_relative = "logs/selection/combined_selection_metrics.json"
        predictions_relative = "logs/selection/combined_subject_predictions.csv"
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
    if stage == "final":
        metrics_path = fold_dir / "logs/selection/final_daic_metrics_by_dataset.json"
        write_json_atomic(metrics_path, read_json(heads_root / "logreg/metrics_by_dataset.json"))
    else:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="qwen,gemma4")
    parser.add_argument("--stages", default="cv,final")
    parser.add_argument("--folds", default="0,1,2,3,4")
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
                    build_attempt(fold_dir, backend=backend, modality=modality, stage=stage, fold=fold)
                    created += 1
    print(f"merged training attempts created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
