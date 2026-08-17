from __future__ import annotations

import json
from pathlib import Path

from src.experiment_tracking.canonical import sha256_file
from tools import create_merged_training_attempts as merged_attempts


ATTEMPT_ID = "20260817T111507Z-gemma4_merged_final_seed1337-bfc13b4f-1234abcd"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_repair_final_teacher_forced_evaluation_preserves_lr_and_appends_real_tf(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(merged_attempts, "PROJECT_ROOT", tmp_path)
    fold_dir = (
        tmp_path / "output_model/symmetric_merged/gemma4/harmonized_v1/audio_text/"
        "campaign/final/fold_0"
    )
    output_dir = (
        tmp_path / "outputs/symmetric_merged/gemma4/harmonized_v1/audio_text/"
        "campaign/final/fold_0"
    )

    lr_metrics = {"daic": {"macro_f1": 0.78, "positive_f1": 0.70}}
    lr_predictions = "dataset,subject_id,label,prediction\ndaic,s1,0,0\n"
    old_metrics = fold_dir / "logs/selection/final_daic_metrics_by_dataset.json"
    old_predictions = fold_dir / "logs/selection/final_daic_predictions.csv"
    _write_json(old_metrics, lr_metrics)
    old_predictions.write_text(lr_predictions, encoding="utf-8")
    _write_json(output_dir / "heads/logreg/metrics_by_dataset.json", lr_metrics)
    (output_dir / "heads/logreg/predictions_subject_level.csv").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "heads/logreg/predictions_subject_level.csv").write_text(lr_predictions, encoding="utf-8")

    tf_metrics = {
        "macro_f1": 0.72,
        "positive_f1": 0.62,
        "accuracy": 0.76,
        "precision": 0.60,
        "recall": 0.64,
        "binary_strict_macro_f1": 0.725,
        "binary_strict_positive_f1": 0.625,
        "binary_strict_accuracy": 0.765,
        "binary_strict_precision": 0.61,
        "binary_strict_recall": 0.64,
        "num_subjects": 47,
        "prediction_backend": "original_teacher_forced",
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation_level": "subject",
    }
    tf_root = output_dir / "gemma4/daic"
    _write_json(tf_root / "metrics_original_teacher_forced.json", tf_metrics)
    (tf_root / "predictions_subject_level.csv").write_text(
        "subject_id,label,prediction\ns1,0,0\n", encoding="utf-8"
    )
    _write_json(output_dir / "slurm_provenance.json", {"scheduler": {"SLURM_JOB_ID": "44684476"}})

    bad_evaluation = {
        "evaluation_id": "eval-111111111111111111111111",
        "dataset": "daic",
        "split_name": "test",
        "split_protocol": "daic_official_train_fit_locked_test_evaluation",
        "checkpoint_role": "best_model",
        "checkpoint_path": "best_model",
        "backend": "original_teacher_forced",
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "subject_level",
        "metric_namespace": "headline/binary_strict",
        "metrics_artifact_path": "logs/selection/final_daic_metrics_by_dataset.json",
        "predictions_artifact_path": "logs/selection/final_daic_predictions.csv",
        "metrics": [],
        "locally_verified": True,
        "reportable": True,
        "warnings": [],
    }
    _write_json(fold_dir / "evaluations.json", {
        "schema_version": "audiollm.evaluations.v1",
        "attempt_id": ATTEMPT_ID,
        "fold": 0,
        "evaluations": [bad_evaluation],
    })
    _write_json(fold_dir / "artifacts.json", {
        "schema_version": "audiollm.artifacts.v1",
        "attempt_id": ATTEMPT_ID,
        "fold": 0,
        "artifacts": [],
    })

    audit = merged_attempts.repair_final_teacher_forced_evaluation(
        fold_dir, backend="gemma4", modality="audio_text"
    )
    assert audit["postprocess_slurm_job_id"] == "44684476"
    corrected_evaluations = json.loads((fold_dir / "evaluations.json").read_text())["evaluations"]
    assert len(corrected_evaluations) == 2
    invalidated, corrected = corrected_evaluations
    assert invalidated["reportable"] is False
    assert invalidated["locally_verified"] is False
    assert merged_attempts.INVALID_FINAL_TF_WARNING in invalidated["warnings"]
    assert corrected["reportable"] is True
    assert corrected["dataset"] == "daic"
    assert corrected["metrics_artifact_path"].startswith("logs/postprocess/")
    values = {metric["name"]: metric["value"] for metric in corrected["metrics"]}
    assert values["macro_f1"] == 0.725
    assert values["positive_f1"] == 0.625
    assert sha256_file(old_metrics) == sha256_file(output_dir / "heads/logreg/metrics_by_dataset.json")

    # The correction is idempotent and never creates duplicate evaluations.
    merged_attempts.repair_final_teacher_forced_evaluation(
        fold_dir, backend="gemma4", modality="audio_text"
    )
    corrected_evaluations = json.loads((fold_dir / "evaluations.json").read_text())["evaluations"]
    assert len(corrected_evaluations) == 2
