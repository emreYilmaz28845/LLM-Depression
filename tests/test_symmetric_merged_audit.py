from __future__ import annotations

import json
from pathlib import Path

from src.merged.audit import (
    _audit_feature_test_protection,
    _audit_head_inner_folds,
    _resolve_qwen_prediction_path,
    _audit_training_artifacts,
)
from src.merged.protocol import DATASETS, canonical_sha256


def _write_inner_artifacts(root: Path) -> tuple[Path, Path, set[str]]:
    rows = [
        {"subject_id": f"dataset::{index}", "label": index % 2}
        for index in range(6)
    ]
    assignments = [
        {"fold": 0, "train_row_indices": [2, 3, 4, 5], "validation_row_indices": [0, 1]},
        {"fold": 1, "train_row_indices": [0, 1, 4, 5], "validation_row_indices": [2, 3]},
        {"fold": 2, "train_row_indices": [0, 1, 2, 3], "validation_row_indices": [4, 5]},
    ]
    for item in assignments:
        item["train_subject_ids"] = [rows[index]["subject_id"] for index in item["train_row_indices"]]
        item["validation_subject_ids"] = [rows[index]["subject_id"] for index in item["validation_row_indices"]]
    payload = {"schema_version": "test", "inner_folds": 3, "seed": 1337, "folds": assignments}
    payload["assignments_hash"] = canonical_sha256(payload)
    inner_path = root / "inner_folds.json"
    rows_path = root / "outer_train_rows.jsonl"
    inner_path.write_text(json.dumps(payload), encoding="utf-8")
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return inner_path, rows_path, {row["subject_id"] for row in rows}


def test_head_inner_fold_audit_checks_exact_grouped_coverage(tmp_path: Path) -> None:
    inner_path, rows_path, subjects = _write_inner_artifacts(tmp_path)
    failures: list[str] = []
    _audit_head_inner_folds(
        inner_path,
        rows_path,
        expected_subjects=subjects,
        failures=failures,
        fold=0,
    )
    assert failures == []

    payload = json.loads(inner_path.read_text(encoding="utf-8"))
    payload["folds"][1]["validation_row_indices"] = [0, 3]
    payload["folds"][1]["validation_subject_ids"] = ["dataset::0", "dataset::3"]
    payload["assignments_hash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "assignments_hash"}
    )
    inner_path.write_text(json.dumps(payload), encoding="utf-8")
    failures = []
    _audit_head_inner_folds(
        inner_path,
        rows_path,
        expected_subjects=subjects,
        failures=failures,
        fold=0,
    )
    assert "head_inner_validation_row_coverage:0" in failures


def test_training_artifact_audit_checks_schedule_and_selection_mean(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "weighting_audit.json").write_text(
        json.dumps(
            {
                "datasets": list(DATASETS),
                "equal_dataset_totals": True,
                "natural_class_prevalence_preserved": True,
                "no_sampling": True,
                "no_duplication": True,
                "mean_loss_weight": 1.0,
            }
        ),
        encoding="utf-8",
    )
    schedule = {
        "epoch": 1,
        "example_count": 4,
        "sample_occurrence_counts": {str(index): 1 for index in range(4)},
        "blocks": [
            {"example_indices": [0, 1]},
            {"example_indices": [2, 3]},
        ],
    }
    (logs / "schedule_audit.json").write_text(json.dumps({"epochs": [schedule]}), encoding="utf-8")
    metrics = {dataset: {"macro_f1": 0.5} for dataset in DATASETS}
    (logs / "training_history.json").write_text(
        json.dumps([{"epoch": 1, "component_selection_metrics": metrics, "mean_dataset_macro_f1": 0.5}]),
        encoding="utf-8",
    )
    failures: list[str] = []
    _audit_training_artifacts(tmp_path, stage="cv", failures=failures, fold=0)
    assert failures == []

    history = json.loads((logs / "training_history.json").read_text(encoding="utf-8"))
    history[0]["mean_dataset_macro_f1"] = 0.4
    (logs / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    failures = []
    _audit_training_artifacts(tmp_path, stage="cv", failures=failures, fold=0)
    assert "selection_mean_mismatch:0:1" in failures


def test_final_feature_audit_allows_official_test_only_in_holdout() -> None:
    official = {"daic::999"}
    failures: list[str] = []
    _audit_feature_test_protection(
        stage="final",
        train_subjects={"daic::1", "cmdc::1"},
        holdout_subjects=official,
        daic_official_subjects=official,
        failures=failures,
        fold=0,
    )
    assert failures == []

    failures = []
    _audit_feature_test_protection(
        stage="final",
        train_subjects=official,
        holdout_subjects=set(),
        daic_official_subjects=official,
        failures=failures,
        fold=0,
    )
    assert failures == ["official_test_in_final_training_features:0"]


def test_compact_audit_uses_local_qwen_prediction_path(tmp_path: Path) -> None:
    fold_root = tmp_path / "fold_0"
    local_prediction = fold_root / "qwen" / "daic" / "predictions_subject_level.csv"
    local_prediction.parent.mkdir(parents=True)
    local_prediction.write_text("subject_id,label,prediction\n", encoding="utf-8")

    resolved = _resolve_qwen_prediction_path(
        fold_root, "daic", "/gpfs/projects/etur92/remote/qwen/daic"
    )
    assert resolved == local_prediction
