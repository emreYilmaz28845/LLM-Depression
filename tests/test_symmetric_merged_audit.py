from __future__ import annotations

import json
from pathlib import Path

from src.merged.audit import _audit_head_inner_folds
from src.merged.protocol import canonical_sha256


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
