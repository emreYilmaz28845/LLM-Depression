from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook

from scripts.build_clean_workbook import build_turkish_question_condition
from tools.turkish_question_condition_report import _metric, build_tables


def test_locked_report_table_cardinality_and_seed_aggregation() -> None:
    rows = []
    inputs = (
        ("audio_only", "not_applicable"),
        ("text_only", "native"),
        ("text_only", "english"),
        ("audio_text", "native"),
        ("audio_text", "english"),
    )
    for condition in ("pos_only", "negative_only"):
        for model in ("qwen", "gemma4"):
            for modality, transcript in inputs:
                for route in ("teacher_forced", "logreg", "xgb_optuna100"):
                    for seed in (7, 1337, 2024):
                        for fold in range(5):
                            rows.append(
                                {
                                    "condition": condition,
                                    "model": "Gemma 4" if model == "gemma4" else "Qwen",
                                    "model_token": model,
                                    "modality": modality,
                                    "transcript_condition": transcript,
                                    "route": route,
                                    "backend": "test-backend",
                                    "seed": seed,
                                    "fold": fold,
                                    "macro_f1": 0.5 + 0.01 * (seed == 1337) + 0.001 * fold,
                                    "positive_f1": 0.4 + 0.001 * fold,
                                    "support": 120,
                                    "invalid_count": 0,
                                    "attempt_id": f"attempt-{len(rows)}",
                                    "evaluation_id": f"evaluation-{len(rows)}",
                                    "provenance_key": f"prov-{len(rows)}",
                                    "run_name": "test",
                                    "fold_dir": "/tmp/test",
                                }
                            )
    tables = build_tables(rows)
    assert [len(tables[key]) for key in ("table1_dataset_condition", "table2_translation", "table3_translation_interaction", "table4_model_comparison", "table5_seed_details", "table6_fold_details")] == [30, 24, 12, 30, 180, 900]
    assert all(row["fold_count"] == 5 for row in tables["table5_seed_details"])
    assert all(row["complete_seed_fold_count"] == 15 for row in tables["table1_dataset_condition"])


def test_workbook_sheet_uses_validated_report_values(tmp_path: Path) -> None:
    rows = []
    inputs = (
        ("audio_only", "not_applicable"),
        ("text_only", "native"),
        ("text_only", "english"),
        ("audio_text", "native"),
        ("audio_text", "english"),
    )
    for condition in ("pos_only", "negative_only"):
        for model in ("qwen", "gemma4"):
            for modality, transcript in inputs:
                for route in ("teacher_forced", "logreg", "xgb_optuna100"):
                    for seed in (7, 1337, 2024):
                        for fold in range(5):
                            rows.append({
                                "condition": condition,
                                "model": "Gemma 4" if model == "gemma4" else "Qwen",
                                "model_token": model,
                                "modality": modality,
                                "transcript_condition": transcript,
                                "route": route,
                                "backend": "test-backend",
                                "seed": seed,
                                "fold": fold,
                                "macro_f1": 0.5,
                                "positive_f1": 0.4,
                                "support": 120,
                                "invalid_count": 0,
                                "attempt_id": f"attempt-{len(rows)}",
                                "evaluation_id": f"evaluation-{len(rows)}",
                                "provenance_key": f"prov-{len(rows)}",
                                "run_name": "test",
                                "fold_dir": "/tmp/test",
                            })
    tables = build_tables(rows)
    report = {
        "schema_version": "audiollm.turkish_question_condition_report.v1",
        "group_id": "test-group",
        "deployment_id": "test-deployment",
        "source_git_sha": "test-sha",
        "validation": {"status": "passed"},
        "tables": tables,
        "job_audit": {"execution": {"by_job_type": {job: {field: 1 for field in ("planned", "submitted", "successful", "failed", "cancelled", "retried", "superseded", "locally_validated", "reportable")} for job in ("train", "evaluation", "hidden_extraction", "hidden_classifier")}}},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    workbook = Workbook()
    workbook.remove(workbook.active)
    build_turkish_question_condition(workbook, report_path=report_path)
    sheet = workbook["Turkish PosOnly vs NegOnly"]
    assert sheet["A1"].value == "Turkish positive-only questions versus negative-only"
    assert sheet["D6"].value == tables["table1_dataset_condition"][0]["pos_only_macro_f1_mean"]
    assert any("Table 7" in str(cell.value) for row in sheet.iter_rows() for cell in row)


def test_strict_metric_parser_accepts_teacher_forced_label_text() -> None:
    metrics, support, invalid = _metric([
        {"label": "1", "subject_id": "p1", "prediction_text": "Depressed"},
        {"label": "0", "subject_id": "p2", "prediction_text": "Non-depressed"},
        {"label": "1", "subject_id": "p3", "prediction_text": "INVALID"},
    ])
    assert support == 3
    assert invalid == 1
    assert metrics["macro_f1"] == 2 / 3
    assert metrics["positive_f1"] == 2 / 3
