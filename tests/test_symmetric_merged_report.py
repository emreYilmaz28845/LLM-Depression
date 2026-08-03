from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
import pytest

from src.merged.report import (
    collect_pooled_stage_rows,
    generate_reports,
    update_workbook,
    validate_workbook,
)


DATASETS = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")
MODALITIES = ("audio_text", "audio_only", "text_only")
METHODS = ("qwen", "logreg", "xgb_fixed", "xgb_optuna")


def _write_config(root: Path, modality: str) -> Path:
    config = {
        "name": f"synthetic_{modality}",
        "protocol": "symmetric_merged",
        "modality": modality,
        "output_dirs": {
            "merged_root": str(root / "merged" / modality),
            "run_root": str(root / "models" / modality),
        },
    }
    path = root / f"{modality}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_predictions(root: Path, run_id: str, modality: str) -> None:
    for fold in range(5):
        fold_root = root / "merged" / modality / run_id / "cv" / f"fold_{fold}"
        for dataset in DATASETS:
            rows = [
                {
                    "subject_id": f"{dataset}::subject_{fold}_{label}",
                    "label": label,
                    "prediction": label,
                    "probability": float(label),
                    "prediction_text": "0" if label == 0 else "1",
                }
                for label in (0, 1)
            ]
            qwen_path = fold_root / "qwen" / dataset / "predictions_subject_level.csv"
            qwen_path.parent.mkdir(parents=True, exist_ok=True)
            with qwen_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for method in ("logreg", "xgb_fixed", "xgb_optuna"):
            rows = [
                {
                    "dataset": dataset,
                    "subject_id": f"{dataset}::subject_{fold}_{label}",
                    "label": label,
                    "prediction": label,
                    "probability": float(label),
                }
                for dataset in DATASETS
                for label in (0, 1)
            ]
            head_path = fold_root / "heads" / method / "predictions_subject_level.jsonl"
            head_path.parent.mkdir(parents=True, exist_ok=True)
            head_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

    final_root = root / "merged" / modality / run_id / "final" / "fold_0"
    final_rows = [
        {
            "subject_id": f"daic::official_{label}",
            "label": label,
            "prediction": label,
            "probability": float(label),
            "prediction_text": "0" if label == 0 else "1",
        }
        for label in (0, 1)
    ]
    qwen_path = final_root / "qwen" / "daic" / "predictions_subject_level.csv"
    qwen_path.parent.mkdir(parents=True, exist_ok=True)
    with qwen_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    for method in ("logreg", "xgb_fixed", "xgb_optuna"):
        rows = [
            {
                "dataset": "daic",
                "subject_id": f"daic::official_{label}",
                "label": label,
                "prediction": label,
                "probability": float(label),
            }
            for label in (0, 1)
        ]
        head_path = final_root / "heads" / method / "predictions_subject_level.jsonl"
        head_path.parent.mkdir(parents=True, exist_ok=True)
        head_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_report_writes_pooled_csv_and_validates_workbook(tmp_path: Path) -> None:
    run_id = "synthetic_report_run"
    configs = [_write_config(tmp_path, modality) for modality in MODALITIES]
    for modality in MODALITIES:
        _write_predictions(tmp_path, run_id, modality)

    result = generate_reports(
        configs,
        run_id=run_id,
        output_dir=tmp_path / "report",
        workbook_path=tmp_path / "report.xlsx",
    )

    assert result["cv_rows"] == 3 * 5 * 5 * 4
    assert result["cv_pooled_rows"] == 3 * 5 * 4
    assert Path(tmp_path / "report" / "symmetric_merged_cv_pooled.csv").is_file()
    metadata = json.loads(
        (tmp_path / "report" / "symmetric_merged_execution_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["runtime_storage"]["job_accounting_fields"] == [
        "elapsed",
        "max_rss",
        "allocated_cpus",
        "allocated_tres",
        "node_list",
    ]
    assert result["workbook_validation"]["status"] == "passed"
    assert set(result["workbook_validation"]["sheets"]) == {
        "Merged Symmetric CV",
        "Merged DAIC Official",
    }


def test_report_rejects_incomplete_cv_prediction_coverage(tmp_path: Path) -> None:
    run_id = "incomplete_report_run"
    config = _write_config(tmp_path, "audio_text")
    _write_predictions(tmp_path, run_id, "audio_text")
    missing = (
        tmp_path
        / "merged"
        / "audio_text"
        / run_id
        / "cv"
        / "fold_3"
        / "heads"
        / "xgb_optuna"
        / "predictions_subject_level.jsonl"
    )
    missing.unlink()

    with pytest.raises(ValueError, match="Incomplete pooled CV prediction coverage"):
        collect_pooled_stage_rows(config, run_id=run_id, stage="cv")


def test_workbook_validation_allows_excel_float_rounding(tmp_path: Path) -> None:
    rows = [{"Metric": 0.23529411764705882}]
    workbook_path = tmp_path / "rounded.xlsx"
    update_workbook(workbook_path, rows, rows)

    from openpyxl import load_workbook

    workbook = load_workbook(workbook_path)
    workbook["Merged Symmetric CV"]["A2"] = 0.2352941176470588
    workbook.save(workbook_path)

    result = validate_workbook(workbook_path, cv_rows=rows, final_rows=rows)
    assert result["status"] == "passed"
