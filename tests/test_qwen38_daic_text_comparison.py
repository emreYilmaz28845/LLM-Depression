from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from tools.build_qwen38_daic_text_comparison import COLUMNS, build, main

RUN_DIR = "output_model/campaign/text_only/daic/example_run/fold_0"


def _write_run(project_root: Path, *, evaluation_view: str | None = "harmonized_all_windows_full_coverage") -> Path:
    run_dir = project_root / RUN_DIR
    (run_dir / "best_model" / "standalone_eval").mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text(
        "\n".join(
            [
                "config:",
                "  dataset: daic",
                "fold: 0",
                "selection_protocol:",
                "  metric_name: inner_val_macro_f1",
                "  metric_mode: max",
                "evaluation:",
                "  aggregation_level: subject",
                "tracking:",
                "  attempt_id: 20260920T000000Z-example-run-abcdef12-01234567",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "evaluations.json").write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "evaluation_id": "eval-1",
                        "backend": "likelihood",
                        "evaluation_view": evaluation_view,
                        "aggregation": "subject_level",
                        "checkpoint_role": "best_model",
                        "locally_verified": evaluation_view is not None,
                        "warnings": [] if evaluation_view else ["evaluation view not recorded in evidence"],
                    }
                ],
                "schema_version": "audiollm.evaluations.v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "jobs.jsonl").write_text(
        json.dumps({"job_key": "train", "slurm_job_id": "46222537", "event_type": "SUBMITTED"})
        + "\n"
        + json.dumps({"job_key": "best_eval", "slurm_job_id": "46222538", "event_type": "SUBMITTED"})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "best_model" / "standalone_eval" / "metrics_likelihood.json").write_text(
        json.dumps(
            {
                "binary_strict_macro_f1": 0.7552083333333333,
                "binary_strict_positive_f1": 0.6666666666666666,
                "binary_strict_confusion_matrix": [[27, 6], [4, 10]],
                "macro_recall": 0.7662337662337663,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_row_reads_verified_artifacts(monkeypatch, tmp_path: Path) -> None:
    _write_run(tmp_path)
    monkeypatch.setattr(
        "tools.build_qwen38_daic_text_comparison.ROWS",
        [
            {
                "model": "Qwen3.8-27B",
                "run_dir": RUN_DIR,
                "backend": "likelihood",
            }
        ],
    )
    rows, provenance = build(tmp_path)
    row = rows[0]
    assert row["model"] == "Qwen3.8-27B"
    assert row["run_name"] == "example_run"
    assert row["attempt_id"].startswith("20260920T000000Z-example-run")
    assert row["selection_metric"] == "inner_val_macro_f1"
    assert row["selection_metric_mode"] == "max"
    assert row["evaluation_view"] == "harmonized_all_windows_full_coverage"
    assert row["macro_f1"] == "0.755208"
    assert row["positive_f1"] == "0.666667"
    # UAR is the mean class recall of the binary_strict confusion matrix.
    assert row["uar"] == "0.766234"
    assert row["jobs"] == "46222537,46222538"
    assert row["checkpoint_role"] == "best_model"
    assert row["notes"] == ""
    assert provenance["schema"] == "audiollm.qwen38_daic_text_comparison.v1"


def test_missing_view_and_metrics_stay_blank_with_reasons(monkeypatch, tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, evaluation_view=None)
    (run_dir / "best_model" / "standalone_eval" / "metrics_likelihood.json").unlink()
    monkeypatch.setattr(
        "tools.build_qwen38_daic_text_comparison.ROWS",
        [{"model": "Qwen3.8-27B", "run_dir": RUN_DIR, "backend": "likelihood"}],
    )
    rows, _ = build(tmp_path)
    row = rows[0]
    assert row["evaluation_view"] == ""
    assert row["macro_f1"] == "" and row["positive_f1"] == "" and row["uar"] == ""
    assert "evaluation_view not recorded in local evidence" in row["notes"]
    assert "metrics_likelihood.json missing" in row["notes"]
    assert "not locally verified" in row["notes"]


def test_absent_run_dir_is_reported_not_invented(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tools.build_qwen38_daic_text_comparison.ROWS",
        [{"model": "Qwen3.8-27B", "run_dir": "output_model/missing/fold_0", "backend": "likelihood"}],
    )
    rows, _ = build(tmp_path)
    assert rows[0]["macro_f1"] == ""
    assert "run directory not present" in rows[0]["notes"]


def test_smoke_runs_are_refused(monkeypatch, tmp_path: Path) -> None:
    smoke_dir = "output_model/campaign/text_only/daic/qwen38_text_only_smoke1ep_20260920/fold_0"
    _write_run(tmp_path)
    (tmp_path / "output_model/campaign/text_only/daic/qwen38_text_only_smoke1ep_20260920").mkdir()
    (tmp_path / "output_model/campaign/text_only/daic/qwen38_text_only_smoke1ep_20260920/fold_0").mkdir()
    monkeypatch.setattr(
        "tools.build_qwen38_daic_text_comparison.ROWS",
        [{"model": "Qwen3.8-27B", "run_dir": smoke_dir, "backend": "likelihood"}],
    )
    with pytest.raises(ValueError, match="Refusing to include smoke run"):
        build(tmp_path)


def test_main_writes_csv_markdown_and_json(monkeypatch, tmp_path: Path) -> None:
    _write_run(tmp_path)
    monkeypatch.setattr(
        "tools.build_qwen38_daic_text_comparison.ROWS",
        [{"model": "Qwen3.8-27B", "run_dir": RUN_DIR, "backend": "likelihood"}],
    )
    out_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_qwen38_daic_text_comparison.py",
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert main() == 0
    with (out_dir / "comparison.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == COLUMNS
    assert rows[0]["macro_f1"] == "0.755208"
    markdown = (out_dir / "comparison.md").read_text(encoding="utf-8")
    assert "Qwen3.8-27B" in markdown
    assert "no winner claim" in markdown
    provenance = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
    assert provenance["rows"][0]["model"] == "Qwen3.8-27B"
