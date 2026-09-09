"""CV headline summaries must not switch to pooled F1 by dataset or model."""
import json
import pytest
from tools import native_en_text_heads_report as report
from tools import summarize_gemma4_harmonized_heads as gemma
from scripts import build_clean_workbook as workbook


@pytest.mark.parametrize("dataset", ["d3tec", "androids_interview", "cmdc", "turkish"])
@pytest.mark.parametrize("backbone", ["qwen", "gemma4"])
def test_every_standalone_cell_uses_equal_fold_weights(monkeypatch, dataset, backbone):
    records = [{"seed": seed, "fold": fold} for seed in report.TRAINING_SEEDS for fold in range(5)]
    monkeypatch.setattr(report, "_fold_metrics", lambda r, ds: {"macro_f1": r["fold"] / 4, "positive_f1": r["fold"] / 8})
    monkeypatch.setattr(report, "_record_provenance", lambda r, dataset=None: r)
    monkeypatch.setattr(report, "_pooled_metrics", lambda *a: pytest.fail("pooled CV is forbidden"))
    cell = report._aggregate_cell(records, endpoint="standalone", condition="native", backbone=backbone, method="logreg", dataset=dataset)
    assert "mean" in cell["aggregation"] and "pooled" not in cell["aggregation"]
    for row in cell["seed_rows"]:
        assert row["macro_f1"] == pytest.approx(.5)
        assert row["positive_f1"] == pytest.approx(.25)


def test_missing_or_duplicate_folds_fail_closed():
    records = [{"seed": seed, "fold": fold} for seed in report.TRAINING_SEEDS for fold in [0, 1, 2, 3, 3]]
    with pytest.raises(report.ReportError, match="exactly folds"):
        report._aggregate_cell(records, endpoint="standalone", condition="native", backbone="qwen", method="logreg", dataset="d3tec")
    with pytest.raises(ValueError, match="exactly folds"):
        gemma.group_report({}, "d3tec", "audio_text")


def test_workbook_rejects_old_pooled_native_english_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema_version": "native_en_text_heads_v2_report.v2", "status": "passed", "summary": [{"endpoint": "standalone", "aggregation": "pooled subject-level"}] * 44, "seed_details": [{}] * 132}))
    with pytest.raises(ValueError, match="requires fold-mean"):
        workbook._load_native_en_report(path)


@pytest.mark.parametrize("dataset", ["androids_interview", "cmdc", "d3tec", "daic", "turkish"])
@pytest.mark.parametrize("backbone", ["qwen", "gemma4"])
def test_merged_cv_per_dataset_cell_uses_equal_fold_weights(monkeypatch, dataset, backbone):
    records = [{"seed": seed, "fold": fold} for seed in report.TRAINING_SEEDS for fold in range(5)]
    monkeypatch.setattr(report, "_fold_metrics", lambda r, ds=None: {"macro_f1": r["fold"] / 4, "positive_f1": r["fold"] / 8})
    monkeypatch.setattr(report, "_record_provenance", lambda r, dataset=None: r)
    monkeypatch.setattr(report, "_pooled_metrics", lambda *a: pytest.fail("pooled CV is forbidden"))
    cell = report._aggregate_cell(records, endpoint="merged_cv", condition="native", backbone=backbone, method="logreg", dataset=dataset, per_dataset=True)
    assert cell["aggregation"] == report.MERGED_CV_PER_DATASET_AGGREGATION
    assert "mean" in cell["aggregation"] and "pooled" not in cell["aggregation"]
    for row in cell["seed_rows"]:
        assert row["macro_f1"] == pytest.approx(.5)
        assert row["positive_f1"] == pytest.approx(.25)


def test_workbook_cv_source_labels_and_translation_tables_use_mean():
    for dataset in ("D3TEC", "Androids Interview", "CMDC", "Turkish"):
        assert "mean" in workbook.STANDALONE_QWEN_SOURCE[dataset][1].lower()
        for modality in workbook.MODALITIES:
            row = workbook.HARMONIZED_EN_QWEN[dataset, modality]
            assert "mean" in row[4] and "pooled" not in row[4]
            assert row[0] == workbook.STANDALONE_QWEN[dataset, modality]
            if dataset in ("D3TEC", "Androids Interview"):
                assert row[1] == workbook.STANDALONE_QWEN_POSF1[dataset, modality]
