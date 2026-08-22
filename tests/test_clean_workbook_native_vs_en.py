from openpyxl import Workbook

from scripts import build_clean_workbook as builder


# ---------------------------------------------------------------------------
# Pending mode (no deterministic study report yet): the legacy single-seed
# snapshot is preserved verbatim.
# ---------------------------------------------------------------------------


def test_native_vs_en_sheet_uses_current_paired_values() -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)

    builder.build_native_vs_en_study(workbook, None)
    # Pending mode falls back to the legacy builder inside main(); exercise
    # both entry points so the fallback contract stays explicit.
    legacy = Workbook()
    legacy.remove(legacy.active)
    builder.build_native_vs_english(legacy)

    sheet = legacy["Native vs EN"]
    assert sheet.max_row == 29
    assert [sheet.cell(4, column).value for column in range(1, 11)] == [
        "Dataset",
        "Modality",
        "Method",
        "Qwen native",
        "Qwen EN",
        "Qwen Δ (EN − native)",
        "Gemma native",
        "Gemma EN",
        "Gemma Δ (EN − native)",
        "Direction",
    ]

    # D3TEC, Audio + Text, XGBoost: current native and Optuna-100 EN values.
    assert sheet.cell(7, 1).value == "D3TEC"
    assert sheet.cell(7, 2).value == "Audio + Text"
    assert sheet.cell(7, 3).value == "XGBoost"
    assert sheet.cell(7, 4).value == builder.QWEN_OPTUNA[("D3TEC", "Audio + Text")]
    assert sheet.cell(7, 5).value == builder.EN_XGB[("D3TEC", "Audio + Text")][0]
    assert sheet.cell(7, 7).value == builder.GEMMA_OPTUNA[("D3TEC", "Audio + Text")]
    assert sheet.cell(7, 8).value == builder.EN_XGB[("D3TEC", "Audio + Text")][1]

    pending = workbook["Native vs EN"]
    assert "pending" in str(pending.cell(2, 1).value).lower()
    headers = [pending.cell(4, column).value for column in range(1, 20)]
    assert headers[0] == "Endpoint"
    assert pending.max_row == 4  # title + note + header only; no data rows


def test_native_vs_en_sheet_omits_audio_only_shared_control() -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)

    builder.build_native_vs_english(workbook)

    modalities = {
        row[0]
        for row in workbook["Native vs EN"].iter_rows(
            min_row=5, max_row=28, min_col=2, max_col=2, values_only=True
        )
    }
    assert modalities == {"Audio + Text", "Text only"}


# ---------------------------------------------------------------------------
# Study-driven mode: locked three-seed layout (24 summary + 72 seed rows).
# ---------------------------------------------------------------------------


def _make_report() -> dict:
    seeds = ["7", "1337", "2024"]
    comparisons = []
    for endpoint, dataset in (
        [("standalone", d) for d in ("d3tec", "androids_interview", "cmdc", "turkish")]
        + [("merged_cv", ""), ("final_daic", "daic")]
    ):
        for backbone in ("qwen", "gemma4"):
            for head in ("logreg", "xgb_optuna100"):
                comp = {
                    "endpoint": endpoint,
                    "dataset": dataset,
                    "backbone": backbone,
                    "head": head,
                }
                for metric in ("macro_f1", "positive_f1"):
                    base = 0.5
                    comp["native"] = comp.get("native", {})
                    comp["english"] = comp.get("english", {})
                    comp.setdefault("paired_delta", {})
                    comp["native"][metric] = {
                        "mean": base,
                        "sample_sd": 0.01,
                        "per_seed": {s: base for s in seeds},
                    }
                    comp["english"][metric] = {
                        "mean": base + 0.05,
                        "sample_sd": 0.02,
                        "per_seed": {s: base + 0.05 for s in seeds},
                    }
                    comp["paired_delta"][metric] = {
                        "mean": 0.05,
                        "sample_sd": 0.0,
                        "per_seed": {s: 0.05 for s in seeds},
                    }
                comparisons.append(comp)
    return {
        "schema_version": "audiollm.native_en_study_report.v1",
        "group_id": "native-en-text-heads-20260822",
        "seeds": [7, 1337, 2024],
        "comparisons": comparisons,
        "evidence_index": [],
    }


def test_study_mode_writes_24_summary_and_72_seed_rows() -> None:
    report = _make_report()
    assert len(report["comparisons"]) == 24

    workbook = Workbook()
    workbook.remove(workbook.active)
    builder.build_native_vs_en_study(workbook, report)

    sheet = workbook["Native vs EN"]
    assert sheet.max_row - 4 == 24
    headers = [sheet.cell(4, col).value for col in range(1, 20)]
    assert headers[:4] == ["Endpoint", "Dataset", "Backbone", "Head"]
    assert headers[-3:] == ["Aggregation", "Seeds", "Status"]

    seeds_sheet = workbook["Native vs EN Seeds"]
    assert seeds_sheet.max_row - 4 == 72
    seed_headers = [seeds_sheet.cell(4, col).value for col in range(1, 15)]
    assert seed_headers[:5] == ["Endpoint", "Dataset", "Backbone", "Head", "Seed"]
    assert seed_headers[-3:] == ["Split seed", "Head seed", "Evaluation view"]

    # Spot-check one merged CV row's ordering and values.
    merged_rows = [
        r for r in range(5, sheet.max_row + 1)
        if sheet.cell(r, 1).value == "Merged CV"
    ]
    assert len(merged_rows) == 4
    first = merged_rows[0]
    assert sheet.cell(first, 5).value == 0.5          # native macro mean
    assert sheet.cell(first, 7).value == 0.55         # english macro mean
    assert sheet.cell(first, 9).value == 0.05         # delta macro mean
    assert sheet.cell(first, 18).value == 3           # seeds count
    assert sheet.cell(first, 19).value == "REPORTABLE"


def test_register_cell_values_for_validate_selected() -> None:
    report = _make_report()
    cell_values: dict = {}
    builder.register_native_en_cell_values(cell_values, report)
    assert cell_values, "study cells must be registered for validation"
    sample_key = next(iter(cell_values))
    assert sample_key.endswith("|macro") or sample_key.endswith("|pos")
    value = cell_values[sample_key]
    assert isinstance(value, float)


def test_report_schema_mismatch_refused(tmp_path) -> None:
    bad = tmp_path / "report.json"
    payload = {"schema_version": "audiollm.other.v1"}
    bad.write_text(json.dumps({"schema_version": "audiollm.other.v1"}))
    import os

    old = os.environ.pop(builder.NATIVE_EN_STUDY_REPORT_ENV, None)
    try:
        os.environ[builder.NATIVE_EN_STUDY_REPORT_ENV] = str(bad)
        import pytest

        with pytest.raises(ValueError, match="schema_version"):
            builder.load_native_en_report()
    finally:
        if old is None:
            os.environ.pop(builder.NATIVE_EN_STUDY_REPORT_ENV, None)
        else:
            os.environ[builder.NATIVE_EN_STUDY_REPORT_ENV] = old


import json  # noqa: E402  (used by fixtures above)
