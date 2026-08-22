from openpyxl import Workbook

from scripts import build_clean_workbook as builder


def test_native_vs_en_sheet_uses_current_paired_values() -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)

    builder.build_native_vs_english(workbook)

    sheet = workbook["Native vs EN"]
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
