"""Build depression_results_clean.xlsx — the trustworthy, provenance-carrying results workbook.

Generated 2026-08-06. Every headline value was verified by recomputation from local
artifacts (final_summary.json, coverage audits, matrix audits, merged-run prediction
files) or matched to an audited experiment document. Sources per row: Provenance sheet.

No values are hand-edited after generation; to change a number, change this script and
regenerate. This is the report layer; the artifact layer is the files referenced in
the Provenance sheet.

Modes:
  default    depression_results_clean.xlsx  (headline: Qwen + LogReg + XGBoost fixed only)
  --detailed depression_results_clean_detailed.xlsx in docs/archive/results_20260806/
             (adds XGBoost Optuna and Subject-OS columns/rows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "depression_results_clean.xlsx"
OUT_DETAILED = PROJECT_ROOT / "docs/archive/results_20260806/depression_results_clean_detailed.xlsx"

# --------------------------------------------------------------------------- styles
DARK = PatternFill("solid", fgColor="203864")
BLUE = PatternFill("solid", fgColor="D9EAF7")
BODY = PatternFill("solid", fgColor="EAF2F8")
NOTE = PatternFill("solid", fgColor="FFF4E5")
POS_FILL = PatternFill("solid", fgColor="C6EFCE")
NEG_FILL = PatternFill("solid", fgColor="FFC7CE")
NEUTRAL_FILL = PatternFill("solid", fgColor="F2F2F2")
WHITE = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=20, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Calibri", size=10, color="1F1F1F")
SMALL_FONT = Font(name="Calibri", size=9, color="1F1F1F")
THIN = Side(style="thin", color="B4C6E7")
BORDER = Border(bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")
WRAP = Alignment(horizontal="left", vertical="center", wrap_text=True)


def _section(ws, row: int, label: str, span: int) -> None:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    cell = ws.cell(row, 1, label)
    cell.font = WHITE
    cell.fill = DARK
    cell.alignment = LEFT
    ws.row_dimensions[row].height = 22


def _header_row(ws, row: int, headers: list[str]) -> None:
    for col, value in enumerate(headers, start=1):
        cell = ws.cell(row, col, value)
        cell.font = WHITE
        cell.fill = DARK
        cell.alignment = CENTER
        cell.border = BORDER


def _body_cell(ws, row: int, col: int, value: Any, *, fmt: str | None = None) -> None:
    cell = ws.cell(row, col, value)
    cell.font = BODY_FONT
    cell.fill = BODY
    cell.alignment = CENTER
    cell.border = BORDER
    if fmt and isinstance(value, (int, float)):
        cell.number_format = fmt


DELTA_EPS = 0.0005


def _delta_fill(value: float | None) -> PatternFill | None:
    """Green = positive delta (EN better / merged better), red = negative, neutral ~0."""
    if value is None:
        return None
    if value > DELTA_EPS:
        return POS_FILL
    if value < -DELTA_EPS:
        return NEG_FILL
    return NEUTRAL_FILL


def _delta_cell(ws, row: int, col: int, value: float) -> None:
    _body_cell(ws, row, col, round(value, 4), fmt="0.0000")
    fill = _delta_fill(value)
    if fill is not None:
        ws.cell(row, col).fill = fill


def _title(ws, text: str, span: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=span)
    ws["A1"] = text
    ws["A1"].font = TITLE_FONT
    ws["A1"].fill = DARK
    ws["A1"].alignment = CENTER
    ws.row_dimensions[1].height = 32


def _note(ws, row: int, text: str, span: int, *, height: int = 26) -> None:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    cell = ws.cell(row, 1, text)
    cell.font = BODY_FONT
    cell.fill = NOTE
    cell.alignment = WRAP
    ws.row_dimensions[row].height = height


def _widths(ws, widths: dict[str, float]) -> None:
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


# --------------------------------------------------------------------------- data
# Standalone fine-tuned Qwen macro-F1. All values recomputed 2026-08-06 from the
# local artifacts listed in PROVENANCE_QWEN. (dataset, modality) -> macro-F1.
STANDALONE_QWEN: dict[tuple[str, str], float] = {
    ("DAIC", "Audio + Text"): 0.840678,
    ("DAIC", "Audio only"): 0.683405,
    ("DAIC", "Text only"): 0.735279,
    ("CMDC", "Audio + Text"): 0.905006,
    ("CMDC", "Audio only"): 0.940341,
    ("CMDC", "Text only"): 0.927452,
    ("Turkish", "Audio + Text"): 0.487747,
    ("Turkish", "Audio only"): 0.612749,
    ("Turkish", "Text only"): 0.511934,
    ("D3TEC", "Audio + Text"): 0.567944,
    ("D3TEC", "Audio only"): 0.569444,
    ("D3TEC", "Text only"): 0.503205,
    ("Androids Interview", "Audio + Text"): 0.843666,
    ("Androids Interview", "Audio only"): 0.895023,
    ("Androids Interview", "Text only"): 0.869516,
}

# Standalone hidden-state heads: (dataset, modality) -> (logreg, xgb_fixed, xgb_optuna, subject_os)
STANDALONE_HEADS: dict[tuple[str, str], tuple[float | None, float | None, float | None, float | None]] = {
    ("DAIC", "Audio + Text"): (0.816, 0.746, None, None),
    ("DAIC", "Audio only"): (0.659, 0.585, None, None),
    ("DAIC", "Text only"): (0.755, 0.787, None, None),
    ("CMDC", "Audio + Text"): (0.985434, 0.970566, 0.985434, None),
    ("CMDC", "Audio only"): (0.985434, 0.985434, 0.985434, None),
    ("CMDC", "Text only"): (0.985434, 0.970566, 0.971154, None),
    ("Turkish", "Audio + Text"): (0.498433, 0.435231, 0.536134, 0.472693),
    ("Turkish", "Audio only"): (0.577778, 0.49, 0.557453, 0.524445),
    ("Turkish", "Text only"): (0.555556, 0.623016, 0.604021, 0.599636),
    ("D3TEC", "Audio + Text"): (0.564324, 0.510748, 0.550363, None),
    ("D3TEC", "Audio only"): (0.558893, 0.544118, 0.569444, None),
    ("D3TEC", "Text only"): (0.580208, 0.481714, 0.526219, None),
    ("Androids Interview", "Audio + Text"): (0.835608, 0.887521, 0.852118, None),
    ("Androids Interview", "Audio only"): (0.844086, 0.844086, 0.835608, None),
    ("Androids Interview", "Text only"): (0.843666, 0.834721, 0.825038, None),
}

# Merged symmetric runs. modality -> (run_id, official DAIC macro per method,
# pooled-CV macro per (dataset, method)).
MERGED_RUNS: dict[str, dict[str, Any]] = {
    "audio_text": {
        "run_id": "merged_retrain_randomk_20260805_260064c",
        "official": {"qwen": 0.810484, "logreg": 0.714207, "xgb_fixed": 0.825464, "xgb_optuna": None},
        "cv": {
            ("daic", "qwen"): 0.626694, ("cmdc", "qwen"): 0.955369, ("turkish", "qwen"): 0.517677,
            ("d3tec", "qwen"): 0.643678, ("androids_interview", "qwen"): 0.809836,
            ("daic", "logreg"): 0.646488, ("cmdc", "logreg"): 0.970566, ("turkish", "logreg"): 0.514472,
            ("d3tec", "logreg"): 0.489237, ("androids_interview", "logreg"): 0.810119,
            ("daic", "xgb_fixed"): 0.671042, ("cmdc", "xgb_fixed"): 0.955369, ("turkish", "xgb_fixed"): 0.451659,
            ("d3tec", "xgb_fixed"): 0.626604, ("androids_interview", "xgb_fixed"): 0.861410,
        },
    },
    "audio_only": {
        "run_id": "merged_retrain_randomk_ao_20260805_7c21a9e",
        "official": {"qwen": 0.571228, "logreg": 0.734163, "xgb_fixed": 0.584596, "xgb_optuna": None},
        "cv": {
            ("daic", "qwen"): 0.604757, ("cmdc", "qwen"): 0.970566, ("turkish", "qwen"): 0.564270,
            ("d3tec", "qwen"): 0.588095, ("androids_interview", "qwen"): 0.827586,
            ("daic", "logreg"): 0.599097, ("cmdc", "logreg"): 0.970566, ("turkish", "logreg"): 0.448659,
            ("d3tec", "logreg"): 0.564324, ("androids_interview", "logreg"): 0.836097,
            ("daic", "xgb_fixed"): 0.582911, ("cmdc", "xgb_fixed"): 0.970566, ("turkish", "xgb_fixed"): 0.455816,
            ("d3tec", "xgb_fixed"): 0.609244, ("androids_interview", "xgb_fixed"): 0.913148,
        },
    },
    "text_only": {
        "run_id": "symmetric_merged_smoke_6fba6e632653",
        "official": {"qwen": 0.809275, "logreg": 0.734163, "xgb_fixed": 0.787330, "xgb_optuna": 0.787330},
        "cv": {
            ("daic", "qwen"): 0.620828, ("cmdc", "qwen"): 0.985434, ("turkish", "qwen"): 0.568750,
            ("d3tec", "qwen"): 0.522190, ("androids_interview", "qwen"): 0.784082,
            ("daic", "logreg"): 0.690141, ("cmdc", "logreg"): 0.970566, ("turkish", "logreg"): 0.635797,
            ("d3tec", "logreg"): 0.493544, ("androids_interview", "logreg"): 0.792857,
            ("daic", "xgb_fixed"): 0.671362, ("cmdc", "xgb_fixed"): 0.970566, ("turkish", "xgb_fixed"): 0.621818,
            ("d3tec", "xgb_fixed"): 0.558893, ("androids_interview", "xgb_fixed"): 0.783695,
            ("daic", "xgb_optuna"): 0.677704, ("cmdc", "xgb_optuna"): 0.970566, ("turkish", "xgb_optuna"): 0.621818,
            ("d3tec", "xgb_optuna"): 0.561665, ("androids_interview", "xgb_optuna"): 0.783695,
        },
    },
}

# ---- Harmonized campaign (2026-08-09/10) ----------------------------------
# Recipe harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1,
# campaign harmonized_v1_prod_20260809T171705Z_d1e8130b (Issue #12 / PR #10,
# source d1e8130b -> 22f297d -> 0a10063; retries under _r1 run names,
# retry registry retry_r1_jobs.tsv). Backend original_teacher_forced,
# headline/binary_strict, best_model checkpoints, seed 1337.
# Aggregations follow the workbook conventions: D3TEC/Androids pooled
# subject-level 5-fold; Turkish/CMDC 5-fold mean; DAIC fixed official test.
# All values recomputed 2026-08-10 from the local artifacts in the
# Provenance sheet. Optuna/Subject-OS were not run (fixed heads only).

HARMONIZED_STANDALONE_QWEN: dict[tuple[str, str], float] = {
    ("DAIC", "Audio + Text"): 0.7353,
    ("DAIC", "Audio only"): 0.5392,
    ("DAIC", "Text only"): 0.7353,
    ("CMDC", "Audio + Text"): 0.9700,
    ("CMDC", "Audio only"): 0.9516,
    ("CMDC", "Text only"): 0.9713,
    ("Turkish", "Audio + Text"): 0.6666,
    ("Turkish", "Audio only"): 0.5137,
    ("Turkish", "Text only"): 0.6502,
    ("D3TEC", "Audio + Text"): 0.5589,
    ("D3TEC", "Audio only"): 0.6649,
    ("D3TEC", "Text only"): 0.6113,
    ("Androids Interview", "Audio + Text"): 0.8606,
    ("Androids Interview", "Audio only"): 0.8690,
    ("Androids Interview", "Text only"): 0.7317,
}

# Harmonized standalone fixed heads: (dataset, modality) -> (logreg_raw, xgb_raw), 5-fold mean macro-F1.
HARMONIZED_STANDALONE_HEADS: dict[tuple[str, str], tuple[float, float]] = {
    ("DAIC", "Audio + Text"): (0.7432, 0.7353),
    ("DAIC", "Audio only"): (0.5583, 0.5190),
    ("DAIC", "Text only"): (0.7235, 0.7457),
    ("CMDC", "Audio + Text"): (0.9614, 0.9700),
    ("CMDC", "Audio only"): (0.9841, 0.9225),
    ("CMDC", "Text only"): (0.9420, 0.9683),
    ("Turkish", "Audio + Text"): (0.6289, 0.6325),
    ("Turkish", "Audio only"): (0.5209, 0.4271),
    ("Turkish", "Text only"): (0.5875, 0.5234),
    ("D3TEC", "Audio + Text"): (0.4988, 0.5585),
    ("D3TEC", "Audio only"): (0.6031, 0.5404),
    ("D3TEC", "Text only"): (0.4651, 0.5911),
    ("Androids Interview", "Audio + Text"): (0.8745, 0.8656),
    ("Androids Interview", "Audio only"): (0.8512, 0.8235),
    ("Androids Interview", "Text only"): (0.8326, 0.8241),
}

# Harmonized merged runs: official = DAIC protected test (47 subjects,
# daic_official_test_only); cv = 5-fold mean of per-fold holdout macro-F1.
HARMONIZED_MERGED: dict[str, dict[str, Any]] = {
    "audio_text": {
        "run_id": "harmonized_v1_prod_20260809T171705Z_d1e8130b",
        "epochs": 2,
        "official": {"qwen": 0.7631, "logreg": 0.7432, "xgb_fixed": 0.7432},
        "cv": {
            ("daic", "qwen"): 0.6540, ("cmdc", "qwen"): 0.9262, ("turkish", "qwen"): 0.5133,
            ("d3tec", "qwen"): 0.6071, ("androids_interview", "qwen"): 0.7874,
            ("daic", "logreg"): 0.6634, ("cmdc", "logreg"): 0.9421, ("turkish", "logreg"): 0.4786,
            ("d3tec", "logreg"): 0.6115, ("androids_interview", "logreg"): 0.8062,
            ("daic", "xgb_fixed"): 0.6835, ("cmdc", "xgb_fixed"): 0.9577, ("turkish", "xgb_fixed"): 0.4940,
            ("d3tec", "xgb_fixed"): 0.5737, ("androids_interview", "xgb_fixed"): 0.7842,
        },
    },
    "audio_only": {
        "run_id": "harmonized_v1_prod_20260809T171705Z_d1e8130b",
        "epochs": 4,
        "official": {"qwen": 0.5332, "logreg": 0.6189, "xgb_fixed": 0.5554},
        "cv": {
            ("daic", "qwen"): 0.6018, ("cmdc", "qwen"): 0.9500, ("turkish", "qwen"): 0.4516,
            ("d3tec", "qwen"): 0.5925, ("androids_interview", "qwen"): 0.8564,
            ("daic", "logreg"): 0.5586, ("cmdc", "logreg"): 0.9683, ("turkish", "logreg"): 0.4820,
            ("d3tec", "logreg"): 0.5387, ("androids_interview", "logreg"): 0.8441,
            ("daic", "xgb_fixed"): 0.5631, ("cmdc", "xgb_fixed"): 0.9366, ("turkish", "xgb_fixed"): 0.4484,
            ("d3tec", "xgb_fixed"): 0.5378, ("androids_interview", "xgb_fixed"): 0.8531,
        },
    },
    "text_only": {
        "run_id": "harmonized_v1_prod_20260809T171705Z_d1e8130b",
        "epochs": 5,
        "official": {"qwen": 0.7756, "logreg": 0.7157, "xgb_fixed": 0.7552},
        "cv": {
            ("daic", "qwen"): 0.7219, ("cmdc", "qwen"): 0.9539, ("turkish", "qwen"): 0.6515,
            ("d3tec", "qwen"): 0.6528, ("androids_interview", "qwen"): 0.7903,
            ("daic", "logreg"): 0.7168, ("cmdc", "logreg"): 0.9698, ("turkish", "logreg"): 0.6684,
            ("d3tec", "logreg"): 0.5733, ("androids_interview", "logreg"): 0.7714,
            ("daic", "xgb_fixed"): 0.7093, ("cmdc", "xgb_fixed"): 0.9405, ("turkish", "xgb_fixed"): 0.6288,
            ("d3tec", "xgb_fixed"): 0.5622, ("androids_interview", "xgb_fixed"): 0.8079,
        },
    },
}

DATASET_LABELS = {
    "daic": "DAIC",
    "cmdc": "CMDC",
    "turkish": "Turkish",
    "d3tec": "D3TEC",
    "androids_interview": "Androids Interview",
}
MODALITY_LABELS = {"audio_text": "Audio + Text", "audio_only": "Audio only", "text_only": "Text only"}
METHOD_LABELS = {"qwen": "Fine-tuned Qwen", "logreg": "LogReg head", "xgb_fixed": "XGBoost fixed", "xgb_optuna": "XGBoost Optuna"}
METHOD_LABELS_SHORT = {"qwen": "qwen", "logreg": "logreg", "xgb_fixed": "xgb_fixed", "xgb_optuna": "xgb_optuna"}

# EN translation: (dataset, modality) -> (native macro, translated macro). Native =
# 5-fold mean of the native-language runs (verified locally). Translated = MN5
# final_summary.json values (docs/ENGLISH_TRANSLATION_RESULTS_2026-08-06.md §2).
EN_DATA: dict[tuple[str, str], tuple[float, float]] = {
    ("CMDC", "Audio only"): (0.940341, 0.9558),
    ("CMDC", "Audio + Text"): (0.905006, 0.95),
    ("CMDC", "Text only"): (0.927452, 0.9563),
    ("Turkish", "Audio only"): (0.612749, 0.5771),
    ("Turkish", "Audio + Text"): (0.487747, 0.4864),
    ("Turkish", "Text only"): (0.511934, 0.4578),
    ("D3TEC", "Audio only"): (0.534786, 0.541),
    ("D3TEC", "Audio + Text"): (0.530093, 0.6108),
    ("D3TEC", "Text only"): (0.479264, 0.3892),
    ("Androids Interview", "Audio only"): (0.8868, 0.8842),
    ("Androids Interview", "Audio + Text"): (0.8335, 0.769),
    ("Androids Interview", "Text only"): (0.8537, 0.7848),
}

EN_NATIVE_RUNS = {
    "CMDC": {
        "Audio only": "cmdc_posf1_tf_cmdc_audio_only_selmacrof1_tf",
        "Audio + Text": "cmdc_posf1_tf_cmdc_audio_text_selmacrof1_tf",
        "Text only": "cmdc_posf1_tf_cmdc_text_only_selmacrof1_tf",
    },
    "Turkish": {
        "Audio only": "t17_posf1_tf_qwen3asr_turkish_t17_audio_only_selposf1_tf_qwen3asr",
        "Audio + Text": "t17_posf1_tf_qwen3asr_turkish_t17_audio_text_selposf1_tf_qwen3asr",
        "Text only": "t17_posf1_tf_qwen3asr_turkish_t17_text_only_selposf1_tf_qwen3asr",
    },
    "D3TEC": {
        "Audio only": "d3tec_prod_20260728T135954Z_d3tec_audio_only_rotary",
        "Audio + Text": "d3tec_prod_20260728T135954Z_d3tec_audio_text_rotary",
        "Text only": "d3tec_prod_20260728T135954Z_d3tec_text_only",
    },
    "Androids Interview": {
        "Audio only": "androids_interview_prod_20260730T145948Z_androids_interview_audio_only",
        "Audio + Text": "androids_interview_prod_20260730T145948Z_androids_interview_audio_text_segment_aligned",
        "Text only": "androids_interview_prod_20260730T145948Z_androids_interview_text_only",
    },
}
EN_EN_RUNS = {
    "CMDC": {
        "Audio only": "en_seq2_cmdc_audio_only_v1",
        "Audio + Text": "en_seq2_cmdc_audio_text_v1",
        "Text only": "en_seq2_cmdc_text_only_v1",
    },
    "Turkish": {
        "Audio only": "en_seq_turkish_audio_only_v1",
        "Audio + Text": "en_seq_turkish_audio_text_v1",
        "Text only": "en_seq_turkish_text_only_v1",
    },
    "D3TEC": {
        "Audio only": "en_seq_d3tec_audio_only_v1",
        "Audio + Text": "en_seq_d3tec_audio_text_v1",
        "Text only": "en_seq_d3tec_text_only_v1",
    },
    "Androids Interview": {
        "Audio only": "en_seq_androids_audio_only_v1",
        "Audio + Text": "en_seq_androids_audio_text_v1",
        "Text only": "en_seq_androids_text_only_v1",
    },
}

# EN-translation hidden-state heads (logreg_raw / xgb_raw), pooled 5-fold
# subject-level macro-F1 (Summary-sheet convention). Computed 2026-08-06 from
# outputs/hidden_classifiers/... (matrix configs/features/translation_en_matrix.yaml;
# jobs 44363856-44363935 + 44364706-44364711 reruns). Native heads = the same
# hidden-head pipeline on the native checkpoints; D3TEC native heads are the NEW
# rotary-recipe baselines (matched to the EN D3TEC rotary recipe).
EN_HEADS_NATIVE: dict[tuple[str, str], tuple[float, float]] = {
    ("CMDC", "Audio only"): (0.985434, 0.985434),
    ("CMDC", "Audio + Text"): (0.985434, 0.970566),
    ("CMDC", "Text only"): (0.985434, 0.970566),
    ("Turkish", "Audio only"): (0.577778, 0.490000),
    ("Turkish", "Audio + Text"): (0.498433, 0.435231),
    ("Turkish", "Text only"): (0.555556, 0.623016),
    ("D3TEC", "Audio only"): (0.536325, 0.489237),
    ("D3TEC", "Audio + Text"): (0.576681, 0.507937),
    ("D3TEC", "Text only"): (0.580208, 0.481714),
    ("Androids Interview", "Audio only"): (0.844086, 0.844086),
    ("Androids Interview", "Audio + Text"): (0.835608, 0.887521),
    ("Androids Interview", "Text only"): (0.843666, 0.834721),
}
EN_HEADS_EN: dict[tuple[str, str], tuple[float, float]] = {
    ("CMDC", "Audio only"): (0.985434, 0.985434),
    ("CMDC", "Audio + Text"): (0.985434, 0.970566),
    ("CMDC", "Text only"): (0.985434, 0.985434),
    ("Turkish", "Audio only"): (0.503582, 0.509380),
    ("Turkish", "Audio + Text"): (0.615349, 0.430000),
    ("Turkish", "Text only"): (0.555717, 0.555556),
    ("D3TEC", "Audio only"): (0.536325, 0.569444),
    ("D3TEC", "Audio + Text"): (0.583669, 0.563494),
    ("D3TEC", "Text only"): (0.580208, 0.591568),
    ("Androids Interview", "Audio only"): (0.826762, 0.836195),
    ("Androids Interview", "Audio + Text"): (0.861410, 0.810119),
    ("Androids Interview", "Text only"): (0.869907, 0.844086),
}

STANDALONE_QWEN_SOURCE = {
    "DAIC": (
        "daic_main_k4_control_20260804_f26dd45/fold_0/best_model (A+T); "
        "subject_audio/daic/daic_replicates_20ep_s1337_daic_audio_only_selposf1_tf (A-only); "
        "text_only/daic/daic_replicates_20ep_s1337_daic_text_only_selposf1_tf (T-only)",
        "Official test, 47 subjects, full-coverage K4-bundle view",
        "outputs/daic_k4_coverage_audit/*/coverage_audit.json + daic_modality_coverage_comparison.csv",
    ),
    "CMDC": (
        "cmdc_posf1_tf_cmdc_{audio_text,audio_only,text_only}_selmacrof1_tf",
        "5-fold CV mean, teacher-forced, subject-level, pos-F1 selection",
        "output_model/*/cmdc/*/final_summary.json",
    ),
    "Turkish": (
        "t17_posf1_tf_qwen3asr_turkish_t17_{audio_text,audio_only,text_only}_selposf1_tf_qwen3asr",
        "5-fold CV mean, teacher-forced, subject-level, pos-F1 selection, train_val protocol",
        "output_model/*/turkish_t17_qwen3asr/*/final_summary.json",
    ),
    "D3TEC": (
        "d3tec_prod_20260728T135954Z_d3tec_{audio_text,audio_only}_normalized + d3tec_text_only",
        "Pooled 5-fold subject-level (62 subjects), response-normalized, macro-F1 selection",
        "outputs/d3tec_matrix/d3tec_prod_20260728T135954Z/d3tec_matrix_audit.json",
    ),
    "Androids Interview": (
        "androids_interview_prod_20260730T145948Z_androids_interview_{audio_only,audio_text_segment_aligned,text_only}",
        "Pooled 5-fold subject-level (116 subjects), segment-aligned, macro-F1 selection",
        "output_model/experiments/androids_interview/*/final_summary.json",
    ),
}

DATASETS = ["DAIC", "CMDC", "Turkish", "D3TEC", "Androids Interview"]
MODALITIES = ["Audio + Text", "Audio only", "Text only"]
HEAD_METHODS = [("logreg", "LogReg head"), ("xgb_fixed", "XGBoost fixed"), ("xgb_optuna", "XGBoost Optuna")]


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


# --------------------------------------------------------------------------- sheets
def build_summary(wb: Workbook, *, detailed: bool) -> None:
    ws = wb.create_sheet("Summary")
    _widths(ws, {"A": 30, "B": 15, "C": 15, "D": 15, "E": 15, "F": 17})
    _title(ws, "Depression Detection — Macro-F1 Summary (clean, verified)", 6)
    _note(
        ws, 2,
        "Macro-F1 only; higher is better. Every value verified 2026-08-06 by recomputation from local artifacts; "
        "per-cell provenance (run, aggregation, eval view, artifact): Provenance sheet. "
        "DAIC = official test, full-coverage K4-bundle view (tuned by positive-F1). CMDC/Turkish = 5-fold CV mean, "
        "teacher-forced, pos-F1 selection. D3TEC = pooled 5-fold, response-normalized, macro-F1 selection "
        "(rotary variants are a different recipe and are NOT this column). Androids = pooled 5-fold, segment-aligned, "
        "macro-F1 selection. Blank = artifact unavailable (DAIC Optuna heads not run)."
        + ("" if detailed else " Optuna/Subject-OS columns are in the detailed workbook (docs/archive/results_20260806/)."),
        6, height=60,
    )
    headers = ["Evaluation / Modality", "Fine-tuned Qwen", "LogReg head", "XGBoost fixed"]
    if detailed:
        headers += ["XGBoost Optuna", "XGBoost Subject OS\n(3-seed mean)"]
    _header_row(ws, 4, headers)
    row = 5
    for dataset in DATASETS:
        for modality in MODALITIES:
            label = f"{dataset} — {modality}"
            ws.cell(row, 1, label).font = BODY_FONT
            ws.cell(row, 1).fill = BODY
            ws.cell(row, 1).alignment = LEFT
            ws.cell(row, 1).border = BORDER
            _body_cell(ws, row, 2, STANDALONE_QWEN[(dataset, modality)], fmt="0.0000")
            logreg, xgb, optuna, os_ = STANDALONE_HEADS[(dataset, modality)]
            _body_cell(ws, row, 3, logreg, fmt="0.0000")
            _body_cell(ws, row, 4, xgb, fmt="0.0000")
            if detailed:
                _body_cell(ws, row, 5, optuna, fmt="0.0000")
                _body_cell(ws, row, 6, os_, fmt="0.0000")
            row += 1
    ws.freeze_panes = "A5"


def build_merged_summary(wb: Workbook, *, detailed: bool) -> None:
    ws = wb.create_sheet("Merged Symmetric Summary")
    _widths(ws, {"A": 34, "B": 15, "C": 15, "D": 15, "E": 15})
    _title(ws, "Symmetric Merged — Macro-F1 Summary (clean, verified)", 5)
    _note(
        ws, 2,
        "Macro-F1 only; higher is better. Runs: A+T = merged_retrain_randomk_20260805_260064c (random-K retrain, "
        "full-coverage eval), A-only = merged_retrain_randomk_ao_20260805_7c21a9e, T-only = "
        "symmetric_merged_smoke_6fba6e632653 (never rerun). The A+T/AO retrains replace the smoke run because its "
        "DAIC config used fixed-K4 eval (collapse, 0.484 macro) instead of random-K + full coverage. "
        "CV values = pooled subject-level 5-fold predictions; final values = the protected DAIC official holdout. "
        "Optuna heads were not run on the retrain checkpoints -> blank. "
        "All values recomputed 2026-08-06 from outputs/symmetric_merged/{audio_text,audio_only,text_only}/<run>/.",
        5, height=78,
    )
    methods = [("qwen", "Fine-tuned Qwen"), ("logreg", "LogReg head"), ("xgb_fixed", "XGBoost fixed")]
    if detailed:
        methods.append(("xgb_optuna", "XGBoost Optuna"))
    headers = ["Evaluation / Modality", *(label for _, label in methods)]

    def cv_stats(modality: str, method: str) -> tuple[float | None, float | None]:
        values = [v for (ds, m), v in MERGED_RUNS[modality]["cv"].items() if m == method]
        if not values:
            return None, None
        return sum(values) / len(values), min(values)

    def table(row: int, getter) -> None:
        _header_row(ws, row, headers)
        for offset, (modality, modality_label) in enumerate(zip(["audio_text", "audio_only", "text_only"], MODALITIES), start=1):
            current = row + offset
            ws.cell(current, 1, modality_label).font = BODY_FONT
            ws.cell(current, 1).fill = BODY
            ws.cell(current, 1).alignment = LEFT
            ws.cell(current, 1).border = BORDER
            for col, (method, _) in enumerate(methods, start=2):
                _body_cell(ws, current, col, getter(modality, method), fmt="0.0000")

    _section(ws, 6, "DAIC Official Holdout — Final Macro-F1", 5)
    table(7, lambda mod, method: MERGED_RUNS[mod]["official"].get(method))

    _section(ws, 11, "Five-Dataset CV — Pooled Mean Macro-F1", 5)
    table(12, lambda mod, method: cv_stats(mod, method)[0])

    _section(ws, 16, "Five-Dataset CV — Worst-Dataset Macro-F1", 5)
    table(17, lambda mod, method: cv_stats(mod, method)[1])

    _section(ws, 21, "Five-Dataset CV — Macro-F1 by Dataset and Modality", 5)
    _header_row(ws, 22, headers)
    row = 23
    for modality in ["audio_text", "audio_only", "text_only"]:
        for dataset in ["daic", "cmdc", "turkish", "d3tec", "androids_interview"]:
            label = f"{DATASET_LABELS[dataset]} — {MODALITY_LABELS[modality]}"
            ws.cell(row, 1, label).font = BODY_FONT
            ws.cell(row, 1).fill = BODY
            ws.cell(row, 1).alignment = LEFT
            ws.cell(row, 1).border = BORDER
            for col, (method, _) in enumerate(methods, start=2):
                _body_cell(ws, row, col, MERGED_RUNS[modality]["cv"].get((dataset, method)), fmt="0.0000")
            row += 1
    ws.freeze_panes = "A5"


def build_en(wb: Workbook) -> None:
    ws = wb.create_sheet("EN Translation MacroF1")
    _widths(ws, {"A": 30, "B": 15, "C": 17, "D": 11, "E": 15, "F": 17, "G": 11})
    _title(ws, "English Translation vs Native — Macro-F1 (clean, verified)", 7)
    _note(
        ws, 2,
        "Macro-F1 only; higher is better. Teacher-forced, binary-strict. Qwen3.6-27B English translations of the "
        "native transcripts (audio unchanged). Qwen rows: 5-fold means of positive-F1-selected checkpoints; "
        "native column verified locally from final_summary.json, translated column from MN5 final_summary.json "
        "(output_model_en/, not synced; docs/ENGLISH_TRANSLATION_RESULTS_2026-08-06.md §2). "
        "Head rows: logreg_raw / xgb_raw hidden-state heads, pooled 5-fold subject-level macro-F1 (Summary-sheet "
        "convention), computed 2026-08-06 from outputs/hidden_classifiers (configs/features/translation_en_matrix.yaml, "
        "70 jobs; 6 first-wave jobs failed with CUDA OOM on degraded node as01r2b12 and were rerun with "
        "--exclude=as01r2b12). D3TEC native = rotary recipe (matched baseline; native rotary heads are new). "
        "Androids native heads = audited hidden report values; EN heads = new. D3TEC audio-only features are "
        "identical native vs EN (transcript unused) — equal scores are expected.",
        4, height=100,
    )
    _header_row(ws, 4, ["Evaluation / Modality", "Native Macro-F1", "Translated (EN) Macro-F1", "Δ Macro-F1"])
    row = 5
    for dataset in DATASETS:
        for modality in MODALITIES:
            key = (dataset, modality)
            if key not in EN_DATA:
                continue
            native, translated = EN_DATA[key]
            ws.cell(row, 1, f"{dataset} — {modality}").font = BODY_FONT
            ws.cell(row, 1).fill = BODY
            ws.cell(row, 1).alignment = LEFT
            ws.cell(row, 1).border = BORDER
            _body_cell(ws, row, 2, native, fmt="0.0000")
            _body_cell(ws, row, 3, translated, fmt="0.0000")
            _delta_cell(ws, row, 4, translated - native)
            row += 1

    row += 1
    _section(ws, row, "Hidden-state heads (pooled 5-fold subject-level macro-F1)", 7)
    _header_row(ws, row + 1, ["Evaluation / Modality", "Native LogReg", "EN LogReg", "Δ LogReg",
                              "Native XGB", "EN XGB", "Δ XGB"])
    r = row + 2
    for dataset in DATASETS:
        for modality in MODALITIES:
            key = (dataset, modality)
            if key not in EN_HEADS_NATIVE:
                continue
            nl, nx = EN_HEADS_NATIVE[key]
            el, ex = EN_HEADS_EN[key]
            ws.cell(r, 1, f"{dataset} — {modality}").font = BODY_FONT
            ws.cell(r, 1).fill = BODY
            ws.cell(r, 1).alignment = LEFT
            ws.cell(r, 1).border = BORDER
            _body_cell(ws, r, 2, nl, fmt="0.0000")
            _body_cell(ws, r, 3, el, fmt="0.0000")
            _delta_cell(ws, r, 4, el - nl)
            _body_cell(ws, r, 5, nx, fmt="0.0000")
            _body_cell(ws, r, 6, ex, fmt="0.0000")
            _delta_cell(ws, r, 7, ex - nx)
            r += 1
    ws.freeze_panes = "A5"


def build_merged_vs_standalone(wb: Workbook) -> None:
    ws = wb.create_sheet("Merged vs Standalone MacroF1")
    _widths(ws, {"A": 30, "B": 18, "C": 18, "D": 12, "E": 16})
    _title(ws, "Symmetric Merged vs Standalone — Macro-F1 by Head (clean, verified)", 5)
    _note(
        ws, 2,
        "Values are `standalone / merged`. Macro-F1 only; higher is better. Teacher-forced, binary-strict, "
        "subject-level. Merged: DAIC = protected official holdout of the retrain runs (A+T "
        "merged_retrain_randomk_20260805_260064c, A-only merged_retrain_randomk_ao_20260805_7c21a9e, T-only smoke); "
        "other datasets = the same runs' pooled 5-fold CV. Standalone: Summary sheet. Direction = Δ (merged − "
        "standalone); ~tie for |Δ| < 0.03. D3TEC standalone = normalized recipe. XGBoost Optuna omitted: not run "
        "on the retrain merged checkpoints (only T-only has a merged Optuna artifact).",
        5, height=78,
    )
    heads = [("qwen", "Fine-tuned Qwen head"), ("logreg", "LogReg head"), ("xgb_fixed", "XGBoost fixed head")]

    def merged_value(dataset: str, modality: str, method: str) -> float:
        mod_key = {"Audio + Text": "audio_text", "Audio only": "audio_only", "Text only": "text_only"}[modality]
        ds_key = next(k for k, v in DATASET_LABELS.items() if v == dataset)
        if dataset == "DAIC":
            return MERGED_RUNS[mod_key]["official"].get(method)
        return MERGED_RUNS[mod_key]["cv"].get((ds_key, method))

    row = 4
    for method, label in heads:
        _section(ws, row, label, 5)
        _header_row(ws, row + 1, ["Evaluation / Modality", "Standalone / Merged Macro-F1", "Δ Macro-F1", "Direction"])
        r = row + 2
        for dataset in DATASETS:
            for modality in MODALITIES:
                if method == "qwen":
                    standalone = STANDALONE_QWEN[(dataset, modality)]
                else:
                    head_idx = {"logreg": 0, "xgb_fixed": 1}[method]
                    standalone = STANDALONE_HEADS[(dataset, modality)][head_idx]
                merged = merged_value(dataset, modality, method)
                if standalone is None or merged is None:
                    continue
                delta = merged - standalone
                if abs(delta) < 0.03:
                    direction = "~tie"
                else:
                    direction = "merged better" if delta > 0 else "standalone better"
                ws.cell(r, 1, f"{dataset} — {modality}").font = BODY_FONT
                ws.cell(r, 1).fill = BODY
                ws.cell(r, 1).alignment = LEFT
                ws.cell(r, 1).border = BORDER
                _body_cell(ws, r, 2, f"{standalone:.4f} / {merged:.4f}")
                _delta_cell(ws, r, 3, delta)
                dir_cell = ws.cell(r, 4, direction)
                dir_cell.font = BODY_FONT
                dir_cell.alignment = CENTER
                dir_cell.border = BORDER
                dir_cell.fill = _delta_fill(delta) or BODY
                r += 1
        row = r + 1
    ws.freeze_panes = "A4"


def build_provenance(wb: Workbook, *, detailed: bool) -> None:
    ws = wb.create_sheet("Provenance")
    widths = {"A": 13, "B": 11, "C": 14, "D": 12, "E": 13, "F": 52, "G": 40, "H": 30, "I": 40}
    _widths(ws, widths)
    _title(ws, "Provenance — every headline number maps to a run, aggregation, eval view, and artifact", 9)
    _header_row(
        ws, 2,
        ["Experiment", "Dataset", "Modality", "Method", "Macro-F1", "Source run / checkpoint",
         "Aggregation / eval view", "Local artifact", "Verification (2026-08-06)"],
    )
    row = 3

    def put(exp, dataset, modality, method, value, source, agg, artifact, verified):
        nonlocal row
        values = [exp, dataset, modality, method, value, source, agg, artifact, verified]
        for col, v in enumerate(values, start=1):
            cell = ws.cell(row, col, v)
            cell.font = SMALL_FONT
            cell.alignment = WRAP if col in (6, 7, 8, 9) else LEFT
            if isinstance(v, float):
                cell.number_format = "0.0000"
        row += 1

    for dataset in DATASETS:
        run, agg, artifact = STANDALONE_QWEN_SOURCE[dataset]
        for modality in MODALITIES:
            put("Standalone", dataset, modality, "Fine-tuned Qwen",
                STANDALONE_QWEN[(dataset, modality)], run, agg, artifact,
                "recomputed from local artifact")
    for dataset in DATASETS:
        for modality in MODALITIES:
            logreg, xgb, optuna, os_ = STANDALONE_HEADS[(dataset, modality)]
            for value, method in ((logreg, "LogReg head"), (xgb, "XGBoost fixed")):
                if value is None:
                    continue
                put("Standalone heads", dataset, modality, method, value,
                    "hidden-state heads on frozen Qwen checkpoints (see docs)",
                    "pooled subject-level, 5-fold",
                    f"outputs/hidden_classifiers/{dataset.lower()}/ + docs/D3TEC_HIDDEN_CLASSIFIER_REPORT_2026-07-29.md / "
                    "reports/androids_hidden_classifier_*.md / outputs/daic_coverage_heads/ (MN5)",
                    "matched to audited hidden-classifier outputs")
            if detailed:
                for value, method in ((optuna, "XGBoost Optuna"), (os_, "XGBoost Subject OS (3-seed mean)")):
                    if value is None:
                        continue
                    if method.startswith("XGBoost Subject OS"):
                        put("Standalone heads", dataset, modality, method, value,
                            "Turkish subject-oversampling experiment, ratio 0.75, seeds 1337/2024/7",
                            "3-seed mean, 5-fold pooled",
                            "outputs/turkish_oversampling_* + docs/TURKISH_SUBJECT_OVERSAMPLING_REPORT_2026-07-25.md",
                            "mean of the three seed rows (internally consistent)")
                    else:
                        put("Standalone heads", dataset, modality, method, value,
                            "hidden-state heads on frozen Qwen checkpoints (see docs)",
                            "pooled subject-level, 5-fold",
                            f"outputs/hidden_classifiers/{dataset.lower()}/ + docs/D3TEC_HIDDEN_CLASSIFIER_REPORT_2026-07-29.md / "
                            "reports/androids_hidden_classifier_*.md / outputs/daic_coverage_heads/ (MN5)",
                            "matched to audited hidden-classifier outputs")

    merged_methods = ["qwen", "logreg", "xgb_fixed"] + (["xgb_optuna"] if detailed else [])

    for modality in ["audio_text", "audio_only", "text_only"]:
        run = MERGED_RUNS[modality]["run_id"]
        for method in merged_methods:
            value = MERGED_RUNS[modality]["official"].get(method)
            if value is None:
                put("Merged DAIC official", "DAIC", MODALITY_LABELS[modality], METHOD_LABELS[method], None,
                    run, "protected official holdout (47 subjects), full coverage",
                    f"outputs/symmetric_merged/{modality}/{run}/final/fold_0/",
                    "optuna not run on retrain checkpoints")
            else:
                put("Merged DAIC official", "DAIC", MODALITY_LABELS[modality], METHOD_LABELS[method], value,
                    run, "protected official holdout (47 subjects), full coverage",
                    f"outputs/symmetric_merged/{modality}/{run}/final/fold_0/",
                    "recomputed from prediction files")
        for (ds_key, method), value in sorted(MERGED_RUNS[modality]["cv"].items()):
            put("Merged CV", DATASET_LABELS[ds_key], MODALITY_LABELS[modality], METHOD_LABELS[method], value,
                run, "pooled subject-level 5-fold CV",
                f"outputs/symmetric_merged/{modality}/{run}/cv/fold_*/",
                "recomputed from prediction files")

    for dataset in DATASETS:
        for modality in MODALITIES:
            key = (dataset, modality)
            if key not in EN_DATA:
                continue
            native, translated = EN_DATA[key]
            put("EN translation", dataset, modality, "Native (fold-mean)", native,
                EN_NATIVE_RUNS[dataset][modality],
                "5-fold CV mean, teacher-forced, pos-F1 selection",
                "output_model/{audio_text,audio_only,text_only}/<dataset>/<run>/final_summary.json",
                "recomputed from local final_summary.json")
            put("EN translation", dataset, modality, "Translated EN (fold-mean)", translated,
                EN_EN_RUNS[dataset][modality],
                "5-fold CV mean, teacher-forced, pos-F1 selection, EN transcripts",
                "output_model_en/.../<run>/final_summary.json (MN5, not synced)",
                "matched to docs/ENGLISH_TRANSLATION_RESULTS_2026-08-06.md §2")

    for dataset in DATASETS:
        for modality in MODALITIES:
            key = (dataset, modality)
            if key not in EN_HEADS_NATIVE:
                continue
            nl, nx = EN_HEADS_NATIVE[key]
            el, ex = EN_HEADS_EN[key]
            put("EN heads", dataset, modality, "Native LogReg", nl,
                EN_NATIVE_RUNS[dataset][modality] + " (best-model hidden features)",
                "pooled 5-fold subject-level, logreg_raw",
                f"outputs/hidden_classifiers/{dataset.lower()}/.../<run>/fold_*/logreg_raw/",
                "recomputed from predictions (D3TEC rotary heads are new)")
            put("EN heads", dataset, modality, "EN LogReg", el,
                EN_EN_RUNS[dataset][modality] + " (best-model hidden features)",
                "pooled 5-fold subject-level, logreg_raw",
                f"outputs/hidden_classifiers/{dataset.lower()}/.../en_seq*/fold_*/logreg_raw/",
                "computed 2026-08-06, matrix configs/features/translation_en_matrix.yaml")
            put("EN heads", dataset, modality, "Native XGB", nx,
                EN_NATIVE_RUNS[dataset][modality] + " (best-model hidden features)",
                "pooled 5-fold subject-level, xgb_raw",
                f"outputs/hidden_classifiers/{dataset.lower()}/.../<run>/fold_*/xgb_raw/",
                "recomputed from predictions (D3TEC rotary heads are new)")
            put("EN heads", dataset, modality, "EN XGB", ex,
                EN_EN_RUNS[dataset][modality] + " (best-model hidden features)",
                "pooled 5-fold subject-level, xgb_raw",
                f"outputs/hidden_classifiers/{dataset.lower()}/.../en_seq*/fold_*/xgb_raw/",
                "computed 2026-08-06, matrix configs/features/translation_en_matrix.yaml")

    packed30_rows = [
        ("Packed30 v1 Audio + Text", "Qwen TF", 0.545,
         "daic_participant_p30_audio_text_s1337_05b52c6b (commit 3caa208)",
         "per-chunk mean teacher-forced score margin, official 47-subject test",
         "output_model/experiments/daic_participant_packed30/audio_text/<run>/fold_0/best_model/standalone_eval/"),
        ("Packed30 v1 Audio only", "Qwen TF", 0.468,
         "daic_participant_p30_audio_only_s1337_05b52c6b (commit 3caa208)",
         "per-chunk mean teacher-forced score margin, official 47-subject test",
         "output_model/experiments/daic_participant_packed30/audio_only/<run>/fold_0/best_model/standalone_eval/"),
        ("Joint-K4 Audio + Text", "Qwen TF", 0.7857,
         "daic_participant_p30_jointk4_audio_text_s1337_e3b0f1c3 (commit e3b0f1c; jobs 44365895/44365896/44365897/44365898, audit 44369723)",
         "per-bundle mean teacher-forced score margin, balanced K=4 coverage, official 47-subject test",
         "output_model/experiments/daic_participant_packed30_jointk4/audio_text/<run>/fold_0/best_model/standalone_eval/"),
        ("Joint-K4 Audio only", "Qwen TF", 0.4444,
         "daic_participant_p30_jointk4_audio_only_s1337_e3b0f1c3 (commit e3b0f1c; jobs 44365889/44365891/44365892/44365893, audit 44369722)",
         "per-bundle mean teacher-forced score margin, balanced K=4 coverage, official 47-subject test",
         "output_model/experiments/daic_participant_packed30_jointk4/audio_only/<run>/fold_0/best_model/standalone_eval/"),
    ]
    for condition, method, value, source, aggregation, artifact in packed30_rows:
        put("DAIC packed30 family", "DAIC", condition, method, value, source, aggregation,
            artifact + "metrics_original_teacher_forced.json + predictions_subject_level.csv",
            "recomputed from subject rows (matches metrics JSON); audits 44369722/44369723 PASSED on GPFS + local evidence audit")

    packed30_head_rows = [
        ("Canonical Audio + Text", "LogReg raw", 0.7647, "outputs/daic_coverage_heads/daic_coverage_heads_20260805_04f2e19/audio_text/classical/c2/logreg_raw/"),
        ("Canonical Audio + Text", "XGBoost raw", 0.6429, "outputs/daic_coverage_heads/daic_coverage_heads_20260805_04f2e19/audio_text/classical/c2/xgb_raw/"),
        ("Canonical Audio only", "LogReg raw", 0.5714, "outputs/daic_coverage_heads/daic_coverage_heads_20260805_04f2e19/audio_only/classical/c2/logreg_raw/"),
        ("Canonical Audio only", "XGBoost raw", 0.3636, "outputs/daic_coverage_heads/daic_coverage_heads_20260805_04f2e19/audio_only/classical/c2/xgb_raw/"),
        ("Joint-K4 Audio + Text", "LogReg raw", 0.7333, "hidden_classifiers/audio_text/logreg_raw/"),
        ("Joint-K4 Audio + Text", "XGBoost raw", 0.7692, "hidden_classifiers/audio_text/xgb_raw/"),
        ("Joint-K4 Audio only", "LogReg raw", 0.6471, "hidden_classifiers/audio_only/logreg_raw/"),
        ("Joint-K4 Audio only", "XGBoost raw", 0.2000, "hidden_classifiers/audio_only/xgb_raw/"),
    ]
    for condition, method, value, artifact in packed30_head_rows:
        if condition.startswith("Canonical"):
            source = ("daic_coverage_heads_20260805_04f2e19 canonical checkpoints "
                      "(audio_text=daic_main_k4_control_20260804_f26dd45, audio_only=daic_replicates_20ep_s1337_daic_audio_only_selposf1_tf)")
            aggregation = "mean depressed probability >= 0.5 per subject, 47 test subjects, complete-coverage (c2_balanced) view"
        else:
            source = "daic_participant_p30_jointk4_<mod>_s1337_e3b0f1c3 selected-epoch head fit (107 vectors, weights 1.0)"
            aggregation = "mean depressed probability >= 0.5 per subject, 47 test subjects, 617 bundles"
        put("DAIC packed30 family", "DAIC", condition, method, value, source, aggregation,
            artifact, "recomputed from predictions_subject_level.jsonl (metrics match)")

    _harmonized_provenance(ws, put)
    ws.freeze_panes = "A3"


# --------------------------------------------------------------------------- data
# DAIC participant-packed30 family (runtime participant-only chunks, seed 1337):
#   packed30_v1  = one chunk per prompt, all chunks subject-normalized (commit 3caa208)
#   packed30_jointk4 = joint K=4 random bundle per subject/epoch, balanced-cover
#                     (commit e3b0f1c, run daic_participant_p30_jointk4_<mod>_s1337_e3b0f1c3)
# Values are strict teacher-forced positive-F1 on the official 47-subject test
# (mean per-subject margin; INVALID counts as wrong; INVALID=0). Verified
# locally by recomputation from predictions_subject_level.csv.
PACKED30_QWEN: dict[tuple[str, str], float] = {
    ("DAIC", "Packed30 v1 Audio + Text"): 0.545,
    ("DAIC", "Packed30 v1 Audio only"): 0.468,
    ("DAIC", "Joint-K4 Audio + Text"): 0.7857,
    ("DAIC", "Joint-K4 Audio only"): 0.4444,
}
PACKED30_HEAD_POSF1: dict[tuple[str, str, str], float] = {
    ("DAIC", "Packed30 v1 Audio + Text", "LogReg"): 0.741,
    ("DAIC", "Packed30 v1 Audio + Text", "XGB"): 0.583,
    ("DAIC", "Packed30 v1 Audio only", "LogReg"): 0.333,
    ("DAIC", "Packed30 v1 Audio only", "XGB"): 0.435,
    ("DAIC", "Joint-K4 Audio + Text", "LogReg"): 0.7333,
    ("DAIC", "Joint-K4 Audio + Text", "XGB"): 0.7692,
    ("DAIC", "Joint-K4 Audio only", "LogReg"): 0.6471,
    ("DAIC", "Joint-K4 Audio only", "XGB"): 0.2000,
}
PACKED30_SOURCE = {
    "Packed30 v1 Audio + Text": ("daic_participant_p30_audio_text_s1337_05b52c6b (commit 3caa208)",
                                 "output_model/experiments/daic_participant_packed30/audio_text/"),
    "Packed30 v1 Audio only": ("daic_participant_p30_audio_only_s1337_05b52c6b (commit 3caa208)",
                               "output_model/experiments/daic_participant_packed30/audio_only/"),
    "Joint-K4 Audio + Text": ("daic_participant_p30_jointk4_audio_text_s1337_e3b0f1c3 (commit e3b0f1c)",
                              "output_model/experiments/daic_participant_packed30_jointk4/audio_text/"),
    "Joint-K4 Audio only": ("daic_participant_p30_jointk4_audio_only_s1337_e3b0f1c3 (commit e3b0f1c)",
                            "output_model/experiments/daic_participant_packed30_jointk4/audio_only/"),
}

HARMONIZED_RUN = "harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b"
HARMONIZED_ARTIFACT = (
    "output_model/harmonized_v1/<modality>/<dataset>/<run>[_r1]/fold_<n>/"
    "best_model/standalone_eval(_r1)/metrics_original_teacher_forced.json + predictions_subject_level.csv"
)
HARMONIZED_HEADS_ARTIFACT = (
    "outputs/hidden_classifiers/harmonized_v1/<dataset>/<run>_r1/fold_<n>/{logreg_raw,xgb_raw}/metrics.json"
)
HARMONIZED_AGG = {
    "DAIC": "fixed official test (47 subjects)",
    "CMDC": "5-fold mean",
    "Turkish": "5-fold mean",
    "D3TEC": "pooled subject-level 5-fold",
    "Androids Interview": "pooled subject-level 5-fold",
}


def _harmonized_provenance(ws, put) -> None:
    run = HARMONIZED_RUN
    modality_key = {"Audio + Text": "audio_text", "Audio only": "audio_only", "Text only": "text_only"}
    for dataset in DATASETS:
        for modality in MODALITIES:
            agg = HARMONIZED_AGG[dataset]
            mod_key = modality_key[modality]
            put("Harmonized standalone", dataset, modality, "Fine-tuned Qwen",
                HARMONIZED_STANDALONE_QWEN[(dataset, modality)],
                f"{run}_{dataset.lower().replace(' ', '_')}_{mod_key}"
                f" (campaign harmonized_v1_prod_20260809T171705Z_d1e8130b; first-wave failures replaced by _r1 retries, "
                f"retry_r1_jobs.tsv)",
                f"original_teacher_forced, headline/binary_strict, best_model; {agg}",
                HARMONIZED_ARTIFACT,
                "recomputed 2026-08-10 from local predictions/metrics")
            logreg, xgb = HARMONIZED_STANDALONE_HEADS[(dataset, modality)]
            heads_run = f"{run}_{dataset.lower().replace(' ', '_')}_{mod_key}"
            put("Harmonized heads", dataset, modality, "LogReg head", logreg,
                f"{heads_run} (best-model hidden features, retry _r1 feature dirs)",
                "5-fold mean subject-level, logreg_raw",
                HARMONIZED_HEADS_ARTIFACT,
                "recomputed 2026-08-10 from local fold metrics")
            put("Harmonized heads", dataset, modality, "XGBoost fixed", xgb,
                f"{heads_run} (best-model hidden features, retry _r1 feature dirs)",
                "5-fold mean subject-level, xgb_raw",
                HARMONIZED_HEADS_ARTIFACT,
                "recomputed 2026-08-10 from local fold metrics")
    for modality in ["audio_text", "audio_only", "text_only"]:
        mod_label = MODALITY_LABELS[modality]
        for method, method_label in (("qwen", "Fine-tuned Qwen"), ("logreg", "LogReg head"), ("xgb_fixed", "XGBoost fixed")):
            value = HARMONIZED_MERGED[modality]["official"].get(method)
            put("Harmonized merged final", "DAIC", mod_label, method_label, value,
                f"merged {mod_label}, final stage {HARMONIZED_MERGED[modality]['epochs']} epochs, run {run}",
                "protected official holdout (47 subjects), teacher-forced",
                f"outputs/symmetric_merged/harmonized_v1/{modality}/{run}/final/fold_0/postprocess_complete.json",
                "recomputed 2026-08-10 from local postprocess summary")
        for (ds_key, method), value in sorted(HARMONIZED_MERGED[modality]["cv"].items()):
            put("Harmonized merged CV", DATASET_LABELS[ds_key], mod_label, METHOD_LABELS[method], value,
                f"merged {mod_label} CV run {run}",
                "mean of per-fold holdout macro-F1, 5 folds, teacher-forced",
                f"outputs/symmetric_merged/harmonized_v1/{modality}/{run}/cv/fold_*/postprocess_complete.json + heads/*/metrics_by_dataset.json",
                "recomputed 2026-08-10 from local fold summaries")


def build_packed30(wb: Workbook) -> None:
    ws = wb.create_sheet("DAIC Packed30 Family")
    _widths(ws, {"A": 34, "B": 12, "C": 14, "D": 14, "E": 12, "F": 52, "G": 46})
    _title(ws, "DAIC runtime participant-packed30: canonical vs v1 vs joint-K4 (official 47-subject test, seed 1337, strict teacher-forced)", 7)
    _note(ws, 2, "Qwen verdict = mean teacher-forced score margin per subject (INVALID counts as wrong; INVALID=0). "
                 "Heads = logreg_raw / xgb_raw on hidden features, mean depressed probability >= 0.5 per subject. "
                 "Canonical heads = daic_coverage_heads_20260805_04f2e19 complete-coverage (c2_balanced) view. "
                 "Selected epochs: joint-K4 audio-only 5, audio+text 3; v1 audio-only 3, audio+text 2.", 7)
    _header_row(ws, 3, ["Condition", "Method", "Positive-F1", "Macro-F1", "AUROC (heads)", "Canonical joint-K4 (pos / macro / AUROC)", "Source run", "Local artifact"])
    row = 4
    canonical = {
        "Audio + Text": {"qwen": (0.800, 0.841, None), "logreg": (0.7647, 0.8157, 0.9177), "xgb": (0.6429, 0.7457, 0.8831)},
        "Audio only": {"qwen": (0.522, 0.683, None), "logreg": (0.5714, 0.6586, 0.8225), "xgb": (0.3636, 0.5846, 0.7814)},
    }
    conditions = [
        ("Packed30 v1 Audio + Text", "Audio + Text", {"qwen_pos": 0.545, "qwen_macro": 0.703, "logreg": (0.741, 0.818, 0.846), "xgb": (0.583, 0.720, 0.908)}),
        ("Packed30 v1 Audio only", "Audio only", {"qwen_pos": 0.468, "qwen_macro": 0.468, "logreg": (0.333, 0.510, 0.617), "xgb": (0.435, 0.626, 0.602)}),
        ("Joint-K4 Audio + Text", "Audio + Text", {"qwen_pos": 0.7857, "qwen_macro": 0.8474, "logreg": (0.7333, 0.8042, 0.9156), "xgb": (0.7692, 0.8405, 0.9134)}),
        ("Joint-K4 Audio only", "Audio only", {"qwen_pos": 0.4444, "qwen_macro": 0.5498, "logreg": (0.6471, 0.7235, 0.8074), "xgb": (0.2000, 0.4919, 0.7814)}),
    ]
    for condition, modality_key, values in conditions:
        source, artifact_root = PACKED30_SOURCE[condition]
        canon = canonical[modality_key]
        for method, value in (
            ("Qwen TF", (values["qwen_pos"], values["qwen_macro"], None)),
            ("LogReg raw", values["logreg"]),
            ("XGBoost raw", values["xgb"]),
        ):
            pos, macro, auroc = value
            canon_key = "qwen" if method == "Qwen TF" else "logreg" if method == "LogReg raw" else "xgb"
            canon_values = canon[canon_key]
            _body_cell(ws, row, 1, condition)
            _body_cell(ws, row, 2, method)
            _body_cell(ws, row, 3, pos, fmt="0.0000")
            _body_cell(ws, row, 4, macro, fmt="0.0000")
            _body_cell(ws, row, 5, auroc, fmt="0.0000")
            _body_cell(ws, row, 6, " / ".join("n/a" if v is None else f"{v:0.3f}" for v in canon_values))
            _body_cell(ws, row, 7, source)
            suffix = "logreg_raw/metrics.json" if method == "LogReg raw" else "xgb_raw/metrics.json" if method == "XGBoost raw" else "standalone_eval/metrics_original_teacher_forced.json"
            _body_cell(ws, row, 8, f"{artifact_root}<run>/fold_0/best_model/{suffix} (recomputed from predictions)")
            row += 1
        row += 1
    _note(ws, row, "Canonical = preprocessed joint-K4 (Qwen: 2026-08-05 coverage validation; heads: "
                   "daic_coverage_heads_20260805_04f2e19 c2_balanced, verified locally by recomputation). "
                   "Provenance sheet holds the full mapping (commit, job IDs incl. resubmits, eval view, aggregation, "
                   "audits 44369722/44369723 GPFS PASSED + local evidence audit PASSED).", 7)
    ws.freeze_panes = "A4"


# --------------------------------------------------------------------------- harmonized campaign sheet
def build_harmonized(wb: Workbook) -> None:
    """Harmonized full-training campaign (2026-08-09/10) summary sheet."""
    ws = wb.create_sheet("Harmonized Campaign")
    _widths(ws, {"A": 30, "B": 15, "C": 15, "D": 15, "E": 17})
    _title(ws, "Harmonized Campaign (recipe harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1) — Macro-F1", 5)
    _note(
        ws, 2,
        "Campaign harmonized_v1_prod_20260809T171705Z_d1e8130b (Issue #12 / PR #10); teacher-forced, binary-strict, "
        "best_model, seed 1337; 20-epoch cap, patience 3, macro-F1 selection; audio encoder frozen; no Optuna. "
        "Standalone aggregations: D3TEC/Androids pooled subject-level 5-fold; Turkish/CMDC 5-fold mean; DAIC fixed "
        "official test (47 subjects). First wave had two documented failure classes (NCCL watchdog on long final "
        "evals; standalone-eval KeyError), fixed in PRs #14/#16 and rerun under _r1 attempt names; retry registry "
        "retry_r1_jobs.tsv (148 jobs, all COMPLETED). Merged: five-dataset CV (holdout per fold, 5-fold mean) and "
        "final training on the DAIC protected official test only (daic_official_test_only), final epochs 2/4/5 "
        "(rounded median of CV-selected epochs). Optuna/Subject-OS not run -> blank. All values recomputed "
        "2026-08-10 from local artifacts; per-cell provenance on the Provenance sheet.",
        5, height=110,
    )
    methods = [("qwen", "Fine-tuned Qwen"), ("logreg", "LogReg head"), ("xgb_fixed", "XGBoost fixed")]
    headers = ["Evaluation / Modality", *(label for _, label in methods)]
    _header_row(ws, 4, headers)

    row = 5
    _section(ws, row, "Standalone — Qwen (headline macro-F1)", 4)
    row += 1
    _header_row(ws, row, headers)
    row += 1
    for dataset in DATASETS:
        for modality in MODALITIES:
            ws.cell(row, 1, f"{dataset} — {modality}").font = BODY_FONT
            ws.cell(row, 1).fill = BODY
            ws.cell(row, 1).alignment = LEFT
            ws.cell(row, 1).border = BORDER
            _body_cell(ws, row, 2, HARMONIZED_STANDALONE_QWEN[(dataset, modality)], fmt="0.0000")
            logreg, xgb = HARMONIZED_STANDALONE_HEADS[(dataset, modality)]
            _body_cell(ws, row, 3, logreg, fmt="0.0000")
            _body_cell(ws, row, 4, xgb, fmt="0.0000")
            row += 1

    row += 1
    _section(ws, row, "Merged final — DAIC protected official test (n=47)", 4)
    row += 1
    _header_row(ws, row, headers)
    row += 1
    for modality, modality_label in zip(["audio_text", "audio_only", "text_only"], MODALITIES):
        ws.cell(row, 1, f"Merged {modality_label} (final, {HARMONIZED_MERGED[modality]['epochs']} epochs)").font = BODY_FONT
        ws.cell(row, 1).fill = BODY
        ws.cell(row, 1).alignment = LEFT
        ws.cell(row, 1).border = BORDER
        for col, (method, _) in enumerate(methods, start=2):
            _body_cell(ws, row, col, HARMONIZED_MERGED[modality]["official"].get(method), fmt="0.0000")
        row += 1

    row += 1
    _section(ws, row, "Merged five-dataset CV — mean of per-fold holdout macro-F1", 4)
    row += 1
    _header_row(ws, row, headers)
    row += 1
    for modality in ["audio_text", "audio_only", "text_only"]:
        for dataset in ["daic", "cmdc", "turkish", "d3tec", "androids_interview"]:
            label = f"{DATASET_LABELS[dataset]} — {MODALITY_LABELS[modality]} (merged CV)"
            ws.cell(row, 1, label).font = BODY_FONT
            ws.cell(row, 1).fill = BODY
            ws.cell(row, 1).alignment = LEFT
            ws.cell(row, 1).border = BORDER
            for col, (method, _) in enumerate(methods, start=2):
                _body_cell(ws, row, col, HARMONIZED_MERGED[modality]["cv"].get((dataset, method)), fmt="0.0000")
            row += 1
    ws.freeze_panes = "A5"


# --------------------------------------------------------------------------- main
# --------------------------------------------------------------------------- validation
def validate_selected_results(selected_results: Path, cell_values: dict[tuple[str, str], float | None]) -> tuple[bool, list[str]]:
    """Cross-check an explicit selected-results export against workbook headline cells.

    The workbook stays script-only: values are compared, never overwritten. Missing
    records are listed as legacy-unmigrated; nothing is invented and nothing is zeroed.
    """
    payload = json.loads(Path(selected_results).read_text(encoding="utf-8"))
    selections = payload.get("selections", [])
    legacy_unmigrated: list[str] = []
    mismatches: list[str] = []
    checked = 0
    for selection in selections:
        cell = str(selection.get("cell"))
        dataset_label, modality_label = cell.split("|", 1)
        key = (dataset_label.strip(), modality_label.strip())
        status = selection.get("status")
        if status != "selected":
            legacy_unmigrated.append(f"{cell}: {status} ({selection.get('reason', 'no record')})")
            continue
        expected = cell_values.get(key)
        if expected is None:
            legacy_unmigrated.append(f"{cell}: no workbook cell for selected record")
            continue
        value = selection.get("value")
        if value is None:
            mismatches.append(f"{cell}: selected record has null value")
            continue
        if expected is not None and abs(float(value) - float(expected)) > 1e-6 * max(1.0, abs(float(expected))):
            mismatches.append(
                f"{cell}: registry {float(value):.6f} differs from workbook {float(expected):.6f}"
            )
        checked += 1
    report = [
        *mismatches,
        *[f"legacy-unmigrated: {entry}" for entry in legacy_unmigrated],
    ]
    return (not mismatches, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true",
                        help="include XGBoost Optuna and Subject-OS columns/rows; write the detailed workbook")
    parser.add_argument("--validate-selected", default=None,
                        help="cross-check an explicit selected-results JSON against workbook cells; never rewrites cells")
    args = parser.parse_args()
    detailed = args.detailed

    if args.validate_selected is not None:
        cell_values: dict[tuple[str, str], float | None] = {
            (dataset_label, modality_label): value
            for (dataset_label, modality_label), value in STANDALONE_QWEN.items()
        }
        ok, report = validate_selected_results(Path(args.validate_selected), cell_values)
        print("\n".join(report))
        print(f"checked={sum(1 for s in json.loads(Path(args.validate_selected).read_text())['selections'] if s['status'] == 'selected')} "
              f"mismatches={sum(1 for line in report if 'differs from' in line)} "
              f"legacy_unmigrated={sum(1 for line in report if line.startswith('legacy-unmigrated:'))}")
        return 0 if ok else 1

    wb = Workbook()
    wb.remove(wb.active)
    build_summary(wb, detailed=detailed)
    build_merged_summary(wb, detailed=detailed)
    build_en(wb)
    build_merged_vs_standalone(wb)
    build_packed30(wb)
    build_harmonized(wb)
    build_provenance(wb, detailed=detailed)
    out = OUT_DETAILED if detailed else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
