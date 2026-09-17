#!/usr/bin/env python3
"""Derive the canonical likelihood headline values from saved teacher-forced candidate scores.

This implements the canonical likelihood switch for the workbook: every canonical
standalone cell is recomputed from the per-subject candidate scores that the
existing teacher-forced runs already saved. The decision rule is
``argmax(mean dep_score, mean non_score)`` per subject (the Turkish pooled text
cells carry the locked pair margin instead); CV cells report the unweighted mean
of the per-fold strict subject-level metrics, DAIC reports its single fixed test
fold. No retraining and no new evaluation jobs are involved.

The recomputed teacher-forced values are checked against the workbook's current
teacher-forced anchors; a mismatch fails the derivation instead of adjusting it.
The JSON output is the citable artifact for the workbook's provenance strings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b"
POOLED_REPORT = (
    PROJECT_ROOT
    / "outputs/turkish_pooled_qcond/exp-turkish-pooled-qcond-clean-v1-20260903/production/report_v2/report.json"
)
OUTPUT = PROJECT_ROOT / "outputs/experiment_reports/likelihood_canonical_values/derived_values.json"

PREFERENCE = (
    "best_model/standalone_eval/predictions_subject_level.csv",
    "best_model/standalone_eval_r1/predictions_subject_level.csv",
    "eval/best_validation/predictions_subject_level.csv",
)

# (label, dataset_dir, run_dataset, modality_dir, run_suffixes)
NATIVE_CELLS = (
    ("DAIC", "daic", "daic", "audio_text", ("_r1",)),
    ("DAIC", "daic", "daic", "audio_only", ("_r1",)),
    ("DAIC", "daic", "daic", "text_only", ("_r1",)),
    ("CMDC", "cmdc", "cmdc", "audio_text", ("_r1", "")),
    ("CMDC", "cmdc", "cmdc", "audio_only", ("_r1", "")),
    ("CMDC", "cmdc", "cmdc", "text_only", ("_r1", "")),
    ("D3TEC", "d3tec", "d3tec", "audio_text", ("_r1", "")),
    ("D3TEC", "d3tec", "d3tec", "audio_only", ("", "_r1")),
    ("D3TEC", "d3tec", "d3tec", "text_only", ("_r1", "")),
    ("Androids Interview", "androids", "androids_interview", "audio_text", ("", "_r1")),
    ("Androids Interview", "androids", "androids_interview", "audio_only", ("_r1", "")),
    ("Androids Interview", "androids", "androids_interview", "text_only", ("_r1", "")),
)

# The workbook's current canonical teacher-forced values, used as anchors.
TF_ANCHORS = {
    ("DAIC", "audio_text"): (0.735279, 0.645161, 0.751082),
    ("DAIC", "audio_only"): (0.539216, 0.411765, 0.553030),
    ("DAIC", "text_only"): (0.735279, 0.645161, 0.751082),
    ("CMDC", "audio_text"): (0.970000, 0.960000, 0.970000),
    ("CMDC", "audio_only"): (0.951600, 0.931818, 0.950000),
    ("CMDC", "text_only"): (0.971300, 0.963636, 0.980000),
    ("D3TEC", "audio_text"): (0.540198, 0.491259, 0.551905),
    ("D3TEC", "audio_only"): (0.618056, 0.553074, 0.665714),
    ("D3TEC", "text_only"): (0.557602, 0.550125, 0.608095),
    ("Androids Interview", "audio_text"): (0.856965, 0.878135, 0.885726),
    ("Androids Interview", "audio_only"): (0.862800, 0.883374, 0.884211),
    ("Androids Interview", "text_only"): (0.723957, 0.785324, 0.769083),
}
POOLED_TF_ANCHORS = {
    "Q01": (0.408934, 0.817868),
    "Q02": (0.417359, 0.737282),
    "Q03": (0.437851, 0.795701),
    "Q04": (0.687717, 0.813004),
    "Q05": (0.740398, 0.846364),
}
POOLED_CELL_LABELS = {
    "Q01": ("Turkish", "Audio only", "native"),
    "Q02": ("Turkish", "Text only", "native"),
    "Q03": ("Turkish", "Text only", "english"),
    "Q04": ("Turkish", "Audio + Text", "native"),
    "Q05": ("Turkish", "Audio + Text", "english"),
}


def strict_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Strict subject-level metrics; an INVALID prediction counts as wrong."""
    yp = [p if p in (0, 1) else (1 - t) for t, p in zip(y_true, y_pred)]
    tp = sum(1 for t, p in zip(y_true, yp) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, yp) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, yp) if t == 0 and p == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    return {
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "positive_f1": positive_f1,
        "uar": (recall + negative_recall) / 2,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def likelihood_predictions(rows: list[dict[str, str]]) -> list[int]:
    """Per-subject likelihood decisions from the saved candidate scores."""
    if rows and rows[0].get("dep_score"):
        return [
            1 if float(row["dep_score"]) > float(row["non_score"]) else 0 for row in rows
        ]
    return [1 if float(row["pair_margin"]) > 0 else 0 for row in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_metrics(folds: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(fold[key] for fold in folds) / len(folds) for key in folds[0]}


def derive_native_cell(
    label: str, dataset_dir: str, run_dataset: str, modality: str, run_suffixes: tuple[str, ...]
) -> dict[str, Any]:
    run_dirs = [
        PROJECT_ROOT
        / "output_model/harmonized_v1"
        / modality
        / dataset_dir
        / f"{CAMPAIGN}_{run_dataset}_{modality}{suffix}"
        for suffix in run_suffixes
    ]
    fold_indices = sorted(
        {
            int(fold_dir.name.split("_")[1])
            for run_dir in run_dirs
            if run_dir.is_dir()
            for fold_dir in run_dir.glob("fold_*")
        }
    )
    if not fold_indices:
        raise SystemExit(f"no folds under {[str(path) for path in run_dirs]}")
    tf_folds: list[dict[str, float]] = []
    lh_folds: list[dict[str, float]] = []
    files: list[dict[str, str]] = []
    for fold_index in fold_indices:
        path = None
        for run_dir in run_dirs:
            fold_dir = run_dir / f"fold_{fold_index}"
            for relative in PREFERENCE:
                candidate = fold_dir / relative
                if candidate.is_file():
                    path = candidate
                    break
            if path is not None:
                break
        if path is None:
            raise SystemExit(f"no subject predictions for fold {fold_index} in {[str(path) for path in run_dirs]}")
        rows = read_rows(path)
        y_true = [int(row["label"]) for row in rows]
        tf_pred = [int(row["prediction"]) for row in rows]
        tf_folds.append(strict_metrics(y_true, tf_pred))
        lh_folds.append(strict_metrics(y_true, likelihood_predictions(rows)))
        files.append({"path": str(path), "sha256": sha256_file(path)})
    tf = mean_metrics(tf_folds)
    anchor = TF_ANCHORS[(label, modality)]
    # The workbook anchors are stored at mixed precision (some rounded to 4 decimals).
    if any(abs(tf[key] - anchor[i]) > 5e-5 for i, key in enumerate(("macro_f1", "positive_f1", "uar"))):
        raise SystemExit(f"{label}/{modality}: teacher-forced recomputation {tf} != anchor {anchor}")
    return {
        "dataset": label,
        "modality": modality,
        "pooled_cell": None,
        "transcript_condition": None,
        "aggregation": "single_test_fold" if label == "DAIC" else "five_fold_mean",
        "teacher_forced": tf,
        "likelihood": mean_metrics(lh_folds),
        "files": files,
    }


def derive_pooled_cell(cell_id: str, index: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    entries = [
        entry
        for entry in index.values()
        if entry.get("cell_id") == cell_id
        and entry.get("seed") == 1337
        and str(entry.get("backend")) == "original_teacher_forced"
        and entry.get("backbone") == "qwen"
    ]
    if len(entries) != 5:
        raise SystemExit(f"{cell_id}: expected 5 fold entries, found {len(entries)}")
    tf_folds: list[dict[str, float]] = []
    lh_folds: list[dict[str, float]] = []
    files: list[dict[str, str]] = []
    for entry in sorted(entries, key=lambda item: item["fold"]):
        path = Path(entry["evaluation_predictions_artifact"]["path"])
        rows = read_rows(path)
        y_true = [int(row["label"]) for row in rows]
        tf_folds.append(strict_metrics(y_true, [int(row["prediction"]) for row in rows]))
        lh_folds.append(strict_metrics(y_true, likelihood_predictions(rows)))
        files.append({"path": str(path), "sha256": sha256_file(path)})
    tf = mean_metrics(tf_folds)
    report_row = next(
        row
        for row in report["tables"]["seed_results"]
        if row.get("cell_id") == cell_id and row.get("seed") == 1337 and row.get("route") == "teacher_forced"
    )
    anchor = POOLED_TF_ANCHORS[cell_id]
    if (
        abs(tf["macro_f1"] - anchor[0]) > 1e-6
        or abs(tf["positive_f1"] - anchor[1]) > 1e-6
        or abs(tf["macro_f1"] - float(report_row["combined_macro_f1_fold_mean"])) > 1e-6
    ):
        raise SystemExit(f"{cell_id}: pooled teacher-forced recomputation {tf} != anchors")
    label, modality, transcript = POOLED_CELL_LABELS[cell_id]
    return {
        "dataset": label,
        "modality": modality,
        "pooled_cell": cell_id,
        "transcript_condition": transcript,
        "aggregation": "five_fold_mean",
        "teacher_forced": tf,
        "likelihood": mean_metrics(lh_folds),
        "files": files,
    }


def main() -> int:
    index = json.loads((POOLED_REPORT.parent / "provenance_index.json").read_text(encoding="utf-8"))
    report = json.loads(POOLED_REPORT.read_text(encoding="utf-8"))
    cells = [derive_native_cell(*cell) for cell in NATIVE_CELLS]
    cells.extend(derive_pooled_cell(cell_id, index, report) for cell_id in POOLED_TF_ANCHORS)
    payload = {
        "schema_version": "audiollm.likelihood_canonical_values.v1",
        "description": (
            "Canonical likelihood values derived from the saved per-subject candidate scores of the "
            "teacher-forced runs (argmax of mean dep_score/non_score per subject; Turkish pooled text "
            "cells use the locked pair margin). CV cells are unweighted fold means; DAIC is its fixed test fold."
        ),
        "campaign": CAMPAIGN,
        "pooled_report": str(POOLED_REPORT),
        "cells": cells,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    for cell in cells:
        name = cell["pooled_cell"] or f"{cell['dataset']}/{cell['modality']}"
        tf = cell["teacher_forced"]
        lh = cell["likelihood"]
        print(
            f"{name:24s} tf={tf['macro_f1']:.6f}/{tf['positive_f1']:.6f}/{tf['uar']:.6f} -> "
            f"likelihood={lh['macro_f1']:.6f}/{lh['positive_f1']:.6f}/{lh['uar']:.6f}"
        )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
