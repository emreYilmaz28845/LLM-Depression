"""Unit tests for the canonical UAR field (binary_strict_uar).

UAR (unweighted average recall, balanced accuracy) is reported alongside
Macro-F1 and Positive-F1. It is the unweighted mean of the two class recalls
under the strict convention: an invalid output counts as wrong for its true
class. These tests pin the aggregation, qualification, and validation paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.aggregate import (
    _metrics_from_prediction_rows,
    aggregate_likelihood_predictions,
    aggregate_mean_probability_predictions,
)
from src.experiment_tracking.qualification import _headline_metrics
from src.experiment_tracking.validate import recompute_strict_headline
from src.utils import AGGREGATION_LEVEL_SUBJECT, PREDICTION_MODE_ORIGINAL_TEACHER_FORCED


def test_strict_uar_equals_macro_recall_and_counts_invalid_as_wrong():
    rows = [
        {"subject_id": "s1", "label": 1, "prediction": 1},   # tp
        {"subject_id": "s2", "label": 1, "prediction": 0},   # fn
        {"subject_id": "s3", "label": 0, "prediction": 0},   # tn
        {"subject_id": "s4", "label": 0, "prediction": 1},   # fp
        {"subject_id": "s5", "label": 0, "prediction": -1},  # invalid -> fp
    ]
    metrics = _metrics_from_prediction_rows(
        rows,
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    assert metrics["binary_strict_uar"] == metrics["macro_recall"]
    # positive recall 1/2, negative recall 1/3 (the invalid row is a false positive)
    assert metrics["binary_strict_uar"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_zero_row_aggregation_reports_zero_uar():
    metrics = _metrics_from_prediction_rows(
        [],
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    assert metrics["binary_strict_uar"] == 0.0


def test_likelihood_aggregation_records_uar():
    rows = [
        {"subject_id": "a", "label": 1, "dep_score": 2.0, "non_score": 0.0},
        {"subject_id": "a", "label": 1, "dep_score": 2.0, "non_score": 0.0},
        {"subject_id": "b", "label": 0, "dep_score": 0.5, "non_score": 2.0},
    ]
    _, metrics = aggregate_likelihood_predictions(rows)
    assert metrics["binary_strict_uar"] == metrics["macro_recall"]
    assert metrics["binary_strict_uar"] == pytest.approx(1.0)


def test_mean_probability_aggregation_records_uar():
    rows = [
        {"subject_id": "a", "label": 1, "probability": 0.9},
        {"subject_id": "b", "label": 0, "probability": 0.2},
    ]
    _, metrics = aggregate_mean_probability_predictions(rows)
    assert metrics["binary_strict_uar"] == metrics["macro_recall"]
    assert metrics["binary_strict_uar"] == pytest.approx(1.0)


def test_qualification_extracts_uar_when_present():
    metrics = _headline_metrics(
        {"binary_strict_macro_f1": 0.5, "binary_strict_uar": 0.6, "num_units": 10}
    )
    by_name = {metric.name: metric.value for metric in metrics}
    assert by_name["uar"] == 0.6
    # Older metrics files without the key are tolerated (metric skipped).
    old = _headline_metrics({"binary_strict_macro_f1": 0.5, "num_units": 10})
    assert all(metric.name != "uar" for metric in old)


def test_recompute_strict_headline_reports_uar(tmp_path: Path):
    path = tmp_path / "predictions_subject_level.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "label", "prediction_text"])
        writer.writeheader()
        writer.writerow({"subject_id": "s1", "label": 1, "prediction_text": "Depressed"})
        writer.writerow({"subject_id": "s2", "label": 1, "prediction_text": "Non-depressed"})
        writer.writerow({"subject_id": "s3", "label": 0, "prediction_text": "Non-depressed"})
        writer.writerow({"subject_id": "s4", "label": 0, "prediction_text": "gibberish"})
    recomputed = recompute_strict_headline(path)
    # tp=1, fn=1, tn=1, fp=1 -> both recalls 0.5 -> UAR 0.5
    assert recomputed["binary_strict_uar"] == pytest.approx(0.5)
