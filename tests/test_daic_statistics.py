"""Tests for the DAIC paired statistics module.

Covers the exact McNemar test, the stratified paired bootstrap, the
subject-level paired prediction-swap permutation test, and the Holm correction.
"""

from __future__ import annotations

import pytest

from src.daic_statistics import (
    exact_mcnemar,
    holm_adjust,
    paired_prediction_swap_permutation,
    stratified_paired_bootstrap,
)
from src.metrics import classification_metrics


def _rows(predictions, labels, seed=None):
    rows = []
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        row = {"subject_id": f"s{index}", "label": int(label), "prediction": int(prediction)}
        if seed is not None:
            row["seed"] = seed
        rows.append(row)
    return rows


LABELS = [1] * 20 + [0] * 20


def test_permutation_identical_predictions_is_never_significant():
    perfect = _rows(LABELS, LABELS)
    other = _rows(LABELS, LABELS)
    result = paired_prediction_swap_permutation(perfect, other, iterations=200, seed=1337)
    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["subjects"] == 40 and result["keys"] == 40


def test_permutation_detects_a_large_effect():
    perfect = _rows(LABELS, LABELS)
    broken = _rows([0] * 20 + [1] * 20, LABELS)  # every subject wrong
    result = paired_prediction_swap_permutation(perfect, broken, iterations=500, seed=1337)
    assert result["observed_delta"] < -0.9
    assert result["p_value"] <= 0.05


def test_permutation_is_deterministic_for_a_seed():
    left = _rows([1, 1, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0])
    right = _rows([1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0])
    first = paired_prediction_swap_permutation(left, right, iterations=300, seed=7)
    second = paired_prediction_swap_permutation(left, right, iterations=300, seed=7)
    assert first == second
    assert first["null_mean"] == pytest.approx(second["null_mean"])


def test_permutation_averages_seed_deltas():
    labels = [1, 0, 1, 0]
    seed7_left = _rows([1, 0, 1, 0], labels, seed=7)
    seed7_right = _rows([1, 0, 0, 0], labels, seed=7)
    seed1337_left = _rows([1, 0, 1, 0], labels, seed=1337)
    seed1337_right = _rows([1, 1, 1, 0], labels, seed=1337)
    result = paired_prediction_swap_permutation(
        seed7_left + seed1337_left, seed7_right + seed1337_right, iterations=200, seed=1337
    )
    expected = (
        classification_metrics(labels, [1, 0, 0, 0])["macro_f1"]
        - classification_metrics(labels, [1, 0, 1, 0])["macro_f1"]
        + classification_metrics(labels, [1, 1, 1, 0])["macro_f1"]
        - classification_metrics(labels, [1, 0, 1, 0])["macro_f1"]
    ) / 2
    assert result["observed_delta"] == pytest.approx(expected)
    assert result["keys"] == 8 and result["subjects"] == 4


def test_permutation_requires_matching_keys_and_labels():
    left = _rows([1, 0], [1, 0])
    shorter = _rows([1], [1])
    with pytest.raises(ValueError, match="identical subject/seed keys"):
        paired_prediction_swap_permutation(left, shorter, iterations=10)
    relabeled = [{"subject_id": "s0", "label": 0, "prediction": 1},
                 {"subject_id": "s1", "label": 0, "prediction": 0}]
    with pytest.raises(ValueError, match="label mismatch"):
        paired_prediction_swap_permutation(left, relabeled, iterations=10)
    duplicated = left + left[:1]
    with pytest.raises(ValueError, match="one row per subject/seed key"):
        paired_prediction_swap_permutation(duplicated, left, iterations=10)


def test_permutation_treats_invalid_as_wrong_and_rejects_bad_arguments():
    labels = [1, 0, 1, 0]
    invalid = [{"subject_id": f"s{i}", "label": label, "prediction": -1}
               for i, label in enumerate(labels)]
    mapped = [{"subject_id": f"s{i}", "label": label, "prediction": 1 - label}
              for i, label in enumerate(labels)]
    invalid_result = paired_prediction_swap_permutation(_rows(labels, labels), invalid, iterations=50)
    mapped_result = paired_prediction_swap_permutation(_rows(labels, labels), mapped, iterations=50)
    assert invalid_result["observed_delta"] == pytest.approx(mapped_result["observed_delta"])
    with pytest.raises(ValueError, match="Unsupported permutation metric"):
        paired_prediction_swap_permutation(_rows(labels, labels), _rows(labels, labels),
                                           metric="not_a_metric", iterations=10)
    with pytest.raises(ValueError, match="iterations must be positive"):
        paired_prediction_swap_permutation(_rows(labels, labels), _rows(labels, labels), iterations=0)


def test_exact_mcnemar_counts_and_p_value():
    # Five discordant subjects, all correct only for the comparison side.
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    baseline = _rows([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], labels)
    comparison = _rows([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], labels)
    result = exact_mcnemar(baseline, comparison)
    assert result["baseline_only_correct"] == 0
    assert result["comparison_only_correct"] == 5
    assert result["p_value"] == pytest.approx(2 * (1 / 2**5))
    with pytest.raises(ValueError, match="identical subject keys"):
        exact_mcnemar(baseline, comparison[:3])


def test_stratified_paired_bootstrap_identical_inputs():
    rows = _rows([1, 1, 0, 0], [1, 0, 1, 0])
    result = stratified_paired_bootstrap(rows, list(rows), metric="macro_f1", iterations=200, seed=1337)
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    with pytest.raises(ValueError, match="Unsupported bootstrap metric"):
        stratified_paired_bootstrap(rows, list(rows), metric="not_a_metric", iterations=10)
    with pytest.raises(ValueError, match="both labels"):
        one_class = _rows([1, 1], [1, 1])
        stratified_paired_bootstrap(one_class, list(one_class), iterations=10)


def test_stratified_paired_bootstrap_requires_matching_keys_and_labels():
    left = _rows([1, 0, 1, 0], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="identical subject/seed keys"):
        stratified_paired_bootstrap(left, left[:2], iterations=10)
    relabeled = [{"subject_id": "s0", "label": 0, "prediction": 1},
                 {"subject_id": "s1", "label": 0, "prediction": 0},
                 {"subject_id": "s2", "label": 1, "prediction": 1},
                 {"subject_id": "s3", "label": 0, "prediction": 0}]
    with pytest.raises(ValueError, match="label mismatch"):
        stratified_paired_bootstrap(left, relabeled, iterations=10)


def test_stratified_paired_bootstrap_treats_invalid_as_wrong():
    labels = [1, 0, 1, 0, 1, 0]
    reference = _rows([1, 0, 1, 0, 1, 0], labels)
    invalid = [{"subject_id": f"s{i}", "label": label, "prediction": -1}
               for i, label in enumerate(labels)]
    mapped = [{"subject_id": f"s{i}", "label": label, "prediction": 1 - label}
              for i, label in enumerate(labels)]
    from_invalid = stratified_paired_bootstrap(reference, invalid, iterations=100, seed=1337)
    from_mapped = stratified_paired_bootstrap(reference, mapped, iterations=100, seed=1337)
    assert from_invalid == pytest.approx(from_mapped)


def test_holm_adjust_steps_down_and_validates():
    assert holm_adjust([0.01, 0.03, 0.04]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == pytest.approx([0.5])
    with pytest.raises(ValueError, match="p-values in"):
        holm_adjust([1.2])
