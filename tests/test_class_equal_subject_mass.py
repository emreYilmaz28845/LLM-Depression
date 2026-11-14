"""Tests for the class-equal subject-mass loss weighting option."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from src.data.androids import apply_hierarchical_training_weights
from src.train import (
    CLASS_EQUAL_SUBJECT_MASS,
    _build_weighted_train_sampler,
    apply_class_equal_subject_mass,
    apply_subject_normalized_chunk_weights,
)


def _chunks(spec: dict[str, tuple[int, int]]) -> list[dict]:
    """spec: subject_id -> (label, number_of_chunks)."""
    rows = []
    for subject_id, (label, count) in spec.items():
        for index in range(count):
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "sample_id": f"{subject_id}_c{index}",
                    "response_id": f"{subject_id}_r{index % 2}",
                }
            )
    return rows


def _mass_by(rows, key):
    totals = defaultdict(float)
    for row in rows:
        totals[row[key]] += float(row["loss_weight"])
    return totals


# Imbalanced like DAIC: 1 depressed subject, 3 non-depressed, unequal chunk counts.
SPEC = {"d1": (1, 5), "n1": (0, 2), "n2": (0, 7), "n3": (0, 3)}


def test_subject_normalized_then_class_equal_balances_classes():
    weighted, _ = apply_subject_normalized_chunk_weights(_chunks(SPEC))
    before = _mass_by(weighted, "label")
    assert before[0] == pytest.approx(3 * before[1])  # class prior kept before balancing

    balanced, audit = apply_class_equal_subject_mass(weighted)
    after = _mass_by(balanced, "label")
    assert after[0] == pytest.approx(after[1])
    mean = sum(float(r["loss_weight"]) for r in balanced) / len(balanced)
    assert math.isclose(mean, 1.0, abs_tol=1e-9)
    assert audit["subjects_per_class"] == {"0": 3, "1": 1}

    # Subjects of the same class still carry equal mass.
    subject_mass = _mass_by(balanced, "subject_id")
    assert subject_mass["n1"] == pytest.approx(subject_mass["n2"])
    assert subject_mass["n2"] == pytest.approx(subject_mass["n3"])
    # Within-class ratios are unchanged; the depressed factor is 3x the non-depressed one.
    for before_row, after_row in zip(weighted, balanced):
        assert after_row["pre_class_balance_loss_weight"] == before_row["loss_weight"]
    ratio = {r["label"]: r["loss_weight"] / r["pre_class_balance_loss_weight"] for r in balanced}
    assert ratio[1] / ratio[0] == pytest.approx(3.0)


def test_hierarchical_then_class_equal_balances_classes():
    weighted, _ = apply_hierarchical_training_weights(_chunks(SPEC))
    balanced, _ = apply_class_equal_subject_mass(weighted)
    after = _mass_by(balanced, "label")
    assert after[0] == pytest.approx(after[1])


def test_already_balanced_is_unchanged():
    spec = {"d1": (1, 3), "d2": (1, 4), "n1": (0, 2), "n2": (0, 5)}
    weighted, _ = apply_subject_normalized_chunk_weights(_chunks(spec))
    balanced, _ = apply_class_equal_subject_mass(weighted)
    for a, b in zip(weighted, balanced):
        assert b["loss_weight"] == pytest.approx(a["loss_weight"])


def test_requires_both_classes_and_weights():
    weighted, _ = apply_subject_normalized_chunk_weights(_chunks({"n1": (0, 2), "n2": (0, 3)}))
    with pytest.raises(ValueError, match="both classes"):
        apply_class_equal_subject_mass(weighted)
    with pytest.raises(ValueError, match="loss_weight"):
        apply_class_equal_subject_mass(_chunks(SPEC))


def test_sampler_not_used_for_class_equal_mode():
    config = {"seed": 1, "training": {"class_balance": CLASS_EQUAL_SUBJECT_MASS}}
    assert _build_weighted_train_sampler(_chunks(SPEC), config) is None
