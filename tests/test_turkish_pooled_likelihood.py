"""Unit tests for the Turkish pooled-text likelihood pair rule.

The four pooled text-only configs declare
``subject_score_aggregation: turkish_pooled_text_pair_mean_margin_strict_v1``.
In likelihood mode the same two-condition pair rule must apply over
candidate-score margins: the pair margin is the mean of the two condition
margins, an exact tie stays INVALID and counts as wrong, and a declared
policy must never be silently dropped.
"""

from __future__ import annotations

import pytest

from src.aggregate import (
    TURKISH_POOLED_TEXT_PAIR_POLICY,
    aggregate_predictions,
    aggregate_turkish_pooled_text_condition_likelihood_predictions,
)
from src.utils import AGGREGATION_LEVEL_SUBJECT, PREDICTION_MODE_LIKELIHOOD


def _likelihood_rows(cases: dict[str, tuple[int, float, float]]) -> list[dict[str, object]]:
    """One likelihood row per subject and question condition.

    The margin is encoded as dep_score - non_score with non_score fixed at 0.
    """
    rows = []
    for subject, (label, positive, negative) in cases.items():
        for condition, margin in (
            ("pos_only_t17", positive),
            ("negative_only_t17", negative),
        ):
            rows.append(
                {
                    "subject_id": subject,
                    "sample_id": f"{subject}-{condition}",
                    "label": label,
                    "question_condition": condition,
                    "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
                    "dep_score": margin,
                    "non_score": 0.0,
                }
            )
    return rows


def test_likelihood_pooled_dispatch_applies_the_pair_rule() -> None:
    rows = _likelihood_rows(
        {
            "both_positive": (1, 0.8, 0.7),
            "both_negative": (0, -0.8, -0.7),
            "disagree_positive": (1, 0.6, -0.1),
            "disagree_negative": (0, -0.6, 0.1),
        }
    )
    headline_rows, _, subject_rows, subject_metrics = aggregate_predictions(
        rows,
        mode=PREDICTION_MODE_LIKELIHOOD,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    assert subject_metrics["aggregation_policy"] == TURKISH_POOLED_TEXT_PAIR_POLICY
    assert subject_metrics["prediction_backend"] == PREDICTION_MODE_LIKELIHOOD
    assert subject_metrics["binary_strict_accuracy"] == pytest.approx(1.0)
    assert subject_metrics["invalid_paired_subject_predictions"] == 0
    assert len(subject_rows) == 4
    assert subject_rows is headline_rows
    for row in subject_rows:
        assert row["aggregation_policy"] == TURKISH_POOLED_TEXT_PAIR_POLICY
        assert row["pair_margin"] == pytest.approx(
            (row["positive_margin"] + row["negative_margin"]) / 2.0
        )


def test_exact_pair_tie_is_invalid_and_strictly_wrong() -> None:
    rows = _likelihood_rows({"tie": (1, 0.5, -0.5)})
    _, _, subject_rows, subject_metrics = aggregate_predictions(
        rows,
        mode=PREDICTION_MODE_LIKELIHOOD,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    assert subject_rows[0]["prediction"] == -1
    assert subject_metrics["invalid_paired_subject_predictions"] == 1
    assert subject_metrics["binary_strict_accuracy"] == pytest.approx(0.0)


def test_declared_policy_may_not_be_mixed() -> None:
    rows = _likelihood_rows({"s1": (1, 0.8, 0.7)})
    rows[0]["aggregation_policy"] = "other_policy"
    with pytest.raises(ValueError, match="may not be mixed"):
        aggregate_predictions(
            rows,
            mode=PREDICTION_MODE_LIKELIHOOD,
            aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        )


def test_pair_rule_requires_both_conditions_per_subject() -> None:
    rows = _likelihood_rows({"s1": (1, 0.8, 0.7)})
    rows = [row for row in rows if row["question_condition"] != "negative_only_t17"]
    with pytest.raises(ValueError, match="must contain exactly"):
        aggregate_predictions(
            rows,
            mode=PREDICTION_MODE_LIKELIHOOD,
            aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        )


def test_plain_likelihood_aggregation_is_unchanged_without_the_policy() -> None:
    rows = [
        {"subject_id": "a", "label": 1, "dep_score": 0.2, "non_score": 0.0},
        {"subject_id": "a", "label": 1, "dep_score": 0.4, "non_score": 0.1},
    ]
    _, _, subject_rows, subject_metrics = aggregate_predictions(
        rows,
        mode=PREDICTION_MODE_LIKELIHOOD,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    assert subject_rows[0]["prediction"] == 1
    assert subject_rows[0]["num_samples"] == 2
    assert "pair_margin" not in subject_rows[0]
    assert subject_metrics["invalid_subjects"] == 0


def test_condition_breakdown_likelihood_rows_and_metrics() -> None:
    rows = _likelihood_rows({"correct": (1, 0.8, 0.7), "tie": (0, 0.0, -0.5)})
    condition_rows = [row for row in rows if row["question_condition"] == "pos_only_t17"]
    subject_rows, metrics = aggregate_turkish_pooled_text_condition_likelihood_predictions(
        condition_rows,
        "pos_only_t17",
    )
    assert len(subject_rows) == 2
    assert metrics["question_condition"] == "pos_only_t17"
    assert metrics["aggregation_policy"] == TURKISH_POOLED_TEXT_PAIR_POLICY
    assert metrics["invalid_condition_predictions"] == 1
    assert metrics["prediction_backend"] == PREDICTION_MODE_LIKELIHOOD
