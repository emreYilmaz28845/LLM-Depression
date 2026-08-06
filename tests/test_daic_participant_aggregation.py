from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from src.aggregate import (
    INVALID_PREDICTION,
    aggregate_mean_score_margin_subject_predictions,
    aggregate_original_teacher_forced_predictions,
)
from src.data.daic import PACKED30_PROTOCOL_ID


def sample_row(
    subject_id: str,
    dep_score: float,
    non_score: float,
    label: int = 0,
    protocol_id: str = PACKED30_PROTOCOL_ID,
) -> dict:
    row = {
        "subject_id": subject_id,
        "sample_id": f"{subject_id}_c0",
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "dep_score": dep_score,
        "non_score": non_score,
        "teacher_forced_prediction": 1 if dep_score > non_score else 0,
        "prediction_backend": "original_teacher_forced",
        "evaluation_protocol_name": "teacher_forced_label_span",
    }
    if protocol_id:
        row["protocol_id"] = protocol_id
    return row


def test_mean_margin_positive_predicts_depressed() -> None:
    rows = [
        sample_row("a", 0.6, 0.4, 0),
        sample_row("a", 0.55, 0.45, 0),
        sample_row("b", 0.4, 0.6, 1),
        sample_row("b", 0.45, 0.55, 1),
    ]
    subject_rows, metrics = aggregate_mean_score_margin_subject_predictions(rows)
    assert len(subject_rows) == 2
    by_subject = {row["subject_id"]: row for row in subject_rows}
    assert by_subject["a"]["prediction"] == 1
    assert by_subject["b"]["prediction"] == 0
    assert by_subject["a"]["score_margin"] == pytest.approx(0.15)
    assert metrics["num_subjects"] == 2


def test_mean_margin_exact_zero_is_invalid() -> None:
    rows = [sample_row("a", 0.5, 0.5, 0), sample_row("a", 0.5, 0.5, 0)]
    subject_rows, _ = aggregate_mean_score_margin_subject_predictions(rows)
    assert subject_rows[0]["prediction"] == INVALID_PREDICTION
    assert subject_rows[0]["prediction_text"] == "INVALID"


def test_non_finite_candidate_score_fails() -> None:
    rows = [sample_row("a", float("inf"), 0.5, 0)]
    with pytest.raises(ValueError, match="Non-finite"):
        aggregate_mean_score_margin_subject_predictions(rows)
    rows = [sample_row("a", 0.5, float("nan"), 0)]
    with pytest.raises(ValueError, match="Non-finite"):
        aggregate_mean_score_margin_subject_predictions(rows)


def test_non_finite_margin_fails() -> None:
    rows = [sample_row("a", 1e308, -1e308, 0)]
    with pytest.raises(ValueError, match="Non-finite teacher-forced margin"):
        aggregate_mean_score_margin_subject_predictions(rows)


def test_dispatch_routes_protocol_rows_to_strict_mean_margin() -> None:
    rows = [sample_row("a", 0.7, 0.3, 0), sample_row("a", 0.4, 0.6, 0)]
    subject_rows, metrics = aggregate_original_teacher_forced_predictions(rows)
    assert len(subject_rows) == 1
    assert subject_rows[0]["score_margin"] == pytest.approx(0.1)
    assert subject_rows[0]["prediction"] == 1
    assert metrics["aggregation_method"] == "mean_teacher_forced_score_margin"
    assert "auroc" not in metrics


def test_dispatch_preserves_legacy_mean_score_path() -> None:
    rows = [sample_row("a", 0.7, 0.3, 0, protocol_id="")]
    for row in rows:
        row["subject_score_aggregation"] = "mean_score"
    subject_rows, metrics = aggregate_original_teacher_forced_predictions(rows)
    assert len(subject_rows) == 1
    assert subject_rows[0]["prediction"] == 1
    assert metrics["aggregation_method"] == "mean_teacher_forced_score_margin"
    assert "auroc" in metrics


def test_dispatch_preserves_majority_vote_path() -> None:
    rows = [
        sample_row("a", 0.9, 0.1, 0, protocol_id=""),
        sample_row("a", 0.1, 0.9, 0, protocol_id=""),
    ]
    subject_rows, metrics = aggregate_original_teacher_forced_predictions(rows)
    assert len(subject_rows) == 1
    # Legacy majority vote: equal votes with an exact-zero summed margin -> INVALID.
    assert subject_rows[0]["prediction"] == INVALID_PREDICTION
    assert subject_rows[0]["score_margin"] == 0.0
    assert "auroc" in metrics


def test_subject_normalized_chunk_weights() -> None:
    from src.train import apply_subject_normalized_chunk_weights

    examples = []
    for subject_id, count, label in (("a", 3, 0), ("b", 1, 1)):
        for index in range(count):
            examples.append(
                {
                    "subject_id": subject_id,
                    "sample_id": f"{subject_id}_{index}",
                    "label": label,
                }
            )
    weighted, audit = apply_subject_normalized_chunk_weights(examples)
    assert audit["policy"] == "all_chunks_subject_normalized"
    raw_totals: dict[str, float] = defaultdict(float)
    for example in weighted:
        raw_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    assert {subject_id: round(value, 12) for subject_id, value in raw_totals.items()} == {"a": 1.0, "b": 1.0}
    mean_weight = sum(float(example["loss_weight"]) for example in weighted) / len(weighted)
    assert mean_weight == pytest.approx(1.0, abs=1e-9)
    by_subject = {str(example["subject_id"]): float(example["loss_weight"]) for example in weighted}
    # scale = 4 / (3 * 1/3 + 1) = 2 -> a-chunks 2/3 each, b 2.0
    assert by_subject["a"] == pytest.approx(2 / 3)
    assert by_subject["b"] == pytest.approx(2.0)


def test_subject_normalized_weights_require_unique_sample_ids() -> None:
    from src.train import apply_subject_normalized_chunk_weights

    examples = [
        {"subject_id": "a", "sample_id": "dup", "label": 0},
        {"subject_id": "a", "sample_id": "dup", "label": 0},
    ]
    with pytest.raises(ValueError, match="unique sample IDs"):
        apply_subject_normalized_chunk_weights(examples)


def test_subject_normalized_weights_require_one_label_per_subject() -> None:
    from src.train import apply_subject_normalized_chunk_weights

    examples = [
        {"subject_id": "a", "sample_id": "a_0", "label": 0},
        {"subject_id": "a", "sample_id": "a_1", "label": 1},
    ]
    with pytest.raises(ValueError, match="one label per subject"):
        apply_subject_normalized_chunk_weights(examples)


def test_merged_weighting_recognizes_daic_chunks_as_stable_units() -> None:
    from src.merged.protocol import compute_hierarchical_example_weights

    examples = []
    for subject_id, count in (("300", 3), ("301", 1)):
        for index in range(count):
            examples.append(
                {
                    "dataset": "daic",
                    "subject_id": subject_id,
                    "sample_id": f"{subject_id}_participant_p30_{index:03d}",
                    "label": 0,
                }
            )
    weighted, audit = compute_hierarchical_example_weights(examples, expected_datasets=("daic",))
    assert audit["response_count"] == 4
    subject_totals: dict[str, float] = defaultdict(float)
    for example in weighted:
        subject_totals[str(example["subject_id"])] += float(example["loss_weight"])
    # Merged weighting equalizes subject totals within the dataset (2.0 here).
    assert len(set(round(value, 12) for value in subject_totals.values())) == 1
    mean_weight = sum(float(example["loss_weight"]) for example in weighted) / len(weighted)
    assert mean_weight == pytest.approx(1.0, abs=1e-10)
    assert audit["hierarchical_invariants"]["equal_subject_totals_within_dataset"] is True
    assert audit["hierarchical_invariants"]["equal_response_totals_within_subject"] is True
