from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import soundfile as sf

from src.aggregate import aggregate_response_subject_predictions
from src.data.d3tec import (
    build_d3tec_training_schedule,
    equal_duration_windows,
)
from src.data.runtime import load_audio_array
from src.utils import PREDICTION_MODE_ORIGINAL_TEACHER_FORCED


def _schedule_examples() -> list[dict]:
    examples = []
    for subject_id in ("001", "002"):
        for prompt_id in range(27):
            count = 4 if prompt_id == 0 else (2 if prompt_id == 1 else 1)
            response_id = f"{subject_id}_p{prompt_id}"
            for segment_index in range(count):
                examples.append(
                    {
                        "dataset": "d3tec",
                        "subject_id": subject_id,
                        "response_id": response_id,
                        "sample_id": f"{response_id}_s{segment_index}",
                        "segment_index": segment_index,
                    }
                )
    return examples


def test_equal_duration_windows_are_contiguous_complete_and_bounded() -> None:
    windows = equal_duration_windows(91.0, 30.0)
    assert len(windows) == 4
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 91.0
    assert max(end - start for start, end in windows) <= 30.0
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))


def test_interval_loader_reads_later_segment_not_first(tmp_path) -> None:
    sample_rate = 8000
    audio = np.concatenate(
        [
            np.zeros(sample_rate, dtype=np.float32),
            np.ones(sample_rate, dtype=np.float32) * 0.5,
        ]
    )
    path = tmp_path / "two_regions.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    first = load_audio_array(str(path), sample_rate, None, False, 0.0, 1.0)
    second = load_audio_array(str(path), sample_rate, None, False, 1.0, 2.0)
    assert np.max(np.abs(first)) == 0.0
    assert np.allclose(second, 0.5)


def test_rotary_schedule_is_deterministic_balanced_and_covers_segments() -> None:
    examples = _schedule_examples()
    epochs, audit = build_d3tec_training_schedule(
        examples,
        policy="rotate_one_per_response",
        seed=1337,
        virtual_epochs=8,
    )
    repeated, _ = build_d3tec_training_schedule(
        examples,
        policy="rotate_one_per_response",
        seed=1337,
        virtual_epochs=8,
    )
    assert audit["schedule_sample_ids"] == [
        [row["sample_id"] for row in epoch] for epoch in repeated
    ]
    assert all(len(epoch) == 54 for epoch in epochs)
    for epoch in epochs:
        assert Counter(row["subject_id"] for row in epoch) == {"001": 27, "002": 27}
        assert len({row["response_id"] for row in epoch}) == 54
    counts = Counter(row["sample_id"] for epoch in epochs for row in epoch)
    for example in examples:
        assert counts[example["sample_id"]] >= 2


def test_flat_and_normalized_share_order_and_normalized_totals() -> None:
    examples = _schedule_examples()
    flat, flat_audit = build_d3tec_training_schedule(
        examples,
        policy="all_segments_flat",
        seed=1337,
        virtual_epochs=8,
    )
    normalized, normalized_audit = build_d3tec_training_schedule(
        examples,
        policy="all_segments_response_normalized",
        seed=1337,
        virtual_epochs=8,
    )
    assert flat_audit["schedule_sample_ids"] == normalized_audit["schedule_sample_ids"]
    flat_counts = Counter(row["sample_id"] for epoch in flat for row in epoch)
    assert max(flat_counts.values()) - min(flat_counts.values()) <= 1
    assert math.isclose(normalized_audit["mean_loss_weight"], 1.0, abs_tol=1e-12)
    assert all(
        math.isclose(total, 1.0 / 27.0, abs_tol=1e-12)
        for total in normalized_audit["raw_response_weight_totals"].values()
    )
    assert all(
        math.isclose(total, 1.0, abs_tol=1e-12)
        for total in normalized_audit["raw_subject_weight_totals"].values()
    )
    assert len(flat) == len(normalized) == 8
    assert all(len(epoch) == 54 for epoch in normalized)


def _prediction_row(
    subject_id: str,
    response_id: str,
    segment_index: int,
    prediction: int,
    margin: float,
    label: int,
) -> dict:
    return {
        "subject_id": subject_id,
        "response_id": response_id,
        "prompt_id": int(response_id.rsplit("p", 1)[1]),
        "sample_id": f"{response_id}_s{segment_index}",
        "segment_index": segment_index,
        "label": label,
        "teacher_forced_prediction": prediction,
        "dep_score": margin,
        "non_score": 0.0,
    }


def test_hierarchical_aggregation_is_invariant_to_unequal_segment_counts() -> None:
    rows = []
    # Subject 001: response 0 has many positive segments, but responses 1 and 2
    # are negative. Direct segment voting would be positive; equal-response
    # hierarchical voting must be negative.
    rows.extend(_prediction_row("001", "001_p0", i, 1, 1.0, 0) for i in range(5))
    rows.append(_prediction_row("001", "001_p1", 0, 0, -1.0, 0))
    rows.append(_prediction_row("001", "001_p2", 0, 0, -1.0, 0))
    # Include the other class so metric/AUROC paths are exercised.
    rows.append(_prediction_row("002", "002_p0", 0, 1, 1.0, 1))
    rows.append(_prediction_row("002", "002_p1", 0, 1, 1.0, 1))
    rows.append(_prediction_row("002", "002_p2", 0, 0, -0.2, 1))

    response_rows, response_metrics, subject_rows, subject_metrics = (
        aggregate_response_subject_predictions(
            rows,
            prediction_field="teacher_forced_prediction",
            backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        )
    )
    predictions = {row["subject_id"]: row["prediction"] for row in subject_rows}
    assert predictions == {"001": 0, "002": 1}
    assert len(response_rows) == 6
    assert response_metrics["num_responses"] == 6
    assert subject_metrics["num_subjects"] == 2
    assert subject_metrics["macro_f1"] == 1.0
