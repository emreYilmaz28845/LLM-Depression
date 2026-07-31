from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.features.androids_hidden_policy import (
    ANDROID_AGGREGATION_POLICY,
    aggregate_androids_hidden_predictions,
    androids_training_weights,
    validate_androids_row_inventory,
)


def _audio_row(
    subject: str,
    label: int,
    response: str,
    window_index: int,
    window_count: int,
    probability: float = 0.5,
) -> dict[str, object]:
    sample_id = f"{response}_w{window_index:02d}"
    return {
        "dataset": "androids_interview",
        "modality": "audio_only",
        "sample_id": sample_id,
        "subject_id": subject,
        "recording_id": subject,
        "turn_id": int(response.rsplit("_t", 1)[-1]),
        "turn_key": response,
        "label": label,
        "response_id": response,
        "window_id": sample_id,
        "window_index": window_index,
        "num_windows": window_count,
        "num_segments": window_count,
        "segment_index": window_index,
        "start_time": float(window_index),
        "end_time": float(window_index + 1),
        "segment_duration": 1.0,
        "turn_duration": float(window_count),
        "probability": probability,
    }


def test_androids_weights_equalize_subjects_and_turns_with_variable_windows() -> None:
    rows = [
        _audio_row("c", 0, "c_t1", 0, 1),
        _audio_row("c", 0, "c_t2", 0, 2),
        _audio_row("c", 0, "c_t2", 1, 2),
        _audio_row("p", 1, "p_t1", 0, 1),
        _audio_row("p", 1, "p_t2", 0, 1),
    ]
    weights, audit = androids_training_weights(rows, "audio_only")
    assert np.isclose(weights.mean(), 1.0)
    assert audit["equal_subject_totals"] is True
    assert audit["equal_turn_weight_within_subject"] is True
    assert np.isclose(audit["subject_weight_totals"]["c"], audit["subject_weight_totals"]["p"])
    assert np.isclose(audit["turn_weight_totals"]["c_t2"], audit["turn_weight_totals"]["c_t1"])


def test_androids_aggregation_is_window_then_turn_then_subject_mean() -> None:
    rows = [
        _audio_row("s", 0, "s_t1", 0, 3, 0.6),
        _audio_row("s", 0, "s_t1", 1, 3, 0.6),
        _audio_row("s", 0, "s_t1", 2, 3, 0.6),
        _audio_row("s", 0, "s_t2", 0, 1, 0.1),
        _audio_row("s", 0, "s_t3", 0, 1, 0.1),
    ]
    turns, subjects, metrics = aggregate_androids_hidden_predictions(rows, "audio_only")
    assert [row["probability"] for row in turns] == [0.6, 0.1, 0.1]
    assert np.isclose(subjects[0]["probability"], (0.6 + 0.1 + 0.1) / 3)
    assert subjects[0]["prediction"] == 0
    assert metrics["aggregation_policy"] == ANDROID_AGGREGATION_POLICY


def test_androids_exact_subject_probability_tie_is_wrong_in_headline_metrics() -> None:
    rows = [
        _audio_row("c", 0, "c_t1", 0, 1, 0.5),
        _audio_row("p", 1, "p_t1", 0, 1, 0.5),
    ]
    _, subjects, metrics = aggregate_androids_hidden_predictions(rows, "audio_only")
    assert all(row["prediction"] == -1 for row in subjects)
    assert metrics["tie_count"] == 2
    assert metrics["accuracy"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 1], [1, 0]]


def test_androids_incomplete_turn_is_rejected() -> None:
    rows = [_audio_row("s", 0, "s_t1", 0, 2)]
    with pytest.raises(ValueError, match="incomplete"):
        validate_androids_row_inventory(rows, "audio_only")


def test_androids_text_only_is_one_vector_per_subject() -> None:
    rows = [
        {
            "sample_id": "c",
            "subject_id": "c",
            "label": 0,
            "source_turn_count": 2,
            "source_window_count": 3,
            "probability": 0.2,
        },
        {
            "sample_id": "p",
            "subject_id": "p",
            "label": 1,
            "source_turn_count": 1,
            "source_window_count": 1,
            "probability": 0.8,
        },
    ]
    turns, subjects, metrics = aggregate_androids_hidden_predictions(rows, "text_only")
    assert turns == []
    assert [row["prediction"] for row in subjects] == [0, 1]
    assert metrics["num_subjects"] == 2


def test_androids_inner_assignments_are_subject_disjoint() -> None:
    pytest.importorskip("sklearn")
    from baselines.qwen_hidden_xgb_optuna import build_inner_subject_assignments

    rows = [
        {**_audio_row(f"s{subject}", subject % 2, f"s{subject}_t1", 0, 1), "sample_id": f"s{subject}"}
        for subject in range(12)
    ]
    assignments = build_inner_subject_assignments(rows, inner_folds=3, seed=1337)
    validation = []
    all_subjects = {row["subject_id"] for row in rows}
    for fold in assignments["folds"]:
        train = set(fold["train_subject_ids"])
        val = set(fold["validation_subject_ids"])
        assert not train & val
        assert train | val <= all_subjects
        validation.extend(val)
    assert sorted(validation) == sorted(all_subjects)


def test_androids_fixed_head_collision_and_sample_weight_propagation() -> None:
    pytest.importorskip("sklearn")
    from baselines.androids_hidden_classifier import _check_complete_or_collision, _result_identity

    class FakePipeline:
        def __init__(self) -> None:
            self.kwargs = None

        def fit(self, x, y, **kwargs):
            self.kwargs = kwargs

    fake = FakePipeline()
    weights = np.asarray([0.5, 1.5])
    fake.fit(np.zeros((2, 1)), np.asarray([0, 1]), classifier__sample_weight=weights)
    assert np.array_equal(fake.kwargs["classifier__sample_weight"], weights)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        output.mkdir(exist_ok=True)
        identity = {"schema_version": "test", "result_config_sha256": "x"}
        (output / "result_config.json").write_text(json.dumps(identity), encoding="utf-8")
        with pytest.raises(ValueError, match="incompatible"):
            _check_complete_or_collision(output, {"schema_version": "other"})
