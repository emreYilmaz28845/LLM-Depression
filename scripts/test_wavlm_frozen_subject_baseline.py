#!/usr/bin/env python3
"""Focused no-model tests for the E1b frozen-WavLM baseline."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.wavlm_frozen_subject_baseline import (
    build_subject_features,
    bootstrap_subject_intervals,
    derangement,
    evenly_spaced_indices,
    _extract_vector,
    _fit_select_refit,
    _metrics,
    evaluate_subject_baseline,
    numeric_chunk_key,
    select_fixed_k_rows,
    shuffled_subject_features,
)


class _FakeFeatureExtractor:
    sampling_rate = 16000

    def __call__(self, values, **kwargs):
        del kwargs
        return {"input_values": torch.as_tensor(values).unsqueeze(0)}


class _FakeWavLM:
    def __call__(self, **kwargs):
        del kwargs
        hidden_states = tuple(
            torch.full((1, 3, 2), float(index), dtype=torch.float32)
            for index in range(13)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def _rows(subject_id: str, label: int, partition: str, count: int) -> list[dict]:
    return [
        {
            "subject_id": subject_id,
            "sample_id": f"{subject_id}_random_segment_{index}",
            "chunk_id": f"random_segment_{index}",
            "label": label,
            "partition": partition,
            "audio_path": f"/{subject_id}/{index}.wav",
        }
        for index in range(1, count + 1)
    ]


def main() -> None:
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(15, 4) == [0, 5, 9, 14]
    lexical_trap = _rows("308", 1, "train", 15)
    assert [numeric_chunk_key(row)[0] for row in sorted(lexical_trap, key=numeric_chunk_key)] == list(
        range(1, 16)
    )

    rows = (
        _rows("300", 0, "train", 10)
        + _rows("308", 1, "train", 15)
        + _rows("302", 0, "val", 10)
        + _rows("307", 1, "val", 15)
        + _rows("301", 0, "test", 10)
        + _rows("322", 1, "test", 15)
    )
    selected, selected_ids = select_fixed_k_rows(rows, chunks_per_subject=4)
    assert selected_ids["300"] == [
        "300_random_segment_1",
        "300_random_segment_4",
        "300_random_segment_7",
        "300_random_segment_10",
    ]
    assert selected_ids["308"] == [
        "308_random_segment_1",
        "308_random_segment_6",
        "308_random_segment_10",
        "308_random_segment_15",
    ]
    assert all(len(sample_ids) == 4 for sample_ids in selected_ids.values())

    pooled = _extract_vector(
        np.asarray([0.0, 0.5, -0.5], dtype=np.float32),
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeWavLM(),
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(
        pooled,
        np.asarray([6.0, 6.0, 7.0, 7.0, 8.0, 8.0], dtype=np.float32),
    )

    vectors = {
        row["sample_id"]: np.asarray(
            [float(row["numeric_chunk_number"]), float(row["label"])],
            dtype=np.float32,
        )
        for row in selected
    }
    subjects = build_subject_features(selected, vectors, chunks_per_subject=4)
    assert all(row["feature"].shape == (4,) for row in subjects.values())
    assert all(len(row["sample_ids"]) == 4 for row in subjects.values())

    smoke_evaluation, _, _ = evaluate_subject_baseline(
        subjects,
        c_grid=(0.01,),
        seed=5,
        shuffle_repeats=1,
        bootstrap_repeats=2,
    )
    tied_prevalence = smoke_evaluation["majority_control"]
    assert tied_prevalence["validation_probability_from_train_prevalence"] == 0.5
    assert tied_prevalence["validation_label_from_train"] == 1
    assert tied_prevalence["validation_metrics"]["confusion_matrix"] == [[0, 1], [0, 1]]

    one_class_metrics = _metrics(
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0.2, 0.3], dtype=np.float64),
    )
    assert one_class_metrics["auroc"] is None
    assert one_class_metrics["average_precision"] is None

    mapping = derangement(["a", "b", "c", "d"], np.random.default_rng(1337))
    assert set(mapping) == set(mapping.values())
    assert all(target != source for target, source in mapping.items())

    shuffled, mappings = shuffled_subject_features(subjects, seed=1337)
    assert set(shuffled) == set(subjects)
    for subject_id in subjects:
        assert shuffled[subject_id]["label"] == subjects[subject_id]["label"]
        assert shuffled[subject_id]["partition"] == subjects[subject_id]["partition"]
        assert shuffled[subject_id]["source_subject_id"] != subject_id
    for partition, partition_mapping in mappings.items():
        assert all(target != source for target, source in partition_mapping.items()), partition

    rng = np.random.default_rng(23)
    train_y = np.tile([0, 1], 20)
    val_y = np.tile([0, 1], 10)
    test_y = np.tile([0, 1], 10)
    train_x = rng.normal(size=(len(train_y), 5))
    val_x = rng.normal(size=(len(val_y), 5))
    test_x = rng.normal(size=(len(test_y), 5))
    train_x[:, 0] += train_y
    val_x[:, 0] += val_y
    test_x[:, 0] += test_y
    fit_result, _, _, test_scores = _fit_select_refit(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        c_grid=(0.01, 0.1, 1.0),
        seed=7,
    )
    minimum = min(
        fit_result["candidates"],
        key=lambda row: (row["metrics"]["log_loss"], row["c"]),
    )
    assert fit_result["selected_c"] == minimum["c"]
    first_bootstrap = bootstrap_subject_intervals(test_y, test_scores, repeats=20, seed=11)
    second_bootstrap = bootstrap_subject_intervals(test_y, test_scores, repeats=20, seed=11)
    assert first_bootstrap == second_bootstrap
    assert first_bootstrap["unit"] == "subject"

    print("E1b WavLM baseline no-model tests passed")


if __name__ == "__main__":
    main()
