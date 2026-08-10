from __future__ import annotations

import copy
from collections import Counter, defaultdict

import pytest

from src.aggregate import (
    aggregate_mean_probability_predictions,
    aggregate_original_teacher_forced_predictions,
)
from src.daic_chunking import (
    balanced_joint_bundles,
    build_independent_epoch_schedule,
    gradient_accumulation_for_reference_updates,
    resolve_chunking_controls,
    rotary_indices,
)
from src.data.runtime import AUDIO_PLACEHOLDER, build_examples
from src.features.extract_qwen_hidden import _validate_saved_split
from src.utils import save_json


def config(mode: str = "subject_chunks") -> dict:
    return {
        "dataset": "daic",
        "seed": 1337,
        "prompt": {
            "system": "system",
            "user_template": "{audio_context_block} {label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "sample_mode": mode,
            "train_chunk_policy": "rotary_k" if mode == "subject_chunks" else "random_k",
            "train_chunks_per_subject": 4,
            "eval_chunk_policy": "all" if mode == "subject_chunks" else "fixed_k",
            "eval_chunks_per_subject": "all" if mode == "subject_chunks" else 4,
            "max_audio_seconds_per_chunk": 30.0,
        },
        "evaluation": {"subject_score_aggregation": "mean_score"},
    }


def manifest(subject: str, n: int, label: int = 0) -> list[dict]:
    return [
        {
            "dataset": "daic",
            "subject_id": subject,
            "sample_id": f"{subject}_{index}",
            "chunk_id": str(index),
            "label": label,
            "label_text": "Depressed" if label else "Non-depressed",
            "transcript": "",
            "audio_path": f"/tmp/{subject}_{index}.wav",
        }
        for index in range(n)
    ]


def test_legacy_joint_k_controls_are_backward_compatible() -> None:
    cfg = config("subject_audio")
    for key in (
        "train_chunk_policy",
        "train_chunks_per_subject",
        "eval_chunk_policy",
        "eval_chunks_per_subject",
    ):
        cfg["data"].pop(key)
    cfg["data"]["chunks_per_subject"] = 4
    controls = resolve_chunking_controls(cfg)
    assert controls["train_chunk_policy"] == "random_k"
    assert controls["eval_chunk_policy"] == "fixed_k"
    assert controls["train_chunks_per_subject"] == 4
    assert controls["eval_chunks_per_subject"] == 4


def test_joint_fixed10_is_deterministic_and_count_neutral_in_text() -> None:
    cfg = config("subject_audio")
    cfg["data"].update(
        {
            "train_chunk_policy": "fixed_k",
            "train_chunks_per_subject": 10,
            "eval_chunk_policy": "fixed_k",
            "eval_chunks_per_subject": 10,
            "subject_audio_count_text": "neutral",
        }
    )
    examples = build_examples(manifest("300", 15, label=1), cfg, "train")
    assert len(examples) == 1
    example = examples[0]
    assert example["subject_chunk_ids"] == [str(index) for index in range(15)]
    assert [path.rsplit("_", 1)[-1].removesuffix(".wav") for path in example["audio_paths"]] == [
        "0", "2", "3", "5", "6", "8", "9", "11", "12", "14"
    ]
    assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == 10
    assert "10 segments" not in example["prompt_text"]
    assert "15 segments" not in example["prompt_text"]


def test_independent_fixed10_schedule_is_stable_and_subject_normalized() -> None:
    rows = manifest("dep", 15, label=1) + manifest("non", 10, label=0)
    schedule, audit = build_independent_epoch_schedule(
        rows,
        policy="fixed_k",
        chunks_per_subject=10,
        seed=1337,
        epochs=2,
        loss_weight_rescale="none",
    )
    expected_dep = {"dep_0", "dep_2", "dep_3", "dep_5", "dep_6", "dep_8", "dep_9", "dep_11", "dep_12", "dep_14"}
    for epoch_rows in schedule:
        assert len(epoch_rows) == 20
        assert {row["sample_id"] for row in epoch_rows if row["subject_id"] == "dep"} == expected_dep
        totals = defaultdict(float)
        for row in epoch_rows:
            totals[row["subject_id"]] += row["loss_weight"]
        assert totals == pytest.approx({"dep": 1.0, "non": 1.0})
    assert audit["equal_total_subject_weight"]


def test_fixed10_rejects_subjects_with_too_few_chunks() -> None:
    with pytest.raises(ValueError, match="requires at least 10 chunks"):
        build_independent_epoch_schedule(
            manifest("short", 9), policy="fixed_k", chunks_per_subject=10,
            seed=1337, epochs=1,
        )


def test_independent_example_has_one_placeholder_one_audio_and_30s_cap() -> None:
    examples = build_examples(manifest("300", 1), config(), "train")
    assert len(examples) == 1
    assert examples[0]["prompt_text"].count(AUDIO_PLACEHOLDER) == 1
    assert len(examples[0]["audio_paths"]) == 1
    assert examples[0]["audio_clip_seconds"] == [30.0]


def test_rotary_is_deterministic_distinct_and_balanced_over_20_epochs() -> None:
    for n in (10, 15):
        replay = [
            rotary_indices(n, 4, subject_id="301", seed=1337, epoch=epoch)
            for epoch in range(20)
        ]
        assert replay == [
            rotary_indices(n, 4, subject_id="301", seed=1337, epoch=epoch)
            for epoch in range(20)
        ]
        assert all(len(indices) == len(set(indices)) == 4 for indices in replay)
        counts = Counter(index for indices in replay for index in indices)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_independent_schedules_include_all_and_equalize_subject_weight() -> None:
    rows = []
    for subject, n in (("a", 10), ("b", 15)):
        rows.extend(
            {"subject_id": subject, "sample_id": f"{subject}_{index}", "label": 0}
            for index in range(n)
        )
    rotary, rotary_audit = build_independent_epoch_schedule(
        rows, policy="rotary_k", chunks_per_subject=4, seed=1337, epochs=20
    )
    assert all(len(epoch) == 8 for epoch in rotary)
    assert rotary_audit["equal_total_subject_weight"]
    all_rows, all_audit = build_independent_epoch_schedule(
        rows, policy="all", chunks_per_subject="all", seed=1337, epochs=2
    )
    assert all(len(epoch) == 25 for epoch in all_rows)
    assert all_audit["equal_total_subject_weight"]
    for epoch in all_rows:
        totals = defaultdict(float)
        for row in epoch:
            totals[row["subject_id"]] += row["loss_weight"]
        assert totals == pytest.approx({"a": 1.0, "b": 1.0})


@pytest.mark.parametrize("n,expected_bundles,expected_occurrences", [(10, 5, 2), (15, 15, 4)])
def test_balanced_joint_cover(n: int, expected_bundles: int, expected_occurrences: int) -> None:
    bundles, audit = balanced_joint_bundles([str(index) for index in range(n)], 4)
    assert len(bundles) == expected_bundles
    assert audit["occurrences_per_chunk"] == expected_occurrences
    assert set(audit["coverage_by_chunk_id"].values()) == {expected_occurrences}
    assert len(audit["memberships"]) == expected_bundles * 4


def test_mean_score_and_mean_probability_make_one_prediction_per_subject() -> None:
    qwen_rows = [
        {
            "subject_id": "a",
            "label": 1,
            "dep_score": dep,
            "non_score": non,
            "teacher_forced_prediction": decoded,
            "subject_score_aggregation": "mean_score",
        }
        for dep, non, decoded in ((-1.0, -2.0, 0), (-4.0, -2.0, 0))
    ]
    subjects, metrics = aggregate_original_teacher_forced_predictions(qwen_rows)
    assert len(subjects) == 1
    assert subjects[0]["prediction"] == 0
    assert metrics["aggregation_method"] == "mean_teacher_forced_score_margin"
    classical, classical_metrics = aggregate_mean_probability_predictions(
        [
            {"subject_id": "a", "label": 1, "probability": 0.8},
            {"subject_id": "a", "label": 1, "probability": 0.3},
        ]
    )
    assert len(classical) == 1
    assert classical[0]["prediction"] == 1
    assert classical_metrics["predicted_positive_rate"] == 1.0


def test_optimizer_update_matching() -> None:
    joint = gradient_accumulation_for_reference_updates(
        independent_examples_per_epoch=400,
        reference_subjects=100,
        reference_gradient_accumulation=8,
    )
    assert joint == 31
    assert (400 + joint - 1) // joint == (100 + 8 - 1) // 8


def test_balanced_eval_examples_record_bundle_memberships() -> None:
    cfg = config("subject_audio")
    cfg["data"]["eval_chunk_policy"] = "balanced_joint_cover"
    examples = build_examples(manifest("300", 10), cfg, "test")
    assert len(examples) == 5
    assert {example["bundle_id"] for example in examples} == set(range(5))
    counts = Counter(
        chunk_id
        for example in examples
        for chunk_id in example["bundle_chunk_ids"]
    )
    assert set(counts.values()) == {2}


def test_hidden_split_validation_accepts_only_explicit_smoke_subsets(tmp_path) -> None:
    split_metadata = [
        {"subject_id": "train_a", "partition": "train"},
        {"subject_id": "train_b", "partition": "train"},
        {"subject_id": "test_a", "partition": "test"},
        {"subject_id": "test_b", "partition": "test"},
    ]
    split_path = tmp_path / "split.json"
    save_json(split_metadata, split_path)
    base_config = {
        "dataset": "daic",
        "evaluation": {"subject_score_aggregation": "mean_score"},
        "split": {
            "train_partition": "train",
            "selection_partition": "val",
            "dev_pool_partitions": ["train", "val"],
            "final_eval_partition": "test",
            "smoke_subject_limit": 1,
        },
    }
    saved = {"split_mode": "fixed", "split_metadata_path": str(split_path)}
    assert _validate_saved_split(
        saved,
        base_config,
        {"outer_train": ["train_a"], "final_eval": ["test_a"]},
        0,
        1,
    ) == split_path
    with pytest.raises(ValueError, match="not a subset"):
        _validate_saved_split(
            saved,
            base_config,
            {"outer_train": ["test_a"], "final_eval": ["test_b"]},
            0,
            1,
        )
    production = copy.deepcopy(base_config)
    production["split"].pop("smoke_subject_limit")
    with pytest.raises(ValueError, match="does not match"):
        _validate_saved_split(
            saved,
            production,
            {"outer_train": ["train_a"], "final_eval": ["test_a"]},
            0,
            1,
        )
