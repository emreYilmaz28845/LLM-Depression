from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.daic_chunking import build_independent_epoch_schedule, resolve_chunking_controls
from src.data.runtime import AUDIO_PLACEHOLDER, build_examples
from src.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs/archive/pre_harmonized_posf1_20260809/daic/daic_audio_text_selposf1_tf.yaml"
EXPERIMENT_CONFIG = (
    ROOT
    / "configs/experiments/daic_chunking/daic_audio_text_independent_all_equalrow_selposf1_tf.yaml"
)


def _configs_without_declared_construction_changes() -> tuple[dict, dict]:
    main = copy.deepcopy(load_yaml(MAIN_CONFIG))
    experiment = copy.deepcopy(load_yaml(EXPERIMENT_CONFIG))

    # Output isolation is operational, not a scientific recipe change.
    main["output_dirs"].pop("run_root")
    experiment["output_dirs"].pop("run_root")

    main_data = main["data"]
    experiment_data = experiment["data"]
    assert main_data.pop("sample_mode") == "subject_audio"
    assert main_data.pop("chunks_per_subject") == 4
    # The canonical main recipe now declares the balanced-cover evaluation
    # view; both the independent and joint experiments replace it, so it is
    # part of the construction diff.
    main_data.pop("eval_chunk_policy")
    main_data.pop("eval_chunks_per_subject")
    for key in (
        "sample_mode",
        "train_chunk_policy",
        "train_chunks_per_subject",
        "eval_chunk_policy",
        "eval_chunks_per_subject",
        "loss_weight_rescale",
        "equal_row_weight",
    ):
        experiment_data.pop(key)

    assert experiment["training"].pop("match_joint_optimizer_updates") is False
    return main, experiment


def _manifest(subject: str, count: int, label: int, transcript: str) -> list[dict]:
    return [
        {
            "dataset": "daic",
            "subject_id": subject,
            "sample_id": f"{subject}_{index}",
            "chunk_id": str(index),
            "label": label,
            "label_text": "Depressed" if label else "Non-depressed",
            "transcript": transcript,
            "audio_path": f"/tmp/{subject}_{index}.wav",
        }
        for index in range(count)
    ]


def test_experiment_diff_is_limited_to_independent_example_construction() -> None:
    main, experiment = _configs_without_declared_construction_changes()
    assert experiment == main

    config = load_yaml(EXPERIMENT_CONFIG)
    controls = resolve_chunking_controls(config)
    assert controls == {
        "enabled": True,
        "sample_mode": "subject_chunks",
        "train_chunk_policy": "all",
        "train_chunks_per_subject": "all",
        "eval_chunk_policy": "all",
        "eval_chunks_per_subject": "all",
        "eval_bundles_per_subject": 15,
        "loss_weight_rescale": "none",
        "max_audio_seconds_per_chunk": 30.0,
    }
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["evaluation"].get("subject_score_aggregation") is None


def test_independent_rows_repeat_capped_transcript_and_keep_label_counts() -> None:
    config = load_yaml(EXPERIMENT_CONFIG)
    transcript = "full-subject-transcript-" * 300
    rows = _manifest("non", 10, 0, transcript) + _manifest("dep", 15, 1, transcript)

    examples = build_examples(rows, config, "train")
    assert Counter(example["subject_id"] for example in examples) == {"non": 10, "dep": 15}
    assert all(len(example["audio_paths"]) == 1 for example in examples)
    assert all(example["prompt_text"].count(AUDIO_PLACEHOLDER) == 1 for example in examples)
    assert {len(example["transcript"]) for example in examples} == {4000}
    for subject in ("non", "dep"):
        subject_rows = [example for example in examples if example["subject_id"] == subject]
        assert len({example["transcript"] for example in subject_rows}) == 1
        assert len({example["label"] for example in subject_rows}) == 1

    schedule, audit = build_independent_epoch_schedule(
        examples,
        policy="all",
        chunks_per_subject="all",
        seed=1337,
        epochs=20,
        loss_weight_rescale="none",
        equal_row_weight=True,
    )
    assert all(len(epoch) == 25 for epoch in schedule)
    for epoch in schedule:
        totals = defaultdict(float)
        for row in epoch:
            assert row["loss_weight"] == pytest.approx(1.0)
            totals[row["subject_id"]] += row["loss_weight"]
        assert totals == pytest.approx({"non": 10.0, "dep": 15.0})
    assert audit["subject_weighting"] == "equal_row"
    assert not audit["equal_total_subject_weight"]


@pytest.mark.parametrize(
    "non_depressed,depressed,expected_subjects,expected_rows",
    [
        (77, 30, 107, 1220),
        (23, 12, 35, 410),
        (33, 14, 47, 540),
    ],
)
def test_expected_daic_partition_sizes(
    non_depressed: int,
    depressed: int,
    expected_subjects: int,
    expected_rows: int,
) -> None:
    counts = Counter({0: non_depressed, 1: depressed})
    assert sum(counts.values()) == expected_subjects
    assert counts[0] * 10 + counts[1] * 15 == expected_rows
