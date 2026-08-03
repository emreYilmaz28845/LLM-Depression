from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
import pytest

from src.aggregate import aggregate_margin_predictions
from src.daic_chunking import (
    build_independent_epoch_schedule,
    build_joint_epoch_schedule,
    deterministic_subject_order,
    fixed_count_balanced_joint_bundles,
    matched_k_resamples,
)
from src.daic_mil import streaming_subject_mil_backward
from src.daic_comprehensive_audit import audit_matrix, audit_oof_predictions, audit_slurm
from src.data.runtime import AUDIO_PLACEHOLDER, build_examples


ROOT = Path(__file__).resolve().parents[1]


def rows(subject: str, n: int, label: int) -> list[dict]:
    return [
        {"subject_id": subject, "sample_id": f"{subject}_{i}", "label": label}
        for i in range(n)
    ]


def test_fixed15_has_exact_six_and_four_occurrences() -> None:
    for n, expected in ((10, 6), (15, 4)):
        bundles, audit = fixed_count_balanced_joint_bundles([str(i) for i in range(n)], 4, 15)
        assert len(bundles) == 15
        assert set(audit["coverage_by_chunk_id"].values()) == {expected}


@pytest.mark.parametrize("k", [2, 4, 8])
def test_joint_k_placeholder_audio_path_agreement(k: int) -> None:
    cfg = {
        "dataset": "daic", "seed": 1337,
        "prompt": {"system": "system", "user_template": "{audio_context_block} {label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {"use_audio": True, "use_text": False, "sample_mode": "subject_audio",
                 "train_chunk_policy": "random_k", "train_chunks_per_subject": k,
                 "eval_chunk_policy": "fixed_k", "eval_chunks_per_subject": k,
                 "max_audio_seconds_per_chunk": 30.0},
    }
    manifest_rows = [
        {"dataset": "daic", "subject_id": "300", "sample_id": f"300_{i}", "chunk_id": str(i),
         "label": 0, "label_text": "Non-depressed", "transcript": "", "audio_path": f"/tmp/300_{i}.wav"}
        for i in range(10)
    ]
    example = build_examples(manifest_rows, cfg, "selection")[0]
    assert len(example["audio_paths"]) == k
    assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == k


def test_fixed_count_rejects_unequal_coverage() -> None:
    try:
        fixed_count_balanced_joint_bundles([str(i) for i in range(11)], 4, 15)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("unequal coverage was accepted")


def test_independent_raw_and_mean_one_weights_and_flat_negative_control() -> None:
    examples = rows("a", 10, 0) + rows("b", 15, 1)
    schedules, audit = build_independent_epoch_schedule(
        examples, policy="all", chunks_per_subject="all", seed=1337, epochs=1,
        loss_weight_rescale="mean_one",
    )
    assert audit["epoch_mean_effective_weights"] == [1.0]
    raw = defaultdict(float)
    for row in schedules[0]:
        raw[row["subject_id"]] += row["raw_loss_weight"]
    assert raw == pytest.approx({"a": 1.0, "b": 1.0})
    flat, flat_audit = build_independent_epoch_schedule(
        examples, policy="all", chunks_per_subject="all", seed=1337, epochs=1,
        loss_weight_rescale="mean_one", equal_row_weight=True,
    )
    assert flat_audit["equal_total_subject_weight"] is False
    effective = defaultdict(float)
    for row in flat[0]:
        effective[row["subject_id"]] += row["loss_weight"]
    assert effective["b"] > effective["a"]


def test_joint_rotary_replay_and_balanced_cover_subject_totals() -> None:
    subjects = []
    for subject, n, label in (("a", 10, 0), ("b", 15, 1)):
        subjects.append({
            "subject_id": subject, "sample_id": subject, "label": label,
            "subject_chunk_paths": [f"/{subject}/{i}.wav" for i in range(n)],
            "subject_chunk_ids": [str(i) for i in range(n)],
            "audio_clip_seconds": [30.0] * 4,
        })
    one, _ = build_joint_epoch_schedule(subjects, policy="joint_rotary_k", k=4, seed=1337, epochs=3)
    two, _ = build_joint_epoch_schedule(subjects, policy="joint_rotary_k", k=4, seed=1337, epochs=3)
    assert [[row["bundle_chunk_ids"] for row in epoch] for epoch in one] == [
        [row["bundle_chunk_ids"] for row in epoch] for epoch in two
    ]
    cover, _ = build_joint_epoch_schedule(subjects, policy="joint_balanced_cover", k=4, seed=1337, epochs=1)
    totals = defaultdict(float)
    for row in cover[0]:
        totals[row["subject_id"]] += row["raw_loss_weight"]
    assert totals == pytest.approx({"a": 1.0, "b": 1.0})


def test_subject_order_is_deterministic_and_epoch_specific() -> None:
    ids = [str(i) for i in range(30)]
    assert deterministic_subject_order(ids, seed=1337, epoch=0) == deterministic_subject_order(ids, seed=1337, epoch=0)
    assert deterministic_subject_order(ids, seed=1337, epoch=0) != deterministic_subject_order(ids, seed=1337, epoch=1)


def test_secondary_aggregations_emit_one_row_per_subject() -> None:
    sample_rows = [
        {"subject_id": subject, "sample_id": f"{subject}_{i}", "label": label,
         "dep_score": margin, "non_score": 0.0}
        for subject, label, values in (("a", 0, [-2.0, -1.0, 3.0]), ("b", 1, [1.0, 2.0, -1.0]))
        for i, margin in enumerate(values)
    ]
    for method in ("mean_score", "median_score", "trimmed_mean_10", "majority_margin_tiebreak", "max_score"):
        subjects, metrics = aggregate_margin_predictions(sample_rows, method)
        assert len(subjects) == 2
        assert metrics["aggregation_method"] == method


def test_matched_resampling_is_cached_deterministic_and_complete() -> None:
    samples = [
        {"subject_id": "a", "sample_id": f"a_{i}", "label": 0, "dep_score": float(i), "non_score": 5.0}
        for i in range(15)
    ]
    first = matched_k_resamples(samples, k=10, iterations=20, seed=1337)
    assert first == matched_k_resamples(samples, k=10, iterations=20, seed=1337)
    assert len(first) == 20
    assert all(len(set(row["sample_ids"])) == 10 for row in first)


def test_streaming_mil_gradient_matches_direct_graph_and_waits_for_subject() -> None:
    parameter = torch.tensor(0.4, requires_grad=True)
    features = [torch.tensor(value) for value in (1.0, -0.5, 2.0)]
    calls = []

    def margin(feature):
        return parameter * feature

    streaming_subject_mil_backward(
        features, label=1, margin_fn=margin,
        backward_fn=lambda value: (calls.append(float(value.detach())), value.backward())[1],
    )
    streaming_grad = parameter.grad.detach().clone()
    assert len(calls) == len(features)
    direct_parameter = torch.tensor(0.4, requires_grad=True)
    direct = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.stack([direct_parameter * value for value in features]).mean(), torch.tensor(1.0)
    )
    direct.backward()
    assert torch.allclose(streaming_grad, direct_parameter.grad, atol=1e-7)


def test_core_matrix_expands_to_360_tasks_with_stable_folds() -> None:
    module_spec = importlib.util.spec_from_file_location(
        "build_matrix", ROOT / "scripts/build_daic_comprehensive_matrix.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    spec = yaml.safe_load((ROOT / "configs/experiments/daic_chunking/comprehensive_matrix.yaml").read_text())
    matrix = module.expand(spec, "unit_test", "core")
    assert matrix["task_count"] == 360
    assert matrix["kind_counts"] == {"train": 90, "evaluation": 90, "hidden": 90, "classical": 90}
    train = [row for row in matrix["tasks"] if row["kind"] == "train"]
    assert {row["fold"] for row in train} == set(range(5))
    assert {row["seed"] for row in train} == {1337, 2027, 3407}
    assert {row["overrides"]["split.seed"] for row in train} == {1337}
    assert audit_matrix(matrix) == []
    broken = {**matrix, "tasks": matrix["tasks"][:-1]}
    assert any("incomplete_cell" in item or "kind_counts_mismatch" in item for item in audit_matrix(broken))


def test_audit_rejects_duplicate_oof_and_failed_slurm_rows() -> None:
    oof = [
        {"protocol_id": "jr4", "seed": 1337, "fold": 0, "subject_id": "a", "label": 0, "prediction": 0},
        {"protocol_id": "jr4", "seed": 1337, "fold": 0, "subject_id": "a", "label": 0, "prediction": 0},
    ]
    failures = audit_oof_predictions(
        oof, expected_subject_ids={"a", "b"}, protocols={"jr4"}, seeds={1337}, folds={0, 1}
    )
    assert any("coverage" in item for item in failures)
    assert any("duplicate" in item for item in failures)
    assert audit_slurm(
        [{"task_id": "x", "state": "FAILED", "exit_code": "1:0"}], {"x"}
    ) == ["slurm_failure:x:FAILED:1:0"]
