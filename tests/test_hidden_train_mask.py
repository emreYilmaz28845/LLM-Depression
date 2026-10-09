"""Tests for the hidden-classifier training-mask validation and filter.

The mask baseline binds the fit's authoritative training pool (the parent
split's ``train_subject_ids``); the head's ``outer_train`` partition is
train + inner-val, so the checks require the masked subjects to equal the
split's training subjects exactly and the extra pool subjects to be exactly
the split's inner-val subjects.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import qwen_hidden_classifier as clf  # noqa: E402
from src.data.window_cap import build_mask, compute_selection_sha256  # noqa: E402

SEED = 1337


def _cache_rows(plan: dict[str, int]) -> tuple[np.ndarray, list[dict]]:
    rows: list[dict] = []
    vectors: list[list[float]] = []
    for subject_id, count in plan.items():
        for index in range(count):
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": f"{subject_id}_w{index:03d}",
                    "label": 0 if subject_id.endswith("1") else 1,
                }
            )
            vectors.append([float(index), float(len(subject_id))])
    return np.asarray(vectors, dtype=np.float32), rows


def _baseline_examples(plan: dict[str, int]) -> list[dict]:
    rows = []
    for subject_id, count in plan.items():
        for index in range(count):
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": f"{subject_id}_w{index:03d}",
                    "label": 0 if subject_id.endswith("1") else 1,
                    "raw_loss_weight": 1.0 / count,
                    "loss_weight": 1.0 / count,
                }
            )
    return rows


def _baseline_sha(rows: list[dict]) -> str:
    """Mirror the training hook's baseline membership hash exactly."""
    canonical = sorted([str(row["subject_id"]), str(row["sample_id"])] for row in rows)
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mask_for(plan: dict[str, int], fraction: float = 0.5) -> dict:
    _, rows = _cache_rows(plan)
    return build_mask(
        _baseline_examples(plan),
        fraction=fraction,
        sampling_seed=SEED,
        baseline_input_sha256=_baseline_sha(rows),
    )


def _rehash(mask: dict) -> None:
    mask["selection_sha256"] = compute_selection_sha256(
        mask["algorithm_version"], mask["sampling_seed"], mask["fraction"], mask["subjects"]
    )


def _split(train_plan: dict, val_plan: dict | None = None) -> dict:
    return {
        "train_subject_ids": sorted(train_plan),
        "val_inner_subject_ids": sorted(val_plan or {}),
    }


def test_valid_mask_passes_validation_and_filters_rows() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    clf._validate_train_mask(mask, expected_selection_sha256=mask["selection_sha256"])
    filtered_x, filtered_rows, metadata = clf._apply_train_mask(
        x, rows, mask, authoritative_split=_split(plan)
    )
    selected = {sid for ids in mask["subjects"].values() for sid in ids}
    assert [row["sample_id"] for row in filtered_rows] == [
        row["sample_id"] for row in rows if row["sample_id"] in selected
    ]
    assert filtered_x.shape[0] == len(filtered_rows) == 4
    assert metadata["selected_rows"] == 4
    assert metadata["available_rows"] == 6
    assert metadata["selection_sha256"] == mask["selection_sha256"]
    assert metadata["baseline_input_sha256"] == mask["baseline_input_sha256"]


def test_mutated_membership_with_stale_hash_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    mask = _mask_for(plan)
    all_ids = sorted(row["sample_id"] for row in _baseline_examples(plan))
    selected = {sid for ids in mask["subjects"].values() for sid in ids}
    replacement = next(sample_id for sample_id in all_ids if sample_id not in selected)
    mask["subjects"]["s1"][0] = replacement  # membership changed, hash left stale
    with pytest.raises(ValueError, match="does not match its membership"):
        clf._validate_train_mask(mask)
    with pytest.raises(ValueError, match="does not match its membership"):
        clf._validate_train_mask(mask, expected_selection_sha256=mask["selection_sha256"])


def test_swapped_subject_association_is_refused() -> None:
    plan = {"s1": 4, "s2": 4}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan, fraction=0.5)
    moved = mask["subjects"]["s1"].pop(0)
    mask["subjects"]["s2"].append(moved)
    _rehash(mask)  # membership hash is consistent; only the association is wrong
    clf._validate_train_mask(mask)
    with pytest.raises(ValueError, match="assigns sample"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_wrong_baseline_membership_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["baseline_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="baseline_input_sha256"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_missing_baseline_hash_is_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["baseline_input_sha256"] = None
    with pytest.raises(ValueError, match="baseline_input_sha256"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_missing_sample_id_is_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan, fraction=0.25)
    mask["subjects"]["s1"] = ["s1_w999"]
    _rehash(mask)
    with pytest.raises(ValueError, match="missing from outer_train"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_training_pool_subset_with_extra_val_subjects_binds_restricted_membership() -> None:
    """The head pool may include extra (val) subjects; the baseline binds the pool subset."""
    train_plan = {"s1": 3, "s2": 3}
    val_plan = {"s9": 2}
    x, rows = _cache_rows({**train_plan, **val_plan})
    mask = _mask_for(train_plan, fraction=0.5)
    filtered_x, filtered_rows, metadata = clf._apply_train_mask(
        x, rows, mask, authoritative_split=_split(train_plan, val_plan)
    )
    selected = {sid for ids in mask["subjects"].values() for sid in ids}
    assert {row["sample_id"] for row in filtered_rows} == selected
    assert metadata["available_rows"] == 6  # training-pool rows, not the 8-row outer_train
    assert all(row["subject_id"] != "s9" for row in filtered_rows)


def test_mask_train_subject_with_val_pool_is_allowed() -> None:
    train_plan = {"s1": 3}
    val_plan = {"s2": 3}
    x, rows = _cache_rows({**train_plan, **val_plan})
    mask = _mask_for(train_plan, fraction=0.5)
    filtered_x, filtered_rows, metadata = clf._apply_train_mask(
        x, rows, mask, authoritative_split=_split(train_plan, val_plan)
    )
    assert {row["subject_id"] for row in filtered_rows} == {"s1"}
    assert metadata["available_rows"] == 3
    assert metadata["selected_subject_count"] == 1


def test_missing_mask_training_subject_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    training_examples = _baseline_examples({"s1": 3})
    mask = build_mask(
        training_examples,
        fraction=0.5,
        sampling_seed=SEED,
        baseline_input_sha256=_baseline_sha(training_examples),
    )
    with pytest.raises(ValueError, match="missing_training_subjects"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_mask_subject_not_in_split_is_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan, fraction=0.25)
    mask["subjects"]["s9"] = [mask["subjects"]["s1"][0]]
    _rehash(mask)
    with pytest.raises(ValueError, match="do not equal the authoritative training split"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_authoritative_split_requires_hash_match(tmp_path) -> None:
    split_file = tmp_path / "split_used.json"
    split_file.write_text(
        json.dumps({"train_subject_ids": ["s1"], "val_inner_subject_ids": ["s2"]}),
        encoding="utf-8",
    )
    sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
    loaded = clf._load_authoritative_split(
        tmp_path, {"saved_split": str(split_file), "saved_split_sha256": sha}
    )
    assert loaded["train_subject_ids"] == ["s1"]
    with pytest.raises(ValueError, match="hash mismatch"):
        clf._load_authoritative_split(
            tmp_path, {"saved_split": str(split_file), "saved_split_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="no saved_split"):
        clf._load_authoritative_split(tmp_path, {})


def test_final_eval_partition_is_untouched_by_mask_application(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    train_plan = {"s1": 3, "s2": 3}
    np.savez(cache / "outer_train.npz", vectors=np.zeros((6, 2), dtype=np.float32))
    train_rows = [
        {
            "subject_id": subject,
            "sample_id": f"{subject}_w{i:03d}",
            "label": 0 if subject == "s1" else 1,
        }
        for subject, count in train_plan.items()
        for i in range(count)
    ]
    (cache / "outer_train_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in train_rows), encoding="utf-8"
    )
    np.savez(cache / "final_eval.npz", vectors=np.ones((2, 2), dtype=np.float32))
    eval_rows = [
        {"subject_id": "s9", "sample_id": f"s9_w{i}", "label": 0} for i in range(2)
    ]
    (cache / "final_eval_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in eval_rows), encoding="utf-8"
    )
    x, rows = clf._load_partition(cache, "outer_train")
    mask = _mask_for(train_plan, fraction=0.5)
    clf._apply_train_mask(x, rows, mask, authoritative_split=_split(train_plan))
    eval_x, eval_loaded = clf._load_partition(cache, "final_eval")
    assert [row["sample_id"] for row in eval_loaded] == [
        row["sample_id"] for row in eval_rows
    ]
    assert eval_x.shape[0] == 2


def test_empty_subject_selections_are_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["subjects"] = {}
    with pytest.raises(ValueError, match="no subject selections"):
        clf._apply_train_mask(x, rows, mask, authoritative_split=_split(plan))


def test_duplicate_sample_id_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    mask = _mask_for(plan)
    mask["subjects"]["s2"].append(mask["subjects"]["s1"][0])
    with pytest.raises(ValueError, match="more than once"):
        clf._validate_train_mask(mask)


def test_wrong_algorithm_identity_is_refused() -> None:
    plan = {"s1": 3}
    mask = _mask_for(plan)
    mask["algorithm_version"] = "sha256-subject-permutation-v0"
    with pytest.raises(ValueError, match="algorithm_version"):
        clf._validate_train_mask(mask)


def test_expected_hash_mismatch_is_refused() -> None:
    plan = {"s1": 3}
    mask = _mask_for(plan)
    with pytest.raises(ValueError, match="!= expected"):
        clf._validate_train_mask(mask, expected_selection_sha256="0" * 64)
