"""Tests for the hidden-classifier training-mask validation and filter."""

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


def test_valid_mask_passes_validation_and_filters_rows() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    clf._validate_train_mask(mask, expected_selection_sha256=mask["selection_sha256"])
    filtered_x, filtered_rows, metadata = clf._apply_train_mask(x, rows, mask)
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
    all_ids = sorted(
        row["sample_id"] for row in _baseline_examples(plan)
    )
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
        clf._apply_train_mask(x, rows, mask)


def test_wrong_baseline_membership_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["baseline_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="baseline_input_sha256"):
        clf._apply_train_mask(x, rows, mask)


def test_missing_baseline_hash_is_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["baseline_input_sha256"] = None
    with pytest.raises(ValueError, match="baseline_input_sha256"):
        clf._apply_train_mask(x, rows, mask)


def test_missing_sample_id_is_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan, fraction=0.25)
    mask["subjects"]["s1"] = ["s1_w999"]
    _rehash(mask)
    with pytest.raises(ValueError, match="missing from outer_train"):
        clf._apply_train_mask(x, rows, mask)


def test_subject_set_mismatch_is_refused() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan, fraction=0.25)
    del mask["subjects"]["s2"]
    _rehash(mask)
    with pytest.raises(ValueError, match="subject set does not match"):
        clf._apply_train_mask(x, rows, mask)


def test_empty_subject_selections_are_refused() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = _mask_for(plan)
    mask["subjects"] = {}
    with pytest.raises(ValueError, match="no subject selections"):
        clf._apply_train_mask(x, rows, mask)


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
