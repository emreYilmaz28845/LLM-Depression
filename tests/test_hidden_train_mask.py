"""Tests for the hidden-classifier training-mask filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import qwen_hidden_classifier as clf  # noqa: E402
from src.data.window_cap import build_mask  # noqa: E402


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


def test_mask_filters_rows_and_reports_metadata() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    mask = build_mask(_baseline_examples(plan), fraction=0.5, sampling_seed=1337)
    filtered_x, filtered_rows, metadata = clf._apply_train_mask(x, rows, mask)
    selected = {sid for ids in mask["subjects"].values() for sid in ids}
    assert [row["sample_id"] for row in filtered_rows] == [
        row["sample_id"] for row in rows if row["sample_id"] in selected
    ]
    assert filtered_x.shape[0] == len(filtered_rows) == 4
    assert metadata["selected_rows"] == 4
    assert metadata["available_rows"] == 6
    assert metadata["selection_sha256"] == mask["selection_sha256"]


def test_mask_with_missing_sample_id_is_rejected() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    mask = build_mask(_baseline_examples(plan), fraction=0.25, sampling_seed=1337)
    mask["subjects"]["s1"] = ["s1_w999"]
    with pytest.raises(ValueError, match="missing from outer_train"):
        clf._apply_train_mask(x, rows, mask)


def test_mask_subject_mismatch_is_rejected() -> None:
    plan = {"s1": 3, "s2": 3}
    x, rows = _cache_rows(plan)
    # Build a mask that only covers s1 while the cache has s1 and s2.
    mask = build_mask(_baseline_examples({"s1": 3}), fraction=0.25, sampling_seed=1337)
    with pytest.raises(ValueError, match="subject set does not match"):
        clf._apply_train_mask(x, rows, mask)


def test_empty_mask_selections_are_rejected() -> None:
    plan = {"s1": 3}
    x, rows = _cache_rows(plan)
    with pytest.raises(ValueError, match="no subject selections"):
        clf._apply_train_mask(x, rows, {"subjects": {}})
