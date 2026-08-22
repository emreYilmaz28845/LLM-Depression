from __future__ import annotations

from src.data.split_utils import CV_PROTOCOL_TRAIN_VAL, SPLIT_MODE_CV, SPLIT_MODE_FIXED
from src.evaluate import _resolve_cv_evaluation_subject_ids, _resolved_split_name


def test_fixed_mode_uses_final_eval_partition() -> None:
    config = {"split": {"final_eval_partition": "test"}}
    assert _resolved_split_name(config, SPLIT_MODE_FIXED, None, 0) == "test"


def test_cv_train_val_uses_val() -> None:
    assert _resolved_split_name({}, SPLIT_MODE_CV, CV_PROTOCOL_TRAIN_VAL, 3) == "val"


def test_cv_train_val_test_uses_test() -> None:
    assert _resolved_split_name({}, SPLIT_MODE_CV, "train_val_test", 3) == "test"


def test_cv_train_val_evaluates_saved_selection_subjects() -> None:
    split_payload = {
        "selection_subject_ids": ["outer-val-2", "outer-val-1"],
        "final_eval_subject_ids": [],
    }
    assert _resolve_cv_evaluation_subject_ids(split_payload, CV_PROTOCOL_TRAIN_VAL) == [
        "outer-val-1",
        "outer-val-2",
    ]


def test_cv_train_val_test_evaluates_saved_final_subjects() -> None:
    split_payload = {
        "selection_subject_ids": ["selection-1"],
        "final_eval_subject_ids": ["test-2", "test-1"],
    }
    assert _resolve_cv_evaluation_subject_ids(split_payload, "train_val_test") == [
        "test-1",
        "test-2",
    ]
