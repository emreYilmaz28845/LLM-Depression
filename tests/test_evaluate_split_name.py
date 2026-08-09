from __future__ import annotations

from src.data.split_utils import CV_PROTOCOL_TRAIN_VAL, SPLIT_MODE_CV, SPLIT_MODE_FIXED
from src.evaluate import _resolved_split_name


def test_fixed_mode_uses_final_eval_partition() -> None:
    config = {"split": {"final_eval_partition": "test"}}
    assert _resolved_split_name(config, SPLIT_MODE_FIXED, None, 0) == "test"


def test_cv_train_val_uses_val() -> None:
    assert _resolved_split_name({}, SPLIT_MODE_CV, CV_PROTOCOL_TRAIN_VAL, 3) == "val"


def test_cv_train_val_test_uses_test() -> None:
    assert _resolved_split_name({}, SPLIT_MODE_CV, "train_val_test", 3) == "test"
