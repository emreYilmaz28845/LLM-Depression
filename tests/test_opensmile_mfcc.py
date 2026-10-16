from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from baselines.qwen_hidden_classifier import resolve_prediction_backend
from src.aggregate import HIERARCHICAL_MEAN_CLASSIFIER_POLICY, aggregate_binary_classifier_predictions
from src.features import optuna100_policy
from src.features.extract_opensmile_mfcc import (
    load_locked_folds,
    parse_path_maps,
    remap_manifest_rows,
    run_smile,
)
from src.features.hidden_classifier_policy import classifier_aggregation_policy

SMILE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mffcc_config" / "avec2013_3.conf"


def _row(subject: str, response: str | None, probability: float, label: int, index: int) -> dict:
    row = {
        "dataset": "cmdc",
        "sample_id": f"{subject}_{index}",
        "subject_id": subject,
        "label": label,
        "probability": probability,
        "predicted_class": int(probability >= 0.5),
        "aggregation_policy": HIERARCHICAL_MEAN_CLASSIFIER_POLICY,
    }
    if response is not None:
        row["response_id"] = response
    return row


def test_hierarchical_mean_differs_from_majority_vote():
    # Two weak "no" responses and one strong "yes": the response-mean is
    # (0.45 + 0.45 + 0.95) / 3 = 0.617, so the subject is Depressed. A
    # majority vote over responses would say Non-depressed.
    rows = [
        _row("s1", "s1::a", 0.45, 1, 0),
        _row("s1", "s1::b", 0.45, 1, 1),
        _row("s1", "s1::c", 0.95, 1, 2),
        _row("s2", "s2::a", 0.10, 0, 3),
    ]
    subjects, metrics = aggregate_binary_classifier_predictions(rows, prediction_backend="opensmile_mfcc_xgb_raw")
    by_subject = {row["subject_id"]: row for row in subjects}
    assert by_subject["s1"]["prediction"] == 1
    assert by_subject["s1"]["probability"] == pytest.approx((0.45 + 0.45 + 0.95) / 3)
    assert metrics["aggregation_policy"] == HIERARCHICAL_MEAN_CLASSIFIER_POLICY
    assert metrics["binary_strict_macro_f1"] == 1.0

    voted = [{**row, "aggregation_policy": "legacy"} for row in rows]
    vote_subjects, _ = aggregate_binary_classifier_predictions(voted)
    assert {row["subject_id"]: row["prediction"] for row in vote_subjects}["s1"] == 0


def test_windows_are_averaged_inside_a_response_first():
    # Response a has three windows (mean 0.3), response b one window (0.8).
    # Response-first mean is 0.55 -> Depressed; a flat window mean (0.425) is not.
    rows = [
        _row("s1", "s1::a", 0.3, 1, 0),
        _row("s1", "s1::a", 0.3, 1, 1),
        _row("s1", "s1::a", 0.3, 1, 2),
        _row("s1", "s1::b", 0.8, 1, 3),
    ]
    subjects, _ = aggregate_binary_classifier_predictions(rows)
    assert subjects[0]["prediction"] == 1
    assert subjects[0]["probability"] == pytest.approx(0.55)


def test_rows_without_response_id_are_their_own_response():
    rows = [_row("s1", None, 0.6, 1, 0), _row("s1", None, 0.3, 1, 1)]
    subjects, _ = aggregate_binary_classifier_predictions(rows)
    assert subjects[0]["probability"] == pytest.approx(0.45)
    assert subjects[0]["prediction"] == 0


def test_policy_override_is_opt_in_and_validated():
    # Pooled Turkish audio defaults to its own rule; the MFCC caches opt into the mean.
    pooled = {"dataset": "turkish", "dataset_variant": "pooled_t17", "input_modality": "audio_only"}
    assert classifier_aggregation_policy(pooled) != HIERARCHICAL_MEAN_CLASSIFIER_POLICY
    assert (
        classifier_aggregation_policy({**pooled, "classifier_aggregation_policy": HIERARCHICAL_MEAN_CLASSIFIER_POLICY})
        == HIERARCHICAL_MEAN_CLASSIFIER_POLICY
    )
    with pytest.raises(ValueError):
        classifier_aggregation_policy({"classifier_aggregation_policy": "something_else"})


def test_opensmile_prediction_backends():
    assert resolve_prediction_backend({"model_backend": "opensmile_mfcc"}, "xgb_raw") == "opensmile_mfcc_xgb_raw"
    assert optuna100_policy.prediction_backend("opensmile_mfcc") == "opensmile_mfcc_xgb_optuna100"
    assert optuna100_policy.prediction_backend("qwen2audio") == optuna100_policy.QWEN_PREDICTION_BACKEND


def test_locked_folds_and_path_maps(tmp_path):
    payload = {"0": {"final_eval_subject_ids": ["a", "b"]}, "1": {"final_eval_subject_ids": ["c"]}}
    path = tmp_path / "folds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    folds = load_locked_folds(path, digest, {"a", "b", "c"})
    assert folds["0"] == {"outer_train_subject_ids": ["c"], "final_eval_subject_ids": ["a", "b"]}
    with pytest.raises(ValueError):
        load_locked_folds(path, "0" * 64, {"a", "b", "c"})
    with pytest.raises(ValueError):
        load_locked_folds(path, digest, {"a", "b", "c", "d"})

    maps = parse_path_maps(["/gpfs/x/=/local/y/"])
    rows = remap_manifest_rows([{"audio_path": "/gpfs/x/1.wav", "audio_paths": ["/gpfs/x/1.wav"]}], maps)
    assert rows[0] == {"audio_path": "/local/y/1.wav", "audio_paths": ["/local/y/1.wav"]}
    with pytest.raises(ValueError):
        parse_path_maps(["no-separator"])


@pytest.mark.skipif(shutil.which("SMILExtract") is None, reason="SMILExtract is not installed")
def test_run_smile_returns_608_finite_functionals():
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(16000 * 3)).astype(np.float32)
    names, vector = run_smile(
        audio,
        smile_bin=shutil.which("SMILExtract"),
        config_text=SMILE_CONFIG.read_text(encoding="utf-8"),
        sample_id="synthetic",
    )
    assert len(names) == vector.size == 608
    assert "frameTime" not in names and "frameIndex" not in names
    assert np.isfinite(vector).all()
