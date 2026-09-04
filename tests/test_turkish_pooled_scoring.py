from __future__ import annotations

import pytest

from src.aggregate import TURKISH_POOLED_TEXT_PAIR_POLICY
from tools.score_turkish_pooled import ScoreError, score_rows


def _teacher_rows() -> list[dict[str, object]]:
    cases = {
        "both_positive": (0.8, 0.7),
        "both_negative": (-0.8, -0.7),
        "disagree_positive": (0.6, -0.1),
        "disagree_negative": (-0.6, 0.1),
    }
    labels = {"both_positive": 1, "both_negative": 0, "disagree_positive": 1, "disagree_negative": 0}
    rows = []
    for subject, (positive, negative) in cases.items():
        for condition, margin in (("pos_only_t17", positive), ("negative_only_t17", negative)):
            rows.append({
                "subject_id": subject, "sample_id": f"{subject}-{condition}", "label": labels[subject],
                "question_condition": condition, "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
                "teacher_forced_valid": True, "teacher_forced_prediction": int(margin > 0),
                "dep_score": margin, "non_score": 0.0,
            })
    return rows


def test_teacher_pair_rule_handles_all_margin_cases() -> None:
    result = score_rows(_teacher_rows(), route="teacher_forced", modality="text_only", view="combined", backend="original_teacher_forced")
    assert result["metrics"]["num_subjects"] == 4
    assert result["metrics"]["invalid_subjects"] == 0
    assert result["metrics"]["aggregation_policy"] == TURKISH_POOLED_TEXT_PAIR_POLICY


def test_invalid_teacher_component_is_strictly_wrong_without_replacement_prediction() -> None:
    rows = _teacher_rows()
    rows[0]["teacher_forced_valid"] = False
    rows[0]["teacher_forced_prediction"] = -1
    result = score_rows(rows, route="teacher_forced", modality="text_only", view="combined", backend="original_teacher_forced")
    assert result["metrics"]["invalid_subjects"] == 1
    assert result["metrics"]["binary_strict_accuracy"] < 1.0


def test_classifier_pair_rule_uses_probability_half_margins() -> None:
    rows = []
    for subject, label, positive, negative in (("s1", 1, 0.9, 0.2), ("s2", 0, 0.2, 0.1)):
        for condition, probability in (("pos_only_t17", positive), ("negative_only_t17", negative)):
            rows.append({
                "subject_id": subject, "sample_id": f"{subject}-{condition}", "label": label,
                "question_condition": condition, "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
                "probability": probability, "predicted_class": int(probability >= 0.5),
            })
    result = score_rows(rows, route="logreg", modality="text_only", view="combined", backend="qwen_hidden_logreg_raw")
    assert result["metrics"]["num_subjects"] == 2
    assert result["metrics"]["aggregation_policy"] == TURKISH_POOLED_TEXT_PAIR_POLICY
    assert result["backend"] == "qwen_hidden_logreg_raw"


def test_mixed_pair_policy_is_rejected() -> None:
    rows = _teacher_rows()
    rows[0]["aggregation_policy"] = "other"
    with pytest.raises((ScoreError, ValueError)):
        score_rows(rows, route="teacher_forced", modality="text_only", view="combined", backend="original_teacher_forced")


def test_audio_views_filter_raw_conditions_before_response_subject_mean_score() -> None:
    rows = []
    for subject, label, values in (("s1", 1, {"pos_only_t17": (0.9,), "negative_only_t17": (0.1, 0.8)}), ("s2", 0, {"pos_only_t17": (0.1,), "negative_only_t17": (0.2,)})):
        for condition, probabilities in values.items():
            for unit, probability in enumerate(probabilities):
                rows.append({
                    "dataset": "turkish", "dataset_variant": "pooled_t17", "modality": "audio_only",
                    "subject_id": subject, "sample_id": f"{subject}-{condition}-{unit}", "label": label,
                    "question_condition": condition, "response_id": f"{subject}-{condition}-unit{unit}",
                    "num_segments": 1, "probability": probability, "predicted_class": int(probability >= 0.5),
                })
    result = score_rows(rows, route="logreg", modality="audio_only", view="combined", backend="qwen_hidden_logreg_raw")
    assert result["metrics"]["num_subjects"] == 2
    assert result["metrics"]["aggregation_policy"] == "turkish_pooled_audio_response_subject_mean_score_margin_v1"
