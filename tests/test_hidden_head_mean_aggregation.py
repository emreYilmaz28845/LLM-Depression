from __future__ import annotations

from src.aggregate import (
    HIERARCHICAL_MEAN_CLASSIFIER_POLICY,
    TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
    aggregate_binary_classifier_predictions,
    aggregate_binary_classifier_response_rows,
)
from src.features.hidden_classifier_policy import (
    D3TEC_AGGREGATION_POLICY,
    PACKED30_AGGREGATION_POLICY,
    PACKED30_PROTOCOL_ID,
    classifier_aggregation_policy,
)


def _rows(policy: str) -> list[dict]:
    # One response votes Depressed confidently, two vote Non-depressed weakly:
    # majority vote says 0, the response-then-subject mean (0.617) says 1.
    windows = [("s1::r0", [0.95, 0.95, 0.95]), ("s1::r1", [0.45]), ("s1::r2", [0.45])]
    rows = []
    for response_id, probabilities in windows:
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "dataset": "d3tec",
                    "modality": "audio_only",
                    "sample_id": f"{response_id}_w{index}",
                    "subject_id": "s1",
                    "response_id": response_id,
                    "label": 1,
                    "probability": probability,
                    "predicted_class": int(probability >= 0.5),
                    "aggregation_policy": policy,
                }
            )
    rows.append(
        {
            "dataset": "d3tec",
            "modality": "audio_only",
            "sample_id": "s2::r0_w0",
            "subject_id": "s2",
            "response_id": "s2::r0",
            "label": 0,
            "probability": 0.2,
            "predicted_class": 0,
            "aggregation_policy": policy,
        }
    )
    return rows


def test_new_runs_declare_the_mean_policy_for_window_datasets() -> None:
    for dataset in ("d3tec", "cmdc", "androids_interview"):
        for modality in ("audio_only", "audio_text"):
            assert (
                classifier_aggregation_policy({"dataset": dataset, "input_modality": modality})
                == HIERARCHICAL_MEAN_CLASSIFIER_POLICY
            )
    assert (
        classifier_aggregation_policy(
            {"dataset": "turkish", "dataset_variant": "pooled_t17", "input_modality": "audio_only"}
        )
        == TURKISH_POOLED_AUDIO_AGGREGATION_POLICY
    )
    assert (
        classifier_aggregation_policy({"protocol_id": PACKED30_PROTOCOL_ID, "input_modality": "audio_only"})
        == PACKED30_AGGREGATION_POLICY
    )


def test_mean_policy_averages_windows_then_responses() -> None:
    subjects, metrics = aggregate_binary_classifier_predictions(_rows(HIERARCHICAL_MEAN_CLASSIFIER_POLICY))
    by_id = {row["subject_id"]: row for row in subjects}
    assert by_id["s1"]["prediction"] == 1
    assert abs(by_id["s1"]["probability"] - (0.95 + 0.45 + 0.45) / 3) < 1e-12
    assert by_id["s1"]["response_count"] == 3 and by_id["s1"]["sample_count"] == 5
    assert by_id["s2"]["prediction"] == 0
    assert metrics["binary_strict_macro_f1"] == 1.0
    assert metrics["aggregation_method"] == HIERARCHICAL_MEAN_CLASSIFIER_POLICY


def test_legacy_rows_keep_majority_vote() -> None:
    subjects, _ = aggregate_binary_classifier_predictions(_rows(D3TEC_AGGREGATION_POLICY))
    assert {row["subject_id"]: row["prediction"] for row in subjects}["s1"] == 0


def test_mean_policy_response_rows_average_windows() -> None:
    responses, _ = aggregate_binary_classifier_response_rows(_rows(HIERARCHICAL_MEAN_CLASSIFIER_POLICY))
    by_id = {row["response_id"]: row for row in responses}
    assert by_id["s1::r0"]["prediction"] == 1 and by_id["s1::r0"]["num_segments"] == 3
    assert by_id["s1::r1"]["prediction"] == 0
