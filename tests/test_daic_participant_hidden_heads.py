from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.aggregate import aggregate_binary_classifier_predictions
from src.data.daic import PACKED30_PROTOCOL_ID
from src.features.extract_qwen_hidden import _require_best_model
from src.features.hidden_classifier_policy import (
    DAIC_SUBJECT_WEIGHT_POLICY,
    PACKED30_AGGREGATION_POLICY,
    classifier_aggregation_policy,
    is_packed30_rows,
    response_normalized_sample_weights,
)
from src.features.qwen_hidden_collator import (
    PromptOnlyExtractionCollator,
    load_prompt_audio,
)


def classifier_row(subject_id: str, probability: float, label: int = 0) -> dict:
    return {
        "subject_id": subject_id,
        "sample_id": f"{subject_id}_c0",
        "label": label,
        "probability": probability,
        "predicted_class": int(probability >= 0.5),
        "classifier_aggregation": PACKED30_AGGREGATION_POLICY,
        "protocol_id": PACKED30_PROTOCOL_ID,
    }


def test_is_packed30_rows_and_aggregation_policy() -> None:
    assert is_packed30_rows({"protocol_id": PACKED30_PROTOCOL_ID})
    assert not is_packed30_rows({"protocol_id": "other"})
    assert not is_packed30_rows({})
    assert (
        classifier_aggregation_policy({"protocol_id": PACKED30_PROTOCOL_ID, "input_modality": "audio_only"})
        == PACKED30_AGGREGATION_POLICY
    )
    assert (
        classifier_aggregation_policy({"protocol_id": PACKED30_PROTOCOL_ID, "input_modality": "text_only"})
        == PACKED30_AGGREGATION_POLICY
    )
    assert (
        classifier_aggregation_policy({"dataset": "d3tec", "input_modality": "text_only"})
        == "one_prediction_per_subject"
    )


def test_head_fit_weights_are_subject_normalized_mean_one() -> None:
    rows = [
        {"subject_id": "300", "sample_id": "300_c0"},
        {"subject_id": "300", "sample_id": "300_c1"},
        {"subject_id": "300", "sample_id": "300_c2"},
        {"subject_id": "301", "sample_id": "301_c0"},
    ]
    weights, audit = response_normalized_sample_weights(
        rows, {"dataset": "daic", "protocol_id": PACKED30_PROTOCOL_ID}
    )
    assert audit["policy"] == DAIC_SUBJECT_WEIGHT_POLICY
    assert weights.mean() == pytest.approx(1.0)
    totals = {subject: 0.0 for subject in ("300", "301")}
    for row, weight in zip(rows, weights):
        totals[row["subject_id"]] += weight
    assert totals["300"] == pytest.approx(totals["301"])
    assert audit["equal_subject_totals"] is True


def test_mean_probability_classifier_aggregation_threshold_0_5() -> None:
    rows = [
        classifier_row("300", 0.9, 0),
        classifier_row("300", 0.2, 0),
        classifier_row("301", 0.4, 1),
        classifier_row("301", 0.6, 1),
    ]
    subject_rows, metrics = aggregate_binary_classifier_predictions(rows)
    assert len(subject_rows) == 2
    by_subject = {row["subject_id"]: row for row in subject_rows}
    assert by_subject["300"]["prediction"] == 1
    assert by_subject["300"]["probability"] == pytest.approx(0.55)
    # Mean probability exactly 0.5 is Depressed at the fixed >= 0.5 threshold.
    assert by_subject["301"]["prediction"] == 1
    assert by_subject["301"]["probability"] == pytest.approx(0.5)
    assert metrics["aggregation_method"] == "mean_depressed_probability_threshold_0_5"
    assert metrics["prediction_backend"] == "qwen_hidden_classifier"
    assert "auroc" in metrics


def test_legacy_classifier_aggregation_unchanged_without_marker() -> None:
    rows = [
        {
            "subject_id": "300",
            "sample_id": "300_c0",
            "label": 0,
            "probability": 0.6,
            "predicted_class": 1,
        },
        {
            "subject_id": "300",
            "sample_id": "300_c1",
            "label": 0,
            "probability": 0.2,
            "predicted_class": 0,
        },
    ]
    subject_rows, metrics = aggregate_binary_classifier_predictions(rows)
    assert len(subject_rows) == 1
    assert subject_rows[0]["aggregation_method"] == "majority_vote_probability_margin_tie_break"
    # Legacy path: tied votes broken by summed probability margin (0.2 - 0.6 < 0).
    assert subject_rows[0]["prediction"] == 0


def test_extractor_refuses_last_model(tmp_path: Path) -> None:
    best = tmp_path / "best_model"
    last = tmp_path / "last_model"
    best.mkdir()
    last.mkdir()
    _require_best_model(best)
    with pytest.raises(ValueError, match="best_model"):
        _require_best_model(last)


def test_extraction_collator_keeps_protocol_id_external(tmp_path: Path) -> None:
    import soundfile as sf

    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros(4800, dtype=np.float32), 16000, subtype="PCM_16")
    example = {
        "dataset": "daic",
        "sample_id": "300_participant_p30_000",
        "subject_id": "300",
        "label": 0,
        "partition": "outer_train",
        "fold": 0,
        "prompt_text": "prompt <|AUDIO|>",
        "protocol_id": PACKED30_PROTOCOL_ID,
        "chunk_index": 0,
        "num_chunks": 3,
        "audio_path": str(wav),
        "audio_spans": [{"start_frame": 0, "end_frame": 4800, "source_row_index": 0}],
        "participant_sample_count": 4800,
    }
    loaded = load_prompt_audio(example, 16000, False)
    assert loaded["audio_arrays"][0].shape == (4800,)

    class FakeProcessor:
        def __init__(self) -> None:
            self.feature_extractor = type("FE", (), {"sampling_rate": 16000})()

        def __call__(self, **kwargs):
            return {"input_ids": np.zeros((1, 8), dtype=np.int64), "attention_mask": np.ones((1, 8), dtype=np.int64)}

    model_inputs, metadata = PromptOnlyExtractionCollator(FakeProcessor())([loaded])
    assert "labels" not in model_inputs
    assert "protocol_id" in metadata[0]
    assert metadata[0]["protocol_id"] == PACKED30_PROTOCOL_ID
    assert metadata[0]["chunk_index"] == 0
    assert metadata[0]["num_chunks"] == 3
