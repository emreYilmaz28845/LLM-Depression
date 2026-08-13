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


def test_mean_probability_backend_is_configurable_and_defaults_to_qwen() -> None:
    rows = [
        classifier_row("300", 0.9, 0),
        classifier_row("300", 0.2, 0),
        classifier_row("301", 0.4, 1),
        classifier_row("301", 0.6, 1),
    ]
    _, metrics = aggregate_binary_classifier_predictions(rows)
    assert metrics["prediction_backend"] == "qwen_hidden_classifier"
    _, metrics_gemma = aggregate_binary_classifier_predictions(
        rows, prediction_backend="gemma4_hidden_logreg_raw"
    )
    assert metrics_gemma["prediction_backend"] == "gemma4_hidden_logreg_raw"
    subject_rows, _ = aggregate_binary_classifier_predictions(
        rows, prediction_backend="gemma4_hidden_xgb_raw"
    )
    assert all(
        row["prediction_backend"] == "gemma4_hidden_xgb_raw" for row in subject_rows
    )


def test_gemma_daic_contract_refuses_wrong_subject_counts(tmp_path: Path) -> None:
    from baselines.qwen_hidden_classifier import _enforce_gemma_daic_contract

    metadata = {"input_modality": "text_only"}
    too_few_train = [
        {"subject_id": str(index), "label": index % 2} for index in range(100)
    ]
    with pytest.raises(ValueError, match="107 official training subjects"):
        _enforce_gemma_daic_contract(
            metadata,
            too_few_train,
            [{"subject_id": str(index)} for index in range(47)],
            {str(index) for index in range(100)},
        )
    train = [
        {"subject_id": str(index), "label": index % 2} for index in range(107)
    ]
    with pytest.raises(ValueError, match="47 official test subjects"):
        _enforce_gemma_daic_contract(
            metadata,
            train,
            [{"subject_id": str(index)} for index in range(20)],
            {str(index) for index in range(107)},
        )


def test_gemma_daic_contract_refuses_incomplete_chunk_coverage() -> None:
    from baselines.qwen_hidden_classifier import (
        _enforce_complete_chunk_coverage,
        _enforce_gemma_daic_contract,
    )

    metadata = {"input_modality": "audio_only"}
    train = []
    for subject in range(107):
        for chunk in range(14):
            train.append(
                {
                    "subject_id": str(subject),
                    "label": subject % 2,
                    "chunk_index": chunk,
                    "num_chunks": 14,
                }
            )
    test = [
        {"subject_id": str(index), "label": index % 2, "chunk_index": 0, "num_chunks": 1}
        for index in range(47)
    ]
    train_subjects = {str(subject) for subject in range(107)}
    _enforce_gemma_daic_contract(metadata, train, test, train_subjects)
    missing = [dict(row) for row in train]
    missing[3]["chunk_index"] = 13  # duplicate index 13, drops nothing but duplicates
    missing[3] = dict(missing[3])
    missing[3]["chunk_index"] = 5  # no-op keep
    del missing[3]["chunk_index"]
    with pytest.raises(ValueError, match="require chunk_index"):
        _enforce_complete_chunk_coverage(missing, "outer_train")
    dup = [dict(row) for row in train]
    dup[0]["chunk_index"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        _enforce_complete_chunk_coverage(dup, "outer_train")
    dropped = train[:-1]
    with pytest.raises(ValueError, match="missing"):
        _enforce_complete_chunk_coverage(dropped, "outer_train")


def test_gemma_fixed_head_identity_and_backends(tmp_path: Path) -> None:
    from src.utils import save_json, write_jsonl

    from baselines.qwen_hidden_classifier import (
        GEMMA4_LOGREG_PREDICTION_BACKEND,
        GEMMA4_VARIANTS,
        GEMMA4_XGB_PREDICTION_BACKEND,
        resolve_prediction_backend,
    )

    qwen_metadata = {"model_backend": None, "input_modality": "text_only"}
    assert (
        resolve_prediction_backend(qwen_metadata, "logreg_raw")
        == "qwen_hidden_classifier"
    )
    assert (
        resolve_prediction_backend(qwen_metadata, "xgb_pca32")
        == "qwen_hidden_classifier"
    )
    gemma_metadata = {"model_backend": "gemma4", "input_modality": "text_only"}
    assert (
        resolve_prediction_backend(gemma_metadata, "logreg_raw")
        == GEMMA4_LOGREG_PREDICTION_BACKEND
    )
    assert (
        resolve_prediction_backend(gemma_metadata, "xgb_raw")
        == GEMMA4_XGB_PREDICTION_BACKEND
    )
    with pytest.raises(ValueError, match="only"):
        resolve_prediction_backend(gemma_metadata, "xgb_pca32")
    assert set(GEMMA4_VARIANTS) == {"logreg_raw", "xgb_raw"}

    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(107, 8)).astype(np.float32)
    test_x = rng.normal(size=(47, 8)).astype(np.float32) + 50.0
    np.savez_compressed(cache / "outer_train.npz", vectors=train_x)
    np.savez_compressed(cache / "final_eval.npz", vectors=test_x)
    train_rows = [
        {"sample_id": f"tr{i}", "subject_id": f"tr{i}", "label": i % 2}
        for i in range(107)
    ]
    test_rows = [
        {"sample_id": f"te{i}", "subject_id": f"te{i}", "label": i % 2}
        for i in range(47)
    ]
    write_jsonl(train_rows, cache / "outer_train_rows.jsonl")
    write_jsonl(test_rows, cache / "final_eval_rows.jsonl")
    save_json(
        {
            "dataset": "daic",
            "input_modality": "text_only",
            "condition": "text_only",
            "fold": 0,
            "model_backend": "gemma4",
            "checkpoint_dir": "synthetic/best_model",
            "protocol_id": PACKED30_PROTOCOL_ID,
        },
        cache / "extraction_metadata.json",
    )
    from baselines.qwen_hidden_classifier import run_variant

    summaries = [
        run_variant(cache, output, variant, seed=1337)
        for variant in ("logreg_raw", "xgb_raw")
    ]
    for variant, summary in zip(("logreg_raw", "xgb_raw"), summaries):
        variant_dir = output / variant
        metadata = __import__("json").loads(
            (variant_dir / "classifier_metadata.json").read_text()
        )
        expected_backend = (
            GEMMA4_LOGREG_PREDICTION_BACKEND
            if variant == "logreg_raw"
            else GEMMA4_XGB_PREDICTION_BACKEND
        )
        assert metadata["prediction_backend"] == expected_backend
        assert metadata["aggregation_policy"] == PACKED30_AGGREGATION_POLICY
        assert metadata["threshold"] == 0.5
        sample_rows = [
            __import__("json").loads(line)
            for line in (variant_dir / "predictions_subject_level.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(sample_rows) == 47
        assert all(
            row["prediction_backend"] == expected_backend for row in sample_rows
        )
        metrics = __import__("json").loads(
            (variant_dir / "metrics.json").read_text()
        )
        assert metrics["prediction_backend"] == expected_backend
        result_config = __import__("json").loads(
            (variant_dir / "result_config.json").read_text()
        )
        assert result_config["model_backend"] == "gemma4"
        assert result_config["prediction_backend"] == expected_backend


def test_officialdev_contract_refuses_wrong_subject_counts(tmp_path: Path) -> None:
    from baselines.qwen_hidden_classifier import _enforce_officialdev_contract

    metadata = {"input_modality": "text_only"}
    too_few_train = [
        {"subject_id": str(index), "label": index % 2} for index in range(85)
    ]
    with pytest.raises(ValueError, match="86 inner-training"):
        _enforce_officialdev_contract(
            metadata,
            too_few_train,
            [{"subject_id": str(index)} for index in range(35)],
            {str(index) for index in range(85)},
        )
    train = [
        {"subject_id": str(index), "label": index % 2} for index in range(86)
    ]
    with pytest.raises(ValueError, match="35 official development"):
        _enforce_officialdev_contract(
            metadata,
            train,
            [{"subject_id": str(index)} for index in range(20)],
            {str(index) for index in range(86)},
        )
    # Both classes in fit and eval sets are still required by run_variant.
    from baselines.qwen_hidden_classifier import (
        _enforce_complete_chunk_coverage,
    )

    audio_metadata = {"input_modality": "audio_only"}
    audio_train = []
    for subject in range(86):
        for chunk in range(2):
            audio_train.append(
                {
                    "subject_id": str(subject),
                    "label": subject % 2,
                    "chunk_index": chunk,
                    "num_chunks": 2,
                }
            )
    audio_test = [
        {"subject_id": str(index), "label": index % 2, "chunk_index": 0, "num_chunks": 1}
        for index in range(35)
    ]
    _enforce_officialdev_contract(
        audio_metadata, audio_train, audio_test, {str(s) for s in range(86)}
    )
    dropped = audio_train[:-1]
    with pytest.raises(ValueError, match="missing"):
        _enforce_complete_chunk_coverage(dropped, "outer_train")


def test_officialdev_qwen_backend_identity_and_run(tmp_path: Path) -> None:
    from src.utils import save_json, write_jsonl

    from baselines.qwen_hidden_classifier import (
        QWEN_LOGREG_PREDICTION_BACKEND,
        QWEN_PREDICTION_BACKEND,
        QWEN_XGB_PREDICTION_BACKEND,
        resolve_prediction_backend,
        run_variant,
    )

    officialdev_metadata = {
        "model_backend": "qwen2audio",
        "input_modality": "text_only",
        "evaluation_provenance": {
            "evaluation_protocol": "daic_official_train_inner_split_dev_evaluation"
        },
    }
    assert (
        resolve_prediction_backend(officialdev_metadata, "logreg_raw")
        == QWEN_LOGREG_PREDICTION_BACKEND
    )
    assert (
        resolve_prediction_backend(officialdev_metadata, "xgb_raw")
        == QWEN_XGB_PREDICTION_BACKEND
    )
    with pytest.raises(ValueError, match="only"):
        resolve_prediction_backend(officialdev_metadata, "xgb_pca32")
    # The locked default remains for Qwen caches outside the campaign.
    plain_metadata = {"model_backend": "qwen2audio", "input_modality": "text_only"}
    assert (
        resolve_prediction_backend(plain_metadata, "logreg_raw")
        == QWEN_PREDICTION_BACKEND
    )

    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    rng = np.random.default_rng(11)
    train_x = rng.normal(size=(86, 8)).astype(np.float32)
    test_x = rng.normal(size=(35, 8)).astype(np.float32) + 50.0
    np.savez_compressed(cache / "outer_train.npz", vectors=train_x)
    np.savez_compressed(cache / "final_eval.npz", vectors=test_x)
    train_rows = [
        {"sample_id": f"tr{i}", "subject_id": f"tr{i}", "label": i % 2}
        for i in range(86)
    ]
    test_rows = [
        {"sample_id": f"te{i}", "subject_id": f"te{i}", "label": i % 2}
        for i in range(35)
    ]
    write_jsonl(train_rows, cache / "outer_train_rows.jsonl")
    write_jsonl(test_rows, cache / "final_eval_rows.jsonl")
    save_json(
        {
            "dataset": "daic",
            "input_modality": "text_only",
            "condition": "text_only",
            "fold": 0,
            "model_backend": "qwen2audio",
            "checkpoint_dir": "synthetic/best_model",
            "protocol_id": PACKED30_PROTOCOL_ID,
            "evaluation_provenance": {
                "evaluation_protocol": "daic_official_train_inner_split_dev_evaluation"
            },
        },
        cache / "extraction_metadata.json",
    )
    summaries = [
        run_variant(cache, output, variant, seed=1337)
        for variant in ("logreg_raw", "xgb_raw")
    ]
    for variant, summary in zip(("logreg_raw", "xgb_raw"), summaries):
        variant_dir = output / variant
        metadata = __import__("json").loads(
            (variant_dir / "classifier_metadata.json").read_text()
        )
        expected_backend = (
            QWEN_LOGREG_PREDICTION_BACKEND
            if variant == "logreg_raw"
            else QWEN_XGB_PREDICTION_BACKEND
        )
        assert metadata["prediction_backend"] == expected_backend
        assert metadata["aggregation_policy"] == PACKED30_AGGREGATION_POLICY
        assert metadata["threshold"] == 0.5
        subject_rows = [
            __import__("json").loads(line)
            for line in (variant_dir / "predictions_subject_level.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(subject_rows) == 35
        assert all(row["prediction_backend"] == expected_backend for row in subject_rows)
        metrics = __import__("json").loads(
            (variant_dir / "metrics.json").read_text()
        )
        assert metrics["prediction_backend"] == expected_backend
