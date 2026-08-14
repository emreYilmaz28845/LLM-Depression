"""Model-free Gemma harmonized-contract tests for the four non-DAIC datasets.

Covers Section 9.1 of docs/QWEN_GEMMA_COVERAGE_COMPLETION_RUNBOOK.md:
- D3TEC, Androids, CMDC, Turkish require ``harmonized_response_windows``;
- DAIC continues to require ``participant_speech_packed30``;
- every example path carries ``prompt_system_text`` / ``prompt_user_text``;
- audio examples stay finite mono float32 16 kHz and at most 480,000 samples;
- native and English conditions produce the same prompt-field contract;
- Qwen example content (prompt_text/training_text) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    build_examples,
    build_prompt_text,
    render_user_prompt_text,
)
from src.model.gemma4_io import (
    GEMMA4_AUDIO_SAMPLING_RATE,
    GEMMA4_HARMONIZED_SAMPLE_MODE,
    GEMMA4_LORA_TARGET_REGEX,
    GEMMA4_MAX_AUDIO_SAMPLES,
    GEMMA4_MODEL_REVISION,
    prepare_gemma4_example,
    validate_gemma4_audio,
    validate_gemma4_config,
)

NON_DAIC_DATASETS = ("d3tec", "androids_interview", "cmdc", "turkish")
MODALITIES = ("audio_text", "audio_only", "text_only")


def harmonized_config(
    dataset: str,
    modality: str = "audio_text",
    *,
    english: bool = False,
) -> dict:
    use_audio = modality != "text_only"
    use_text = modality != "audio_only"
    data: dict = {
        "use_audio": use_audio,
        "use_text": use_text,
        "sample_mode": GEMMA4_HARMONIZED_SAMPLE_MODE,
        "segment_seconds": 30.0,
        "audio_text_transcript_scope": "full_subject",
        "train_chunk_policy": "all_windows_hierarchical_weighted",
        "eval_chunk_policy": "all_windows",
        "transcript_max_chars": 0,
        "allow_empty_transcript": False,
    }
    if dataset == "turkish":
        data["t17"] = True
    config: dict = {
        "model_backend": "gemma4",
        "dataset": dataset,
        "model_revision": GEMMA4_MODEL_REVISION,
        "prompt": {
            "system": "You are a psychologist analyzing speech and transcript information for depression screening.",
            "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}",
            "prompt_language": "english",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": data,
        "audio_adapter": {"enabled": False, "adapter_dim": 512, "dropout": 0.1, "train_projector": False},
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": GEMMA4_LORA_TARGET_REGEX,
        },
        "training": {
            "bf16": True,
            "gradient_checkpointing": True,
            "selection_metric": "inner_val_macro_f1",
            "selection_metric_mode": "max",
        },
        "evaluation": {
            "sample_prediction_mode": "original_teacher_forced",
            "headline_mode": "original_teacher_forced",
            "evaluation_view": "harmonized_all_windows_full_coverage",
        },
    }
    if english:
        config["transcripts"] = {
            "variant": "english",
            "cache_path": "/gpfs/translations/harmonized_en_complete_v1/placeholder/accepted.jsonl",
            "minimum_status": "automatic_low",
            "require_complete": True,
            "include_failed": False,
        }
    return config


def make_row(
    dataset: str,
    subject: str,
    sample: str,
    audio_path: Path,
    transcript: str,
) -> dict:
    row: dict = {
        "dataset": dataset,
        "subject_id": subject,
        "sample_id": sample,
        "label": 0,
        "label_text": "Non-depressed",
        "transcript": transcript,
        "audio_path": str(audio_path),
    }
    if dataset == "d3tec":
        prompt_id = int(sample.split("_q")[-1])
        row.update({"prompt_id": prompt_id, "response_id": f"{subject}__{prompt_id}", "full_response_transcript": transcript})
    elif dataset == "androids_interview":
        turn_id = int(sample.split("_q")[-1])
        row.update({"turn_id": turn_id, "response_id": f"{subject}__r{turn_id}", "full_turn_transcript": transcript})
    else:
        row["question_id"] = sample
    return row


@pytest.mark.parametrize("dataset", NON_DAIC_DATASETS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_harmonized_configs_validate_for_all_datasets_and_modalities(
    dataset: str, modality: str
) -> None:
    validate_gemma4_config(harmonized_config(dataset, modality))


@pytest.mark.parametrize("dataset", NON_DAIC_DATASETS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_english_condition_configs_validate(dataset: str, modality: str) -> None:
    validate_gemma4_config(harmonized_config(dataset, modality, english=True))


def test_non_daic_rejects_packed30_sample_mode() -> None:
    config = harmonized_config("cmdc")
    config["data"]["sample_mode"] = "participant_speech_packed30"
    with pytest.raises(ValueError, match="sample_mode"):
        validate_gemma4_config(config)


def test_non_daic_rejects_wrong_segment_seconds() -> None:
    config = harmonized_config("turkish")
    config["data"]["segment_seconds"] = 31.0
    with pytest.raises(ValueError, match="segment_seconds"):
        validate_gemma4_config(config)


def test_androids_audio_text_requires_full_subject_scope() -> None:
    config = harmonized_config("androids_interview", "audio_text")
    config["data"]["audio_text_transcript_scope"] = "full_turn"
    with pytest.raises(ValueError, match="audio_text_transcript_scope"):
        validate_gemma4_config(config)


def test_daic_rejects_harmonized_sample_mode() -> None:
    config = harmonized_config("daic", "audio_text")
    config["data"]["sample_mode"] = GEMMA4_HARMONIZED_SAMPLE_MODE
    with pytest.raises(ValueError, match="participant_speech_packed30"):
        validate_gemma4_config(config)


def test_daic_keeps_packed30_constraints() -> None:
    config = harmonized_config("daic", "audio_text")
    config["data"]["sample_mode"] = "participant_speech_packed30"
    config["data"]["participant_chunk_samples"] = 480000
    config["data"]["audio_text_transcript_scope"] = "full_participant"
    validate_gemma4_config(config)
    config["data"]["participant_chunk_samples"] = 240000
    with pytest.raises(ValueError, match="participant_chunk_samples"):
        validate_gemma4_config(config)


def test_unsupported_dataset_rejected() -> None:
    with pytest.raises(ValueError, match="dataset"):
        validate_gemma4_config(harmonized_config("eatd"))


def _write_wav(path: Path, seconds: float) -> None:
    sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)


@pytest.mark.parametrize("dataset", NON_DAIC_DATASETS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_harmonized_examples_carry_raw_prompt_fields(
    tmp_path: Path, dataset: str, modality: str
) -> None:
    wav = tmp_path / "unit.wav"
    _write_wav(wav, 8.0)
    rows = [
        make_row(dataset, "s1", f"s1_q1", wav, "first response"),
        make_row(dataset, "s1", f"s1_q2", wav, "second response"),
        make_row(dataset, "s2", f"s2_q1", wav, "third response"),
    ]
    for index, row in enumerate(rows):
        if index % 2:
            row["label"] = 1
            row["label_text"] = "Depressed"
    config = harmonized_config(dataset, modality)
    examples = build_examples(rows, config, "train")
    assert examples
    for example in examples:
        assert example["prompt_system_text"] == config["prompt"]["system"]
        assert isinstance(example["prompt_user_text"], str) and example["prompt_user_text"].strip()
        assert example["prompt_user_text"] == render_user_prompt_text(
            config, example["transcript"], is_subject_bundle=False
        )
        if modality != "text_only":
            assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == 1


@pytest.mark.parametrize("dataset", NON_DAIC_DATASETS)
def test_harmonized_english_examples_carry_english_prompt_fields(
    tmp_path: Path, dataset: str
) -> None:
    wav = tmp_path / "unit.wav"
    _write_wav(wav, 8.0)
    rows = [
        make_row(dataset, "s1", "s1_q1", wav, "English translation of the answer"),
    ]
    config = harmonized_config(dataset, "text_only", english=True)
    examples = build_examples(rows, config, "train")
    assert len(examples) == 1
    example = examples[0]
    assert "English translation of the answer" in example["prompt_user_text"]
    assert "English translation of the answer" in example["prompt_text"]


def test_qwen_example_content_is_unchanged(tmp_path: Path) -> None:
    """Adding the raw prompt fields must not alter Qwen prompt/training text."""
    wav = tmp_path / "unit.wav"
    _write_wav(wav, 8.0)
    dataset = "cmdc"
    rows = [make_row(dataset, "s1", "s1_q1", wav, "transcript text")]
    config = harmonized_config(dataset, "audio_text")
    config["model_backend"] = "qwen2audio"
    config["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    examples = build_examples(rows, config, "train")
    assert len(examples) == 1
    example = examples[0]
    expected_user = render_user_prompt_text(config, "transcript text", is_subject_bundle=False)
    expected_prompt = build_prompt_text(
        system_prompt=config["prompt"]["system"],
        user_text=expected_user,
        num_audios=1,
        use_audio=True,
        audio_placeholder=AUDIO_PLACEHOLDER,
    )
    assert example["prompt_text"] == expected_prompt
    assert example["training_text"] == f"{expected_prompt}Non-depressed<|im_end|>\n"
    # The raw fields are pure additions: Qwen prompt rendering is untouched.
    assert example["prompt_system_text"] == config["prompt"]["system"]
    assert example["prompt_user_text"] == expected_user


@pytest.mark.parametrize("dataset", NON_DAIC_DATASETS)
@pytest.mark.parametrize("modality", ("audio_text", "audio_only"))
def test_gemma_prepare_accepts_harmonized_audio_examples(
    tmp_path: Path, dataset: str, modality: str
) -> None:
    from types import SimpleNamespace

    from tests.test_gemma4_io import FakeProcessor

    wav = tmp_path / "unit.wav"
    _write_wav(wav, 8.0)
    rows = [make_row(dataset, "s1", "s1_q1", wav, "first response")]
    config = harmonized_config(dataset, modality)
    examples = build_examples(rows, config, "train")
    assert len(examples) == 1
    prepared = prepare_gemma4_example(examples[0], config, FakeProcessor())
    assert prepared["prompt_text"].startswith("<bos><|turn>system")
    assert prepared["training_text"].startswith(prepared["prompt_text"])
    assert prepared["prompt_user_text"] == examples[0]["prompt_user_text"]


def test_harmonized_window_audio_stays_within_gemma_limits(tmp_path: Path) -> None:
    # A 30 s window at 16 kHz is exactly 480,000 samples: valid. Anything above
    # must be rejected by the Gemma audio validator, never truncated.
    assert 30 * GEMMA4_AUDIO_SAMPLING_RATE == GEMMA4_MAX_AUDIO_SAMPLES
    full = np.zeros(GEMMA4_MAX_AUDIO_SAMPLES, dtype=np.float32)
    assert validate_gemma4_audio(full, GEMMA4_AUDIO_SAMPLING_RATE) is full
    with pytest.raises(ValueError, match="480000"):
        validate_gemma4_audio(
            np.zeros(GEMMA4_MAX_AUDIO_SAMPLES + 1, dtype=np.float32),
            GEMMA4_AUDIO_SAMPLING_RATE,
        )
