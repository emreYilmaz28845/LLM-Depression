from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.model.gemma4_io import (
    GEMMA4_EVALUATION_VIEW,
    GEMMA4_LORA_TARGET_REGEX,
    GEMMA4_MODEL_REVISION,
    GEMMA4_TURN_TERMINATOR,
    Gemma4SFTCollator,
    expected_gemma4_audio_tokens,
    prepare_gemma4_example,
    prepare_gemma4_examples,
    validate_gemma4_audio,
    validate_gemma4_config,
)

AUDIO_MARKER = "<|audio|>"


def gemma_config(**overrides) -> dict:
    config = {
        "model_backend": "gemma4",
        "dataset": "daic",
        "model_revision": GEMMA4_MODEL_REVISION,
        "prompt": {
            "system": "system",
            "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "sample_mode": "participant_speech_packed30",
            "participant_chunk_samples": 480000,
            "inter_span_silence_samples": 0,
            "audio_text_transcript_scope": "full_participant",
            "train_chunk_policy": "all_chunks_subject_normalized",
            "eval_chunk_policy": "all_chunks_mean_score",
            "loss_weight_rescale": "mean_one",
            "equal_row_weight": False,
            "transcript_max_chars": 0,
            "allow_empty_transcript": False,
        },
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
            "evaluation_view": GEMMA4_EVALUATION_VIEW,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


class FakeProcessor:
    """Char-level processor fake that preserves the exact prompt-prefix property.

    ``apply_chat_template`` renders the official generation prompt with the
    ``<|audio|>`` marker; ``__call__`` expands the marker into
    ``ceil(samples/640)`` audio token ids, keeping prompt ids an exact prefix of
    training ids, and returns the Gemma tensor key set.
    """

    def __init__(self, audio_token="<|audio|>", pad_token_id=0, sampling_rate=16000):
        self.audio_token = audio_token
        self.audio_token_id = 258881
        self.pad_token_id = pad_token_id
        self.feature_extractor = SimpleNamespace(sampling_rate=sampling_rate)
        self.chat_template = "fake"
        self.tokenizer = SimpleNamespace(
            pad_token_id=pad_token_id,
            convert_tokens_to_ids=lambda token: self.audio_token_id,
        )
        self._captured: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self._captured.append({"messages": messages, "kwargs": dict(kwargs)})
        system = messages[0]["content"]
        user_content = messages[-1]["content"]
        if isinstance(user_content, list):
            text = "".join(item["text"] for item in user_content if item.get("type") == "text")
            marker = AUDIO_MARKER if any(item.get("type") == "audio" for item in user_content) else ""
        else:
            text = str(user_content)
            marker = ""
        return (
            f"<bos><|turn>system\n{system}<turn|>\n"
            f"<|turn>user\n{marker}{text}<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )

    def _encode(self, text: str, audio_list: list[np.ndarray]) -> list[int]:
        ids: list[int] = []
        parts = text.split(AUDIO_MARKER)
        for index, part in enumerate(parts):
            ids.extend(ord(char) for char in part)
            if index < len(parts) - 1:
                frames = int(math.ceil(len(audio_list[index % len(audio_list)]) / 640))
                ids.extend([self.audio_token_id] * frames)
        return ids

    def __call__(self, text=None, audio=None, sampling_rate=None, padding=False, return_tensors=None, return_mm_token_type_ids=True, **kwargs):
        if isinstance(text, str):
            text = [text]
        audio_list = list(audio or [])
        batch: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "mm_token_type_ids": [],
            "input_features": [],
            "input_features_mask": [],
        }
        for sample_text in list(text):
            ids = self._encode(sample_text, audio_list)
            batch["input_ids"].append(ids)
            batch["attention_mask"].append([1] * len(ids))
            mm = [0] * len(ids)
            position = 0
            for wav in audio_list:
                frames = int(math.ceil(len(wav) / 640))
                if AUDIO_MARKER in sample_text:
                    position = sample_text.index(AUDIO_MARKER, position) + 1
                    for offset in range(frames):
                        mm[position + offset] = 3
            batch["mm_token_type_ids"].append(mm)
            for wav in audio_list:
                frames = int(math.ceil(len(wav) / 640))
                batch["input_features"].append(np.zeros((frames, 640), dtype=np.float32))
                batch["input_features_mask"].append(np.ones(frames, dtype=bool))
        if not audio_list:
            batch.pop("input_features")
            batch.pop("input_features_mask")
        return {key: np.asarray(value) for key, value in batch.items()}


def make_example(modality: str = "audio_text", label: int = 1, with_audio: bool = True) -> dict:
    example = {
        "dataset": "daic",
        "subject_id": "300",
        "sample_id": "300__000",
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "internal_label_text": "Depressed" if label else "Non-depressed",
        "transcript": "full participant transcript",
        "input_modality": modality,
        "prompt_system_text": "You are a psychologist analyzing speech and transcript information for depression screening.",
        "prompt_user_text": "Based on the audio and transcript, determine whether the subject is Depressed or Non-depressed.",
        "loss_weight": 1.25,
    }
    if with_audio:
        example["audio_paths"] = ["/tmp/fake.wav"]
    return example


class TestValidateGemma4Audio:
    def test_accepts_valid_waveform(self) -> None:
        wav = np.zeros(480000, dtype=np.float32)
        assert validate_gemma4_audio(wav, 16000) is wav

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            validate_gemma4_audio(np.zeros((2, 640), dtype=np.float32), 16000)

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            validate_gemma4_audio(np.zeros(640, dtype=np.float64), 16000)

    def test_rejects_wrong_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="16000"):
            validate_gemma4_audio(np.zeros(640, dtype=np.float32), 8000)

    def test_rejects_non_finite(self) -> None:
        wav = np.zeros(640, dtype=np.float32)
        wav[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_gemma4_audio(wav, 16000)

    def test_rejects_out_of_range(self) -> None:
        wav = np.ones(640, dtype=np.float32) * 2.0
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            validate_gemma4_audio(wav, 16000)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            validate_gemma4_audio(np.zeros(0, dtype=np.float32), 16000)

    def test_rejects_over_480000_without_truncating(self) -> None:
        wav = np.zeros(480001, dtype=np.float32)
        with pytest.raises(ValueError, match="480000"):
            validate_gemma4_audio(wav, 16000)
        assert len(wav) == 480001

    def test_expected_audio_tokens(self) -> None:
        assert expected_gemma4_audio_tokens(640) == 1
        assert expected_gemma4_audio_tokens(641) == 2
        assert expected_gemma4_audio_tokens(480000) == 750


class TestPrepareGemma4Example:
    def test_requires_raw_system_and_user_fields(self) -> None:
        processor = FakeProcessor()
        config = gemma_config()
        example = make_example()
        del example["prompt_system_text"]
        with pytest.raises(ValueError, match="prompt_system_text"):
            prepare_gemma4_example(example, config, processor)
        example = make_example()
        del example["prompt_user_text"]
        with pytest.raises(ValueError, match="prompt_user_text"):
            prepare_gemma4_example(example, config, processor)

    def test_message_structure_audio_text(self) -> None:
        processor = FakeProcessor()
        example = prepare_gemma4_example(make_example("audio_text"), gemma_config(), processor)
        messages = processor._captured[-1]["messages"]
        assert messages[0]["role"] == "system"
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "audio"}
        assert "audio" not in user_content[0]
        assert user_content[1]["type"] == "text"
        assert AUDIO_MARKER in example["prompt_text"]

    def test_message_structure_audio_only(self) -> None:
        processor = FakeProcessor()
        example = prepare_gemma4_example(make_example("audio_only"), gemma_config(), processor)
        messages = processor._captured[-1]["messages"]
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "audio"}
        assert user_content[1]["type"] == "text"
        assert AUDIO_MARKER in example["prompt_text"]

    def test_message_structure_text_only_has_no_audio_item(self) -> None:
        processor = FakeProcessor()
        config = gemma_config()
        config["data"]["use_audio"] = False
        config["data"]["use_text"] = True
        example = prepare_gemma4_example(make_example("text_only", with_audio=False), config, processor)
        messages = processor._captured[-1]["messages"]
        assert isinstance(messages[1]["content"], str)
        assert AUDIO_MARKER not in example["prompt_text"]

    def test_render_kwargs_disable_thinking(self) -> None:
        processor = FakeProcessor()
        prepare_gemma4_example(make_example(), gemma_config(), processor)
        kwargs = processor._captured[-1]["kwargs"]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert kwargs["tokenize"] is False

    def test_training_text_is_prompt_plus_label_and_turn_terminator(self) -> None:
        processor = FakeProcessor()
        example = prepare_gemma4_example(make_example(label=0), gemma_config(), processor)
        assert example["training_text"] == example["prompt_text"] + "Non-depressed" + GEMMA4_TURN_TERMINATOR

    def test_only_backend_prompt_fields_change(self) -> None:
        processor = FakeProcessor()
        source = make_example(label=1)
        original = dict(source)
        prepared = prepare_gemma4_example(source, gemma_config(), processor)
        for key, value in original.items():
            assert prepared[key] == value, key
        assert "prompt_text" in prepared and "training_text" in prepared
        assert prepared["prompt_text"].startswith("<bos><|turn>system")
        assert prepared["training_text"].endswith(GEMMA4_TURN_TERMINATOR)


class TestGemma4SFTCollator:
    def _collator(self, processor: FakeProcessor | None = None, debug: bool = False) -> Gemma4SFTCollator:
        return Gemma4SFTCollator(processor=processor or FakeProcessor(), debug=debug)

    def test_batch_size_one_audio_labels_mask_prompt(self) -> None:
        processor = FakeProcessor()
        example = make_example(label=1)
        example["audio_arrays"] = [np.zeros(640, dtype=np.float32)]
        prepared = prepare_gemma4_example(example, gemma_config(), processor)
        prepared["audio_arrays"] = example["audio_arrays"]
        batch = Gemma4SFTCollator(processor=processor)([prepared])
        prompt_len = len(processor._encode(prepared["prompt_text"], prepared["audio_arrays"]))
        assert batch["input_ids"].shape[0] == 1
        assert int((batch["labels"][0, :prompt_len] == -100).sum()) == prompt_len
        assert int((batch["labels"][0, prompt_len:] == -100).sum()) == 0
        assert "input_features" in batch
        assert batch["input_features"].dtype == torch_float32()
        assert batch["input_features_mask"].dtype == torch_bool()

    def test_text_only_has_no_audio_keys(self) -> None:
        processor = FakeProcessor()
        config = gemma_config()
        config["data"]["use_audio"] = False
        config["data"]["use_text"] = True
        example = prepare_gemma4_example(make_example("text_only", with_audio=False), config, processor)
        batch = Gemma4SFTCollator(processor=processor)([example])
        assert "input_features" not in batch
        assert "input_features_mask" not in batch

    def test_multi_audio_rejected(self) -> None:
        processor = FakeProcessor()
        example = make_example()
        example["audio_arrays"] = [np.zeros(640, dtype=np.float32), np.zeros(640, dtype=np.float32)]
        with pytest.raises(ValueError, match="exactly one waveform"):
            Gemma4SFTCollator(processor=processor)([example])

    def test_invalid_audio_rejected(self) -> None:
        processor = FakeProcessor()
        example = make_example()
        example["audio_arrays"] = [np.zeros(640, dtype=np.float64)]
        with pytest.raises(ValueError, match="float32"):
            Gemma4SFTCollator(processor=processor)([example])

    def test_right_padding_for_batch(self) -> None:
        processor = FakeProcessor()
        collator = Gemma4SFTCollator(processor=processor)
        short = make_example(label=1)
        short["sample_id"] = "s1"
        short["audio_arrays"] = [np.zeros(640, dtype=np.float32)]
        short = prepare_gemma4_example(short, gemma_config(), processor)
        long = make_example(label=0)
        long["sample_id"] = "s2"
        long["audio_arrays"] = [np.zeros(1280, dtype=np.float32)]
        long = prepare_gemma4_example(long, gemma_config(), processor)
        batch = collator([short, long])
        assert batch["input_ids"].shape[0] == 2
        expected_seq = max(
            len(processor._encode(short["training_text"], short["audio_arrays"])),
            len(processor._encode(long["training_text"], long["audio_arrays"])),
        )
        assert batch["input_ids"].shape[1] == expected_seq
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
        assert batch["input_features"].shape[1] == 2  # padded to longest frames
        assert batch["input_features"].shape[0] == 2

    def test_loss_weight_retained(self) -> None:
        processor = FakeProcessor()
        example = make_example()
        example["audio_arrays"] = [np.zeros(640, dtype=np.float32)]
        example = prepare_gemma4_example(example, gemma_config(), processor)
        batch = Gemma4SFTCollator(processor=processor)([example])
        assert float(batch["loss_weight"][0]) == 1.25

    def test_debug_info_exposed(self) -> None:
        processor = FakeProcessor()
        collator = Gemma4SFTCollator(processor=processor, debug=True)
        example = make_example()
        example["audio_arrays"] = [np.zeros(640, dtype=np.float32)]
        example = prepare_gemma4_example(example, gemma_config(), processor)
        collator([example])
        debug = collator.last_debug_example
        assert debug is not None
        assert debug["sample_id"] == example["sample_id"]
        assert "decoded_training_text" in debug
        assert "decoded_prompt_text" in debug
        assert "input_ids" in debug
        assert "labels" in debug
        assert "unmasked_token_ids" in debug


def torch_float32():
    import torch

    return torch.float32


def torch_bool():
    import torch

    return torch.bool


class TestValidateGemma4Config:
    def test_valid_configs_pass(self) -> None:
        validate_gemma4_config(gemma_config())
        text_config = gemma_config()
        text_config["data"]["use_audio"] = False
        text_config["data"]["use_text"] = True
        validate_gemma4_config(text_config)

    def test_non_gemma_backend_is_noop(self) -> None:
        config = gemma_config()
        config["model_backend"] = "qwen2audio"
        validate_gemma4_config(config)

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"dataset": "cmdc"}, "dataset"),
            ({"model_revision": "deadbeef"}, "model_revision"),
        ],
    )
    def test_rejects_wrong_identity(self, override, message) -> None:
        with pytest.raises(ValueError, match=message):
            validate_gemma4_config(gemma_config(**override))

    def test_rejects_non_packed30_sample_mode(self) -> None:
        config = gemma_config()
        config["data"]["sample_mode"] = "subject_audio"
        with pytest.raises(ValueError, match="sample_mode"):
            validate_gemma4_config(config)

    def test_rejects_wrong_chunk_samples(self) -> None:
        config = gemma_config()
        config["data"]["participant_chunk_samples"] = 240000
        with pytest.raises(ValueError, match="participant_chunk_samples"):
            validate_gemma4_config(config)

    def test_rejects_wrong_transcript_scope(self) -> None:
        config = gemma_config()
        config["data"]["use_audio"] = True
        config["data"]["use_text"] = True
        config["data"]["audio_text_transcript_scope"] = "chunk_aligned"
        with pytest.raises(ValueError, match="audio_text_transcript_scope"):
            validate_gemma4_config(config)

    def test_rejects_audio_adapter_enabled(self) -> None:
        config = gemma_config()
        config["audio_adapter"]["enabled"] = True
        with pytest.raises(ValueError, match="audio_adapter.enabled"):
            validate_gemma4_config(config)

    def test_rejects_train_projector(self) -> None:
        config = gemma_config()
        config["audio_adapter"]["train_projector"] = True
        with pytest.raises(ValueError, match="train_projector"):
            validate_gemma4_config(config)

    def test_rejects_tune_audio_encoder(self) -> None:
        config = gemma_config()
        config["lora"]["tune_audio_encoder"] = True
        with pytest.raises(ValueError, match="tune_audio_encoder"):
            validate_gemma4_config(config)

    def test_rejects_fp32(self) -> None:
        config = gemma_config()
        config["training"]["bf16"] = False
        with pytest.raises(ValueError, match="bf16"):
            validate_gemma4_config(config)

    def test_rejects_no_gradient_checkpointing(self) -> None:
        config = gemma_config()
        config["training"]["gradient_checkpointing"] = False
        with pytest.raises(ValueError, match="gradient_checkpointing"):
            validate_gemma4_config(config)

    def test_rejects_wrong_selection_metric(self) -> None:
        config = gemma_config()
        config["training"]["selection_metric"] = "inner_val_positive_f1"
        with pytest.raises(ValueError, match="selection_metric"):
            validate_gemma4_config(config)

    def test_rejects_wrong_headline_mode(self) -> None:
        config = gemma_config()
        config["evaluation"]["headline_mode"] = "likelihood"
        with pytest.raises(ValueError, match="headline_mode"):
            validate_gemma4_config(config)

    def test_rejects_wrong_evaluation_view(self) -> None:
        config = gemma_config()
        config["evaluation"]["evaluation_view"] = "fixed_k4"
        with pytest.raises(ValueError, match="evaluation_view"):
            validate_gemma4_config(config)

    def test_rejects_wrong_lora_target(self) -> None:
        config = gemma_config()
        config["lora"]["target_modules"] = ".*v_proj.*"
        with pytest.raises(ValueError, match="target_modules"):
            validate_gemma4_config(config)

    def test_rejects_paired_batching(self) -> None:
        config = gemma_config()
        config["evaluation"]["candidate_batching"] = "paired"
        with pytest.raises(ValueError, match="paired"):
            validate_gemma4_config(config)


def test_prepare_gemma4_examples_keeps_non_gemma_examples_untouched() -> None:
    config = gemma_config()
    config["model_backend"] = "qwen2audio"
    source = [make_example()]
    prepared = prepare_gemma4_examples(source, config, FakeProcessor())
    assert prepared == source
