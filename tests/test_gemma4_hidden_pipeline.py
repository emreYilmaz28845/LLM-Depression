from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.features.extract_qwen_hidden import (
    BACKEND_HIDDEN_SIZES,
    CACHE_SCHEMA_VERSION_GEMMA4,
    CACHE_SCHEMA_VERSION_QWEN,
    QWEN_HIDDEN_SIZES,
    _apply_subject_selection,
    _backend_cache_schema,
    _decoder_hidden_size,
    _parent_attempt_id,
    load_subject_selection,
)
from src.features.gemma4_hidden_collator import (
    GEMMA4_MODEL_INPUT_KEYS,
    Gemma4PromptOnlyExtractionCollator,
)
from src.model.gemma4_io import GEMMA4_MODEL_REVISION
from src.utils import save_json, write_jsonl

AUDIO_MARKER = "<|audio|>"


class FakeGemmaProcessor:
    """Char-level Gemma processor fake with the exact prompt template and
    multimodal key set used by the real pinned processor."""

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
            text = "".join(
                item["text"] for item in user_content if item.get("type") == "text"
            )
            marker = (
                AUDIO_MARKER
                if any(item.get("type") == "audio" for item in user_content)
                else ""
            )
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

    def __call__(
        self,
        text=None,
        audio=None,
        sampling_rate=None,
        padding=False,
        return_tensors=None,
        return_mm_token_type_ids=True,
        **kwargs,
    ):
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


def make_example(
    modality: str = "audio_text", label: int = 1, with_audio: bool = True
) -> dict:
    example = {
        "dataset": "daic",
        "subject_id": "300",
        "sample_id": "300__000",
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "internal_label_text": "Depressed" if label else "Non-depressed",
        "transcript": "full participant transcript",
        "input_modality": modality,
        "partition": "outer_train",
        "fold": 0,
        "prompt_system_text": (
            "You are a psychologist analyzing speech and transcript information "
            "for depression screening."
        ),
        "prompt_user_text": (
            "Based on the audio and transcript, determine whether the subject is "
            "Depressed or Non-depressed."
        ),
    }
    processor = FakeGemmaProcessor()
    user_content = (
        [{"type": "audio"}, {"type": "text", "text": example["prompt_user_text"]}]
        if with_audio
        else example["prompt_user_text"]
    )
    example["prompt_text"] = processor.apply_chat_template(
        [
            {"role": "system", "content": example["prompt_system_text"]},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    if with_audio:
        example["audio_paths"] = ["/tmp/fake.wav"]
        example["audio_arrays"] = [np.zeros(640, dtype=np.float32)]
    else:
        example["audio_paths"] = []
        example["audio_arrays"] = []
    return example


class TestGemma4PromptOnlyCollator:
    def test_audio_batch_retains_exact_multimodal_keys(self):
        collator = Gemma4PromptOnlyExtractionCollator(FakeGemmaProcessor())
        model_inputs, metadata = collator([make_example(with_audio=True)])
        self_assert = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "input_features",
            "input_features_mask",
        }
        assert set(model_inputs) == self_assert
        assert model_inputs["input_ids"].ndim == 2
        assert model_inputs["mm_token_type_ids"].shape == model_inputs["input_ids"].shape
        assert model_inputs["input_features"].dtype == torch.float32
        assert model_inputs["input_features_mask"].dtype == torch.bool
        assert metadata[0]["sample_id"] == "300__000"
        assert metadata[0]["prompt_text"]

    def test_text_only_omits_audio_feature_tensors(self):
        collator = Gemma4PromptOnlyExtractionCollator(FakeGemmaProcessor())
        model_inputs, _ = collator([make_example(with_audio=False)])
        assert {"input_features", "input_features_mask"}.isdisjoint(model_inputs)
        assert {"input_ids", "attention_mask", "mm_token_type_ids"}.issubset(model_inputs)

    def test_gold_labels_subject_and_sample_ids_never_reach_model_inputs(self):
        collator = Gemma4PromptOnlyExtractionCollator(FakeGemmaProcessor())
        example = make_example()
        example["sample_id"] = "leak-sample"
        example["subject_id"] = "leak-subject"
        example["label"] = 1
        example["training_text"] = example["prompt_text"] + "Depressed<turn|>\n"
        model_inputs, _ = collator([example])
        decoded = "".join(
            chr(int(token)) for token in model_inputs["input_ids"][0].tolist()
        )
        assert "labels" not in model_inputs
        assert "leak-sample" not in decoded
        assert "leak-subject" not in decoded
        assert "Depressed<turn|>" not in decoded

    def test_prompt_uses_gemma_template_with_no_answer_text(self):
        processor = FakeGemmaProcessor()
        collator = Gemma4PromptOnlyExtractionCollator(processor)
        example = make_example(label=1)
        example["training_text"] = example["prompt_text"] + "Depressed<turn|>\n"
        model_inputs, _ = collator([example])
        decoded = "".join(
            chr(int(token)) for token in model_inputs["input_ids"][0].tolist()
        )
        assert "<|turn>system" in decoded
        assert "<|turn>user" in decoded
        assert "<|turn>model" in decoded
        assert "Depressed<turn|>" not in decoded


class TestBackendHiddenSizes:
    def test_gemma4_accepts_3840_only(self):
        gemma_model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=3840)),
            base_model=None,
        )
        config = {"model_backend": "gemma4"}
        assert _decoder_hidden_size(gemma_model, config) == 3840
        bad = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096)),
            base_model=None,
        )
        with pytest.raises(ValueError, match="3840"):
            _decoder_hidden_size(bad, config)

    def test_qwen_sizes_preserved(self):
        text_model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=3584), base_model=None
        )
        audio_model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096)),
            base_model=None,
        )
        assert _decoder_hidden_size(text_model, {"model_backend": "text"}) == 3584
        assert _decoder_hidden_size(audio_model, {"model_backend": "qwen2audio"}) == 4096
        with pytest.raises(ValueError, match="expected one of"):
            _decoder_hidden_size(audio_model, {"model_backend": "text"})
        with pytest.raises(ValueError, match="expected one of"):
            _decoder_hidden_size(text_model, {"model_backend": "qwen2audio"})

    def test_unset_backend_keeps_legacy_qwen_union(self):
        model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096)),
            base_model=None,
        )
        assert _decoder_hidden_size(model, {}) == 4096

    def test_backend_size_tables_are_exact(self):
        assert BACKEND_HIDDEN_SIZES == {"gemma4": {3840}, "qwen2audio": {4096}, "text": {3584}}
        assert QWEN_HIDDEN_SIZES == {3584, 4096}


class TestCacheSchemaAndIdentity:
    def test_gemma_cache_schema_is_distinct(self):
        assert _backend_cache_schema({"model_backend": "gemma4"}) == CACHE_SCHEMA_VERSION_GEMMA4
        assert CACHE_SCHEMA_VERSION_GEMMA4 == "gemma4_hidden_cache.v1"
        assert _backend_cache_schema({}) == CACHE_SCHEMA_VERSION_QWEN
        assert _backend_cache_schema({"model_backend": "qwen2audio"}) == CACHE_SCHEMA_VERSION_QWEN
        assert CACHE_SCHEMA_VERSION_QWEN == "qwen_hidden_cache.v2"

    def test_parent_attempt_id_reads_modern_metadata(self, tmp_path: Path):
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir(parents=True)
        (fold_dir / "best_model").mkdir()
        assert _parent_attempt_id(fold_dir / "best_model") is None
        save_json({"attempt_id": "20260812T031624Z-test-abc12345-00000000"}, fold_dir / "metadata.json")
        assert (
            _parent_attempt_id(fold_dir / "best_model")
            == "20260812T031624Z-test-abc12345-00000000"
        )


class TestSubjectSelection:
    def test_requires_both_partitions_and_hashes_file(self, tmp_path: Path):
        selection_path = tmp_path / "smoke_selection.json"
        save_json(
            {"outer_train": ["303", "304"], "final_eval": ["310", "311"]},
            selection_path,
        )
        selection = load_subject_selection(selection_path)
        assert selection["outer_train"] == ["303", "304"]
        assert selection["final_eval"] == ["310", "311"]
        assert len(selection["sha256"]) == 64

    def test_refuses_missing_partition_or_duplicates(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        save_json({"outer_train": ["303"]}, bad)
        with pytest.raises(ValueError, match="missing keys"):
            load_subject_selection(bad)
        dup = tmp_path / "dup.json"
        save_json({"outer_train": ["303", "303"], "final_eval": ["310"]}, dup)
        with pytest.raises(ValueError, match="duplicates"):
            load_subject_selection(dup)

    def test_none_returns_none(self):
        assert load_subject_selection(None) is None

    def test_restricts_saved_split_and_refuses_unknown_subjects(self):
        saved = {
            "outer_train": ["a", "b", "c"],
            "final_eval": ["x", "y"],
        }
        selection = {"outer_train": ["b", "c"], "final_eval": ["y"], "sha256": "0" * 64}
        assert _apply_subject_selection(saved, selection) == {
            "outer_train": ["b", "c"],
            "final_eval": ["y"],
        }
        assert _apply_subject_selection(saved, None) == saved
        bad = {"outer_train": ["zzz"], "final_eval": ["y"], "sha256": "0" * 64}
        with pytest.raises(ValueError, match="not in the saved split"):
            _apply_subject_selection(saved, bad)
