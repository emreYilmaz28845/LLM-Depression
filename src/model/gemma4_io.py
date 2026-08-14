from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils import (
    INPUT_MODALITY_AUDIO_ONLY,
    INPUT_MODALITY_AUDIO_TEXT,
    INPUT_MODALITY_TEXT_ONLY,
    MODEL_BACKEND_GEMMA4,
    get_logger,
    resolve_input_modality,
    resolve_model_backend,
)

LOGGER = get_logger(__name__)

GEMMA4_AUDIO_SAMPLING_RATE = 16000
GEMMA4_AUDIO_SAMPLES_PER_TOKEN = 640
GEMMA4_MAX_AUDIO_SAMPLES = 480000
GEMMA4_MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA4_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
GEMMA4_EXPECTED_LORA_MODULES = 288
GEMMA4_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
GEMMA4_TURN_TERMINATOR = "<turn|>\n"
GEMMA4_SUPPORTED_DATASETS = {"daic", "d3tec", "turkish", "androids_interview", "cmdc"}
GEMMA4_DAIC_SAMPLE_MODE = "participant_speech_packed30"
GEMMA4_HARMONIZED_SAMPLE_MODE = "harmonized_response_windows"

AUDIO_TOLERANCE = 1e-6


def validate_gemma4_audio(
    waveform: Any, sampling_rate: int, require_unit_range: bool = True
) -> np.ndarray:
    """Validate one waveform before the Gemma processor touches it.

    Raises ``ValueError`` with the exact violated invariant. Never truncates.

    The ``[-1, 1]`` range check is part of the DAIC packed30 contract only.
    The harmonized non-DAIC contract (runbook Section 9.1) requires finite
    mono float32 16 kHz audio of at most 480,000 samples; real-world CMDC
    sources can exceed the unit range by a few percent, so callers with
    ``require_unit_range=False`` skip only that check.
    """
    if not isinstance(waveform, np.ndarray):
        raise ValueError(
            f"Gemma audio must be a NumPy array, got {type(waveform).__name__}."
        )
    if waveform.ndim != 1:
        raise ValueError(
            f"Gemma audio must be one-dimensional (mono), got ndim={waveform.ndim}."
        )
    if waveform.dtype != np.float32:
        raise ValueError(
            f"Gemma audio must be float32, got {waveform.dtype}."
        )
    if int(sampling_rate) != GEMMA4_AUDIO_SAMPLING_RATE:
        raise ValueError(
            f"Gemma audio requires {GEMMA4_AUDIO_SAMPLING_RATE} Hz, got {int(sampling_rate)}."
        )
    if not np.isfinite(waveform).all():
        raise ValueError("Gemma audio contains non-finite samples (NaN/Inf).")
    if waveform.size < 1:
        raise ValueError("Gemma audio must contain at least one sample.")
    if waveform.size > GEMMA4_MAX_AUDIO_SAMPLES:
        raise ValueError(
            f"Gemma audio exceeds {GEMMA4_MAX_AUDIO_SAMPLES} samples "
            f"({waveform.size}); the repository must not feed the backend more "
            "than one packed30 window. Do not truncate here."
        )
    if float(waveform.min()) < -1.0 - AUDIO_TOLERANCE or float(waveform.max()) > 1.0 + AUDIO_TOLERANCE:
        if require_unit_range:
            raise ValueError(
                f"Gemma audio must lie within [-1, 1] (small tolerance allowed), "
                f"got min={float(waveform.min()):.6f} max={float(waveform.max()):.6f}."
            )
    return waveform


def expected_gemma_audio_tokens(num_samples: int) -> int:
    return int(math.ceil(int(num_samples) / GEMMA4_AUDIO_SAMPLES_PER_TOKEN))


def expected_gemma4_audio_tokens(num_samples: int) -> int:
    return int(math.ceil(int(num_samples) / GEMMA4_AUDIO_SAMPLES_PER_TOKEN))


def _gemma4_messages(
    system_text: str,
    user_text: str,
    modality: str,
) -> list[dict[str, Any]]:
    if modality == INPUT_MODALITY_TEXT_ONLY:
        user_content: Any = user_text
    else:
        # The audio content item carries no value: the chat template only needs
        # the type marker, and the collator passes the already-loaded waveform
        # arrays to the processor. Never put file paths in chat messages.
        user_content = [
            {"type": "audio"},
            {"type": "text", "text": user_text},
        ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def render_gemma4_prompt(
    processor,
    system_text: str,
    user_text: str,
    modality: str,
) -> str:
    messages = _gemma4_messages(system_text, user_text, modality)
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def prepare_gemma4_example(
    example: dict[str, Any],
    config: dict[str, Any],
    processor,
) -> dict[str, Any]:
    """Render the Gemma prompt and training text for one example.

    Requires ``prompt_system_text`` and ``prompt_user_text`` on the example.
    Returns a shallow copy updated only in the backend-rendered prompt fields;
    subject IDs, labels, transcripts, weights, and audio plans are untouched.
    """
    system_text = example.get("prompt_system_text")
    user_text = example.get("prompt_user_text")
    if not isinstance(system_text, str) or not system_text.strip():
        raise ValueError(
            f"Gemma example {example.get('sample_id', '')} is missing "
            "prompt_system_text."
        )
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError(
            f"Gemma example {example.get('sample_id', '')} is missing "
            "prompt_user_text."
        )
    modality = resolve_input_modality(config)
    prompt_text = render_gemma4_prompt(processor, system_text, user_text, modality)
    label_text = example["internal_label_text"]
    prepared = dict(example)
    prepared["prompt_text"] = prompt_text
    prepared["training_text"] = (
        f"{prompt_text}{label_text}{GEMMA4_TURN_TERMINATOR}"
    )
    return prepared


def prepare_gemma4_examples(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    processor,
) -> list[dict[str, Any]]:
    return [
        prepare_gemma4_example(example, config, processor)
        if _is_gemma_backend(config)
        else example
        for example in examples
    ]


def _is_gemma_backend(config: dict[str, Any]) -> bool:
    return resolve_model_backend(config) == MODEL_BACKEND_GEMMA4


class Gemma4SFTCollator:
    """Gemma 4 unified SFT collator with the exact prompt-prefix label mask.

    - Right padding during training (manual, tokenizer-agnostic).
    - Exactly one waveform per audio example; text-only has no audio keys.
    - Retains ``mm_token_type_ids`` and boolean ``input_features_mask``.
    - Labels: prompt positions and pad positions are ``-100``; the loss covers
      the label span and the official turn terminator only.
    - ``loss_weight`` is retained per example for the existing weighted loss.
    """

    def __init__(self, processor, debug: bool = False, require_unit_range: bool = True):
        self.processor = processor
        self.debug = debug
        self.require_unit_range = require_unit_range
        self.last_debug_example: dict[str, Any] | None = None

    def _processor_kwargs(
        self, text: str, audio: list[np.ndarray] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": text,
            "return_tensors": None,
            "padding": False,
            "return_mm_token_type_ids": True,
        }
        if audio is not None:
            kwargs["audio"] = audio
            kwargs["sampling_rate"] = int(self.processor.feature_extractor.sampling_rate)
        return kwargs

    def _process_single(self, example: dict[str, Any]) -> dict[str, Any]:
        audio_arrays = example.get("audio_arrays") or []
        modality_has_audio = bool(example.get("audio_paths") or example.get("audio_spans") or example.get("audio_span_groups") or example.get("audio_path"))
        if modality_has_audio:
            if len(audio_arrays) != 1:
                raise ValueError(
                    f"Gemma collator requires exactly one waveform per audio example; "
                    f"sample_id={example.get('sample_id', '')} has {len(audio_arrays)}."
                )
            for array in audio_arrays:
                validate_gemma4_audio(
                    array, GEMMA4_AUDIO_SAMPLING_RATE,
                    require_unit_range=self.require_unit_range,
                )
            audio: list[np.ndarray] | None = audio_arrays
        else:
            audio = None
        processor_kwargs = self._processor_kwargs(example["training_text"], audio)
        prompt_kwargs = self._processor_kwargs(example["prompt_text"], audio)
        processed_full = self.processor(**processor_kwargs)
        processed_prompt = self.processor(**prompt_kwargs)
        input_ids = np.asarray(processed_full["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(processed_full["attention_mask"], dtype=np.int64)
        prompt_ids = np.asarray(processed_prompt["input_ids"], dtype=np.int64)
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[0]
        if prompt_ids.ndim == 2:
            prompt_ids = prompt_ids[0]
        prompt_len = int(len(prompt_ids))
        if not np.array_equal(prompt_ids, input_ids[:prompt_len]):
            raise ValueError(
                f"Gemma prompt token IDs are not an exact prefix of training "
                f"token IDs for sample_id={example.get('sample_id', '')}. "
                "Fix prompt construction; never guess mask indices."
            )
        labels = input_ids.copy()
        labels[:prompt_len] = -100
        output: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": np.asarray(
                processed_full["mm_token_type_ids"], dtype=np.int64
            ),
            "prompt_len": prompt_len,
            "sample_id": example["sample_id"],
            "subject_id": example["subject_id"],
            "label": int(example["label"]),
        }
        if audio is not None:
            output["input_features"] = np.asarray(
                processed_full["input_features"], dtype=np.float32
            )
            output["input_features_mask"] = np.asarray(
                processed_full["input_features_mask"], dtype=bool
            )
        if self.debug and self.last_debug_example is None:
            self.last_debug_example = {
                "sample_id": example["sample_id"],
                "decoded_training_text": example["training_text"],
                "decoded_prompt_text": example["prompt_text"],
                "input_ids": input_ids.tolist(),
                "labels": labels.tolist(),
                "unmasked_token_ids": [
                    int(token_id) for token_id in labels[prompt_len:] if token_id != -100
                ],
            }
        return output

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed_items = [self._process_single(example) for example in batch]
        max_seq_len = max(len(item["input_ids"]) for item in processed_items)
        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        batch_input_ids: list[np.ndarray] = []
        batch_attention_mask: list[np.ndarray] = []
        batch_labels: list[np.ndarray] = []
        batch_mm_token_type_ids: list[np.ndarray] = []
        for item in processed_items:
            pad_len = max_seq_len - len(item["input_ids"])
            batch_input_ids.append(np.pad(item["input_ids"], (0, pad_len), constant_values=pad_token_id))
            batch_attention_mask.append(np.pad(item["attention_mask"], (0, pad_len), constant_values=0))
            batch_labels.append(np.pad(item["labels"], (0, pad_len), constant_values=-100))
            mm_ids = np.asarray(item["mm_token_type_ids"], dtype=np.int64)
            if mm_ids.ndim == 2:
                mm_ids = mm_ids[0]
            batch_mm_token_type_ids.append(np.pad(mm_ids, (0, pad_len), constant_values=0))

        batch_dict: dict[str, Any] = {
            "input_ids": torch.tensor(np.stack(batch_input_ids), dtype=torch.long),
            "attention_mask": torch.tensor(np.stack(batch_attention_mask), dtype=torch.long),
            "labels": torch.tensor(np.stack(batch_labels), dtype=torch.long),
            "mm_token_type_ids": torch.tensor(
                np.stack(batch_mm_token_type_ids), dtype=torch.long
            ),
        }
        if any("loss_weight" in example for example in batch):
            batch_dict["loss_weight"] = torch.tensor(
                [float(example.get("loss_weight", 1.0)) for example in batch],
                dtype=torch.float32,
            )

        if "input_features" in processed_items[0]:
            all_features = [
                np.asarray(item["input_features"], dtype=np.float32)[0]
                for item in processed_items
            ]
            all_masks = [
                np.asarray(item["input_features_mask"], dtype=bool)[0]
                for item in processed_items
            ]
            max_frames = max(int(mask.shape[0]) for mask in all_masks)
            padded_features: list[np.ndarray] = []
            padded_masks: list[np.ndarray] = []
            for features, mask in zip(all_features, all_masks):
                pad_frames = max_frames - int(mask.shape[0])
                padded_features.append(
                    np.pad(features, ((0, pad_frames), (0, 0)), constant_values=0.0)
                )
                padded_masks.append(np.pad(mask, (0, pad_frames), constant_values=False))
            batch_dict["input_features"] = torch.tensor(
                np.stack(padded_features), dtype=torch.float32
            )
            batch_dict["input_features_mask"] = torch.tensor(
                np.stack(padded_masks), dtype=torch.bool
            )
        return batch_dict


def validate_gemma4_config(config: dict[str, Any]) -> None:
    """Fail unless every Gemma-specific config invariant from runbook §9.6 holds.

    DAIC keeps its packed30 contract exactly. D3TEC, Turkish, Androids, and
    CMDC use the harmonized response-window recipe. Everything else (freeze,
    BF16, gradient checkpointing, macro-F1 selection, teacher-forced
    evaluation, pinned revision, exact LoRA regex) is shared across datasets.
    """
    if resolve_model_backend(config) != MODEL_BACKEND_GEMMA4:
        return
    errors: list[str] = []

    def _require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    dataset = str(config.get("dataset", "")).lower()
    _require(
        dataset in GEMMA4_SUPPORTED_DATASETS,
        "dataset must be one of "
        + ", ".join(sorted(GEMMA4_SUPPORTED_DATASETS)),
    )
    modality = resolve_input_modality(config)
    _require(
        modality in {INPUT_MODALITY_TEXT_ONLY, INPUT_MODALITY_AUDIO_ONLY, INPUT_MODALITY_AUDIO_TEXT},
        "modality must be text-only, audio-only, or audio+text",
    )
    data_cfg = config.get("data", {})
    if dataset == "daic":
        _require(
            str(data_cfg.get("sample_mode", "")).lower() == GEMMA4_DAIC_SAMPLE_MODE,
            f"data.sample_mode must be {GEMMA4_DAIC_SAMPLE_MODE} for daic",
        )
        if modality != INPUT_MODALITY_TEXT_ONLY:
            _require(
                int(data_cfg.get("participant_chunk_samples", 0) or 0) == 480000,
                "data.participant_chunk_samples must be 480000 for audio modes",
            )
        if modality == INPUT_MODALITY_AUDIO_TEXT:
            _require(
                str(data_cfg.get("audio_text_transcript_scope", "")).lower() == "full_participant",
                "data.audio_text_transcript_scope must be full_participant",
            )
    else:
        _require(
            str(data_cfg.get("sample_mode", "")).lower() == GEMMA4_HARMONIZED_SAMPLE_MODE,
            f"data.sample_mode must be {GEMMA4_HARMONIZED_SAMPLE_MODE} for "
            f"dataset={dataset}",
        )
        segment_seconds = float(data_cfg.get("segment_seconds", 0.0) or 0.0)
        _require(
            0.0 < segment_seconds <= 30.0,
            "data.segment_seconds must be in (0, 30] for harmonized response windows",
        )
        if dataset == "androids_interview" and modality == INPUT_MODALITY_AUDIO_TEXT:
            _require(
                str(data_cfg.get("audio_text_transcript_scope", "")).lower()
                == "full_subject",
                "data.audio_text_transcript_scope must be full_subject for "
                "androids_interview audio+text",
            )
    audio_adapter_cfg = config.get("audio_adapter") or {}
    _require(
        not bool(audio_adapter_cfg.get("enabled", False)),
        "audio_adapter.enabled must be false",
    )
    _require(
        not bool(audio_adapter_cfg.get("train_projector", False)),
        "audio_adapter.train_projector must be false",
    )
    _require(
        not bool((config.get("lora") or {}).get("tune_audio_encoder", False)),
        "lora.tune_audio_encoder must be false (audio encoder tuning disabled)",
    )
    training_cfg = config.get("training", {})
    _require(bool(training_cfg.get("bf16", False)), "training.bf16 must be true")
    _require(
        bool(training_cfg.get("gradient_checkpointing", False)),
        "training.gradient_checkpointing must be true",
    )
    _require(
        str(training_cfg.get("selection_metric", "")) == "inner_val_macro_f1",
        "training.selection_metric must be inner_val_macro_f1",
    )
    _require(
        str(training_cfg.get("selection_metric_mode", "")).lower() == "max",
        "training.selection_metric_mode must be max",
    )
    evaluation_cfg = config.get("evaluation", {})
    _require(
        str(evaluation_cfg.get("sample_prediction_mode", "")) == "original_teacher_forced",
        "evaluation.sample_prediction_mode must be original_teacher_forced",
    )
    _require(
        str(evaluation_cfg.get("headline_mode", "")) == "original_teacher_forced",
        "evaluation.headline_mode must be original_teacher_forced",
    )
    _require(
        evaluation_cfg.get("evaluation_view") == GEMMA4_EVALUATION_VIEW,
        f"evaluation.evaluation_view must be {GEMMA4_EVALUATION_VIEW}",
    )
    lora_cfg = config.get("lora", {})
    _require(
        lora_cfg.get("target_modules") == GEMMA4_LORA_TARGET_REGEX,
        "lora.target_modules must be the exact Gemma 4 decoder regex",
    )
    revision = config.get("model_revision")
    _require(
        isinstance(revision, str) and revision == GEMMA4_MODEL_REVISION,
        f"model_revision must be the pinned revision {GEMMA4_MODEL_REVISION}",
    )
    batching = str(evaluation_cfg.get("candidate_batching", "sequential")).strip().lower()
    _require(
        batching in {"", "sequential"},
        "candidate paired batching is not allowed in Gemma configs",
    )
    if errors:
        raise ValueError("Invalid gemma4 config:\n- " + "\n- ".join(errors))
