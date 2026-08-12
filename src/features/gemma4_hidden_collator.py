from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.model.gemma4_io import validate_gemma4_audio

# The Gemma 4 unified model consumes these keys from the processor. Audio keys
# are present only when the modality actually carries audio; text-only batches
# omit them cleanly.
GEMMA4_MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "input_features",
    "input_features_mask",
}


class Gemma4PromptOnlyExtractionCollator:
    """Process Gemma prompt-only examples while keeping labels as external
    metadata.

    Mirrors the locked Qwen collator contract for the Gemma backend: batch
    size one, prompt text only, no gold label, no generation, and none of the
    external metadata fields ever reaches the model inputs.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        if len(batch) != 1:
            raise ValueError("Gemma hidden extraction requires batch size 1.")
        example = batch[0]
        audio = example.get("audio_arrays") or None
        if audio is not None:
            if len(audio) != 1:
                raise ValueError(
                    f"Gemma extraction requires exactly one waveform per audio "
                    f"example; sample_id={example.get('sample_id', '')} has {len(audio)}."
                )
            sampling_rate = int(self.processor.feature_extractor.sampling_rate)
            validate_gemma4_audio(audio[0], sampling_rate)
        kwargs: dict[str, Any] = {
            "text": example["prompt_text"],
            "return_tensors": "pt",
            "padding": False,
            "return_mm_token_type_ids": True,
        }
        if audio is not None:
            kwargs["audio"] = audio
            kwargs["sampling_rate"] = sampling_rate
        processed = self.processor(**kwargs)
        model_inputs = {
            key: value
            for key, value in processed.items()
            if key in GEMMA4_MODEL_INPUT_KEYS
        }
        forbidden = {"labels", "sample_id", "subject_id"}.intersection(model_inputs)
        if forbidden:
            raise AssertionError(
                f"Forbidden keys leaked into Gemma model inputs: {sorted(forbidden)}"
            )
        for name, value in model_inputs.items():
            if isinstance(value, np.ndarray):
                model_inputs[name] = torch.as_tensor(value)
            elif not isinstance(value, torch.Tensor):
                raise AssertionError(
                    f"Gemma processor returned non-tensor key {name!r}: {type(value).__name__}."
                )
        metadata = [
            {
                "dataset": str(example["dataset"]),
                "sample_id": str(example["sample_id"]),
                "subject_id": str(example["subject_id"]),
                "label": int(example["label"]),
                "partition": str(example["partition"]),
                "fold": int(example["fold"]),
                "prompt_text": str(example["prompt_text"]),
            }
        ]
        # These fields remain external metadata. In particular, prompt_id is
        # never rendered into the prompt by this collator.
        for key in (
            "response_id",
            "turn_key",
            "question_id",
            "window_id",
            "prompt_id",
            "segment_index",
            "num_segments",
            "chunk_id",
            "bundle_id",
            "bundle_chunk_ids",
            "bundle_coverage_count",
            "effective_k",
            "chunk_schedule_epoch",
            "protocol_id",
            "chunk_index",
            "num_chunks",
        ):
            if key in example:
                metadata[0][key] = example[key]
        return model_inputs, metadata
