from __future__ import annotations

from typing import Any

import numpy as np
import torch


MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "input_features",
    "feature_attention_mask",
}


class PromptOnlyExtractionCollator:
    """Process prompt-only examples while keeping labels as external metadata."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        if len(batch) != 1:
            raise ValueError("Primary hidden extraction requires batch size 1.")
        example = batch[0]
        audio = example.get("audio_arrays") or None
        kwargs: dict[str, Any] = {
            "text": example["prompt_text"],
            "return_tensors": "pt",
            "padding": False,
        }
        if audio is not None:
            feature_extractor = getattr(self.processor, "feature_extractor", None)
            if feature_extractor is None:
                raise ValueError("Audio input requires a processor feature extractor.")
            kwargs["audio"] = audio
            kwargs["sampling_rate"] = int(feature_extractor.sampling_rate)
        processed = self.processor(**kwargs)
        model_inputs = {key: value for key, value in processed.items() if key in MODEL_INPUT_KEYS}
        forbidden = {"labels", "sample_id", "subject_id"}.intersection(model_inputs)
        if forbidden:
            raise AssertionError(f"Forbidden keys leaked into model inputs: {sorted(forbidden)}")
        metadata = [{
            "dataset": str(example["dataset"]),
            "sample_id": str(example["sample_id"]),
            "subject_id": str(example["subject_id"]),
            "label": int(example["label"]),
            "partition": str(example["partition"]),
            "fold": int(example["fold"]),
            "prompt_text": str(example["prompt_text"]),
        }]
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


def load_prompt_audio(example: dict[str, Any], sampling_rate: int | None, silence_audio: bool) -> dict[str, Any]:
    """Attach the deterministic evaluation audio arrays expected by the collator."""
    from src.data.runtime import (
        load_audio_array,
        load_audio_spans_array,
        load_span_group_audio_arrays,
        uses_audio_spans,
    )

    output = dict(example)
    if example.get("audio_span_groups"):
        output["audio_arrays"] = load_span_group_audio_arrays(
            output, sampling_rate, silence_audio
        )
        return output
    if uses_audio_spans(example):
        if sampling_rate is None:
            raise ValueError("Audio examples require a processor sampling rate.")
        output["audio_arrays"] = [
            load_audio_spans_array(
                output["audio_path"],
                output["audio_spans"],
                sampling_rate,
                silence_audio,
                output.get("participant_sample_count"),
            )
        ]
        return output
    paths = output.get("audio_paths") or []
    if paths:
        if sampling_rate is None:
            raise ValueError("Audio examples require a processor sampling rate.")
        start_times = list(output.get("audio_start_times") or [None] * len(paths))
        end_times = list(output.get("audio_end_times") or [None] * len(paths))
        output["audio_arrays"] = [
            load_audio_array(path, sampling_rate, max_seconds, silence_audio, start_time, end_time)
            for path, max_seconds, start_time, end_time in zip(
                paths, output["audio_clip_seconds"], start_times, end_times
            )
        ]
    else:
        output["audio_arrays"] = []
    return output
