"""Processor-input zero padding for audio below the verified minimum.

Provenance (2026-10-08, isolated CPU probes on the MN5 login node, ``qwen3omni``
environment, transformers 5.3.0, real Qwen3-Omni Thinker checkpoint):

- The Qwen3-Omni processor (``Qwen3OmniMoeProcessor`` + ``WhisperFeatureExtractor``,
  n_fft=400, hop=160) raises ``RuntimeError: Padding size should be less than the
  corresponding input dimension`` for waveforms of 200 samples or fewer. A
  201-sample waveform produces one mel frame and passes the processor.
- The Thinker audio encoder accepts that one-frame input: a random-weight forward
  called exactly as ``get_audio_features`` calls it (``input_features`` of shape
  ``(mel, frames)``, ``feature_lens=[1]``) returns one hidden state.

The window15 DAIC packing preserves the canonical fixed-15 packing and its
short-tail convention, so a participant's final chunk can be shorter than the
processor minimum (observed 160 samples). The 15s recipes opt in through
``data.processor_min_audio_samples: 201``: every waveform below the minimum is
zero-padded numerically at the processor input only. Manifest frame counts,
boundaries, chunk membership, subject/unit weights and coverage stay unchanged,
and the training-side audit records the real and padded lengths. Baseline 30s
behavior is unchanged because the key is absent by default.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

MIN_PROCESSOR_SAMPLES = 201


def resolve_min_audio_samples(config: dict[str, Any] | None) -> int:
    """Return the opt-in processor-input minimum; 0 (default) disables padding."""
    if not config:
        return 0
    data = config.get("data") or {}
    value = data.get("processor_min_audio_samples", 0)
    if value is None or value == "":
        return 0
    try:
        minimum = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"data.processor_min_audio_samples must be an integer, got {value!r}"
        ) from exc
    if minimum < 0:
        raise ValueError("data.processor_min_audio_samples must be >= 0")
    if 0 < minimum < MIN_PROCESSOR_SAMPLES:
        raise ValueError(
            f"data.processor_min_audio_samples={minimum} is below the verified "
            f"processor minimum of {MIN_PROCESSOR_SAMPLES} samples"
        )
    return minimum


def pad_audio_array(audio: np.ndarray, minimum: int) -> tuple[np.ndarray, int]:
    """Zero-pad one 1-D waveform to ``minimum`` samples.

    Returns ``(audio, padded_samples)``; ``padded_samples`` is 0 when the
    waveform already has at least ``minimum`` samples.
    """
    if minimum <= 0:
        return audio, 0
    array = np.asarray(audio)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D waveform, got shape {array.shape}")
    missing = int(minimum) - int(array.shape[0])
    if missing <= 0:
        return audio, 0
    return np.pad(array, (0, missing), mode="constant", constant_values=0.0), missing


def pad_audio_arrays(
    audio_arrays: Iterable[np.ndarray], minimum: int
) -> tuple[list[np.ndarray], int]:
    """Pad every waveform in a list; returns ``(arrays, total_padded_samples)``."""
    arrays = list(audio_arrays)
    if minimum <= 0:
        return arrays, 0
    padded_total = 0
    padded_arrays: list[np.ndarray] = []
    for array in arrays:
        padded, missing = pad_audio_array(array, minimum)
        padded_arrays.append(padded)
        padded_total += missing
    return padded_arrays, padded_total
