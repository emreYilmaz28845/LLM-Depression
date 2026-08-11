"""Real Gemma4UnifiedProcessor contract tests (require transformers 5.14.1 + snapshot).

These tests exercise the pinned processor exactly as the runbook table
specifies. They skip cleanly when the pinned Transformers classes or the local
snapshot are unavailable; a skip is not the MN5 acceptance gate.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers", minversion="5.14.0")
Gemma4UnifiedProcessor = getattr(transformers, "Gemma4UnifiedProcessor", None)
if Gemma4UnifiedProcessor is None:
    pytest.skip("requires transformers 5.14.1 with Gemma4UnifiedProcessor", allow_module_level=True)

from src.model.gemma4_io import (  # noqa: E402
    GEMMA4_TURN_TERMINATOR,
    Gemma4SFTCollator,
)

SNAPSHOT_CANDIDATES = [
    Path(os.environ.get("GEMMA4_SNAPSHOT_DIR", "")),
    Path("/media/emre/Backup/AudioLLM/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    Path("/tmp/opencode/gemma4_snapshot_tf"),
]


def _find_snapshot() -> Path | None:
    for candidate in SNAPSHOT_CANDIDATES:
        if candidate.is_dir() and (candidate / "tokenizer.json").is_file():
            return candidate
    return None


SNAPSHOT = _find_snapshot()

pytestmark = pytest.mark.skipif(
    SNAPSHOT is None, reason="pinned Gemma snapshot not available locally"
)


@pytest.fixture(scope="module")
def processor():
    return Gemma4UnifiedProcessor.from_pretrained(SNAPSHOT, local_files_only=True)


def _waveform(samples: int) -> np.ndarray:
    rng = np.random.default_rng(1337)
    return ((rng.random(samples).astype(np.float32) - 0.5) * 0.1).astype(np.float32)


def _prompt_and_training(processor, modality: str = "audio_text"):
    system = "You are a psychologist analyzing speech audio for depression screening."
    user = "Based on the audio, determine whether the subject is Depressed or Non-depressed."
    if modality == "text_only":
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "audio"},
                {"type": "text", "text": user},
            ]},
        ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return prompt, prompt + "Depressed" + GEMMA4_TURN_TERMINATOR


def _process(processor, text: str, wav: np.ndarray | None):
    kwargs = {"text": text, "padding": False, "return_tensors": None}
    if wav is not None:
        kwargs["audio"] = [wav]
        kwargs["sampling_rate"] = 16000
    return processor(**kwargs)


@pytest.mark.parametrize(
    "samples, expected_frames",
    [(640, 1), (641, 2), (480000, 750)],
)
def test_feature_shape_table(processor, samples: int, expected_frames: int) -> None:
    prompt, training = _prompt_and_training(processor)
    wav = _waveform(samples)
    full = _process(processor, training, wav)
    features = np.asarray(full["input_features"], dtype=np.float32)
    mask = np.asarray(full["input_features_mask"], dtype=bool)
    assert features.shape == (1, expected_frames, 640)
    assert int(mask.sum()) == expected_frames
    assert features.dtype == np.float32
    assert mask.dtype == np.bool_
    assert np.asarray(full["mm_token_type_ids"]).shape == np.asarray(full["input_ids"]).shape


def test_batch_shape_table(processor) -> None:
    prompt, training = _prompt_and_training(processor)
    wav1 = _waveform(640)
    wav2 = _waveform(1280)
    batch = processor(
        text=[training, training],
        audio=[wav1, wav2],
        sampling_rate=16000,
        padding=True,
        return_tensors=None,
    )
    features = np.asarray(batch["input_features"], dtype=np.float32)
    masks = np.asarray(batch["input_features_mask"], dtype=bool)
    assert features.shape == (2, 2, 640)
    assert masks[0].sum() == 1
    assert masks[1].sum() == 2


def test_audio_placeholder_count_matches_frames(processor) -> None:
    prompt, training = _prompt_and_training(processor)
    for samples in (640, 641, 480000):
        wav = _waveform(samples)
        full = _process(processor, training, wav)
        ids = np.asarray(full["input_ids"]).reshape(-1)
        expected = math.ceil(samples / 640)
        assert int((ids == processor.audio_token_id).sum()) == expected


def test_text_only_returns_no_audio_tensors(processor) -> None:
    prompt, training = _prompt_and_training(processor, modality="text_only")
    full = _process(processor, training, None)
    assert "input_features" not in full
    assert "input_features_mask" not in full


def test_prompt_ids_exact_prefix_for_all_modalities(processor) -> None:
    for modality in ("text_only", "audio_only", "audio_text"):
        prompt, training = _prompt_and_training(processor, modality=modality)
        wav = _waveform(480000) if modality != "text_only" else None
        full = _process(processor, training, wav)
        prm = _process(processor, prompt, wav)
        full_ids = np.asarray(full["input_ids"]).reshape(-1)
        prm_ids = np.asarray(prm["input_ids"]).reshape(-1)
        assert np.array_equal(prm_ids, full_ids[: len(prm_ids)]), modality


def test_only_response_span_unmasked(processor) -> None:
    prompt, training = _prompt_and_training(processor, modality="audio_text")
    wav = _waveform(640)
    full = _process(processor, training, wav)
    prm = _process(processor, prompt, wav)
    full_ids = np.asarray(full["input_ids"]).reshape(-1)
    prm_ids = np.asarray(prm["input_ids"]).reshape(-1)
    prompt_len = len(prm_ids)
    labels = full_ids.copy()
    labels[:prompt_len] = -100
    assert int((labels[prompt_len:] == -100).sum()) == 0
    unmasked_text = np.asarray(full["input_ids"]).reshape(-1)[prompt_len:]
    assert (unmasked_text == processor.audio_token_id).sum() == 0


def test_collator_with_real_processor(processor) -> None:
    prompt, training = _prompt_and_training(processor)
    wav = _waveform(480000)
    example = {
        "sample_id": "300__000",
        "subject_id": "300",
        "label": 1,
        "internal_label_text": "Depressed",
        "prompt_text": prompt,
        "training_text": training,
        "audio_paths": ["/tmp/fake.wav"],
        "audio_arrays": [wav],
        "loss_weight": 1.0,
    }
    collator = Gemma4SFTCollator(processor=processor)
    batch = collator([example])
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_features"].shape == (1, 750, 640)
    assert batch["input_features_mask"].dtype == torch_bool()
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    prompt_len = len(np.asarray(_process(processor, prompt, wav)["input_ids"]).reshape(-1))
    assert int((batch["labels"][0, :prompt_len] == -100).sum()) == prompt_len


def torch_bool():
    import torch

    return torch.bool
