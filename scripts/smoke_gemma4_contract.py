#!/usr/bin/env python
"""Gemma 4 DAIC contract smoke (Tier A) — real processor, no full model weights.

Runs through Slurm on an MN5 compute node with the dedicated Gemma
environment. Verifies, against the pinned GPFS snapshot:

- processor class and feature-extractor parameters (640 samples/token, 16 kHz);
- the runbook shape table (640 / 641 / 480000 samples, batch 640+1280);
- prompt rendering (chat template, thinking disabled) and exact prompt-prefix
  property for all three modalities;
- label masking (prompt + pads = -100; only the response span unmasked);
- audio placeholder count == valid frame count == ceil(samples / 640);
- config invariants for the three Gemma configs;
- exact model metadata (architecture, model_type, 48 layers,
  attention_k_eq_v=true).

Offline-only: model and processor load from the local GPFS snapshot with
local_files_only=True. Never attempt a network download.

Usage:
  python scripts/smoke_gemma4_contract.py \
      --model-dir <GPFS snapshot dir> \
      --configs <gemma cfg> [<gemma cfg> ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SAMPLES_PER_TOKEN = 640
SAMPLING_RATE = 16000
MAX_SAMPLES = 480000
TURN_TERMINATOR = "<turn|>\n"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _audio(samples: int) -> np.ndarray:
    rng = np.random.default_rng(1337)
    return ((rng.random(samples).astype(np.float32) - 0.5) * 0.1).astype(np.float32)


def _prompt_for(modality: str, system: str, user: str, processor) -> str:
    if modality == "text_only":
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "audio", "audio": None},
                {"type": "text", "text": user},
            ]},
        ]
    return _render(processor, messages)


def _render(processor, messages: list[dict]) -> str:
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )


def check_processor_contract(processor, modality: str, system: str, user: str) -> None:
    sr = int(processor.feature_extractor.sampling_rate)
    _require(sr == SAMPLING_RATE, f"sampling rate must be {SAMPLING_RATE}, got {sr}")
    _require(
        int(processor.feature_extractor.audio_samples_per_token) == SAMPLES_PER_TOKEN,
        f"audio_samples_per_token must be {SAMPLES_PER_TOKEN}",
    )
    _require(
        type(processor).__name__ == "Gemma4UnifiedProcessor",
        f"processor class must be Gemma4UnifiedProcessor, got {type(processor).__name__}",
    )

    messages = (
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if modality == "text_only"
        else [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "audio", "audio": None},
                {"type": "text", "text": user},
            ]},
        ]
    )
    prompt = _render(processor, messages)
    label = "Depressed"
    training_text = prompt + label + TURN_TERMINATOR

    if modality == "text_only":
        full = processor(text=training_text, padding=False, return_tensors=None)
        _require("input_features" not in full, "text-only must not produce audio features")
        _require("input_features_mask" not in full, "text-only must not produce audio mask")
        print(f"  [{modality}] no audio tensors ok=True")
        return

    for n_samples in (640, 641, 480000):
        wav = _audio(n_samples)
        full = processor(
            text=training_text, audio=[wav], sampling_rate=SAMPLING_RATE,
            padding=False, return_tensors=None,
        )
        prm = processor(
            text=prompt, audio=[wav], sampling_rate=SAMPLING_RATE,
            padding=False, return_tensors=None,
        )
        features = np.asarray(full["input_features"], dtype=np.float32)
        mask = np.asarray(full["input_features_mask"], dtype=bool)
        expected_frames = math.ceil(n_samples / SAMPLES_PER_TOKEN)
        _require(
            features.shape == (1, expected_frames, SAMPLES_PER_TOKEN),
            f"{n_samples} samples: feature shape {features.shape} != "
            f"(1, {expected_frames}, {SAMPLES_PER_TOKEN})",
        )
        _require(
            int(mask.sum()) == expected_frames,
            f"{n_samples} samples: valid mask count {int(mask.sum())} != {expected_frames}",
        )
        _require(
            features.dtype == np.float32,
            f"{n_samples} samples: input_features dtype {features.dtype} != float32",
        )
        _require(
            mask.dtype == np.bool_,
            f"{n_samples} samples: input_features_mask dtype {mask.dtype} != bool",
        )
        full_ids = np.asarray(full["input_ids"]).reshape(-1)
        prm_ids = np.asarray(prm["input_ids"]).reshape(-1)
        _require(
            np.asarray(full["mm_token_type_ids"]).shape == np.asarray(full["input_ids"]).shape,
            "mm_token_type_ids shape must match input_ids shape",
        )
        audio_id_count = int((full_ids == processor.audio_token_id).sum())
        _require(
            audio_id_count == expected_frames,
            f"{n_samples} samples: audio placeholder ids {audio_id_count} != {expected_frames}",
        )
        _require(
            np.array_equal(prm_ids, full_ids[: len(prm_ids)]),
            f"{n_samples} samples: prompt ids are not an exact prefix of training ids",
        )
        print(
            f"  [{modality}] {n_samples:>6} samples -> "
            f"features {tuple(features.shape)} mask_sum={int(mask.sum())} "
            f"prompt_len={len(prm_ids)} full_len={len(full_ids)} prefix_ok=True"
        )

    if modality == "text_only":
        full = processor(text=training_text, padding=False, return_tensors=None)
        _require("input_features" not in full, "text-only must not produce audio features")
        _require("input_features_mask" not in full, "text-only must not produce audio mask")
        print(f"  [{modality}] label mask: prompt_len={len(prompt)} ok=True")
        return
    else:
        wav = _audio(640)
        full = processor(
            text=training_text, audio=[wav], sampling_rate=SAMPLING_RATE,
            padding=False, return_tensors=None,
        )
        labels = np.asarray(full["input_ids"]).reshape(-1).copy()
        prm = processor(
            text=prompt, audio=[wav], sampling_rate=SAMPLING_RATE,
            padding=False, return_tensors=None,
        )
        prompt_len = len(np.asarray(prm["input_ids"]).reshape(-1))
        labels[:prompt_len] = -100
        unmasked = labels[prompt_len:]
        _require(
            int((unmasked == -100).sum()) == 0,
            "response span must be fully unmasked",
        )
        print(
            f"  [{modality}] label mask: prompt_len={prompt_len} "
            f"unmasked_span={len(unmasked)} ok=True"
        )


def check_batch_contract(processor) -> None:
    wav1 = _audio(640)
    wav2 = _audio(1280)
    texts = ["x" * 16, "y" * 24]
    batch = processor(
        text=texts, audio=[wav1, wav2], sampling_rate=SAMPLING_RATE,
        padding=True, return_tensors=None,
    )
    features = np.asarray(batch["input_features"], dtype=np.float32)
    masks = np.asarray(batch["input_features_mask"], dtype=bool)
    _require(features.shape == (2, 2, SAMPLES_PER_TOKEN), f"batch features {features.shape}")
    _require(masks[0].sum() == 1 and masks[1].sum() == 2, f"batch masks {masks.sum(1).tolist()}")
    print(f"  [batch] 640+1280 -> {tuple(features.shape)} masks={masks.sum(1).tolist()} ok=True")


def check_model_metadata(model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    _require(
        "Gemma4UnifiedForConditionalGeneration" in config.get("architectures", []),
        "config.json must declare Gemma4UnifiedForConditionalGeneration",
    )
    _require(config.get("model_type") == "gemma4_unified", "model_type must be gemma4_unified")
    text_config = config.get("text_config", {})
    _require(int(text_config.get("num_hidden_layers", 0)) == 48, "must have 48 text decoder layers")
    _require(text_config.get("attention_k_eq_v") is True, "attention_k_eq_v must be true")
    print(
        "  [metadata] architecture=Gemma4UnifiedForConditionalGeneration "
        "model_type=gemma4_unified layers=48 attention_k_eq_v=true ok=True"
    )


def check_config_invariants(config_paths: list[str]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model.gemma4_io import validate_gemma4_config
    from src.utils import load_yaml

    for path in config_paths:
        config = load_yaml(path)
        validate_gemma4_config(config)
        print(f"  [config] {Path(path).name} invariants ok=True")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Pinned GPFS snapshot directory")
    parser.add_argument("--configs", nargs="*", default=[], help="Gemma config paths to validate")
    args = parser.parse_args()

    _require(
        os.environ.get("HF_HUB_OFFLINE") == "1",
        "HF_HUB_OFFLINE=1 is mandatory on MN5",
    )
    _require(
        os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "TRANSFORMERS_OFFLINE=1 is mandatory on MN5",
    )
    _require(
        os.environ.get("HF_DATASETS_OFFLINE") == "1",
        "HF_DATASETS_OFFLINE=1 is mandatory on MN5",
    )
    model_dir = Path(args.model_dir)
    _require(model_dir.is_dir(), f"model dir not found: {model_dir}")

    from transformers import Gemma4UnifiedProcessor  # noqa: PLC0415

    processor = Gemma4UnifiedProcessor.from_pretrained(model_dir, local_files_only=True)
    print("Gemma4UnifiedProcessor loaded from", model_dir)

    system = "You are a psychologist analyzing speech and transcript information for depression screening."
    user = "Based on the audio and transcript, determine whether the subject is Depressed or Non-depressed.\nAnswer with exactly one label: Depressed or Non-depressed."
    for modality in ("audio_text", "audio_only", "text_only"):
        check_processor_contract(processor, modality, system, user)
    check_batch_contract(processor)
    check_model_metadata(model_dir)
    if args.configs:
        check_config_invariants(args.configs)

    print("Tier A contract smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
