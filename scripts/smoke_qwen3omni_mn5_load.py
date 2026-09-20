#!/usr/bin/env python
"""Qwen3-Omni MN5 load smoke — real BF16 weights, sharded with device_map="auto".

This answers one question only: does the dedicated offline environment load the
real Qwen3-Omni snapshot on H100s and run the repository's own inference path?
It deliberately contains no training code and no FSDP: the weights are split
across the visible GPUs by accelerate's device_map so a single process can hold
~66 GB of parameters. That is enough for a load/forward/scoring check; real
training still needs FSDP or DeepSpeed ZeRO.

Checks:
- the offline environment imports (torch, transformers 5.x, peft, accelerate);
- Qwen3OmniMoeProcessor loads from the GPFS snapshot;
- the Thinker loads BF16 sharded across the GPUs, with no CPU/disk offload;
- one repository-shaped audio+text batch (src.data.runtime prompt builder +
  Qwen2AudioSFTCollator) produces a finite loss;
- teacher-forced Depressed / Non-depressed margins are finite and distinct;
- per-GPU memory and the device map are recorded to JSON.

Synthetic waveform and synthetic transcript only; no subject content is logged.
Offline only: every load uses local_files_only=True.

Usage:
  python scripts/smoke_qwen3omni_mn5_load.py --model-dir <snapshot>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SAMPLING_RATE = 16000
CHUNK_SAMPLES = 480000  # harmonized packed30 chunk: 30 s at 16 kHz
LABEL_TEXTS = ("Depressed", "Non-depressed")
SYNTH_TRANSCRIPT = (
    "I have been feeling low for a few weeks. Sleeping is hard and I stopped meeting friends."
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _synthetic_waveform(samples: int) -> np.ndarray:
    rng = np.random.default_rng(1337)
    return ((rng.random(samples).astype(np.float32) - 0.5) * 0.1).astype(np.float32)


def _config(model_dir: str) -> dict:
    return {
        "model_backend": "qwen3omni",
        "model_name_or_path": model_dir,
        "dataset": "daic",
        "prompt": {
            "system": (
                "You are a psychologist analyzing speech and transcript information "
                "for depression screening."
            ),
            "user_template": (
                "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, "
                "determine whether the subject is {label_descriptor}.\n{label_instruction}"
            ),
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {"use_audio": True, "use_text": True},
        "audio_adapter": {"enabled": False, "adapter_dim": 512, "dropout": 0.1, "train_projector": False},
    }


def _load_model(model_dir: str):
    """Load the Thinker sharded across the visible GPUs, without any repo training code."""
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": "auto",
        "local_files_only": True,
    }
    try:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_dir, **kwargs)
        print("loaded the standalone Thinker directly (talker never built)", flush=True)
        return model, "thinker_direct"
    except Exception as exc:  # noqa: BLE001 - fall back to the full checkpoint, then take .thinker
        print(f"direct Thinker load failed ({type(exc).__name__}: {exc}); trying the full omni model", flush=True)

    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_dir, **kwargs)
    if hasattr(full, "disable_talker"):
        full.disable_talker()
    thinker = full.thinker
    print("loaded the full omni model, disabled the talker, and kept .thinker", flush=True)
    return thinker, "talker_disabled"


def _device_map_summary(model) -> dict:
    raw = getattr(model, "hf_device_map", None)
    if not raw:
        return {}
    counts: dict[str, int] = {}
    for value in raw.values():
        if isinstance(value, (list, tuple)):
            key = f"{list(value)}"
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision-note", default="")
    args = parser.parse_args()

    _require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 is mandatory")
    _require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE=1 is mandatory")
    _require(os.environ.get("HF_DATASETS_OFFLINE") == "1", "HF_DATASETS_OFFLINE=1 is mandatory")
    _require(torch.cuda.is_available(), "this smoke requires CUDA")
    _require(Path(args.model_dir, "config.json").is_file(), f"model config missing: {args.model_dir}")

    gpu_count = torch.cuda.device_count()
    _require(gpu_count >= 1, "no CUDA device visible")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.runtime import (
        build_prompt_text,
        build_training_text,
        render_user_prompt_text,
        resolve_audio_placeholder,
    )
    from src.model.collator import Qwen2AudioSFTCollator
    from src.model.runtime import load_processor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _config(args.model_dir)
    system_prompt = config["prompt"]["system"]
    user_text = render_user_prompt_text(config, transcript=SYNTH_TRANSCRIPT)
    audio_placeholder = resolve_audio_placeholder(config)
    prompt_text = build_prompt_text(
        system_prompt, user_text, 1, True, audio_placeholder=audio_placeholder
    )
    training_text = build_training_text(prompt_text, LABEL_TEXTS[0])
    print(f"audio placeholder in use: {audio_placeholder!r}", flush=True)
    print(f"prompt chars: {len(prompt_text)}", flush=True)

    started = time.time()
    print(f"loading processor from {args.model_dir}", flush=True)
    processor = load_processor(args.model_dir, config)
    print(f"processor: {type(processor).__name__}", flush=True)
    _require(
        hasattr(processor, "feature_extractor") and processor.feature_extractor is not None,
        "processor has no audio feature_extractor; the collator cannot resolve a sampling rate",
    )

    print(f"loading model (bf16, sdpa, device_map=auto) across {gpu_count} GPU(s)", flush=True)
    model, load_mode = _load_model(args.model_dir)
    model.eval()
    load_seconds = time.time() - started
    print(f"model loaded in {load_seconds:.1f} s", flush=True)

    device_map = _device_map_summary(model)
    print(f"device map: {device_map}", flush=True)
    offloaded = [key for key in device_map if key in {"cpu", "disk"}]
    _require(not offloaded, f"weights were offloaded away from the GPUs: {offloaded}")

    per_gpu_allocated = {
        index: round(torch.cuda.memory_allocated(index) / 1024 ** 3, 3)
        for index in range(gpu_count)
    }
    print(f"allocated per GPU (GiB): {per_gpu_allocated}", flush=True)

    waveform = _synthetic_waveform(CHUNK_SAMPLES)
    example = {
        "sample_id": "SYNTH_AUDIO_TEXT",
        "subject_id": "SYNTH",
        "label": 1,
        "audio_arrays": [waveform],
        "prompt_text": prompt_text,
        "training_text": training_text,
    }
    collator = Qwen2AudioSFTCollator(processor=processor, debug=False)
    batch = collator([example])
    for key in ("input_ids", "attention_mask", "labels", "input_features", "feature_attention_mask"):
        _require(key in batch, f"collator output is missing {key}")
    prompt_ids = processor(
        text=[prompt_text], audio=[waveform], sampling_rate=SAMPLING_RATE, padding=False, return_tensors="pt"
    )["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    print(
        "collated batch: "
        f"input_ids {tuple(batch['input_ids'].shape)}, "
        f"input_features {tuple(batch['input_features'].shape)}, "
        f"prompt_len {prompt_len}",
        flush=True,
    )
    masked = int((batch["labels"] == -100).sum())
    _require(masked == prompt_len, f"label masking covers {masked} positions, expected {prompt_len}")
    unmasked_ids = [int(token) for token in batch["labels"][batch["labels"] != -100]]
    unmasked_text = processor.tokenizer.decode(unmasked_ids)
    _require(
        unmasked_text.strip().startswith(LABEL_TEXTS[0]),
        f"unmasked label tokens do not start with {LABEL_TEXTS[0]!r}: {unmasked_text!r}",
    )
    print(f"unmasked label text: {unmasked_text!r}", flush=True)

    target_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def _to_model(tensors: dict) -> dict:
        """Move to the first device and cast audio features to the model dtype.

        The repository collator emits float32 audio features (the Qwen2-Audio path
        relies on Whisper's encoder casting them internally). The Qwen3-Omni audio
        encoder does not, so ``input_features`` must be cast here; without it the
        conv2d layers raise "Input type (float) and bias type (c10::BFloat16)
        should be the same".
        """
        moved = {key: value.to(target_device) for key, value in tensors.items() if key != "loss_weight"}
        if "input_features" in moved:
            moved["input_features"] = moved["input_features"].to(model_dtype)
        return moved

    model_batch = _to_model(batch)
    print(f"input_features dtype after cast: {model_batch['input_features'].dtype}", flush=True)
    with torch.no_grad():
        outputs = model(**model_batch)
    loss = float(outputs.loss.detach().float().cpu())
    print(f"forward loss: {loss:.6f}", flush=True)
    _require(torch.isfinite(torch.tensor(loss)), f"forward loss is not finite: {loss}")

    scores: dict[str, float] = {}
    with torch.no_grad():
        for candidate in LABEL_TEXTS:
            inputs = processor(
                text=[prompt_text + candidate],
                audio=[waveform],
                sampling_rate=SAMPLING_RATE,
                padding=False,
                return_tensors="pt",
            )
            moved = _to_model(inputs)
            logits = model(**moved).logits[0]
            selected = logits[prompt_len - 1 : inputs["input_ids"].shape[1] - 1]
            log_probs = torch.log_softmax(selected.float(), dim=-1)
            target_ids = inputs["input_ids"][0, prompt_len:].to(log_probs.device)
            scores[candidate] = float(log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean().item())
    margin = scores[LABEL_TEXTS[0]] - scores[LABEL_TEXTS[1]]
    print(f"teacher-forced scores: {scores}", flush=True)
    print(f"margin (Depressed - Non-depressed): {margin:+.6f}", flush=True)
    _require(all(np.isfinite(value) for value in scores.values()), f"non-finite candidate score: {scores}")
    _require(abs(margin) > 1e-6, "candidate margin is exactly zero; scoring is not discriminating")

    peak_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    result = {
        "model_dir": args.model_dir,
        "revision_note": args.revision_note,
        "load_mode": load_mode,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gpu_count": gpu_count,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "load_seconds": round(load_seconds, 1),
        "device_map": device_map,
        "allocated_per_gpu_gib": per_gpu_allocated,
        "peak_allocated_gib": round(peak_allocated, 3),
        "collated_input_ids_shape": list(batch["input_ids"].shape),
        "collated_input_features_shape": list(batch["input_features"].shape),
        "prompt_len": prompt_len,
        "forward_loss": loss,
        "candidate_scores": scores,
        "margin": margin,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "accelerate": __import__("accelerate").__version__,
        },
    }
    stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    result_path = output_dir / f"qwen3omni_load_smoke_{stamp}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"result JSON: {result_path}", flush=True)
    print("\nQwen3-Omni MN5 load smoke: ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
