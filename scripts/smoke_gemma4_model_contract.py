#!/usr/bin/env python
"""Gemma 4 DAIC model contract smoke (Tier B) — one H100, real 12B weights.

Runs through Slurm in the dedicated Gemma environment with the pinned GPFS
snapshot. For each modality (text-only, audio-only, audio+text):

- one processor batch;
- one forward pass with labels;
- one backward pass;
- one optimizer step on LoRA parameters;
- finite loss and finite gradients;
- exact 288-module LoRA audit (no vision/embedding/audio-projection adaptation,
  every non-LoRA parameter frozen);
- adapter and processor save;
- fresh base plus adapter reload;
- teacher-forced candidate scoring after reload.

Uses synthetic text and a synthetic waveform only. No DAIC subject content is
logged. Offline-only: everything loads with local_files_only=True.

Usage:
  python scripts/smoke_gemma4_model_contract.py --model-dir <snapshot>
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SAMPLES_PER_TOKEN = 640
SAMPLING_RATE = 16000
MAX_SAMPLES = 480000
TURN_TERMINATOR = "<turn|>\n"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _synthetic_waveform(samples: int) -> np.ndarray:
    rng = np.random.default_rng(1337)
    return ((rng.random(samples).astype(np.float32) - 0.5) * 0.1).astype(np.float32)


def _synthetic_user_text(modality: str) -> str:
    if modality == "text_only":
        return "Based on the transcript, determine whether the subject is Depressed or Non-depressed."
    if modality == "audio_only":
        return "Based on the audio, determine whether the subject is Depressed or Non-depressed."
    return "Based on the audio and transcript, determine whether the subject is Depressed or Non-depressed."


def _messages(modality: str, system: str, user: str):
    if modality == "text_only":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "audio"},
            {"type": "text", "text": user},
        ]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--work-dir", default="/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/outputs/gemma4_smoke_tierB")
    parser.add_argument("--modalities", nargs="*", default=["text_only", "audio_only", "audio_text"])
    args = parser.parse_args()

    _require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 is mandatory")
    _require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE=1 is mandatory")
    _require(os.environ.get("HF_DATASETS_OFFLINE") == "1", "HF_DATASETS_OFFLINE=1 is mandatory")
    _require(torch.cuda.is_available(), "Tier B requires one CUDA GPU")
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model.gemma4_io import GEMMA4_LORA_TARGET_REGEX, prepare_gemma4_example
    from src.model.gemma4_lora import load_model_for_inference, load_model_for_training, load_processor

    config_base = {
        "model_backend": "gemma4",
        "dataset": "daic",
        "model_revision": REVISION,
        "prompt": {
            "system": "You are a psychologist analyzing speech and transcript information for depression screening.",
            "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "audio_adapter": {"enabled": False, "adapter_dim": 512, "dropout": 0.1, "train_projector": False},
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none", "target_modules": GEMMA4_LORA_TARGET_REGEX},
        "training": {"bf16": True, "gradient_checkpointing": True, "selection_metric": "inner_val_macro_f1", "selection_metric_mode": "max"},
        "evaluation": {
            "sample_prediction_mode": "original_teacher_forced",
            "headline_mode": "original_teacher_forced",
            "evaluation_view": "harmonized_all_windows_full_coverage",
        },
    }

    processor = load_processor(args.model_dir, config_base)
    print("processor loaded:", type(processor).__name__)

    for modality in args.modalities:
        config = dict(config_base)
        config["data"] = {
            "use_audio": modality != "text_only",
            "use_text": modality != "audio_only",
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
        }
        modality_dir = work_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Tier B modality: {modality} ===")

        torch.cuda.reset_peak_memory_stats()
        model = load_model_for_training(args.model_dir, config)
        model.to(device)

        # Exact 288-module audit through the repository loader.
        from peft.tuners.tuners_utils import inspect_matched_modules  # noqa: PLC0415

        matched = {str(name) for name in inspect_matched_modules(model.base_model)["matched"]}
        _require(len(matched) == 288, f"{modality}: expected 288 adapted modules, got {len(matched)}")
        vision = [n for n in matched if "vision" in n or "embed_vision" in n]
        embeddings = [n for n in matched if "embed_tokens" in n or "lm_head" in n or "embed_audio" in n]
        _require(not vision, f"{modality}: vision modules adapted: {vision[:4]}")
        _require(not embeddings, f"{modality}: embedding/lm-head/audio modules adapted: {embeddings[:4]}")
        non_lora_trainable = [
            name for name, p in model.named_parameters() if "lora_" not in name and p.requires_grad
        ]
        _require(not non_lora_trainable, f"{modality}: non-LoRA trainable params: {non_lora_trainable[:4]}")
        audio_proj_trainable = any(
            "embed_audio.embedding_projection" in name and p.requires_grad
            for name, p in model.named_parameters()
        )
        _require(not audio_proj_trainable, f"{modality}: audio projection must be frozen")
        lora_trainable = sum(
            int(p.numel()) for name, p in model.named_parameters() if "lora_" in name and p.requires_grad
        )
        _require(lora_trainable > 0, f"{modality}: no trainable LoRA parameter")
        print(f"  audit: matched={len(matched)} lora_trainable_params={lora_trainable}")

        # One processor batch (synthetic).
        system = config_base["prompt"]["system"]
        user = _synthetic_user_text(modality)
        example = {
            "dataset": "daic",
            "subject_id": "SYNTH",
            "sample_id": f"SYNTH_{modality}",
            "label": 1,
            "label_text": "Depressed",
            "internal_label_text": "Depressed",
            "transcript": "Synthetic transcript text for the Gemma contract smoke.",
            "input_modality": modality,
            "prompt_system_text": system,
            "prompt_user_text": user,
            "loss_weight": 1.0,
        }
        waveform = _synthetic_waveform(480000) if modality != "text_only" else None
        prepared = prepare_gemma4_example(example, config, processor)
        if waveform is not None:
            batch = processor(
                text=[prepared["training_text"]],
                audio=[waveform],
                sampling_rate=SAMPLING_RATE,
                padding=False,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
        else:
            batch = processor(text=[prepared["training_text"]], padding=False, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}

        # Forward with labels.
        prompt_ids = processor(
            text=[prepared["prompt_text"]],
            audio=[waveform] if waveform is not None else None,
            sampling_rate=SAMPLING_RATE if waveform is not None else None,
            padding=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        prompt_len = int(prompt_ids.shape[1])
        labels = batch["input_ids"].clone()
        labels[0, :prompt_len] = -100
        batch["labels"] = labels
        _require(
            torch.equal(prompt_ids[0], batch["input_ids"][0, :prompt_len]),
            f"{modality}: prompt ids are not an exact prefix of training ids",
        )

        outputs = model(**batch)
        loss = outputs.loss
        _require(torch.isfinite(loss), f"{modality}: loss is not finite: {loss}")
        print(f"  forward: loss={float(loss.detach().cpu()):.6f}")

        # Backward + optimizer step on LoRA parameters.
        loss.backward()
        finite_grads = 0
        total_grads = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            total_grads += 1
            if torch.isfinite(parameter.grad).all():
                finite_grads += 1
        _require(total_grads > 0, f"{modality}: no gradients produced")
        _require(
            finite_grads == total_grads,
            f"{modality}: {total_grads - finite_grads}/{total_grads} gradients are not finite",
        )
        print(f"  backward: finite_grads={finite_grads}/{total_grads}")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2.0e-4,
        )
        optimizer.step()
        optimizer.zero_grad()
        print("  optimizer step: OK")

        # Adapter + processor save.
        adapter_dir = modality_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(adapter_dir)
        print(f"  saved adapter + processor -> {adapter_dir}")

        # Fresh base + adapter reload, teacher-forced candidate scoring.
        del model
        torch.cuda.empty_cache()
        reloaded = load_model_for_inference(args.model_dir, adapter_dir, config)
        reloaded.to(device)
        reloaded.eval()
        scores = {}
        with torch.no_grad():
            for candidate in ("Depressed", "Non-depressed"):
                candidate_text = prepared["prompt_text"] + candidate
                if waveform is not None:
                    inputs = processor(
                        text=[candidate_text],
                        audio=[waveform],
                        sampling_rate=SAMPLING_RATE,
                        padding=False,
                        return_tensors="pt",
                    )
                else:
                    inputs = processor(text=[candidate_text], padding=False, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = reloaded(**inputs).logits[0]
                selected = logits[prompt_len - 1 : inputs["input_ids"].shape[1] - 1]
                log_probs = torch.log_softmax(selected, dim=-1)
                target_ids = inputs["input_ids"][0, prompt_len:]
                scores[candidate] = float(
                    log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean().item()
                )
        print(f"  teacher-forced scores after reload: {scores}")
        _require(all(math.isfinite(value) for value in scores.values()), f"{modality}: non-finite score")
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  peak CUDA memory: {peak:.2f} GiB")
        with (modality_dir / "peak_memory_gib.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"{peak:.4f}\n")

        # Release CUDA state before the next modality loads another 12B model.
        del reloaded, inputs, logits, scores, batch, labels, outputs, loss, optimizer, prompt_ids, prepared
        if waveform is not None:
            del waveform
        torch.cuda.empty_cache()
        print(f"  released CUDA state (allocated now {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB)")

    print("\nTier B model contract smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
