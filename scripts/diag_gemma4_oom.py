#!/usr/bin/env python
"""Diagnose the Tier B audio_only OOM: print dtypes, shapes, memory summary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model.gemma4_io import GEMMA4_LORA_TARGET_REGEX, prepare_gemma4_example
    from src.model.gemma4_lora import load_model_for_training, load_processor

    config = {
        "model_backend": "gemma4",
        "dataset": "daic",
        "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "prompt": {"system": "s", "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "audio_adapter": {"enabled": False, "adapter_dim": 512, "dropout": 0.1, "train_projector": False},
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none", "target_modules": GEMMA4_LORA_TARGET_REGEX},
        "training": {"bf16": True, "gradient_checkpointing": True, "selection_metric": "inner_val_macro_f1", "selection_metric_mode": "max"},
        "evaluation": {"sample_prediction_mode": "original_teacher_forced", "headline_mode": "original_teacher_forced", "evaluation_view": "harmonized_all_windows_full_coverage"},
        "data": {
            "use_audio": True, "use_text": False, "sample_mode": "participant_speech_packed30",
            "participant_chunk_samples": 480000, "inter_span_silence_samples": 0,
            "audio_text_transcript_scope": "full_participant",
            "train_chunk_policy": "all_chunks_subject_normalized", "eval_chunk_policy": "all_chunks_mean_score",
            "loss_weight_rescale": "mean_one", "equal_row_weight": False,
            "transcript_max_chars": 0, "allow_empty_transcript": False,
        },
    }
    processor = load_processor(args.model_dir, config)
    model = load_model_for_training(args.model_dir, config)
    model.to("cuda")
    param_dtypes = {}
    for name, p in model.named_parameters():
        param_dtypes.setdefault(str(p.dtype), 0)
        param_dtypes[str(p.dtype)] += 1
    print("param dtype counts:", param_dtypes)
    print("base model dtype:", next(model.base_model.model.parameters()).dtype)
    print("gradient_checkpointing enabled:", model.base_model.model.is_gradient_checkpointing if hasattr(model.base_model.model, "is_gradient_checkpointing") else "n/a")

    example = {
        "dataset": "daic", "subject_id": "SYNTH", "sample_id": "SYNTH_audio_only",
        "label": 1, "label_text": "Depressed", "internal_label_text": "Depressed",
        "transcript": "Synthetic transcript text for the Gemma contract smoke.",
        "input_modality": "audio_only",
        "prompt_system_text": "You are a psychologist analyzing speech audio for depression screening.",
        "prompt_user_text": "Based on the audio, determine whether the subject is Depressed or Non-depressed.",
        "loss_weight": 1.0,
    }
    prepared = prepare_gemma4_example(example, config, processor)
    rng = np.random.default_rng(1337)
    waveform = ((rng.random(480000).astype(np.float32) - 0.5) * 0.1).astype(np.float32)
    batch = processor(text=[prepared["training_text"]], audio=[waveform], sampling_rate=16000, padding=False, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    prompt_ids = processor(text=[prepared["prompt_text"]], audio=[waveform], sampling_rate=16000, padding=False, return_tensors="pt")["input_ids"].to("cuda")
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][0, : prompt_ids.shape[1]] = -100
    print("input_ids:", tuple(batch["input_ids"].shape), "seq_len:", int(batch["input_ids"].shape[1]))
    print("input_features:", tuple(batch["input_features"].shape), batch["input_features"].dtype)
    print("input_features_mask:", tuple(batch["input_features_mask"].shape), batch["input_features_mask"].dtype)
    print("mm_token_type_ids:", tuple(batch["mm_token_type_ids"].shape))
    print("audio_token_id count in ids:", int((batch["input_ids"][0] == processor.audio_token_id).sum()))
    print("model.use_cache:", model.config.use_cache, "| base use_cache:", model.base_model.model.config.use_cache)

    try:
        outputs = model(**batch)
        print("forward OK, loss:", float(outputs.loss.detach().cpu()))
        torch.cuda.reset_peak_memory_stats()
        outputs.loss.backward()
        print("backward OK")
    except torch.cuda.OutOfMemoryError:
        print("OOM CAUGHT")
        print(torch.cuda.memory_summary(abbreviated=False))
        raise SystemExit(1)
    print("peak allocated GiB:", torch.cuda.max_memory_allocated() / 1024 ** 3)


if __name__ == "__main__":
    main()
