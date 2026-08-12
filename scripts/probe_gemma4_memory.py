#!/usr/bin/env python
"""Probe: does the intended audio+text BF16 LoRA config fit with expandable_segments?

Loads the real longest-transcript DAIC packed30 row from the manifest, renders
the Gemma training text, and runs one forward + backward on one GPU (the exact
production per-device step). Measures peak memory. Exits 1 on OOM.
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model.gemma4_io import GEMMA4_LORA_TARGET_REGEX, prepare_gemma4_example
    from src.model.gemma4_lora import load_model_for_training, load_processor
    from src.data.runtime import load_audio_spans_array, render_user_prompt_text

    config = {
        "model_backend": "gemma4",
        "dataset": "daic",
        "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "prompt": {"system": "You are a psychologist analyzing speech and transcript information for depression screening.", "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "audio_adapter": {"enabled": False, "adapter_dim": 512, "dropout": 0.1, "train_projector": False},
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "bias": "none", "target_modules": GEMMA4_LORA_TARGET_REGEX},
        "training": {"bf16": True, "gradient_checkpointing": True, "selection_metric": "inner_val_macro_f1", "selection_metric_mode": "max"},
        "evaluation": {"sample_prediction_mode": "original_teacher_forced", "headline_mode": "original_teacher_forced", "evaluation_view": "harmonized_all_windows_full_coverage"},
        "data": {
            "use_audio": True, "use_text": True, "sample_mode": "participant_speech_packed30",
            "participant_chunk_samples": 480000, "inter_span_silence_samples": 0,
            "audio_text_transcript_scope": "full_participant",
            "train_chunk_policy": "all_chunks_subject_normalized", "eval_chunk_policy": "all_chunks_mean_score",
            "loss_weight_rescale": "mean_one", "equal_row_weight": False,
            "transcript_max_chars": 0, "allow_empty_transcript": False,
        },
    }

    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8") if line.strip()]
    longest = max(rows, key=lambda r: len(str(r.get("full_participant_transcript", ""))))
    print("longest row:", longest["subject_id"], longest["sample_id"], "transcript_chars:", len(str(longest["full_participant_transcript"])))

    processor = load_processor(args.model_dir, config)
    model = load_model_for_training(args.model_dir, config)
    model.to("cuda")
    model.train()

    waveform = load_audio_spans_array(
        longest["audio_path"], list(longest["audio_spans"]), 16000, False, longest.get("participant_sample_count")
    )
    print("waveform samples:", len(waveform))

    example = {
        "dataset": "daic", "subject_id": longest["subject_id"], "sample_id": longest["sample_id"],
        "label": int(longest["label"]), "label_text": longest["label_text"],
        "internal_label_text": "Depressed" if int(longest["label"]) == 1 else "Non-depressed",
        "transcript": str(longest["full_participant_transcript"]),
        "input_modality": "audio_text",
        "prompt_system_text": config["prompt"]["system"],
        "prompt_user_text": render_user_prompt_text(
            config,
            str(longest["full_participant_transcript"]),
            is_subject_bundle=False,
        ),
        "loss_weight": 1.0,
    }
    prepared = prepare_gemma4_example(example, config, processor)
    batch = processor(text=[prepared["training_text"]], audio=[waveform], sampling_rate=16000, padding=False, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    prompt_ids = processor(text=[prepared["prompt_text"]], audio=[waveform], sampling_rate=16000, padding=False, return_tensors="pt")["input_ids"].to("cuda")
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][0, : prompt_ids.shape[1]] = -100
    print("seq_len:", int(batch["input_ids"].shape[1]), "audio tokens:", int((batch["input_ids"][0] == processor.audio_token_id).sum()))

    torch.cuda.reset_peak_memory_stats()
    try:
        outputs = model(**batch)
        loss = outputs.loss
        print("forward loss:", float(loss.detach().cpu()))
        loss.backward()
        print("backward OK")
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise SystemExit("non-finite gradient")
    except torch.cuda.OutOfMemoryError:
        print("OOM CAUGHT")
        print(torch.cuda.memory_summary(abbreviated=False))
        raise SystemExit(2)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print("peak allocated GiB:", round(peak, 2))
    Path(args.output).write_text(json.dumps({
        "subject_id": longest["subject_id"],
        "sample_id": longest["sample_id"],
        "seq_len": int(batch["input_ids"].shape[1]),
        "transcript_chars": len(str(longest["full_participant_transcript"])),
        "peak_allocated_gib": peak,
        "expandable_segments": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "ok": True,
    }, indent=2) + "\n", encoding="utf-8")
    print("PROBE OK")


if __name__ == "__main__":
    main()
