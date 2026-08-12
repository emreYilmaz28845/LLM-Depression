from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.features.gemma4_hidden_collator import (
    GEMMA4_MODEL_INPUT_KEYS,
    Gemma4PromptOnlyExtractionCollator,
)
from src.features.pooling import last_valid_token
from src.model.gemma4_io import (
    GEMMA4_AUDIO_SAMPLING_RATE,
    validate_gemma4_audio,
)
from src.model.runtime import load_model_for_inference, load_processor


def _synthetic_waveform(seconds: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(1234)
    samples = int(seconds * GEMMA4_AUDIO_SAMPLING_RATE)
    waveform = 0.2 * rng.standard_normal(samples).astype(np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    return validate_gemma4_audio(waveform, GEMMA4_AUDIO_SAMPLING_RATE)


def _synthetic_example(modality: str) -> dict[str, Any]:
    example = {
        "dataset": "daic",
        "subject_id": "000",
        "sample_id": "smoke_000",
        "label": 0,
        "partition": "final_eval",
        "fold": 0,
        "prompt_system_text": (
            "You are a psychologist analyzing speech and transcript information "
            "for depression screening."
        ),
        "prompt_user_text": (
            "Based on the provided material, determine whether the subject is "
            "Depressed or Non-depressed."
        ),
    }
    return example


def _render_prompt(processor, example: dict[str, Any], modality: str) -> str:
    from src.model.gemma4_io import render_gemma4_prompt

    return render_gemma4_prompt(
        processor, example["prompt_system_text"], example["prompt_user_text"], modality
    )


def _parent_config(adapter_path: Path) -> dict[str, Any]:
    """Load the parent run_config.yaml beside the adapter so the strict Gemma
    config validation sees the exact production config."""
    import yaml

    run_config_path = adapter_path.parent / "run_config.yaml"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"parent run_config.yaml not found: {run_config_path}")
    payload = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid parent run_config.yaml: {run_config_path}")
    return payload["config"]


def run_contract(
    *,
    model_path: str,
    adapter_path: Path,
    modality: str,
    output_dir: Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    parent_config = _parent_config(adapter_path)
    processor = load_processor(model_path, parent_config)
    model = load_model_for_inference(
        model_path, adapter_path=str(adapter_path), config=parent_config
    )
    device = torch.device(device_name)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    example = _synthetic_example(modality)
    example["prompt_text"] = _render_prompt(processor, example, modality)
    if modality in ("audio_only", "audio_text"):
        example["audio_paths"] = ["<synthetic>"]
        example["audio_arrays"] = [_synthetic_waveform()]
    else:
        example["audio_paths"] = []
        example["audio_arrays"] = []

    collator = Gemma4PromptOnlyExtractionCollator(processor)
    model_inputs, metadata = collator([example])
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    if "labels" in model_inputs:
        raise AssertionError("labels must never reach the model in the contract smoke")
    if not metadata[0]["prompt_text"]:
        raise AssertionError("empty prompt rendered")

    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1]
    if hidden is None:
        raise AssertionError("outputs.hidden_states[-1] is missing")
    if tuple(hidden.shape[:2]) != tuple(model_inputs["attention_mask"].shape):
        raise AssertionError(
            f"hidden {tuple(hidden.shape[:2])} does not align with input mask "
            f"{tuple(model_inputs['attention_mask'].shape)}"
        )
    vector = (
        last_valid_token(hidden, model_inputs["attention_mask"])
        .cpu()
        .numpy()[0]
        .astype(np.float32, copy=False)
    )
    if vector.shape != (3840,):
        raise AssertionError(f"expected (3840,) pooled vector, got {vector.shape}")
    if not bool(np.isfinite(vector).all()):
        raise AssertionError("pooled vector contains non-finite values")
    if vector.dtype != np.float32:
        raise AssertionError(f"pooled vector dtype is {vector.dtype}, expected float32")

    with torch.inference_mode():
        repeated = model(
            **model_inputs,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    repeated_vector = (
        last_valid_token(repeated.hidden_states[-1], model_inputs["attention_mask"])
        .cpu()
        .numpy()[0]
        .astype(np.float32, copy=False)
    )
    max_abs_diff = float(np.max(np.abs(vector - repeated_vector)))
    if not np.allclose(vector, repeated_vector, rtol=1e-5, atol=1e-5):
        raise AssertionError(
            f"determinism tolerance failed with max_abs_diff={max_abs_diff}"
        )

    peak_mb = float(torch.cuda.max_memory_allocated(device=device)) / (1024 * 1024)
    return {
        "status": "passed",
        "modality": modality,
        "model_path": model_path,
        "adapter_path": str(adapter_path),
        "pooled_vector_dimension": int(vector.shape[0]),
        "pooled_vector_dtype": str(vector.dtype),
        "finite": True,
        "determinism_max_abs_diff": max_abs_diff,
        "determinism_rtol": 1e-5,
        "determinism_atol": 1e-5,
        "peak_gpu_memory_mb": peak_mb,
        "keys_reaching_model": sorted(model_inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 fixed-head Tier A contract smoke.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("audio_text", "audio_only", "text_only"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    result = run_contract(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        modality=args.modality,
        output_dir=args.output,
    )
    result["elapsed_seconds"] = round(time.time() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
