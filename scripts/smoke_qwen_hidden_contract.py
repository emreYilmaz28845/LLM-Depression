"""Qwen prompt-only hidden-state forward contract smoke.

Loads the pinned base model plus a completed DAIC adapter and runs one
prompt-only forward pass with synthetic input (synthetic 16 kHz waveform for
audio modalities, synthetic text otherwise). Requires: offline loading
succeeds, no labels or generation are used, ``outputs.hidden_states[-1]``
exists, the last-valid-prompt-token pooled vector is finite float32 with the
backend dimension (4096 audio, 3584 text), and the repeated forward is
deterministic within the locked tolerance. Records peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.features.pooling import last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator
from src.model.runtime import load_model_for_inference, load_processor, resolve_processor_sampling_rate
from src.utils import resolve_model_backend

QWEN_SYSTEM_PROMPT = (
    "You are a psychologist analyzing speech audio for depression screening."
)
QWEN_AUDIO_PROMPT = (
    "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
    "Based on the provided material, determine whether the subject is "
    "Depressed or Non-depressed.\nAnswer with exactly one label."
)
QWEN_TEXT_PROMPT = (
    "Based on the provided transcript, determine whether the subject is "
    "Depressed or Non-depressed.\nAnswer with exactly one label."
)


def _synthetic_waveform(seconds: float = 10.0, sampling_rate: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(1234)
    samples = int(seconds * sampling_rate)
    waveform = 0.2 * rng.standard_normal(samples).astype(np.float32)
    return np.clip(waveform, -1.0, 1.0)


def _synthetic_example(modality: str) -> dict[str, Any]:
    example = {
        "dataset": "daic",
        "subject_id": "000",
        "sample_id": "smoke_000",
        "label": 0,
        "partition": "final_eval",
        "fold": 0,
    }
    if modality == "text_only":
        example["prompt_text"] = (
            "<|im_start|>system\n" + QWEN_SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n" + QWEN_TEXT_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
        )
        example["audio_paths"] = []
        example["audio_arrays"] = []
    else:
        example["prompt_text"] = (
            "<|im_start|>system\n" + QWEN_SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n" + QWEN_AUDIO_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
        )
        example["audio_paths"] = ["<synthetic>"]
        example["audio_arrays"] = [_synthetic_waveform()]
    return example


def _parent_config(adapter_path: Path) -> dict[str, Any]:
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
    model = load_model_for_inference(model_path, adapter_path=str(adapter_path), config=parent_config)
    device = torch.device(device_name)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    expected_dimension = 3584 if modality == "text_only" else 4096
    sampling_rate = resolve_processor_sampling_rate(processor)
    example = _synthetic_example(modality)
    if modality in ("audio_only", "audio_text"):
        example["audio_arrays"] = [
            _synthetic_waveform(sampling_rate=sampling_rate)
        ]

    collator = PromptOnlyExtractionCollator(processor)
    model_inputs, metadata = collator([example])
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    if "labels" in model_inputs:
        raise AssertionError("labels must never reach the model in the contract smoke")
    if not metadata[0]["prompt_text"]:
        raise AssertionError("empty prompt rendered")
    if modality == "text_only" and "input_features" in model_inputs:
        raise AssertionError("text-only contract must omit audio feature tensors")

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
    from src.features.pooling import aligned_attention_mask

    mask, _ = aligned_attention_mask(
        hidden, model_inputs["attention_mask"], getattr(outputs, "attention_mask", None)
    )
    vector = last_valid_token(hidden, mask).cpu().numpy()[0].astype(np.float32, copy=False)
    if vector.shape != (expected_dimension,):
        raise AssertionError(f"expected ({expected_dimension},) pooled vector, got {vector.shape}")
    if not bool(np.isfinite(vector).all()):
        raise AssertionError("pooled vector contains non-finite values")
    if vector.dtype != np.float32:
        raise AssertionError(f"pooled vector must be float32, got {vector.dtype}")

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
        raise AssertionError(f"determinism check failed: max_abs_diff={max_abs_diff}")

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    backend = resolve_model_backend(parent_config)
    result = {
        "status": "passed",
        "backend": backend,
        "modality": modality,
        "model_path": model_path,
        "adapter_path": str(adapter_path),
        "vector_dimension": expected_dimension,
        "vector_dtype": "float32",
        "vector_finite": True,
        "determinism_rtol": 1e-5,
        "determinism_atol": 1e-5,
        "determinism_max_abs_diff": max_abs_diff,
        "peak_gpu_memory_gb": round(peak_memory_gb, 3),
        "labels_used": False,
        "generation_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contract_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("audio_only", "audio_text", "text_only"))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_contract(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        modality=args.modality,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
