"""Verify A/B labels at the rendered prompt boundary without model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.runtime import (  # noqa: E402
    build_prompt_text,
    build_training_text,
    render_user_prompt_text,
    resolve_audio_placeholder,
)
from src.utils import load_yaml, resolve_label_config  # noqa: E402


def _ids(processor, text: str, audio: list[np.ndarray] | None) -> list[int]:
    kwargs = {"text": text, "return_tensors": None, "padding": False}
    if audio is not None:
        kwargs["audio"] = audio
        kwargs["sampling_rate"] = int(processor.feature_extractor.sampling_rate)
    encoded = processor(**kwargs)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def verify_config(config_path: Path, processor) -> dict[str, object]:
    config = load_yaml(config_path)
    labels = resolve_label_config(config)
    if labels["internal_positive_label"] != "A" or labels["internal_negative_label"] != "B":
        raise ValueError(f"{config_path}: expected internal A/B labels")
    use_audio = bool(config["data"]["use_audio"])
    user_text = render_user_prompt_text(config, "A short sample transcript.")
    prompt = build_prompt_text(
        config["prompt"]["system"], user_text, int(use_audio), use_audio,
        audio_placeholder=resolve_audio_placeholder(config),
    )
    audio = None
    if use_audio:
        sampling_rate = int(processor.feature_extractor.sampling_rate)
        audio = [np.zeros(sampling_rate, dtype=np.float32)]
    prompt_ids = _ids(processor, prompt, audio)
    label_ids: dict[str, int] = {}
    for label in ("A", "B"):
        candidate_ids = _ids(processor, prompt + label, audio)
        training_ids = _ids(processor, build_training_text(prompt, label), audio)
        if candidate_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{config_path}: {label} changes the prompt token prefix")
        continuation = candidate_ids[len(prompt_ids):]
        if len(continuation) != 1:
            raise ValueError(f"{config_path}: {label} uses {len(continuation)} tokens at the prompt boundary")
        if training_ids[:len(prompt_ids)] != prompt_ids or training_ids[len(prompt_ids)] != continuation[0]:
            raise ValueError(f"{config_path}: training and likelihood label tokens differ for {label}")
        label_ids[label] = continuation[0]
    if label_ids["A"] == label_ids["B"]:
        raise ValueError(f"{config_path}: A and B have the same token ID")
    return {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "model_backend": "qwen2audio" if use_audio else "qwen2_text",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "label_token_ids": label_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", required=True, type=Path)
    parser.add_argument("--audio-model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted((PROJECT_ROOT / "configs/main").glob("*likelihood_ab_v1.yaml"))
    if len(paths) != 15:
        parser.error(f"expected 15 core A/B configs, found {len(paths)}")
    text_processor = AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True)
    audio_processor = AutoProcessor.from_pretrained(args.audio_model_dir, local_files_only=True)
    results = []
    for path in paths:
        config = load_yaml(path)
        if config.get("model_backend") not in (None, "qwen2audio", "text"):
            raise ValueError(f"{path}: unsupported model backend for this audit")
        processor = audio_processor if config["data"]["use_audio"] else text_processor
        results.append(verify_config(path, processor))
    payload = {
        "status": "passed",
        "configs_checked": len(results),
        "text_tokenizer_sha256": hashlib.sha256((args.text_model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "audio_tokenizer_sha256": hashlib.sha256((args.audio_model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
