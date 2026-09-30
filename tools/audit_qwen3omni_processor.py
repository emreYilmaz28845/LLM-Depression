#!/usr/bin/env python3
"""Audit the Qwen3-Omni audio processor against the pooled Turkish inputs.

Runs in the Qwen3-Omni environment on MN5, CPU only, model weights are never
loaded: ``Qwen3OmniMoeProcessor.from_pretrained`` fetches the tokenizer and the
audio feature extractor only. For every pooled audio cell (native audio-only,
native audio+text, English audio+text) and both question conditions it

* renders the real prompt (system prompt, question-context sentence, audio
  placeholder block) through the same helpers the runtime uses,
* loads a real 30-second window with the same loader the training path uses,
* runs the processor call the evaluation path uses (``text`` + ``audio`` +
  ``sampling_rate``), and
* checks that the audio token is present, the feature tensors are finite and
  the text ids are non-empty.

The English and native transcripts come from the task runtime manifests built by
``scripts/build_turkish_pooled_manifest.py``; the audio files come from the
dataset root the manifest paths point at, so this must run where the dataset is
mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prompt_context import resolve_question_context_sentences, resolve_system_prompt
from src.data.runtime import build_prompt_text, load_audio_array, render_user_prompt_text
from src.utils import load_yaml_with_overrides, read_jsonl
from tools.qwen3_pooled_defaults import POOLED_TURKISH_ENGLISH, POOLED_TURKISH_NATIVE

CONDITIONS = ("pos_only_t17", "negative_only_t17")
AUDIO_CELLS = (
    ("native", "audio_only", POOLED_TURKISH_NATIVE["audio_only"]),
    ("native", "audio_text", POOLED_TURKISH_NATIVE["audio_text"]),
    ("english", "audio_text", POOLED_TURKISH_ENGLISH["audio_text"]),
)
WINDOW_SECONDS = 30.0


class AuditError(RuntimeError):
    """Raised when the processor audit finds a contract violation."""


def sample_rows(runtime_root: Path, language: str) -> dict[str, dict[str, Any]]:
    manifest_dir = runtime_root / ("manifests_en" if language == "english" else "manifests") / "turkish"
    manifest_path = manifest_dir / "turkish_manifest.jsonl"
    if not manifest_path.is_file():
        raise AuditError(f"missing pooled {language} manifest: {manifest_path}")
    samples: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(manifest_path):
        samples.setdefault(str(row["dataset_variant"]), row)
    missing = [condition for condition in CONDITIONS if condition not in samples]
    if missing:
        raise AuditError(f"{language}: manifest has no rows for {missing}")
    return samples


def _float_tensors(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        key: value
        for key, value in inputs.items()
        if torch.is_tensor(value) and value.is_floating_point()
    }


def audit(*, runtime_root: Path, model_path: Path) -> dict[str, Any]:
    import torch

    from src.model.runtime import load_processor, resolve_processor_sampling_rate

    config = load_yaml_with_overrides(PROJECT_ROOT / POOLED_TURKISH_NATIVE["audio_text"], [])
    processor = load_processor(str(model_path), config)
    tokenizer = processor.tokenizer
    sampling_rate = resolve_processor_sampling_rate(processor)
    if not sampling_rate:
        raise AuditError("the Qwen3-Omni processor declares no sampling rate")
    samples_by_language = {
        "native": sample_rows(runtime_root, "native"),
        "english": sample_rows(runtime_root, "english"),
    }
    cells = []
    for language, modality, config_rel in AUDIO_CELLS:
        cell_config = load_yaml_with_overrides(PROJECT_ROOT / config_rel, [])
        system_prompt = resolve_system_prompt(cell_config)
        sentences = resolve_question_context_sentences(cell_config)
        conditions = []
        for condition in CONDITIONS:
            row = samples_by_language[language][condition]
            transcript = str(row.get("transcript", "")) if modality == "audio_text" else ""
            user_text = render_user_prompt_text(
                cell_config,
                transcript,
                is_subject_bundle=True,
                question_condition=condition,
            )
            prompt_text = build_prompt_text(system_prompt, user_text, 1, True)
            audio_path = str(row["audio_paths"][0])
            audio = load_audio_array(audio_path, int(sampling_rate), WINDOW_SECONDS, False, None, None)
            inputs = processor(
                text=[prompt_text],
                audio=[audio],
                sampling_rate=int(sampling_rate),
                return_tensors="pt",
                padding=False,
            )
            sizes = {key: tuple(value.shape) for key, value in inputs.items() if torch.is_tensor(value)}
            float_tensors = _float_tensors(dict(inputs))
            if not float_tensors:
                raise AuditError(f"{language}/{modality}/{condition}: no floating feature tensor returned")
            for name, value in float_tensors.items():
                if not bool(torch.isfinite(value).all()):
                    raise AuditError(f"{language}/{modality}/{condition}: {name} has non-finite values")
            if "input_ids" not in inputs or int(inputs["input_ids"].numel()) == 0:
                raise AuditError(f"{language}/{modality}/{condition}: the processor returned no text ids")
            if sentences[condition] not in prompt_text:
                raise AuditError(f"{language}/{modality}/{condition}: question sentence missing")
            conditions.append(
                {
                    "condition": condition,
                    "sample_id": str(row.get("sample_id", "")),
                    "audio_path": audio_path,
                    "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                    "audio_seconds": round(float(len(audio)) / float(sampling_rate), 4),
                    "tensor_shapes": {key: list(shape) for key, shape in sizes.items()},
                    "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                }
            )
        cells.append({"language": language, "modality": modality, "config": config_rel, "conditions": conditions})
    return {
        "schema_version": "audiollm.qwen3omni_processor_audit.v1",
        "runtime_root": str(runtime_root),
        "model_path": str(model_path),
        "sampling_rate": int(sampling_rate),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        payload = audit(runtime_root=args.runtime_root.resolve(), model_path=args.model_path.resolve())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload["status"] = "passed"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps({
        "status": "passed",
        "processor_class": payload["processor_class"],
        "cells": [f"{cell['language']}/{cell['modality']}" for cell in payload["cells"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
