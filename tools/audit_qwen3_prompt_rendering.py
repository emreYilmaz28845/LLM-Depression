#!/usr/bin/env python3
"""Render the Turkish pooled prompts for the five default cells and audit them.

CPU only, no model weights. Two layers of checks:

* prompt content, for every cell: the condition sentence is the configured
  question-context sentence, the subject id and the subject's score never reach
  the rendered text, and the two answer labels are present;
* backend rendering, when ``--model-path`` points at the pinned Qwen3.8
  snapshot: the text-only cells are rendered through the real chat template
  (thinking disabled), the generation prompt and the training text must agree on
  the label boundary (the backend raises otherwise), each answer label must be a
  single token, and the prompt must fit the model context.

The pool manifests come from the task runtime built by
``scripts/build_turkish_pooled_manifest.py``.
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
from src.data.runtime import build_prompt_text, render_user_prompt_text
from src.utils import load_yaml_with_overrides, read_jsonl, resolve_label_config
from tools.qwen3_pooled_defaults import POOLED_TURKISH_ENGLISH, POOLED_TURKISH_NATIVE

CONDITIONS = ("pos_only_t17", "negative_only_t17")
AUDIO_MODALITIES = {"audio_only", "audio_text"}


class AuditError(RuntimeError):
    """Raised when a rendered prompt violates the contract."""


def cells() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for modality, rel in sorted(POOLED_TURKISH_NATIVE.items()):
        rows.append(("native", modality, rel))
    for modality, rel in sorted(POOLED_TURKISH_ENGLISH.items()):
        rows.append(("english", modality, rel))
    return rows


def manifest_rows(runtime_root: Path, language: str) -> list[dict[str, Any]]:
    manifest_dir = runtime_root / ("manifests_en" if language == "english" else "manifests") / "turkish"
    manifest_path = manifest_dir / "turkish_manifest.jsonl"
    if not manifest_path.is_file():
        raise AuditError(f"missing pooled {language} manifest: {manifest_path}")
    return read_jsonl(manifest_path)


def sample_rows(runtime_root: Path, language: str) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for row in manifest_rows(runtime_root, language):
        condition = str(row["dataset_variant"])
        samples.setdefault(condition, row)
    missing = [condition for condition in CONDITIONS if condition not in samples]
    if missing:
        raise AuditError(f"{language}: manifest has no rows for {missing}")
    return samples


def audit_cell(
    language: str,
    modality: str,
    config_rel: str,
    samples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config = load_yaml_with_overrides(PROJECT_ROOT / config_rel, [])
    system_prompt = resolve_system_prompt(config)
    sentences = resolve_question_context_sentences(config)
    labels_cfg = resolve_label_config(config)
    use_audio = modality in AUDIO_MODALITIES
    records = []
    for condition in CONDITIONS:
        row = samples[condition]
        transcript = str(row.get("transcript", ""))
        subject_id = str(row["subject_id"])
        score = str(float(row["score"]))
        user_text = render_user_prompt_text(
            config,
            transcript,
            is_subject_bundle=use_audio,
            question_condition=condition,
        )
        if sentences[condition] not in user_text:
            raise AuditError(
                f"{language}/{modality}/{condition}: the configured question-context sentence "
                "is missing from the rendered user text"
            )
        other = "negative_only_t17" if condition == "pos_only_t17" else "pos_only_t17"
        if sentences[other] in user_text:
            raise AuditError(
                f"{language}/{modality}/{condition}: the other condition's sentence leaked in"
            )
        scaffolding = f"{system_prompt}\n{user_text.replace(transcript, '<transcript>')}"
        if subject_id and subject_id in scaffolding:
            raise AuditError(f"{language}/{modality}/{condition}: subject id leaked into the prompt")
        if score in scaffolding:
            raise AuditError(f"{language}/{modality}/{condition}: the subject score leaked into the prompt")
        for label in (labels_cfg["internal_positive_label"], labels_cfg["internal_negative_label"]):
            if label not in user_text:
                raise AuditError(f"{language}/{modality}/{condition}: answer label {label!r} is missing")
        prompt_text = build_prompt_text(
            system_prompt, user_text, 1 if use_audio else 0, use_audio
        )
        records.append(
            {
                "condition": condition,
                "sample_id": str(row.get("sample_id", "")),
                "subject_id": subject_id,
                "label": int(row["label"]),
                "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "question_context": sentences[condition],
            }
        )
    return {
        "config": config_rel,
        "language": language,
        "modality": modality,
        "backend": config.get("model_backend"),
        "records": records,
        "labels": {
            "positive": labels_cfg["internal_positive_label"],
            "negative": labels_cfg["internal_negative_label"],
        },
    }


def _qwen38_context_limit(model_path: Path, tokenizer) -> int:
    """Declared text context length, or 0 when the snapshot does not declare a usable one."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    candidates = [config, getattr(config, "text_config", None)]
    for candidate in candidates:
        value = getattr(candidate, "max_position_embeddings", None)
        if isinstance(value, int) and 0 < value < 10**9:
            return int(value)
    value = int(getattr(tokenizer, "model_max_length", 0) or 0)
    return value if 0 < value < 10**9 else 0


def audit_qwen38_rendering(
    model_path: Path, records: list[dict[str, Any]], samples: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    from src.model.qwen38_lora import (
        load_processor,
        render_qwen38_prompt,
        render_qwen38_training_text,
    )

    processor = load_processor(str(model_path))
    tokenizer = processor.tokenizer
    context_limit = _qwen38_context_limit(model_path, tokenizer)
    audits = []
    for record in records:
        config = load_yaml_with_overrides(PROJECT_ROOT / record["config"], [])
        labels_cfg = resolve_label_config(config)
        condition = record["records"][0]["condition"]
        row = samples[condition]
        system_text = resolve_system_prompt(config)
        user_text = render_user_prompt_text(
            config,
            str(row.get("transcript", "")),
            is_subject_bundle=False,
            question_condition=condition,
        )
        prompt_text = render_qwen38_prompt(processor, system_text, user_text)
        label_tokens = {}
        for name in ("internal_positive_label", "internal_negative_label"):
            label_text = labels_cfg[name]
            token_ids = tokenizer.encode(label_text, add_special_tokens=False)
            if len(token_ids) != 1:
                raise AuditError(
                    f"{record['config']}: answer label {label_text!r} is {len(token_ids)} tokens, "
                    "not one"
                )
            label_tokens[label_text] = token_ids[0]
        training_text = render_qwen38_training_text(
            processor, system_text, user_text, labels_cfg["internal_negative_label"]
        )
        if not training_text.startswith(prompt_text):
            raise AuditError(
                f"{record['config']}: the Qwen3.8 generation prompt and training text disagree"
            )
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if context_limit and prompt_tokens >= context_limit:
            raise AuditError(f"{record['config']}: prompt length {prompt_tokens} exceeds {context_limit}")
        audits.append(
            {
                "config": record["config"],
                "condition": condition,
                "prompt_tokens": prompt_tokens,
                "context_limit": context_limit,
                "label_tokens": label_tokens,
            }
        )
    return {"model_path": str(model_path), "cells": audits}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--model-path", type=Path,
                        help="pinned Qwen3.8 snapshot for tokenizer/render checks (no weights)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    runtime_root = args.runtime_root.resolve()
    audits: list[dict[str, Any]] = []
    try:
        samples_by_language = {
            "native": sample_rows(runtime_root, "native"),
            "english": sample_rows(runtime_root, "english"),
        }
        text_only_records: list[dict[str, Any]] = []
        for language, modality, config_rel in cells():
            record = audit_cell(language, modality, config_rel, samples_by_language[language])
            audits.append(record)
            if modality == "text_only":
                text_only_records.append(record)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "schema_version": "audiollm.qwen3_prompt_rendering_audit.v1",
        "runtime_root": str(runtime_root),
        "cells": audits,
        "qwen38_rendering": None,
    }
    if args.model_path is not None:
        try:
            payload["qwen38_rendering"] = audit_qwen38_rendering(
                args.model_path.resolve(), text_only_records, samples_by_language["native"]
            )
            # The English text-only cell must render with the English transcripts.
            payload["qwen38_rendering"]["english_text_only"] = audit_qwen38_rendering(
                args.model_path.resolve(),
                [record for record in text_only_records if record["language"] == "english"],
                samples_by_language["english"],
            )["cells"]
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
        "cells": [f"{record['language']}/{record['modality']}" for record in audits],
        "qwen38_rendering": payload["qwen38_rendering"] is not None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
