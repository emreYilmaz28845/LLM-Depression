#!/usr/bin/env python3
"""Audit the eight Qwen3 English-transcript cells' render and notice contract.

Weight-free and CPU only. For every generated English cell the audit checks:

* the cell differs from its native counterpart only inside the derivation
  allowlist (transcript overlay, versioned translation notice, manifest/split/
  run roots and the recipe marker), so the original audio contract is inherited
  unchanged;
* the transcript overlay keeps the accepted English policy (variant english,
  minimum_status automatic_low, require_complete true, include_failed false)
  and the documented cache contract (an accepted.jsonl for the translated
  datasets, the inert pooled placeholder for the prebuilt pooled cell);
* the versioned translation notice is present exactly once, immediately before
  the transcript block, with the exact sentence for the modality: text-only
  never claims audio, audio+text states that the audio remains in the original
  language;
* the native counterpart carries no notice and its rendered prompt never claims
  a translation;
* training-inner and final-eval rendering are identical for the same cell.

It records current config and prompt hashes and the native diff, so the result
can be compared with older readiness audits without trusting stale hashes.
No training and no submission happen here; the only files written are the
``--output`` JSON and a temporary silence window under the lane output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_qwen3_english_configs as generator  # noqa: E402
from src.data.prompt_context import (  # noqa: E402
    TRANSLATION_NOTICE_SENTENCES,
    TRANSLATION_NOTICE_VERSION,
    prompt_context_record,
    resolve_translation_notice,
)
from src.data.runtime import build_examples, render_user_prompt_text, resolve_audio_placeholder  # noqa: E402
from src.experiment_tracking.canonical import sha256_file  # noqa: E402
from src.utils import INPUT_MODALITY_AUDIO_TEXT, INPUT_MODALITY_TEXT_ONLY, load_yaml_with_overrides  # noqa: E402

SCHEMA_VERSION = "audiollm.qwen3_english_rendering_audit.v1"
TRANSCRIPT_MARKER = "The transcript of the subject's speech is:"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/qwen3_multiseed_english_20261002/english_render_audit.json"
)
DEFAULT_TMP_DIR = PROJECT_ROOT / "outputs/qwen3_multiseed_english_20261002/render_audit_tmp"
POOLED_DATASET_VARIANT = "pooled_t17"


class AuditError(RuntimeError):
    """Raised when the English render/notice contract is violated."""


def _expected_notice(modality: str) -> str:
    key = INPUT_MODALITY_TEXT_ONLY if modality == "text_only" else INPUT_MODALITY_AUDIO_TEXT
    return TRANSLATION_NOTICE_SENTENCES[TRANSLATION_NOTICE_VERSION][key]


def _silence_window(tmp_dir: Path) -> str:
    import numpy as np
    import soundfile as sf

    path = tmp_dir / "silence_30s.wav"
    if not path.is_file():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.zeros(30 * 16000, dtype="float32"), 16000)
    return str(path)


def _row(dataset: str, variant: str | None, audio_path: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "sample_id": f"{dataset}-s1",
        "subject_id": "s1",
        "label": 1,
        "label_text": "Depressed",
        "transcript": "a translated transcript",
        "audio_path": audio_path,
        "transcript_variant": "english",
        "language": "en",
    }
    if variant is not None:
        row["dataset_variant"] = variant
    return row


def audit_cell(
    cell: tuple, *, tmp_dir: Path, tokenizer: Any | None = None
) -> dict[str, Any]:
    slug, source_name, target_name, _manifest_dir, _run_dir, modality, folds, separate_eval, cache_dir = cell
    config_path = generator.PROJECT_ROOT / "configs/main" / target_name
    native_path = generator.PROJECT_ROOT / "configs/main" / source_name
    config = load_yaml_with_overrides(config_path, [])
    native_config = load_yaml_with_overrides(native_path, [])

    changed = generator.diff_paths(
        yaml.safe_load(native_path.read_text(encoding="utf-8")),
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    disallowed = [path for path in changed if path not in generator.ALLOWED_DIFF_PATHS]

    transcripts = config.get("transcripts") or {}
    policy_expected = dict(generator.TRANSCRIPTS_POLICY)
    if cache_dir is None:
        policy_expected["cache_path"] = generator.POOLED_TRANSCRIPTS_CACHE_PLACEHOLDER
    transcript_policy_ok = all(transcripts.get(key) == value for key, value in policy_expected.items())

    notice = resolve_translation_notice(config)
    native_notice = resolve_translation_notice(native_config)
    expected_notice = _expected_notice(modality)

    audio_path = _silence_window(tmp_dir)
    if str(config.get("dataset_variant", "")) == POOLED_DATASET_VARIANT:
        rows = [
            _row(str(config["dataset"]), condition, audio_path)
            for condition in ("pos_only_t17", "negative_only_t17")
        ]
    else:
        rows = [_row(str(config["dataset"]), None, audio_path)]
    train = build_examples(rows, config, "train_inner")
    evaluation = build_examples(rows, config, "final_eval")

    rendered = render_user_prompt_text(
        config, "a translated transcript", question_condition="pos_only_t17"
    )
    native_rendered = render_user_prompt_text(
        native_config, "a native transcript", question_condition="pos_only_t17"
    )
    record = prompt_context_record(config)
    native_record = prompt_context_record(native_config)
    train_prompt = train[0]["prompt_text"]
    eval_prompt = evaluation[0]["prompt_text"]

    tokenizer_audit = None
    if tokenizer is not None and str(config.get("model_backend")) == "qwen38":
        from src.model.qwen38_lora import prepare_qwen38_examples

        class _Adapter:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
                return self._inner.apply_chat_template(*args, **kwargs)

        prepared = prepare_qwen38_examples(train, config, _Adapter(tokenizer))
        prompt_ids = tokenizer.encode(prepared[0]["prompt_text"], add_special_tokens=False)
        notice_ids = tokenizer.encode(notice or "", add_special_tokens=False)
        tokenizer_audit = {
            "prompt_tokens": len(prompt_ids),
            "notice_tokens": len(notice_ids),
            "notice_survives_tokenization": bool(notice_ids)
            and any(
                prompt_ids[index : index + len(notice_ids)] == notice_ids
                for index in range(max(len(prompt_ids) - len(notice_ids) + 1, 0))
            ),
            "prompt_within_8192": len(prompt_ids) <= 8192,
        }
    elif tokenizer is not None:
        tokenizer_audit = {
            "status": "deferred: the Qwen3-Omni processor audit runs in the Qwen3-Omni environment"
        }

    notice_occurrences = rendered.count(notice or "")
    notice_before_transcript = bool(notice) and (
        rendered.rindex(notice) < rendered.index(TRANSCRIPT_MARKER)
    )
    audio_placeholder = resolve_audio_placeholder(config)
    audio_placeholder_count = train_prompt.count(audio_placeholder) if audio_placeholder else 0
    expected_audio_placeholders = 1 if modality in ("audio_only", "audio_text") else 0
    failures = []
    if disallowed:
        failures.append(f"diff outside the English allowlist: {disallowed}")
    if not transcript_policy_ok:
        failures.append("transcript policy is not the accepted English policy")
    if notice != expected_notice:
        failures.append("translation notice text does not match the modality sentence")
    if notice_occurrences != 1:
        failures.append(f"translation notice occurs {notice_occurrences} times in the rendered prompt")
    if notice and not notice_before_transcript:
        failures.append("translation notice is not directly before the transcript block")
    if native_notice is not None:
        failures.append("the native counterpart carries a translation notice")
    if "English translation" in native_rendered:
        failures.append("the native rendered prompt claims a translation")
    if train_prompt != eval_prompt:
        failures.append("training-inner and final-eval prompts differ")
    if audio_placeholder_count != expected_audio_placeholders:
        failures.append(
            f"audio placeholder count {audio_placeholder_count} != {expected_audio_placeholders}"
        )
    if tokenizer_audit and tokenizer_audit.get("prompt_tokens") is not None:
        if not tokenizer_audit.get("notice_survives_tokenization"):
            failures.append("the notice does not survive tokenization")
        if not tokenizer_audit.get("prompt_within_8192"):
            failures.append("the rendered prompt exceeds 8192 tokens")

    return {
        "cell_id": slug,
        "config": f"configs/main/{target_name}",
        "config_sha256": sha256_file(config_path),
        "native_config": f"configs/main/{source_name}",
        "native_config_sha256": sha256_file(native_path),
        "modality": modality,
        "dataset": str(config["dataset"]),
        "backend": config.get("model_backend"),
        "folds": list(folds),
        "separate_eval": separate_eval,
        "changed_paths_vs_native": changed,
        "allowed_diff_paths": sorted(generator.ALLOWED_DIFF_PATHS),
        "transcript_policy": dict(transcripts),
        "notice": {
            "version": record["translation_notice_version"],
            "expected_version": TRANSLATION_NOTICE_VERSION,
            "text": notice,
            "sha256": record["translation_notice_sha256"],
            "occurrences_in_rendered_prompt": notice_occurrences,
            "offset_before_transcript_block": notice_before_transcript,
        },
        "system_prompt_sha256": record["system_prompt_sha256"],
        "native": {
            "notice": native_record["translation_notice"],
            "system_prompt_sha256": native_record["system_prompt_sha256"],
            "rendered_carries_notice": "English translation" in native_rendered,
        },
        "rendered_user_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "train_prompt_sha256": hashlib.sha256(train_prompt.encode("utf-8")).hexdigest(),
        "eval_prompt_sha256": hashlib.sha256(eval_prompt.encode("utf-8")).hexdigest(),
        "train_equals_eval_prompt": train_prompt == eval_prompt,
        "audio_placeholder_count": audio_placeholder_count,
        "tokenizer_audit": tokenizer_audit,
        "failures": failures,
    }


def run_audit(*, tmp_dir: Path, tokenizer: Any | None = None) -> dict[str, Any]:
    cells = [
        audit_cell(cell, tmp_dir=tmp_dir, tokenizer=tokenizer) for cell in generator.CELLS
    ]
    failures = [
        f"{cell['cell_id']}: {failure}" for cell in cells for failure in cell["failures"]
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prompt_context_version": generator.PROMPT_CONTEXT_VERSION,
        "translation_notice_version": TRANSLATION_NOTICE_VERSION,
        "cells": cells,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument(
        "--qwen38-model-path",
        type=Path,
        default=None,
        help="optional local Qwen3.8 snapshot for the weight-free tokenizer check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = None
    if args.qwen38_model_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.qwen38_model_path), local_files_only=True
        )
    payload = run_audit(tmp_dir=args.tmp_dir, tokenizer=tokenizer)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cells": [f"{cell['cell_id']}" for cell in payload["cells"]],
                "failures": payload["failures"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
