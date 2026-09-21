#!/usr/bin/env python3
"""Remeasure prompt-context length for the final prompt-context recipe.

The older ``full_transcript_context_audit.md`` measured the short baseline
prompt with a real Qwen2-Audio processor. This audit reruns the measurement for
the final prompt-context texts, which are longer, and reports the worst case per
subject: one 30-second audio window plus the subject's full transcript, with the
longer of the two target-label variants.

Text tokens are counted with a Qwen-family tokenizer when one is supplied. The
Qwen2-Audio processor is not available on the local workstation, so the audio
expansion uses the repository's own length formulas:

* Qwen2-Audio: ``qwen2audio_audio_token_length(3000)`` (verified against the real
  processor in the earlier audit);
* Gemma 4: ``expected_gemma4_audio_tokens(480000)``.

Every number produced without the real processor is labelled so it cannot be
mistaken for a processor measurement; the smoke runs record the real
``audio_budget_audit_*.json`` from the training jobs.

The diagnostic reads shared manifests read-only and never modifies them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.data.runtime import (  # noqa: E402
    _harmonized_subject_transcripts,
    build_prompt_text,
    build_training_text,
    qwen2audio_audio_token_length,
    render_user_prompt_text,
    resolve_audio_placeholder,
)
from src.model.gemma4_io import expected_gemma4_audio_tokens  # noqa: E402
from src.utils import internal_label_text_from_int, read_jsonl, resolve_input_modality  # noqa: E402

AUDIO_TOKEN = "<|AUDIO|>"
QWEN2AUDIO_CONTEXT_LIMIT = 8192
SAFETY_MARGIN = 128
DATASET_MANIFEST_NAMES = {
    "androids_interview": "androids_interview_manifest.jsonl",
    "d3tec": "d3tec_manifest.jsonl",
    "daic": "daic_manifest.jsonl",
    "cmdc": "cmdc_manifest.jsonl",
    "turkish": "turkish_manifest.jsonl",
}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_for(matrix: dict[str, Any], dataset: str, modality: str, backbone: str) -> dict[str, Any]:
    for cell in matrix["standalone"]:
        if (
            cell["dataset"] == dataset
            and cell["modality"] == modality
            and cell["backbone"] == backbone
        ):
            return yaml.safe_load((PROJECT_ROOT / cell["config"]).read_text(encoding="utf-8"))
    raise KeyError(f"no matrix cell for {dataset}/{modality}/{backbone}")


def _subject_transcripts(rows: list[dict[str, Any]], dataset: str) -> dict[str, str]:
    if dataset == "daic":
        transcripts: dict[str, str] = {}
        for row in rows:
            transcripts[str(row["subject_id"])] = str(row["full_participant_transcript"])
        return transcripts
    return _harmonized_subject_transcripts(rows, dataset)


def _turkish_condition_transcripts(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        condition = str(row.get("dataset_variant", "")).strip()
        grouped.setdefault(condition, []).append(row)
    return {
        condition: _harmonized_subject_transcripts(condition_rows, "turkish")
        for condition, condition_rows in grouped.items()
    }


def _render_worst_case(config: dict[str, Any], transcript: str, question_condition: str | None) -> dict[str, Any]:
    modality = resolve_input_modality(config)
    user_text = render_user_prompt_text(
        config, transcript, question_condition=question_condition
    )
    prompt_text = build_prompt_text(
        system_prompt=config["prompt"]["system"],
        user_text=user_text,
        num_audios=1 if modality != "text_only" else 0,
        use_audio=modality != "text_only",
        audio_placeholder=resolve_audio_placeholder(config),
    )
    return {"prompt_text": prompt_text, "user_text": user_text, "modality": modality}


def _training_text(config: dict[str, Any], prompt_text: str, label: int) -> str:
    return build_training_text(prompt_text, internal_label_text_from_int(config, label))


def _text_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _audio_tokens(backbone: str) -> int:
    if backbone == "gemma4":
        return int(expected_gemma4_audio_tokens(480_000))
    return int(qwen2audio_audio_token_length(3000))


def audit_dataset(
    dataset: str,
    *,
    matrix: dict[str, Any],
    manifest_root: Path,
    tokenizer,
    manifest_override: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_override or (
        manifest_root
        / "manifests_harmonized"
        / _dataset_dir(dataset)
        / DATASET_MANIFEST_NAMES["turkish" if dataset == "turkish" else dataset]
    )
    rows = read_jsonl(manifest_path)
    result: dict[str, Any] = {
        "dataset": dataset,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "manifest_rows": len(rows),
    }
    token_assumptions = {
        backbone: {
            "audio_tokens_per_window": _audio_tokens(backbone),
            "source": "formula" if backbone != "gemma4" else "expected_gemma4_audio_tokens",
        }
        for backbone in ("qwen", "gemma4")
    }
    result["audio_token_assumptions"] = token_assumptions

    if dataset == "turkish":
        conditions = _turkish_condition_transcripts(rows)
        per_condition: dict[str, Any] = {}
        for condition, transcripts in sorted(conditions.items()):
            entry: dict[str, Any] = {}
            for backbone in ("qwen", "gemma4"):
                config = _config_for(matrix, dataset, "text_only", backbone)
                effective: list[float] = []
                text_tokens: list[float] = []
                longest = {"subject_id": None, "tokens": -1}
                for subject_id, transcript in transcripts.items():
                    worst = 0
                    for label in (0, 1):
                        prompt = _render_worst_case(config, transcript, condition)
                        payload = _training_text(config, prompt["prompt_text"], label)
                        tokens = _text_tokens(tokenizer, payload)
                        worst = max(worst, tokens)
                    effective.append(worst)
                    text_tokens.append(_text_tokens(tokenizer, transcript))
                    if worst > longest["tokens"]:
                        longest = {"subject_id": subject_id, "tokens": worst}
                entry[backbone] = {
                    "subjects": len(transcripts),
                    "text_tokens": _stats(text_tokens),
                    "effective_sequence_tokens": _stats(effective),
                    "longest_subject": longest,
                    "context_limit": QWEN2AUDIO_CONTEXT_LIMIT,
                    "over_limit": sum(1 for value in effective if value > QWEN2AUDIO_CONTEXT_LIMIT),
                    "over_margin_limit": sum(
                        1
                        for value in effective
                        if value > QWEN2AUDIO_CONTEXT_LIMIT - SAFETY_MARGIN
                    ),
                }
            per_condition[condition] = entry
        result["conditions"] = per_condition
        result["subjects"] = len({str(row["subject_id"]) for row in rows})
        return result

    transcripts = _subject_transcripts(rows, dataset)
    for modality in ("audio_text", "text_only"):
        entry: dict[str, Any] = {}
        for backbone in ("qwen", "gemma4"):
            config = _config_for(matrix, dataset, modality, backbone)
            audio_tokens = _audio_tokens(backbone)
            effective: list[float] = []
            text_tokens: list[float] = []
            longest = {"subject_id": None, "tokens": -1}
            for subject_id, transcript in transcripts.items():
                worst = 0
                for label in (0, 1):
                    prompt = _render_worst_case(config, transcript, None)
                    payload = _training_text(config, prompt["prompt_text"], label)
                    tokens = _text_tokens(tokenizer, payload)
                    placeholder = payload.count(AUDIO_TOKEN)
                    expected_placeholder = 1 if modality != "text_only" else 0
                    if placeholder != expected_placeholder:
                        raise RuntimeError(
                            f"{dataset}/{modality}/{backbone}/{subject_id}: "
                            f"{placeholder} audio placeholders, expected {expected_placeholder}"
                        )
                    tokens = tokens - placeholder + (audio_tokens if placeholder else 0)
                    worst = max(worst, tokens)
                effective.append(worst)
                text_tokens.append(_text_tokens(tokenizer, transcript))
                if worst > longest["tokens"]:
                    longest = {"subject_id": subject_id, "tokens": worst}
            entry[backbone] = {
                "subjects": len(transcripts),
                "text_tokens": _stats(text_tokens),
                "effective_sequence_tokens": _stats(effective),
                "longest_subject": longest,
                "context_limit": QWEN2AUDIO_CONTEXT_LIMIT,
                "over_limit": sum(1 for value in effective if value > QWEN2AUDIO_CONTEXT_LIMIT),
                "over_margin_limit": sum(
                    1 for value in effective if value > QWEN2AUDIO_CONTEXT_LIMIT - SAFETY_MARGIN
                ),
            }
        result[modality] = entry
    result["subjects"] = len(transcripts)
    return result


def _dataset_dir(dataset: str) -> str:
    return "androids" if dataset == "androids_interview" else dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/promptcontext/matrix.yaml",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="evidence root that contains manifests_harmonized/ (read-only)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Qwen-family tokenizer directory used as the text-token proxy",
    )
    parser.add_argument(
        "--turkish-manifest",
        type=Path,
        default=None,
        help=(
            "pooled Turkish manifest to audit instead of "
            "manifests_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-samples",
        type=Path,
        default=None,
        help="also write one rendered prompt sample per cell (synthetic transcript)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    if args.tokenizer is None:
        print("ERROR: --tokenizer is required for text-token counting", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    datasets = ["androids_interview", "d3tec", "daic", "cmdc", "turkish"]
    report = {
        "matrix": str(args.matrix),
        "recipe_id": matrix["recipe_id"],
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_proxy": (
            "Qwen2-family text tokenizer; the Qwen2-Audio and Gemma processors are not "
            "installed locally, so audio expansion uses the repository formulas and every "
            "Gemma text count is a proxy that the MN5 smoke re-measures"
        ),
        "context_limit": QWEN2AUDIO_CONTEXT_LIMIT,
        "context_limit_source": "Qwen2-Audio text_config.max_position_embeddings (full_transcript_context_audit.md)",
        "safety_margin": SAFETY_MARGIN,
        "datasets": {},
    }
    for dataset in datasets:
        report["datasets"][dataset] = audit_dataset(
            dataset,
            matrix=matrix,
            manifest_root=args.manifest_root,
            tokenizer=tokenizer,
            manifest_override=args.turkish_manifest if dataset == "turkish" else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    previous = args.output.read_text(encoding="utf-8") if args.output.is_file() else None
    if previous is not None and previous != payload:
        print(f"ERROR: refusing to overwrite a differing audit: {args.output}", file=sys.stderr)
        return 1
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(args.output), "max_effective_tokens": _max_tokens(report)}, indent=2))
    if args.prompt_samples:
        write_prompt_samples(matrix, args.prompt_samples)
    return 0


def _max_tokens(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset, entry in report["datasets"].items():
        if dataset == "turkish":
            summary[dataset] = {
                condition: {
                    backbone: values["effective_sequence_tokens"]["max"]
                    for backbone, values in backbones.items()
                }
                for condition, backbones in entry["conditions"].items()
            }
            continue
        summary[dataset] = {
            modality: {
                backbone: values["effective_sequence_tokens"]["max"]
                for backbone, values in backbones.items()
                if isinstance(values, dict) and "effective_sequence_tokens" in values
            }
            for modality, backbones in entry.items()
            if modality in ("audio_text", "text_only")
        }
    return summary


def prompt_samples(matrix: dict[str, Any], transcript: str) -> list[dict[str, Any]]:
    """Render one representative prompt per cell with a synthetic transcript."""
    samples: list[dict[str, Any]] = []
    for cell in matrix["standalone"]:
        config = yaml.safe_load((PROJECT_ROOT / cell["config"]).read_text(encoding="utf-8"))
        conditions = (
            ("pos_only_t17", "negative_only_t17")
            if cell["dataset"] == "turkish"
            else (None,)
        )
        for condition in conditions:
            rendered = _render_worst_case(config, transcript, condition)
            samples.append(
                {
                    "dataset": cell["dataset"],
                    "modality": cell["modality"],
                    "backbone": cell["backbone"],
                    "config": cell["config"],
                    "question_condition": condition,
                    "system": config["prompt"]["system"],
                    "user": rendered["user_text"],
                    "label_instruction": "Answer with exactly one label: Depressed or Non-depressed.",
                }
            )
    return samples


def write_prompt_samples(matrix: dict[str, Any], path: Path) -> None:
    transcript = "[synthetic example transcript; no subject data]"
    samples = prompt_samples(matrix, transcript)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"transcript": transcript, "samples": samples}, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    previous = path.read_text(encoding="utf-8") if path.is_file() else None
    if previous is not None and previous != payload:
        raise SystemExit(f"ERROR: refusing to overwrite differing prompt samples: {path}")
    path.write_text(payload, encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    lines = [
        "# Prompt-context rendered prompt samples",
        "",
        f"Synthetic transcript used for every sample: `{transcript}`",
        "",
    ]
    for sample in samples:
        condition = f" ({sample['question_condition']})" if sample["question_condition"] else ""
        lines += [
            f"## {sample['dataset']} / {sample['modality']} / {sample['backbone']}{condition}",
            "",
            "```text",
            sample["system"],
            "",
            "--- user ---",
            sample["user"],
            "```",
            "",
        ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"prompt samples written: {path} and {markdown_path} ({len(samples)} samples)")


if __name__ == "__main__":
    raise SystemExit(main())
