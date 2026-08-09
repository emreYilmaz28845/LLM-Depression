#!/usr/bin/env python3
"""Audit whether one 30-second audio chunk plus a full transcript fits Qwen2-Audio.

This is a CPU-only diagnostic. It loads the processor/tokenizer, never the model,
and does not modify dataset files. Detailed per-subject output is written under
``outputs/`` because it contains dataset identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoConfig, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    build_prompt_text,
    build_training_text,
    qwen2audio_audio_token_length,
    render_user_prompt_text,
)
from src.utils import internal_label_text_from_int


DEFAULT_MODEL = Path(
    "/home/emre/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
    "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
)
STANDARD_CONFIG: dict[str, Any] = {
    "prompt": {
        "system": "You are a psychologist analyzing speech and transcript information for depression screening.",
        "user_template": (
            "{audio_context_block}\n"
            "{transcript_block}Based on the {decision_basis}, determine whether the subject is "
            "{label_descriptor}.\n{label_instruction}"
        ),
    },
    "labels": {"label_vocab_version": "legacy_english_labels"},
    "data": {"use_audio": True, "use_text": True},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def natural_key(value: Any) -> tuple[Any, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def manifest_subjects(path: Path) -> set[str]:
    return {str(row["subject_id"]) for row in read_jsonl(path)}


def join_grouped(rows: Iterable[tuple[str, tuple[Any, ...], str]]) -> dict[str, str]:
    grouped: dict[str, list[tuple[tuple[Any, ...], str]]] = defaultdict(list)
    for subject_id, order, text in rows:
        text = str(text).strip()
        if text:
            grouped[str(subject_id)].append((order, text))
    return {
        subject_id: "\n".join(text for _, text in sorted(items, key=lambda item: item[0]))
        for subject_id, items in grouped.items()
    }


def load_daic(subjects: set[str], root: Path) -> tuple[dict[str, str], list[Path]]:
    records: list[tuple[str, tuple[Any, ...], str]] = []
    sources: list[Path] = []
    for subject_id in sorted(subjects, key=natural_key):
        path = root / f"{subject_id}_TRANSCRIPT.csv"
        if not path.exists():
            continue
        sources.append(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for index, row in enumerate(reader):
                if str(row.get("speaker", "")).strip().lower() == "participant":
                    records.append(
                        (
                            subject_id,
                            (float(row.get("start_time") or index), index),
                            str(row.get("value", "")),
                        )
                    )
    return join_grouped(records), sources


def load_d3tec(subjects: set[str], path: Path) -> dict[str, str]:
    return join_grouped(
        (
            str(row["subject_id"]),
            (natural_key(row.get("prompt_id", "")), natural_key(row.get("sample_id", ""))),
            str(row.get("transcript", "")),
        )
        for row in read_jsonl(path)
        if str(row.get("subject_id")) in subjects
    )


def load_androids(subjects: set[str], path: Path) -> dict[str, str]:
    return join_grouped(
        (
            str(row["subject_id"]),
            (
                natural_key(row.get("recording_id", "")),
                natural_key(row.get("turn_id", "")),
                natural_key(row.get("sample_id", "")),
            ),
            str(row.get("transcript", "")),
        )
        for row in read_jsonl(path)
        if str(row.get("subject_id")) in subjects
    )


def load_manifest_transcripts(path: Path) -> dict[str, str]:
    rows = read_jsonl(path)
    return join_grouped(
        (
            str(row["subject_id"]),
            (
                natural_key(row.get("question_id", "")),
                natural_key(row.get("audio_path", "")),
                natural_key(row.get("sample_id", "")),
            ),
            str(row.get("transcript", "")),
        )
        for row in rows
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def prompt_for(transcript: str, label: int) -> str:
    user_text = render_user_prompt_text(STANDARD_CONFIG, transcript, is_subject_bundle=False)
    prompt = build_prompt_text(
        STANDARD_CONFIG["prompt"]["system"],
        user_text,
        num_audios=1,
        use_audio=True,
        audio_placeholder=AUDIO_PLACEHOLDER,
    )
    return build_training_text(prompt, internal_label_text_from_int(STANDARD_CONFIG, label))


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def audit_dataset(
    name: str,
    transcripts: dict[str, str],
    expected_subjects: set[str],
    tokenizer: Any,
    audio_token_id: int,
    audio_tokens: int,
    context_limit: int,
    safety_margin: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing = sorted(expected_subjects - set(transcripts), key=natural_key)
    extra = sorted(set(transcripts) - expected_subjects, key=natural_key)
    details: list[dict[str, Any]] = []
    for subject_id in sorted(expected_subjects & set(transcripts), key=natural_key):
        transcript = transcripts[subject_id]
        candidates = []
        for label in (0, 1):
            token_ids = tokenizer(prompt_for(transcript, label), add_special_tokens=False)["input_ids"]
            placeholder_count = sum(int(token_id == audio_token_id) for token_id in token_ids)
            if placeholder_count != 1:
                raise RuntimeError(
                    f"{name}/{subject_id}: expected one <|AUDIO|> token, found {placeholder_count}"
                )
            effective_tokens = len(token_ids) - placeholder_count + audio_tokens
            candidates.append((effective_tokens, len(token_ids), label))
        effective_tokens, text_tokens_with_placeholder, worst_label = max(candidates)
        details.append(
            {
                "dataset": name,
                "subject_id": subject_id,
                "transcript_chars": len(transcript),
                "transcript_token_count": len(
                    tokenizer(transcript, add_special_tokens=False)["input_ids"]
                ),
                "text_tokens_with_audio_placeholder_and_label": text_tokens_with_placeholder,
                "audio_embedding_tokens_30sec": audio_tokens,
                "effective_multimodal_tokens": effective_tokens,
                "context_limit": context_limit,
                "remaining_tokens": context_limit - effective_tokens,
                "fits": effective_tokens <= context_limit,
                "fits_with_safety_margin": effective_tokens <= context_limit - safety_margin,
                "worst_case_label": worst_label,
            }
        )
    lengths = [row["effective_multimodal_tokens"] for row in details]
    transcript_lengths = [row["transcript_token_count"] for row in details]
    summary = {
        "dataset": name,
        "expected_subjects": len(expected_subjects),
        "audited_subjects": len(details),
        "missing_transcript_subjects": len(missing),
        "extra_transcript_subjects_ignored": len(extra),
        "context_limit": context_limit,
        "safety_margin": safety_margin,
        "audio_embedding_tokens_30sec": audio_tokens,
        "transcript_tokens_min": min(transcript_lengths) if transcript_lengths else None,
        "transcript_tokens_mean": statistics.mean(transcript_lengths) if transcript_lengths else None,
        "transcript_tokens_median": statistics.median(transcript_lengths)
        if transcript_lengths
        else None,
        "transcript_tokens_p95": percentile(transcript_lengths, 0.95)
        if transcript_lengths
        else None,
        "transcript_tokens_max": max(transcript_lengths) if transcript_lengths else None,
        "effective_tokens_min": min(lengths) if lengths else None,
        "effective_tokens_mean": statistics.mean(lengths) if lengths else None,
        "effective_tokens_median": statistics.median(lengths) if lengths else None,
        "effective_tokens_p95": percentile(lengths, 0.95) if lengths else None,
        "effective_tokens_max": max(lengths) if lengths else None,
        "subjects_over_limit": sum(not row["fits"] for row in details),
        "subjects_over_limit_pct": 100.0 * sum(not row["fits"] for row in details) / len(details)
        if details
        else None,
        "subjects_over_margin_limit": sum(not row["fits_with_safety_margin"] for row in details),
        "subjects_over_margin_limit_pct": 100.0
        * sum(not row["fits_with_safety_margin"] for row in details)
        / len(details)
        if details
        else None,
        "missing_subject_ids": missing,
    }
    return details, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/full_transcript_context_audit"
    )
    parser.add_argument("--safety-margin", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = {
        "D3TEC": PROJECT_ROOT / "outputs/manifests_d3tec/d3tec_manifest.jsonl",
        "Turkish": PROJECT_ROOT / "outputs/manifests_t17_qwen3asr/turkish_manifest.jsonl",
        "Androids": PROJECT_ROOT
        / "outputs/manifests_androids_interview/androids_interview_manifest.jsonl",
        "DAIC-WOZ": PROJECT_ROOT / "outputs/manifests/daic_manifest.jsonl",
        "CMDC": PROJECT_ROOT / "outputs/manifests/cmdc_manifest.jsonl",
    }
    subjects = {name: manifest_subjects(path) for name, path in manifests.items()}
    transcript_sources = {
        "D3TEC": Path(
            "/media/emre/Backup/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET/"
            "transcripts_qwen3_asr_spanish.jsonl"
        ),
        "Androids": Path(
            "/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus/"
            "interview_transcripts_qwen3_asr_italian.jsonl"
        ),
    }
    daic, daic_sources = load_daic(
        subjects["DAIC-WOZ"], Path("/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/unprocessed")
    )
    transcripts = {
        "D3TEC": load_d3tec(subjects["D3TEC"], transcript_sources["D3TEC"]),
        "Turkish": load_manifest_transcripts(manifests["Turkish"]),
        "Androids": load_androids(subjects["Androids"], transcript_sources["Androids"]),
        "DAIC-WOZ": daic,
        "CMDC": load_manifest_transcripts(manifests["CMDC"]),
    }

    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True)
    tokenizer = processor.tokenizer
    audio_token_id = int(tokenizer.convert_tokens_to_ids("<|AUDIO|>"))
    if tokenizer.unk_token_id is not None and audio_token_id == int(tokenizer.unk_token_id):
        raise RuntimeError("The selected processor does not recognize Qwen2-Audio's <|AUDIO|> token.")
    context_limit = int(config.text_config.max_position_embeddings)
    audio_tokens = qwen2audio_audio_token_length(3000)
    validation_text = prompt_for("Processor validation transcript.", 0)
    processed_validation = processor(
        text=validation_text,
        audio=[np.zeros(480_000, dtype=np.float32)],
        sampling_rate=16_000,
        return_tensors=None,
        padding=False,
    )
    validation_ids = np.asarray(processed_validation["input_ids"]).reshape(-1)
    expanded_audio_tokens = int(np.sum(validation_ids == audio_token_id))
    validation_mel_frames = int(
        np.asarray(processed_validation["feature_attention_mask"]).reshape(-1).sum()
    )
    if expanded_audio_tokens != audio_tokens or validation_mel_frames != 3000:
        raise RuntimeError(
            "Processor validation disagrees with the Qwen2-Audio length formula: "
            f"expanded_audio_tokens={expanded_audio_tokens}, mel_frames={validation_mel_frames}."
        )

    all_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in ("D3TEC", "Turkish", "Androids", "DAIC-WOZ", "CMDC"):
        detail, summary = audit_dataset(
            name,
            transcripts[name],
            subjects[name],
            tokenizer,
            audio_token_id,
            audio_tokens,
            context_limit,
            args.safety_margin,
        )
        all_details.extend(detail)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "subject_context_lengths.jsonl"
    with detail_path.open("w", encoding="utf-8") as handle:
        for row in all_details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.output_dir / "summary.json"
    provenance = {
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": sha256(args.model_path / "config.json"),
        "tokenizer_config_sha256": sha256(args.model_path / "tokenizer_config.json"),
        "context_limit": context_limit,
        "audio_assumption_seconds": 30.0,
        "audio_mel_frames": 3000,
        "audio_embedding_tokens": audio_tokens,
        "processor_validation": {
            "zero_audio_samples": 480000,
            "sampling_rate": 16000,
            "feature_attention_mask_frames": validation_mel_frames,
            "expanded_audio_placeholder_tokens": expanded_audio_tokens,
        },
        "prompt_policy": "one 30-second audio chunk plus full participant transcript",
        "prompt_template": STANDARD_CONFIG["prompt"],
        "manifest_sources": {name: str(path.resolve()) for name, path in manifests.items()},
        "transcript_sources": {
            "D3TEC": str(transcript_sources["D3TEC"]),
            "Turkish": str(manifests["Turkish"].resolve()),
            "Androids": str(transcript_sources["Androids"]),
            "DAIC-WOZ": f"{len(daic_sources)} *_TRANSCRIPT.csv files under unprocessed",
            "CMDC": str(manifests["CMDC"].resolve()),
        },
        "manifest_sha256": {name: sha256(path) for name, path in manifests.items()},
        "transcript_source_sha256": {
            "D3TEC": sha256(transcript_sources["D3TEC"]),
            "Turkish": sha256(manifests["Turkish"]),
            "Androids": sha256(transcript_sources["Androids"]),
            "DAIC-WOZ_combined": combined_sha256(daic_sources),
            "CMDC": sha256(manifests["CMDC"]),
        },
    }
    summary_path.write_text(
        json.dumps({"provenance": provenance, "datasets": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = args.output_dir / "summary.csv"
    public_fields = [key for key in summaries[0] if key != "missing_subject_ids"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in public_fields} for row in summaries)
    print(json.dumps({"summary": str(summary_path), "datasets": summaries}, indent=2))


if __name__ == "__main__":
    main()
