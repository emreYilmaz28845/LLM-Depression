"""Label-free translation-unit export from canonical native manifests.

Reads the built native manifest JSONL for CMDC, Turkish, D3TEC, and ANDROIDS
Interview and emits a common unit schema. Never exports depression labels,
scores, folds, diagnoses, or class metadata. Overlength units are split at
sentence boundaries with stable part identities; context is capped
deterministically, never truncated mid-sentence.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    read_jsonl,
    resolve_project_path,
    sha256_text,
    write_jsonl,
)


LOGGER = get_logger(__name__)

TRANSLATION_DATASETS = ("cmdc", "turkish", "d3tec", "androids_interview")

TARGET_LANGUAGE = "en"

SOURCE_LANGUAGE_BY_DATASET = {
    "cmdc": "zh",
    "turkish": "tr",
    "d3tec": "es",
    "androids_interview": "it",
}

# (manifest field, scope) pairs in deterministic emission order.
DATASET_UNIT_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "cmdc": (("transcript", "response"),),
    "turkish": (("transcript", "audio_chunk"),),
    "d3tec": (
        ("full_response_transcript", "full_response"),
        ("segment_transcript", "segment"),
    ),
    "androids_interview": (
        ("full_turn_transcript", "full_turn"),
        ("segment_transcript", "segment"),
    ),
}

# Fields allowed to flow from the native manifest into a translation unit.
_SAFE_ROW_FIELDS = frozenset(
    {
        "dataset",
        "subject_id",
        "sample_id",
        "response_id",
        "turn_key",
        "transcript",
        "segment_transcript",
        "full_response_transcript",
        "full_turn_transcript",
        "chunk_id",
        "segment_index",
        "window_index",
    }
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?；;…\n])\s*|(?<=\.)(?=\s+[A-Z\"'“«(])|(?<=\.)(?=$)"
)


def split_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    return pieces or ([text.strip()] if text.strip() else [])


def split_source_for_budget(
    text: str,
    max_chars: int,
    max_tokens: int | None = None,
    tokenize: Callable[[str], int] | None = None,
) -> list[str]:
    if not text.strip():
        return [""]
    if tokenize is not None and max_tokens is not None:
        if tokenize(text) <= max_tokens:
            return [text]
    elif len(text) <= max_chars:
        return [text]
    sentences = split_sentences(text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if tokenize is not None and max_tokens is not None:
            candidate_len = tokenize(sentence)
            if candidate_len > max_tokens:
                raise ValueError(
                    "Single sentence exceeds the translation budget and cannot be "
                    f"split further ({candidate_len} tokens). Refusing to truncate."
                )
            if current and current_len + candidate_len > max_tokens:
                parts.append(" ".join(current).strip())
                current = [sentence]
                current_len = candidate_len
            else:
                current.append(sentence)
                current_len += candidate_len
        else:
            if sentence_len > max_chars:
                raise ValueError(
                    "Single sentence exceeds the translation budget and cannot be "
                    f"split further ({sentence_len} chars). Refusing to truncate."
                )
            if current and current_len + sentence_len > max_chars:
                parts.append(" ".join(current).strip())
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += sentence_len
    if current:
        parts.append(" ".join(current).strip())
    return parts or [text]


def _sha256_join(parts: list[str]) -> str:
    return sha256_text("\n".join(parts))


def unit_rows_for_dataset(
    manifest_rows: list[dict[str, Any]],
    dataset: str,
    *,
    max_source_tokens: int = 12000,
    max_source_chars: int | None = None,
    max_context_chars: int = 6000,
    tokenize: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    if dataset not in TRANSLATION_DATASETS:
        raise ValueError(f"Unsupported translation dataset: {dataset!r}")
    field_specs = DATASET_UNIT_FIELDS[dataset]
    source_language = SOURCE_LANGUAGE_BY_DATASET[dataset]

    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        rows_by_subject.setdefault(str(row["subject_id"]), []).append(row)

    def ordered_subject_rows(subject_id: str) -> list[dict[str, Any]]:
        rows = rows_by_subject.get(subject_id, [])
        if dataset == "turkish":
            def turkish_key(row: dict[str, Any]) -> tuple[int, str]:
                chunk_id = str(row.get("chunk_id", "")).strip()
                return (int(chunk_id) if chunk_id.isdigit() else 10**9, str(row["sample_id"]))
            return sorted(rows, key=turkish_key)
        if dataset == "d3tec":
            return sorted(
                rows, key=lambda row: (int(row.get("prompt_id", 0)), int(row.get("segment_index", 0)))
            )
        if dataset == "androids_interview":
            return sorted(
                rows, key=lambda row: (int(row.get("turn_id", 0)), int(row.get("window_index", 0)))
            )
        return sorted(rows, key=lambda row: str(row["sample_id"]))

    def unit_id_for(row: dict[str, Any], field: str) -> str:
        if field in {"full_response_transcript", "full_turn_transcript"}:
            return str(row["response_id"])
        return str(row["sample_id"])

    def context_for(row: dict[str, Any], field: str, subject_rows: list[dict[str, Any]]) -> tuple[str, str]:
        if field in {"full_response_transcript", "full_turn_transcript"}:
            return "", ""
        if dataset == "d3tec":
            return str(row.get("response_id", "")), str(row.get("full_response_transcript", ""))
        if dataset == "androids_interview":
            return str(row.get("response_id", "")), str(row.get("full_turn_transcript", ""))
        index = subject_rows.index(row) if row in subject_rows else 0
        neighbors = subject_rows[max(0, index - 2) : index]
        context_text = "\n\n".join(str(neighbor.get("transcript", "")).strip() for neighbor in neighbors)
        if len(context_text) > max_context_chars:
            context_text = context_text[-max_context_chars:]
        context_id = f"{row['subject_id']}_context" if context_text else ""
        return context_id, context_text

    units: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    seen_full: set[str] = set()
    for subject_id in sorted(rows_by_subject):
        subject_rows = ordered_subject_rows(subject_id)
        for row in subject_rows:
            for field, scope in field_specs:
                source_text = str(row.get(field, "")).strip()
                if not source_text:
                    raise ValueError(
                        f"Empty {field} for {dataset} unit {row.get('sample_id')}: "
                        "empty native transcripts cannot be translated."
                    )
                unit_id = unit_id_for(row, field)
                if scope.startswith("full_") and unit_id in seen_full:
                    continue
                if scope.startswith("full_"):
                    seen_full.add(unit_id)
                context_id, context_text = context_for(row, field, subject_rows)
                parts = split_source_for_budget(
                    source_text,
                    max_chars=max_source_chars if max_source_chars is not None else max(4000, len(source_text)),
                    max_tokens=max_source_tokens,
                    tokenize=tokenize,
                )
                for part_index, part_text in enumerate(parts):
                    key = (unit_id, field, part_index)
                    if key in seen:
                        raise ValueError(f"Duplicate translation unit key: {key}")
                    seen.add(key)
                    units.append(
                        {
                            "dataset": dataset,
                            "unit_id": unit_id,
                            "field": field,
                            "scope": scope,
                            "source_language": source_language,
                            "target_language": TARGET_LANGUAGE,
                            "source_text": part_text,
                            "source_sha256": sha256_text(part_text),
                            "context_id": context_id,
                            "context_text": context_text,
                            "context_sha256": sha256_text(context_text) if context_text else "",
                            "part_index": part_index,
                            "part_count": len(parts),
                        }
                    )
    units.sort(
        key=lambda unit: (
            unit["unit_id"],
            next(
                (index for index, (field, _) in enumerate(field_specs) if field == unit["field"]),
                10**9,
            ),
            unit["part_index"],
        )
    )
    return units


def export_units(
    manifest_path: str | Path,
    dataset: str,
    out_path: str | Path,
    *,
    max_source_tokens: int = 12000,
    max_source_chars: int | None = None,
    max_context_chars: int = 6000,
    tokenize: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    for row in rows:
        if str(row.get("dataset", "")).lower() != dataset:
            raise ValueError(
                f"Manifest at {manifest_path} contains dataset={row.get('dataset')}, "
                f"expected {dataset}."
            )
    units = unit_rows_for_dataset(
        rows,
        dataset,
        max_source_tokens=max_source_tokens,
        max_source_chars=max_source_chars,
        max_context_chars=max_context_chars,
        tokenize=tokenize,
    )
    write_jsonl(units, out_path)
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit["field"]] = counts.get(unit["field"], 0) + 1
    parts = sum(1 for unit in units if unit["part_count"] > 1)
    return {
        "dataset": dataset,
        "unit_count": len(units),
        "field_counts": counts,
        "split_units": parts,
        "source_units_path": str(out_path),
    }


def _load_tokenizer(tokenizer_path: str | None) -> Callable[[str], int] | None:
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Token-based budgets unavailable (%s); using character budgets.", exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export label-free translation units from a native manifest.")
    parser.add_argument("--dataset", required=True, choices=TRANSLATION_DATASETS)
    parser.add_argument("--manifest", required=True, help="Native manifest JSONL path.")
    parser.add_argument("--out", required=True, help="Output units JSONL path.")
    parser.add_argument("--profile", help="Optional length-profile JSON output.")
    parser.add_argument("--tokenizer", help="Qwen tokenizer path for token budgets (default: char budgets).")
    parser.add_argument("--max-source-tokens", type=int, default=12000)
    parser.add_argument("--max-source-chars", type=int, default=None, help="Force char-based splitting budget (default: none).")
    parser.add_argument("--max-context-chars", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    tokenize = _load_tokenizer(args.tokenizer)
    audit = export_units(
        resolve_project_path(args.manifest),
        args.dataset,
        args.out,
        max_source_tokens=args.max_source_tokens,
        max_source_chars=args.max_source_chars,
        max_context_chars=args.max_context_chars,
        tokenize=tokenize,
    )
    LOGGER.info("Exported %s units -> %s (split units: %s)", audit["unit_count"], args.out, audit["split_units"])
    if args.profile:
        rows = read_jsonl(args.out)
        profile = {
            "dataset": args.dataset,
            "source_token_budget": args.max_source_tokens,
            "tokenizer": args.tokenizer or "chars-fallback",
            "fields": {},
        }
        for field in DATASET_UNIT_FIELDS[args.dataset]:
            field_units = [row for row in rows if row["field"] == field[0]]
            profile["fields"][field[0]] = {
                "unit_count": len(field_units),
                "source_chars": {
                    "min": min((len(row["source_text"]) for row in field_units), default=0),
                    "max": max((len(row["source_text"]) for row in field_units), default=0),
                    "mean": (
                        sum(len(row["source_text"]) for row in field_units) / len(field_units)
                        if field_units
                        else 0
                    ),
                },
                "context_chars": {
                    "max": max((len(row["context_text"]) for row in field_units), default=0),
                },
                "split_parts": sum(1 for row in field_units if row["part_count"] > 1),
            }
        ensure_dir(Path(args.profile).parent)
        with Path(args.profile).open("w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        LOGGER.info("Length profile written -> %s", args.profile)


if __name__ == "__main__":
    main()
