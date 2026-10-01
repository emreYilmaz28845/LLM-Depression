#!/usr/bin/env python3
"""Audit native/English manifest equivalence for the Qwen3 English cells.

An English-transcript cell must differ from its native counterpart by the
transcript text alone. Everything the model does not see as text has to stay
identical: the same rows, the same subjects, the same labels, the same audio
files. This tool compares a native manifest with its English manifest and fails
closed on any drift, on a row that never received a translation, and on a row
whose translation equals the native text (a native fallback).

It also records the translation provenance markers the overlay writes
(``language``, ``transcript_variant``, ``translation_sha256``) and refuses
geriatri rows in the pooled family.

Run it where both manifests exist. Locally the caches and datasets are under
``/media/emre/Backup/AudioLLM``; on the cluster they live in the task runtime.
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

from src.utils import read_jsonl  # noqa: E402

SCHEMA_VERSION = "audiollm.en_manifest_equivalence.v1"
IDENTITY_FIELDS = ("subject_id", "label", "audio_path")
TRANSLATION_MARKERS = ("language", "transcript_variant", "translation_sha256")
GERIATRI_MARKER = "geriatri"


class AuditError(RuntimeError):
    """Raised when a manifest pair cannot be audited at all."""


def _index(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise AuditError(f"{path}: row without sample_id")
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = row
    if duplicates:
        raise AuditError(f"{path}: duplicate sample_ids: {sorted(set(duplicates))[:5]}")
    return indexed


def audit_pair(dataset: str, native_path: Path, english_path: Path) -> dict[str, Any]:
    if not native_path.is_file():
        raise AuditError(f"missing native manifest: {native_path}")
    if not english_path.is_file():
        raise AuditError(f"missing English manifest: {english_path}")
    native = _index(read_jsonl(native_path), native_path)
    english = _index(read_jsonl(english_path), english_path)

    failures: list[str] = []
    native_only = sorted(set(native) - set(english))
    english_only = sorted(set(english) - set(native))
    if native_only:
        failures.append(f"rows present only in the native manifest: {native_only[:5]}")
    if english_only:
        failures.append(f"rows present only in the English manifest: {english_only[:5]}")

    mismatches: dict[str, list[str]] = {field: [] for field in IDENTITY_FIELDS}
    for sample_id in sorted(set(native) & set(english)):
        for field in IDENTITY_FIELDS:
            if native[sample_id].get(field) != english[sample_id].get(field):
                mismatches[field].append(sample_id)
    for field, samples in mismatches.items():
        if samples:
            failures.append(f"{field} differs for {len(samples)} rows, e.g. {samples[:5]}")

    missing_markers: dict[str, list[str]] = {marker: [] for marker in TRANSLATION_MARKERS}
    empty_translations: list[str] = []
    identical_text: list[str] = []
    geriatri_rows: list[str] = []
    native_languages: dict[str, int] = {}
    for sample_id, row in english.items():
        for marker in TRANSLATION_MARKERS:
            if not str(row.get(marker) or "").strip():
                missing_markers[marker].append(sample_id)
        if not str(row.get("transcript") or "").strip():
            empty_translations.append(sample_id)
        if str(row.get("language") or "") != "en":
            failures.append(f"row {sample_id}: language={row.get('language')!r}, expected 'en'")
        if str(row.get("transcript_variant") or "") != "english":
            failures.append(
                f"row {sample_id}: transcript_variant={row.get('transcript_variant')!r}, "
                "expected 'english'"
            )
        if sample_id in native:
            if str(native[sample_id].get("transcript") or "") == str(row.get("transcript") or ""):
                identical_text.append(sample_id)
            language = str(native[sample_id].get("language") or "unknown")
            native_languages[language] = native_languages.get(language, 0) + 1
        if GERIATRI_MARKER in sample_id.lower() or GERIATRI_MARKER in str(
            row.get("subject_id") or ""
        ).lower():
            geriatri_rows.append(sample_id)
    for marker, samples in missing_markers.items():
        if samples:
            failures.append(f"{len(samples)} English rows lack {marker}, e.g. {samples[:5]}")
    if empty_translations:
        failures.append(f"{len(empty_translations)} English rows have an empty transcript")
    if identical_text:
        failures.append(
            f"{len(identical_text)} English rows repeat the native transcript (native fallback), "
            f"e.g. {identical_text[:5]}"
        )
    if geriatri_rows:
        failures.append(f"geriatri rows are not part of this family: {geriatri_rows[:5]}")

    return {
        "dataset": dataset,
        "native_manifest": str(native_path),
        "english_manifest": str(english_path),
        "rows": {"native": len(native), "english": len(english)},
        "native_source_languages": native_languages,
        "identical_text_rows": identical_text,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }


def parse_pairs(values: list[list[str]]) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for value in values:
        if len(value) != 3:
            raise AuditError("--pair takes exactly three values: dataset native_manifest english_manifest")
        pairs.append((value[0], Path(value[1]), Path(value[2])))
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("DATASET", "NATIVE", "ENGLISH"),
        required=True,
        help="one manifest pair to audit (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pairs = parse_pairs(args.pair)
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    audit: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "pairs": []}
    failed = 0
    for dataset, native_path, english_path in pairs:
        try:
            record = audit_pair(dataset, native_path, english_path)
        except AuditError as error:
            record = {"dataset": dataset, "status": "failed", "failures": [str(error)]}
        audit["pairs"].append(record)
        print(f"{dataset}: {record['status']} ({record.get('rows')})")
        for failure in record["failures"]:
            print(f"  ERROR: {failure}")
        if record["status"] != "passed":
            failed += 1
    audit["status"] = "passed" if failed == 0 else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
