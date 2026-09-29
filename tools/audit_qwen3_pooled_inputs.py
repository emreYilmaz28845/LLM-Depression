#!/usr/bin/env python3
"""Audit the built pooled Turkish inputs in a task runtime against the recorded contract.

The pooled native/English manifests are built by
``scripts/build_turkish_pooled_manifest.py`` inside the task runtime. This audit
re-verifies the built files independently of the builder: row and subject
counts, the 37/83 label split at threshold 17, the five-fold map, the
native/English pairing with valid translation hashes, and the canonical
manifest/fold hashes. When the recorded production hashes are supplied it also
proves the runtime reproduces the identities the pooled campaign recorded.

CPU only; no model, tokenizer or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_json, read_jsonl, sha256_file, sha256_jsonl_rows

EXPECTED_ROWS = 2221
EXPECTED_SUBJECTS = 120
EXPECTED_LABELS = {0: 37, 1: 83}
EXPECTED_CONDITIONS = {"pos_only_t17": 1051, "negative_only_t17": 1170}
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_THRESHOLD = 17.0
IDENTITY_IGNORED_FIELDS = {
    "transcript", "transcript_original", "language", "source_language",
    "transcript_variant", "translation_model", "translation_status", "translation_sha256",
}
# Recorded production identities (the pooled campaign's provenance index).
RECORDED_NATIVE_MANIFEST_HASH = "37e991526986d9693c9620682719a6b54c7d30ec86b53152ae23dde167271b70"
RECORDED_FOLD_FILE_SHA256 = "3262a009db52c6d049e223947a9be6ce119a31e816b5c3072ce84b3ad92ecd58"


class AuditError(RuntimeError):
    """Raised when the built runtime violates the pooled input contract."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_fold_mapping(folds_path: Path) -> dict[str, int]:
    payload = read_json(folds_path)
    if isinstance(payload, dict) and "folds" in payload:
        payload = payload["folds"]
    mapping: dict[str, int] = {}
    for raw_fold, data in payload.items():
        fold = int(raw_fold)
        for subject in data["final_eval_subject_ids"]:
            subject_id = str(subject)
            if subject_id in mapping:
                raise AuditError(f"subject {subject_id} appears in more than one fold")
            mapping[subject_id] = fold
    return mapping


def _identity_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in IDENTITY_IGNORED_FIELDS}


def _audit_language(runtime_root: Path, language: str) -> dict[str, Any]:
    manifest_dir = runtime_root / ("manifests_en" if language == "english" else "manifests") / "turkish"
    split_dir = runtime_root / ("splits_en" if language == "english" else "splits") / "turkish"
    manifest_path = manifest_dir / "turkish_manifest.jsonl"
    metadata_path = split_dir / "turkish_manifest_metadata.json"
    folds_path = split_dir / "turkish_folds.json"
    for path in (manifest_path, metadata_path, folds_path):
        if not path.is_file():
            raise AuditError(f"missing pooled {language} artifact: {path}")
    rows = read_jsonl(manifest_path)
    metadata = read_json(metadata_path)
    if metadata.get("transcript_variant") != language:
        raise AuditError(
            f"{language}: metadata transcript_variant is {metadata.get('transcript_variant')!r}"
        )
    if len(rows) != EXPECTED_ROWS:
        raise AuditError(f"{language}: {len(rows)} rows, expected {EXPECTED_ROWS}")
    conditions = Counter(str(row["dataset_variant"]) for row in rows)
    if dict(conditions) != EXPECTED_CONDITIONS:
        raise AuditError(f"{language}: condition counts are {dict(conditions)}")
    subject_labels: dict[str, int] = {}
    subject_scores: dict[str, float] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        score = float(row["score"])
        if float(row["threshold"]) != EXPECTED_THRESHOLD:
            raise AuditError(f"{language}: threshold is not {EXPECTED_THRESHOLD} in {row['sample_id']}")
        if not str(row.get("transcript", "")).strip():
            raise AuditError(f"{language}: empty transcript in {row['sample_id']}")
        if subject_labels.setdefault(subject_id, label) != label:
            raise AuditError(f"{language}: inconsistent label for subject {subject_id}")
        if subject_scores.setdefault(subject_id, score) != score:
            raise AuditError(f"{language}: inconsistent score for subject {subject_id}")
        if language == "english":
            translation = str(row.get("transcript", ""))
            declared = str(row.get("translation_sha256", ""))
            if str(row.get("language")) != "en" or str(row.get("transcript_variant")) != "english":
                raise AuditError(f"{language}: missing English markers in {row['sample_id']}")
            if declared != hashlib.sha256(translation.encode("utf-8")).hexdigest():
                raise AuditError(f"{language}: translation hash mismatch in {row['sample_id']}")
    if len(subject_labels) != EXPECTED_SUBJECTS:
        raise AuditError(f"{language}: {len(subject_labels)} subjects, expected {EXPECTED_SUBJECTS}")
    labels = Counter(subject_labels.values())
    if dict(sorted(labels.items())) != EXPECTED_LABELS:
        raise AuditError(f"{language}: subject labels are {dict(sorted(labels.items()))}")
    manifest_hash = sha256_jsonl_rows(rows)
    if metadata.get("manifest_hash") != manifest_hash:
        raise AuditError(f"{language}: metadata manifest_hash does not match the rows")
    mapping = _load_fold_mapping(folds_path)
    if len(mapping) != EXPECTED_SUBJECTS or set(mapping.values()) != set(EXPECTED_FOLDS):
        raise AuditError(f"{language}: fold map does not cover the five folds and 120 subjects")
    if set(mapping) != set(subject_labels):
        raise AuditError(f"{language}: fold subjects and manifest subjects differ")
    canonical_mapping = _canonical_sha(sorted(mapping.items()))
    if metadata.get("fold_hash") != canonical_mapping:
        raise AuditError(f"{language}: metadata fold_hash does not match the fold map")
    return {
        "language": language,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_hash": manifest_hash,
        "rows": len(rows),
        "subjects": len(subject_labels),
        "label_counts": {str(key): value for key, value in sorted(labels.items())},
        "condition_counts": dict(sorted(conditions.items())),
        "folds_path": str(folds_path),
        "folds_file_sha256": sha256_file(folds_path),
        "fold_canonical_mapping_sha256": canonical_mapping,
        "fold_counts": {str(fold): sum(1 for value in mapping.values() if value == fold) for fold in EXPECTED_FOLDS},
        "_rows": rows,
        "_mapping": mapping,
    }


def audit(runtime_root: Path) -> dict[str, Any]:
    native = _audit_language(runtime_root, "native")
    english = _audit_language(runtime_root, "english")
    native_rows = {str(row["sample_id"]): row for row in native.pop("_rows")}
    english_rows = {str(row["sample_id"]): row for row in english.pop("_rows")}
    native_mapping = native.pop("_mapping")
    english_mapping = english.pop("_mapping")
    if set(native_rows) != set(english_rows):
        raise AuditError("native and English sample-id sets differ")
    if native_mapping != english_mapping:
        raise AuditError("native and English fold maps differ")
    for sample_id in sorted(native_rows):
        if _identity_projection(native_rows[sample_id]) != _identity_projection(english_rows[sample_id]):
            raise AuditError(f"native/English identity projection differs for {sample_id}")
    return {
        "schema_version": "audiollm.qwen3_pooled_inputs_audit.v1",
        "runtime_root": str(runtime_root),
        "native": native,
        "english": english,
        "pairing": {
            "paired_rows": len(native_rows),
            "identity_projection_equal": True,
            "fold_maps_equal": True,
        },
        "recorded_contract": {
            "native_manifest_hash": RECORDED_NATIVE_MANIFEST_HASH,
            "fold_file_sha256": RECORDED_FOLD_FILE_SHA256,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--expect-recorded-hashes", action="store_true",
                        help="also require the recorded production manifest and fold identities")
    parser.add_argument("--output", type=Path, help="write the audit JSON here")
    args = parser.parse_args(argv)

    try:
        payload = audit(args.runtime_root.resolve())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.expect_recorded_hashes:
        problems = []
        if payload["native"]["manifest_hash"] != RECORDED_NATIVE_MANIFEST_HASH:
            problems.append(
                f"native manifest hash {payload['native']['manifest_hash']} != recorded "
                f"{RECORDED_NATIVE_MANIFEST_HASH}"
            )
        if payload["native"]["folds_file_sha256"] != RECORDED_FOLD_FILE_SHA256:
            problems.append(
                f"fold file sha256 {payload['native']['folds_file_sha256']} != recorded "
                f"{RECORDED_FOLD_FILE_SHA256}"
            )
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 2
    payload["status"] = "passed"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps({
        "status": "passed",
        "native_manifest_hash": payload["native"]["manifest_hash"],
        "english_manifest_hash": payload["english"]["manifest_hash"],
        "fold_file_sha256": payload["native"]["folds_file_sha256"],
        "rows": payload["native"]["rows"],
        "subjects": payload["native"]["subjects"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
