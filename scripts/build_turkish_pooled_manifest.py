#!/usr/bin/env python3
"""Build the Turkish pooled manifest from four already-validated source pairs.

This command is deliberately source-manifest-only.  It never reads raw audio,
metadata CSVs, transcript stores, translation caches, or fold generators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import manifest_build_signature
from src.utils import load_yaml_with_overrides, read_json, read_jsonl, sha256_file, sha256_jsonl_rows


POSITIVE = "pos_only_t17"
NEGATIVE = "negative_only_t17"
CONDITIONS = (POSITIVE, NEGATIVE)
EXPECTED_ROWS = {POSITIVE: 1051, NEGATIVE: 1170}
EXPECTED_SUBJECTS = 120
EXPECTED_LABELS = {0: 37, 1: 83}
EXPECTED_FOLDS = {0, 1, 2, 3, 4}
SCHEMA_VERSION = "audiollm.turkish_pooled_manifest.v1"
AUDIT_SCHEMA_VERSION = "audiollm.turkish_pooled_manifest_audit.v1"


class ManifestError(ValueError):
    """Raised for a source or destination contract violation."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON source {path}: {exc}") from exc


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest source {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ManifestError(f"manifest source is not a row list: {path}")
    return rows


def _resolve_split_path(path: Path) -> tuple[Path, dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict) and payload.get("folds_path"):
        nested = Path(str(payload["folds_path"]))
        if not nested.is_absolute():
            nested = path.parent / nested
        nested = nested.resolve()
        payload = _read_json(nested)
        return nested, payload
    if isinstance(payload, dict) and isinstance(payload.get("folds"), dict):
        return path, payload["folds"]
    if not isinstance(payload, dict):
        raise ManifestError(f"split source is not an object: {path}")
    return path, payload


def _split_mapping(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    resolved_path, payload = _resolve_split_path(path)
    mapping: dict[str, int] = {}
    duplicate: list[str] = []
    for raw_fold, raw_payload in payload.items():
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"non-integer fold in {resolved_path}: {raw_fold!r}") from exc
        if fold not in EXPECTED_FOLDS:
            raise ManifestError(f"unexpected fold {fold} in {resolved_path}")
        if not isinstance(raw_payload, dict):
            raise ManifestError(f"fold payload is not an object in {resolved_path}: {fold}")
        subjects = raw_payload.get("final_eval_subject_ids")
        if not isinstance(subjects, list):
            raise ManifestError(f"fold {fold} has no final_eval_subject_ids in {resolved_path}")
        for subject in subjects:
            subject_id = str(subject)
            if subject_id in mapping:
                duplicate.append(subject_id)
            mapping[subject_id] = fold
    if duplicate:
        raise ManifestError(f"subjects occur in multiple folds in {resolved_path}: {sorted(set(duplicate))[:10]}")
    if len(mapping) != EXPECTED_SUBJECTS:
        raise ManifestError(f"split {resolved_path} has {len(mapping)} subjects, expected {EXPECTED_SUBJECTS}")
    if set(mapping.values()) != EXPECTED_FOLDS:
        raise ManifestError(f"split {resolved_path} does not contain folds 0..4")
    canonical = sorted((subject, fold) for subject, fold in mapping.items())
    return mapping, {
        "path": str(resolved_path),
        "raw_sha256": sha256_file(resolved_path),
        "canonical_mapping_sha256": _canonical_sha(canonical),
        "subject_count": len(mapping),
        "fold_counts": dict(sorted(Counter(mapping.values()).items())),
    }


def _audio_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("audio_paths")
    if paths is None:
        paths = [row.get("audio_path")] if row.get("audio_path") else []
    return [str(path) for path in paths if str(path).strip()]


def _identity_projection(row: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "transcript", "transcript_original", "language", "source_language",
        "transcript_variant", "translation_model", "translation_status",
        "translation_sha256",
    }
    return {key: value for key, value in row.items() if key not in ignored}


def _source_contract(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    language: str,
    expected_rows: int,
) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ManifestError(f"{condition}/{language} has {len(rows)} rows, expected {expected_rows}")
    required = {"dataset", "dataset_variant", "sample_id", "subject_id", "label", "score", "threshold", "transcript"}
    sample_ids: set[str] = set()
    subject_values: dict[str, dict[str, Any]] = {}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ManifestError(f"{condition}/{language} row is missing fields {missing}")
        if str(row.get("dataset", "")).lower() != "turkish":
            raise ManifestError(f"{condition}/{language} has a non-Turkish row")
        if str(row.get("dataset_variant", "")).strip() != condition:
            raise ManifestError(
                f"{condition}/{language} row {row.get('sample_id', '')} has condition "
                f"{row.get('dataset_variant')!r}"
            )
        sample_id = str(row["sample_id"])
        subject_id = str(row["subject_id"])
        if not sample_id or not subject_id:
            raise ManifestError(f"{condition}/{language} has an empty sample or subject ID")
        if sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id in {condition}/{language}: {sample_id}")
        sample_ids.add(sample_id)
        label = int(row["label"])
        if label not in (0, 1):
            raise ManifestError(f"non-binary label in {condition}/{language}: {sample_id}")
        score = float(row["score"])
        threshold = float(row["threshold"])
        if not (math.isfinite(score) and math.isfinite(threshold)):
            raise ManifestError(f"non-finite score or threshold in {condition}/{language}: {sample_id}")
        if threshold != 17.0:
            raise ManifestError(f"threshold is not 17 in {condition}/{language}: {sample_id}")
        if not str(row.get("transcript", "")).strip():
            raise ManifestError(f"empty transcript in {condition}/{language}: {sample_id}")
        paths = _audio_paths(row)
        if not paths:
            raise ManifestError(f"missing audio path in {condition}/{language}: {sample_id}")
        missing_audio = [path for path in paths if not Path(path).is_file()]
        if missing_audio:
            raise ManifestError(
                f"missing audio in {condition}/{language}: {missing_audio[0]}"
            )
        subject_value = {"label": label, "score": score, "threshold": threshold}
        previous = subject_values.setdefault(subject_id, subject_value)
        if previous != subject_value:
            raise ManifestError(
                f"subject {subject_id} has inconsistent label/score in {condition}/{language}"
            )
    if len(subject_values) != EXPECTED_SUBJECTS:
        raise ManifestError(
            f"{condition}/{language} has {len(subject_values)} subjects, expected {EXPECTED_SUBJECTS}"
        )
    labels = Counter(value["label"] for value in subject_values.values())
    if dict(sorted(labels.items())) != EXPECTED_LABELS:
        raise ManifestError(
            f"{condition}/{language} subject labels are {dict(sorted(labels.items()))}, "
            f"expected {EXPECTED_LABELS}"
        )
    return subject_values, {
        "sample_id_count": len(sample_ids),
        "sample_id_sha256": _canonical_sha(sorted(sample_ids)),
        "sample_ids": sorted(sample_ids),
    }, {
        "rows": len(rows),
        "subjects": len(subject_values),
        "subject_label_counts": dict(sorted(labels.items())),
        "condition_counts": {condition: len(rows)},
        "sample_ids": sorted(sample_ids),
    }


def _check_translation_pair(
    native_rows: list[dict[str, Any]],
    english_rows: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[str, Any]:
    native_by_id = {str(row["sample_id"]): row for row in native_rows}
    english_by_id = {str(row["sample_id"]): row for row in english_rows}
    if set(native_by_id) != set(english_by_id):
        raise ManifestError(f"{condition}: native and English sample IDs do not pair one-to-one")
    valid_hashes = 0
    for sample_id in sorted(native_by_id):
        native = native_by_id[sample_id]
        english = english_by_id[sample_id]
        if _identity_projection(native) != _identity_projection(english):
            raise ManifestError(f"{condition}: non-transcript identity differs for {sample_id}")
        translation = str(english.get("transcript", "")).strip()
        declared = str(english.get("translation_sha256", "")).strip()
        if not declared:
            raise ManifestError(f"{condition}: translation hash missing for {sample_id}")
        if not translation:
            raise ManifestError(f"{condition}: English transcript is empty for {sample_id}")
        if declared != hashlib.sha256(translation.encode("utf-8")).hexdigest():
            raise ManifestError(f"{condition}: translation hash mismatch for {sample_id}")
        if str(english.get("language", "")) != "en" or str(english.get("transcript_variant", "")) != "english":
            raise ManifestError(f"{condition}: English language markers are missing for {sample_id}")
        valid_hashes += 1
    return {
        "paired_rows": len(native_by_id),
        "translation_hash_valid_rows": valid_hashes,
        "pairing_sha256": _canonical_sha(sorted(native_by_id)),
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _build_signature(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    config = load_yaml_with_overrides(config_path, [])
    if not isinstance(config, dict):
        raise ManifestError(f"pooled signature config is not an object: {config_path}")
    return manifest_build_signature(config)


def _write_once_files(
    files: dict[Path, bytes], *, check_only: bool, audit_path: Path
) -> None:
    output_files = {path: payload for path, payload in files.items() if path != audit_path}
    destinations = {path.parent for path in output_files}
    for directory in destinations:
        # Output directories contain only the four direct artifacts.  Do not
        # recursively inspect the audit parent: it may also contain the four
        # read-only source trees supplied to this command.
        existing = {
            path for path in directory.iterdir() if path.is_file()
        } if directory.exists() else set()
        extra = sorted(existing - set(output_files))
        if extra:
            raise ManifestError(f"destination contains incompatible existing files: {extra[0]}")
        for path, payload in output_files.items():
            if path.exists() and path.read_bytes() != payload:
                raise ManifestError(f"refusing to overwrite incompatible existing output: {path}")
    if audit_path.exists() and audit_path.read_bytes() != files[audit_path]:
        raise ManifestError(f"refusing to overwrite incompatible audit output: {audit_path}")
    if check_only:
        return
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)


def _output_files(
    manifest_dir: Path,
    split_dir: Path,
    rows: list[dict[str, Any]],
    folds: dict[str, Any],
    *,
    language: str,
    build_signature: dict[str, Any] | None,
    source_hashes: dict[str, Any],
    source_split_hash: str,
    source_manifest_hash: str,
) -> dict[Path, bytes]:
    manifest_hash = sha256_jsonl_rows(rows)
    manifest_path = manifest_dir / "turkish_manifest.jsonl"
    folds_path = split_dir / "turkish_folds.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "turkish",
        "dataset_variant": "pooled_t17",
        "transcript_variant": language,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_hash": manifest_hash,
        "manifest_file_sha256": hashlib.sha256((b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
        ))).hexdigest(),
        "manifest_row_count": len(rows),
        "manifest_subject_count": EXPECTED_SUBJECTS,
        "build_signature": build_signature,
        "folds_path": str(folds_path.resolve()),
        "fold_hash": source_split_hash,
        "split_source": "reused_canonical_source_folds",
        "split_source_notes": "Existing five-fold subject mapping reused only after four-way semantic identity audit.",
        "source_manifest_hash": source_manifest_hash,
        "source_split_hash": source_split_hash,
        "source_hashes": source_hashes,
        "condition_counts": dict(Counter(str(row["dataset_variant"]) for row in rows)),
        "question_condition_values": list(CONDITIONS),
        "translation_pairing": None if language == "native" else {
            "required": True,
            "paired_rows": sum(1 for row in rows if str(row.get("translation_sha256", ""))),
        },
    }
    manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )
    return {
        manifest_path: manifest_bytes,
        manifest_dir / "turkish_manifest.csv": _csv_bytes(rows),
        folds_path: _json_bytes(folds),
        split_dir / "turkish_manifest_metadata.json": _json_bytes(metadata),
    }


def _validate_sources(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_specs = {
        POSITIVE: {
            "native": (Path(args.positive_native_manifest), Path(args.positive_native_split)),
            "english": (Path(args.positive_english_manifest), Path(args.positive_english_split)),
        },
        NEGATIVE: {
            "native": (Path(args.negative_native_manifest), Path(args.negative_native_split)),
            "english": (Path(args.negative_english_manifest), Path(args.negative_english_split)),
        },
    }
    mappings: dict[str, dict[str, int]] = {}
    split_audit: dict[str, Any] = {}
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifests_audit: dict[str, Any] = {}
    subject_values: dict[str, dict[str, Any]] = {}
    all_sample_ids_by_language: dict[str, set[str]] = {"native": set(), "english": set()}
    for condition in CONDITIONS:
        for language in ("native", "english"):
            manifest_path, split_path = source_specs[condition][language]
            if not manifest_path.is_file() or not split_path.is_file():
                raise ManifestError(f"missing explicit {condition}/{language} source: {manifest_path} / {split_path}")
            rows = _read_rows(manifest_path)
            rows_by_key[(condition, language)] = rows
            mapping, split_record = _split_mapping(split_path)
            mappings[f"{condition}/{language}"] = mapping
            split_audit[f"{condition}/{language}"] = {
                **split_record,
                "input_path": str(split_path.resolve()),
                "input_sha256": sha256_file(split_path),
            }
            values, sample_audit, manifest_record = _source_contract(
                rows, condition=condition, language=language,
                expected_rows=EXPECTED_ROWS[condition],
            )
            manifests_audit[f"{condition}/{language}"] = {
                "input_path": str(manifest_path.resolve()),
                "input_sha256": sha256_file(manifest_path),
                "canonical_manifest_hash": sha256_jsonl_rows(rows),
                "sample_audit": sample_audit,
                **manifest_record,
            }
            if set(values) != set(mapping):
                raise ManifestError(f"{condition}/{language}: manifest and split subject sets differ")
            if language == "native":
                if not subject_values:
                    subject_values = values
                elif values != subject_values:
                    raise ManifestError(f"{condition}: subject labels or scores differ from the positive source")
            if all_sample_ids_by_language[language].intersection(sample_audit["sample_ids"]):
                raise ManifestError(f"sample IDs are duplicated across pooled source manifests: {condition}/{language}")
            all_sample_ids_by_language[language].update(sample_audit["sample_ids"])
    first_mapping = next(iter(mappings.values()))
    for name, mapping in mappings.items():
        if mapping != first_mapping:
            raise ManifestError(f"canonical subject fold mapping differs for {name}")
    _check_translation_pair(rows_by_key[(POSITIVE, "native")], rows_by_key[(POSITIVE, "english")], condition=POSITIVE)
    _check_translation_pair(rows_by_key[(NEGATIVE, "native")], rows_by_key[(NEGATIVE, "english")], condition=NEGATIVE)
    for condition in CONDITIONS:
        native_values = {
            str(row["subject_id"]): {"label": int(row["label"]), "score": float(row["score"])}
            for row in rows_by_key[(condition, "native")]
        }
        if native_values != {
            subject: {"label": int(value["label"]), "score": float(value["score"])}
            for subject, value in subject_values.items()
        }:
            raise ManifestError(f"{condition}: native labels or threshold scores differ between conditions")
    pooled = {
        "native": rows_by_key[(POSITIVE, "native")] + rows_by_key[(NEGATIVE, "native")],
        "english": rows_by_key[(POSITIVE, "english")] + rows_by_key[(NEGATIVE, "english")],
    }
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "expected": {
            "rows": dict(EXPECTED_ROWS),
            "pooled_rows": sum(EXPECTED_ROWS.values()),
            "subjects": EXPECTED_SUBJECTS,
            "subject_labels": {str(key): value for key, value in EXPECTED_LABELS.items()},
            "folds": sorted(EXPECTED_FOLDS),
        },
        "source_manifests": manifests_audit,
        "source_splits": split_audit,
        "canonical_split_mapping_sha256": _canonical_sha(sorted(first_mapping.items())),
        "pooled_manifest_hashes": {name: sha256_jsonl_rows(rows) for name, rows in pooled.items()},
        "pooled_row_counts": {name: len(rows) for name, rows in pooled.items()},
        "pooled_subject_counts": {name: len({str(row['subject_id']) for row in rows}) for name, rows in pooled.items()},
        "pooled_condition_counts": {
            name: dict(sorted(Counter(str(row["dataset_variant"]) for row in rows).items()))
            for name, rows in pooled.items()
        },
        "translation_pairing_counts": {
            condition: EXPECTED_ROWS[condition] for condition in CONDITIONS
        },
        "transcript_asymmetry": {
            POSITIVE: "positive native transcripts are unreviewed",
            NEGATIVE: "negative native transcripts are reviewed",
        },
        "checks": {
            "four_split_mappings_identical": True,
            "native_english_sample_pairing": True,
            "translation_hashes_valid": True,
            "audio_paths_exist": True,
            "labels_and_scores_match": True,
            "source_condition_tags_preserved": True,
        },
    }
    audit["audit_sha256"] = _canonical_sha(audit)
    return pooled, {"folds": {str(fold): {
        "final_eval_subject_ids": sorted(subject for subject, value in first_mapping.items() if value == fold),
        "outer_train_subject_ids": sorted(subject for subject, value in first_mapping.items() if value != fold),
    } for fold in sorted(EXPECTED_FOLDS)}}, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("positive-native-manifest", "positive native source manifest"),
        ("positive-native-split", "positive native source fold file"),
        ("negative-native-manifest", "negative native source manifest"),
        ("negative-native-split", "negative native source fold file"),
        ("positive-english-manifest", "positive English source manifest"),
        ("positive-english-split", "positive English source fold file"),
        ("negative-english-manifest", "negative English source manifest"),
        ("negative-english-split", "negative English source fold file"),
    ):
        parser.add_argument(f"--{name}", required=True, type=Path, help=help_text)
    parser.add_argument("--native-output-dir", required=True, type=Path)
    parser.add_argument("--english-output-dir", required=True, type=Path)
    parser.add_argument("--native-split-output-dir", required=True, type=Path)
    parser.add_argument("--english-split-output-dir", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--english-config", type=Path)
    parser.add_argument("--check-only", action="store_true", help="validate and print the audit without writing any file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pooled, fold_payload, audit = _validate_sources(args)
    native_signature = _build_signature(args.native_config)
    english_signature = _build_signature(args.english_config)
    folds = fold_payload["folds"]
    native_files = _output_files(
        args.native_output_dir.resolve(), args.native_split_output_dir.resolve(), pooled["native"], folds,
        language="native", build_signature=native_signature,
        source_hashes=audit["source_manifests"],
        source_split_hash=audit["canonical_split_mapping_sha256"],
        source_manifest_hash=audit["pooled_manifest_hashes"]["native"],
    )
    english_files = _output_files(
        args.english_output_dir.resolve(), args.english_split_output_dir.resolve(), pooled["english"], folds,
        language="english", build_signature=english_signature,
        source_hashes=audit["source_manifests"],
        source_split_hash=audit["canonical_split_mapping_sha256"],
        source_manifest_hash=audit["pooled_manifest_hashes"]["english"],
    )
    audit = {
        **audit,
        "outputs": {
            "native": {str(path): hashlib.sha256(payload).hexdigest() for path, payload in native_files.items()},
            "english": {str(path): hashlib.sha256(payload).hexdigest() for path, payload in english_files.items()},
        },
    }
    audit["audit_sha256"] = _canonical_sha({key: value for key, value in audit.items() if key != "audit_sha256"})
    all_files = {**native_files, **english_files}
    all_files[args.audit_output.resolve()] = _json_bytes(audit)
    _write_once_files(all_files, check_only=args.check_only, audit_path=args.audit_output.resolve())
    result = {
        "status": "passed",
        "check_only": bool(args.check_only),
        "audit_sha256": audit["audit_sha256"],
        "native_manifest_hash": audit["pooled_manifest_hashes"]["native"],
        "english_manifest_hash": audit["pooled_manifest_hashes"]["english"],
        "pooled_rows": sum(EXPECTED_ROWS.values()),
        "subjects": EXPECTED_SUBJECTS,
        "condition_counts": audit["pooled_condition_counts"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
