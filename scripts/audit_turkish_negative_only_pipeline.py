#!/usr/bin/env python3
"""Build and audit the Turkish negative-only native and English manifests."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_harmonized_en_mn5 import equivalence_audit
from src.data.build_manifest import build_for_config
from src.translation.units import unit_rows_for_dataset
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)


EXPECTED_METADATA_SHA256 = "196bb9b706ff477587559f98b444c8f522f1bb86cc7381698eb64a354e557df0"
EXPECTED_RAW_TRANSCRIPT_SHA256 = "3d99f48b2dbbb6e27040d5e5e561d0cfd35d70e1888543bc75182a58ce22f99e"
EXPECTED_REVIEWED_TRANSCRIPT_SHA256 = "80dce20e9e36062596344a96aca821a79631e098b0b017394f730457155b8798"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"
EXPECTED_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
EXPECTED_ACCEPTED_STATUSES = {
    "automatic_high",
    "automatic_medium",
    "automatic_low",
    "human_verified",
}


def _manifest_evidence(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    metadata = read_json(metadata_path)
    manifest_path = resolve_project_path(metadata["manifest_path"])
    return metadata, read_jsonl(manifest_path)


def _subject_scores_from_source(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            subject = unicodedata.normalize("NFC", str(row["patient_id"]).strip())
            score = float(row["depresyon_skoru"])
            previous = scores.setdefault(subject, score)
            if previous != score:
                raise ValueError(f"Source metadata has mixed depression scores for one subject: {path}")
    return scores


def _normalized_fold_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalized_fold_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized_fold_payload(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFD", value)
    return value


def _audit_native(
    config_path: Path,
    *,
    reference_metadata: Path | None,
    reference_folds: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    config = load_yaml_with_overrides(config_path, [])
    metadata, rows = _manifest_evidence(config_path)
    root = Path(config["dataset_root"])
    source_metadata = root / str(config["metadata_csv"])
    transcript_path = root / str(config["transcript_file"])
    raw_transcript_path = root / "whisper_transcripts_qwen3_asr.jsonl"
    transcript_audit_path = root / "whisper_transcripts_qwen3_asr_reviewed.audit.json"
    transcript_audit = read_json(transcript_audit_path) if transcript_audit_path.is_file() else {}
    subjects = {str(row["subject_id"]): int(row["label"]) for row in rows}
    sample_counts = Counter(int(row["label"]) for row in rows)
    subject_counts = Counter(subjects.values())
    windows = 0
    for row in rows:
        info = sf.info(str(row["audio_path"]))
        duration = float(info.frames / info.samplerate)
        windows += max(1, math.ceil(duration / 30.0))

    checks = {
        "manifest_rows": len(rows) == 1170,
        "manifest_subjects": len(subjects) == 120,
        "sample_labels": sample_counts == Counter({0: 353, 1: 817}),
        "subject_labels": subject_counts == Counter({0: 37, 1: 83}),
        "dataset_variant": {str(row.get("dataset_variant", "")) for row in rows}
        == {"negative_only_t17"},
        "transcripts_nonempty": all(str(row.get("transcript", "")).strip() for row in rows),
        "language_tr": {str(row.get("language", "")) for row in rows} == {"tr"},
        "audio_paths_exist": all(Path(str(row["audio_path"])).is_file() for row in rows),
        "harmonized_windows": windows == 1172,
        "metadata_sha256": sha256_file(source_metadata) == EXPECTED_METADATA_SHA256,
        "raw_transcript_sha256": (
            raw_transcript_path.is_file()
            and sha256_file(raw_transcript_path) == EXPECTED_RAW_TRANSCRIPT_SHA256
        ),
        "reviewed_transcript_sha256": (
            sha256_file(transcript_path) == EXPECTED_REVIEWED_TRANSCRIPT_SHA256
        ),
        "reviewed_transcript_audit": (
            transcript_audit.get("schema_version") == "reviewed_transcript_corrections.v1"
            and transcript_audit.get("source_sha256") == EXPECTED_RAW_TRANSCRIPT_SHA256
            and transcript_audit.get("output_sha256") == sha256_file(transcript_path)
            and int(transcript_audit.get("row_count", -1)) == 1170
            and int(transcript_audit.get("correction_count", -1)) == 1
        ),
        "metadata_record_count": int(metadata.get("manifest_row_count", -1)) == len(rows),
        "metadata_subject_count": int(metadata.get("manifest_subject_count", -1)) == len(subjects),
    }
    if reference_metadata is not None:
        checks["reference_subject_scores"] = (
            _subject_scores_from_source(source_metadata)
            == _subject_scores_from_source(reference_metadata)
        )
    if reference_folds is not None:
        current_folds = read_json(resolve_project_path(metadata["folds_path"]))
        checks["reference_outer_folds"] = (
            _normalized_fold_payload(current_folds)
            == _normalized_fold_payload(read_json(reference_folds))
        )
    failures.extend(name for name, passed in checks.items() if not passed)

    units = unit_rows_for_dataset(rows, "turkish")
    unit_keys = {
        (str(unit["unit_id"]), str(unit["field"]), int(unit["part_index"]))
        for unit in units
    }
    forbidden = {"label", "score", "fold", "depresyon_skoru", "label_t17", "target_t17"}
    unit_checks = {
        "translation_units": len(units) == 1170,
        "translation_unit_keys": len(unit_keys) == len(units),
        "translation_units_label_free": all(not forbidden.intersection(unit) for unit in units),
        "translation_units_unsplit": all(int(unit["part_count"]) == 1 for unit in units),
    }
    failures.extend(name for name, passed in unit_checks.items() if not passed)
    return (
        {
            "config": str(config_path),
            "metadata_path": str(source_metadata),
            "metadata_sha256": sha256_file(source_metadata),
            "transcript_path": str(transcript_path),
            "transcript_sha256": sha256_file(transcript_path),
            "manifest_path": metadata["manifest_path"],
            "manifest_hash": metadata.get("manifest_hash"),
            "fold_hash": metadata.get("fold_hash"),
            "rows": len(rows),
            "subjects": len(subjects),
            "sample_labels": {str(key): value for key, value in sorted(sample_counts.items())},
            "subject_labels": {str(key): value for key, value in sorted(subject_counts.items())},
            "harmonized_audio_windows": windows,
            "translation_units": len(units),
            "checks": {**checks, **unit_checks},
        },
        metadata,
        rows,
        failures,
    )


def _audit_translation(
    cache_root: Path,
    expected_units: int,
    *,
    require_retry_provenance: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    required = ("units.jsonl", "candidates.jsonl", "accepted.jsonl", "rejected.jsonl", "audit.json")
    missing = [name for name in required if not (cache_root / name).is_file()]
    if missing:
        return {"cache_root": str(cache_root), "missing_files": missing}, [
            f"translation cache missing {name}" for name in missing
        ]
    units = read_jsonl(cache_root / "units.jsonl")
    candidates = read_jsonl(cache_root / "candidates.jsonl")
    accepted = read_jsonl(cache_root / "accepted.jsonl")
    rejected = read_jsonl(cache_root / "rejected.jsonl")
    audit = read_json(cache_root / "audit.json")
    repair_provenance = None
    if require_retry_provenance:
        repair_path = cache_root / "repair_provenance.json"
        if not repair_path.is_file():
            failures.append("translation_repair_provenance_missing")
        else:
            repair_provenance = read_json(repair_path)
    unit_hashes = {
        (str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0))): str(row["source_sha256"])
        for row in units
    }
    accepted_hashes = {
        (str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0))): str(row["source_sha256"])
        for row in accepted
    }
    statuses = {str(row.get("status", "")) for row in accepted}
    checks = {
        "units": len(units) == expected_units,
        "candidates": len(candidates) == expected_units,
        "accepted": len(accepted) == expected_units,
        "rejected": not rejected,
        "source_hashes": accepted_hashes == unit_hashes,
        "statuses": not (statuses - EXPECTED_ACCEPTED_STATUSES),
        "model": str(audit.get("model")) == EXPECTED_MODEL,
        "model_revision": str(audit.get("model_revision")) == EXPECTED_MODEL_REVISION,
        "audit_units": int(audit.get("unit_count", -1)) == expected_units,
        "audit_candidates": int(audit.get("candidate_count", -1)) == expected_units,
        "audit_accepted": int(audit.get("accepted_cache_record_count", -1)) == expected_units,
        "audit_extra_candidates": int(audit.get("extra_candidates", -1)) == 0,
    }
    if require_retry_provenance and repair_provenance is not None:
        schema = repair_provenance.get("schema_version")
        common = {
            "repair_units": int(repair_provenance.get("unit_count", -1)) == expected_units,
            "repair_parent_hashes": set(repair_provenance.get("parent_hashes", {}))
            == {"units.jsonl", "candidates.jsonl", "accepted.jsonl", "rejected.jsonl", "audit.json"},
        }
        if schema == "translation_validation_retry.v1":
            pending = int(repair_provenance.get("pending_rejected_count", -1))
            retained = int(repair_provenance.get("retained_candidate_count", -1))
            checks.update(
                {
                    "repair_schema": True,
                    **common,
                    "repair_coverage": pending > 0 and retained + pending == expected_units,
                    "repair_seed": isinstance(repair_provenance.get("retry_seed"), int),
                    "repair_directives_hash": (
                        (cache_root / "validation_retries.jsonl").is_file()
                        and repair_provenance.get("validation_retry_directives_sha256")
                        == sha256_file(cache_root / "validation_retries.jsonl")
                    ),
                }
            )
        elif schema == "translation_reviewed_correction.v1":
            corrected = int(repair_provenance.get("corrected_candidate_count", -1))
            retained = int(repair_provenance.get("retained_candidate_count", -1))
            checks.update(
                {
                    "repair_schema": True,
                    **common,
                    "repair_coverage": corrected > 0 and retained + corrected == expected_units,
                    "repair_source_changes": int(
                        repair_provenance.get("source_changed_unit_count", -1)
                    ) == corrected,
                    "repair_reviewed_hash": (
                        (cache_root / "reviewed.jsonl").is_file()
                        and repair_provenance.get("reviewed_sha256")
                        == sha256_file(cache_root / "reviewed.jsonl")
                    ),
                    "repair_units_hash": repair_provenance.get("units_sha256")
                    == sha256_file(cache_root / "units.jsonl"),
                    "repair_candidates_hash": repair_provenance.get("candidates_sha256")
                    == sha256_file(cache_root / "candidates.jsonl"),
                }
            )
        else:
            checks["repair_schema"] = False
    failures.extend(f"translation_{name}" for name, passed in checks.items() if not passed)
    return {
        "cache_root": str(cache_root),
        "units": len(units),
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "statuses": sorted(statuses),
        "accepted_sha256": sha256_file(cache_root / "accepted.jsonl"),
        "audit_sha256": sha256_file(cache_root / "audit.json"),
        "repair_provenance_sha256": (
            sha256_file(cache_root / "repair_provenance.json")
            if (cache_root / "repair_provenance.json").is_file()
            else None
        ),
        "model": audit.get("model"),
        "model_revision": audit.get("model_revision"),
        "checks": checks,
    }, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    native_config = resolve_project_path(args.native_config)
    english_config = resolve_project_path(args.english_config)
    if args.build_native:
        build_for_config(native_config, [])
    native, native_meta, native_rows, failures = _audit_native(
        native_config,
        reference_metadata=args.reference_metadata,
        reference_folds=args.reference_folds,
    )

    translation = None
    if args.require_translation or args.build_english:
        translation, translation_failures = _audit_translation(
            args.translation_cache,
            1170,
            require_retry_provenance=True,
        )
        failures.extend(translation_failures)

    equivalence = None
    english = None
    if args.build_english:
        if failures:
            failures.append("english_build_skipped_due_to_failed_prerequisite")
        else:
            build_for_config(english_config, [])
            en_meta, en_rows = _manifest_evidence(english_config)
            equivalence = equivalence_audit(native_meta, native_rows, en_meta, en_rows)
            failures.extend(f"english_equivalence: {item}" for item in equivalence["failures"])
            english = {
                "config": str(english_config),
                "manifest_path": en_meta["manifest_path"],
                "manifest_hash": en_meta.get("manifest_hash"),
                "fold_hash": en_meta.get("fold_hash"),
                "rows": len(en_rows),
                "subjects": len({str(row["subject_id"]) for row in en_rows}),
                "overlay": en_meta.get("transcript_overlay"),
            }

    return {
        "schema_version": "turkish_negative_only_pipeline_audit.v1",
        "status": "passed" if not failures else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": os.environ.get("PIPELINE_SOURCE_COMMIT"),
            "git_branch": os.environ.get("PIPELINE_SOURCE_BRANCH"),
        },
        "native": native,
        "translation": translation,
        "english": english,
        "equivalence": equivalence,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-config",
        default="configs/main/turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml",
    )
    parser.add_argument(
        "--english-config",
        default="configs/main/turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
    )
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument("--build-english", action="store_true")
    parser.add_argument("--require-translation", action="store_true")
    parser.add_argument(
        "--translation-cache",
        type=Path,
        default=Path(
            os.environ.get(
                "TURKISH_NEGATIVE_ONLY_TRANSLATION_CACHE",
                "/gpfs/projects/etur92/ozu647717/AudioLLM/translations/harmonized_en_complete_v3/turkish_negative_only_t17",
            )
        ),
    )
    parser.add_argument("--reference-metadata", type=Path)
    parser.add_argument("--reference-folds", type=Path)
    parser.add_argument("--audit-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = run(args)
    save_json(audit, args.audit_path)
    print(json.dumps({"status": audit["status"], "audit": str(args.audit_path), "failures": audit["failures"]}, indent=2))
    if audit["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
