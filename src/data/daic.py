from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.split_utils import (
    build_partition_scoped_stratified_folds,
    resolve_dev_pool_partitions,
    resolve_outer_fold_count,
    subject_fold_report,
)
from src.data.validation import is_quarantined_missing
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)
LEGACY_CHUNK_RE = re.compile(r"^(?P<subject_id>\d+)_(?P<chunk_id>\d+)\.wav$")
PREPROCESSED_SEGMENT_RE = re.compile(
    r"^(?P<subject_id>\d+)_(?P<segment_kind>random_segment|segment)_(?P<chunk_id>\d+)\.wav$"
)
PREPROCESSED_TRAIN_DEV_VARIANT = "preprocessed_train_dev"
PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT = "preprocessed_full_transcript_all_splits"
PREPROCESSED_TEST_FULL_TRANSCRIPT_VARIANT = "preprocessed_test_full_transcript"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        filtered_lines = [line for line in handle if line.strip()]
    return list(csv.DictReader(filtered_lines))


def _sample_id_from_audio_name(audio_name: str) -> str | None:
    if LEGACY_CHUNK_RE.match(audio_name) or PREPROCESSED_SEGMENT_RE.match(audio_name):
        return Path(audio_name).stem
    return None


def _subject_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def _chunk_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[1]


def _first_present_key(row: dict[str, str], *candidates: str) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def _required_row_value(row: dict[str, str], *candidates: str) -> str:
    key = _first_present_key(row, *candidates)
    if key is None:
        raise KeyError(
            "None of the expected columns were found in DAIC summary row. "
            f"Expected one of: {', '.join(candidates)} | available: {', '.join(sorted(row.keys()))}"
        )
    return str(row[key]).strip()


def _resolve_transcript_path(base_dir: Path, config: dict[str, Any]) -> Path:
    configured = str(config.get("transcript_path", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            base_dir / "preprocessed_whisper_transcripts.jsonl",
            base_dir / "whisper_transcripts.jsonl",
            base_dir.parent / "preprocessed_whisper_transcripts.jsonl",
            base_dir.parent / "whisper_transcripts.jsonl",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Could not locate a DAIC transcript cache. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_whisper_transcripts(path: Path) -> dict[str, dict[str, Any]]:
    transcripts: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_name = Path(str(row["audio_path"])).name
            sample_id = _sample_id_from_audio_name(audio_name)
            if sample_id is None:
                continue
            transcripts[sample_id] = {
                "transcript": str(row["transcript"]).strip(),
                "language": row.get("language", ""),
                "audio_name": audio_name,
            }
    return transcripts


def _build_subject_meta_from_rows(split_rows: dict[str, list[dict[str, str]]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    subject_meta: dict[str, dict[str, Any]] = {}
    split_lookup: dict[str, str] = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            subject_id = _required_row_value(row, "Participant_ID", "participant_id", "subject_id", "patient_id")
            label = int(_required_row_value(row, "PHQ8_Binary", "PHQ_Binary", "is_depressed", "label"))
            score_key = _first_present_key(row, "PHQ8_Score", "PHQ_Score", "PHQ8_SUM", "PHQ_SUM", "phq8_score", "phq_score")
            score_value = str(row.get(score_key, "")).strip() if score_key else ""
            gender_key = _first_present_key(row, "Gender", "gender")
            subject_meta[subject_id] = {
                "label": label,
                "label_text": label_text_from_int(label),
                "score": int(score_value) if score_value else "",
                "gender": row.get(gender_key) if gender_key else None,
            }
            split_lookup[subject_id] = split_name
    return subject_meta, split_lookup


def _discover_legacy_wavs(wav_dir: Path, subject_meta: dict[str, dict[str, Any]]) -> dict[str, Path]:
    wav_map: dict[str, Path] = {}
    for wav_path in sorted(wav_dir.glob("*.wav")):
        match = LEGACY_CHUNK_RE.match(wav_path.name)
        if not match:
            continue
        subject_id = match.group("subject_id")
        if subject_id not in subject_meta:
            continue
        wav_map[Path(wav_path.name).stem] = wav_path
    return wav_map


def _parse_segment_files(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _discover_preprocessed_wavs(
    base_dir: Path,
    split_rows: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    split_dirs = {
        "train": base_dir / "train_audio_segments",
        "val": base_dir / "dev_audio_segments",
        "test": base_dir / "test_audio_segments",
    }
    wav_map: dict[str, Path] = {}
    extra_file_audit: list[dict[str, Any]] = []

    expected_names_by_split: dict[str, set[str]] = {}
    for split_name, rows in split_rows.items():
        expected_names: set[str] = set()
        for row in rows:
            expected_names.update(_parse_segment_files(row.get("segment_files", "")))
        expected_names_by_split[split_name] = expected_names

    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            raise FileNotFoundError(f"DAIC preprocessed split directory is missing: {split_dir}")
        expected_names = expected_names_by_split.get(split_name, set())
        for wav_path in sorted(split_dir.glob("*/*.wav")):
            sample_id = wav_path.stem
            if _sample_id_from_audio_name(wav_path.name) is None:
                extra_file_audit.append(
                    {
                        "split": split_name,
                        "audio_path": str(wav_path),
                        "reason": "unexpected_filename_format",
                    }
                )
                continue
            if expected_names and wav_path.name not in expected_names:
                extra_file_audit.append(
                    {
                        "split": split_name,
                        "audio_path": str(wav_path),
                        "reason": "not_referenced_by_preprocessing_summary",
                    }
                )
                continue
            wav_map[sample_id] = wav_path
    return wav_map, extra_file_audit


def _build_full_transcript_rows(
    rows: list[dict[str, str]],
    transcript_source_path: Path,
) -> dict[str, dict[str, Any]]:
    transcripts: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject_id = _required_row_value(row, "Participant_ID", "participant_id", "subject_id", "patient_id")
        transcript = _required_row_value(row, "full_transcript")
        for segment_name in _parse_segment_files(row.get("segment_files", "")):
            sample_id = _sample_id_from_audio_name(segment_name)
            if sample_id is None:
                continue
            transcripts[sample_id] = {
                "transcript": transcript,
                "language": "",
                "audio_name": segment_name,
                "transcript_source_path": str(transcript_source_path),
            }
    return transcripts


def _finalize_manifest(
    *,
    quarantine: dict[str, Any],
    transcript_path: Path,
    transcripts: dict[str, dict[str, Any]],
    wav_map: dict[str, Path],
    subject_meta: dict[str, dict[str, Any]],
    split_lookup: dict[str, str],
) -> dict[str, Any]:
    sample_ids = sorted(
        set(wav_map) | {sample_id for sample_id in transcripts if _subject_id_from_sample_id(sample_id) in subject_meta}
    )
    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    missing_audio = 0
    missing_transcript = 0
    matched_samples = 0
    per_split_counts = defaultdict(Counter)

    for sample_id in sample_ids:
        subject_id = _subject_id_from_sample_id(sample_id)
        chunk_id = _chunk_id_from_sample_id(sample_id)
        split_name = split_lookup[subject_id]
        meta = subject_meta[subject_id]
        wav_path = wav_map.get(sample_id)
        transcript_entry = transcripts.get(sample_id)
        transcript_found = bool(transcript_entry and transcript_entry["transcript"])
        audio_found = wav_path is not None and wav_path.exists()
        if not audio_found:
            missing_audio += 1
        if not transcript_found:
            missing_transcript += 1
        join_audit_rows.append(
            {
                "subject_id": subject_id,
                "chunk_id": chunk_id,
                "sample_id": sample_id,
                "wav_path": str(wav_path) if wav_path else "",
                "audio_found": audio_found,
                "transcript_found": transcript_found,
                "split": split_name,
                "label": meta["label"],
                "label_text": meta["label_text"],
            }
        )
        if not audio_found or not transcript_found:
            if is_quarantined_missing(quarantine, "daic", sample_id):
                continue
            if not audio_found:
                raise FileNotFoundError(f"DAIC missing canonical audio for sample_id={sample_id}")
            raise ValueError(f"DAIC missing canonical transcript for sample_id={sample_id}")
        matched_samples += 1
        per_split_counts[split_name][meta["label"]] += 1
        manifest_rows.append(
            {
                "dataset": "daic",
                "subject_id": subject_id,
                "sample_id": sample_id,
                "audio_path": str(wav_path),
                "audio_paths": [str(wav_path)],
                "transcript": transcript_entry["transcript"],
                "transcript_path": str(transcript_entry.get("transcript_source_path", transcript_path)),
                "label": meta["label"],
                "label_text": meta["label_text"],
                "score": meta["score"],
                "split_original": split_name,
                "fold": "",
                "chunk_id": chunk_id,
                "question_id": "",
                "start_time": "",
                "end_time": "",
                "language": transcript_entry.get("language", ""),
                "gender": meta.get("gender"),
                "modality_mode": "single_audio_single_text",
            }
        )

    LOGGER.info(
        "DAIC join audit | missing_audio=%s missing_transcript=%s matched_samples=%s",
        missing_audio,
        missing_transcript,
        matched_samples,
    )
    for split_name, counts in sorted(per_split_counts.items()):
        LOGGER.info(
            "DAIC split=%s | depressed_samples=%s non_depressed_samples=%s",
            split_name,
            counts[1],
            counts[0],
        )

    subject_partition_rows = [
        {
            "subject_id": subject_id,
            "partition": split_name,
            "label": subject_meta[subject_id]["label"],
            "label_text": subject_meta[subject_id]["label_text"],
        }
        for subject_id, split_name in sorted(split_lookup.items())
    ]

    return {
        "manifest_rows": manifest_rows,
        "join_audit_rows": join_audit_rows,
        "subject_partition_rows": subject_partition_rows,
    }


def _build_preprocessed_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    split_rows = {
        "train": _load_csv_rows(base_dir / "train_preprocessing_summary.csv"),
        "val": _load_csv_rows(base_dir / "dev_preprocessing_summary.csv"),
        "test": _load_csv_rows(base_dir / "test_preprocessing_summary.csv"),
    }
    subject_meta, split_lookup = _build_subject_meta_from_rows(split_rows)
    transcript_path = base_dir
    transcripts: dict[str, dict[str, Any]] = {}
    for split_name, summary_name in (
        ("train", "train_preprocessing_summary.csv"),
        ("val", "dev_preprocessing_summary.csv"),
        ("test", "test_preprocessing_summary.csv"),
    ):
        summary_path = base_dir / summary_name
        transcripts.update(_build_full_transcript_rows(split_rows[split_name], summary_path))
    wav_map, extra_file_audit = _discover_preprocessed_wavs(base_dir, split_rows)
    result = _finalize_manifest(
        quarantine=quarantine,
        transcript_path=transcript_path,
        transcripts=transcripts,
        wav_map=wav_map,
        subject_meta=subject_meta,
        split_lookup=split_lookup,
    )
    result["extra_file_audit"] = extra_file_audit
    result["split_source"] = PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT
    result["split_source_notes"] = (
        "Uses train_preprocessing_summary.csv, dev_preprocessing_summary.csv, and "
        "test_preprocessing_summary.csv from the preprocessed DAIC layout. Each chunk "
        "listed in segment_files is paired with the participant's full_transcript from "
        "its split summary row, so train, val, and test all use repeated full transcripts."
    )
    return result


def _build_preprocessed_test_full_transcript_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    transcript_path = base_dir / "test_preprocessing_summary.csv"
    split_rows = {
        "test": _load_csv_rows(transcript_path),
    }
    subject_meta, split_lookup = _build_subject_meta_from_rows(split_rows)
    transcripts = _build_full_transcript_rows(split_rows["test"], transcript_path)
    wav_map, extra_file_audit = _discover_preprocessed_wavs(base_dir, split_rows)
    result = _finalize_manifest(
        quarantine=quarantine,
        transcript_path=transcript_path,
        transcripts=transcripts,
        wav_map=wav_map,
        subject_meta=subject_meta,
        split_lookup=split_lookup,
    )
    result["extra_file_audit"] = extra_file_audit
    result["split_source"] = PREPROCESSED_TEST_FULL_TRANSCRIPT_VARIANT
    result["split_source_notes"] = (
        "Uses test_preprocessing_summary.csv and preprocessed/test_audio_segments. "
        "Each test chunk is paired with the participant's full_transcript rather than "
        "a segment-aligned transcript."
    )
    return result


def _build_legacy_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    official_dir = base_dir / "minimal_zips"
    wav_dir = official_dir / "preprocessed_audios"
    transcript_path = _resolve_transcript_path(base_dir, config)

    split_rows = {
        "train": _load_csv_rows(official_dir / "train_split_Depression_AVEC2017 (1).csv"),
        "dev": _load_csv_rows(official_dir / "dev_split_Depression_AVEC2017.csv"),
        "test": _load_csv_rows(official_dir / "full_test_split (1).csv"),
    }
    subject_meta, split_lookup = _build_subject_meta_from_rows(split_rows)
    transcripts = _load_whisper_transcripts(transcript_path)
    wav_map = _discover_legacy_wavs(wav_dir, subject_meta)
    result = _finalize_manifest(
        quarantine=quarantine,
        transcript_path=transcript_path,
        transcripts=transcripts,
        wav_map=wav_map,
        subject_meta=subject_meta,
        split_lookup=split_lookup,
    )
    result["split_source"] = "official_train_dev_test"
    result["split_source_notes"] = "Uses the original DAIC split CSV files under minimal_zips."
    return result


def build_daic_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    manifest_variant = str(config.get("manifest_variant", "")).strip().lower()
    result: dict[str, Any]
    if manifest_variant in {PREPROCESSED_TRAIN_DEV_VARIANT, PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT}:
        LOGGER.info("Building DAIC manifest from preprocessed train/dev/test full-transcript layout: %s", base_dir)
        result = _build_preprocessed_manifest(config, quarantine)
    elif manifest_variant == PREPROCESSED_TEST_FULL_TRANSCRIPT_VARIANT:
        LOGGER.info("Building DAIC manifest from preprocessed test layout with repeated full transcripts: %s", base_dir)
        result = _build_preprocessed_test_full_transcript_manifest(config, quarantine)
    elif manifest_variant:
        raise ValueError(
            f"Unsupported DAIC manifest_variant={manifest_variant!r}. "
            f"Expected one of: {PREPROCESSED_TRAIN_DEV_VARIANT!r}, "
            f"{PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT!r}, "
            f"{PREPROCESSED_TEST_FULL_TRANSCRIPT_VARIANT!r}, or empty for auto-detect."
        )
    elif (base_dir / "train_preprocessing_summary.csv").exists():
        LOGGER.info("Building DAIC manifest from preprocessed full-transcript all-splits layout: %s", base_dir)
        result = _build_preprocessed_manifest(config, quarantine)
    else:
        LOGGER.info("Building DAIC manifest from legacy layout: %s", base_dir)
        result = _build_legacy_manifest(config, quarantine)

    partition_rows = result.get("subject_partition_rows") or []
    if not partition_rows:
        return result
    dev_pool_partitions = resolve_dev_pool_partitions(config)
    dev_partition_set = set(dev_pool_partitions)
    dev_subject_ids = sorted([row["subject_id"] for row in partition_rows if row["partition"] in dev_partition_set])
    if not dev_subject_ids:
        LOGGER.info(
            "Skipping DAIC CV fold generation because no subjects were found in development-pool partitions: %s",
            dev_pool_partitions,
        )
        return result
    final_eval_partition = str(config["split"]["final_eval_partition"])
    final_eval_subject_ids = sorted([row["subject_id"] for row in partition_rows if row["partition"] == final_eval_partition])
    overlap = sorted(set(dev_subject_ids).intersection(final_eval_subject_ids))
    if overlap:
        LOGGER.warning(
            "Skipping DAIC CV fold generation because development-pool partitions %s overlap final_eval_partition=%s.",
            dev_pool_partitions,
            final_eval_partition,
        )
        return result
    subject_labels = {row["subject_id"]: int(row["label"]) for row in partition_rows}
    folds = build_partition_scoped_stratified_folds(
        partition_rows=partition_rows,
        subject_labels=subject_labels,
        dev_pool_partitions=dev_pool_partitions,
        final_eval_partition=final_eval_partition,
        n_splits=resolve_outer_fold_count(config),
        seed=int(config["split"]["seed"]),
    )
    result["folds"] = folds
    result["fold_report"] = subject_fold_report(folds, subject_labels)
    return result
