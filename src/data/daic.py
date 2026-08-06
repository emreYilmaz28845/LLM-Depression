from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from src.data.split_utils import (
    build_partition_scoped_stratified_folds,
    resolve_dev_pool_partitions,
    resolve_outer_fold_count,
    subject_fold_report,
)
from src.data.validation import is_quarantined_missing
from src.utils import ensure_dir, get_logger, label_text_from_int, sha256_file, write_jsonl


LOGGER = get_logger(__name__)
LEGACY_CHUNK_RE = re.compile(r"^(?P<subject_id>\d+)_(?P<chunk_id>\d+)\.wav$")
PREPROCESSED_SEGMENT_RE = re.compile(
    r"^(?P<subject_id>\d+)_(?P<segment_kind>random_segment|segment)_(?P<chunk_id>\d+)\.wav$"
)
PREPROCESSED_TRAIN_DEV_VARIANT = "preprocessed_train_dev"
PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT = "preprocessed_full_transcript_all_splits"
PREPROCESSED_TEST_FULL_TRANSCRIPT_VARIANT = "preprocessed_test_full_transcript"

PACKED30_PROTOCOL_ID = "daic_participant_speech_packed30_v1"
PACKED30_MANIFEST_VARIANT = "unprocessed_participant_speech_packed30_v1"
PACKED30_SCHEMA_VERSION = "daic_participant_speech_packed30_manifest.v1"
PACKED30_CHUNK_SAMPLES = 480000
PACKED30_SAMPLE_RATE = 16000
PACKED30_DEFAULT_TRANSCRIPT_MAX_CHARS = 4000
PACKED30_MANIFEST_JSONL = "daic_participant_speech_packed30_manifest.jsonl"
PACKED30_SUBJECTS_JSONL = "daic_participant_speech_packed30_subjects.jsonl"
PACKED30_JOIN_AUDIT_JSONL = "daic_participant_speech_packed30_join_audit.jsonl"
PACKED30_CORPUS_AUDIT_JSON = "daic_participant_speech_packed30_corpus_audit.json"
PACKED30_METADATA_JSON = "daic_participant_speech_packed30_metadata.json"
PACKED30_OFFICIAL_SPLIT_ORDER = ("train", "val", "test")
PACKED30_EXPECTED_TOTALS = {
    "subjects": {"train": 107, "val": 35, "test": 47, "total": 189},
    "blank_lines": 14,
    "nonblank_rows": 47400,
    "excluded_non_participant_rows": 14996,
    "excluded_empty_participant_rows": 27,
    "retained_rows": 32373,
    "retained_frames": 1406614000,
    "speech_seconds": 87913.375,
    "chunks": 3021,
    "chunks_by_split_label": {
        "train": {0: 1148, 1: 450},
        "val": {0: 363, 1: 240},
        "test": {0: 595, 1: 225},
    },
    "chunks_per_subject_range": (3, 43),
    "final_chunk_samples_range": (1440, 478480),
}
# The four known source rows that extend past their WAV bounds. The audit MUST
# validate timestamps for all rows (including excluded ones) and match this
# allowlist exactly; any other invalid row is a STOP condition.
PACKED30_INVALID_ROW_ALLOWLIST = (
    {
        "subject_id": "381",
        "start_time": 1078.550,
        "stop_time": 1089.320,
        "speaker": "Participant",
        "reason": "empty value; excluded before packing",
    },
    {
        "subject_id": "402",
        "start_time": 965.496,
        "stop_time": 967.936,
        "speaker": "Ellie",
        "reason": "excluded speaker",
    },
    {
        "subject_id": "402",
        "start_time": 968.853,
        "stop_time": 970.293,
        "speaker": "Ellie",
        "reason": "excluded speaker",
    },
    {
        "subject_id": "402",
        "start_time": 971.641,
        "stop_time": 972.251,
        "speaker": "Ellie",
        "reason": "excluded speaker",
    },
)


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


def _parse_participant_transcript_tsv(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Parse a DAIC time-aligned transcript TSV (UTF-8-SIG, tab delimiter).

    Returns ``(parsed_rows, blank_line_count)``. Each parsed row carries
    ``source_row_index`` (zero-based index among non-blank data rows after the
    header), ``start_time``/``stop_time`` as raw strings, a whitespace-stripped
    ``speaker`` and ``value``. The only allowed non-empty speakers are ``Ellie``
    and ``Participant``.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty DAIC transcript: {path}")
    header = [str(cell).strip() for cell in rows[0]]
    if header[:4] != ["start_time", "stop_time", "speaker", "value"]:
        raise ValueError(f"Unexpected DAIC transcript header {header!r}: {path}")
    parsed: list[dict[str, Any]] = []
    blank_lines = 0
    for raw_row in rows[1:]:
        if not any(str(cell).strip() for cell in raw_row):
            blank_lines += 1
            continue
        if len(raw_row) != 4:
            raise ValueError(f"Malformed DAIC transcript row {raw_row!r}: {path}")
        parsed.append(
            {
                "source_row_index": len(parsed),
                "start_time": str(raw_row[0]).strip(),
                "stop_time": str(raw_row[1]).strip(),
                "speaker": str(raw_row[2]).strip(),
                "value": str(raw_row[3]).strip(),
            }
        )
    for row in parsed:
        if row["speaker"] not in {"Ellie", "Participant"}:
            raise ValueError(
                f"Unexpected speaker {row['speaker']!r} in {path}. "
                "Only Ellie and Participant are allowed."
            )
    return parsed, blank_lines


def _parse_row_timestamps(row: dict[str, Any]) -> tuple[tuple[float, float] | None, str | None]:
    try:
        start_time = float(row["start_time"])
        stop_time = float(row["stop_time"])
    except (TypeError, ValueError):
        return None, "non_finite_or_unparsable_time"
    if not (math.isfinite(start_time) and math.isfinite(stop_time)):
        return None, "non_finite_or_unparsable_time"
    return (start_time, stop_time), None


def _validate_row_interval(
    row: dict[str, Any], wav_frames: int
) -> tuple[tuple[float, float] | None, str | None]:
    """Validate the source timestamp contract for a row (step 4-7 of Section 3.2).

    Returns ``((start_time, stop_time), None)`` on success or
    ``(None, reason)`` on failure. A timestamp contract failure means the row is
    invalid (and, unless allowlisted, fails the build).
    """
    times, parse_reason = _parse_row_timestamps(row)
    if times is None:
        return None, parse_reason
    start_time, stop_time = times
    if not (0.0 <= start_time < stop_time <= wav_frames / PACKED30_SAMPLE_RATE):
        return None, "out_of_wav_bounds"
    return times, None


def _round3(value: float) -> float:
    return round(float(value), 3)


def _allowlist_keys() -> set[tuple[str, float, float, str]]:
    return {
        (
            str(item["subject_id"]),
            _round3(float(item["start_time"])),
            _round3(float(item["stop_time"])),
            str(item["speaker"]),
        )
        for item in PACKED30_INVALID_ROW_ALLOWLIST
    }


def _audit_subject_source_rows(
    subject_id: str,
    wav_frames: int,
    parsed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the Section 3.2 filter/validation to one subject's transcript rows.

    Every parsed row is timestamp-validated (including rows later excluded by
    the speaker/non-empty filter). The invalid rows MUST match the locked
    allowlist exactly. Retained Participant intervals are then checked for
    chronological non-overlap (equal adjacency allowed).
    """
    invalid_rows: list[dict[str, Any]] = []
    for row in parsed_rows:
        times, reason = _validate_row_interval(row, wav_frames)
        if times is None:
            invalid_rows.append({**row, "invalid_reason": reason})

    def _invalid_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
        times, _ = _parse_row_timestamps(row)
        if times is None:
            return (subject_id, str(row["start_time"]), str(row["stop_time"]), str(row["speaker"]))
        start_time, stop_time = times
        return (subject_id, _round3(start_time), _round3(stop_time), str(row["speaker"]))

    observed_keys = {_invalid_row_key(row) for row in invalid_rows}
    expected_keys = {key for key in _allowlist_keys() if key[0] == subject_id}
    if observed_keys != expected_keys:
        raise ValueError(
            f"DAIC packed30 invalid-source-row set mismatch for subject_id={subject_id}: "
            f"observed={sorted(observed_keys)} expected={sorted(expected_keys)}"
        )
    invalid_row_indices = [int(row["source_row_index"]) for row in invalid_rows]
    if len(invalid_row_indices) != len(set(invalid_row_indices)):
        raise ValueError(
            f"DAIC packed30 invalid rows are not uniquely identified for subject_id={subject_id}."
        )
    invalid_row_indices = set(invalid_row_indices)

    retained_intervals: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in parsed_rows:
        source_row_index = int(row["source_row_index"])
        if source_row_index in invalid_row_indices:
            exclusions.append(
                {
                    "subject_id": subject_id,
                    "source_row_index": source_row_index,
                    "start_time": float(row["start_time"]),
                    "stop_time": float(row["stop_time"]),
                    "speaker": row["speaker"],
                    "reason": "invalid_allowlisted_row",
                    "invalid_reason": next(
                        invalid["invalid_reason"]
                        for invalid in invalid_rows
                        if int(invalid["source_row_index"]) == source_row_index
                    ),
                }
            )
            continue
        if row["speaker"] != "Participant":
            exclusions.append(
                {
                    "subject_id": subject_id,
                    "source_row_index": source_row_index,
                    "start_time": float(row["start_time"]),
                    "stop_time": float(row["stop_time"]),
                    "speaker": row["speaker"],
                    "reason": "excluded_non_participant",
                }
            )
            continue
        if not row["value"]:
            exclusions.append(
                {
                    "subject_id": subject_id,
                    "source_row_index": source_row_index,
                    "start_time": float(row["start_time"]),
                    "stop_time": float(row["stop_time"]),
                    "speaker": row["speaker"],
                    "reason": "excluded_empty_participant_text",
                }
            )
            continue
        times, reason = _validate_row_interval(row, wav_frames)
        if times is None:
            raise AssertionError(
                f"Retained Participant row failed validation for subject_id={subject_id}: "
                f"row_index={source_row_index} reason={reason}"
            )
        start_time, stop_time = times
        start_frame = int(round(start_time * PACKED30_SAMPLE_RATE))
        end_frame = int(round(stop_time * PACKED30_SAMPLE_RATE))
        if not (0 <= start_frame < end_frame <= wav_frames):
            raise ValueError(
                f"Converted frame interval out of WAV bounds for subject_id={subject_id} "
                f"row_index={source_row_index}: [{start_frame}, {end_frame}) not within [0, {wav_frames})."
            )
        retained_intervals.append(
            {
                "source_row_index": source_row_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "stop_time": stop_time,
                "value": row["value"],
            }
        )
    retained_intervals.sort(key=lambda interval: interval["start_frame"])
    previous_end: int | None = None
    for interval in retained_intervals:
        if previous_end is not None and interval["start_frame"] < previous_end:
            raise ValueError(
                f"Retained Participant intervals overlap for subject_id={subject_id}: "
                f"row_index={interval['source_row_index']} starts at {interval['start_frame']} "
                f"but previous interval ends at {previous_end}."
            )
        previous_end = interval["end_frame"]
    return {
        "invalid_rows": invalid_rows,
        "retained_intervals": retained_intervals,
        "exclusions": exclusions,
    }


def _pack_retained_intervals(
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Divide the chronological participant-speech stream into 480000-sample chunks.

    Implements the Section 3.2 state machine exactly: no silence is inserted,
    long turns are split at chunk boundaries, short turns are concatenated, and
    the final partial chunk is kept whenever it contains at least one sample.
    """
    chunks: list[dict[str, Any]] = []
    current_spans: list[dict[str, Any]] = []
    current_samples = 0
    chunk_index = 0
    for interval in intervals:
        cursor = int(interval["start_frame"])
        end_frame = int(interval["end_frame"])
        while cursor < end_frame:
            capacity = PACKED30_CHUNK_SAMPLES - current_samples
            take = min(capacity, end_frame - cursor)
            current_spans.append(
                {
                    "start_frame": cursor,
                    "end_frame": cursor + take,
                    "source_row_index": int(interval["source_row_index"]),
                    "source_start_time": float(interval["start_time"]),
                    "source_stop_time": float(interval["stop_time"]),
                }
            )
            cursor += take
            current_samples += take
            if current_samples == PACKED30_CHUNK_SAMPLES:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "participant_sample_count": PACKED30_CHUNK_SAMPLES,
                        "spans": current_spans,
                    }
                )
                chunk_index += 1
                current_spans = []
                current_samples = 0
    if current_samples > 0:
        chunks.append(
            {
                "chunk_index": chunk_index,
                "participant_sample_count": current_samples,
                "spans": current_spans,
            }
        )
    return chunks


def _truncate_packed30_text(text: str, max_chars: int) -> tuple[str, dict[str, Any] | None]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, None
    return text[:max_chars], {
        "transcript_original_chars": len(text),
        "transcript_kept_chars": max_chars,
        "transcript_truncated": True,
    }


def _chunk_transcript_text(spans: list[dict[str, Any]], intervals_by_row: dict[int, str]) -> str:
    values: list[str] = []
    for span in spans:
        value = intervals_by_row.get(int(span["source_row_index"]), "")
        if value:
            values.append(value)
    return "\n".join(values)


def _source_row_span_coverage(intervals: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (retained_frames, retained_row_count) for a subject."""
    retained_frames = sum(int(interval["end_frame"]) - int(interval["start_frame"]) for interval in intervals)
    return retained_frames, len(intervals)


def _point_biserial_diagnostics(values: list[float], labels: list[int]) -> dict[str, Any]:
    """Point-biserial r, two-sided p-value, and Mann-Whitney AUROC.

    Audit diagnostics only (Section 8); they are never used as model features or
    protocol decisions. Requires scipy (present in the local audit environment).
    """
    from scipy.stats import mannwhitneyu, pointbiserialr

    if len(values) != len(labels):
        raise ValueError("Diagnostic value/label vectors must have equal length.")
    if len(set(labels)) != 2:
        raise ValueError("Diagnostic cohort must contain both labels.")
    zero_values = [value for value, label in zip(values, labels) if int(label) == 0]
    one_values = [value for value, label in zip(values, labels) if int(label) == 1]
    r_value, p_value = pointbiserialr(labels, values)
    try:
        u_stat, mw_p_value = mannwhitneyu(
            one_values,
            zero_values,
            alternative="two-sided",
            method="asymptotic",
        )
    except ValueError:
        u_stat, mw_p_value = float("nan"), float("nan")
    n0, n1 = len(zero_values), len(one_values)
    mw_auroc = float(u_stat / (n0 * n1)) if n0 and n1 else float("nan")
    import statistics

    return {
        "class_count": {"0": len(zero_values), "1": len(one_values)},
        "class_median": {
            "0": float(statistics.median(zero_values)),
            "1": float(statistics.median(one_values)),
        },
        "point_biserial_r": float(r_value),
        "point_biserial_p_value": float(p_value),
        "mann_whitney_auroc_label1_vs_label0": float(mw_auroc),
        "mann_whitney_p_value": float(mw_p_value),
    }


def _source_hash(path: Path) -> str:
    return sha256_file(path)


def _build_participant_packed30_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    """Build the v1 participant-only packed-30s manifest and corpus audit.

    Source audio is the original ``unprocessed/<subject>_AUDIO.wav`` only and
    labels/splits come from the sibling ``minimal_zips/`` official CSVs only.
    The manifest records ordered integer-frame spans; no derived WAVs are
    materialized.
    """
    from src.data.build_manifest import manifest_build_signature

    unprocessed_root = Path(config["dataset_root"])
    label_root = Path(config["label_root"])
    if not unprocessed_root.is_dir():
        raise FileNotFoundError(f"DAIC unprocessed dataset root is missing: {unprocessed_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"DAIC label root is missing: {label_root}")
    daic_quarantine = (quarantine.get("datasets") or {}).get("daic", {}) or {}
    if daic_quarantine.get("missing_samples"):
        raise ValueError(
            "v1 forbids DAIC quarantines; configs/quarantines.yaml daic.missing_samples "
            "MUST remain empty for the participant-packed30 protocol."
        )
    split_rows = {
        "train": _load_csv_rows(label_root / "train_split_Depression_AVEC2017 (1).csv"),
        "val": _load_csv_rows(label_root / "dev_split_Depression_AVEC2017.csv"),
        "test": _load_csv_rows(label_root / "full_test_split (1).csv"),
    }
    subject_meta, split_lookup = _build_subject_meta_from_rows(split_rows)
    for split_name, rows in split_rows.items():
        expected = PACKED30_EXPECTED_TOTALS["subjects"][split_name]
        if len(rows) != expected:
            raise ValueError(
                f"DAIC official split {split_name!r} has {len(rows)} subjects; expected {expected}."
            )
    split_order = {name: index for index, name in enumerate(PACKED30_OFFICIAL_SPLIT_ORDER)}
    ordered_subjects = sorted(
        subject_meta,
        key=lambda subject_id: (split_order[split_lookup[subject_id]], int(subject_id)),
    )
    subject_ids_by_split: dict[str, list[str]] = defaultdict(list)
    for subject_id in ordered_subjects:
        subject_ids_by_split[split_lookup[subject_id]].append(subject_id)
    all_subject_ids = list(ordered_subjects)
    transcript_max_chars = int(config.get("data", {}).get("transcript_max_chars", PACKED30_DEFAULT_TRANSCRIPT_MAX_CHARS) or 0)
    if transcript_max_chars <= 0:
        transcript_max_chars = PACKED30_DEFAULT_TRANSCRIPT_MAX_CHARS

    manifest_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    chunks_by_split_label: dict[str, Counter] = defaultdict(Counter)
    final_chunk_samples: list[int] = []
    chunks_per_subject: dict[str, int] = {}
    retained_row_counts: dict[str, int] = {}
    blank_lines_total = 0
    excluded_non_participant_total = 0
    excluded_empty_total = 0
    invalid_allowlisted_total = 0
    wav_info_by_subject: dict[str, dict[str, Any]] = {}

    for subject_id in ordered_subjects:
        split_name = split_lookup[subject_id]
        meta = subject_meta[subject_id]
        wav_path = unprocessed_root / f"{subject_id}_AUDIO.wav"
        transcript_path = unprocessed_root / f"{subject_id}_TRANSCRIPT.csv"
        if not wav_path.is_file():
            raise FileNotFoundError(f"DAIC packed30 missing source WAV: {wav_path}")
        if not transcript_path.is_file():
            raise FileNotFoundError(f"DAIC packed30 missing source transcript: {transcript_path}")
        info = sf.info(wav_path)
        if int(info.samplerate) != PACKED30_SAMPLE_RATE or int(info.channels) != 1:
            raise ValueError(
                f"DAIC packed30 requires 16 kHz mono WAVs; subject_id={subject_id} is "
                f"{info.samplerate} Hz / {info.channels} channel(s): {wav_path}"
            )
        if str(info.subtype) != "PCM_16":
            raise ValueError(
                f"DAIC packed30 requires PCM16 WAVs; subject_id={subject_id} subtype={info.subtype!r}."
            )
        wav_info_by_subject[subject_id] = {
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "frames": int(info.frames),
            "duration_seconds": float(info.frames / PACKED30_SAMPLE_RATE),
        }
        parsed_rows, blank_lines = _parse_participant_transcript_tsv(transcript_path)
        blank_lines_total += blank_lines
        audit = _audit_subject_source_rows(subject_id, int(info.frames), parsed_rows)
        invalid_allowlisted_total += len(audit["invalid_rows"])
        excluded_non_participant_total += sum(
            1 for exclusion in audit["exclusions"] if exclusion["reason"] == "excluded_non_participant"
        )
        excluded_empty_total += sum(
            1
            for exclusion in audit["exclusions"]
            if exclusion["reason"] == "excluded_empty_participant_text"
        )
        intervals = audit["retained_intervals"]
        retained_frames, retained_rows = _source_row_span_coverage(intervals)
        retained_row_counts[subject_id] = retained_rows
        chunks = _pack_retained_intervals(intervals)
        chunks_per_subject[subject_id] = len(chunks)
        values_by_row = {int(interval["source_row_index"]): interval["value"] for interval in intervals}
        full_transcript = "\n".join(interval["value"] for interval in intervals)
        full_transcript, truncation_log = _truncate_packed30_text(full_transcript, transcript_max_chars)
        full_transcript_sha256 = hashlib.sha256(full_transcript.encode("utf-8")).hexdigest()
        num_chunks = len(chunks)
        for chunk in chunks:
            chunk_index = int(chunk["chunk_index"])
            sample_id = f"{subject_id}_participant_p30_{chunk_index:03d}"
            chunks_by_split_label[split_name][int(meta["label"])] += 1
            if chunk_index == num_chunks - 1:
                final_chunk_samples.append(int(chunk["participant_sample_count"]))
            manifest_rows.append(
                {
                    "schema_version": PACKED30_SCHEMA_VERSION,
                    "protocol_id": PACKED30_PROTOCOL_ID,
                    "dataset": "daic",
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "audio_path": str(wav_path),
                    "audio_spans": chunk["spans"],
                    "participant_sample_count": int(chunk["participant_sample_count"]),
                    "chunk_index": chunk_index,
                    "num_chunks": num_chunks,
                    "chunk_transcript": _chunk_transcript_text(chunk["spans"], values_by_row),
                    "full_participant_transcript": full_transcript,
                    "full_participant_transcript_sha256": full_transcript_sha256,
                    "transcript": full_transcript,
                    "label": int(meta["label"]),
                    "label_text": meta["label_text"],
                    "split_original": split_name,
                }
            )
        for exclusion in audit["exclusions"]:
            join_audit_rows.append(
                {
                    "subject_id": subject_id,
                    "source_row_index": int(exclusion["source_row_index"]),
                    "start_time": exclusion["start_time"],
                    "stop_time": exclusion["stop_time"],
                    "speaker": exclusion["speaker"],
                    "value_empty": False,
                    "status": exclusion["reason"],
                    "invalid_reason": exclusion.get("invalid_reason", ""),
                }
            )
        for interval in intervals:
            assigned_chunks = [
                int(chunk["chunk_index"])
                for chunk in chunks
                if any(int(span["source_row_index"]) == int(interval["source_row_index"]) for span in chunk["spans"])
            ]
            join_audit_rows.append(
                {
                    "subject_id": subject_id,
                    "source_row_index": int(interval["source_row_index"]),
                    "start_time": interval["start_time"],
                    "stop_time": interval["stop_time"],
                    "speaker": "Participant",
                    "value_empty": False,
                    "status": "retained",
                    "invalid_reason": "",
                    "frame_start": int(interval["start_frame"]),
                    "frame_end": int(interval["end_frame"]),
                    "chunk_indices": assigned_chunks,
                }
            )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "split_original": split_name,
                "label": int(meta["label"]),
                "label_text": meta["label_text"],
                "audio_path": str(wav_path),
                "transcript_path": str(transcript_path),
                "wav_frames": int(info.frames),
                "wav_duration_seconds": float(info.frames / PACKED30_SAMPLE_RATE),
                "retained_participant_rows": retained_rows,
                "retained_frames": retained_frames,
                "participant_speech_seconds": float(retained_frames / PACKED30_SAMPLE_RATE),
                "chunk_count": num_chunks,
                "final_chunk_samples": (
                    chunks[-1]["participant_sample_count"] if chunks else 0
                ),
                "full_participant_transcript_sha256": full_transcript_sha256,
                "full_participant_transcript_chars": len(full_transcript),
                "full_participant_transcript_truncated": bool(truncation_log),
                "audio_sha256": _source_hash(wav_path),
                "transcript_sha256": _source_hash(transcript_path),
            }
        )

    manifest_rows.sort(
        key=lambda row: (
            split_order[row["split_original"]],
            int(row["subject_id"]),
            int(row["chunk_index"]),
        )
    )
    total_retained_frames = sum(int(row["participant_sample_count"]) for row in manifest_rows)
    total_retained_rows = sum(retained_row_counts.values())
    total_chunks = len(manifest_rows)
    total_subjects = len(subject_rows)
    expected = PACKED30_EXPECTED_TOTALS
    if blank_lines_total != expected["blank_lines"]:
        raise ValueError(
            f"DAIC packed30 blank-line count {blank_lines_total} != expected {expected['blank_lines']}."
        )
    if excluded_non_participant_total != expected["excluded_non_participant_rows"]:
        raise ValueError(
            f"DAIC packed30 excluded non-participant rows {excluded_non_participant_total} "
            f"!= expected {expected['excluded_non_participant_rows']}."
        )
    if excluded_empty_total != expected["excluded_empty_participant_rows"]:
        raise ValueError(
            f"DAIC packed30 excluded empty rows {excluded_empty_total} != expected "
            f"{expected['excluded_empty_participant_rows']}."
        )
    nonblank_rows = total_retained_rows + excluded_non_participant_total + excluded_empty_total + invalid_allowlisted_total
    if nonblank_rows != expected["nonblank_rows"]:
        raise ValueError(f"DAIC packed30 nonblank row total {nonblank_rows} != expected {expected['nonblank_rows']}.")
    if total_retained_rows != expected["retained_rows"]:
        raise ValueError(f"DAIC packed30 retained rows {total_retained_rows} != expected {expected['retained_rows']}.")
    if total_retained_frames != expected["retained_frames"]:
        raise ValueError(
            f"DAIC packed30 retained frames {total_retained_frames} != expected {expected['retained_frames']}."
        )
    if total_chunks != expected["chunks"]:
        raise ValueError(f"DAIC packed30 emitted chunks {total_chunks} != expected {expected['chunks']}.")
    observed_chunks_by_split_label = {
        split_name: dict(chunks_by_split_label.get(split_name, Counter()))
        for split_name in PACKED30_OFFICIAL_SPLIT_ORDER
    }
    if observed_chunks_by_split_label != expected["chunks_by_split_label"]:
        raise ValueError(
            f"DAIC packed30 chunk split/label totals mismatch: {observed_chunks_by_split_label}"
        )
    if len(final_chunk_samples) != total_subjects or not final_chunk_samples:
        raise ValueError("DAIC packed30 expects exactly one final chunk per subject.")
    if (min(final_chunk_samples), max(final_chunk_samples)) != expected["final_chunk_samples_range"]:
        raise ValueError(
            f"DAIC packed30 final-chunk range {(min(final_chunk_samples), max(final_chunk_samples))} "
            f"!= expected {expected['final_chunk_samples_range']}."
        )
    chunk_count_values = sorted(chunks_per_subject.values())
    if (min(chunk_count_values), max(chunk_count_values)) != expected["chunks_per_subject_range"]:
        raise ValueError(
            f"DAIC packed30 per-subject chunk range {(min(chunk_count_values), max(chunk_count_values))} "
            f"!= expected {expected['chunks_per_subject_range']}."
        )
    coverage_total = sum(int(row["participant_sample_count"]) for row in manifest_rows)
    if coverage_total != total_retained_frames:
        raise ValueError(
            f"DAIC packed30 chunk sample total {coverage_total} != retained frame total {total_retained_frames}."
        )

    def diagnostics_for(feature_name: str, feature_fn) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        for cohort_name, cohort_ids in (
            ("train", subject_ids_by_split["train"]),
            ("val", subject_ids_by_split["val"]),
            ("test", subject_ids_by_split["test"]),
            ("all", all_subject_ids),
        ):
            values = [feature_fn(subject_id) for subject_id in cohort_ids]
            labels = [int(subject_meta[subject_id]["label"]) for subject_id in cohort_ids]
            diagnostics[cohort_name] = _point_biserial_diagnostics(values, labels)
        return diagnostics

    speech_seconds_by_subject = {row["subject_id"]: float(row["participant_speech_seconds"]) for row in subject_rows}
    feature_functions = {
        "full_interview_duration_seconds": lambda subject_id: wav_info_by_subject[subject_id]["duration_seconds"],
        "participant_speech_seconds": lambda subject_id: speech_seconds_by_subject[subject_id],
        "retained_participant_row_count": lambda subject_id: float(retained_row_counts[subject_id]),
        "chunk_count": lambda subject_id: float(chunks_per_subject[subject_id]),
    }
    diagnostics = {
        feature_name: diagnostics_for(feature_name, feature_fn)
        for feature_name, feature_fn in feature_functions.items()
    }

    invalid_rows_audit = []
    for row in join_audit_rows:
        if row["status"] != "invalid_allowlisted_row":
            continue
        allowlisted = next(
            item
            for item in PACKED30_INVALID_ROW_ALLOWLIST
            if item["subject_id"] == str(row["subject_id"])
            and _round3(float(item["start_time"])) == _round3(float(row["start_time"]))
            and _round3(float(item["stop_time"])) == _round3(float(row["stop_time"]))
        )
        invalid_rows_audit.append(
            {
                "subject_id": str(row["subject_id"]),
                "start_time": float(row["start_time"]),
                "stop_time": float(row["stop_time"]),
                "speaker": str(row["speaker"]),
                "reason": str(allowlisted["reason"]),
                "invalid_reason": str(row["invalid_reason"]),
            }
        )

    corpus_audit = {
        "protocol_id": PACKED30_PROTOCOL_ID,
        "schema_version": PACKED30_SCHEMA_VERSION,
        "totals": {
            "subjects": dict(Counter(split_lookup[subject_id] for subject_id in ordered_subjects)),
            "subjects_by_label": dict(Counter(int(subject_meta[subject_id]["label"]) for subject_id in ordered_subjects)),
            "retained_rows": total_retained_rows,
            "retained_frames": total_retained_frames,
            "speech_seconds": float(total_retained_frames / PACKED30_SAMPLE_RATE),
            "chunks": total_chunks,
            "chunks_by_split_label": observed_chunks_by_split_label,
            "blank_lines": blank_lines_total,
            "nonblank_rows": nonblank_rows,
            "excluded_non_participant_rows": excluded_non_participant_total,
            "excluded_empty_participant_rows": excluded_empty_total,
            "invalid_allowlisted_rows": invalid_allowlisted_total,
            "chunks_per_subject_range": [min(chunk_count_values), max(chunk_count_values)],
            "final_chunk_samples_range": [min(final_chunk_samples), max(final_chunk_samples)],
            "final_chunk_count": len(final_chunk_samples),
        },
        "invalid_rows": invalid_rows_audit,
        "diagnostics": diagnostics,
        "source_hashes": {
            "wav_subject_sha256": {row["subject_id"]: row["audio_sha256"] for row in subject_rows},
            "transcript_subject_sha256": {row["subject_id"]: row["transcript_sha256"] for row in subject_rows},
        },
        "locked_contract": {
            "chunk_samples": PACKED30_CHUNK_SAMPLES,
            "sample_rate": PACKED30_SAMPLE_RATE,
            "expected_totals": expected,
            "invalid_row_allowlist": list(PACKED30_INVALID_ROW_ALLOWLIST),
        },
    }

    artifact_dir = Path(config["output_dirs"]["manifest_dir"])
    artifact_dir = ensure_dir(artifact_dir)
    manifest_path = artifact_dir / PACKED30_MANIFEST_JSONL
    subjects_path = artifact_dir / PACKED30_SUBJECTS_JSONL
    join_audit_path = artifact_dir / PACKED30_JOIN_AUDIT_JSONL
    corpus_audit_path = artifact_dir / PACKED30_CORPUS_AUDIT_JSON

    from src.utils import save_json

    write_jsonl(manifest_rows, manifest_path)
    write_jsonl(subject_rows, subjects_path)
    write_jsonl(join_audit_rows, join_audit_path)
    save_json(corpus_audit, corpus_audit_path)

    packed30_metadata = {
        "schema_version": PACKED30_SCHEMA_VERSION,
        "protocol_id": PACKED30_PROTOCOL_ID,
        "manifest_variant": PACKED30_MANIFEST_VARIANT,
        "dataset": "daic",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _jsonl_sha256(manifest_rows),
        "manifest_row_count": len(manifest_rows),
        "subject_count": total_subjects,
        "subject_counts_by_split": dict(
            Counter(split_lookup[subject_id] for subject_id in ordered_subjects)
        ),
        "source_file_hashes": {
            "wav": {row["subject_id"]: row["audio_sha256"] for row in subject_rows},
            "transcript": {row["subject_id"]: row["transcript_sha256"] for row in subject_rows},
        },
        "packing_constants": {
            "chunk_samples": PACKED30_CHUNK_SAMPLES,
            "sample_rate": PACKED30_SAMPLE_RATE,
            "inter_span_silence_samples": 0,
            "transcript_max_chars": transcript_max_chars,
        },
        "artifact_paths": {
            "manifest": str(manifest_path),
            "subjects": str(subjects_path),
            "join_audit": str(join_audit_path),
            "corpus_audit": str(corpus_audit_path),
            "metadata": str(artifact_dir / PACKED30_METADATA_JSON),
        },
    }

    subject_partition_rows = [
        {
            "subject_id": subject_id,
            "partition": split_lookup[subject_id],
            "label": int(subject_meta[subject_id]["label"]),
            "label_text": subject_meta[subject_id]["label_text"],
        }
        for subject_id in ordered_subjects
    ]

    LOGGER.info(
        "DAIC packed30 build | subjects=%s retained_rows=%s retained_frames=%s chunks=%s "
        "blank_lines=%s",
        total_subjects,
        total_retained_rows,
        total_retained_frames,
        total_chunks,
        blank_lines_total,
    )
    return {
        "manifest_rows": manifest_rows,
        "subject_rows": subject_rows,
        "subject_partition_rows": subject_partition_rows,
        "join_audit_rows": [
            {
                "subject_id": row["subject_id"],
                "source_row_index": row["source_row_index"],
                "status": row["status"],
                "speaker": row["speaker"],
                "start_time": row["start_time"],
                "stop_time": row["stop_time"],
            }
            for row in join_audit_rows
        ],
        "split_source": "official_train_dev_test_participant_packed30",
        "split_source_notes": (
            "Official minimal_zips split CSVs joined by Participant_ID; labels "
            "PHQ8_Binary (train/dev) and PHQ_Binary (test); participant-only "
            "runtime 30-second packing over original unprocessed recordings."
        ),
        "packed30": {
            "artifact_paths": packed30_metadata["artifact_paths"],
            "metadata": packed30_metadata,
            "corpus_audit": corpus_audit,
            "build_signature": manifest_build_signature(config),
        },
    }


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    serialized = "\n".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_daic_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    manifest_variant = str(config.get("manifest_variant", "")).strip().lower()
    result: dict[str, Any]
    if manifest_variant == PACKED30_MANIFEST_VARIANT:
        LOGGER.info("Building DAIC participant-packed30 manifest: %s", base_dir)
        result = _build_participant_packed30_manifest(config, quarantine)
    elif manifest_variant in {PREPROCESSED_TRAIN_DEV_VARIANT, PREPROCESSED_FULL_TRANSCRIPT_ALL_SPLITS_VARIANT}:
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
