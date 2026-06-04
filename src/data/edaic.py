from __future__ import annotations

import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.validation import is_quarantined_missing
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)
SEGMENT_RE = re.compile(r"^(?P<subject_id>\d+)_(?P<segment_kind>random_segment|segment)_(?P<chunk_id>\d+)\.wav$")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        filtered_lines = [line for line in handle if line.strip()]
    return list(csv.DictReader(filtered_lines))


def _required_row_value(row: dict[str, str], key: str) -> str:
    if key not in row:
        raise KeyError(f"Missing required EDAIC summary column {key!r}. Available: {', '.join(sorted(row.keys()))}")
    return str(row[key]).strip()


def _sample_id_from_audio_name(audio_name: str) -> str | None:
    if SEGMENT_RE.match(audio_name):
        return Path(audio_name).stem
    return None


def _subject_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def _chunk_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[1]


def _parse_segment_files(row: dict[str, str], summary_path: Path) -> list[str]:
    raw_value = _required_row_value(row, "segment_files")
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse segment_files in {summary_path}: {raw_value!r}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"segment_files must be a non-empty list in {summary_path}: {raw_value!r}")
    return [str(item).strip() for item in parsed if str(item).strip()]


def _build_expected_rows(
    split_name: str,
    summary_path: Path,
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    expected_samples: dict[str, dict[str, Any]] = {}
    subject_meta: dict[str, dict[str, Any]] = {}
    split_lookup: dict[str, str] = {}

    for row in rows:
        subject_id = _required_row_value(row, "participant_id")
        label = int(_required_row_value(row, "is_depressed"))
        transcript = _required_row_value(row, "full_transcript")
        if not transcript:
            raise ValueError(f"EDAIC full_transcript is empty in {summary_path} for participant_id={subject_id}")
        segment_files = _parse_segment_files(row, summary_path)
        declared_num_segments = int(_required_row_value(row, "num_segments"))
        if len(segment_files) != declared_num_segments:
            raise ValueError(
                f"EDAIC segment count mismatch in {summary_path} for participant_id={subject_id}: "
                f"declared={declared_num_segments} actual={len(segment_files)}"
            )

        subject_meta[subject_id] = {
            "label": label,
            "label_text": label_text_from_int(label),
            "score": "",
            "gender": None,
        }
        split_lookup[subject_id] = split_name

        for audio_name in segment_files:
            match = SEGMENT_RE.match(audio_name)
            if match is None:
                raise ValueError(
                    f"EDAIC segment filename does not match expected pattern in {summary_path}: {audio_name!r}"
                )
            if match.group("subject_id") != subject_id:
                raise ValueError(
                    f"EDAIC segment filename subject mismatch in {summary_path}: "
                    f"participant_id={subject_id} audio_name={audio_name!r}"
                )
            sample_id = Path(audio_name).stem
            if sample_id in expected_samples:
                raise ValueError(f"Duplicate EDAIC sample_id detected across summaries: {sample_id}")
            expected_samples[sample_id] = {
                "subject_id": subject_id,
                "audio_name": audio_name,
                "transcript": transcript,
                "transcript_path": str(summary_path),
                "split_name": split_name,
                "label": label,
                "label_text": label_text_from_int(label),
            }

    return expected_samples, subject_meta, split_lookup


def _discover_split_wavs(
    split_name: str,
    split_dir: Path,
    expected_names: set[str],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"EDAIC preprocessed split directory is missing: {split_dir}")

    wav_map: dict[str, Path] = {}
    extra_file_audit: list[dict[str, Any]] = []
    for wav_path in sorted(split_dir.glob("*/*.wav")):
        sample_id = _sample_id_from_audio_name(wav_path.name)
        if sample_id is None:
            extra_file_audit.append(
                {
                    "split": split_name,
                    "audio_path": str(wav_path),
                    "reason": "unexpected_filename_format",
                }
            )
            continue
        if wav_path.name not in expected_names:
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


def build_edaic_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    split_specs = [
        ("train", base_dir / "train_preprocessing_summary.csv", base_dir / "train_audio_segments"),
        ("val", base_dir / "dev_preprocessing_summary.csv", base_dir / "dev_audio_segments"),
        ("test", base_dir / "test_preprocessing_summary.csv", base_dir / "test_audio_segments"),
    ]

    expected_samples: dict[str, dict[str, Any]] = {}
    subject_meta: dict[str, dict[str, Any]] = {}
    split_lookup: dict[str, str] = {}
    wav_map: dict[str, Path] = {}
    extra_file_audit: list[dict[str, Any]] = []

    for split_name, summary_path, split_dir in split_specs:
        rows = _load_csv_rows(summary_path)
        split_expected_samples, split_subject_meta, split_split_lookup = _build_expected_rows(split_name, summary_path, rows)
        expected_samples.update(split_expected_samples)
        subject_meta.update(split_subject_meta)
        split_lookup.update(split_split_lookup)
        split_wav_map, split_extra_file_audit = _discover_split_wavs(
            split_name=split_name,
            split_dir=split_dir,
            expected_names={payload["audio_name"] for payload in split_expected_samples.values()},
        )
        wav_map.update(split_wav_map)
        extra_file_audit.extend(split_extra_file_audit)

    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    per_split_counts = defaultdict(Counter)
    missing_audio = 0
    missing_transcript = 0

    for sample_id in sorted(expected_samples):
        payload = expected_samples[sample_id]
        subject_id = payload["subject_id"]
        meta = subject_meta[subject_id]
        wav_path = wav_map.get(sample_id)
        transcript = payload["transcript"]
        audio_found = wav_path is not None and wav_path.exists()
        transcript_found = bool(transcript)
        if not audio_found:
            missing_audio += 1
        if not transcript_found:
            missing_transcript += 1
        join_audit_rows.append(
            {
                "subject_id": subject_id,
                "chunk_id": _chunk_id_from_sample_id(sample_id),
                "sample_id": sample_id,
                "wav_path": str(wav_path) if wav_path else "",
                "audio_found": audio_found,
                "transcript_found": transcript_found,
                "split": payload["split_name"],
                "label": meta["label"],
                "label_text": meta["label_text"],
            }
        )
        if not audio_found or not transcript_found:
            if is_quarantined_missing(quarantine, "edaic", sample_id):
                continue
            if not audio_found:
                raise FileNotFoundError(f"EDAIC missing canonical audio for sample_id={sample_id}")
            raise ValueError(f"EDAIC missing canonical transcript for sample_id={sample_id}")

        per_split_counts[payload["split_name"]][meta["label"]] += 1
        manifest_rows.append(
            {
                "dataset": "edaic",
                "subject_id": subject_id,
                "sample_id": sample_id,
                "audio_path": str(wav_path),
                "audio_paths": [str(wav_path)],
                "transcript": transcript,
                "transcript_path": payload["transcript_path"],
                "label": meta["label"],
                "label_text": meta["label_text"],
                "score": meta["score"],
                "split_original": payload["split_name"],
                "fold": "",
                "chunk_id": _chunk_id_from_sample_id(sample_id),
                "question_id": "",
                "start_time": "",
                "end_time": "",
                "language": "",
                "gender": meta["gender"],
                "modality_mode": "single_audio_single_text",
            }
        )

    LOGGER.info(
        "EDAIC join audit | missing_audio=%s missing_transcript=%s matched_samples=%s",
        missing_audio,
        missing_transcript,
        len(manifest_rows),
    )
    for split_name, counts in sorted(per_split_counts.items()):
        LOGGER.info(
            "EDAIC split=%s | depressed_samples=%s non_depressed_samples=%s",
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
        "subject_partition_rows": subject_partition_rows,
        "join_audit_rows": join_audit_rows,
        "extra_file_audit": extra_file_audit,
        "split_source": "preprocessed_full_transcript_all_splits",
        "split_source_notes": (
            "Uses train_preprocessing_summary.csv, dev_preprocessing_summary.csv, and "
            "test_preprocessing_summary.csv from the preprocessed EDAIC layout. Each chunk "
            "listed in segment_files is paired with the participant's full_transcript from "
            "the corresponding summary row."
        ),
    }
