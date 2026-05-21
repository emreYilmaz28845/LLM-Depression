from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.validation import is_quarantined_missing
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)
CHUNK_RE = re.compile(r"^(?P<subject_id>\d+)_(?P<chunk_id>\d+)\.wav$")


def _load_official_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        filtered_lines = [line for line in handle if line.strip()]
    return list(csv.DictReader(filtered_lines))


def _load_whisper_transcripts(path: Path) -> dict[str, dict[str, Any]]:
    transcripts: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_name = Path(row["audio_path"]).name
            match = CHUNK_RE.match(audio_name)
            if not match:
                continue
            sample_id = f"{match.group('subject_id')}_{match.group('chunk_id')}"
            transcripts[sample_id] = {
                "transcript": row["transcript"].strip(),
                "language": row.get("language", ""),
                "audio_name": audio_name,
            }
    return transcripts


def build_daic_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    official_dir = base_dir / "minimal_zips"
    wav_dir = official_dir / "preprocessed_audios"
    transcript_path = base_dir / "whisper_transcripts.jsonl"

    split_specs = {
        "train": official_dir / "train_split_Depression_AVEC2017 (1).csv",
        "dev": official_dir / "dev_split_Depression_AVEC2017.csv",
        "test": official_dir / "full_test_split (1).csv",
    }
    split_rows = {name: _load_official_split_csv(path) for name, path in split_specs.items()}
    subject_meta: dict[str, dict[str, Any]] = {}
    split_lookup: dict[str, str] = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            subject_id = row["Participant_ID"]
            label_key = "PHQ8_Binary" if "PHQ8_Binary" in row else "PHQ_Binary"
            score_key = "PHQ8_Score"
            label = int(row[label_key])
            subject_meta[subject_id] = {
                "label": label,
                "label_text": label_text_from_int(label),
                "score": int(row[score_key]),
                "gender": row.get("Gender"),
            }
            split_lookup[subject_id] = split_name

    transcripts = _load_whisper_transcripts(transcript_path)
    wav_map: dict[str, Path] = {}
    for wav_path in sorted(wav_dir.glob("*.wav")):
        match = CHUNK_RE.match(wav_path.name)
        if not match:
            continue
        subject_id = match.group("subject_id")
        if subject_id not in subject_meta:
            continue
        sample_id = f"{subject_id}_{match.group('chunk_id')}"
        wav_map[sample_id] = wav_path

    sample_ids = sorted(set(wav_map) | {sample_id for sample_id in transcripts if sample_id.split("_", 1)[0] in subject_meta})
    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    missing_audio = 0
    missing_transcript = 0
    matched_samples = 0
    per_split_counts = defaultdict(Counter)

    for sample_id in sample_ids:
        subject_id, chunk_id = sample_id.split("_", 1)
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
                "transcript_path": str(transcript_path),
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

