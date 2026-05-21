from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils import LABEL_DEPRESSED, LABEL_NON_DEPRESSED, get_logger, load_yaml


LOGGER = get_logger(__name__)


def load_quarantine(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    data = load_yaml(path)
    return data or {}


def _dataset_missing_sample_ids(quarantine: dict[str, Any], dataset: str) -> set[str]:
    dataset_cfg = (quarantine.get("datasets") or {}).get(dataset, {})
    missing = dataset_cfg.get("missing_samples", [])
    sample_ids: set[str] = set()
    for item in missing:
        sample_ids.add(str(item["sample_id"]))
    return sample_ids


def is_quarantined_missing(quarantine: dict[str, Any], dataset: str, sample_id: str) -> bool:
    return sample_id in _dataset_missing_sample_ids(quarantine, dataset)


def assert_clean_labels(rows: list[dict[str, Any]]) -> None:
    allowed_labels = {0, 1}
    allowed_text = {LABEL_DEPRESSED, LABEL_NON_DEPRESSED}
    bad_rows = [
        row["sample_id"]
        for row in rows
        if row["label"] not in allowed_labels or row["label_text"] not in allowed_text
    ]
    if bad_rows:
        raise ValueError(f"Found invalid labels for sample ids: {bad_rows[:10]}")


def assert_audio_exists(rows: list[dict[str, Any]]) -> None:
    missing_paths: list[str] = []
    for row in rows:
        audio_paths = row.get("audio_paths") or [row["audio_path"]]
        for audio_path in audio_paths:
            if not audio_path:
                missing_paths.append(f"{row['sample_id']}::EMPTY_AUDIO_PATH")
                continue
            if not Path(audio_path).exists():
                missing_paths.append(f"{row['sample_id']}::{audio_path}")
    if missing_paths:
        raise FileNotFoundError(f"Missing audio paths detected: {missing_paths[:20]}")


def assert_transcripts(rows: list[dict[str, Any]], allow_empty: bool = False) -> None:
    empty_ids = [row["sample_id"] for row in rows if not str(row.get("transcript", "")).strip()]
    if empty_ids and not allow_empty:
        raise ValueError(f"Found empty transcripts for sample ids: {empty_ids[:20]}")


def print_random_rows(rows: list[dict[str, Any]], dataset: str, seed: int, limit: int = 10) -> None:
    preview_rows = rows[:]
    random.Random(seed).shuffle(preview_rows)
    LOGGER.info("Random manifest preview for %s:", dataset)
    for row in preview_rows[:limit]:
        transcript = str(row.get("transcript", ""))[:200].replace("\n", " ")
        preview = {
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
            "audio_path": row.get("audio_path"),
            "transcript": transcript,
            "label_text": row["label_text"],
        }
        LOGGER.info(json.dumps(preview, ensure_ascii=False))


def print_class_counts(rows: list[dict[str, Any]], dataset: str, partition_field: str | None = None) -> None:
    subject_labels = {}
    sample_counter = Counter()
    subject_counter = Counter()
    partition_sample_counts: dict[str, Counter] = defaultdict(Counter)
    partition_subjects: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        label = int(row["label"])
        sample_counter[label] += 1
        subject_labels[row["subject_id"]] = label
        partition = str(row.get(partition_field, "ALL")) if partition_field else "ALL"
        partition_sample_counts[partition][label] += 1
        partition_subjects[partition][row["subject_id"]] = label
    for label in subject_labels.values():
        subject_counter[label] += 1
    LOGGER.info(
        "%s sample counts | depressed=%s non_depressed=%s",
        dataset,
        sample_counter[1],
        sample_counter[0],
    )
    LOGGER.info(
        "%s subject counts | depressed=%s non_depressed=%s",
        dataset,
        subject_counter[1],
        subject_counter[0],
    )
    for partition, counter in sorted(partition_sample_counts.items()):
        subject_partition_counter = Counter(partition_subjects[partition].values())
        LOGGER.info(
            "%s partition=%s | sample depressed=%s non_depressed=%s | subject depressed=%s non_depressed=%s",
            dataset,
            partition,
            counter[1],
            counter[0],
            subject_partition_counter[1],
            subject_partition_counter[0],
        )


def assert_subject_partition_uniqueness(assignments: list[dict[str, Any]], subject_key: str = "subject_id") -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        locations[row[subject_key]].add(str(row["partition"]))
    overlaps = {subject_id: parts for subject_id, parts in locations.items() if len(parts) > 1}
    if overlaps:
        raise ValueError(f"Subjects found in multiple partitions: {list(overlaps.items())[:10]}")

