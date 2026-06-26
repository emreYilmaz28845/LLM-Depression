#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUFFIXED_SCIENTIFIC_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+\.\d+$"
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _load_transcripts(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            basename = Path(str(payload["audio_path"])).name
            rows[basename] = {
                "transcript": str(payload.get("transcript", "")).strip(),
                "language": str(payload.get("language", "")).strip(),
            }
    return rows


def inspect_dataset(root: Path, metadata_csv: str) -> dict[str, Any]:
    metadata_path = root / metadata_csv
    transcript_path = root / "whisper_transcripts_repaired.jsonl"
    audio_dir = root / "all-files"
    transcripts = _load_transcripts(transcript_path)
    disk_audio = {path.name for path in audio_dir.glob("*.wav")}

    csv.field_size_limit(max(csv.field_size_limit(), 10**9))
    metadata_rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    subject_labels: dict[str, int] = {}
    files_per_subject: Counter[str] = Counter()
    file_labels: Counter[int] = Counter()
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    missing_audio: list[str] = []
    missing_transcripts: list[str] = []
    empty_transcripts: list[str] = []
    language_anomalies: list[dict[str, str]] = []
    feature_dimensions: Counter[int] = Counter()
    malformed_feature_values = 0
    subject_scores: dict[str, set[float]] = defaultdict(set)

    metadata_basenames: set[str] = set()
    for row in metadata_rows:
        basename = Path(row["file_name"]).name
        metadata_basenames.add(basename)
        subject_id = row["patient_id"].strip()
        label = int(float(row["label_t25"]))
        score = float(row["depresyon_skoru"])
        expected = int(score >= 25.0)
        if label != expected:
            raise ValueError(f"Threshold mismatch: {basename}")
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            raise ValueError(f"Mixed labels for subject {subject_id}")
        subject_labels[subject_id] = label
        subject_scores[subject_id].add(score)
        files_per_subject[subject_id] += 1
        file_labels[label] += 1

        audio_path = audio_dir / basename
        if not audio_path.exists():
            missing_audio.append(basename)
        else:
            info = sf.info(audio_path)
            durations.append(float(info.frames / info.samplerate))
            sample_rates[int(info.samplerate)] += 1

        transcript = transcripts.get(basename)
        if transcript is None:
            missing_transcripts.append(basename)
        else:
            if not transcript["transcript"]:
                empty_transcripts.append(basename)
            if transcript["language"] != "tr":
                language_anomalies.append(
                    {"file": basename, "language": transcript["language"]}
                )

        features = [value for value in row.get("features", "").split(",") if value.strip()]
        feature_dimensions[len(features)] += 1
        malformed_feature_values += sum(
            1
            for value in features
            if _is_malformed_feature_value(value)
        )

    subject_label_counts = Counter(subject_labels.values())
    file_counts = list(files_per_subject.values())
    count_by_label = {
        label: [count for subject, count in files_per_subject.items() if subject_labels[subject] == label]
        for label in (0, 1)
    }
    return {
        "root": str(root),
        "metadata_csv": str(metadata_path),
        "labeled_files": len(metadata_rows),
        "audio_files_on_disk": len(disk_audio),
        "transcript_rows": len(transcripts),
        "unlabeled_audio_files": len(disk_audio - metadata_basenames),
        "subjects": len(subject_labels),
        "file_class_counts": {
            "depressed": file_labels[1],
            "non_depressed": file_labels[0],
        },
        "subject_class_counts": {
            "depressed": subject_label_counts[1],
            "non_depressed": subject_label_counts[0],
        },
        "files_per_subject": {
            "min": min(file_counts),
            "median": statistics.median(file_counts),
            "mean": statistics.mean(file_counts),
            "max": max(file_counts),
            "depressed_mean": statistics.mean(count_by_label[1]),
            "non_depressed_mean": statistics.mean(count_by_label[0]),
        },
        "duration_seconds": {
            "min": min(durations),
            "p25": _percentile(durations, 0.25),
            "median": statistics.median(durations),
            "p75": _percentile(durations, 0.75),
            "p95": _percentile(durations, 0.95),
            "max": max(durations),
            "total_hours": sum(durations) / 3600.0,
            "files_over_30_seconds": sum(value > 30.0 for value in durations),
        },
        "sample_rates": dict(sorted(sample_rates.items())),
        "feature_dimensions": dict(sorted(feature_dimensions.items())),
        "suffixed_feature_values": malformed_feature_values,
        "missing_audio": missing_audio,
        "missing_transcripts": missing_transcripts,
        "empty_transcripts": empty_transcripts,
        "language_anomalies": language_anomalies,
        "subjects_with_multiple_scores": sorted(
            subject_id for subject_id, values in subject_scores.items() if len(values) > 1
        ),
    }


def _is_malformed_feature_value(value: str) -> bool:
    try:
        float(value)
        return False
    except ValueError:
        return bool(SUFFIXED_SCIENTIFIC_NUMBER.match(value.strip()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Turkish depression dataset.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--metadata-csv",
        default="metadata_turkish_t25_binary_merged.csv",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_dataset(Path(args.root), args.metadata_csv)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
