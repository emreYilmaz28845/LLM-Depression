from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from src.data.split_utils import deterministic_inner_split
from src.utils import label_text_from_int


ANDROIDS_INTERVIEW_SUBJECT_COUNT = 116
ANDROIDS_INTERVIEW_PATIENT_COUNT = 64
ANDROIDS_INTERVIEW_CONTROL_COUNT = 52
ANDROIDS_INTERVIEW_TURN_COUNT = 874
ANDROIDS_INTERVIEW_WINDOW_COUNT = 1302
ANDROIDS_DEFAULT_SEGMENT_SECONDS = 30.0

_RECORDING_RE = re.compile(
    r"^(?P<numeric_id>\d+)_(?P<condition>[CP])(?P<gender>[FM])"
    r"(?P<age>\d+)_(?P<education>\d+|x)$",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_androids_recording_id(recording_id: str) -> dict[str, Any]:
    text = str(recording_id).strip().strip("'\"")
    match = _RECORDING_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid ANDROIDS recording ID: {recording_id!r}")
    condition = match.group("condition").upper()
    numeric_id = f"{int(match.group('numeric_id')):02d}"
    education_raw = match.group("education").lower()
    return {
        "recording_id": text,
        "subject_id": f"{numeric_id}_{condition}",
        "numeric_subject_id": numeric_id,
        "condition_code": condition,
        "label": int(condition == "P"),
        "label_text": label_text_from_int(int(condition == "P")),
        "diagnosis": "depression" if condition == "P" else "control",
        "gender": match.group("gender").upper(),
        "age": int(match.group("age")),
        "education_level": int(education_raw) if education_raw.isdigit() else None,
        "education_level_raw": education_raw,
    }


def parse_androids_turn_path(path: str | Path) -> dict[str, Any]:
    audio_path = Path(path)
    recording = parse_androids_recording_id(audio_path.parent.name)
    prefix = f"{recording['recording_id']}_"
    if not audio_path.stem.startswith(prefix):
        raise ValueError(
            f"ANDROIDS interview clip does not match parent recording: {audio_path}"
        )
    turn_text = audio_path.stem[len(prefix) :]
    if not turn_text.isdigit() or int(turn_text) <= 0:
        raise ValueError(f"Invalid ANDROIDS interview turn ID: {audio_path.name}")
    turn_id = int(turn_text)
    turn_key = audio_path.stem
    return {
        **recording,
        "turn_id": turn_id,
        "turn_key": turn_key,
        "response_id": turn_key,
    }


def equal_duration_windows(
    duration: float,
    segment_seconds: float = ANDROIDS_DEFAULT_SEGMENT_SECONDS,
) -> list[tuple[float, float]]:
    if duration <= 0:
        raise ValueError(f"Audio duration must be positive, got {duration}.")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    count = max(1, int(math.ceil(duration / segment_seconds)))
    width = duration / count
    return [
        (index * width, duration if index == count - 1 else (index + 1) * width)
        for index in range(count)
    ]


def androids_window_id(turn_key: str, window_index: int) -> str:
    return f"{turn_key}_w{int(window_index):02d}"


def discover_androids_interview_windows(
    dataset_root: str | Path,
    *,
    segment_seconds: float = ANDROIDS_DEFAULT_SEGMENT_SECONDS,
    enforce_corpus_contract: bool = True,
) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    audio_root = root / "Interview-Task" / "audio_clip"
    audio_files = sorted(audio_root.rglob("*.wav"), key=lambda path: str(path))
    rows: list[dict[str, Any]] = []
    subject_ids: set[str] = set()
    class_counts: Counter[int] = Counter()
    seen_recordings: set[str] = set()
    for audio_path in audio_files:
        identity = parse_androids_turn_path(audio_path)
        subject_ids.add(identity["subject_id"])
        if identity["recording_id"] not in seen_recordings:
            class_counts[int(identity["label"])] += 1
            seen_recordings.add(identity["recording_id"])
        info = sf.info(audio_path)
        duration = float(info.frames / info.samplerate)
        windows = equal_duration_windows(duration, segment_seconds)
        for window_index, (start_time, end_time) in enumerate(windows):
            rows.append(
                {
                    "dataset": "androids_interview",
                    "task": "interview",
                    **identity,
                    "sample_id": androids_window_id(identity["turn_key"], window_index),
                    "window_id": androids_window_id(identity["turn_key"], window_index),
                    "window_index": window_index,
                    "segment_index": window_index,
                    "num_windows": len(windows),
                    "num_segments": len(windows),
                    "audio_path": str(audio_path),
                    "start_time": start_time,
                    "end_time": end_time,
                    "segment_duration": end_time - start_time,
                    "turn_duration": duration,
                }
            )
    if enforce_corpus_contract:
        expected_classes = Counter({0: ANDROIDS_INTERVIEW_CONTROL_COUNT, 1: ANDROIDS_INTERVIEW_PATIENT_COUNT})
        if (
            len(subject_ids) != ANDROIDS_INTERVIEW_SUBJECT_COUNT
            or len(seen_recordings) != ANDROIDS_INTERVIEW_SUBJECT_COUNT
            or len(audio_files) != ANDROIDS_INTERVIEW_TURN_COUNT
            or len(rows) != ANDROIDS_INTERVIEW_WINDOW_COUNT
            or class_counts != expected_classes
        ):
            raise ValueError(
                "Unexpected ANDROIDS Interview corpus: "
                f"subjects={len(subject_ids)} classes={dict(class_counts)} "
                f"turns={len(audio_files)} windows={len(rows)}; expected "
                f"{ANDROIDS_INTERVIEW_SUBJECT_COUNT}, {dict(expected_classes)}, "
                f"{ANDROIDS_INTERVIEW_TURN_COUNT}, {ANDROIDS_INTERVIEW_WINDOW_COUNT}."
            )
    return rows


def build_androids_interview_official_folds(
    fold_list_path: str | Path,
    subject_by_recording: dict[str, str],
) -> dict[int, dict[str, list[str]]]:
    path = Path(fold_list_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or len(rows[0]) < 12 or rows[0][7].strip().lower() != "interview":
        raise ValueError(f"Could not locate the Interview fold block in {path}.")
    expected_headers = [f"fold{index}" for index in range(1, 6)]
    if [value.strip().lower() for value in rows[1][7:12]] != expected_headers:
        raise ValueError(f"Unexpected ANDROIDS Interview fold headers in {path}.")

    expected_recordings = set(subject_by_recording)
    observed_recordings: set[str] = set()
    folds: dict[int, dict[str, list[str]]] = {}
    for fold_index, column in enumerate(range(7, 12)):
        recordings = [
            row[column].strip().strip("'\"")
            for row in rows[2:]
            if len(row) > column and row[column].strip()
        ]
        duplicates = sorted(
            recording for recording, count in Counter(recordings).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate recordings in Interview fold {fold_index}: {duplicates}")
        unknown = sorted(set(recordings) - expected_recordings)
        if unknown:
            raise ValueError(f"Unknown recordings in Interview fold {fold_index}: {unknown}")
        overlap = sorted(observed_recordings.intersection(recordings))
        if overlap:
            raise ValueError(f"Interview held-out fold overlap: {overlap}")
        observed_recordings.update(recordings)
        heldout_subjects = sorted(subject_by_recording[recording] for recording in recordings)
        if len(set(heldout_subjects)) != len(heldout_subjects):
            raise ValueError(f"Multiple recordings map to one subject in Interview fold {fold_index}.")
        folds[fold_index] = {
            "outer_train_subject_ids": sorted(set(subject_by_recording.values()) - set(heldout_subjects)),
            "final_eval_subject_ids": heldout_subjects,
            "official_interview_recording_ids": recordings,
        }
    if observed_recordings != expected_recordings:
        raise ValueError(
            "Interview fold coverage mismatch: "
            f"missing={sorted(expected_recordings - observed_recordings)} "
            f"extra={sorted(observed_recordings - expected_recordings)}"
        )
    return folds


def _load_transcripts(
    path: Path,
    *,
    id_field: str,
    expected_language: set[str] = frozenset({"it", "italian"}),
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ANDROIDS transcript JSONL: {path}")
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stable_id = str(row.get(id_field, "")).strip()
            if not stable_id:
                raise ValueError(f"Missing {id_field} in {path}:{line_number}")
            if stable_id in result:
                raise ValueError(f"Duplicate ANDROIDS transcript ID: {stable_id}")
            transcript = str(row.get("transcript", "")).strip()
            if not transcript:
                raise ValueError(f"Empty ANDROIDS transcript: {stable_id}")
            language = str(row.get("language", "")).strip().lower()
            detected = str(row.get("asr_detected_language", "")).strip().lower()
            if language not in expected_language or (detected and detected not in expected_language):
                raise ValueError(f"Non-Italian ANDROIDS transcript: {stable_id}")
            result[stable_id] = {**row, "transcript": transcript}
    return result


def _fold_report(
    folds: dict[int, dict[str, list[str]]],
    subject_meta: dict[str, dict[str, Any]],
    *,
    seed: int,
    inner_val_ratio: float,
) -> list[dict[str, Any]]:
    labels = {subject_id: int(row["label"]) for subject_id, row in subject_meta.items()}
    report: list[dict[str, Any]] = []
    for fold, payload in sorted(folds.items()):
        inner = deterministic_inner_split(
            labels,
            payload["outer_train_subject_ids"],
            seed=seed + fold,
            val_ratio=inner_val_ratio,
        )
        partitions = {
            "train_inner": inner["train_inner_subject_ids"],
            "val_inner": inner["val_inner_subject_ids"],
            "outer_test": payload["final_eval_subject_ids"],
        }
        report.append(
            {
                "fold": fold,
                "partitions": {
                    name: {
                        "subject_ids": values,
                        "label_counts": dict(Counter(subject_meta[value]["label_text"] for value in values)),
                    }
                    for name, values in partitions.items()
                },
            }
        )
    return report


def build_androids_interview_manifest(
    config: dict[str, Any],
    quarantine: dict[str, Any],
) -> dict[str, Any]:
    del quarantine
    root = Path(config["dataset_root"])
    data_cfg = config.get("data", {})
    segment_seconds = float(data_cfg.get("segment_seconds", ANDROIDS_DEFAULT_SEGMENT_SECONDS))
    if not math.isclose(
        segment_seconds,
        ANDROIDS_DEFAULT_SEGMENT_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("ANDROIDS Interview requires data.segment_seconds=30.")
    if str(data_cfg.get("segment_partition", "equal_duration")) != "equal_duration":
        raise ValueError("ANDROIDS Interview supports only equal_duration segmentation.")
    if int(config.get("split", {}).get("outer_folds", 5)) != 5:
        raise ValueError("ANDROIDS Interview requires the five official folds.")

    windows = discover_androids_interview_windows(root, segment_seconds=segment_seconds)
    full_path = Path(
        config.get("full_transcript_path")
        or root / "interview_transcripts_qwen3_asr_italian.jsonl"
    )
    segment_path = Path(
        config.get("segment_transcript_path")
        or root / "interview_transcripts_qwen3_asr_italian_segments.jsonl"
    )
    full_transcripts = _load_transcripts(full_path, id_field="sample_id")
    segment_transcripts = _load_transcripts(segment_path, id_field="sample_id")

    subject_meta: dict[str, dict[str, Any]] = {}
    subject_by_recording: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    used_full: set[str] = set()
    used_segments: set[str] = set()
    for window in windows:
        turn_key = str(window["turn_key"])
        window_id = str(window["window_id"])
        full = full_transcripts.get(turn_key)
        segment = segment_transcripts.get(window_id)
        if full is None or segment is None:
            raise ValueError(
                f"ANDROIDS transcript coverage failure for {window_id}: "
                f"full={full is not None} segment={segment is not None}"
            )
        if str(full.get("audio_path", "")) != str(window["audio_path"]):
            raise ValueError(f"ANDROIDS full-turn audio path mismatch for {turn_key}.")
        for field in ("start_time", "end_time"):
            if field not in segment or not math.isclose(
                float(segment[field]), float(window[field]), rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"ANDROIDS interval mismatch for {window_id} field={field}: "
                    f"cache={segment.get(field)!r} canonical={window[field]!r}"
                )
        if str(segment.get("audio_path", "")) != str(window["audio_path"]):
            raise ValueError(f"ANDROIDS audio path mismatch for {window_id}.")
        used_full.add(turn_key)
        used_segments.add(window_id)
        subject_id = str(window["subject_id"])
        subject_meta.setdefault(
            subject_id,
            {
                key: window[key]
                for key in (
                    "subject_id",
                    "numeric_subject_id",
                    "recording_id",
                    "condition_code",
                    "diagnosis",
                    "gender",
                    "age",
                    "education_level",
                    "education_level_raw",
                    "label",
                    "label_text",
                )
            },
        )
        subject_by_recording[str(window["recording_id"])] = subject_id
        manifest_rows.append(
            {
                **window,
                "audio_paths": [window["audio_path"]],
                "transcript": segment["transcript"],
                "segment_transcript": segment["transcript"],
                "full_turn_transcript": full["transcript"],
                "transcript_path": str(segment_path),
                "full_transcript_path": str(full_path),
                "language": "it",
                "split_original": "",
                "fold": "",
                "question_id": str(window["turn_id"]),
                "prompt_id": int(window["turn_id"]),
                "chunk_id": int(window["window_index"]),
                "modality_mode": "single_audio_window_dual_transcript",
            }
        )
        join_audit_rows.append(
            {
                "sample_id": window_id,
                "response_id": turn_key,
                "audio_found": Path(window["audio_path"]).is_file(),
                "segment_transcript_found": True,
                "full_turn_transcript_found": True,
                "aligned_start_time": window["start_time"],
                "aligned_end_time": window["end_time"],
            }
        )
    extra_full = sorted(set(full_transcripts) - used_full)
    extra_segments = sorted(set(segment_transcripts) - used_segments)
    if extra_full or extra_segments:
        raise ValueError(
            f"Unexpected ANDROIDS transcript rows: extra_full={extra_full[:10]} "
            f"extra_segments={extra_segments[:10]}"
        )

    folds = build_androids_interview_official_folds(
        root / "fold-lists.csv", subject_by_recording
    )
    seed = int(config.get("split", {}).get("seed", config["seed"]))
    fold_report = _fold_report(
        folds,
        subject_meta,
        seed=seed,
        inner_val_ratio=float(config.get("split", {}).get("inner_val_ratio", 0.2)),
    )
    windows_per_turn = Counter(row["response_id"] for row in manifest_rows)
    turns_per_subject = Counter(row["subject_id"] for row in {r["response_id"]: r for r in manifest_rows}.values())
    chunk_audit = {
        "segment_seconds": segment_seconds,
        "segment_partition": "equal_duration",
        "subjects": len(subject_meta),
        "class_counts": dict(Counter(int(row["label"]) for row in subject_meta.values())),
        "turns": len(windows_per_turn),
        "windows": len(manifest_rows),
        "turns_per_subject_distribution": dict(sorted(Counter(turns_per_subject.values()).items())),
        "windows_per_turn_distribution": dict(sorted(Counter(windows_per_turn.values()).items())),
        "max_segment_duration": max(float(row["segment_duration"]) for row in manifest_rows),
        "coverage_contiguous_complete": True,
    }
    fold_hash = hashlib.sha256(
        json.dumps(folds, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_rows": manifest_rows,
        "subject_rows": [subject_meta[key] for key in sorted(subject_meta)],
        "folds": folds,
        "fold_report": fold_report,
        "join_audit_rows": join_audit_rows,
        "chunk_window_audit": chunk_audit,
        "fold_hash": fold_hash,
        "split_source": "official_androids_interview_folds",
        "split_source_notes": "Only columns 8-12 under the Interview header in fold-lists.csv are parsed.",
        "transcript_paths": {
            "full_turn": str(full_path),
            "segment_aligned": str(segment_path),
        },
        "source_hashes": {
            "fold_lists_sha256": _sha256_file(root / "fold-lists.csv"),
            "full_turn_transcripts_sha256": _sha256_file(full_path),
            "segment_transcripts_sha256": _sha256_file(segment_path),
            "full_turn_transcript_report_sha256": (
                _sha256_file(full_path.with_suffix(".report.json"))
                if full_path.with_suffix(".report.json").is_file()
                else None
            ),
            "segment_transcript_report_sha256": (
                _sha256_file(segment_path.with_suffix(".report.json"))
                if segment_path.with_suffix(".report.json").is_file()
                else None
            ),
        },
    }


def apply_androids_training_weights(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not examples:
        raise ValueError("Cannot weight an empty ANDROIDS training partition.")
    turns_by_subject: dict[str, set[str]] = defaultdict(set)
    windows_by_turn: Counter[str] = Counter()
    turn_subject: dict[str, str] = {}
    for example in examples:
        subject_id = str(example["subject_id"])
        turn_key = str(example.get("response_id", "")).strip()
        if not turn_key:
            raise ValueError("ANDROIDS training weights require response_id on every window.")
        if turn_key in turn_subject and turn_subject[turn_key] != subject_id:
            raise ValueError(f"ANDROIDS turn {turn_key} spans multiple subjects.")
        turn_subject[turn_key] = subject_id
        turns_by_subject[subject_id].add(turn_key)
        windows_by_turn[turn_key] += 1

    raw_weights = [
        1.0
        / (
            len(turns_by_subject[str(example["subject_id"])])
            * windows_by_turn[str(example["response_id"])]
        )
        for example in examples
    ]
    scale = len(raw_weights) / sum(raw_weights)
    weighted: list[dict[str, Any]] = []
    for example, raw_weight in zip(examples, raw_weights):
        weighted.append(
            {**example, "raw_loss_weight": raw_weight, "loss_weight": raw_weight * scale}
        )
    raw_turn_totals: defaultdict[str, float] = defaultdict(float)
    raw_subject_totals: defaultdict[str, float] = defaultdict(float)
    for example in weighted:
        raw_turn_totals[str(example["response_id"])] += float(example["raw_loss_weight"])
        raw_subject_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    for subject_id, total in raw_subject_totals.items():
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"ANDROIDS subject weight total is not one for {subject_id}: {total}"
            )
    for turn_key, total in raw_turn_totals.items():
        expected = 1.0 / len(turns_by_subject[turn_subject[turn_key]])
        if not math.isclose(total, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"ANDROIDS turn weight total mismatch for {turn_key}: "
                f"{total} != {expected}"
            )
    audit = {
        "formula": "1 / (turns_for_subject * windows_for_parent_turn), rescaled_to_mean_one",
        "sample_count": len(weighted),
        "subject_count": len(turns_by_subject),
        "turn_count": len(windows_by_turn),
        "mean_loss_weight": sum(float(row["loss_weight"]) for row in weighted) / len(weighted),
        "raw_subject_weight_totals": dict(sorted(raw_subject_totals.items())),
        "raw_turn_weight_totals": dict(sorted(raw_turn_totals.items())),
    }
    return weighted, audit
