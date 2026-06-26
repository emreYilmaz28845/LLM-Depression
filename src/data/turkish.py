from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.split_utils import (
    assign_stratified_group_folds,
    deterministic_inner_split,
    subject_fold_report,
    validate_non_overlapping_folds,
)
from src.data.validation import is_quarantined_missing
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)
DEFAULT_METADATA_CSV = "metadata_turkish_t25_binary_merged.csv"
DEFAULT_TRANSCRIPT_FILE = "whisper_transcripts_repaired.jsonl"
SOURCE_THRESHOLD = 25.0


def _optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


def _load_whisper_transcripts(path: Path) -> dict[str, dict[str, str]]:
    transcripts: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            basename = Path(str(payload.get("audio_path", ""))).name
            if not basename:
                raise ValueError(f"Turkish transcript row {line_number} has no audio basename.")
            if basename in transcripts:
                raise ValueError(f"Duplicate Turkish transcript basename: {basename}")
            transcripts[basename] = {
                "transcript": str(payload.get("transcript", "")).strip(),
                "language": str(payload.get("language", "")).strip(),
            }
    return transcripts


def _parse_chunk_id(basename: str, subject_id: str) -> str:
    pattern = re.compile(rf"^{re.escape(subject_id)}-[^-]+-([^-]+)-.+\.wav$", re.IGNORECASE)
    match = pattern.match(basename)
    if match is None:
        raise ValueError(
            f"Turkish filename does not match patient_id | patient_id={subject_id!r} file={basename!r}"
        )
    return match.group(1)


def verify_turkish_split_integrity(
    manifest_rows: list[dict[str, Any]],
    folds: dict[int, dict[str, list[str]]],
    inner_val_ratio: float,
    seed: int,
) -> None:
    if not manifest_rows:
        raise ValueError("Turkish manifest is empty.")

    all_subjects = {str(row["subject_id"]) for row in manifest_rows}
    validate_non_overlapping_folds(folds, sorted(all_subjects))

    labels: dict[str, int] = {}
    sample_subjects: dict[str, set[str]] = defaultdict(set)
    file_subjects: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in labels and labels[subject_id] != label:
            raise ValueError(f"Turkish subject has mixed labels: {subject_id}")
        labels[subject_id] = label
        sample_subjects[str(row["sample_id"])].add(subject_id)
        file_subjects[Path(str(row["audio_path"])).name].add(subject_id)

    duplicate_samples = {key: value for key, value in sample_subjects.items() if len(value) != 1}
    if duplicate_samples:
        raise ValueError(f"Turkish sample ids map to multiple subjects: {list(duplicate_samples.items())[:5]}")
    duplicate_files = {key: value for key, value in file_subjects.items() if len(value) != 1}
    if duplicate_files:
        raise ValueError(f"Turkish audio files map to multiple subjects: {list(duplicate_files.items())[:5]}")

    holdout_counts: Counter[str] = Counter()
    for fold_idx, payload in sorted(folds.items()):
        outer_train = set(payload["outer_train_subject_ids"])
        final_eval = set(payload["final_eval_subject_ids"])
        overlap = outer_train.intersection(final_eval)
        if overlap:
            raise ValueError(f"Turkish fold {fold_idx} train/eval overlap: {sorted(overlap)[:5]}")
        if outer_train.union(final_eval) != all_subjects:
            raise ValueError(f"Turkish fold {fold_idx} does not partition every subject.")
        holdout_counts.update(final_eval)

        inner = deterministic_inner_split(
            labels,
            sorted(outer_train),
            seed=int(seed) + int(fold_idx),
            val_ratio=float(inner_val_ratio),
        )
        train_inner = set(inner["train_inner_subject_ids"])
        val_inner = set(inner["val_inner_subject_ids"])
        if train_inner.intersection(val_inner):
            raise ValueError(f"Turkish fold {fold_idx} inner train/val overlap.")
        if val_inner.intersection(final_eval):
            raise ValueError(f"Turkish fold {fold_idx} inner val/final eval overlap.")

    if set(holdout_counts) != all_subjects or any(count != 1 for count in holdout_counts.values()):
        raise ValueError("Each Turkish subject must be held out in exactly one fold.")


def build_turkish_manifest(
    config: dict[str, Any],
    quarantine: dict[str, Any],
) -> dict[str, Any]:
    root = Path(config["dataset_root"])
    metadata_path = root / str(config.get("metadata_csv", DEFAULT_METADATA_CSV))
    transcript_path = root / str(config.get("transcript_file", DEFAULT_TRANSCRIPT_FILE))
    audio_dir = root / str(config.get("audio_dir", "all-files"))
    threshold = float(config.get("threshold", SOURCE_THRESHOLD))
    split_cfg = config.get("split", {})
    n_splits = int(split_cfg.get("outer_folds", 5))
    split_seed = int(split_cfg.get("seed", config.get("seed", 1337)))
    inner_val_ratio = float(split_cfg.get("inner_val_ratio", 0.2))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Turkish metadata CSV not found: {metadata_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Turkish transcript JSONL not found: {transcript_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Turkish audio directory not found: {audio_dir}")

    transcripts = _load_whisper_transcripts(transcript_path)
    csv.field_size_limit(max(csv.field_size_limit(), 10**9))

    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    extra_file_audit: list[dict[str, Any]] = []
    subject_labels: dict[str, int] = {}
    subject_metadata: dict[str, dict[str, Any]] = {}
    metadata_basenames: set[str] = set()

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "file_name",
            "label",
            "depresyon_skoru",
            "anksiyete_skoru",
            "patient_id",
            "w2v2_predicted_score",
            "label_t25",
            "target_t25",
        }
        missing_fields = sorted(required_fields - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f"Turkish metadata CSV is missing fields: {missing_fields}")

        for row_number, source_row in enumerate(reader, start=2):
            basename = Path(str(source_row["file_name"])).name
            if not basename:
                raise ValueError(f"Turkish metadata row {row_number} has an empty file_name.")
            if basename in metadata_basenames:
                raise ValueError(f"Duplicate Turkish metadata basename: {basename}")
            metadata_basenames.add(basename)

            sample_id = Path(basename).stem
            subject_id = str(source_row["patient_id"]).strip()
            if not subject_id:
                raise ValueError(f"Turkish metadata row {row_number} has an empty patient_id.")
            chunk_id = _parse_chunk_id(basename, subject_id)
            audio_path = audio_dir / basename
            transcript_payload = transcripts.get(basename)
            transcript = str((transcript_payload or {}).get("transcript", "")).strip()
            language = str((transcript_payload or {}).get("language", "")).strip()
            audio_found = audio_path.is_file()
            transcript_found = transcript_payload is not None
            transcript_nonempty = bool(transcript)

            score = float(source_row["depresyon_skoru"])
            source_label = int(float(source_row["label_t25"]))
            source_target = str(source_row["target_t25"]).strip().lower()
            expected_source_label = int(score >= SOURCE_THRESHOLD)
            expected_source_target = "depressed" if expected_source_label else "non_depressed"
            if source_label != expected_source_label or source_target != expected_source_target:
                raise ValueError(
                    f"Turkish source label mismatch for {basename}: score={score} "
                    f"label_t25={source_label} target_t25={source_target!r}"
                )
            label = int(score >= threshold)
            if threshold == SOURCE_THRESHOLD and label != source_label:
                raise ValueError(f"Turkish threshold-derived label mismatch for {basename}.")

            join_audit_rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "file_name": basename,
                    "audio_found": audio_found,
                    "transcript_found": transcript_found,
                    "transcript_nonempty": transcript_nonempty,
                    "transcript_language": language,
                    "label": label,
                }
            )

            if not audio_found or not transcript_found or not transcript_nonempty:
                if is_quarantined_missing(quarantine, "turkish", sample_id):
                    extra_file_audit.append(
                        {
                            "file": basename,
                            "sample_id": sample_id,
                            "reason": "quarantined_missing_required_input",
                            "audio_found": audio_found,
                            "transcript_found": transcript_found,
                            "transcript_nonempty": transcript_nonempty,
                        }
                    )
                    continue
                raise FileNotFoundError(
                    f"Turkish required input missing for {basename} | audio={audio_found} "
                    f"transcript={transcript_found} nonempty={transcript_nonempty}"
                )

            if language != "tr":
                extra_file_audit.append(
                    {
                        "file": basename,
                        "sample_id": sample_id,
                        "reason": "non_tr_language_tag",
                        "language": language,
                    }
                )

            previous_label = subject_labels.setdefault(subject_id, label)
            if previous_label != label:
                raise ValueError(f"Turkish subject has mixed labels: {subject_id}")

            comorbid = int(float(source_row["label"]))
            subject_metadata.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "label": label,
                    "label_text": label_text_from_int(label),
                    "score": score,
                    "comorbid": comorbid,
                    "anxiety_score": _optional_float(source_row["anksiyete_skoru"]),
                    "age": _optional_float(source_row.get("age")),
                    "marital_status": _optional_float(source_row.get("medeni_hal")),
                    "education": _optional_float(source_row.get("egitim")),
                    "occupation": _optional_float(source_row.get("meslek")),
                },
            )

            manifest_rows.append(
                {
                    "dataset": "turkish",
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "audio_paths": [str(audio_path)],
                    "transcript": transcript,
                    "transcript_path": str(transcript_path),
                    "label": label,
                    "label_text": label_text_from_int(label),
                    "score": score,
                    "split_original": "all",
                    "fold": "",
                    "chunk_id": chunk_id,
                    "question_id": "",
                    "start_time": "",
                    "end_time": "",
                    "language": "tr",
                    "gender": None,
                    "modality_mode": "single_audio_single_text",
                    "comorbid": comorbid,
                    "anxiety_score": _optional_float(source_row["anksiyete_skoru"]),
                    "threshold": threshold,
                    "w2v2_predicted_score": _optional_float(source_row["w2v2_predicted_score"]),
                }
            )

    disk_audio_basenames = {path.name for path in audio_dir.glob("*.wav")}
    for basename in sorted(disk_audio_basenames - metadata_basenames):
        transcript_payload = transcripts.get(basename)
        extra_file_audit.append(
            {
                "file": basename,
                "sample_id": Path(basename).stem,
                "reason": "no_label_row",
                "audio_found": True,
                "transcript_found": transcript_payload is not None,
                "language": str((transcript_payload or {}).get("language", "")).strip(),
            }
        )
    for basename in sorted(set(transcripts) - metadata_basenames - disk_audio_basenames):
        extra_file_audit.append(
            {
                "file": basename,
                "sample_id": Path(basename).stem,
                "reason": "transcript_without_audio_or_label",
                "audio_found": False,
                "transcript_found": True,
                "language": transcripts[basename]["language"],
            }
        )

    folds = assign_stratified_group_folds(subject_labels, n_splits=n_splits, seed=split_seed)
    verify_turkish_split_integrity(manifest_rows, folds, inner_val_ratio, split_seed)
    fold_report = subject_fold_report(folds, subject_labels)
    subject_sample_counts = Counter(str(row["subject_id"]) for row in manifest_rows)
    subject_rows = []
    for subject_id in sorted(subject_metadata):
        subject_rows.append(
            {
                **subject_metadata[subject_id],
                "num_samples": int(subject_sample_counts[subject_id]),
            }
        )

    sample_counts = Counter(int(row["label"]) for row in manifest_rows)
    subject_counts = Counter(subject_labels.values())
    LOGGER.info(
        "Turkish counts | samples=%s (depressed=%s non_depressed=%s) | "
        "subjects=%s (depressed=%s non_depressed=%s) | unlabeled_audio=%s",
        len(manifest_rows),
        sample_counts[1],
        sample_counts[0],
        len(subject_labels),
        subject_counts[1],
        subject_counts[0],
        len(disk_audio_basenames - metadata_basenames),
    )

    return {
        "manifest_rows": manifest_rows,
        "subject_rows": subject_rows,
        "folds": folds,
        "fold_report": fold_report,
        "join_audit_rows": join_audit_rows,
        "extra_file_audit": extra_file_audit,
        "split_source": "stratified_group_cv",
        "split_source_notes": (
            f"{n_splits}-fold subject-level CV grouped by patient_id; "
            f"inner_val_ratio={inner_val_ratio}; seed={split_seed}"
        ),
    }
