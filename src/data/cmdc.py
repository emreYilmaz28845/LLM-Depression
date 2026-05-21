from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.split_utils import (
    assign_stratified_group_folds,
    build_cmdc_official_folds,
    cmdc_fold_report,
    validate_non_overlapping_folds,
)
from src.data.validation import is_quarantined_missing
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)


def build_cmdc_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    subject_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir()])
    manifest_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    extra_file_audit: list[dict[str, Any]] = []
    sample_counter = Counter()
    subject_counter = Counter()

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        label = 1 if subject_id.startswith("MDD") else 0
        subject_counter[label] += 1
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "label_text": label_text_from_int(label),
            }
        )
        wav_by_qid = {path.stem: path for path in subject_dir.glob("Q*.wav")}
        txt_by_qid = {path.stem: path for path in subject_dir.glob("Q*.txt")}
        all_qids = sorted(set(wav_by_qid) | set(txt_by_qid))
        for qid in all_qids:
            sample_id = f"{subject_id}_{qid}"
            has_wav = qid in wav_by_qid
            has_txt = qid in txt_by_qid
            if not has_wav or not has_txt:
                if is_quarantined_missing(quarantine, "cmdc", sample_id):
                    extra_file_audit.append(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "question_id": qid,
                            "wav_found": has_wav,
                            "transcript_found": has_txt,
                            "quarantined": True,
                        }
                    )
                    continue
                raise FileNotFoundError(
                    f"CMDC missing canonical pair for sample_id={sample_id} | wav_found={has_wav} transcript_found={has_txt}"
                )
            transcript = txt_by_qid[qid].read_text(encoding="utf-8", errors="ignore").strip()
            manifest_rows.append(
                {
                    "dataset": "cmdc",
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "audio_path": str(wav_by_qid[qid]),
                    "audio_paths": [str(wav_by_qid[qid])],
                    "transcript": transcript,
                    "transcript_path": str(txt_by_qid[qid]),
                    "label": label,
                    "label_text": label_text_from_int(label),
                    "score": "",
                    "split_original": "",
                    "fold": "",
                    "chunk_id": "",
                    "question_id": qid,
                    "start_time": "",
                    "end_time": "",
                    "language": "zh",
                    "modality_mode": "single_audio_single_text",
                }
            )
            sample_counter[label] += 1

    subject_ids = [row["subject_id"] for row in subject_rows]
    subject_labels = {row["subject_id"]: int(row["label"]) for row in subject_rows}
    workbook_folds = build_cmdc_official_folds(subject_ids, base_dir / "5FoldSplitInfo.xlsx", strict=False)
    workbook_fold_report = cmdc_fold_report(workbook_folds, subject_labels)
    split_source = "official_workbook"
    split_source_notes = "Accepted workbook fold definitions."
    try:
        validate_non_overlapping_folds(workbook_folds, subject_ids)
        folds = workbook_folds
        fold_report = workbook_fold_report
    except ValueError as exc:
        split_source = "deterministic_subject_level_fallback"
        split_source_notes = (
            "Workbook fold definitions failed the zero-overlap proof and were replaced with a deterministic "
            "subject-level 5-fold stratified split."
        )
        LOGGER.warning("CMDC workbook folds failed validation: %s", exc)
        folds = assign_stratified_group_folds(subject_labels, n_splits=5, seed=int(config["seed"]))
        validate_non_overlapping_folds(folds, subject_ids)
        fold_report = cmdc_fold_report(folds, subject_labels)
    for item in fold_report:
        LOGGER.info(
            "CMDC fold=%s | heldout_mdd=%s heldout_hc=%s outer_train_mdd=%s outer_train_hc=%s outer_train=%s",
            item["fold"],
            item["heldout_mdd_count"],
            item["heldout_hc_count"],
            item["outer_train_class_counts"]["Depressed"],
            item["outer_train_class_counts"]["Non-depressed"],
            item["outer_train_count"],
        )
        LOGGER.info("CMDC fold=%s heldout MDD ids: %s", item["fold"], ", ".join(item["heldout_mdd_subject_ids"]))
        LOGGER.info("CMDC fold=%s heldout HC ids: %s", item["fold"], ", ".join(item["heldout_hc_subject_ids"]))

    LOGGER.info(
        "CMDC pooled counts | depressed_subjects=%s non_depressed_subjects=%s depressed_samples=%s non_depressed_samples=%s",
        subject_counter[1],
        subject_counter[0],
        sample_counter[1],
        sample_counter[0],
    )
    return {
        "manifest_rows": manifest_rows,
        "subject_rows": subject_rows,
        "folds": folds,
        "fold_report": fold_report,
        "workbook_fold_report": workbook_fold_report,
        "split_source": split_source,
        "split_source_notes": split_source_notes,
        "extra_file_audit": extra_file_audit,
    }
