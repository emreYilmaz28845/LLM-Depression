from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.data.split_utils import assign_stratified_group_folds
from src.utils import get_logger, label_text_from_int


LOGGER = get_logger(__name__)
CANONICAL_RESPONSES = ["negative", "neutral", "positive"]


def build_eatd_manifest(config: dict[str, Any], _quarantine: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["dataset_root"])
    manifest_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    extra_file_audit: list[dict[str, Any]] = []
    raw_mismatches: list[dict[str, Any]] = []
    subject_counter = Counter()
    sample_counter = Counter()

    for subset_name in ["train", "validation"]:
        subset_dir = base_dir / subset_name
        for subject_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
            local_subject_id = subject_dir.name
            subject_id = f"{subset_name}__{local_subject_id}"
            raw_sds = float((subject_dir / "label.txt").read_text(encoding="utf-8").strip())
            index_sds = float((subject_dir / "new_label.txt").read_text(encoding="utf-8").strip())
            if abs(index_sds - raw_sds * 1.25) > 1e-6:
                raw_mismatches.append(
                    {
                        "subject_id": subject_id,
                        "raw_sds": raw_sds,
                        "index_sds": index_sds,
                    }
                )
            label = int(index_sds >= 53.0)
            subject_counter[label] += 1
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "source_subset": subset_name,
                    "source_subject_dir": str(subject_dir),
                    "label": label,
                    "label_text": label_text_from_int(label),
                    "raw_sds": raw_sds,
                    "index_sds": index_sds,
                }
            )
            wav_by_name = {path.stem: path for path in subject_dir.glob("*.wav") if not path.name.endswith("_out.wav")}
            txt_by_name = {
                path.stem: path
                for path in subject_dir.glob("*.txt")
                if path.name not in {"label.txt", "new_label.txt"}
            }
            for txt_name, txt_path in sorted(txt_by_name.items()):
                if txt_name not in CANONICAL_RESPONSES:
                    extra_file_audit.append(
                        {
                            "subject_id": subject_id,
                            "file_path": str(txt_path),
                            "note": "Ignoring non-canonical extra transcript file.",
                        }
                    )
            for response_name in CANONICAL_RESPONSES:
                wav_path = wav_by_name.get(response_name)
                txt_path = txt_by_name.get(response_name)
                if wav_path is None or txt_path is None:
                    raise FileNotFoundError(
                        f"EATD missing canonical response for subject_id={subject_id} response={response_name}"
                    )
                transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                manifest_rows.append(
                    {
                        "dataset": "eatd",
                        "subject_id": subject_id,
                        "sample_id": f"{subject_id}_{response_name}",
                        "audio_path": str(wav_path),
                        "audio_paths": [str(wav_path)],
                        "transcript": transcript,
                        "transcript_path": str(txt_path),
                        "label": label,
                        "label_text": label_text_from_int(label),
                        "score": index_sds,
                        "raw_sds": raw_sds,
                        "index_sds": index_sds,
                        "split_original": subset_name,
                        "fold": "",
                        "chunk_id": "",
                        "question_id": response_name,
                        "start_time": "",
                        "end_time": "",
                        "language": "zh",
                        "modality_mode": "single_audio_single_text",
                    }
                )
                sample_counter[label] += 1

    if raw_mismatches:
        raise ValueError(f"EATD raw/index SDS mismatch detected: {raw_mismatches[:10]}")

    if subject_counter[1] != 30 or subject_counter[0] != 132:
        raise ValueError(
            f"EATD pooled class count mismatch | depressed={subject_counter[1]} non_depressed={subject_counter[0]}"
        )
    LOGGER.info(
        "EATD pooled counts | depressed_subjects=%s non_depressed_subjects=%s depressed_samples=%s non_depressed_samples=%s",
        subject_counter[1],
        subject_counter[0],
        sample_counter[1],
        sample_counter[0],
    )

    subject_labels = {row["subject_id"]: int(row["label"]) for row in subject_rows}
    folds = assign_stratified_group_folds(subject_labels, n_splits=3, seed=int(config["seed"]))
    return {
        "manifest_rows": manifest_rows,
        "subject_rows": subject_rows,
        "folds": folds,
        "extra_file_audit": extra_file_audit,
    }

