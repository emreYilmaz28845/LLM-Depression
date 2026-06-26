#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - existing env already uses soundfile
    sf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir  # noqa: E402


DEFAULT_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/Turkish")
DEFAULT_METADATA = "metadata_turkish_t25_binary_merged.csv"
DEFAULT_TRANSCRIPTS = "whisper_transcripts.jsonl"
DEFAULT_AUDIO_DIR = "all-files"

WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû']+", re.UNICODE)
NATURAL_SPLIT_RE = re.compile(r"(\d+)")
TERMINAL_PUNCTUATION = (".", "!", "?", "…")
ELLIPSIS_ENDINGS = ("...", "…")
ALLOWED_TRANSCRIPT_CHARS = set(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ÇĞİÖŞÜçğıöşüÂÎÛâîû"
    ".,!?;:'\"()[]{}<>-/%&+*=#@_ \n\r\t…"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Turkish Whisper transcripts.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA)
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "reviews" / "turkish_transcripts"),
    )
    return parser.parse_args()


def natural_sort_key(value: str) -> list[Any]:
    parts = NATURAL_SPLIT_RE.split(value)
    return [int(part) if part.isdigit() else part for part in parts]


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def parse_subject_id_from_filename(basename: str) -> str:
    parts = basename.split("-")
    if not parts:
        raise ValueError(f"Could not parse subject id from filename {basename!r}")
    return parts[0]


def parse_question_index(basename: str) -> int | None:
    parts = Path(basename).stem.split("-")
    if len(parts) < 3:
        return None
    raw_value = parts[2]
    return int(raw_value) if raw_value.isdigit() else None


def parse_prompt_group(basename: str) -> str:
    parts = Path(basename).stem.split("-")
    if len(parts) < 4:
        return ""
    return parts[3]


def repeated_clause_details(text: str) -> dict[str, int]:
    clauses: list[str] = []
    for chunk in re.split(r"[,.!?;:\n]+", text.lower()):
        words = tokenize(chunk)
        if len(words) >= 3:
            clauses.append(" ".join(words))
    clause_counts = Counter(clauses)
    return {clause: count for clause, count in clause_counts.items() if count > 1}


def repeated_ngram_ratio(words: list[str], n: int) -> float:
    if len(words) < n * 2:
        return 0.0
    ngrams = [" ".join(words[idx : idx + n]) for idx in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def transcript_has_odd_chars(text: str) -> tuple[bool, str]:
    odd_chars = sorted({char for char in text if char not in ALLOWED_TRANSCRIPT_CHARS})
    return bool(odd_chars), "".join(odd_chars)


def read_duration_seconds(audio_path: Path) -> float | None:
    if sf is None:
        return None
    info = sf.info(audio_path)
    if info.samplerate <= 0:
        return None
    return float(info.frames / info.samplerate)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    csv.field_size_limit(max(csv.field_size_limit(), 10**9))
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basename = Path(str(row["file_name"])).name
            rows[basename] = {
                "subject_id": str(row["patient_id"]).strip(),
                "row": row,
            }
    return rows


def build_rows(
    root: Path,
    metadata: dict[str, dict[str, Any]],
    transcript_path: Path,
    audio_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            basename = Path(str(payload.get("audio_path", ""))).name
            if not basename:
                raise ValueError(f"Transcript row {line_number} has no audio basename.")
            metadata_entry = metadata.get(basename)
            subject_id = (
                metadata_entry["subject_id"]
                if metadata_entry is not None
                else parse_subject_id_from_filename(basename)
            )
            text = str(payload.get("transcript", "")).strip()
            words = tokenize(text)
            duration_seconds = read_duration_seconds(audio_dir / basename)
            odd_chars_present, odd_chars = transcript_has_odd_chars(text)
            rows.append(
                {
                    "audio_filename": basename,
                    "subject_id": subject_id,
                    "question_index": parse_question_index(basename),
                    "prompt_group": parse_prompt_group(basename),
                    "in_metadata": metadata_entry is not None,
                    "language": str(payload.get("language", "")).strip(),
                    "transcript": text,
                    "transcript_preview": text[:220],
                    "word_count": len(words),
                    "char_count": len(text),
                    "duration_seconds": duration_seconds,
                    "words_per_second": (
                        len(words) / duration_seconds
                        if duration_seconds not in (None, 0.0)
                        else None
                    ),
                    "repeated_clause_details": repeated_clause_details(text),
                    "repeated_bigram_ratio": repeated_ngram_ratio(words, n=2),
                    "repeated_trigram_ratio": repeated_ngram_ratio(words, n=3),
                    "unique_token_ratio": (len(set(words)) / len(words)) if words else 0.0,
                    "ends_with_ellipsis": text.endswith(ELLIPSIS_ENDINGS),
                    "starts_lowercase": bool(text[:1].islower()),
                    "ends_with_terminal_punct": text.endswith(TERMINAL_PUNCTUATION),
                    "odd_chars_present": odd_chars_present,
                    "odd_chars": odd_chars,
                }
            )
    rows.sort(key=lambda row: (row["subject_id"], natural_sort_key(row["audio_filename"])))
    return rows


def annotate_subject_context(rows: list[dict[str, Any]]) -> None:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)

    for subject_rows in by_subject.values():
        word_counts = [int(row["word_count"]) for row in subject_rows]
        median_words = statistics.median(word_counts)
        if len(word_counts) >= 4:
            sorted_counts = sorted(word_counts)
            q1 = statistics.median(sorted_counts[: len(sorted_counts) // 2])
            q3 = statistics.median(sorted_counts[(len(sorted_counts) + 1) // 2 :])
            iqr = max(1.0, q3 - q1)
        else:
            q1 = min(word_counts)
            q3 = max(word_counts)
            iqr = max(1.0, q3 - q1)

        for index, row in enumerate(subject_rows):
            prev_row = subject_rows[index - 1] if index > 0 else None
            next_row = subject_rows[index + 1] if index + 1 < len(subject_rows) else None
            likely_continuation = False
            if row["word_count"] <= 6:
                if prev_row is not None and (
                    prev_row["ends_with_ellipsis"] or not prev_row["ends_with_terminal_punct"]
                ):
                    likely_continuation = True
                if row["starts_lowercase"]:
                    likely_continuation = True
                if next_row is not None and next_row["starts_lowercase"]:
                    likely_continuation = True

            row["subject_file_count"] = len(subject_rows)
            row["subject_word_count_median"] = median_words
            row["subject_word_count_q1"] = q1
            row["subject_word_count_q3"] = q3
            row["subject_word_count_iqr"] = iqr
            row["subject_word_count_outlier_low"] = (
                row["word_count"] < median_words - max(12.0, 1.5 * iqr)
            )
            row["subject_word_count_outlier_high"] = (
                row["word_count"] > median_words + max(18.0, 1.5 * iqr)
            )
            row["previous_audio_filename"] = prev_row["audio_filename"] if prev_row else ""
            row["next_audio_filename"] = next_row["audio_filename"] if next_row else ""
            row["likely_continuation_fragment"] = likely_continuation
            row["subject_context_semantic_break"] = (
                row["word_count"] <= 6
                and median_words >= 20
                and not likely_continuation
            )


def assign_quality(row: dict[str, Any]) -> None:
    issues: list[str] = []
    notes: list[str] = []

    duration_seconds = row["duration_seconds"]
    long_clip = duration_seconds is not None and duration_seconds >= 15.0
    word_count = int(row["word_count"])
    language = str(row["language"])
    repetition_strong = (
        row["repeated_trigram_ratio"] >= 0.20
        or any(count >= 2 for count in row["repeated_clause_details"].values())
    )
    possible_hallucination = (
        repetition_strong and row["unique_token_ratio"] < 0.45 and word_count >= 15
    )
    wrong_context = (
        (language not in ("", "tr") or row["odd_chars_present"])
        and word_count <= 8
    )

    if language not in ("", "tr"):
        issues.append("language_mismatch")
        notes.append(f"language tag is {language!r}, not 'tr'")
    if row["odd_chars_present"]:
        issues.append("odd_characters")
        notes.append(f"unexpected characters {row['odd_chars']!r}")
    if word_count <= 6:
        issues.append("too_short")
        if duration_seconds is None:
            notes.append(f"{word_count} words")
        elif long_clip:
            notes.append(f"{word_count} words over {duration_seconds:.1f}s clip")
        else:
            notes.append(f"{word_count} words in short {duration_seconds:.1f}s tail clip")
    if row["ends_with_ellipsis"] or row["likely_continuation_fragment"]:
        issues.append("truncated")
        if row["likely_continuation_fragment"]:
            notes.append("looks like a clipped continuation fragment")
        else:
            notes.append("ends with ellipsis")
    if repetition_strong:
        issues.append("repetition")
        notes.append("contains repeated clauses or repeated n-grams")
    if possible_hallucination:
        issues.append("possible_hallucination")
        notes.append("repetition pattern looks stronger than normal hesitation")
    if wrong_context:
        issues.append("wrong_context")
        notes.append("breaks otherwise Turkish/transcript context")
    if row["subject_context_semantic_break"] and "wrong_context" in issues:
        issues.append("semantic_break")
        notes.append("stands out sharply from the rest of the subject session")

    # Deduplicate while keeping stable order.
    stable_issues: list[str] = []
    for issue in issues:
        if issue not in stable_issues:
            stable_issues.append(issue)
    stable_notes: list[str] = []
    for note in notes:
        if note not in stable_notes:
            stable_notes.append(note)

    if "wrong_context" in stable_issues or (
        "language_mismatch" in stable_issues and "odd_characters" in stable_issues
    ):
        quality_label = "very_bad"
    elif "possible_hallucination" in stable_issues:
        quality_label = "likely_bad"
    elif "too_short" in stable_issues and long_clip:
        quality_label = "needs_audio_check"
    elif stable_issues:
        quality_label = "minor_issue"
    else:
        quality_label = "ok"

    row["issue_codes"] = stable_issues
    row["quality_label"] = quality_label
    row["note"] = "; ".join(stable_notes)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_counts = Counter(row["quality_label"] for row in rows)
    labeled_rows = [row for row in rows if row["in_metadata"]]
    labeled_counts = Counter(row["quality_label"] for row in labeled_rows)
    all_issue_counts = Counter(
        issue for row in rows for issue in row["issue_codes"]
    )
    labeled_issue_counts = Counter(
        issue for row in labeled_rows for issue in row["issue_codes"]
    )
    high_risk_labels = {"very_bad", "likely_bad", "needs_audio_check"}
    subjects_with_high_risk: Counter[str] = Counter(
        row["subject_id"] for row in labeled_rows if row["quality_label"] in high_risk_labels
    )
    question_index_counts = Counter(
        row["question_index"]
        for row in labeled_rows
        if row["quality_label"] in high_risk_labels and row["question_index"] is not None
    )

    def serialize_counts(counter: Counter[str]) -> dict[str, dict[str, float]]:
        total = sum(counter.values())
        rendered: dict[str, dict[str, float]] = {}
        for key in ("ok", "minor_issue", "needs_audio_check", "likely_bad", "very_bad"):
            count = int(counter.get(key, 0))
            rendered[key] = {
                "count": count,
                "percent": (100.0 * count / total) if total else 0.0,
            }
        return rendered

    examples: list[dict[str, Any]] = []
    for label in ("very_bad", "likely_bad", "needs_audio_check"):
        label_rows = [row for row in labeled_rows if row["quality_label"] == label]
        for row in label_rows[:5]:
            examples.append(
                {
                    "audio_filename": row["audio_filename"],
                    "subject_id": row["subject_id"],
                    "quality_label": row["quality_label"],
                    "issue_codes": row["issue_codes"],
                    "note": row["note"],
                    "transcript_preview": row["transcript_preview"],
                }
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "all_transcripts": len(rows),
            "labeled_transcripts": len(labeled_rows),
            "unlabeled_transcripts": len(rows) - len(labeled_rows),
            "subjects_all": len({row["subject_id"] for row in rows}),
            "subjects_labeled": len({row["subject_id"] for row in labeled_rows}),
        },
        "quality_counts_all": serialize_counts(all_counts),
        "quality_counts_labeled": serialize_counts(labeled_counts),
        "issue_counts_all": dict(all_issue_counts.most_common()),
        "issue_counts_labeled": dict(labeled_issue_counts.most_common()),
        "subjects_with_many_high_risk_labeled": [
            {"subject_id": subject_id, "high_risk_count": count}
            for subject_id, count in subjects_with_high_risk.most_common()
            if count >= 1
        ],
        "high_risk_question_indices_labeled": [
            {"question_index": int(question_index), "count": int(count)}
            for question_index, count in question_index_counts.most_common()
        ],
        "examples_labeled": examples,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "audio_filename",
        "subject_id",
        "question_index",
        "prompt_group",
        "in_metadata",
        "language",
        "duration_seconds",
        "word_count",
        "char_count",
        "words_per_second",
        "subject_word_count_median",
        "subject_word_count_outlier_low",
        "subject_word_count_outlier_high",
        "likely_continuation_fragment",
        "quality_label",
        "issue_codes",
        "note",
        "transcript_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["issue_codes"] = "|".join(row["issue_codes"])
            writer.writerow(csv_row)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    metadata_path = root / str(args.metadata_csv)
    transcript_path = root / str(args.transcripts)
    audio_dir = root / str(args.audio_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript JSONL not found: {transcript_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    metadata = load_metadata(metadata_path)
    rows = build_rows(
        root=root,
        metadata=metadata,
        transcript_path=transcript_path,
        audio_dir=audio_dir,
    )
    annotate_subject_context(rows)
    for row in rows:
        assign_quality(row)

    report_csv = output_dir / "turkish_transcript_review.csv"
    report_jsonl = output_dir / "turkish_transcript_review.jsonl"
    summary_json = output_dir / "turkish_transcript_review_summary.json"

    write_csv(rows, report_csv)
    write_jsonl(rows, report_jsonl)
    summary = build_summary(rows)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"report_csv": str(report_csv), "report_jsonl": str(report_jsonl), "summary_json": str(summary_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
