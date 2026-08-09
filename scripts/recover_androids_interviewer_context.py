#!/usr/bin/env python3
"""Recover interviewer context preceding each Androids participant turn.

The Androids corpus provides full two-speaker interview WAVs and manual
participant-turn boundaries, but its existing ASR caches cover participant
clips only.  This tool treats the complement immediately before each
participant turn as *interviewer context*, transcribes it locally, and maps the
result back to that turn.

The context can contain a question, a follow-up, an acknowledgement, silence,
or a mixture.  The output therefore requires review before a field is promoted
to verified ``question_text``.

Canonical audio is never changed.  Temporary context WAVs are deleted after
each ASR batch.  Derived outputs default to the gitignored ``outputs/`` tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transcribe_turkish_qwen3asr import Qwen3ASRBackend
from src.data.androids import parse_androids_recording_id


DEFAULT_DATASET_ROOT = Path(
    "/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "androids_question_recovery"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"


@dataclass(frozen=True)
class InterviewerContextSpan:
    context_id: str
    recording_id: str
    subject_id: str
    turn_id: int
    full_audio_path: str
    participant_clip_path: str
    context_start: float
    context_end: float
    context_duration: float
    participant_start: float
    participant_end: float
    participant_duration: float
    sample_rate: int
    channels: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _timing_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def discover_interviewer_context_spans(
    dataset_root: Path,
    *,
    timing_path: Path | None = None,
    duration_tolerance: float = 0.02,
) -> list[InterviewerContextSpan]:
    """Map each participant turn to the non-participant interval before it."""

    timing_path = timing_path or dataset_root / "interview_timedata.csv"
    if not timing_path.is_file():
        raise FileNotFoundError(f"Androids timing file not found: {timing_path}")

    spans: list[InterviewerContextSpan] = []
    seen_recordings: set[str] = set()
    for line_number, row in enumerate(_timing_rows(timing_path), start=1):
        if not row or not row[0].strip():
            continue
        recording_id = row[0].strip()
        if recording_id in seen_recordings:
            raise ValueError(f"Duplicate Androids timing row for {recording_id}")
        seen_recordings.add(recording_id)

        identity = parse_androids_recording_id(recording_id)
        values = [value.strip() for value in row[1:] if value.strip()]
        if len(values) % 2:
            raise ValueError(
                f"Odd number of turn timestamps for {recording_id} on line {line_number}"
            )
        try:
            timestamps = [float(value) for value in values]
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestamp for {recording_id} on line {line_number}"
            ) from exc

        group = "HC" if identity["condition_code"] == "C" else "PT"
        full_audio_path = (
            dataset_root / "Interview-Task" / "audio" / group / f"{recording_id}.wav"
        )
        if not full_audio_path.is_file():
            raise FileNotFoundError(f"Androids full interview not found: {full_audio_path}")
        audio_info = sf.info(full_audio_path)
        audio_duration = float(audio_info.frames) / float(audio_info.samplerate)

        previous_participant_end = 0.0
        for turn_id, (participant_start, participant_end) in enumerate(
            zip(timestamps[::2], timestamps[1::2]), start=1
        ):
            if participant_start < previous_participant_end:
                raise ValueError(
                    f"Overlapping or unordered participant turns for {recording_id} turn {turn_id}"
                )
            if participant_end <= participant_start:
                raise ValueError(
                    f"Non-positive participant turn for {recording_id} turn {turn_id}"
                )
            if participant_end > audio_duration + duration_tolerance:
                raise ValueError(
                    f"Participant turn exceeds full audio for {recording_id} turn {turn_id}: "
                    f"{participant_end:.6f} > {audio_duration:.6f}"
                )

            participant_clip_path = (
                dataset_root
                / "Interview-Task"
                / "audio_clip"
                / recording_id
                / f"{recording_id}_{turn_id}.wav"
            )
            if not participant_clip_path.is_file():
                raise FileNotFoundError(
                    f"Androids participant clip not found: {participant_clip_path}"
                )

            context_start = previous_participant_end
            context_end = participant_start
            spans.append(
                InterviewerContextSpan(
                    context_id=f"{recording_id}_before_turn_{turn_id:02d}",
                    recording_id=recording_id,
                    subject_id=str(identity["subject_id"]),
                    turn_id=turn_id,
                    full_audio_path=str(full_audio_path),
                    participant_clip_path=str(participant_clip_path),
                    context_start=context_start,
                    context_end=context_end,
                    context_duration=context_end - context_start,
                    participant_start=participant_start,
                    participant_end=participant_end,
                    participant_duration=participant_end - participant_start,
                    sample_rate=int(audio_info.samplerate),
                    channels=int(audio_info.channels),
                )
            )
            previous_participant_end = participant_end

    if not spans:
        raise ValueError(f"No Androids interviewer context spans found in {timing_path}")
    return spans


def build_span_report(
    spans: Sequence[InterviewerContextSpan],
    *,
    dataset_root: Path,
    timing_path: Path,
) -> dict[str, Any]:
    durations = sorted(span.context_duration for span in spans)

    def percentile(fraction: float) -> float:
        return durations[round((len(durations) - 1) * fraction)]

    return {
        "dataset": "androids",
        "task": "interviewer_context_recovery",
        "dataset_root": str(dataset_root),
        "timing_path": str(timing_path),
        "timing_sha256": _sha256_file(timing_path),
        "mapping_rule": (
            "For participant turn i, interviewer context is the full-interview interval "
            "from the end of participant turn i-1 (or 0 for the first turn) to the start "
            "of participant turn i."
        ),
        "context_semantics": (
            "Candidate context may contain a question, follow-up, acknowledgement, silence, "
            "or a mixture. It is not verified question text."
        ),
        "num_recordings": len({span.recording_id for span in spans}),
        "num_participant_turns": len(spans),
        "num_context_spans": len(spans),
        "context_duration_total_sec": sum(durations),
        "context_duration_min_sec": durations[0],
        "context_duration_median_sec": percentile(0.5),
        "context_duration_p95_sec": percentile(0.95),
        "context_duration_max_sec": durations[-1],
        "zero_duration_contexts": sum(duration == 0 for duration in durations),
    }


def _load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            context_id = str(row.get("context_id", ""))
            if not context_id or context_id in rows:
                raise ValueError(
                    f"Invalid or duplicate context_id in {path} line {line_number}: {context_id!r}"
                )
            rows[context_id] = row
    return rows


def _validate_resume_rows(
    existing: dict[str, dict[str, Any]],
    spans: dict[str, InterviewerContextSpan],
) -> None:
    unknown = sorted(set(existing) - set(spans))
    if unknown:
        raise ValueError(f"Resume output contains unknown context IDs: {unknown[:5]}")
    for context_id, row in existing.items():
        span = spans[context_id]
        for key in ("context_start", "context_end", "participant_start", "participant_end"):
            if not math.isclose(float(row[key]), float(getattr(span, key)), abs_tol=1e-6):
                raise ValueError(f"Resume interval mismatch for {context_id}: {key}")


def _slice_context(span: InterviewerContextSpan, destination: Path) -> None:
    info = sf.info(span.full_audio_path)
    start_frame = max(0, int(math.floor(span.context_start * info.samplerate)))
    end_frame = min(info.frames, int(math.floor(span.context_end * info.samplerate)))
    audio, sample_rate = sf.read(
        span.full_audio_path,
        start=start_frame,
        stop=end_frame,
        dtype="float32",
        always_2d=False,
    )
    sf.write(destination, audio, sample_rate, subtype="PCM_16")


def transcribe_contexts(
    spans: Sequence[InterviewerContextSpan],
    *,
    output_path: Path,
    backend: Any,
    model_id: str,
    batch_size: int,
    min_context_seconds: float,
    resume: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if output_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )
    span_map = {span.context_id: span for span in spans}
    existing = _load_existing_rows(output_path) if output_path.exists() and resume else {}
    _validate_resume_rows(existing, span_map)
    completed = dict(existing)

    pending = [span for span in spans if span.context_id not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        eligible = [span for span in batch if span.context_duration >= min_context_seconds]
        results_by_id: dict[str, Any] = {}
        if eligible:
            with tempfile.TemporaryDirectory(prefix="androids_interviewer_context_") as temp_dir:
                paths: list[str] = []
                for span in eligible:
                    path = Path(temp_dir) / f"{span.context_id}.wav"
                    _slice_context(span, path)
                    paths.append(str(path))
                results = list(backend.transcribe(audio=paths, language=["Italian"] * len(paths)))
                if len(results) != len(eligible):
                    raise RuntimeError(
                        f"ASR returned {len(results)} results for {len(eligible)} contexts"
                    )
                results_by_id = dict(zip((span.context_id for span in eligible), results))

        for span in batch:
            result = results_by_id.get(span.context_id)
            text = str(getattr(result, "text", "")).strip() if result is not None else ""
            detected_language = (
                str(getattr(result, "language", "")).strip() if result is not None else ""
            )
            row = {
                **asdict(span),
                "dataset": "androids",
                "task": "interviewer_context_recovery",
                "mapping_scope": "immediately_preceding_nonparticipant_interval",
                "interviewer_context_transcript": text,
                "question_text_verified": False,
                "manual_review_required": True,
                "review_reason": (
                    "The interval can contain a question, follow-up, acknowledgement, silence, "
                    "or a mixture."
                ),
                "asr_status": (
                    "TRANSCRIBED"
                    if result is not None
                    else "SKIPPED_CONTEXT_SHORTER_THAN_MINIMUM"
                ),
                "asr_model": model_id if result is not None else "",
                "asr_language_arg": "Italian" if result is not None else "",
                "asr_detected_language": detected_language,
            }
            completed[span.context_id] = row

        _write_jsonl(output_path, (completed[span.context_id] for span in spans if span.context_id in completed))
        print(f"Recovered {len(completed)}/{len(spans)} interviewer contexts", flush=True)

    return [completed[span.context_id] for span in spans]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--timing-path", type=Path)
    parser.add_argument(
        "--span-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "androids_interviewer_context_spans.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "androids_interviewer_context_qwen3_asr_italian.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "androids_interviewer_context_report.json",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-context-seconds", type=float, default=0.15)
    parser.add_argument(
        "--skip-forced-aligner-dependency",
        action="store_true",
        help=(
            "Stub qwen-asr's unused nagisa import. Safe only because this tool never "
            "requests forced-alignment timestamps."
        ),
    )
    return parser.parse_args()


def _stub_unused_nagisa_dependency() -> None:
    """Allow ASR-only qwen-asr use without the optional DyNet runtime.

    qwen-asr imports its forced-aligner module eagerly. That module imports
    ``nagisa``, whose DyNet runtime is not needed unless forced alignment is
    actually called. This recovery already has corpus-supplied timestamps and
    never asks qwen-asr for word timestamps.
    """

    stub = types.ModuleType("nagisa")

    def _unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "nagisa is disabled for Androids interviewer-context recovery; "
            "forced alignment is not supported in this job."
        )

    stub.tagging = _unsupported  # type: ignore[attr-defined]
    sys.modules["nagisa"] = stub


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.min_context_seconds < 0:
        raise ValueError("--min-context-seconds cannot be negative")

    dataset_root = args.dataset_root.resolve()
    timing_path = (args.timing_path or dataset_root / "interview_timedata.csv").resolve()
    all_spans = discover_interviewer_context_spans(
        dataset_root, timing_path=timing_path
    )
    spans = all_spans[: args.limit] if args.limit is not None else all_spans
    _write_jsonl(args.span_manifest, (asdict(span) for span in spans))
    report = build_span_report(spans, dataset_root=dataset_root, timing_path=timing_path)
    report["limited_run"] = args.limit is not None
    report["full_inventory_num_context_spans"] = len(all_spans)

    if not args.extract_only:
        if args.skip_forced_aligner_dependency:
            _stub_unused_nagisa_dependency()
        backend = Qwen3ASRBackend(
            args.model,
            device=args.device,
            dtype=args.dtype,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation or None,
        )
        rows = transcribe_contexts(
            spans,
            output_path=args.out,
            backend=backend,
            model_id=args.model,
            batch_size=args.batch_size,
            min_context_seconds=args.min_context_seconds,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        report.update(
            {
                "transcript_output_path": str(args.out),
                "transcript_output_sha256": _sha256_file(args.out),
                "asr_model": args.model,
                "asr_rows": len(rows),
                "asr_transcribed_rows": sum(row["asr_status"] == "TRANSCRIBED" for row in rows),
                "asr_nonempty_rows": sum(bool(row["interviewer_context_transcript"]) for row in rows),
                "manual_review_required_rows": sum(row["manual_review_required"] for row in rows),
            }
        )

    report["span_manifest_path"] = str(args.span_manifest)
    report["span_manifest_sha256"] = _sha256_file(args.span_manifest)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
