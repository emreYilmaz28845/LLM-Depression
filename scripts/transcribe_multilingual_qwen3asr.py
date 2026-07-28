#!/usr/bin/env python3
"""Transcribe D3TEC and ANDROIDS with the proven Turkish Qwen3-ASR backend.

This is a dataset-aware wrapper around ``transcribe_turkish_qwen3asr.py``.  It
reuses that runner's crash-safe batch writing, per-file retry, resume support,
duration QC, and report generation while keeping the completed Turkish workflow
unchanged.

Presets:

* ``d3tec``: Spanish transcripts from the canonical SM-27 recordings.  Each row
  also records the normalized paired iPhone path.
* ``androids_interview``: Italian transcripts from participant-only interview
  turns under ``Interview-Task/audio_clip`` (not the interviewer-containing
  whole recordings).
* ``androids_reading``: Italian transcripts from the reading recordings.

Examples:

    python scripts/transcribe_multilingual_qwen3asr.py --preset d3tec --limit 4
    python scripts/transcribe_multilingual_qwen3asr.py --preset androids_interview
    python scripts/transcribe_multilingual_qwen3asr.py --preset androids_reading

Use ``--out`` for smoke artifacts.  A pre-existing output is never truncated
unless ``--overwrite`` is explicit; use ``--resume`` to continue a partial run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import transcribe_turkish_qwen3asr as core  # noqa: E402
from src.data.d3tec import discover_d3tec_response_windows  # noqa: E402
from src.utils import configure_logging, ensure_dir, get_logger  # noqa: E402


LOGGER = get_logger(__name__)

D3TEC_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET")
ANDROIDS_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus")


@dataclass(frozen=True)
class Preset:
    name: str
    dataset: str
    audio_dir: Path
    output_path: Path
    language_arg: str
    language_tag: str
    expected_detected_languages: frozenset[str]
    recursive: bool
    default_batch_size: int
    default_max_new_tokens: int
    default_chunk_threshold_seconds: float
    default_chunk_seconds: float
    enrich: Callable[[dict[str, Any]], dict[str, Any]]


def _d3tec_enrichment(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"]))
    match = re.fullmatch(r"(?P<subject>\d+)(?:_(?P<prompt>\d+))?", audio_path.stem)
    if match is None:
        raise ValueError(f"Unexpected D3TEC SM-27 basename: {audio_path.name}")
    subject_id = match.group("subject")
    prompt_id = match.group("prompt") or "0"
    iphone_stem = f"{subject_id}_cel" + (f"_{prompt_id}" if prompt_id != "0" else "")
    paired_path = D3TEC_ROOT / "iPhoneSE2020" / f"{iphone_stem}.wav"
    return {
        **row,
        "dataset": "d3tec",
        "subject_id": subject_id,
        "sample_id": f"{subject_id}_{prompt_id}",
        "prompt_id": prompt_id,
        "source_device": "SM-27",
        "paired_iphone_audio_path": str(paired_path),
        "paired_iphone_audio_found": paired_path.is_file(),
    }


def _parse_androids_speaker(recording_id: str) -> dict[str, Any]:
    parts = recording_id.split("_")
    if len(parts) != 3 or len(parts[1]) < 2:
        raise ValueError(f"Unexpected ANDROIDS recording id: {recording_id}")
    numeric_id, demographics, education = parts
    condition = demographics[0]
    if condition not in {"C", "P"}:
        raise ValueError(f"Unexpected ANDROIDS condition in recording id: {recording_id}")
    return {
        "subject_id": f"{numeric_id}_{condition}",
        "recording_id": recording_id,
        "condition_code": condition,
        "diagnosis": "control" if condition == "C" else "depression",
        "gender": demographics[1] if len(demographics) >= 2 else "",
        "age": int(demographics[2:]) if demographics[2:].isdigit() else None,
        "education_level": int(education) if education.isdigit() else None,
    }


def _androids_interview_enrichment(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"]))
    recording_id = audio_path.parent.name
    prefix = f"{recording_id}_"
    if not audio_path.stem.startswith(prefix):
        raise ValueError(
            f"ANDROIDS interview clip does not match its parent recording: {audio_path}"
        )
    turn_text = audio_path.stem[len(prefix) :]
    if not turn_text.isdigit():
        raise ValueError(f"Unexpected ANDROIDS interview turn id: {audio_path.name}")
    speaker = _parse_androids_speaker(recording_id)
    return {
        **row,
        "dataset": "androids",
        "task": "interview",
        **speaker,
        "turn_id": int(turn_text),
        "sample_id": audio_path.stem,
    }


def _androids_reading_enrichment(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"]))
    speaker = _parse_androids_speaker(audio_path.stem)
    return {
        **row,
        "dataset": "androids",
        "task": "reading",
        **speaker,
        "sample_id": f"{audio_path.stem}_reading",
    }


PRESETS: dict[str, Preset] = {
    "d3tec": Preset(
        name="d3tec",
        dataset="d3tec",
        audio_dir=D3TEC_ROOT / "SM-27",
        output_path=D3TEC_ROOT / "transcripts_qwen3_asr_spanish.jsonl",
        language_arg="Spanish",
        language_tag="es",
        expected_detected_languages=frozenset({"spanish", "es"}),
        recursive=False,
        default_batch_size=16,
        default_max_new_tokens=512,
        default_chunk_threshold_seconds=120.0,
        default_chunk_seconds=60.0,
        enrich=_d3tec_enrichment,
    ),
    "androids_interview": Preset(
        name="androids_interview",
        dataset="androids",
        audio_dir=ANDROIDS_ROOT / "Interview-Task" / "audio_clip",
        output_path=ANDROIDS_ROOT / "interview_transcripts_qwen3_asr_italian.jsonl",
        language_arg="Italian",
        language_tag="it",
        expected_detected_languages=frozenset({"italian", "it"}),
        recursive=True,
        default_batch_size=8,
        default_max_new_tokens=512,
        default_chunk_threshold_seconds=120.0,
        default_chunk_seconds=60.0,
        enrich=_androids_interview_enrichment,
    ),
    "androids_reading": Preset(
        name="androids_reading",
        dataset="androids",
        audio_dir=ANDROIDS_ROOT / "Reading-Task" / "audio",
        output_path=ANDROIDS_ROOT / "reading_transcripts_qwen3_asr_italian.jsonl",
        language_arg="Italian",
        language_tag="it",
        expected_detected_languages=frozenset({"italian", "it"}),
        recursive=True,
        default_batch_size=8,
        default_max_new_tokens=512,
        default_chunk_threshold_seconds=120.0,
        default_chunk_seconds=60.0,
        enrich=_androids_reading_enrichment,
    ),
}


class _CombinedResult:
    def __init__(self, text: str, language: str) -> None:
        self.text = text
        self.language = language


class LongAudioChunkingTranscriber:
    """Split exceptionally long recordings before ASR and reassemble their text.

    Qwen3-ASR handled the 111-second D3TEC smoke file well, but the 530-second
    ANDROIDS turn produced a severely incomplete result.  This wrapper leaves
    ordinary recordings untouched and splits only files above ``threshold`` into
    contiguous WAV chunks.  Chunk files live in a temporary directory and are
    deleted after each transcribe call.
    """

    def __init__(
        self,
        backend: core.Transcriber,
        *,
        threshold_seconds: float,
        chunk_seconds: float,
        segment_batch_size: int,
    ) -> None:
        if threshold_seconds <= 0 or chunk_seconds <= 0:
            raise ValueError("Long-audio chunk thresholds must be positive.")
        self.backend = backend
        self.threshold_seconds = float(threshold_seconds)
        self.chunk_seconds = float(chunk_seconds)
        self.segment_batch_size = max(1, int(segment_batch_size))

    def transcribe(self, audio: Sequence[str], language: Any) -> Sequence[Any]:
        paths = [Path(path) for path in audio]
        language_values = (
            list(language)
            if isinstance(language, (list, tuple))
            else [language] * len(paths)
        )
        if len(language_values) != len(paths):
            raise ValueError("Language arguments do not align with audio paths.")

        results: list[Any | None] = [None] * len(paths)
        short_indices: list[int] = []
        for index, path in enumerate(paths):
            duration = core.read_duration_seconds(path)
            if duration is None or duration <= self.threshold_seconds:
                short_indices.append(index)

        if short_indices:
            short_results = self.backend.transcribe(
                audio=[str(paths[index]) for index in short_indices],
                language=[language_values[index] for index in short_indices],
            )
            if len(short_results) != len(short_indices):
                raise RuntimeError("Qwen3-ASR returned the wrong number of short-audio results.")
            for index, result in zip(short_indices, short_results):
                results[index] = result

        for index, path in enumerate(paths):
            if results[index] is not None:
                continue
            results[index] = self._transcribe_long(path, language_values[index])

        return [result for result in results if result is not None]

    def _transcribe_long(self, path: Path, language_value: Any) -> _CombinedResult:
        if core.sf is None:
            raise RuntimeError("soundfile is required to chunk long recordings.")
        audio, sample_rate = core.sf.read(str(path), dtype="float32", always_2d=False)
        frames_per_chunk = max(1, int(round(self.chunk_seconds * sample_rate)))
        n_chunks = int(math.ceil(len(audio) / frames_per_chunk))
        LOGGER.info(
            "Long-audio ASR | %s | %.1fs -> %d chunk(s) of at most %.1fs",
            path.name,
            len(audio) / sample_rate,
            n_chunks,
            self.chunk_seconds,
        )
        texts: list[str] = []
        detected_languages: list[str] = []
        with tempfile.TemporaryDirectory(prefix="qwen3asr_chunks_") as tmp:
            tmp_path = Path(tmp)
            segment_paths: list[Path] = []
            for chunk_index in range(n_chunks):
                start = chunk_index * frames_per_chunk
                end = min(len(audio), start + frames_per_chunk)
                segment_path = tmp_path / f"{path.stem}_part_{chunk_index + 1:03d}.wav"
                core.sf.write(str(segment_path), audio[start:end], sample_rate, subtype="PCM_16")
                segment_paths.append(segment_path)

            for start in range(0, len(segment_paths), self.segment_batch_size):
                batch = segment_paths[start : start + self.segment_batch_size]
                batch_results = self.backend.transcribe(
                    audio=[str(segment) for segment in batch],
                    language=[language_value] * len(batch),
                )
                if len(batch_results) != len(batch):
                    raise RuntimeError(
                        f"Qwen3-ASR returned {len(batch_results)}/{len(batch)} long-audio segments."
                    )
                for result in batch_results:
                    text = str(getattr(result, "text", "") or "").strip()
                    if text:
                        texts.append(text)
                    detected = str(getattr(result, "language", "") or "").strip()
                    if detected:
                        detected_languages.append(detected)
        detected_language = detected_languages[0] if detected_languages else ""
        return _CombinedResult(text="\n".join(texts), language=detected_language)


def discover_files(preset: Preset) -> list[Path]:
    iterator = preset.audio_dir.rglob("*.wav") if preset.recursive else preset.audio_dir.glob("*.wav")
    files = sorted(iterator, key=lambda path: core.natural_sort_key(str(path.relative_to(preset.audio_dir))))
    seen: dict[str, Path] = {}
    for path in files:
        if path.name in seen:
            raise ValueError(
                f"Duplicate basename {path.name!r}: {seen[path.name]} vs {path}. "
                "Resume and coverage use basenames as stable identifiers."
            )
        seen[path.name] = path
    return files


def _language_aware_review(
    preset: Preset,
    *,
    text: str,
    duration_sec: float | None,
    detected_language: str,
    failed: bool,
) -> list[str]:
    codes: list[str] = []
    if failed:
        codes.append("transcription_failed")
    if not text:
        codes.append("empty_transcript")
    if duration_sec is not None and duration_sec < core.ULTRA_SHORT_SEC:
        codes.append("ultra_short_clip")
    if (
        text
        and duration_sec is not None
        and duration_sec >= core.MIN_CHAR_RATE_DUR_SEC
        and (len(text) / duration_sec) < core.MIN_CHAR_RATE
    ):
        codes.append("low_char_rate")
    detected = core._normalize_lang(detected_language)
    if detected and detected not in preset.expected_detected_languages:
        codes.append("language_mismatch")
    return codes


def _configure_core_for_preset(preset: Preset) -> None:
    core.OUTPUT_LANGUAGE_TAG = preset.language_tag

    def review(**kwargs: Any) -> list[str]:
        return _language_aware_review(preset, **kwargs)

    core.manual_review_reason_codes = review


def enrich_output(path: Path, preset: Preset) -> list[dict[str, Any]]:
    rows = core._read_rows(path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched_row = preset.enrich(row)
        duration = row.get("audio_duration_sec")
        threshold = row.get("asr_chunk_threshold_seconds")
        chunk_seconds = row.get("asr_chunk_seconds")
        if duration is not None and threshold and chunk_seconds:
            enriched_row["asr_chunked"] = float(duration) > float(threshold)
            enriched_row["asr_chunk_count"] = (
                int(math.ceil(float(duration) / float(chunk_seconds)))
                if enriched_row["asr_chunked"]
                else 1
            )
        enriched.append(enriched_row)
    enriched.sort(key=lambda row: core.natural_sort_key(Path(str(row["audio_path"])).name))
    tmp_path = path.with_suffix(path.suffix + ".enrich.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    reread = core._read_rows(tmp_path)
    if len(reread) != len(enriched):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Refusing to replace {path}: enrichment rewrite has "
            f"{len(reread)}/{len(enriched)} rows."
        )
    os.replace(tmp_path, path)
    return enriched


def select_files(
    all_files: Sequence[Path],
    *,
    includes: Sequence[str],
    limit: int | None,
) -> list[Path]:
    files = list(all_files)
    if includes:
        files = [
            path
            for path in files
            if any(
                fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)
                for pattern in includes
            )
        ]
        if not files:
            raise ValueError(f"No audio files matched --include patterns: {list(includes)}")
    if limit is not None:
        files = files[:limit]
    return files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--out", help="Output JSONL; defaults to the preset's dataset-root artifact.")
    parser.add_argument("--model", default=core.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--chunk-threshold-seconds",
        type=float,
        help="Split recordings longer than this before ASR (preset default: 120).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        help="Length of contiguous long-audio ASR chunks (preset default: 60).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--d3tec-segments",
        action="store_true",
        help=(
            "For preset=d3tec, transcribe the canonical <=30s equal-duration "
            "segment windows and emit segment-aligned rows."
        ),
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="Maximum D3TEC segment duration used with --d3tec-segments.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only transcribe paths/basenames matching this glob; repeatable (smoke/review use).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print the selected files and exit without loading the model.",
    )
    return parser.parse_args(argv)


def _rewrite_segment_rows(
    out_path: Path,
    windows_by_sample: dict[str, dict[str, Any]],
    preset: Preset,
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in core._read_rows(out_path):
        basename = Path(str(row["audio_path"])).stem
        stable_id = str(row.get("sample_id") or basename)
        window = windows_by_sample.get(stable_id) or windows_by_sample.get(basename)
        if window is None:
            raise ValueError(f"ASR output does not match a canonical D3TEC segment: {stable_id}")
        subject_id = str(window["subject_id"])
        prompt_id = int(window["prompt_id"])
        iphone_suffix = "" if prompt_id == 0 else f"_{prompt_id}"
        paired = D3TEC_ROOT / "iPhoneSE2020" / f"{subject_id}_cel{iphone_suffix}.wav"
        rewritten.append(
            {
                **row,
                **window,
                "audio_path": str(window["audio_path"]),
                "language": preset.language_tag,
                "source_device": "SM-27",
                "paired_iphone_audio_path": str(paired),
                "paired_iphone_audio_found": paired.is_file(),
                "audio_duration_sec": float(window["segment_duration"]),
                "segment_partition": "equal_duration",
                "segment_seconds": float(window["segment_duration"]),
            }
        )
    rewritten.sort(key=lambda row: core.natural_sort_key(str(row["sample_id"])))
    tmp = out_path.with_suffix(out_path.suffix + ".segments.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if len(core._read_rows(tmp)) != len(rewritten):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Refusing to replace incomplete D3TEC segment transcript output.")
    os.replace(tmp, out_path)
    return rewritten


def _run_d3tec_segments(args: argparse.Namespace, preset: Preset, out_path: Path) -> dict[str, Any]:
    if args.preset != "d3tec":
        raise ValueError("--d3tec-segments is valid only with --preset d3tec.")
    windows = discover_d3tec_response_windows(
        D3TEC_ROOT,
        segment_seconds=float(args.segment_seconds),
    )
    windows_by_sample = {str(row["sample_id"]): row for row in windows}
    virtual_paths = [Path(f"{row['sample_id']}.wav") for row in windows]
    selected_virtual = select_files(
        virtual_paths,
        includes=args.include,
        limit=args.limit,
    )
    selected_ids = [path.stem for path in selected_virtual]
    limited = bool(args.include) or args.limit is not None
    if args.list_files:
        for stable_id in selected_ids:
            row = windows_by_sample[stable_id]
            print(
                f"{stable_id}\t{row['audio_path']}\t"
                f"{float(row['start_time']):.6f}\t{float(row['end_time']):.6f}"
            )
        return {"n_files": len(selected_ids)}

    _configure_core_for_preset(preset)
    batch_size = args.batch_size or preset.default_batch_size
    max_new_tokens = args.max_new_tokens or preset.default_max_new_tokens
    with tempfile.TemporaryDirectory(prefix="d3tec_asr_segments_") as tmp_name:
        tmp_dir = Path(tmp_name)
        materialized: list[Path] = []
        for stable_id in selected_ids:
            window = windows_by_sample[stable_id]
            source = Path(str(window["audio_path"]))
            info = core.sf.info(str(source))
            start = int(round(float(window["start_time"]) * info.samplerate))
            stop = int(round(float(window["end_time"]) * info.samplerate))
            audio, sample_rate = core.sf.read(
                str(source),
                start=start,
                stop=stop,
                dtype="float32",
                always_2d=False,
            )
            target = tmp_dir / f"{stable_id}.wav"
            core.sf.write(str(target), audio, sample_rate, subtype="PCM_16")
            materialized.append(target)

        # Resume matching in the core utility is basename-based. Existing public
        # rows retain the original WAV path, so temporarily present their stable
        # sample IDs as materialized basenames before invoking the crash-safe core.
        if args.resume and out_path.exists():
            resume_rows = core._read_rows(out_path)
            resume_tmp = out_path.with_suffix(out_path.suffix + ".resume.tmp")
            with resume_tmp.open("w", encoding="utf-8") as handle:
                for row in resume_rows:
                    stable_id = str(row["sample_id"])
                    handle.write(
                        json.dumps(
                            {**row, "audio_path": str(tmp_dir / f"{stable_id}.wav")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            os.replace(resume_tmp, out_path)

        backend = core.Qwen3ASRBackend(
            args.model,
            device=args.device,
            dtype=args.dtype,
            max_inference_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            attn_implementation=args.attn,
        )
        result = core.transcribe_all(
            backend,
            materialized,
            out_path=out_path,
            model_id=args.model,
            language_arg=preset.language_arg,
            batch_size=batch_size,
            resume=bool(args.resume),
        )
        rows = _rewrite_segment_rows(out_path, windows_by_sample, preset)

    row_ids = [str(row["sample_id"]) for row in rows]
    duplicate_ids = sorted(
        stable_id for stable_id, count in Counter(row_ids).items() if count > 1
    )
    expected_ids = set(selected_ids) if limited else set(windows_by_sample)
    observed_ids = set(row_ids)
    empty_ids = [str(row["sample_id"]) for row in rows if not str(row.get("transcript", "")).strip()]
    language_mismatches = [
        str(row["sample_id"])
        for row in rows
        if str(row.get("language", "")).lower() not in {"es", "spanish"}
        or (
            row.get("asr_detected_language")
            and str(row["asr_detected_language"]).lower() not in {"es", "spanish"}
        )
    ]
    report = {
        "preset": "d3tec_segments",
        "dataset": "d3tec",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_file": str(out_path),
        "asr_model": args.model,
        "asr_language_arg": preset.language_arg,
        "language_tag": preset.language_tag,
        "segment_partition": "equal_duration",
        "segment_seconds": float(args.segment_seconds),
        "n_canonical_windows": len(windows),
        "n_selected_files": len(selected_ids),
        "n_rows": len(rows),
        "n_unique_sample_ids": len(observed_ids),
        "n_empty_transcripts": len(empty_ids),
        "n_language_mismatches": len(language_mismatches),
        "n_failures": len(result["failures"]),
        "coverage_complete": (
            observed_ids == expected_ids
            and not duplicate_ids
            and not empty_ids
            and not language_mismatches
            and not result["failures"]
        ),
        "limited_run": limited,
        "missing_sample_ids": sorted(expected_ids - observed_ids),
        "extra_sample_ids": sorted(observed_ids - expected_ids),
        "duplicate_sample_ids": duplicate_ids,
        "empty_sample_ids": empty_ids,
        "language_mismatch_sample_ids": language_mismatches,
        "failed_files": result["failures"],
    }
    report_path = core.write_report(report, out_path)
    LOGGER.info(
        "Completed D3TEC segment ASR | rows=%d coverage=%s report=%s",
        len(rows),
        report["coverage_complete"],
        report_path,
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    if args.out:
        out_path = Path(args.out)
    elif args.d3tec_segments:
        out_path = D3TEC_ROOT / "transcripts_qwen3_asr_spanish_segments.jsonl"
    else:
        out_path = preset.output_path
    ensure_dir(out_path.parent)

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if out_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --resume or explicitly use --overwrite."
        )

    if args.d3tec_segments:
        return _run_d3tec_segments(args, preset, out_path)

    all_files = discover_files(preset)
    selected_files = select_files(all_files, includes=args.include, limit=args.limit)
    limited = bool(args.include) or args.limit is not None
    LOGGER.info(
        "Preset=%s | discovered=%d selected=%d | language=%s/%s | output=%s",
        preset.name,
        len(all_files),
        len(selected_files),
        preset.language_arg,
        preset.language_tag,
        out_path,
    )
    if args.list_files:
        for path in selected_files:
            print(path)
        return {"n_files": len(selected_files)}

    _configure_core_for_preset(preset)
    batch_size = args.batch_size or preset.default_batch_size
    max_new_tokens = args.max_new_tokens or preset.default_max_new_tokens
    chunk_threshold = args.chunk_threshold_seconds or preset.default_chunk_threshold_seconds
    chunk_seconds = args.chunk_seconds or preset.default_chunk_seconds
    backend = core.Qwen3ASRBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        attn_implementation=args.attn,
    )
    transcriber = LongAudioChunkingTranscriber(
        backend,
        threshold_seconds=chunk_threshold,
        chunk_seconds=chunk_seconds,
        segment_batch_size=batch_size,
    )
    result = core.transcribe_all(
        transcriber,
        selected_files,
        out_path=out_path,
        model_id=args.model,
        language_arg=preset.language_arg,
        batch_size=batch_size,
        resume=bool(args.resume),
    )
    raw_rows = core._read_rows(out_path)
    for row in raw_rows:
        row["asr_chunk_threshold_seconds"] = chunk_threshold
        row["asr_chunk_seconds"] = chunk_seconds
    raw_tmp = out_path.with_suffix(out_path.suffix + ".chunkmeta.tmp")
    with raw_tmp.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if len(core._read_rows(raw_tmp)) != len(raw_rows):
        raw_tmp.unlink(missing_ok=True)
        raise RuntimeError("Refusing to replace output: chunk-metadata rewrite is incomplete.")
    os.replace(raw_tmp, out_path)
    final_rows = enrich_output(out_path, preset)
    report = core.build_report(
        rows=final_rows,
        all_files=all_files,
        failures=result["failures"],
        model_id=args.model,
        language_arg=preset.language_arg,
        out_path=out_path,
        limited=limited,
    )
    report.update(
        {
            "preset": preset.name,
            "dataset": preset.dataset,
            "language_tag": preset.language_tag,
            "audio_root": str(preset.audio_dir),
            "n_selected_files": len(selected_files),
            "asr_chunk_threshold_seconds": chunk_threshold,
            "asr_chunk_seconds": chunk_seconds,
            "n_chunked_files": sum(bool(row.get("asr_chunked")) for row in final_rows),
        }
    )
    report_path = core.write_report(report, out_path)
    LOGGER.info(
        "Completed %s | rows=%d empty=%d flagged=%d failures=%d coverage=%s | report=%s",
        preset.name,
        report["n_rows"],
        report["n_empty_transcripts"],
        report["n_manual_review_recommended"],
        report["n_failures"],
        report["coverage_complete"],
        report_path,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
