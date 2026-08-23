#!/usr/bin/env python3
"""Re-transcribe the Turkish dataset with Qwen3-ASR, emitting a loader-compatible JSONL.

See docs/qwen3_asr_turkish_retranscription_plan.md. This script is standalone: no repo
code imports it, and it does NOT touch the secap conda env. Run it inside the separate
``qwen3asr`` env (see scripts/transcribe_turkish_qwen3asr.sh) on the local RTX 4090.

Output schema (one JSON object per wav) — the loader in src/data/turkish.py only reads
``audio_path`` (-> basename), ``transcript``, ``language``, ``repair_status``; the other
fields are harmless provenance/QC carried for parity with the repaired-file style:

    {
      "audio_path":   "/.../Turkish/all-files/aa1-1-1-ank.wav",  # absolute, loader takes .name
      "transcript":   "İyiyim teşekkür ederim ...",
      "language":     "tr",                # fixed, matches the Whisper convention
      "repair_status":"QWEN3ASR_RAW",      # distinguishable from Whisper rows; loader stores the string
      "asr_model":    "Qwen/Qwen3-ASR-1.7B",
      "asr_language_arg": "Turkish",       # the forced --language arg ("" when auto-detect)
      "asr_detected_language": "Turkish",  # what the model reported
      "audio_duration_sec": 20.0,
      "n_chars": 152,
      "manual_review_recommended": false,
      "manual_review_reason_codes": []
    }

Safety / reproducibility:
  * Writes to a NEW filename; legacy whisper_transcripts*.jsonl are never touched.
  * Optionally filters audio through a metadata CSV allowlist.
  * Enforces unique basenames (mirrors the loader invariant) and full selected coverage.
  * Greedy decode (qwen-asr default) + sorted inputs + recorded asr_model => reproducible.
  * Crash-safe + resumable: each batch is written + fsync'd straight to ``<out>`` (the single
    source of truth — no journal/consolidate/delete step that could silently lose data), then
    a guarded in-place sort tidies it. ``--resume`` keeps successful rows and retries empty or
    failed rows. Existing output is never truncated unless ``--overwrite`` is explicit.

Usage (real run):
    python scripts/transcribe_turkish_qwen3asr.py            # all defaults
    python scripts/transcribe_turkish_qwen3asr.py --resume   # continue after an interruption
    python scripts/transcribe_turkish_qwen3asr.py --limit 4  # smoke test on the first 4 clips
    python scripts/transcribe_turkish_qwen3asr.py --metadata-csv /path/to/metadata.csv

Plumbing self-test (no GPU / no model download, uses a fake transcriber):
    python scripts/transcribe_turkish_qwen3asr.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

try:  # soundfile is only needed for the duration-based QC; degrade gracefully without it.
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised only when soundfile is absent
    sf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/Turkish")
DEFAULT_AUDIO_DIR = DEFAULT_ROOT / "all-files"
DEFAULT_OUT = DEFAULT_ROOT / "whisper_transcripts_qwen3_asr.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_LANGUAGE = "Turkish"
REPAIR_STATUS = "QWEN3ASR_RAW"
OUTPUT_LANGUAGE_TAG = "tr"  # what the loader expects; distinct from the ASR --language arg

# QC thresholds (tunable; only affect the manual_review_* flags, never the transcript).
ULTRA_SHORT_SEC = 0.5  # clips this short (min in the corpus is 0.31 s) are inherently risky
MIN_CHAR_RATE_DUR_SEC = 5.0  # only judge char-rate on clips with enough audio to speak
MIN_CHAR_RATE = 1.0  # chars/sec below this on a >=5 s clip looks suspiciously sparse

NATURAL_SPLIT_RE = re.compile(r"(\d+)")


def natural_sort_key(value: str) -> list[Any]:
    """Human-friendly ordering so aa1-2-... sorts before aa1-10-... (stable, deterministic)."""
    return [int(part) if part.isdigit() else part for part in NATURAL_SPLIT_RE.split(value)]


# --------------------------------------------------------------------------------------
# Transcriber backends
# --------------------------------------------------------------------------------------
class Transcriber(Protocol):
    """The slice of the qwen-asr API this script depends on (see model card)."""

    def transcribe(self, audio: Sequence[str], language: Any) -> Sequence[Any]:
        ...


class _Result(Protocol):
    text: str
    language: str


class Qwen3ASRBackend:
    """Thin wrapper over ``qwen_asr.Qwen3ASRModel`` (transformers backend).

    The qwen-asr API (verified against the Qwen/Qwen3-ASR-1.7B model card):
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(model_id, dtype=..., device_map=...,
                                              max_inference_batch_size=..., max_new_tokens=...)
        results = model.transcribe(audio=[paths...], language=[...]|None)
        results[i].text / results[i].language
    Greedy decoding is the default, which is what we want for reproducibility.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        dtype: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
        attn_implementation: str | None,
    ) -> None:
        import torch  # imported lazily so --self-test needs no torch
        from qwen_asr import Qwen3ASRModel

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported --dtype {dtype!r}; choose one of {sorted(dtype_map)}.")

        kwargs: dict[str, Any] = {
            "dtype": dtype_map[dtype],
            "device_map": device,
            "max_inference_batch_size": int(max_inference_batch_size),
            "max_new_tokens": int(max_new_tokens),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        LOGGER.info("Loading %s | %s", model_id, kwargs)
        self._model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)

    def transcribe(self, audio: Sequence[str], language: Any) -> Sequence[Any]:
        return self._model.transcribe(audio=list(audio), language=language)


class _FakeResult:
    def __init__(self, text: str, language: str) -> None:
        self.text = text
        self.language = language


class _FakeTranscriber:
    """Deterministic stand-in for --self-test: derives text from the filename, no model needed."""

    def transcribe(self, audio: Sequence[str], language: Any) -> Sequence[_FakeResult]:
        detected = language[0] if isinstance(language, (list, tuple)) and language else (language or "Turkish")
        return [_FakeResult(text=f"merhaba {Path(path).stem}", language=str(detected)) for path in audio]


# --------------------------------------------------------------------------------------
# Audio discovery & duration
# --------------------------------------------------------------------------------------
def discover_wavs(audio_dir: Path) -> list[Path]:
    """Sorted, basename-deduplicated list of *.wav under ``audio_dir`` (fail fast on collisions)."""
    files = sorted(audio_dir.glob("*.wav"), key=lambda p: natural_sort_key(p.name))
    seen: dict[str, Path] = {}
    for path in files:
        if path.name in seen:
            raise ValueError(
                f"Duplicate audio basename {path.name!r}: {seen[path.name]} vs {path} "
                "(the loader keys on basename and rejects duplicates)."
            )
        seen[path.name] = path
    return files


def _normalized_basename(value: str) -> str:
    return unicodedata.normalize("NFC", Path(str(value)).name).casefold()


def load_metadata_allowlist(path: Path) -> list[str]:
    """Read unique audio basenames from a metadata CSV ``file_name`` column."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    selected: list[str] = []
    seen: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError(f"Metadata CSV is missing required field 'file_name': {path}")
        for row_number, row in enumerate(reader, start=2):
            basename = Path(str(row.get("file_name", "")).strip()).name
            if not basename:
                raise ValueError(f"Metadata CSV row {row_number} has an empty file_name: {path}")
            key = _normalized_basename(basename)
            if key in seen:
                raise ValueError(
                    f"Duplicate normalized metadata basename at row {row_number}: "
                    f"{seen[key]!r} and {basename!r}"
                )
            seen[key] = basename
            selected.append(basename)
    if not selected:
        raise ValueError(f"Metadata CSV has no selected audio rows: {path}")
    return selected


def select_allowlisted_wavs(all_files: Sequence[Path], allowlist: Sequence[str]) -> list[Path]:
    """Resolve an allowlist to disk paths while tolerating Unicode composition differences."""
    disk_by_key: dict[str, Path] = {}
    for path in all_files:
        key = _normalized_basename(path.name)
        if key in disk_by_key:
            raise ValueError(
                f"Audio filenames collide after Unicode normalization: "
                f"{disk_by_key[key].name!r} and {path.name!r}"
            )
        disk_by_key[key] = path

    missing = [name for name in allowlist if _normalized_basename(name) not in disk_by_key]
    if missing:
        raise FileNotFoundError(
            f"Metadata selects {len(missing)} missing WAV file(s); first entries: {missing[:5]}"
        )
    selected_keys = {_normalized_basename(name) for name in allowlist}
    return [path for path in all_files if _normalized_basename(path.name) in selected_keys]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_duration_seconds(audio_path: Path) -> float | None:
    if sf is None:
        return None
    try:
        info = sf.info(str(audio_path))
    except Exception as exc:  # pragma: no cover - corrupt/unreadable audio is rare
        LOGGER.warning("Could not read duration for %s: %s", audio_path.name, exc)
        return None
    if info.samplerate <= 0:
        return None
    return float(info.frames / info.samplerate)


# --------------------------------------------------------------------------------------
# Row construction & QC
# --------------------------------------------------------------------------------------
def _normalize_lang(value: str) -> str:
    return str(value or "").strip().lower()


def manual_review_reason_codes(
    *, text: str, duration_sec: float | None, detected_language: str, failed: bool
) -> list[str]:
    """Heuristic QC flags. They never alter the transcript — only surface clips for review."""
    codes: list[str] = []
    if failed:
        codes.append("transcription_failed")
    if not text:
        codes.append("empty_transcript")
    if duration_sec is not None and duration_sec < ULTRA_SHORT_SEC:
        codes.append("ultra_short_clip")
    if (
        text
        and duration_sec is not None
        and duration_sec >= MIN_CHAR_RATE_DUR_SEC
        and (len(text) / duration_sec) < MIN_CHAR_RATE
    ):
        codes.append("low_char_rate")
    if detected_language and _normalize_lang(detected_language) not in {"turkish", "tr"}:
        codes.append("language_mismatch")
    return codes


def build_row(
    *,
    audio_path: Path,
    text: str,
    detected_language: str,
    duration_sec: float | None,
    model_id: str,
    language_arg: str,
    failed: bool = False,
) -> dict[str, Any]:
    text = (text or "").strip()
    codes = manual_review_reason_codes(
        text=text, duration_sec=duration_sec, detected_language=detected_language, failed=failed
    )
    return {
        "audio_path": str(audio_path),
        "transcript": text,
        "language": OUTPUT_LANGUAGE_TAG,
        "repair_status": REPAIR_STATUS,
        "asr_model": model_id,
        "asr_language_arg": language_arg,
        "asr_detected_language": str(detected_language or "").strip(),
        "audio_duration_sec": duration_sec,
        "n_chars": len(text),
        "manual_review_recommended": bool(codes),
        "manual_review_reason_codes": codes,
    }


# --------------------------------------------------------------------------------------
# Resume / journal helpers
# --------------------------------------------------------------------------------------
def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _row_is_successful(row: dict[str, Any]) -> bool:
    codes = set(row.get("manual_review_reason_codes") or [])
    return bool(str(row.get("transcript", "")).strip()) and "transcription_failed" not in codes


def _write_rows_atomically(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        _append_rows(handle, rows)
    if len(_read_rows(tmp_path)) != len(rows):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Refusing to replace {path}: temporary rewrite was incomplete.")
    os.replace(tmp_path, path)


def prepare_resume_output(out_path: Path, files: Sequence[Path]) -> set[str]:
    """Keep one successful row per selected WAV and discard retryable failed rows."""
    expected = {_normalized_basename(path.name): path.name for path in files}
    successful: dict[str, dict[str, Any]] = {}
    for row in _read_rows(out_path):
        basename = Path(str(row.get("audio_path", ""))).name
        if not basename:
            continue
        key = _normalized_basename(basename)
        if key not in expected:
            raise ValueError(
                f"Resume output contains a row outside the selected audio set: {basename!r}"
            )
        if _row_is_successful(row):
            successful[key] = row

    ordered = sorted(
        successful.values(),
        key=lambda row: natural_sort_key(Path(str(row["audio_path"])).name),
    )
    _write_rows_atomically(out_path, ordered)
    return set(successful)


def _append_rows(handle, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def finalize_sorted(out_path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    """Best-effort in-place sort of the (already-written) output by basename.

    Crucially, this NEVER risks the data: rows are written incrementally + fsync'd to
    ``out_path`` as the source of truth. If a re-read can't see every row (a transient
    network/fuse read-after-write hiccup — the exact failure that silently zeroed an
    earlier full run), we leave the complete-but-unsorted file in place instead of
    swapping in a short one. Sorting is cosmetic (diff-friendliness); data integrity wins.
    """
    rows = _read_rows(out_path)
    if len(rows) < expected_rows:
        LOGGER.warning(
            "finalize_sorted: re-read saw %d/%d rows; leaving %s unsorted (data is intact on disk).",
            len(rows),
            expected_rows,
            out_path,
        )
        return rows

    by_basename = {Path(str(r.get("audio_path", ""))).name: r for r in rows}
    by_basename.pop("", None)
    ordered = sorted(by_basename.values(), key=lambda r: natural_sort_key(Path(str(r["audio_path"])).name))

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Verify the temp file is complete BEFORE swapping it in — never replace good data
    # with a short rewrite.
    if len(_read_rows(tmp_path)) < len(ordered):
        LOGGER.warning("finalize_sorted: temp rewrite looked incomplete; keeping original %s.", out_path)
        tmp_path.unlink(missing_ok=True)
        return rows
    os.replace(tmp_path, out_path)
    return ordered


# --------------------------------------------------------------------------------------
# Core transcription loop (backend-agnostic so it is unit-testable without a GPU)
# --------------------------------------------------------------------------------------
def transcribe_all(
    transcriber: Transcriber,
    files: Sequence[Path],
    *,
    out_path: Path,
    model_id: str,
    language_arg: str,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    done: set[str] = prepare_resume_output(out_path, files) if resume and out_path.exists() else set()
    if resume:
        LOGGER.info(
            "Resume: %d successful clip(s) already in %s; retrying failed or missing rows.",
            len(done),
            out_path.name,
        )

    pending = [f for f in files if _normalized_basename(f.name) not in done]
    LOGGER.info("To transcribe: %d / %d clip(s) (batch_size=%d).", len(pending), len(files), batch_size)

    failures: list[dict[str, str]] = []
    n_written = 0
    started = time.monotonic()

    # Write rows STRAIGHT to the output file (append on resume, truncate on a fresh run),
    # flushing + fsync'ing after every batch. The data file is the single source of truth:
    # there is no journal/consolidate/delete step that could silently lose it.
    with out_path.open("a" if resume else "w", encoding="utf-8") as out:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            rows = _transcribe_batch(
                transcriber, batch, model_id=model_id, language_arg=language_arg, failures=failures
            )
            _append_rows(out, rows)
            n_written += len(rows)
            done_count = len(done) + n_written
            elapsed = time.monotonic() - started
            rate = n_written / elapsed if elapsed > 0 else 0.0
            LOGGER.info(
                "Progress %d/%d (%.1f%%) | %.2f clip/s | last=%s",
                done_count,
                len(files),
                100.0 * done_count / max(len(files), 1),
                rate,
                batch[-1].name,
            )

    expected = len(done) + n_written
    final_rows = finalize_sorted(out_path, expected_rows=expected)
    # Hard post-condition: refuse to report success on a silently-empty/short file.
    if n_written > 0 and not final_rows:
        raise RuntimeError(
            f"Transcribed {n_written} clip(s) but {out_path} reads back empty; "
            "aborting so the result is never silently lost."
        )
    if len(final_rows) < expected:
        LOGGER.warning("Final file has %d row(s), expected >= %d (see warnings above).", len(final_rows), expected)
    LOGGER.info("Wrote %d row(s) -> %s", len(final_rows), out_path)
    return {"rows": final_rows, "failures": failures}


def _transcribe_batch(
    transcriber: Transcriber,
    batch: Sequence[Path],
    *,
    model_id: str,
    language_arg: str,
    failures: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Transcribe one batch; on a batch-level error, isolate failures file-by-file."""
    paths = [str(p) for p in batch]
    language = [language_arg] * len(batch) if language_arg else None
    try:
        results = transcriber.transcribe(audio=paths, language=language)
    except Exception as exc:  # noqa: BLE001 - retry per-file so one bad clip can't sink the batch
        LOGGER.warning("Batch of %d failed (%s); retrying file-by-file.", len(batch), exc)
        return [
            _transcribe_single(transcriber, path, model_id=model_id, language_arg=language_arg, failures=failures)
            for path in batch
        ]

    if len(results) != len(batch):
        raise RuntimeError(
            f"qwen-asr returned {len(results)} results for {len(batch)} inputs; "
            "cannot align transcripts to files safely."
        )
    rows: list[dict[str, Any]] = []
    for path, result in zip(batch, results):
        rows.append(
            build_row(
                audio_path=path,
                text=getattr(result, "text", ""),
                detected_language=getattr(result, "language", ""),
                duration_sec=read_duration_seconds(path),
                model_id=model_id,
                language_arg=language_arg,
            )
        )
    return rows


def _transcribe_single(
    transcriber: Transcriber,
    path: Path,
    *,
    model_id: str,
    language_arg: str,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    language = [language_arg] if language_arg else None
    try:
        results = transcriber.transcribe(audio=[str(path)], language=language)
        result = results[0]
        return build_row(
            audio_path=path,
            text=getattr(result, "text", ""),
            detected_language=getattr(result, "language", ""),
            duration_sec=read_duration_seconds(path),
            model_id=model_id,
            language_arg=language_arg,
        )
    except Exception as exc:  # noqa: BLE001 - record and emit an empty, review-flagged row
        LOGGER.error("Failed to transcribe %s: %s", path.name, exc)
        failures.append({"file": path.name, "error": str(exc)})
        return build_row(
            audio_path=path,
            text="",
            detected_language="",
            duration_sec=read_duration_seconds(path),
            model_id=model_id,
            language_arg=language_arg,
            failed=True,
        )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def build_report(
    *,
    rows: Sequence[dict[str, Any]],
    all_files: Sequence[Path],
    failures: Sequence[dict[str, str]],
    model_id: str,
    language_arg: str,
    out_path: Path,
    limited: bool,
    inventory_files: Sequence[Path] | None = None,
    selected_file_count: int | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    row_basenames = [Path(str(r["audio_path"])).name for r in rows]
    row_basename_set = set(row_basenames)
    expected_basenames = {f.name for f in all_files}
    inventory_basenames = {f.name for f in (inventory_files or all_files)}

    reason_counts: dict[str, int] = {}
    n_flagged = 0
    n_empty = 0
    for row in rows:
        codes = row.get("manual_review_reason_codes") or []
        if codes:
            n_flagged += 1
        if not str(row.get("transcript", "")).strip():
            n_empty += 1
        for code in codes:
            reason_counts[code] = reason_counts.get(code, 0) + 1

    missing = sorted(expected_basenames - row_basename_set)  # selected wavs with no row
    extra = sorted(row_basename_set - expected_basenames)  # rows outside this run's selection
    duplicate = len(row_basenames) != len(row_basename_set)
    coverage_complete = (not limited) and not missing and not extra and not duplicate
    n_failures = sum(
        "transcription_failed" in set(row.get("manual_review_reason_codes") or [])
        for row in rows
    )
    selected_count = selected_file_count if selected_file_count is not None else len(expected_basenames)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_file": str(out_path),
        "asr_model": model_id,
        "asr_language_arg": language_arg,
        "metadata_csv": str(metadata_path) if metadata_path is not None else None,
        "metadata_sha256": sha256_file(metadata_path) if metadata_path is not None else None,
        "output_sha256": sha256_file(out_path),
        "n_audio_files_on_disk": len(inventory_basenames),
        "n_selected_audio_files": selected_count,
        "n_unselected_audio_files": len(inventory_basenames) - selected_count,
        "n_rows": len(rows),
        "n_unique_basenames": len(row_basename_set),
        "n_empty_transcripts": n_empty,
        "n_manual_review_recommended": n_flagged,
        "n_failures": n_failures,
        "manual_review_reason_counts": reason_counts,
        "coverage_complete": coverage_complete,
        "limited_run": limited,
        "missing_basenames": missing[:50],
        "n_missing": len(missing),
        "extra_basenames": extra[:50],
        "n_extra": len(extra),
        "duplicate_basenames": duplicate,
        "failed_files": list(failures)[:200],
    }


def write_report(report: dict[str, Any], out_path: Path) -> Path:
    report_path = out_path.with_name(out_path.stem + ".report.json")
    ensure_dir(report_path.parent)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, report_path)
    return report_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="Directory of *.wav clips.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path (new filename; never clobbers Whisper).")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional CSV allowlist; only WAV basenames from its file_name column are transcribed.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id, e.g. Qwen/Qwen3-ASR-1.7B or -0.6B.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Force decode language ('' => auto-detect).")
    parser.add_argument("--batch-size", type=int, default=16, help="Clips per transcribe() call.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Decode cap per clip.")
    parser.add_argument("--device", default="cuda:0", help="device_map for from_pretrained.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default=None, help="attn_implementation, e.g. flash_attention_2.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful rows and retry missing, empty, or failed rows in --out.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output; mutually exclusive with --resume.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: only the first N clips (skips coverage gate).")
    parser.add_argument("--self-test", action="store_true", help="Run plumbing on a fake transcriber + temp dir; no GPU.")
    return parser.parse_args(argv)


def _run(transcriber: Transcriber, args: argparse.Namespace) -> dict[str, Any]:
    audio_dir = Path(args.audio_dir)
    out_path = Path(args.out)
    metadata_path = Path(args.metadata_csv) if getattr(args, "metadata_csv", None) else None
    resume = bool(getattr(args, "resume", False))
    overwrite = bool(getattr(args, "overwrite", False))
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if out_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --resume or explicit --overwrite."
        )
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    ensure_dir(out_path.parent)

    inventory_files = discover_wavs(audio_dir)
    if not inventory_files:
        raise FileNotFoundError(f"No *.wav found under {audio_dir}")
    selected_files = inventory_files
    if metadata_path is not None:
        selected_files = select_allowlisted_wavs(
            inventory_files,
            load_metadata_allowlist(metadata_path),
        )
    limited = args.limit is not None
    files = selected_files[: args.limit] if limited else selected_files

    result = transcribe_all(
        transcriber,
        files,
        out_path=out_path,
        model_id=args.model,
        language_arg=args.language,
        batch_size=max(1, int(args.batch_size)),
        resume=resume,
    )

    # Coverage/QC are judged against the FULL final file (resume merges prior rows).
    final_rows = _read_rows(out_path)
    report = build_report(
        rows=final_rows,
        all_files=files,
        failures=result["failures"],
        model_id=args.model,
        language_arg=args.language,
        out_path=out_path,
        limited=limited,
        inventory_files=inventory_files,
        selected_file_count=len(selected_files),
        metadata_path=metadata_path,
    )
    report_path = write_report(report, out_path)
    LOGGER.info("Report -> %s", report_path)
    LOGGER.info(
        "Summary | rows=%d unique=%d empty=%d flagged=%d failures=%d coverage_complete=%s",
        report["n_rows"],
        report["n_unique_basenames"],
        report["n_empty_transcripts"],
        report["n_manual_review_recommended"],
        report["n_failures"],
        report["coverage_complete"],
    )
    if not limited and not report["coverage_complete"]:
        LOGGER.warning(
            "Coverage incomplete: missing=%d extra=%d duplicate=%s. See %s.",
            report["n_missing"],
            report["n_extra"],
            report["duplicate_basenames"],
            report_path,
        )
    return report


def report_passes_full_run(report: dict[str, Any]) -> bool:
    return bool(
        report.get("coverage_complete")
        and int(report.get("n_empty_transcripts", 0)) == 0
        and int(report.get("n_failures", 0)) == 0
    )


def _self_test() -> int:
    """End-to-end plumbing check with a fake transcriber: discovery, batching, resume, report."""
    import tempfile
    import wave

    print("[self-test] building a temp corpus ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_dir = tmp_path / "all-files"
        audio_dir.mkdir()
        names = ["aa1-1-1-ank.wav", "aa1-2-2-ank.wav", "aa1-10-3-ank.wav", "bb2-1-1-ist.wav", "cc3-1-1-izm.wav"]
        for name in names:
            with wave.open(str(audio_dir / name), "wb") as wav:  # ~1 s of silence so soundfile reads a duration
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 16000)
        out_path = tmp_path / "whisper_transcripts_qwen3_asr.jsonl"
        metadata_path = tmp_path / "metadata.csv"
        selected_names = names[:4]
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file_name"])
            writer.writeheader()
            for name in selected_names:
                writer.writerow({"file_name": f"nested/{name}"})

        common = dict(
            audio_dir=str(audio_dir),
            out=str(out_path),
            metadata_csv=str(metadata_path),
            model="fake/qwen3asr",
            language="Turkish",
            batch_size=2,
            max_new_tokens=256,
            device="cpu",
            dtype="bfloat16",
            attn=None,
            limit=None,
            self_test=True,
            overwrite=False,
        )

        # Pass 1: only the first 2 selected clips, then resume to cover the allowlist.
        args1 = argparse.Namespace(resume=False, **{**common, "limit": 2})
        _run(_FakeTranscriber(), args1)
        rows_after_1 = _read_rows(out_path)
        assert len(rows_after_1) == 2, f"expected 2 rows after limited pass, got {len(rows_after_1)}"

        args2 = argparse.Namespace(resume=True, **common)
        report = _run(_FakeTranscriber(), args2)
        assert not out_path.with_suffix(out_path.suffix + ".tmp").exists(), "temp file not cleaned up"
        assert report["n_rows"] == len(selected_names), report
        assert report["n_unique_basenames"] == len(selected_names), "basenames not unique"
        assert report["n_audio_files_on_disk"] == len(names), report
        assert report["n_selected_audio_files"] == len(selected_names), report
        assert report["n_unselected_audio_files"] == 1, report
        assert report["coverage_complete"], f"coverage should be complete: {report}"
        assert report["n_failures"] == 0, "fake transcriber should not fail"
        assert report_passes_full_run(report), report

        # Ordering is natural-sorted (aa1-2 before aa1-10) and resume did not duplicate rows.
        ordered = [Path(r["audio_path"]).name for r in _read_rows(out_path)]
        assert ordered == sorted(selected_names, key=natural_sort_key), f"unexpected order: {ordered}"

        # Schema spot-check on one row.
        row = _read_rows(out_path)[0]
        for key in ("audio_path", "transcript", "language", "repair_status"):
            assert key in row, f"missing loader key {key}"
        assert row["language"] == OUTPUT_LANGUAGE_TAG and row["repair_status"] == REPAIR_STATUS

        try:
            _run(_FakeTranscriber(), argparse.Namespace(resume=False, **common))
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output should require --resume or --overwrite")
    print("[self-test] PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()

    transcriber = Qwen3ASRBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn,
    )
    report = _run(transcriber, args)
    if not args.limit and not report_passes_full_run(report):
        LOGGER.error(
            "Full-run acceptance failed: coverage=%s empty=%s failures=%s",
            report["coverage_complete"],
            report["n_empty_transcripts"],
            report["n_failures"],
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
