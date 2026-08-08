#!/usr/bin/env python3
"""Transcribe DAIC-WOZ (full interviews), CMDC, ANDROIDS and D3TEC with the Gemini API.

Standalone script: no repo code imports it, no GPU/torch needed. It calls the
Gemini API (Google AI Studio key) over REST using only stdlib + ``requests``.

Presets (language of the transcription prompt is preset-specific):

* ``daic``  -- English. Participant-only turns ``<subject>_<segment>.wav`` under
  ``DAIC-WOZ/minimal_zips/preprocessed_audios`` (interviewer removed; NOT the
  full ``unprocessed`` interviews and NOT the 30 s random segments).
* ``cmdc``  -- Chinese. Q1..Q12 ``Q*.wav`` files per subject (HC##/MDD##).
* ``androids_interview`` -- Italian. Participant-only interview turns under
  ``Interview-Task/audio_clip`` (NOT the reading task).
* ``d3tec`` -- Spanish. SM-27 response recordings ``<subject>[_<prompt>].wav``.

Free-tier friendly design (10 RPM / 250 RPD / 250k TPM on gemini-3.5-flash):

* Clips of one subject are sent as ONE request with N inline audio parts,
  so the request/day quota is spent per subject, not per clip (DAIC 189
  subjects + CMDC 78 + ANDROIDS 116 + D3TEC 124 -> ~510 requests, roughly
  2 days at 250 requests/day instead of ~25 per-file).
* A rate limiter enforces ``--min-interval-sec`` (default 7 s -> <=10 RPM) and
  ``--daily-limit`` (default 250): when the cap is hit the script sleeps until
  the next 08:00 UTC (midnight Pacific, when RPD resets) and continues.
* A failed subject batch falls back to per-file requests so one bad clip does
  not waste a subject.

Output schema (one JSON object per wav; the ANDROIDS loader reads
``audio_path`` (-> basename), ``transcript``, ``language``, ``sample_id``; the
other fields are provenance/QC):

    {
      "audio_path":   "/.../unprocessed/300_AUDIO.wav",
      "transcript":   "...",
      "language":     "en",
      "asr_model":    "gemini-3.5-flash",
      "asr_backend":  "gemini_api",
      "asr_language_arg": "English",
      "asr_detected_language": "",
      "audio_duration_sec": 1350.2,
      "n_chars": 8920,
      "manual_review_recommended": false,
      "manual_review_reason_codes": [],
      "dataset": "daic", "subject_id": "300", "sample_id": "300_AUDIO",
      ...
    }

Safety / reproducibility:

* Writes to NEW filenames; existing qwen3_asr / whisper JSONLs are never touched.
* Enforces unique stable ids (dataset-relative paths; CMDC repeats basenames
  across subjects) and full coverage of the selected files.
* Crash-safe + resumable: each subject batch is appended and fsync'd straight
  to ``<out>``; ``--resume`` skips clips already present in ``<out>``.
* The model must return a JSON object whose keys match the requested filenames
  exactly; anything else is re-requested per file.

Examples:

    export GEMINI_API_KEY=...   # Google AI Studio API key (free tier works)
    python scripts/transcribe_gemini_api.py --preset daic --resume
    python scripts/transcribe_gemini_api.py --preset d3tec --limit 2
    python scripts/transcribe_gemini_api.py --self-test   # no network
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

LOGGER = logging.getLogger("transcribe_gemini_api")

DAIC_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/minimal_zips/preprocessed_audios")
CMDC_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/CMDC")
ANDROIDS_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus")
D3TEC_ROOT = Path("/media/emre/Backup/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET")

DEFAULT_MODEL = "gemini-3.5-flash"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ULTRA_SHORT_SEC = 0.5
MIN_CHAR_RATE_DUR_SEC = 5.0
MIN_CHAR_RATE = 2.0  # chars per second below which a transcript is suspicious
MAX_RETRIES = 3


def natural_sort_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def read_duration_seconds(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class Preset:
    name: str
    dataset: str
    audio_dir: Path
    output_path: Path
    language_arg: str
    language_tag: str
    expected_detected_languages: frozenset[str]
    default_batch_size: int
    batch_key: Callable[[Path], str]
    enrich: Callable[[dict[str, Any], Path], dict[str, Any]]


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


def _subject_batch_key(path: Path) -> str:
    """Group audio files by their parent (subject) directory."""
    return str(path.parent)


def _enrich_daic(row: dict[str, Any], path: Path) -> dict[str, Any]:
    match = re.fullmatch(r"(?P<subject>\d+)_(?P<segment>\d+)\.wav", path.name)
    if match is None:
        raise ValueError(f"Unexpected DAIC preprocessed audio basename: {path.name}")
    return {
        **row,
        "dataset": "daic",
        "audio_scope": "participant_turns",
        "subject_id": match.group("subject"),
        "segment_id": int(match.group("segment")),
        "sample_id": path.stem,
    }


def _enrich_cmdc(row: dict[str, Any], path: Path) -> dict[str, Any]:
    subject_id = path.parent.name
    question_match = re.fullmatch(r"(Q\d+)\.wav", path.name)
    if question_match is None:
        raise ValueError(f"Unexpected CMDC audio basename: {path.name}")
    return {
        **row,
        "dataset": "cmdc",
        "subject_id": subject_id,
        "question_id": question_match.group(1),
        "sample_id": path.stem,
    }


def _enrich_androids(row: dict[str, Any], path: Path) -> dict[str, Any]:
    recording_id = path.parent.name
    prefix = f"{recording_id}_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"ANDROIDS interview clip does not match its parent recording: {path}")
    turn_text = path.stem[len(prefix):]
    if not turn_text.isdigit():
        raise ValueError(f"Unexpected ANDROIDS interview turn id: {path.name}")
    speaker = _parse_androids_speaker(recording_id)
    return {
        **row,
        "dataset": "androids",
        "task": "interview",
        **speaker,
        "turn_id": int(turn_text),
        "sample_id": path.stem,
    }


def _enrich_d3tec(row: dict[str, Any], path: Path) -> dict[str, Any]:
    match = re.fullmatch(r"(?P<subject>\d+)(?:_(?P<prompt>\d+))?", path.stem)
    if match is None:
        raise ValueError(f"Unexpected D3TEC SM-27 basename: {path.name}")
    subject_id = match.group("subject")
    prompt_id = match.group("prompt") or "0"
    return {
        **row,
        "dataset": "d3tec",
        "source_device": "SM-27",
        "subject_id": subject_id,
        "prompt_id": prompt_id,
        "sample_id": path.stem,
    }


PRESETS: dict[str, Preset] = {
    "daic": Preset(
        name="daic",
        dataset="daic",
        audio_dir=DAIC_ROOT,
        output_path=DAIC_ROOT / "transcripts_gemini_en.jsonl",
        language_arg="English",
        language_tag="en",
        expected_detected_languages=frozenset({"en", "english"}),
        default_batch_size=15,
        batch_key=lambda path: path.stem.split("_")[0],
        enrich=_enrich_daic,
    ),
    "cmdc": Preset(
        name="cmdc",
        dataset="cmdc",
        audio_dir=CMDC_ROOT,
        output_path=CMDC_ROOT / "transcripts_gemini_zh.jsonl",
        language_arg="Chinese (Mandarin)",
        language_tag="zh",
        expected_detected_languages=frozenset({"zh", "chinese", "mandarin"}),
        default_batch_size=6,
        batch_key=_subject_batch_key,
        enrich=_enrich_cmdc,
    ),
    "androids_interview": Preset(
        name="androids_interview",
        dataset="androids",
        audio_dir=ANDROIDS_ROOT / "Interview-Task" / "audio_clip",
        output_path=ANDROIDS_ROOT / "interview_transcripts_gemini_italian.jsonl",
        language_arg="Italian",
        language_tag="it",
        expected_detected_languages=frozenset({"it", "italian"}),
        default_batch_size=11,
        batch_key=_subject_batch_key,
        enrich=_enrich_androids,
    ),
    "d3tec": Preset(
        name="d3tec",
        dataset="d3tec",
        audio_dir=D3TEC_ROOT / "SM-27",
        output_path=D3TEC_ROOT / "transcripts_gemini_spanish.jsonl",
        language_arg="Spanish",
        language_tag="es",
        expected_detected_languages=frozenset({"es", "spanish"}),
        default_batch_size=15,
        batch_key=lambda path: path.stem.split("_")[0],
        enrich=_enrich_d3tec,
    ),
}


class GeminiAPIError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, daily_quota: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.daily_quota = daily_quota


class GeminiClient:
    """REST client with a free-tier-aware rate limiter."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        min_interval_sec: float = 7.0,
        daily_limit: int = 250,
    ) -> None:
        if not api_key:
            raise ValueError("No Gemini API key: set GEMINI_API_KEY or pass --api-key.")
        self.api_key = api_key
        self.model = model
        self.min_interval_sec = float(min_interval_sec)
        self.daily_limit = int(daily_limit)
        self._requests_today = 0
        self._last_request_at = 0.0
        self._day_started_at = time.time()

    def _new_day_at(self) -> datetime:
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        return nxt

    def _wait_for_quota(self) -> None:
        while self._requests_today >= self.daily_limit:
            wake_at = self._new_day_at()
            wait_sec = (wake_at - datetime.now(timezone.utc)).total_seconds()
            LOGGER.info(
                "Daily limit %d reached (%d requests today); sleeping %.1f h until %s UTC",
                self.daily_limit, self._requests_today, wait_sec / 3600, wake_at.strftime("%H:%M"),
            )
            time.sleep(min(wait_sec, 3600.0))
        elapsed = time.time() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

    def transcribe_batch(self, paths: Sequence[Path], language: str) -> dict[str, str]:
        """Transcribe N clips in one request; returns {basename: transcript}."""
        parts: list[dict[str, Any]] = []
        for path in paths:
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": payload,
                    }
                }
            )
        names = [path.name for path in paths]
        parts.append(
            {
                "text": (
                    f"Transcribe the following audio clips verbatim in {language}, "
                    "including every speaker. Do not translate, summarize, or add "
                    "any commentary. Output EXACTLY one line per clip, in any "
                    f"order, using this format: <filename><TAB><transcript>. "
                    f"Filenames: {json.dumps(names)}. Do not output anything "
                    "other than those lines."
                )
            }
        )
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0},
        }
        response_text = self._generate(body)
        return self._parse_mapping(response_text, names)

    @staticmethod
    def _parse_mapping(response_text: str, names: Sequence[str]) -> dict[str, str]:
        """Parse the one-line-per-clip format: ``<filename><TAB><transcript>``.

        Tab-separated lines keep verbatim speech safe: double quotes, pipes and
        other characters inside transcripts cannot corrupt the structure.
        """
        raw = response_text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
        mapping: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            name, sep, transcript = line.partition("\t")
            if not sep or not name:
                raise GeminiAPIError(f"Malformed Gemini output line: {line[:120]!r}")
            mapping[name] = transcript.strip()
        expected = set(names)
        if set(mapping) != expected:
            raise GeminiAPIError(
                f"Gemini key mismatch: got {sorted(mapping)}, expected {sorted(expected)}"
            )
        return {name: text for name, text in mapping.items() if text}

    def _generate(self, body: dict[str, Any]) -> str:
        self._wait_for_quota()
        url = API_ENDPOINT.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key}
        for attempt in range(1, MAX_RETRIES + 1):
            self._last_request_at = time.time()
            self._requests_today += 1
            try:
                response = requests.post(url, headers=headers, json=body, timeout=600)
            except requests.RequestException as exc:
                raise GeminiAPIError(f"Network error: {exc}", retryable=True) from exc
            if response.status_code == 200:
                return self._extract_text(response.json())
            if response.status_code == 429:
                if "free_tier_requests" in response.text:
                    raise GeminiAPIError(
                        "Free-tier daily request quota exhausted for this model "
                        f"({self.model}); sleeping until the reset.",
                        daily_quota=True,
                    )
                if attempt < MAX_RETRIES:
                    LOGGER.warning("429 (quota); retry %d/%d after 60 s", attempt, MAX_RETRIES)
                    time.sleep(60)
                    continue
                raise GeminiAPIError(
                    f"429 quota exhausted after {MAX_RETRIES} attempts: {response.text[:300]}",
                    retryable=True,
                )
            if 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES:
                    LOGGER.warning("HTTP %d; retry %d/%d after 10 s", response.status_code, attempt, MAX_RETRIES)
                    time.sleep(10)
                    continue
            raise GeminiAPIError(
                f"HTTP {response.status_code}: {response.text[:300]}", retryable=False
            )
        raise GeminiAPIError("Unreachable", retryable=True)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiAPIError(
                f"Malformed Gemini response: {json.dumps(payload)[:300]}", retryable=False
            ) from exc


def stable_key(path: Path, preset: Preset) -> str:
    """Stable per-clip id: dataset-relative path (CMDC repeats basenames across subjects)."""
    return str(path.relative_to(preset.audio_dir))


def discover_files(preset: Preset) -> list[Path]:
    files = sorted(
        (path for path in preset.audio_dir.rglob("*.wav")),
        key=lambda path: natural_sort_key(stable_key(path, preset)),
    )
    seen: dict[str, Path] = {}
    for path in files:
        key = stable_key(path, preset)
        if key in seen:
            raise ValueError(
                f"Duplicate stable id {key!r}: {seen[key]} vs {path}. "
                "Resume and coverage use dataset-relative paths as stable identifiers."
            )
        seen[key] = path
    return files


def select_files(
    all_files: Sequence[Path],
    preset: Preset,
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
                re.fullmatch(pattern, path.name)
                or re.fullmatch(pattern, stable_key(path, preset))
                or re.fullmatch(pattern, str(path))
                for pattern in includes
            )
        ]
        if not files:
            raise ValueError(f"No audio files matched --include patterns: {list(includes)}")
    if limit is not None:
        files = files[:limit]
    return files


def review_codes(
    preset: Preset,
    *,
    text: str,
    duration_sec: float | None,
    failed: bool,
) -> list[str]:
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
    return codes


def make_row(
    path: Path,
    preset: Preset,
    *,
    model: str,
    transcript: str,
    duration_sec: float | None,
    reason_codes: Sequence[str],
    asr_detected_language: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "audio_path": str(path),
        "transcript": transcript,
        "language": preset.language_tag,
        "asr_model": model,
        "asr_backend": "gemini_api",
        "asr_language_arg": preset.language_arg,
        "asr_detected_language": asr_detected_language,
        "audio_duration_sec": duration_sec,
        "n_chars": len(transcript),
        "manual_review_recommended": bool(reason_codes),
        "manual_review_reason_codes": list(reason_codes),
    }
    return preset.enrich(row, path)


def append_rows(out_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_stable_ids(out_path: Path, preset: Preset) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = Path(str(row.get("audio_path", "")))
            try:
                done.add(stable_key(audio_path, preset))
            except ValueError:
                done.add(audio_path.name)
    return done


def group_by_subject(files: Sequence[Path], preset: Preset) -> list[list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(preset.batch_key(path), []).append(path)
    ordered = sorted(groups, key=natural_sort_key)
    return [groups[key] for key in ordered]


class FakeGeminiClient:
    """Self-test transport: returns one dummy transcript per requested name."""

    def __init__(self, **_: Any) -> None:
        pass

    def transcribe_batch(self, paths: Sequence[Path], language: str) -> dict[str, str]:
        return {path.name: f"dummy transcript for {path.stem}" for path in paths}


def run_pipeline(
    preset: Preset,
    client: Any,
    out_path: Path,
    *,
    batch_size: int,
    resume: bool,
    include: Sequence[str],
    limit: int | None,
    list_files: bool,
    model: str,
) -> dict[str, Any]:
    all_files = discover_files(preset)
    selected_files = select_files(all_files, preset, includes=include, limit=limit)
    limited = bool(include) or limit is not None
    LOGGER.info(
        "Preset=%s | discovered=%d selected=%d | language=%s/%s | model=%s | output=%s",
        preset.name, len(all_files), len(selected_files),
        preset.language_arg, preset.language_tag, model, out_path,
    )
    if list_files:
        for path in selected_files:
            print(path)
        return {"n_files": len(selected_files)}

    if out_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --resume to continue it."
        )
    done = read_stable_ids(out_path, preset) if resume else set()
    pending = [path for path in selected_files if stable_key(path, preset) not in done]
    LOGGER.info("Resume state | done=%d pending=%d", len(done), len(pending))

    failures: list[str] = []
    written = 0
    subjects = group_by_subject(pending, preset)
    for subject_files in subjects:
        n_batches = (len(subject_files) + batch_size - 1) // batch_size
        for batch_index, batch_start in enumerate(
            range(0, len(subject_files), batch_size), start=1
        ):
            batch = subject_files[batch_start:batch_start + batch_size]
            while True:
                try:
                    durations = {path.name: read_duration_seconds(path) for path in batch}
                    transcripts = _transcribe_batch_with_fallback(
                        client, batch, preset, failures
                    )
                    break
                except GeminiAPIError as exc:
                    if not exc.daily_quota:
                        raise
                    client._requests_today = client.daily_limit
                    client._wait_for_quota()
                    LOGGER.info("Daily quota reset reached; resuming %s", batch[0].parent.name)
            rows: list[dict[str, Any]] = []
            for path in batch:
                if path.name in failures:
                    continue
                transcript = transcripts.get(path.name, "")
                codes = review_codes(
                    preset,
                    text=transcript,
                    duration_sec=durations[path.name],
                    failed=False,
                )
                rows.append(
                    make_row(
                        path,
                        preset,
                        model=model,
                        transcript=transcript,
                        duration_sec=durations[path.name],
                        reason_codes=codes,
                    )
                )
                written += 1
            append_rows(out_path, rows)
            n_ok = len(batch) - sum(1 for p in batch if p.name in failures)
            LOGGER.info(
                "subject=%s | batch %d/%d | ok=%d/%d | total_written=%d",
                batch[0].parent.name, batch_index, n_batches, n_ok, len(batch), written,
            )

    rows = _read_rows(out_path)
    row_stable_ids = {stable_key(Path(row["audio_path"]), preset) for row in rows}
    empty_ids = [row["sample_id"] for row in rows if not str(row.get("transcript", "")).strip()]
    report = {
        "preset": preset.name,
        "dataset": preset.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_file": str(out_path),
        "asr_backend": "gemini_api",
        "asr_model": model,
        "asr_language_arg": preset.language_arg,
        "language_tag": preset.language_tag,
        "audio_root": str(preset.audio_dir),
        "n_discovered_files": len(all_files),
        "n_selected_files": len(selected_files),
        "n_rows": len(rows),
        "n_written_this_run": written,
        "n_empty_transcripts": len(empty_ids),
        "n_failures": len(failures),
        "coverage_complete": (
            {stable_key(path, preset) for path in pending}.issubset(row_stable_ids)
            and not failures
            and not empty_ids
        ),
        "limited_run": limited,
        "failed_files": failures,
        "empty_sample_ids": empty_ids,
        "missing_files": sorted(
            {stable_key(path, preset) for path in pending} - row_stable_ids
        ),
    }
    return report


def _transcribe_batch_with_fallback(
    client: Any,
    batch: Sequence[Path],
    preset: Preset,
    failures: list[str],
) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    try:
        transcripts.update(client.transcribe_batch(batch, preset.language_arg))
        LOGGER.info(
            "batch ok | %d/%d clips | %s",
            len(transcripts), len(batch), batch[0].parent.name,
        )
    except GeminiAPIError as exc:
        if exc.daily_quota:
            raise
        LOGGER.warning("Subject batch failed (%s); falling back per file", exc)
        for path in batch:
            try:
                transcripts.update(client.transcribe_batch([path], preset.language_arg))
            except GeminiAPIError as per_file_exc:
                if per_file_exc.daily_quota:
                    raise
                failures.append(path.name)
                LOGGER.error("Per-file transcription failed: %s | %s", path.name, per_file_exc)
    return transcripts


def _read_rows(out_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_report(report: dict[str, Any], out_path: Path) -> Path:
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--out", help="Output JSONL; defaults to the preset's dataset-root artifact.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--batch-size", type=int, help="Clips per request; preset default if unset.")
    parser.add_argument("--min-interval-sec", type=float, default=7.0)
    parser.add_argument("--daily-limit", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only transcribe paths/basenames matching this regex; repeatable (smoke/review use).",
    )
    parser.add_argument(
        "--list-files", action="store_true",
        help="Print the selected files and exit without any network call.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the plumbing self-test (fake client, no network, no API key) and exit.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    out_path = Path(args.out) if args.out else preset.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.overwrite and out_path.exists():
        out_path.unlink()

    batch_size = args.batch_size or preset.default_batch_size
    client = (
        None
        if args.list_files
        else GeminiClient(
            args.api_key,
            model=args.model,
            min_interval_sec=args.min_interval_sec,
            daily_limit=args.daily_limit,
        )
    )
    report = run_pipeline(
        preset,
        client=client,
        out_path=out_path,
        batch_size=batch_size,
        resume=bool(args.resume),
        include=args.include,
        limit=args.limit,
        list_files=bool(args.list_files),
        model=args.model,
    )
    if args.list_files:
        return report
    report_path = write_report(report, out_path)
    LOGGER.info(
        "Completed %s | rows=%d failures=%d coverage=%s | report=%s",
        preset.name, report["n_rows"], report["n_failures"],
        report["coverage_complete"], report_path,
    )
    return report


def self_test() -> int:
    """Verify discovery, grouping, batching, parsing and row schema without network."""
    parse = GeminiClient._parse_mapping
    parsed = parse('a.wav\thello\nb.wav\tyo', ["a.wav", "b.wav"])
    assert parsed == {"a.wav": "hello", "b.wav": "yo"}, parsed
    quoted = parse('a.wav\the said "hello" there, it\'s fine.', ["a.wav"])
    assert quoted == {"a.wav": 'he said "hello" there, it\'s fine.'}, quoted
    reordered = parse('b.wav\tsecond\na.wav\tfirst', ["a.wav", "b.wav"])
    assert reordered == {"a.wav": "first", "b.wav": "second"}, reordered
    fenced = parse('```\na.wav\thi\n```', ["a.wav"])
    assert fenced == {"a.wav": "hi"}, fenced
    try:
        parse('a.wav\thello', ["a.wav", "b.wav"])
        raise AssertionError("key mismatch must raise")
    except GeminiAPIError:
        pass
    try:
        parse('a.wav hello', ["a.wav"])
        raise AssertionError("missing tab must raise")
    except GeminiAPIError:
        pass

    for preset in PRESETS.values():
        files = discover_files(preset)
        groups = group_by_subject(files[: min(len(files), 25)], preset)
        assert groups and all(groups), f"empty subject group for {preset.name}"
        assert all(g[0].parent == p.parent for g in groups for p in g), preset.name
        row = make_row(
            files[0], preset,
            model=DEFAULT_MODEL,
            transcript="sample",
            duration_sec=10.0,
            reason_codes=[],
        )
        for key in ("audio_path", "transcript", "language", "sample_id", "dataset"):
            assert key in row, f"{preset.name} row missing {key}"
        assert row["language"] == preset.language_tag, preset.name
        assert row["asr_model"] == DEFAULT_MODEL, preset.name

    with tempfile.TemporaryDirectory(prefix="gemini_selftest_") as tmp:
        out = Path(tmp) / "out.jsonl"
        preset = PRESETS["daic"]
        sample = discover_files(preset)[:2]
        for path in sample:
            append_rows(
                out,
                [make_row(path, preset, model=DEFAULT_MODEL, transcript="x",
                          duration_sec=1.0, reason_codes=[])],
            )
        done = read_stable_ids(out, preset)
        assert done == {stable_key(path, preset) for path in sample}, done
    LOGGER.info("self-test passed for presets %s", sorted(PRESETS))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.preset is None:
        raise SystemExit("--preset is required (choices: %s)" % ", ".join(sorted(PRESETS)))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
