"""Re-window saved Qwen3-Omni manifests for the wav2vec2 audio baseline.

The wav2vec2 baseline must see the same natural units and the same speech as
the Qwen3-Omni frozen-feature routes, only cut into shorter windows. Both
helpers start from the exact manifest rows of the Omni reference run, so no
raw-corpus rebuild (and no transcript join) is involved:

* ``rewindow_harmonized_rows`` (D3TEC, ANDROIDS Interview): these manifests
  hold one row per 30 s window of a natural unit (one WAV per response or
  turn). The unit's windows are checked to tile ``[0, duration]`` and the unit
  is re-split with the same equal-duration rule at ``segment_seconds``.
  CMDC and Turkish cut windows at runtime, so they need only the config's
  ``data.segment_seconds``.
* ``repack_daic_packed_rows`` (DAIC): packed30 rows hold ordered frame spans of
  the participant-only speech stream. The spans are merged back into the
  retained source intervals and re-packed with the manifest builder's own
  ``_pack_retained_intervals`` at ``chunk_samples``. This equals building the
  manifest at that chunk size, because packing depends only on the ordered
  retained intervals.

Segment-level transcripts are dropped from re-windowed rows: they belong to
the old windows, and the baseline is audio-only.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

from src.data.daic import PACKED30_SAMPLE_RATE, _pack_retained_intervals

HARMONIZED_REWINDOW_DATASETS = ("d3tec", "androids_interview")
RUNTIME_WINDOW_DATASETS = ("cmdc", "turkish")
TILE_TOLERANCE_SECONDS = 1e-6
SEGMENT_TRANSCRIPT_FIELDS = ("transcript", "segment_transcript", "chunk_transcript", "window_transcript")


def equal_duration_windows(duration: float, segment_seconds: float) -> list[tuple[float, float]]:
    """Same rule as ``src.data.d3tec``/``androids``/runtime: ceil(d/s) equal windows."""
    if duration <= 0:
        raise ValueError(f"Unit duration must be positive, got {duration}.")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    count = max(1, int(math.ceil(duration / segment_seconds)))
    width = duration / count
    return [(i * width, duration if i == count - 1 else (i + 1) * width) for i in range(count)]


def _unit_key(row: dict[str, Any]) -> str:
    value = str(row.get("response_id") or "").strip()
    if not value:
        raise ValueError(f"Harmonized manifest row {row.get('sample_id')!r} has no response_id.")
    return value


def rewindow_harmonized_rows(
    rows: list[dict[str, Any]], dataset: str, segment_seconds: float
) -> list[dict[str, Any]]:
    if dataset not in HARMONIZED_REWINDOW_DATASETS:
        raise ValueError(f"Manifest re-windowing is only for {HARMONIZED_REWINDOW_DATASETS}, got {dataset!r}.")
    units: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        units.setdefault(_unit_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for unit_id, unit_rows in units.items():
        unit_rows = sorted(unit_rows, key=lambda r: int(r["segment_index"]))
        paths = {str(r["audio_path"]) for r in unit_rows}
        subjects = {str(r["subject_id"]) for r in unit_rows}
        if len(paths) != 1 or len(subjects) != 1:
            raise ValueError(f"Unit {unit_id} spans several audio files or subjects.")
        expected = int(unit_rows[0]["num_segments"])
        if [int(r["segment_index"]) for r in unit_rows] != list(range(expected)):
            raise ValueError(f"Unit {unit_id} windows are not segment_index 0..{expected - 1}.")
        cursor = 0.0
        for r in unit_rows:
            if abs(float(r["start_time"]) - cursor) > TILE_TOLERANCE_SECONDS:
                raise ValueError(f"Unit {unit_id} windows do not tile the unit from 0 s.")
            cursor = float(r["end_time"])
        duration = cursor
        windows = equal_duration_windows(duration, segment_seconds)
        base_id = str(unit_rows[0]["sample_id"]).rsplit("_", 1)[0]
        for index, (start, end) in enumerate(windows):
            new = dict(unit_rows[0])
            for field in SEGMENT_TRANSCRIPT_FIELDS:
                if field in new:
                    new[field] = ""
            new.update(
                {
                    "sample_id": f"{base_id}_w{int(round(segment_seconds))}s{index:02d}",
                    "segment_index": index,
                    "num_segments": len(windows),
                    "start_time": start,
                    "end_time": end,
                    "segment_duration": end - start,
                    "source_unit_duration": duration,
                    "source_window_count": expected,
                    "rewindow_segment_seconds": float(segment_seconds),
                }
            )
            if "window_index" in new:
                new["window_index"] = index
            if "num_windows" in new:
                new["num_windows"] = len(windows)
            if "window_id" in new:
                new["window_id"] = new["sample_id"]
            output.append(new)
    if len({r["sample_id"] for r in output}) != len(output):
        raise ValueError("Re-windowed sample IDs are not unique.")
    return output


def _subject_intervals(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge a subject's packed spans back into the retained source intervals."""
    intervals: list[dict[str, Any]] = []
    for row in sorted(subject_rows, key=lambda r: int(r["chunk_index"])):
        for span in row["audio_spans"]:
            start, end = int(span["start_frame"]), int(span["end_frame"])
            source = int(span["source_row_index"])
            if end <= start:
                raise ValueError("Empty or reversed packed span.")
            last = intervals[-1] if intervals else None
            if last is not None and start < last["end_frame"]:
                raise ValueError("Packed spans are not in chronological order.")
            if last is not None and last["source_row_index"] == source and last["end_frame"] == start:
                last["end_frame"] = end
            else:
                intervals.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                        "source_row_index": source,
                        "start_time": float(span["source_start_time"]),
                        "stop_time": float(span["source_stop_time"]),
                    }
                )
    return intervals


def repack_daic_packed_rows(rows: list[dict[str, Any]], chunk_samples: int) -> list[dict[str, Any]]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    by_subject: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in sorted(rows, key=lambda r: (int(r["subject_id"]), int(r["chunk_index"]))):
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    seconds = chunk_samples / PACKED30_SAMPLE_RATE
    tag = f"p{int(seconds)}" if float(seconds).is_integer() else f"s{chunk_samples}"
    output: list[dict[str, Any]] = []
    for subject, subject_rows in by_subject.items():
        if len({str(r["audio_path"]) for r in subject_rows}) != 1:
            raise ValueError(f"DAIC subject {subject} spans several audio files.")
        source_samples = sum(int(r["participant_sample_count"]) for r in subject_rows)
        intervals = _subject_intervals(subject_rows)
        if sum(i["end_frame"] - i["start_frame"] for i in intervals) != source_samples:
            raise ValueError(f"DAIC subject {subject}: merged spans lose samples.")
        chunks = _pack_retained_intervals(intervals, chunk_samples=chunk_samples)
        if sum(int(c["participant_sample_count"]) for c in chunks) != source_samples:
            raise ValueError(f"DAIC subject {subject}: re-packing changes the speech total.")
        template = dict(subject_rows[0])
        template["chunk_transcript"] = ""
        for chunk in chunks:
            new = dict(template)
            index = int(chunk["chunk_index"])
            new.update(
                {
                    "sample_id": f"{subject}_participant_{tag}_{index:03d}",
                    "audio_spans": chunk["spans"],
                    "participant_sample_count": int(chunk["participant_sample_count"]),
                    "chunk_index": index,
                    "num_chunks": len(chunks),
                    "repack_chunk_samples": int(chunk_samples),
                    "source_num_chunks": len(subject_rows),
                }
            )
            output.append(new)
    return output
