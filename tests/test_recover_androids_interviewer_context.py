from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.recover_androids_interviewer_context import (
    _stub_unused_nagisa_dependency,
    build_span_report,
    discover_interviewer_context_spans,
    transcribe_contexts,
)


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text
        self.language = "Italian"


class _Backend:
    def transcribe(self, audio, language):
        assert language == ["Italian"] * len(audio)
        return [_Result(f"context {Path(path).stem}") for path in audio]


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "Androids-Corpus"
    full_dir = root / "Interview-Task" / "audio" / "HC"
    clip_dir = root / "Interview-Task" / "audio_clip" / "01_CF56_1"
    full_dir.mkdir(parents=True)
    clip_dir.mkdir(parents=True)
    sample_rate = 100
    full = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    sf.write(full_dir / "01_CF56_1.wav", full, sample_rate)
    sf.write(clip_dir / "01_CF56_1_1.wav", full[200:400], sample_rate)
    sf.write(clip_dir / "01_CF56_1_2.wav", full[600:900], sample_rate)
    with (root / "interview_timedata.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(["01_CF56_1", "2.0", "4.0", "6.0", "9.0"])
    return root


def test_discovers_preceding_context_and_maps_it_to_turn(tmp_path) -> None:
    root = _fixture(tmp_path)
    spans = discover_interviewer_context_spans(root)
    assert len(spans) == 2
    assert (spans[0].context_start, spans[0].context_end) == (0.0, 2.0)
    assert (spans[1].context_start, spans[1].context_end) == (4.0, 6.0)
    assert spans[1].participant_start == 6.0
    assert spans[1].participant_end == 9.0
    report = build_span_report(
        spans,
        dataset_root=root,
        timing_path=root / "interview_timedata.csv",
    )
    assert report["num_recordings"] == 1
    assert report["num_context_spans"] == 2
    assert report["context_duration_total_sec"] == 4.0


def test_transcription_output_is_context_not_verified_question(tmp_path) -> None:
    root = _fixture(tmp_path)
    spans = discover_interviewer_context_spans(root)
    output = tmp_path / "contexts.jsonl"
    rows = transcribe_contexts(
        spans,
        output_path=output,
        backend=_Backend(),
        model_id="fake/asr",
        batch_size=2,
        min_context_seconds=0.15,
        resume=False,
        overwrite=False,
    )
    assert len(rows) == 2
    assert all(row["interviewer_context_transcript"] for row in rows)
    assert all(row["question_text_verified"] is False for row in rows)
    persisted = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["context_id"] for row in persisted] == [span.context_id for span in spans]


def test_short_context_is_kept_but_not_sent_to_asr(tmp_path) -> None:
    root = _fixture(tmp_path)
    spans = discover_interviewer_context_spans(root)
    short = spans[0].__class__(
        **{
            **spans[0].__dict__,
            "context_end": 0.05,
            "context_duration": 0.05,
        }
    )
    output = tmp_path / "short.jsonl"
    rows = transcribe_contexts(
        [short],
        output_path=output,
        backend=_Backend(),
        model_id="fake/asr",
        batch_size=1,
        min_context_seconds=0.15,
        resume=False,
        overwrite=False,
    )
    assert rows[0]["asr_status"] == "SKIPPED_CONTEXT_SHORTER_THAN_MINIMUM"
    assert rows[0]["interviewer_context_transcript"] == ""


def test_nagisa_stub_fails_closed_if_forced_alignment_is_called() -> None:
    import sys

    original = sys.modules.get("nagisa")
    try:
        _stub_unused_nagisa_dependency()
        try:
            sys.modules["nagisa"].tagging("test")
        except RuntimeError as exc:
            assert "forced alignment is not supported" in str(exc)
        else:
            raise AssertionError("Disabled forced alignment unexpectedly ran")
    finally:
        if original is None:
            sys.modules.pop("nagisa", None)
        else:
            sys.modules["nagisa"] = original
