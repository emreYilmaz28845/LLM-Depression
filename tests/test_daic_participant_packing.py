from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.daic import (
    PACKED30_CHUNK_SAMPLES,
    PACKED30_INVALID_ROW_ALLOWLIST,
    PACKED30_MANIFEST_VARIANT,
    PACKED30_PROTOCOL_ID,
    PACKED30_SAMPLE_RATE,
    _audit_subject_source_rows,
    _jsonl_sha256,
    _pack_retained_intervals,
    _parse_participant_transcript_tsv,
    _truncate_packed30_text,
)


def write_wav(path: Path, seconds: float) -> None:
    frames = int(seconds * PACKED30_SAMPLE_RATE)
    audio = np.zeros(frames, dtype=np.float32)
    sf.write(path, audio, PACKED30_SAMPLE_RATE, subtype="PCM_16")


def make_tsv(path: Path, rows: list[tuple[str, str, str, str]], blank_lines: int = 0) -> None:
    lines = ["start_time\tstop_time\tspeaker\tvalue"]
    for start, stop, speaker, value in rows:
        lines.append(f"{start}\t{stop}\t{speaker}\t{value}")
    lines.extend([""] * blank_lines)
    path.write_text("\ufeff" + "\n".join(lines) + "\n", encoding="utf-8")


def participant_row(start: float, stop: float, value: str = "hello world") -> tuple[str, str, str, str]:
    return (f"{start:.3f}", f"{stop:.3f}", "Participant", value)


def test_tsv_parsing_utf8_sig_tab_and_blank_lines(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    make_tsv(
        tsv,
        [participant_row(1.0, 2.0), participant_row(3.0, 4.0, "second")],
        blank_lines=2,
    )
    parsed, blank = _parse_participant_transcript_tsv(tsv)
    assert blank == 2
    assert [row["source_row_index"] for row in parsed] == [0, 1]
    assert [row["speaker"] for row in parsed] == ["Participant", "Participant"]
    assert parsed[0]["start_time"] == "1.000" and parsed[0]["value"] == "hello world"


def test_tsv_parsing_rejects_unknown_speaker(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    make_tsv(tsv, [("1.0", "2.0", "Interviewer", "hello")])
    with pytest.raises(ValueError, match="Unexpected speaker"):
        _parse_participant_transcript_tsv(tsv)


def test_subject_381_allowlisted_invalid_row_and_filtering(tmp_path: Path) -> None:
    wav = tmp_path / "381_AUDIO.wav"
    write_wav(wav, 30.0)
    tsv = tmp_path / "381_TRANSCRIPT.csv"
    make_tsv(
        tsv,
        [
            ("10.0", "12.0", "Ellie", "hello"),
            participant_row(14.0, 16.0),
            ("1078.550", "1089.320", "Participant", ""),
            participant_row(20.0, 22.0),
        ],
    )
    parsed, _ = _parse_participant_transcript_tsv(tsv)
    audit = _audit_subject_source_rows("381", wav.stat().st_size and 480000, parsed)
    assert len(audit["invalid_rows"]) == 1
    assert audit["invalid_rows"][0]["invalid_reason"] == "out_of_wav_bounds"
    assert audit["invalid_rows"][0]["speaker"] == "Participant"
    assert audit["invalid_rows"][0]["value"] == ""
    reasons = {exclusion["reason"] for exclusion in audit["exclusions"]}
    assert reasons == {"excluded_non_participant", "invalid_allowlisted_row"}
    intervals = audit["retained_intervals"]
    assert len(intervals) == 2
    assert intervals[0]["start_frame"] == 224000
    assert intervals[0]["end_frame"] == 256000
    assert intervals[1]["start_frame"] == 320000
    assert intervals[1]["end_frame"] == 352000


def test_subject_402_allowlisted_ellie_invalid_rows(tmp_path: Path) -> None:
    wav = tmp_path / "402_AUDIO.wav"
    write_wav(wav, 30.0)
    tsv = tmp_path / "402_TRANSCRIPT.csv"
    make_tsv(
        tsv,
        [
            ("965.496", "967.936", "Ellie", "out of bounds"),
            participant_row(10.0, 12.0),
            ("968.853", "970.293", "Ellie", "out of bounds"),
            participant_row(14.0, 16.0),
            ("971.641", "972.251", "Ellie", "out of bounds"),
        ],
    )
    parsed, _ = _parse_participant_transcript_tsv(tsv)
    audit = _audit_subject_source_rows("402", 480000, parsed)
    assert len(audit["invalid_rows"]) == 3
    assert all(row["speaker"] == "Ellie" for row in audit["invalid_rows"])
    assert len(audit["retained_intervals"]) == 2


def test_unexpected_invalid_row_fails_build(tmp_path: Path) -> None:
    wav = tmp_path / "303_AUDIO.wav"
    write_wav(wav, 30.0)
    tsv = tmp_path / "303_TRANSCRIPT.csv"
    make_tsv(tsv, [participant_row(10.0, 12.0), participant_row(5000.0, 5010.0)])
    parsed, _ = _parse_participant_transcript_tsv(tsv)
    with pytest.raises(ValueError, match="invalid-source-row set mismatch"):
        _audit_subject_source_rows("303", 480000, parsed)


def test_overlap_failure_and_adjacency(tmp_path: Path) -> None:
    wav = tmp_path / "w.wav"
    write_wav(wav, 30.0)
    tsv = tmp_path / "t.tsv"
    make_tsv(tsv, [participant_row(10.0, 12.0), participant_row(11.5, 13.0)])
    parsed, _ = _parse_participant_transcript_tsv(tsv)
    with pytest.raises(ValueError, match="overlap"):
        _audit_subject_source_rows("999", 480000, parsed)

    tsv2 = tmp_path / "t2.tsv"
    make_tsv(tsv2, [participant_row(10.0, 12.0), participant_row(12.0, 13.0)])
    parsed2, _ = _parse_participant_transcript_tsv(tsv2)
    audit = _audit_subject_source_rows("999", 480000, parsed2)
    assert len(audit["retained_intervals"]) == 2
    assert audit["retained_intervals"][0]["end_frame"] == audit["retained_intervals"][1]["start_frame"]


def test_non_finite_time_fails(tmp_path: Path) -> None:
    wav = tmp_path / "w.wav"
    write_wav(wav, 30.0)
    tsv = tmp_path / "t.tsv"
    make_tsv(tsv, [participant_row(10.0, 12.0), ("abc", "12.0", "Participant", "bad")])
    parsed, _ = _parse_participant_transcript_tsv(tsv)
    with pytest.raises(ValueError, match="invalid-source-row set mismatch"):
        _audit_subject_source_rows("999", 480000, parsed)


def _interval(start: float, stop: float, row_index: int = 0) -> dict:
    return {
        "source_row_index": row_index,
        "start_frame": int(round(start * PACKED30_SAMPLE_RATE)),
        "end_frame": int(round(stop * PACKED30_SAMPLE_RATE)),
        "start_time": start,
        "stop_time": stop,
        "value": f"row {row_index}",
    }


def test_packing_exact_boundary_chunk() -> None:
    chunks = _pack_retained_intervals([_interval(0.0, 30.0)])
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["participant_sample_count"] == PACKED30_CHUNK_SAMPLES
    assert len(chunks[0]["spans"]) == 1


def test_packing_splits_long_turn_across_chunks() -> None:
    chunks = _pack_retained_intervals([_interval(0.0, 37.5)])
    assert len(chunks) == 2
    assert chunks[0]["participant_sample_count"] == PACKED30_CHUNK_SAMPLES
    assert chunks[1]["participant_sample_count"] == 120000
    assert chunks[0]["chunk_index"] == 0 and chunks[1]["chunk_index"] == 1
    first_span = chunks[0]["spans"][0]
    second_span = chunks[1]["spans"][0]
    assert first_span["end_frame"] == PACKED30_CHUNK_SAMPLES
    assert second_span["start_frame"] == PACKED30_CHUNK_SAMPLES
    assert first_span["source_row_index"] == second_span["source_row_index"] == 0


def test_packing_concatenates_short_turns_and_keeps_final_partial() -> None:
    chunks = _pack_retained_intervals(
        [_interval(0.0, 12.5, 0), _interval(12.5, 31.25, 1), _interval(31.25, 31.75, 2)]
    )
    assert len(chunks) == 2
    assert chunks[0]["participant_sample_count"] == PACKED30_CHUNK_SAMPLES
    assert chunks[1]["participant_sample_count"] == 28000
    assert {span["source_row_index"] for span in chunks[0]["spans"]} == {0, 1}
    assert {span["source_row_index"] for span in chunks[1]["spans"]} == {1, 2}
    assert chunks[1]["spans"][0]["source_row_index"] == 1


def test_packing_preserves_every_sample_exactly_once() -> None:
    intervals = [_interval(0.25, 8.0, 0), _interval(9.0, 45.0, 1), _interval(45.0, 46.0, 2)]
    chunks = _pack_retained_intervals(intervals)
    total = sum(chunk["participant_sample_count"] for chunk in chunks)
    retained = sum(int(interval["end_frame"]) - int(interval["start_frame"]) for interval in intervals)
    assert total == retained
    frames = [frame for chunk in chunks for span in chunk["spans"] for frame in range(span["start_frame"], span["end_frame"])]
    assert len(frames) == len(set(frames)) == retained


def test_packing_no_inserted_silence() -> None:
    intervals = [_interval(1.0, 2.0, 0), _interval(3.0, 4.0, 1)]
    chunks = _pack_retained_intervals(intervals)
    assert len(chunks) == 1
    assert chunks[0]["participant_sample_count"] == 32000
    span_frames = sum(span["end_frame"] - span["start_frame"] for span in chunks[0]["spans"])
    assert span_frames == chunks[0]["participant_sample_count"]


def test_jsonl_sha256_is_deterministic() -> None:
    rows = [{"b": 2, "a": 1}, {"a": [3, 1], "c": "x"}]
    assert _jsonl_sha256(rows) == _jsonl_sha256(rows)
    assert len(_jsonl_sha256(rows)) == 64


def test_truncation_matches_existing_deterministic_behavior() -> None:
    text = "x" * 5000
    truncated, log = _truncate_packed30_text(text, 4000)
    assert truncated == text[:4000]
    assert log == {"transcript_original_chars": 5000, "transcript_kept_chars": 4000, "transcript_truncated": True}
    unchanged, no_log = _truncate_packed30_text("short", 4000)
    assert unchanged == "short" and no_log is None


def test_locked_allowlist_constant_is_consistent() -> None:
    assert len(PACKED30_INVALID_ROW_ALLOWLIST) == 4
    assert {item["subject_id"] for item in PACKED30_INVALID_ROW_ALLOWLIST} == {"381", "402"}
    reasons = {item["reason"] for item in PACKED30_INVALID_ROW_ALLOWLIST}
    assert reasons == {"empty value; excluded before packing", "excluded speaker"}


DAIC_UNPROCESSED_ROOT = Path(
    os.environ.get(
        "DAIC_UNPROCESSED_ROOT",
        "/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/unprocessed",
    )
)
DAIC_LABEL_ROOT = Path(
    os.environ.get(
        "DAIC_LABEL_ROOT",
        "/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/minimal_zips",
    )
)
HAS_CORPUS = (
    DAIC_UNPROCESSED_ROOT.is_dir()
    and DAIC_LABEL_ROOT.is_dir()
    and len(list(DAIC_UNPROCESSED_ROOT.glob("*_AUDIO.wav"))) == 189
)


@pytest.mark.skipif(
    not HAS_CORPUS,
    reason="Requires the real DAIC corpus (189 subjects) for the locked contract totals.",
)
def test_full_builder_contract_and_rebuild_hashes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAIC_UNPROCESSED_ROOT", str(DAIC_UNPROCESSED_ROOT))
    monkeypatch.setenv("DAIC_LABEL_ROOT", str(DAIC_LABEL_ROOT))
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from src.data.build_manifest import build_for_config

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/daic_participant_packed30/daic_participant_packed30_audio_only.yaml"
    )
    manifest_dir = tmp_path / "outputs" / "manifests_daic_participant_packed30"
    split_dir = tmp_path / "outputs" / "splits_daic_participant_packed30"

    def run() -> dict[str, str]:
        build_for_config(config_path, [])
        return {
            name: hashlib.sha256((manifest_dir / name).read_bytes()).hexdigest()
            for name in (
                "daic_participant_speech_packed30_manifest.jsonl",
                "daic_participant_speech_packed30_corpus_audit.json",
                "daic_participant_speech_packed30_metadata.json",
            )
        }

    first = run()
    second = run()
    assert first == second, "Rebuilding twice must be byte-identical"
    import json

    rows = [
        json.loads(line)
        for line in (manifest_dir / "daic_participant_speech_packed30_manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3021
    assert all(str(row["protocol_id"]) == PACKED30_PROTOCOL_ID for row in rows)
    assert len({str(row["sample_id"]) for row in rows}) == 3021
    assert {str(row["split_original"]) for row in rows} == {"train", "val", "test"}
    assert sum(int(row["participant_sample_count"]) for row in rows) == 1406614000
