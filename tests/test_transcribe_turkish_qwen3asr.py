from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import pytest

from scripts import transcribe_turkish_qwen3asr as asr


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def _write_metadata(path: Path, names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "depresyon_skoru"])
        writer.writeheader()
        for name in names:
            writer.writerow({"file_name": f"nested/{name}", "depresyon_skoru": "17"})


def _args(audio_dir: Path, metadata: Path, out: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "audio_dir": str(audio_dir),
        "metadata_csv": str(metadata),
        "out": str(out),
        "model": "fake/qwen3asr",
        "language": "Turkish",
        "batch_size": 2,
        "max_new_tokens": 32,
        "device": "cpu",
        "dtype": "bfloat16",
        "attn": None,
        "resume": False,
        "overwrite": False,
        "limit": None,
        "self_test": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_metadata_allowlist_filters_inventory_and_records_hashes(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    names = ["aa1-2-1-ank.wav", "aa1-2-2-ank.wav", "excluded-2-1-ank.wav"]
    for name in names:
        _write_wav(audio_dir / name)
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, names[:2])
    out = tmp_path / "transcripts.jsonl"

    report = asr._run(asr._FakeTranscriber(), _args(audio_dir, metadata, out))

    assert report["n_audio_files_on_disk"] == 3
    assert report["n_selected_audio_files"] == 2
    assert report["n_unselected_audio_files"] == 1
    assert report["n_rows"] == 2
    assert report["n_empty_transcripts"] == 0
    assert report["n_failures"] == 0
    assert report["coverage_complete"] is True
    assert report["metadata_sha256"] == asr.sha256_file(metadata)
    assert report["output_sha256"] == asr.sha256_file(out)
    assert {Path(str(row["audio_path"])).name for row in _read_rows(out)} == set(names[:2])

    with pytest.raises(FileExistsError, match="--resume or explicit --overwrite"):
        asr._run(asr._FakeTranscriber(), _args(audio_dir, metadata, out))


def test_metadata_allowlist_rejects_duplicates_and_missing_wavs(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "aç1-2-1-depr.wav")

    duplicate = tmp_path / "duplicate.csv"
    _write_metadata(duplicate, ["aç1-2-1-depr.wav", "ac\u03271-2-1-depr.wav"])
    with pytest.raises(ValueError, match="Duplicate normalized metadata basename"):
        asr.load_metadata_allowlist(duplicate)

    missing = tmp_path / "missing.csv"
    _write_metadata(missing, ["aç1-2-1-depr.wav", "missing-2-1-depr.wav"])
    with pytest.raises(FileNotFoundError, match="selects 1 missing WAV"):
        asr.select_allowlisted_wavs(
            asr.discover_wavs(audio_dir),
            asr.load_metadata_allowlist(missing),
        )


def test_resume_retries_failed_rows_and_removes_duplicates(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    names = ["aa1-2-1-ank.wav", "aa1-2-2-ank.wav"]
    for name in names:
        _write_wav(audio_dir / name)
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, names)
    out = tmp_path / "transcripts.jsonl"

    success = asr.build_row(
        audio_path=audio_dir / names[0],
        text="başarılı",
        detected_language="Turkish",
        duration_sec=1.0,
        model_id="fake/qwen3asr",
        language_arg="Turkish",
    )
    failed = asr.build_row(
        audio_path=audio_dir / names[1],
        text="",
        detected_language="",
        duration_sec=1.0,
        model_id="fake/qwen3asr",
        language_arg="Turkish",
        failed=True,
    )
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in [success, success, failed]) + "\n",
        encoding="utf-8",
    )

    report = asr._run(
        asr._FakeTranscriber(),
        _args(audio_dir, metadata, out, resume=True),
    )
    rows = _read_rows(out)

    assert len(rows) == 2
    assert len({Path(str(row["audio_path"])).name for row in rows}) == 2
    assert all(str(row["transcript"]).strip() for row in rows)
    assert report["n_failures"] == 0
    assert report["n_empty_transcripts"] == 0
    assert asr.report_passes_full_run(report)

    unchanged = out.read_bytes()
    second_report = asr._run(
        asr._FakeTranscriber(),
        _args(audio_dir, metadata, out, resume=True),
    )
    assert out.read_bytes() == unchanged
    assert second_report["n_rows"] == 2


def test_resume_rejects_rows_outside_current_selection(tmp_path: Path) -> None:
    selected = tmp_path / "selected.wav"
    other = tmp_path / "other.wav"
    _write_wav(selected)
    _write_wav(other)
    out = tmp_path / "transcripts.jsonl"
    row = asr.build_row(
        audio_path=other,
        text="başka",
        detected_language="Turkish",
        duration_sec=1.0,
        model_id="fake/qwen3asr",
        language_arg="Turkish",
    )
    out.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the selected audio set"):
        asr.prepare_resume_output(out, [selected])


def test_full_run_acceptance_rejects_empty_or_failed_rows() -> None:
    assert not asr.report_passes_full_run(
        {"coverage_complete": True, "n_empty_transcripts": 1, "n_failures": 0}
    )
    assert not asr.report_passes_full_run(
        {"coverage_complete": True, "n_empty_transcripts": 0, "n_failures": 1}
    )
    assert not asr.report_passes_full_run(
        {"coverage_complete": False, "n_empty_transcripts": 0, "n_failures": 0}
    )
