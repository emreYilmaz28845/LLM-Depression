import numpy as np
import pytest
import torch

from src.data.daic import _pack_retained_intervals
from src.features.wav2vec2_windows import (
    equal_duration_windows,
    repack_daic_packed_rows,
    rewindow_harmonized_rows,
)


def _harmonized_rows(durations, segment_seconds=30.0, dataset="d3tec"):
    rows = []
    for unit, duration in enumerate(durations):
        windows = equal_duration_windows(duration, segment_seconds)
        for index, (start, end) in enumerate(windows):
            rows.append({
                "dataset": dataset, "subject_id": "001", "label": 1,
                "response_id": f"001::001_p{unit}", "sample_id": f"001_p{unit}_s{index}",
                "audio_path": f"/x/001_{unit}.wav", "segment_index": index, "num_segments": len(windows),
                "start_time": start, "end_time": end, "segment_duration": end - start,
                "transcript": "old window text", "full_response_transcript": "unit text",
            })
    return rows


def test_rewindow_at_source_length_reproduces_windows():
    rows = _harmonized_rows([12.0, 31.0, 95.5])
    out = rewindow_harmonized_rows(rows, "d3tec", 30.0)
    assert [(r["start_time"], r["end_time"]) for r in out] == [(r["start_time"], r["end_time"]) for r in rows]


def test_rewindow_20s_tiles_each_unit_equally():
    rows = _harmonized_rows([12.0, 31.0, 95.5])
    out = rewindow_harmonized_rows(rows, "d3tec", 20.0)
    by_unit = {}
    for r in out:
        by_unit.setdefault(r["response_id"], []).append(r)
    counts = {k: len(v) for k, v in by_unit.items()}
    assert counts == {"001::001_p0": 1, "001::001_p1": 2, "001::001_p2": 5}
    for unit_rows in by_unit.values():
        assert unit_rows[0]["start_time"] == 0.0
        assert all(r["segment_duration"] <= 20.0 + 1e-9 for r in unit_rows)
        widths = {round(r["segment_duration"], 9) for r in unit_rows}
        assert len(widths) == 1
        assert all(r["transcript"] == "" and r["full_response_transcript"] == "unit text" for r in unit_rows)
    assert len({r["sample_id"] for r in out}) == len(out)


def test_rewindow_rejects_gapped_unit():
    rows = _harmonized_rows([45.0])
    rows[1]["start_time"] += 0.5
    with pytest.raises(ValueError, match="tile"):
        rewindow_harmonized_rows(rows, "d3tec", 20.0)


def test_rewindow_refuses_runtime_window_datasets():
    with pytest.raises(ValueError):
        rewindow_harmonized_rows(_harmonized_rows([10.0]), "cmdc", 20.0)


def _intervals():
    spans, cursor = [], 1000
    for index, length in enumerate([5000, 700000, 120, 330000, 64000, 1]):
        spans.append({"start_frame": cursor, "end_frame": cursor + length, "source_row_index": index,
                      "start_time": cursor / 16000, "stop_time": (cursor + length) / 16000})
        cursor += length + 777
    return spans


def _packed_rows(chunk_samples):
    rows = []
    chunks = _pack_retained_intervals(_intervals(), chunk_samples=chunk_samples)
    for chunk in chunks:
        rows.append({"subject_id": "300", "label": 0, "audio_path": "/x/300_AUDIO.wav",
                     "sample_id": f"300_participant_p30_{chunk['chunk_index']:03d}",
                     "audio_spans": chunk["spans"], "participant_sample_count": chunk["participant_sample_count"],
                     "chunk_index": chunk["chunk_index"], "num_chunks": len(chunks), "chunk_transcript": "t",
                     "protocol_id": "daic_participant_speech_packed30_v1"})
    return rows


@pytest.mark.parametrize("target", [480000, 320000, 240000])
def test_repack_equals_direct_packing(target):
    out = repack_daic_packed_rows(_packed_rows(480000), target)
    direct = _pack_retained_intervals(_intervals(), chunk_samples=target)
    assert [r["audio_spans"] for r in out] == [c["spans"] for c in direct]
    assert [r["participant_sample_count"] for r in out] == [c["participant_sample_count"] for c in direct]
    assert all(r["chunk_transcript"] == "" for r in out)
    assert len({r["sample_id"] for r in out}) == len(out)


def test_pool_hidden_mean_and_population_variance():
    from src.features.extract_wav2vec2 import pool_hidden

    hidden = torch.randn(1, 37, 8)
    pooled = pool_hidden(hidden)
    h = hidden[0].double().numpy()
    np.testing.assert_allclose(pooled["mean"], h.mean(0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pooled["meanvar"], np.concatenate([h.mean(0), h.var(0)]), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        pool_hidden(torch.randn(2, 5, 8))


def test_remap_audio_paths_requires_every_path():
    from src.features.extract_wav2vec2 import remap_audio_paths

    rows = [{"audio_path": "/gpfs/a.wav"}, {"audio_paths": ["/gpfs/b.wav"], "audio_path": ""}]
    out = remap_audio_paths(rows, {"/gpfs/a.wav": "/local/a.wav", "/gpfs/b.wav": "/local/b.wav"})
    assert out[0]["audio_path"] == "/local/a.wav" and out[1]["audio_paths"] == ["/local/b.wav"]
    assert rows[0]["audio_path"] == "/gpfs/a.wav"
    with pytest.raises(ValueError, match="not in the verified path map"):
        remap_audio_paths(rows, {"/gpfs/a.wav": "/local/a.wav"})
