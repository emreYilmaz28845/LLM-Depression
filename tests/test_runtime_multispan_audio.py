from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.runtime import (
    AudioTextDataset,
    load_audio_spans_array,
    uses_audio_spans,
)


def make_wav(path: Path, frames: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(frames).astype(np.float32) * 0.1
    sf.write(path, audio, 16000, subtype="PCM_16")


def span_example(wav_path: str, spans: list[dict], expected: int | None = None) -> dict:
    return {
        "audio_path": wav_path,
        "audio_spans": spans,
        "participant_sample_count": expected,
        "sample_id": "x_participant_p30_000",
        "subject_id": "x",
        "label": 0,
        "label_text": "Non-depressed",
        "internal_label_text": "Non-depressed",
        "transcript": "",
        "prompt_text": "p",
        "training_text": "t",
        "dataset": "daic",
        "input_modality": "audio_only",
        "audio_paths": [],
        "audio_clip_seconds": [],
        "audio_start_times": [],
        "audio_end_times": [],
    }


def test_load_audio_spans_concatenates_in_order(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 8000)
    first, _ = sf.read(wav, start=0, frames=2000, dtype="float32")
    second, _ = sf.read(wav, start=6000, frames=2000, dtype="float32")
    spans = [
        {"start_frame": 0, "end_frame": 2000},
        {"start_frame": 6000, "end_frame": 8000},
    ]
    loaded = load_audio_spans_array(str(wav), spans, 16000, False, expected_samples=4000)
    assert loaded.shape == (4000,)
    np.testing.assert_array_equal(loaded[:2000], first)
    np.testing.assert_array_equal(loaded[2000:], second)


def test_load_audio_spans_validates_sample_count(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 8000)
    spans = [{"start_frame": 0, "end_frame": 2000}, {"start_frame": 6000, "end_frame": 8000}]
    with pytest.raises(ValueError, match="expected 3999"):
        load_audio_spans_array(str(wav), spans, 16000, False, expected_samples=3999)


def test_load_audio_spans_rejects_out_of_bounds_span(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 8000)
    with pytest.raises(ValueError, match="outside"):
        load_audio_spans_array(str(wav), [{"start_frame": 7000, "end_frame": 9000}], 16000, False, 2000)


def test_load_audio_spans_rejects_empty_span_list(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 8000)
    with pytest.raises(ValueError, match="at least one span"):
        load_audio_spans_array(str(wav), [], 16000, False)


def test_load_audio_spans_silence_path(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 8000)
    loaded = load_audio_spans_array(
        str(wav), [{"start_frame": 0, "end_frame": 1000}], 16000, True, expected_samples=1000
    )
    assert loaded.shape == (1000,)
    assert not loaded.any()


def test_uses_audio_spans_predicate() -> None:
    assert uses_audio_spans({"audio_spans": [{"start_frame": 0, "end_frame": 10}]})
    assert not uses_audio_spans({"audio_spans": []})
    assert not uses_audio_spans({"audio_paths": ["/x.wav"]})
    assert not uses_audio_spans({})


def test_audio_text_dataset_loads_span_example_once(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 48000)
    example = span_example(
        str(wav),
        [{"start_frame": 0, "end_frame": 20000}, {"start_frame": 40000, "end_frame": 48000}],
        expected=28000,
    )
    dataset = AudioTextDataset([example], processor_sampling_rate=16000)
    item = dataset[0]
    assert len(item["audio_arrays"]) == 1
    assert item["audio_arrays"][0].shape == (28000,)
    assert item["participant_sample_count"] == 28000


def test_audio_text_dataset_span_max_30s_waveform(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 500000)
    example = span_example(
        str(wav),
        [{"start_frame": 0, "end_frame": 480000}],
        expected=480000,
    )
    item = AudioTextDataset([example], processor_sampling_rate=16000)[0]
    assert item["audio_arrays"][0].shape == (480000,)


def test_legacy_path_examples_are_unaffected(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    make_wav(wav, 16000)
    example = {
        "audio_paths": [str(wav)],
        "audio_clip_seconds": [1.0],
        "audio_start_times": [0.0],
        "audio_end_times": [None],
        "sample_id": "legacy",
        "subject_id": "s",
        "label": 0,
        "label_text": "Non-depressed",
        "internal_label_text": "Non-depressed",
        "transcript": "",
        "prompt_text": "p",
        "training_text": "t",
        "dataset": "daic",
        "input_modality": "audio_only",
    }
    item = AudioTextDataset([example], processor_sampling_rate=16000)[0]
    assert item["audio_arrays"][0].shape == (16000,)
