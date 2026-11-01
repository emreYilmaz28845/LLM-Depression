import numpy as np
import pytest

from scripts.probe_audio_recording_shortcut import (
    ConditionAudio,
    binary_metrics,
    complement_intervals,
    fit_length,
    gather,
    pair_other_class,
    speech_intervals,
    token_speech_mask,
)

SR = 16000


def _tone_with_gap() -> np.ndarray:
    """0.5 s tone, 0.5 s near-silence, 0.5 s tone."""
    t = np.arange(int(0.5 * SR)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    quiet = np.full(int(0.5 * SR), 1e-4, dtype=np.float32)
    return np.concatenate([tone, quiet, tone])


def test_speech_and_room_tone_split():
    audio = _tone_with_gap()
    intervals = speech_intervals(audio, top_db=30)
    speech = gather(audio, intervals)
    tone = gather(audio, complement_intervals(intervals, len(audio)))
    assert speech.size + tone.size == audio.size
    assert 0.9 * SR < speech.size < 1.2 * SR
    assert float(np.max(np.abs(tone))) < 0.05


def test_complement_intervals_edges():
    assert complement_intervals(np.array([[2, 4], [6, 8]]), 10).tolist() == [[0, 2], [4, 6], [8, 10]]
    assert complement_intervals(np.zeros((0, 2), dtype=int), 5).tolist() == [[0, 5]]


def test_fit_length_tiles_and_crops():
    assert fit_length(np.array([1, 2], dtype=np.float32), 5).tolist() == [1, 2, 1, 2, 1]
    assert fit_length(np.arange(6, dtype=np.float32), 3).tolist() == [0, 1, 2]
    assert fit_length(np.zeros(0, dtype=np.float32), 3).tolist() == [0, 0, 0]


def test_token_speech_mask():
    mask = token_speech_mask(4, 100, np.array([[0, 50]]))
    assert mask.tolist() == [True, True, False, False]


def test_pair_other_class():
    pairs = pair_other_class({"a": 0, "b": 0, "c": 1})
    assert pairs == {"a": "c", "b": "c", "c": "a"}
    with pytest.raises(ValueError):
        pair_other_class({"a": 0})


def test_binary_metrics():
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["uar"] == pytest.approx(0.75)
    assert metrics["positive_f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert binary_metrics([0, 1], [-1, -1])["uar"] == 0


def test_condition_audio_keeps_length_and_level():
    audio = _tone_with_gap()
    examples = [
        {"sample_id": "s1", "subject_id": "a", "label": 0},
        {"sample_id": "s2", "subject_id": "b", "label": 1},
    ]
    sources = {"s1": audio, "s2": audio * 0.1}
    provider = ConditionAudio(lambda ex, sr, silence: [sources[ex["sample_id"]]], examples, SR, 30.0, 0.2)
    for condition in ("real", "speech_only", "room_tone", "silence", "other_subject"):
        provider.condition = condition
        (result,) = provider(examples[0], SR, False)
        assert result.shape == audio.shape
    provider.condition = "room_tone"
    (tone,) = provider(examples[0], SR, False)
    assert float(np.max(np.abs(tone))) < 0.05
    provider.condition = "other_subject"
    (other,) = provider(examples[0], SR, False)
    assert np.allclose(other, audio * 0.1)
    provider.condition = "silence"
    assert not provider(examples[0], SR, False)[0].any()
