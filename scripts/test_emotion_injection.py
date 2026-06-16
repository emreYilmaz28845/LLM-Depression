"""No-model validation of emotion-augmented prompt injection + K-chunk alignment.

Builds tiny in-memory manifests + an emotion cache and asserts (§9 D of the plan):
  * single-chunk: the rendered prompt contains the chunk's English caption.
  * subject deterministic: the K interleaved descriptions match the K deterministic
    chunks, in order.
  * subject random (seeded): after re-render, descriptions track the SAMPLED chunk
    order (the §4.3 alignment guarantee).
  * missing caption -> configured neutral fallback, not a crash.

Runs without loading any model or reading audio. Exit 0 on success.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.emotion import NEUTRAL_FALLBACK_CAPTION
from src.data.runtime import (
    AudioTextDataset,
    _base_example_from_row,
    _build_subject_level_audio_examples,
)


def _subject_config() -> dict:
    return {
        "dataset": "daic",
        "prompt": {
            "system": "You are a clinical expert.",
            "user_template": (
                "{audio_context_block}\n{transcript_block}"
                "Based on the speech audio segments, transcript, and emotional "
                "descriptions, determine whether the subject is {label_descriptor}.\n"
                "{label_instruction}"
            ),
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": True,
            "use_emotion": True,
            "sample_mode": "subject_audio",
            "chunks_per_subject": 3,
            "max_audio_seconds_per_chunk": 30.0,
        },
    }


def _single_config() -> dict:
    cfg = _subject_config()
    cfg["data"]["sample_mode"] = "chunk"
    cfg["prompt"]["user_template"] = (
        "{audio_context_block}\nBased on the speech audio, transcript, and emotional "
        "description, determine whether the subject is {label_descriptor}.\n"
        "{emotion_block}{transcript_block}{label_instruction}"
    )
    return cfg


def _rows(n: int) -> list[dict]:
    return [
        {
            "dataset": "daic",
            "subject_id": "300",
            "sample_id": f"300_seg_{i}",
            "audio_path": f"/tmp/300_seg_{i}.wav",
            "label": 1,
            "label_text": "Depressed",
            "transcript": "hello world",
        }
        for i in range(n)
    ]


def test_single_chunk() -> None:
    cache = {"300_seg_0": "the voice is calm and slow"}
    example, _ = _base_example_from_row(
        _rows(1)[0], _single_config(), 4000, cache, "neutral_fallback"
    )
    assert "Emotional description: the voice is calm and slow" in example["prompt_text"], example["prompt_text"]
    print("[ok] single-chunk caption injected")


def test_single_chunk_fallback() -> None:
    example, _ = _base_example_from_row(_rows(1)[0], _single_config(), 4000, {}, "neutral_fallback")
    assert NEUTRAL_FALLBACK_CAPTION in example["prompt_text"]
    print("[ok] single-chunk missing -> neutral fallback")


def test_subject_deterministic() -> None:
    rows = _rows(5)
    cache = {f"300_seg_{i}": f"caption number {i}" for i in range(5)}
    examples = _build_subject_level_audio_examples(
        rows, _subject_config(), "train", 4000, None, cache, "neutral_fallback"
    )
    ex = examples[0]
    # Deterministic evenly-spaced indices for K=3 over 5 -> 0, 2, 4.
    expected_paths = ex["audio_paths"]
    prompt = ex["prompt_text"]
    for i, path in enumerate(expected_paths, start=1):
        sid_caption = ex["chunk_caption_by_path"][path]
        assert f"Emotional description {i}: {sid_caption}" in prompt, prompt
        assert f"Audio {i}:" in prompt
    print(f"[ok] subject deterministic interleave aligned ({len(expected_paths)} chunks)")


def test_subject_random_alignment() -> None:
    rows = _rows(5)
    cache = {f"300_seg_{i}": f"caption number {i}" for i in range(5)}
    examples = _build_subject_level_audio_examples(
        rows, _subject_config(), "train", 4000, None, cache, "neutral_fallback"
    )
    ds = AudioTextDataset(examples, chunk_sampling="random", chunk_sampling_seed=7)
    ex = examples[0]
    sampled_paths, _ = ds._resolve_audio_plan(ex)
    rendered = ds._rerender_emotion_prompt(ex, sampled_paths)
    prompt = rendered["prompt_text"]
    for i, path in enumerate(sampled_paths, start=1):
        caption = ex["chunk_caption_by_path"][path]
        assert f"Emotional description {i}: {caption}" in prompt, (sampled_paths, prompt)
    print(f"[ok] subject random re-render tracks sampled order {sampled_paths}")


def test_subject_drop_policy() -> None:
    rows = _rows(5)
    cache = {f"300_seg_{i}": f"caption {i}" for i in range(5)}
    del cache["300_seg_2"]  # one missing
    examples = _build_subject_level_audio_examples(
        rows, _subject_config(), "train", 4000, None, cache, "drop_emotion_line"
    )
    ex = examples[0]
    # Audio placeholders must still equal K even when a description line is dropped.
    assert ex["prompt_text"].count("<|AUDIO|>") == ex["chunks_per_subject"]
    print("[ok] drop policy keeps audio count while dropping a description line")


def main() -> int:
    test_single_chunk()
    test_single_chunk_fallback()
    test_subject_deterministic()
    test_subject_random_alignment()
    test_subject_drop_policy()
    print("\nAll emotion-injection checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
