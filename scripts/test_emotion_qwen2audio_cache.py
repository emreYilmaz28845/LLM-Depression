"""No-model validation that a Qwen2-Audio emotion cache is a drop-in for SECap.

A Qwen2-Audio cache row has ``emotion_en`` populated directly (English, no
translation) and NO ``emotion_zh`` field. This test proves:
  * ``load_emotion_cache`` reads ``emotion_en`` from the Qwen schema, and the
    existing prompt-injection path renders the caption unchanged.
  * ``build_emotion_cache validate`` reports full coverage for a Qwen cache even
    though it carries no ``emotion_zh`` (the validator only requires ``emotion_en``).

Runs without loading any model or reading audio. Exit 0 on success.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.emotion import load_emotion_cache
from src.data.runtime import _base_example_from_row
from src.emotion import build_emotion_cache


def _single_config() -> dict:
    return {
        "dataset": "daic",
        "prompt": {
            "system": "You are a clinical expert.",
            "user_template": (
                "{audio_context_block}\nBased on the speech audio, transcript, and emotional "
                "description, determine whether the subject is {label_descriptor}.\n"
                "{emotion_block}{transcript_block}{label_instruction}"
            ),
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": True,
            "use_emotion": True,
            "sample_mode": "chunk",
            "max_audio_seconds_per_chunk": 30.0,
        },
    }


def _qwen_cache_rows() -> list[dict]:
    # Qwen schema: emotion_en set directly, emotion_zh absent.
    return [
        {
            "dataset": "daic",
            "subject_id": "300",
            "sample_id": "300_seg_0",
            "audio_path": "/tmp/300_seg_0.wav",
            "emotion_en": "the voice sounds tense and fast-paced",
            "emotion_source": "qwen2audio",
            "emotion_prompt": "Describe the speaker's emotional state ...",
            "qwen2audio_model": "Qwen2-Audio-7B-Instruct",
            "generation_config": {"max_new_tokens": 48, "do_sample": False, "num_beams": 1},
            "caption_ok": True,
        }
    ]


def _row() -> dict:
    return {
        "dataset": "daic",
        "subject_id": "300",
        "sample_id": "300_seg_0",
        "audio_path": "/tmp/300_seg_0.wav",
        "label": 1,
        "label_text": "Depressed",
        "transcript": "hello world",
    }


def test_qwen_cache_injects() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "daic_qwen2audio_en.jsonl"
        with cache_path.open("w", encoding="utf-8") as handle:
            for row in _qwen_cache_rows():
                handle.write(json.dumps(row) + "\n")

        cache = load_emotion_cache(cache_path, caption_field="emotion_en")
        assert cache["300_seg_0"] == "the voice sounds tense and fast-paced"
        example, _ = _base_example_from_row(_row(), _single_config(), 4000, cache, "neutral_fallback")
        assert "Emotional description: the voice sounds tense and fast-paced" in example["prompt_text"], (
            example["prompt_text"]
        )
    print("[ok] Qwen2-Audio cache (emotion_en, no emotion_zh) injects via existing path")


def test_qwen_cache_validates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "daic_qwen2audio_en.jsonl"
        manifest_path = Path(tmp) / "daic_manifest.jsonl"
        with cache_path.open("w", encoding="utf-8") as handle:
            for row in _qwen_cache_rows():
                handle.write(json.dumps(row) + "\n")
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_row()) + "\n")

        # --strict raises a SystemExit if any manifest sample_id lacks emotion_en.
        rc = build_emotion_cache.main(
            ["validate", "--cache", str(cache_path), "--manifest", str(manifest_path), "--strict"]
        )
        assert rc == 0
    print("[ok] Qwen2-Audio cache passes build_emotion_cache validate --strict")


def main() -> int:
    test_qwen_cache_injects()
    test_qwen_cache_validates()
    print("\nAll Qwen2-Audio cache checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
