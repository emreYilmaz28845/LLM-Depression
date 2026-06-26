"""Emotion-augmented prompting support (training/eval side).

This module consumes a *frozen* offline emotion cache and injects natural-language
emotional descriptions into the depression classification prompt. It NEVER
imports SECap or any model: it only reads a text cache keyed by ``sample_id``.
See ``secap_implementation.md`` and the offline extraction drivers under
``src/emotion/`` for how the cache is produced.

Two prompt mechanics are supported (see the plan, §3):

* single-chunk (`sample_mode: chunk`): one caption rendered into the user text via
  the ``{emotion_block}`` template placeholder.
* subject K-chunk (`sample_mode: subject_audio`): per-chunk captions interleaved
  with the ``Audio i:`` placeholders, so each clip is bound to its description.
  Because training re-samples K of N chunks per epoch, the interleaved block is
  re-rendered in ``AudioTextDataset.__getitem__`` to track the sampled chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import get_logger, read_jsonl


LOGGER = get_logger(__name__)

# Fixed neutral sentence injected when a chunk has no reliable caption.
# Kept content-free so the model can learn to ignore it rather than crash.
NEUTRAL_FALLBACK_CAPTION = "No reliable emotional description is available for this segment."

MISSING_POLICY_NEUTRAL = "neutral_fallback"
MISSING_POLICY_DROP = "drop_emotion_line"
MISSING_POLICY_ERROR = "error"
SUPPORTED_MISSING_POLICIES = (
    MISSING_POLICY_NEUTRAL,
    MISSING_POLICY_DROP,
    MISSING_POLICY_ERROR,
)

AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"


def use_emotion(config: dict[str, Any]) -> bool:
    """Whether emotion-augmented prompting is enabled (``data.use_emotion``).

    Emotion is derived from audio, so it requires ``data.use_audio``.
    """
    data_cfg = config.get("data", {})
    enabled = bool(data_cfg.get("use_emotion", False))
    if enabled and not bool(data_cfg.get("use_audio", False)):
        raise ValueError(
            "data.use_emotion=true requires data.use_audio=true "
            "(emotional descriptions are derived from audio)."
        )
    return enabled


def resolve_missing_policy(config: dict[str, Any]) -> str:
    policy = str(config.get("data", {}).get("emotion_on_missing", MISSING_POLICY_NEUTRAL)).strip()
    if policy not in SUPPORTED_MISSING_POLICIES:
        raise ValueError(
            f"Unsupported data.emotion_on_missing={policy!r}. "
            f"Expected one of {', '.join(SUPPORTED_MISSING_POLICIES)}."
        )
    return policy


def load_emotion_cache(path: str | Path, caption_field: str = "emotion_en") -> dict[str, str | None]:
    """Load the frozen emotion cache as ``{sample_id -> caption_field}``.

    The caption field may be ``None`` (for example, translation failed); callers
    apply the configured missing-caption policy. Rows lacking the requested key
    map to ``None`` so they are treated as missing rather than crashing.
    """
    caption_field = str(caption_field or "emotion_en").strip()
    if not caption_field:
        raise ValueError("Emotion caption field must be non-empty.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Emotion cache not found: {path}. Run the offline SECap extraction + "
            f"translation (src/emotion/) before training with data.use_emotion=true."
        )
    mapping: dict[str, str | None] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        caption = row.get(caption_field)
        if isinstance(caption, str):
            caption = caption.strip() or None
        else:
            caption = None
        mapping[str(sample_id)] = caption
    LOGGER.info("Loaded emotion cache %s field=%s (%d rows).", path, caption_field, len(mapping))
    return mapping


def report_cache_coverage(
    cache: dict[str, str | None], sample_ids: list[str]
) -> dict[str, int]:
    """Coverage stats of ``cache`` against the manifest's ``sample_ids``."""
    missing_row = 0
    null_caption = 0
    for sample_id in sample_ids:
        if sample_id not in cache:
            missing_row += 1
        elif cache[sample_id] is None:
            null_caption += 1
    stats = {
        "total": len(sample_ids),
        "missing_row": missing_row,
        "null_caption": null_caption,
        "covered": len(sample_ids) - missing_row - null_caption,
    }
    if missing_row or null_caption:
        LOGGER.warning(
            "Emotion cache coverage: %d/%d covered (%d missing rows, %d null captions).",
            stats["covered"],
            stats["total"],
            missing_row,
            null_caption,
        )
    return stats


def resolve_caption(
    cache: dict[str, str | None] | None,
    sample_id: str,
    policy: str,
) -> str | None:
    """Resolve the caption for ``sample_id`` honoring the missing policy.

    Returns the caption string, or the neutral fallback, or ``None`` (meaning the
    description line should be dropped entirely). ``error`` policy raises.
    """
    caption = cache.get(sample_id) if cache else None
    if caption:
        return caption
    if policy == MISSING_POLICY_ERROR:
        raise ValueError(f"No emotion caption for sample_id={sample_id!r} (policy=error).")
    if policy == MISSING_POLICY_DROP:
        return None
    return NEUTRAL_FALLBACK_CAPTION


def single_chunk_emotion_block(caption: str | None) -> str:
    """Render the ``{emotion_block}`` placeholder for single-chunk prompts.

    Returns an empty string when the caption was dropped, so the line vanishes.
    """
    if not caption:
        return ""
    return f"Emotional description: {caption}\n"


def interleaved_audio_emotion_block(
    captions: list[str | None], audio_placeholder: str = AUDIO_PLACEHOLDER
) -> str:
    """Interleave ``Audio i:`` placeholders with ``Emotional description i:`` lines.

    ``captions[i]`` aligns with audio ``i+1`` (the i-th sampled/baked chunk). A
    ``None`` caption drops only that chunk's description line (drop policy),
    keeping the audio placeholder so the audio-token count still matches the
    number of audio arrays. ``audio_placeholder`` defaults to the Qwen2-Audio
    string; the Qwen3-Omni backend passes its native placeholder so the per-chunk
    blocks tokenize as real special tokens rather than literal text.
    """
    lines: list[str] = []
    for index, caption in enumerate(captions, start=1):
        lines.append(f"Audio {index}: {audio_placeholder}")
        if caption:
            lines.append(f"Emotional description {index}: {caption}")
    return "\n".join(lines)
