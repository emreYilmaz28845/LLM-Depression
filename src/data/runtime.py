from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.eatd import CANONICAL_RESPONSES
from src.data.emotion import (
    interleaved_audio_emotion_block,
    load_emotion_cache,
    report_cache_coverage,
    resolve_caption,
    resolve_missing_policy,
    single_chunk_emotion_block,
    use_emotion,
)
from src.daic_chunking import (
    balanced_joint_bundles,
    evenly_spaced_indices,
    fixed_count_balanced_joint_bundles,
    resolve_chunking_controls,
)
from src.utils import (
    get_logger,
    INPUT_MODALITY_AUDIO_ONLY,
    INPUT_MODALITY_AUDIO_TEXT,
    INPUT_MODALITY_TEXT_ONLY,
    MODEL_BACKEND_QWEN3OMNI,
    internal_label_text_from_int,
    prompt_label_descriptor,
    prompt_label_instruction,
    read_jsonl,
    resolve_input_modality,
    resolve_model_backend,
    save_json,
    write_jsonl,
)


LOGGER = get_logger(__name__)
# Qwen2-Audio uses <|audio_bos|><|AUDIO|><|audio_eos|>; Qwen3-Omni uses distinct
# single special tokens <|audio_start|><|audio_pad|><|audio_end|> (verified via the
# smoke gate, QWEN3_OMNI_IMPLEMENTATION.md §5.1). Reusing the wrong string tokenizes
# as literal text and silently misaligns audio features, so the placeholder must
# follow the resolved model_backend. AUDIO_PLACEHOLDER stays the Qwen2-Audio default
# for backward compatibility; resolve_audio_placeholder() picks per backend.
QWEN2AUDIO_AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
QWEN3OMNI_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
AUDIO_PLACEHOLDER = QWEN2AUDIO_AUDIO_PLACEHOLDER

JOINT_PACKED30_MODE = "participant_speech_packed30_joint"
JOINT_PACKED30_RECIPE_ID = "runtime_packed30_joint_random_k4_fullcover_v1"
JOINT_PACKED30_MAX_CHUNK_SAMPLES = 480000
JOINT_PACKED30_SAMPLE_RATE = 16000
JOINT_PACKED30_REQUIRED_K = 4
JOINT_PACKED30_CONTEXT_SENTINEL = "__JOINT_BUNDLE_AUDIO_CONTEXT__"


def resolve_audio_placeholder(config: dict[str, Any]) -> str:
    if resolve_model_backend(config) == MODEL_BACKEND_QWEN3OMNI:
        return QWEN3OMNI_AUDIO_PLACEHOLDER
    return AUDIO_PLACEHOLDER
DEFAULT_SINGLE_AUDIO_CONTEXT = "The subject's speech audio is provided."
DEFAULT_SUBJECT_AUDIO_CONTEXT = "The subject's speech audio is provided in three responses: negative, neutral, and positive."


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def build_subject_label_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in rows:
        mapping[row["subject_id"]] = int(row["label"])
    return mapping


def filter_rows_by_subjects(rows: list[dict[str, Any]], subject_ids: list[str]) -> list[dict[str, Any]]:
    subject_set = set(subject_ids)
    return [row for row in rows if row["subject_id"] in subject_set]


def _audio_prompt_block(num_audios: int, audio_placeholder: str = AUDIO_PLACEHOLDER) -> str:
    return "\n".join(f"Audio {index}: {audio_placeholder}" for index in range(1, num_audios + 1))


def _modality_flags(input_modality: str) -> tuple[bool, bool]:
    return (
        input_modality in {INPUT_MODALITY_AUDIO_TEXT, INPUT_MODALITY_AUDIO_ONLY},
        input_modality in {INPUT_MODALITY_AUDIO_TEXT, INPUT_MODALITY_TEXT_ONLY},
    )


def _decision_basis(input_modality: str) -> str:
    if input_modality == INPUT_MODALITY_AUDIO_TEXT:
        return "audio and transcript"
    if input_modality == INPUT_MODALITY_AUDIO_ONLY:
        return "audio"
    if input_modality == INPUT_MODALITY_TEXT_ONLY:
        return "transcript"
    raise ValueError(f"Unsupported input modality: {input_modality}")


def _audio_context_block(use_audio: bool, is_subject_bundle: bool) -> str:
    if not use_audio:
        return ""
    if is_subject_bundle:
        return DEFAULT_SUBJECT_AUDIO_CONTEXT
    return DEFAULT_SINGLE_AUDIO_CONTEXT


def _transcript_block(use_text: bool, transcript: str) -> str:
    if not use_text:
        return ""
    return f"The transcript of the subject's speech is:\n{transcript}\n\n"


def _user_prompt_template(config: dict[str, Any], is_subject_bundle: bool) -> str:
    prompt_cfg = config.get("prompt", {})
    if is_subject_bundle:
        template = str(prompt_cfg.get("subject_user_template", "")).strip()
        if template:
            return template
    template = str(prompt_cfg.get("user_template", "")).strip()
    if template:
        return template
    if is_subject_bundle:
        raise ValueError("Missing prompt.user_template or prompt.subject_user_template in config.")
    raise ValueError("Missing prompt.user_template in config.")


def render_user_prompt_text(
    config: dict[str, Any],
    transcript: str,
    *,
    is_subject_bundle: bool = False,
    audio_context_override: str | None = None,
    emotion_block: str = "",
) -> str:
    input_modality = resolve_input_modality(config)
    use_audio, use_text = _modality_flags(input_modality)
    template = _user_prompt_template(config, is_subject_bundle)
    audio_context_block = (
        audio_context_override
        if audio_context_override is not None
        else _audio_context_block(use_audio, is_subject_bundle)
    )
    placeholder_values = {
        "transcript": transcript,
        "audio_context_block": audio_context_block,
        "transcript_block": _transcript_block(use_text, transcript),
        "decision_basis": _decision_basis(input_modality),
        "label_descriptor": prompt_label_descriptor(config),
        "label_instruction": prompt_label_instruction(config),
        "emotion_block": emotion_block,
    }
    try:
        return template.format_map(placeholder_values).strip()
    except KeyError as exc:
        available = ", ".join(sorted(placeholder_values))
        raise ValueError(
            f"Unknown placeholder {exc.args[0]!r} in prompt template. "
            f"Available placeholders: {available}."
        ) from exc


def build_prompt_text(
    system_prompt: str,
    user_text: str,
    num_audios: int,
    use_audio: bool,
    emotion_captions: list[str | None] | None = None,
    audio_placeholder: str = AUDIO_PLACEHOLDER,
) -> str:
    if use_audio and num_audios:
        if emotion_captions is not None:
            # Interleave Audio i: with Emotional description i: (subject mode).
            if len(emotion_captions) != num_audios:
                raise ValueError(
                    f"emotion_captions length ({len(emotion_captions)}) must match "
                    f"num_audios ({num_audios})."
                )
            audio_block = interleaved_audio_emotion_block(
                emotion_captions, audio_placeholder=audio_placeholder
            )
        else:
            audio_block = _audio_prompt_block(num_audios, audio_placeholder=audio_placeholder)
        user_text = audio_block + "\n" + user_text
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_training_text(prompt_text: str, label_text: str) -> str:
    return f"{prompt_text}{label_text}<|im_end|>\n"


def _truncate_text(text: str, max_chars: int) -> tuple[str, dict[str, Any] | None]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, None
    return text[:max_chars], {
        "transcript_original_chars": len(text),
        "transcript_kept_chars": max_chars,
        "transcript_truncated": True,
    }


def _resolve_subject_transcript(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    mode_name: str,
) -> str:
    transcript_values = [str(row["transcript"]).strip() for row in rows]
    multi_transcript = str(config.get("data", {}).get("multi_transcript", "strict")).strip().lower()
    if multi_transcript == "concat":
        separator = str(config.get("data", {}).get("multi_transcript_separator", "\n"))
        return separator.join(value for value in transcript_values if value)
    if multi_transcript != "strict":
        raise ValueError(
            f"Unsupported data.multi_transcript={multi_transcript!r}. Expected 'strict' or 'concat'."
        )
    unique_values = set(transcript_values)
    if len(unique_values) != 1:
        raise ValueError(
            f"{mode_name} expects exactly one transcript per subject unless "
            f"data.multi_transcript=concat. Found {len(unique_values)} transcripts "
            f"for subject_id={rows[0]['subject_id']}."
        )
    return transcript_values[0]


def _ordered_subject_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows and str(rows[0].get("dataset", "")).lower() == "androids_interview":
        return sorted(
            rows,
            key=lambda row: (
                int(row.get("turn_id", 0)),
                int(row.get("window_index", row.get("segment_index", 0))),
            ),
        )
    if rows and str(rows[0].get("dataset", "")).lower() == "d3tec":
        return sorted(
            rows,
            key=lambda row: (
                int(row.get("prompt_id", 0)),
                int(row.get("segment_index", 0)),
            ),
        )
    if rows and str(rows[0].get("dataset", "")).lower() == "turkish":
        def turkish_key(row: dict[str, Any]) -> tuple[int, str]:
            chunk_id = str(row.get("chunk_id", "")).strip()
            return (int(chunk_id) if chunk_id.isdigit() else 10**9, str(row["sample_id"]))

        return sorted(rows, key=turkish_key)
    if rows and str(rows[0].get("dataset", "")).lower() == "daic":
        def daic_key(row: dict[str, Any]) -> tuple[int, int, str]:
            raw_chunk_id = str(row.get("chunk_id", "")).strip()
            match = re.search(r"(\d+)$", raw_chunk_id)
            return (
                0 if match else 1,
                int(match.group(1)) if match else 10**9,
                str(row["sample_id"]),
            )

        return sorted(rows, key=daic_key)
    return sorted(rows, key=lambda item: item["sample_id"])


def _subject_mode_audio_plan(audio_paths: list[str], max_per_response: float, max_total: float) -> dict[str, Any]:
    durations = []
    for audio_path in audio_paths:
        info = sf.info(audio_path)
        durations.append(float(info.frames / info.samplerate))
    capped = [min(duration, max_per_response) for duration in durations]
    total = sum(capped)
    if total > max_total:
        scale = max_total / total if total else 1.0
        kept = [value * scale for value in capped]
    else:
        kept = capped
    return {
        "audio_original_seconds": durations,
        "audio_kept_seconds": kept,
        "audio_total_original_seconds": sum(durations),
        "audio_total_kept_seconds": sum(kept),
        "audio_truncated": any(abs(source - target) > 1e-6 for source, target in zip(durations, kept)),
    }


def _base_example_from_row(
    row: dict[str, Any],
    config: dict[str, Any],
    transcript_max_chars: int,
    emotion_cache: dict[str, str | None] | None = None,
    emotion_policy: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_modality = resolve_input_modality(config)
    use_audio, use_text = _modality_flags(input_modality)
    transcript = row["transcript"] if use_text else ""
    if use_text and str(row.get("dataset", "")).lower() == "androids_interview":
        scope = str(
            config.get("data", {}).get(
                "audio_text_transcript_scope", "segment_aligned"
            )
        ).strip().lower()
        if scope == "segment_aligned":
            transcript = str(row["segment_transcript"])
        elif scope == "full_turn":
            transcript = str(row["full_turn_transcript"])
        else:
            raise ValueError(
                "Unsupported data.audio_text_transcript_scope="
                f"{scope!r}. Expected 'segment_aligned' or 'full_turn'."
            )
    transcript, transcript_log = _truncate_text(transcript, transcript_max_chars)
    emotion_block = ""
    if emotion_cache is not None and use_audio:
        caption = resolve_caption(emotion_cache, str(row["sample_id"]), emotion_policy)
        emotion_block = single_chunk_emotion_block(caption)
    user_text = render_user_prompt_text(
        config, transcript, is_subject_bundle=False, emotion_block=emotion_block
    )
    example_internal_label = row.get("internal_label_text") or internal_label_text_from_int(config, int(row["label"]))
    prompt_text = build_prompt_text(
        system_prompt=config["prompt"]["system"],
        user_text=user_text,
        num_audios=1 if use_audio else 0,
        use_audio=use_audio,
        audio_placeholder=resolve_audio_placeholder(config),
    )
    example = {
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "sample_id": row["sample_id"],
        "label": int(row["label"]),
        "label_text": row["label_text"],
        "internal_label_text": example_internal_label,
        "transcript": transcript,
        "audio_paths": [row["audio_path"]] if use_audio else [],
        "audio_clip_seconds": [None] if use_audio else [],
        "audio_start_times": [float(row.get("start_time", 0.0) or 0.0)] if use_audio else [],
        "audio_end_times": (
            [float(row["end_time"]) if row.get("end_time") not in (None, "") else None]
            if use_audio
            else []
        ),
        "input_modality": input_modality,
        "prompt_text": prompt_text,
        "training_text": build_training_text(prompt_text, example_internal_label),
        "question_id": row.get("question_id", ""),
        "response_id": row.get("response_id", ""),
        "prompt_id": row.get("prompt_id", row.get("question_id", "")),
        "segment_index": row.get("segment_index", 0),
        "num_segments": row.get("num_segments", 1),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "segment_duration": row.get("segment_duration", ""),
    }
    return example, transcript_log


def _build_subject_level_text_only_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    transcript_max_chars: int,
    truncation_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[row["subject_id"]].append(row)

    examples: list[dict[str, Any]] = []
    truncation_logs: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = _ordered_subject_rows(grouped[subject_id])
        label_values = {int(row["label"]) for row in rows}
        if len(label_values) != 1:
            raise ValueError(
                f"Text-only subject mode expects exactly one label per subject. "
                f"Found {len(label_values)} labels for subject_id={subject_id}."
            )

        canonical_row = rows[0]
        dataset_name = str(canonical_row.get("dataset", "")).lower()
        if dataset_name == "androids_interview":
            turn_rows: dict[int, dict[str, Any]] = {}
            for row in rows:
                turn_id = int(row["turn_id"])
                prior = turn_rows.setdefault(turn_id, row)
                if (
                    str(prior["full_turn_transcript"]).strip()
                    != str(row["full_turn_transcript"]).strip()
                ):
                    raise ValueError(
                        "Inconsistent ANDROIDS full transcript within "
                        f"response_id={row['response_id']}."
                    )
                if str(prior["response_id"]) != str(row["response_id"]):
                    raise ValueError(
                        f"ANDROIDS turn_id={turn_id} maps to multiple parent turns "
                        f"for subject_id={subject_id}."
                    )
            subject_transcript = "\n\n".join(
                f"[Turn {turn_id}]\n"
                f"{str(turn_rows[turn_id]['full_turn_transcript']).strip()}"
                for turn_id in sorted(turn_rows)
            )
        elif dataset_name == "d3tec":
            response_rows: dict[int, dict[str, Any]] = {}
            for row in rows:
                prompt_id = int(row["prompt_id"])
                prior = response_rows.setdefault(prompt_id, row)
                if str(prior["full_response_transcript"]).strip() != str(row["full_response_transcript"]).strip():
                    raise ValueError(
                        f"Inconsistent D3TEC full transcript within response_id={row['response_id']}."
                    )
            expected_prompts = set(range(27))
            if set(response_rows) != expected_prompts:
                raise ValueError(
                    f"D3TEC text-only mode requires prompts 0-26 for {subject_id}; "
                    f"found={sorted(response_rows)}"
                )
            subject_transcript = "\n\n".join(
                f"[Response {prompt_id}]\n{str(response_rows[prompt_id]['full_response_transcript']).strip()}"
                for prompt_id in range(27)
            )
        else:
            subject_transcript = _resolve_subject_transcript(
                rows,
                config,
                mode_name="Text-only subject mode",
            )
        transcript, transcript_log = _truncate_text(subject_transcript, transcript_max_chars)
        user_text = render_user_prompt_text(config, transcript, is_subject_bundle=False)
        internal_label_text = canonical_row.get("internal_label_text") or internal_label_text_from_int(
            config,
            int(canonical_row["label"]),
        )
        prompt_text = build_prompt_text(
            system_prompt=config["prompt"]["system"],
            user_text=user_text,
            num_audios=0,
            use_audio=False,
        )
        examples.append(
            {
                "dataset": canonical_row["dataset"],
                "subject_id": subject_id,
                "sample_id": subject_id,
                "label": int(canonical_row["label"]),
                "label_text": canonical_row["label_text"],
                "internal_label_text": internal_label_text,
                "transcript": transcript,
                "audio_paths": [],
                "audio_clip_seconds": [],
                "audio_start_times": [],
                "audio_end_times": [],
                "input_modality": INPUT_MODALITY_TEXT_ONLY,
                "prompt_text": prompt_text,
                "training_text": build_training_text(prompt_text, internal_label_text),
                "question_id": canonical_row.get("question_id", ""),
                "protocol_id": canonical_row.get("protocol_id", ""),
            }
        )
        if transcript_log:
            truncation_logs.append(
                {
                    "partition": partition_name,
                    "subject_id": subject_id,
                    "sample_id": subject_id,
                    **transcript_log,
                }
            )

    if truncation_log_path:
        write_jsonl(truncation_logs, truncation_log_path)
    return examples


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Deterministic, reproducible selection of `count` indices spread across `total`.

    Used to pick the held-out evaluation view of a subject's chunks so that
    validation/test never depend on random sampling.
    """
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / (count - 1)
    indices = [int(round(i * step)) for i in range(count)]
    # Guard against rounding collisions at the boundaries.
    deduped: list[int] = []
    for idx in indices:
        if idx not in deduped:
            deduped.append(idx)
    fallback = 0
    while len(deduped) < count and fallback < total:
        if fallback not in deduped:
            deduped.append(fallback)
        fallback += 1
    return sorted(deduped)


def _build_subject_level_audio_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    transcript_max_chars: int,
    truncation_log_path: str | Path | None = None,
    emotion_cache: dict[str, str | None] | None = None,
    emotion_policy: str | None = None,
) -> list[dict[str, Any]]:
    """One example per subject with a FIXED number of audio chunks (K).

    The example carries both:
      - ``audio_paths``: a deterministic evenly-spaced K-chunk view used verbatim
        by the (deterministic) evaluation path, and
      - ``subject_chunk_paths``: the subject's full chunk list, from which the
        training dataset samples K chunks stochastically per epoch.

    ``chunks_per_subject`` is fixed per subject so the number of ``<|AUDIO|>``
    placeholders in the prompt always matches the number of audio arrays, whether
    the view is the deterministic eval view or a random training view.
    """
    input_modality = resolve_input_modality(config)
    use_audio, use_text = _modality_flags(input_modality)
    if not use_audio:
        raise ValueError("sample_mode=subject_audio requires data.use_audio=true.")
    data_cfg = config["data"]
    controls = resolve_chunking_controls(config)
    configured_k = (
        controls["train_chunks_per_subject"]
        if "train" in partition_name.lower()
        else controls["eval_chunks_per_subject"]
    )
    chunks_per_subject = (
        10**9 if configured_k == "all" else int(configured_k)
    )
    if chunks_per_subject < 1:
        raise ValueError("data.chunks_per_subject must be >= 1 for subject_audio mode.")
    raw_cap = data_cfg.get("max_audio_seconds_per_chunk", 30.0)
    max_seconds_per_chunk = float(raw_cap) if raw_cap else None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[row["subject_id"]].append(row)

    examples: list[dict[str, Any]] = []
    truncation_logs: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = _ordered_subject_rows(grouped[subject_id])
        label_values = {int(row["label"]) for row in rows}
        if len(label_values) != 1:
            raise ValueError(
                f"subject_audio mode expects exactly one label per subject. "
                f"Found {len(label_values)} labels for subject_id={subject_id}."
            )
        canonical_row = rows[0]
        chunk_paths = [row["audio_path"] for row in rows]
        chunk_ids = [str(row.get("chunk_id", row["sample_id"])) for row in rows]
        if (
            controls["train_chunk_policy"] == "fixed_k"
            and "train" in partition_name.lower()
            and configured_k != "all"
            and len(chunk_paths) < int(configured_k)
        ):
            raise ValueError(
                f"fixed_k requires at least {int(configured_k)} chunks for subject_id={subject_id}; "
                f"found {len(chunk_paths)}."
            )
        emotion_on = emotion_cache is not None
        chunk_caption_by_path: dict[str, str | None] = {}
        if emotion_on:
            for row in rows:
                chunk_caption_by_path[row["audio_path"]] = resolve_caption(
                    emotion_cache, str(row["sample_id"]), emotion_policy
                )

        transcript = ""
        transcript_log: dict[str, Any] | None = None
        if use_text:
            subject_transcript = _resolve_subject_transcript(
                rows,
                config,
                mode_name="subject_audio mode with use_text",
            )
            transcript, transcript_log = _truncate_text(subject_transcript, transcript_max_chars)

        effective_k = min(chunks_per_subject, len(chunk_paths))
        deterministic_indices = _evenly_spaced_indices(len(chunk_paths), effective_k)
        deterministic_paths = [chunk_paths[index] for index in deterministic_indices]
        clip_seconds = [max_seconds_per_chunk] * effective_k

        count_text_mode = str(data_cfg.get("subject_audio_count_text", "explicit")).strip().lower()
        if count_text_mode == "explicit":
            audio_context = (
                f"The subject's speech audio is provided in {effective_k} "
                f"segment{'s' if effective_k != 1 else ''} sampled from the interview."
            )
        elif count_text_mode == "neutral":
            audio_context = (
                "The subject's speech audio is provided in multiple segments sampled "
                "from the interview."
            )
        else:
            raise ValueError(
                "data.subject_audio_count_text must be 'explicit' or 'neutral'."
            )
        user_text = render_user_prompt_text(
            config,
            transcript,
            is_subject_bundle=True,
            audio_context_override=audio_context,
        )
        internal_label_text = canonical_row.get("internal_label_text") or internal_label_text_from_int(
            config,
            int(canonical_row["label"]),
        )
        deterministic_captions = (
            [chunk_caption_by_path[path] for path in deterministic_paths] if emotion_on else None
        )
        audio_placeholder = resolve_audio_placeholder(config)
        prompt_text = build_prompt_text(
            system_prompt=config["prompt"]["system"],
            user_text=user_text,
            num_audios=effective_k,
            use_audio=True,
            emotion_captions=deterministic_captions,
            audio_placeholder=audio_placeholder,
        )
        example = {
            "dataset": canonical_row["dataset"],
            "subject_id": subject_id,
            "sample_id": subject_id,
            "label": int(canonical_row["label"]),
            "label_text": canonical_row["label_text"],
            "internal_label_text": internal_label_text,
            "transcript": transcript,
            "audio_paths": deterministic_paths,
            "audio_clip_seconds": clip_seconds,
            "subject_chunk_paths": chunk_paths,
            "subject_chunk_ids": chunk_ids,
            "subject_chunk_clip_seconds": [max_seconds_per_chunk] * len(chunk_paths),
            "chunks_per_subject": effective_k,
            "input_modality": input_modality,
            "prompt_text": prompt_text,
            "training_text": build_training_text(prompt_text, internal_label_text),
            "question_id": "subject_audio_bundle",
        }
        if emotion_on:
            # Carry everything the training dataset needs to re-render the
            # interleaved prompt after it randomly re-samples K chunks per epoch,
            # so each Emotional description i tracks the audio actually fed (§4.3).
            example["chunk_caption_by_path"] = chunk_caption_by_path
            example["emotion_user_text"] = user_text
            example["emotion_system_prompt"] = config["prompt"]["system"]
            example["emotion_internal_label_text"] = internal_label_text
            example["emotion_audio_placeholder"] = audio_placeholder
        if (
            controls["eval_chunk_policy"] in {"balanced_joint_cover", "fixed_count_balanced_joint_cover"}
            and "train" not in partition_name.lower()
        ):
            if controls["eval_chunk_policy"] == "fixed_count_balanced_joint_cover":
                bundles, coverage = fixed_count_balanced_joint_bundles(
                    chunk_ids, effective_k, controls["eval_bundles_per_subject"]
                )
            else:
                bundles, coverage = balanced_joint_bundles(chunk_ids, effective_k)
            for bundle_id, indices in enumerate(bundles):
                bundle = dict(example)
                bundle_paths = [chunk_paths[index] for index in indices]
                bundle["sample_id"] = f"{subject_id}__bundle_{bundle_id:03d}"
                bundle["audio_paths"] = bundle_paths
                bundle["audio_clip_seconds"] = [max_seconds_per_chunk] * len(indices)
                bundle["bundle_id"] = bundle_id
                bundle["bundle_chunk_ids"] = [chunk_ids[index] for index in indices]
                bundle["bundle_coverage_count"] = coverage["occurrences_per_chunk"]
                if emotion_on:
                    captions = [
                        bundle["chunk_caption_by_path"].get(path)
                        for path in bundle_paths
                    ]
                    bundle_prompt = build_prompt_text(
                        system_prompt=bundle["emotion_system_prompt"],
                        user_text=bundle["emotion_user_text"],
                        num_audios=len(bundle_paths),
                        use_audio=True,
                        emotion_captions=captions,
                        audio_placeholder=bundle.get(
                            "emotion_audio_placeholder", AUDIO_PLACEHOLDER
                        ),
                    )
                    bundle["prompt_text"] = bundle_prompt
                    bundle["training_text"] = build_training_text(
                        bundle_prompt, bundle["emotion_internal_label_text"]
                    )
                examples.append(bundle)
        else:
            examples.append(example)
        log_row = {
            "partition": partition_name,
            "subject_id": subject_id,
            "sample_id": subject_id,
            "num_chunks_available": len(chunk_paths),
            "chunks_per_subject": effective_k,
            "deterministic_eval_chunk_indices": deterministic_indices,
            "max_audio_seconds_per_chunk": max_seconds_per_chunk,
        }
        if transcript_log:
            log_row.update(transcript_log)
        truncation_logs.append(log_row)

    if truncation_log_path:
        write_jsonl(truncation_logs, truncation_log_path)
    return examples


def _build_participant_speech_packed30_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    truncation_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """One example per packed30 chunk (audio-only and audio+text).

    Each example carries ordered ``audio_spans`` over the source WAV plus the
    locked full participant transcript; the runtime multi-span loader
    reconstructs exactly one waveform of at most 480000 samples.
    """
    input_modality = resolve_input_modality(config)
    use_audio, use_text = _modality_flags(input_modality)
    if not use_audio:
        raise ValueError(
            "sample_mode=participant_speech_packed30 requires data.use_audio=true."
        )
    audio_placeholder = resolve_audio_placeholder(config)
    examples: list[dict[str, Any]] = []
    truncation_logs: list[dict[str, Any]] = []
    ordered_rows = sorted(
        manifest_rows,
        key=lambda row: (int(row["subject_id"]), int(row["chunk_index"]), str(row["sample_id"])),
    )
    for row in ordered_rows:
        label = int(row["label"])
        internal_label_text = internal_label_text_from_int(config, label)
        transcript = str(row["full_participant_transcript"]) if use_text else ""
        user_text = render_user_prompt_text(config, transcript, is_subject_bundle=False)
        prompt_text = build_prompt_text(
            system_prompt=config["prompt"]["system"],
            user_text=user_text,
            num_audios=1,
            use_audio=True,
            audio_placeholder=audio_placeholder,
        )
        examples.append(
            {
                "dataset": "daic",
                "subject_id": row["subject_id"],
                "sample_id": row["sample_id"],
                "label": label,
                "label_text": row["label_text"],
                "internal_label_text": internal_label_text,
                "transcript": transcript,
                "audio_path": row["audio_path"],
                "audio_spans": list(row["audio_spans"]),
                "participant_sample_count": int(row["participant_sample_count"]),
                "audio_paths": [],
                "audio_clip_seconds": [],
                "audio_start_times": [],
                "audio_end_times": [],
                "protocol_id": row["protocol_id"],
                "chunk_id": f"{int(row['chunk_index']):03d}",
                "chunk_index": int(row["chunk_index"]),
                "num_chunks": int(row["num_chunks"]),
                "chunk_transcript": str(row["chunk_transcript"]),
                "full_participant_transcript_sha256": str(
                    row["full_participant_transcript_sha256"]
                ),
                "input_modality": input_modality,
                "prompt_text": prompt_text,
                "training_text": build_training_text(prompt_text, internal_label_text),
                "question_id": "",
            }
        )
        if len(transcript) > int(config["data"].get("transcript_max_chars", 4000) or 4000):
            truncation_logs.append(
                {
                    "partition": partition_name,
                    "subject_id": row["subject_id"],
                    "sample_id": row["sample_id"],
                    "transcript_original_chars": len(str(row["full_participant_transcript"])),
                    "transcript_kept_chars": len(transcript),
                    "transcript_truncated": True,
                }
            )
    if truncation_log_path:
        write_jsonl(truncation_logs, truncation_log_path)
    return examples


def _bundle_audio_context(num_segments: int) -> str:
    return (
        f"The subject's speech audio is provided in {int(num_segments)} "
        f"segment{'s' if int(num_segments) != 1 else ''} sampled from the interview."
    )


def render_joint_packed30_bundle(
    example: dict[str, Any], bundle_size: int
) -> tuple[str, str]:
    """Render the joint packed30 prompt/training text for a bundle of ``bundle_size``
    span groups.

    The prompt must contain exactly ``bundle_size`` audio placeholders and may
    describe the current bundle size (never the subject's total chunk count).
    Training (epoch schedules), evaluation (balanced-cover bundles), and hidden
    extraction all route through this single renderer.

    ``data.audio_text_transcript_scope`` selects the transcript source:
    ``full_participant`` (default) renders the locked full participant
    transcript once; ``chunk_aligned`` joins the current bundle's own
    ``chunk_transcript`` texts, forcing the audio chunks and the text to refer
    to the same spans.
    """
    bundle_size = int(bundle_size)
    if bundle_size < 1:
        raise ValueError("A joint packed30 bundle must contain at least one span group.")
    template = str(example["prompt_user_template"])
    values = dict(example["prompt_template_values"])
    values["audio_context_block"] = _bundle_audio_context(bundle_size)
    scope = str(example.get("prompt_transcript_scope") or "full_participant").strip().lower()
    if scope == "chunk_aligned":
        chunk_texts = [
            str(group.get("chunk_transcript", "")).strip()
            for group in (example.get("audio_span_groups") or [])
        ]
        joined = "\n\n".join(text for text in chunk_texts if text)
        max_chars = int(example.get("prompt_transcript_max_chars") or 0)
        joined, _ = _truncate_text(joined, max_chars)
        values["transcript"] = joined
        values["transcript_block"] = _transcript_block(True, joined)
    user_text = template.format_map(values).strip()
    prompt_text = build_prompt_text(
        system_prompt=example["prompt_system"],
        user_text=user_text,
        num_audios=bundle_size,
        use_audio=True,
        audio_placeholder=example["prompt_audio_placeholder"],
    )
    return prompt_text, build_training_text(prompt_text, example["prompt_internal_label_text"])


def load_span_group_audio_arrays(
    example: dict[str, Any],
    sampling_rate: int,
    silence_audio: bool,
) -> list[np.ndarray]:
    """Shared loader converting every ``audio_span_groups`` member into exactly one
    waveform through the multi-span packed30 loader.

    Enforces the locked invariant: number of audio placeholders in the prompt
    == len(audio_span_groups) == number of waveforms passed to the processor.
    """
    groups = list(example.get("audio_span_groups") or [])
    if not groups:
        raise ValueError("load_span_group_audio_arrays requires at least one span group.")
    if sampling_rate is None:
        raise ValueError("Audio examples require a processor sampling rate.")
    placeholder = str(example.get("prompt_audio_placeholder", AUDIO_PLACEHOLDER))
    prompt_placeholders = str(example.get("prompt_text", "")).count(placeholder)
    if prompt_placeholders != len(groups):
        raise ValueError(
            f"Joint packed30 placeholder/group mismatch for sample_id="
            f"{example.get('sample_id', '')}: prompt has {prompt_placeholders} "
            f"audio placeholders but {len(groups)} span groups."
        )
    arrays = []
    for group in groups:
        array = load_audio_spans_array(
            str(group["audio_path"]),
            list(group["audio_spans"]),
            sampling_rate,
            silence_audio,
            int(group.get("participant_sample_count")) if group.get("participant_sample_count") is not None else None,
        )
        arrays.append(array)
    return arrays


def _build_participant_speech_packed30_joint_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    truncation_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """One source example per subject plus deterministic balanced-cover bundles
    for validation/test.

    Training carries the complete ordered ``subject_chunk_span_groups`` pool;
    the joint epoch scheduler materializes one random K=4 bundle per subject per
    epoch and re-renders the prompt for the current bundle size. Validation and
    test use the existing cyclic ``balanced_joint_bundles`` algorithm with
    K=min(4, N); every chunk appears and every chunk within a subject has
    exactly equal occurrence count.
    """
    input_modality = resolve_input_modality(config)
    use_audio, use_text = _modality_flags(input_modality)
    if not use_audio:
        raise ValueError(
            f"sample_mode={JOINT_PACKED30_MODE} requires data.use_audio=true."
        )
    data_cfg = config["data"]
    requested_k = int(data_cfg.get("train_chunks_per_subject", JOINT_PACKED30_REQUIRED_K))
    eval_k = int(data_cfg.get("eval_chunks_per_subject", JOINT_PACKED30_REQUIRED_K))
    transcript_max_chars = int(data_cfg.get("transcript_max_chars", 0) or 0)
    transcript_scope = str(
        data_cfg.get("audio_text_transcript_scope", "full_participant")
    ).strip().lower()
    if transcript_scope not in ("full_participant", "chunk_aligned"):
        raise ValueError(
            f"Unsupported data.audio_text_transcript_scope={transcript_scope!r} "
            "for sample_mode=participant_speech_packed30_joint. "
            "Expected 'full_participant' or 'chunk_aligned'."
        )
    audio_placeholder = resolve_audio_placeholder(config)
    prompt_system = str(config["prompt"]["system"])
    user_template = _user_prompt_template(config, is_subject_bundle=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["subject_id"])].append(row)

    examples: list[dict[str, Any]] = []
    truncation_logs: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = sorted(
            grouped[subject_id],
            key=lambda row: (int(row["subject_id"]), int(row["chunk_index"]), str(row["sample_id"])),
        )
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(
                f"{JOINT_PACKED30_MODE} expects exactly one label per subject; "
                f"found {len(labels)} for subject_id={subject_id}."
            )
        official_splits = {str(row.get("split_original", "")) for row in rows}
        if len(official_splits) != 1:
            raise ValueError(
                f"{JOINT_PACKED30_MODE} expects exactly one official split per "
                f"subject; found {sorted(official_splits)} for subject_id={subject_id}."
            )
        transcripts = {str(row["full_participant_transcript"]).strip() for row in rows}
        hashes = {str(row["full_participant_transcript_sha256"]) for row in rows}
        if len(transcripts) != 1 or len(hashes) != 1:
            raise ValueError(
                f"{JOINT_PACKED30_MODE} expects one consistent full participant "
                f"transcript and hash per subject; subject_id={subject_id}."
            )
        chunk_indices = [int(row["chunk_index"]) for row in rows]
        if chunk_indices != list(range(len(chunk_indices))):
            raise ValueError(
                f"{JOINT_PACKED30_MODE} requires unique consecutive integer chunk "
                f"indices for subject_id={subject_id}; found {chunk_indices}."
            )
        for row in rows:
            sample_count = int(row.get("participant_sample_count", 0))
            if not (1 <= sample_count <= JOINT_PACKED30_MAX_CHUNK_SAMPLES):
                raise ValueError(
                    f"{JOINT_PACKED30_MODE} requires 1 <= participant_sample_count "
                    f"<= {JOINT_PACKED30_MAX_CHUNK_SAMPLES}; subject_id={subject_id} "
                    f"sample_id={row['sample_id']} got {sample_count}."
                )
        groups = [
            {
                "audio_path": str(row["audio_path"]),
                "audio_spans": list(row["audio_spans"]),
                "participant_sample_count": int(row["participant_sample_count"]),
                "chunk_id": f"{int(row['chunk_index']):03d}",
                "chunk_index": int(row["chunk_index"]),
                "chunk_transcript": str(row["chunk_transcript"]),
            }
            for row in rows
        ]
        canonical_row = rows[0]
        label = int(canonical_row["label"])
        internal_label_text = internal_label_text_from_int(config, label)
        transcript = str(canonical_row["full_participant_transcript"]) if use_text else ""
        transcript, transcript_log = _truncate_text(transcript, transcript_max_chars)
        values = {
            "transcript": transcript,
            "audio_context_block": JOINT_PACKED30_CONTEXT_SENTINEL,
            "transcript_block": _transcript_block(use_text, transcript),
            "decision_basis": _decision_basis(input_modality),
            "label_descriptor": prompt_label_descriptor(config),
            "label_instruction": prompt_label_instruction(config),
            "emotion_block": "",
        }
        deterministic_k = min(requested_k, len(groups))
        deterministic_indices = _evenly_spaced_indices(len(groups), deterministic_k)
        deterministic_groups = [groups[index] for index in deterministic_indices]
        source = {
            "dataset": "daic",
            "subject_id": subject_id,
            "sample_id": subject_id,
            "label": label,
            "label_text": canonical_row["label_text"],
            "internal_label_text": internal_label_text,
            "transcript": transcript,
            "full_participant_transcript_sha256": str(
                canonical_row["full_participant_transcript_sha256"]
            ),
            "audio_span_groups": deterministic_groups,
            "subject_chunk_span_groups": groups,
            "audio_paths": [],
            "audio_clip_seconds": [],
            "audio_start_times": [],
            "audio_end_times": [],
            "protocol_id": canonical_row["protocol_id"],
            "num_chunks": len(groups),
            "chunk_index": 0,
            "input_modality": input_modality,
            "prompt_user_template": user_template,
            "prompt_template_values": values,
            "prompt_system": prompt_system,
            "prompt_audio_placeholder": audio_placeholder,
            "prompt_internal_label_text": internal_label_text,
            "prompt_transcript_scope": transcript_scope,
            "prompt_transcript_max_chars": transcript_max_chars,
            "question_id": "",
        }
        source["prompt_text"], source["training_text"] = render_joint_packed30_bundle(
            source, len(deterministic_groups)
        )
        if transcript_log:
            truncation_logs.append(
                {
                    "partition": partition_name,
                    "subject_id": subject_id,
                    "sample_id": subject_id,
                    **transcript_log,
                }
            )

        if "train" in partition_name.lower():
            examples.append(source)
            continue

        chunk_ids = [group["chunk_id"] for group in groups]
        effective_eval_k = min(eval_k, len(groups))
        bundles, coverage = balanced_joint_bundles(chunk_ids, effective_eval_k)
        for bundle_id, indices in enumerate(bundles):
            bundle = dict(source)
            bundle_groups = [groups[index] for index in indices]
            bundle["sample_id"] = f"{subject_id}__bundle_{bundle_id:03d}"
            bundle["audio_span_groups"] = bundle_groups
            bundle["bundle_id"] = bundle_id
            bundle["bundle_chunk_ids"] = [chunk_ids[index] for index in indices]
            bundle["bundle_coverage_count"] = coverage["occurrences_per_chunk"]
            bundle["effective_k"] = len(bundle_groups)
            bundle["prompt_text"], bundle["training_text"] = render_joint_packed30_bundle(
                bundle, len(bundle_groups)
            )
            examples.append(bundle)

    if truncation_log_path:
        write_jsonl(truncation_logs, truncation_log_path)
    return examples


def build_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    truncation_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    dataset_name = str(config["dataset"]).lower()
    sample_mode = str(config["data"].get("sample_mode", "response")).lower()
    input_modality = resolve_input_modality(config)
    transcript_max_chars = int(config["data"].get("transcript_max_chars", 0) or 0)
    truncation_logs: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    if sample_mode == "participant_speech_packed30":
        if dataset_name != "daic":
            raise ValueError(
                "sample_mode=participant_speech_packed30 requires dataset=daic."
            )
        if input_modality == INPUT_MODALITY_TEXT_ONLY:
            return _build_subject_level_text_only_examples(
                manifest_rows,
                config,
                partition_name,
                transcript_max_chars,
                truncation_log_path=truncation_log_path,
            )
        return _build_participant_speech_packed30_examples(
            manifest_rows,
            config,
            partition_name,
            truncation_log_path=truncation_log_path,
        )

    if sample_mode == JOINT_PACKED30_MODE:
        if dataset_name != "daic":
            raise ValueError(
                f"sample_mode={JOINT_PACKED30_MODE} requires dataset=daic."
            )
        if input_modality == INPUT_MODALITY_TEXT_ONLY:
            return _build_subject_level_text_only_examples(
                manifest_rows,
                config,
                partition_name,
                transcript_max_chars,
                truncation_log_path=truncation_log_path,
            )
        return _build_participant_speech_packed30_joint_examples(
            manifest_rows,
            config,
            partition_name,
            truncation_log_path=truncation_log_path,
        )

    emotion_cache: dict[str, str | None] | None = None
    emotion_policy: str | None = None
    if use_emotion(config):
        cache_path = config["data"].get("emotion_cache_path")
        if not cache_path:
            raise ValueError("data.use_emotion=true requires data.emotion_cache_path.")
        emotion_cache = load_emotion_cache(
            cache_path,
            caption_field=str(config["data"].get("emotion_caption_field", "emotion_en")),
        )
        emotion_policy = resolve_missing_policy(config)
        report_cache_coverage(
            emotion_cache, [str(row["sample_id"]) for row in manifest_rows]
        )

    if input_modality == INPUT_MODALITY_TEXT_ONLY and dataset_name in {
        "androids_interview",
        "daic",
        "d3tec",
        "edaic",
        "turkish",
    }:
        return _build_subject_level_text_only_examples(
            manifest_rows,
            config,
            partition_name,
            transcript_max_chars,
            truncation_log_path=truncation_log_path,
        )

    if sample_mode == "subject_audio" and dataset_name in {"daic", "edaic", "turkish"}:
        return _build_subject_level_audio_examples(
            manifest_rows,
            config,
            partition_name,
            transcript_max_chars,
            truncation_log_path=truncation_log_path,
            emotion_cache=emotion_cache,
            emotion_policy=emotion_policy,
        )

    if sample_mode in {"subject_chunks", "subject_mil"} and dataset_name == "daic":
        controls = resolve_chunking_controls(config)
        selected_rows = _ordered_subject_rows(list(manifest_rows))
        if controls["eval_chunk_policy"] == "matched_k" and "train" not in partition_name.lower():
            grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected_rows:
                grouped_rows[str(row["subject_id"])].append(row)
            selected_rows = [
                rows[index]
                for subject_id in sorted(grouped_rows)
                for rows in [_ordered_subject_rows(grouped_rows[subject_id])]
                for index in evenly_spaced_indices(len(rows), int(controls["eval_chunks_per_subject"]))
            ]
        for row in selected_rows:
            example, transcript_log = _base_example_from_row(
                row, config, transcript_max_chars, emotion_cache, emotion_policy
            )
            example["audio_clip_seconds"] = [
                controls["max_audio_seconds_per_chunk"]
            ]
            example["chunk_id"] = str(row.get("chunk_id", row["sample_id"]))
            examples.append(example)
            if transcript_log:
                truncation_logs.append(
                    {
                        "partition": partition_name,
                        "subject_id": row["subject_id"],
                        "sample_id": row["sample_id"],
                        **transcript_log,
                    }
                )
        if truncation_log_path:
            write_jsonl(truncation_logs, truncation_log_path)
        return examples

    if dataset_name != "eatd" or sample_mode == "response":
        for row in sorted(manifest_rows, key=lambda item: item["sample_id"]):
            example, transcript_log = _base_example_from_row(
                row, config, transcript_max_chars, emotion_cache, emotion_policy
            )
            if transcript_log:
                truncation_logs.append(
                    {
                        "partition": partition_name,
                        "subject_id": row["subject_id"],
                        "sample_id": row["sample_id"],
                        **transcript_log,
                    }
                )
            examples.append(example)
    else:
        use_audio, use_text = _modality_flags(input_modality)
        max_per_response = float(config["data"]["max_audio_seconds_per_response"])
        max_total = float(config["data"]["max_total_audio_seconds"])
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in manifest_rows:
            grouped[row["subject_id"]][row["question_id"]] = row

        for subject_id in sorted(grouped):
            rows_by_response = grouped[subject_id]
            ordered_rows = [rows_by_response[name] for name in CANONICAL_RESPONSES]
            transcript_sections: list[str] = []
            for response_name, row in zip(CANONICAL_RESPONSES, ordered_rows):
                if use_text:
                    transcript_sections.append(f"[{response_name.capitalize()} response]\n{row['transcript']}")
            combined_transcript = "\n\n".join(transcript_sections) if use_text else ""
            combined_transcript, transcript_log = _truncate_text(combined_transcript, transcript_max_chars)
            audio_paths = [row["audio_path"] for row in ordered_rows] if use_audio else []
            audio_plan = _subject_mode_audio_plan(audio_paths, max_per_response, max_total) if use_audio else {
                "audio_original_seconds": [],
                "audio_kept_seconds": [],
                "audio_total_original_seconds": 0.0,
                "audio_total_kept_seconds": 0.0,
                "audio_truncated": False,
            }
            emotion_captions = None
            if emotion_cache is not None and use_audio:
                emotion_captions = [
                    resolve_caption(emotion_cache, str(row["sample_id"]), emotion_policy)
                    for row in ordered_rows
                ]
            user_text = render_user_prompt_text(config, combined_transcript, is_subject_bundle=True)
            internal_label_text = internal_label_text_from_int(config, int(ordered_rows[0]["label"]))
            prompt_text = build_prompt_text(
                system_prompt=config["prompt"]["system"],
                user_text=user_text,
                num_audios=len(audio_paths),
                use_audio=use_audio,
                emotion_captions=emotion_captions,
                audio_placeholder=resolve_audio_placeholder(config),
            )
            example = {
                "dataset": "eatd",
                "subject_id": subject_id,
                "sample_id": subject_id,
                "label": int(ordered_rows[0]["label"]),
                "label_text": ordered_rows[0]["label_text"],
                "internal_label_text": internal_label_text,
                "transcript": combined_transcript,
                "audio_paths": audio_paths,
                "audio_clip_seconds": audio_plan["audio_kept_seconds"],
                "input_modality": input_modality,
                "prompt_text": prompt_text,
                "training_text": build_training_text(prompt_text, internal_label_text),
                "question_id": "subject_bundle",
                "bundle_question_ids": CANONICAL_RESPONSES,
            }
            examples.append(example)
            log_row = {
                "partition": partition_name,
                "subject_id": subject_id,
                "sample_id": subject_id,
                **audio_plan,
            }
            if transcript_log:
                log_row.update(transcript_log)
            truncation_logs.append(log_row)

    if truncation_log_path:
        write_jsonl(truncation_logs, truncation_log_path)
    return examples


def qwen2audio_audio_token_length(mel_frames: int) -> int:
    """Number of audio embedding tokens Qwen2-Audio emits for ``mel_frames`` mel frames.

    Mirrors ``Qwen2AudioForConditionalGeneration._get_feat_extract_output_lengths``
    (two stride-2 convolutions). A full 30s clip = 3000 mel frames -> 750 tokens.
    Used by the audio-budget audit as a fallback when the processor does not expand
    the ``<|AUDIO|>`` placeholder in ``input_ids``.
    """
    input_lengths = (int(mel_frames) - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return max(0, int(output_lengths))


def load_audio_array(
    audio_path: str,
    target_sr: int,
    max_seconds: float | None,
    silence_audio: bool,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    info = sf.info(audio_path)
    duration = float(info.frames / info.samplerate)
    start_seconds = max(0.0, float(start_time or 0.0))
    end_seconds = duration if end_time is None else min(duration, float(end_time))
    if end_seconds <= start_seconds:
        raise ValueError(
            f"Invalid audio interval for {audio_path}: start_time={start_seconds} end_time={end_seconds}"
        )
    interval_seconds = end_seconds - start_seconds
    keep_seconds = min(float(max_seconds), interval_seconds) if max_seconds else interval_seconds
    start_frame = min(info.frames - 1, max(0, int(round(start_seconds * info.samplerate))))
    keep_frames = max(1, min(info.frames - start_frame, int(round(keep_seconds * info.samplerate))))
    if silence_audio:
        audio = np.zeros(keep_frames, dtype=np.float32)
        source_sr = info.samplerate
    else:
        audio, source_sr = sf.read(
            audio_path,
            start=start_frame,
            frames=keep_frames,
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    if int(source_sr) != int(target_sr):
        import librosa

        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def uses_audio_spans(example: dict[str, Any]) -> bool:
    """True when an example carries ordered ``audio_spans`` (packed30 protocol).

    Span examples load ONE concatenated waveform from the source WAV instead of
    per-path arrays; all existing loaders route through this predicate so the
    legacy per-path path stays untouched.
    """
    return bool(example.get("audio_spans"))


def load_audio_spans_array(
    audio_path: str,
    spans: list[dict[str, Any]],
    target_sr: int,
    silence_audio: bool,
    expected_samples: int | None = None,
) -> np.ndarray:
    """Load and concatenate end-exclusive native frame intervals (packed30).

    Reads each ``[start_frame, end_frame)`` interval at the WAV's native sample
    rate, concatenates the arrays in manifest order without inserted silence,
    and validates the resulting sample count against ``expected_samples``.
    """
    if not spans:
        raise ValueError("load_audio_spans_array requires at least one span.")
    info = sf.info(audio_path)
    source_sr = int(info.samplerate)
    arrays: list[np.ndarray] = []
    for span in spans:
        start_frame = int(span["start_frame"])
        end_frame = int(span["end_frame"])
        if not (0 <= start_frame < end_frame <= int(info.frames)):
            raise ValueError(
                f"Span [{start_frame}, {end_frame}) is outside {audio_path} "
                f"({info.frames} frames)."
            )
        if silence_audio:
            arrays.append(np.zeros(end_frame - start_frame, dtype=np.float32))
        else:
            audio, read_sr = sf.read(
                audio_path,
                start=start_frame,
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=False,
            )
            if read_sr != source_sr:
                raise ValueError(
                    f"Span read sample rate changed for {audio_path}: {read_sr} != {source_sr}."
                )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            arrays.append(np.asarray(audio, dtype=np.float32))
    concatenated = np.concatenate(arrays, axis=0)
    if expected_samples is not None and int(concatenated.shape[0]) != int(expected_samples):
        raise ValueError(
            f"Concatenated span waveform has {concatenated.shape[0]} samples; "
            f"expected {int(expected_samples)} for {audio_path}."
        )
    if int(source_sr) != int(target_sr):
        import librosa

        concatenated = librosa.resample(
            concatenated, orig_sr=source_sr, target_sr=target_sr
        )
    return np.asarray(concatenated, dtype=np.float32)


_LIBROSA_EFFECTS = "unset"


def _load_librosa_effects():
    """Lazily import ``librosa.effects`` (pulls in numba). Returns the module or
    ``None`` if unavailable, warning once so pitch/time-stretch degrade to a no-op
    instead of crashing training."""
    global _LIBROSA_EFFECTS
    if _LIBROSA_EFFECTS == "unset":
        try:
            from librosa import effects

            _LIBROSA_EFFECTS = effects
        except Exception as exc:  # noqa: BLE001 - any import failure -> skip these effects
            LOGGER.warning(
                "librosa.effects unavailable (%s); pitch_shift/time_stretch augmentation disabled, "
                "noise/gain still active.",
                exc,
            )
            _LIBROSA_EFFECTS = None
    return _LIBROSA_EFFECTS


def apply_audio_augment(
    audio: np.ndarray,
    sampling_rate: int,
    cfg: dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Train-only waveform acoustic augmentation.

    Attacks the speaker/channel/site shortcut the audio path overfits to. Each
    effect is applied independently with probability ``cfg["prob"]``; its
    magnitude is drawn uniformly from the configured ``[lo, hi]`` range. A range
    set to ``None`` (or absent) disables that effect. Determinism is preserved by
    routing all randomness through the dataset's seeded ``rng``/``np_rng``, so the
    per-epoch augmentation view is reproducible for a given ``seed``.

    NOTE: this must only ever run on the TRAIN dataset. Eval loads audio through a
    separate path (``src/evaluate.py`` backends, not ``AudioTextDataset``), so the
    determinism rule (handoff §3) holds as long as the augment cfg is passed only
    to the training dataset.
    """
    if not cfg or not cfg.get("enabled", False) or audio.size == 0:
        return audio
    prob = float(cfg.get("prob", 0.5))
    out = np.asarray(audio, dtype=np.float32)
    orig_len = out.shape[0]

    def _draw(key: str) -> float | None:
        rng_range = cfg.get(key)
        if rng_range is None or rng.random() >= prob:
            return None
        lo, hi = float(rng_range[0]), float(rng_range[1])
        return rng.uniform(lo, hi)

    # Spectral/temporal first (change content), then level, then channel noise.
    # pitch_shift/time_stretch need librosa.effects (-> numba); if that stack is
    # unavailable we skip them with a one-time warning rather than crash training.
    # Noise/gain below are pure-NumPy and always available.
    n_steps = _draw("pitch_semitones")
    rate = _draw("time_stretch")
    if n_steps is not None or (rate is not None and rate > 0):
        effects = _load_librosa_effects()
        if effects is not None:
            if n_steps is not None:
                out = effects.pitch_shift(out, sr=sampling_rate, n_steps=n_steps)
            if rate is not None and rate > 0:
                out = effects.time_stretch(out, rate=rate)
    gain_db = _draw("gain_db")
    if gain_db is not None:
        out = out * float(10.0 ** (gain_db / 20.0))
    snr_db = _draw("noise_snr_db")
    if snr_db is not None:
        signal_power = float(np.mean(out**2))
        if signal_power > 0:
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = np_rng.standard_normal(out.shape[0]).astype(np.float32) * float(np.sqrt(noise_power))
            out = out + noise

    # Keep length <= original so the audio-token budget (and VRAM) stays bounded
    # even when time_stretch lengthens the clip.
    if out.shape[0] > orig_len:
        out = out[:orig_len]
    np.clip(out, -1.0, 1.0, out=out)
    return np.asarray(out, dtype=np.float32)


class AudioTextDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        processor_sampling_rate: int | None = None,
        silence_audio: bool = False,
        chunk_sampling: str | None = None,
        chunk_sampling_seed: int = 1337,
        audio_augment: dict[str, Any] | None = None,
    ):
        """``chunk_sampling`` controls subject_audio chunk selection.

        - ``None`` / ``"deterministic"`` (default): use the example's baked
          ``audio_paths`` verbatim. This MUST be used for validation/test so
          reported metrics never depend on random sampling.
        - ``"random"``: for subject_audio examples, draw ``chunks_per_subject``
          chunks from the subject's full ``subject_chunk_paths`` on every access,
          giving a fresh view each epoch. Training-only.

        ``audio_augment`` (train-only) applies waveform acoustic augmentation to
        each loaded chunk; see ``apply_audio_augment``. Leave ``None`` for
        selection/eval datasets.
        """
        if chunk_sampling not in (None, "deterministic", "random"):
            raise ValueError(f"Unsupported chunk_sampling={chunk_sampling!r}.")
        self.examples = examples
        self.processor_sampling_rate = int(processor_sampling_rate) if processor_sampling_rate is not None else None
        self.silence_audio = bool(silence_audio)
        self.chunk_sampling = chunk_sampling
        self.audio_augment = audio_augment if (audio_augment and audio_augment.get("enabled")) else None
        self._rng = random.Random(int(chunk_sampling_seed))
        self._np_rng = np.random.default_rng(int(chunk_sampling_seed))

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve_audio_plan(self, example: dict[str, Any]) -> tuple[list[str], list[Any]]:
        audio_paths = example["audio_paths"]
        clip_seconds = example["audio_clip_seconds"]
        if (
            self.chunk_sampling == "random"
            and example.get("subject_chunk_paths")
            and example.get("chunks_per_subject")
        ):
            k = int(example["chunks_per_subject"])
            pool = list(example["subject_chunk_paths"])
            if len(pool) >= k:
                audio_paths = self._rng.sample(pool, k)
            else:
                audio_paths = pool + self._rng.choices(pool, k=k - len(pool))
            cap = clip_seconds[0] if clip_seconds else None
            clip_seconds = [cap] * len(audio_paths)
        return audio_paths, clip_seconds

    def _rerender_emotion_prompt(
        self, example: dict[str, Any], audio_paths: list[str]
    ) -> dict[str, Any]:
        """Re-render the interleaved emotion prompt to match the sampled chunks.

        Under ``chunk_sampling="random"`` the audio order differs from the baked
        deterministic view, so the per-chunk ``Emotional description i`` lines must
        be rebuilt from the captions of the chunks actually sampled (§4.3). Eval is
        deterministic and keeps the baked text. Cheap string formatting; the
        collator reads ``prompt_text``/``training_text`` fresh each call.
        """
        caption_map = example["chunk_caption_by_path"]
        captions = [caption_map.get(path) for path in audio_paths]
        prompt_text = build_prompt_text(
            system_prompt=example["emotion_system_prompt"],
            user_text=example["emotion_user_text"],
            num_audios=len(audio_paths),
            use_audio=True,
            emotion_captions=captions,
            audio_placeholder=example.get("emotion_audio_placeholder", AUDIO_PLACEHOLDER),
        )
        return {
            "prompt_text": prompt_text,
            "training_text": build_training_text(
                prompt_text, example["emotion_internal_label_text"]
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        audio_paths, clip_seconds = self._resolve_audio_plan(example)
        rerendered: dict[str, Any] = {}
        if self.chunk_sampling == "random" and example.get("chunk_caption_by_path"):
            rerendered = self._rerender_emotion_prompt(example, audio_paths)
        if example.get("audio_span_groups"):
            audio_arrays = load_span_group_audio_arrays(
                example,
                self.processor_sampling_rate,
                self.silence_audio,
            )
            if self.audio_augment and not self.silence_audio:
                audio_arrays = [
                    apply_audio_augment(
                        array,
                        self.processor_sampling_rate,
                        self.audio_augment,
                        self._rng,
                        self._np_rng,
                    )
                    for array in audio_arrays
                ]
        elif uses_audio_spans(example):
            if self.processor_sampling_rate is None:
                raise ValueError("Audio examples require a processor sampling rate.")
            audio_arrays = [
                load_audio_spans_array(
                    example["audio_path"],
                    example["audio_spans"],
                    self.processor_sampling_rate,
                    self.silence_audio,
                    example.get("participant_sample_count"),
                )
            ]
            if self.audio_augment and not self.silence_audio:
                audio_arrays = [
                    apply_audio_augment(
                        array,
                        self.processor_sampling_rate,
                        self.audio_augment,
                        self._rng,
                        self._np_rng,
                    )
                    for array in audio_arrays
                ]
        elif audio_paths:
            if self.processor_sampling_rate is None:
                raise ValueError("Audio examples require a processor sampling rate.")
            start_times = list(example.get("audio_start_times") or [None] * len(audio_paths))
            end_times = list(example.get("audio_end_times") or [None] * len(audio_paths))
            audio_arrays = [
                load_audio_array(
                    audio_path,
                    self.processor_sampling_rate,
                    max_seconds,
                    self.silence_audio,
                    start_time,
                    end_time,
                )
                for audio_path, max_seconds, start_time, end_time in zip(
                    audio_paths, clip_seconds, start_times, end_times
                )
            ]
            if self.audio_augment and not self.silence_audio:
                audio_arrays = [
                    apply_audio_augment(
                        array, self.processor_sampling_rate, self.audio_augment, self._rng, self._np_rng
                    )
                    for array in audio_arrays
                ]
        else:
            audio_arrays = []
        return {
            **example,
            **rerendered,
            "audio_arrays": audio_arrays,
        }


def save_partition_subjects(
    output_path: str | Path,
    train_subject_ids: list[str],
    selection_subject_ids: list[str],
    final_eval_subject_ids: list[str],
    subject_labels: dict[str, int],
    train_split_name: str = "train_inner",
    selection_split_name: str = "val_inner",
    final_eval_split_name: str = "final_eval",
) -> dict[str, Any]:
    train_depressed = sum(subject_labels[subject_id] for subject_id in train_subject_ids)
    selection_depressed = sum(subject_labels[subject_id] for subject_id in selection_subject_ids)
    final_eval_depressed = sum(subject_labels[subject_id] for subject_id in final_eval_subject_ids)
    payload = {
        "split_names": {
            "train": train_split_name,
            "selection": selection_split_name,
            "final_eval": final_eval_split_name,
        },
        "train_subject_ids": train_subject_ids,
        "selection_subject_ids": selection_subject_ids,
        "final_eval_subject_ids": final_eval_subject_ids,
        "train_inner_subject_ids": train_subject_ids,
        "val_inner_subject_ids": selection_subject_ids,
        "class_counts": {
            train_split_name: {
                "depressed": train_depressed,
                "non_depressed": len(train_subject_ids) - train_depressed,
            },
            selection_split_name: {
                "depressed": selection_depressed,
                "non_depressed": len(selection_subject_ids) - selection_depressed,
            },
            final_eval_split_name: {
                "depressed": final_eval_depressed,
                "non_depressed": len(final_eval_subject_ids) - final_eval_depressed,
            },
        },
    }
    save_json(payload, output_path)
    return payload
