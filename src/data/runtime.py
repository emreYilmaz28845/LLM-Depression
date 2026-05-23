from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.eatd import CANONICAL_RESPONSES
from src.utils import (
    get_logger,
    internal_label_text_from_int,
    prompt_label_descriptor,
    prompt_label_instruction,
    read_jsonl,
    save_json,
    write_jsonl,
)


LOGGER = get_logger(__name__)
AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"


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


def _audio_prompt_block(num_audios: int) -> str:
    return "\n".join(f"Audio {index}: {AUDIO_PLACEHOLDER}" for index in range(1, num_audios + 1))


def _single_example_user_text_with_config(
    config: dict[str, Any],
    use_audio: bool,
    use_text: bool,
    transcript: str,
) -> str:
    label_descriptor = prompt_label_descriptor(config)
    label_instruction = prompt_label_instruction(config)
    if use_audio and use_text:
        return (
            "The subject's speech audio is provided.\n"
            f"The transcript of the subject's speech is:\n{transcript}\n\n"
            f"Based on the audio and transcript, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    if use_audio:
        return (
            "The subject's speech audio is provided.\n\n"
            f"Based on the audio, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    if use_text:
        return (
            f"The transcript of the subject's speech is:\n{transcript}\n\n"
            f"Based on the transcript, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    raise ValueError("At least one of use_audio or use_text must be enabled.")


def _subject_example_user_text_with_config(
    config: dict[str, Any],
    use_audio: bool,
    use_text: bool,
    transcript: str,
) -> str:
    label_descriptor = prompt_label_descriptor(config)
    label_instruction = prompt_label_instruction(config)
    if use_audio and use_text:
        return (
            "The subject's speech audio is provided in three responses: negative, neutral, and positive.\n"
            f"The transcript of the subject's speech is:\n{transcript}\n\n"
            f"Based on the audio and transcript, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    if use_audio:
        return (
            "The subject's speech audio is provided in three responses: negative, neutral, and positive.\n\n"
            f"Based on the audio, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    if use_text:
        return (
            f"The transcript of the subject's speech is:\n{transcript}\n\n"
            f"Based on the transcript, determine whether the subject is {label_descriptor}.\n"
            f"{label_instruction}"
        )
    raise ValueError("At least one of use_audio or use_text must be enabled.")


def build_prompt_text(
    system_prompt: str,
    user_text: str,
    num_audios: int,
    use_audio: bool,
) -> str:
    if use_audio and num_audios:
        user_text = _audio_prompt_block(num_audios) + "\n" + user_text
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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    use_audio = bool(config["data"]["use_audio"])
    use_text = bool(config["data"]["use_text"])
    transcript = row["transcript"] if use_text else ""
    transcript, transcript_log = _truncate_text(transcript, transcript_max_chars)
    user_text = _single_example_user_text_with_config(config, use_audio, use_text, transcript)
    example_internal_label = row.get("internal_label_text") or internal_label_text_from_int(config, int(row["label"]))
    prompt_text = build_prompt_text(
        system_prompt=config["prompt"]["system"],
        user_text=user_text,
        num_audios=1 if use_audio else 0,
        use_audio=use_audio,
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
        "prompt_text": prompt_text,
        "training_text": build_training_text(prompt_text, example_internal_label),
        "question_id": row.get("question_id", ""),
    }
    return example, transcript_log


def build_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_name: str,
    truncation_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    dataset_name = str(config["dataset"]).lower()
    sample_mode = str(config["data"].get("sample_mode", "response")).lower()
    transcript_max_chars = int(config["data"].get("transcript_max_chars", 0) or 0)
    truncation_logs: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    if dataset_name != "eatd" or sample_mode == "response":
        for row in sorted(manifest_rows, key=lambda item: item["sample_id"]):
            example, transcript_log = _base_example_from_row(row, config, transcript_max_chars)
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
        use_audio = bool(config["data"]["use_audio"])
        use_text = bool(config["data"]["use_text"])
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
            user_text = _subject_example_user_text_with_config(config, use_audio, use_text, combined_transcript)
            internal_label_text = internal_label_text_from_int(config, int(ordered_rows[0]["label"]))
            prompt_text = build_prompt_text(
                system_prompt=config["prompt"]["system"],
                user_text=user_text,
                num_audios=len(audio_paths),
                use_audio=use_audio,
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


def load_audio_array(audio_path: str, target_sr: int, max_seconds: float | None, silence_audio: bool) -> np.ndarray:
    info = sf.info(audio_path)
    keep_seconds = min(max_seconds, float(info.frames / info.samplerate)) if max_seconds else float(info.frames / info.samplerate)
    keep_frames = max(1, int(round(keep_seconds * info.samplerate)))
    if silence_audio:
        audio = np.zeros(keep_frames, dtype=np.float32)
        source_sr = info.samplerate
    else:
        audio, source_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[:keep_frames]
    if int(source_sr) != int(target_sr):
        import librosa

        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


class AudioTextDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], processor_sampling_rate: int, silence_audio: bool = False):
        self.examples = examples
        self.processor_sampling_rate = int(processor_sampling_rate)
        self.silence_audio = bool(silence_audio)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        audio_arrays = [
            load_audio_array(audio_path, self.processor_sampling_rate, max_seconds, self.silence_audio)
            for audio_path, max_seconds in zip(example["audio_paths"], example["audio_clip_seconds"])
        ]
        return {
            **example,
            "audio_arrays": audio_arrays,
        }


def save_partition_subjects(
    output_path: str | Path,
    train_inner_subject_ids: list[str],
    val_inner_subject_ids: list[str],
    final_eval_subject_ids: list[str],
    subject_labels: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "train_inner_subject_ids": train_inner_subject_ids,
        "val_inner_subject_ids": val_inner_subject_ids,
        "final_eval_subject_ids": final_eval_subject_ids,
        "class_counts": {
            "train_inner": {
                "depressed": sum(subject_labels[subject_id] for subject_id in train_inner_subject_ids),
                "non_depressed": len(train_inner_subject_ids) - sum(subject_labels[subject_id] for subject_id in train_inner_subject_ids),
            },
            "val_inner": {
                "depressed": sum(subject_labels[subject_id] for subject_id in val_inner_subject_ids),
                "non_depressed": len(val_inner_subject_ids) - sum(subject_labels[subject_id] for subject_id in val_inner_subject_ids),
            },
            "final_eval": {
                "depressed": sum(subject_labels[subject_id] for subject_id in final_eval_subject_ids),
                "non_depressed": len(final_eval_subject_ids) - sum(subject_labels[subject_id] for subject_id in final_eval_subject_ids),
            },
        },
    }
    save_json(payload, output_path)
    return payload
