from __future__ import annotations

from pathlib import Path

import pytest

from src.data.daic import PACKED30_PROTOCOL_ID
from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    build_examples,
    load_audio_spans_array,
    uses_audio_spans,
)


def packed30_config(**data_overrides) -> dict:
    config = {
        "dataset": "daic",
        "seed": 1337,
        "protocol_id": PACKED30_PROTOCOL_ID,
        "prompt": {
            "system": "system",
            "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "sample_mode": "participant_speech_packed30",
            "participant_chunk_samples": 480000,
            "inter_span_silence_samples": 0,
            "train_chunk_policy": "all_chunks_subject_normalized",
            "eval_chunk_policy": "all_chunks_mean_score",
            "loss_weight_rescale": "mean_one",
            "equal_row_weight": False,
            "transcript_max_chars": 4000,
        },
        "evaluation": {
            "sample_prediction_mode": "original_teacher_forced",
            "aggregation_level": "subject",
            "subject_score_aggregation": "mean_score",
        },
    }
    config["data"].update(data_overrides)
    return config


def packed30_row(
    subject_id: str,
    chunk_index: int,
    num_chunks: int,
    label: int = 0,
    transcript: str = "full transcript",
    wav_path: str = "/tmp/x.wav",
) -> dict:
    return {
        "schema_version": "daic_participant_speech_packed30_manifest.v1",
        "protocol_id": PACKED30_PROTOCOL_ID,
        "dataset": "daic",
        "subject_id": subject_id,
        "sample_id": f"{subject_id}_participant_p30_{chunk_index:03d}",
        "audio_path": wav_path,
        "audio_spans": [
            {
                "start_frame": chunk_index * 1000,
                "end_frame": chunk_index * 1000 + 1000,
                "source_row_index": chunk_index,
                "source_start_time": 1.0,
                "source_stop_time": 2.0,
            }
        ],
        "participant_sample_count": 1000,
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "chunk_transcript": "row text",
        "full_participant_transcript": transcript,
        "full_participant_transcript_sha256": __import__("hashlib").sha256(transcript.encode()).hexdigest(),
        "transcript": transcript,
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "split_original": "train",
    }


def test_audio_only_one_example_per_chunk() -> None:
    rows = [packed30_row("300", 0, 3, 0), packed30_row("300", 1, 3, 0), packed30_row("300", 2, 3, 0)]
    examples = build_examples(rows, packed30_config(), "train")
    assert len(examples) == 3
    assert all(example["sample_id"].endswith("_participant_p30_%03d" % index) for index, example in enumerate(examples))
    assert all(uses_audio_spans(example) for example in examples)
    assert all(example["participant_sample_count"] == 1000 for example in examples)
    assert all(example["audio_paths"] == [] for example in examples)
    assert all(example["protocol_id"] == PACKED30_PROTOCOL_ID for example in examples)


def test_every_audio_prompt_has_exactly_one_placeholder() -> None:
    rows = [packed30_row("300", 0, 2, 0), packed30_row("300", 1, 2, 0)]
    examples = build_examples(rows, packed30_config(), "train")
    for example in examples:
        assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == 1
        assert example["prompt_text"].count("<|AUDIO|>") == 1


def test_audio_text_repeats_full_participant_transcript_with_hash() -> None:
    rows = [
        packed30_row("300", 0, 2, 0, transcript="line1\nline2"),
        packed30_row("300", 1, 2, 0, transcript="line1\nline2"),
    ]
    examples = build_examples(rows, packed30_config(use_audio=True, use_text=True), "train")
    assert len(examples) == 2
    assert {example["transcript"] for example in examples} == {"line1\nline2"}
    assert len({example["full_participant_transcript_sha256"] for example in examples}) == 1
    assert "line1\nline2" in examples[0]["prompt_text"]


def test_text_only_one_example_per_subject() -> None:
    rows = [
        packed30_row("300", 0, 3, 0, transcript="t0"),
        packed30_row("300", 1, 3, 0, transcript="t0"),
        packed30_row("300", 2, 3, 0, transcript="t0"),
        packed30_row("301", 0, 1, 1, transcript="t1"),
    ]
    examples = build_examples(
        rows, packed30_config(use_audio=False, use_text=True, sample_mode="subject"), "train"
    )
    assert len(examples) == 2
    assert sorted(example["subject_id"] for example in examples) == ["300", "301"]
    assert all(example["transcript"] in {"t0", "t1"} for example in examples)


def test_packed30_text_only_routes_to_one_example_per_subject() -> None:
    rows = [packed30_row("300", 0, 2, 0), packed30_row("300", 1, 2, 0)]
    examples = build_examples(
        rows, packed30_config(use_audio=False, use_text=True), "train"
    )
    assert len(examples) == 1
    assert examples[0]["sample_id"] == "300"
    assert examples[0]["transcript"] == "full transcript"


def test_packed30_requires_daic_dataset() -> None:
    config = packed30_config()
    config["dataset"] = "cmdc"
    rows = [packed30_row("300", 0, 1, 0)]
    with pytest.raises(ValueError, match="requires dataset=daic"):
        build_examples(rows, config, "train")


def test_packed30_examples_load_single_span_waveform(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "s.wav"
    audio = np.zeros(480000, dtype=np.float32)
    sf.write(wav, audio, 16000, subtype="PCM_16")
    rows = [
        {
            **packed30_row("300", 0, 1, 0, wav_path=str(wav)),
            "audio_spans": [{"start_frame": 0, "end_frame": 480000, "source_row_index": 0, "source_start_time": 0.0, "source_stop_time": 30.0}],
            "participant_sample_count": 480000,
        }
    ]
    examples = build_examples(rows, packed30_config(), "train")
    loaded = load_audio_spans_array(
        examples[0]["audio_path"], examples[0]["audio_spans"], 16000, False, 480000
    )
    assert loaded.shape == (480000,)


def test_canonical_daic_subject_chunks_behavior_unchanged() -> None:
    config = {
        "dataset": "daic",
        "seed": 1337,
        "prompt": {"system": "system", "user_template": "{audio_context_block} {label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "sample_mode": "subject_chunks",
            "train_chunk_policy": "rotary_k",
            "train_chunks_per_subject": 4,
            "eval_chunk_policy": "all",
            "eval_chunks_per_subject": "all",
            "max_audio_seconds_per_chunk": 30.0,
        },
        "evaluation": {"subject_score_aggregation": "mean_score"},
    }
    rows = [
        {
            "dataset": "daic",
            "subject_id": "300",
            "sample_id": f"300_{index}",
            "chunk_id": str(index),
            "label": 0,
            "label_text": "Non-depressed",
            "transcript": "",
            "audio_path": f"/tmp/300_{index}.wav",
        }
        for index in range(4)
    ]
    examples = build_examples(rows, config, "train")
    assert len(examples) == 4
    assert all("audio_paths" in example and not uses_audio_spans(example) for example in examples)
    assert all("protocol_id" not in example for example in examples)


def test_androids_response_examples_unchanged() -> None:
    config = {
        "dataset": "androids_interview",
        "seed": 1337,
        "prompt": {"system": "system", "user_template": "{audio_context_block} {label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": True,
            "sample_mode": "response_segments",
            "segment_seconds": 30.0,
            "audio_text_transcript_scope": "segment_aligned",
            "transcript_max_chars": 4000,
        },
    }
    rows = [
        {
            "dataset": "androids_interview",
            "subject_id": "and1",
            "sample_id": "and1_000",
            "response_id": "1",
            "turn_id": 1,
            "segment_transcript": "seg text",
            "full_turn_transcript": "full text",
            "transcript": "seg text",
            "label": 0,
            "label_text": "Non-depressed",
            "audio_path": "/tmp/and1_000.wav",
        }
    ]
    examples = build_examples(rows, config, "train")
    assert len(examples) == 1
    assert examples[0]["transcript"] == "seg text"
    assert "protocol_id" not in examples[0]
    assert not uses_audio_spans(examples[0])
