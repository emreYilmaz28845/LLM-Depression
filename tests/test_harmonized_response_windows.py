from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.androids import apply_hierarchical_training_weights
from src.data.runtime import AUDIO_PLACEHOLDER, build_examples


def config(dataset: str, *, use_audio: bool = True, use_text: bool = True) -> dict:
    return {
        "dataset": dataset,
        "seed": 1337,
        "prompt": {
            "system": "system",
            "user_template": "{audio_context_block}\n{transcript_block}{label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": use_audio,
            "use_text": use_text,
            "sample_mode": "harmonized_response_windows",
            "segment_seconds": 30.0,
            "audio_text_transcript_scope": "full_subject",
            "transcript_max_chars": 0,
        },
    }


def row(subject: str, sample: str, audio: Path, transcript: str, question: str) -> dict:
    return {
        "dataset": "cmdc",
        "subject_id": subject,
        "sample_id": sample,
        "question_id": question,
        "label": 0,
        "label_text": "Non-depressed",
        "audio_path": str(audio),
        "transcript": transcript,
    }


def test_cmdc_long_response_uses_all_nonoverlapping_windows_and_full_transcript(
    tmp_path: Path,
) -> None:
    long_audio = tmp_path / "q1.wav"
    short_audio = tmp_path / "q2.wav"
    sf.write(long_audio, np.zeros(65 * 16000, dtype=np.float32), 16000)
    sf.write(short_audio, np.zeros(10 * 16000, dtype=np.float32), 16000)
    rows = [
        row("s1", "s1_q1", long_audio, "first answer", "Q1"),
        row("s1", "s1_q2", short_audio, "second answer", "Q2"),
    ]

    examples = build_examples(rows, config("cmdc"), "train")

    assert len(examples) == 4
    assert [example["response_id"] for example in examples] == [
        "s1::Q1",
        "s1::Q1",
        "s1::Q1",
        "s1::Q2",
    ]
    spans = [(example["start_time"], example["end_time"]) for example in examples]
    assert spans[:3] == [
        (0.0, 65.0 / 3.0),
        (65.0 / 3.0, 130.0 / 3.0),
        (130.0 / 3.0, 65.0),
    ]
    assert spans[3] == (0.0, 10.0)
    assert {example["transcript"] for example in examples} == {
        "first answer\nsecond answer"
    }
    assert all(example["prompt_text"].count(AUDIO_PLACEHOLDER) == 1 for example in examples)


def test_text_only_emits_one_full_transcript_example_per_subject(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    rows = [
        row("s1", "s1_q1", audio, "first", "Q1"),
        row("s1", "s1_q2", audio, "second", "Q2"),
        row("s2", "s2_q1", audio, "third", "Q1"),
    ]
    rows[2]["label"] = 1
    rows[2]["label_text"] = "Depressed"

    examples = build_examples(
        rows, config("cmdc", use_audio=False, use_text=True), "train"
    )

    assert len(examples) == 2
    assert {example["subject_id"] for example in examples} == {"s1", "s2"}
    assert {example["transcript"] for example in examples} == {"first\nsecond", "third"}
    assert all(AUDIO_PLACEHOLDER not in example["prompt_text"] for example in examples)


def test_hierarchical_weights_equalize_subjects_and_responses(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    examples = [
        {"subject_id": "s1", "response_id": "r1", "sample_id": "a"},
        {"subject_id": "s1", "response_id": "r1", "sample_id": "b"},
        {"subject_id": "s1", "response_id": "r2", "sample_id": "c"},
        {"subject_id": "s2", "response_id": "r3", "sample_id": "d"},
    ]

    weighted, audit = apply_hierarchical_training_weights(examples)

    by_subject = {}
    for example in weighted:
        by_subject.setdefault(example["subject_id"], 0.0)
        by_subject[example["subject_id"]] += example["raw_loss_weight"]
    assert by_subject == {"s1": 1.0, "s2": 1.0}
    assert audit["raw_source_unit_weight_totals"]["r1"] == 0.5
    assert audit["raw_source_unit_weight_totals"]["r2"] == 0.5
