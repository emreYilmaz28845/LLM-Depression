from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from src.aggregate import aggregate_response_subject_predictions
from src.data.androids import (
    androids_window_id,
    apply_androids_training_weights,
    build_androids_interview_official_folds,
    discover_androids_interview_windows,
    equal_duration_windows,
    parse_androids_recording_id,
    parse_androids_turn_path,
)
from src.data.runtime import build_examples
from src.utils import PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
from scripts.transcribe_multilingual_qwen3asr import _validate_androids_resume_rows


def _config(*, use_audio: bool, use_text: bool, scope: str = "segment_aligned") -> dict:
    return {
        "dataset": "androids_interview",
        "prompt": {
            "system": "screen",
            "user_template": "{audio_context_block}\n{transcript_block}{decision_basis} {label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": use_audio,
            "use_text": use_text,
            "sample_mode": "response_segments" if use_audio else "subject",
            "audio_text_transcript_scope": scope,
            "transcript_max_chars": 12000,
        },
    }


def _row(turn_id: int, window_index: int, *, subject_id: str = "05_P") -> dict:
    turn_key = f"05_PM53_4_{turn_id}"
    return {
        "dataset": "androids_interview",
        "subject_id": subject_id,
        "sample_id": androids_window_id(turn_key, window_index),
        "response_id": turn_key,
        "turn_id": turn_id,
        "window_index": window_index,
        "segment_index": window_index,
        "num_segments": 2,
        "audio_path": "/tmp/turn.wav",
        "start_time": window_index * 10.0,
        "end_time": (window_index + 1) * 10.0,
        "segment_duration": 10.0,
        "segment_transcript": f"segment {turn_id}-{window_index}",
        "full_turn_transcript": f"full turn {turn_id}",
        "transcript": f"segment {turn_id}-{window_index}",
        "label": 1,
        "label_text": "Depressed",
        "question_id": str(turn_id),
        "prompt_id": turn_id,
    }


def test_identity_turn_window_and_condition_parsing() -> None:
    identity = parse_androids_recording_id("05_PM53_4")
    assert identity == {
        "recording_id": "05_PM53_4",
        "subject_id": "05_P",
        "numeric_subject_id": "05",
        "condition_code": "P",
        "label": 1,
        "label_text": "Depressed",
        "diagnosis": "depression",
        "gender": "M",
        "age": 53,
        "education_level": 4,
        "education_level_raw": "4",
    }
    parsed = parse_androids_turn_path(
        Path("/corpus/audio_clip/05_PM53_4/05_PM53_4_10.wav")
    )
    assert parsed["turn_id"] == 10
    assert parsed["response_id"] == "05_PM53_4_10"
    assert androids_window_id(parsed["turn_key"], 3) == "05_PM53_4_10_w03"


def test_equal_windows_and_discovery_are_complete_contiguous_and_bounded(tmp_path) -> None:
    windows = equal_duration_windows(91.0, 30.0)
    assert len(windows) == 4
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 91.0
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    assert max(end - start for start, end in windows) <= 30.0

    audio_dir = tmp_path / "Interview-Task" / "audio_clip" / "01_CF56_1"
    audio_dir.mkdir(parents=True)
    audio = np.zeros(31 * 8000, dtype=np.float32)
    sf.write(audio_dir / "01_CF56_1_1.wav", audio, 8000)
    rows = discover_androids_interview_windows(
        tmp_path, enforce_corpus_contract=False
    )
    assert len(rows) == 2
    assert rows[0]["start_time"] == 0.0
    assert rows[-1]["end_time"] == 31.0
    assert rows[0]["end_time"] == rows[1]["start_time"]
    assert max(row["segment_duration"] for row in rows) <= 30.0


def test_only_interview_fold_columns_are_parsed_and_cover_subjects(tmp_path) -> None:
    fold_path = tmp_path / "fold-lists.csv"
    rows = [
        ["Read", "", "", "", "", "", "", "Interview", "", "", "", ""],
        ["fold1", "fold2", "fold3", "fold4", "fold5", "", "", "fold1", "fold2", "fold3", "fold4", "fold5"],
        ["'read_ignored'", "", "", "", "", "", "", "'01_CF56_1'", "'02_PM53_4'", "'03_CF44_2'", "'04_PM35_3'", "'05_CF41_3'"],
    ]
    with fold_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    mapping = {
        "01_CF56_1": "01_C",
        "02_PM53_4": "02_P",
        "03_CF44_2": "03_C",
        "04_PM35_3": "04_P",
        "05_CF41_3": "05_C",
    }
    folds = build_androids_interview_official_folds(fold_path, mapping)
    assert len(folds) == 5
    assert {
        subject
        for payload in folds.values()
        for subject in payload["final_eval_subject_ids"]
    } == set(mapping.values())
    assert all("read_ignored" not in str(payload) for payload in folds.values())


def test_runtime_selects_segment_or_full_turn_transcript() -> None:
    rows = [_row(1, 0), _row(1, 1)]
    aligned = build_examples(rows, _config(use_audio=True, use_text=True), "train")
    full = build_examples(
        rows,
        _config(use_audio=True, use_text=True, scope="full_turn"),
        "train",
    )
    assert [row["transcript"] for row in aligned] == ["segment 1-0", "segment 1-1"]
    assert [row["transcript"] for row in full] == ["full turn 1", "full turn 1"]
    assert [row["audio_start_times"] for row in aligned] == [[0.0], [10.0]]
    audio_only = build_examples(
        rows,
        _config(use_audio=True, use_text=False, scope="not_a_real_scope"),
        "train",
    )
    assert all(row["transcript"] == "" for row in audio_only)


def test_segment_asr_resume_accepts_raw_fsynced_rows_but_rejects_bad_intervals() -> None:
    window = _row(1, 0)
    window["window_id"] = window["sample_id"]
    windows = {window["sample_id"]: window}
    _validate_androids_resume_rows(
        [{"audio_path": f"/temporary/{window['sample_id']}.wav"}],
        windows,
    )
    bad = {
        "sample_id": window["sample_id"],
        "audio_path": window["audio_path"],
        "start_time": 1.0,
        "end_time": window["end_time"],
    }
    try:
        _validate_androids_resume_rows([bad], windows)
    except ValueError as exc:
        assert "Interval-mismatched" in str(exc)
    else:
        raise AssertionError("Interval-mismatched resume row was accepted.")


def test_text_only_deduplicates_windows_and_orders_turns_numerically() -> None:
    rows = [_row(10, 1), _row(2, 0), _row(10, 0), _row(2, 1)]
    examples = build_examples(
        rows, _config(use_audio=False, use_text=True), "train"
    )
    assert len(examples) == 1
    transcript = examples[0]["transcript"]
    assert transcript.count("full turn 2") == 1
    assert transcript.count("full turn 10") == 1
    assert transcript.index("[Turn 2]") < transcript.index("[Turn 10]")


def test_training_weights_equalize_subjects_and_turns_within_subject() -> None:
    examples = [
        _row(1, 0, subject_id="05_P"),
        _row(1, 1, subject_id="05_P"),
        _row(2, 0, subject_id="05_P"),
        {**_row(1, 0, subject_id="01_C"), "response_id": "01_CF56_1_1"},
    ]
    weighted, audit = apply_androids_training_weights(examples)
    assert math.isclose(audit["mean_loss_weight"], 1.0)
    assert set(audit["raw_subject_weight_totals"].values()) == {1.0}
    turn_totals: defaultdict[str, float] = defaultdict(float)
    for row in weighted:
        turn_totals[row["response_id"]] += row["raw_loss_weight"]
    assert math.isclose(turn_totals["05_PM53_4_1"], 0.5)
    assert math.isclose(turn_totals["05_PM53_4_2"], 0.5)
    assert math.isclose(turn_totals["01_CF56_1_1"], 1.0)


def test_window_turn_subject_aggregation_does_not_multiply_long_turn_votes() -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "subject_id": "05_P",
                "response_id": "05_PM53_4_1",
                "sample_id": f"long_w{index}",
                "num_segments": 5,
                "label": 0,
                "teacher_forced_prediction": 1,
                "dep_score": 1.0,
                "non_score": 0.0,
            }
        )
    for turn in (2, 3):
        rows.append(
            {
                "subject_id": "05_P",
                "response_id": f"05_PM53_4_{turn}",
                "sample_id": f"short_{turn}",
                "num_segments": 1,
                "label": 0,
                "teacher_forced_prediction": 0,
                "dep_score": 0.0,
                "non_score": 1.0,
            }
        )
    _, _, subjects, _ = aggregate_response_subject_predictions(
        rows,
        prediction_field="teacher_forced_prediction",
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        score_average=True,
    )
    assert subjects[0]["prediction"] == 0


def test_androids_hierarchy_uses_equal_weight_scores_not_window_majority() -> None:
    rows = []
    for index, (prediction, margin) in enumerate(((1, 0.1), (1, 0.1), (0, -1.0))):
        rows.append(
            {
                "subject_id": "01_C",
                "response_id": "01_CF56_1_1",
                "sample_id": f"window_{index}",
                "num_segments": 3,
                "label": 0,
                "teacher_forced_prediction": prediction,
                "dep_score": margin,
                "non_score": 0.0,
            }
        )
    responses, _, subjects, _ = aggregate_response_subject_predictions(
        rows,
        prediction_field="teacher_forced_prediction",
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        score_average=True,
    )
    assert responses[0]["prediction"] == 0
    assert subjects[0]["prediction"] == 0
