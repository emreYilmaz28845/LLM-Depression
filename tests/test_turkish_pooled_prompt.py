"""Turkish pooled question-context prompt rendering (plan Step 1)."""

from __future__ import annotations

import pytest

from src.data.runtime import (
    QUESTION_CONTEXT_SENTENCES,
    _base_example_from_row,
    build_examples,
    render_user_prompt_text,
)


def _config(modality: str = "audio_text", tagged: bool = True) -> dict:
    use_audio = modality in ("audio_only", "audio_text")
    use_text = modality in ("text_only", "audio_text")
    system = "You are a psychologist analyzing speech and transcript information for depression screening."
    if modality == "audio_only":
        system = "You are a psychologist analyzing speech audio for depression screening."
    elif modality == "text_only":
        system = "You are a psychologist analyzing transcript information for depression screening."
    template = "{audio_context_block}\n{transcript_block}Based on audio and transcript."
    if tagged:
        template = template.replace(
            "{transcript_block}", "{question_context}{transcript_block}", 1
        )
    return {
        "dataset": "turkish",
        "dataset_variant": "pooled_t17",
        "prompt": {"system": system, "user_template": template, "prompt_language": "english"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": use_audio,
            "use_text": use_text,
            "sample_mode": "harmonized_response_windows",
            "segment_seconds": 30.0,
        },
    }


def _row(sample_id: str, subject_id: str, variant: str, transcript: str = "merhaba dunya") -> dict:
    return {
        "dataset": "turkish",
        "subject_id": subject_id,
        "sample_id": sample_id,
        "audio_path": f"/tmp/{sample_id}.wav",
        "transcript": transcript,
        "label": 1,
        "label_text": "Depressed",
        "dataset_variant": variant,
    }


def test_question_context_sentences_are_locked() -> None:
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] == (
        "The following speech is the subject's response to positive interview questions."
    )
    assert QUESTION_CONTEXT_SENTENCES["negative_only_t17"] == (
        "The following speech is the subject's response to negative interview questions."
    )


def test_tagged_template_renders_correct_sentence_per_condition() -> None:
    config = _config("audio_text", tagged=True)
    pos = render_user_prompt_text(
        config, "transcript here", question_condition="pos_only_t17"
    )
    neg = render_user_prompt_text(
        config, "transcript here", question_condition="negative_only_t17"
    )
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] in pos
    assert QUESTION_CONTEXT_SENTENCES["negative_only_t17"] in neg
    assert QUESTION_CONTEXT_SENTENCES["negative_only_t17"] not in pos
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] not in neg
    # tag sits on its own line immediately before the transcript block
    assert "questions.\nThe transcript of the subject's speech is:" in pos
    assert config["prompt"]["system"] not in pos  # system prompt untouched by renderer


def test_tagged_template_fail_closed_on_missing_or_unknown_condition() -> None:
    config = _config("audio_text", tagged=True)
    with pytest.raises(ValueError):
        render_user_prompt_text(config, "transcript here", question_condition=None)
    with pytest.raises(ValueError):
        render_user_prompt_text(config, "transcript here", question_condition="pooled_t17")
    with pytest.raises(ValueError):
        render_user_prompt_text(config, "transcript here", question_condition="bogus")


def test_legacy_untagged_template_is_byte_identical() -> None:
    config = _config("audio_text", tagged=False)
    before = render_user_prompt_text(config, "transcript here")
    after_none = render_user_prompt_text(config, "transcript here", question_condition=None)
    after_pos = render_user_prompt_text(
        config, "transcript here", question_condition="pos_only_t17"
    )
    assert before == after_none == after_pos


def test_audio_only_tagged_template_carries_tag_without_transcript() -> None:
    config = _config("audio_only", tagged=True)
    text = render_user_prompt_text(config, "", question_condition="negative_only_t17")
    assert QUESTION_CONTEXT_SENTENCES["negative_only_t17"] in text
    assert "The transcript of the subject's speech is:" not in text


def test_base_example_threads_row_variant() -> None:
    config = _config("audio_text", tagged=True)
    example, _ = _base_example_from_row(
        _row("s1-1-1", "s1", "pos_only_t17"), config, 0
    )
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] in example["prompt_text"]
    assert example["dataset_variant"] == "pos_only_t17"
    assert example["question_condition"] == "pos_only_t17"


def test_text_only_pooled_builds_separate_per_condition_examples() -> None:
    config = _config("text_only", tagged=True)
    rows = [
        _row("s1-1-1", "s1", "pos_only_t17", transcript="positive words here"),
        _row("s1-2-1", "s1", "negative_only_t17", transcript="negative words here"),
    ]
    examples = build_examples(rows, config, partition_name="final_eval")
    assert len(examples) == 2
    by_condition = {example["question_condition"]: example for example in examples}
    assert set(by_condition) == {"pos_only_t17", "negative_only_t17"}
    assert (
        QUESTION_CONTEXT_SENTENCES["pos_only_t17"]
        in by_condition["pos_only_t17"]["prompt_text"]
    )
    assert (
        QUESTION_CONTEXT_SENTENCES["negative_only_t17"]
        in by_condition["negative_only_t17"]["prompt_text"]
    )
    assert "positive words here" in by_condition["pos_only_t17"]["prompt_text"]
    assert "negative words here" in by_condition["negative_only_t17"]["prompt_text"]
    # distinct samples, shared subject for pair scoring
    sample_ids = {example["sample_id"] for example in examples}
    assert len(sample_ids) == 2
    assert {example["subject_id"] for example in examples} == {"s1"}
