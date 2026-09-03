from __future__ import annotations

import pytest

from src.data.runtime import QUESTION_CONTEXT_SENTENCES, build_examples, render_user_prompt_text
from src.utils import load_yaml


CONFIG = load_yaml(
    __import__("pathlib").Path(__file__).parents[1]
    / "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml"
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "turkish", "dataset_variant": condition, "sample_id": f"{condition}-s{subject}",
            "subject_id": f"s{subject}", "label": 1, "label_text": "Depressed",
            "transcript": f"transcript-{condition}-{subject}", "audio_path": "",
        }
        for subject in ("1", "2")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]


def test_pooled_text_examples_are_exactly_one_pair_per_subject() -> None:
    examples = build_examples(_rows(), CONFIG, "train")
    assert len(examples) == 4
    assert {example["sample_id"] for example in examples} == {
        "s1::pos_only_t17", "s1::negative_only_t17", "s2::pos_only_t17", "s2::negative_only_t17",
    }
    for example in examples:
        condition = str(example["question_condition"])
        sentence = QUESTION_CONTEXT_SENTENCES[condition]
        user_text = str(example["prompt_user_text"])
        assert sentence in user_text
        assert user_text.index(sentence) < user_text.index("The transcript of the subject's speech is:")
        assert user_text.split("The transcript of the subject's speech is:", 1)[0].rstrip().endswith(sentence)


def test_tagged_prompt_rejects_missing_or_unknown_condition() -> None:
    with pytest.raises(ValueError, match="question_condition"):
        render_user_prompt_text(CONFIG, "x", question_condition=None)
    with pytest.raises(ValueError, match="question_condition"):
        render_user_prompt_text(CONFIG, "x", question_condition="other")


def test_untagged_existing_prompt_stays_untagged() -> None:
    config = {**CONFIG, "prompt": {**CONFIG["prompt"], "user_template": "{transcript_block}Based on the transcript, decide."}}
    rendered = render_user_prompt_text(config, "x", question_condition="pos_only_t17")
    assert "positive interview questions" not in rendered
