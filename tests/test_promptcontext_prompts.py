from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_promptcontext_qwen38_configs import CELLS
from src.data.prompt_context import (
    DATASET_CONTEXT_BLOCKS,
    PROMPT_CONTEXT_VERSION,
    QUESTION_CONTEXT_SENTENCE_SETS,
    SHARED_INSTRUCTION_TEXT_ONLY,
    resolve_question_context_sentences,
    resolve_system_prompt,
)
from src.data.runtime import QUESTION_CONTEXT_SENTENCES, build_examples, render_user_prompt_text

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
LEGACY_POOLED = MAIN / "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml"
LEGACY_TEXT_ONLY = MAIN / "d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml"

NEW_CONFIGS = {cell[0]: MAIN / cell[2] for cell in CELLS}
CONTEXT_KEYS = {cell[0]: cell[4] for cell in CELLS}

# A single distinctive sentence per dataset block, used to prove that exactly the
# selected dataset's context reaches the prompt.
CONTEXT_MARKERS = {
    "androids": "This is an Italian interview with a human interviewer.",
    "d3tec": "This is Spanish speech from a non-interactive slideshow of 27 tasks.",
    "daic": "This is an English semi-structured interview with the virtual interviewer Ellie.",
    "cmdc": "This is Mandarin speech from a face-to-face, symptom-focused interview.",
    "turkish_pooled": "This is Turkish speech from one of two question sets completed by the same participants.",
}

AUDIO_CLAIMS = (
    "speech audio",
    "audio window",
    "vocal feature",
    "The audio is",
    "audio is provided",
    "Audio 1:",
    "<|AUDIO|>",
    "<|audio_bos|>",
    "<|audio_start|>",
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _new_config(cell_id: str) -> dict:
    return _load(NEW_CONFIGS[cell_id])


def _pooled_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "turkish",
            "dataset_variant": condition,
            "sample_id": f"{condition}-s{subject}",
            "subject_id": f"s{subject}",
            "label": 1,
            "label_text": "Depressed",
            "transcript": f"transcript-{condition}-{subject}",
            "audio_path": "",
        }
        for subject in ("1", "2")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]


def test_legacy_configs_keep_their_inline_prompt_and_sentences() -> None:
    for path in (LEGACY_POOLED, LEGACY_TEXT_ONLY):
        config = _load(path)
        assert "version" not in config["prompt"]
        assert resolve_system_prompt(config) == config["prompt"]["system"]
        assert "You are a psychologist analyzing transcript information for depression screening." in resolve_system_prompt(config)

    pooled = _load(LEGACY_POOLED)
    assert resolve_question_context_sentences(pooled) is QUESTION_CONTEXT_SENTENCE_SETS["legacy_v1"]
    rendered = render_user_prompt_text(pooled, "x", question_condition="pos_only_t17")
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] in rendered
    assert "This recording answers a question from the positive set." not in rendered


def test_legacy_question_context_wording_is_frozen() -> None:
    assert QUESTION_CONTEXT_SENTENCE_SETS["legacy_v1"] == {
        "pos_only_t17": "The following speech is the subject's response to positive interview questions.",
        "negative_only_t17": "The following speech is the subject's response to negative interview questions.",
    }
    assert QUESTION_CONTEXT_SENTENCES == QUESTION_CONTEXT_SENTENCE_SETS["legacy_v1"]


def test_new_configs_render_the_shared_instruction_and_exactly_one_context() -> None:
    for cell_id, path in NEW_CONFIGS.items():
        config = _load(path)
        assert "system" not in config["prompt"], cell_id
        assert config["prompt"]["version"] == PROMPT_CONTEXT_VERSION
        assert config["prompt"]["dataset_context"] == CONTEXT_KEYS[cell_id]
        system_prompt = resolve_system_prompt(config)
        assert system_prompt.startswith(SHARED_INSTRUCTION_TEXT_ONLY)
        assert SHARED_INSTRUCTION_TEXT_ONLY in system_prompt
        for other_key, marker in CONTEXT_MARKERS.items():
            if other_key == CONTEXT_KEYS[cell_id]:
                assert system_prompt.count(marker) == 1
                continue
            assert marker not in system_prompt, f"{cell_id} leaked {other_key} context"
        assert system_prompt.count("Recording context:") == 1


def test_text_only_prompts_never_claim_audio() -> None:
    for cell_id, path in NEW_CONFIGS.items():
        config = _load(path)
        user_text = render_user_prompt_text(config, "a transcript", question_condition="pos_only_t17")
        combined = resolve_system_prompt(config) + "\n" + user_text
        for claim in AUDIO_CLAIMS:
            assert claim not in combined, f"{cell_id}: audio claim {claim!r}"
        assert "Based on the transcript, determine whether" in user_text
        assert not user_text.startswith("The subject's speech audio is provided")


def test_transcript_placement_and_label_contract() -> None:
    for cell_id, path in NEW_CONFIGS.items():
        config = _load(path)
        user_text = render_user_prompt_text(config, "the transcript body", question_condition="pos_only_t17")
        assert "The transcript of the subject's speech is:\nthe transcript body" in user_text
        assert user_text.index("the transcript body") < user_text.index("Based on the transcript")
        assert "Depressed" in user_text and "Non-depressed" in user_text
        assert config["labels"]["label_vocab_version"] == "legacy_english_labels"


def test_new_pooled_config_selects_the_positive_and_negative_sentences() -> None:
    config = _new_config("turkish_pooled")
    examples = build_examples(_pooled_rows(), config, "train")
    assert len(examples) == 4
    sentences = QUESTION_CONTEXT_SENTENCE_SETS[PROMPT_CONTEXT_VERSION]
    for example in examples:
        condition = str(example["question_condition"])
        user_text = str(example["prompt_user_text"])
        assert sentences[condition] in user_text
        other = "negative_only_t17" if condition == "pos_only_t17" else "pos_only_t17"
        assert sentences[other] not in user_text
        assert user_text.index(sentences[condition]) < user_text.index(
            "The transcript of the subject's speech is:"
        )
        assert "BDI ≥ 17" in resolve_system_prompt(config)


def test_new_pooled_config_fails_closed_on_missing_or_unknown_condition() -> None:
    config = _new_config("turkish_pooled")
    with pytest.raises(ValueError, match="question_condition"):
        render_user_prompt_text(config, "x", question_condition=None)
    with pytest.raises(ValueError, match="question_condition"):
        render_user_prompt_text(config, "x", question_condition="other")


def test_pooled_conditions_share_one_subject_identity_per_label() -> None:
    config = _new_config("turkish_pooled")
    examples = build_examples(_pooled_rows(), config, "train_inner")
    for subject in ("s1", "s2"):
        subject_examples = [example for example in examples if example["subject_id"] == subject]
        assert len(subject_examples) == 2
        assert {example["question_condition"] for example in subject_examples} == {
            "pos_only_t17",
            "negative_only_t17",
        }
        assert {example["label"] for example in subject_examples} == {1}
        assert all(example["sample_id"].startswith(f"{subject}::") for example in subject_examples)


def test_unknown_prompt_version_context_key_and_sentence_set_fail_closed() -> None:
    config = _new_config("d3tec")
    broken_version = {**config, "prompt": {**config["prompt"], "version": "other"}}
    with pytest.raises(ValueError, match="prompt.version"):
        resolve_system_prompt(broken_version)
    broken_context = {**config, "prompt": {**config["prompt"], "dataset_context": "other"}}
    with pytest.raises(ValueError, match="dataset_context"):
        resolve_system_prompt(broken_context)
    missing_context = {**config, "prompt": {k: v for k, v in config["prompt"].items() if k != "dataset_context"}}
    with pytest.raises(ValueError, match="dataset_context"):
        resolve_system_prompt(missing_context)
    broken_sentences = {
        **config,
        "prompt": {**config["prompt"], "question_context_version": "other"},
    }
    with pytest.raises(ValueError, match="question_context_version"):
        resolve_question_context_sentences(broken_sentences)


def test_training_and_evaluation_render_the_same_prompt() -> None:
    config = _new_config("daic")
    rows = [
        {
            "dataset": "daic",
            "sample_id": "p1_c000",
            "subject_id": "1",
            "chunk_index": 0,
            "label": 0,
            "label_text": "Non-depressed",
            "transcript": "hello",
            "full_participant_transcript": "hello",
            "chunk_transcript": "hello",
            "full_participant_transcript_sha256": "0" * 64,
            "participant_sample_count": 1,
            "audio_path": "x.wav",
            "audio_spans": [{"start_frame": 0, "end_frame": 1}],
        }
    ]
    train = build_examples(rows, config, "train_inner")
    evaluation = build_examples(rows, config, "final_eval")
    assert train and evaluation
    assert train[0]["prompt_text"] == evaluation[0]["prompt_text"]
    assert train[0]["prompt_system_text"] == resolve_system_prompt(config)
    assert evaluation[0]["prompt_system_text"] == resolve_system_prompt(config)


def test_rendering_is_deterministic() -> None:
    for path in NEW_CONFIGS.values():
        config = _load(path)
        first = resolve_system_prompt(config)
        second = resolve_system_prompt(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert first == second
        assert render_user_prompt_text(config, "body", question_condition="pos_only_t17") == render_user_prompt_text(
            config, "body", question_condition="pos_only_t17"
        )
