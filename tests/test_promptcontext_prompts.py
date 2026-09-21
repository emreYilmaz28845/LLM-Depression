"""Rendered-prompt audits for the prompt-context family.

These tests render the exact text every cell sends to the model and check the
properties the runbook requires: the shared instruction is identical across
datasets, each example carries only its own recording context, a modality never
claims an input it does not receive, the label instruction is unchanged, the
Turkish condition sentence is versioned, and Qwen and Gemma see the same
semantic system/user content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_promptcontext_configs import (
    COMMON_INSTRUCTION,
    DATASET_CONTEXT,
    MODALITIES,
    STANDALONE_DATASET_ORDER,
    merged_target_name,
    source_name,
    target_name,
)
from src.data.runtime import (
    QUESTION_CONTEXT_SENTENCES,
    QUESTION_CONTEXT_SENTENCE_SETS,
    build_examples,
    build_prompt_text,
    render_user_prompt_text,
    resolve_audio_placeholder,
    resolve_question_context_sentences,
)
from src.model.gemma4_io import _gemma4_messages
from src.utils import INPUT_MODALITY_TEXT_ONLY, load_yaml, resolve_input_modality

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
MERGED = ROOT / "configs/experiments/merged"
TRANSCRIPT = "I have been sleeping badly and I do not enjoy things anymore."
LABEL_INSTRUCTION = "Answer with exactly one label: Depressed or Non-depressed."

STANDALONE_CELLS = [
    (dataset, modality, gemma)
    for dataset in STANDALONE_DATASET_ORDER
    for modality in MODALITIES
    for gemma in (False, True)
]


def _config(dataset: str, modality: str, gemma: bool) -> dict:
    return load_yaml(MAIN / target_name(dataset, modality, gemma=gemma))


def _rendered(config: dict) -> tuple[str, str]:
    modality = resolve_input_modality(config)
    question_condition = "pos_only_t17" if config["dataset"] == "turkish" else None
    user_text = render_user_prompt_text(
        config, TRANSCRIPT, question_condition=question_condition
    )
    prompt_text = build_prompt_text(
        system_prompt=config["prompt"]["system"],
        user_text=user_text,
        num_audios=1 if modality != INPUT_MODALITY_TEXT_ONLY else 0,
        use_audio=modality != INPUT_MODALITY_TEXT_ONLY,
        audio_placeholder=resolve_audio_placeholder(config),
    )
    return user_text, prompt_text


@pytest.mark.parametrize("dataset,modality,gemma", STANDALONE_CELLS)
def test_every_cell_renders_the_common_instruction_and_its_own_context(
    dataset: str, modality: str, gemma: bool
) -> None:
    config = _config(dataset, modality, gemma)
    user_text, prompt_text = _rendered(config)
    system = config["prompt"]["system"]
    assert system == f"{COMMON_INSTRUCTION}\n\n{DATASET_CONTEXT[dataset]}"
    assert prompt_text.startswith("<|im_start|>system\n")
    assert COMMON_INSTRUCTION in prompt_text
    assert DATASET_CONTEXT[dataset] in prompt_text
    for other, block in DATASET_CONTEXT.items():
        if other != dataset:
            assert block not in prompt_text
    assert LABEL_INSTRUCTION in prompt_text
    assert user_text.strip().endswith(LABEL_INSTRUCTION)


@pytest.mark.parametrize("dataset,modality,gemma", STANDALONE_CELLS)
def test_modalities_never_claim_inputs_they_do_not_receive(
    dataset: str, modality: str, gemma: bool
) -> None:
    config = _config(dataset, modality, gemma)
    user_text, prompt_text = _rendered(config)
    placeholder = resolve_audio_placeholder(config)
    transcript_marker = "The transcript of the subject's speech is:"
    audio_marker = "The subject's speech audio is provided."
    if modality == "text_only":
        assert placeholder not in prompt_text
        assert audio_marker not in user_text
        assert transcript_marker in user_text
        assert "Based on the transcript" in user_text
    else:
        assert prompt_text.count(placeholder) == 1
        assert audio_marker in user_text
        assert "Based on the audio and transcript" in user_text or "Based on the audio" in user_text
        if modality == "audio_only":
            assert transcript_marker not in user_text
            assert TRANSCRIPT not in user_text
        else:
            assert transcript_marker in user_text


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("gemma", (False, True))
def test_backends_receive_the_same_semantic_prompt(modality: str, gemma: bool) -> None:
    qwen = _config("cmdc", modality, False)
    target = qwen if not gemma else _config("cmdc", modality, True)
    assert target["prompt"] == qwen["prompt"]
    assert render_user_prompt_text(target, TRANSCRIPT) == render_user_prompt_text(
        qwen, TRANSCRIPT
    )
    assert target["prompt"]["system"] == qwen["prompt"]["system"]
    messages = _gemma4_messages(
        target["prompt"]["system"], render_user_prompt_text(target, TRANSCRIPT), modality
    )
    assert messages[0] == {"role": "system", "content": qwen["prompt"]["system"]}
    content = messages[1]["content"]
    if modality == INPUT_MODALITY_TEXT_ONLY:
        assert isinstance(content, str)
    else:
        assert [item["type"] for item in content] == ["audio", "text"]
        assert content[1]["text"] == render_user_prompt_text(qwen, TRANSCRIPT)


def test_turkish_condition_sentences_are_versioned() -> None:
    assert set(QUESTION_CONTEXT_SENTENCE_SETS) == {"legacy_v1", "promptcontext_v1"}
    assert QUESTION_CONTEXT_SENTENCE_SETS["legacy_v1"] is QUESTION_CONTEXT_SENTENCES
    new = QUESTION_CONTEXT_SENTENCE_SETS["promptcontext_v1"]
    assert new["pos_only_t17"].startswith("Positive set: This recording answers a question")
    assert new["negative_only_t17"].startswith("Negative set: This recording answers a question")
    assert "is not the participant's depression label" in new["pos_only_t17"]
    assert "is not the participant's depression label" in new["negative_only_t17"]
    for condition in ("pos_only_t17", "negative_only_t17"):
        assert new[condition] != QUESTION_CONTEXT_SENTENCES[condition]


def test_turkish_cells_use_the_new_sentences_and_legacy_configs_keep_the_old_ones() -> None:
    transcript_marker = "The transcript of the subject's speech is:"
    for modality in MODALITIES:
        for gemma in (False, True):
            config = _config("turkish", modality, gemma)
            sentences = resolve_question_context_sentences(config)
            assert sentences is QUESTION_CONTEXT_SENTENCE_SETS["promptcontext_v1"]
            for condition, sentence in sentences.items():
                user_text = render_user_prompt_text(
                    config, TRANSCRIPT, question_condition=condition
                )
                assert sentence in user_text
                if modality != "audio_only":
                    assert user_text.index(sentence) < user_text.index(transcript_marker)
    legacy = load_yaml(MAIN / source_name("turkish", "text_only"))
    assert resolve_question_context_sentences(legacy) is QUESTION_CONTEXT_SENTENCES
    rendered = render_user_prompt_text(
        legacy, TRANSCRIPT, question_condition="pos_only_t17"
    )
    assert QUESTION_CONTEXT_SENTENCES["pos_only_t17"] in rendered
    assert QUESTION_CONTEXT_SENTENCE_SETS["promptcontext_v1"]["pos_only_t17"] not in rendered


def test_unknown_question_context_version_is_refused() -> None:
    config = _config("turkish", "text_only", False)
    config = {**config, "prompt": {**config["prompt"], "question_context_version": "nope"}}
    with pytest.raises(ValueError, match="question_context_version"):
        resolve_question_context_sentences(config)


def test_pooled_builder_emits_both_conditions_with_the_new_sentence() -> None:
    config = _config("turkish", "text_only", False)
    rows = [
        {
            "dataset": "turkish",
            "dataset_variant": condition,
            "sample_id": f"{subject}-{condition}",
            "subject_id": subject,
            "label": 1,
            "label_text": "Depressed",
            "transcript": f"{condition} transcript for {subject}",
            "audio_path": "",
        }
        for subject in ("s1", "s2")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]
    examples = build_examples(rows, config, "val")
    assert len(examples) == 4
    sentences = QUESTION_CONTEXT_SENTENCE_SETS["promptcontext_v1"]
    for example in examples:
        condition = str(example["question_condition"])
        assert sentences[condition] in example["prompt_user_text"]
        assert example["prompt_system_text"] == f"{COMMON_INSTRUCTION}\n\n{DATASET_CONTEXT['turkish']}"
        assert DATASET_CONTEXT["turkish"] in example["prompt_text"]
        assert example["prompt_text"].count("<|im_start|>assistant") == 1
