"""The versioned English-translation notice in rendered prompts.

Native prompts stay byte-identical; an English cell renders one notice block
immediately before the transcript block, exactly once, in training, evaluation
and hidden-extraction example building alike (all three go through
``build_examples``). The notice must never claim an audio input that text-only
does not receive, and a config that declares it must fail closed against a
native (non-English) transcript row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.data.prompt_context import (
    TRANSLATION_NOTICE_SENTENCES,
    TRANSLATION_NOTICE_VERSION,
    prompt_context_record,
    resolve_translation_notice,
)
from src.data.runtime import build_examples, render_user_prompt_text

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"

EN_CELLS = {
    "d3tec_text_only": "d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
    "d3tec_audio_text": "d3tec_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
    "androids_text_only": "androids_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
    "androids_audio_text": "androids_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
    "cmdc_text_only": "cmdc_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
    "cmdc_audio_text": "cmdc_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
    "turkish_text_only": "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
    "turkish_audio_text": "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
}
NATIVE_COUNTERPARTS = {
    "d3tec_text_only": "d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "d3tec_audio_text": "d3tec_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "androids_text_only": "androids_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "androids_audio_text": "androids_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "daic_audio_text": "daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "turkish_audio_text": "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
}
TRANSCRIPT_MARKER = "The transcript of the subject's speech is:"

# The prompt hash recorded by the existing seed-1337 DAIC audio+text run. A
# native prompt that changes would invalidate that run's reuse classification,
# so this pins the native wording.
RECORDED_DAIC_AUDIO_TEXT_PROMPT_SHA256 = (
    "ed0c9145f184c26cd70a7627c6b4ff1e9ce1f9272a8d4683839b6fe41463b0fe"
)


def _load(name: str) -> dict:
    return yaml.safe_load((MAIN / name).read_text(encoding="utf-8"))


def _row(*, dataset: str, variant: str | None = None) -> dict:
    row = {
        "dataset": dataset,
        "sample_id": f"{dataset}-s1",
        "subject_id": "s1",
        "label": 1,
        "label_text": "Depressed",
        "transcript": "a translated transcript",
        "audio_path": "",
        "transcript_variant": "english",
        "language": "en",
    }
    if variant is not None:
        row["dataset_variant"] = variant
    return row


@pytest.mark.parametrize("cell_id", sorted(EN_CELLS))
def test_notice_renders_once_immediately_before_the_transcript_block(cell_id: str) -> None:
    config = _load(EN_CELLS[cell_id])
    notice = resolve_translation_notice(config)
    assert notice == TRANSLATION_NOTICE_SENTENCES[TRANSLATION_NOTICE_VERSION][
        "text_only" if "text_only" in cell_id else "audio_text"
    ]
    user_text = render_user_prompt_text(
        config, "a translated transcript", question_condition="pos_only_t17"
    )
    assert user_text.count(notice) == 1
    assert user_text.index(notice) < user_text.index(TRANSCRIPT_MARKER)
    assert user_text.index(notice) + len(notice) + 1 == user_text.index(TRANSCRIPT_MARKER)
    # The native instruction, the dataset context and the label instruction are
    # untouched: only the notice was added. The instruction lives in the system
    # prompt, which carries no notice at all.
    system_prompt = prompt_context_record(config)["system_prompt"]
    assert "You are classifying a participant's depression study label" in system_prompt
    assert notice not in system_prompt
    assert "Answer with exactly one label" in user_text


@pytest.mark.parametrize("cell_id", ["d3tec_text_only", "cmdc_text_only", "turkish_text_only"])
def test_text_only_notice_never_claims_an_audio_input(cell_id: str) -> None:
    notice = resolve_translation_notice(_load(EN_CELLS[cell_id]))
    assert "audio" not in notice.lower()
    user_text = render_user_prompt_text(
        _load(EN_CELLS[cell_id]), "a translated transcript", question_condition="pos_only_t17"
    )
    assert "The audio remains in the original language." not in user_text


@pytest.mark.parametrize("cell_id", ["d3tec_audio_text", "androids_audio_text", "turkish_audio_text"])
def test_audio_text_notice_states_the_audio_language(cell_id: str) -> None:
    notice = resolve_translation_notice(_load(EN_CELLS[cell_id]))
    assert "The audio remains in the original language." in notice


def test_native_counterparts_carry_no_notice() -> None:
    for cell_id, name in sorted(NATIVE_COUNTERPARTS.items()):
        config = _load(name)
        assert resolve_translation_notice(config) is None, cell_id
        user_text = render_user_prompt_text(
            config, "a native transcript", question_condition="pos_only_t17"
        )
        assert "English translation" not in user_text, cell_id
        record = prompt_context_record(config)
        assert record["translation_notice"] is None
        assert record["translation_notice_sha256"] is None


def test_native_prompt_hash_matches_the_recorded_seed_1337_run() -> None:
    config = _load(NATIVE_COUNTERPARTS["daic_audio_text"])
    record = prompt_context_record(config)
    assert record["system_prompt_sha256"] == RECORDED_DAIC_AUDIO_TEXT_PROMPT_SHA256


def test_notice_is_recorded_in_the_prompt_context_record() -> None:
    config = _load(EN_CELLS["d3tec_audio_text"])
    record = prompt_context_record(config)
    assert record["translation_notice_version"] == TRANSLATION_NOTICE_VERSION
    assert record["translation_notice"] == resolve_translation_notice(config)
    assert record["translation_notice_sha256"] == hashlib.sha256(
        record["translation_notice"].encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("condition", ["pos_only_t17", "negative_only_t17"])
def test_turkish_question_context_survives_the_notice(condition: str) -> None:
    config = _load(EN_CELLS["turkish_audio_text"])
    user_text = render_user_prompt_text(config, "çeviri", question_condition=condition)
    notice = resolve_translation_notice(config)
    assert user_text.count(notice) == 1
    expected = {
        "pos_only_t17": "Positive set: This recording answers a question from the positive set.",
        "negative_only_t17": "Negative set: This recording answers a question from the negative set.",
    }[condition]
    unexpected = {
        "pos_only_t17": "Negative set:",
        "negative_only_t17": "Positive set:",
    }[condition]
    assert expected in user_text
    assert unexpected not in user_text
    # The question context still precedes the transcript block, and the notice
    # sits directly above the transcript text itself.
    assert user_text.index(expected) < user_text.index(notice) < user_text.index(TRANSCRIPT_MARKER)


def test_train_eval_and_hidden_extraction_render_the_same_notice_prompt() -> None:
    config = _load(EN_CELLS["androids_text_only"])
    rows = [_row(dataset="androids_interview")]
    train = build_examples(rows, config, "train_inner")
    evaluation = build_examples(rows, config, "final_eval")
    assert train[0]["prompt_text"] == evaluation[0]["prompt_text"]
    notice = resolve_translation_notice(config)
    assert train[0]["prompt_text"].count(notice) == 1
    assert train[0]["prompt_user_text"].count(notice) == 1
    assert train[0]["prompt_system_text"] == evaluation[0]["prompt_system_text"]
    assert notice not in train[0]["prompt_system_text"]


def test_unknown_notice_version_fails_closed() -> None:
    config = _load(EN_CELLS["cmdc_text_only"])
    broken = {**config, "prompt": {**config["prompt"], "translation_notice_version": "other"}}
    with pytest.raises(ValueError, match="translation_notice_version"):
        resolve_translation_notice(broken)


def test_notice_on_a_native_transcript_config_fails_closed() -> None:
    config = _load(NATIVE_COUNTERPARTS["d3tec_audio_text"])
    broken = {**config, "prompt": {**config["prompt"], "translation_notice_version": TRANSLATION_NOTICE_VERSION}}
    with pytest.raises(ValueError, match="transcripts.variant"):
        resolve_translation_notice(broken)


def test_notice_on_audio_only_fails_closed() -> None:
    config = _load(EN_CELLS["d3tec_audio_text"])
    broken = {
        **config,
        "data": {**config["data"], "use_text": False, "use_audio": True},
    }
    with pytest.raises(ValueError, match="modality"):
        resolve_translation_notice(broken)


def test_template_without_a_transcript_block_fails_closed() -> None:
    config = _load(EN_CELLS["cmdc_text_only"])
    broken = {
        **config,
        "prompt": {**config["prompt"], "user_template": "Judge the participant: {label_descriptor}."},
    }
    with pytest.raises(ValueError, match="transcript_block"):
        render_user_prompt_text(broken, "text", question_condition="pos_only_t17")


def test_native_fallback_rows_fail_closed_for_a_notice_config() -> None:
    config = _load(EN_CELLS["cmdc_text_only"])
    native_row = _row(dataset="cmdc")
    native_row["transcript_variant"] = "original"
    with pytest.raises(ValueError, match="transcript_variant"):
        build_examples([native_row], config, "train_inner")
    missing_marker = _row(dataset="cmdc")
    del missing_marker["transcript_variant"]
    with pytest.raises(ValueError, match="transcript_variant"):
        build_examples([missing_marker], config, "train_inner")
    wrong_language = _row(dataset="cmdc")
    wrong_language["language"] = "zh"
    with pytest.raises(ValueError, match="language"):
        build_examples([wrong_language], config, "train_inner")


def test_native_configs_accept_native_rows() -> None:
    config = _load(NATIVE_COUNTERPARTS["d3tec_text_only"])
    native_row = _row(dataset="d3tec")
    native_row["transcript_variant"] = "original"
    native_row["language"] = "es"
    examples = build_examples([native_row], config, "train_inner")
    assert len(examples) == 1
    assert "English translation" not in examples[0]["prompt_text"]
