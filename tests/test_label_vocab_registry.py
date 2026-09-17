"""Registry and legend-rendering tests for the label-vocabulary families."""

import pytest

from src.utils import (
    LABEL_VOCAB_VERSION_BINARY_01,
    LABEL_VOCAB_VERSION_LEGACY,
    LABEL_VOCAB_VERSION_SHORT_AB,
    LABEL_VOCAB_VERSION_TRUEFALSE,
    LABEL_VOCAB_VERSION_YESNO,
    LEGEND_STYLE_LABEL_VOCAB_VERSIONS,
    SUPPORTED_LABEL_VOCAB_VERSIONS,
    _default_label_config_for_version,
    prompt_label_instruction,
    resolve_label_config,
)

LEGEND_VOCABS = {
    LABEL_VOCAB_VERSION_SHORT_AB: ("A", "B"),
    LABEL_VOCAB_VERSION_BINARY_01: ("1", "0"),
    LABEL_VOCAB_VERSION_TRUEFALSE: ("True", "False"),
    LABEL_VOCAB_VERSION_YESNO: ("Yes", "No"),
}


def test_new_vocab_versions_are_registered() -> None:
    for version in (
        LABEL_VOCAB_VERSION_BINARY_01,
        LABEL_VOCAB_VERSION_TRUEFALSE,
        LABEL_VOCAB_VERSION_YESNO,
    ):
        assert version in SUPPORTED_LABEL_VOCAB_VERSIONS
    assert set(LEGEND_STYLE_LABEL_VOCAB_VERSIONS) == set(LEGEND_VOCABS)


def test_default_label_configs_for_new_vocabs() -> None:
    for version, (positive, negative) in LEGEND_VOCABS.items():
        resolved = _default_label_config_for_version(version)
        assert resolved["label_vocab_version"] == version
        assert resolved["internal_positive_label"] == positive
        assert resolved["internal_negative_label"] == negative
        assert resolved["external_positive_label"] == "Depressed"
        assert resolved["external_negative_label"] == "Non-depressed"


def test_legend_rendering_for_every_legend_style_vocab() -> None:
    for version, (positive, negative) in LEGEND_VOCABS.items():
        config = {"labels": {"label_vocab_version": version}}
        assert prompt_label_instruction(config) == (
            "Use this label legend:\n"
            f"{positive} = Depressed\n"
            f"{negative} = Non-depressed\n"
            f"Answer with exactly one label: {positive} or {negative}."
        )


def test_legacy_vocab_keeps_answer_only_instruction() -> None:
    config = {"labels": {"label_vocab_version": LABEL_VOCAB_VERSION_LEGACY}}
    assert prompt_label_instruction(config) == (
        "Answer with exactly one label: Depressed or Non-depressed."
    )


def test_explicit_tokens_for_new_vocab_are_preserved() -> None:
    resolved = resolve_label_config(
        {
            "labels": {
                "label_vocab_version": LABEL_VOCAB_VERSION_BINARY_01,
                "internal_positive_label": "1",
                "internal_negative_label": "0",
            }
        }
    )
    assert resolved["internal_positive_label"] == "1"
    assert resolved["internal_negative_label"] == "0"


def test_unsupported_vocab_version_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported labels.label_vocab_version"):
        resolve_label_config({"labels": {"label_vocab_version": "not_a_label_vocab"}})
    with pytest.raises(ValueError, match="Unsupported labels.label_vocab_version"):
        _default_label_config_for_version("not_a_label_vocab")
