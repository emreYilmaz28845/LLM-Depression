"""Structural tests for the DAIC label-vocabulary likelihood configs.

Each generated config must differ from its A/B source only in the labels block,
the file name, recipe_id, and run_root.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from src.utils import load_yaml, prompt_label_instruction

ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "configs" / "labels"
MODALITIES = ("audio_only", "text_only", "audio_text")
VOCABS = {
    "01": ("binary_01_labels", "1", "0"),
    "truefalse": ("truefalse_labels", "True", "False"),
    "yesno": ("yesno_labels", "Yes", "No"),
}


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("tag", tuple(VOCABS))
def test_label_vocab_config_differs_from_ab_source_only_in_declared_fields(
    modality: str, tag: str
) -> None:
    source_path = LABELS_DIR / f"daic_{modality}_harmonized_selmacrof1_likelihood_ab_v1.yaml"
    candidate_path = LABELS_DIR / f"daic_{modality}_harmonized_selmacrof1_likelihood_{tag}_v1.yaml"
    source = load_yaml(source_path)
    candidate = load_yaml(candidate_path)
    vocab_id, positive, negative = VOCABS[tag]

    expected = deepcopy(source)
    expected["recipe_id"] = (
        f"harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_{tag}_v1"
    )
    expected["labels"] = {
        "label_vocab_version": vocab_id,
        "internal_positive_label": positive,
        "internal_negative_label": negative,
        "external_positive_label": "Depressed",
        "external_negative_label": "Non-depressed",
    }
    source_run_root = Path(source["output_dirs"]["run_root"])
    assert source_run_root.parent.name == modality
    expected["output_dirs"]["run_root"] = str(
        source_run_root.parents[2] / "label_vocab_v1" / modality / "daic"
    )

    assert candidate == expected
    assert prompt_label_instruction(candidate) == (
        "Use this label legend:\n"
        f"{positive} = Depressed\n"
        f"{negative} = Non-depressed\n"
        f"Answer with exactly one label: {positive} or {negative}."
    )
    assert candidate["evaluation"]["sample_prediction_mode"] == "likelihood"
    assert candidate["evaluation"]["headline_mode"] == "likelihood"


def test_nine_label_vocab_configs_exist_under_configs_labels() -> None:
    names = [
        path.name
        for path in LABELS_DIR.glob("daic_*_harmonized_selmacrof1_likelihood_*_v1.yaml")
    ]
    generated = [
        name
        for name in names
        if any(name.endswith(f"likelihood_{tag}_v1.yaml") for tag in VOCABS)
    ]
    assert len(generated) == 9, generated
