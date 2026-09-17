from pathlib import Path

from src.utils import load_yaml, prompt_label_instruction, resolve_label_config


ROOT = Path(__file__).resolve().parents[1]


def test_daic_ab_likelihood_changes_only_label_and_scoring_protocol() -> None:
    baseline = load_yaml(ROOT / "configs/main/daic_text_only_harmonized_selmacrof1_tf.yaml")
    candidate = load_yaml(ROOT / "configs/labels/daic_text_only_harmonized_selmacrof1_likelihood_ab.yaml")

    labels = resolve_label_config(candidate)
    assert (labels["internal_positive_label"], labels["internal_negative_label"]) == ("A", "B")
    assert prompt_label_instruction(candidate) == (
        "Use this label legend:\nA = Depressed\nB = Non-depressed\n"
        "Answer with exactly one label: A or B."
    )
    assert candidate["evaluation"]["sample_prediction_mode"] == "likelihood"
    assert candidate["evaluation"]["headline_mode"] == "likelihood"
    assert candidate["evaluation"]["aggregation_level"] == "subject"

    for key in ("dataset", "seed", "protocol_id", "manifest_variant", "split", "data", "training", "lora", "prompt"):
        assert candidate[key] == baseline[key], key
