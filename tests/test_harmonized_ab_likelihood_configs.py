from copy import deepcopy
from pathlib import Path

from src.utils import load_yaml, prompt_label_instruction


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
ARCHIVE_MAIN = ROOT / "configs/archive/pre_likelihood_20260917/main"
LABELS_DIR = ROOT / "configs/labels"
RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_ab_v1"
LABELS = {
    "label_vocab_version": "short_internal_ab_labels",
    "internal_positive_label": "A",
    "internal_negative_label": "B",
    "external_positive_label": "Depressed",
    "external_negative_label": "Non-depressed",
}


def _key(config: dict) -> tuple:
    data = config["data"]
    return (
        config["dataset"], config.get("dataset_variant"),
        bool(data["use_audio"]), bool(data["use_text"]),
    )


def test_core_ab_family_preserves_harmonized_data_and_training_recipe() -> None:
    baselines = {}
    for path in ARCHIVE_MAIN.glob("*harmonized_selmacrof1_tf*.yaml"):
        name = path.name
        if (
            name.endswith("_en.yaml") or "_gemma4_12b" in name
            or "_officialdev" in name or "turkish_negative_only" in name
            or "turkish_pooled" in name or name.startswith("turkish_t17_")
        ):
            continue
        config = load_yaml(path)
        assert _key(config) not in baselines
        baselines[_key(config)] = config
    assert len(baselines) == 15

    paths = sorted(LABELS_DIR.glob("*likelihood_ab_v1.yaml"))
    assert len(paths) == 15
    seen = set()
    run_roots = set()
    for path in paths:
        candidate = load_yaml(path)
        key = _key(candidate)
        assert key in baselines and key not in seen, path
        seen.add(key)
        expected = deepcopy(baselines[key])
        expected["recipe_id"] = RECIPE
        expected["labels"] = LABELS
        # Managed runs write to <campaign>/<modality>/<dataset>, where dataset is
        # the config's dataset value (androids_interview, turkish).
        baseline_run_root = Path(expected["output_dirs"]["run_root"])
        output_model_root = baseline_run_root.parents[2]
        expected["output_dirs"]["run_root"] = str(
            output_model_root / "likelihood_ab_v1" / baseline_run_root.parent.name / candidate["dataset"]
        )
        expected["evaluation"]["sample_prediction_mode"] = "likelihood"
        expected["evaluation"]["headline_mode"] = "likelihood"
        expected["evaluation"]["evaluation_view"] = "harmonized_all_windows_full_coverage"
        assert candidate == expected, path
        assert "A = Depressed\nB = Non-depressed" in prompt_label_instruction(candidate)
        assert candidate["training"]["selection_metric"] == "inner_val_macro_f1"
        assert candidate["training"]["early_stopping"]["metric"] == "inner_val_macro_f1"
        run_roots.add(candidate["output_dirs"]["run_root"])
    assert seen == set(baselines)
    assert len(run_roots) == 15
