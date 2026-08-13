from pathlib import Path

from src.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1"
DATASETS = {"d3tec", "turkish", "androids_interview", "daic", "cmdc"}


def harmonized_configs():
    return sorted(
        path
        for path in MAIN.glob("*harmonized_selmacrof1_tf*.yaml")
        if not path.name.endswith("_en.yaml")
        and "_gemma4_12b" not in path.name
        and "_officialdev" not in path.name
    )


def test_harmonized_main_has_five_datasets_by_three_modalities() -> None:
    configs = [load_yaml(path) for path in harmonized_configs()]
    assert len(configs) == 15
    assert {config["dataset"] for config in configs} == DATASETS
    for dataset in DATASETS:
        members = [config for config in configs if config["dataset"] == dataset]
        modalities = {
            (bool(config["data"]["use_audio"]), bool(config["data"]["use_text"]))
            for config in members
        }
        assert modalities == {(True, False), (False, True), (True, True)}


def test_gemma4_variants_exist_for_daic_only() -> None:
    gemma = sorted(MAIN.glob("*harmonized_selmacrof1_tf_gemma4_12b.yaml"))
    assert len(gemma) == 3
    assert all("daic_" in path.name for path in gemma)


def test_harmonized_selection_and_teacher_forced_recipe_is_locked() -> None:
    for path in harmonized_configs():
        config = load_yaml(path)
        assert config["recipe_id"] == RECIPE
        assert config["quarantine_path"].endswith("configs/quarantines.yaml")
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["selection_metric_mode"] == "max"
        assert config["training"]["early_stopping"]["metric"] == "inner_val_macro_f1"
        assert config["training"]["early_stopping"]["mode"] == "max"
        assert config["training"]["class_balance"] == "none"
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
        assert config["evaluation"]["headline_mode"] == "original_teacher_forced"


def test_audio_configs_are_single_window_all_coverage() -> None:
    for path in harmonized_configs():
        config = load_yaml(path)
        data = config["data"]
        if not data["use_audio"]:
            assert config["evaluation"]["aggregation_level"] == "subject"
            continue
        if config["dataset"] == "daic":
            assert data["sample_mode"] == "participant_speech_packed30"
            assert data["participant_chunk_samples"] == 480000
            assert data["train_chunk_policy"] == "all_chunks_subject_normalized"
            assert data["eval_chunk_policy"] == "all_chunks_mean_score"
            assert config["manifest_variant"] == "unprocessed_participant_speech_packed30_v1"
            assert config["evaluation"]["aggregation_level"] == "subject"
        else:
            assert data["sample_mode"] == "harmonized_response_windows"
            assert data["segment_seconds"] == 30.0
            assert data["train_chunk_policy"] == "all_windows_hierarchical_weighted"
            assert data["eval_chunk_policy"] == "all_windows"
            assert config["evaluation"]["aggregation_level"] == "response_subject"


def test_superseded_main_configs_are_archived_and_edaic_is_untouched() -> None:
    archive = ROOT / "configs/archive/pre_harmonized_posf1_20260809"
    assert len(list(archive.glob("*/*.yaml"))) == 9
    assert len(list(MAIN.glob("edaic_*_selposf1_tf.yaml"))) == 3
    assert not [
        path
        for path in MAIN.glob("*selposf1*.yaml")
        if path.name.startswith(("cmdc_", "daic_", "turkish_"))
    ]
