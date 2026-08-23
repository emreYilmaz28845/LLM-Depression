from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "main"
GROUP = ROOT / "experiments" / "definitions" / "turkish-mixed-vs-negonly-native-en-multimodal-heads-v1-20260823.yaml"

GROUP_ID = "turkish-mixed-vs-negonly-native-en-multimodal-heads-v1-20260823"
VIEW = "harmonized_all_windows_full_coverage"
NATIVE_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1"
EN_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1"
GEMMA_TARGET = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"


CELLS = {
    "M01": ("mixed", "not_applicable", "audio_only", "qwen", "turkish_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "M02": ("mixed", "native", "text_only", "qwen", "turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "M03": ("mixed", "english", "text_only", "qwen", "turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "M04": ("mixed", "native", "audio_text", "qwen", "turkish_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "M05": ("mixed", "english", "audio_text", "qwen", "turkish_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "M06": ("mixed", "not_applicable", "audio_only", "gemma4", "turkish_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "M07": ("mixed", "native", "text_only", "gemma4", "turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "M08": ("mixed", "english", "text_only", "gemma4", "turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
    "M09": ("mixed", "native", "audio_text", "gemma4", "turkish_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "M10": ("mixed", "english", "audio_text", "gemma4", "turkish_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
    "N01": ("negative_only", "not_applicable", "audio_only", "qwen", "turkish_negative_only_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "N02": ("negative_only", "native", "text_only", "qwen", "turkish_negative_only_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "N03": ("negative_only", "english", "text_only", "qwen", "turkish_negative_only_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "N04": ("negative_only", "native", "audio_text", "qwen", "turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "N05": ("negative_only", "english", "audio_text", "qwen", "turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "N06": ("negative_only", "not_applicable", "audio_only", "gemma4", "turkish_negative_only_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "N07": ("negative_only", "native", "text_only", "gemma4", "turkish_negative_only_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "N08": ("negative_only", "english", "text_only", "gemma4", "turkish_negative_only_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
    "N09": ("negative_only", "native", "audio_text", "gemma4", "turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "N10": ("negative_only", "english", "audio_text", "gemma4", "turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
}


def _load(name: str) -> dict:
    return yaml.safe_load((MAIN / name).read_text(encoding="utf-8"))


def _config_for(cell: str) -> dict:
    return _load(CELLS[cell][4])


def test_group_definition_records_exact_twenty_cells() -> None:
    group = yaml.safe_load(GROUP.read_text(encoding="utf-8"))
    assert group["group_id"] == GROUP_ID
    cells = group["scope"]["backbone_cells"]
    assert len(cells) == 20
    assert {cell["id"] for cell in cells} == set(CELLS)
    assert all(cell["transcript_condition"] != "english" or cell["modality"] != "audio_only" for cell in cells)
    assert group["expected_seeds"] == [7, 1337, 2024]
    assert group["expected_folds"] == [0, 1, 2, 3, 4]
    assert group["primary_metric"]["evaluation_view"] == VIEW


def test_all_cells_use_locked_harmonized_recipe() -> None:
    assert len(CELLS) == 20
    for cell_id, (recording, transcript, modality, backbone, _) in CELLS.items():
        config = _config_for(cell_id)
        assert config["dataset"] == "turkish"
        if recording == "negative_only":
            assert config["dataset_variant"] == "negative_only_t17"
            assert config["metadata_schema"] == "minimal_t17"
            assert "Turkish_Negative_Only" in config["dataset_root"]
        else:
            assert "dataset_variant" not in config
            assert "Turkish_Negative_Only" not in config["dataset_root"]
        assert config["threshold"] == 17
        assert config["split"] == {"mode": "cv", "cv_protocol": "train_val", "outer_folds": 5, "inner_val_ratio": 0.2, "seed": 1337}
        assert config["recipe_id"] == (EN_RECIPE if transcript == "english" else NATIVE_RECIPE)
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["selection_metric_mode"] == "max"
        assert config["training"]["class_balance"] == "none"
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
        assert config["evaluation"]["headline_mode"] == "original_teacher_forced"
        assert config["evaluation"]["evaluation_view"] == VIEW
        if config["data"]["use_audio"]:
            assert config["audio_adapter"]["enabled"] is False
            assert config["audio_adapter"]["train_projector"] is False
        assert (config["data"]["use_audio"], config["data"]["use_text"]) == {
            "audio_only": (True, False),
            "text_only": (False, True),
            "audio_text": (True, True),
        }[modality]
        if transcript == "english":
            assert config["transcripts"]["variant"] == "english"
            assert config["transcripts"]["require_complete"] is True
            assert "accepted.jsonl" in config["transcripts"]["cache_path"]
        else:
            assert "transcripts" not in config or config["transcripts"].get("variant", "original") == "original"
        if backbone == "gemma4":
            assert config["model_backend"] == "gemma4"
            assert config["model_revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
            assert config["lora"]["target_modules"] == GEMMA_TARGET
        else:
            assert "model_backend" not in config
            assert isinstance(config["lora"]["target_modules"], list)


def test_negative_gemma_configs_are_qwen_derived_with_only_approved_backend_diffs() -> None:
    pairs = {"N01": "N06", "N02": "N07", "N03": "N08", "N04": "N09", "N05": "N10"}
    allowed_top_level = {"model_name_or_path", "model_backend", "model_revision", "output_dirs", "lora"}
    for qwen_cell, gemma_cell in pairs.items():
        qwen = _config_for(qwen_cell)
        gemma = _config_for(gemma_cell)
        assert set(qwen) ^ set(gemma) <= {"model_backend", "model_revision"}
        for key in sorted(set(qwen) & set(gemma)):
            if key in allowed_top_level:
                continue
            assert qwen[key] == gemma[key], f"unexpected {key} difference for {qwen_cell}/{gemma_cell}"
        assert "gemma-4-12B-it" in gemma["model_name_or_path"]
        assert qwen["output_dirs"]["run_root"] != gemma["output_dirs"]["run_root"]
        assert qwen["output_dirs"]["manifest_dir"] == gemma["output_dirs"]["manifest_dir"]
        assert qwen["output_dirs"]["split_dir"] == gemma["output_dirs"]["split_dir"]


def test_no_english_audio_only_config_exists() -> None:
    assert not any(
        "audio_only" in name and "_en" in name
        for name in (path.name for path in MAIN.glob("turkish*_harmonized_selmacrof1_tf*.yaml"))
    )
