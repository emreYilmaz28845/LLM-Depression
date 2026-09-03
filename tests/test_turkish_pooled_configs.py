"""Pooled Turkish config family invariants (plan Step 3)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "main"

VIEW = "harmonized_all_windows_full_coverage"
NATIVE_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1_qcond_v1"
EN_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1_qcond_v1"
GEMMA_TARGET = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"

# pooled cell id -> (transcript_condition, modality, backbone, config file)
CELLS = {
    "Q01": ("not_applicable", "audio_only", "qwen", "turkish_pooled_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "Q02": ("native", "text_only", "qwen", "turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "Q03": ("english", "text_only", "qwen", "turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "Q04": ("native", "audio_text", "qwen", "turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml"),
    "Q05": ("english", "audio_text", "qwen", "turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    "G01": ("not_applicable", "audio_only", "gemma4", "turkish_pooled_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "G02": ("native", "text_only", "gemma4", "turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "G03": ("english", "text_only", "gemma4", "turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
    "G04": ("native", "audio_text", "gemma4", "turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"),
    "G05": ("english", "audio_text", "gemma4", "turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"),
}


def _load(name: str) -> dict:
    return yaml.safe_load((MAIN / name).read_text(encoding="utf-8"))


def test_exactly_ten_pooled_configs_and_no_english_audio_only() -> None:
    pooled = sorted(path.name for path in MAIN.glob("turkish_pooled_t17_*.yaml"))
    assert len(pooled) == 10
    assert set(pooled) == {cell[3] for cell in CELLS.values()}
    assert not any("audio_only" in name and "_en" in name for name in pooled)


def test_pooled_cells_use_locked_qcond_recipe() -> None:
    assert len(CELLS) == 10
    for cell_id, (transcript, modality, backbone, name) in CELLS.items():
        config = _load(name)
        assert config["dataset"] == "turkish", cell_id
        assert config["dataset_variant"] == "pooled_t17", cell_id
        assert config["threshold"] == 17, cell_id
        assert config["recipe_id"] == (EN_RECIPE if transcript == "english" else NATIVE_RECIPE), cell_id
        assert config["split"] == {"mode": "cv", "cv_protocol": "train_val", "outer_folds": 5, "inner_val_ratio": 0.2, "seed": 1337}, cell_id
        assert config["training"]["selection_metric"] == "inner_val_macro_f1", cell_id
        assert config["training"]["selection_metric_mode"] == "max", cell_id
        assert config["training"]["class_balance"] == "none", cell_id
        assert config["training"]["num_train_epochs"] == 20, cell_id
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced", cell_id
        assert config["evaluation"]["headline_mode"] == "original_teacher_forced", cell_id
        assert config["evaluation"]["evaluation_view"] == VIEW, cell_id
        if config["data"]["use_audio"]:
            assert config["audio_adapter"]["enabled"] is False, cell_id
            assert config["audio_adapter"]["train_projector"] is False, cell_id
        assert (config["data"]["use_audio"], config["data"]["use_text"]) == {
            "audio_only": (True, False),
            "text_only": (False, True),
            "audio_text": (True, True),
        }[modality], cell_id
        # tag placeholder on its own line before the transcript block
        template = config["prompt"]["user_template"]
        assert "{question_context}{transcript_block}" in template, cell_id
        assert config["prompt"]["system"].startswith("You are a psychologist"), cell_id
        if transcript == "english":
            assert config["transcripts"]["variant"] == "english", cell_id
            assert config["transcripts"]["require_complete"] is True, cell_id
        else:
            assert "transcripts" not in config, cell_id
        if backbone == "gemma4":
            assert config["model_backend"] == "gemma4", cell_id
            assert config["model_revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", cell_id
            assert config["lora"]["target_modules"] == GEMMA_TARGET, cell_id
        else:
            assert "model_backend" not in config, cell_id
        # own pooled roots, never the pos_only / negative_only dirs
        assert "turkish_pooled_t17_qwen3asr" in config["output_dirs"]["manifest_dir"], cell_id
        assert "turkish_pooled_t17_qwen3asr" in config["output_dirs"]["split_dir"], cell_id
        assert "turkish_pooled_t17_qwen3asr" in config["output_dirs"]["run_root"], cell_id
        assert "pos_only" not in config["output_dirs"]["run_root"], cell_id
        assert "negative_only" not in config["output_dirs"]["run_root"], cell_id


def test_pooled_gemma_configs_share_inputs_with_qwen() -> None:
    pairs = [
        ("Q01", "G01"), ("Q02", "G02"), ("Q03", "G03"), ("Q04", "G04"), ("Q05", "G05"),
    ]
    for qwen_cell, gemma_cell in pairs:
        qwen = _load(CELLS[qwen_cell][3])
        gemma = _load(CELLS[gemma_cell][3])
        assert qwen["output_dirs"]["manifest_dir"] == gemma["output_dirs"]["manifest_dir"]
        assert qwen["output_dirs"]["split_dir"] == gemma["output_dirs"]["split_dir"]
        assert qwen["output_dirs"]["run_root"] != gemma["output_dirs"]["run_root"]
        assert "gemma-4-12B-it" in gemma["model_name_or_path"]
