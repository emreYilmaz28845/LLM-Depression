from __future__ import annotations

from pathlib import Path

import yaml

from src.turkish_pooled_qcond import load_cells


ROOT = Path(__file__).parents[1]


def test_locked_turkish_pooled_matrix_has_ten_distinct_cells() -> None:
    cells = load_cells(ROOT)
    assert len(cells) == 10
    assert {cell.cell_id for cell in cells} == {f"Q{i:02d}" for i in range(1, 6)} | {f"G{i:02d}" for i in range(1, 6)}
    assert len({cell.config for cell in cells}) == 10
    assert sum(cell.backbone == "qwen" for cell in cells) == 5
    assert sum(cell.backbone == "gemma4" for cell in cells) == 5


def test_every_pooled_config_has_the_tag_and_locked_recipe() -> None:
    for cell in load_cells(ROOT):
        config = yaml.safe_load((ROOT / cell.config).read_text(encoding="utf-8"))
        assert config["dataset"] == "turkish"
        assert config["dataset_variant"] == "pooled_t17"
        assert config["recipe_id"].endswith("_qcond_v1")
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
        assert config["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage"
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["selection_metric_mode"] == "max"
        assert config["training"]["class_balance"] == "none"
        assert "{question_context}" in config["prompt"]["user_template"]
        assert config["data"]["segment_seconds"] <= 30.0
        if config["data"]["use_audio"]:
            assert config["data"]["train_chunk_policy"] == "all_windows_hierarchical_weighted"
            assert config["data"]["eval_chunk_policy"] == "all_windows"
            assert config.get("audio_adapter", {}).get("enabled") is False
            assert config.get("audio_adapter", {}).get("train_projector") is False
        if cell.modality == "text_only":
            assert config["evaluation"]["subject_score_aggregation"] == "turkish_pooled_text_pair_mean_margin_strict_v1"


def test_qwen_and_gemma_share_input_contract_but_not_output_roots() -> None:
    cells = {cell.cell_id: yaml.safe_load((ROOT / cell.config).read_text(encoding="utf-8")) for cell in load_cells(ROOT)}
    for qwen_id, gemma_id in ((f"Q{i:02d}", f"G{i:02d}") for i in range(1, 6)):
        qwen, gemma = cells[qwen_id], cells[gemma_id]
        assert qwen["data"] == gemma["data"]
        assert qwen["split"] == gemma["split"]
        assert qwen["evaluation"] == gemma["evaluation"]
        assert qwen["output_dirs"]["run_root"] != gemma["output_dirs"]["run_root"]
