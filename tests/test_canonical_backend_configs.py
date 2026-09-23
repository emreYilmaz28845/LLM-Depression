from __future__ import annotations

from pathlib import Path

import yaml

from scripts import build_canonical_backend_configs as builder
from src.model.qwen38_lora import QWEN38_LORA_TARGET_REGEX
from src.model.qwen3omni_lora import QWEN3OMNI_LORA_TARGET_REGEX


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_backend_configs_match_generator() -> None:
    assert builder.build(check=True) == 0


def test_canonical_backend_assignment_and_resource_contracts() -> None:
    for stem, _context, _output_dataset in builder.CELLS:
        for modality in ("audio_only", "audio_text", "text_only"):
            path = ROOT / "configs/main" / builder._filename(stem, modality)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert config["prompt"]["version"] == "promptcontext_v1"
            assert config["training"]["strategy"] == "fsdp"
            assert config["training"]["activation_offload"] == "cpu"
            assert config["training"]["run_final_eval_in_train"] is False
            assert config["evaluation"]["sample_prediction_mode"] == "likelihood"
            assert config["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage"
            assert config["evaluation"]["inference_dtype"] == "bf16"

            if modality == "text_only":
                assert config["model_backend"] == "qwen38"
                assert "Qwen3.8-27B" in config["model_name_or_path"]
                assert config["lora"]["target_modules"] == QWEN38_LORA_TARGET_REGEX
                assert "resources" not in config
            else:
                assert config["model_backend"] == "qwen3omni"
                assert "Qwen3-Omni-30B-A3B-Instruct" in config["model_name_or_path"]
                assert config["model_attn_implementation"] == "sdpa"
                assert config["lora"]["target_modules"] == QWEN3OMNI_LORA_TARGET_REGEX
                assert config["resources"] == {"eval_nodes": 1, "eval_gpus_per_node": 4}
