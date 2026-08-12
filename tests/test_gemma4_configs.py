from __future__ import annotations

from pathlib import Path

from src.model.gemma4_io import (
    GEMMA4_EVALUATION_VIEW,
    GEMMA4_LORA_TARGET_REGEX,
    GEMMA4_MODEL_REVISION,
    validate_gemma4_config,
)
from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"

GEMMA_CONFIGS = {
    modality: f"daic_{modality}_harmonized_selmacrof1_tf_gemma4_12b.yaml"
    for modality in ("text_only", "audio_only", "audio_text")
}
QWEN_CONFIGS = {
    modality: f"daic_{modality}_harmonized_selmacrof1_tf.yaml"
    for modality in ("text_only", "audio_only", "audio_text")
}

# Keys the Gemma config is allowed to add relative to its Qwen counterpart.
ALLOWED_NEW_KEYS = ("model_backend", "model_revision")

# Top-level keys whose values are allowed to differ (everything else must be
# byte-identical after env expansion).
ALLOWED_DIFFERING_KEYS = {
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "output_dirs",
    "lora",
    "evaluation",
}

# Nested differences allowed inside the differing top-level keys.
ALLOWED_OUTPUT_DIRS_DIFF = ("run_root",)
ALLOWED_LORA_DIFF = ("target_modules",)
ALLOWED_EVALUATION_DIFF = ("evaluation_view",)


def _deep_diff(source: dict, target: dict, path: str = "") -> list[str]:
    differences: list[str] = []
    for key in sorted(set(source) | set(target)):
        key_path = f"{path}.{key}" if path else key
        if key not in source:
            differences.append(f"{key_path}: added in Gemma config")
            continue
        if key not in target:
            differences.append(f"{key_path}: missing in Gemma config")
            continue
        if isinstance(source[key], dict) and isinstance(target[key], dict):
            differences.extend(_deep_diff(source[key], target[key], key_path))
        elif source[key] != target[key]:
            differences.append(f"{key_path}: {source[key]!r} != {target[key]!r}")
    return differences


def _approved_differences(modality: str) -> list[str]:
    source = load_yaml(MAIN / QWEN_CONFIGS[modality])
    target = load_yaml(MAIN / GEMMA_CONFIGS[modality])
    differences: list[str] = []
    for key in sorted(set(source) | set(target)):
        if key not in target:
            differences.append(f"{key}: missing in Gemma config")
            continue
        if key not in source:
            if key in ALLOWED_NEW_KEYS:
                continue
            differences.append(f"{key}: unexpected new key in Gemma config")
            continue
        if key in ALLOWED_DIFFERING_KEYS:
            if key == "output_dirs":
                for sub_key in sorted(set(source[key]) | set(target[key])):
                    if sub_key not in target[key]:
                        differences.append(f"output_dirs.{sub_key}: missing in Gemma config")
                    elif sub_key not in source[key]:
                        differences.append(f"output_dirs.{sub_key}: added in Gemma config")
                    elif sub_key not in ALLOWED_OUTPUT_DIRS_DIFF and source[key][sub_key] != target[key][sub_key]:
                        differences.append(
                            f"output_dirs.{sub_key}: {source[key][sub_key]!r} != {target[key][sub_key]!r}"
                        )
            elif key == "lora":
                for sub_key in sorted(set(source[key]) | set(target[key])):
                    if sub_key not in target[key]:
                        differences.append(f"lora.{sub_key}: missing in Gemma config")
                    elif sub_key not in source[key]:
                        differences.append(f"lora.{sub_key}: added in Gemma config")
                    elif sub_key not in ALLOWED_LORA_DIFF and source[key][sub_key] != target[key][sub_key]:
                        differences.append(
                            f"lora.{sub_key}: {source[key][sub_key]!r} != {target[key][sub_key]!r}"
                        )
            elif key == "evaluation":
                for sub_key in sorted(set(source[key]) | set(target[key])):
                    if sub_key not in target[key]:
                        differences.append(f"evaluation.{sub_key}: missing in Gemma config")
                    elif sub_key not in source[key]:
                        if sub_key in ALLOWED_EVALUATION_DIFF:
                            continue
                        differences.append(f"evaluation.{sub_key}: added in Gemma config")
                    elif sub_key not in ALLOWED_EVALUATION_DIFF and source[key][sub_key] != target[key][sub_key]:
                        differences.append(
                            f"evaluation.{sub_key}: {source[key][sub_key]!r} != {target[key][sub_key]!r}"
                        )
            # model_backend / model_name_or_path / model_revision value
            # differences are the approved backbone switch; the nested keys
            # above are audited sub-key by sub-key.
            continue
        if source[key] != target[key]:
            differences.append(f"{key}: {source[key]!r} != {target[key]!r}")
    return differences


def test_gemma_configs_exist_and_validate() -> None:
    for modality, name in GEMMA_CONFIGS.items():
        config = load_yaml(MAIN / name)
        validate_gemma4_config(config)
        assert config["model_backend"] == "gemma4"
        assert config["model_revision"] == GEMMA4_MODEL_REVISION
        assert config["evaluation"]["evaluation_view"] == GEMMA4_EVALUATION_VIEW
        assert config["lora"]["target_modules"] == GEMMA4_LORA_TARGET_REGEX
        assert "harmonized_v1_gemma4" in config["output_dirs"]["run_root"]


def test_gemma_configs_share_daic_manifest_and_split_dirs() -> None:
    for modality, name in GEMMA_CONFIGS.items():
        gemma = load_yaml(MAIN / name)
        qwen = load_yaml(MAIN / QWEN_CONFIGS[modality])
        assert gemma["output_dirs"]["manifest_dir"] == qwen["output_dirs"]["manifest_dir"]
        assert gemma["output_dirs"]["split_dir"] == qwen["output_dirs"]["split_dir"]


def test_gemma_config_differences_are_limited_to_approved_list() -> None:
    for modality in GEMMA_CONFIGS:
        differences = _approved_differences(modality)
        assert not differences, f"{modality}: unexpected differences: {differences}"


def test_gemma_configs_preserve_scientific_invariants() -> None:
    for modality, name in GEMMA_CONFIGS.items():
        gemma = load_yaml(MAIN / name)
        qwen = load_yaml(MAIN / QWEN_CONFIGS[modality])
        for key in ("dataset", "seed", "recipe_id", "protocol_id", "manifest_variant", "labels", "prompt", "split"):
            assert gemma[key] == qwen[key], f"{modality}: {key} changed"
        assert gemma["data"] == qwen["data"], f"{modality}: data changed"
        assert gemma["training"] == qwen["training"], f"{modality}: training changed"
        assert gemma.get("audio_adapter") == qwen.get("audio_adapter"), f"{modality}: audio_adapter changed"
        for key in ("sample_prediction_mode", "headline_mode", "aggregation_level", "subject_score_aggregation"):
            if key not in qwen["evaluation"]:
                assert key not in gemma["evaluation"], f"{modality}: evaluation.{key} added"
                continue
            assert gemma["evaluation"][key] == qwen["evaluation"][key], f"{modality}: evaluation.{key} changed"
        assert gemma["evaluation"]["evaluation_view"] == GEMMA4_EVALUATION_VIEW


def test_gemma_lora_regex_matches_six_modules_per_layer_across_48_layers() -> None:
    import re

    pattern = re.compile(GEMMA4_LORA_TARGET_REGEX)
    matched = []
    for layer in range(48):
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            name = f"model.language_model.layers.{layer}.{module}"
            assert pattern.fullmatch(name), f"regex must match {name}"
            matched.append(name)
        assert not pattern.fullmatch(f"model.language_model.layers.{layer}.self_attn.v_proj")
        assert not pattern.fullmatch(f"model.language_model.layers.{layer}.input_layernorm")
    assert len(matched) == 288


def test_archived_gemma_configs_do_not_exist() -> None:
    archive = ROOT / "configs/archive"
    assert not list(archive.rglob("*gemma4*"))
