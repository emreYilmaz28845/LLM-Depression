#!/usr/bin/env python3
"""Build the canonical 15-cell configs with the current default backends.

Text-only cells use Qwen3.8-27B. Audio-only and audio+text cells use the
Qwen3-Omni-30B-A3B Thinker. The pre-migration Qwen2 configs are retained under
``configs/archive/pre_default_backbone_20260923`` and are the immutable source
for this deterministic conversion.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prompt_context import PROMPT_CONTEXT_VERSION, resolve_system_prompt
from src.model.qwen38_lora import (
    QWEN38_EVALUATION_VIEW,
    QWEN38_LORA_TARGET_REGEX,
    QWEN38_MODEL_REVISION,
    validate_qwen38_config,
)
from src.model.qwen3omni_lora import (
    QWEN3OMNI_EVALUATION_VIEW,
    QWEN3OMNI_LORA_TARGET_REGEX,
    validate_qwen3omni_config,
)

MAIN = ROOT / "configs/main"
ARCHIVE = ROOT / "configs/archive/pre_default_backbone_20260923"

QWEN38_PATH = "${QWEN38_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
OMNI_PATH = "${QWEN3_OMNI_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"

# filename, prompt context key, output dataset directory
CELLS = (
    ("d3tec", "d3tec", "d3tec"),
    ("androids", "androids", "androids_interview"),
    ("daic", "daic", "daic"),
    ("cmdc", "cmdc", "cmdc"),
    ("turkish_pos_only_t17", "turkish_pooled", "turkish_pos_only_t17_qwen3asr"),
)


def _filename(stem: str, modality: str) -> str:
    suffix = "_qwen3asr" if stem.startswith("turkish_") else ""
    return f"{stem}_{modality}_harmonized_selmacrof1_likelihood_v1{suffix}.yaml"


def _prompt(config: dict[str, Any], context: str, *, turkish: bool) -> dict[str, Any]:
    old = config["prompt"]
    result: dict[str, Any] = {
        "version": PROMPT_CONTEXT_VERSION,
        "dataset_context": context,
    }
    if turkish:
        result["question_context_version"] = PROMPT_CONTEXT_VERSION
    result["user_template"] = old["user_template"]
    result["prompt_language"] = old.get("prompt_language", "english")
    return result


def derive(source: dict[str, Any], *, modality: str, context: str, output_dataset: str) -> dict[str, Any]:
    config = copy.deepcopy(source)
    turkish = str(config.get("dataset")) == "turkish"
    config["recipe_id"] = f"{config['recipe_id']}_promptcontext_v1"
    config["prompt"] = _prompt(config, context, turkish=turkish)

    training = config["training"]
    training["strategy"] = "fsdp"
    training["activation_offload"] = "cpu"
    training["run_final_eval_in_train"] = False

    evaluation = config["evaluation"]
    evaluation["evaluation_view"] = "harmonized_all_windows_full_coverage"
    evaluation["inference_dtype"] = "bf16"

    if modality == "text_only":
        config["model_backend"] = "qwen38"
        config["model_name_or_path"] = QWEN38_PATH
        config["model_revision"] = QWEN38_MODEL_REVISION
        config["lora"]["target_modules"] = QWEN38_LORA_TARGET_REGEX
        config["output_dirs"]["run_root"] = (
            f"${{PROJECT_ROOT}}/output_model/promptcontext_v1_qwen38_likelihood/"
            f"text_only/{output_dataset}"
        )
        validate_qwen38_config(config)
    else:
        config["model_backend"] = "qwen3omni"
        config["model_name_or_path"] = OMNI_PATH
        config["model_attn_implementation"] = "sdpa"
        config.pop("model_revision", None)
        config["lora"]["target_modules"] = QWEN3OMNI_LORA_TARGET_REGEX
        config["output_dirs"]["run_root"] = (
            f"${{PROJECT_ROOT}}/output_model/promptcontext_v1_qwen3omni_likelihood/"
            f"{modality}/{output_dataset}"
        )
        config["resources"] = {"eval_nodes": 1, "eval_gpus_per_node": 4}
        validate_qwen3omni_config(config)

    resolve_system_prompt(config)
    return config


def render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000)


def build(*, check: bool) -> int:
    mismatches: list[str] = []
    for stem, context, output_dataset in CELLS:
        for modality in ("audio_only", "audio_text", "text_only"):
            name = _filename(stem, modality)
            source_path = ARCHIVE / name
            target_path = MAIN / name
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing archived Qwen2 source: {source_path}")
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            expected = render(
                derive(source, modality=modality, context=context, output_dataset=output_dataset)
            )
            if check:
                if not target_path.is_file() or target_path.read_text(encoding="utf-8") != expected:
                    mismatches.append(str(target_path.relative_to(ROOT)))
            else:
                target_path.write_text(expected, encoding="utf-8")

    if mismatches:
        print("Canonical backend config drift:")
        for path in mismatches:
            print(f"- {path}")
        return 1
    print("Canonical backend configs are current." if check else "Wrote 15 canonical backend configs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
