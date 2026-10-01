#!/usr/bin/env python3
"""Build the canonical default configs with the current default backends.

Text-only cells use Qwen3.8-27B. Audio-only and audio+text cells use the
Qwen3-Omni-30B-A3B Thinker. The pre-migration Qwen2 configs are retained under
``configs/archive/pre_default_backbone_20260923`` and are the immutable source
for this deterministic conversion.

The canonical default family is twelve generated files (four datasets times
three modalities). The Turkish default cell no longer lives here: it is the
existing **pooled** pos+neg family
(``configs/main/turkish_pooled_t17_*_promptcontext_v1_*``), which was produced
by the prompt-context generators and is the migration's native source contract.
This script verifies that pooled default's canonical-backend contract instead of
rewriting it; the historical Qwen2 pooled configs and the positive-only Turkish
configs stay untouched.
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
from src.experiment_tracking.manifest_policy import MANIFEST_POLICY_PREBUILT
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
)

# The Turkish default cell: the three existing pooled Qwen3 configs. They are
# the migration's native source contract and are verified, never regenerated.
POOLED_TURKISH_DEFAULTS = {
    "audio_only": (
        "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
    "audio_text": (
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
    "text_only": (
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
        "_promptcontext_v1_qwen38_27b.yaml"
    ),
}
POOLED_TURKISH_TEXT_AGGREGATION = "turkish_pooled_text_pair_mean_margin_strict_v1"

# The audio lane's default shape: two nodes of four GPUs with per-rank
# accumulation 16, which is the shape the existing pooled Qwen3-Omni baseline
# cells were submitted with. The FSDP recipe keeps an effective global batch of
# 128 (1 x 16 x 8); the text lane keeps one node of four GPUs with accumulation
# 32 (1 x 32 x 4).
AUDIO_TRAIN_NODES = 2
AUDIO_GRADIENT_ACCUMULATION_STEPS = 16
AUDIO_RESOURCES = {
    "train_nodes": AUDIO_TRAIN_NODES,
    "eval_nodes": 1,
    "eval_gpus_per_node": 4,
}


def _filename(stem: str, modality: str) -> str:
    return f"{stem}_{modality}_harmonized_selmacrof1_likelihood_v1.yaml"


def _prompt(config: dict[str, Any], context: str) -> dict[str, Any]:
    old = config["prompt"]
    return {
        "version": PROMPT_CONTEXT_VERSION,
        "dataset_context": context,
        "user_template": old["user_template"],
        "prompt_language": old.get("prompt_language", "english"),
    }


def derive(source: dict[str, Any], *, modality: str, context: str, output_dataset: str) -> dict[str, Any]:
    config = copy.deepcopy(source)
    config["recipe_id"] = f"{config['recipe_id']}_promptcontext_v1"
    config["prompt"] = _prompt(config, context)

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
        config["resources"] = dict(AUDIO_RESOURCES)
        training["gradient_accumulation_steps"] = AUDIO_GRADIENT_ACCUMULATION_STEPS
        validate_qwen3omni_config(config)

    resolve_system_prompt(config)
    return config


def render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000)


def verify_pooled_turkish_defaults() -> list[str]:
    """Fail unless the three pooled Qwen3 Turkish configs satisfy the canonical contract.

    These files are the migration's native source contract: they are verified in
    place, never regenerated or rewritten by this script.
    """
    failures: list[str] = []
    for modality, name in sorted(POOLED_TURKISH_DEFAULTS.items()):
        path = MAIN / name
        if not path.is_file():
            failures.append(f"missing pooled Turkish default: {path.name}")
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt = config.get("prompt") or {}
        training = config.get("training") or {}
        evaluation = config.get("evaluation") or {}
        output_dirs = config.get("output_dirs") or {}
        problems: list[str] = []
        if str(config.get("dataset")) != "turkish" or str(config.get("dataset_variant")) != "pooled_t17":
            problems.append("dataset must be turkish / pooled_t17")
        if prompt.get("version") != PROMPT_CONTEXT_VERSION:
            problems.append("prompt.version must be promptcontext_v1")
        if prompt.get("dataset_context") != "turkish_pooled":
            problems.append("prompt.dataset_context must be turkish_pooled")
        if prompt.get("question_context_version") != PROMPT_CONTEXT_VERSION:
            problems.append("prompt.question_context_version must be promptcontext_v1")
        if "{question_context}" not in str(prompt.get("user_template", "")):
            problems.append("prompt.user_template must carry {question_context}")
        if "system" in prompt:
            problems.append("prompt must not carry inline system text")
        if training.get("strategy") != "fsdp":
            problems.append("training.strategy must be fsdp")
        if training.get("activation_offload") != "cpu":
            problems.append("training.activation_offload must be cpu")
        if training.get("run_final_eval_in_train") is not False:
            problems.append("training.run_final_eval_in_train must be false")
        if evaluation.get("sample_prediction_mode") != "likelihood":
            problems.append("evaluation.sample_prediction_mode must be likelihood")
        if evaluation.get("evaluation_view") != "harmonized_all_windows_full_coverage":
            problems.append("evaluation.evaluation_view must be harmonized_all_windows_full_coverage")
        if str(evaluation.get("inference_dtype", "")).strip().lower() != "bf16":
            problems.append("evaluation.inference_dtype must be bf16")
        if config.get("manifest_policy") != MANIFEST_POLICY_PREBUILT:
            problems.append(f"manifest_policy must be {MANIFEST_POLICY_PREBUILT}")
        if modality == "text_only":
            expected_run_root = (
                "${PROJECT_ROOT}/output_model/promptcontext_v1_qwen38_likelihood/text_only/turkish"
            )
            if config.get("model_backend") != "qwen38":
                problems.append("model_backend must be qwen38")
            if evaluation.get("subject_score_aggregation") != POOLED_TURKISH_TEXT_AGGREGATION:
                problems.append(
                    f"evaluation.subject_score_aggregation must be {POOLED_TURKISH_TEXT_AGGREGATION}"
                )
            if "resources" in config:
                problems.append("the text-only pooled default must not declare resources")
        else:
            expected_run_root = (
                "${PROJECT_ROOT}/output_model/promptcontext_v1_qwen3omni_likelihood/"
                f"{modality}/turkish"
            )
            if config.get("model_backend") != "qwen3omni":
                problems.append("model_backend must be qwen3omni")
            if evaluation.get("aggregation_level") != "response_subject":
                problems.append("evaluation.aggregation_level must be response_subject")
            if evaluation.get("hierarchical_score_aggregation") != "mean":
                problems.append("evaluation.hierarchical_score_aggregation must be mean")
            if config.get("resources") != AUDIO_RESOURCES:
                problems.append(
                    "resources must declare the audio lane shape "
                    f"{sorted(AUDIO_RESOURCES)}"
                )
            if training.get("gradient_accumulation_steps") != AUDIO_GRADIENT_ACCUMULATION_STEPS:
                problems.append(
                    "training.gradient_accumulation_steps must be "
                    f"{AUDIO_GRADIENT_ACCUMULATION_STEPS} for the two-node audio lane"
                )
        if output_dirs.get("run_root") != expected_run_root:
            problems.append(f"output_dirs.run_root must be {expected_run_root}")
        try:
            validate_qwen38_config(config)
            validate_qwen3omni_config(config)
        except ValueError as exc:
            problems.append(f"backend validation failed: {exc}")
        if problems:
            failures.extend(f"{path.name}: {problem}" for problem in problems)
    return failures


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
    pooled_failures = verify_pooled_turkish_defaults()
    if pooled_failures:
        print("Pooled Turkish default drift:")
        for failure in pooled_failures:
            print(f"- {failure}")
        return 1
    if check:
        print("Canonical backend configs are current; pooled Turkish defaults verified.")
    else:
        print("Wrote 12 canonical backend configs; verified 3 pooled Turkish defaults.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
