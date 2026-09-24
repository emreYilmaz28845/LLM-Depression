#!/usr/bin/env python3
"""Generate the Qwen3-Omni prompt-context production configs under ``configs/main``.

Two families come out of the same deterministic machinery:

* the two DAIC pilot cells, derived from the archived pre-default-backbone
  Qwen2-Audio DAIC sources, so PR #261's documented reproduction command keeps
  working unchanged;
* the two Turkish pooled t17 cells, derived from the current ``configs/main``
  Turkish pooled Qwen2-Audio likelihood sources (the pooled recipe is not one of
  PR #262's 15 canonical cells).

Each config is derived from a canonical Qwen2-Audio source and changes only the
documented difference set of its family, so the pair isolates the backbone (and,
for the prompt, the prompt-context recipe) instead of silently moving the recipe:

* ``model_backend``/``model_name_or_path`` select the verified offline
  Qwen3-Omni snapshot, and ``model_attn_implementation: sdpa`` is required
  because the offline MN5 wheelhouse ships no flash-attn;
* the prompt switches from the inline ``prompt.system`` to the shared
  ``promptcontext_v1`` instruction plus the centralized DAIC recording-context
  block (``src/data/prompt_context.py``);
* ``lora.target_modules`` becomes the anchored Thinker decoder regex (attention
  projections plus the non-routed shared expert; routers and routed experts stay
  frozen);
* training switches to the shared FSDP strategy with the measured
  activation-offload choice, and ``run_final_eval_in_train`` turns false because
  FSDP cannot do the in-train single-GPU held-out evaluation;
* evaluation declares the explicit view, the BF16 inference dtype the 30B
  Thinker needs, and the resource shape used for the sharded standalone
  evaluation.

Split protocol, seed, label contract, manifest identity, windowing, weighting,
checkpoint selection, early stopping and the canonical epoch count are inherited
unchanged. The script is deterministic and idempotent: ``--check`` reports every
difference from the derived content without writing, and an existing file with
different content is never overwritten silently.

The structured diff audit lists the changed leaf paths of both configs and fails
when a change falls outside the allowlist. It is written under ``outputs/``
(not tracked).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prompt_context import PROMPT_CONTEXT_VERSION, resolve_system_prompt
from src.experiment_tracking.manifest_policy import MANIFEST_POLICY_PREBUILT
from src.model.qwen3omni_lora import (
    QWEN3OMNI_EVALUATION_VIEW,
    QWEN3OMNI_LORA_TARGET_REGEX,
    validate_qwen3omni_config,
)

MAIN = PROJECT_ROOT / "configs/main"
PRE_DEFAULT_BACKBONE_ARCHIVE = PROJECT_ROOT / "configs/archive/pre_default_backbone_20260923"
DEFAULT_AUDIT_OUTPUT = (
    PROJECT_ROOT / "outputs/qwen3omni_daic_config_diff/config_diff_audit.json"
)

QWEN3_OMNI_MODEL_PATH = (
    "${QWEN3_OMNI_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"
)
QWEN3_OMNI_ATTN_IMPLEMENTATION = "sdpa"
RECIPE_SUFFIX = "_promptcontext_v1"
RUN_ROOT_CAMPAIGN = "promptcontext_v1_qwen3omni_likelihood"
INFERENCE_DTYPE = "bf16"
# Measured production choices; the pilot's memory gate and bounded sweep set them
# and the reason is recorded in the run evidence.
ACTIVATION_OFFLOAD = "cpu"
EVAL_NODES = 1
EVAL_GPUS_PER_NODE = 4

# (slug, source config name, target config name, modality)
CELLS = (
    (
        "daic_audio_only",
        "daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "daic_audio_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "audio_only",
    ),
    (
        "daic_audio_text",
        "daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
        "daic_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "audio_text",
    ),
)

# Every diff between a source config and its derived config must be one of these
# leaf paths. Anything else fails the audit instead of shipping a silent change.
ALLOWED_DIFF_PATHS = frozenset(
    {
        "model_backend",
        "model_name_or_path",
        "model_attn_implementation",
        "recipe_id",
        "output_dirs.run_root",
        "prompt.system",
        "prompt.version",
        "prompt.dataset_context",
        "prompt.user_template",
        "prompt.prompt_language",
        "lora.target_modules",
        "training.strategy",
        "training.activation_offload",
        "training.run_final_eval_in_train",
        "evaluation.evaluation_view",
        "evaluation.inference_dtype",
        "resources",
        "resources.eval_nodes",
        "resources.eval_gpus_per_node",
    }
)

# Turkish pooled t17 cells. Their sources are current ``configs/main`` configs, so
# they are read from the main config directory rather than the archive.
POOLED_SOURCE_DIR = MAIN
POOLED_CELLS = (
    (
        "turkish_pooled_audio_only",
        "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml",
        "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "audio_only",
    ),
    (
        "turkish_pooled_audio_text",
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml",
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "audio_text",
    ),
)

# The pooled family adds the prompt question-context version and the prebuilt
# manifest policy to the DAIC difference set. The pooled recipe's manifest is
# built outside the worker, so the submission must never rebuild it.
POOLED_ALLOWED_DIFF_PATHS = ALLOWED_DIFF_PATHS | {
    "manifest_policy",
    "prompt.question_context_version",
}

POOLED_DATASET_CONTEXT = "turkish_pooled"
POOLED_RUN_ROOT_DATASET_DIR = "turkish"


class GenerationError(RuntimeError):
    """Raised when a source config or a derived diff breaks the contract."""


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


def diff_paths(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    """Leaf paths that differ between two configs."""
    source_flat = _flatten(source)
    target_flat = _flatten(target)
    return sorted(
        path
        for path in set(source_flat) | set(target_flat)
        if source_flat.get(path, "<missing>") != target_flat.get(path, "<missing>")
    )


def derive(source: dict[str, Any], cell: tuple) -> dict[str, Any]:
    slug, source_name, _target, modality = cell
    config = copy.deepcopy(source)
    prompt = config.get("prompt") or {}
    template = prompt.get("user_template")
    language = prompt.get("prompt_language", "english")
    if not template:
        raise GenerationError(f"{source_name}: prompt.user_template is missing")

    config["model_backend"] = "qwen3omni"
    config["model_name_or_path"] = QWEN3_OMNI_MODEL_PATH
    config["model_attn_implementation"] = QWEN3_OMNI_ATTN_IMPLEMENTATION
    config["recipe_id"] = f"{config['recipe_id']}{RECIPE_SUFFIX}"
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{RUN_ROOT_CAMPAIGN}/{modality}/daic"
    )
    config["prompt"] = {
        "version": PROMPT_CONTEXT_VERSION,
        "dataset_context": "daic",
        "user_template": template,
        "prompt_language": language,
    }

    lora = config["lora"]
    if set(lora) - {"rank", "alpha", "dropout", "bias", "target_modules"}:
        raise GenerationError(f"{source_name}: unexpected lora keys {sorted(lora)}")
    lora["target_modules"] = QWEN3OMNI_LORA_TARGET_REGEX

    training = config["training"]
    training["strategy"] = "fsdp"
    training["activation_offload"] = ACTIVATION_OFFLOAD
    training["run_final_eval_in_train"] = False

    config["evaluation"]["evaluation_view"] = QWEN3OMNI_EVALUATION_VIEW
    config["evaluation"]["inference_dtype"] = INFERENCE_DTYPE

    config["resources"] = {
        "eval_nodes": EVAL_NODES,
        "eval_gpus_per_node": EVAL_GPUS_PER_NODE,
    }

    changed = diff_paths(source, config)
    disallowed = [path for path in changed if path not in ALLOWED_DIFF_PATHS]
    if disallowed:
        raise GenerationError(f"{slug}: diff outside the allowlist: {disallowed}")
    if "prompt.system" in _flatten(config):
        raise GenerationError(f"{slug}: derived config must not carry prompt.system")
    validate_qwen3omni_config(config)
    resolve_system_prompt(config)
    return reorder(config)


def derive_pooled(source: dict[str, Any], cell: tuple) -> dict[str, Any]:
    """Derive one Turkish pooled t17 Qwen3-Omni prompt-context config.

    Split protocol, leakage unit, windowing, hierarchical weights, label contract,
    checkpoint selection, early stopping, dataset roots, transcript source and
    subject aggregation are inherited unchanged: the pooled audio cells keep the
    source recipe's response-subject mean aggregation, because the locked
    pair-margin rule applies to the pooled text-only cell only.
    """
    slug, source_name, _target, modality = cell
    config = copy.deepcopy(source)
    prompt = config.get("prompt") or {}
    template = prompt.get("user_template")
    language = prompt.get("prompt_language", "english")
    if not template:
        raise GenerationError(f"{source_name}: prompt.user_template is missing")
    if "{question_context}" not in str(template):
        raise GenerationError(f"{source_name}: pooled template must carry {{question_context}}")

    config["model_backend"] = "qwen3omni"
    config["model_name_or_path"] = QWEN3_OMNI_MODEL_PATH
    config["model_attn_implementation"] = QWEN3_OMNI_ATTN_IMPLEMENTATION
    config["recipe_id"] = f"{config['recipe_id']}{RECIPE_SUFFIX}"
    config["manifest_policy"] = MANIFEST_POLICY_PREBUILT
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{RUN_ROOT_CAMPAIGN}/{modality}/{POOLED_RUN_ROOT_DATASET_DIR}"
    )
    config["prompt"] = {
        "version": PROMPT_CONTEXT_VERSION,
        "dataset_context": POOLED_DATASET_CONTEXT,
        "question_context_version": PROMPT_CONTEXT_VERSION,
        "user_template": template,
        "prompt_language": language,
    }

    lora = config["lora"]
    if set(lora) - {"rank", "alpha", "dropout", "bias", "target_modules"}:
        raise GenerationError(f"{source_name}: unexpected lora keys {sorted(lora)}")
    lora["target_modules"] = QWEN3OMNI_LORA_TARGET_REGEX

    training = config["training"]
    training["strategy"] = "fsdp"
    training["activation_offload"] = ACTIVATION_OFFLOAD
    training["run_final_eval_in_train"] = False

    config["evaluation"]["evaluation_view"] = QWEN3OMNI_EVALUATION_VIEW
    config["evaluation"]["inference_dtype"] = INFERENCE_DTYPE

    config["resources"] = {
        "eval_nodes": EVAL_NODES,
        "eval_gpus_per_node": EVAL_GPUS_PER_NODE,
    }
    if str(config.get("dataset_variant")) != "pooled_t17":
        raise GenerationError(f"{slug}: pooled source must declare dataset_variant=pooled_t17")

    changed = diff_paths(source, config)
    disallowed = [path for path in changed if path not in POOLED_ALLOWED_DIFF_PATHS]
    if disallowed:
        raise GenerationError(f"{slug}: diff outside the pooled allowlist: {disallowed}")
    if "prompt.system" in _flatten(config):
        raise GenerationError(f"{slug}: derived config must not carry prompt.system")
    validate_qwen3omni_config(config)
    resolve_system_prompt(config)
    return reorder(config, POOLED_TOP_LEVEL_ORDER)


TOP_LEVEL_ORDER = (
    "dataset",
    "seed",
    "recipe_id",
    "protocol_id",
    "manifest_variant",
    "model_backend",
    "model_name_or_path",
    "model_attn_implementation",
    "dataset_root",
    "label_root",
    "quarantine_path",
    "output_dirs",
    "prompt",
    "labels",
    "data",
    "split",
    "lora",
    "audio_adapter",
    "training",
    "evaluation",
    "resources",
)

# The pooled configs mirror the canonical Turkish pooled config's field order,
# including the keys the DAIC family does not carry.
POOLED_TOP_LEVEL_ORDER = (
    "dataset",
    "dataset_variant",
    "seed",
    "recipe_id",
    "model_backend",
    "model_name_or_path",
    "model_attn_implementation",
    "dataset_root",
    "metadata_csv",
    "transcript_file",
    "threshold",
    "metadata_schema",
    "quarantine_path",
    "manifest_policy",
    "output_dirs",
    "prompt",
    "labels",
    "data",
    "split",
    "lora",
    "audio_adapter",
    "training",
    "evaluation",
    "resources",
)

SECTION_ORDER = {
    "training": (
        "dist_timeout_minutes",
        "strategy",
        "activation_offload",
        "num_train_epochs",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "bf16",
        "gradient_checkpointing",
        "run_final_eval_in_train",
        "dataloader_num_workers",
        "audio_budget_audit",
        "max_grad_norm",
        "class_balance",
        "selection_metric",
        "selection_metric_mode",
        "early_stopping",
    ),
    "evaluation": (
        "sample_prediction_mode",
        "headline_mode",
        "aggregation_level",
        "subject_score_aggregation",
        "hierarchical_score_aggregation",
        "evaluation_view",
        "inference_dtype",
        "generation_max_new_tokens",
        "num_beams",
        "do_sample",
        "evaluate_last_checkpoint",
    ),
    "resources": ("eval_nodes", "eval_gpus_per_node"),
}


def _ordered(mapping: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    result = {key: mapping[key] for key in order if key in mapping}
    result.update({key: value for key, value in mapping.items() if key not in result})
    return result


def reorder(
    config: dict[str, Any], top_level_order: tuple[str, ...] = TOP_LEVEL_ORDER
) -> dict[str, Any]:
    """Keep the derived configs readable in the canonical field order."""
    for section, order in SECTION_ORDER.items():
        if isinstance(config.get(section), dict):
            config[section] = _ordered(config[section], order)
    return _ordered(config, top_level_order)


def _str_representer(dumper: yaml.Dumper, value: str):
    """Render multi-line strings as literal blocks so prompts stay readable."""
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


class _ConfigDumper(yaml.SafeDumper):
    """Local dumper: the block-scalar style must not leak into other callers."""


_ConfigDumper.add_representer(str, _str_representer)


def render(config: dict[str, Any]) -> str:
    return yaml.dump(
        config,
        Dumper=_ConfigDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def emit(target: Path, config: dict[str, Any], *, check_only: bool, failures: list[str]) -> None:
    rendered = render(config)
    if yaml.safe_load(rendered) != config:
        failures.append(f"rendered config does not round-trip: {target}")
        return
    if target.is_file():
        if target.read_text(encoding="utf-8") != rendered:
            failures.append(f"existing file differs from derived content: {target}")
        return
    if check_only:
        failures.append(f"missing derived file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {target.relative_to(PROJECT_ROOT)}")


FAMILIES = (
    ("daic", PRE_DEFAULT_BACKBONE_ARCHIVE, CELLS, derive, ALLOWED_DIFF_PATHS),
    (
        "turkish_pooled",
        POOLED_SOURCE_DIR,
        POOLED_CELLS,
        derive_pooled,
        POOLED_ALLOWED_DIFF_PATHS,
    ),
)


def _emit_cell(
    *,
    family: str,
    source_dir: Path,
    cell: tuple,
    derive_fn: Any,
    allowed_paths: frozenset[str],
    check_only: bool,
    failures: list[str],
    audit: dict[str, Any],
) -> None:
    slug, source_name, target_name, modality = cell
    source_path = source_dir / source_name
    if not source_path.is_file():
        raise GenerationError(f"missing canonical source config: {source_path}")
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    config = derive_fn(source, cell)
    target = MAIN / target_name
    emit(target, config, check_only=check_only, failures=failures)
    changed = diff_paths(source, config)
    audit["configs"].append(
        {
            "cell_id": slug,
            "family": family,
            "modality": modality,
            "source": str(source_path.relative_to(PROJECT_ROOT)),
            "config": str(target.relative_to(PROJECT_ROOT)),
            "changed_paths": changed,
            "allowed_paths": sorted(allowed_paths),
            "allowed": not (set(changed) - set(allowed_paths)),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help="structured diff audit path (default: outputs/.../config_diff_audit.json)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    audit: dict[str, Any] = {
        "schema_version": "audiollm.qwen3omni_config_diff.v1",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "family_allowed_paths": {
            family: sorted(allowed) for family, _source, _cells, _derive, allowed in FAMILIES
        },
        "activation_offload": ACTIVATION_OFFLOAD,
        "evaluation_shape": {"nodes": EVAL_NODES, "gpus_per_node": EVAL_GPUS_PER_NODE},
        "configs": [],
    }
    for family, source_dir, cells, derive_fn, allowed_paths in FAMILIES:
        for cell in cells:
            _emit_cell(
                family=family,
                source_dir=source_dir,
                cell=cell,
                derive_fn=derive_fn,
                allowed_paths=allowed_paths,
                check_only=args.check,
                failures=failures,
                audit=audit,
            )

    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.audit_output}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
