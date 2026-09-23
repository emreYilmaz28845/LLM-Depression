#!/usr/bin/env python3
"""Generate the Qwen3.8 standalone prompt-context family (five text-only cells).

Every cell is derived from its canonical source config and changes only the
allowed fields, so the family represents a prompt change on the Qwen3.8 backend:

* the prompt recipe version and the dataset context key replace the inline
  ``prompt.system`` text (``src/data/prompt_context.py``);
* the Turkish pooled cell additionally selects the versioned question-context
  sentence set and the prebuilt-manifest policy;
* the recipe id, the output root and the explicit evaluation view/dtype are new;
* the Qwen3.8 backend requires ``model_backend``, the pinned model path and
  revision, the PR #259 LoRA target set, the FSDP training strategy and the
  CPU activation offload the validated PR #259 production run used.

Split protocol, seed, label contract, transcript source, aggregation, windowing,
early stopping and the canonical epoch count are inherited unchanged. The script
is deterministic and idempotent: ``--check`` reports every difference from the
derived content without writing, and an existing file with different content is
never overwritten silently.

The structured diff audit lists the changed leaf paths of every cell and fails
when a change is outside the allowlist. It is written next to a deterministic
JSON summary under ``outputs/`` (not tracked).
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

MAIN = PROJECT_ROOT / "configs/main"
PRE_DEFAULT_BACKBONE_ARCHIVE = PROJECT_ROOT / "configs/archive/pre_default_backbone_20260923"
MATRIX = PROJECT_ROOT / "configs/experiments/promptcontext_qwen38/matrix.yaml"
DEFAULT_AUDIT_OUTPUT = (
    PROJECT_ROOT / "outputs/prompt_context_config_diff/config_diff_audit.json"
)


def source_config_path(source_name: str) -> Path:
    """Use the preserved Qwen2 source after canonical filenames change backend."""
    archived = PRE_DEFAULT_BACKBONE_ARCHIVE / source_name
    return archived if archived.is_file() else MAIN / source_name

QWEN38_MODEL_PATH = "${QWEN38_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
QWEN38_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN38_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
RECIPE_SUFFIX = "_promptcontext_v1"
RUN_ROOT_CAMPAIGN = "promptcontext_v1_qwen38_likelihood"
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
INFERENCE_DTYPE = "bf16"
ACTIVATION_OFFLOAD = "cpu"
MANIFEST_POLICY_PREBUILT = "prebuilt"

# (slug, source config name, target config name, dataset dir, dataset-context key, folds, pooled)
CELLS = (
    (
        "daic",
        "daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml",
        "daic_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "daic",
        "daic",
        (0,),
        False,
    ),
    (
        "d3tec",
        "d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "d3tec",
        "d3tec",
        (0, 1, 2, 3, 4),
        False,
    ),
    (
        "androids",
        "androids_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "androids_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        # The run root carries the real dataset name so the managed submission's
        # dataset qualifier, the config and the runtime override all agree.
        "androids_interview",
        "androids",
        (0, 1, 2, 3, 4),
        False,
    ),
    (
        "cmdc",
        "cmdc_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "cmdc_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "cmdc",
        "cmdc",
        (0, 1, 2, 3, 4),
        False,
    ),
    (
        "turkish_pooled",
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml",
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "turkish",
        "turkish_pooled",
        (0, 1, 2, 3, 4),
        True,
    ),
)

# Every diff between a source config and its derived cell must be one of these
# leaf paths. Anything else fails the audit instead of shipping a silent change.
ALLOWED_DIFF_PATHS = frozenset(
    {
        "model_backend",
        "model_name_or_path",
        "model_revision",
        "recipe_id",
        "output_dirs.run_root",
        "prompt.system",
        "prompt.version",
        "prompt.dataset_context",
        "prompt.question_context_version",
        "prompt.user_template",
        "prompt.prompt_language",
        "lora.target_modules",
        "training.strategy",
        "training.activation_offload",
        "training.run_final_eval_in_train",
        "evaluation.evaluation_view",
        "evaluation.inference_dtype",
        "manifest_policy",
    }
)


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
    slug, source_name, _target, dataset_dir, context_key, _folds, pooled = cell
    config = copy.deepcopy(source)
    prompt = config.get("prompt") or {}
    template = prompt.get("user_template")
    language = prompt.get("prompt_language", "english")
    if not template:
        raise GenerationError(f"{source_name}: prompt.user_template is missing")

    config["model_backend"] = "qwen38"
    config["model_name_or_path"] = QWEN38_MODEL_PATH
    config["model_revision"] = QWEN38_MODEL_REVISION
    config["recipe_id"] = f"{config['recipe_id']}{RECIPE_SUFFIX}"
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{RUN_ROOT_CAMPAIGN}/text_only/{dataset_dir}"
    )
    new_prompt: dict[str, Any] = {
        "version": PROMPT_CONTEXT_VERSION,
        "dataset_context": context_key,
    }
    if pooled:
        new_prompt["question_context_version"] = PROMPT_CONTEXT_VERSION
    new_prompt["user_template"] = template
    new_prompt["prompt_language"] = language
    config["prompt"] = new_prompt

    lora = config["lora"]
    if set(lora) - {"rank", "alpha", "dropout", "bias", "target_modules"}:
        raise GenerationError(f"{source_name}: unexpected lora keys {sorted(lora)}")
    lora["target_modules"] = QWEN38_LORA_TARGET_REGEX

    training = config["training"]
    training["strategy"] = "fsdp"
    training["activation_offload"] = ACTIVATION_OFFLOAD
    training["run_final_eval_in_train"] = False

    config["evaluation"]["evaluation_view"] = EVALUATION_VIEW
    config["evaluation"]["inference_dtype"] = INFERENCE_DTYPE

    if pooled:
        config["manifest_policy"] = MANIFEST_POLICY_PREBUILT

    changed = diff_paths(source, config)
    disallowed = [path for path in changed if path not in ALLOWED_DIFF_PATHS]
    if disallowed:
        raise GenerationError(
            f"{slug}: diff outside the allowlist: {disallowed}"
        )
    if "prompt.system" in _flatten(config):
        raise GenerationError(f"{slug}: derived config must not carry prompt.system")
    resolve_system_prompt(config)
    return reorder(config)


def build_matrix() -> dict[str, Any]:
    cells = []
    for cell in CELLS:
        slug, source_name, target, _dataset_dir, context_key, folds, _pooled = cell
        source = yaml.safe_load(source_config_path(source_name).read_text(encoding="utf-8"))
        config = derive(source, cell)
        dataset = str(config["dataset"])
        cells.append(
            {
                "cell_id": slug,
                "dataset": dataset,
                "dataset_variant": config.get("dataset_variant"),
                "modality": "text_only",
                "backbone": "qwen38",
                "prompt_context_dataset": context_key,
                "recipe_id": config["recipe_id"],
                "config": f"configs/main/{target}",
                "folds": list(folds),
                "separate_eval": True,
            }
        )
    return {
        "name": "promptcontext_v1_qwen38_standalone_matrix",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "seed": 1337,
        "max_epochs": 20,
        "checkpoint_selection": "inner_val_macro_f1",
        "evaluation_backend": "likelihood",
        "evaluation_view": EVALUATION_VIEW,
        "experiments": cells,
    }


TOP_LEVEL_ORDER = (
    "dataset",
    "dataset_variant",
    "seed",
    "recipe_id",
    "protocol_id",
    "manifest_variant",
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "dataset_root",
    "label_root",
    "full_transcript_path",
    "segment_transcript_path",
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
    "training",
    "evaluation",
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
        "evaluation_view",
        "inference_dtype",
        "generation_max_new_tokens",
        "num_beams",
        "do_sample",
        "evaluate_last_checkpoint",
    ),
}


def _ordered(mapping: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    result = {key: mapping[key] for key in order if key in mapping}
    result.update({key: value for key, value in mapping.items() if key not in result})
    return result


def reorder(config: dict[str, Any]) -> dict[str, Any]:
    """Keep the derived configs readable in the canonical field order."""
    for section, order in SECTION_ORDER.items():
        if isinstance(config.get(section), dict):
            config[section] = _ordered(config[section], order)
    return _ordered(config, TOP_LEVEL_ORDER)


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
        "schema_version": "audiollm.promptcontext_config_diff.v1",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "configs": [],
    }
    for cell in CELLS:
        slug, source_name, target_name, dataset_dir, context_key, folds, pooled = cell
        source_path = source_config_path(source_name)
        if not source_path.is_file():
            raise GenerationError(f"missing canonical source config: {source_path}")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config = derive(source, cell)
        target = MAIN / target_name
        emit(target, config, check_only=args.check, failures=failures)
        audit["configs"].append(
            {
                "cell_id": slug,
                "source": f"configs/main/{source_name}",
                "config": f"configs/main/{target.name}",
                "changed_paths": diff_paths(source, config),
                "allowed": True,
                "dataset_context": context_key,
                "manifest_policy": config.get("manifest_policy", "build"),
                "folds": list(folds),
            }
        )

    matrix = build_matrix()
    emit(MATRIX, matrix, check_only=args.check, failures=failures)
    audit["matrix"] = "configs/experiments/promptcontext_qwen38/matrix.yaml"
    audit["training_fits"] = sum(len(cell[5]) for cell in CELLS)

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
