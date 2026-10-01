#!/usr/bin/env python3
"""Generate the Qwen3 English-transcript default family (eight cells).

Each cell is derived from the current **native** Qwen3 default config of the
same dataset and modality, so the English family isolates exactly one input
change: the transcript overlay. The derivation is deterministic and the audit
fails when a diff falls outside the allowlist:

* ``recipe_id`` gains the ``_en`` marker;
* a ``transcripts`` overlay block selects the accepted English translation
  (``variant: english``, ``minimum_status: automatic_low``,
  ``require_complete: true``, ``include_failed: false``). The Turkish pooled
  cells keep the accepted pooled source contract: their manifest is prebuilt
  from the pooled source manifests, so ``cache_path`` stays the documented
  inert placeholder and must never be pointed at a pos-only translation cache;
* the manifest/split directories move to the isolated English roots
  (``outputs/manifests_harmonized_en/``, ``outputs/splits_harmonized_en/``);
* the output root moves to the English Qwen3 campaign
  (``output_model/promptcontext_v1_qwen38_likelihood_en/`` and
  ``output_model/promptcontext_v1_qwen3omni_likelihood_en/``).

Everything else is inherited byte for byte: pinned model identity and revision,
prompt context (including the Turkish pooled question-context sentences), LoRA
policy, FSDP training shape, likelihood evaluation view/dtype, split protocol,
seed, label contract, windowing and aggregation.

The script also writes the default English matrix (eight cells, fixed heads
empty because Qwen3 head execution is explicit-only and never dispatched by the
harmonized launchers) and is idempotent: ``--check`` reports every difference
without writing, and an existing file with different content is never
overwritten silently. The structured diff audit is written under ``outputs/``
(not tracked).

This generator only prepares configs. It does not train, submit or evaluate
anything, and it never rewrites the historical Qwen2 English configs or the
legacy English matrix.
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

from src.data.prompt_context import (
    PROMPT_CONTEXT_VERSION,
    TRANSLATION_NOTICE_VERSION,
    resolve_system_prompt,
    resolve_translation_notice,
)
from src.experiment_tracking.manifest_policy import MANIFEST_POLICY_PREBUILT, validate_manifest_policy
from src.model.qwen38_lora import validate_qwen38_config
from src.model.qwen3omni_lora import validate_qwen3omni_config

MAIN = PROJECT_ROOT / "configs/main"
MATRIX = PROJECT_ROOT / "configs/experiments/harmonized/english_translation_matrix.yaml"
DEFAULT_AUDIT_OUTPUT = PROJECT_ROOT / "outputs/qwen3_english_configs/config_diff_audit.json"

EN_MANIFEST_ROOT = "${PROJECT_ROOT}/outputs/manifests_harmonized_en"
EN_SPLIT_ROOT = "${PROJECT_ROOT}/outputs/splits_harmonized_en"
QWEN38_EN_RUN_ROOT = "${PROJECT_ROOT}/output_model/promptcontext_v1_qwen38_likelihood_en"
QWEN3OMNI_EN_RUN_ROOT = "${PROJECT_ROOT}/output_model/promptcontext_v1_qwen3omni_likelihood_en"
TRANSLATION_ROOT = (
    "${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
    "/harmonized_en_complete_v1"
)
# The accepted pooled English contract keeps this placeholder: the pooled
# English manifest is prebuilt from the pooled source manifests and the worker
# never reads a worker-side translation cache. The value must not be replaced
# with a pos-only accepted cache.
POOLED_TRANSCRIPTS_CACHE_PLACEHOLDER = "pooled_source_manifest_translations"
EN_RECIPE_SUFFIX = "_en"
# Head execution for the Qwen3 backbones needs a separate hidden-extraction
# support task; the default English matrix therefore declares no fixed heads.
EXPECTED_FIXED_HEADS: list[str] = []

# (slug, source config name, target config name, manifest dataset dir, run dataset dir,
#  modality, folds, separate_eval, transcripts cache dir or None for pooled prebuilt)
CELLS = (
    (
        "d3tec_text_only",
        "d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
        "d3tec",
        "d3tec",
        "text_only",
        (0, 1, 2, 3, 4),
        True,
        "d3tec",
    ),
    (
        "d3tec_audio_text",
        "d3tec_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
        "d3tec_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
        "d3tec",
        "d3tec",
        "audio_text",
        (0, 1, 2, 3, 4),
        True,
        "d3tec",
    ),
    (
        "androids_text_only",
        "androids_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "androids_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
        "androids",
        "androids_interview",
        "text_only",
        (0, 1, 2, 3, 4),
        True,
        "androids_interview",
    ),
    (
        "androids_audio_text",
        "androids_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
        "androids_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
        "androids",
        "androids_interview",
        "audio_text",
        (0, 1, 2, 3, 4),
        True,
        "androids_interview",
    ),
    (
        "cmdc_text_only",
        "cmdc_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "cmdc_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
        "cmdc",
        "cmdc",
        "text_only",
        (0, 1, 2, 3, 4),
        False,
        "cmdc",
    ),
    (
        "cmdc_audio_text",
        "cmdc_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
        "cmdc_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
        "cmdc",
        "cmdc",
        "audio_text",
        (0, 1, 2, 3, 4),
        False,
        "cmdc",
    ),
    (
        "turkish_pooled_text_only",
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
        "turkish_pooled_t17_qwen3asr",
        "turkish",
        "text_only",
        (0, 1, 2, 3, 4),
        False,
        None,
    ),
    (
        "turkish_pooled_audio_text",
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
        "turkish_pooled_t17_qwen3asr",
        "turkish",
        "audio_text",
        (0, 1, 2, 3, 4),
        False,
        None,
    ),
)

# Every diff between a native source config and its derived English cell must be
# one of these leaf paths. Anything else fails the audit.
ALLOWED_DIFF_PATHS = frozenset(
    {
        "recipe_id",
        "transcripts",
        "transcripts.variant",
        "transcripts.cache_path",
        "transcripts.minimum_status",
        "transcripts.require_complete",
        "transcripts.include_failed",
        "output_dirs.manifest_dir",
        "output_dirs.split_dir",
        "output_dirs.run_root",
        "prompt.translation_notice_version",
    }
)

TRANSCRIPTS_POLICY = {
    "variant": "english",
    "minimum_status": "automatic_low",
    "require_complete": True,
    "include_failed": False,
}


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


def transcripts_block(cell: tuple) -> dict[str, Any]:
    slug, _source, _target, _manifest_dir, _run_dir, _modality, _folds, _eval, cache_dir = cell
    block: dict[str, Any] = dict(TRANSCRIPTS_POLICY)
    if cache_dir is None:
        block["cache_path"] = POOLED_TRANSCRIPTS_CACHE_PLACEHOLDER
    else:
        block["cache_path"] = f"{TRANSLATION_ROOT}/{cache_dir}/accepted.jsonl"
    return block


def derive(source: dict[str, Any], cell: tuple) -> dict[str, Any]:
    slug, source_name, _target, manifest_dir, run_dir, modality, _folds, _separate, cache_dir = cell
    config = copy.deepcopy(source)
    if str(config.get("dataset_variant", "") or "") == "pooled_t17":
        if cache_dir is not None:
            raise GenerationError(f"{slug}: pooled cells must use the pooled transcript contract")
    elif cache_dir is None:
        raise GenerationError(f"{slug}: non-pooled cells need an accepted translation cache")

    config["recipe_id"] = f"{config['recipe_id']}{EN_RECIPE_SUFFIX}"
    config["transcripts"] = transcripts_block(cell)
    config["prompt"]["translation_notice_version"] = TRANSLATION_NOTICE_VERSION
    config["output_dirs"]["manifest_dir"] = f"{EN_MANIFEST_ROOT}/{manifest_dir}"
    config["output_dirs"]["split_dir"] = f"{EN_SPLIT_ROOT}/{manifest_dir}"
    if modality == "text_only":
        config["output_dirs"]["run_root"] = f"{QWEN38_EN_RUN_ROOT}/text_only/{run_dir}"
    else:
        config["output_dirs"]["run_root"] = f"{QWEN3OMNI_EN_RUN_ROOT}/audio_text/{run_dir}"

    changed = diff_paths(source, config)
    disallowed = [path for path in changed if path not in ALLOWED_DIFF_PATHS]
    if disallowed:
        raise GenerationError(f"{slug}: diff outside the allowlist: {disallowed}")

    transcripts = config["transcripts"]
    if transcripts["variant"] != "english":
        raise GenerationError(f"{slug}: transcripts.variant must be english")
    if cache_dir is None:
        policy = validate_manifest_policy(config)
        if policy != MANIFEST_POLICY_PREBUILT:
            raise GenerationError(f"{slug}: pooled cells must stay on the prebuilt manifest policy")
        if transcripts["cache_path"] != POOLED_TRANSCRIPTS_CACHE_PLACEHOLDER:
            raise GenerationError(
                f"{slug}: the pooled English cell must keep the documented inert cache placeholder"
            )
    elif not str(transcripts["cache_path"]).endswith("/accepted.jsonl"):
        raise GenerationError(f"{slug}: the English overlay must point at an accepted cache")

    validate_qwen38_config(config)
    validate_qwen3omni_config(config)
    resolve_system_prompt(config)
    if resolve_translation_notice(config) is None:
        raise GenerationError(f"{slug}: the English cell must carry the translation notice")
    return reorder(config)


TOP_LEVEL_ORDER = (
    "dataset",
    "dataset_variant",
    "seed",
    "recipe_id",
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "model_attn_implementation",
    "dataset_root",
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
    "audio_adapter",
    "training",
    "evaluation",
    "resources",
    "transcripts",
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
    "transcripts": ("variant", "cache_path", "minimum_status", "require_complete", "include_failed"),
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


def emit(
    target: Path,
    content: str,
    *,
    check_only: bool,
    failures: list[str],
    replace: bool = False,
    replacements: list[str] | None = None,
) -> None:
    """Write the derived content, refusing to change an existing file silently.

    An existing file with different content is a failure unless the caller
    passes the explicit one-time ``replace`` transition, which records the
    replaced path so the audit shows exactly what changed.
    """
    if target.is_file():
        if target.read_text(encoding="utf-8") != content:
            if replace and not check_only:
                target.write_text(content, encoding="utf-8")
                print(f"replaced {target.relative_to(PROJECT_ROOT)}")
                if replacements is not None:
                    replacements.append(str(target.relative_to(PROJECT_ROOT)))
            else:
                failures.append(f"existing file differs from derived content: {target}")
        return
    if check_only:
        failures.append(f"missing derived file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote {target.relative_to(PROJECT_ROOT)}")


def emit_config(
    target: Path,
    config: dict[str, Any],
    *,
    check_only: bool,
    failures: list[str],
    replace: bool = False,
    replacements: list[str] | None = None,
) -> None:
    rendered = render(config)
    if yaml.safe_load(rendered) != config:
        failures.append(f"rendered config does not round-trip: {target}")
        return
    emit(
        target,
        rendered,
        check_only=check_only,
        failures=failures,
        replace=replace,
        replacements=replacements,
    )


def target_matrix_differs(target: Path, content: str) -> bool:
    return target.is_file() and target.read_text(encoding="utf-8") != content


def build_matrix() -> dict[str, Any]:
    return {
        "name": "harmonized_v1_en_standalone_matrix_qwen3",
        "recipe_id": "harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_v1_promptcontext_v1_en",
        "recipe_note": (
            "Turkish pooled cells carry the qcond variant "
            "(..._likelihood_qcond_v1_promptcontext_v1_en)."
        ),
        "seed": 1337,
        "max_epochs": 20,
        "checkpoint_selection": "inner_val_macro_f1",
        "fixed_heads": list(EXPECTED_FIXED_HEADS),
        "optuna": False,
        "experiments": [
            {
                "config": f"configs/main/{cell[2]}",
                "folds": list(cell[6]),
                "separate_eval": cell[7],
            }
            for cell in CELLS
        ],
    }


def render_matrix(matrix: dict[str, Any]) -> str:
    lines = [
        f"name: {matrix['name']}",
        f"recipe_id: {matrix['recipe_id']}",
        f"recipe_note: {matrix['recipe_note']}",
        f"seed: {matrix['seed']}",
        f"max_epochs: {matrix['max_epochs']}",
        f"checkpoint_selection: {matrix['checkpoint_selection']}",
        f"fixed_heads: {matrix['fixed_heads']}",
        f"optuna: {'true' if matrix['optuna'] else 'false'}",
        "experiments:",
    ]
    for item in matrix["experiments"]:
        folds = ", ".join(str(fold) for fold in item["folds"])
        separate = "true" if item["separate_eval"] else "false"
        lines.append(
            f"  - {{config: {item['config']}, folds: [{folds}], separate_eval: {separate}}}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument(
        "--replace-matrix",
        action="store_true",
        help=(
            "one-time transition: replace the default English matrix even when it "
            "differs from the derived content (the pre-Qwen3 selection is preserved "
            "in english_translation_matrix_legacy_qwen2.yaml)"
        ),
    )
    parser.add_argument(
        "--replace-cells",
        action="store_true",
        help=(
            "one-time transition: replace existing generated English cells even when they "
            "differ from the derived content (used when the derived contract changes, for "
            "example a new prompt.translation_notice_version); the replaced paths are "
            "recorded in the audit"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help="structured diff audit path (default: outputs/.../config_diff_audit.json)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    replacements: list[str] = []
    audit: dict[str, Any] = {
        "schema_version": "audiollm.qwen3_english_config_diff.v1",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "translation_notice_version": TRANSLATION_NOTICE_VERSION,
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "matrix": str(MATRIX.relative_to(PROJECT_ROOT)),
        "configs": [],
    }
    for cell in CELLS:
        slug, source_name, target_name, manifest_dir, run_dir, modality, folds, separate_eval, cache_dir = cell
        source_path = MAIN / source_name
        if not source_path.is_file():
            raise GenerationError(f"missing native source config: {source_path}")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config = derive(source, cell)
        target = MAIN / target_name
        emit_config(
            target,
            config,
            check_only=args.check,
            failures=failures,
            replace=args.replace_cells,
            replacements=replacements,
        )
        audit["configs"].append(
            {
                "cell_id": slug,
                "modality": modality,
                "source": f"configs/main/{source_name}",
                "config": f"configs/main/{target_name}",
                "changed_paths": diff_paths(source, config),
                "allowed": True,
                "manifest_dir": config["output_dirs"]["manifest_dir"],
                "split_dir": config["output_dirs"]["split_dir"],
                "run_root": config["output_dirs"]["run_root"],
                "recipe_id": config["recipe_id"],
                "cache_path": config["transcripts"]["cache_path"],
                "folds": list(folds),
                "separate_eval": separate_eval,
            }
        )

    matrix = build_matrix()
    matrix_rendered = render_matrix(matrix)
    if yaml.safe_load(matrix_rendered) != matrix:
        raise GenerationError("rendered English matrix does not round-trip")
    matrix_replaced = False
    if target_matrix_differs(MATRIX, matrix_rendered):
        if args.replace_matrix and not args.check:
            print(f"replacing the default English matrix: {MATRIX.relative_to(PROJECT_ROOT)}")
            MATRIX.write_text(matrix_rendered, encoding="utf-8")
            matrix_replaced = True
        else:
            failures.append(f"existing file differs from derived content: {MATRIX}")
    emit(MATRIX, matrix_rendered, check_only=args.check, failures=failures)
    audit["matrix_configs"] = [item["config"] for item in matrix["experiments"]]
    audit["matrix_replaced"] = matrix_replaced
    audit["cells_replaced"] = replacements
    audit["matrix_legacy"] = "configs/experiments/harmonized/english_translation_matrix_legacy_qwen2.yaml"
    audit["fixed_heads"] = matrix["fixed_heads"]
    audit["training_folds"] = sum(len(item["folds"]) for item in matrix["experiments"])
    audit["separate_eval_folds"] = sum(
        len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"]
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
    try:
        raise SystemExit(main())
    except GenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
