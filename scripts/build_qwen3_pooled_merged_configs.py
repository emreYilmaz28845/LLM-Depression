#!/usr/bin/env python3
"""Generate the five Qwen3 pooled merged contracts.

The contracts are derived from the existing symmetric-merged configs, so the
merged methodology is inherited by construction and the diff audit proves it:
protocol settings, selection metric (``mean_dataset_macro_f1``), folds and heads
config stay byte-identical, and the training/execution shape changes only by the
explicit FSDP recipe and evaluation shape documented below. Only the documented
contract fields change:

* explicit ``model_backend`` identity (Qwen3.8-27B for text-only, Qwen3-Omni
  30B-A3B Thinker for the audio modalities) with the pinned model path and, for
  Qwen3.8, the pinned revision;
* the five components point at the current native default cells; the Turkish
  component is the pooled Qwen3 config, and the English contracts swap the four
  translated datasets to their Qwen3 English cells while DAIC keeps its native
  English input (text-only and audio+text; no English audio-only contract,
  which would be input-identical to its native counterpart);
* the pooled English component keeps the prebuilt manifest contract;
* isolated merged roots under ``symmetric_merged/qwen3_pooled_{native,english}``;
* the explicit FSDP training recipe (``training.strategy: fsdp`` with CPU
  activation offload) and, for the audio routes, the sharded evaluation shape
  (``execution.postprocess_gpus: 4``) that matches the components' declared
  ``resources.eval_gpus_per_node``;
* a per-contract ``status`` line (``smoke_only`` until the route passes its own
  bounded GPU smoke chain, then ``execute_verified``). The config field is
  documentation; the submission guard keys on the contract identity and its own
  readiness table and ignores nothing here.

The script is deterministic and idempotent: ``--check`` reports every difference
without writing, and an existing file with different content is never overwritten
silently. The structured diff audit is written under ``outputs/`` (not tracked).
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

from scripts import build_qwen3_english_configs as english_configs  # noqa: E402

MERGED = PROJECT_ROOT / "configs/experiments/merged"
MAIN = PROJECT_ROOT / "configs/main"
DEFAULT_AUDIT_OUTPUT = PROJECT_ROOT / "outputs/qwen3_pooled_merged_configs/config_diff_audit.json"

QWEN38_MODEL_PATH = "${QWEN38_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
QWEN38_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN3OMNI_MODEL_PATH = (
    "${QWEN3_OMNI_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"
)
QWEN3OMNI_ATTN_IMPLEMENTATION = "sdpa"

BLOCKED_STATUS = "blocked_prerequisite"
SMOKE_ONLY_STATUS = "smoke_only"
EXECUTE_VERIFIED_STATUS = "execute_verified"

# Per-contract readiness. ``smoke_only`` means the bounded smoke stage is the
# verification mechanism and the multi-fold production guard stays closed; a
# route moves to ``execute_verified`` only after its own GPU smoke chain
# passed. The submission guard reads its own readiness table (contract name +
# resolved backend + modality), so this field is documentation and is kept in
# sync with it.
CELL_STATUS: dict[str, tuple[str, str]] = {
    "native_text_only": (
        EXECUTE_VERIFIED_STATUS,
        "Qwen3 merged FSDP/postprocess GPU smoke chain passed (run qwen3_merged_smoke_text_only_20260930_r3: "
        "train 46844006 and postprocess 46845076 COMPLETED 0:0; deployment "
        "feat-qwen3-merged-fsdp-postprocess-20260930-20260930T142929Z-c41d8e76-91bf3e5a, source "
        "c41d8e76c12979c358499a8b84fce035f67b6478), so the cv and final stages are executable, and the "
        "route passed its bounded hidden-feature audit on that smoke checkpoint, so its head kind is "
        "open.",
    ),
    "native_audio_text": (
        SMOKE_ONLY_STATUS,
        "The declared shape moved to the two-node lane (execution.train_nodes 2 with "
        "gradient_accumulation_steps 16) after the recorded one-node chain, so that chain's evidence "
        "no longer matches the declared shape; a smoke chain in the two-node shape is pending.",
    ),
    "native_audio_only": (
        SMOKE_ONLY_STATUS,
        "The declared shape moved to the two-node lane (execution.train_nodes 2 with "
        "gradient_accumulation_steps 16) after the recorded one-node chain, so that chain's evidence "
        "no longer matches the declared shape; a smoke chain in the two-node shape is pending.",
    ),
    "english_text_only": (
        EXECUTE_VERIFIED_STATUS,
        "Qwen3 merged FSDP/postprocess GPU smoke chain passed (run "
        "qwen3_multiseed_smoke_en_text_20260930_r1: train 46852254 and postprocess 46852255 "
        "COMPLETED 0:0; deployment "
        "feat-qwen3-multiseed-matrix-readiness-20260930-20260930T183953Z-4ff77c53-521d2e6a, source "
        "4ff77c53ebd3808671af551a58287136bd1726e5), so the cv and final stages are executable; the "
        "four translated components render the versioned translation notice, and the route passed its "
        "bounded hidden-feature audit on that smoke checkpoint, so its head kind is open.",
    ),
    "english_audio_text": (
        SMOKE_ONLY_STATUS,
        "The declared shape moved to the two-node lane (execution.train_nodes 2 with "
        "gradient_accumulation_steps 16) after the recorded one-node chain, so that chain's evidence "
        "no longer matches the declared shape; a smoke chain in the two-node shape is pending.",
    ),
}

NATIVE_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_v1_promptcontext_v1"
ENGLISH_RECIPE = f"{NATIVE_RECIPE}_en"

# The pooled Qwen3 component per modality.
POOLED_COMPONENT = {
    "audio_only": (
        "configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
    "audio_text": (
        "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
    "text_only": (
        "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
        "_promptcontext_v1_qwen38_27b.yaml"
    ),
}
POOLED_COMPONENT_EN = {
    "text_only": (
        "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
        "_promptcontext_v1_en_qwen38_27b.yaml"
    ),
    "audio_text": (
        "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_en_qwen3omni_30b_a3b.yaml"
    ),
}
POOLED_MANIFEST = {
    "native": (
        "outputs/manifests_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest.jsonl",
        "outputs/splits_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest_metadata.json",
    ),
    "english": (
        "outputs/manifests_harmonized_en/turkish_pooled_t17_qwen3asr/turkish_manifest.jsonl",
        "outputs/splits_harmonized_en/turkish_pooled_t17_qwen3asr/turkish_manifest_metadata.json",
    ),
}

# The five components, in the existing merged order.
def native_components(modality: str) -> list[dict[str, str]]:
    return [
        {
            "name": "daic",
            "config": f"configs/main/daic_{modality}_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/daic/daic_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/daic/daic_manifest_metadata.json",
        },
        {
            "name": "cmdc",
            "config": f"configs/main/cmdc_{modality}_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/cmdc/cmdc_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/cmdc/cmdc_manifest_metadata.json",
        },
        {
            "name": "turkish",
            "config": POOLED_COMPONENT[modality],
            "manifest_path": POOLED_MANIFEST["native"][0],
            "metadata_path": POOLED_MANIFEST["native"][1],
        },
        {
            "name": "d3tec",
            "config": f"configs/main/d3tec_{modality}_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/d3tec/d3tec_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/d3tec/d3tec_manifest_metadata.json",
        },
        {
            "name": "androids_interview",
            "config": f"configs/main/androids_{modality}_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/androids/androids_interview_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/androids/androids_interview_manifest_metadata.json",
        },
    ]


def english_components(modality: str) -> list[dict[str, str]]:
    """The five English components for one modality, in the merged order.

    DAIC keeps its native English input; the other four datasets use their
    generated English cells (their config name and English manifest/split roots
    come from the English generator's cell table, so the two generators cannot
    drift apart silently).
    """
    english_cells = {
        cell[4]: (cell[2], cell[3])
        for cell in english_configs.CELLS
        if cell[5] == modality
    }
    missing = {"cmdc", "d3tec", "androids_interview", "turkish"} - set(english_cells)
    if missing:
        raise GenerationError(
            f"the English generator has no {modality} cell for {sorted(missing)}"
        )
    components: list[dict[str, str]] = [
        {
            "name": "daic",
            "config": f"configs/main/daic_{modality}_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/daic/daic_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/daic/daic_manifest_metadata.json",
        },
        {
            "name": "cmdc",
            "config": f"configs/main/{english_cells['cmdc'][0]}",
            "manifest_path": f"outputs/manifests_harmonized_en/{english_cells['cmdc'][1]}/cmdc_manifest.jsonl",
            "metadata_path": f"outputs/splits_harmonized_en/{english_cells['cmdc'][1]}/cmdc_manifest_metadata.json",
        },
        {
            "name": "turkish",
            "config": POOLED_COMPONENT_EN[modality],
            "manifest_path": POOLED_MANIFEST["english"][0],
            "metadata_path": POOLED_MANIFEST["english"][1],
        },
        {
            "name": "d3tec",
            "config": f"configs/main/{english_cells['d3tec'][0]}",
            "manifest_path": f"outputs/manifests_harmonized_en/{english_cells['d3tec'][1]}/d3tec_manifest.jsonl",
            "metadata_path": f"outputs/splits_harmonized_en/{english_cells['d3tec'][1]}/d3tec_manifest_metadata.json",
        },
        {
            "name": "androids_interview",
            "config": f"configs/main/{english_cells['androids_interview'][0]}",
            "manifest_path": (
                "outputs/manifests_harmonized_en/"
                f"{english_cells['androids_interview'][1]}/androids_interview_manifest.jsonl"
            ),
            "metadata_path": (
                "outputs/splits_harmonized_en/"
                f"{english_cells['androids_interview'][1]}/androids_interview_manifest_metadata.json"
            ),
        },
    ]
    return components


# (slug, legacy source config, target config, modality, backend, language)
CELLS = (
    (
        "native_text_only",
        "symmetric_merged_harmonized_text_only_likelihood_v1.yaml",
        "symmetric_merged_qwen3_pooled_native_text_only.yaml",
        "text_only",
        "qwen38",
        "native",
    ),
    (
        "native_audio_only",
        "symmetric_merged_harmonized_audio_only_likelihood_v1.yaml",
        "symmetric_merged_qwen3_pooled_native_audio_only.yaml",
        "audio_only",
        "qwen3omni",
        "native",
    ),
    (
        "native_audio_text",
        "symmetric_merged_harmonized_audio_text_likelihood_v1.yaml",
        "symmetric_merged_qwen3_pooled_native_audio_text.yaml",
        "audio_text",
        "qwen3omni",
        "native",
    ),
    (
        "english_text_only",
        "symmetric_merged_harmonized_text_only_likelihood_v1.yaml",
        "symmetric_merged_qwen3_pooled_english_text_only.yaml",
        "text_only",
        "qwen38",
        "english",
    ),
    (
        "english_audio_text",
        "symmetric_merged_harmonized_audio_text_likelihood_v1.yaml",
        "symmetric_merged_qwen3_pooled_english_audio_text.yaml",
        "audio_text",
        "qwen3omni",
        "english",
    ),
)

ALLOWED_DIFF_PATHS = frozenset(
    {
        "name",
        "model_backend",
        "model_name_or_path",
        "model_revision",
        "model_attn_implementation",
        "recipe_id",
        "status",
        "status_reason",
        "output_dirs.merged_root",
        "output_dirs.run_root",
        # The Qwen3 contracts carry their FSDP recipe and evaluation shape
        # explicitly so the planner, the workers and the guard all read the same
        # declared contract.
        "training.strategy",
        "training.activation_offload",
        "training.gradient_accumulation_steps",
        "execution.postprocess_gpus",
        "execution.train_nodes",
    }
)
COMPONENT_PATHS = ("config", "manifest_path", "metadata_path")


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
    """Leaf paths that differ, excluding the component arrays.

    Component differences are reported per field by ``component_diff`` so the
    audit reads in the same granularity as the allowlist.
    """
    source_flat = _flatten(source)
    target_flat = _flatten(target)
    for key in list(source_flat):
        if key == "components" or key.startswith("components."):
            del source_flat[key]
    for key in list(target_flat):
        if key == "components" or key.startswith("components."):
            del target_flat[key]
    return sorted(
        path
        for path in set(source_flat) | set(target_flat)
        if source_flat.get(path, "<missing>") != target_flat.get(path, "<missing>")
    )


def component_diff(source_components: list[dict], target_components: list[dict]) -> list[str]:
    changes: list[str] = []
    source_by_name = {str(item["name"]): item for item in source_components}
    for item in target_components:
        name = str(item["name"])
        source = source_by_name.get(name, {})
        for key in COMPONENT_PATHS:
            if source.get(key) != item.get(key):
                changes.append(f"components[{name}].{key}")
    return changes


def derive(source: dict[str, Any], cell: tuple) -> dict[str, Any]:
    slug, _source_name, _target, modality, backend, language = cell
    config = copy.deepcopy(source)
    if modality == "text_only":
        config["model_backend"] = "qwen38"
        config["model_name_or_path"] = QWEN38_MODEL_PATH
        config["model_revision"] = QWEN38_MODEL_REVISION
        config.pop("model_attn_implementation", None)
    else:
        config["model_backend"] = "qwen3omni"
        config["model_name_or_path"] = QWEN3OMNI_MODEL_PATH
        config["model_attn_implementation"] = QWEN3OMNI_ATTN_IMPLEMENTATION
        config.pop("model_revision", None)

    config["recipe_id"] = NATIVE_RECIPE if language == "native" else ENGLISH_RECIPE
    config["components"] = (
        english_components(modality) if language == "english" else native_components(modality)
    )
    campaign = "qwen3_pooled_native" if language == "native" else "qwen3_pooled_english"
    config["output_dirs"]["merged_root"] = (
        f"${{PROJECT_ROOT}}/outputs/symmetric_merged/{campaign}/{modality}"
    )
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/symmetric_merged/{campaign}_likelihood/{modality}"
    )
    # The merged contract carries its FSDP recipe explicitly, and the audio
    # routes declare the sharded evaluation shape their components require (the
    # 30B Thinker does not fit one H100 in bf16).
    config.setdefault("training", {})["strategy"] = "fsdp"
    config["training"]["activation_offload"] = "cpu"
    if modality != "text_only":
        config.setdefault("execution", {})["postprocess_gpus"] = 4
        # Every Qwen3-Omni route runs the two-node lane: two four-GPU nodes with
        # per-rank accumulation 16, the shape the pooled Qwen3-Omni baselines
        # were submitted with. The effective global batch stays 128.
        config["execution"]["train_nodes"] = 2
        config["training"]["gradient_accumulation_steps"] = 16
    config["status"], config["status_reason"] = CELL_STATUS[slug]
    if language == "english":
        config["name"] = f"symmetric_merged_qwen3_pooled_english_{modality}"
    else:
        config["name"] = f"symmetric_merged_qwen3_pooled_native_{modality}"

    changed = diff_paths(source, config) + component_diff(source["components"], config["components"])
    disallowed = [
        path
        for path in changed
        if path not in ALLOWED_DIFF_PATHS and not path.startswith("components[")
    ]
    if disallowed:
        raise GenerationError(f"{slug}: diff outside the allowlist: {disallowed}")

    # Contract checks: five components, no mixed backend family, pooled Turkish,
    # English overlay on the translated datasets for the English contract.
    if len(config["components"]) != 5:
        raise GenerationError(f"{slug}: merged contracts need exactly five components")
    for component in config["components"]:
        path = PROJECT_ROOT / component["config"]
        if not path.is_file():
            raise GenerationError(f"{slug}: missing component config {component['config']}")
        component_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        component_backend = str(component_config.get("model_backend") or "")
        if component_backend != backend:
            raise GenerationError(
                f"{slug}: component {component['name']} resolves to {component_backend!r}, "
                f"expected {backend!r} (mixed merged backends are not allowed)"
            )
        if component["name"] == "turkish":
            if str(component_config.get("dataset_variant")) != "pooled_t17":
                raise GenerationError(f"{slug}: the Turkish component must be the pooled family")
            if language == "english":
                if str((component_config.get("transcripts") or {}).get("variant")) != "english":
                    raise GenerationError(f"{slug}: the English Turkish component must carry the overlay")
        elif language == "english" and component["name"] != "daic":
            if str((component_config.get("transcripts") or {}).get("variant")) != "english":
                raise GenerationError(
                    f"{slug}: English components other than DAIC must carry the English overlay"
                )
    return reorder(config)


TOP_LEVEL_ORDER = (
    "name",
    "protocol",
    "recipe_id",
    "modality",
    "seed",
    "status",
    "status_reason",
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "model_attn_implementation",
    "components",
    "output_dirs",
    "protocol_settings",
    "training",
    "heads",
    "execution",
)


def _ordered(mapping: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    result = {key: mapping[key] for key in order if key in mapping}
    result.update({key: value for key, value in mapping.items() if key not in result})
    return result


def reorder(config: dict[str, Any]) -> dict[str, Any]:
    return _ordered(config, TOP_LEVEL_ORDER)


def render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument(
        "--replace-contracts",
        action="store_true",
        help=(
            "one-time transition: replace existing generated contracts even when they differ from "
            "the derived content (for example a readiness status flip backed by a recorded smoke "
            "chain); the replaced paths are recorded in the audit"
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
        "schema_version": "audiollm.qwen3_pooled_merged_config_diff.v1",
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "component_paths": list(COMPONENT_PATHS),
        "status": BLOCKED_STATUS,
        "configs": [],
    }
    for cell in CELLS:
        slug, source_name, target_name, modality, backend, language = cell
        source_path = MERGED / source_name
        if not source_path.is_file():
            raise GenerationError(f"missing legacy merged source: {source_path}")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config = derive(source, cell)
        rendered = render(config)
        if yaml.safe_load(rendered) != config:
            raise GenerationError(f"{slug}: rendered config does not round-trip")
        target = MERGED / target_name
        emit(
            target,
            rendered,
            check_only=args.check,
            failures=failures,
            replace=args.replace_contracts,
            replacements=replacements,
        )
        audit["contracts_replaced"] = replacements
        audit["configs"].append(
            {
                "cell_id": slug,
                "language": language,
                "modality": modality,
                "backend": backend,
                "source": f"configs/experiments/merged/{source_name}",
                "config": f"configs/experiments/merged/{target_name}",
                "changed_paths": diff_paths(source, config),
                "component_changes": component_diff(source["components"], config["components"]),
                "allowed": True,
            }
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
