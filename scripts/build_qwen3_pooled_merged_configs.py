#!/usr/bin/env python3
"""Generate the four Qwen3 pooled merged contracts.

The contracts are derived from the existing symmetric-merged configs, so the
merged methodology is inherited by construction and the diff audit proves it:
protocol settings, selection metric (``mean_dataset_macro_f1``), folds, heads
config, training shape and execution block stay byte-identical. Only the
documented contract fields change:

* explicit ``model_backend`` identity (Qwen3.8-27B for text-only, Qwen3-Omni
  30B-A3B Thinker for the audio modalities) with the pinned model path and, for
  Qwen3.8, the pinned revision;
* the five components point at the current native default cells; the Turkish
  component is the pooled Qwen3 config, and the English text-only contract swaps
  the four translated datasets to their Qwen3 English cells while DAIC keeps its
  native English input;
* the pooled English component keeps the prebuilt manifest contract;
* isolated merged roots under ``symmetric_merged/qwen3_pooled_{native,english}``;
* an explicit ``status: blocked_prerequisite`` line, because Qwen3 merged
  execution (FSDP training and postprocess) is not implemented yet. The config
  field is documentation; the submission route blocks by component backend and
  ignores nothing here.

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
BLOCKED_REASON = (
    "Qwen3 merged execution is not implemented yet: the merged trainer is still the DDP "
    "path and the hidden/postprocess steps do not support Qwen3.8/Qwen3-Omni shards or "
    "processors. The submission route refuses GPU jobs for this contract until the "
    "separate Qwen3 merged FSDP/postprocess support task lands."
)

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


def english_text_only_components() -> list[dict[str, str]]:
    return [
        {
            "name": "daic",
            "config": "configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
            "manifest_path": "outputs/manifests_harmonized/daic/daic_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized/daic/daic_manifest_metadata.json",
        },
        {
            "name": "cmdc",
            "config": "configs/main/cmdc_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
            "manifest_path": "outputs/manifests_harmonized_en/cmdc/cmdc_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized_en/cmdc/cmdc_manifest_metadata.json",
        },
        {
            "name": "turkish",
            "config": POOLED_COMPONENT_EN["text_only"],
            "manifest_path": POOLED_MANIFEST["english"][0],
            "metadata_path": POOLED_MANIFEST["english"][1],
        },
        {
            "name": "d3tec",
            "config": "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
            "manifest_path": "outputs/manifests_harmonized_en/d3tec/d3tec_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized_en/d3tec/d3tec_manifest_metadata.json",
        },
        {
            "name": "androids_interview",
            "config": "configs/main/androids_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
            "manifest_path": "outputs/manifests_harmonized_en/androids/androids_interview_manifest.jsonl",
            "metadata_path": "outputs/splits_harmonized_en/androids/androids_interview_manifest_metadata.json",
        },
    ]


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
        english_text_only_components() if language == "english" else native_components(modality)
    )
    campaign = "qwen3_pooled_native" if language == "native" else "qwen3_pooled_english"
    config["output_dirs"]["merged_root"] = (
        f"${{PROJECT_ROOT}}/outputs/symmetric_merged/{campaign}/{modality}"
    )
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/symmetric_merged/{campaign}_likelihood/{modality}"
    )
    config["status"] = BLOCKED_STATUS
    config["status_reason"] = BLOCKED_REASON
    if language == "english":
        config["name"] = "symmetric_merged_qwen3_pooled_english_text_only"
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


def emit(target: Path, content: str, *, check_only: bool, failures: list[str]) -> None:
    if target.is_file():
        if target.read_text(encoding="utf-8") != content:
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
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help="structured diff audit path (default: outputs/.../config_diff_audit.json)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
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
        emit(target, rendered, check_only=args.check, failures=failures)
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
