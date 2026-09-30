#!/usr/bin/env python3
"""Validate the Qwen3 Turkish pooled-default selection and emit its selection map.

The map resolves ``(family, language, modality)`` to an exact config, backend,
manifest contract and readiness line, and the validators fail closed when the
default routes drift:

* the native 15-cell matrix keeps four datasets on the canonical Qwen3 configs
  and spends its three Turkish slots on the pooled Qwen3 configs (no pos-only
  default slot, no legacy English config);
* the English 8-cell matrix selects exactly the generated Qwen3 English family;
* the five Turkish standalone cells (three native pooled, two English pooled)
  keep the pooled source contract, the question context and the aggregation
  rules;
* the four merged contracts resolve each of their five components to one Qwen3
  backend family, keep the mean-dataset-macro-F1 selection contract, and stay
  explicitly blocked for GPU execution until the support task lands;
* head execution stays explicit-only: the default matrices never declare fixed
  heads, and Qwen3 head jobs run only through the dedicated smoke submitter.

``--emit`` writes the machine-readable map next to the private task outputs; the
validators themselves are the tracked contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_qwen3_english_configs as english_configs  # noqa: E402
from scripts import build_qwen3_pooled_merged_configs as merged_configs  # noqa: E402

MAIN = PROJECT_ROOT / "configs/main"
NATIVE_MATRIX = PROJECT_ROOT / "configs/experiments/harmonized/standalone_matrix.yaml"
ENGLISH_MATRIX = PROJECT_ROOT / "configs/experiments/harmonized/english_translation_matrix.yaml"
NATIVE_MATRIX_LEGACY = (
    PROJECT_ROOT / "configs/experiments/harmonized/standalone_matrix_legacy_pos_only.yaml"
)
ENGLISH_MATRIX_LEGACY = (
    PROJECT_ROOT / "configs/experiments/harmonized/english_translation_matrix_legacy_qwen2.yaml"
)

POOLED_TURKISH_NATIVE = {
    "text_only": (
        "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
        "_promptcontext_v1_qwen38_27b.yaml"
    ),
    "audio_only": (
        "configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
    "audio_text": (
        "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    ),
}
POOLED_TURKISH_ENGLISH = {
    "text_only": (
        "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
        "_promptcontext_v1_en_qwen38_27b.yaml"
    ),
    "audio_text": (
        "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
        "_promptcontext_v1_en_qwen3omni_30b_a3b.yaml"
    ),
}
MERGED_CONTRACTS = tuple(f"configs/experiments/merged/{cell[2]}" for cell in merged_configs.CELLS)
NEW_MODEL_BACKENDS = frozenset({"qwen38", "qwen3omni"})
EXPECTED_BACKEND_BY_MODALITY = {"text_only": "qwen38", "audio_only": "qwen3omni", "audio_text": "qwen3omni"}
POOLED_TEXT_AGGREGATION = "turkish_pooled_text_pair_mean_margin_strict_v1"

READINESS = (
    ("turkish/native/text_only", "ready: existing REPORTABLE pooled Qwen3 runs"),
    ("turkish/native/audio_only", "ready: existing REPORTABLE pooled Qwen3 runs"),
    ("turkish/native/audio_text", "ready: existing REPORTABLE pooled Qwen3 runs"),
    ("turkish/english/text_only", "data and config ready; Qwen3.8 rendering audit prepared"),
    ("turkish/english/audio_text", "data and config ready; Qwen3-Omni processor audit prepared"),
    (
        "merged/native",
        "implementation landed; native text_only and audio_text run their bounded GPU smoke chains in this task, "
        "native audio_only waits for its own chain, and the merged head kind stays deferred until Qwen3 "
        "hidden-feature support is verified for merged checkpoints",
    ),
    (
        "merged/english",
        "implementation landed; the English text and English audio+text contracts wait for "
        "their own GPU smoke chains, and the merged head kind stays deferred until Qwen3 "
        "hidden-feature support is verified for merged checkpoints",
    ),
    (
        "merged/head",
        "deferred for every Qwen3 merged route until Qwen3 hidden-feature support is verified "
        "for merged checkpoints",
    ),
    ("heads", "explicit-only: Qwen3 hidden extraction landed; Qwen3 head jobs run through the dedicated smoke submitter"),
)


class SelectionError(RuntimeError):
    """Raised when the selection cannot be resolved at all."""


def load_config(rel: str) -> dict[str, Any]:
    path = PROJECT_ROOT / rel
    if not path.is_file():
        raise SelectionError(f"missing config: {rel}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_matrix(rel: Path) -> dict[str, Any]:
    if not rel.is_file():
        raise SelectionError(f"missing matrix: {rel}")
    return yaml.safe_load(rel.read_text(encoding="utf-8"))


def resolve_selection() -> dict[str, Any]:
    """The explicit (family, language, modality) -> contract map."""
    cells = []
    for language, table in (("native", POOLED_TURKISH_NATIVE), ("english", POOLED_TURKISH_ENGLISH)):
        for modality, rel in sorted(table.items()):
            config = load_config(rel)
            cells.append(
                {
                    "family": "turkish_pooled",
                    "language": language,
                    "modality": modality,
                    "config": rel,
                    "backend": config.get("model_backend"),
                    "manifest_contract": config.get("manifest_policy", "build"),
                    "recipe_id": config.get("recipe_id"),
                }
            )
    for dataset, stem in (
        ("daic", "daic"),
        ("d3tec", "d3tec"),
        ("androids_interview", "androids"),
        ("cmdc", "cmdc"),
        ("turkish", "turkish"),
    ):
        for modality in ("audio_only", "audio_text", "text_only"):
            rel = f"configs/main/{stem}_{modality}_harmonized_selmacrof1_likelihood_v1.yaml"
            if not (PROJECT_ROOT / rel).is_file():
                continue
            config = load_config(rel)
            cells.append(
                {
                    "family": "canonical",
                    "language": "native",
                    "modality": modality,
                    "dataset": dataset,
                    "config": rel,
                    "backend": config.get("model_backend"),
                    "manifest_contract": config.get("manifest_policy", "build"),
                    "recipe_id": config.get("recipe_id"),
                }
            )
        for cell in english_configs.CELLS:
            # The Turkish English cells are listed by the pooled table above.
            if stem == "turkish" or not cell[2].startswith(f"{stem}_"):
                continue
            rel = f"configs/main/{cell[2]}"
            config = load_config(rel)
            cells.append(
                {
                    "family": "english_qwen3" if dataset != "turkish" else "turkish_pooled",
                    "language": "english",
                    "modality": cell[5],
                    "dataset": dataset,
                    "config": rel,
                    "backend": config.get("model_backend"),
                    "manifest_contract": config.get("manifest_policy", "build"),
                    "recipe_id": config.get("recipe_id"),
                }
            )
    return {"schema_version": "audiollm.qwen3_pooled_defaults.selection.v1", "cells": cells, "merged": list(MERGED_CONTRACTS)}


def validate_native_matrix() -> list[str]:
    failures: list[str] = []
    matrix = load_matrix(NATIVE_MATRIX)
    if len(matrix["experiments"]) != 15:
        failures.append(f"native matrix must keep 15 cells, found {len(matrix['experiments'])}")
    selected = [str(item["config"]) for item in matrix["experiments"]]
    for modality, rel in POOLED_TURKISH_NATIVE.items():
        if rel not in selected:
            failures.append(f"native matrix must select the pooled Turkish cell {rel}")
    if len(selected) != len(set(selected)):
        failures.append("native matrix selects a config twice")
    if matrix.get("fixed_heads") != []:
        failures.append("native matrix must not declare fixed heads (Qwen3 head execution is explicit-only)")
    for item in matrix["experiments"]:
        rel = str(item["config"])
        config = load_config(rel)
        backend = str(config.get("model_backend") or "")
        if backend not in NEW_MODEL_BACKENDS:
            failures.append(f"native default cell is not a Qwen3 backend: {rel}")
        dataset = str(config["dataset"])
        if dataset == "turkish":
            if str(config.get("dataset_variant")) != "pooled_t17":
                failures.append(f"Turkish default slot is not pooled: {rel}")
            if config.get("manifest_policy") != "prebuilt":
                failures.append(f"pooled Turkish default must keep the prebuilt manifest policy: {rel}")
        elif "pos_only" in rel or rel.endswith("_en.yaml"):
            failures.append(f"legacy Turkish/English config leaked into the native default: {rel}")
    return failures


def validate_english_matrix() -> list[str]:
    failures: list[str] = []
    matrix = load_matrix(ENGLISH_MATRIX)
    if len(matrix["experiments"]) != 8:
        failures.append(f"English matrix must keep 8 cells, found {len(matrix['experiments'])}")
    if matrix.get("fixed_heads") != []:
        failures.append("English default matrix must not declare fixed heads")
    expected = {
        f"configs/main/{cell[2]}"
        for cell in english_configs.CELLS
    }
    selected = {str(item["config"]) for item in matrix["experiments"]}
    if selected != expected:
        failures.append(
            "English matrix selection differs from the generated Qwen3 English family: "
            f"missing={sorted(expected - selected)} extra={sorted(selected - expected)}"
        )
    for rel in sorted(selected):
        config = load_config(rel)
        backend = str(config.get("model_backend") or "")
        if backend not in NEW_MODEL_BACKENDS:
            failures.append(f"English default cell is not a Qwen3 backend: {rel}")
        transcripts = config.get("transcripts") or {}
        if transcripts.get("variant") != "english":
            failures.append(f"English default cell has no English overlay: {rel}")
        if not str(config.get("recipe_id", "")).endswith("_en"):
            failures.append(f"English default cell recipe must carry the _en marker: {rel}")
    train_folds = sum(len(item["folds"]) for item in matrix["experiments"])
    eval_folds = sum(len(item["folds"]) for item in matrix["experiments"] if item.get("separate_eval"))
    if (train_folds, eval_folds) != (40, 20):
        failures.append(f"English matrix fold scope changed: train={train_folds} eval={eval_folds}")
    return failures


def validate_turkish_cells() -> list[str]:
    failures: list[str] = []
    for language, table in (("native", POOLED_TURKISH_NATIVE), ("english", POOLED_TURKISH_ENGLISH)):
        for modality, rel in sorted(table.items()):
            config = load_config(rel)
            expected_backend = EXPECTED_BACKEND_BY_MODALITY[modality]
            if config.get("model_backend") != expected_backend:
                failures.append(f"{rel}: backend must be {expected_backend}")
            if str(config.get("dataset")) != "turkish" or str(config.get("dataset_variant")) != "pooled_t17":
                failures.append(f"{rel}: must be the pooled Turkish family")
            prompt = config.get("prompt") or {}
            if prompt.get("dataset_context") != "turkish_pooled":
                failures.append(f"{rel}: prompt context must be turkish_pooled")
            if "{question_context}" not in str(prompt.get("user_template", "")):
                failures.append(f"{rel}: prompt must carry the question context")
            evaluation = config.get("evaluation") or {}
            if modality == "text_only":
                if evaluation.get("subject_score_aggregation") != POOLED_TEXT_AGGREGATION:
                    failures.append(f"{rel}: text-only pooled aggregation rule changed")
            else:
                if evaluation.get("aggregation_level") != "response_subject":
                    failures.append(f"{rel}: audio pooled aggregation level changed")
                if evaluation.get("hierarchical_score_aggregation") != "mean":
                    failures.append(f"{rel}: audio pooled hierarchical aggregation changed")
    # The English cells must inherit their scientific fields from the native cell.
    for modality, en_rel in POOLED_TURKISH_ENGLISH.items():
        native = load_config(POOLED_TURKISH_NATIVE[modality])
        english = load_config(en_rel)
        for key in ("split", "data", "labels", "lora", "training", "evaluation", "audio_adapter"):
            if english.get(key) != native.get(key):
                failures.append(f"{en_rel}: {key} must be inherited unchanged from the native cell")
        if english.get("manifest_policy") != "prebuilt":
            failures.append(f"{en_rel}: pooled English cell must stay on the prebuilt manifest policy")
        cache_path = str((english.get("transcripts") or {}).get("cache_path", ""))
        if cache_path != english_configs.POOLED_TRANSCRIPTS_CACHE_PLACEHOLDER:
            failures.append(f"{en_rel}: the pooled English cache placeholder must stay inert")
    return failures


def validate_merged_contracts() -> list[str]:
    failures: list[str] = []
    expected_status = {
        f"configs/experiments/merged/{cell[2]}": merged_configs.CELL_STATUS[cell[0]]
        for cell in merged_configs.CELLS
    }
    for rel in MERGED_CONTRACTS:
        config = load_config(rel)
        status, _reason = expected_status[rel]
        if config.get("status") != status:
            failures.append(
                f"{rel}: merged contract status must stay {status!r} until its own GPU smoke chain passes"
            )
        if not str(config.get("status_reason") or "").strip():
            failures.append(f"{rel}: merged contract needs a status_reason")
        training = config.get("training") or {}
        if str(training.get("strategy") or "") != "fsdp":
            failures.append(f"{rel}: merged contract must declare training.strategy=fsdp")
        if str(training.get("activation_offload") or "") != "cpu":
            failures.append(f"{rel}: merged contract must declare training.activation_offload=cpu")
        modality = str(config.get("modality") or "")
        expected_gpus = 1 if modality == "text_only" else 4
        if int((config.get("execution") or {}).get("postprocess_gpus", 1)) != expected_gpus:
            failures.append(
                f"{rel}: execution.postprocess_gpus must be {expected_gpus} for modality {modality!r}"
            )
        settings = config.get("protocol_settings") or {}
        if settings.get("selection_metric") != "mean_dataset_macro_f1":
            failures.append(f"{rel}: merged selection metric must stay mean_dataset_macro_f1")
        components = config.get("components") or []
        if len(components) != 5:
            failures.append(f"{rel}: merged contract needs exactly five components")
        backend = str(config.get("model_backend") or "")
        if backend not in NEW_MODEL_BACKENDS:
            failures.append(f"{rel}: merged contract must declare its Qwen3 backend explicitly")
        for component in components:
            component_config = load_config(str(component["config"]))
            component_backend = str(component_config.get("model_backend") or "")
            if component_backend != backend:
                failures.append(
                    f"{rel}: component {component['name']} resolves to {component_backend!r}, "
                    f"expected {backend!r} (mixed merged backends are not allowed)"
                )
            if component["name"] == "turkish" and str(component_config.get("dataset_variant")) != "pooled_t17":
                failures.append(f"{rel}: the Turkish component must be the pooled family")
        if "english" in rel:
            for component in components:
                if component["name"] in ("daic", "turkish"):
                    continue
                component_config = load_config(str(component["config"]))
                if (component_config.get("transcripts") or {}).get("variant") != "english":
                    failures.append(f"{rel}: component {component['name']} needs the English overlay")
    return failures


def validate_head_scope() -> list[str]:
    failures: list[str] = []
    for rel in list(POOLED_TURKISH_NATIVE.values()) + list(POOLED_TURKISH_ENGLISH.values()):
        config = load_config(rel)
        if str(config.get("model_backend") or "") in NEW_MODEL_BACKENDS:
            continue
        failures.append(f"{rel}: expected a Qwen3 backbone for the pooled default cell")
    return failures


def validate_all() -> dict[str, list[str]]:
    return {
        "native_matrix": validate_native_matrix(),
        "english_matrix": validate_english_matrix(),
        "turkish_cells": validate_turkish_cells(),
        "merged_contracts": validate_merged_contracts(),
        "head_scope": validate_head_scope(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate and print a summary")
    parser.add_argument("--emit", type=Path, help="write the selection map JSON here")
    args = parser.parse_args(argv)

    failures = validate_all()
    total = sum(len(items) for items in failures.values())
    for section, items in failures.items():
        for item in items:
            print(f"ERROR {section}: {item}", file=sys.stderr)
    if args.emit is not None:
        payload = resolve_selection()
        payload["readiness"] = [{"contract": name, "state": state} for name, state in READINESS]
        payload["failures"] = {section: items for section, items in failures.items()}
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.emit}")
    if total:
        print(f"Qwen3 pooled default selection has {total} failure(s).", file=sys.stderr)
        return 1
    print("Qwen3 pooled default selection is consistent (native 15, English 8, five merged contracts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
