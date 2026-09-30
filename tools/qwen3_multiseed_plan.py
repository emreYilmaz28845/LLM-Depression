#!/usr/bin/env python3
"""Build the machine-readable Qwen3 multiseed route selection map.

The readiness task needs one explicit record per route: which config runs, with
which backend, pinned model revision, prompt contract, manifest and split
contract, evaluation contract, evaluation shape, fold set and output root. The
map is built from the tracked sources of truth (the two default matrices, the
generators' cell tables, the canonical configs and the merged contracts), never
from run names.

``--check`` fails closed when the route set drifts from the declared shape:
every native and English matrix cell must resolve to a Qwen3 backend with the
expected recipe, every merged contract must exist with five components resolved
to one backend family, and the merged submission guard's readiness table must
agree with the contracts' documentation ``status`` field.

The production job manifest (three seeds, all folds, merged smoke/cv/final job
graphs) is a separate phase and is emitted from the same route records.
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

from src.data.prompt_context import prompt_context_record  # noqa: E402
from src.experiment_tracking.canonical import sha256_file  # noqa: E402
from src.experiment_tracking.submit import resolve_evaluation_shape  # noqa: E402
from src.utils import load_yaml_with_overrides, resolve_input_modality, resolve_model_backend  # noqa: E402
from scripts import build_qwen3_pooled_merged_configs as merged_configs  # noqa: E402
from tools import qwen3_pooled_defaults as pooled_defaults  # noqa: E402

SCHEMA_VERSION = "audiollm.qwen3_multiseed_selection_map.v1"
PLANNED_SEEDS = (7, 1337, 2024)
SPLIT_SEED = 1337


class PlanError(RuntimeError):
    """Raised when the route map cannot be built or fails its checks."""


def _config_record(config: dict[str, Any], raw_config: dict[str, Any]) -> dict[str, Any]:
    """Record one route.

    ``config`` is the env-expanded config used for validation; ``raw_config`` is
    the file as authored, so template paths (``${PROJECT_ROOT}``, model path
    env defaults) are recorded as templates instead of one checkout's absolute
    path.
    """
    prompt = config.get("prompt") or {}
    evaluation = config.get("evaluation") or {}
    split = config.get("split") or {}
    raw_output_dirs = raw_config.get("output_dirs") or {}
    raw_prompt = raw_config.get("prompt") or {}
    prompt_record = None
    try:
        prompt_record = prompt_context_record(config)
    except (KeyError, TypeError, ValueError):
        prompt_record = None
    shape = resolve_evaluation_shape(config)
    return {
        "model_backend": resolve_model_backend(config),
        "model_name_or_path": raw_config.get("model_name_or_path"),
        "model_name_or_path_resolved": config.get("model_name_or_path"),
        "model_revision": config.get("model_revision"),
        "model_attn_implementation": config.get("model_attn_implementation"),
        "recipe_id": config.get("recipe_id"),
        "dataset": config.get("dataset"),
        "dataset_variant": config.get("dataset_variant"),
        "modality": resolve_input_modality(config),
        "prompt": {
            "version": prompt.get("version"),
            "dataset_context": prompt.get("dataset_context"),
            "question_context_version": prompt.get("question_context_version"),
            "prompt_language": prompt.get("prompt_language"),
            "user_template": raw_prompt.get("user_template"),
            "system_prompt_sha256": (prompt_record or {}).get("system_prompt_sha256"),
        },
        "labels": config.get("labels"),
        "manifest": {
            "policy": config.get("manifest_policy", "build"),
            "dir": raw_output_dirs.get("manifest_dir"),
            "split_dir": raw_output_dirs.get("split_dir"),
            "transcripts_variant": (config.get("transcripts") or {}).get("variant"),
        },
        "split": {
            "mode": split.get("mode"),
            "cv_protocol": split.get("cv_protocol"),
            "outer_folds": split.get("outer_folds"),
            "seed": split.get("seed"),
            "inner_val_ratio": split.get("inner_val_ratio"),
            "final_eval_partition": split.get("final_eval_partition"),
        },
        "evaluation": {
            "sample_prediction_mode": evaluation.get("sample_prediction_mode"),
            "evaluation_view": evaluation.get("evaluation_view"),
            "aggregation_level": evaluation.get("aggregation_level"),
            "subject_score_aggregation": evaluation.get("subject_score_aggregation"),
            "hierarchical_score_aggregation": evaluation.get("hierarchical_score_aggregation"),
            "selection_metric": (config.get("training") or {}).get("selection_metric"),
            "selection_metric_mode": (config.get("training") or {}).get("selection_metric_mode"),
        },
        "resources": {
            "eval_nodes": shape.get("nodes"),
            "eval_gpus_per_node": shape.get("gpus_per_node"),
            "sharded": shape.get("sharded"),
            "declared": bool(config.get("resources")),
        },
        "training": {
            "strategy": (config.get("training") or {}).get("strategy"),
            "activation_offload": (config.get("training") or {}).get("activation_offload"),
            "per_device_train_batch_size": (config.get("training") or {}).get("per_device_train_batch_size"),
            "gradient_accumulation_steps": (config.get("training") or {}).get("gradient_accumulation_steps"),
            "num_train_epochs": (config.get("training") or {}).get("num_train_epochs"),
        },
        "run_root": raw_output_dirs.get("run_root"),
    }


def _matrix_folds(matrix_rel: str) -> dict[str, list[int]]:
    matrix = yaml.safe_load((PROJECT_ROOT / matrix_rel).read_text(encoding="utf-8")) or {}
    result: dict[str, list[int]] = {}
    for entry in matrix.get("experiments") or []:
        result[str(entry["config"])] = [int(fold) for fold in entry.get("folds") or []]
    return result


def build_selection_map() -> dict[str, Any]:
    selection = pooled_defaults.resolve_selection()
    native_folds = _matrix_folds("configs/experiments/harmonized/standalone_matrix.yaml")
    english_folds = _matrix_folds("configs/experiments/harmonized/english_translation_matrix.yaml")
    folds_by_config = {**native_folds, **english_folds}

    routes: list[dict[str, Any]] = []
    for cell in selection["cells"]:
        rel = str(cell["config"])
        raw_config = yaml.safe_load((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        config = load_yaml_with_overrides(PROJECT_ROOT / rel, [])
        record = _config_record(config, raw_config)
        dataset = cell.get("dataset") or ("turkish" if cell["family"].startswith("turkish_pooled") else "unknown")
        routes.append(
            {
                "route_id": f"{dataset}_{record['modality']}_{cell['language']}",
                "family": cell["family"],
                "language": cell["language"],
                "config": rel,
                "config_sha256": sha256_file(PROJECT_ROOT / rel),
                "folds": folds_by_config.get(rel, [0]),
                **record,
            }
        )

    merged_routes: list[dict[str, Any]] = []
    readiness = {}
    from scripts import submit_symmetric_merged as merged_submit  # noqa: E402

    readiness = merged_submit.QWEN3_CONTRACT_READINESS
    for cell in merged_configs.CELLS:
        slug, _source, target, modality, backend, language = cell
        rel = f"configs/experiments/merged/{target}"
        config = load_yaml_with_overrides(PROJECT_ROOT / rel, [])
        raw_config = yaml.safe_load((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        components = [
            {
                "name": item["name"],
                "config": item["config"],
                "transcripts_variant": (
                    yaml.safe_load((PROJECT_ROOT / item["config"]).read_text(encoding="utf-8")).get(
                        "transcripts"
                    )
                    or {}
                ).get("variant"),
            }
            for item in config.get("components") or []
        ]
        contract_name = str(config.get("name"))
        entry = readiness.get(contract_name) or {}
        merged_routes.append(
            {
                "route_id": f"merged_{language}_{modality}",
                "family": "merged",
                "language": language,
                "config": rel,
                "config_sha256": sha256_file(PROJECT_ROOT / rel),
                "stages": {"smoke": [0], "cv": [0, 1, 2, 3, 4], "final": [0]},
                "backend": backend,
                "modality": modality,
                "status": config.get("status"),
                "status_reason": config.get("status_reason"),
                "guard": {
                    "production_ready": entry.get("production_ready"),
                    "head_ready": entry.get("head_ready"),
                    "declared": bool(entry),
                },
                "components": components,
                "protocol_settings": config.get("protocol_settings"),
                "training": {
                    "strategy": (config.get("training") or {}).get("strategy"),
                    "activation_offload": (config.get("training") or {}).get("activation_offload"),
                },
                "execution": config.get("execution"),
                "run_roots": raw_config.get("output_dirs"),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "selection_map": "tools/qwen3_pooled_defaults.py::resolve_selection",
            "merged_builder": "scripts/build_qwen3_pooled_merged_configs.py::CELLS",
            "guard": "scripts/submit_symmetric_merged.py::QWEN3_CONTRACT_READINESS",
        },
        "planned_seeds": list(PLANNED_SEEDS),
        "split_seed": SPLIT_SEED,
        "routes": routes,
        "merged_routes": merged_routes,
    }


def check_selection_map(selection_map: dict[str, Any], *, require_english_audio_text: bool) -> list[str]:
    failures: list[str] = []
    routes = selection_map["routes"]
    cell_ids = {route["route_id"] for route in routes}
    if len(routes) != 23:
        failures.append(f"route map must carry the 15 native + 8 English cells, found {len(routes)}")
    for route in routes:
        if route["model_backend"] not in {"qwen38", "qwen3omni"}:
            failures.append(f"{route['config']}: not a Qwen3 backend")
        if route["evaluation"]["sample_prediction_mode"] != "likelihood":
            failures.append(f"{route['config']}: decision rule must stay likelihood")
        if route["evaluation"]["evaluation_view"] != "harmonized_all_windows_full_coverage":
            failures.append(f"{route['config']}: evaluation view changed")
        if route["split"]["seed"] != SPLIT_SEED:
            failures.append(f"{route['config']}: split seed must stay {SPLIT_SEED}")
        if not route["folds"]:
            failures.append(f"{route['config']}: no folds declared")
        expected_modality = {"qwen38": "text_only"}.get(route["model_backend"])
        if expected_modality and route["modality"] != expected_modality:
            failures.append(f"{route['config']}: {route['model_backend']} must be text_only")
        expected_eval_gpus = 1 if route["modality"] == "text_only" else 4
        if int(route["resources"]["eval_gpus_per_node"]) != expected_eval_gpus:
            failures.append(
                f"{route['config']}: evaluation shape must be {expected_eval_gpus} GPU(s) "
                f"for {route['modality']}"
            )
        if route["language"] == "english" and route["manifest"]["transcripts_variant"] != "english":
            failures.append(f"{route['config']}: English route without the English overlay")
    native = [route for route in routes if route["language"] == "native"]
    english = [route for route in routes if route["language"] == "english"]
    if len(native) != 15:
        failures.append(f"native route count must be 15, found {len(native)}")
    if len(english) != 8:
        failures.append(f"English route count must be 8, found {len(english)}")

    merged = selection_map["merged_routes"]
    merged_ids = {route["route_id"] for route in merged}
    expected_merged = {
        "merged_native_text_only",
        "merged_native_audio_only",
        "merged_native_audio_text",
        "merged_english_text_only",
    }
    if require_english_audio_text:
        expected_merged.add("merged_english_audio_text")
    missing = expected_merged - merged_ids
    if missing:
        failures.append(f"merged contracts missing: {sorted(missing)}")
    for route in merged:
        if len(route["components"]) != 5:
            failures.append(f"{route['config']}: merged contracts need five components")
        if not route["guard"]["declared"]:
            failures.append(f"{route['config']}: merged contract absent from the submission guard")
        documentation_status = route["status"]
        guard_ready = route["guard"]["production_ready"]
        if guard_ready and documentation_status != "execute_verified":
            failures.append(f"{route['config']}: guard and documentation status disagree")
        if route["training"]["strategy"] != "fsdp":
            failures.append(f"{route['config']}: merged contract must declare FSDP")
        expected_gpus = 1 if route["modality"] == "text_only" else 4
        postprocess_gpus = int((route["execution"] or {}).get("postprocess_gpus", 1))
        if postprocess_gpus != expected_gpus:
            failures.append(
                f"{route['config']}: postprocess_gpus must be {expected_gpus} for {route['modality']}"
            )
        if route["language"] == "english":
            for component in route["components"]:
                if component["name"] in ("daic", "turkish"):
                    continue
                if component["transcripts_variant"] != "english":
                    failures.append(f"{route['config']}: component {component['name']} lacks the English overlay")
        for expected_component in ("daic", "cmdc", "turkish", "d3tec", "androids_interview"):
            if expected_component not in [item["name"] for item in route["components"]]:
                failures.append(f"{route['config']}: missing component {expected_component}")
    if cell_ids & merged_ids:  # pragma: no cover - defensive
        failures.append("route ids collide between standalone and merged maps")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-selection-map", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--require-english-audio-text-contract",
        action="store_true",
        help="require the English audio+text merged contract (added in the merged-route phase)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selection_map = build_selection_map()
    except (PlanError, KeyError, TypeError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    failures = check_selection_map(
        selection_map, require_english_audio_text=args.require_english_audio_text_contract
    )
    if args.emit_selection_map is not None:
        args.emit_selection_map.parent.mkdir(parents=True, exist_ok=True)
        args.emit_selection_map.write_text(
            json.dumps(selection_map, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.emit_selection_map}")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    if args.check or failures:
        print(
            f"selection map: {len(selection_map['routes'])} standalone routes, "
            f"{len(selection_map['merged_routes'])} merged contracts, {len(failures)} failure(s)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
