"""Tests for the Qwen3 multiseed readiness tools.

``tools/qwen3_multiseed_inventory.py`` classifies existing seed-1337 runs, and
``tools/qwen3_multiseed_plan.py`` builds the route selection map. Both decide
from recorded provenance, so the tests build synthetic run directories and
perturb exactly one property at a time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.data.prompt_context import resolve_system_prompt
from src.utils import load_yaml_with_overrides
from tools import qwen3_multiseed_inventory as inventory
from tools import qwen3_multiseed_plan as planning

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = "configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1.yaml"


def _reference_config() -> dict:
    return load_yaml_with_overrides(REPO_ROOT / REFERENCE_CONFIG, [])


def _reduced_run_config(perturb: dict | None = None) -> dict:
    config = load_yaml_with_overrides(REPO_ROOT / REFERENCE_CONFIG, [])
    for dotted, value in (perturb or {}).items():
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return config


def _write_run(tmp_path: Path, *, perturb: dict | None = None, state: str = "REPORTABLE",
               with_evidence: bool = True, metrics_value: float = 1.0, run_name: str = "run_one") -> Path:
    config = _reduced_run_config(perturb)
    fold_dir = tmp_path / "scan" / "campaign" / "text_only" / "daic" / run_name / "fold_0"
    fold_dir.mkdir(parents=True)
    payload = {
        "config": config,
        "prompt_context": {
            "version": "promptcontext_v1",
            "dataset_context": "daic",
            "question_context_version": "legacy_v1",
            "input_modality": "text_only",
            "system_prompt_sha256": hashlib.sha256(
                resolve_system_prompt(_reference_config()).encode("utf-8")
            ).hexdigest(),
        },
        "input_modality": "text_only",
        "manifest_hash": "a" * 64,
        "split_metadata_hash": "b" * 64,
        "manifest_path": "/remote/manifest.jsonl",
        "tracking": {"attempt_id": "attempt-1"},
        "config_overrides": [],
        "training_strategy": {"world_size": 4, "effective_global_batch_size": 128},
    }
    (fold_dir / "run_config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    (fold_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (fold_dir / "jobs.jsonl").write_text(
        json.dumps({"event_type": "COMPLETED", "slurm_job_id": "1", "exit_code": "0:0"}) + "\n",
        encoding="utf-8",
    )
    if with_evidence:
        checkpoint = fold_dir / "best_model"
        (checkpoint / "standalone_eval").mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        (checkpoint / "standalone_eval" / "predictions_subject_level.csv").write_text(
            "subject_id,label,prediction_text\n"
            "1,1,Depressed\n"
            "2,0,Non-depressed\n",
            encoding="utf-8",
        )
        (checkpoint / "standalone_eval" / "metrics_likelihood.json").write_text(
            json.dumps(
                {
                    "binary_strict_macro_f1": metrics_value,
                    "binary_strict_positive_f1": metrics_value,
                    "binary_strict_uar": metrics_value,
                }
            ),
            encoding="utf-8",
        )
    return fold_dir


def _write_matrix(tmp_path: Path) -> Path:
    matrix = tmp_path / "configs/experiments/harmonized/standalone_matrix.yaml"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {"config": REFERENCE_CONFIG, "folds": [0], "separate_eval": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    return matrix


@pytest.fixture()
def synthetic_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs/main").mkdir(parents=True, exist_ok=True)
    (tmp_path / REFERENCE_CONFIG).write_bytes((REPO_ROOT / REFERENCE_CONFIG).read_bytes())
    return tmp_path


def _run_inventory(tmp_path: Path, matrix: Path) -> dict:
    args = inventory.parse_args(
        [
            "--matrix",
            str(matrix),
            "--scan-root",
            str(tmp_path / "scan"),
            "--output",
            str(tmp_path / "inventory.json"),
        ]
    )
    return inventory.build_inventory(args)


def test_reduce_config_drops_paths_at_every_depth() -> None:
    config = _reference_config()
    reduced = inventory.reduce_config(config)
    flat = inventory.flatten(reduced)
    assert not [path for path in flat if path.endswith(("_path", "_dir", "_root", "_file", "_csv"))]
    assert "output_dirs" not in reduced
    assert reduced["recipe_id"] == config["recipe_id"]
    assert reduced["split"]["seed"] == config["split"]["seed"]


def test_shape_paths_are_separated_from_science() -> None:
    assert inventory.is_shape_path("training.gradient_accumulation_steps")
    assert inventory.is_shape_path("resources.eval_gpus_per_node")
    assert not inventory.is_shape_path("training.learning_rate")
    assert not inventory.is_shape_path("evaluation.evaluation_view")


def test_inventory_marks_a_matching_run_reusable(synthetic_project: Path) -> None:
    _write_run(synthetic_project)
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    cell = inventory_data["cells"][0]
    assert cell["cell_verdict"] == "reusable"
    fold = cell["folds"][0]
    assert fold["verdict"] == "reusable"
    candidate = fold["candidates"][0]
    assert candidate["checks"]["prompt_matches"] is True
    assert candidate["checks"]["recomputation_matches"] is True


def test_inventory_marks_a_shape_only_run_pending_decision(synthetic_project: Path) -> None:
    _write_run(synthetic_project, perturb={"training.gradient_accumulation_steps": 16})
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    fold = inventory_data["cells"][0]["folds"][0]
    assert fold["verdict"] == "reusable_pending_shape_decision"
    assert fold["candidates"][0]["checks"]["shape_differences"] == [
        "training.gradient_accumulation_steps"
    ]
    assert fold["candidates"][0]["checks"]["scientific_differences"] == []


def test_inventory_marks_a_prompt_change_incompatible(synthetic_project: Path) -> None:
    _write_run(synthetic_project, perturb={"prompt.version": "promptcontext_v0"})
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    fold = inventory_data["cells"][0]["folds"][0]
    assert fold["verdict"] == "incompatible"
    assert "science" in fold["candidates"][0]["blocking_checks"]


def test_inventory_marks_missing_evidence_incomplete(synthetic_project: Path) -> None:
    _write_run(synthetic_project, with_evidence=False)
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    fold = inventory_data["cells"][0]["folds"][0]
    assert fold["verdict"] == "incomplete"
    assert "local_evidence" in fold["candidates"][0]["missing_checks"]


def test_inventory_marks_a_recomputation_mismatch_incomplete(synthetic_project: Path) -> None:
    _write_run(synthetic_project, metrics_value=0.5)
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    fold = inventory_data["cells"][0]["folds"][0]
    assert fold["verdict"] == "incomplete"
    assert "recomputation" in fold["candidates"][0]["missing_checks"]


def test_inventory_excludes_marked_campaigns(synthetic_project: Path) -> None:
    _write_run(synthetic_project, run_name="q3dlv_label_vocab_run")
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    fold = inventory_data["cells"][0]["folds"][0]
    assert fold["verdict"] == "missing"
    assert len(fold["excluded_candidates"]) == 1
    assert fold["excluded_candidates"][0]["exclusion_markers"] == ["q3dlv", "label_vocab"]


def test_inventory_reports_missing_cells(synthetic_project: Path) -> None:
    (synthetic_project / "scan").mkdir(parents=True, exist_ok=True)
    inventory_data = _run_inventory(synthetic_project, _write_matrix(synthetic_project))
    assert inventory_data["cells"][0]["cell_verdict"] == "missing"
    assert inventory_data["summary"]["cell_verdicts"]["missing"] == 1


def test_selection_map_carries_every_route_and_every_merged_contract() -> None:
    selection_map = planning.build_selection_map()
    native = [route for route in selection_map["routes"] if route["language"] == "native"]
    english = [route for route in selection_map["routes"] if route["language"] == "english"]
    assert len(native) == 15
    assert len(english) == 8
    merged_ids = {route["route_id"] for route in selection_map["merged_routes"]}
    assert merged_ids == {
        "merged_native_text_only",
        "merged_native_audio_only",
        "merged_native_audio_text",
        "merged_english_text_only",
        "merged_english_audio_text",
    }
    assert planning.check_selection_map(selection_map, require_english_audio_text=True) == []


def test_selection_map_requires_the_english_audio_text_contract_when_asked() -> None:
    selection_map = planning.build_selection_map()
    without = json.loads(json.dumps(selection_map))
    without["merged_routes"] = [
        route
        for route in without["merged_routes"]
        if route["route_id"] != "merged_english_audio_text"
    ]
    failures = planning.check_selection_map(without, require_english_audio_text=True)
    assert any("merged_english_audio_text" in failure for failure in failures)


def test_selection_map_checks_fail_closed_on_drift() -> None:
    selection_map = planning.build_selection_map()
    drifted = json.loads(json.dumps(selection_map))
    drifted["routes"][0]["evaluation"]["sample_prediction_mode"] = "original_teacher_forced"
    drifted["routes"][0]["resources"]["eval_gpus_per_node"] = 8
    failures = planning.check_selection_map(drifted, require_english_audio_text=False)
    assert any("decision rule must stay likelihood" in failure for failure in failures)
    assert any("evaluation shape must be" in failure for failure in failures)


def test_production_manifest_plans_three_seeds_and_marks_reuse() -> None:
    selection_map = planning.build_selection_map()
    inventory = {
        "cells": [
            {
                "cell_id": "d3tec_text_only_qwen38",
                "dataset": "d3tec",
                "modality": "text_only",
                "cell_verdict": "reusable",
                "folds": [
                    {"fold": 0, "selected_run": "run_f0", "selected_attempt_id": "attempt-f0"},
                ],
            },
            {
                "cell_id": "d3tec_audio_text_qwen3omni",
                "dataset": "d3tec",
                "modality": "audio_text",
                "cell_verdict": "reusable_pending_shape_decision",
                "folds": [],
            },
        ]
    }
    manifest = planning.build_production_manifest(selection_map, inventory=inventory)
    assert manifest["status"] == "planned_not_submitted"
    assert manifest["seeds"] == [7, 1337, 2024]
    assert manifest["summary"]["standalone_jobs"] == 468
    # 5 contracts x 3 seeds x 6 stage-folds x (train + postprocess + head): every
    # contract passed its hidden-feature audit, so the head kind is planned.
    assert manifest["summary"]["merged_jobs"] == 270
    assert manifest["summary"]["total_planned_jobs"] == manifest["summary"]["standalone_jobs"] + manifest["summary"]["merged_jobs"]
    d3tec_text = next(
        route for route in manifest["standalone"] if route["route_id"] == "d3tec_text_only_native"
    )
    seed_1337 = [job for job in d3tec_text["jobs"] if job["seed"] == 1337 and job["kind"] == "train"]
    assert seed_1337[0]["reuse_seed_1337"]["attempt_id"] == "attempt-f0"
    assert seed_1337[0]["reuse_seed_1337"]["open_decision"] is None
    other_seed = [
        job for job in d3tec_text["jobs"] if job["seed"] == 2024 and job["kind"] == "train"
    ]
    assert other_seed[0]["reuse_seed_1337"] is None
    # Every contract passed its hidden-feature audit, so each merged plan carries
    # its train, postprocess and head jobs.
    assert all(route["head_ready"] for route in manifest["merged"])
    assert all(
        job["kind"] in {"train", "postprocess", "head"}
        for route in manifest["merged"]
        for job in route["jobs"]
    )
    assert all(
        job["dependency"] != "head"
        for route in manifest["merged"]
        for job in route["jobs"]
        if job["kind"] == "train"
    )


def test_production_manifest_records_waiting_head_jobs() -> None:
    selection_map = planning.build_selection_map()
    head_matrix = {
        "routes": [
            {
                "route_id": "d3tec_text_only_native",
                "config": "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
                "jobs": [
                    {
                        "seed": 1337,
                        "fold": 0,
                        "parent_status": "waiting_for_checkpoint",
                        "reason": "checkpoint not present",
                    }
                ],
            }
        ]
    }
    manifest = planning.build_production_manifest(selection_map, head_matrix=head_matrix)
    assert manifest["summary"]["standalone_head_jobs_planned"] == 0
    assert manifest["summary"]["standalone_head_jobs_waiting"] == 1
    assert manifest["standalone_heads"][0]["status"] == "waiting_for_checkpoint"


def test_selection_map_records_templates_not_expanded_paths() -> None:
    selection_map = planning.build_selection_map()
    manifest_dirs = {route["manifest"]["dir"] for route in selection_map["routes"]}
    assert any(str(value).startswith("${PROJECT_ROOT}") for value in manifest_dirs)
    assert not any(str(value).startswith("/home/") for value in manifest_dirs)
