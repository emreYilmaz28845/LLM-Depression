"""Pooled campaign matrix invariants (plan Step 6)."""

from __future__ import annotations

import json
from pathlib import Path

from src.turkish_pooled_qcond import (
    EVALUATION_VIEW,
    GROUP_ID,
    SMOKE_CELL_IDS,
    build_plan,
    load_cells,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "d74dbe35da92e310b89ae91df0556f37b1c03f76"


def test_production_plan_has_locked_cardinality_and_dependencies() -> None:
    plan = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert plan["group_id"] == GROUP_ID
    assert plan["expected_counts"] == {
        "cells": 10,
        "seeds": 3,
        "folds": 5,
        "backbone_fold_runs": 150,
        "teacher_forced_evaluations": 150,
        "logreg_routes": 150,
        "xgb_routes": 150,
        "slurm_jobs": 600,
        "xgb_completed_trials": 100,
    }
    assert len(plan["jobs"]) == 600
    by_key = {job["job_key"]: job for job in plan["jobs"]}
    assert len(by_key) == 600
    for job in plan["jobs"]:
        if job["job_type"] == "train":
            assert job["resource_shape"].startswith("1 node, 4 tasks, 4 H100")
        elif job["job_type"] in {"evaluation", "hidden_extraction_logreg"}:
            assert "1 H100" in job["resource_shape"]
        else:
            assert "0 GPU" in job["resource_shape"]
        assert job["evaluation_view"] == EVALUATION_VIEW
        assert job["recording_condition"] == "pooled"
    for job in plan["jobs"]:
        if job["route"] == "teacher_forced":
            assert by_key[job["dependencies"][0]]["route"] == "backbone_train"
        if job["route"] == "logreg":
            assert by_key[job["dependencies"][0]]["route"] == "teacher_forced"
        if job["route"] == "xgb_optuna100":
            assert by_key[job["dependencies"][0]]["route"] == "logreg"


def test_pooled_cells_share_two_prebuilt_manifest_pairs() -> None:
    plan = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    pairs = {(job["manifest_dir"], job["split_dir"]) for job in plan["jobs"] if job["job_type"] == "train"}
    assert len(pairs) == 2
    assert all("pooled" in manifest for manifest, _ in pairs)
    english = {(m, s) for m, s in pairs if "pooled_en" in m}
    native = {(m, s) for m, s in pairs if "pooled_en" not in m}
    assert len(english) == 1 and len(native) == 1


def test_smoke_plan_is_the_locked_two_cell_subset() -> None:
    plan = build_plan(stage="smoke", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert plan["selected_cell_ids"] == list(SMOKE_CELL_IDS)
    assert plan["expected_counts"] == {
        "cells": 2,
        "seeds": 1,
        "folds": 1,
        "backbone_fold_runs": 2,
        "teacher_forced_evaluations": 2,
        "logreg_routes": 2,
        "xgb_routes": 2,
        "slurm_jobs": 8,
        "xgb_completed_trials": 2,
    }
    assert all(job["seed"] == 1337 and job["fold"] == 0 for job in plan["jobs"])
    assert all(job["recording_condition"] == "pooled" for job in plan["jobs"])
    assert all("--set=training.num_train_epochs=1" in job["overrides"] for job in plan["jobs"] if job["job_type"] == "train")
    assert all("--set=split.smoke_subject_limit=6" in job["overrides"] for job in plan["jobs"] if job["job_type"] == "train")


def test_plan_generation_is_byte_identical_and_hash_sidecar_is_checked(tmp_path: Path) -> None:
    first = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    second = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert first == second
    target, sidecar = write_plan(first, tmp_path / "plan.json")
    first_bytes = target.read_bytes()
    write_plan(second, target)
    assert target.read_bytes() == first_bytes
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["plan_sha256"] == first["plan_sha256"]


def test_matrix_loads_ten_pooled_cells() -> None:
    cells = load_cells(ROOT)
    assert len(cells) == 10
    assert {cell.cell_id for cell in cells} == {f"Q{i:02d}" for i in range(1, 6)} | {f"G{i:02d}" for i in range(1, 6)}
    assert all(cell.recording_condition == "pooled" for cell in cells)
