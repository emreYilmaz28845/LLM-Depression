from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.turkish_question_condition import (
    EVALUATION_VIEW,
    GROUP_ID,
    SMOKE_CELL_IDS,
    build_plan,
    load_cells,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "e176da5e0595464bc44320d32e04f7fe0a7adf5e"


def test_production_plan_has_locked_cardinality_and_dependencies() -> None:
    plan = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert plan["group_id"] == GROUP_ID
    assert plan["expected_counts"] == {
        "cells": 20,
        "seeds": 3,
        "folds": 5,
        "backbone_fold_runs": 300,
        "teacher_forced_evaluations": 300,
        "logreg_routes": 300,
        "xgb_routes": 300,
        "slurm_jobs": 1200,
        "xgb_completed_trials": 100,
    }
    assert len(plan["jobs"]) == 1200
    by_key = {job["job_key"]: job for job in plan["jobs"]}
    assert len(by_key) == 1200
    for job in plan["jobs"]:
        if job["job_type"] == "train":
            assert job["resource_shape"].startswith("1 node, 4 tasks, 4 H100")
        elif job["job_type"] in {"evaluation", "hidden_extraction_logreg"}:
            assert "1 H100" in job["resource_shape"]
        else:
            assert "0 GPU" in job["resource_shape"]
        assert job["evaluation_view"] == EVALUATION_VIEW
    for job in plan["jobs"]:
        if job["route"] == "teacher_forced":
            assert by_key[job["dependencies"][0]]["route"] == "backbone_train"
        if job["route"] == "logreg":
            assert by_key[job["dependencies"][0]]["route"] == "teacher_forced"
        if job["route"] == "xgb_optuna100":
            assert by_key[job["dependencies"][0]]["route"] == "logreg"


def test_smoke_plan_is_the_locked_six_cell_negative_only_subset() -> None:
    plan = build_plan(stage="smoke", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert plan["selected_cell_ids"] == list(SMOKE_CELL_IDS)
    assert plan["expected_counts"] == {
        "cells": 6,
        "seeds": 1,
        "folds": 1,
        "backbone_fold_runs": 6,
        "teacher_forced_evaluations": 6,
        "logreg_routes": 6,
        "xgb_routes": 6,
        "slurm_jobs": 24,
        "xgb_completed_trials": 2,
    }
    assert all(job["seed"] == 1337 and job["fold"] == 0 for job in plan["jobs"])
    assert all(job["recording_condition"] == "negative_only" for job in plan["jobs"])
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


def test_matrix_rejects_english_audio_only_if_group_is_changed(tmp_path: Path) -> None:
    group = ROOT / "experiments/definitions/turkish-pos_only-vs-negonly-native-en-multimodal-heads-v1-20260903.yaml"
    changed = tmp_path / "experiments/definitions" / group.name
    changed.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    text = group.read_text(encoding="utf-8").replace(
        "transcript_condition: not_applicable, modality: audio_only, backbone: qwen",
        "transcript_condition: english, modality: audio_only, backbone: qwen",
        1,
    )
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="audio-only cell"):
        load_cells(tmp_path)
