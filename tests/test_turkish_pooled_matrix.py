from __future__ import annotations

from pathlib import Path

from src.turkish_pooled_qcond import build_plan
from tools.turkish_pooled_qcond import _preflight_script


ROOT = Path(__file__).parents[1]
SOURCE_SHA = "a" * 40


def test_smoke_and_production_plans_have_locked_cardinality_and_disjoint_names() -> None:
    smoke = build_plan(stage="smoke", source_sha=SOURCE_SHA, repo_root=ROOT)
    production = build_plan(stage="production", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert smoke["expected_counts"]["slurm_jobs"] == 8
    assert production["expected_counts"]["slurm_jobs"] == 600
    assert smoke["expected_counts"]["xgb_completed_trials"] == 2
    assert production["expected_counts"]["xgb_completed_trials"] == 100
    smoke_names = {job["run_name"] for job in smoke["jobs"]}
    production_names = {job["run_name"] for job in production["jobs"]}
    assert smoke_names.isdisjoint(production_names)
    assert all(job["run_name"].startswith("tpq_smoke_") for job in smoke["jobs"])
    assert all(job["run_name"].startswith("tpq_prod_") for job in production["jobs"])


def test_matrix_resources_dependencies_and_remote_preflight_are_safe() -> None:
    plan = build_plan(stage="smoke", source_sha=SOURCE_SHA, repo_root=ROOT)
    assert {job["resource_shape"] for job in plan["jobs"]} == {
        "1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4 (DDP)",
        "1 node, 1 task, 1 H100",
        "1 node, 1 task, 0 GPU, 20 CPUs",
    }
    assert plan["jobs"][1]["dependencies"] == [f"{plan['jobs'][0]['run_name']}:train"]
    assert plan["protocol"]["text_pair_policy"] == "turkish_pooled_text_pair_mean_margin_strict_v1"
    deployment = {"deployed_code_path": "/gpfs/projects/etur92/ozu647717/deployments/test/code"}
    script = _preflight_script(deployment, stage="smoke")
    assert "smoke.json" in script
    assert "--require-models --require-environment" in script
    assert "--delete" not in script
