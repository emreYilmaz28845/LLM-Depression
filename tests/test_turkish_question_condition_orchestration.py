from __future__ import annotations

from pathlib import Path
import subprocess

from src.turkish_question_condition import build_plan
from tools.turkish_question_condition import (
    GROUP_ID,
    _make_submission_plan,
    _remote_submission_script,
)


def _preflight() -> dict[str, object]:
    return {
        "status": "passed",
        "group_id": GROUP_ID,
        "pairs": [
            {
                "condition": condition,
                "language": language,
                "metadata_manifest_hash": f"manifest-{condition}-{language}",
                "metadata_sha256": f"split-{condition}-{language}",
                "folds": {"fold_hash": f"fold-{condition}"},
            }
            for condition in ("mixed", "negative_only")
            for language in ("native", "english")
        ],
    }


def _deployment() -> dict[str, object]:
    return {
        "deployment_id": "exp-turkish-full-negonly-multimodal-20260823T000000Z-e176da5e-abcdef12",
        "experiment_id": "exp-turkish-full-negonly-multimodal-20260823",
        "git_commit": "e176da5e0595464bc44320d32e04f7fe0a7adf5e",
        "git_branch_at_deploy": "agent/exp-turkish-full-negonly-multimodal",
        "git_dirty": False,
        "source_manifest_sha256": "source-manifest",
        "deployed_code_path": "/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/fake/code",
    }


def test_submission_plan_expands_the_locked_smoke() -> None:
    matrix = build_plan(stage="smoke", source_sha=_deployment()["git_commit"], repo_root=Path(__file__).parents[1])
    plan = _make_submission_plan(matrix=matrix, deployment=_deployment(), preflight=_preflight())
    assert len(plan["backbones"]) == 6
    assert len(plan["head_jobs"]) == 12
    assert plan["expected_counts"]["slurm_jobs"] == 24
    assert {item["backbone_attempt_id"] for item in plan["backbones"]}
    assert all(item["context"]["group_id"] == GROUP_ID for item in plan["head_jobs"])
    assert all(item["config"]["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage" for item in plan["head_jobs"])
    assert all(item["trials"] in (0, 2) for item in plan["head_jobs"])


def test_submission_script_is_dependency_aware_and_non_destructive() -> None:
    matrix = build_plan(stage="smoke", source_sha=_deployment()["git_commit"], repo_root=Path(__file__).parents[1])
    plan = _make_submission_plan(matrix=matrix, deployment=_deployment(), preflight=_preflight())
    script = _remote_submission_script(plan, _deployment())
    assert "--dependency=afterok:$eval_0" in script
    assert "--dependency=afterok:$logreg_0" in script
    assert "scripts/run_turkish_question_logreg_slurm.sh" in script
    assert "scripts/run_turkish_question_xgb_slurm.sh" in script
    assert "--delete" not in script
    assert "__SUBMISSION_COMPLETE__ 6 12" in script
    checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
