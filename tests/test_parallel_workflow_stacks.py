from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.compare import ComparisonError, plan_integration


def _git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True, check=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(["init", "-q", "-b", "main"], r)
    _git(["config", "user.email", "t@t"], r)
    _git(["config", "user.name", "T"], r)
    (r / "configs").mkdir()
    (r / "configs" / "base.yaml").write_text("evaluation:\n  evaluation_view: view_a\nthreshold: 0.5\n")
    (r / "src.txt").write_text("hello\n")
    _git(["add", "."], r)
    _git(["commit", "-q", "-m", "base"], r)
    return r


def _branch(repo, name, edits, base=None):
    """Create a branch from base (default HEAD) applying file edits."""
    start = base or "HEAD"
    sha = _git(["rev-parse", start], repo).stdout.strip()
    _git(["checkout", "-q", "-b", name, sha], repo)
    for rel, content in edits.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", f"changes on {name}"], repo)
    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "main"], repo)
    return head


def test_stacked_child_records_exact_parent_sha_without_merge(repo):
    parent_sha = _branch(repo, "agent/exp-base",
                         {"src.txt": "parent change\n"})
    # child branches from the parent's exact unmerged SHA
    child_sha = _branch(repo, "agent/exp-base-pooling",
                        {"src.txt": "parent change\nchild change\n"},
                        base=parent_sha)
    plan = plan_integration(repo, branch_a="agent/exp-base", branch_b="agent/exp-base-pooling", base="main")
    assert plan["stacked"] is True
    assert plan["merge_base_ab"] == parent_sha
    # the child's ancestry contains the parent commit; main is not involved
    assert child_sha != parent_sha


def test_competing_lanes_with_git_conflict_are_detected(repo):
    _branch(repo, "agent/exp-a", {"src.txt": "lane A\n"})
    _branch(repo, "agent/exp-b", {"src.txt": "lane B\n"})
    plan = plan_integration(repo, branch_a="agent/exp-a", branch_b="agent/exp-b", base="main")
    assert plan["git_conflicts"], "conflicting edits to the same file must be reported"
    assert plan["automatic_merge_authorized"] is False


def test_clean_git_merge_with_semantic_conflict_is_flagged(repo):
    # Git merges cleanly: disjoint-line edits to the SAME contract file.
    _branch(repo, "agent/exp-a", {"configs/base.yaml": "# touched by a\nevaluation:\n  evaluation_view: view_a\nthreshold: 0.5\n"})
    _branch(repo, "agent/exp-b", {"configs/base.yaml": "evaluation:\n  evaluation_view: view_a\nthreshold: 0.6\n"})
    plan = plan_integration(repo, branch_a="agent/exp-a", branch_b="agent/exp-b", base="main")
    assert not plan["git_conflicts"]
    assert "configs/base.yaml" in plan["semantic_conflict_candidates"]
    assert plan["cross_feature_tests_required"] is True
    assert plan["automatic_merge_authorized"] is False
    assert "decision required" in plan["decision"]
    plan = plan_integration(repo, branch_a="agent/exp-a", branch_b="agent/exp-b", base="main")
    assert not plan["git_conflicts"]
    assert "configs/base.yaml" in plan["semantic_conflict_candidates"]
    assert plan["cross_feature_tests_required"] is True
    assert plan["automatic_merge_authorized"] is False
    assert "decision required" in plan["decision"]


def test_non_overlapping_non_contract_changes_need_no_cross_tests(repo):
    _branch(repo, "agent/exp-a", {"docs_notes_a.md": "a\n"})
    _branch(repo, "agent/exp-b", {"docs_notes_b.md": "b\n"})
    plan = plan_integration(repo, branch_a="agent/exp-a", branch_b="agent/exp-b", base="main")
    assert not plan["git_conflicts"]
    assert plan["semantic_conflict_candidates"] == []
    assert plan["cross_feature_tests_required"] is False


def test_missing_reference_fails_closed(repo):
    with pytest.raises(ComparisonError, match="not found"):
        plan_integration(repo, branch_a="nope", branch_b="main")


def test_pin_definition_records_parent_for_stacks(tmp_path):
    """The pin/definition contract records parent branch and exact parent SHA."""
    pin = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": "exp-child-20260821",
        "worktree": "/home/emre/worktrees/LLM-Depression-exp-child",
        "branch": "agent/exp-child",
        "allowed_paths": ["/home/emre/worktrees/LLM-Depression-exp-child"],
        "protected_paths": ["/home/emre/Projects/AudioLLM/Teacher-System"],
        "parent_branch": "agent/exp-base",
        "parent_sha": "1234567890abcdef1234567890abcdef12345678",
    }
    assert pin["parent_branch"] == "agent/exp-base"
    assert len(pin["parent_sha"]) == 40
