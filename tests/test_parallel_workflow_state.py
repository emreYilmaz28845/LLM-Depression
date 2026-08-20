from __future__ import annotations
import json, subprocess, pathlib, hashlib, tempfile, sys, os

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "parallel_workflow_state.py"
AUDIT_TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "audit_parallel_workflow_implementation.py"
STATE_SCHEMA = "audiollm.parallel_workflow_execution.v1"

def run_tool(*args):
    cmd = [sys.executable, str(TOOL)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def init_state(tmp_path, execution_id="20260820T205735Z-parallel-workflow-2d995f4c"):
    runbook = pathlib.Path(__file__).resolve().parents[1] / "docs" / "PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md"
    if not runbook.exists():
        runbook = pathlib.Path("/home/emre/Projects/AudioLLM/LLM-Depression/docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md")
    if not runbook.exists():
        runbook = pathlib.Path("docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md")
    state_path = tmp_path / "state.json"
    res = run_tool("init", "--runbook", str(runbook), "--execution-id", execution_id, "--output", str(state_path))
    assert res.returncode == 0, res.stderr
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["schema_version"] == STATE_SCHEMA
    assert data["execution_id"] == execution_id
    assert data["phases"]["0"]["status"] == "IN_PROGRESS"
    return state_path

def test_init_and_show(tmp_path):
    state_path = init_state(tmp_path)
    res = run_tool("show", "--state", str(state_path))
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["execution_id"] == "20260820T205735Z-parallel-workflow-2d995f4c"

def test_record_and_pass_phase0_requires_evidence(tmp_path):
    state_path = init_state(tmp_path)
    res = run_tool("pass", "--state", str(state_path), "--phase", "0", "--next-phase", "1")
    assert res.returncode != 0
    assert "missing required evidence" in res.stderr.lower() or "missing" in res.stderr.lower()
    res = run_tool("record", "--state", str(state_path), "--phase", "0", "--evidence", "grant_journal: docs/agent-journal/2026-08-20.md")
    assert res.returncode == 0
    res = run_tool("pass", "--state", str(state_path), "--phase", "0", "--next-phase", "1")
    assert res.returncode != 0
    for ev in [
        "baseline: git status 2d995f4 branch main",
        "worktree: /home/emre/worktrees/LLM-Depression-feat-parallel-workflow branch agent/feat-parallel-experiment-workflow",
        "state.json: outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json sha 44635",
        "test: tests/test_parallel_workflow_state.py passed",
        "PR: https://github.com/emre/LLM-Depression/pull/999 head sha abc merge sha def",
        "audit: tools/audit_parallel_workflow_implementation.py --allow-incomplete passed",
    ]:
        run_tool("record", "--state", str(state_path), "--phase", "0", "--evidence", ev)
    res = run_tool("pass", "--state", str(state_path), "--phase", "0", "--next-phase", "1")
    assert res.returncode == 0, res.stderr
    data = json.loads(state_path.read_text())
    assert data["phases"]["0"]["status"] == "PASSED"
    assert data["phases"]["1"]["status"] == "IN_PROGRESS"
    assert data["current_phase"] == 1

def test_rejects_skipping_phase(tmp_path):
    state_path = init_state(tmp_path)
    res = run_tool("enter", "--state", str(state_path), "--phase", "2", "--next-action", "skip")
    assert res.returncode != 0
    assert "previous phase" in res.stderr.lower() or "cannot" in res.stderr.lower()

def test_rejects_completion_with_pending_phases(tmp_path):
    state_path = init_state(tmp_path)
    audit_fake = tmp_path / "audit.json"
    audit_fake.write_text("{}")
    res = run_tool("complete", "--state", str(state_path), "--audit", str(audit_fake))
    assert res.returncode != 0
    assert "not passed" in res.stderr.lower() or "cannot" in res.stderr.lower()

def test_rejects_execution_id_change(tmp_path):
    state_path = init_state(tmp_path)
    data = json.loads(state_path.read_text())
    data["execution_id"] = "tampered-id"
    tmp_tampered = tmp_path / "tampered.json"
    tmp_tampered.write_text(json.dumps(data))
    res = subprocess.run([sys.executable, str(AUDIT_TOOL), "--state", str(tmp_tampered), "--allow-incomplete"], capture_output=True, text=True)
    assert res.returncode != 0

def test_atomic_write_and_resume(tmp_path):
    state_path = init_state(tmp_path)
    run_tool("record", "--state", str(state_path), "--phase", "0", "--evidence", "grant_journal: docs/agent-journal/2026-08-20.md")
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert any("grant_journal" in ev for ev in data["phases"]["0"]["evidence"])
    res = run_tool("record", "--state", str(state_path), "--phase", "0", "--evidence", "grant_journal: docs/agent-journal/2026-08-20.md")
    assert res.returncode == 0
    data2 = json.loads(state_path.read_text())
    assert data2["phases"]["0"]["evidence"].count("grant_journal: docs/agent-journal/2026-08-20.md") == 1

def test_hard_stop_blocks_progress(tmp_path):
    state_path = init_state(tmp_path)
    res = run_tool("hard-stop", "--state", str(state_path), "--phase", "0", "--reason", "test blocker", "--evidence", "outputs/blocker.json")
    assert res.returncode == 0
    data = json.loads(state_path.read_text())
    assert data["status"] == "HARD_STOP"
    res = run_tool("pass", "--state", str(state_path), "--phase", "0", "--next-phase", "1")
    assert res.returncode != 0
    assert "hard_stop" in res.stderr.lower()

def test_enter_rejects_when_previous_not_passed(tmp_path):
    state_path = init_state(tmp_path)
    for ev in [
        "grant_journal: docs/agent-journal/2026-08-20.md",
        "baseline: git status",
        "worktree: /home/emre/worktrees/LLM-Depression-feat-parallel-workflow",
        "state.json: outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json",
        "test: tests/test_parallel_workflow_state.py",
        "PR: https://github.com/test/pr/1",
        "audit: audit passed",
    ]:
        run_tool("record", "--state", str(state_path), "--phase", "0", "--evidence", ev)
    res = run_tool("pass", "--state", str(state_path), "--phase", "0", "--next-phase", "1")
    assert res.returncode == 0
    res = run_tool("enter", "--state", str(state_path), "--phase", "3", "--next-action", "skip")
    assert res.returncode != 0
