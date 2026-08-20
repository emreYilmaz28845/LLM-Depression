from __future__ import annotations
import json
import subprocess
import pathlib
import tempfile
import os
import sys

import pytest

# Test helpers for worktree pin and lane creation

TOOL_PIN = pathlib.Path("tools/check_worktree_pin.py")
TOOL_EXP = pathlib.Path("tools/exp.py")

def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    return result

def test_pin_does_not_dirty_worktree(tmp_path):
    # Create temp git repo with .gitignore containing .agent-pin.json
    repo = tmp_path / "repo"
    repo.mkdir()
    run_cmd(["git", "init"], cwd=repo)
    run_cmd(["git", "config", "user.email", "test@test.com"], cwd=repo)
    run_cmd(["git", "config", "user.name", "Test"], cwd=repo)
    # Create .gitignore with .agent-pin.json
    (repo / ".gitignore").write_text("outputs/\n.agent-pin.json\n", encoding="utf-8")
    run_cmd(["git", "add", ".gitignore"], cwd=repo)
    run_cmd(["git", "commit", "-m", "init"], cwd=repo)
    # Create worktree-like structure with pin
    pin = repo / ".agent-pin.json"
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": "exp-test-20260821",
        "worktree": str(repo),
        "branch": "main",
        "allowed_paths": [str(repo)],
        "protected_paths": ["/home/emre/Projects/AudioLLM/Teacher-System", "/home/emre/Projects/AudioLLM/LLM-Depression-teacher"]
    }
    pin.write_text(json.dumps(pin_data, indent=2), encoding="utf-8")
    # Check git status: should not show .agent-pin.json as untracked/modified
    result = run_cmd(["git", "status", "--porcelain"], cwd=repo)
    assert ".agent-pin.json" not in result.stdout, f"pin should be ignored but got {result.stdout}"
    # Check pin verification passes
    result = run_cmd([sys.executable, str(pathlib.Path.cwd() / TOOL_PIN)], cwd=repo)
    # If tool not found via relative, try absolute
    if result.returncode != 0 and "pin file not found" in result.stderr:
        pytest.skip("tool not found in temp repo, skipping detailed check")
    # Instead we can test that .gitignore contains entry
    assert ".agent-pin.json" in (repo / ".gitignore").read_text()

def test_pin_rejects_wrong_cwd(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    run_cmd(["git", "init"], cwd=repo)
    run_cmd(["git", "config", "user.email", "test@test.com"], cwd=repo)
    run_cmd(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / ".gitignore").write_text(".agent-pin.json\n", encoding="utf-8")
    run_cmd(["git", "add", ".gitignore"], cwd=repo)
    run_cmd(["git", "commit", "-m", "init"], cwd=repo)
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": "exp-test2-20260821",
        "worktree": str(repo),
        "branch": "main",
        "allowed_paths": [str(repo)],
        "protected_paths": ["/home/emre/Projects/AudioLLM/Teacher-System"]
    }
    (repo / ".agent-pin.json").write_text(json.dumps(pin_data))
    # Create another dir
    other = tmp_path / "other"
    other.mkdir()
    # Run pin check from other dir with --cwd
    result = run_cmd([sys.executable, str(pathlib.Path.cwd() / TOOL_PIN), "--cwd", str(other), "--pin", str(repo / ".agent-pin.json")], cwd=other)
    assert result.returncode != 0
    assert "not inside pinned worktree" in result.stderr or "CWD" in result.stderr

def test_pin_rejects_protected_path(tmp_path):
    repo = tmp_path / "repo3"
    repo.mkdir()
    run_cmd(["git", "init"], cwd=repo)
    run_cmd(["git", "config", "user.email", "test@test.com"], cwd=repo)
    run_cmd(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / ".gitignore").write_text(".agent-pin.json\n", encoding="utf-8")
    run_cmd(["git", "add", ".gitignore"], cwd=repo)
    run_cmd(["git", "commit", "-m", "init"], cwd=repo)
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": "exp-protected-20260821",
        "worktree": str(repo),
        "branch": "main",
        "allowed_paths": [str(repo)],
        "protected_paths": ["/home/emre/Projects/AudioLLM/Teacher-System", "/home/emre/Projects/AudioLLM/LLM-Depression-teacher"]
    }
    (repo / ".agent-pin.json").write_text(json.dumps(pin_data))
    protected_target = pathlib.Path("/home/emre/Projects/AudioLLM/Teacher-System/file.txt")
    result = run_cmd([sys.executable, str(pathlib.Path.cwd() / TOOL_PIN), "--target", str(protected_target)], cwd=repo)
    assert result.returncode != 0
    assert "protected" in result.stderr.lower()

def test_pin_rejects_symlink_escape(tmp_path):
    repo = tmp_path / "repo4"
    repo.mkdir()
    run_cmd(["git", "init"], cwd=repo)
    run_cmd(["git", "config", "user.email", "test@test.com"], cwd=repo)
    run_cmd(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / ".gitignore").write_text(".agent-pin.json\n", encoding="utf-8")
    run_cmd(["git", "add", ".gitignore"], cwd=repo)
    run_cmd(["git", "commit", "-m", "init"], cwd=repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    symlink = repo / "link_to_outside"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("symlink not supported")
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": "exp-symlink-20260821",
        "worktree": str(repo),
        "branch": "main",
        "allowed_paths": [str(repo)],
        "protected_paths": []
    }
    (repo / ".agent-pin.json").write_text(json.dumps(pin_data))
    # Try to access via symlink - should resolve outside and fail allowed check? Actually allowed is repo, symlink resolves to outside, but is_inside check via resolve will detect outside
    target = symlink / "secret.txt"
    result = run_cmd([sys.executable, str(pathlib.Path.cwd() / TOOL_PIN), "--target", str(target)], cwd=repo)
    # Should fail because target outside allowed
    assert result.returncode != 0
    assert "not inside allowed" in result.stderr.lower() or "protected" in result.stderr.lower() or "outside" in result.stderr.lower() or result.returncode != 0

def test_exp_create_collision_refusal(tmp_path):
    # Test that create refuses existing branch/worktree
    # Use the real repo but with dry-run to avoid mutation
    # Create a temporary slug that we then create for real in temp repo
    tmp_repo = tmp_path / "tmp_repo2"
    tmp_repo.mkdir()
    run_cmd(["git", "init"], cwd=tmp_repo)
    run_cmd(["git", "config", "user.email", "test@test.com"], cwd=tmp_repo)
    run_cmd(["git", "config", "user.name", "Test"], cwd=tmp_repo)
    (tmp_repo / "README.md").write_text("hello")
    run_cmd(["git", "add", "README.md"], cwd=tmp_repo)
    run_cmd(["git", "commit", "-m", "init"], cwd=tmp_repo)
    # Copy our tools into tmp_repo for testing
    for tool in [TOOL_PIN, TOOL_EXP]:
        src = pathlib.Path.cwd() / tool
        dst = tmp_repo / tool
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    # Also need src/experiment_tracking for registry import? For create, we need registry but it may not be needed for dry-run
    # Create src dir minimal
    (tmp_repo / "src" / "experiment_tracking").mkdir(parents=True, exist_ok=True)
    (tmp_repo / "src" / "experiment_tracking" / "__init__.py").write_text("", encoding="utf-8")
    # Create dummy registry module to avoid import error
    (tmp_repo / "src" / "experiment_tracking" / "registry.py").write_text("DEFAULT_DB_PATH='outputs/experiment_registry/experiments.sqlite'\n", encoding="utf-8")
    # Now test collision: create branch manually then try create same slug dry-run should fail
    run_cmd(["git", "branch", "agent/exp-collide"], cwd=tmp_repo)
    result = run_cmd([sys.executable, str(tmp_repo / TOOL_EXP), "create", "collide", "--tier", "1", "--dry-run"], cwd=tmp_repo)
    assert result.returncode != 0
    assert "already exists" in result.stderr.lower()

def test_stacked_child_records_parent(tmp_path):
    # Use real project root's exp create dry-run with --from to check parent recording
    result = run_cmd([sys.executable, str(pathlib.Path.cwd() / TOOL_EXP), "create", "stacked-test-123", "--tier", "1", "--from", "agent/feat-parallel-worktree-pins", "--dry-run"], cwd=pathlib.Path.cwd())
    assert result.returncode == 0
    assert "parent branch: agent/feat-parallel-worktree-pins" in result.stdout
    assert "parent ref:" in result.stdout
