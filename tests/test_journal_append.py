from __future__ import annotations

import fcntl
import json
import os
import pathlib
import subprocess
import sys
from datetime import date, datetime, timezone

import pytest

from tools.journal_append import (
    Identity,
    JournalError,
    build_entry,
    derive_identity,
    journal_file,
    journal_root,
)

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "journal_append.py"


def run_tool(args, env, cwd=None, stdin=None):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=env,
        input=stdin,
    )


def make_env(tmp_path, root):
    env = dict(os.environ)
    env["AGENT_JOURNAL_ROOT"] = str(root)
    env.pop("PYTHONPATH", None)
    return env


def make_journal_root(tmp_path):
    root = tmp_path / "Agent-Journal"
    (root / "LLM-Depression").mkdir(parents=True)
    return root


def init_git_repo(path):
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("journal repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True
    )


def test_identity_from_pin(tmp_path):
    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / ".agent-pin.json").write_text(
        json.dumps(
            {
                "schema_version": "audiollm.agent_pin.v1",
                "experiment_id": "exp-pilot-a-20260912",
                "branch": "agent/exp-pilot-a",
                "worktree": str(lane),
            }
        ),
        encoding="utf-8",
    )
    identity = derive_identity(lane)
    assert identity.label == "agent/exp-pilot-a"
    assert "experiment_id=exp-pilot-a-20260912" in identity.detail
    assert f"worktree={lane}" in identity.detail


def test_identity_from_pin_prefers_git_toplevel(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    (repo / ".agent-pin.json").write_text(
        json.dumps({"experiment_id": "exp-x-20260101", "branch": "agent/exp-x"}),
        encoding="utf-8",
    )
    nested = repo / "src"
    nested.mkdir()
    assert derive_identity(nested).label == "agent/exp-x"


def test_identity_from_git_branch_and_sha(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    identity = derive_identity(repo)
    assert identity.label.startswith("main@")
    assert "branch=main" in identity.detail


def test_identity_refuses_without_pin_or_git(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(JournalError):
        derive_identity(plain)


def test_broken_pin_fails_closed(tmp_path):
    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / ".agent-pin.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(JournalError):
        derive_identity(lane)


def test_year_rotation_uses_istanbul_date(tmp_path):
    root = make_journal_root(tmp_path)
    assert journal_file(date(2027, 1, 1), root).name == "agent-journal-2027.md"
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)

    before = run_tool(
        ["--title", "late", "--body-file", "-", "--no-git", "--now", "2026-12-31T20:30:00Z"],
        env,
        cwd=repo,
        stdin="body\n",
    )
    assert before.returncode == 0, before.stderr
    after = run_tool(
        ["--title", "rollover", "--body-file", "-", "--no-git", "--now", "2026-12-31T21:30:00Z"],
        env,
        cwd=repo,
        stdin="body\n",
    )
    assert after.returncode == 0, after.stderr
    assert (root / "LLM-Depression" / "agent-journal-2026.md").is_file()
    assert (root / "LLM-Depression" / "agent-journal-2027.md").is_file()


def test_concurrent_appends_keep_every_entry(tmp_path):
    root = make_journal_root(tmp_path)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)

    count = 8
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(TOOL),
                "--title",
                f"entry-{index}",
                "--body-file",
                "-",
                "--no-git",
            ],
            cwd=str(repo),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(count)
    ]
    for process, index in zip(processes, range(count)):
        process.communicate(f"body {index}\n")
    for process in processes:
        assert process.returncode == 0

    text = (root / "LLM-Depression" / "agent-journal-2026.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert len(headings) == count
    for index in range(count):
        assert text.count(f"— entry-{index}\n") == 1


def test_lock_timeout_refuses_to_write(tmp_path):
    root = make_journal_root(tmp_path)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    target = root / "LLM-Depression" / "agent-journal-2026.md"
    target.write_text("# Agent journal — 2026\n\n", encoding="utf-8")

    fd = os.open(target, os.O_WRONLY | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = run_tool(
            [
                "--title",
                "blocked",
                "--body-file",
                "-",
                "--no-git",
                "--lock-timeout",
                "0.2",
            ],
            env,
            cwd=repo,
            stdin="body\n",
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert result.returncode == 1
    assert "could not lock" in result.stderr
    assert "blocked" not in target.read_text(encoding="utf-8")


def test_empty_title_or_body_refused(tmp_path):
    root = make_journal_root(tmp_path)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)

    assert run_tool(["--title", "  ", "--body-file", "-"], env, repo, "body\n").returncode == 1
    assert run_tool(["--title", "ok", "--body-file", "-"], env, repo, "\n").returncode == 1
    assert not (root / "LLM-Depression" / "agent-journal-2026.md").exists()


def test_missing_journal_dir_fails_closed(tmp_path):
    root = tmp_path / "Agent-Journal"
    root.mkdir()
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "x", "--body-file", "-", "--no-git"], env, repo, "body\n")
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_dry_run_touches_nothing(tmp_path):
    root = make_journal_root(tmp_path)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(
        ["--title", "planned", "--body-file", "-", "--dry-run"], env, repo, "body\n"
    )
    assert result.returncode == 0
    assert "planned" in result.stdout
    assert not (root / "LLM-Depression" / "agent-journal-2026.md").exists()
    log = subprocess.run(
        ["git", "-C", str(root), "log", "--oneline"], capture_output=True, text=True
    ).stdout
    assert "journal:" not in log


def test_commit_created_and_missing_remote_is_not_fatal(tmp_path):
    root = make_journal_root(tmp_path)
    init_git_repo(root)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "committed", "--body-file", "-"], env, repo, "body\n")
    assert result.returncode == 0
    assert "push skipped" in result.stderr
    log = subprocess.run(
        ["git", "-C", str(root), "log", "--oneline"], capture_output=True, text=True
    ).stdout
    assert "journal:" in log
    assert "committed" in log


def test_sync_without_remote_exits_nonzero(tmp_path):
    root = make_journal_root(tmp_path)
    init_git_repo(root)
    env = make_env(tmp_path, root)
    result = run_tool(["--sync"], env)
    assert result.returncode == 1
    assert "no remote" in result.stderr


def test_sync_pushes_to_local_bare_remote(tmp_path):
    root = make_journal_root(tmp_path)
    init_git_repo(root)
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", str(bare)], check=True)
    env = make_env(tmp_path, root)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "backed up", "--body-file", "-"], env, repo, "body\n")
    assert result.returncode == 0
    assert "push: pushed" in result.stdout
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    log = subprocess.run(
        ["git", "-C", str(bare), "log", "--oneline", branch],
        capture_output=True,
        text=True,
    ).stdout
    assert "backed up" in log


def test_build_entry_contains_identity_and_stamp():
    identity = Identity(label="agent/exp-x", detail="experiment_id=exp-x branch=agent/exp-x")
    entry = build_entry(
        "A title", "Body text.", identity, datetime(2026, 9, 12, 11, 30, tzinfo=timezone.utc)
    )
    assert entry.startswith("## 2026-09-12 11:30Z — agent/exp-x — A title\n")
    assert "Agent: experiment_id=exp-x branch=agent/exp-x\n" in entry
    assert entry.endswith("Body text.\n\n")


def test_journal_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_JOURNAL_ROOT", str(tmp_path / "custom"))
    assert journal_root() == (tmp_path / "custom").resolve()
