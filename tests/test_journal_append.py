from __future__ import annotations

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
    legacy_journal_file,
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


def make_fake_sb(tmp_path):
    """A stand-in `sb` that records its argv, stdin, cwd and env."""
    record = tmp_path / "sb-calls.jsonl"
    script = tmp_path / "fake_sb.py"
    script.write_text(
        "import json, os, sys\n"
        f"rec = {str(record)!r}\n"
        "data = {'argv': sys.argv[1:], 'stdin': sys.stdin.read() if '-' in sys.argv else '',\n"
        "        'cwd': os.getcwd(), 'agent': os.environ.get('SECONDBRAIN_AGENT'),\n"
        "        'root': os.environ.get('SECONDBRAIN_ROOT')}\n"
        "open(rec, 'a').write(json.dumps(data) + '\\n')\n"
        "print('logged j-test')\n",
        encoding="utf-8",
    )
    return f"{sys.executable} {script}", record


def calls(record):
    return [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []


def sb_env(tmp_path, agent="codex"):
    root = tmp_path / "SecondBrain"
    env = make_env(tmp_path, root)
    command, record = make_fake_sb(tmp_path)
    env["SECONDBRAIN_SB"] = command
    if agent:
        env["SECONDBRAIN_AGENT"] = agent
    else:
        env.pop("SECONDBRAIN_AGENT", None)
    return env, record, root


def test_journal_file_is_monthly_in_secondbrain(tmp_path):
    root = tmp_path / "SecondBrain"
    assert journal_file(date(2026, 10, 12), root) == (
        root / "20-Projects" / "LLM-Depression" / "journal" / "LLM-Depression-2026-10.md"
    )
    assert legacy_journal_file(date(2026, 1, 1), tmp_path).name == "agent-journal-2026.md"


def test_entry_is_forwarded_to_sb_log(tmp_path):
    env, record, root = sb_env(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "  A title ", "--body-file", "-"], env, repo, "Body text.\n")
    assert result.returncode == 0, result.stderr
    assert "logged j-test" in result.stdout
    (call,) = calls(record)
    assert call["argv"] == ["log", "--project", "LLM-Depression", "--title", "A title", "--body-file", "-"]
    assert call["stdin"] == "Body text.\n"
    assert call["agent"] == "codex"
    assert call["root"] == str(root.resolve())
    assert pathlib.Path(call["cwd"]).resolve() == repo.resolve()


def test_missing_writer_identity_is_refused(tmp_path):
    env, record, _ = sb_env(tmp_path, agent=None)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "x", "--body-file", "-"], env, repo, "body\n")
    assert result.returncode == 1
    assert "SECONDBRAIN_AGENT" in result.stderr
    assert calls(record) == []


def test_empty_title_or_body_refused(tmp_path):
    env, record, _ = sb_env(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    assert run_tool(["--title", "  ", "--body-file", "-"], env, repo, "body\n").returncode == 1
    assert run_tool(["--title", "ok", "--body-file", "-"], env, repo, "\n").returncode == 1
    assert calls(record) == []


def test_dry_run_touches_nothing(tmp_path):
    env, record, _ = sb_env(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "planned", "--body-file", "-", "--dry-run"], env, repo, "body\n")
    assert result.returncode == 0
    assert "planned" in result.stdout and "LLM-Depression-" in result.stdout
    assert calls(record) == []


def test_sync_forwards_to_sb_sync(tmp_path):
    env, record, _ = sb_env(tmp_path)
    result = run_tool(["--sync"], env)
    assert result.returncode == 0
    assert calls(record)[0]["argv"] == ["sync"]


def test_missing_sb_command_fails_clearly(tmp_path):
    env, _, _ = sb_env(tmp_path)
    env["SECONDBRAIN_SB"] = str(tmp_path / "does-not-exist")
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    result = run_tool(["--title", "x", "--body-file", "-"], env, repo, "body\n")
    assert result.returncode == 1
    assert "`sb` command" in result.stderr


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
