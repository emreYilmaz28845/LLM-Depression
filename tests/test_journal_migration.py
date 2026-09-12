from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from tools.journal_append import Identity, append_entry, build_entry
from tools.migrate_agent_journal import (
    MigrationError,
    demote_headings,
    execute,
    load_sources,
    plan,
    source_date,
    verify,
)


def make_root(tmp_path):
    root = tmp_path / "Agent-Journal"
    (root / "LLM-Depression").mkdir(parents=True)
    return root


def write_sources(tmp_path, files):
    source = tmp_path / "agent-journal"
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text, encoding="utf-8")
    return source


def test_demote_headings_skips_fenced_code():
    text = (
        "# 2026-01-02\n"
        "\n"
        "## Entry\n"
        "\n"
        "```bash\n"
        "# not a heading\n"
        "echo hi\n"
        "```\n"
        "\n"
        "#### deep\n"
        "###### six\n"
    )
    out = demote_headings(text)
    assert out.startswith("## 2026-01-02\n")
    assert "### Entry\n" in out
    assert "```bash\n# not a heading\necho hi\n```\n" in out
    assert "##### deep\n" in out
    assert "###### six\n" in out


def test_source_date_rejects_bad_names(tmp_path):
    bad = tmp_path / "notes.md"
    bad.write_text("x\n", encoding="utf-8")
    with pytest.raises(MigrationError):
        source_date(bad)


def test_plan_execute_verify_roundtrip(tmp_path):
    source = write_sources(
        tmp_path,
        {
            "2026-01-02.md": "# 2026-01-02\n\n## First\n\nbody one\n",
            "2026-01-03.md": "# 2026-01-03\n\n## Second\n\nbody two\n",
        },
    )
    root = make_root(tmp_path)
    planned = plan(source, root)
    assert len(planned) == 1
    target, content = planned[0]
    assert target == root / "LLM-Depression" / "agent-journal-2026.md"
    assert content.startswith("# Agent journal — 2026\n\n## 2026-01-02\n")
    assert "## 2026-01-03\n" in content
    assert not target.exists()

    execute(source, root)
    assert target.is_file()
    messages = verify(source, root)
    assert "2/2 sources preserved" in messages[0]


def test_execute_refuses_nonempty_target(tmp_path):
    source = write_sources(
        tmp_path, {"2026-01-02.md": "# 2026-01-02\n\n## First\n\nbody\n"}
    )
    root = make_root(tmp_path)
    execute(source, root)
    with pytest.raises(MigrationError):
        execute(source, root)
    execute(source, root, force=True)


def test_verify_detects_tampering_inside_migrated_content(tmp_path):
    source = write_sources(
        tmp_path, {"2026-01-02.md": "# 2026-01-02\n\n## First\n\nbody\n"}
    )
    root = make_root(tmp_path)
    execute(source, root)
    target = root / "LLM-Depression" / "agent-journal-2026.md"
    target.write_text(
        target.read_text(encoding="utf-8").replace("body\n", "tampered\n"),
        encoding="utf-8",
    )
    with pytest.raises(MigrationError):
        verify(source, root)


def test_verify_accepts_entries_appended_after_migration(tmp_path):
    source = write_sources(
        tmp_path, {"2026-01-02.md": "# 2026-01-02\n\n## First\n\nbody\n"}
    )
    root = make_root(tmp_path)
    execute(source, root)
    target = root / "LLM-Depression" / "agent-journal-2026.md"

    identity = Identity(label="agent/exp-x", detail="experiment_id=exp-x branch=agent/exp-x")
    entry = build_entry(
        "Later entry", "Body.", identity, datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc)
    )
    append_entry(target, entry, lock_timeout=5.0, header="# Agent journal — 2026\n\n")

    messages = verify(source, root)
    assert "1/1 sources preserved" in messages[0]
    assert "appended afterwards" in messages[0]


def test_missing_source_and_empty_source_fail_closed(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(MigrationError):
        plan(tmp_path / "nope", root)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MigrationError):
        plan(empty, root)


def test_empty_source_file_fails_closed(tmp_path):
    source = write_sources(tmp_path, {"2026-01-02.md": "\n"})
    root = make_root(tmp_path)
    with pytest.raises(MigrationError):
        load_sources(source)


def test_sources_split_per_year(tmp_path):
    source = write_sources(
        tmp_path,
        {
            "2025-12-31.md": "# 2025-12-31\n\n## Old\n\nbody\n",
            "2026-01-01.md": "# 2026-01-01\n\n## New\n\nbody\n",
        },
    )
    root = make_root(tmp_path)
    targets = [target.name for target, _ in plan(source, root)]
    assert targets == ["agent-journal-2025.md", "agent-journal-2026.md"]
    assert date(2026, 1, 1).year == 2026
