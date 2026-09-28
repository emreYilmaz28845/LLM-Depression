"""Tests for the Qwen3 DAIC label-vocabulary resume ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.qwen3_daic_label_vocab_state import (
    LedgerError,
    main,
    sha256_file,
)


def _init(tmp_path: Path) -> Path:
    ledger = tmp_path / "state.json"
    code = main([
        "--ledger", str(ledger), "init",
        "--baseline-sha", "10fc786bf99cd0e7c7b1c0571339a8a7944ea4d6",
        "--experiment-id", "feat-qwen3-daic-label-vocab-20260928",
        "--slug", "qwen3-daic-label-vocab",
        "--branch", "agent/feat-qwen3-daic-label-vocab",
        "--worktree", "/tmp/does-not-need-to-exist",
        "--group-definition", "experiments/definitions/qwen3-daic-label-vocab-20260928.yaml",
        "--campaign", "qwen3_daic_label_vocab_v1",
        "--smoke-campaign", "qwen3_daic_label_vocab_smoke_v1",
        "--group-id", "qwen3-daic-label-vocab-20260928",
    ])
    assert code == 0
    return ledger


def test_init_writes_all_phases_and_refuses_to_overwrite(tmp_path: Path) -> None:
    ledger = _init(tmp_path)
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert record["schema_version"] == "audiollm.qwen3_daic_label_vocab.state.v1"
    assert [entry["status"] for entry in record["phases"]] == ["pending"] * len(record["phases"])
    assert record["grant"]["merges_allowed"] is False
    assert record["baseline"]["origin_main_sha"].startswith("10fc786")

    before = ledger.read_text(encoding="utf-8")
    assert main([
        "--ledger", str(ledger), "init",
        "--baseline-sha", "x", "--experiment-id", "x", "--slug", "x",
        "--branch", "x", "--worktree", "x",
        "--group-definition", "x", "--campaign", "x", "--smoke-campaign", "x", "--group-id", "x",
    ]) == 1
    assert ledger.read_text(encoding="utf-8") == before


def test_phase_status_and_job_events_are_recorded_once(tmp_path: Path) -> None:
    ledger = _init(tmp_path)
    assert main(["--ledger", str(ledger), "phase", "configs", "in_progress", "--note", "generator"]) == 0
    assert main([
        "--ledger", str(ledger), "job", "--attempt-id", "a1", "--job-key", "train",
        "--slurm-job-id", "44394029", "--state", "SUBMITTED",
    ]) == 0
    assert main([
        "--ledger", str(ledger), "job", "--attempt-id", "a1", "--job-key", "train",
        "--slurm-job-id", "44394029", "--state", "SUBMITTED",
    ]) == 0
    assert main([
        "--ledger", str(ledger), "job", "--attempt-id", "a1", "--job-key", "train",
        "--slurm-job-id", "44394029", "--state", "COMPLETED", "--detail", "0:0",
    ]) == 0

    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert [(event["state"]) for event in record["jobs"]] == ["SUBMITTED", "COMPLETED"]
    assert next(entry for entry in record["phases"] if entry["id"] == "configs")["status"] == "in_progress"
    assert [item["event"] for item in record["history"]].count("job") == 2


def test_evidence_records_hash_and_skips_identical_repeats(tmp_path: Path) -> None:
    ledger = _init(tmp_path)
    artifact = tmp_path / "report.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    expected = sha256_file(artifact)
    assert main([
        "--ledger", str(ledger), "evidence", "--attempt-id", "a1", "--kind", "report",
        "--path", str(artifact),
    ]) == 0
    assert main([
        "--ledger", str(ledger), "evidence", "--attempt-id", "a1", "--kind", "report",
        "--path", str(artifact),
    ]) == 0
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(record["evidence"]) == 1
    assert record["evidence"][0]["sha256"] == expected


def test_unknown_phase_and_status_are_refused(tmp_path: Path) -> None:
    ledger = _init(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    assert main(["--ledger", str(ledger), "phase", "not-a-phase", "complete"]) == 1
    assert main(["--ledger", str(ledger), "phase", "configs", "not-a-status"]) == 1
    assert ledger.read_text(encoding="utf-8") == before


def test_writes_are_atomic_and_leave_no_temporary_siblings(tmp_path: Path) -> None:
    ledger = _init(tmp_path)
    for status in ("in_progress", "complete"):
        assert main(["--ledger", str(ledger), "phase", "worktree", status]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert record["phases"][1]["id"] == "worktree"
    assert record["phases"][1]["status"] == "complete"


def test_hard_stop_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = _init(tmp_path)
    assert main([
        "--ledger", str(ledger), "hard-stop", "--reason", "token gate failed",
        "--cell", "daic_text_only_qwen38_27b_en", "--decision", "ask the user about the EN label",
    ]) == 0
    assert main(["--ledger", str(ledger), "decision", "keep EN as legacy_english_labels"]) == 0
    assert main(["--ledger", str(ledger), "show"]) == 0
    output = capsys.readouterr().out
    assert "HARD STOP: token gate failed" in output
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert record["decisions"][0]["text"] == "keep EN as legacy_english_labels"
