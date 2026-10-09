"""Focused tests for the incremental window15 cycle driver."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import qwen3_window15_cycle as cycle  # noqa: E402


class _Recorder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, timeout=None):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_cycle_has_no_shared_lock_usage() -> None:
    # the driver must never hold the lane submission lock; the guard takes it itself
    assert not hasattr(cycle, "lane_submission_lock")


def test_cycle_fit_phase_uses_latest_attempt_and_exact_0_0(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"key": "k|7|0", "status": "submitted", "attempt_id": "old-attempt"}),
                json.dumps({"key": "k|7|0", "status": "submitted", "attempt_id": "new-attempt"}),
                json.dumps({"key": "head:x|7|0", "status": "held", "attempt_id": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "old-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "11"},
                    {"attempt_id": "old-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "12"},
                    {"attempt_id": "new-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "21"},
                    {"attempt_id": "new-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "22"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cycle, "LEDGER", ledger)
    monkeypatch.setattr(cycle, "EXECUTION_LEDGER", exec_ledger)
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: {job_id: {"state": "COMPLETED", "exit": "0:0"} for job_id in ids},
    )
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run_exp", recorder)

    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["completed_fits"] == 1
    assert result["collected"] == 1 and result["validated"] == 1
    assert recorder.calls[0][:4] == ["collect", "feat-qwen3-window15-20261008", "--attempt-id", "new-attempt"]
    assert any("validate" in call for call in recorder.calls)

    # a non-0:0 pair is not collected
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: {job_id: {"state": "COMPLETED", "exit": "1:0"} for job_id in ids},
    )
    result = cycle.phase_collect_fits(max_fits=4, dry_run=True)
    assert result["completed_fits"] == 0


def test_cycle_matrix_rebuild_passes_pre_filtered_map(tmp_path, monkeypatch) -> None:
    audit = {"keys_total": 126, "keys": [{"key": "a|7|0", "status": "eligible"}], "status_counts": {}}
    parent_map = {
        "entries": [
            {"route_id": "a", "parent_training_seed": 7, "fold": 0, "config": "c", "fold_dir": "d", "parent_attempt_id": "x"},
        ]
    }
    monkeypatch.setattr(cycle, "build_plan", lambda *a, **k: (parent_map, audit))
    recorder = _Recorder()

    def fake_run(cmd, timeout=None):
        recorder.calls.append(list(cmd))
        if "scp" in cmd and "window15_heads_matrix.json" in " ".join(cmd):
            target = Path(cmd[-1])
            target.write_text(json.dumps({"summary": {"resolved": 1}}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cycle, "_run", fake_run)
    monkeypatch.setattr(cycle, "_latest_head_deployment", lambda: "feat-qwen3-window15-20261008-xyz")
    result = cycle.phase_matrix(dry_run=False)
    assert result["status"] == "ok"
    remote = next(" ".join(call) for call in recorder.calls if "ssh" in call)
    assert "--build-matrix" in remote and "--parent-map-pre-filtered" in remote


def test_cycle_head_submit_skips_without_resolved_parents(tmp_path, monkeypatch) -> None:
    matrix = tmp_path / "window15_heads_matrix.json"
    matrix.write_text(json.dumps({"summary": {"resolved": 0}}), encoding="utf-8")
    monkeypatch.setattr(cycle, "OUT_DIR", tmp_path)
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run", recorder)
    result = cycle.phase_head_submit(max_head_fits=4, dry_run=False)
    assert result["status"] == "no resolved parents"
    assert recorder.calls == []


def test_cycle_head_collect_validates_only_bound_0_0_attempts(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "head_submissions.jsonl"
    registry.write_text(
        "\n".join(
            [
                json.dumps({"registry_key": "a|7|0", "attempt_id": "head-1", "extract_job_id": "1", "classifier_job_id": "2"}),
                json.dumps({"registry_key": "b|7|0", "attempt_id": "head-2", "extract_job_id": "3", "classifier_job_id": "4"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cycle, "head_registry_entries", lambda path=None: [json.loads(line) for line in registry.read_text().splitlines()])
    monkeypatch.setattr(cycle, "EVIDENCE", tmp_path)
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: (
            {"1": {"state": "COMPLETED", "exit": "0:0"}, "2": {"state": "COMPLETED", "exit": "0:0"}}
            if set(ids) == {"1", "2"}
            else {"3": {"state": "RUNNING", "exit": "0:0"}, "4": {"state": "PENDING", "exit": "0:0"}}
        ),
    )
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run", recorder)
    result = cycle.phase_head_collect(max_head_fits=4, dry_run=False)
    assert result["pending_head_attempts"] == 1
    assert result["collected"] == 1 and result["validated"] == 1
    joined = [" ".join(call) for call in recorder.calls]
    assert any("head-1" in call and "collect" in call for call in joined)
    assert not any("head-2" in call for call in joined)


def test_cycle_run_logs_bounded_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cycle, "CYCLE_LOG", tmp_path / "cycle_log.jsonl")
    for name in (
        "phase_reconcile",
        "phase_collect_fits",
        "phase_matrix",
        "phase_head_submit",
        "phase_head_collect",
        "phase_coverage",
    ):
        monkeypatch.setattr(cycle, name, lambda *a, **k: {"status": "ok"})
    args = types.SimpleNamespace(max_fits=2, max_head_fits=2, dry_run=False)
    assert cycle.run_cycle(args) == 0
    records = [json.loads(line) for line in (tmp_path / "cycle_log.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert "at_utc" in records[0] and records[0]["reconcile"]["status"] == "ok"
