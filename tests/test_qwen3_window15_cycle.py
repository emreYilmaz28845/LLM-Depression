"""Focused tests for the incremental window15 cycle driver."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

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


KEY = "daic_audio_only_native|7|0"


def _audit_item(status: str = "waiting_training") -> dict:
    return {
        "key": KEY,
        "status": status,
        "modality": "audio_only",
        "dataset": "daic",
        "run_name": "q3w15_daic_audio_only_native_s7_f0",
        "fold": 0,
        "attempt_id": "new-attempt",
    }


def _fixture(tmp_path, monkeypatch, fold_state: str | None = None):
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"key": KEY, "status": "submitted", "attempt_id": "old-attempt"}),
                json.dumps({"key": KEY, "status": "submitted", "attempt_id": "new-attempt"}),
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
    run_root = tmp_path / "output_model"
    fold_dir = run_root / "audio_only" / "daic" / "q3w15_daic_audio_only_native_s7_f0" / "fold_0"
    if fold_state is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "status.json").write_text(json.dumps({"state": fold_state, "attempt_id": "new-attempt"}), encoding="utf-8")
        (fold_dir / "metadata.json").write_text(json.dumps({"attempt_id": "new-attempt"}), encoding="utf-8")
    monkeypatch.setattr(cycle, "LEDGER", ledger)
    monkeypatch.setattr(cycle, "EXECUTION_LEDGER", exec_ledger)
    monkeypatch.setattr(cycle, "RUN_ROOT", run_root)
    monkeypatch.setattr(cycle, "build_plan", lambda *a, **k: ({}, {"keys": [_audit_item()]}))
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: {job_id: {"state": "COMPLETED", "exit": "0:0"} for job_id in ids},
    )
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run_exp", recorder)
    return recorder, fold_dir


def test_cycle_has_no_shared_lock_usage() -> None:
    assert not hasattr(cycle, "lane_submission_lock")


def test_cycle_collect_then_status_then_validate_with_exact_state(tmp_path, monkeypatch) -> None:
    recorder, fold_dir = _fixture(tmp_path, monkeypatch, fold_state=None)

    # confirmation requires the exact local validated state; simulate the
    # official validate advancing the fold only when it runs
    def fake_run(args, *a, **k):
        recorder.calls.append(list(args))
        if "validate" in args:
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": "new-attempt"}), encoding="utf-8")
            (fold_dir / "metadata.json").write_text(json.dumps({"attempt_id": "new-attempt"}), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cycle, "_run_exp", fake_run)
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["completed_fits"] == 1
    assert result["collected"] == 1 and result["validated"] == 1
    assert result["results"][-1]["stage"] == "validated"
    # collection is followed by exactly one official status reconcile before validate
    order = [" ".join(call) for call in recorder.calls]
    assert any("collect" in call for call in order)
    assert order.index(next(call for call in order if " status " in f" {call} ")) > order.index(
        next(call for call in order if "collect" in call)
    )


def test_cycle_rc0_noop_does_not_count_validated(tmp_path, monkeypatch) -> None:
    recorder, _ = _fixture(tmp_path, monkeypatch, fold_state=None)
    # collect and validate both rc0, but no local validated state ever appears
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["collected"] == 1
    assert result["validated"] == 0
    assert result["results"][-1]["stage"] == "blocked_validate"
    assert result["results"][-1]["final_state"] is None


def test_cycle_needs_collect_policy(tmp_path, monkeypatch) -> None:
    # SYNCED_LOCALLY: collection already done, validate only
    recorder, _ = _fixture(tmp_path, monkeypatch, fold_state="SYNCED_LOCALLY")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["results"][-1]["collect_skipped"] is True
    assert not any("collect" in " ".join(call) for call in recorder.calls)

    # COMPLETED_ON_MN5 is not collection proof: collect runs
    recorder2, _ = _fixture(tmp_path, monkeypatch, fold_state="COMPLETED_ON_MN5")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["results"][-1]["collect_skipped"] is False
    assert any("collect" in " ".join(call) for call in recorder2.calls)

    # validated folds are skipped entirely
    recorder3, _ = _fixture(tmp_path, monkeypatch, fold_state="LOCALLY_VALIDATED")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["skipped_validated"] == 1 and result["completed_fits"] == 0
    assert recorder3.calls == []


def test_cycle_bounded_unit_exception_keeps_other_units(tmp_path, monkeypatch) -> None:
    second_key = "d3tec_audio_only_native|7|0"
    recorder, _ = _fixture(tmp_path, monkeypatch, fold_state=None)
    with (tmp_path / "submissions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": second_key, "status": "submitted", "attempt_id": "second"}) + "\n")
    with (tmp_path / "state.json").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "jobs": [
                        {"attempt_id": "new-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "21"},
                        {"attempt_id": "new-attempt", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "22"},
                        {"attempt_id": "second", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "31"},
                        {"attempt_id": "second", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "32"},
                    ]
                }
            )
        )
    audit = {"keys": [_audit_item(), {**_audit_item(), "key": second_key, "dataset": "d3tec", "run_name": "r2", "attempt_id": "second"}]}
    monkeypatch.setattr(cycle, "build_plan", lambda *a, **k: ({}, audit))

    def flaky_accounting(ids, runner=None):
        if "31" in ids:
            raise cycle.AdmissionError("accounting unavailable for second unit")
        return {job_id: {"state": "COMPLETED", "exit": "0:0"} for job_id in ids}

    monkeypatch.setattr(cycle, "query_top_level_accounting", flaky_accounting)
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert any(entry["key"] == second_key for entry in result["errors"])
    assert result["completed_fits"] == 1  # the first unit still processed


def test_cycle_matrix_failure_skips_head_submit(tmp_path, monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(cycle, "phase_reconcile", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(cycle, "phase_collect_fits", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(cycle, "phase_matrix", lambda *a, **k: {"status": "remote build failed"})
    monkeypatch.setattr(
        cycle, "phase_head_submit", lambda *a, matrix_ok=True, **k: calls.append(matrix_ok) or {"status": "skipped"})
    monkeypatch.setattr(cycle, "phase_head_collect", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(cycle, "phase_coverage", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(cycle, "CYCLE_LOG", tmp_path / "cycle_log.jsonl")
    args = types.SimpleNamespace(max_fits=2, max_head_fits=2, dry_run=False)
    assert cycle.run_cycle(args) == 0
    assert calls == [False]

    # direct phase call refuses without running the guard
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run", recorder)
    result = cycle.phase_head_submit(4, False, matrix_ok=False)
    assert result["status"].startswith("skipped")
    assert recorder.calls == []


def test_cycle_head_collect_confirms_exact_state(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "head_submissions.jsonl"
    registry.write_text(
        json.dumps({"registry_key": "a|7|0", "attempt_id": "head-1", "extract_job_id": "1", "classifier_job_id": "2"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cycle, "head_registry_entries", lambda path=None: [json.loads(line) for line in registry.read_text().splitlines()])
    monkeypatch.setattr(cycle, "EVIDENCE", tmp_path)
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: {job_id: {"state": "COMPLETED", "exit": "0:0"} for job_id in ids},
    )
    mirror = tmp_path / "head_attempts" / "head-1"
    recorder = _Recorder()

    def fake_run(cmd, timeout=None):
        recorder.calls.append(list(cmd))
        if "collect" in cmd:
            mirror.mkdir(parents=True, exist_ok=True)
            (mirror / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": "head-1"}), encoding="utf-8")
            (mirror / "metadata.json").write_text(json.dumps({"attempt_id": "head-1"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cycle, "_run", fake_run)
    result = cycle.phase_head_collect(max_head_fits=4, dry_run=False)
    assert result["collected"] == 1 and result["validated"] == 1
    assert result["results"][-1]["stage"] == "validated"


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
    assert len(records) == 1 and "at_utc" in records[0]


def test_cycle_stale_or_blank_identity_is_not_skipped(tmp_path, monkeypatch) -> None:
    # LOCALLY_VALIDATED state with a blank status attempt id: stale evidence,
    # never skipped; the validation confirmation must also refuse it
    recorder, fold_dir = _fixture(tmp_path, monkeypatch, fold_state="LOCALLY_VALIDATED")
    (fold_dir / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": ""}), encoding="utf-8")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["skipped_validated"] == 0
    assert result["completed_fits"] == 1
    assert result["validated"] == 0
    assert result["results"][-1]["stage"] == "blocked_validate"

    # metadata missing: status alone is not proof
    recorder2, fold_dir2 = _fixture(tmp_path, monkeypatch, fold_state="LOCALLY_VALIDATED")
    (fold_dir2 / "metadata.json").unlink()
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["skipped_validated"] == 0
    assert result["completed_fits"] == 1
    assert result["validated"] == 0

    # SYNCED_LOCALLY with broken identity must NOT skip collection
    recorder3, fold_dir3 = _fixture(tmp_path, monkeypatch, fold_state="SYNCED_LOCALLY")
    (fold_dir3 / "metadata.json").write_text(json.dumps({"attempt_id": "old-attempt"}), encoding="utf-8")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["results"][-1]["collect_skipped"] is False

    # SYNCED_LOCALLY with exact identity may skip collection
    recorder4, _ = _fixture(tmp_path, monkeypatch, fold_state="SYNCED_LOCALLY")
    result = cycle.phase_collect_fits(max_fits=4, dry_run=False)
    assert result["results"][-1]["collect_skipped"] is True


def test_cycle_head_stale_identity_is_not_confirmed(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "head_submissions.jsonl"
    registry.write_text(
        json.dumps({"registry_key": "a|7|0", "attempt_id": "head-1", "extract_job_id": "1", "classifier_job_id": "2"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cycle, "head_registry_entries", lambda path=None: [json.loads(line) for line in registry.read_text().splitlines()])
    monkeypatch.setattr(cycle, "EVIDENCE", tmp_path)
    monkeypatch.setattr(
        cycle,
        "query_top_level_accounting",
        lambda ids, runner=None: {job_id: {"state": "COMPLETED", "exit": "0:0"} for job_id in ids},
    )
    mirror = tmp_path / "head_attempts" / "head-1"
    mirror.mkdir(parents=True)
    # DONE state but the metadata belongs to a different/stale attempt
    (mirror / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": "head-1"}), encoding="utf-8")
    (mirror / "metadata.json").write_text(json.dumps({"attempt_id": "old-head"}), encoding="utf-8")
    recorder = _Recorder()
    monkeypatch.setattr(cycle, "_run", recorder)
    result = cycle.phase_head_collect(max_head_fits=4, dry_run=False)
    # stale mirror is not skipped; the fake collect/validate never fix identity
    assert result["pending_head_attempts"] == 1
    assert result["validated"] == 0
    assert result["results"][-1]["stage"] == "blocked_validate"


def test_cycle_lock_prevents_overlap_without_mutation(tmp_path, monkeypatch) -> None:
    import fcntl

    lock_path = tmp_path / "cycle.lock"
    monkeypatch.setattr(cycle, "CYCLE_LOCK", lock_path)
    monkeypatch.setattr(cycle, "CYCLE_LOG", tmp_path / "cycle_log.jsonl")
    called: list[str] = []
    for name in (
        "phase_reconcile",
        "phase_collect_fits",
        "phase_matrix",
        "phase_head_submit",
        "phase_head_collect",
        "phase_coverage",
    ):
        monkeypatch.setattr(cycle, name, lambda *a, _n=name, **k: called.append(_n) or {"status": "ok"})
    args = types.SimpleNamespace(max_fits=2, max_head_fits=2, dry_run=False)

    hold = lock_path.open("w")
    fcntl.flock(hold, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert cycle.run_cycle(args) == 0
        assert called == []  # busy cycle performs no phase and no mutation
        assert not (tmp_path / "cycle_log.jsonl").exists()
    finally:
        fcntl.flock(hold, fcntl.LOCK_UN)
        hold.close()

    # the released lock lets the next bounded cycle run normally
    assert cycle.run_cycle(args) == 0
    assert called  # phases ran
