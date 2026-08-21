from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.monitor import (
    MonitorError,
    SchedulerClient,
    check_resume_no_duplicate_submission,
    classify_failure,
    parse_sacct,
    parse_squeue,
    plan_retry,
    reconcile_job,
    validate_lifecycle_advancement,
)


def test_parse_sacct_folds_steps_and_keeps_top_level():
    text = (
        "101|COMPLETED|0:0|00:06:10|as02r5b05\n"
        "101.batch|COMPLETED|0:0|00:06:09|\n"
        "102|CANCELLED by 100|0:15|00:14:28|as01r3b25\n"
        "103|FAILED|1:0|00:40:39|as01r1b30\n"
        "104|RUNNING|0:0|00:01:02|as02r5b16\n"
    )
    jobs = parse_sacct(text)
    assert set(jobs) == {"101", "102", "103", "104"}
    assert jobs["101"]["State"] == "COMPLETED" and jobs["101"]["ExitCode"] == "0:0"
    assert jobs["102"]["ExitCode"] == "0:15"
    assert jobs["104"]["State"] == "RUNNING"


def test_parse_squeue():
    jobs = parse_squeue("105 PENDING 0:00 llm-depression-train\n106 RUNNING 00:03:11 llm-depression-eval\n")
    assert jobs["105"]["State"] == "PENDING"
    assert jobs["106"]["State"] == "RUNNING"
    assert parse_squeue("") == {}


class FakeScheduler(SchedulerClient):
    def __init__(self, queue_text="", sacct_text=""):
        super().__init__(host="ozu647717@alogin2.bsc.es", runner=self._dispatch)
        self.queue_text = queue_text
        self.sacct_text = sacct_text
        self.calls: list[list[str]] = []

    def _dispatch(self, argv):
        self.calls.append(argv)
        script = argv[-1]
        if script.startswith("squeue"):
            return subprocess.CompletedProcess(argv, 0, self.queue_text, "")
        if script.startswith("sacct"):
            return subprocess.CompletedProcess(argv, 0, self.sacct_text, "")
        return subprocess.CompletedProcess(argv, 255, "", "unexpected")


def test_scheduler_client_queries_remote_host_via_ssh():
    sched = FakeScheduler(
        queue_text="107 RUNNING 00:01:00 train\n",
        sacct_text="106|COMPLETED|0:0|00:06:55|as02r5b16\n",
    )
    queue = sched.squeue(["107"])
    acct = sched.sacct(["106"])
    assert queue["107"]["State"] == "RUNNING"
    assert acct["106"]["ExitCode"] == "0:0"
    for call in sched.calls:
        assert call[0] == "ssh"
        assert "ozu647717@alogin2.bsc.es" in call
        assert not any("transfer1" in str(part) for part in call)


def test_reconcile_requires_completed_and_zero_exit():
    record = {"slurm_job_id": "101", "job_key": "train", "attempt_id": "a1"}
    rec = reconcile_job(record, {}, parse_sacct("101|COMPLETED|0:0|00:06:10|n1"), artifacts_ok=True)
    assert rec.terminal_success is True
    # nonzero exit under COMPLETED is a problem
    rec2 = reconcile_job(record, {}, parse_sacct("101|COMPLETED|1:0|00:06:10|n1"), artifacts_ok=True)
    assert rec2.terminal_success is False
    assert rec2.problems and "nonzero ExitCode" in rec2.problems[0]
    # completed but artifacts missing is a problem
    rec3 = reconcile_job(record, {}, parse_sacct("101|COMPLETED|0:0|00:06:10|n1"), artifacts_ok=False)
    assert rec3.terminal_success is False
    assert any("artifacts missing" in p for p in rec3.problems)
    # still queued is not terminal success even if accounting shows COMPLETED stale row
    rec4 = reconcile_job(record, parse_squeue("101 RUNNING 00:01 x"),
                         parse_sacct("101|COMPLETED|0:0|00:06:10|n1"), artifacts_ok=True)
    assert rec4.terminal_success is False


def test_failure_classification_matrix():
    assert classify_failure("NODE_FAIL", "0:9") == "transient_infrastructure"
    assert classify_failure("TIMEOUT", "0:9") == "transient_infrastructure"
    assert classify_failure("OUT_OF_MEMORY", "0:9") == "transient_infrastructure"
    assert classify_failure("PREEMPTED", "0:9") == "transient_infrastructure"
    assert classify_failure("FAILED", "1:0") == "deterministic_code_config"
    assert classify_failure("CANCELLED by 999", "0:15") == "cancelled_dependency"
    assert classify_failure("", "1:0") == "deterministic_code_config"
    assert classify_failure("SUSPENDED", "0:0") == "unknown"


def test_cancelled_dependency_detected_from_dependency_state():
    record = {
        "slurm_job_id": "200", "job_key": "best_eval",
        "dependency_job_ids": ["199"],
    }
    acct = parse_sacct("199|CANCELLED by 200|0:15|00:14:28|n1\n200|CANCELLED by 199|0:15|00:00:01|n1\n")
    rec = reconcile_job(record, {}, acct, artifacts_ok=None)
    assert rec.classification == "cancelled_dependency"


def test_retry_budget_one_unchanged_transient_only():
    new_id = iter(["attempt-retry-1", "attempt-retry-2"])
    failed = reconcile_job({"slurm_job_id": "55", "job_key": "train"}, {},
                           parse_sacct("55|NODE_FAIL|0:9|00:00:10|n1"), artifacts_ok=None)
    plan = plan_retry(failed, [], lambda: next(new_id))
    assert plan["allowed"] is True and plan["kind"] == "unchanged_transient_retry"
    assert plan["resubmission_of_job_id"] == "55"
    with pytest.raises(MonitorError, match="retry budget exhausted"):
        plan_retry(failed, [{"resubmission_of_job_id": "55"}], lambda: next(new_id))
    det = reconcile_job({"slurm_job_id": "56", "job_key": "train"}, {},
                        parse_sacct("56|FAILED|1:0|00:40:39|n1"), artifacts_ok=None)
    plan_det = plan_retry(det, [], lambda: next(new_id))
    assert plan_det["kind"] == "code_fix_new_deployment_new_attempt"
    unk = reconcile_job({"slurm_job_id": "57", "job_key": "train"}, {},
                        parse_sacct("57|SUSPENDED|0:0|00:00:01|n1"), artifacts_ok=None)
    plan_unk = plan_retry(unk, [], lambda: next(new_id))
    assert plan_unk["allowed"] is False


def test_resume_refuses_duplicate_submission():
    events = [
        {"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": "101"},
    ]
    latest = check_resume_no_duplicate_submission(events, "train")
    assert latest is not None and latest["slurm_job_id"] == "101"
    assert check_resume_no_duplicate_submission([], "train") is None


def test_invalid_lifecycle_advancement_refused():
    validate_lifecycle_advancement("SUBMITTED", "RUNNING")
    validate_lifecycle_advancement("RUNNING", "COMPLETED_ON_MN5")
    with pytest.raises(MonitorError, match="invalid lifecycle advancement"):
        validate_lifecycle_advancement("RUNNING", "REPORTABLE")
    with pytest.raises(MonitorError, match="invalid lifecycle advancement"):
        validate_lifecycle_advancement("PLANNED", "REPORTABLE")


def test_scheduler_failure_returns_nonzero_signal():
    class Broken(SchedulerClient):
        def __init__(self):
            super().__init__(host="h", runner=lambda argv: subprocess.CompletedProcess(argv, 1, "", "ssh: connect failed"))

    with pytest.raises(MonitorError, match="squeue failed"):
        Broken().squeue(["1"])
