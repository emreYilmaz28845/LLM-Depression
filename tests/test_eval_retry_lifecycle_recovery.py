"""Failure-before-retry lifecycle recovery regression tests.

A fold can reach FAILED when ``exp.py status`` mirrors an original failed
evaluation before a later linked retry completes. The intended protocol is an
append-only same-attempt retry: a SUBMITTED event linked through
``resubmission_of_job_id`` plus a later cleanly COMPLETED terminal event. The
verified recovery gate may then move FAILED to COMPLETED_ON_MN5, and only after
that do the normal stepwise gates apply. Unlinked, wrong-attempt, wrong-parent,
failed-latest and out-of-order retries must stay blocked, and the generic
lifecycle transition table must keep refusing FAILED -> COMPLETED_ON_MN5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_tracking import lifecycle  # noqa: E402
from src.experiment_tracking.lifecycle import new_job_event  # noqa: E402
from src.experiment_tracking.validate import (  # noqa: E402
    finish_gates,
    recover_failed_attempt_from_verified_retry,
)
from tests.test_parallel_workflow_validate import KW, _build_attempt  # noqa: E402

ATTEMPT = "20260821T000000Z-run1-abcdef01-12345678"


def _event(**kwargs):
    return new_job_event(attempt_id=ATTEMPT, fold=0, **kwargs)


def _fail_fold(fold: Path, *, linked: bool = True, wrong_attempt: bool = False,
               failed_latest: bool = False, reorder: bool = False,
               dependencies: tuple[str, ...] = ("101",), no_failure: bool = False,
               train_retry: bool = False) -> Path:
    train = _event(job_key="train", job_type="train", event_type="COMPLETED",
                   slurm_job_id="101", status="COMPLETED")
    train["exit_code"] = "0:0"
    eval_failed = _event(job_key="best_eval", job_type="evaluation", event_type="FAILED",
                         slurm_job_id="102", status="FAILED")
    eval_failed["exit_code"] = "1:0"
    eval_completed = _event(job_key="best_eval", job_type="evaluation", event_type="COMPLETED",
                            slurm_job_id="102", status="COMPLETED")
    eval_completed["exit_code"] = "0:0"
    retry_attempt = "20260821T000000Z-other-attempt-deadbeef-99999999" if wrong_attempt else ATTEMPT
    submitted = new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="PENDING",
        resubmission_of_job_id=("102" if linked else None),
        dependency_job_ids=list(dependencies),
    )
    if failed_latest:
        retry_terminal = new_job_event(
            job_key="best_eval", job_type="evaluation", event_type="FAILED",
            attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="FAILED",
            dependency_job_ids=list(dependencies),
        )
        retry_terminal["exit_code"] = "1:0"
    else:
        retry_terminal = new_job_event(
            job_key="best_eval", job_type="evaluation", event_type="COMPLETED",
            attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="COMPLETED",
            dependency_job_ids=list(dependencies),
        )
        retry_terminal["exit_code"] = "0:0"
    if train_retry:
        train_failed = _event(job_key="train", job_type="train", event_type="FAILED",
                              slurm_job_id="104", status="FAILED")
        train_failed["exit_code"] = "1:0"
        train_retry_submitted = _event(job_key="train", job_type="train", event_type="SUBMITTED",
                                       slurm_job_id="105", status="PENDING",
                                       resubmission_of_job_id="104")
        train_retry_completed = _event(job_key="train", job_type="train", event_type="COMPLETED",
                                       slurm_job_id="105", status="COMPLETED")
        train_retry_completed["exit_code"] = "0:0"
        events = [train_failed, train_retry_submitted, train_retry_completed, eval_completed]
    elif no_failure:
        events = [train, submitted, retry_terminal]
    elif reorder:
        events = [train, submitted, retry_terminal, eval_failed]
    else:
        events = [train, eval_failed, submitted, retry_terminal]
    (fold / "jobs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    status["state"] = "FAILED"
    status["history"].append({"from": "RUNNING", "to": "FAILED", "at_utc": status["updated_at_utc"]})
    (fold / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return fold


def test_failed_then_linked_retry_recovers_and_finishes(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path))
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is True
    assert recovery["state"] == "COMPLETED_ON_MN5"
    assert recovery["retry_jobs"] == [
        {"job_key": "best_eval", "failed_job_id": "102", "retry_job_id": "103"}
    ]
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "COMPLETED_ON_MN5"
    assert status["history"][-1]["to"] == "COMPLETED_ON_MN5"
    assert status["history"][-1]["verified_recovery"]["verified_retry_jobs"][0]["retry_job_id"] == "103"
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is True
    assert finish["state"] == "REPORTABLE"


def test_finish_gates_recovers_directly_from_failed(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path))
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is True
    assert finish["state"] == "REPORTABLE"


def test_train_leg_linked_retry_recovers(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), train_retry=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is True
    assert recovery["retry_jobs"][0]["job_key"] == "train"
    assert recovery["retry_jobs"][0]["retry_job_id"] == "105"


def test_unlinked_retry_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), linked=False)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "no linked submitted retry" in recovery["reason"]
    assert finish_gates(fold, **KW)["ok"] is False


def test_wrong_attempt_retry_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), wrong_attempt=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "identity does not match" in recovery["reason"]


def test_failed_latest_retry_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), failed_latest=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "latest terminal event" in recovery["reason"]


def test_failure_after_retry_success_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), reorder=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "latest terminal event" in recovery["reason"]


def test_wrong_parent_dependency_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=("999",))
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "dependency link" in recovery["reason"]


def test_failed_state_without_failed_event_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), no_failure=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "no linked retry evidence" in recovery["reason"]


def test_late_failure_after_success_blocks_finish(tmp_path: Path) -> None:
    fold = _build_attempt(tmp_path)
    late_failure = _event(job_key="best_eval", job_type="evaluation", event_type="FAILED",
                          slurm_job_id="102", status="FAILED")
    late_failure["exit_code"] = "1:0"
    with (fold / "jobs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(late_failure) + "\n")
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is False
    assert "latest COMPLETED 0:0" in finish["next_action"]


def test_generic_transition_still_refuses_failed_to_completed(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), linked=False)
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    record = lifecycle.StatusRecord.from_dict(status)
    with pytest.raises(lifecycle.InvalidTransitionError):
        record.transition("COMPLETED_ON_MN5")
    with pytest.raises(lifecycle.InvalidTransitionError):
        record.recover_failed_to_completed(reason="no evidence", verification={})


def test_status_schema_requires_payload_for_failed_recovery(tmp_path: Path) -> None:
    from src.experiment_tracking.schemas import validate_status

    fold = _fail_fold(_build_attempt(tmp_path))
    bare = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    bare["state"] = "COMPLETED_ON_MN5"
    bare["history"].append(
        {"from": "FAILED", "to": "COMPLETED_ON_MN5", "at_utc": bare["updated_at_utc"]}
    )
    ok, errors = validate_status(bare)
    assert ok is False
    assert any("verified_recovery" in error for error in errors)
    recovered = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    recovered["state"] = "COMPLETED_ON_MN5"
    recovered["history"].append(
        {
            "from": "FAILED",
            "to": "COMPLETED_ON_MN5",
            "at_utc": recovered["updated_at_utc"],
            "reason": "verified linked retry",
            "verified_recovery": {
                "verified_retry_jobs": [
                    {"job_key": "best_eval", "failed_job_id": "102", "retry_job_id": "103"}
                ]
            },
        }
    )
    ok, errors = validate_status(recovered)
    assert ok is True, errors
