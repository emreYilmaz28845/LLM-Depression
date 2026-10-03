"""Failure-before-retry lifecycle recovery regression tests.

A fold can reach FAILED when ``exp.py status`` mirrors an original failed
evaluation before a later linked retry completes. The intended protocol is an
append-only same-attempt retry: a SUBMITTED event linked through
``resubmission_of_job_id`` plus a later cleanly COMPLETED terminal event. The
verified recovery gate may then move FAILED to COMPLETED_ON_MN5, and only after
that do the normal stepwise gates apply.

The gate requires an explicit exact ``0:0`` exit code, verifies chronology from
immutable ``at_utc`` timestamps (late-appended official evidence with real
submission times stays valid), and requires either a scheduler dependency on the
completed train job or a verified attempt contract for parent linkage.

Unlinked, wrong-attempt, wrong-parent, failed-latest, missing-exit,
fake-0:0-suffix, chronologically-impossible and out-of-order retries stay
blocked, and the generic lifecycle transition table must keep refusing
FAILED -> COMPLETED_ON_MN5.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_tracking import lifecycle  # noqa: E402
from src.experiment_tracking import validate as validate_module  # noqa: E402
from src.experiment_tracking.lifecycle import new_job_event  # noqa: E402
from src.experiment_tracking.validate import (  # noqa: E402
    finish_gates,
    recover_failed_attempt_from_verified_retry,
)
from tests.test_parallel_workflow_validate import KW, _build_attempt  # noqa: E402

ATTEMPT = "20260821T000000Z-run1-abcdef01-12345678"
OTHER_ATTEMPT = "20260821T000000Z-other-attempt-deadbeef-99999999"
BASE = datetime(2026, 10, 3, 2, 0, 0, tzinfo=timezone.utc)


def _t(minutes: int) -> datetime:
    return BASE + timedelta(minutes=minutes)


def _event(**kwargs):
    at = kwargs.pop("at")
    return new_job_event(attempt_id=ATTEMPT, fold=0, at_utc=at, **kwargs)


def _write_fold_events(fold: Path, events: list[dict]) -> None:
    (fold / "jobs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    status["state"] = "FAILED"
    status["history"].append({"from": "RUNNING", "to": "FAILED", "at_utc": status["updated_at_utc"]})
    (fold / "status.json").write_text(json.dumps(status), encoding="utf-8")


def _fail_fold(
    fold: Path,
    *,
    linked: bool = True,
    wrong_attempt: bool = False,
    failed_latest: bool = False,
    late_appended_evidence: bool = False,
    dependencies: tuple[str, ...] = ("101",),
    no_failure: bool = False,
    train_retry: bool = False,
    retry_exit_code: str | None = "0:0",
    retry_submitted_minute: int | None = None,
) -> Path:
    train_submitted = _event(job_key="train", job_type="train", event_type="SUBMITTED",
                             slurm_job_id="101", status="PENDING", at=_t(0))
    train_completed = _event(job_key="train", job_type="train", event_type="COMPLETED",
                             slurm_job_id="101", status="COMPLETED", at=_t(40))
    train_completed["exit_code"] = "0:0"
    eval_submitted = _event(job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
                            slurm_job_id="102", status="PENDING", dependency_job_ids=["101"],
                            at=_t(5))
    eval_failed = _event(job_key="best_eval", job_type="evaluation", event_type="FAILED",
                         slurm_job_id="102", status="FAILED", at=_t(50))
    eval_failed["exit_code"] = "1:0"
    eval_completed = _event(job_key="best_eval", job_type="evaluation", event_type="COMPLETED",
                            slurm_job_id="102", status="COMPLETED", at=_t(50))
    eval_completed["exit_code"] = "0:0"
    submitted_minute = retry_submitted_minute if retry_submitted_minute is not None else 60
    retry_attempt = OTHER_ATTEMPT if wrong_attempt else ATTEMPT
    retry_submitted = new_job_event(
        job_key="best_eval", job_type="evaluation", event_type="SUBMITTED",
        attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="PENDING",
        resubmission_of_job_id=("102" if linked else None),
        dependency_job_ids=list(dependencies),
        at_utc=_t(submitted_minute),
    )
    if failed_latest:
        retry_terminal = new_job_event(
            job_key="best_eval", job_type="evaluation", event_type="FAILED",
            attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="FAILED",
            dependency_job_ids=list(dependencies), at_utc=_t(90),
        )
        retry_terminal["exit_code"] = "1:0"
    else:
        retry_terminal = new_job_event(
            job_key="best_eval", job_type="evaluation", event_type="COMPLETED",
            attempt_id=retry_attempt, fold=0, slurm_job_id="103", status="COMPLETED",
            dependency_job_ids=list(dependencies), at_utc=_t(90),
        )
        retry_terminal["exit_code"] = retry_exit_code

    if train_retry:
        train_failed = _event(job_key="train", job_type="train", event_type="FAILED",
                              slurm_job_id="101", status="FAILED", at=_t(30))
        train_failed["exit_code"] = "1:0"
        train_retry_submitted = _event(job_key="train", job_type="train", event_type="SUBMITTED",
                                       slurm_job_id="105", status="PENDING",
                                       resubmission_of_job_id="101", at=_t(35))
        train_retry_completed = _event(job_key="train", job_type="train", event_type="COMPLETED",
                                       slurm_job_id="105", status="COMPLETED", at=_t(45))
        train_retry_completed["exit_code"] = "0:0"
        events = [
            train_submitted, train_failed, train_retry_submitted, train_retry_completed,
            eval_submitted, eval_completed,
        ]
    elif no_failure:
        events = [train_submitted, train_completed, eval_submitted, retry_submitted, retry_terminal]
    elif late_appended_evidence:
        # The official link and the failed terminal are appended after the retry
        # terminal, but carry their real historical timestamps.
        events = [
            train_submitted, train_completed, eval_submitted, retry_terminal, eval_failed,
            retry_submitted,
        ]
    else:
        events = [
            train_submitted, train_completed, eval_submitted, eval_failed, retry_submitted,
            retry_terminal,
        ]
    _write_fold_events(fold, events)
    return fold


def _make_contract(
    tmp_path: Path,
    fold: Path,
    *,
    checkpoint_tail: str = "best_model",
    attempt: str = ATTEMPT,
    fold_number: int = 0,
) -> Path:
    contract_dir = tmp_path / "outputs" / "exp_submit" / attempt
    contract_dir.mkdir(parents=True, exist_ok=True)
    resolved = fold.resolve()
    parts = resolved.parts
    rel = Path(*parts[parts.index("output_model"):]).as_posix()
    contract = {
        "attempt_id": attempt,
        "fold": fold_number,
        "kind": "standalone_backbone",
        "local_fold_rel": rel,
        "checkpoint_dir": f"/gpfs/x/AudioLLM/LLM-Depression/{rel}/{checkpoint_tail}",
        "standalone_eval_dir": f"/gpfs/x/AudioLLM/LLM-Depression/{rel}/best_model/standalone_eval",
        "qualifiers": {"checkpoint_role": "best_model"},
    }
    path = contract_dir / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_failed_then_linked_retry_recovers_and_finishes(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path))
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is True
    assert recovery["state"] == "COMPLETED_ON_MN5"
    entry = recovery["retry_jobs"][0]
    assert entry["job_key"] == "best_eval"
    assert entry["failed_job_id"] == "102"
    assert entry["retry_job_id"] == "103"
    assert entry["predecessor_submitted_at_utc"] < entry["retry_submitted_at_utc"]
    assert entry["retry_submitted_at_utc"] < entry["retry_completed_at_utc"]
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


def test_late_appended_retry_evidence_still_recovers(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), late_appended_evidence=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is True
    assert recovery["state"] == "COMPLETED_ON_MN5"


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
    assert "latest terminal event by at_utc" in recovery["reason"]


def test_missing_exit_code_retry_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), retry_exit_code=None)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "not a COMPLETED 0:0" in recovery["reason"]


def test_fake_zero_exit_code_prefix_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), retry_exit_code="0:01")
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "not a COMPLETED 0:0" in recovery["reason"]


def test_retry_submitted_before_predecessor_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), retry_submitted_minute=1)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "chronology invalid" in recovery["reason"]


def test_retry_submitted_after_completion_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), retry_submitted_minute=95)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "chronology invalid" in recovery["reason"]


def test_wrong_parent_dependency_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=("999",))
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "dependency link" in recovery["reason"]


def test_empty_parent_link_without_contract_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "EVAL_PARENT_SUBMIT_ROOT", tmp_path / "outputs" / "exp_submit")
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=())
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "no verified parent contract" in recovery["reason"]


def test_empty_parent_link_with_contract_recovers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "EVAL_PARENT_SUBMIT_ROOT", tmp_path / "outputs" / "exp_submit")
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=())
    contract = _make_contract(tmp_path, fold)
    assert contract.is_file()
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is True
    assert finish_gates(fold, **KW)["ok"] is True


def test_contract_with_wrong_checkpoint_dir_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "EVAL_PARENT_SUBMIT_ROOT", tmp_path / "outputs" / "exp_submit")
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=())
    _make_contract(tmp_path, fold, checkpoint_tail="last_model")
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "best_model checkpoint" in recovery["reason"]


def test_malformed_contract_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "EVAL_PARENT_SUBMIT_ROOT", tmp_path / "outputs" / "exp_submit")
    fold = _fail_fold(_build_attempt(tmp_path), dependencies=())
    path = _make_contract(tmp_path, fold)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["fold"] = "not-a-number"
    path.write_text(json.dumps(contract), encoding="utf-8")
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "contract fold is not an integer" in recovery["reason"]


def test_failed_state_without_failed_event_is_blocked(tmp_path: Path) -> None:
    fold = _fail_fold(_build_attempt(tmp_path), no_failure=True)
    recovery = recover_failed_attempt_from_verified_retry(fold)
    assert recovery["recovered"] is False
    assert "no linked retry evidence" in recovery["reason"]


def test_late_failure_after_success_blocks_finish(tmp_path: Path) -> None:
    fold = _build_attempt(tmp_path)
    late_failure = _event(job_key="best_eval", job_type="evaluation", event_type="FAILED",
                          slurm_job_id="102", status="FAILED", at=_t(120))
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
