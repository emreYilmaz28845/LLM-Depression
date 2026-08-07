from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment_tracking import lifecycle, schemas
from src.experiment_tracking.lifecycle import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    StatusRecord,
    append_job_event,
    latest_job_event,
    new_job_event,
    read_job_events,
    read_status,
    write_status,
)

ATTEMPT_ID = "20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b"


def test_every_allowed_transition_succeeds() -> None:
    for from_state, targets in ALLOWED_TRANSITIONS.items():
        for to_state in sorted(targets):
            record = StatusRecord(ATTEMPT_ID, 0, state=from_state)
            assert record.transition(to_state) == to_state
            entry = record.history[-1]
            assert entry["from"] == from_state
            assert entry["to"] == to_state


def test_skipped_transitions_fail() -> None:
    record = StatusRecord(ATTEMPT_ID, 0, state="PLANNED")
    for skipped in ("SUBMITTED", "RUNNING", "COMPLETED_ON_MN5", "REPORTABLE"):
        with pytest.raises(InvalidTransitionError):
            record.transition(skipped)
    assert record.state == "PLANNED"
    assert record.history == []


def test_backward_transitions_fail() -> None:
    record = StatusRecord(ATTEMPT_ID, 0, state="COMPLETED_ON_MN5")
    for backward in ("RUNNING", "SUBMITTED", "DEPLOYED", "PLANNED"):
        with pytest.raises(InvalidTransitionError):
            record.transition(backward)
    assert record.state == "COMPLETED_ON_MN5"


def test_failed_and_cancelled_never_complete() -> None:
    for state in ("FAILED", "CANCELLED"):
        record = StatusRecord(ATTEMPT_ID, 0, state=state)
        for completion in ("RUNNING", "COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"):
            with pytest.raises(InvalidTransitionError):
                record.transition(completion)
        assert record.transition("SUPERSEDED") == "SUPERSEDED"


def test_full_forward_chain_succeeds() -> None:
    record = StatusRecord(ATTEMPT_ID, 0)
    chain = (
        "DEPLOYED",
        "SUBMITTED",
        "RUNNING",
        "COMPLETED_ON_MN5",
        "SYNCED_LOCALLY",
        "LOCALLY_VALIDATED",
        "REPORTABLE",
    )
    for state in chain:
        record.transition(state)
    assert record.state == "REPORTABLE"
    assert len(record.history) == len(chain)


def test_status_record_round_trip_validates(tmp_path: Path) -> None:
    record = StatusRecord(ATTEMPT_ID, 0)
    record.transition("DEPLOYED")
    path = tmp_path / "status.json"
    write_status(path, record)
    restored = StatusRecord.from_dict(read_status(path))
    assert restored.to_dict() == record.to_dict()
    ok, errors = schemas.validate_status(restored.to_dict())
    assert ok, errors


def test_job_events_append_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    submitted = new_job_event(
        job_key="train",
        job_type="train",
        event_type="SUBMITTED",
        attempt_id=ATTEMPT_ID,
        fold=0,
        slurm_job_id="1843921",
        status="PENDING",
    )
    started = new_job_event(
        job_key="train",
        job_type="train",
        event_type="STARTED",
        attempt_id=ATTEMPT_ID,
        fold=0,
        slurm_job_id="1843921",
        status="RUNNING",
    )
    append_job_event(path, submitted)
    append_job_event(path, started)
    events = read_job_events(path)
    assert [event["event_type"] for event in events] == ["SUBMITTED", "STARTED"]
    assert latest_job_event(events, "train")["event_type"] == "STARTED"
    assert latest_job_event(events, "evaluation") is None
    evaluation = new_job_event(
        job_key="evaluation",
        job_type="evaluation",
        event_type="SUBMITTED",
        attempt_id=ATTEMPT_ID,
        fold=0,
        slurm_job_id="1843922",
        dependency_job_ids=["1843921"],
        status="PENDING",
    )
    append_job_event(path, evaluation)
    assert len(read_job_events(path)) == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert latest_job_event(read_job_events(path), "evaluation")["dependency_job_ids"] == ["1843921"]


def test_append_job_event_rejects_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    with pytest.raises(ValueError, match="event_id"):
        append_job_event(
            path,
            {
                "schema_version": "audiollm.job_event.v1",
                "event_id": "not-a-uuid",
                "attempt_id": ATTEMPT_ID,
                "fold": 0,
                "job_key": "train",
                "job_type": "train",
                "event_type": "SUBMITTED",
                "dependency_job_ids": [],
                "at_utc": "2026-08-07T08:35:22.000000Z",
            },
        )
    assert not path.exists()


def test_status_validator_rejects_illegal_history() -> None:
    record = StatusRecord(ATTEMPT_ID, 0)
    record.transition("DEPLOYED")
    payload = record.to_dict()
    payload["history"][0]["to"] = "RUNNING"
    payload["state"] = "RUNNING"
    ok, errors = schemas.validate_status(payload)
    assert not ok
    assert any("not allowed" in error for error in errors)


def test_superseded_is_terminal_for_completed_states() -> None:
    for state in ("COMPLETED_ON_MN5", "IMPORTED_LEGACY", "FAILED", "CANCELLED", "REPORTABLE"):
        record = StatusRecord(ATTEMPT_ID, 0, state=state)
        assert record.transition("SUPERSEDED") == "SUPERSEDED"
        assert "SUPERSEDED" not in ALLOWED_TRANSITIONS.get("SUPERSEDED", frozenset())
