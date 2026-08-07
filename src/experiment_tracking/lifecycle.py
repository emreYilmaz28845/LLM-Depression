from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .canonical import (
    append_jsonl_atomic,
    format_utc_timestamp,
    read_json,
    read_jsonl,
    utc_now,
    write_json_atomic,
)
from .constants import LIFECYCLE_STATES, SCHEMA_VERSION_JOB_EVENT, SCHEMA_VERSION_STATUS

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "PLANNED": frozenset({"DEPLOYED"}),
    "DEPLOYED": frozenset({"SUBMITTED"}),
    "SUBMITTED": frozenset({"RUNNING", "FAILED", "CANCELLED"}),
    "RUNNING": frozenset({"COMPLETED_ON_MN5", "FAILED", "CANCELLED"}),
    "COMPLETED_ON_MN5": frozenset({"SYNCED_LOCALLY", "SUPERSEDED"}),
    "SYNCED_LOCALLY": frozenset({"LOCALLY_VALIDATED"}),
    "LOCALLY_VALIDATED": frozenset({"REPORTABLE"}),
    "IMPORTED_LEGACY": frozenset({"LOCALLY_VALIDATED", "SUPERSEDED"}),
    "FAILED": frozenset({"SUPERSEDED"}),
    "CANCELLED": frozenset({"SUPERSEDED"}),
    "REPORTABLE": frozenset({"SUPERSEDED"}),
}


def is_allowed_transition(from_state: str, to_state: str) -> bool:
    return to_state in ALLOWED_TRANSITIONS.get(from_state, frozenset())


def allowed_next_states(state: str) -> tuple[str, ...]:
    return tuple(sorted(ALLOWED_TRANSITIONS.get(state, frozenset())))


class InvalidTransitionError(ValueError):
    pass


class StatusRecord:
    def __init__(
        self,
        attempt_id: str,
        fold: int,
        state: str = "PLANNED",
        at_utc: datetime | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state: {state!r}")
        self.attempt_id = attempt_id
        self.fold = fold
        self.state = state
        self.updated_at_utc = format_utc_timestamp(at_utc if at_utc is not None else utc_now())
        self.history: list[dict[str, Any]] = list(history) if history else []

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "StatusRecord":
        obj = cls(
            attempt_id=record["attempt_id"],
            fold=record["fold"],
            state=record["state"],
            history=record.get("history"),
        )
        if isinstance(record.get("updated_at_utc"), str):
            obj.updated_at_utc = record["updated_at_utc"]
        return obj

    def transition(self, to_state: str, reason: str | None = None, at_utc: datetime | None = None) -> str:
        if to_state not in LIFECYCLE_STATES:
            raise InvalidTransitionError(
                f"unknown lifecycle state: {to_state!r}"
            )
        if not is_allowed_transition(self.state, to_state):
            raise InvalidTransitionError(
                f"transition {self.state!r} -> {to_state!r} is not allowed"
            )
        timestamp = format_utc_timestamp(at_utc if at_utc is not None else utc_now())
        entry = {"from": self.state, "to": to_state, "at_utc": timestamp, "reason": reason}
        self.history.append(entry)
        self.state = to_state
        self.updated_at_utc = timestamp
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION_STATUS,
            "attempt_id": self.attempt_id,
            "fold": self.fold,
            "state": self.state,
            "updated_at_utc": self.updated_at_utc,
            "history": [dict(entry) for entry in self.history],
        }


def write_status(path: str | Path, record: StatusRecord) -> None:
    write_json_atomic(path, record.to_dict())


def read_status(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def new_job_event(
    *,
    job_key: str,
    job_type: str,
    event_type: str,
    attempt_id: str,
    fold: int,
    slurm_job_id: str | None = None,
    slurm_array_job_id: str | None = None,
    slurm_array_task_id: str | None = None,
    dependency_job_ids: list[str] | None = None,
    status: str | None = None,
    reason: str | None = None,
    resubmission_of_job_id: str | None = None,
    event_id: str | None = None,
    at_utc: datetime | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_JOB_EVENT,
        "event_id": event_id if event_id is not None else str(uuid.uuid4()),
        "attempt_id": attempt_id,
        "fold": fold,
        "job_key": job_key,
        "job_type": job_type,
        "event_type": event_type,
        "slurm_job_id": slurm_job_id,
        "slurm_array_job_id": slurm_array_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "dependency_job_ids": list(dependency_job_ids or []),
        "status": status,
        "at_utc": format_utc_timestamp(at_utc if at_utc is not None else utc_now()),
        "reason": reason,
        "resubmission_of_job_id": resubmission_of_job_id,
    }


def append_job_event(path: str | Path, event: dict[str, Any]) -> None:
    from .schemas import validate_job_event

    ok, errors = validate_job_event(event)
    if not ok:
        raise ValueError("invalid job event: " + "; ".join(errors))
    append_jsonl_atomic(path, event)


def read_job_events(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def latest_job_event(events: list[dict[str, Any]], job_key: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("job_key") == job_key:
            return event
    return None
