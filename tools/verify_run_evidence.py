from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.evidence import (
    EvidenceVerificationError,
    register_local_artifacts,
    set_metadata_supersedes,
    verify_artifacts_locally,
    verify_evaluations_locally,
)
from src.experiment_tracking.lifecycle import (
    InvalidTransitionError,
    StatusRecord,
    append_job_event,
    read_job_events,
    read_status,
    write_status,
)
from src.experiment_tracking.schemas import validate_status


def _fold_dir(value: str) -> Path:
    target = Path(value)
    if not target.is_dir() or not (target / "run_config.yaml").is_file():
        raise argparse.ArgumentTypeError(
            f"not a fold directory (missing run_config.yaml): {target}"
        )
    return target


def _cmd_verify_artifacts(fold_dir: Path) -> int:
    result = verify_artifacts_locally(fold_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_evaluations(fold_dir: Path) -> int:
    result = verify_evaluations_locally(fold_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_register_artifacts(fold_dir: Path, entries_file: Path) -> int:
    try:
        entries = json.loads(entries_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        print(f"error: unreadable artifact entries file: {error}", file=sys.stderr)
        return 2
    result = register_local_artifacts(fold_dir, entries)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_append_events(fold_dir: Path, events_file: Path) -> int:
    try:
        events = json.loads(events_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        print(f"error: unreadable events file: {error}", file=sys.stderr)
        return 2
    if not isinstance(events, list) or not events:
        print("error: events file must contain a non-empty JSON array", file=sys.stderr)
        return 2
    jobs_path = fold_dir / "jobs.jsonl"
    existing = read_job_events(jobs_path)
    known_event_ids = {event["event_id"] for event in existing}
    known_job_keys = {event["job_key"] for event in existing}
    appended = 0
    for event in events:
        if event.get("event_id") in known_event_ids:
            continue
        if event.get("job_key") not in known_job_keys:
            print(
                f"error: event references unknown job_key {event.get('job_key')!r}; "
                "terminal events may only extend existing jobs",
                file=sys.stderr,
            )
            return 2
        try:
            append_job_event(jobs_path, event)
        except ValueError as error:
            print(f"error: refused invalid job event: {error}", file=sys.stderr)
            return 2
        appended += 1
    print(json.dumps({"appended_events": appended, "total_events": len(existing) + appended}, indent=2))
    return 0


def _cmd_transition(fold_dir: Path, to_state: str, reason: str) -> int:
    status_path = fold_dir / "status.json"
    try:
        record = StatusRecord.from_dict(read_status(status_path))
    except (KeyError, ValueError) as error:
        print(f"error: unreadable status.json: {error}", file=sys.stderr)
        return 2
    ok, errors = validate_status(read_status(status_path))
    if not ok:
        print(f"error: invalid status.json: {'; '.join(errors)}", file=sys.stderr)
        return 2
    try:
        record.transition(to_state, reason=reason)
    except InvalidTransitionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    write_status(status_path, record)
    print(json.dumps({"state": record.state, "history_length": len(record.history)}, indent=2))
    return 0


def _cmd_set_supersedes(fold_dir: Path, attempt_id: str) -> int:
    result = set_metadata_supersedes(fold_dir, attempt_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct run tracking evidence through repository lifecycle APIs. "
        "Never rewrites metrics or predictions; every action is schema-validated and "
        "idempotent, and transitions must follow the lifecycle state machine."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    verify_artifacts = subparsers.add_parser(
        "verify-artifacts",
        help="Flip artifacts.json exists_locally/locally_verified flags from local hashes",
    )
    verify_artifacts.add_argument("fold_dir", type=_fold_dir)
    verify_artifacts.set_defaults(func=_cmd_verify_artifacts)

    verify_evaluations = subparsers.add_parser(
        "verify-evaluations",
        help="Mark evaluations.json locally_verified/reportable only when their artifacts are verified",
    )
    verify_evaluations.add_argument("fold_dir", type=_fold_dir)
    verify_evaluations.set_defaults(func=_cmd_verify_evaluations)

    register_artifacts = subparsers.add_parser(
        "register-artifacts",
        help="Register existing hash-verified local files in artifacts.json without overwriting evidence",
    )
    register_artifacts.add_argument("fold_dir", type=_fold_dir)
    register_artifacts.add_argument("entries_file", type=Path, help="JSON array of artifact descriptors")
    register_artifacts.set_defaults(func=_cmd_register_artifacts)

    append_events = subparsers.add_parser(
        "append-events",
        help="Append validated terminal job events to jobs.jsonl (append-only; existing events never change)",
    )
    append_events.add_argument("fold_dir", type=_fold_dir)
    append_events.add_argument("events_file", type=Path, help="JSON array of job events")
    append_events.set_defaults(func=_cmd_append_events)

    transition = subparsers.add_parser(
        "transition",
        help="Advance the lifecycle state through a valid transition (e.g. LOCALLY_VALIDATED -> REPORTABLE)",
    )
    transition.add_argument("fold_dir", type=_fold_dir)
    transition.add_argument("to_state", help="target lifecycle state")
    transition.add_argument("--reason", required=True)
    transition.set_defaults(func=_cmd_transition)

    set_supersedes = subparsers.add_parser(
        "set-supersedes",
        help="Record the superseded attempt id in metadata.json (only when absent)",
    )
    set_supersedes.add_argument("fold_dir", type=_fold_dir)
    set_supersedes.add_argument("attempt_id")
    set_supersedes.set_defaults(func=_cmd_set_supersedes)

    args = parser.parse_args()
    try:
        if args.action == "verify-artifacts":
            return _cmd_verify_artifacts(args.fold_dir)
        if args.action == "verify-evaluations":
            return _cmd_verify_evaluations(args.fold_dir)
        if args.action == "register-artifacts":
            return _cmd_register_artifacts(args.fold_dir, args.entries_file)
        if args.action == "append-events":
            return _cmd_append_events(args.fold_dir, args.events_file)
        if args.action == "transition":
            return _cmd_transition(args.fold_dir, args.to_state, args.reason)
        if args.action == "set-supersedes":
            return _cmd_set_supersedes(args.fold_dir, args.attempt_id)
        parser.error(f"unknown action: {args.action}")
        return 2
    except (EvidenceVerificationError, InvalidTransitionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
