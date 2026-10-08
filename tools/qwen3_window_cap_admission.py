#!/usr/bin/env python3
"""Fail-closed lane admission and guarded submission for the window-cap campaign.

This is the tool that production fits, refills and masked head legs must go
through. It never merely reports: ``run-submit`` performs the admission check
and then runs the real submit command in the same invocation, so a submission
cannot bypass the guard.

Invariants (mirroring the reviewed cross-lane pattern):

* every scheduler query is a raw SSH command whose return code is checked;
  a failed or malformed query raises instead of reading an empty queue;
* the authoritative ownership sources (submission ledger, lane state, head
  registry when present, local fold sidecars) must exist and parse; a missing
  or malformed source fails closed rather than silently losing job ids;
* own job ids resolve from the full user queue first, then ``sacct``; ids
  absent from both count as nonterminal;
* every submit appends a durable conservative reservation *before* the command
  runs. Full delivery records the parsed ids; partial or timed-out delivery
  keeps the full reservation counted even when some ids are known, until a
  reconciliation event proves the reserved jobs terminal;
* the exact admission conditions are ``user_queue < 350`` and
  ``own_nonterminal + jobs_this_submit * wave_fits <= 80``, re-checked on every
  invocation; oversized waves are refused.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LANE_CAP_NONTERMINAL = 80
USER_QUEUE_STOP = 350
JOBS_PER_FIT = 2
JOBS_PER_HEAD_CHAIN = 2
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "OUT_OF_MEMORY",
    "REVOKED",
}
RESERVATION_ACTIVE_STATUSES = {"reserved", "partial", "uncertain"}
DEFAULT_SCHEDULER = "ozu647717@alogin2.bsc.es"
DEFAULT_USER = "ozu647717"


class AdmissionError(RuntimeError):
    """Raised when admission cannot be proven or the lane is full."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json_strict(path: Path, what: str) -> dict:
    """Load an authoritative ownership source; fail closed when absent/broken."""
    if not path.is_file():
        raise AdmissionError(
            f"{what} is missing at {path}; refusing to treat unknown ownership as empty"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"{what} is unreadable or malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdmissionError(f"{what} at {path} is not a JSON object")
    return payload


def parse_delimited(output: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        job_id, state = line.split("|", 1)
        job_id = job_id.strip()
        state = state.strip().split()[0] if state.strip() else ""
        if not job_id or not state:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        states[job_id] = state
    return states


def run_ssh(command: str, *, scheduler: str, user: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", scheduler, command],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise AdmissionError(
            f"scheduler query failed (rc={result.returncode}): {result.stderr.strip()[:200]}"
        )
    return result.stdout


def user_queue(*, scheduler: str, user: str) -> dict[str, str]:
    return parse_delimited(
        run_ssh(f"squeue -u {user} -h -o '%i|%T'", scheduler=scheduler, user=user)
    )


def read_reservations(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise AdmissionError(f"reservations file malformed at {path}: {exc}") from exc
        if not isinstance(entry, dict):
            raise AdmissionError(f"reservations entry is not an object in {path}")
        entries.append(entry)
    return entries


def append_reservation(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def reservation_statuses(entries: list[dict]) -> dict[str, dict]:
    """Latest status per reservation id, with the full history preserved."""
    statuses: dict[str, dict] = {}
    for entry in entries:
        reservation_id = str(entry.get("reservation_id") or "")
        if not reservation_id:
            continue
        event = str(entry.get("event") or "")
        if event == "reserved":
            statuses[reservation_id] = dict(entry, status="reserved")
        elif reservation_id in statuses:
            statuses[reservation_id].update(entry)
            statuses[reservation_id]["status"] = event
    return statuses


def active_reservations(entries: list[dict]) -> dict[str, dict]:
    return {
        reservation_id: record
        for reservation_id, record in reservation_statuses(entries).items()
        if record.get("status") in RESERVATION_ACTIVE_STATUSES
    }


def _chunks(items: list[str], size: int = 200):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def resolve_states(job_ids: list[str], *, scheduler: str, user: str) -> tuple[dict[str, str], int]:
    queue = user_queue(scheduler=scheduler, user=user)
    states = {job_id: queue[job_id] for job_id in job_ids if job_id in queue}
    missing = [job_id for job_id in job_ids if job_id not in states]
    for chunk in _chunks(missing):
        output = run_ssh(
            f"sacct -j {','.join(chunk)} -n -P -o JobIDRaw,State",
            scheduler=scheduler,
            user=user,
        )
        for line in output.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            job_id, state = line.split("|", 1)
            job_id = job_id.strip()
            if job_id in chunk and job_id not in states:
                states[job_id] = state.strip().split()[0] if state.strip() else "UNKNOWN"
    return states, len(queue)


def collect_ownership(
    *,
    ledger_path: Path,
    state_path: Path,
    head_registry: Path,
    local_run_root: Path,
) -> dict:
    """Authoritative own-attempt identity; missing sources fail closed."""
    ledger = load_json_strict(ledger_path, "submission ledger")
    state = load_json_strict(state_path, "lane state")
    attempts: set[str] = set()
    for entry in (state.get("job_inventory") or []):
        if entry.get("attempt_id"):
            attempts.add(str(entry["attempt_id"]))
    deployments = {
        str(value)
        for value in (state.get("deployments") or {}).values()
        if value
    }
    jobs: dict[str, dict] = {}
    uncertain_records = 0
    for record in ledger.get("jobs") or []:
        attempt_id = str(record.get("attempt_id") or "")
        deployment_id = str(record.get("deployment_id") or "")
        if attempts and attempt_id not in attempts and deployment_id not in deployments:
            continue
        job_id = record.get("slurm_job_id")
        if job_id:
            jobs[str(job_id)] = {
                "source": "ledger",
                "attempt_id": attempt_id,
                "job_key": record.get("job_key"),
                "event_type": record.get("event_type"),
            }
        elif record.get("event_type") == "SUBMITTED":
            uncertain_records += 1
    for entry in (state.get("job_inventory") or []):
        for key in ("train_job", "eval_job", "extract_job", "classifier_job"):
            if entry.get(key):
                jobs.setdefault(
                    str(entry[key]),
                    {
                        "source": "state_inventory",
                        "attempt_id": entry.get("attempt_id"),
                        "job_key": key,
                    },
                )
    if head_registry.is_file():
        try:
            lines = head_registry.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AdmissionError(f"head registry unreadable at {head_registry}: {exc}") from exc
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError as exc:
                raise AdmissionError(
                    f"head registry malformed at {head_registry}: {exc}"
                ) from exc
            for key in ("extract_job_id", "classifier_job_id"):
                if entry.get(key):
                    jobs.setdefault(
                        str(entry[key]),
                        {
                            "source": "head_registry",
                            "attempt_id": entry.get("attempt_id"),
                            "job_key": key,
                        },
                    )
    if local_run_root.is_dir():
        for sidecar in sorted(local_run_root.glob("*/*/*/fold_*/jobs.jsonl")):
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise AdmissionError(f"fold sidecar malformed at {sidecar}: {exc}") from exc
                job_id = event.get("slurm_job_id")
                if job_id:
                    jobs.setdefault(
                        str(job_id),
                        {
                            "source": "local_jobs_sidecar",
                            "attempt_id": event.get("attempt_id"),
                            "job_key": event.get("job_key"),
                        },
                    )
    return {
        "attempts": attempts,
        "deployments": deployments,
        "jobs": jobs,
        "uncertain_records": uncertain_records,
    }


def snapshot(
    *,
    ledger_path: Path,
    state_path: Path,
    head_registry: Path,
    reservations_path: Path,
    local_run_root: Path,
    scheduler: str,
    user: str,
) -> dict:
    ownership = collect_ownership(
        ledger_path=ledger_path,
        state_path=state_path,
        head_registry=head_registry,
        local_run_root=local_run_root,
    )
    reservations = read_reservations(reservations_path)
    active = active_reservations(reservations)
    reserved_jobs = sum(int(record.get("jobs_reserved") or 0) for record in active.values())
    reservation_job_ids = {
        str(job_id)
        for record in reservation_statuses(reservations).values()
        for job_id in (record.get("job_ids") or {}).values()
        if job_id
    }
    job_ids = sorted(set(ownership["jobs"]) | reservation_job_ids)
    states, queue_count = resolve_states(job_ids, scheduler=scheduler, user=user)
    nonterminal = sorted(
        job_id for job_id in job_ids if states.get(job_id, "UNKNOWN") not in TERMINAL_STATES
    )
    unresolved = sorted(job_id for job_id in job_ids if job_id not in states)
    own_nonterminal = (
        len(nonterminal)
        + JOBS_PER_FIT * ownership["uncertain_records"]
        + reserved_jobs
    )
    return {
        "recorded_at_utc": _now(),
        "user_queue_size": queue_count,
        "own_recorded_jobs": len(job_ids),
        "own_nonterminal_jobs": nonterminal,
        "own_unresolved_jobs": unresolved,
        "uncertain_submitted_records": ownership["uncertain_records"],
        "active_reservations": {
            reservation_id: {
                "status": record.get("status"),
                "jobs_reserved": record.get("jobs_reserved"),
                "job_ids": record.get("job_ids"),
            }
            for reservation_id, record in sorted(active.items())
        },
        "reserved_jobs": reserved_jobs,
        "own_nonterminal_count": own_nonterminal,
        "lane_cap": LANE_CAP_NONTERMINAL,
        "user_stop": USER_QUEUE_STOP,
        "head_chain_headroom": max(
            0, (LANE_CAP_NONTERMINAL - own_nonterminal) // JOBS_PER_HEAD_CHAIN
        ),
        "job_provenance": ownership["jobs"],
        "states": states,
    }


def admission_check(snap: dict, jobs_this_submit: int, wave_fits: int = 1) -> None:
    if jobs_this_submit < 1 or wave_fits < 1:
        raise AdmissionError("jobs_this_submit and wave_fits must be at least 1")
    total = jobs_this_submit * wave_fits
    if snap["user_queue_size"] >= USER_QUEUE_STOP:
        raise AdmissionError(
            f"user queue {snap['user_queue_size']} is at or above the {USER_QUEUE_STOP} stop"
        )
    if snap["own_nonterminal_count"] + total > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"no lane headroom: own nonterminal {snap['own_nonterminal_count']} + {total} "
            f"> {LANE_CAP_NONTERMINAL}"
        )
    if snap["own_unresolved_jobs"]:
        raise AdmissionError(
            "own job ids unresolved in queue and sacct; refusing to treat them as terminal: "
            + ", ".join(snap["own_unresolved_jobs"][:8])
        )


def parse_delivered_job_ids(output: str) -> dict[str, str]:
    """Extract submitted job ids from exp.py and head-dispatch outputs."""
    delivered: dict[str, str] = {}
    match = re.search(r"submitted jobs: (\{[^}]*\})", output)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, dict):
                delivered.update({str(k): str(v) for k, v in parsed.items()})
        except (ValueError, SyntaxError):
            pass
    for key in ("EXTRACT_ID", "CLASSIFIER_ID"):
        for job_id in re.findall(rf"{key}=(\d+)", output):
            delivered[f"{key.lower()}:{job_id}"] = job_id
    return delivered


def run_submit(
    argv: list[str],
    *,
    jobs_this_submit: int,
    wave_fits: int,
    kind: str,
    attempt_hint: str | None,
    ledger_path: Path,
    state_path: Path,
    head_registry: Path,
    reservations_path: Path,
    local_run_root: Path,
    scheduler: str,
    user: str,
    submit_timeout: int,
    delivery_files: list[Path] | None = None,
) -> int:
    """Admission check, durable reservation, then the real submit command."""
    snap = snapshot(
        ledger_path=ledger_path,
        state_path=state_path,
        head_registry=head_registry,
        reservations_path=reservations_path,
        local_run_root=local_run_root,
        scheduler=scheduler,
        user=user,
    )
    admission_check(snap, jobs_this_submit, wave_fits)
    reservation_id = f"{_now().replace(':', '').replace('-', '')}-{os.urandom(3).hex()}"
    append_reservation(
        reservations_path,
        {
            "event": "reserved",
            "reservation_id": reservation_id,
            "at_utc": _now(),
            "kind": kind,
            "attempt_hint": attempt_hint,
            "jobs_reserved": jobs_this_submit * wave_fits,
            "own_nonterminal_before": snap["own_nonterminal_count"],
            "user_queue_before": snap["user_queue_size"],
            "argv": argv,
        },
    )
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=submit_timeout
        )
    except subprocess.TimeoutExpired as exc:
        append_reservation(
            reservations_path,
            {
                "event": "uncertain",
                "reservation_id": reservation_id,
                "at_utc": _now(),
                "error": f"submit command timed out after {submit_timeout}s",
            },
        )
        raise AdmissionError(
            f"submit command timed out; reservation {reservation_id} stays counted "
            f"until reconciled"
        ) from exc
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    delivered = parse_delivered_job_ids(output)
    for delivery_file in delivery_files or []:
        if not delivery_file.is_file():
            continue
        try:
            delivered.update(
                parse_delivered_job_ids(delivery_file.read_text(encoding="utf-8"))
            )
        except OSError as exc:
            raise AdmissionError(
                f"delivery evidence file unreadable at {delivery_file}: {exc}"
            ) from exc
    expected = jobs_this_submit * wave_fits
    values = list(delivered.values())
    unique_ids = set(values)
    numeric = all(job_id.isdigit() for job_id in values)
    duplicates = len(values) != len(unique_ids)
    oversized = len(values) > expected
    proof_ok = (
        result.returncode == 0
        and len(values) == expected
        and len(unique_ids) == expected
        and numeric
    )
    if proof_ok:
        append_reservation(
            reservations_path,
            {
                "event": "delivered",
                "reservation_id": reservation_id,
                "at_utc": _now(),
                "job_ids": delivered,
                "proof": {
                    "expected": expected,
                    "parsed": len(values),
                    "unique": len(unique_ids),
                    "numeric": numeric,
                    "rc": result.returncode,
                },
            },
        )
        print(output.rstrip())
        return 0
    reasons: list[str] = []
    if result.returncode != 0:
        reasons.append(f"rc={result.returncode}")
    if duplicates:
        reasons.append("duplicate job ids")
    if oversized:
        reasons.append(f"oversized delivery {len(values)} for expected {expected}")
    if len(values) < expected:
        reasons.append(f"short delivery {len(values)} of {expected}")
    if not numeric:
        reasons.append("non-numeric job id")
    # Only a short delivery of unique numeric ids is a partial known subset;
    # duplicate, oversized or malformed deliveries are not proven legs and stay
    # uncertain. Every failure keeps the full reservation counted.
    event = (
        "partial"
        if values and numeric and not duplicates and not oversized
        else "uncertain"
    )
    append_reservation(
        reservations_path,
        {
            "event": event,
            "reservation_id": reservation_id,
            "at_utc": _now(),
            "job_ids": delivered,
            "proof": {
                "expected": expected,
                "parsed": len(values),
                "unique": len(unique_ids),
                "numeric": numeric,
                "rc": result.returncode,
            },
            "error": "; ".join(reasons) or "delivery not proven",
        },
    )
    print(output.rstrip(), file=sys.stderr)
    raise AdmissionError(
        f"submit {event} delivery ({'; '.join(reasons) or 'not proven'}); "
        f"reservation {reservation_id} stays counted until reconciled"
    )


def reconcile(
    reservations_path: Path,
    *,
    reservation_id: str | None,
    manual_note: str | None,
    scheduler: str,
    user: str,
    job_ids: str | None = None,
) -> list[dict]:
    entries = read_reservations(reservations_path)
    statuses = reservation_statuses(entries)
    results: list[dict] = []
    for rid, record in sorted(statuses.items()):
        if reservation_id and rid != reservation_id:
            continue
        if record.get("status") not in RESERVATION_ACTIVE_STATUSES:
            continue
        if job_ids is not None:
            if reservation_id is None:
                raise AdmissionError("--job-ids requires --reservation-id")
            supplied = [token.strip() for token in job_ids.split(",") if token.strip()]
            if not supplied or not all(token.isdigit() for token in supplied):
                raise AdmissionError("operator job ids must be non-empty numeric ids")
            if len(set(supplied)) != len(supplied):
                raise AdmissionError("operator job ids contain duplicates")
            reserved = int(record.get("jobs_reserved") or 0)
            if len(supplied) > reserved:
                raise AdmissionError(
                    f"operator supplied {len(supplied)} ids for a {reserved}-job reservation"
                )
            event = "delivered" if len(supplied) == reserved else "partial"
            append_reservation(
                reservations_path,
                {
                    "event": event,
                    "reservation_id": rid,
                    "at_utc": _now(),
                    "job_ids": {
                        f"operator_{index}": job_id
                        for index, job_id in enumerate(supplied)
                    },
                    "source": "operator_supplied",
                    "manual_note": manual_note,
                },
            )
            results.append(
                {
                    "reservation_id": rid,
                    "resolution": event,
                    "job_ids": supplied,
                    "source": "operator_supplied",
                }
            )
            continue
        distinct_ids = sorted(
            {str(job_id) for job_id in (record.get("job_ids") or {}).values() if job_id}
        )
        if manual_note and (reservation_id is None or rid == reservation_id):
            append_reservation(
                reservations_path,
                {
                    "event": "reconciled",
                    "reservation_id": rid,
                    "at_utc": _now(),
                    "manual_note": manual_note,
                },
            )
            results.append({"reservation_id": rid, "resolution": "manual", "note": manual_note})
            continue
        if not distinct_ids:
            results.append(
                {"reservation_id": rid, "resolution": "unresolved", "reason": "no job ids known"}
            )
            continue
        states, _ = resolve_states(distinct_ids, scheduler=scheduler, user=user)
        if all(states.get(job_id) in TERMINAL_STATES for job_id in distinct_ids) and (
            int(record.get("jobs_reserved") or 0) <= len(distinct_ids)
        ):
            append_reservation(
                reservations_path,
                {
                    "event": "reconciled",
                    "reservation_id": rid,
                    "at_utc": _now(),
                    "resolved_ids": distinct_ids,
                    "states": {job_id: states.get(job_id) for job_id in distinct_ids},
                },
            )
            results.append({"reservation_id": rid, "resolution": "terminal", "states": states})
        else:
            results.append(
                {
                    "reservation_id": rid,
                    "resolution": "still active",
                    "states": {job_id: states.get(job_id) for job_id in distinct_ids},
                }
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--head-registry", type=Path, required=True)
    parser.add_argument("--reservations", type=Path, required=True)
    parser.add_argument("--local-run-root", type=Path, required=True)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--submit-timeout", type=int, default=3600)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("evidence")
    check = sub.add_parser("check")
    check.add_argument("--jobs-this-submit", type=int, required=True)
    check.add_argument("--wave-fits", type=int, default=1)
    run = sub.add_parser("run-submit")
    run.add_argument("--jobs-this-submit", type=int, required=True)
    run.add_argument("--wave-fits", type=int, default=1)
    run.add_argument("--kind", choices=("fit", "head", "aux"), required=True)
    run.add_argument("--attempt-hint", default=None)
    run.add_argument(
        "--delivery-file",
        action="append",
        type=Path,
        default=[],
        help="file the command writes with EXTRACT_ID/CLASSIFIER_ID or submitted-jobs evidence",
    )
    run.add_argument("submit_command", nargs=argparse.REMAINDER)
    rec = sub.add_parser("reconcile")
    rec.add_argument("--reservation-id", default=None)
    rec.add_argument("--manual-note", default=None)
    rec.add_argument(
        "--job-ids",
        default=None,
        help="comma-separated distinct numeric ids proven for an uncertain reservation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = dict(
        ledger_path=args.ledger,
        state_path=args.state,
        head_registry=args.head_registry,
        reservations_path=args.reservations,
        local_run_root=args.local_run_root,
        scheduler=args.scheduler,
        user=args.user,
    )
    try:
        if args.command == "status":
            print(json.dumps(snapshot(**common), indent=1, sort_keys=True))
            return 0
        if args.command == "evidence":
            snap = snapshot(**common)
            print(json.dumps(snap, indent=1, sort_keys=True))
            return 0
        if args.command == "check":
            snap = snapshot(**common)
            admission_check(snap, args.jobs_this_submit, args.wave_fits)
            print(
                f"ADMISSION OK: own_nonterminal={snap['own_nonterminal_count']} + "
                f"{args.jobs_this_submit * args.wave_fits} <= {LANE_CAP_NONTERMINAL}; "
                f"user_queue={snap['user_queue_size']} < {USER_QUEUE_STOP}"
            )
            return 0
        if args.command == "run-submit":
            command = list(args.submit_command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                raise AdmissionError("run-submit needs a command after --")
            return run_submit(
                command,
                jobs_this_submit=args.jobs_this_submit,
                wave_fits=args.wave_fits,
                kind=args.kind,
                attempt_hint=args.attempt_hint,
                submit_timeout=args.submit_timeout,
                delivery_files=list(args.delivery_file or []),
                **common,
            )
        if args.command == "reconcile":
            results = reconcile(
                args.reservations,
                reservation_id=args.reservation_id,
                manual_note=args.manual_note,
                job_ids=args.job_ids,
                scheduler=args.scheduler,
                user=args.user,
            )
            print(json.dumps(results, indent=1, sort_keys=True))
            return 0
        raise AdmissionError(f"unknown command {args.command!r}")
    except AdmissionError as exc:
        print(f"ADMISSION REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
