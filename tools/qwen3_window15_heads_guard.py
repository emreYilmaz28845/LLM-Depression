#!/usr/bin/env python3
"""Guarded window15 head dispatch: lane admission around tools/qwen3_heads_dispatch.py.

The generic dispatcher owns the head plan/submit/collect/validate machinery;
this wrapper owns the lane's resource and evidence guards so a head wave obeys
exactly the same rules as training refills:

- an exclusive flock submission lock (one submitter per lane at a time);
- only keys whose planner audit status is ``eligible`` (a validated treatment
  parent) are ever submitted; keys already present in the generic registry are
  skipped, so healthy attempts are never duplicated and nothing is retried
  blindly;
- before the wave both admission counts are re-checked with the dispatch
  driver's authoritative functions: ``own_nonterminal + 2 per key <= 80`` and
  ``user < 350``; the real ``bsc_quota`` storage gate (>= 500 GB project
  reserve, >= 50 GB local) must pass;
- every attempted key must return numeric ``EXTRACT_ID`` and ``CLASSIFIER_ID``
  values in the raw submit output; anything else is recorded as uncertain with
  the raw tail preserved and the wave stops;
- deliveries are appended to the lane ledger, so the durable watcher and all
  later wave accounting see head jobs as owned.

Read-only subcommands (status/collect/validate/finish/coverage) delegate to the
generic tool with the lane's registry.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_window15_dispatch import (  # noqa: E402
    AdmissionError,
    append_record,
    own_job_ids,
    own_nonterminal_count,
    query_job_states,
    storage_admission,
)

HEADS_TOOL = LANE / "tools/qwen3_heads_dispatch.py"


AUDIT = LANE / "outputs/qwen3_window15_20261008/heads/heads_plan_audit.json"
LOCK = LANE / "outputs/qwen3_window15_20261008/heads/submit.lock"
JOBS_PER_KEY = 2


class GuardError(RuntimeError):
    pass


def _baseline_registry() -> Path:
    return LANE / "outputs/qwen3_window15_20261008/head_submissions.jsonl"


def _registry_keys(registry_path: Path) -> set[tuple[str, str]]:
    """(route_id, seed, fold) keys already present in the generic registry."""
    settled: set[tuple[str, str]] = set()
    if not registry_path.is_file():
        return settled
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = entry.get("key") or entry.get("parent_key") or ""
        if isinstance(key, str) and key:
            settled.add(tuple(key.split("|", 2)) if "|" in key else (key,))
        route = entry.get("route_id")
        seed = entry.get("parent_training_seed", entry.get("seed"))
        fold = entry.get("fold")
        if route is not None and seed is not None and fold is not None:
            settled.add((str(route), str(int(seed)), str(int(fold))))
    return settled


def eligible_keys(audit_path: Path = AUDIT) -> list[dict]:
    if not audit_path.is_file():
        raise GuardError(f"planner audit missing: {audit_path}; run tools/qwen3_window15_heads_plan.py first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("keys_total") != 126:
        raise GuardError("planner audit does not cover 126 keys")
    return [item for item in audit.get("keys", []) if item.get("status") == "eligible"]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=LANE, capture_output=True, text=True, timeout=3600)


def parse_submit_ids(output: str) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in output.splitlines():
        line = line.strip()
        match = re.match(r"^=== JOB (.+) ===$", line)
        if match:
            current = match.group(1)
            results[current] = {}
            continue
        if current is None:
            continue
        if line.startswith("ERROR="):
            results[current]["error"] = line.split("=", 1)[1]
        elif line.startswith("EXTRACT_ID="):
            results[current]["extract"] = line.split("=", 1)[1].strip()
        elif line.startswith("CLASSIFIER_ID="):
            results[current]["classifier"] = line.split("=", 1)[1].strip()
    return results


def _numeric_ids(entry: dict[str, str]) -> bool:
    return bool(
        re.fullmatch(r"\d+", str(entry.get("extract", "")))
        and re.fullmatch(r"\d+", str(entry.get("classifier", "")))
    )


def command_submit(args: argparse.Namespace) -> int:
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("REFUSED: another head submitter holds the lane lock")
        return 2
    try:
        eligible = eligible_keys()
        settled = _registry_keys(args.registry)
        pending = [
            item
            for item in eligible
            if (item["route_id"], str(item["seed"]), str(item["fold"])) not in settled
            and item["key"] not in {f"{r}|{s}|{f}" for r, s, f in settled}
        ]
        if args.only:
            wanted = set(args.only)
            pending = [item for item in pending if item["key"] in wanted]
        if not pending:
            print(json.dumps({"status": "ok", "pending": 0, "note": "no eligible unsubmitted keys"}))
            return 0

        job_ids, uncertain = own_job_ids()
        states, user_queue = query_job_states(job_ids)
        own_nonterminal = own_nonterminal_count(job_ids, states, uncertain)
        storage = storage_admission()

        limit = int(args.limit) if args.limit else len(pending)
        capacity_keys = max(0, (80 - own_nonterminal) // JOBS_PER_KEY)
        limit = min(limit, capacity_keys, len(pending))
        if user_queue >= 350:
            print(f"REFUSED: user queue {user_queue} at or above 350")
            return 2
        if limit < 1:
            print(f"REFUSED: no lane headroom (own {own_nonterminal} + {JOBS_PER_KEY} > 80)")
            return 2
        batch = pending[:limit]
        print(
            "heads admission: " + json.dumps(
                {
                    "own_nonterminal": own_nonterminal,
                    "user_queue": user_queue,
                    "eligible": len(eligible),
                    "pending": len(pending),
                    "batch": len(batch),
                    "jobs_per_key": JOBS_PER_KEY,
                    "storage_remaining_gb": storage["gpfs_projects_remaining_gb"],
                },
                sort_keys=True,
            )
        )
        if args.dry_run:
            print(json.dumps({"status": "dry-run", "keys": [item["key"] for item in batch]}, sort_keys=True))
            return 0

        plan_cmd = [
            sys.executable, str(HEADS_TOOL), "plan",
            "--matrix", str(args.matrix),
            "--campaign", "qwen3_window15_20261008",
            "--output", str(args.plan),
        ]
        plan_result = _run(plan_cmd)
        if plan_result.returncode != 0:
            print(f"REFUSED: generic plan failed rc={plan_result.returncode}: {plan_result.stderr.strip()[:300]}")
            return 2

        cmd = [
            sys.executable, str(HEADS_TOOL), "submit",
            "--plan", str(args.plan),
            "--registry", str(args.registry),
            "--deployment-id", str(args.deployment_id),
            "--limit", str(len(batch)),
            "--execute",
        ]
        for item in batch:
            cmd += ["--key", item["key"]]
        result = _run(cmd)
        raw = (result.stdout or "") + "\n" + (result.stderr or "")
        parsed = parse_submit_ids(result.stdout or "")
        delivered = 0
        for item in batch:
            entry = parsed.get(item["key"], {})
            if result.returncode == 0 and _numeric_ids(entry):
                append_record(
                    {
                        "key": f"head:{item['key']}",
                        "run_name": str(item.get("run_name", "")),
                        "ts": int(time.time()),
                        "status": "submitted",
                        "attempt_id": None,
                        "job_ids": {"extract": entry["extract"], "classifier": entry["classifier"]},
                        "parent_attempt_id": item.get("attempt_id"),
                    }
                )
                delivered += 1
            else:
                append_record(
                    {
                        "key": f"head:{item['key']}",
                        "run_name": str(item.get("run_name", "")),
                        "ts": int(time.time()),
                        "status": "uncertain",
                        "reason": f"submit rc={result.returncode}; missing numeric extract/classifier ids",
                        "attempt_id": None,
                        "job_ids": {},
                        "parent_attempt_id": item.get("attempt_id"),
                        "tail": raw[-800:],
                    }
                )
                break
        print(json.dumps({"status": "ok", "delivered": delivered, "attempted": len(batch)}, sort_keys=True))
        return 0 if delivered == len(batch) else 1
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def command_delegate(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(HEADS_TOOL), args.command, "--registry", str(args.registry)]
    for passthrough in args.rest or []:
        cmd.append(passthrough)
    return _run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="guarded head submit (dry-run first)")
    submit.add_argument("--matrix", type=Path, required=True)
    submit.add_argument("--plan", type=Path, default=LANE / "outputs/qwen3_window15_20261008/head_dispatch_plan_v1.json")
    submit.add_argument("--registry", type=Path, default=_baseline_registry())
    submit.add_argument("--deployment-id", required=True)
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument("--only", action="append", default=None)
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=command_submit)

    for name in ("status", "collect", "validate", "finish", "coverage"):
        delegate = sub.add_parser(name, help=f"delegate to tools/qwen3_heads_dispatch.py {name}")
        delegate.add_argument("--registry", type=Path, default=_baseline_registry())
        delegate.add_argument("--rest", nargs=argparse.REMAINDER)
        delegate.set_defaults(func=command_delegate)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (GuardError, AdmissionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
