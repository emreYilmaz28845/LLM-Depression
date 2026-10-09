#!/usr/bin/env python3
"""Submit bounded production masked-head waves through the admission guard.

Tracked implementation of the window-cap heads wave driver. The lane runtime
shim ``outputs/qwen3_train_window_cap_20261008/production_heads_wave.py`` calls
this tool with ``--campaign-dir <lane dir>`` so the running watcher keeps its
exact behavior.

Safety layers, in order:

1. pause marker: refuses unless ``heads_dispatch_enabled`` exists;
2. exclusive lane submission lock (``submission.lock``) held across rebuild,
   eligibility, admission and dispatch so head waves and base refills cannot
   race on budget;
3. fresh plan: the generator always rewrites the plan with a new build token;
   the driver refuses a missing or stale token;
4. arm-explicit keys: the 378-row matrix must yield 378 distinct expected
   keys, and every wave key must be inside that set with no duplicates;
5. storage gate: the exact-GB 500/50 gate runs directly before dispatch;
6. guard: ``admit.sh run-submit`` reserves the full 2*N chain budget and
   proves delivery from the dispatcher's ``submit_output.log``
   (``EXTRACT_ID=`` / ``CLASSIFIER_ID=`` lines). Registered chains that never
   produced both job ids are reported as stranded, never blind-retried.

Without ``--execute`` nothing is submitted and no reservation is made.

Usage:
  python tools/qwen3_window_cap_heads_wave.py --campaign-dir <dir> [--wave-fits N] [--execute]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "qwen3_train_window_cap_20261008"
DEFAULT_DEPLOYMENT = (
    "feat-qwen3-train-window-cap-20261008-20261008T114534Z-2f104779-db057646"
)
DEFAULT_CAMPAIGN = "qwen3_train_window_cap_20261008"
PLAN_MAX_AGE_SECONDS = 900

try:  # package import (tests, lane shim)
    from . import qwen3_window_cap_head_plan as planner
except ImportError:  # direct script execution
    sys.path.insert(0, str(TOOLS_DIR))
    import qwen3_window_cap_head_plan as planner

# Configured per campaign dir by configure(); defaults keep the operator path.
LANE = DEFAULT_CAMPAIGN_DIR
PLAN = LANE / "heads_production_plan.json"
WAVE_PLAN = LANE / "heads_production_wave_plan.json"
REGISTRY = LANE / "head_submissions.jsonl"
SUBMISSIONS = LANE / "heads_production_submissions.jsonl"
SUBMIT_OUTPUT = LANE / "head_submit_evidence" / "submit_output.log"
ENABLE_MARKER = LANE / "heads_dispatch_enabled"
SUBMISSION_LOCK = LANE / "submission.lock"
STORAGE_GATE = PROJECT_ROOT / "tools" / "qwen3_window_cap_storage_gate.py"
DISPATCH_TOOL = TOOLS_DIR / "qwen3_heads_dispatch.py"
DEPLOYMENT = DEFAULT_DEPLOYMENT
CAMPAIGN = DEFAULT_CAMPAIGN


def configure(
    campaign_dir: Path | str | None = None,
    *,
    deployment: str | None = None,
    campaign: str | None = None,
) -> None:
    """Point the wave driver at one campaign dir."""
    global LANE, PLAN, WAVE_PLAN, REGISTRY, SUBMISSIONS, SUBMIT_OUTPUT
    global ENABLE_MARKER, SUBMISSION_LOCK, DEPLOYMENT, CAMPAIGN
    LANE = Path(campaign_dir or DEFAULT_CAMPAIGN_DIR).resolve()
    PLAN = LANE / "heads_production_plan.json"
    WAVE_PLAN = LANE / "heads_production_wave_plan.json"
    REGISTRY = LANE / "head_submissions.jsonl"
    SUBMISSIONS = LANE / "heads_production_submissions.jsonl"
    SUBMIT_OUTPUT = LANE / "head_submit_evidence" / "submit_output.log"
    ENABLE_MARKER = LANE / "heads_dispatch_enabled"
    SUBMISSION_LOCK = LANE / "submission.lock"
    if deployment is not None:
        DEPLOYMENT = str(deployment)
    if campaign is not None:
        CAMPAIGN = str(campaign)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def acquire_submission_lock():
    """Non-blocking exclusive lane submission lock; None when busy."""
    handle = SUBMISSION_LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def registry_state() -> dict[str, dict]:
    state: dict[str, dict] = {}
    if REGISTRY.is_file():
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry.get("registry_key")
            if key:
                state[str(key)] = entry
    return state


def stranded_chains(state: dict[str, dict]) -> list[str]:
    """Registered chains that never produced both job ids (need repair)."""
    stranded = []
    for key, entry in sorted(state.items()):
        if (
            entry.get("error")
            or not entry.get("extract_job_id")
            or not entry.get("classifier_job_id")
        ):
            stranded.append(
                f"{key} (error={entry.get('error')!r}, "
                f"extract={entry.get('extract_job_id')}, "
                f"classifier={entry.get('classifier_job_id')})"
            )
    return stranded


def rebuild_plan() -> int:
    proc = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "qwen3_window_cap_head_plan.py"),
            "--campaign-dir",
            str(LANE),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def plan_age_seconds(plan: dict) -> float | None:
    created = plan.get("created_at_utc")
    if not created:
        return None
    try:
        stamp = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def build_wave(plan: dict, wave_fits: int, registered: set[str]) -> dict:
    routes = []
    for route in plan.get("routes") or []:
        jobs = [
            job
            for job in route.get("jobs") or []
            if f"{route['route_id']}|{int(job['seed'])}|{int(job['fold'])}"
            not in registered
        ]
        if jobs:
            routes.append({**route, "jobs": jobs})
    pending_total = sum(len(route["jobs"]) for route in routes)
    kept = []
    taken = 0
    for route in routes:
        if taken >= wave_fits:
            break
        kept.append({**route, "jobs": route["jobs"][: wave_fits - taken]})
        taken += len(kept[-1]["jobs"])
    wave = {**plan, "routes": kept, "created_at_utc": now()}
    wave["summary"] = {
        "resolved": taken,
        "pending_ready_chains": pending_total,
        "waiting_for_checkpoint": plan.get("summary", {}).get("waiting_for_checkpoint"),
        "blocked_failed_parent": plan.get("summary", {}).get("blocked_failed_parent"),
        "expected_keys": plan.get("summary", {}).get("expected_keys"),
    }
    return wave


def wave_keys(wave: dict) -> list[str]:
    keys = []
    for route in wave.get("routes") or []:
        for job in route.get("jobs") or []:
            keys.append(f"{route['route_id']}|{int(job['seed'])}|{int(job['fold'])}")
    return keys


def guard_command(wave_plan: Path, wave_fits: int) -> list[str]:
    return [
        "bash",
        str(LANE / "admit.sh"),
        "run-submit",
        "--jobs-this-submit",
        "2",
        "--wave-fits",
        str(wave_fits),
        "--kind",
        "head",
        "--delivery-file",
        str(SUBMIT_OUTPUT),
        "--",
        sys.executable,
        str(DISPATCH_TOOL),
        "submit",
        "--plan",
        str(wave_plan),
        "--campaign",
        CAMPAIGN,
        "--deployment-id",
        DEPLOYMENT,
        "--execute",
    ]


def storage_ok() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(STORAGE_GATE), "--json"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip() or "refused")[:300]
    return True, proc.stdout.strip()[:300]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--wave-fits", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--deployment", default=None)
    parser.add_argument("--campaign", default=None)
    args = parser.parse_args(argv)
    configure(args.campaign_dir, deployment=args.deployment, campaign=args.campaign)

    if not ENABLE_MARKER.is_file():
        print(
            "production head dispatch paused: missing "
            f"{ENABLE_MARKER.name} (enable only after the fix self-verification passes)"
        )
        return 0
    lock = acquire_submission_lock()
    if lock is None:
        print("lane submission lock busy; skipping heads wave this cycle")
        return 0
    try:
        rc = rebuild_plan()
        if rc != 0:
            return rc
        if not PLAN.is_file():
            print("no plan written by the generator; refusing")
            return 1
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        token = plan.get("build_token")
        age = plan_age_seconds(plan)
        if not token or age is None or age > PLAN_MAX_AGE_SECONDS:
            print(
                f"stale or untokened plan refused (token={token!r}, age={age}); "
                "readiness may have been withdrawn"
            )
            return 1
        matrix = json.loads(planner.MATRIX.read_text(encoding="utf-8"))
        expected = planner.assert_matrix_keys(matrix)
        state = registry_state()
        stranded = stranded_chains(state)
        if stranded:
            print(
                f"WARNING: {len(stranded)} registered head chain(s) need repair "
                "(not retried):"
            )
            for line in stranded[:10]:
                print(f"  {line}")
        wave = build_wave(plan, args.wave_fits, set(state))
        keys = wave_keys(wave)
        if len(set(keys)) != len(keys):
            print("duplicate keys inside the wave; refusing")
            return 1
        unknown = [key for key in keys if key not in expected]
        if unknown:
            print(f"wave keys outside the 378 expected keys; refusing: {unknown[:5]}")
            return 1
        taken = wave["summary"]["resolved"]
        if taken == 0:
            print(
                "no pending ready head chains "
                f"(pending ready: {wave['summary']['pending_ready_chains']}, "
                f"waiting for checkpoints: {wave['summary']['waiting_for_checkpoint']})"
            )
            return 0
        ok, text = storage_ok()
        if not ok:
            print(f"storage gate refused head wave: {text}")
            return 1
        WAVE_PLAN.write_text(json.dumps(wave, indent=1) + "\n", encoding="utf-8")
        command = guard_command(WAVE_PLAN, taken)
        print(
            f"wave chains: {taken} (pending ready: "
            f"{wave['summary']['pending_ready_chains']}) plan token={token}"
        )
        print("guard command:", " ".join(command))
        if not args.execute:
            print("dry-run only; no submission performed")
            return 0
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        record = {
            "at_utc": now(),
            "wave_chains": taken,
            "wave_plan": str(WAVE_PLAN),
            "plan_build_token": token,
            "command": command,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-2000:],
            "stranded_chains": stranded,
        }
        with SUBMISSIONS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
