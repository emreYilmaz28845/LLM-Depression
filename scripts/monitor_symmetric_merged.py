#!/usr/bin/env python3
"""Refresh a symmetric merged registry from squeue/sacct and terminal logs."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_json, save_json


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def _command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _state(job_id: str) -> tuple[str, str]:
    accounting = _command(
        ["sacct", "-X", "-n", "-P", "-j", str(job_id), "--format=State,ExitCode"]
    )
    if accounting:
        line = next((line for line in accounting.splitlines() if line.strip()), "")
        fields = line.split("|")
        if fields:
            state = fields[0].split("+", 1)[0].strip().upper().split(None, 1)[0]
            return state, fields[1].strip() if len(fields) > 1 else ""
    queue = _command(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
    if queue:
        return queue.splitlines()[0].strip().upper(), ""
    return "UNKNOWN", ""


def refresh_registry(path: Path) -> dict[str, Any]:
    registry = read_json(path)
    failures: list[dict[str, Any]] = []
    terminal = True
    for job in registry.get("jobs", []):
        job_id = job.get("job_id")
        if not job_id or str(job_id).startswith("dry_"):
            if job.get("state") == "planned_dry_run":
                job["observed_state"] = "DRY_RUN"
            continue
        state, exit_code = _state(str(job_id))
        job["observed_state"] = state
        job["exit_code"] = exit_code
        if state not in TERMINAL_SUCCESS | TERMINAL_FAILURE:
            terminal = False
        if state in TERMINAL_FAILURE or (state == "COMPLETED" and exit_code not in {"", "0:0"}):
            failures.append({"job_key": job.get("job_key"), "job_id": job_id, "state": state, "exit_code": exit_code})
    registry["terminal"] = terminal
    registry["observed_failures"] = failures
    registry["registry_status"] = "failed" if failures else "terminal_success" if terminal else "active"
    save_json(registry, path)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--watch-seconds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        result = refresh_registry(args.registry)
        print(json.dumps({
            "registry": str(args.registry),
            "registry_status": result["registry_status"],
            "terminal": result["terminal"],
            "failures": result["observed_failures"],
        }, indent=2), flush=True)
        if args.watch_seconds <= 0 or result["terminal"]:
            break
        time.sleep(min(args.watch_seconds, 60))
        args.watch_seconds = min(args.watch_seconds, 60)


if __name__ == "__main__":
    main()
