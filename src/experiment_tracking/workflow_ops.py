from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any

def check_squeue(job_ids: list[str]) -> dict[str, str]:
    # Call squeue and return dict of job_id -> state
    if not job_ids:
        return {}
    try:
        result = subprocess.run(["squeue", "-j", ",".join(job_ids), "-o", "%i %T", "-h"], capture_output=True, text=True, timeout=10)
        states = {}
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        return states
    except Exception:
        return {}

def check_sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    try:
        result = subprocess.run(["sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State,ExitCode", "--noheader", "-P"], capture_output=True, text=True, timeout=10)
        states = {}
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 3:
                states[parts[0]] = {"State": parts[1], "ExitCode": parts[2]}
        return states
    except Exception:
        return {}

def classify_failure(exit_code: str, state: str) -> str:
    if state in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"):
        return "transient_infrastructure" if state in ("NODE_FAIL", "PREEMPTED", "BOOT_FAIL") or "NodeFail" in state else "deterministic"
    if "1:0" in exit_code or "0:0" not in exit_code:
        return "deterministic"
    return "unknown"

