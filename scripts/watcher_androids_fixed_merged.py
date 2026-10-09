#!/usr/bin/env python3
"""Durable lane watcher for the Worker 1 corrected merged baseline.

Read-only monitoring: builds the lane's own job-ID list from the remote run
registries, refreshes every registry with the official monitor, samples job
states from the scheduler with return codes checked, and writes
``watcher_status.json`` (plus ``watcher_alert.json`` on failures or SSH query
errors).  An exclusive ``fcntl`` lock on ``watcher.lock`` guarantees a single
durable watcher; create ``watcher.stop`` to end it cleanly.

The watcher never submits, cancels or retries anything.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from scripts.dispatch_androids_fixed_merged import (  # noqa: E402
    AdmissionError,
    BSC_QUOTA_COMMAND,
    parse_bsc_quota_projects,
)
EVIDENCE = LANE / "outputs/qwen3_androids_official_folds_20261008"
RUNTIME = "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/feat-qwen3-androids-official-folds-20261008"
DEPLOYMENT_CODE = "/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/feat-qwen3-androids-official-folds-20261008-20261008T135256Z-0893c683-01b3e92d/code"
SCHEDULER = "ozu647717@alogin1.bsc.es"
FILE_HOST = "ozu647717@transfer1.bsc.es"
INTERVAL_SECONDS = 900
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}
FAILURE_STATES = {"FAILED", "NODE_FAIL", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "PREEMPTED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ssh(host: str, command: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def registry_paths() -> list[str]:
    result = ssh(FILE_HOST, f"ls {RUNTIME}/registries/qmsm_*.json 2>/dev/null")
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def own_job_ids(paths: list[str]) -> tuple[list[str], str]:
    listing = ",".join(paths)
    script = (
        "import json;"
        "ids=set();"
        f"paths={paths!r};"
        "[ids.update(str(j.get('job_id')) for j in json.load(open(p)).get('jobs',[]) "
        "if str(j.get('job_id','')).isdigit()) for p in paths];"
        "print(','.join(sorted(ids)))"
    )
    result = ssh(FILE_HOST, f"python3 -c \"{script}\"")
    if result.returncode != 0:
        return [], f"id_query_failed rc={result.returncode}: {result.stderr.strip()[:200]}"
    ids = [value for value in result.stdout.strip().split(",") if value.isdigit()]
    return ids, ""


def refresh_registries(paths: list[str]) -> str:
    for path in paths:
        result = ssh(
            SCHEDULER,
            "module purge >/dev/null 2>&1; module load bsc/1.0 miniforge/24.3.0-0 >/dev/null 2>&1; "
            "source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate; "
            f"timeout 60 python {DEPLOYMENT_CODE}/scripts/monitor_symmetric_merged.py --registry {path}",
        )
        if result.returncode != 0:
            return f"registry refresh failed for {Path(path).name} rc={result.returncode}"
    return ""


def job_states(ids: list[str]) -> tuple[dict[str, str], str]:
    if not ids:
        return {}, ""
    states: dict[str, str] = {}
    for index in range(0, len(ids), 200):
        chunk = ids[index : index + 200]
        result = ssh(
            SCHEDULER,
            f"sacct -j {','.join(chunk)} --format=JobID,State -P 2>/dev/null "
            "| awk -F'|' '$1 !~ /\\./ {print $1\"|\"$2}'",
        )
        if result.returncode != 0:
            return {}, f"sacct query failed rc={result.returncode}: {result.stderr.strip()[:200]}"
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 2 and parts[0] in set(chunk):
                states[parts[0]] = parts[1]
    return states, ""


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    lock_handle = (EVIDENCE / "watcher.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("watcher lock is held by another process; exiting")
        return 0
    (EVIDENCE / "watcher.pid").write_text(str(os.getpid()), encoding="utf-8")
    checks = 0
    while not (EVIDENCE / "watcher.stop").exists():
        checks += 1
        errors: list[str] = []
        paths = registry_paths()
        if not paths:
            errors.append("no registries found or file-host query failed")
        refresh_error = refresh_registries(paths) if paths else ""
        if refresh_error:
            errors.append(refresh_error)
        ids, id_error = own_job_ids(paths) if paths else ([], "")
        if id_error:
            errors.append(id_error)
        states, state_error = job_states(ids)
        if state_error:
            errors.append(state_error)
        counts: dict[str, int] = {}
        for state in states.values():
            counts[state] = counts.get(state, 0) + 1
        unresolved = len(ids) - len(states)
        nonterminal = sum(1 for value in states.values() if value not in TERMINAL) + unresolved
        failures = sorted(
            job_id for job_id, value in states.items() if value in FAILURE_STATES
        )
        storage: dict = {}
        quota_result = ssh(SCHEDULER, BSC_QUOTA_COMMAND)
        if quota_result.returncode != 0:
            errors.append(
                f"bsc_quota query failed rc={quota_result.returncode}: "
                f"{quota_result.stderr.strip()[:200]}"
            )
        else:
            try:
                parsed = parse_bsc_quota_projects(quota_result.stdout)
                try:
                    local_free = shutil.disk_usage(LANE).free / (1024 ** 3)
                except OSError as exc:
                    raise AdmissionError(f"local disk query failed: {exc}") from exc
                storage = {
                    **parsed,
                    "project_remaining_gib": parsed["soft_quota_gib"]
                    - parsed["usage_gib"]
                    - parsed["in_doubt_gib"],
                    "local_free_gib": local_free,
                }
            except AdmissionError as exc:
                errors.append(f"storage parse failed: {exc}")
        status = {
            "schema_version": "audiollm.androids_fixed_merged.watcher_status.v1",
            "utc": utc_now(),
            "checks": checks,
            "watcher_pid": os.getpid(),
            "registry_count": len(paths),
            "own_job_ids": len(ids),
            "resolved_job_ids": len(states),
            "unresolved_job_ids": unresolved,
            "nonterminal": nonterminal,
            "counts": counts,
            "failures": failures,
            "storage": storage,
            "errors": errors,
            "next_check_after_utc": datetime.fromtimestamp(
                time.time() + INTERVAL_SECONDS, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(EVIDENCE / "watcher_status.json", status)
        if failures or errors:
            write_json(
                EVIDENCE / "watcher_alert.json",
                {"utc": utc_now(), "failures": failures, "errors": errors, "checks": checks},
            )
        time.sleep(INTERVAL_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
