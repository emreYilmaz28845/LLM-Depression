#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


TERMINAL_SUCCESS = ("COMPLETED", "0:0")


def _registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or "job_id" not in rows[0]:
        raise ValueError(f"Invalid or empty D3TEC job registry: {path}")
    return rows


def _elapsed_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported Slurm elapsed value: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _sacct(host: str, job_ids: list[str]) -> list[dict[str, str]]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        (
            "sacct -X -n -P "
            f"-j {','.join(job_ids)} "
            "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(
        completed.stdout.splitlines(),
        delimiter="|",
        fieldnames=["job_id", "state", "exit_code", "elapsed", "start", "end"],
    )
    return [
        {key: (value or "").strip() for key, value in row.items()}
        for row in reader
        if row.get("job_id")
    ]


def capture(
    registry_path: Path,
    scheduler_host: str,
    output_path: Path,
    storage_path: Path | None = None,
) -> dict[str, Any]:
    registry = _registry(registry_path)
    job_ids = list(dict.fromkeys(row["job_id"] for row in registry))
    accounting_rows = _sacct(scheduler_host, job_ids)
    by_id = {row["job_id"]: row for row in accounting_rows}
    missing = [job_id for job_id in job_ids if job_id not in by_id]
    if missing:
        raise ValueError(f"Missing top-level sacct rows for job IDs: {missing}")

    jobs = []
    for registry_row in registry:
        row = {**registry_row, **by_id[registry_row["job_id"]]}
        row["elapsed_seconds"] = _elapsed_seconds(row["elapsed"])
        jobs.append(row)
    failures = [
        row
        for row in jobs
        if (row["state"], row["exit_code"]) != TERMINAL_SUCCESS
    ]
    if failures:
        raise ValueError(f"Non-successful registered jobs: {failures[:3]}")

    payload: dict[str, Any] = {
        "schema_version": "d3tec_hidden_slurm_accounting.v1",
        "registry": str(registry_path),
        "scheduler_host": scheduler_host,
        "jobs": jobs,
        "state_counts": dict(Counter(row["state"] for row in jobs)),
        "stage_runtime_seconds": dict(
            sorted(
                {
                    stage: sum(
                        row["elapsed_seconds"] for row in jobs if row["stage"] == stage
                    )
                    for stage in {row["stage"] for row in jobs}
                }.items()
            )
        ),
        "total_job_runtime_seconds": sum(row["elapsed_seconds"] for row in jobs),
        "retries": [],
    }
    if storage_path is not None:
        payload["storage"] = json.loads(storage_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture terminal top-level Slurm accounting for a D3TEC registry."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scheduler-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path)
    args = parser.parse_args()
    payload = capture(
        args.registry,
        args.scheduler_host,
        args.output,
        args.storage,
    )
    print(
        f"Captured {len(payload['jobs'])} successful jobs "
        f"({payload['total_job_runtime_seconds']} aggregate seconds)."
    )


if __name__ == "__main__":
    main()
