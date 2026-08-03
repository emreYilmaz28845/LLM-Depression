from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from src.utils import save_json


def source_commits_match(left: Any, right: Any) -> bool:
    """Compare Git commit IDs while accepting unambiguous abbreviated forms."""

    left_value = str(left or "").strip().lower()
    right_value = str(right or "").strip().lower()
    if not left_value or not right_value:
        return False
    if not all(character in "0123456789abcdef" for character in left_value + right_value):
        return left_value == right_value
    shorter, longer = sorted((left_value, right_value), key=len)
    return len(shorter) >= 7 and longer.startswith(shorter)


def write_slurm_provenance(path: str | Path, *, worker: str, **payload: Any) -> None:
    """Record scheduler, host, interpreter, and worker identity beside artifacts."""

    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_SUBMIT_DIR",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_QOS",
    )
    source_commit = os.environ.get("SOURCE_COMMIT") or os.environ.get("SYMMETRIC_MERGED_SOURCE_COMMIT")
    provenance = {
        "schema_version": "symmetric_merged_slurm_provenance.v1",
        "worker": worker,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
        "scheduler": {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None},
        **payload,
    }
    if source_commit:
        provenance["source_commit"] = str(source_commit)
    save_json(provenance, path)
