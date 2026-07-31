from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from src.utils import save_json


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
