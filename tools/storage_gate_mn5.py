#!/usr/bin/env python3
"""MN5 storage admission gate: real project quota plus local reserve.

The project quota must be read from ``bsc_quota`` for the owning group
(``gpfs_projects GRP`` row: Usage, soft Quota, hard Limit, In doubt). Shared
filesystem occupancy (``df``) does not enforce the granted project soft-quota
reserve and must never be used as the admission signal. The gate fails closed
when the SSH read fails, when the group header is absent, or when the
``gpfs_projects`` row cannot be parsed.

Admission requires ``soft - usage - in_doubt >= min_project_free_gb`` (the
500 GB project reserve) and local free space ``>= min_local_free_gb`` (the
50 GB local reserve). Units follow the displayed 1024-based convention
(``3.91 TB`` == 4000 GB soft quota).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_TRANSFER_HOST = "ozu647717@transfer1.bsc.es"
DEFAULT_GROUP = "etur92"
DEFAULT_MIN_PROJECT_FREE_GB = 500.0
DEFAULT_MIN_LOCAL_FREE_GB = 50.0

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROJECT_ROW = re.compile(
    r"^gpfs_projects\s+GRP\s+([\d.]+)\s*(KB|MB|GB|TB)\s+([\d.]+)\s*(KB|MB|GB|TB)\s+"
    r"([\d.]+)\s*(KB|MB|GB|TB)\s+([\d.]+)\s*(KB|MB|GB|TB)"
)
_UNIT_GB = {"KB": 1.0 / (1024**2), "MB": 1.0 / 1024, "GB": 1.0, "TB": 1024.0}


class StorageGateError(RuntimeError):
    """The gate could not verify storage (verification) or refused (threshold)."""

    def __init__(self, message: str, kind: str = "verification") -> None:
        super().__init__(message)
        self.kind = kind


def parse_bsc_quota_projects(text: str, group: str = DEFAULT_GROUP) -> dict[str, float]:
    """Parse the ``gpfs_projects GRP`` quota row for the owning group."""
    cleaned = "\n".join(_ANSI.sub("", line) for line in text.splitlines())
    if f"group {group}" not in cleaned:
        raise StorageGateError(f"bsc_quota output does not mention group {group}")
    for line in cleaned.splitlines():
        match = _PROJECT_ROW.match(line.strip())
        if not match:
            continue
        usage = float(match.group(1)) * _UNIT_GB[match.group(2)]
        soft = float(match.group(3)) * _UNIT_GB[match.group(4)]
        hard = float(match.group(5)) * _UNIT_GB[match.group(6)]
        in_doubt = float(match.group(7)) * _UNIT_GB[match.group(8)]
        return {
            "usage_gb": usage,
            "soft_quota_gb": soft,
            "hard_limit_gb": hard,
            "in_doubt_gb": in_doubt,
            "remaining_gb": soft - usage - in_doubt,
        }
    raise StorageGateError("no gpfs_projects GRP row found in bsc_quota output")


def read_bsc_quota(
    host: str = DEFAULT_TRANSFER_HOST,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> str:
    runner = runner or subprocess.run
    try:
        result = runner(
            ["ssh", "-o", "BatchMode=yes", host, "bsc_quota"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # SSH failure is never an admission signal
        raise StorageGateError(f"bsc_quota query failed: {type(exc).__name__}: {exc}") from exc
    if result.returncode != 0:
        raise StorageGateError(
            f"bsc_quota exited rc={result.returncode}: {(result.stderr or '').strip()[:200]}"
        )
    return (result.stdout or "") + "\n" + (result.stderr or "")


def evaluate_gate(
    quota: dict[str, float],
    local_free_bytes: int,
    min_project_free_gb: float = DEFAULT_MIN_PROJECT_FREE_GB,
    min_local_free_gb: float = DEFAULT_MIN_LOCAL_FREE_GB,
) -> tuple[bool, str]:
    local_free_gb = local_free_bytes / (1024**3)
    if quota["remaining_gb"] < min_project_free_gb:
        return (
            False,
            f"project reserve {quota['remaining_gb']:.2f} GB below required "
            f"{min_project_free_gb:.0f} GB",
        )
    if local_free_gb < min_local_free_gb:
        return (
            False,
            f"local free {local_free_gb:.2f} GB below required {min_local_free_gb:.0f} GB",
        )
    return True, "ok"


def check(
    *,
    host: str = DEFAULT_TRANSFER_HOST,
    group: str = DEFAULT_GROUP,
    min_project_free_gb: float = DEFAULT_MIN_PROJECT_FREE_GB,
    min_local_free_gb: float = DEFAULT_MIN_LOCAL_FREE_GB,
    local_path: Path | str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> dict[str, Any]:
    """Verify storage admission; raise :class:`StorageGateError` when refused."""
    text = read_bsc_quota(host, runner=runner)
    quota = parse_bsc_quota_projects(text, group=group)
    path = str(local_path or Path.home())
    local_free_bytes = shutil.disk_usage(path).free
    local_free_gb = local_free_bytes / (1024**3)
    ok, reason = evaluate_gate(quota, local_free_bytes, min_project_free_gb, min_local_free_gb)
    evidence: dict[str, Any] = {
        "host": host,
        "group": group,
        "usage_gb": round(quota["usage_gb"], 2),
        "in_doubt_gb": round(quota["in_doubt_gb"], 2),
        "soft_quota_gb": round(quota["soft_quota_gb"], 2),
        "hard_limit_gb": round(quota["hard_limit_gb"], 2),
        "remaining_gb": round(quota["remaining_gb"], 2),
        "min_project_free_gb": min_project_free_gb,
        "local_path": path,
        "local_free_gb": round(local_free_gb, 2),
        "min_local_free_gb": min_local_free_gb,
        "ok": ok,
        "reason": reason,
    }
    if not ok:
        raise StorageGateError(reason, kind="threshold")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_TRANSFER_HOST)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--min-project-free-gb", type=float, default=DEFAULT_MIN_PROJECT_FREE_GB)
    parser.add_argument("--min-local-free-gb", type=float, default=DEFAULT_MIN_LOCAL_FREE_GB)
    parser.add_argument("--local-path", type=Path, default=None)
    args = parser.parse_args()
    try:
        evidence = check(
            host=args.host,
            group=args.group,
            min_project_free_gb=args.min_project_free_gb,
            min_local_free_gb=args.min_local_free_gb,
            local_path=args.local_path,
        )
    except StorageGateError as error:
        print(f"storage gate refused ({error.kind}): {error}")
        return 1 if error.kind == "threshold" else 2
    print("storage gate:", json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
