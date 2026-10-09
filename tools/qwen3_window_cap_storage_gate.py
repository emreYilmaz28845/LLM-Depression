#!/usr/bin/env python3
"""Fail-closed storage gate for the window-cap refill watcher.

Queries the real BSC project quota (`bsc_quota projects`) and enforces the
granted reserve: remaining under the group soft quota (quota - usage -
in_doubt) must stay at or above the project reserve (default 500GB), and the
local filesystem must keep at least 50GB available. Any SSH failure, missing
`gpfs_projects` row or unparseable value refuses the gate; the shared
filesystem occupancy from `df` is never used as a quota signal.

Usage:
  python tools/qwen3_window_cap_storage_gate.py --json
  python tools/qwen3_window_cap_storage_gate.py --parse-file sample.txt
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_HOST = "ozu647717@transfer1.bsc.es"
DEFAULT_FILESYSTEM = "gpfs_projects"
DEFAULT_MIN_REMAINING_GB = 500.0
DEFAULT_MIN_LOCAL_GB = 50.0
DEFAULT_LOCAL_PATH = "/home/emre/Projects"
BYTES_PER_GB = 1024 ** 3
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SIZE_UNITS = {
    "B": 1.0 / BYTES_PER_GB,
    "KB": 1024.0 / BYTES_PER_GB,
    "MB": 1024.0 ** 2 / BYTES_PER_GB,
    "GB": 1.0,
    "TB": 1024.0,
    "PB": 1024.0 ** 2,
}


class StorageGateError(RuntimeError):
    """Raised when the storage gate cannot be proven; always fail closed."""


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _to_gb(value: str, unit: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise StorageGateError(f"unparseable size {value!r} with unit {unit!r}") from exc
    unit = unit.strip().upper()
    if unit not in _SIZE_UNITS:
        raise StorageGateError(f"unknown size unit {unit!r}")
    return number * _SIZE_UNITS[unit]


def parse_bsc_quota(output: str, filesystem: str = DEFAULT_FILESYSTEM) -> dict:
    """Parse the group row for one filesystem from `bsc_quota projects`."""
    if not output.strip():
        raise StorageGateError("bsc_quota output is empty")
    text = strip_ansi(output)
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != filesystem:
            continue
        if len(tokens) < 10:
            raise StorageGateError(f"{filesystem} row is too short: {line!r}")
        if tokens[1] != "GRP":
            raise StorageGateError(f"{filesystem} row is not a group row: {line!r}")
        return {
            "filesystem": filesystem,
            "usage_gb": _to_gb(tokens[2], tokens[3]),
            "quota_gb": _to_gb(tokens[4], tokens[5]),
            "limit_gb": _to_gb(tokens[6], tokens[7]),
            "in_doubt_gb": _to_gb(tokens[8], tokens[9]),
        }
    raise StorageGateError(f"no {filesystem} row found in bsc_quota output")


def evaluate(
    parsed: dict,
    *,
    min_remaining_gb: float = DEFAULT_MIN_REMAINING_GB,
    local_free_gb: float | None = None,
    min_local_gb: float = DEFAULT_MIN_LOCAL_GB,
) -> tuple[bool, str, float]:
    remaining = parsed["quota_gb"] - parsed["usage_gb"] - parsed["in_doubt_gb"]
    if remaining < min_remaining_gb:
        return (
            False,
            f"project soft-quota remaining {remaining:.1f}GB < {min_remaining_gb:.1f}GB",
            remaining,
        )
    if local_free_gb is not None and local_free_gb < min_local_gb:
        return (
            False,
            f"local free {local_free_gb:.1f}GB < {min_local_gb:.1f}GB",
            remaining,
        )
    local_text = "not checked" if local_free_gb is None else f"{local_free_gb:.1f}GB"
    return (
        True,
        f"project remaining {remaining:.1f}GB >= {min_remaining_gb:.1f}GB; "
        f"local free {local_text}",
        remaining,
    )


def query_quota(host: str) -> str:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host,
            "bsc_quota projects",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise StorageGateError(
            f"bsc_quota query failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:200]}"
        )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--filesystem", default=DEFAULT_FILESYSTEM)
    parser.add_argument("--min-remaining-gb", type=float, default=DEFAULT_MIN_REMAINING_GB)
    parser.add_argument("--min-local-gb", type=float, default=DEFAULT_MIN_LOCAL_GB)
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--parse-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = (
            args.parse_file.read_text(encoding="utf-8")
            if args.parse_file is not None
            else query_quota(args.host)
        )
        parsed = parse_bsc_quota(output, args.filesystem)
        local_free_gb = shutil.disk_usage(args.local_path).free / BYTES_PER_GB
        ok, reason, remaining = evaluate(
            parsed,
            min_remaining_gb=args.min_remaining_gb,
            local_free_gb=local_free_gb,
            min_local_gb=args.min_local_gb,
        )
        verdict = {
            "ok": ok,
            "reason": reason,
            "remaining_gb": remaining,
            "local_free_gb": local_free_gb,
            **parsed,
        }
        if args.json:
            print(json.dumps(verdict, indent=1, sort_keys=True))
        else:
            print(("STORAGE GATE OK" if ok else "STORAGE GATE REFUSED") + f": {reason}")
        return 0 if ok else 1
    except StorageGateError as exc:
        print(f"STORAGE GATE REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
