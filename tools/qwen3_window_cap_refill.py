#!/usr/bin/env python3
"""Bounded production refill driver for the window-cap campaign.

Tracked implementation of the window-cap refill driver. The lane runtime shim
``outputs/qwen3_train_window_cap_20261008/production_refill.py`` calls this
tool with ``--campaign-dir <lane dir> --ledger <parallel ledger>`` so the
running watcher keeps its exact behavior.

Every production fit is submitted through the tracked admission guard
(``run-submit``), which re-checks own + leg <= 80 and user < 350 on each leg,
records a durable reservation before the command, and proves the delivery with
exact unique numeric job ids. The driver itself never submits directly.

Safety layers: a non-blocking exclusive lane submission lock
(``submission.lock``) shared with the heads wave driver, and the exact-GB
500/50 storage gate run directly before any submission.

Usage:
  python tools/qwen3_window_cap_refill.py --campaign-dir <dir> --ledger <state.json> --max-fits 6 --dry-run
  python tools/qwen3_window_cap_refill.py --campaign-dir <dir> --ledger <state.json> --max-fits 6
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "qwen3_train_window_cap_20261008"
DEFAULT_LOCAL_RUN_ROOT = PROJECT_ROOT / "output_model" / "qwen3_train_window_cap_20261008"
DEFAULT_DEPLOYMENT = (
    "feat-qwen3-train-window-cap-20261008-20261008T114534Z-2f104779-db057646"
)
DEFAULT_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate"
DEFAULT_LANE_ID = "feat-qwen3-train-window-cap-20261008"
DEFAULT_CAMPAIGN = "qwen3_train_window_cap_20261008"
MAX_WAVE_FITS = 40

# Configured per campaign dir by configure(); defaults keep the operator path.
LANE = DEFAULT_CAMPAIGN_DIR
MATRIX = LANE / "production_matrix.json"
SUBMISSIONS = LANE / "production_submissions.jsonl"
GUARD = PROJECT_ROOT / "tools" / "qwen3_window_cap_admission.py"
STORAGE_GATE = PROJECT_ROOT / "tools" / "qwen3_window_cap_storage_gate.py"
SUBMISSION_LOCK = LANE / "submission.lock"
LEDGER: Path | None = None
LOCAL_RUN_ROOT = DEFAULT_LOCAL_RUN_ROOT
DEPLOYMENT = DEFAULT_DEPLOYMENT
ENV_ACTIVATE = DEFAULT_ENV_ACTIVATE
LANE_ID = DEFAULT_LANE_ID
CAMPAIGN = DEFAULT_CAMPAIGN


def configure(
    campaign_dir: Path | str | None = None,
    *,
    ledger: Path | str | None = None,
    local_run_root: Path | str | None = None,
    deployment: str | None = None,
    env_activate: str | None = None,
    lane_id: str | None = None,
    campaign: str | None = None,
    guard: Path | str | None = None,
    storage_gate: Path | str | None = None,
) -> None:
    """Point the refill driver at one campaign dir and admission context."""
    global LANE, MATRIX, SUBMISSIONS, GUARD, STORAGE_GATE, SUBMISSION_LOCK
    global LEDGER, LOCAL_RUN_ROOT, DEPLOYMENT, ENV_ACTIVATE, LANE_ID, CAMPAIGN
    LANE = Path(campaign_dir or DEFAULT_CAMPAIGN_DIR).resolve()
    MATRIX = LANE / "production_matrix.json"
    SUBMISSIONS = LANE / "production_submissions.jsonl"
    SUBMISSION_LOCK = LANE / "submission.lock"
    if ledger is not None:
        LEDGER = Path(ledger)
    if local_run_root is not None:
        LOCAL_RUN_ROOT = Path(local_run_root)
    if deployment is not None:
        DEPLOYMENT = str(deployment)
    if env_activate is not None:
        ENV_ACTIVATE = str(env_activate)
    if lane_id is not None:
        LANE_ID = str(lane_id)
    if campaign is not None:
        CAMPAIGN = str(campaign)
    if guard is not None:
        GUARD = Path(guard)
    if storage_gate is not None:
        STORAGE_GATE = Path(storage_gate)


def submitted_run_names() -> set[str]:
    names: set[str] = set()
    root = PROJECT_ROOT / "outputs" / "exp_submit"
    if root.is_dir():
        for entry in root.iterdir():
            contract = entry / "contract.json"
            if contract.is_file():
                try:
                    names.add(
                        str(json.loads(contract.read_text(encoding="utf-8"))["run_name"])
                    )
                except (ValueError, KeyError):
                    continue
    if SUBMISSIONS.is_file():
        for line in SUBMISSIONS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                names.add(str(json.loads(line)["run_name"]))
    return names


def acquire_submission_lock():
    """Non-blocking exclusive lane submission lock; None when busy."""
    handle = SUBMISSION_LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


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


def admit_paths() -> dict[str, str]:
    return {
        "--ledger": str(LEDGER),
        "--state": str(LANE / "state.json"),
        "--head-registry": str(LANE / "head_submissions.jsonl"),
        "--reservations": str(LANE / "admission_reservations.jsonl"),
        "--local-run-root": str(LOCAL_RUN_ROOT),
    }


def build_command(row: dict) -> list[str]:
    command = [sys.executable, str(GUARD)]
    for key, value in admit_paths().items():
        command += [key, value]
    command += [
        "run-submit",
        "--jobs-this-submit",
        "2",
        "--wave-fits",
        "1",
        "--kind",
        "fit",
        "--attempt-hint",
        row["run_name"],
        "--",
        sys.executable,
        "tools/exp.py",
        "submit",
        LANE_ID,
        "--config",
        row["config"],
        "--fold",
        str(row["fold"]),
        "--seed",
        str(row["seed"]),
        "--run-name",
        row["run_name"],
        "--campaign",
        CAMPAIGN,
        "--modality",
        row["modality"],
        "--dataset",
        row["dataset"],
        "--deployment-id",
        DEPLOYMENT,
        "--env-activate",
        ENV_ACTIVATE,
        "--execute",
    ]
    if row.get("manifest_policy") == "prebuilt":
        command += ["--manifest-policy", "prebuilt"]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--local-run-root", type=Path, default=None)
    parser.add_argument("--deployment", default=None)
    parser.add_argument("--env-activate", default=None)
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--max-fits", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-route", default=None)
    args = parser.parse_args(argv)
    configure(
        args.campaign_dir,
        ledger=args.ledger,
        local_run_root=args.local_run_root,
        deployment=args.deployment,
        env_activate=args.env_activate,
        lane_id=args.lane_id,
        campaign=args.campaign,
    )
    if args.max_fits < 1 or args.max_fits > MAX_WAVE_FITS:
        raise SystemExit(f"--max-fits must be 1..{MAX_WAVE_FITS}")
    if not args.dry_run:
        if LEDGER is None:
            raise SystemExit("--ledger is required for real submissions")
        lock = acquire_submission_lock()
        if lock is None:
            print("lane submission lock busy; skipping refill this cycle")
            return 0
        ok, text = storage_ok()
        if not ok:
            print(f"storage gate refused refill: {text}")
            return 1
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    done = submitted_run_names()
    pending = [
        row
        for row in matrix["treatments"]
        if row["run_name"] not in done
        and (args.only_route is None or row["route_id"] == args.only_route)
    ]
    if not pending:
        print("no pending production fits")
        return 0
    selected = pending[: args.max_fits]
    print(f"pending={len(pending)} selected={len(selected)} (each fit 2 jobs)")
    for row in selected:
        command = build_command(row)
        if args.dry_run:
            print("DRY-RUN:", " ".join(command))
            continue
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        entry = {
            "run_name": row["run_name"],
            "route_id": row["route_id"],
            "dataset": row["dataset"],
            "modality": row["modality"],
            "seed": row["seed"],
            "fold": row["fold"],
            "arm": row["arm"],
            "fraction": row["fraction"],
            "manifest_policy": row.get("manifest_policy"),
            "deployment_id": DEPLOYMENT,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-600:],
            "stderr_tail": proc.stderr[-600:],
        }
        match = re.search(r'"attempt_id": "([^"]+)"', proc.stdout)
        if match:
            entry["attempt_id"] = match.group(1)
        with SUBMISSIONS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        print(row["run_name"], "rc=", proc.returncode, entry.get("attempt_id", "-"))
        if proc.returncode != 0:
            print("stopping wave on first failure", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
