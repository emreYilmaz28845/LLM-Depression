#!/usr/bin/env python3
"""Bounded collect + validate for terminal production fits, with real skipping.

Correctness rules:

- A fully completed fit is skipped only on authoritative evidence for the
  exact current attempt: a durable validation receipt whose ``(logical cell,
  attempt_id)`` matches, or a managed ``status.json`` whose own ``attempt_id``
  and ``fold`` match the record and whose state is ``LOCALLY_VALIDATED`` /
  ``REPORTABLE``. A receipt from a prior attempt, or a lifecycle sidecar that
  cannot prove the exact attempt and fold, never causes a skip. Claimed log
  text is never enough.
- Collect success and validation success are tracked separately. A fit counts
  as validated only when both succeeded; a failed collect or validate is
  recorded in an append-only failure ledger with its return code and tail.
- A fit that fails collection or validation twice is marked for explicit
  diagnosis and skipped by later passes so it can never starve other cells;
  its training attempt is never reissued here.
- The per-pass limit applies only to cells that are actually attempted, so
  validated cells do not consume the budget and later cells cannot starve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import qwen3_legacy_prompt_dispatch as dispatch  # noqa: E402

EVIDENCE = LANE / "outputs/qwen3_legacy_prompt_20261008"
RUN_ROOT = LANE / "output_model/qwen3_legacy_prompt_20261008"
RECEIPTS = EVIDENCE / "validation_receipts.jsonl"
FAILURES = EVIDENCE / "validation_failures.jsonl"
DIAGNOSIS_THRESHOLD = 2
VALIDATED_STATES = {"LOCALLY_VALIDATED", "REPORTABLE"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def successful_receipts(receipts_path: Path = RECEIPTS) -> set[tuple[str, str]]:
    """Successful receipts as exact ``(logical cell, attempt id)`` pairs.

    A receipt without an attempt id cannot prove which delivery was validated
    and is therefore never usable for skipping.
    """
    return {
        (record["key"], str(record["attempt_id"]))
        for record in load_jsonl(receipts_path)
        if record.get("stage") == "validate"
        and record.get("ok") is True
        and record.get("attempt_id")
    }


def successful_keys(receipts_path: Path = RECEIPTS) -> set[str]:
    """Logical cells with at least one successful validation receipt (reported only)."""
    return {key for key, _attempt in successful_receipts(receipts_path)}


def failure_counts(failures_path: Path = FAILURES) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in load_jsonl(failures_path):
        if record.get("ok") is False:
            counts[record["key"]] = counts.get(record["key"], 0) + 1
    return counts


def lifecycle_validated(run_root: Path = RUN_ROOT) -> set[tuple[str, str, int]]:
    """Validated lifecycle proofs as exact ``(run name, attempt id, fold)``.

    The managed fold ``status.json`` is authoritative: it stores its own
    ``attempt_id`` and ``fold``. A sidecar that is validated but lacks either
    field, or that belongs to a different attempt, never causes a skip.
    """
    proofs: set[tuple[str, str, int]] = set()
    for status in run_root.glob("*/*/*/fold_*/status.json"):
        try:
            data = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("state") not in VALIDATED_STATES:
            continue
        attempt_id = data.get("attempt_id")
        fold = data.get("fold")
        if attempt_id is None or fold is None:
            continue
        proofs.add((status.parents[1].name, str(attempt_id), int(fold)))
    return proofs


def fold_number(key: str) -> int | None:
    try:
        return int(key.rsplit("|f", 1)[1])
    except (IndexError, ValueError):
        return None


def fully_completed(records: list[dict], states: dict[str, str]) -> list[dict]:
    done = []
    for record in records:
        if record.get("status") != "submitted":
            continue
        jobs = record.get("job_ids") or {}
        train = states.get(str(jobs.get("train")), "UNKNOWN")
        best = states.get(str(jobs.get("best_eval")), "UNKNOWN")
        if train == "COMPLETED" and best == "COMPLETED":
            done.append(record)
    return done


def skip_reason(
    record: dict,
    *,
    receipts: set[tuple[str, str]],
    lifecycle: set[tuple[str, str, int]],
    failures: dict[str, int],
) -> str | None:
    attempt_id = str(record.get("attempt_id") or "")
    if attempt_id and (record["key"], attempt_id) in receipts:
        return "receipt"
    fold = fold_number(record["key"])
    run_name = record.get("run_name")
    if attempt_id and fold is not None and (run_name, attempt_id, fold) in lifecycle:
        return "lifecycle"
    if failures.get(record["key"], 0) >= DIAGNOSIS_THRESHOLD:
        return "needs_diagnosis"
    return None


def plan_pass(
    records: list[dict],
    states: dict[str, str],
    *,
    receipts: set[tuple[str, str]],
    lifecycle: set[tuple[str, str, int]],
    failures: dict[str, int],
    limit: int,
) -> tuple[list[dict], dict[str, int]]:
    """Deterministic attempt list plus skipped counters; limit applies to attempts."""
    counters = {"skipped_receipt": 0, "skipped_lifecycle": 0, "skipped_diagnosis": 0}
    to_process: list[dict] = []
    for record in fully_completed(records, states):
        reason = skip_reason(record, receipts=receipts, lifecycle=lifecycle, failures=failures)
        if reason == "receipt":
            counters["skipped_receipt"] += 1
            continue
        if reason == "lifecycle":
            counters["skipped_lifecycle"] += 1
            continue
        if reason == "needs_diagnosis":
            counters["skipped_diagnosis"] += 1
            continue
        if len(to_process) >= limit:
            continue
        to_process.append(record)
    return to_process, counters


def run_pass(
    records: list[dict],
    states: dict[str, str],
    *,
    receipts_path: Path,
    failures_path: Path,
    lifecycle: set[tuple[str, str, int]],
    collect: Callable[[dict], int],
    validate: Callable[[dict], int],
    limit: int,
) -> dict[str, int]:
    receipts = successful_receipts(receipts_path)
    failures = failure_counts(failures_path)
    to_process, counters = plan_pass(
        records,
        states,
        receipts=receipts,
        lifecycle=lifecycle,
        failures=failures,
        limit=limit,
    )
    summary = {
        "attempted": 0,
        "collected_ok": 0,
        "validated_ok": 0,
        "failed_collect": 0,
        "failed_validate": 0,
        **counters,
    }
    for record in to_process:
        summary["attempted"] += 1
        collect_rc = collect(record)
        if collect_rc != 0:
            append_jsonl(
                failures_path,
                {
                    "key": record["key"],
                    "attempt_id": record.get("attempt_id"),
                    "stage": "collect",
                    "ok": False,
                    "rc": collect_rc,
                    "ts": int(time.time()),
                },
            )
            summary["failed_collect"] += 1
            continue
        summary["collected_ok"] += 1
        validate_rc = validate(record)
        if validate_rc != 0:
            append_jsonl(
                failures_path,
                {
                    "key": record["key"],
                    "attempt_id": record.get("attempt_id"),
                    "stage": "validate",
                    "ok": False,
                    "rc": validate_rc,
                    "ts": int(time.time()),
                },
            )
            summary["failed_validate"] += 1
            continue
        append_jsonl(
            receipts_path,
            {
                "key": record["key"],
                "attempt_id": record.get("attempt_id"),
                "stage": "validate",
                "ok": True,
                "rc": 0,
                "ts": int(time.time()),
            },
        )
        summary["validated_ok"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    records = [r for r in dispatch.ledger_records() if r.get("status") == "submitted"]
    if not records:
        print("no submitted ledger records")
        return 0
    ids = [str(value) for record in records for value in (record.get("job_ids") or {}).values()]
    states, user = dispatch.query_job_states(ids)

    status = subprocess.run(
        [sys.executable, "tools/exp.py", "status", "feat-qwen3-legacy-prompt-20261008"],
        cwd=LANE,
        capture_output=True,
        text=True,
        timeout=900,
    )

    def collect(record: dict) -> int:
        result = subprocess.run(
            [
                sys.executable,
                "tools/exp.py",
                "collect",
                "feat-qwen3-legacy-prompt-20261008",
                "--attempt-id",
                record["attempt_id"],
                "--execute",
            ],
            cwd=LANE,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            print(f"{record['key']} collect rc={result.returncode}")
        return result.returncode

    def validate(record: dict) -> int:
        result = subprocess.run(
            [sys.executable, "tools/exp.py", "validate", "--attempt-id", record["attempt_id"]],
            cwd=LANE,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            print(f"{record['key']} validate rc={result.returncode}: {(result.stdout + result.stderr)[-200:]}")
        return result.returncode

    summary = run_pass(
        records,
        states,
        receipts_path=RECEIPTS,
        failures_path=FAILURES,
        lifecycle=lifecycle_validated(),
        collect=collect,
        validate=validate,
        limit=args.limit,
    )
    print(
        "pass summary: "
        + json.dumps(
            {
                "lane_status_rc": status.returncode,
                "user_queue": user,
                "fully_completed": len(fully_completed(records, states)),
                **summary,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
