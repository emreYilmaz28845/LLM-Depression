#!/usr/bin/env python3
"""Lane-owned guarded integration for standalone fixed-head chains.

A head chain is exactly two scheduler jobs per validated parent training leg:
hidden-feature extraction plus the fixed ``logreg_raw``/``xgb_raw`` classifier
(head seed 1337, no Optuna). This wrapper integrates head dispatch with the
same lane budget and delivery proof as training fits:

- Eligibility requires the exact accepted parent: the dispatch plan's resolved
  parent attempt id must equal the validated attempt id of that training cell
  (durable validation receipt or managed lifecycle proof, exact identity).
- Every head submission must prove exactly two distinct numeric job ids
  (``extract_job_id`` and ``classifier_job_id``); anything else is recorded as
  an append-only uncertain record and stops the pass without retry.
- Before every leg the live own-nonterminal count (which already includes head
  deliveries from the registry and head attempt sidecars) and the user queue
  are reconciled; ``own + 2 <= 80`` and ``user < 350`` are required.
- When eligible head chains exist and at least two own slots are open, they are
  submitted before new training fits within the same budget. The whole pass
  runs under one exclusive lane lock so refill and head admission can never
  race.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import collect_legacy_prompt_cells as collector  # noqa: E402
from tools import qwen3_legacy_prompt_dispatch as dispatch  # noqa: E402

EVIDENCE = LANE / "outputs/qwen3_legacy_prompt_20261008"
PLAN = EVIDENCE / "head_dispatch_plan_v1.json"
REGISTRY = EVIDENCE / "head_submissions.jsonl"
LEDGER = EVIDENCE / "submissions.jsonl"
LOCK = EVIDENCE / ".submit.lock"
HEAD_JOB_FIELDS = ("extract_job_id", "classifier_job_id")


class HeadGuardError(RuntimeError):
    """Raised when head admission cannot be verified or the budget is exceeded."""


def parse_head_submit_output(output: str) -> dict[str, dict[str, str]]:
    """Parse the shared tool's ``=== JOB <key> ===`` blocks."""
    results: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in output.splitlines():
        line = line.strip()
        match = re.match(r"^=== JOB (.+) ===$", line)
        if match:
            current = match.group(1)
            results[current] = {}
            continue
        if current is None:
            continue
        if line.startswith("EXTRACT_ID="):
            results[current]["extract_job_id"] = line.split("=", 1)[1].strip()
        elif line.startswith("CLASSIFIER_ID="):
            results[current]["classifier_job_id"] = line.split("=", 1)[1].strip()
        elif line.startswith("ERROR="):
            results[current]["error"] = line.split("=", 1)[1].strip()
    return results


def valid_head_delivery(entry: dict[str, str]) -> bool:
    """Exactly two distinct numeric job ids for extract and classifier."""
    if entry.get("error"):
        return False
    if set(entry) - {"extract_job_id", "classifier_job_id"}:
        return False
    extract = entry.get("extract_job_id", "")
    classifier = entry.get("classifier_job_id", "")
    if not extract.isdigit() or not classifier.isdigit():
        return False
    return extract != classifier


def parse_cell(key: str) -> tuple[str, int, int] | None:
    parts = key.split("|")
    if len(parts) != 3:
        return None
    route = parts[0]
    seed = parts[1].lstrip("s")
    fold = parts[2].lstrip("f")
    if not seed.isdigit() or not fold.isdigit():
        return None
    return route, int(seed), int(fold)


def validated_cells(
    receipts_path: Path = collector.RECEIPTS,
    lifecycle: set[tuple[str, str, int]] | None = None,
    ledger_path: Path = LEDGER,
) -> dict[tuple[str, int, int], str]:
    """Exact attempt ids with authoritative validation for training cells."""
    receipts = collector.successful_receipts(receipts_path)
    proofs = lifecycle if lifecycle is not None else collector.lifecycle_validated()
    validated: dict[tuple[str, int, int], str] = {}
    for record in dispatch.ledger_records(ledger_path):
        if record.get("status") != "submitted" or not record.get("attempt_id"):
            continue
        cell = parse_cell(record["key"])
        if cell is None:
            continue
        attempt = str(record["attempt_id"])
        if (record["key"], attempt) in receipts or (
            record.get("run_name"),
            attempt,
            cell[2],
        ) in proofs:
            validated[cell] = attempt
    return validated


def eligible_head_jobs(
    plan: dict,
    validated: dict[tuple[str, int, int], str],
    submitted_keys: set[str],
) -> list[dict]:
    """Resolved plan entries whose exact parent attempt is validated and unsubmitted."""
    jobs: list[dict] = []
    for route in plan.get("routes") or []:
        for job in route.get("jobs") or []:
            if job.get("parent_status") != "resolved":
                continue
            key = job.get("registry_key")
            parent = job.get("parent") or {}
            cell = ((route.get("route_id") or route.get("dataset")), int(job["seed"]), int(job["fold"]))
            # plan registry keys are route|seed|fold; normalize to the same cell
            key_cell = parse_cell(key or "")
            cell = key_cell if key_cell else cell
            if key in submitted_keys:
                continue
            attempt = validated.get(cell)
            if not attempt or attempt != parent.get("attempt_id"):
                continue
            jobs.append(
                {
                    "key": key,
                    "cell": cell,
                    "parent_attempt_id": parent.get("attempt_id"),
                    "parent_adapter_sha256": parent.get("adapter_sha256"),
                }
            )
    return jobs


def registry_submitted_keys(registry_path: Path = REGISTRY) -> set[str]:
    if not registry_path.exists():
        return set()
    keys: set[str] = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["registry_key"])
    return keys


def head_ledger_record(
    job: dict, entry: dict[str, str], attempt_id: str | None, status: str, tail: str = ""
) -> dict:
    record = {
        "key": f"head::{job['key']}",
        "stage": "head",
        "ts": int(time.time()),
        "status": status,
        "attempt_id": attempt_id,
        "parent_attempt_id": job.get("parent_attempt_id"),
        "job_ids": (
            {
                "extract": entry.get("extract_job_id"),
                "classifier": entry.get("classifier_job_id"),
            }
            if status == "submitted"
            else {}
        ),
    }
    if status != "submitted":
        record["reason"] = "invalid or missing head delivery"
        record["tail"] = tail[-600:]
    return record


def run_campaign_pass(
    *,
    head_jobs: list[dict],
    fit_jobs: list[dict],
    reconcile: Callable[[], tuple[int, int]],
    submit_head: Callable[[dict], dict],
    submit_fit: Callable[[dict], dict],
    on_record: Callable[[dict], None],
    max_fits: int,
) -> dict[str, int]:
    """Heads first, then fits, all inside the shared live budget."""
    summary = {"heads_submitted": 0, "fits_submitted": 0, "refused": 0}
    pending_heads = list(head_jobs)
    pending_fits = list(fit_jobs)
    while pending_heads or (pending_fits and summary["fits_submitted"] < max_fits):
        own, user = reconcile()
        try:
            dispatch.per_fit_admission(own, user)
        except dispatch.AdmissionError as error:
            summary["refused"] = 1
            summary["reason"] = str(error)
            break
        if pending_heads:
            record = submit_head(pending_heads.pop(0))
            on_record(record)
            if record.get("status") != "submitted":
                summary["refused"] = 1
                summary["reason"] = "head delivery not proven; stopping pass"
                break
            summary["heads_submitted"] += 1
            continue
        record = submit_fit(pending_fits.pop(0))
        on_record(record)
        if record.get("status") != "submitted":
            summary["refused"] = 1
            summary["reason"] = "fit delivery not proven; stopping pass"
            break
        summary["fits_submitted"] += 1
    return summary


def submit_head_via_shared(job: dict, deployment_id: str) -> dict:
    command = [
        sys.executable,
        "tools/qwen3_heads_dispatch.py",
        "submit",
        "--key",
        job["key"],
        "--deployment-id",
        deployment_id,
        "--execute",
    ]
    try:
        result = subprocess.run(command, cwd=LANE, capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return head_ledger_record(job, {}, None, "uncertain", tail=str(exc))
    parsed = parse_head_submit_output(result.stdout or "")
    entry = parsed.get(job["key"]) or {}
    attempt_id = _latest_attempt(job["key"])
    if result.returncode == 0 and valid_head_delivery(entry) and attempt_id:
        return head_ledger_record(job, entry, attempt_id, "submitted")
    return head_ledger_record(
        job,
        entry,
        attempt_id,
        "uncertain",
        tail=(result.stdout or "") + (result.stderr or ""),
    )


def _latest_attempt(key: str) -> str | None:
    if not REGISTRY.exists():
        return None
    latest = None
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("registry_key") == key:
                latest = entry.get("attempt_id")
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fits", type=int, default=40)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {"routes": []}
        validated = validated_cells()
        heads = eligible_head_jobs(plan, validated, registry_submitted_keys())
        records = [r for r in dispatch.ledger_records() if r.get("status") == "submitted"]
        settled = dispatch.settled_keys()
        matrix = json.loads((EVIDENCE / "matrix.json").read_text(encoding="utf-8"))
        fits = [f for f in matrix["fits"] if f["key"] not in settled]

        def reconcile() -> tuple[int, int]:
            job_ids, uncertain, unknown = dispatch.own_job_ids()
            states, user = dispatch.query_job_states(job_ids)
            dispatch.require_reconciled(job_ids, states, unknown)
            return dispatch.own_nonterminal_count(job_ids, states, uncertain), user

        def record(entry: dict) -> None:
            with LEDGER.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

        if not args.execute:
            print(json.dumps({"planned_heads": len(heads), "planned_fits": len(fits)}, sort_keys=True))
            return 0
        summary = run_campaign_pass(
            head_jobs=heads,
            fit_jobs=fits,
            reconcile=reconcile,
            submit_head=lambda job: submit_head_via_shared(job, args.deployment_id),
            submit_fit=lambda fit: dispatch.submit_fit(fit),
            on_record=record,
            max_fits=args.max_fits,
        )
        print("campaign pass:", json.dumps(summary, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
