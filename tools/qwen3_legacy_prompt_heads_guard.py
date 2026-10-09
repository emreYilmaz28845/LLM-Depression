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
HEAD_JOB_FIELDS = ("extract_job_id", "classifier_job_id")
TREATMENT_PLANNER = LANE / "tools/qwen3_legacy_prompt_head_plan.py"
RUNTIME_CACHE_ROOT = (
    "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
    "feat-qwen3-legacy-prompt-20261008/heads_cache"
)


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


def head_recorded_keys(ledger_path: Path = LEDGER) -> set[str]:
    """Every head key already recorded in the append-only ledger, any status.

    A reserved/uncertain/failed head record excludes the key from eligibility
    until explicit evidence-based reconciliation or a deliberate new attempt;
    the guard never reissues an unchanged head key automatically.
    """
    keys: set[str] = set()
    if not ledger_path.exists():
        return keys
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("stage") == "head" and str(record.get("key", "")).startswith("head::"):
            keys.add(str(record["key"])[len("head::") :])
    return keys


def plan_resolved_attempts(plan: dict) -> dict[tuple[str, int, int], str]:
    """Resolved plan entries as ``cell -> exact parent attempt``.

    Real shared dispatch plans omit ``registry_key``; the cell is then
    normalized from the route's ``route_id`` (or ``dataset``) plus the job's
    seed/fold, exactly like ``eligible_head_jobs``. Keeping the parent attempt
    means a replaced attempt (a new validated parent for the same cell) also
    requests a plan refresh.
    """
    resolved: dict[tuple[str, int, int], str] = {}
    for route in plan.get("routes") or []:
        for job in route.get("jobs") or []:
            if job.get("parent_status") != "resolved":
                continue
            cell = parse_cell(job.get("registry_key") or "")
            if cell is None:
                route_id = route.get("route_id") or route.get("dataset")
                if route_id is None or "seed" not in job or "fold" not in job:
                    continue
                cell = (str(route_id), int(job["seed"]), int(job["fold"]))
            resolved[cell] = str((job.get("parent") or {}).get("attempt_id") or "")
    return resolved


def plan_resolved_cells(plan: dict) -> set[tuple[str, int, int]]:
    return set(plan_resolved_attempts(plan))


def plan_needs_refresh(validated: dict[tuple[str, int, int], str], plan: dict | None) -> bool:
    """True when a validated cell lacks a resolved entry or the attempt changed."""
    if plan is None:
        return True
    resolved = plan_resolved_attempts(plan)
    return any(resolved.get(cell) != attempt for cell, attempt in validated.items())


def maybe_refresh_plan(
    validated: dict[tuple[str, int, int], str],
    *,
    plan_path: Path = PLAN,
    matrix_path: Path,
    evidence_dir: Path = EVIDENCE,
    run_root: Path = dispatch.RUN_ROOT,
    runtime_cache_root: str,
    refresh_interval: int = 3600,
) -> bool:
    """Rebuild matrix + dispatch plan when newly validated parents need it.

    Throttled so repeated passes with unresolvable parents (for example while
    adapters are not hashable on this host) do not rebuild every pass. Returns
    True when a rebuild was attempted.
    """
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    if not plan_needs_refresh(validated, plan):
        return False
    stamp = evidence_dir / ".head_plan_refresh_stamp"
    if stamp.exists() and time.time() - stamp.stat().st_mtime < refresh_interval:
        return False
    if not TREATMENT_PLANNER.exists():
        # The generic shared matrix is a control-config/English inventory and
        # must never overwrite the treatment plan; a lane-owned treatment
        # planner is required before any refresh can happen.
        print(
            "lane-owned treatment planner unavailable; plan refresh skipped "
            "(the generic control matrix is never used)"
        )
        return False
    commands = [
        # Step 1: lane-owned treatment-only planner with in-place GPFS proofs.
        [
            sys.executable,
            str(TREATMENT_PLANNER),
            "--emit-matrix",
            str(matrix_path),
            "--run-root",
            str(run_root),
            "--cache-root",
            runtime_cache_root,
        ],
        # Step 2: shared dispatch plan over the treatment-only matrix emitted
        # by step 1. The shared matrix inventory tool is never invoked here.
        [
            sys.executable,
            "tools/qwen3_heads_dispatch.py",
            "plan",
            "--matrix",
            str(matrix_path),
            "--language",
            "native",
        ],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=LANE, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"plan refresh step failed rc={result.returncode}: {command[1]}")
            return False
    stamp.write_text(json.dumps({"ts": int(time.time()), "validated": len(validated)}) + "\n", encoding="utf-8")
    return True


def approved_head_keys(matrix_path: Path = EVIDENCE / "matrix.json") -> set[str]:
    """The exact approved Native head keys for the 189-fit treatment matrix.

    The generic head inventory contains English and other lanes; the executable
    dispatch is restricted to the approved 15 Native routes / 189 cells, tied
    to treatment parents. Keys are normalized to the plan's ``route|seed|fold``.
    """
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for fit in matrix.get("fits") or []:
        cell = parse_cell(fit.get("key") or "")
        if cell is not None:
            keys.add(f"{cell[0]}|{cell[1]}|{cell[2]}")
    return keys


def eligible_head_jobs(
    plan: dict,
    validated: dict[tuple[str, int, int], str],
    submitted_keys: set[str],
    recorded_keys: set[str] | None = None,
    approved_keys: set[str] | None = None,
) -> list[dict]:
    """Resolved plan entries whose exact parent attempt is validated and unrecorded."""
    recorded = recorded_keys or set()
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
            # Shared dispatch plans (and this lane's treatment planner output)
            # do not necessarily carry an explicit registry_key; derive the
            # canonical key from the route/seed/fold triple in that case.
            key = key or f"{cell[0]}|{cell[1]}|{cell[2]}"
            if key in submitted_keys or key in recorded:
                continue
            if approved_keys is not None and key not in approved_keys:
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
    job: dict,
    entry: dict[str, str],
    attempt_id: str | None,
    status: str,
    tail: str = "",
    deployment_id: str | None = None,
) -> dict:
    parsed_ids = {}
    for source_key, target_key in (
        ("extract_job_id", "extract"),
        ("classifier_job_id", "classifier"),
    ):
        value = entry.get(source_key)
        if value is not None and str(value).strip():
            parsed_ids[target_key] = str(value).strip()
    record = {
        "key": f"head::{job['key']}",
        "stage": "head",
        "ts": int(time.time()),
        "status": status,
        "attempt_id": attempt_id,
        "parent_attempt_id": job.get("parent_attempt_id"),
        # Known IDs are always preserved, including on uncertain outcomes, so a
        # potentially delivered head job can never escape the own accounting.
        "job_ids": parsed_ids,
    }
    if deployment_id is not None:
        record["requested_deployment_id"] = deployment_id
    if status != "submitted":
        record["reason"] = "invalid or missing head delivery"
        record["tail"] = tail[-800:]
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


def _registry_records(registry_path: Path = REGISTRY) -> list[dict]:
    """All shared head-registry entries, newest last; missing file yields none."""
    if not registry_path.exists():
        return []
    records: list[dict] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _fresh_registry_delivery(
    key: str,
    before_attempts: set[str],
    job: dict,
    deployment_id: str,
    registry_path: Path = REGISTRY,
) -> dict | None:
    """The single fresh exact registry proof for one head key, or None.

    Freshness is the attempt-id set difference since this call started: the
    shared submit tool mints a new attempt per submission and writes the
    registry entry only after it parsed exact remote job ids. The entry must
    also match the head key, the parent attempt, the requested deployment, and
    carry two distinct numeric job ids.
    """
    fresh = [
        entry
        for entry in _registry_records(registry_path)
        if entry.get("registry_key") == key
        and str(entry.get("attempt_id") or "").strip() not in before_attempts
    ]
    if len(fresh) != 1:
        return None
    entry = fresh[0]
    attempt = str(entry.get("attempt_id") or "").strip()
    if not attempt:
        return None  # a blank attempt is never a fresh delivery proof
    if str(entry.get("deployment_id") or "") != str(deployment_id):
        return None
    expected_parent = str(job.get("parent_attempt_id") or "").strip()
    entry_parent = str(entry.get("parent_attempt_id") or "").strip()
    if not expected_parent or not entry_parent or entry_parent != expected_parent:
        return None
    if entry.get("error"):
        return None
    extract = str(entry.get("extract_job_id") or "")
    classifier = str(entry.get("classifier_job_id") or "")
    if not extract.isdigit() or not classifier.isdigit() or extract == classifier:
        return None
    return entry


def _submitted_from_registry(job: dict, entry: dict, deployment_id: str) -> dict:
    return head_ledger_record(
        job,
        {
            "extract_job_id": str(entry["extract_job_id"]),
            "classifier_job_id": str(entry["classifier_job_id"]),
        },
        str(entry["attempt_id"]),
        "submitted",
        deployment_id=deployment_id,
    )


def submit_head_via_shared(
    job: dict,
    deployment_id: str,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    registry_path: Path = REGISTRY,
) -> dict:
    """Submit one head chain and prove delivery from the shared registry.

    The shared CLI prints a human summary; its exact delivery proof is the
    registry entry (key, fresh attempt, parent attempt, deployment, two
    distinct numeric job ids). A fresh registry entry is authoritative even
    when stdout parsing or the return code say otherwise; without it the
    delivery is uncertain and the key stays reserved.
    """
    before = {
        str(entry.get("attempt_id") or "").strip()
        for entry in _registry_records(registry_path)
        if entry.get("registry_key") == job["key"]
    }

    def proof() -> dict | None:
        return _fresh_registry_delivery(
            job["key"], before, job, deployment_id, registry_path
        )

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
    runner = runner or subprocess.run
    partial = ""
    result: subprocess.CompletedProcess | None = None
    try:
        result = runner(command, cwd=LANE, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        partial = _text(getattr(exc, "output", None) or getattr(exc, "stdout", ""))
        partial += "\n" + _text(getattr(exc, "stderr", ""))
    except Exception as exc:  # fail closed, but still consume a fresh proof
        partial = f"{type(exc).__name__}: {exc}"

    fresh = proof()
    if fresh is not None:
        return _submitted_from_registry(job, fresh, deployment_id)
    # No exact fresh registry proof: the delivery stays uncertain even when the
    # stdout happens to look structured. Known IDs from stdout are preserved
    # for ownership, never promoted to submitted without the registry proof.
    if result is not None:
        stdout = result.stdout or ""
        entry = parse_head_submit_output(stdout).get(job["key"]) or {}
        return head_ledger_record(
            job,
            entry,
            _latest_attempt(job["key"], registry_path),
            "uncertain",
            tail=stdout + (result.stderr or ""),
            deployment_id=deployment_id,
        )
    entry = parse_head_submit_output(partial).get(job["key"]) or {}
    return head_ledger_record(
        job,
        entry,
        _latest_attempt(job["key"], registry_path),
        "uncertain",
        tail=partial,
        deployment_id=deployment_id,
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _latest_attempt(key: str, registry_path: Path = REGISTRY) -> str | None:
    latest = None
    for entry in _registry_records(registry_path):
        if entry.get("registry_key") == key:
            latest = entry.get("attempt_id")
    return latest


def reconcile_delivered_heads(
    ledger_path: Path = LEDGER,
    registry_path: Path = REGISTRY,
) -> dict[str, int]:
    """Record already-delivered head chains proven by the shared registry.

    One-shot recovery for uncertain records whose delivery actually succeeded
    (the shared CLI wrote exact ids to ``head_submissions.jsonl`` while the
    guard parser only saw a summary). It never reissues a job: it appends a
    ``submitted`` record with the exact stored attempt/id pair for the same
    key and attempt. History stays append-only and the key stays reserved.
    """
    records = dispatch.ledger_records(ledger_path)
    latest: dict[str, dict] = {}
    order: list[str] = []
    for record in records:
        if record.get("stage") != "head":
            continue
        key = str(record.get("key") or "")
        if not key.startswith("head::"):
            continue
        if key not in latest:
            order.append(key)
        latest[key] = record
    registry = _registry_records(registry_path)
    summary = {"reconciled": 0, "already_submitted": 0, "unresolved": 0}
    for key in order:
        record = latest[key]
        if record.get("status") == "submitted":
            summary["already_submitted"] += 1
            continue
        cell_key = key[len("head::") :]
        attempt = str(record.get("attempt_id") or "").strip()
        parent_attempt = str(record.get("parent_attempt_id") or "").strip()
        if not attempt or not parent_attempt:
            # Missing identity can never be reconciled by matching empty strings.
            summary["unresolved"] += 1
            continue
        requested_deployment = record.get("requested_deployment_id")
        candidates = []
        for entry in registry:
            if str(entry.get("registry_key") or "") != cell_key or entry.get("error"):
                continue
            if str(entry.get("attempt_id") or "").strip() != attempt:
                continue
            if str(entry.get("parent_attempt_id") or "").strip() != parent_attempt:
                continue
            if requested_deployment is not None and str(
                entry.get("deployment_id") or ""
            ) != str(requested_deployment):
                continue
            extract = str(entry.get("extract_job_id") or "")
            classifier = str(entry.get("classifier_job_id") or "")
            if extract.isdigit() and classifier.isdigit() and extract != classifier:
                candidates.append(entry)
        # Exactly one consistent proof: same key, same attempt, same parent,
        # matching requested deployment when the record carries one.
        if len(candidates) != 1:
            summary["unresolved"] += 1
            continue
        candidate = candidates[0]
        job = {"key": cell_key, "parent_attempt_id": record.get("parent_attempt_id")}
        delivered = head_ledger_record(
            job,
            {
                "extract_job_id": str(candidate["extract_job_id"]),
                "classifier_job_id": str(candidate["classifier_job_id"]),
            },
            attempt,
            "submitted",
            deployment_id=(
                str(candidate.get("deployment_id"))
                if candidate.get("deployment_id")
                else None
            ),
        )
        delivered["reconciled"] = True
        delivered["reconciliation_source"] = "head_submissions.jsonl exact registry proof"
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(delivered) + "\n")
        summary["reconciled"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fits", type=int, default=40)
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reconcile-heads", action="store_true")
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.reconcile_heads:
        with dispatch.acquire_submit_lock():
            summary = reconcile_delivered_heads()
        print("head reconciliation:", json.dumps(summary, sort_keys=True))
        return 0
    if not args.deployment_id:
        parser.error("--deployment-id is required for submission passes")

    with dispatch.acquire_submit_lock():
        validated = validated_cells()
        maybe_refresh_plan(
            validated,
            plan_path=PLAN,
            matrix_path=EVIDENCE / "heads_matrix.json",
            runtime_cache_root=RUNTIME_CACHE_ROOT,
        )
        plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {"routes": []}
        heads = eligible_head_jobs(
            plan,
            validated,
            registry_submitted_keys(),
            head_recorded_keys(),
            approved_head_keys(),
        )
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
