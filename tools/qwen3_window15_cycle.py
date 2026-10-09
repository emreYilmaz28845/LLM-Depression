#!/usr/bin/env python3
"""Incremental window15 cycle driver: fits -> matrix -> heads.

One bounded cycle, idempotent and safe to run from the durable watcher:

1. **reconcile**: ``exp.py status`` reconciles the lane's exact attempts and
   mirrors terminal scheduler evidence into the canonical fold sidecars through
   the official append-only API (sidecars may lag while jobs already completed;
   the generic parent resolver tolerates a lagging lifecycle pointer when the
   job history is clean, so this step is what keeps the resolver honest).
2. **fit collect + validate**: for each newest submitted fit whose exact
   top-level train and best_eval jobs are live COMPLETED ``0:0``, run the
   official compact collection and validation so the parent becomes eligible.
3. **matrix rebuild**: rebuild the planner audit and the audit-filtered eligible
   parent map, then run the generic remote parent resolver to emit a fresh
   matrix. The matrix is rebuilt every cycle because an older one can still
   carry unresolved parents.
4. **guarded head submit**: only when at least one eligible, unblocked key
   exists; the guard acquires the shared lane submission lock itself, so this
   driver never holds the lock (no nested locking).
5. **head collect + validate**: for head attempts whose extract and classifier
   jobs are COMPLETED ``0:0``, run the compact collection and official head
   validation.
6. **coverage + cycle record**: audit every planned key and append one JSON
   record to ``cycle_log.jsonl``.

``--dry-run`` performs no mutation and prints the plan of the cycle.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_window15_dispatch import (  # noqa: E402
    AdmissionError,
    EXECUTION_LEDGER,
    EXP_SUBMIT_ROOT,
    LEDGER,
    exec_ledger_lane_jobs,
    head_registry_entries,
)
from tools.qwen3_window15_heads_guard import query_top_level_accounting  # noqa: E402
from tools.qwen3_window15_heads_plan import (  # noqa: E402
    CONTRACT,
    OUT_DIR,
    build_plan,
    eligible_parent_map,
)

EVIDENCE = LANE / "outputs/qwen3_window15_20261008"
CYCLE_LOG = EVIDENCE / "cycle_log.jsonl"
RUN_ROOT = LANE / "output_model/qwen3_window15_20261008"
REMOTE_RUNTIME = "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/feat-qwen3-window15-20261008"
CAMPAIGN_REMOTE = "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/qwen3_window15_20261008"
VALIDATED_STATES = {"LOCALLY_VALIDATED", "REPORTABLE"}


class CycleError(RuntimeError):
    pass


def _latest_head_deployment() -> str:
    deploys = sorted(
        path for path in (LANE / "outputs/exp_deploy").glob("feat-qwen3-window15-*") if path.is_dir()
    )
    if not deploys:
        raise CycleError("no lane deployment record found")
    record = json.loads((deploys[-1] / "deployment.json").read_text(encoding="utf-8"))
    return str(record["deployment_id"])


def _fit_attempts(ledger_path: Path | None = None) -> dict[str, dict]:
    """Newest submitted attempt per treatment fit key (smokes excluded)."""
    latest: dict[str, dict] = {}
    path = ledger_path or LEDGER
    if not path.is_file():
        return latest
    _, audit = build_plan(CONTRACT, path)
    treatment_keys = {item["key"] for item in audit["keys"]}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("key") or "")
        if key not in treatment_keys:
            continue
        if record.get("status") == "submitted" and record.get("attempt_id"):
            latest[key] = record
    return latest


def _fit_job_ids(attempt_id: str, exec_ledger_path: Path | None = None) -> dict[str, str] | None:
    """Exact train/best_eval ids from the authoritative execution ledger."""
    jobs = exec_ledger_lane_jobs(exec_ledger_path or EXECUTION_LEDGER)
    ids = {
        str(job.get("job_key")): str(job.get("slurm_job_id"))
        for job in jobs
        if str(job.get("attempt_id")) == attempt_id
        and str(job.get("job_key")) in {"train", "best_eval"}
        and str(job.get("slurm_job_id") or "").strip()
    }
    if set(ids) != {"train", "best_eval"}:
        return None
    if not all(re.fullmatch(r"\d+", value) for value in ids.values()):
        return None
    return ids


def _local_state(fold_dir: Path) -> str | None:
    path = fold_dir / "status.json"
    if not path.is_file():
        return None
    return str((json.loads(path.read_text(encoding="utf-8")) or {}).get("state") or "") or None


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=LANE, capture_output=True, text=True, timeout=timeout)


def _run_exp(args: list[str]) -> subprocess.CompletedProcess:
    return _run([sys.executable, "tools/exp.py", *args])


def phase_reconcile(dry_run: bool) -> dict:
    if dry_run:
        return {"status": "dry-run"}
    result = _run_exp(["status", "feat-qwen3-window15-20261008"])
    return {"status": "ok" if result.returncode == 0 else f"rc={result.returncode}"}


DONE_STATES = {"LOCALLY_VALIDATED", "REPORTABLE"}
# COMPLETED_ON_MN5 is NOT collection proof; SYNCED_LOCALLY means collection is done.
SKIP_COLLECT_STATES = {"SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"}


def _expected_fold_dir(item: dict) -> Path:
    return (
        RUN_ROOT
        / str(item["modality"])
        / str(item["dataset"])
        / str(item["run_name"])
        / f"fold_{int(item['fold'])}"
    )


def _confirmed_state(fold_dir: Path, attempt: str) -> tuple[bool, str | None]:
    """Exact local lifecycle state plus attempt identity for one fold."""
    state = _local_state(fold_dir)
    metadata_path = fold_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    confirmed = state in DONE_STATES and str(metadata.get("attempt_id") or "") == attempt
    return confirmed, state


def phase_collect_fits(max_fits: int, dry_run: bool) -> dict:
    attempts = _fit_attempts()
    _, audit = build_plan(CONTRACT, LEDGER)
    audit_by_key = {item["key"]: item for item in audit["keys"]}
    units: list[dict] = []
    skipped_validated = 0
    errors: list[dict] = []
    for key, record in sorted(attempts.items()):
        item = audit_by_key.get(key)
        if item is None:
            continue
        fold_dir = _expected_fold_dir(item)
        state = _local_state(fold_dir) if fold_dir.is_dir() else None
        if state in DONE_STATES:
            skipped_validated += 1
            continue
        try:
            ids = _fit_job_ids(str(record["attempt_id"]))
            if not ids:
                continue
            accounting = query_top_level_accounting(sorted(ids.values()))
        except AdmissionError as exc:
            errors.append({"key": key, "error": str(exc)})
            continue  # bounded: other ready units continue
        if all(
            accounting.get(job_id, {}).get("state") == "COMPLETED"
            and accounting.get(job_id, {}).get("exit") == "0:0"
            for job_id in ids.values()
        ):
            units.append(
                {
                    "key": key,
                    "attempt_id": str(record["attempt_id"]),
                    "ids": ids,
                    "needs_collect": state not in SKIP_COLLECT_STATES,
                }
            )
    if dry_run:
        return {
            "status": "dry-run",
            "completed_fits": len(units),
            "skipped_validated": skipped_validated,
            "errors": errors,
        }
    collected = validated = 0
    results: list[dict] = []
    for unit in units[:max_fits]:
        result = {"attempt_id": unit["attempt_id"], "key": unit["key"], "collect_skipped": not unit["needs_collect"]}
        try:
            if unit["needs_collect"]:
                collect = _run_exp(
                    ["collect", "feat-qwen3-window15-20261008", "--attempt-id", unit["attempt_id"], "--execute"]
                )
                result["collect_rc"] = collect.returncode
                if collect.returncode != 0:
                    result["stage"] = "blocked_collect"
                    results.append(result)
                    continue
                collected += 1
                # ONE official status reconcile AFTER collection before validation
                _run_exp(["status", "feat-qwen3-window15-20261008"])
            validate = _run_exp(["validate", "--attempt-id", unit["attempt_id"]])
            result["validate_rc"] = validate.returncode
            item = audit_by_key.get(unit["key"])
            fold_dir = _expected_fold_dir(item) if item else Path("/nonexistent")
            confirmed, state = _confirmed_state(fold_dir, unit["attempt_id"])
            result["final_state"] = state
            if validate.returncode == 0 and confirmed:
                validated += 1
                result["stage"] = "validated"
            else:
                result["stage"] = "blocked_validate"
        except Exception as exc:  # bounded: one unit cannot abort the cycle
            result["stage"] = f"error {type(exc).__name__}: {exc}"
        results.append(result)
    return {
        "status": "ok",
        "completed_fits": len(units),
        "skipped_validated": skipped_validated,
        "collected": collected,
        "validated": validated,
        "errors": errors,
        "results": results[-8:],
    }


def phase_matrix(dry_run: bool) -> dict:
    parent_map, audit = build_plan(CONTRACT, LEDGER)
    filtered = eligible_parent_map(parent_map, audit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered_path = OUT_DIR / "parent_map.eligible.json"
    filtered_path.write_text(json.dumps(filtered, indent=1, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "heads_plan_audit.json").write_text(json.dumps(audit, indent=1, sort_keys=True), encoding="utf-8")
    if dry_run:
        return {"status": "dry-run", "eligible": len(filtered["entries"]), "audit": audit["status_counts"]}
    deployment_id = _latest_head_deployment()
    deployment_dir = LANE / "outputs/exp_deploy" / deployment_id
    code_root = f"/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/{deployment_id}/code"
    scp = _run(
        [
            "scp", "-q", str(filtered_path),
            f"ozu647717@transfer1.bsc.es:{REMOTE_RUNTIME}/planning/window15_parent_map_eligible.json",
        ],
        timeout=600,
    )
    if scp.returncode != 0:
        return {"status": f"scp rc={scp.returncode}"}
    remote = (
        f"source /gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate && "
        f"cd {code_root} && "
        f"python tools/qwen3_window15_heads_plan.py --build-matrix "
        f"--contract {REMOTE_RUNTIME}/planning/treatment_contract.json "
        f"--parent-map {REMOTE_RUNTIME}/planning/window15_parent_map_eligible.json "
        f"--parent-map-pre-filtered "
        f"--matrix-out {REMOTE_RUNTIME}/head_cache/window15_heads_matrix.json "
        f"--cache-root {REMOTE_RUNTIME}/head_cache "
        f"--campaign-root {CAMPAIGN_REMOTE} --seed 7 --seed 1337 --seed 2024"
    )
    build = _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "ozu647717@alogin2.bsc.es", remote], timeout=1800)
    if build.returncode != 0:
        return {"status": "remote build failed", "tail": build.stderr.strip()[-200:]}
    pull = _run(
        [
            "scp", "-q",
            f"ozu647717@transfer1.bsc.es:{REMOTE_RUNTIME}/head_cache/window15_heads_matrix.json",
            str(OUT_DIR / "window15_heads_matrix.json"),
        ],
        timeout=600,
    )
    if pull.returncode != 0:
        return {"status": f"matrix pull rc={pull.returncode}"}
    matrix = json.loads((OUT_DIR / "window15_heads_matrix.json").read_text(encoding="utf-8"))
    return {"status": "ok", "eligible": len(filtered["entries"]), "summary": matrix.get("summary", {})}


def phase_head_submit(max_head_fits: int, dry_run: bool, matrix_ok: bool = True) -> dict:
    if not matrix_ok:
        return {"status": "skipped: fresh matrix rebuild failed; previous matrix not reused"}
    matrix_path = OUT_DIR / "window15_heads_matrix.json"
    if not matrix_path.is_file():
        return {"status": "no matrix"}
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    resolved = int((matrix.get("summary") or {}).get("resolved") or 0)
    if resolved < 1:
        return {"status": "no resolved parents", "resolved": 0}
    deployment_id = _latest_head_deployment()
    base = [
        sys.executable, "tools/qwen3_window15_heads_guard.py", "submit",
        "--matrix", str(matrix_path),
        "--deployment-id", deployment_id,
        "--limit", str(max_head_fits),
    ]
    if dry_run:
        result = _run(base + ["--dry-run"])
        return {"status": "dry-run", "rc": result.returncode, "tail": result.stdout.strip()[-200:]}
    result = _run(base)
    return {"status": "ok" if result.returncode == 0 else f"rc={result.returncode}", "tail": result.stdout.strip()[-200:]}


def _head_confirmed(mirror: Path, attempt: str) -> tuple[bool, str | None]:
    state = _local_state(mirror)
    metadata_path = mirror / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("attempt_id") or "") not in ("", attempt):
            return False, state
    return state in DONE_STATES, state


def phase_head_collect(max_head_fits: int, dry_run: bool) -> dict:
    units: list[dict] = []
    errors: list[dict] = []
    for entry in head_registry_entries():
        attempt_id = str(entry.get("attempt_id") or "")
        extract = str(entry.get("extract_job_id") or "")
        classifier = str(entry.get("classifier_job_id") or "")
        if not attempt_id or not extract.isdigit() or not classifier.isdigit():
            continue
        mirror = EVIDENCE / "head_attempts" / attempt_id
        state = _local_state(mirror) if mirror.is_dir() else None
        if state in DONE_STATES:
            continue
        try:
            accounting = query_top_level_accounting([extract, classifier])
        except AdmissionError as exc:
            errors.append({"attempt_id": attempt_id, "error": str(exc)})
            continue  # bounded: other ready units continue
        if all(
            accounting.get(job_id, {}).get("state") == "COMPLETED"
            and accounting.get(job_id, {}).get("exit") == "0:0"
            for job_id in (extract, classifier)
        ):
            units.append({"attempt_id": attempt_id, "needs_collect": state not in SKIP_COLLECT_STATES})
    if dry_run:
        return {"status": "dry-run", "pending_head_attempts": len(units), "errors": errors}
    collected = validated = 0
    results: list[dict] = []
    for unit in units[:max_head_fits]:
        result = {"attempt_id": unit["attempt_id"], "collect_skipped": not unit["needs_collect"]}
        try:
            if unit["needs_collect"]:
                collect = _run(
                    [
                        sys.executable, "tools/qwen3_window15_heads_guard.py", "collect",
                        "--rest", "--attempt-id", unit["attempt_id"],
                    ]
                )
                result["collect_rc"] = collect.returncode
                if collect.returncode != 0:
                    result["stage"] = "blocked_collect"
                    results.append(result)
                    continue
                collected += 1
            validate = _run(
                [
                    sys.executable, "tools/qwen3_window15_heads_guard.py", "validate",
                    "--rest", "--attempt-id", unit["attempt_id"],
                ]
            )
            result["validate_rc"] = validate.returncode
            mirror = EVIDENCE / "head_attempts" / unit["attempt_id"]
            confirmed, state = _head_confirmed(mirror, unit["attempt_id"])
            result["final_state"] = state
            if validate.returncode == 0 and confirmed:
                validated += 1
                result["stage"] = "validated"
            else:
                result["stage"] = "blocked_validate"
        except Exception as exc:  # bounded: one unit cannot abort the cycle
            result["stage"] = f"error {type(exc).__name__}: {exc}"
        results.append(result)
    return {
        "status": "ok",
        "pending_head_attempts": len(units),
        "collected": collected,
        "validated": validated,
        "errors": errors,
        "results": results[-8:],
    }


def phase_coverage(dry_run: bool) -> dict:
    if dry_run:
        return {"status": "dry-run"}
    output = EVIDENCE / "heads_coverage_audit.json"
    result = _run(
        [
            sys.executable, "tools/qwen3_window15_heads_guard.py", "coverage",
            "--rest", "--output", str(output),
        ]
    )
    return {"status": "ok" if result.returncode == 0 else f"rc={result.returncode}", "output": str(output)}


def run_cycle(args: argparse.Namespace) -> int:
    record: dict = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def bounded(name: str, func) -> None:
        try:
            record[name] = func()
        except Exception as exc:  # every phase stays bounded and reported
            record[name] = {"status": f"error {type(exc).__name__}: {exc}"}

    bounded("reconcile", lambda: phase_reconcile(args.dry_run))
    bounded("fit_collect_validate", lambda: phase_collect_fits(args.max_fits, args.dry_run))
    bounded("matrix_rebuild", lambda: phase_matrix(args.dry_run))
    matrix_ok = record["matrix_rebuild"].get("status") == "ok"
    bounded(
        "head_submit",
        lambda: phase_head_submit(args.max_head_fits, args.dry_run, matrix_ok=matrix_ok),
    )
    bounded("head_collect_validate", lambda: phase_head_collect(args.max_head_fits, args.dry_run))
    bounded("coverage", lambda: phase_coverage(args.dry_run))
    print(json.dumps(record, sort_keys=True))
    if not args.dry_run:
        CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CYCLE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one bounded cycle")
    run.add_argument("--max-fits", type=int, default=4)
    run.add_argument("--max-head-fits", type=int, default=4)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=run_cycle)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
