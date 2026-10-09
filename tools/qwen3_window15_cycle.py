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
HEAD_VALIDATED_STATES = {"LOCALLY_VALIDATED", "REPORTABLE"}


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
    """Newest submitted attempt per training fit key."""
    latest: dict[str, dict] = {}
    path = ledger_path or LEDGER
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("key") or "")
        if key.startswith("head:") or key.startswith("seed:") or not key:
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


def phase_collect_fits(max_fits: int, dry_run: bool) -> dict:
    attempts = _fit_attempts()
    completed: list[dict] = []
    for key, record in sorted(attempts.items()):
        attempt_id = str(record["attempt_id"])
        ids = _fit_job_ids(attempt_id)
        if not ids:
            continue
        try:
            accounting = query_top_level_accounting(sorted(ids.values()))
        except AdmissionError as exc:
            return {"status": f"accounting failed: {exc}"}
        if all(
            accounting.get(job_id, {}).get("state") == "COMPLETED"
            and accounting.get(job_id, {}).get("exit") == "0:0"
            for job_id in ids.values()
        ):
            completed.append({"key": key, "attempt_id": attempt_id, "ids": ids})
    if dry_run:
        return {"status": "dry-run", "completed_fits": len(completed)}
    collected = validated = 0
    results: list[dict] = []
    for item in completed[:max_fits]:
        collect = _run_exp(
            ["collect", "feat-qwen3-window15-20261008", "--attempt-id", item["attempt_id"], "--execute"]
        )
        if collect.returncode != 0:
            results.append({"attempt_id": item["attempt_id"], "collect": f"rc={collect.returncode}"})
            continue
        collected += 1
        validate = _run_exp(["validate", "--attempt-id", item["attempt_id"]])
        if validate.returncode == 0:
            validated += 1
            results.append({"attempt_id": item["attempt_id"], "collect": "ok", "validate": "ok"})
        else:
            results.append({"attempt_id": item["attempt_id"], "collect": "ok", "validate": f"rc={validate.returncode}"})
    return {
        "status": "ok",
        "completed_fits": len(completed),
        "collected": collected,
        "validated": validated,
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


def phase_head_submit(max_head_fits: int, dry_run: bool) -> dict:
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


def phase_head_collect(max_head_fits: int, dry_run: bool) -> dict:
    done: list[dict] = []
    for entry in head_registry_entries():
        attempt_id = str(entry.get("attempt_id") or "")
        extract = str(entry.get("extract_job_id") or "")
        classifier = str(entry.get("classifier_job_id") or "")
        if not attempt_id or not extract.isdigit() or not classifier.isdigit():
            continue
        mirror = EVIDENCE / "head_attempts" / attempt_id
        state = _local_state(mirror) if mirror.is_dir() else None
        if state in HEAD_VALIDATED_STATES:
            continue
        try:
            accounting = query_top_level_accounting([extract, classifier])
        except AdmissionError as exc:
            return {"status": f"accounting failed: {exc}"}
        if all(
            accounting.get(job_id, {}).get("state") == "COMPLETED"
            and accounting.get(job_id, {}).get("exit") == "0:0"
            for job_id in (extract, classifier)
        ):
            done.append({"attempt_id": attempt_id})
    if dry_run:
        return {"status": "dry-run", "pending_head_attempts": len(done)}
    collected = validated = 0
    for item in done[:max_head_fits]:
        collect = _run(
            [
                sys.executable, "tools/qwen3_window15_heads_guard.py", "collect",
                "--rest", "--attempt-id", item["attempt_id"],
            ]
        )
        if collect.returncode != 0:
            continue
        collected += 1
        validate = _run(
            [
                sys.executable, "tools/qwen3_window15_heads_guard.py", "validate",
                "--rest", "--attempt-id", item["attempt_id"],
            ]
        )
        if validate.returncode == 0:
            validated += 1
    return {"status": "ok", "pending_head_attempts": len(done), "collected": collected, "validated": validated}


def phase_coverage(dry_run: bool) -> dict:
    if dry_run:
        return {"status": "dry-run"}
    result = _run(
        [sys.executable, "tools/qwen3_window15_heads_guard.py", "coverage", "--rest"]
    )
    return {"status": "ok" if result.returncode == 0 else f"rc={result.returncode}"}


def run_cycle(args: argparse.Namespace) -> int:
    record: dict = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    phases = (
        ("reconcile", lambda: phase_reconcile(args.dry_run)),
        ("fit_collect_validate", lambda: phase_collect_fits(args.max_fits, args.dry_run)),
        ("matrix_rebuild", lambda: phase_matrix(args.dry_run)),
        ("head_submit", lambda: phase_head_submit(args.max_head_fits, args.dry_run)),
        ("head_collect_validate", lambda: phase_head_collect(args.max_head_fits, args.dry_run)),
        ("coverage", lambda: phase_coverage(args.dry_run)),
    )
    for name, func in phases:
        try:
            record[name] = func()
        except Exception as exc:  # every phase stays bounded and reported
            record[name] = {"status": f"error {type(exc).__name__}: {exc}"}
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
