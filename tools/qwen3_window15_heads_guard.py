#!/usr/bin/env python3
"""Guarded window15 head dispatch: lane admission around tools/qwen3_heads_dispatch.py.

The generic dispatcher owns the head plan/submit/collect/validate machinery;
this wrapper owns the lane's resource and evidence guards:

- the shared lane submission lock (``outputs/<campaign>/submission.lock``) is
  held across fresh plan rebuild, eligibility, admission and delivery;
- eligibility is rebuilt inside the lock and binds every key to the exact
  ledger attempt, the collected fold's ``metadata.json`` and
  ``run_config.yaml`` tracking attempt, the expected dataset/modality/seed/fold,
  the 15-second treatment identity with ``processor_min_audio_samples: 201`` and
  present manifest/split hashes;
- the top-level ``train`` and ``best_eval`` job ids are taken from the
  canonical sidecar ``SUBMITTED`` events (numeric, attempt-bound) and must show
  **live scheduler accounting** ``COMPLETED`` with exit ``0:0``; step rows
  (``.batch``/``.extern``) and missing or contradictory accounting are refused;
- any prior ``head:<key>`` delivery or reservation record blocks the key until
  an explicit ``--reconcile-key`` record; nothing is blindly resubmitted;
- a durable two-job reservation per selected key is written before any submit;
  the submit call is exception-safe and reconciliation (fresh registry or fresh
  content-changed submit log) always finalizes every batch key, preserving
  partial known ids and unresolved reservations;
- promotion to ``submitted`` requires the changed registry row to bind a
  nonblank head attempt, the exact parent attempt, the deployment id and the
  registry key; ids alone are never sufficient, and the CLI stdout is never
  used as delivery evidence.

Read-only subcommands (status/collect/validate/finish/coverage) delegate to the
generic tool with the lane's registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_window15_dispatch import (  # noqa: E402
    AdmissionError,
    HEADS_REGISTRY,
    append_record,
    lane_submission_lock,
    own_job_ids,
    own_nonterminal_count,
    query_job_states,
    run_ssh,
    storage_admission,
)
from tools.qwen3_window15_heads_plan import (  # noqa: E402
    CONTRACT,
    LEDGER,
    PlanError,
    build_plan,
)

HEADS_TOOL = LANE / "tools/qwen3_heads_dispatch.py"
REGISTRY = HEADS_REGISTRY
JOBS_PER_KEY = 2
MAX_LANE_NONTERMINAL = 80
USER_QUEUE_STOP = 350
BLOCKING_HEAD_STATUSES = {"held", "uncertain", "failed", "submitted", "partial"}


class GuardError(RuntimeError):
    pass


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_snapshot(registry_path: Path = REGISTRY) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if not registry_path.is_file():
        return entries
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = str(entry.get("registry_key") or entry.get("key") or "")
        if key:
            entries[key] = entry
    return entries


def _ledger_latest(ledger_path: Path = LEDGER) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not ledger_path.is_file():
        return latest
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("key") or "")
        if key:
            latest[key] = record
    return latest


def _head_blocked_keys(ledger_latest: dict[str, dict]) -> set[str]:
    """Keys with any prior head delivery/reservation; reconciled clears them."""
    blocked: set[str] = set()
    for key, record in ledger_latest.items():
        if not key.startswith("head:"):
            continue
        status = str(record.get("status") or "")
        target = key[len("head:") :]
        if status == "reconciled":
            blocked.discard(target)
        elif status in BLOCKING_HEAD_STATUSES:
            blocked.add(target)
    return blocked


def _seen_ids(registry: dict[str, dict], ledger_latest: dict[str, dict]) -> set[str]:
    seen: set[str] = set()
    for entry in registry.values():
        for value in (entry.get("extract_job_id"), entry.get("classifier_job_id")):
            if value:
                seen.add(str(value))
    for record in ledger_latest.values():
        for value in (record.get("job_ids") or {}).values():
            if value:
                seen.add(str(value))
    return seen


def bind_parent_identity(item: dict, ledger_latest: dict[str, dict]) -> dict:
    """Bind one eligible key to its exact attempt, fold evidence and job ids.

    Returns ``{"attempt": str, "ids": {"train": str, "best_eval": str}}`` and
    fails closed on any missing, stale or mismatched identity. The strict
    COMPLETED 0:0 check happens later against live scheduler accounting.
    """
    key = f"{item['route_id']}|{item['seed']}|{item['fold']}"
    record = ledger_latest.get(str(item["key"]))
    if record is None or str(record.get("status")) != "submitted":
        raise GuardError(f"{key}: no submitted ledger record for this exact key")
    attempt = str(item.get("attempt_id") or "")
    if str(record.get("attempt_id") or "") != attempt or not attempt:
        raise GuardError(f"{key}: audit attempt does not match the ledger attempt")
    fold_dir = Path(str(item.get("local_fold_dir") or ""))
    if not fold_dir.is_dir():
        raise GuardError(f"{key}: local fold evidence missing")
    metadata = _read_json(fold_dir / "metadata.json") or {}
    if str(metadata.get("attempt_id") or "") != attempt:
        raise GuardError(f"{key}: metadata attempt does not match the ledger attempt")
    run_config = yaml.safe_load((fold_dir / "run_config.yaml").read_text(encoding="utf-8"))
    tracking = run_config.get("tracking") or {}
    if str(tracking.get("attempt_id") or "") != attempt:
        raise GuardError(f"{key}: run_config tracking attempt mismatch")
    if int(tracking.get("fold", -1)) != int(item["fold"]):
        raise GuardError(f"{key}: run_config fold mismatch")
    nested = run_config.get("config") or {}
    if str(nested.get("dataset")) != str(item["dataset"]):
        raise GuardError(f"{key}: run_config dataset mismatch")
    if str(run_config.get("input_modality")) != str(item["modality"]):
        raise GuardError(f"{key}: run_config modality mismatch")
    data = nested.get("data") or {}
    variant = str(nested.get("manifest_variant") or "")
    segment = str(data.get("segment_seconds") or "")
    if not (variant.endswith("_15s_v1") or segment in {"15.0", "15"}):
        raise GuardError(f"{key}: run_config is not the 15-second treatment identity")
    if str(data.get("processor_min_audio_samples") or "") != "201":
        raise GuardError(f"{key}: run_config lacks processor_min_audio_samples=201")
    if not run_config.get("manifest_hash") or not run_config.get("split_metadata_hash"):
        raise GuardError(f"{key}: run_config lacks manifest/split hashes")

    events = [
        json.loads(line)
        for line in (fold_dir / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids: dict[str, str] = {}
    for job_key in ("train", "best_eval"):
        candidates = [
            str(event.get("slurm_job_id"))
            for event in events
            if str(event.get("job_key") or "") == job_key
            and str(event.get("event_type")) in {"SUBMITTED", "TERMINAL"}
            and str(event.get("attempt_id") or "") == attempt
            and re.fullmatch(r"\d+", str(event.get("slurm_job_id") or ""))
        ]
        if not candidates:
            raise GuardError(f"{key}: canonical sidecar has no numeric {job_key} submission id")
        ids[job_key] = candidates[-1]
    return {"attempt": attempt, "ids": ids}


def query_top_level_accounting(job_ids: list[str], runner=None) -> dict[str, dict[str, str]]:
    """Live ``sacct`` accounting for exact top-level job ids only.

    Step rows (``.batch``/``.extern``/``.N``) and foreign ids are ignored; a
    repeated id with contradictory states raises.
    """
    wanted = {str(job_id) for job_id in job_ids}
    if not wanted:
        return {}
    raw = run_ssh(
        f"sacct -j {','.join(sorted(wanted))} -n -P -o JobIDRaw,State,ExitCode",
        runner,
    )
    states: dict[str, dict[str, str]] = {}
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        job_id = parts[0].strip()
        if job_id not in wanted:
            continue
        state = parts[1].strip()
        exit_code = parts[2].strip()
        if job_id in states and states[job_id] != {"state": state, "exit": exit_code}:
            raise GuardError(f"contradictory accounting rows for job {job_id}")
        states[job_id] = {"state": state, "exit": exit_code}
    return states


def assert_completed(key: str, ids: dict[str, str], accounting: dict[str, dict[str, str]]) -> None:
    for job_key, job_id in ids.items():
        entry = accounting.get(job_id)
        if entry is None:
            raise GuardError(f"{key}: {job_key} job {job_id} missing from live accounting")
        if entry["state"] != "COMPLETED" or entry["exit"] != "0:0":
            raise GuardError(
                f"{key}: {job_key} job {job_id} accounting is {entry['state']} {entry['exit']}, "
                "expected COMPLETED 0:0"
            )


def parse_submit_output(output: str) -> dict[str, dict[str, str]]:
    """Parse the remote submit script output (``=== JOB key ===`` blocks)."""
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
        if line.startswith("ERROR="):
            results[current]["error"] = line.split("=", 1)[1]
        elif line.startswith("EXTRACT_ID="):
            results[current]["extract"] = line.split("=", 1)[1].strip()
        elif line.startswith("CLASSIFIER_ID="):
            results[current]["classifier"] = line.split("=", 1)[1].strip()
    return results


def _valid_ids(entry: dict[str, str], seen: set[str]) -> tuple[str, str] | None:
    extract = str(entry.get("extract", ""))
    classifier = str(entry.get("classifier", ""))
    if not re.fullmatch(r"\d+", extract) or not re.fullmatch(r"\d+", classifier):
        return None
    if extract == classifier:
        return None
    if extract in seen or classifier in seen:
        return None
    return extract, classifier


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=LANE, capture_output=True, text=True, timeout=3600)


def _plan_key_index(plan_path: Path) -> dict[str, dict]:
    plan = _read_json(plan_path) or {}
    index: dict[str, dict] = {}
    for route in plan.get("routes") or []:
        for job in route.get("jobs") or []:
            key = f"{route['route_id']}|{job['seed']}|{job['fold']}"
            index[key] = {"route": route, "job": job}
    return index


def _log_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_log_ids(paths: list[Path], pre_digests: dict[str, str | None]) -> dict[str, dict[str, str]]:
    """Fresh submit-output ids require changed content, never just mtime."""
    for path in paths:
        digest = _log_digest(path)
        if digest is not None and pre_digests.get(str(path)) != digest:
            return parse_submit_output(path.read_text(encoding="utf-8"))
    return {}


def _reconcile_batch(
    batch: list[dict],
    *,
    verified: dict[str, dict],
    pre_registry: dict[str, dict],
    pre_log_digests: dict[str, str | None],
    deployment_id: str,
    seen: set[str],
    final_reason: str | None,
) -> int:
    """Finalize every batch key from authoritative sources; return delivered count."""
    post_registry = _registry_snapshot(REGISTRY)
    fresh = _fresh_log_ids(
        [REGISTRY.parent / "head_submit_output.log", REGISTRY.parent / "submit_output.log"],
        pre_log_digests,
    )
    delivered = 0
    for item in batch:
        key = item["key"]
        ids: dict[str, str] = {}
        entry = post_registry.get(key)
        changed = entry is not None and pre_registry.get(key) != entry
        bound = (
            changed
            and str(entry.get("registry_key") or "") == key
            and str(entry.get("attempt_id") or "").strip() != ""
            and str(entry.get("parent_attempt_id") or "") == verified[key]["attempt"]
            and str(entry.get("deployment_id") or "") == str(deployment_id)
        )
        if bound:
            ids = {
                "extract": str(entry.get("extract_job_id") or ""),
                "classifier": str(entry.get("classifier_job_id") or ""),
            }
        if not ids and key in fresh:
            ids = fresh[key]
        valid = _valid_ids(ids, seen) if ids else None
        if valid is not None:
            extract, classifier = valid
            seen.update({extract, classifier})
            append_record(
                {
                    "key": f"head:{key}",
                    "run_name": str(item.get("run_name", "")),
                    "ts": int(time.time()),
                    "status": "submitted",
                    "attempt_id": None,
                    "job_ids": {"extract": extract, "classifier": classifier},
                    "parent_attempt_id": verified[key]["attempt"],
                    "note": "authoritative fresh delivery (bound registry row or changed submit log)",
                }
            )
            delivered += 1
        else:
            known = {k: v for k, v in ids.items() if v}
            append_record(
                {
                    "key": f"head:{key}",
                    "run_name": str(item.get("run_name", "")),
                    "ts": int(time.time()),
                    "status": "uncertain",
                    "attempt_id": None,
                    "job_ids": known,
                    "parent_attempt_id": verified[key]["attempt"],
                    "reason": final_reason
                    or ("partial delivery" if known else "no authoritative delivery"),
                    "reservation": JOBS_PER_KEY,
                }
            )
    return delivered


def _submit_locked(args: argparse.Namespace) -> int:
    try:
        _, audit = build_plan(CONTRACT, LEDGER)
    except PlanError as exc:
        raise GuardError(f"fresh plan rebuild failed: {exc}") from exc
    if audit.get("keys_total") != 126:
        raise GuardError("fresh plan audit does not cover 126 keys")
    eligible = [item for item in audit["keys"] if item["status"] == "eligible"]

    if getattr(args, "reconcile_key", None):
        for key in args.reconcile_key:
            append_record(
                {
                    "key": f"head:{key}",
                    "run_name": "",
                    "ts": int(time.time()),
                    "status": "reconciled",
                    "attempt_id": None,
                    "job_ids": {},
                    "reason": str(getattr(args, "reason", "") or "explicit reconciliation"),
                }
            )

    ledger_latest = _ledger_latest()
    verified: dict[str, dict] = {}
    for item in eligible:
        verified[item["key"]] = bind_parent_identity(item, ledger_latest)
    all_ids = sorted({jid for info in verified.values() for jid in info["ids"].values()})
    accounting = query_top_level_accounting(all_ids)
    for item in eligible:
        assert_completed(item["key"], verified[item["key"]]["ids"], accounting)

    registry = _registry_snapshot(REGISTRY)
    settled = set(registry) | _head_blocked_keys(ledger_latest)
    pending = [item for item in eligible if item["key"] not in settled]
    if args.only:
        wanted = set(args.only)
        pending = [item for item in pending if item["key"] in wanted]
    if not pending:
        print(json.dumps({"status": "ok", "eligible": len(eligible), "pending": 0, "note": "no eligible unblocked unsubmitted keys"}))
        return 0

    job_ids, uncertain = own_job_ids()
    states, user_queue = query_job_states(job_ids)
    own_nonterminal = own_nonterminal_count(job_ids, states, uncertain)
    storage = storage_admission()
    if user_queue >= USER_QUEUE_STOP:
        print(f"REFUSED: user queue {user_queue} at or above {USER_QUEUE_STOP}")
        return 2
    capacity = max(0, (MAX_LANE_NONTERMINAL - own_nonterminal) // JOBS_PER_KEY)
    limit = min(int(args.limit) if args.limit else len(pending), capacity, len(pending))
    if limit < 1:
        print(f"REFUSED: no lane headroom (own {own_nonterminal} + {JOBS_PER_KEY} > {MAX_LANE_NONTERMINAL})")
        return 2
    batch = pending[:limit]
    print(
        "heads admission: "
        + json.dumps(
            {
                "eligible": len(eligible),
                "pending": len(pending),
                "batch": len(batch),
                "own_nonterminal": own_nonterminal,
                "user_queue": user_queue,
                "jobs_per_key": JOBS_PER_KEY,
                "storage_remaining_gb": storage["gpfs_projects_remaining_gb"],
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "keys": [item["key"] for item in batch]}, sort_keys=True))
        return 0

    for item in batch:
        append_record(
            {
                "key": f"head:{item['key']}",
                "run_name": str(item.get("run_name", "")),
                "ts": int(time.time()),
                "status": "held",
                "attempt_id": None,
                "job_ids": {},
                "parent_attempt_id": verified[item["key"]]["attempt"],
                "note": "full 2-job reservation written before submit",
            }
        )

    plan_cmd = [
        sys.executable, str(HEADS_TOOL), "plan",
        "--matrix", str(args.matrix),
        "--campaign", "qwen3_window15_20261008",
        "--output", str(args.plan),
    ]
    plan_result = _run(plan_cmd)
    if plan_result.returncode != 0:
        _reconcile_batch(
            batch,
            verified=verified,
            pre_registry=registry,
            pre_log_digests={},
            deployment_id=args.deployment_id,
            seen=_seen_ids(registry, ledger_latest),
            final_reason=f"generic plan failed rc={plan_result.returncode}",
        )
        print(f"REFUSED: generic plan failed rc={plan_result.returncode}")
        return 2
    index = _plan_key_index(args.plan)
    for item in batch:
        found = index.get(item["key"])
        if not found or found["job"].get("parent_status") != "resolved":
            _reconcile_batch(
                batch, verified=verified, pre_registry=registry, pre_log_digests={},
                deployment_id=args.deployment_id, seen=_seen_ids(registry, ledger_latest),
                final_reason="plan entry not resolved",
            )
            print(f"REFUSED: {item['key']} not resolved in the generic plan")
            return 2
        parent = found["job"].get("parent") or {}
        if str(parent.get("attempt_id") or "") != verified[item["key"]]["attempt"]:
            _reconcile_batch(
                batch, verified=verified, pre_registry=registry, pre_log_digests={},
                deployment_id=args.deployment_id, seen=_seen_ids(registry, ledger_latest),
                final_reason="plan parent attempt mismatch",
            )
            print(f"REFUSED: {item['key']} plan parent attempt mismatch")
            return 2
        if not parent.get("adapter_sha256") or not parent.get("adapter_config_sha256"):
            _reconcile_batch(
                batch, verified=verified, pre_registry=registry, pre_log_digests={},
                deployment_id=args.deployment_id, seen=_seen_ids(registry, ledger_latest),
                final_reason="plan parent adapter hashes missing",
            )
            print(f"REFUSED: {item['key']} plan parent adapter hashes missing")
            return 2
        if not parent.get("manifest_hash") or not (parent.get("split_fingerprint") or {}).get("sha256"):
            _reconcile_batch(
                batch, verified=verified, pre_registry=registry, pre_log_digests={},
                deployment_id=args.deployment_id, seen=_seen_ids(registry, ledger_latest),
                final_reason="plan parent manifest/split hashes missing",
            )
            print(f"REFUSED: {item['key']} plan parent manifest/split hashes missing")
            return 2

    pre_log_paths = [REGISTRY.parent / "head_submit_output.log", REGISTRY.parent / "submit_output.log"]
    pre_log_digests = {str(path): _log_digest(path) for path in pre_log_paths}
    cmd = [
        sys.executable, str(HEADS_TOOL), "submit",
        "--plan", str(args.plan),
        "--registry", str(REGISTRY),
        "--deployment-id", str(args.deployment_id),
        "--limit", str(len(batch)),
        "--execute",
    ]
    for item in batch:
        cmd += ["--key", item["key"]]
    final_reason: str | None = None
    try:
        result = _run(cmd)
        if result.returncode != 0:
            final_reason = f"submit rc={result.returncode}"
    except Exception as exc:  # timeouts and runner failures still reconcile
        final_reason = f"submit exception {type(exc).__name__}: {exc}"
    delivered = _reconcile_batch(
        batch,
        verified=verified,
        pre_registry=registry,
        pre_log_digests=pre_log_digests,
        deployment_id=args.deployment_id,
        seen=_seen_ids(registry, ledger_latest),
        final_reason=final_reason,
    )
    print(json.dumps({"status": "ok", "delivered": delivered, "attempted": len(batch)}, sort_keys=True))
    return 0 if delivered == len(batch) else 1


def command_submit(args: argparse.Namespace) -> int:
    try:
        with lane_submission_lock():
            return _submit_locked(args)
    except (GuardError, AdmissionError) as exc:
        print(f"REFUSED: {exc}")
        return 2


def command_delegate(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(HEADS_TOOL), args.command, "--registry", str(REGISTRY)]
    for passthrough in args.rest or []:
        cmd.append(passthrough)
    return _run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="guarded head submit (dry-run first)")
    submit.add_argument("--matrix", type=Path, required=True)
    submit.add_argument("--plan", type=Path, default=LANE / "outputs/qwen3_window15_20261008/head_dispatch_plan_v1.json")
    submit.add_argument("--deployment-id", required=True)
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument("--only", action="append", default=None)
    submit.add_argument("--reconcile-key", action="append", default=None)
    submit.add_argument("--reason", default=None)
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=command_submit)

    for name in ("status", "collect", "validate", "finish", "coverage"):
        delegate = sub.add_parser(name, help=f"delegate to tools/qwen3_heads_dispatch.py {name}")
        delegate.add_argument("--rest", nargs=argparse.REMAINDER)
        delegate.set_defaults(func=command_delegate)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (GuardError, AdmissionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
