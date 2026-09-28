#!/usr/bin/env python3
"""Resume ledger for the Qwen3 DAIC label-vocabulary campaign.

The campaign runs many Slurm jobs across a long horizon, so the agent must be
able to resume after a restart or a context compaction without resubmitting
healthy work. This tool is the single writer of the ignored ledger at
``outputs/qwen3_daic_label_vocab/state.json``.

Every mutation is atomic (write a sibling temporary file, then ``os.replace``)
and is recorded in an append-only ``history`` list. Repeated identical job or
evidence events are ignored instead of duplicated. ``init`` refuses to
overwrite an existing ledger so recorded history is never lost by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "outputs" / "qwen3_daic_label_vocab" / "state.json"
SCHEMA_VERSION = "audiollm.qwen3_daic_label_vocab.state.v1"

PHASES = (
    "grant",
    "worktree",
    "configs",
    "token_audit",
    "acceptance_pr",
    "prebuilt_inputs",
    "deployment",
    "smoke",
    "production",
    "analysis",
    "handoff",
)
PHASE_STATUSES = ("pending", "in_progress", "complete", "blocked", "hard_stop")


class LedgerError(Exception):
    """Raised when the ledger cannot be read or written safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LedgerError(f"ledger does not exist: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise LedgerError(f"ledger is not valid JSON: {exc}") from exc
    if record.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError(f"unexpected ledger schema_version: {record.get('schema_version')!r}")
    return record


def save_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _record(record: dict[str, Any], event: dict[str, Any]) -> None:
    event["at_utc"] = utc_now()
    record.setdefault("history", []).append(event)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(record: dict[str, Any], phase_id: str) -> dict[str, Any]:
    for entry in record.setdefault("phases", []):
        if entry["id"] == phase_id:
            return entry
    entry = {"id": phase_id, "status": "pending", "note": ""}
    record["phases"].append(entry)
    return entry


def command_init(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    if path.exists() and not args.allow_existing:
        raise LedgerError(f"ledger already exists: {path} (refusing to overwrite)")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": "qwen3-daic-label-vocab",
        "plan": "docs/QWEN3_DAIC_LABEL_VOCAB_EXECUTION_PLAN.md",
        "created_at_utc": utc_now(),
        "grant": {
            "granted_in_chat": True,
            "journal_reference": args.journal_reference,
            "merges_allowed": False,
            "notes": "Exact grant text is recorded in the agent journal entry of the same date.",
        },
        "baseline": {
            "origin_main_sha": args.baseline_sha,
            "recorded_at_utc": utc_now(),
        },
        "lane": {
            "experiment_id": args.experiment_id,
            "slug": args.slug,
            "branch": args.branch,
            "worktree": args.worktree,
            "tier": args.tier,
            "group_definition": args.group_definition,
            "pull_request": None,
        },
        "campaign": {
            "name": args.campaign,
            "smoke_campaign": args.smoke_campaign,
            "group_id": args.group_id,
            "dataset": "daic",
            "fold": args.fold,
            "seeds": [7, 1337, 2024],
            "smoke_seed": 1337,
        },
        "phases": [],
        "cells": [],
        "deployment": None,
        "jobs": [],
        "evidence": [],
        "decisions": [],
        "hard_stop": None,
        "history": [],
    }
    for phase_id in PHASES:
        _phase(record, phase_id)
    _record(record, {"event": "init"})
    save_ledger(path, record)
    print(f"initialized {path}")
    return 0


def command_phase(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    if args.phase not in PHASES:
        raise LedgerError(f"unknown phase {args.phase!r}; expected one of {list(PHASES)}")
    if args.status not in PHASE_STATUSES:
        raise LedgerError(f"unknown status {args.status!r}; expected one of {list(PHASE_STATUSES)}")
    entry = _phase(record, args.phase)
    entry["status"] = args.status
    if args.note:
        entry["note"] = args.note
    _record(record, {"event": "phase", "phase": args.phase, "status": args.status, "note": args.note or ""})
    save_ledger(path, record)
    print(f"phase {args.phase}: {args.status}")
    return 0


def command_deployment(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    record["deployment"] = {
        "deployment_id": args.deployment_id,
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "source_manifest_sha256": args.source_manifest_sha256,
        "recorded_at_utc": utc_now(),
    }
    _record(record, {"event": "deployment", "deployment_id": args.deployment_id, "git_commit": args.git_commit})
    save_ledger(path, record)
    print(f"deployment {args.deployment_id} recorded")
    return 0


def command_cell(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    entry = {
        "cell": args.cell,
        "modality": args.modality,
        "model": args.model,
        "config": args.config,
        "config_sha256": args.config_sha256,
        "run_root": args.run_root,
        "arms": args.arms.split(","),
        "recorded_at_utc": utc_now(),
    }
    record["cells"] = [item for item in record["cells"] if item["cell"] != args.cell] + [entry]
    _record(record, {"event": "cell", "cell": args.cell, "config": args.config})
    save_ledger(path, record)
    print(f"cell {args.cell} recorded")
    return 0


def command_job(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    event = {
        "attempt_id": args.attempt_id,
        "logical_run": args.logical_run,
        "job_key": args.job_key,
        "slurm_job_id": args.slurm_job_id,
        "state": args.state,
        "detail": args.detail or "",
    }
    key = (event["attempt_id"], event["job_key"], event["slurm_job_id"], event["state"])
    for existing in record["jobs"]:
        if (existing["attempt_id"], existing["job_key"], existing["slurm_job_id"], existing["state"]) == key:
            print("job event already recorded; nothing to do")
            return 0
    event["at_utc"] = utc_now()
    record["jobs"].append(event)
    _record(record, {"event": "job", **{k: event[k] for k in ("attempt_id", "job_key", "slurm_job_id", "state")}})
    save_ledger(path, record)
    print(f"job event recorded: {event['attempt_id']} {event['job_key']} {event['slurm_job_id']} {event['state']}")
    return 0


def command_evidence(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    evidence_path = Path(args.path)
    digest = args.sha256 or (sha256_file(evidence_path) if evidence_path.is_file() else "")
    if not digest:
        raise LedgerError(f"cannot hash evidence file: {evidence_path}")
    for existing in record["evidence"]:
        if existing["path"] == args.path and existing["sha256"] == digest:
            print("evidence already recorded; nothing to do")
            return 0
    entry = {
        "attempt_id": args.attempt_id,
        "kind": args.kind,
        "path": args.path,
        "sha256": digest,
        "recorded_at_utc": utc_now(),
    }
    record["evidence"].append(entry)
    _record(record, {"event": "evidence", "attempt_id": args.attempt_id, "kind": args.kind, "path": args.path})
    save_ledger(path, record)
    print(f"evidence recorded: {args.kind} {args.path}")
    return 0


def command_decision(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    record["decisions"].append({"text": args.text, "at_utc": utc_now()})
    _record(record, {"event": "decision", "text": args.text})
    save_ledger(path, record)
    print("decision recorded")
    return 0


def command_hard_stop(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    record = load_ledger(path)
    record["hard_stop"] = {
        "reason": args.reason,
        "cell": args.cell or "",
        "evidence": list(args.evidence or []),
        "smallest_needed_decision": args.decision or "",
        "recorded_at_utc": utc_now(),
    }
    _record(record, {"event": "hard_stop", "reason": args.reason})
    save_ledger(path, record)
    print("hard stop recorded")
    return 0


def command_show(args: argparse.Namespace) -> int:
    record = load_ledger(Path(args.ledger))
    lane = record["lane"]
    print(f"task: {record['task']}")
    print(f"lane: {lane['experiment_id']} branch={lane['branch']}")
    print(f"worktree: {lane['worktree']}")
    print(f"pr: {lane['pull_request']}")
    print(f"baseline: {record['baseline']['origin_main_sha']}")
    print(f"campaign: {record['campaign']['name']} (smoke {record['campaign']['smoke_campaign']})")
    print("phases:")
    for entry in record["phases"]:
        note = f" - {entry['note']}" if entry.get("note") else ""
        print(f"  {entry['id']}: {entry['status']}{note}")
    print(f"cells: {len(record['cells'])}")
    print(f"job events: {len(record['jobs'])}")
    print(f"evidence records: {len(record['evidence'])}")
    if record.get("hard_stop"):
        print(f"HARD STOP: {record['hard_stop']['reason']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--baseline-sha", required=True)
    init.add_argument("--experiment-id", required=True)
    init.add_argument("--slug", required=True)
    init.add_argument("--branch", required=True)
    init.add_argument("--worktree", required=True)
    init.add_argument("--tier", type=int, default=2)
    init.add_argument("--group-definition", required=True)
    init.add_argument("--campaign", required=True)
    init.add_argument("--smoke-campaign", required=True)
    init.add_argument("--group-id", required=True)
    init.add_argument("--fold", type=int, default=0)
    init.add_argument("--journal-reference", default="Agent-Journal/LLM-Depression/agent-journal-2026.md")
    init.add_argument("--allow-existing", action="store_true")
    init.set_defaults(func=command_init)

    phase = sub.add_parser("phase")
    phase.add_argument("phase")
    phase.add_argument("status")
    phase.add_argument("--note", default="")
    phase.set_defaults(func=command_phase)

    deployment = sub.add_parser("deployment")
    deployment.add_argument("--deployment-id", required=True)
    deployment.add_argument("--git-commit", required=True)
    deployment.add_argument("--git-branch", default="")
    deployment.add_argument("--source-manifest-sha256", default="")
    deployment.set_defaults(func=command_deployment)

    cell = sub.add_parser("cell")
    cell.add_argument("--cell", required=True)
    cell.add_argument("--modality", required=True)
    cell.add_argument("--model", required=True)
    cell.add_argument("--config", required=True)
    cell.add_argument("--config-sha256", required=True)
    cell.add_argument("--run-root", required=True)
    cell.add_argument("--arms", default="ab,01,truefalse,yesno,en")
    cell.set_defaults(func=command_cell)

    job = sub.add_parser("job")
    job.add_argument("--attempt-id", required=True)
    job.add_argument("--logical-run", default="")
    job.add_argument("--job-key", required=True)
    job.add_argument("--slurm-job-id", required=True)
    job.add_argument("--state", required=True)
    job.add_argument("--detail", default="")
    job.set_defaults(func=command_job)

    evidence = sub.add_parser("evidence")
    evidence.add_argument("--attempt-id", required=True)
    evidence.add_argument("--kind", required=True)
    evidence.add_argument("--path", required=True)
    evidence.add_argument("--sha256", default="")
    evidence.set_defaults(func=command_evidence)

    decision = sub.add_parser("decision")
    decision.add_argument("text")
    decision.set_defaults(func=command_decision)

    hard_stop = sub.add_parser("hard-stop")
    hard_stop.add_argument("--reason", required=True)
    hard_stop.add_argument("--cell", default="")
    hard_stop.add_argument("--evidence", action="append")
    hard_stop.add_argument("--decision", default="")
    hard_stop.set_defaults(func=command_hard_stop)

    show = sub.add_parser("show")
    show.set_defaults(func=command_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except LedgerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
