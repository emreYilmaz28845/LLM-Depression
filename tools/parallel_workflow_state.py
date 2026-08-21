from __future__ import annotations
import argparse, json, hashlib, datetime, pathlib, sys, os, re, tempfile

SCHEMA_VERSION = "audiollm.parallel_workflow_execution.v1"
PHASES = list(range(14))

# Required evidence substrings per phase for pass validation
REQUIRED_EVIDENCE_PATTERNS = {
    0: ["grant_journal", "baseline", "worktree", "state.json", "test", "PR", "audit"],
    # For later phases, require at least 1 evidence, specific checks enforced by auditor
}

def utc_now_str():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(path: pathlib.Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def load_state(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_state_schema(state: dict) -> list[str]:
    errors = []
    if state.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    required_top = ["execution_id", "runbook_path", "runbook_sha256_at_start", "grant_journal_path", "status", "current_phase", "phases", "branches", "prs", "deployments", "attempts", "jobs", "hard_stop", "updated_at_utc"]
    for k in required_top:
        if k not in state:
            errors.append(f"missing key {k}")
    if "phases" in state:
        if not isinstance(state["phases"], dict):
            errors.append("phases must be dict")
        else:
            for i in PHASES:
                key = str(i)
                if key not in state["phases"]:
                    errors.append(f"missing phase {key}")
                else:
                    ph = state["phases"][key]
                    if ph.get("status") not in ("PENDING", "IN_PROGRESS", "PASSED"):
                        errors.append(f"phase {key} invalid status {ph.get('status')}")
                    if "evidence" not in ph or not isinstance(ph["evidence"], list):
                        errors.append(f"phase {key} evidence must be list")
                    if "next_action" not in ph:
                        errors.append(f"phase {key} missing next_action")
    if state.get("status") not in ("ACTIVE", "HARD_STOP", "COMPLETE"):
        errors.append(f"invalid status {state.get('status')}")
    return errors

def cmd_init(args):
    runbook_path = pathlib.Path(args.runbook)
    state_path = pathlib.Path(args.output)
    execution_id = args.execution_id
    if state_path.exists():
        print(f"ERROR: state file already exists: {state_path}", file=sys.stderr)
        return 1
    if not runbook_path.exists():
        print(f"ERROR: runbook not found: {runbook_path}", file=sys.stderr)
        return 1
    sha = sha256_file(runbook_path)
    grant_journal = args.grant_journal or "docs/agent-journal/2026-08-20.md"
    now = utc_now_str()
    state = {
        "schema_version": SCHEMA_VERSION,
        "execution_id": execution_id,
        "runbook_path": str(runbook_path),
        "runbook_sha256_at_start": sha,
        "grant_journal_path": grant_journal,
        "status": "ACTIVE",
        "current_phase": 0,
        "phases": {str(i): {"status": "PENDING", "evidence": [], "next_action": ""} for i in PHASES},
        "branches": [],
        "prs": [],
        "deployments": [],
        "attempts": [],
        "jobs": [],
        "hard_stop": None,
        "updated_at_utc": now,
    }
    state["phases"]["0"]["status"] = "IN_PROGRESS"
    state["phases"]["0"]["next_action"] = "Implement state tool and auditor skeleton, validate, PR, merge"
    for i in range(1,14):
        state["phases"][str(i)]["next_action"] = f"Enter Phase {i} after Phase {i-1} PASSED"
    atomic_write_json(state_path, state)
    print(f"initialized {state_path} execution_id={execution_id} runbook_sha={sha}")
    return 0

def cmd_show(args):
    state_path = pathlib.Path(args.state)
    if not state_path.exists():
        print(f"ERROR: state not found {state_path}", file=sys.stderr)
        return 1
    state = load_state(state_path)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0

def cmd_enter(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    phase = args.phase
    if state.get("hard_stop") is not None:
        print(f"ERROR: cannot enter phase {phase} while HARD_STOP is active; resolve hard stop first", file=sys.stderr)
        return 1
    if phase not in PHASES:
        print(f"ERROR: invalid phase {phase}", file=sys.stderr)
        return 1
    # Must not skip phases: all previous phases must be PASSED, except if phase 0 is IN_PROGRESS already
    # Check ordering: current_phase should be phase-1 if entering next, or phase itself if re-entering?
    # For Phase 0, if already IN_PROGRESS, entering again is idempotent? We should reject if already PASSED or IN_PROGRESS unless phase == current_phase
    # Also ensure previous phases PASSED
    for i in range(phase):
        if state["phases"][str(i)]["status"] != "PASSED":
            print(f"ERROR: cannot enter phase {phase}: previous phase {i} is {state['phases'][str(i)]['status']}, not PASSED", file=sys.stderr)
            return 1
    ph = state["phases"][str(phase)]
    if ph["status"] == "PASSED":
        print(f"ERROR: phase {phase} already PASSED", file=sys.stderr)
        return 1
    if ph["status"] == "IN_PROGRESS" and state["current_phase"] == phase:
        # already in progress, allow updating next_action
        pass
    else:
        if ph["status"] != "PENDING":
            print(f"ERROR: phase {phase} is {ph['status']}, expected PENDING", file=sys.stderr)
            return 1
        # entering new phase: current_phase must be phase-1 or phase==0 with current 0
        if phase != 0 and state["current_phase"] != phase-1:
            # Also allow if current_phase == phase (already) but we handled
            print(f"ERROR: current_phase is {state['current_phase']}, cannot enter phase {phase} (previous not PASSED or skipping)", file=sys.stderr)
            return 1
        if state["status"] != "ACTIVE":
            print(f"ERROR: status is {state['status']}, cannot enter", file=sys.stderr)
            return 1
    # update
    ph["status"] = "IN_PROGRESS"
    if args.next_action:
        ph["next_action"] = args.next_action
    state["current_phase"] = phase
    state["updated_at_utc"] = utc_now_str()
    # execution_id and runbook SHA immutability check handled by not changing them, but verify no change attempted
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"entered phase {phase}")
    return 0

def cmd_record(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    phase = args.phase
    if phase not in PHASES:
        print(f"ERROR: invalid phase {phase}", file=sys.stderr)
        return 1
    if state["phases"][str(phase)]["status"] not in ("IN_PROGRESS", "PASSED"):
        print(f"WARNING: recording evidence for phase {phase} which is {state['phases'][str(phase)]['status']}", file=sys.stderr)
    ph = state["phases"][str(phase)]
    ev = args.evidence
    if ev in ph["evidence"]:
        print(f"evidence already recorded for phase {phase}: {ev}")
        return 0
    ph["evidence"].append(ev)
    state["updated_at_utc"] = utc_now_str()
    atomic_write_json(state_path, state)
    print(f"recorded evidence for phase {phase}: {ev}")
    return 0

def check_required_evidence(phase: int, evidence: list[str]) -> list[str]:
    patterns = REQUIRED_EVIDENCE_PATTERNS.get(phase, [])
    if not patterns:
        # require at least one evidence for phases beyond 0
        if len(evidence) == 0:
            return [f"phase {phase} requires at least one evidence"]
        return []
    missing = []
    lower_evs = [e.lower() for e in evidence]
    for pat in patterns:
        pat_low = pat.lower()
        if not any(pat_low in ev for ev in lower_evs):
            missing.append(pat)
    if missing:
        return [f"phase {phase} missing required evidence patterns: {missing}"]
    return []

def cmd_pass(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    phase = args.phase
    next_phase = args.next_phase
    if state.get("hard_stop") is not None:
        print(f"ERROR: cannot pass phase {phase} while HARD_STOP is active", file=sys.stderr)
        return 1
    if phase not in PHASES or next_phase not in PHASES and next_phase != 14:
        # next_phase may be 14 meaning complete? But spec uses next_phase <14
        if next_phase != 14:
            print(f"ERROR: invalid phase numbers phase={phase} next_phase={next_phase}", file=sys.stderr)
            return 1
    if next_phase != phase + 1:
        print(f"ERROR: pass must go to next sequential phase: {phase} -> {next_phase} is not +1", file=sys.stderr)
        return 1
    ph = state["phases"][str(phase)]
    if ph["status"] != "IN_PROGRESS":
        print(f"ERROR: phase {phase} is {ph['status']}, must be IN_PROGRESS to pass", file=sys.stderr)
        return 1
    # Check required evidence
    errs = check_required_evidence(phase, ph["evidence"])
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    # Also ensure not skipping: all previous phases must be PASSED which is true if phase is IN_PROGRESS sequentially
    for i in range(phase):
        if state["phases"][str(i)]["status"] != "PASSED":
            print(f"ERROR: cannot pass phase {phase} because previous phase {i} not PASSED", file=sys.stderr)
            return 1
    # Mark passed
    ph["status"] = "PASSED"
    ph["next_action"] = ph.get("next_action", "") + " (PASSED)"
    state["updated_at_utc"] = utc_now_str()
    # Enter next phase if not beyond 13
    if next_phase in PHASES:
        nxt = state["phases"][str(next_phase)]
        if nxt["status"] != "PENDING":
            print(f"ERROR: next phase {next_phase} is {nxt['status']}, expected PENDING", file=sys.stderr)
            return 1
        nxt["status"] = "IN_PROGRESS"
        if nxt["next_action"] == "":
            nxt["next_action"] = f"Execute Phase {next_phase}"
        state["current_phase"] = next_phase
    else:
        # No next phase, stay at current but will be completed later via complete command
        state["current_phase"] = phase
    atomic_write_json(state_path, state)
    print(f"phase {phase} PASSED, entered phase {next_phase if next_phase in PHASES else 'COMPLETE_PENDING'}")
    return 0

def cmd_hard_stop(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    phase = args.phase
    if phase not in PHASES:
        print(f"ERROR: invalid phase {phase}", file=sys.stderr)
        return 1
    if state["status"] == "HARD_STOP" and not args.force:
        print(f"ERROR: already in HARD_STOP; use --force to overwrite or resolve via resume", file=sys.stderr)
        return 1
    state["status"] = "HARD_STOP"
    state["hard_stop"] = {
        "phase": phase,
        "reason": args.reason,
        "evidence": args.evidence,
        "at_utc": utc_now_str()
    }
    state["updated_at_utc"] = utc_now_str()
    atomic_write_json(state_path, state)
    print(f"HARD_STOP set at phase {phase}: {args.reason}")
    return 0

def cmd_complete(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    audit_path = pathlib.Path(args.audit) if args.audit else None
    if state.get("hard_stop") is not None:
        print(f"ERROR: cannot complete while HARD_STOP is active", file=sys.stderr)
        return 1
    if state.get("status") == "COMPLETE":
        print("ERROR: execution already COMPLETE", file=sys.stderr)
        return 1
    # Phase 13 must be the active phase; 0-12 PASSED.
    for i in range(13):
        if state["phases"][str(i)]["status"] != "PASSED":
            print(f"ERROR: cannot mark COMPLETE: phase {i} not PASSED", file=sys.stderr)
            return 1
    if state["phases"]["13"]["status"] != "IN_PROGRESS":
        print("ERROR: cannot mark COMPLETE: Phase 13 must be IN_PROGRESS (preterminal gate)", file=sys.stderr)
        return 1
    # An approved preterminal audit is mandatory and consumed atomically.
    if audit_path is None or not audit_path.is_file():
        print(f"ERROR: approved preterminal audit file required: {audit_path}", file=sys.stderr)
        return 1
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: preterminal audit unreadable: {e}", file=sys.stderr)
        return 1
    if audit.get("passed") is not True or audit.get("mode") != "preterminal":
        print("ERROR: audit is not a passing preterminal audit (mode=preterminal, passed=true)", file=sys.stderr)
        return 1
    audit_sha = sha256_file(audit_path)
    state["completion"] = {
        "preterminal_audit_path": str(audit_path.resolve()),
        "preterminal_audit_sha256": audit_sha,
        "completed_at_utc": utc_now_str(),
    }
    state["phases"]["13"]["status"] = "PASSED"
    state["status"] = "COMPLETE"
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after complete: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"execution {state['execution_id']} marked COMPLETE (preterminal audit {audit_sha[:12]} consumed)")
    return 0

def cmd_reopen(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    phase = args.phase
    if phase not in PHASES:
        print(f"ERROR: invalid phase {phase}", file=sys.stderr)
        return 1
    if state.get("status") not in ("COMPLETE", "HARD_STOP", "ACTIVE"):
        print(f"ERROR: cannot reopen from status {state.get('status')}", file=sys.stderr)
        return 1
    import copy
    snapshot = copy.deepcopy(state)
    invalidations = state.get("invalidations", [])
    invalidations.append({
        "at_utc": utc_now_str(),
        "reopened_phase": phase,
        "reason": args.reason,
        "evidence": args.evidence,
        "previous_status": snapshot.get("status"),
        "previous_current_phase": snapshot.get("current_phase"),
        "previous_phases_snapshot": {k: v.get("status") for k, v in snapshot.get("phases", {}).items()},
    })
    state["invalidations"] = invalidations
    for i in PHASES:
        key = str(i)
        if i < phase:
            if state["phases"][key]["status"] != "PASSED":
                print(f"WARNING: phase {i} is {state['phases'][key]['status']}, expected PASSED for reopen to {phase}", file=sys.stderr)
        elif i == phase:
            state["phases"][key]["status"] = "IN_PROGRESS"
            state["phases"][key]["evidence"] = []
            state["phases"][key]["next_action"] = f"Re-enter Phase {phase} after invalidation: {args.reason}"
        else:
            state["phases"][key]["status"] = "PENDING"
            state["phases"][key]["evidence"] = []
            state["phases"][key]["next_action"] = f"Enter Phase {i} after Phase {i-1} PASSED (reopened)"
    state["current_phase"] = phase
    state["status"] = "ACTIVE"
    state["hard_stop"] = None
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after reopen: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"reopened to phase {phase} due to: {args.reason}")
    return 0


def cmd_record_job(args):
    """Append a structured job record to the ledger (append-only, atomic)."""
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    jobs = state.setdefault("jobs", [])
    record = {
        "at_utc": utc_now_str(),
        "attempt_id": args.attempt_id,
        "job_key": args.job_key,
        "job_type": args.job_type,
        "event_type": args.event_type,
        "slurm_job_id": args.slurm_job_id,
        "status": args.status,
        "fold": args.fold,
    }
    for extra in ("exit_code", "dependency_job_ids", "reason", "deployment_id", "evaluation_view", "backend", "aggregation", "metrics_path"):
        value = getattr(args, extra, None)
        if value is not None:
            record[extra] = value
    jobs.append(record)
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after record-job: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"recorded job event {args.event_type} {args.job_key} {args.slurm_job_id or '-'} for {args.attempt_id}")
    return 0


def cmd_record_pr(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    state.setdefault("prs", []).append({
        "pr_url": args.pr_url,
        "phase": args.phase,
        "head_sha": args.head_sha,
        "merge_sha": args.merge_sha,
        "recorded_at_utc": utc_now_str(),
    })
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after record-pr: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"recorded pr {args.pr_url}")
    return 0


def cmd_remove_evidence(args):
    """Remove a malformed evidence entry, recording the correction explicitly."""
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    ph = state["phases"][str(args.phase)]
    entry = args.entry
    if entry not in ph.get("evidence", []):
        print(f"ERROR: evidence entry not found in phase {args.phase}: {entry}", file=sys.stderr)
        return 1
    ph["evidence"] = [e for e in ph["evidence"] if e != entry]
    corrections = state.setdefault("evidence_corrections", [])
    corrections.append({
        "at_utc": utc_now_str(),
        "phase": args.phase,
        "removed_entry": entry,
        "reason": args.reason,
        "replacement": args.replacement,
    })
    if args.replacement:
        ph["evidence"].append(args.replacement)
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after remove-evidence: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"removed malformed evidence entry from phase {args.phase}")
    return 0


def cmd_correct_pr(args):
    """Append a corrected head/merge SHA for a PR; latest record wins."""
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    prs = state.setdefault("prs", [])
    prior = [p for p in prs if p.get("pr_url") == args.pr_url]
    if not prior:
        print(f"ERROR: no prior record for {args.pr_url}", file=sys.stderr)
        return 1
    prs.append({
        "pr_url": args.pr_url,
        "phase": args.phase if args.phase is not None else prior[-1].get("phase"),
        "head_sha": args.head_sha,
        "merge_sha": args.merge_sha,
        "corrects_recorded_at_utc": prior[-1].get("recorded_at_utc"),
        "recorded_at_utc": utc_now_str(),
    })
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after correct-pr: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"corrected pr record for {args.pr_url}")
    return 0


def cmd_record_deployment(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    state.setdefault("deployments", []).append({
        "deployment_id": args.deployment_id,
        "experiment_id": args.experiment_id,
        "git_commit": args.git_commit,
        "source_manifest_sha256": args.source_manifest_sha256,
        "deployed_code_path": args.deployed_code_path,
        "created_at_utc": utc_now_str(),
    })
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after record-deployment: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"recorded deployment {args.deployment_id}")
    return 0


def cmd_record_attempt(args):
    state_path = pathlib.Path(args.state)
    state = load_state(state_path)
    state.setdefault("attempts", []).append({
        "attempt_id": args.attempt_id,
        "deployment_id": args.deployment_id,
        "experiment_id": args.experiment_id,
        "fold": args.fold,
        "status": args.status,
        "created_at_utc": utc_now_str(),
    })
    state["updated_at_utc"] = utc_now_str()
    errors = validate_state_schema(state)
    if errors:
        print(f"ERROR: schema validation failed after record-attempt: {errors}", file=sys.stderr)
        return 1
    atomic_write_json(state_path, state)
    print(f"recorded attempt {args.attempt_id} status {args.status}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Manage parallel workflow execution ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    # init
    p_init = sub.add_parser("init", help="initialize new state")
    p_init.add_argument("--runbook", required=True)
    p_init.add_argument("--execution-id", required=True)
    p_init.add_argument("--output", required=True)
    p_init.add_argument("--grant-journal", default=None)
    p_init.set_defaults(func=cmd_init)
    # show
    p_show = sub.add_parser("show", help="show state")
    p_show.add_argument("--state", required=True)
    p_show.set_defaults(func=cmd_show)
    # enter
    p_enter = sub.add_parser("enter", help="enter phase")
    p_enter.add_argument("--state", required=True)
    p_enter.add_argument("--phase", type=int, required=True)
    p_enter.add_argument("--next-action", required=True)
    p_enter.set_defaults(func=cmd_enter)
    # record
    p_record = sub.add_parser("record", help="record evidence")
    p_record.add_argument("--state", required=True)
    p_record.add_argument("--phase", type=int, required=True)
    p_record.add_argument("--evidence", required=True)
    p_record.set_defaults(func=cmd_record)
    # pass
    p_pass = sub.add_parser("pass", help="mark phase passed and enter next")
    p_pass.add_argument("--state", required=True)
    p_pass.add_argument("--phase", type=int, required=True)
    p_pass.add_argument("--next-phase", type=int, required=True)
    p_pass.set_defaults(func=cmd_pass)
    # hard-stop
    p_hard = sub.add_parser("hard-stop", help="set hard stop")
    p_hard.add_argument("--state", required=True)
    p_hard.add_argument("--phase", type=int, required=True)
    p_hard.add_argument("--reason", required=True)
    p_hard.add_argument("--evidence", required=True)
    p_hard.add_argument("--force", action="store_true")
    p_hard.set_defaults(func=cmd_hard_stop)
    # complete
    p_complete = sub.add_parser("complete", help="mark execution complete")
    p_complete.add_argument("--state", required=True)
    p_complete.add_argument("--audit", required=True)
    p_complete.set_defaults(func=cmd_complete)

    p_reopen = sub.add_parser("reopen", help="invalidate false completion and reopen to earliest affected phase (preserves history)")
    p_reopen.add_argument("--state", required=True)
    p_reopen.add_argument("--phase", type=int, required=True, help="earliest affected phase to reopen (e.g., 5)")
    p_reopen.add_argument("--reason", required=True, help="reason for invalidation")
    p_reopen.add_argument("--evidence", required=True, help="evidence path or ID for invalidation")
    p_reopen.set_defaults(func=cmd_reopen)

    p_job = sub.add_parser("record-job", help="append a structured job record to the ledger (append-only)")
    p_job.add_argument("--state", required=True)
    p_job.add_argument("--attempt-id", required=True)
    p_job.add_argument("--job-key", required=True)
    p_job.add_argument("--job-type", required=True)
    p_job.add_argument("--event-type", default="SUBMITTED")
    p_job.add_argument("--slurm-job-id", default=None)
    p_job.add_argument("--status", required=True)
    p_job.add_argument("--fold", type=int, default=0)
    for extra in ("exit_code", "dependency_job_ids", "reason", "deployment_id", "evaluation_view", "backend", "aggregation", "metrics_path"):
        p_job.add_argument(f"--{extra.replace('_', '-')}", default=None)
    p_job.set_defaults(func=cmd_record_job)

    p_pr = sub.add_parser("record-pr", help="append a structured merged-PR record")
    p_pr.add_argument("--state", required=True)
    p_pr.add_argument("--pr-url", required=True)
    p_pr.add_argument("--phase", type=int, required=True)
    p_pr.add_argument("--head-sha", required=True)
    p_pr.add_argument("--merge-sha", required=True)
    p_pr.set_defaults(func=cmd_record_pr)

    p_rem = sub.add_parser("remove-evidence", help="remove a malformed evidence entry (records the correction)")
    p_rem.add_argument("--state", required=True)
    p_rem.add_argument("--phase", type=int, required=True)
    p_rem.add_argument("--entry", required=True)
    p_rem.add_argument("--reason", required=True)
    p_rem.add_argument("--replacement", default=None)
    p_rem.set_defaults(func=cmd_remove_evidence)

    p_cpr = sub.add_parser("correct-pr", help="append a corrected head/merge SHA for a previously recorded PR")
    p_cpr.add_argument("--state", required=True)
    p_cpr.add_argument("--pr-url", required=True)
    p_cpr.add_argument("--phase", type=int, default=None)
    p_cpr.add_argument("--head-sha", required=True)
    p_cpr.add_argument("--merge-sha", required=True)
    p_cpr.set_defaults(func=cmd_correct_pr)

    p_dep = sub.add_parser("record-deployment", help="append a structured deployment record")
    p_dep.add_argument("--state", required=True)
    p_dep.add_argument("--deployment-id", required=True)
    p_dep.add_argument("--experiment-id", required=True)
    p_dep.add_argument("--git-commit", required=True)
    p_dep.add_argument("--source-manifest-sha256", required=True)
    p_dep.add_argument("--deployed-code-path", required=True)
    p_dep.set_defaults(func=cmd_record_deployment)

    p_att = sub.add_parser("record-attempt", help="append a structured attempt record")
    p_att.add_argument("--state", required=True)
    p_att.add_argument("--attempt-id", required=True)
    p_att.add_argument("--deployment-id", required=True)
    p_att.add_argument("--experiment-id", required=True)
    p_att.add_argument("--fold", type=int, default=0)
    p_att.add_argument("--status", default="PLANNED")
    p_att.set_defaults(func=cmd_record_attempt)

    args = parser.parse_args()
    # Additional immutability checks before func?
    # For show/init we bypass
    if args.command in ("enter", "record", "pass", "hard-stop", "complete"):
        # Load and check that execution_id and runbook_sha not changed externally by comparing to file? Actually we can't detect change without prior snapshot.
        # We enforce that those fields are not modified to different values via direct edit: we could store original and on each operation ensure not changed arbitrarily.
        # Simpler: just ensure schema valid and hard_stop clearing not allowed silently.
        # The hard_stop clearing check is in complete/enter/pass already refusing if HARD_STOP.
        pass
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
