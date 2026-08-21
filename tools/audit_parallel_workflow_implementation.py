"""Implementation auditor for the parallel workflow execution ledger.

Modes:
- skeleton/preterminal development: --allow-incomplete validates schema,
  ordering, evidence existence, and refuses terminal completion.
- preterminal: requires status ACTIVE, phases 0-12 PASSED and Phase 13
  IN_PROGRESS; verifies structured evidence (PR SHAs, deployments, attempts,
  jobs), local artifact existence/hashes, full-suite evidence tied to the
  final SHA, CLI/docs agreement, clean auditor source, and optionally live
  scheduler accounting. Emits the audit JSON that `state complete` consumes.
- terminal: read-only verification of a COMPLETE ledger including the stored
  preterminal audit hash. The audit artifact never contains its own hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

SCHEMA_VERSION = "audiollm.parallel_workflow_execution.v1"
PHASES = list(range(14))
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_ID = re.compile(r"^[0-9TZ]+-[a-z0-9._-]+$")
AUDITOR_FILES = ("tools/audit_parallel_workflow_implementation.py", "tools/parallel_workflow_state.py")

WORKFLOW_COMMANDS = (
    "create", "deploy", "verify-deployment", "submit", "status",
    "collect", "validate", "compare", "finish", "plan-integration",
)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_path(entry: str) -> bool:
    if not isinstance(entry, str) or not entry.strip():
        return False
    if entry.startswith("http://") or entry.startswith("https://"):
        return False
    if len(entry) > 256 or any(c.isspace() for c in entry):
        return False
    return bool("/" in entry or entry.endswith((".log", ".json", ".txt", ".md", ".yaml")))


def check_evidence_exists(evidence: list[str], base: pathlib.Path) -> list[str]:
    missing = []
    for ev in evidence:
        if not ev or not isinstance(ev, str):
            missing.append(f"empty evidence entry: {ev}")
            continue
        if not _looks_like_path(ev):
            continue
        p = pathlib.Path(ev)
        if not p.is_absolute():
            p = base / ev
        try:
            exists = p.exists()
            size = p.stat().st_size if exists else None
        except OSError:
            missing.append(f"evidence path is not a valid filesystem path: {ev[:80]}")
            continue
        if not exists:
            missing.append(f"evidence path does not exist: {ev}")
        elif size == 0:
            missing.append(f"evidence file is empty: {ev}")
    return missing


def _git(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)


def check_clean_source(repo_root: pathlib.Path) -> list[str]:
    errors = []
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"], repo_root)
    if status.returncode != 0:
        return [f"git status failed: {status.stderr.strip()}"]
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        x, y, path = line[0], line[1], line[3:].strip()
        errors.append(f"verification worktree is dirty: {path} ({x}{y})")
        if any(path == f or path.startswith(f) for f in AUDITOR_FILES):
            errors.append(f"auditor/state-tool source is dirty: {path} ({x}{y})")
        if x not in (" ", "?", ""):
            errors.append(f"staged change present during audit: {path}")
    return errors


def check_source_revision(repo_root: pathlib.Path, expected_final_sha: str | None) -> list[str]:
    """Require the verification worktree and local integration ref at one exact SHA."""
    if not expected_final_sha:
        return ["expected final SHA is required for a substantive completion audit"]
    if not HEX40.match(expected_final_sha):
        return [f"expected final SHA is not a full 40-char SHA: {expected_final_sha!r}"]

    errors = []
    for ref, label in (("HEAD", "verification HEAD"), ("origin/main", "origin/main")):
        resolved = _git(["rev-parse", "--verify", ref], repo_root)
        if resolved.returncode != 0:
            errors.append(f"cannot resolve {label}: {resolved.stderr.strip()}")
            continue
        actual = resolved.stdout.strip()
        if actual != expected_final_sha:
            errors.append(f"{label} {actual} does not match expected final SHA {expected_final_sha}")
    return errors


def check_cli_docs_agreement(repo_root: pathlib.Path) -> list[str]:
    errors = []
    help_proc = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "exp.py"), "--help"],
        capture_output=True, text=True,
    )
    help_text = help_proc.stdout + help_proc.stderr
    if help_proc.returncode != 0:
        errors.append(f"exp.py --help failed with exit code {help_proc.returncode}")
    for cmd in WORKFLOW_COMMANDS:
        if f"{cmd}" not in help_text:
            errors.append(f"exp.py --help does not advertise command: {cmd}")
        cmd_help = subprocess.run(
            [sys.executable, str(repo_root / "tools" / "exp.py"), cmd, "--help"],
            capture_output=True, text=True,
        )
        if cmd_help.returncode != 0:
            errors.append(f"exp.py {cmd} --help failed with exit code {cmd_help.returncode}")
        elif "usage:" not in (cmd_help.stdout + cmd_help.stderr).lower():
            errors.append(f"exp.py {cmd} --help did not emit usage text")
    agents = repo_root / "AGENTS.md"
    if agents.is_file():
        agents_text = agents.read_text(encoding="utf-8", errors="replace")
        for cmd in WORKFLOW_COMMANDS:
            if cmd not in agents_text:
                errors.append(f"AGENTS.md does not mention implemented command: {cmd}")
    else:
        errors.append("AGENTS.md not found for CLI/docs agreement check")
    return errors


def check_live_jobs(state: dict, scheduler_host: str = "ozu647717@alogin2.bsc.es") -> list[str]:
    """Every recorded job id must resolve in sacct; none may be active."""
    job_ids = sorted({str(j.get("slurm_job_id")) for j in state.get("jobs", []) if j.get("slurm_job_id")})
    if not job_ids:
        return ["no job ids recorded for live reconciliation"]
    script = "sacct --noheader --parsable -j " + ",".join(job_ids) + " --format=JobIDRaw,State"
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", scheduler_host, script],
        capture_output=True, text=True, timeout=90,
    )
    if proc.returncode != 0:
        return [f"live sacct query failed rc={proc.returncode}: {proc.stderr.strip()}"]
    states: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and "." not in parts[0]:
            states[parts[0]] = parts[1]
    errors = []
    for jid in job_ids:
        if jid not in states:
            errors.append(f"job {jid} missing from live accounting")
        elif states[jid].upper().startswith(("RUNNING", "PENDING", "SUSPENDED")):
            errors.append(f"task-owned job still active: {jid} ({states[jid]})")
    return errors


def check_structured_records(
    state: dict,
    repo_root: pathlib.Path,
    expected_final_sha: str | None = None,
) -> list[str]:
    errors = []
    all_prs = state.get("prs", [])
    if not all_prs:
        errors.append("structured prs records are empty")
    # Latest record per PR URL is authoritative (corrections are appended).
    latest: dict[str, dict[str, Any]] = {}
    for pr in all_prs:
        latest[pr.get("pr_url", "")] = pr
    for url, pr in sorted(latest.items()):
        url = pr.get("pr_url", "")
        if not re.match(r"^https://github\.com/[^/]+/[^/]+/pull/\d+$", url):
            errors.append(f"malformed pr_url: {url!r}")
        if not HEX40.match(pr.get("head_sha", "")):
            errors.append(f"pr head_sha is not a full 40-char SHA: {pr.get('head_sha')!r}")
        if not HEX40.match(pr.get("merge_sha", "")):
            errors.append(f"pr merge_sha is not a full 40-char SHA: {pr.get('merge_sha')!r}")
        head_sha = pr.get("head_sha", "")
        merge_sha = pr.get("merge_sha", "")
        if HEX40.match(head_sha) and HEX40.match(merge_sha):
            for sha, role in ((head_sha, "head"), (merge_sha, "merge")):
                resolved = _git(["cat-file", "-e", f"{sha}^{{commit}}"], repo_root)
                if resolved.returncode != 0:
                    errors.append(f"PR {url} {role} SHA is not a local commit: {sha}")
            parents = _git(["rev-list", "--parents", "-n", "1", merge_sha], repo_root)
            # A normal merge commit must contain the recorded PR head. Squash and
            # rebase merges have one parent, so Git alone cannot prove the head
            # relationship; integration of their recorded merge SHA is still
            # checked below.
            if parents.returncode == 0 and len(parents.stdout.split()) > 2:
                ancestry = _git(["merge-base", "--is-ancestor", head_sha, merge_sha], repo_root)
                if ancestry.returncode != 0:
                    errors.append(f"PR {url} head SHA is not an ancestor of merge commit")
            if expected_final_sha and HEX40.match(expected_final_sha):
                integrated = _git(
                    ["merge-base", "--is-ancestor", merge_sha, expected_final_sha], repo_root
                )
                if integrated.returncode != 0:
                    errors.append(
                        f"PR {url} merge SHA is not integrated into expected final SHA"
                    )

    deployments = state.get("deployments", [])
    if not deployments:
        errors.append("structured deployments records are empty")
    for dep in deployments:
        if not HEX40.match(dep.get("git_commit", "")):
            errors.append(f"deployment {dep.get('deployment_id')} git_commit invalid")
        if not HEX64.match(dep.get("source_manifest_sha256", "")):
            errors.append(f"deployment {dep.get('deployment_id')} source_manifest_sha256 invalid")
        code_path = dep.get("deployed_code_path", "")
        if not code_path.startswith("/gpfs/") or "/code" not in code_path:
            errors.append(f"deployment {dep.get('deployment_id')} deployed_code_path invalid: {code_path}")

    attempts = state.get("attempts", [])
    if not attempts:
        errors.append("structured attempts records are empty")
    reportable = [a for a in attempts if a.get("status") == "REPORTABLE"]
    if not reportable:
        errors.append("no attempt reached REPORTABLE in the ledger")

    jobs = state.get("jobs", [])
    if not jobs:
        errors.append("structured jobs records are empty")
    final_attempts = {a.get("attempt_id") for a in reportable}
    final_jobs = [j for j in jobs if j.get("attempt_id") in final_attempts]
    train_ok = any(
        j.get("job_key") == "train" and j.get("status") == "COMPLETED"
        and str(j.get("exit_code", "")).startswith("0:0")
        for j in final_jobs
    )
    eval_ok = any(
        j.get("job_key") in ("best_eval", "evaluation", "standalone_eval")
        and j.get("job_type") == "evaluation"
        and j.get("status") == "COMPLETED"
        and str(j.get("exit_code", "")).startswith("0:0")
        for j in final_jobs
    )
    if not train_ok:
        errors.append("final smoke attempt lacks COMPLETED 0:0 train job")
    if not eval_ok:
        errors.append("final smoke attempt lacks COMPLETED 0:0 standalone evaluation job")
    cancelled_evals = [
        j for j in final_jobs
        if j.get("job_key") in ("best_eval", "evaluation", "standalone_eval")
        and j.get("status") == "CANCELLED"
    ]
    replaced = any(
        c.get("slurm_job_id") != r.get("slurm_job_id")
        for c in cancelled_evals
        for r in final_jobs
        if r.get("job_key") == c.get("job_key") and r.get("status") == "COMPLETED"
    )
    if cancelled_evals and not replaced:
        errors.append("cancelled evaluation exists without a completed replacement")
    return errors


def check_smoke_artifacts(state: dict, repo_root: pathlib.Path) -> list[str]:
    errors = []
    deployments = state.get("deployments", [])
    if not deployments:
        return ["cannot select final smoke evidence without a deployment record"]
    final_deployment_id = deployments[-1].get("deployment_id")

    # Corrections are append-only; the latest record for an attempt ID wins.
    latest_attempts: dict[str, dict[str, Any]] = {}
    for attempt in state.get("attempts", []):
        latest_attempts[attempt.get("attempt_id", "")] = attempt
    candidates = [
        attempt for attempt in latest_attempts.values()
        if attempt.get("deployment_id") == final_deployment_id
        and attempt.get("status") == "REPORTABLE"
    ]
    if len(candidates) != 1:
        return [
            f"expected exactly one REPORTABLE attempt for final deployment "
            f"{final_deployment_id}, found {len(candidates)}"
        ]
    attempt = candidates[0]
    fold_value = attempt.get("local_fold_path")
    if not fold_value:
        return [
            f"final REPORTABLE attempt {attempt.get('attempt_id')} lacks local_fold_path"
        ]
    fold = pathlib.Path(fold_value)
    if not fold.is_absolute():
        fold = repo_root / fold
    if not fold.is_dir():
        return [f"final smoke local_fold_path is not a directory: {fold}"]

    sidecars = {}
    for name in ("metadata.json", "status.json", "artifacts.json", "evaluations.json"):
        path = fold / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"final smoke sidecar missing/empty: {path}")
            continue
        try:
            sidecars[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"final smoke sidecar is invalid JSON: {path}: {exc}")
    if errors:
        return errors

    attempt_id = attempt.get("attempt_id")
    if sidecars["metadata.json"].get("attempt_id") != attempt_id:
        errors.append("final smoke metadata attempt_id does not match ledger")
    if sidecars["status.json"].get("attempt_id") != attempt_id:
        errors.append("final smoke status attempt_id does not match ledger")
    if sidecars["status.json"].get("state") != "REPORTABLE":
        errors.append("final smoke status sidecar is not REPORTABLE")

    evaluations = sidecars["evaluations.json"].get("evaluations", [])
    reportable = [ev for ev in evaluations if ev.get("reportable") is True]
    if len(reportable) != 1:
        errors.append(
            f"final smoke requires exactly one reportable evaluation, found {len(reportable)}"
        )
        return errors
    evaluation = reportable[0]
    artifact_records = {
        item.get("path"): item for item in sidecars["artifacts.json"].get("artifacts", [])
    }
    for key, label in (
        ("metrics_artifact_path", "standalone metrics"),
        ("predictions_artifact_path", "standalone subject predictions"),
    ):
        relative = evaluation.get(key)
        if not relative:
            errors.append(f"reportable evaluation lacks {key}")
            continue
        path = fold / relative
        record = artifact_records.get(relative)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{label} missing/empty: {path}")
        elif record is None:
            errors.append(f"{label} lacks artifacts.json record: {relative}")
        elif record.get("locally_verified") is not True:
            errors.append(f"{label} is not marked locally verified: {relative}")
        elif record.get("sha256") != sha256_file(path):
            errors.append(f"{label} hash does not match artifacts.json: {relative}")
    return errors


def check_full_suite_evidence(state: dict, repo_root: pathlib.Path, final_sha: str | None) -> list[str]:
    errors = []
    log_path = None
    for phase_key in ("11", "12", "13"):
        ph = state.get("phases", {}).get(phase_key, {})
        for ev in ph.get("evidence", []):
            if isinstance(ev, str) and "full_suite" in ev and _looks_like_path(ev):
                p = pathlib.Path(ev)
                if not p.is_absolute():
                    p = repo_root / ev
                if p.is_file():
                    log_path = p
                    break
        if log_path:
            break
    if log_path is None:
        errors.append("full-suite validation log not found in phase 11-13 evidence")
        return errors
    content = log_path.read_text(encoding="utf-8", errors="replace")
    if log_path.stat().st_size == 0:
        errors.append(f"full-suite log is empty: {log_path}")
    if not re.search(r"\b\d+ passed\b", content.lower()):
        errors.append(f"full-suite log does not record a passing run: {log_path}")
    if final_sha and final_sha not in content:
        errors.append(
            f"full-suite log is not tied to the final merged SHA {final_sha}: {log_path}"
        )
    ec = log_path.parent / (log_path.stem + ".exitcode")
    if not ec.is_file():
        errors.append(f"full-suite exit-code evidence is missing: {ec}")
    else:
        exit_code = ec.read_text(encoding="utf-8", errors="replace").strip()
        if exit_code != "0":
            errors.append(f"full-suite exit code is not exactly zero: {exit_code!r}")
    if "pytest_exit=0" not in content:
        errors.append(f"full-suite log lacks explicit pytest_exit=0 marker: {log_path}")
    return errors


def audit_state(
    state_path: pathlib.Path,
    allow_incomplete: bool = False,
    mode: str = "auto",
    verify_live_jobs: bool = False,
    expected_final_sha: str | None = None,
    repo_root_override: pathlib.Path | None = None,
) -> tuple[bool, list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        state = load_state(state_path)
    except Exception as e:
        return False, [f"failed to load state: {e}"], {}

    if repo_root_override is not None:
        repo_root = pathlib.Path(repo_root_override).resolve()
    else:
        repo_root = state_path.parent.parent.parent.resolve()  # outputs/<exec>/state.json -> repo

    if state.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: {state.get('schema_version')} != {SCHEMA_VERSION}")
    for i in range(14):
        key = str(i)
        if key not in state.get("phases", {}):
            errors.append(f"missing phase {key}")
            continue
        ph = state["phases"][key]
        status = ph.get("status")
        if status not in ("PENDING", "IN_PROGRESS", "PASSED"):
            errors.append(f"phase {key} invalid status {status}")
        if status == "PASSED":
            for j in range(i):
                if state["phases"][str(j)]["status"] != "PASSED":
                    errors.append(f"phase {i} PASSED but previous phase {j} is {state['phases'][str(j)]['status']}")
    in_progress = [k for k, v in state.get("phases", {}).items() if v.get("status") == "IN_PROGRESS"]
    if len(in_progress) > 1:
        errors.append(f"multiple phases IN_PROGRESS: {in_progress}")
    current = state.get("current_phase")
    if current not in PHASES:
        errors.append(f"invalid current_phase {current}")
    exec_id = state.get("execution_id", "")
    if not exec_id or "parallel-workflow" not in exec_id:
        errors.append(f"invalid execution_id {exec_id}")
    if not state.get("runbook_sha256_at_start"):
        errors.append("missing runbook_sha256_at_start")
    status = state.get("status")
    if status not in ("ACTIVE", "HARD_STOP", "COMPLETE"):
        errors.append(f"invalid status {status}")
    if status == "COMPLETE":
        not_passed = [k for k, v in state.get("phases", {}).items() if v.get("status") != "PASSED"]
        if not_passed:
            errors.append(f"status COMPLETE but phases not all PASSED: {not_passed}")

    # Evidence must exist on disk for every passed phase.
    for i in PHASES:
        ph = state["phases"].get(str(i), {})
        if ph.get("status") == "PASSED":
            if len(ph.get("evidence", [])) == 0:
                errors.append(f"phase {i} PASSED but evidence empty")
            else:
                errors.extend(
                    f"phase {i}: {m}" for m in check_evidence_exists(ph["evidence"], repo_root)
                )

    # Mode resolution.
    all_passed = all(v.get("status") == "PASSED" for v in state["phases"].values())
    if mode == "auto":
        mode = "terminal" if status == "COMPLETE" else "preterminal"

    if mode == "preterminal":
        if allow_incomplete:
            pass  # development audits skip substantive gates
        else:
            if status != "ACTIVE":
                errors.append(f"preterminal audit requires ACTIVE status, got {status}")
            if current != 13:
                errors.append(f"preterminal audit requires current_phase 13, got {current}")
            for i in range(13):
                if state["phases"][str(i)].get("status") != "PASSED":
                    errors.append(f"preterminal audit requires phase {i} PASSED")
            if state["phases"]["13"].get("status") != "IN_PROGRESS":
                errors.append("preterminal audit requires Phase 13 IN_PROGRESS")
            errors.extend(check_source_revision(repo_root, expected_final_sha))
            errors.extend(check_structured_records(state, repo_root, expected_final_sha))
            errors.extend(check_smoke_artifacts(state, repo_root))
            errors.extend(check_full_suite_evidence(state, repo_root, expected_final_sha))
            errors.extend(check_cli_docs_agreement(repo_root))
            if verify_live_jobs:
                errors.extend(check_live_jobs(state))
    elif mode == "terminal":
        if status != "COMPLETE":
            errors.append(f"terminal audit requires COMPLETE status, got {status}")
        if not all_passed:
            errors.append("terminal audit requires all phases PASSED")
        stored = state.get("completion", {})
        if not stored.get("preterminal_audit_path"):
            errors.append("COMPLETE state lacks recorded preterminal audit path")
        else:
            pa = pathlib.Path(stored["preterminal_audit_path"])
            if not pa.is_absolute():
                pa = repo_root / pa
            if not pa.is_file():
                errors.append(f"recorded preterminal audit missing: {pa}")
            elif not HEX64.match(stored.get("preterminal_audit_sha256", "")):
                errors.append("recorded preterminal audit hash invalid")
            elif sha256_file(pa) != stored.get("preterminal_audit_sha256"):
                errors.append("recorded preterminal audit hash does not match file (evidence tampering)")
            else:
                try:
                    pa_json = json.loads(pa.read_text(encoding="utf-8"))
                    if pa_json.get("passed") is not True or pa_json.get("mode") != "preterminal":
                        errors.append("stored preterminal audit was not a passing preterminal audit")
                except Exception as e:
                    errors.append(f"stored preterminal audit unreadable: {e}")
        errors.extend(check_source_revision(repo_root, expected_final_sha))
        errors.extend(check_structured_records(state, repo_root, expected_final_sha))
        errors.extend(check_smoke_artifacts(state, repo_root))
        errors.extend(check_full_suite_evidence(state, repo_root, expected_final_sha))
        errors.extend(check_cli_docs_agreement(repo_root))
        if verify_live_jobs:
            errors.extend(check_live_jobs(state))

    # Hard stop consistency.
    if status == "HARD_STOP" and state.get("hard_stop") is None:
        errors.append("status HARD_STOP but hard_stop is null")

    # Clean-source gate applies to substantive audits only.
    if mode in ("preterminal", "terminal") and not allow_incomplete:
        errors.extend(check_clean_source(repo_root))

    return (len(errors) == 0), errors + warnings, state


def main():
    parser = argparse.ArgumentParser(description="Audit parallel workflow implementation ledger")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--mode", choices=["auto", "preterminal", "terminal"], default="auto")
    parser.add_argument("--verify-live-jobs", action="store_true",
                        help="reconcile recorded job ids against live MN5 accounting")
    parser.add_argument("--expected-final-sha", default=None,
                        help="full SHA the full-suite evidence must be tied to")
    parser.add_argument("--repo-root", default=None,
                        help="verification repository for clean-source/CLI/docs gates "
                             "(default: the repository containing the state file)")
    args = parser.parse_args()
    state_path = pathlib.Path(args.state)
    passed, messages, state = audit_state(
        state_path,
        allow_incomplete=args.allow_incomplete,
        mode=args.mode,
        verify_live_jobs=args.verify_live_jobs,
        expected_final_sha=args.expected_final_sha,
        repo_root_override=pathlib.Path(args.repo_root) if args.repo_root else None,
    )
    audit = {
        "schema_version": "audiollm.parallel_workflow_audit.v1",
        "state_path": str(state_path),
        "allow_incomplete": args.allow_incomplete,
        "mode": args.mode if args.mode != "auto" else ("terminal" if state.get("status") == "COMPLETE" else "preterminal"),
        "passed": passed,
        "errors": messages if not passed else [],
        "warnings": [] if passed else messages,
        "status": state.get("status") if state else "UNKNOWN",
        "current_phase": state.get("current_phase") if state else None,
        "phases": {k: v["status"] for k, v in state.get("phases", {}).items()} if state else {},
        "expected_final_sha": args.expected_final_sha,
    }
    out_path = pathlib.Path(args.output) if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"audit written to {out_path} passed={passed}")
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        for m in messages:
            print(f"ERROR: {m}", file=sys.stderr)
        print("audit FAILED")
        return 1
    print("audit PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
