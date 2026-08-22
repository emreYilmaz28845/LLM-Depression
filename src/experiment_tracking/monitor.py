"""Remote-aware monitoring, failure classification, retry planning, resume.

All scheduler queries go through SSH to the documented MN5 scheduler login
(never local squeue/sacct, never transfer1). Terminal success requires
top-level COMPLETED with ExitCode 0:0 plus required artifacts.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

DEFAULT_SCHEDULER_HOST = "ozu647717@alogin2.bsc.es"

TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
TRANSIENT_INFRASTRUCTURE_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "TIMEOUT", "OUT_OF_MEMORY"}


class MonitorError(RuntimeError):
    """Raised on contradictory evidence, unknown jobs, or invalid advancement."""


def parse_squeue(text: str) -> dict[str, dict[str, str]]:
    """Parse `squeue -o "%i %T %M %j"` style output (headerless)."""
    jobs: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 2:
            continue
        job_id = parts[0].lstrip("(").rstrip(")")
        if not job_id.isdigit():
            continue
        jobs[job_id] = {
            "JobIDRaw": job_id,
            "State": parts[1],
            "Elapsed": parts[2] if len(parts) > 2 else "",
            "JobName": parts[3] if len(parts) > 3 else "",
        }
    return jobs


_SACCT_LINE = re.compile(r"^(?P<id>\d+)(?:\.(?P<step>\S+))?\|(?P<rest>.*)$")


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    """Parse `sacct --parsable --noheader --format=JobIDRaw,State,ExitCode,...`.

    Steps (JobID like `123.batch`) are folded away; only top-level rows remain.
    """
    jobs: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line:
            continue
        match = _SACCT_LINE.match(line)
        if not match or match.group("step"):
            continue
        fields = match.group("rest").split("|")
        jobs[match.group("id")] = {
            "JobIDRaw": match.group("id"),
            "State": fields[0] if len(fields) > 0 else "",
            "ExitCode": fields[1] if len(fields) > 1 else "",
            "Elapsed": fields[2] if len(fields) > 2 else "",
            "NodeList": fields[3] if len(fields) > 3 else "",
        }
    return jobs


class SchedulerClient:
    """Queries the remote scheduler login over non-interactive SSH."""

    def __init__(
        self,
        host: str = DEFAULT_SCHEDULER_HOST,
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.host = host
        self._runner = runner

    def _run(self, script: str) -> subprocess.CompletedProcess:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.host, script]
        if self._runner is not None:
            return self._runner(argv)
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)

    def squeue(self, job_ids: list[str]) -> dict[str, dict[str, str]]:
        if not job_ids:
            return {}
        proc = self._run(
            "squeue --noheader -o '%.50i %.10T %.12M %.40j' -j " + ",".join(job_ids)
        )
        if proc.returncode != 0:
            raise MonitorError(f"squeue failed rc={proc.returncode}: {proc.stderr.strip()}")
        return parse_squeue(proc.stdout)

    def sacct(self, job_ids: list[str]) -> dict[str, dict[str, str]]:
        if not job_ids:
            return {}
        proc = self._run(
            "sacct --noheader --parsable -j " + ",".join(job_ids)
            + " --format=JobIDRaw,State,ExitCode,Elapsed,NodeList%30"
        )
        if proc.returncode != 0:
            raise MonitorError(f"sacct failed rc={proc.returncode}: {proc.stderr.strip()}")
        return parse_sacct(proc.stdout)


def classify_failure(state: str, exit_code: str, dependency_cancelled: bool = False) -> str:
    state_u = (state or "").upper()
    if dependency_cancelled or state_u.startswith("CANCELLED"):
        return "cancelled_dependency"
    if any(state_u.startswith(s) for s in TRANSIENT_INFRASTRUCTURE_STATES):
        return "transient_infrastructure"
    if state_u.startswith("FAILED") or state_u in TERMINAL_FAILURE_STATES:
        return "deterministic_code_config"
    if exit_code and not exit_code.startswith("0:0"):
        return "deterministic_code_config"
    return "unknown"


def terminal_event_type(state: str, exit_code: str) -> str:
    state_u = (state or "").upper()
    if state_u.startswith("CANCELLED"):
        return "CANCELLED"
    if state_u == "COMPLETED" and exit_code == "0:0":
        return "COMPLETED"
    return "FAILED"


@dataclass
class JobReconciliation:
    slurm_job_id: str
    job_key: str
    queue_state: str | None = None
    account_state: str | None = None
    exit_code: str | None = None
    elapsed: str | None = None
    node: str | None = None
    artifacts_ok: bool | None = None
    classification: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def terminal_success(self) -> bool:
        return (
            self.account_state == "COMPLETED"
            and self.exit_code == "0:0"
            and self.queue_state is None
            and self.artifacts_ok is True
        )

    @property
    def terminal_failure(self) -> bool:
        state_u = (self.account_state or "").upper()
        return any(state_u.startswith(s) for s in TERMINAL_FAILURE_STATES)


def reconcile_job(
    record: dict[str, Any],
    queue: dict[str, dict[str, str]],
    accounting: dict[str, dict[str, str]],
    artifacts_ok: bool | None,
) -> JobReconciliation:
    job_id = str(record.get("slurm_job_id") or "")
    if not job_id.isdigit():
        raise MonitorError(f"recorded job has no valid Slurm id: {record!r}")
    rec = JobReconciliation(
        slurm_job_id=job_id,
        job_key=str(record.get("job_key", "?")),
        artifacts_ok=artifacts_ok,
    )
    in_queue = queue.get(job_id)
    in_acct = accounting.get(job_id)
    if in_queue:
        rec.queue_state = in_queue["State"]
    if in_acct:
        rec.account_state = in_acct["State"]
        rec.exit_code = in_acct["ExitCode"]
        rec.elapsed = in_acct["Elapsed"]
        rec.node = in_acct["NodeList"]
    if rec.terminal_failure:
        dep_ids = [str(d) for d in record.get("dependency_job_ids", [])]
        dep_cancelled = any(
            accounting.get(dep, {}).get("State", "").startswith("CANCELLED") for dep in dep_ids
        )
        rec.classification = classify_failure(rec.account_state or "", rec.exit_code or "", dep_cancelled)
    elif rec.account_state == "COMPLETED" and rec.exit_code != "0:0":
        rec.classification = classify_failure("", rec.exit_code or "")
    if rec.account_state == "COMPLETED" and rec.exit_code == "0:0" and artifacts_ok is False:
        rec.problems.append("job completed but required artifacts missing")
    if rec.account_state == "COMPLETED" and rec.exit_code != "0:0":
        rec.problems.append(f"COMPLETED with nonzero ExitCode {rec.exit_code}")
    return rec


def plan_retry(
    failed: JobReconciliation,
    existing_retries: list[dict[str, Any]],
    new_attempt_id_fn: Callable[[], str],
) -> dict[str, Any]:
    """Bounded retry: one unchanged retry per demonstrated transient failure."""
    if failed.classification == "transient_infrastructure":
        prior = [r for r in existing_retries if r.get("resubmission_of_job_id") == failed.slurm_job_id]
        if prior:
            raise MonitorError(
                f"retry budget exhausted for job {failed.slurm_job_id}: "
                f"{len(prior)} unchanged retry already recorded; deterministic fix + new deployment required"
            )
        return {
            "allowed": True,
            "kind": "unchanged_transient_retry",
            "resubmission_of_job_id": failed.slurm_job_id,
            "new_attempt_id": new_attempt_id_fn(),
        }
    if failed.classification == "cancelled_dependency":
        return {
            "allowed": True,
            "kind": "resubmit_after_dependency_fix",
            "resubmission_of_job_id": failed.slurm_job_id,
            "new_attempt_id": new_attempt_id_fn(),
            "note": "fix or rerun the cancelled dependency first; new attempt identity required",
        }
    if failed.classification == "deterministic_code_config":
        return {
            "allowed": True,
            "kind": "code_fix_new_deployment_new_attempt",
            "resubmission_of_job_id": failed.slurm_job_id,
            "new_attempt_id": new_attempt_id_fn(),
            "note": "validate a fix, deploy a new source snapshot, then submit the new attempt",
        }
    return {
        "allowed": False,
        "kind": "diagnose_first",
        "classification": failed.classification,
        "note": "unknown failure requires diagnosis before any retry",
    }


def check_resume_no_duplicate_submission(
    job_events: list[dict[str, Any]], job_key: str
) -> dict[str, Any] | None:
    """Return the latest SUBMITTED/terminal event for job_key, else None.

    A caller must refuse to submit when this returns an event whose Slurm job
    may still exist or already terminated successfully.
    """
    latest = None
    for event in reversed(job_events):
        if event.get("job_key") == job_key:
            latest = event
            break
    return latest


def validate_lifecycle_advancement(current: str, target: str) -> None:
    from .lifecycle import InvalidTransitionError, is_allowed_transition

    if not is_allowed_transition(current, target):
        raise MonitorError(
            f"invalid lifecycle advancement {current} -> {target}; "
            f"use official transitions only"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
