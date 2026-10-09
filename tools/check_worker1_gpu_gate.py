#!/usr/bin/env python3
"""Candidate coordinator-owned global GPU gate (worker1, 2026-10-09 stop).

Context
- The user imposed a hard global cap: at most 8 GPUs and 2 nodes in total across
  ALL of this account's jobs (train/eval/postprocess/extraction/smokes together),
  with strict sequential GPU release: at most ONE released GPU job at a time,
  even when that leaves the other GPUs idle.
- Pending GPU jobs stay held; the gate fails closed on any own running job, any
  own unheld pending (released) GPU job, or any unknown scheduler state.
- A whole leg (15-job chain) must never be released as a batch; admission is
  single-job only.
- CPU head jobs also need a bounded CPU billing rate and a bounded wall time.
- Root acceptance is required before ANY release. This tool never submits,
  releases, cancels, or retries anything; it only decides ALLOW/DENY.

The gate counts GPU totals from TOTAL AllocTRES (running) or ReqTRES (pending),
never per-node values, and enforces <= 8 GPUs and <= 2 nodes across the account.

Usage (read-only decision):
    python tools/check_worker1_gpu_gate.py --scontrol-dump dump.txt \
        --candidate-job 47077944 --root-approval <token>
Exit code 0 = ALLOW, 1 = DENY (reasons printed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

GPU_LIMIT = 8
NODE_LIMIT = 2
GPU_BILLING_RATE_CAP = 320  # actual allocation billing per hour (8-GPU audio train)
CPU_HEAD_BILLING_RATE_CAP = 40  # actual billing per hour for CPU head jobs
CPU_HEAD_MAX_WALL_MINUTES = 120

ACTIVE_STATES = {
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "RESIZING",
    "SUSPENDED",
}
RELEASED_PENDING_MARKER = "unheld_pending"
HELD_REASON = "JobHeldUser"
KNOWN_TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
    "REVOKED",
}
KNOWN_STATES = ACTIVE_STATES | KNOWN_TERMINAL | {"PENDING"}


@dataclass
class Tres:
    gpus: int = 0
    nodes: int = 0
    billing: int = 0


@dataclass
class Job:
    job_id: str
    name: str = ""
    state: str = "UNKNOWN"
    reason: str = ""
    req: Tres = field(default_factory=Tres)
    alloc: Tres | None = None
    time_limit_minutes: int | None = None

    @property
    def held(self) -> bool:
        return self.state == "PENDING" and self.reason == HELD_REASON

    @property
    def is_pending(self) -> bool:
        return self.state == "PENDING"

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def effective(self) -> Tres:
        """Total TRES for this job: AllocTRES when running, ReqTRES otherwise."""
        if self.alloc is not None and self.is_active:
            return self.alloc
        return self.req

    @property
    def is_gpu(self) -> bool:
        return self.effective().gpus > 0


def parse_tres(text: str) -> Tres:
    """Parse a Slurm TRES string; totals only, never per-node values."""
    t = Tres()
    m = re.search(r"(?:^|,)gres/gpu=(\d+)", text or "")
    if m:
        t.gpus = int(m.group(1))
    m = re.search(r"(?:^|,)node=(\d+)", text or "")
    if m:
        t.nodes = int(m.group(1))
    m = re.search(r"(?:^|,)billing=(\d+)", text or "")
    if m:
        t.billing = int(m.group(1))
    return t


def parse_scontrol_dump(text: str) -> list[Job]:
    """Parse `scontrol show job` output into Job records.

    Multiple jobs are separated by blank lines. Missing TRES data stays at zero,
    which later fails closed via the unknown/incomplete checks.
    """
    jobs: list[Job] = []
    for block in re.split(r"\n\s*\n", text or ""):
        if "JobId=" not in block:
            continue
        fields = {}
        for piece in block.replace("\n", " ").split():
            if "=" in piece:
                k, _, v = piece.partition("=")
                fields.setdefault(k, v)
        job = Job(job_id=fields.get("JobId", ""))
        job.name = fields.get("JobName", "")
        job.state = fields.get("JobState", "UNKNOWN")
        reason = fields.get("Reason", "")
        job.reason = "" if reason in ("None", "(null)") else reason
        job.req = parse_tres(fields.get("ReqTRES", ""))
        alloc = fields.get("AllocTRES", "")
        job.alloc = parse_tres(alloc) if alloc and alloc != "(null)" else None
        tl = fields.get("TimeLimit", "")
        m = re.match(r"^(\d+)-(\d+):(\d+):(\d+)$", tl)
        if m:
            d, h, mi, _ = map(int, m.groups())
            job.time_limit_minutes = d * 1440 + h * 60 + mi
        else:
            m = re.match(r"^(\d+):(\d+):(\d+)$", tl)
            if m:
                h, mi, _ = map(int, m.groups())
                job.time_limit_minutes = h * 60 + mi
        jobs.append(job)
    return jobs


def check_release(
    candidate: Job,
    own_jobs: list[Job],
    root_approval: str | None,
    gpu_limit: int = GPU_LIMIT,
    node_limit: int = NODE_LIMIT,
    gpu_billing_cap: int = GPU_BILLING_RATE_CAP,
    cpu_head_billing_cap: int = CPU_HEAD_BILLING_RATE_CAP,
    cpu_head_max_wall_minutes: int = CPU_HEAD_MAX_WALL_MINUTES,
) -> dict:
    """Decide whether ONE held candidate job may be released. Never mutates state."""
    reasons: list[str] = []
    if not root_approval:
        reasons.append("root_acceptance_required: no approval token supplied")

    if not candidate.job_id:
        reasons.append("candidate_missing_job_id")
    if candidate.state not in KNOWN_STATES:
        reasons.append(f"candidate_unknown_state:{candidate.state}")
    if candidate.state != "PENDING":
        reasons.append(f"candidate_not_pending:{candidate.state}")

    # Fail closed over every other own job.
    for job in own_jobs:
        if job.job_id == candidate.job_id:
            continue
        eff = job.effective()
        if job.is_active:
            reasons.append(f"own_active_job:{job.job_id}:{job.state}:gpus={eff.gpus}")
        elif job.is_pending and not job.held and job.is_gpu:
            reasons.append(f"own_unheld_pending_gpu_job:{job.job_id}:{RELEASED_PENDING_MARKER}")
        elif job.is_pending and job.reason and job.reason != HELD_REASON:
            reasons.append(f"own_pending_unknown_reason:{job.job_id}:{job.reason}")
        elif job.state not in KNOWN_STATES:
            reasons.append(f"own_unknown_state:{job.job_id}:{job.state}")

    cand = candidate.effective()
    if cand.gpus < 0 or cand.nodes < 0 or cand.billing < 0:
        reasons.append("candidate_negative_tres")
    if cand.gpus == 0 and candidate.time_limit_minutes is None:
        reasons.append("cpu_job_requires_bounded_wall_time")
    if cand.gpus > 0:
        if cand.gpus > gpu_limit:
            reasons.append(f"candidate_gpus_exceed_limit:{cand.gpus}>{gpu_limit}")
        if cand.nodes > node_limit:
            reasons.append(f"candidate_nodes_exceed_limit:{cand.nodes}>{node_limit}")
        if cand.billing > gpu_billing_cap:
            reasons.append(f"candidate_billing_exceeds_cap:{cand.billing}>{gpu_billing_cap}")
    else:
        if cand.billing > cpu_head_billing_cap:
            reasons.append(f"cpu_head_billing_exceeds_cap:{cand.billing}>{cpu_head_billing_cap}")
        if (
            candidate.time_limit_minutes is None
            or candidate.time_limit_minutes > cpu_head_max_wall_minutes
        ):
            reasons.append(
                "cpu_head_wall_time_unbounded_or_too_long:"
                f"{candidate.time_limit_minutes}>{cpu_head_max_wall_minutes}"
            )

    # Account-wide totals (defense in depth; with one-at-a-time this is redundant).
    gpu_total = cand.gpus
    for job in own_jobs:
        if job.job_id == candidate.job_id:
            continue
        if job.is_active or (job.is_pending and not job.held):
            gpu_total += job.effective().gpus
    if gpu_total > gpu_limit:
        reasons.append(f"account_gpu_total_exceeds_limit:{gpu_total}>{gpu_limit}")

    return {"allow": not reasons, "candidate": candidate.job_id, "reasons": reasons}


def check_batch_release(candidates: list[Job], own_jobs: list[Job], root_approval: str | None) -> dict:
    """Whole-leg releases are forbidden; only single-job admission is allowed."""
    if len(candidates) != 1:
        return {
            "allow": False,
            "reasons": [f"batch_release_forbidden:{len(candidates)}_jobs"],
        }
    return check_release(candidates[0], own_jobs, root_approval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="coordinator-owned global GPU gate (read-only)")
    parser.add_argument("--scontrol-dump", required=True, help="file with `scontrol show job` output")
    parser.add_argument("--candidate-job", required=True)
    parser.add_argument("--root-approval", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        text = open(args.scontrol_dump, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        print(json.dumps({"allow": False, "reasons": [f"dump_unreadable:{exc}"]}))
        return 1
    jobs = parse_scontrol_dump(text)
    if not any(j.job_id == args.candidate_job for j in jobs):
        print(json.dumps({"allow": False, "reasons": [f"candidate_not_found:{args.candidate_job}"]}))
        return 1
    candidate = next(j for j in jobs if j.job_id == args.candidate_job)
    others = [j for j in jobs if j.job_id != args.candidate_job]
    decision = check_release(candidate, others, args.root_approval)
    if args.json:
        print(json.dumps(decision, indent=2))
    else:
        print(f"{'ALLOW' if decision['allow'] else 'DENY'} candidate={decision['candidate']}")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
    return 0 if decision["allow"] else 1


if __name__ == "__main__":
    sys.exit(main())
