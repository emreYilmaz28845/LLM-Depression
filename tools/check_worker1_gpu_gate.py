#!/usr/bin/env python3
"""Candidate coordinator-owned global GPU gate (worker1, 2026-10-09 stop).

READ-ONLY DECISION AID. This tool never submits, releases, cancels, or retries.
The real release is coordinator-owned: the coordinator holds an atomic lease
lock and must build a FRESH full own-queue + all-TRES snapshot before every
single release, with all other pending jobs held. The worker cannot release
independently; this tool only answers "would releasing exactly this one held
job be admissible under the current snapshot?".

Rules enforced (user directives, reviewed 2026-10-09):
- Hard global cap: at most 8 GPUs and 2 nodes TOTAL across ALL own-account
  jobs (train/eval/postprocess/extraction/smokes together), counted from
  total AllocTRES (running) or ReqTRES (pending), never per-node values.
- Strictly sequential: at most ONE released GPU job at a time; every other
  GPU job stays held (dependents included).
- Fail closed on: any own active job, any own unheld pending GPU job, any
  unknown state, duplicate/incomplete job records, malformed TRES.
- Candidate must be PENDING with reason exactly JobHeldUser; its ReqTRES must
  be present and valid: node >= 1, billing > 0; a GPU job must declare its
  GPU count (missing/undecipherable declaration fails, never falls through
  as CPU).
- CPU head jobs need bounded CPU billing and bounded wall time.
- Whole-leg batch admission is forbidden (single-job only).
- Root approval token required before any release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

GPU_LIMIT = 8
NODE_LIMIT = 2
GPU_BILLING_RATE_CAP = 320  # allocated billing per hour (8-GPU audio train)
CPU_HEAD_BILLING_RATE_CAP = 40  # allocated billing per hour for CPU head jobs
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

TRES_KEYS = {"gres/gpu": "gpus", "node": "nodes", "billing": "billing"}


class TresParseError(ValueError):
    """Raised on malformed or undecipherable TRES content (fail closed)."""


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
    req_present: bool = True
    alloc_present: bool = False
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
        if self.is_active and self.alloc_present and self.alloc is not None:
            return self.alloc
        return self.req

    @property
    def is_gpu(self) -> bool:
        return self.effective().gpus > 0


def parse_tres(text: str) -> tuple[Tres, bool]:
    """Strict TRES parse. Returns (tres, present). Raises on malformed tokens.

    A token whose key is one we track (gres/gpu, node, billing) must carry a
    non-negative integer; anything else is a hard parse failure so that a
    broken declaration can never silently fall through as a CPU/zero request.
    """
    present = bool(text and text.strip() and text.strip() != "(null)")
    t = Tres()
    if not present:
        return t, False
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise TresParseError(f"token_without_equals:{token}")
        key, _, value = token.partition("=")
        key = key.strip()
        value = value.strip()
        if key in TRES_KEYS:
            if not re.fullmatch(r"\d+", value):
                raise TresParseError(f"malformed_tres_value:{key}={value}")
            setattr(t, TRES_KEYS[key], int(value))
    return t, True


def parse_scontrol_dump(text: str) -> list[Job]:
    """Parse `scontrol show job` output. Fails closed on duplicate/incomplete records."""
    jobs: list[Job] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", text or ""):
        if "JobId=" not in block:
            continue
        fields: dict[str, str] = {}
        for piece in block.replace("\n", " ").split():
            if "=" in piece:
                k, _, v = piece.partition("=")
                fields.setdefault(k, v)
        job_id = fields.get("JobId", "")
        if not job_id:
            raise TresParseError("job_record_without_jobid")
        if job_id in seen:
            raise TresParseError(f"duplicate_job_records:{job_id}")
        seen.add(job_id)
        job = Job(job_id=job_id)
        job.name = fields.get("JobName", "")
        job.state = fields.get("JobState", "UNKNOWN")
        reason = fields.get("Reason", "")
        job.reason = "" if reason in ("None", "(null)") else reason
        if "ReqTRES" not in fields:
            raise TresParseError(f"incomplete_record_missing_reqtres:{job_id}")
        job.req, job.req_present = parse_tres(fields.get("ReqTRES", ""))
        if not job.req_present:
            raise TresParseError(f"empty_reqtres:{job_id}")
        alloc_raw = fields.get("AllocTRES", "")
        if alloc_raw and alloc_raw not in ("(null)",):
            job.alloc, job.alloc_present = parse_tres(alloc_raw)
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
    """Decide whether ONE held candidate may be released. Never mutates state."""
    reasons: list[str] = []
    if not root_approval:
        reasons.append("root_acceptance_required: no approval token supplied")

    if not candidate.job_id:
        reasons.append("candidate_missing_job_id")
    if candidate.state not in KNOWN_STATES:
        reasons.append(f"candidate_unknown_state:{candidate.state}")
    if not (candidate.state == "PENDING" and candidate.reason == HELD_REASON):
        reasons.append(
            f"candidate_must_be_pending_with_exact_{HELD_REASON}:"
            f"state={candidate.state}:reason={candidate.reason or 'None'}"
        )
    if not candidate.req_present:
        reasons.append("candidate_req_tres_missing")
    else:
        req = candidate.req
        if req.nodes < 1:
            reasons.append(f"candidate_nodes_missing_or_zero:{req.nodes}")
        if req.billing <= 0:
            reasons.append(f"candidate_billing_missing_or_zero:{req.billing}")
        if req.gpus > 0:
            if req.gpus > gpu_limit:
                reasons.append(f"candidate_gpus_exceed_limit:{req.gpus}>{gpu_limit}")
            if req.nodes > node_limit:
                reasons.append(f"candidate_nodes_exceed_limit:{req.nodes}>{node_limit}")
            if req.billing > gpu_billing_cap:
                reasons.append(f"candidate_billing_exceeds_cap:{req.billing}>{gpu_billing_cap}")
        else:
            if req.billing > cpu_head_billing_cap:
                reasons.append(f"cpu_head_billing_exceeds_cap:{req.billing}>{cpu_head_billing_cap}")
            if (
                candidate.time_limit_minutes is None
                or candidate.time_limit_minutes > cpu_head_max_wall_minutes
            ):
                reasons.append(
                    "cpu_head_wall_time_unbounded_or_too_long:"
                    f"{candidate.time_limit_minutes}>{cpu_head_max_wall_minutes}"
                )

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

    gpu_total = candidate.req.gpus if candidate.req_present else 0
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
        return {"allow": False, "reasons": [f"batch_release_forbidden:{len(candidates)}_jobs"]}
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
        jobs = parse_scontrol_dump(text)
    except (OSError, TresParseError) as exc:
        print(json.dumps({"allow": False, "reasons": [f"snapshot_unusable:{exc}"]}))
        return 1
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
