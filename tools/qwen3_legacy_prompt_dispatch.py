#!/usr/bin/env python3
"""Fail-closed progressive dispatcher for the Worker 4 legacy-prompt campaign.

Admission semantics (binding):

- The user-wide queue must be below 350, verified from a raw delimited
  ``squeue`` query with the SSH return code checked. A failed SSH command, a
  non-zero scheduler return code, or an unparseable line refuses submission;
  it is never treated as "no jobs".
- The lane's own nonterminal count is computed from every authoritative
  delivered job ID recorded for this lane: the append-only submission ledger
  plus every local fold-sidecar ``jobs.jsonl`` SUBMITTED event (covers smokes,
  auxiliaries, downstream jobs and historical attempts). IDs are reconciled
  against the live scheduler (``squeue`` then ``sacct``); IDs that cannot be
  resolved count as nonterminal (fail closed), and uncertain/failed ledger
  records count conservatively as two jobs.
- Before every single fit (two jobs: train + best_eval) both counts are
  re-checked. Submission proceeds only while ``user_queue < 350`` and
  ``own_nonterminal + 2 <= 80``. Planned waves larger than the 80-job lane
  allocation (or larger than 40 fits) are rejected outright.
- The ledger is append-only. A partial or uncertain submission is preserved
  and never retried automatically.

The same accounting must gate head-chain dispatch: reconcile own nonterminal
jobs and refuse while ``own + 2 > 80`` before each head attempt.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

LANE = Path(__file__).resolve().parents[1]
EVIDENCE = LANE / "outputs/qwen3_legacy_prompt_20261008"
MATRIX = EVIDENCE / "matrix.json"
LEDGER = EVIDENCE / "submissions.jsonl"
RUN_ROOT = LANE / "output_model/qwen3_legacy_prompt_20261008"
CAMPAIGN = "qwen3_legacy_prompt_20261008"
DEPLOYMENT = "feat-qwen3-legacy-prompt-20261008-20261008T112715Z-c51ef4be-2984d970"
ENVIRONMENTS = {
    "text_only": "/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate",
    "audio_only": "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate",
    "audio_text": "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate",
}
SCHEDULER = "ozu647717@alogin2.bsc.es"
USER_QUEUE_STOP = 350
LANE_CAP_NONTERMINAL = 80
WAVE_HARD_CAP_FITS = 40
JOBS_PER_FIT = 2

NONTERMINAL_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "RESIZING",
    "SUSPENDED",
    "REQUEUED",
    "UNKNOWN",
}
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}


class AdmissionError(RuntimeError):
    """Raised when admission cannot be verified or the budget is exceeded."""


def run_ssh(command: str, runner: Callable[..., subprocess.CompletedProcess] | None = None) -> str:
    runner = runner or subprocess.run
    result = runner(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            SCHEDULER,
            command,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise AdmissionError(
            f"scheduler query failed (rc={result.returncode}): {result.stderr.strip()[:200]}"
        )
    return result.stdout


def parse_delimited(output: str) -> dict[str, str]:
    """Parse ``jobid|state`` lines; refuse unparseable output."""
    states: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        job_id, state = line.split("|", 1)
        job_id, state = job_id.strip(), state.strip().split()[0] if state.strip() else ""
        if not job_id or not state:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        states[job_id] = state
    return states


def user_queue_count(runner: Callable[..., subprocess.CompletedProcess] | None = None) -> int:
    output = run_ssh("squeue -u ozu647717 -h -o '%i|%T'", runner)
    return len(parse_delimited(output))


def ledger_records(ledger_path: Path = LEDGER) -> list[dict]:
    if not ledger_path.exists():
        return []
    records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def settled_keys(ledger_path: Path = LEDGER) -> set[str]:
    return {record["key"] for record in ledger_records(ledger_path)}


def own_job_ids(
    ledger_path: Path = LEDGER, run_root: Path = RUN_ROOT
) -> tuple[list[str], int]:
    """Authoritative delivered IDs for this lane plus conservative uncertain count."""
    ids: set[str] = set()
    uncertain = 0
    for record in ledger_records(ledger_path):
        for value in (record.get("job_ids") or {}).values():
            if value:
                ids.add(str(value))
        if record.get("status") in {"uncertain", "failed"}:
            uncertain += 1
    for sidecar in sorted(run_root.glob("*/*/*/fold_*/jobs.jsonl")):
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            job_id = event.get("slurm_job_id")
            if job_id:
                ids.add(str(job_id))
    return sorted(ids), uncertain


def _chunks(items: list[str], size: int = 200) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def query_job_states(
    job_ids: list[str], runner: Callable[..., subprocess.CompletedProcess] | None = None
) -> dict[str, str]:
    states: dict[str, str] = {}
    for chunk in _chunks(job_ids):
        output = run_ssh(f"squeue -j {','.join(chunk)} -h -o '%i|%T'", runner)
        states.update(parse_delimited(output))
    missing = [job_id for job_id in job_ids if job_id not in states]
    for chunk in _chunks(missing):
        output = run_ssh(
            f"sacct -j {','.join(chunk)} -n -P -o JobIDRaw,State",
            runner,
        )
        for line in output.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            job_id, state = line.split("|", 1)
            job_id = job_id.strip()
            if job_id in chunk and job_id not in states:
                states[job_id] = state.strip().split()[0] if state.strip() else "UNKNOWN"
    return states


def own_nonterminal_count(
    job_ids: list[str],
    states: dict[str, str],
    uncertain_records: int = 0,
) -> int:
    count = 0
    for job_id in job_ids:
        state = states.get(job_id, "UNKNOWN")
        if state not in TERMINAL_STATES:
            count += 1
    count += JOBS_PER_FIT * uncertain_records
    return count


def admission(
    own_nonterminal: int,
    user_queue: int,
    max_fits: int,
) -> int:
    """Return the number of fits this call may submit; refuse otherwise."""
    if max_fits < 1:
        raise AdmissionError("max_fits must be at least 1")
    if max_fits > WAVE_HARD_CAP_FITS or max_fits * JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"planned wave of {max_fits} fits exceeds the lane allocation "
            f"({LANE_CAP_NONTERMINAL} nonterminal jobs)"
        )
    if user_queue >= USER_QUEUE_STOP:
        raise AdmissionError(
            f"user queue {user_queue} is at or above the {USER_QUEUE_STOP} stop threshold"
        )
    headroom = (LANE_CAP_NONTERMINAL - own_nonterminal) // JOBS_PER_FIT
    if headroom <= 0:
        raise AdmissionError(
            f"no lane headroom: own nonterminal {own_nonterminal} of {LANE_CAP_NONTERMINAL}"
        )
    return min(max_fits, headroom)


def head_chain_headroom(own_nonterminal: int) -> int:
    """Head-chain capacity at the same 80-job lane budget.

    A head chain is two scheduler jobs per parent key (extract + classifier),
    so the same accounting used for training fits applies; return 0 when no
    full chain fits, and require the caller to refuse in that case.
    """
    if own_nonterminal + JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        return 0
    return (LANE_CAP_NONTERMINAL - own_nonterminal) // JOBS_PER_FIT


def submit_fit(
    fit: dict,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> dict:
    command = [
        sys.executable,
        "tools/exp.py",
        "submit",
        "feat-qwen3-legacy-prompt-20261008",
        "--config",
        fit["config"],
        "--fold",
        str(fit["fold"]),
        "--seed",
        str(fit["seed"]),
        "--run-name",
        fit["run_name"],
        "--campaign",
        CAMPAIGN,
        "--modality",
        fit["modality"],
        "--dataset",
        fit["dataset"],
        "--deployment-id",
        DEPLOYMENT,
        "--env-activate",
        ENVIRONMENTS[fit["modality"]],
        "--manifest-policy",
        "prebuilt",
        "--execute",
    ]
    runner = runner or subprocess.run
    result = runner(command, cwd=LANE, capture_output=True, text=True, timeout=1200)
    record = {
        "key": fit["key"],
        "run_name": fit["run_name"],
        "ts": int(time.time()),
        "status": "uncertain",
        "job_ids": {},
        "attempt_id": None,
    }
    if result.returncode == 0:
        attempt = re.search(r'"attempt_id": "([^"]+)"', result.stdout)
        jobs = re.search(r"submitted jobs: (\{[^}]*\})", result.stdout)
        if attempt and jobs:
            record["attempt_id"] = attempt.group(1)
            record["job_ids"] = ast.literal_eval(jobs.group(1))
            record["status"] = "submitted"
        else:
            record["tail"] = result.stdout[-600:]
    else:
        record["tail"] = (result.stdout + result.stderr)[-600:]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fits", type=int, default=WAVE_HARD_CAP_FITS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--matrix", default=str(MATRIX))
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    settled = settled_keys(Path(args.ledger))
    remaining = [fit for fit in matrix["fits"] if fit["key"] not in settled]

    if args.max_fits > WAVE_HARD_CAP_FITS or args.max_fits * JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        print(
            f"REFUSED: planned wave of {args.max_fits} fits exceeds the lane "
            f"allocation ({LANE_CAP_NONTERMINAL} nonterminal jobs)"
        )
        return 2

    submitted: list[dict] = []
    while len(submitted) < args.max_fits and remaining:
        try:
            job_ids, uncertain = own_job_ids(Path(args.ledger))
            states = query_job_states(job_ids)
            own = own_nonterminal_count(job_ids, states, uncertain)
            user = user_queue_count()
            allowed = admission(own, user, args.max_fits - len(submitted))
        except AdmissionError as error:
            print(f"REFUSED before fit {len(submitted) + 1}: {error}")
            break
        if len(submitted) >= allowed:
            print(f"REFUSED after {len(submitted)} fits: lane headroom reached")
            break
        fit = remaining.pop(0)
        if not args.execute:
            print(f"dry: {fit['key']} own={own} user={user} allowed={allowed}")
            submitted.append({"key": fit["key"], "status": "dry"})
            continue
        record = submit_fit(fit)
        with Path(args.ledger).open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record) + "\n")
        submitted.append(record)
        print(
            f"{fit['key']} {record['status']} {record['job_ids']} "
            f"(own={own}, user={user}, allowed={allowed})",
            flush=True,
        )
        if record["status"] != "submitted":
            print("stopping wave on non-submitted fit (fail-closed)")
            break
    print(f"wave complete: {len(submitted)} fits processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
