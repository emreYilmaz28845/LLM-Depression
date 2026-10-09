#!/usr/bin/env python3
"""Fail-closed progressive dispatcher for the window15 campaign (Worker 2).

Admission semantics (binding; adapted from the reviewed Worker-4 pattern
``tools/qwen3_legacy_prompt_dispatch.py`` at a94fa23 with the window15 lane
identity; the Worker-4 tool itself is never executed from this lane):

- The user-wide queue must be below 350, verified from a raw delimited
  ``squeue`` query with the SSH return code checked. A failed SSH command, a
  non-zero scheduler return code, or an unparseable line refuses submission;
  it is never treated as "no jobs".
- The lane's own nonterminal count is computed from every authoritative
  delivered job ID recorded for this lane: the append-only dispatch ledger
  (``outputs/qwen3_window15_20261008/submissions.jsonl``), every canonical
  execution-ledger job record whose deployment belongs to this lane, every
  ``outputs/exp_submit/*/submit_output.log`` delivery line, and every local
  fold sidecar ``jobs.jsonl`` event. This covers smokes, auxiliaries,
  downstream jobs, heads and historical attempts. Own IDs are resolved by
  fetching the full user queue once and then querying ``sacct`` for the
  remaining IDs; IDs absent from both count as nonterminal (fail closed).
- Partial or uncertain delivery is counted conservatively: dispatch-ledger
  ``uncertain``/``failed`` records reserve two jobs each, and any attempt that
  has a submit log but no complete job graph in the canonical execution
  ledger reserves two more jobs.
- Before every single leg (a production fit is two jobs: train + best_eval)
  both counts are re-checked. The condition is exactly ``user_queue < 350``
  and ``own_nonterminal + leg_jobs <= 80``. Waves are rejected outright when
  they exceed the 40-fit / 80-job lane allocation.
- Every submit outcome that is not a clean parsed success is recorded as
  ``uncertain`` in the append-only ledger with any partial stdout/stderr,
  attempt id and job ids preserved, and its reservation stays in the
  own-nonterminal count until reconciled. Nothing is retried automatically.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

LANE = Path(__file__).resolve().parents[1]
EVIDENCE = LANE / "outputs/qwen3_window15_20261008"
MATRIX = EVIDENCE / "matrix.json"
LEDGER = EVIDENCE / "submissions.jsonl"
CONTRACT = EVIDENCE / "contracts/treatment_contract.json"
RUN_ROOT = LANE / "output_model/qwen3_window15_20261008"
EXP_SUBMIT_ROOT = LANE / "outputs/exp_submit"
HEADS_REGISTRY = EVIDENCE / "head_submissions.jsonl"
LANE_SUBMISSION_LOCK = EVIDENCE / "submission.lock"
SLUG = "feat-qwen3-window15-20261008"
CAMPAIGN = "qwen3_window15_20261008"
DEPLOYMENT_PREFIX = "feat-qwen3-window15-20261008-"
ATTEMPT_MARKER = "q3w15_"
EXECUTION_LEDGER = Path(
    os.environ.get(
        "PARALLEL_WORKFLOW_STATE",
        "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/"
        "parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json",
    )
)
SCHEDULER = "ozu647717@alogin2.bsc.es"
USER_QUEUE_STOP = 350
LANE_CAP_NONTERMINAL = 80
WAVE_HARD_CAP_FITS = 40
JOBS_PER_FIT = 2
QWEN3OMNI_ENV = "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate"
ENVIRONMENTS = {
    "audio_only": QWEN3OMNI_ENV,
    "audio_text": QWEN3OMNI_ENV,
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

Runner = Callable[..., subprocess.CompletedProcess]


class AdmissionError(RuntimeError):
    """Raised when admission cannot be verified or the budget is exceeded."""


@contextlib.contextmanager
def lane_submission_lock(path: Path = LANE_SUBMISSION_LOCK, blocking: bool = False):
    """One exclusive lane submission lock shared by training and head entrypoints.

    Both entrypoints hold this lock across fresh plan rebuild, eligibility,
    admission and delivery so two processes can never submit for the lane at
    the same time.
    """
    handle = Path(path).open("w")
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError as exc:
            raise AdmissionError("another lane submitter holds the submission lock") from exc
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def run_ssh(command: str, runner: Runner | None = None) -> str:
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


def user_queue(runner: Runner | None = None) -> dict[str, str]:
    """The full user queue as job id -> state."""
    return parse_delimited(run_ssh("squeue -u ozu647717 -h -o '%i|%T'", runner))


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_bsc_quota_projects(output: str, *, expected_group: str = "etur92") -> dict[str, float]:
    """Parse the ``gpfs_projects`` GRP row of ``bsc_quota projects``.

    Fail-closed contract (same as the peer lane gates):

    - the section header must name exactly the expected group, exactly once;
    - exactly one ``gpfs_projects`` row of type ``GRP`` inside that section,
      with GB unit tokens on all four size fields (Usage, soft Quota, hard
      Limit, In doubt);
    - every value finite and non-negative; soft quota positive and hard limit
      at least the soft quota.

    Ambiguous output, a missing field, a unit mismatch, NaN/Inf or a negative
    value raises AdmissionError. There is deliberately no default for a missing
    in-doubt value. Shared-filesystem ``df`` output is never a substitute
    because it does not reflect the project quota.
    """
    import math

    cleaned = strip_ansi(output)
    sections = re.split(r"Printing quota for group\s+(\S+?):", cleaned)
    target_sections = [
        sections[index + 1]
        for index in range(1, len(sections) - 1, 2)
        if sections[index] == expected_group
    ]
    if not target_sections:
        raise AdmissionError(
            f"bsc_quota output does not declare group {expected_group!r}"
        )
    if len(target_sections) != 1:
        raise AdmissionError(
            f"bsc_quota output declares group {expected_group!r} {len(target_sections)} times"
        )
    rows = [
        line.split()
        for line in target_sections[0].splitlines()
        if line.split() and line.split()[0] == "gpfs_projects"
    ]
    if not rows:
        raise AdmissionError("bsc_quota output has no gpfs_projects row")
    if len(rows) != 1:
        raise AdmissionError(
            f"bsc_quota output has {len(rows)} gpfs_projects rows; expected exactly one"
        )
    fields = rows[0]
    if len(fields) < 10:
        raise AdmissionError(f"bsc_quota gpfs_projects row too short: {' '.join(fields)!r}")
    if fields[1] != "GRP":
        raise AdmissionError(
            f"bsc_quota gpfs_projects type is {fields[1]!r}, expected 'GRP'"
        )
    values: dict[str, float] = {}
    for name, value_index in (("usage", 2), ("quota", 4), ("limit", 6), ("in_doubt", 8)):
        token = fields[value_index]
        unit = fields[value_index + 1]
        if unit != "GB":
            raise AdmissionError(f"bsc_quota {name} unit is {unit!r}, expected 'GB'")
        try:
            number = float(token)
        except ValueError as exc:
            raise AdmissionError(f"bsc_quota {name} value unparseable: {token!r}") from exc
        if not math.isfinite(number):
            raise AdmissionError(f"bsc_quota {name} value is not finite: {token!r}")
        if number < 0:
            raise AdmissionError(f"bsc_quota {name} value is negative: {token!r}")
        values[name] = number
    if values["quota"] <= 0:
        raise AdmissionError("bsc_quota soft quota must be positive")
    if values["limit"] < values["quota"]:
        raise AdmissionError("bsc_quota hard limit is below the soft quota")
    return {
        "usage_gb": values["usage"],
        "quota_gb": values["quota"],
        "limit_gb": values["limit"],
        "in_doubt_gb": values["in_doubt"],
    }


def storage_admission(
    runner: Runner | None = None,
    *,
    remaining_min_gb: float = 500.0,
    local_min_gb: float = 50.0,
    local_root: Path | None = None,
    local_available_bytes: int | None = None,
) -> dict:
    """Fail-closed storage admission for refills.

    Requires the etur92 ``gpfs_projects`` soft-quota remaining
    (``quota - usage - in_doubt``) to be at least 500 GB and the local
    filesystem holding the lane to have at least 50 GB available. Any SSH or
    parse failure refuses. Shared-filesystem occupancy (``df``) is deliberately
    not used for the project side.
    """
    import math

    for label, value in (("remaining_min_gb", remaining_min_gb), ("local_min_gb", local_min_gb)):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise AdmissionError(f"{label} must be finite and non-negative")
    raw = run_ssh("bsc_quota projects --unit GB --no-color", runner)
    parsed = parse_bsc_quota_projects(raw)
    remaining = parsed["quota_gb"] - parsed["usage_gb"] - parsed["in_doubt_gb"]
    if not math.isfinite(remaining):
        raise AdmissionError("bsc_quota remaining is not finite")
    if remaining < remaining_min_gb:
        raise AdmissionError(
            f"gpfs_projects remaining {remaining:.2f} GB is below the {remaining_min_gb:.0f} GB reserve"
        )
    if local_available_bytes is None:
        import shutil

        local_available_bytes = shutil.disk_usage(str(local_root or LANE)).free
    local_bytes = float(local_available_bytes)
    if not math.isfinite(local_bytes) or local_bytes < 0:
        raise AdmissionError("local available bytes must be finite and non-negative")
    local_available_gb = local_bytes / (1024 ** 3)
    if local_available_gb < local_min_gb:
        raise AdmissionError(
            f"local available {local_available_gb:.2f} GB is below the {local_min_gb:.0f} GB reserve"
        )
    return {
        "gpfs_projects_usage_gb": parsed["usage_gb"],
        "gpfs_projects_quota_gb": parsed["quota_gb"],
        "gpfs_projects_in_doubt_gb": parsed["in_doubt_gb"],
        "gpfs_projects_remaining_gb": round(remaining, 2),
        "remaining_min_gb": remaining_min_gb,
        "local_available_gb": round(local_available_gb, 2),
        "local_min_gb": local_min_gb,
        "admission": "admissible",
    }


def ledger_records(ledger_path: Path = LEDGER) -> list[dict]:
    """Read the lane dispatch ledger.

    A missing file is permitted only while no delivery exists yet (the driver
    creates it on the first record). Unreadable or malformed content fails
    closed instead of silently reporting zero ownership.
    """
    if not ledger_path.exists():
        return []
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AdmissionError(f"lane submission ledger unreadable: {ledger_path}: {exc}") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AdmissionError(
                f"lane submission ledger malformed at line {line_number}: {exc}"
            ) from exc
    return records


def settled_keys(ledger_path: Path = LEDGER) -> set[str]:
    return {record["key"] for record in ledger_records(ledger_path)}


def _extract_delivery(text: str) -> tuple[str | None, dict[str, str]]:
    attempt = re.search(r'"attempt_id": "([^"]+)"', text)
    jobs = re.search(r"submitted jobs: (\{[^}]*\})", text)
    attempt_id = attempt.group(1) if attempt else None
    job_ids: dict[str, str] = {}
    if jobs:
        try:
            parsed = ast.literal_eval(jobs.group(1))
            if isinstance(parsed, dict):
                job_ids = {str(key): str(value) for key, value in parsed.items()}
        except (ValueError, SyntaxError):
            job_ids = {}
    return attempt_id, job_ids


def _complete_job_graph(job_ids: dict[str, str]) -> bool:
    """True only for the exact expected graph: unique numeric train + best_eval.

    Anything else (single ID, duplicate-valued IDs, non-numeric or extra keys)
    is not a proven delivery and must stay uncertain.
    """
    if not isinstance(job_ids, dict) or set(job_ids) != {"train", "best_eval"}:
        return False
    train = str(job_ids.get("train") or "").strip()
    best = str(job_ids.get("best_eval") or "").strip()
    if not train.isdigit() or not best.isdigit():
        return False
    return train != best


def exec_ledger_lane_jobs(exec_ledger_path: Path = EXECUTION_LEDGER) -> list[dict]:
    """Canonical execution-ledger job records belonging to this lane.

    This is required authoritative data. A missing, unreadable, malformed or
    jobs-less execution ledger raises AdmissionError instead of silently
    reporting zero ownership (fail-open ownership loss).
    """
    if not exec_ledger_path.exists():
        raise AdmissionError(f"authoritative execution ledger missing: {exec_ledger_path}")
    try:
        data = json.loads(exec_ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(
            f"authoritative execution ledger unreadable/malformed: {exec_ledger_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or "jobs" not in data:
        raise AdmissionError(
            f"authoritative execution ledger has no jobs section: {exec_ledger_path}"
        )
    lane_jobs = []
    for job in data.get("jobs", []):
        deployment = str(job.get("deployment_id") or "")
        attempt = str(job.get("attempt_id") or "")
        if deployment.startswith(DEPLOYMENT_PREFIX) or ATTEMPT_MARKER in attempt:
            lane_jobs.append(job)
    return lane_jobs


def own_job_ids(
    ledger_path: Path = LEDGER,
    run_root: Path = RUN_ROOT,
    exec_ledger_path: Path = EXECUTION_LEDGER,
    exp_submit_root: Path = EXP_SUBMIT_ROOT,
    heads_registry_path: Path = HEADS_REGISTRY,
) -> tuple[list[str], int]:
    """Authoritative delivered IDs for this lane plus conservative uncertain count."""
    ids: set[str] = set()
    uncertain = 0
    lane_jobs = exec_ledger_lane_jobs(exec_ledger_path)
    exp_logs = (
        sorted(exp_submit_root.glob("*/submit_output.log")) if exp_submit_root.exists() else []
    )
    if not ledger_path.exists() and (lane_jobs or exp_logs):
        raise AdmissionError(
            f"lane submission ledger missing after deliveries exist: {ledger_path}; "
            "seed it from evidence (--seed-ledger)"
        )
    exec_attempt_graphs: dict[str, dict[str, str]] = {}
    for job in lane_jobs:
        attempt = str(job.get("attempt_id") or "")
        job_id = str(job.get("slurm_job_id") or "").strip()
        if attempt and job_id:
            exec_attempt_graphs.setdefault(attempt, {})[str(job.get("job_key") or "job")] = job_id
    # Last record per key wins: a finalized delivery supersedes its reservation.
    latest_by_key: dict[str, dict] = {}
    for record in ledger_records(ledger_path):
        key = str(record.get("key") or "")
        if key:
            latest_by_key[key] = record
        else:
            for value in (record.get("job_ids") or {}).values():
                if value:
                    ids.add(str(value))
            if record.get("status") in {"uncertain", "failed", "held"}:
                uncertain += 1
    for record in latest_by_key.values():
        for value in (record.get("job_ids") or {}).values():
            if value:
                ids.add(str(value))
        if record.get("status") in {"uncertain", "failed", "held"}:
            # The reservation clears only when the authoritative execution
            # ledger proves the attempt's complete distinct delivered-ID graph.
            attempt = str(record.get("attempt_id") or "")
            if not _complete_job_graph(exec_attempt_graphs.get(attempt, {})):
                uncertain += 1
    for job in lane_jobs:
        job_id = job.get("slurm_job_id")
        if job_id:
            ids.add(str(job_id))
    for log in exp_logs:
        _, job_ids = _extract_delivery(log.read_text(encoding="utf-8"))
        for value in job_ids.values():
            if value:
                ids.add(str(value))
    # Historic and current head-chain deliveries from the generic head registry.
    for entry in head_registry_entries(heads_registry_path):
        for value in (entry.get("extract_job_id"), entry.get("classifier_job_id")):
            if value:
                ids.add(str(value))
        for value in (entry.get("job_ids") or {}).values():
            if value:
                ids.add(str(value))
    if run_root.exists():
        for sidecar in sorted(run_root.glob("*/*/*/fold_*/jobs.jsonl")):
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                job_id = event.get("slurm_job_id")
                if job_id:
                    ids.add(str(job_id))
    return sorted(ids), uncertain


def head_registry_entries(registry_path: Path = HEADS_REGISTRY) -> list[dict]:
    """Read the generic head registry; malformed content fails closed."""
    if not registry_path.exists():
        return []
    entries = []
    for line_number, line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AdmissionError(
                f"head registry malformed at line {line_number}: {exc}"
            ) from exc
    return entries


def unresolved_delivery_reservations(
    ledger_path: Path = LEDGER,
    exec_ledger_path: Path = EXECUTION_LEDGER,
    exp_submit_root: Path = EXP_SUBMIT_ROOT,
) -> int:
    """Attempts with a submit log but no complete distinct job graph in the lane ledger.

    ``exp.py submit`` writes ``submit_output.log`` only after it reaches the
    sbatch step and records the complete job graph in the canonical execution
    ledger. A logged attempt without a proven complete distinct-ID graph
    (missing job keys, missing IDs or duplicate IDs) therefore delivered a
    partial or unverified graph; reserve two jobs for it.
    """
    settled_attempts = {
        str(record.get("attempt_id"))
        for record in ledger_records(ledger_path)
        if record.get("status") == "submitted"
        and record.get("attempt_id")
        and _complete_job_graph(record.get("job_ids") or {})
    }
    exec_graphs: dict[str, dict[str, str]] = {}
    for job in exec_ledger_lane_jobs(exec_ledger_path):
        attempt = str(job.get("attempt_id") or "")
        job_id = str(job.get("slurm_job_id") or "").strip()
        if attempt and job_id:
            exec_graphs.setdefault(attempt, {})[str(job.get("job_key") or "job")] = job_id
    reservations = 0
    if exp_submit_root.exists():
        for log in sorted(exp_submit_root.glob("*/submit_output.log")):
            attempt_id, _ = _extract_delivery(log.read_text(encoding="utf-8"))
            if not attempt_id:
                continue
            if _complete_job_graph(exec_graphs.get(attempt_id, {})):
                continue
            if attempt_id in settled_attempts:
                continue
            reservations += 1
    return reservations


def _chunks(items: list[str], size: int = 200) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def append_record(entry: dict, ledger_path: Path = LEDGER) -> None:
    """Append one delivery record to the lane ledger (the only write path)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry) + "\n")


def seed_ledger_from_evidence(
    ledger_path: Path = LEDGER,
    exec_ledger_path: Path = EXECUTION_LEDGER,
    exp_submit_root: Path = EXP_SUBMIT_ROOT,
) -> int:
    """Create the lane ledger from actual execution-ledger/exp_submit evidence.

    Used once before the first driver-run submission when deliveries already
    exist from earlier ``exp.py`` invocations. Records are derived from the
    authoritative sources, never typed by hand. An attempt whose graph is
    incomplete (no train + best_eval pair) is recorded as uncertain so it
    keeps its conservative reservation.
    """
    if ledger_path.exists() and ledger_records(ledger_path):
        raise AdmissionError(f"lane ledger already has records: {ledger_path}")
    lane_jobs = exec_ledger_lane_jobs(exec_ledger_path)
    by_attempt: dict[str, dict[str, str]] = {}
    for job in lane_jobs:
        attempt = str(job.get("attempt_id") or "")
        job_id = job.get("slurm_job_id")
        if attempt and job_id:
            by_attempt.setdefault(attempt, {})[str(job.get("job_key") or "job")] = str(job_id)
    for log in sorted(exp_submit_root.glob("*/submit_output.log")):
        attempt_id, job_ids = _extract_delivery(log.read_text(encoding="utf-8"))
        if attempt_id and job_ids:
            by_attempt.setdefault(attempt_id, {})
            for key, value in job_ids.items():
                by_attempt[attempt_id].setdefault(key, value)
    written = 0
    for attempt_id, job_ids in sorted(by_attempt.items()):
        complete = _complete_job_graph(job_ids)
        append_record(
            {
                "key": f"seed:{attempt_id}",
                "run_name": "",
                "ts": int(time.time()),
                "status": "submitted" if complete else "uncertain",
                "reason": "" if complete else "seeded from evidence; incomplete job graph",
                "attempt_id": attempt_id,
                "job_ids": job_ids,
                "source": "seed_from_execution_ledger",
            },
            ledger_path,
        )
        written += 1
    return written


def query_job_states(
    job_ids: list[str], runner: Runner | None = None
) -> tuple[dict[str, str], int]:
    """Resolve own job states from the full user queue, then sacct for the rest.

    Returns the state map and the user-wide queue count from the same queue
    fetch. IDs present in neither source are left unresolved and later count as
    nonterminal. ``squeue -j`` is deliberately not used: IDs that already left
    the queue can make that query fail.
    """
    queue = user_queue(runner)
    states = {job_id: queue[job_id] for job_id in job_ids if job_id in queue}
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
    return states, len(queue)


def own_nonterminal_count(
    job_ids: list[str],
    states: dict[str, str],
    uncertain_reservations: int = 0,
) -> int:
    count = 0
    for job_id in job_ids:
        state = states.get(job_id, "UNKNOWN")
        if state not in TERMINAL_STATES:
            count += 1
    count += JOBS_PER_FIT * uncertain_reservations
    return count


def per_fit_admission(own_nonterminal: int, user_queue_size: int, leg_jobs: int = JOBS_PER_FIT) -> None:
    """The exact per-leg condition; anything else refuses."""
    if user_queue_size >= USER_QUEUE_STOP:
        raise AdmissionError(
            f"user queue {user_queue_size} is at or above the {USER_QUEUE_STOP} stop threshold"
        )
    if own_nonterminal + int(leg_jobs) > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"no lane headroom: own nonterminal {own_nonterminal} + {leg_jobs} "
            f"exceeds {LANE_CAP_NONTERMINAL}"
        )


def validate_wave_size(fits: list[dict], max_fits: int) -> None:
    if max_fits < 1:
        raise AdmissionError("max_fits must be at least 1")
    if max_fits > WAVE_HARD_CAP_FITS:
        raise AdmissionError(
            f"planned wave of {max_fits} fits exceeds the {WAVE_HARD_CAP_FITS}-fit lane allocation"
        )
    wave_jobs = sum(int(fit.get("leg_jobs", JOBS_PER_FIT)) for fit in fits[:max_fits])
    if wave_jobs > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"planned wave of {wave_jobs} jobs exceeds the {LANE_CAP_NONTERMINAL}-job lane allocation"
        )


def lane_headroom(own_nonterminal: int) -> int:
    """Remaining two-job fits under the 80-job lane budget."""
    if own_nonterminal + JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        return 0
    return (LANE_CAP_NONTERMINAL - own_nonterminal) // JOBS_PER_FIT


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def uncertain_record(fit: dict, reason: str, stdout: str = "", stderr: str = "") -> dict:
    attempt_id, job_ids = _extract_delivery(stdout)
    return {
        "key": fit["key"],
        "run_name": fit["run_name"],
        "ts": int(time.time()),
        "status": "uncertain",
        "reason": reason,
        "attempt_id": attempt_id,
        "job_ids": job_ids,
        "tail": (stdout + "\n" + stderr)[-800:],
    }


def validate_fit(fit: dict, deploy_root: Path | None = None) -> None:
    """Fail closed before sbatch on missing identity or activation."""
    deploy_root = deploy_root if deploy_root is not None else LANE / "outputs/exp_deploy"
    deployment_id = str(fit.get("deployment_id") or "")
    if not deployment_id:
        raise AdmissionError(f"fit {fit.get('key')} has no deployment_id")
    record = deploy_root / deployment_id / "deployment.json"
    if not record.is_file():
        raise AdmissionError(
            f"fit {fit.get('key')} deployment record missing locally: {record}"
        )
    modality = str(fit.get("modality") or "")
    env = str(fit.get("env_activate") or ENVIRONMENTS.get(modality, ""))
    if not env:
        raise AdmissionError(f"fit {fit.get('key')} has no activation environment")
    if modality in ENVIRONMENTS and env != ENVIRONMENTS[modality]:
        raise AdmissionError(
            f"fit {fit.get('key')} activation {env!r} does not match the verified "
            f"environment for {modality}: {ENVIRONMENTS[modality]!r}"
        )
    if not str(fit.get("run_name") or ""):
        raise AdmissionError(f"fit {fit.get('key')} has no run_name")


def submit_fit(fit: dict, runner: Runner | None = None) -> dict:
    modality = str(fit["modality"])
    env = str(fit.get("env_activate") or ENVIRONMENTS.get(modality, ""))
    command = [
        sys.executable,
        "tools/exp.py",
        "submit",
        SLUG,
        "--config",
        fit["config"],
        "--fold",
        str(fit["fold"]),
        "--run-name",
        fit["run_name"],
        "--campaign",
        CAMPAIGN,
        "--modality",
        modality,
        "--dataset",
        fit["dataset"],
        "--deployment-id",
        str(fit["deployment_id"]),
        "--env-activate",
        env,
        "--manifest-policy",
        str(fit.get("manifest_policy", "build")),
    ]
    if fit.get("seed") is not None:
        command += ["--seed", str(fit["seed"])]
    if fit.get("supersedes_attempt_id"):
        command += ["--supersedes-attempt-id", str(fit["supersedes_attempt_id"])]
    for override in fit.get("overrides") or []:
        command += ["--set", str(override)]
    command += ["--execute"]
    runner = runner or subprocess.run
    try:
        result = runner(command, cwd=LANE, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        return uncertain_record(
            fit,
            "timeout",
            _text(getattr(exc, "stdout", "")),
            _text(getattr(exc, "stderr", "")),
        )
    except OSError as exc:
        return uncertain_record(fit, f"oserror: {exc}")
    except Exception as exc:  # fail closed on any unexpected runner failure
        return uncertain_record(fit, f"runner error: {type(exc).__name__}: {exc}")

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode == 0:
        attempt_id, job_ids = _extract_delivery(stdout)
        if attempt_id and _complete_job_graph(job_ids):
            return {
                "key": fit["key"],
                "run_name": fit["run_name"],
                "ts": int(time.time()),
                "status": "submitted",
                "job_ids": job_ids,
                "attempt_id": attempt_id,
            }
        return uncertain_record(
            fit,
            "incomplete or malformed job graph (expected unique numeric train+best_eval)",
            stdout,
            stderr,
        )
    # A non-zero return code may still have delivered remote jobs; preserve
    # everything the output shows and count it conservatively.
    return uncertain_record(fit, f"submit rc={result.returncode}", stdout, stderr)


def run_wave(
    fits: list[dict],
    max_fits: int,
    *,
    reconcile: Callable[[], tuple[int, int]],
    submit: Callable[[dict], dict],
    on_record: Callable[[dict], None] | None = None,
    storage_check: Callable[[], dict] | None = None,
    storage_every: int = 5,
) -> int:
    """Submit up to ``max_fits`` fits; per-fit admission, no capacity double-count.

    When ``storage_check`` is given it is re-run every ``storage_every`` fits
    (and the caller runs it once before the wave) so a bounded batch cannot
    outrun the projects soft-quota reserve.
    """
    processed = 0
    for fit in fits:
        if processed >= max_fits:
            break
        if storage_check is not None and processed > 0 and processed % max(1, int(storage_every)) == 0:
            try:
                storage_check()
            except AdmissionError as error:
                print(f"REFUSED before fit {processed + 1} (storage): {error}")
                break
        own, user = reconcile()
        try:
            per_fit_admission(own, user, int(fit.get("leg_jobs", JOBS_PER_FIT)))
        except AdmissionError as error:
            print(f"REFUSED before fit {processed + 1}: {error}")
            break
        record = submit(fit)
        processed += 1
        if on_record is not None:
            on_record(record)
        if record.get("status") != "submitted":
            print("stopping wave on non-submitted fit (fail-closed)")
            break
        print(
            f"{fit['key']} {record['status']} {record['job_ids']} (own={own}, user={user})",
            flush=True,
        )
    return processed


def reconcile_evidence(
    ledger_path: Path = LEDGER,
    runner: Runner | None = None,
    exec_ledger_path: Path = EXECUTION_LEDGER,
    exp_submit_root: Path = EXP_SUBMIT_ROOT,
    run_root: Path = RUN_ROOT,
) -> dict:
    job_ids, uncertain_records = own_job_ids(
        ledger_path, run_root, exec_ledger_path, exp_submit_root
    )
    partial_reservations = unresolved_delivery_reservations(
        ledger_path, exec_ledger_path, exp_submit_root
    )
    states, user = query_job_states(job_ids, runner)
    uncertain_total = int(uncertain_records) + int(partial_reservations)
    own_nonterminal = own_nonterminal_count(job_ids, states, uncertain_total)
    unresolved = sorted(job_id for job_id in job_ids if job_id not in states)
    return {
        "own_job_ids": job_ids,
        "own_job_count": len(job_ids),
        "own_states": states,
        "own_unresolved_ids": unresolved,
        "uncertain_records": int(uncertain_records),
        "partial_delivery_reservations": int(partial_reservations),
        "own_nonterminal": own_nonterminal,
        "user_queue": int(user),
        "lane_cap_nonterminal": LANE_CAP_NONTERMINAL,
        "user_queue_stop": USER_QUEUE_STOP,
        "lane_headroom_fits": lane_headroom(own_nonterminal),
    }


def _modality_from_config(config: dict) -> str:
    data = config.get("data") or {}
    use_audio = bool(data.get("use_audio", False))
    use_text = bool(data.get("use_text", False))
    if use_audio and use_text:
        return "audio_text"
    if use_audio:
        return "audio_only"
    return "text_only"


def build_matrix(
    contract_path: Path = CONTRACT,
    out_path: Path = MATRIX,
) -> dict:
    import yaml

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fits: list[dict] = []
    for row in contract["rows"]:
        config_path = row["treatment_config"]
        payload = yaml.safe_load((LANE / config_path).read_text(encoding="utf-8")) or {}
        dataset = str(payload["dataset"])
        data_cfg = payload.get("data") or {}
        pooled = str(payload.get("dataset_variant", "")).strip() == "pooled_t17"
        fits.append(
            {
                "key": str(row["registry_key"]),
                "route_id": str(row["route_id"]),
                "run_name": str(row["planned_run_name"]),
                "config": config_path,
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "modality": _modality_from_config(payload),
                "dataset": dataset,
                "env_activate": QWEN3OMNI_ENV,
                "manifest_policy": "prebuilt" if pooled else "build",
                "leg_jobs": JOBS_PER_FIT,
                "overrides": [],
                "supersedes_attempt_id": None,
                "control_attempt_id": str(row.get("control_attempt_id", "")),
            }
        )
    fits.sort(key=lambda fit: (fit["route_id"], fit["seed"], fit["fold"]))
    payload = {
        "schema_version": "audiollm.qwen3_window15_matrix.v1",
        "campaign": CAMPAIGN,
        "expected_fits": len(fits),
        "fits": fits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--max-fits", type=int, default=WAVE_HARD_CAP_FITS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evidence", action="store_true", help="print admission evidence and exit")
    parser.add_argument("--only", action="append", default=[], help="restrict to fit keys (repeatable)")
    parser.add_argument("--deployment-id", default=None, help="override deployment for every fit")
    parser.add_argument("--build-matrix", action="store_true", help="regenerate the production matrix")
    parser.add_argument(
        "--seed-ledger",
        action="store_true",
        help="create the lane ledger from actual execution-ledger/exp_submit evidence",
    )
    args = parser.parse_args()

    if args.seed_ledger:
        written = seed_ledger_from_evidence(Path(args.ledger))
        print(json.dumps({"status": "ok", "seeded_records": written, "ledger": args.ledger}))
        return 0

    if args.build_matrix:
        payload = build_matrix(out_path=Path(args.matrix))
        print(json.dumps({"status": "ok", "fits": len(payload["fits"]), "matrix": args.matrix}))
        return 0

    ledger_path = Path(args.ledger)
    matrix_path = Path(args.matrix)
    if not matrix_path.is_file():
        print(f"REFUSED: matrix missing: {matrix_path}")
        return 2
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    remaining = [fit for fit in matrix["fits"] if fit["key"] not in settled_keys(ledger_path)]
    if args.only:
        wanted = set(args.only)
        remaining = [fit for fit in remaining if fit["key"] in wanted]
    if args.deployment_id:
        for fit in remaining:
            fit["deployment_id"] = args.deployment_id

    try:
        validate_wave_size(remaining, args.max_fits)
    except AdmissionError as error:
        print(f"REFUSED: {error}")
        return 2

    def reconcile() -> tuple[int, int]:
        evidence = reconcile_evidence(ledger_path)
        return int(evidence["own_nonterminal"]), int(evidence["user_queue"])

    if not args.execute or args.evidence:
        evidence = reconcile_evidence(ledger_path)
        evidence["remaining_fits"] = len(remaining)
        evidence["max_fits"] = args.max_fits
        try:
            per_fit_admission(
                int(evidence["own_nonterminal"]),
                int(evidence["user_queue"]),
                int(remaining[0].get("leg_jobs", JOBS_PER_FIT)) if remaining else JOBS_PER_FIT,
            )
            evidence["admission"] = "admissible"
        except AdmissionError as error:
            evidence["admission"] = f"REFUSED: {error}"
        print(json.dumps(evidence, indent=1, sort_keys=True))
        return 0 if evidence["admission"] == "admissible" else 2

    if not remaining:
        print("no unsubmitted fits remain")
        return 0

    try:
        with lane_submission_lock():
            for fit in remaining:
                validate_fit(fit)

            storage_evidence = storage_admission()
            print("storage admission: " + json.dumps(storage_evidence, sort_keys=True))

            def record(entry: dict) -> None:
                append_record(entry, ledger_path)

            processed = run_wave(
                remaining,
                args.max_fits,
                reconcile=reconcile,
                submit=submit_fit,
                on_record=record,
                storage_check=storage_admission,
                storage_every=5,
            )
    except AdmissionError as error:
        print(f"REFUSED: {error}")
        return 2
    print(f"wave complete: {processed} fits processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
