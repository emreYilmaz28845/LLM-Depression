#!/usr/bin/env python3
"""Fail-closed progressive dispatcher for the Worker 4 legacy-prompt campaign.

Admission semantics (binding):

- The user-wide queue must be below 350, verified from a raw delimited
  ``squeue`` query with the SSH return code checked. A failed SSH command, a
  non-zero scheduler return code, or an unparseable line refuses submission;
  it is never treated as "no jobs".
- The lane's own nonterminal count keeps ownership of every unique numeric
  job id recorded for this lane (all ledger records plus every local
  fold-sidecar ``jobs.jsonl`` SUBMITTED event), and each uncertain/failed
  record adds its full two-job reservation; a safe overcount is accepted.
  Own IDs are resolved by fetching the full user queue once (``squeue -u``
  filtered to own IDs, so IDs that already left the queue cannot break the
  query) and then querying ``sacct`` for the remaining IDs. Non-numeric
  preserved ids, or ids that neither source can resolve, stop admission until
  manually reconciled (fail closed); the wave is never allowed to continue on
  an unknown own job.
- Before every single fit (two jobs: train + best_eval) both counts are
  re-checked. The per-fit condition is exactly ``user_queue < 350`` and
  ``own_nonterminal + 2 <= 80``; the loop never compares the processed count
  against a remaining-capacity number. Planned waves larger than the 80-job
  lane allocation (or larger than 40 fits) are rejected outright.
- Every submit outcome that is not a clean parsed success is recorded as
  ``uncertain`` in the append-only ledger with any partial stdout/stderr,
  attempt id and job ids preserved, and its two-job reservation stays in the
  own-nonterminal count until reconciled. A parsed success additionally
  requires exactly the expected keys ``train`` and ``best_eval`` carrying
  distinct numeric job ids plus a non-empty attempt id; a single id, duplicate
  values, malformed ids, extra or missing keys, and a missing attempt id all
  stay uncertain and stop the wave. Nothing is retried automatically.
- The append-only submission ledger is authoritative. A missing ledger refuses
  immediately; it is never treated as "no deliveries" and an empty ledger is
  never silently recreated. A genuine first-ever bootstrap must create an empty
  ledger deliberately and record that decision.
- The same accounting gates head-chain dispatch: reconcile own nonterminal jobs
  and refuse while ``own + 2 > 80`` before each head attempt.
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


def user_queue_count(runner: Runner | None = None) -> int:
    return len(user_queue(runner))


def ledger_records(ledger_path: Path = LEDGER) -> list[dict]:
    """Read the append-only ledger; refuse when the authoritative file is missing.

    A missing ledger is never treated as "no deliveries": the delivered history
    is authoritative for this lane and the accounting must not silently fall
    back to local sidecars only. A genuine first-ever bootstrap must create an
    empty ledger deliberately (``Path.touch``) and record that decision.
    """
    if not ledger_path.exists():
        raise AdmissionError(f"authoritative submission ledger is missing: {ledger_path}")
    records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def settled_keys(ledger_path: Path = LEDGER) -> set[str]:
    return {record["key"] for record in ledger_records(ledger_path)}


def own_job_ids(
    ledger_path: Path = LEDGER, run_root: Path = RUN_ROOT
) -> tuple[list[str], int, list[str]]:
    """Known own job IDs, uncertain-record count, and unknown preserved values.

    Every unique numeric job id from every ledger record (submitted, uncertain
    and failed) and from every local fold-sidecar ``jobs.jsonl`` is preserved in
    the accounting; keeping ownership of a known job always takes priority over
    avoiding a possible overcount. Each uncertain/failed record additionally
    contributes its full two-job reservation, so a partial delivery can
    overcount but never disappears. Non-numeric or blank preserved values are
    returned separately and must stop admission until reconciled.
    """
    ids: set[str] = set()
    unknown: list[str] = []
    uncertain = 0
    for record in ledger_records(ledger_path):
        status = record.get("status")
        if status not in {"submitted", "uncertain", "failed"}:
            continue
        for value in (record.get("job_ids") or {}).values():
            text = str(value).strip()
            if not text:
                continue
            if text.isdigit():
                ids.add(text)
            else:
                unknown.append(text)
        if status in {"uncertain", "failed"}:
            uncertain += 1
    for sidecar in sorted(run_root.glob("*/*/*/fold_*/jobs.jsonl")):
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            job_id = event.get("slurm_job_id")
            if job_id is None or str(job_id).strip() == "":
                continue
            text = str(job_id).strip()
            if text.isdigit():
                ids.add(text)
            else:
                unknown.append(text)
    return sorted(ids), uncertain, sorted(set(unknown))


def reconciliation_failures(
    job_ids: list[str], states: dict[str, str], unknown_ids: list[str]
) -> list[str]:
    failures: list[str] = []
    if unknown_ids:
        failures.append(f"non-numeric preserved ids: {unknown_ids[:10]}")
    unresolved = [job_id for job_id in job_ids if job_id not in states]
    if unresolved:
        failures.append(f"unresolved ids: {unresolved[:10]}")
    return failures


def require_reconciled(
    job_ids: list[str], states: dict[str, str], unknown_ids: list[str]
) -> None:
    """Admission must stop while any own id is unknown or unresolved."""
    failures = reconciliation_failures(job_ids, states, unknown_ids)
    if failures:
        raise AdmissionError("reconciliation required: " + "; ".join(failures))


def _chunks(items: list[str], size: int = 200) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


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
    uncertain_records: int = 0,
) -> int:
    count = 0
    for job_id in job_ids:
        state = states.get(job_id, "UNKNOWN")
        if state not in TERMINAL_STATES:
            count += 1
    count += JOBS_PER_FIT * uncertain_records
    return count


def per_fit_admission(own_nonterminal: int, user_queue_size: int) -> None:
    """The exact per-fit condition; anything else refuses."""
    if user_queue_size >= USER_QUEUE_STOP:
        raise AdmissionError(
            f"user queue {user_queue_size} is at or above the {USER_QUEUE_STOP} stop threshold"
        )
    if own_nonterminal + JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"no lane headroom: own nonterminal {own_nonterminal} of {LANE_CAP_NONTERMINAL}"
        )


def validate_wave_size(max_fits: int) -> None:
    if max_fits < 1:
        raise AdmissionError("max_fits must be at least 1")
    if max_fits > WAVE_HARD_CAP_FITS or max_fits * JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        raise AdmissionError(
            f"planned wave of {max_fits} fits exceeds the lane allocation "
            f"({LANE_CAP_NONTERMINAL} nonterminal jobs)"
        )


def head_chain_headroom(own_nonterminal: int) -> int:
    """Head-chain capacity at the same 80-job lane budget (2 jobs per chain)."""
    if own_nonterminal + JOBS_PER_FIT > LANE_CAP_NONTERMINAL:
        return 0
    return (LANE_CAP_NONTERMINAL - own_nonterminal) // JOBS_PER_FIT


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


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _valid_delivery(job_ids: dict[str, str]) -> bool:
    """Exactly the expected unique numeric train + best_eval pair."""
    if set(job_ids) != {"train", "best_eval"}:
        return False
    train = str(job_ids["train"]).strip()
    best = str(job_ids["best_eval"]).strip()
    if not train.isdigit() or not best.isdigit():
        return False
    return train != best


def uncertain_record(
    fit: dict,
    reason: str,
    stdout: str = "",
    stderr: str = "",
) -> dict:
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


def submit_fit(fit: dict, runner: Runner | None = None) -> dict:
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
    try:
        result = runner(command, cwd=LANE, capture_output=True, text=True, timeout=1200)
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
        if attempt_id and _valid_delivery(job_ids):
            return {
                "key": fit["key"],
                "run_name": fit["run_name"],
                "ts": int(time.time()),
                "status": "submitted",
                "job_ids": job_ids,
                "attempt_id": attempt_id,
            }
        reason = "unparsed success output"
        if attempt_id and not _valid_delivery(job_ids):
            reason = f"invalid delivered job ids: {job_ids!r}"
        return uncertain_record(fit, reason, stdout, stderr)
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
) -> int:
    """Submit up to ``max_fits`` fits; per-fit admission, no capacity double-count."""
    processed = 0
    for fit in fits:
        if processed >= max_fits:
            break
        try:
            own, user = reconcile()
            per_fit_admission(own, user)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fits", type=int, default=WAVE_HARD_CAP_FITS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--matrix", default=str(MATRIX))
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    try:
        validate_wave_size(args.max_fits)
    except AdmissionError as error:
        print(f"REFUSED: {error}")
        return 2

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    settled = settled_keys(ledger_path)
    remaining = [fit for fit in matrix["fits"] if fit["key"] not in settled]
    if not remaining:
        print("no unsubmitted fits remain")
        return 0

    def reconcile() -> tuple[int, int]:
        job_ids, uncertain, unknown = own_job_ids(ledger_path)
        states, user = query_job_states(job_ids)
        require_reconciled(job_ids, states, unknown)
        return own_nonterminal_count(job_ids, states, uncertain), user

    def record(entry: dict) -> None:
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry) + "\n")

    if not args.execute:
        print(f"dry: {len(remaining)} unsubmitted fits, wave cap {args.max_fits}")
        return 0

    processed = run_wave(
        remaining,
        args.max_fits,
        reconcile=reconcile,
        submit=submit_fit,
        on_record=record,
    )
    print(f"wave complete: {processed} fits processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
