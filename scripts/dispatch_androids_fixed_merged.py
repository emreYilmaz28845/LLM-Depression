#!/usr/bin/env python3
"""Fail-closed progressive dispatcher for the Worker 1 corrected merged baseline.

Admission semantics (mirrors the verified cross-lane dispatcher pattern):

- The user-wide queue must be below 350, verified from a raw delimited
  ``squeue`` query with the SSH return code checked. A failed SSH command, a
  non-zero scheduler return code or an unparseable line refuses submission; it
  is never treated as "no jobs".
- The lane's own nonterminal count is computed from every authoritative
  delivered job ID recorded for this lane: all run registries under
  ``<runtime>/registries`` plus archived registries, with unresolved IDs counted
  as nonterminal (fail closed). Uncertain submission records in the append-only
  submissions ledger reserve their leg size until their IDs resolve.
- Before every leg (one route/seed chain: 15 jobs for cv, 3 for smoke/final)
  both counts are re-checked: ``user_queue < 350`` and
  ``own_nonterminal + leg_size <= 80``. The loop never relies on an initial
  wave-size count; it reconciles before each leg and stops on the first refusal.
- Every submit outcome that is not a clean parsed success is recorded as
  ``uncertain`` in the append-only ledger with partial output preserved, and its
  reservation stays in the own-nonterminal count until reconciled. Nothing is
  retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
LEG_JOBS = {"cv": 15, "smoke": 3, "final": 3}
ROUTES = {
    "native_text_only": ("native", "text_only"),
    "native_audio_only": ("native", "audio_only"),
    "native_audio_text": ("native", "audio_text"),
    "english_text_only": ("english", "text_only"),
    "english_audio_text": ("english", "audio_text"),
}
CAMPAIGN_BASE = "qwen3_androids_official_folds_20261008"

Runner = Callable[..., subprocess.CompletedProcess]


class AdmissionError(RuntimeError):
    """Raised when admission cannot be verified or the budget is exceeded."""


def run_ssh(command: str, scheduler: str, runner: Runner | None = None) -> str:
    runner = runner or subprocess.run
    result = runner(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", scheduler, command],
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
    states: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        job_id, state = line.split("|", 1)
        job_id = job_id.strip()
        state = state.strip().split()[0] if state.strip() else ""
        if not job_id or not state:
            raise AdmissionError(f"unparseable scheduler line: {line[:80]!r}")
        states[job_id] = state
    return states


def user_queue(scheduler: str, runner: Runner | None = None) -> dict[str, str]:
    return parse_delimited(
        run_ssh("squeue -u ozu647717 -h -o '%i|%T'", scheduler, runner)
    )


def registry_job_ids(registries_dir: Path) -> set[str]:
    """Every authoritative delivered job ID recorded for this lane.

    Unreadable, malformed or non-mapping registries refuse admission: dropping a
    registry would silently lose own delivered IDs.  Planning (dry-run)
    registries only carry synthetic ``dry_*`` identifiers and are ignored.
    """

    ids: set[str] = set()
    if not registries_dir.is_dir():
        raise AdmissionError(f"registries directory missing: {registries_dir}")
    for path in sorted(registries_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AdmissionError(f"unreadable registry {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise AdmissionError(f"malformed registry (not a mapping): {path}")
        planning = str(payload.get("submission_mode") or "") in {"dry_run", "planned"}
        for job in payload.get("jobs") or []:
            job_id = str(job.get("job_id") or "")
            if job_id.isdigit():
                ids.add(job_id)
            elif job_id and not planning:
                raise AdmissionError(f"non-numeric delivered job id {job_id!r} in {path}")
    return ids


def ledger_records(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def ledger_reservations(ledger_path: Path) -> tuple[set[str], int]:
    """Delivered IDs from the ledger plus conservative reservations.

    An uncertain/failed record reserves the unresolved remainder of its leg: the
    parsed job IDs are counted through normal state resolution, and everything
    the record cannot prove delivered stays reserved.
    """

    ids: set[str] = set()
    reservation = 0
    for record in ledger_records(ledger_path):
        parsed = [
            str(value) for value in (record.get("job_ids") or []) if str(value).isdigit()
        ]
        ids.update(parsed)
        if record.get("status") in {"uncertain", "failed"}:
            expected = int(record.get("reservation") or 0)
            reservation += max(0, expected - len(parsed))
    return ids, reservation


def _chunks(items: list[str], size: int = 200) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def query_job_states(
    job_ids: list[str], scheduler: str, runner: Runner | None = None
) -> tuple[dict[str, str], int]:
    queue = user_queue(scheduler, runner)
    states = {job_id: queue[job_id] for job_id in job_ids if job_id in queue}
    missing = [job_id for job_id in job_ids if job_id not in states]
    for chunk in _chunks(missing):
        output = run_ssh(
            f"sacct -j {','.join(chunk)} -n -P -o JobIDRaw,State", scheduler, runner
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
    job_ids: Iterable[str], states: dict[str, str], reservation: int = 0
) -> int:
    count = sum(1 for job_id in job_ids if states.get(job_id, "UNKNOWN") not in TERMINAL_STATES)
    return count + reservation


def leg_admission(own_nonterminal: int, user_total: int, leg_size: int) -> None:
    if user_total >= 350:
        raise AdmissionError(f"user queue {user_total} is at or above the 350 stop threshold")
    if own_nonterminal + leg_size > 80:
        raise AdmissionError(
            f"no lane headroom: own nonterminal {own_nonterminal} + leg {leg_size} of 80"
        )


def _extract_delivery(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    candidates = [text]
    for index in range(len(text) - 1, -1, -1):
        if text[index] == "{":
            candidates.append(text[index:])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("job_ids"):
            return payload
    return None


def submit_leg(
    *,
    stage: str,
    route: str,
    seed: int,
    deployment_code: str,
    source_commit: str,
    input_root: str,
    pooled_runtime_root: str,
    runtime: str,
    runner: Runner | None = None,
) -> dict:
    campaign_suffix, modality = ROUTES[route]
    run_id = f"qmsm_{route}_s{seed}"
    registry_path = f"{runtime}/registries/{run_id}.json"
    expected = LEG_JOBS.get(stage, 0)
    command = [
        sys.executable,
        "scripts/submit_symmetric_merged.py",
        "--stage",
        stage,
        "--config",
        f"configs/experiments/merged/symmetric_merged_qwen3_pooled_{route}.yaml",
        "--run-id",
        run_id,
        "--registry",
        registry_path,
        "--set",
        f"seed={seed}",
        "--set",
        "protocol_settings.split_seed=1337",
        "--set",
        "heads.fixed_seed=1337",
        "--set",
        f"output_dirs.merged_root={input_root}/outputs/symmetric_merged/{CAMPAIGN_BASE}_{campaign_suffix}/{modality}",
        "--set",
        f"output_dirs.run_root={input_root}/output_model/symmetric_merged/{CAMPAIGN_BASE}_{campaign_suffix}_likelihood/{modality}",
        "--input-root",
        input_root,
        "--pooled-runtime-root",
        pooled_runtime_root,
        "--log-root",
        f"{runtime}/logs/symmetric_merged",
    ]
    if stage == "smoke":
        command += ["--smoke-subjects", "2", "--smoke-epochs", "1", "--smoke-trials", "0"]
    env = dict(os.environ)
    env.update(
        {
            "SYMMETRIC_MERGED_SOURCE_COMMIT": source_commit,
            "QWEN_HIDDEN_DEPS": f"{input_root}/.deps/qwen_hidden",
            "QWEN38_ENV_ACTIVATE": "/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921",
            "QWEN3OMNI_ENV_ACTIVATE": "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni",
        }
    )
    runner = runner or subprocess.run
    try:
        result = runner(
            command, cwd=deployment_code, capture_output=True, text=True, timeout=1200, env=env
        )
    except subprocess.TimeoutExpired as exc:
        return uncertain_record(
            stage,
            route,
            seed,
            "timeout",
            _text(getattr(exc, "stdout", "")),
            _text(getattr(exc, "stderr", "")),
            registry=registry_path,
        )
    except OSError as exc:
        return uncertain_record(stage, route, seed, f"oserror: {exc}", registry=registry_path)
    except Exception as exc:  # noqa: BLE001 - fail closed after possible delivery
        return uncertain_record(
            stage,
            route,
            seed,
            f"unexpected runner error: {type(exc).__name__}: {exc}",
            registry=registry_path,
        )
    stdout, stderr = result.stdout or "", result.stderr or ""
    if result.returncode == 0:
        payload = _extract_delivery(stdout)
        if payload:
            job_ids = [str(value) for value in payload.get("job_ids") or []]
            if len(job_ids) == expected:
                return {
                    "ts": int(time.time()),
                    "status": "submitted",
                    "stage": stage,
                    "route": route,
                    "seed": seed,
                    "run_id": payload.get("run_id"),
                    "registry": registry_path,
                    "job_ids": job_ids,
                }
            return uncertain_record(
                stage,
                route,
                seed,
                f"partial delivery {len(job_ids)}/{expected}",
                stdout,
                stderr,
                job_ids=job_ids,
                registry=registry_path,
            )
        return uncertain_record(
            stage, route, seed, "unparsed success output", stdout, stderr, registry=registry_path
        )
    return uncertain_record(
        stage, route, seed, f"submit rc={result.returncode}", stdout, stderr, registry=registry_path
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def uncertain_record(
    stage: str,
    route: str,
    seed: int,
    reason: str,
    stdout: str = "",
    stderr: str = "",
    *,
    job_ids: Iterable[str] = (),
    registry: str = "",
) -> dict:
    expected = LEG_JOBS.get(stage, 0)
    parsed = [str(value) for value in job_ids if str(value).isdigit()]
    return {
        "ts": int(time.time()),
        "status": "uncertain",
        "stage": stage,
        "route": route,
        "seed": seed,
        "reason": reason,
        "registry": registry,
        "job_ids": parsed,
        "reservation": max(0, expected - len(parsed)),
        "tail": (stdout + "\n" + stderr)[-800:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["smoke", "cv", "final"])
    parser.add_argument("--deployment-code", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pooled-runtime-root", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--scheduler", default="ozu647717@alogin1.bsc.es")
    parser.add_argument("--routes", nargs="*", default=list(ROUTES))
    parser.add_argument("--seeds", nargs="*", type=int, default=[7, 1337, 2024])
    parser.add_argument("--only", default="")
    parser.add_argument("--max-legs", type=int, default=5, help="hard cap on legs per invocation")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    leg_size = LEG_JOBS[args.stage]
    if args.max_legs < 1 or args.max_legs * leg_size > 80:
        print(f"REFUSED: planned wave of {args.max_legs} legs exceeds the 80-job lane allocation")
        return 2

    registries_dir = Path(args.runtime) / "registries"
    ledger_path = Path(args.runtime) / "submissions.jsonl"
    planned = [
        (route, seed)
        for route in args.routes
        for seed in args.seeds
        if not args.only or args.only == route
    ]

    processed = 0
    for route, seed in planned:
        if processed >= args.max_legs:
            break
        try:
            ids = sorted(registry_job_ids(registries_dir))
            ledger_ids, reservation = ledger_reservations(ledger_path)
            ids = sorted(set(ids) | ledger_ids)
            states, user_total = query_job_states(ids, args.scheduler)
            own = own_nonterminal_count(ids, states, reservation)
            leg_admission(own, user_total, leg_size)
        except AdmissionError as error:
            print(f"REFUSED before {route} s{seed}: {error}")
            return 2
        line = f"{route} s{seed}: admission ok (own_nonterminal={own}, user_queue={user_total}, leg={leg_size})"
        if not args.execute:
            print(line)
            processed += 1
            continue
        record = submit_leg(
            stage=args.stage,
            route=route,
            seed=seed,
            deployment_code=args.deployment_code,
            source_commit=args.source_commit,
            input_root=args.input_root,
            pooled_runtime_root=args.pooled_runtime_root,
            runtime=args.runtime,
        )
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record) + "\n")
        processed += 1
        print(f"{line} -> {record['status']} {record.get('job_ids', '')}", flush=True)
        if record.get("status") != "submitted":
            print("stopping on non-submitted leg (fail-closed; no automatic retry)")
            return 1
    print(f"dispatch complete: {processed} leg(s) processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
