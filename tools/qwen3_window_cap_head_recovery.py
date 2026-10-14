#!/usr/bin/env python3
"""Exact affected-head recovery for the window-cap lane (tracked implementation).

The lane wrapper ``outputs/qwen3_train_window_cap_20261008/resubmit_heads.py``
passes the exact configuration (corrected deployment, known affected old
deployments, affected train+val datasets) and calls this module.

Fail-closed rules:

* a key is replaceable only when its latest old attempt's classifier is
  demonstrably terminal FAILED/CANCELLED, or COMPLETED with a durable confound
  record whose exact attempt, full old deployment id and registry key match the
  current entry (never a blanket or boolean flag);
* confound records are only written for exact known affected old deployments
  AND canonical train+val routes (Androids/D3TEC); unknown sources and
  train-only routes (DAIC/CMDC/Turkish) are never auto-classified, they are
  held;
* the scheduler reconcile is enforced (SSH failure, missing or contradictory
  top-level ids abort);
* the authoritative plan is rebuilt under the lane submission lock and must
  carry a fresh token inside the 900 s build gate;
* every wave writes an executable plan subset containing exactly its admitted
  keys, and the budget pre-check refuses any wave that would exceed the cap.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "qwen3_train_window_cap_20261008"
DEFAULT_SCHEDULER = "ozu647717@alogin2.bsc.es"
DEFAULT_CAMPAIGN = "qwen3_train_window_cap_20261008"
# Canonical train+val routes whose mask-only cap dropped inner-val rows.
AFFECTED_POOL_DATASETS = ("androids_interview", "d3tec")
# Exact immutable old head deployments known to carry the wrong pool semantics.
KNOWN_AFFECTED_OLD_DEPLOYMENTS = (
    "feat-qwen3-train-window-cap-20261008-20261008T114534Z-2f104779-db057646",
    "feat-qwen3-train-window-cap-20261008-20261009T140151Z-98d8a91d-0490fe33",
)
DEFAULT_WAVE_KEYS = 5
DEFAULT_OWN_CAP = 80

try:
    from . import qwen3_window_cap_head_plan as planner
    from . import qwen3_window_cap_heads_wave as wave
except ImportError:  # direct script execution
    sys.path.insert(0, str(TOOLS_DIR))
    import qwen3_window_cap_head_plan as planner
    import qwen3_window_cap_heads_wave as wave

# Configured per campaign dir by configure(); defaults keep the operator path.
LANE = DEFAULT_CAMPAIGN_DIR
PLAN = LANE / "heads_production_plan.json"
REGISTRY = LANE / "head_submissions.jsonl"
SUBMIT_OUTPUT = LANE / "head_submit_evidence" / "submit_output.log"
EVIDENCE = LANE / "head_resubmissions.jsonl"
CONFOUNDED = LANE / "head_confounded_attempts.jsonl"
SUBMISSION_LOCK = LANE / "submission.lock"
ENABLE_MARKER = LANE / "heads_dispatch_enabled"
ADMIT = LANE / "admit.sh"
CORRECTED_HEAD_DEPLOYMENT: str | None = None
AFFECTED_OLD_DEPLOYMENTS = KNOWN_AFFECTED_OLD_DEPLOYMENTS
AFFECTED_DATASETS = AFFECTED_POOL_DATASETS
SCHEDULER = DEFAULT_SCHEDULER
CAMPAIGN = DEFAULT_CAMPAIGN
WAVE_KEYS = DEFAULT_WAVE_KEYS
OWN_CAP = DEFAULT_OWN_CAP


def configure(
    campaign_dir: Path | str | None = None,
    *,
    corrected_deployment: str | None = None,
    old_deployments: tuple[str, ...] | None = None,
    affected_datasets: tuple[str, ...] | None = None,
    scheduler_host: str | None = None,
    campaign: str | None = None,
    wave_keys: int | None = None,
    own_cap: int | None = None,
) -> None:
    global LANE, PLAN, REGISTRY, SUBMIT_OUTPUT, EVIDENCE, CONFOUNDED
    global SUBMISSION_LOCK, ENABLE_MARKER, ADMIT
    global CORRECTED_HEAD_DEPLOYMENT, AFFECTED_OLD_DEPLOYMENTS, AFFECTED_DATASETS
    global SCHEDULER, CAMPAIGN, WAVE_KEYS, OWN_CAP
    LANE = Path(campaign_dir or DEFAULT_CAMPAIGN_DIR).resolve()
    PLAN = LANE / "heads_production_plan.json"
    REGISTRY = LANE / "head_submissions.jsonl"
    SUBMIT_OUTPUT = LANE / "head_submit_evidence" / "submit_output.log"
    EVIDENCE = LANE / "head_resubmissions.jsonl"
    CONFOUNDED = LANE / "head_confounded_attempts.jsonl"
    SUBMISSION_LOCK = LANE / "submission.lock"
    ENABLE_MARKER = LANE / "heads_dispatch_enabled"
    ADMIT = LANE / "admit.sh"
    if corrected_deployment is not None:
        CORRECTED_HEAD_DEPLOYMENT = str(corrected_deployment)
    if old_deployments is not None:
        AFFECTED_OLD_DEPLOYMENTS = tuple(str(d) for d in old_deployments)
    if affected_datasets is not None:
        AFFECTED_DATASETS = tuple(str(d).lower() for d in affected_datasets)
    if scheduler_host is not None:
        SCHEDULER = str(scheduler_host)
    if campaign is not None:
        CAMPAIGN = str(campaign)
    if wave_keys is not None:
        WAVE_KEYS = int(wave_keys)
    if own_cap is not None:
        OWN_CAP = int(own_cap)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(entry: dict) -> None:
    with EVIDENCE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# registry / confound selection (pure)
# ---------------------------------------------------------------------------


def latest_registry_entries() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if REGISTRY.is_file():
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            key = str(entry.get("registry_key") or "")
            if "_cap" in key:
                latest[key] = entry
    return latest


def needs_new_attempt(entry: dict) -> bool:
    """Exact full deployment identity comparison (no suffix matching)."""
    if CORRECTED_HEAD_DEPLOYMENT is None:
        return False
    return str(entry.get("deployment_id") or "") != CORRECTED_HEAD_DEPLOYMENT


def route_dataset(key: str) -> str:
    return key.split("|", 1)[0].rsplit("_cap", 1)[0].split("_")[0].lower() if "|" in key else ""


def dataset_of_key(key: str) -> str:
    prefix = key.split("|", 1)[0]
    for dataset in ("androids_interview", "d3tec", "cmdc", "turkish", "daic"):
        if prefix.startswith(dataset):
            return dataset
    return ""


def confounded_records() -> dict[str, dict]:
    """Exact attempts with full confound evidence (never a bare boolean)."""
    records: dict[str, dict] = {}
    if CONFOUNDED.is_file():
        for line in CONFOUNDED.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            evidence = entry.get("mismatch_evidence")
            if (
                entry.get("confounded") is True
                and entry.get("attempt_id")
                and entry.get("deployment_id")
                and entry.get("registry_key")
                and isinstance(evidence, dict)
                and evidence
            ):
                records[str(entry["attempt_id"])] = entry
    return records


def confound_match(record: dict | None, entry: dict, key: str) -> bool:
    """The confound record must match the actual entry's deployment and key."""
    if not record:
        return False
    return (
        str(record.get("deployment_id")) == str(entry.get("deployment_id") or "")
        and str(record.get("registry_key")) == key
    )


def build_confound_record(entry: dict, key: str, classifier_state: str) -> dict | None:
    """Full-evidence confound record, or None when the source is not eligible.

    Only exact known affected old deployments AND canonical train+val routes
    qualify; unknown deployments and train-only routes are never classified.
    """
    deployment = str(entry.get("deployment_id") or "")
    dataset = dataset_of_key(key)
    if deployment not in AFFECTED_OLD_DEPLOYMENTS or dataset not in AFFECTED_DATASETS:
        return None
    return {
        "at_utc": now(),
        "registry_key": key,
        "attempt_id": str(entry.get("attempt_id") or ""),
        "deployment_id": deployment,
        "dataset": dataset,
        "confounded": True,
        "disposition": (
            "terminal_confounded" if classifier_state == "COMPLETED" else "pending_terminal"
        ),
        "mismatch_evidence": {
            "pool_policy": "Androids/D3TEC outer_train includes inner-val; others train only",
            "old_semantics": "mask-only cap dropped inner-val rows",
            "old_deployment": deployment,
            "reference": "control_head_pool_independent_1529.json",
        },
    }


def update_confound_registry(entries: dict[str, dict], states: dict[str, str]) -> int:
    existing = {str(r.get("attempt_id") or "") for r in confounded_records().values()}
    appended = 0
    for key, entry in sorted(entries.items()):
        attempt = str(entry.get("attempt_id") or "")
        if not attempt or attempt in existing:
            continue
        confound = build_confound_record(entry, key, states.get(str(entry.get("classifier_job_id")), "UNKNOWN"))
        if confound is None:
            continue
        with CONFOUNDED.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(confound, sort_keys=True) + "\n")
        appended += 1
    return appended


def select_authorized(
    pending: dict[str, dict], states: dict[str, str], confounded: dict[str, dict]
) -> tuple[list[str], dict[str, str]]:
    authorized: list[str] = []
    held: dict[str, str] = {}
    for key, entry in sorted(pending.items()):
        record_match = confound_match(confounded.get(str(entry.get("attempt_id"))), entry, key)
        ok, why = wave.replacement_gate(
            states.get(str(entry.get("extract_job_id")), "UNKNOWN"),
            states.get(str(entry.get("classifier_job_id")), "UNKNOWN"),
            confounded=record_match,
        )
        if ok:
            authorized.append(key)
        else:
            held[key] = why
    return authorized, held


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def own_nonterminal() -> int:
    proc = subprocess.run(
        ["bash", str(ADMIT), "status"], capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if proc.returncode != 0:
        return 10**6
    return int(json.loads(proc.stdout)["own_nonterminal_count"])


def reconcile_states(ids: list[str]) -> tuple[dict[str, str] | None, str]:
    wanted = sorted({job_id for job_id in ids if str(job_id).isdigit()})
    if not wanted:
        return {}, "ok"
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            SCHEDULER,
            f"sacct -j {','.join(wanted)} -n -P -o JobIDRaw,State,ExitCode",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None, "ssh/sacct query failed"
    parsed = planner.parse_sacct(proc.stdout, set(wanted))
    missing = sorted(set(wanted) - set(parsed))
    if missing:
        return None, f"missing top-level ids: {missing[:5]}"
    contradictory = sorted(
        job_id for job_id, info in parsed.items() if info.get("state") == "CONTRADICTION"
    )
    if contradictory:
        return None, f"contradictory top-level ids: {contradictory[:5]}"
    return {job_id: info["state"] for job_id, info in parsed.items()}, "ok"


def wave_headroom_ok(own: int, keys: int) -> bool:
    """Whole-wave admission: own + 2 * min(wave size, keys) must fit the cap."""
    wave = min(WAVE_KEYS, keys)
    return own + 2 * wave <= OWN_CAP


def guard_command(wave_plan: Path, keys: list[str]) -> list[str]:
    command = [
        "bash",
        str(ADMIT),
        "run-submit",
        "--jobs-this-submit",
        "2",
        "--wave-fits",
        str(len(keys)),
        "--kind",
        "head",
        "--delivery-file",
        str(SUBMIT_OUTPUT),
        "--",
        sys.executable,
        str(TOOLS_DIR / "qwen3_heads_dispatch.py"),
        "submit",
        "--plan",
        str(wave_plan),
        "--campaign",
        CAMPAIGN,
        "--deployment-id",
        str(CORRECTED_HEAD_DEPLOYMENT),
    ]
    for key in keys:
        command += ["--resubmit-key", key]
    command += ["--execute"]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--corrected-deployment", default=None)
    parser.add_argument("--old-deployment", action="append", default=None)
    parser.add_argument("--affected-dataset", action="append", default=None)
    parser.add_argument("--scheduler-host", default=None)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--wave-keys", type=int, default=None)
    parser.add_argument("--own-cap", type=int, default=None)
    args = parser.parse_args(argv)
    configure(
        args.campaign_dir,
        corrected_deployment=args.corrected_deployment,
        old_deployments=tuple(args.old_deployment) if args.old_deployment else None,
        affected_datasets=tuple(args.affected_dataset) if args.affected_dataset else None,
        scheduler_host=args.scheduler_host,
        campaign=args.campaign,
        wave_keys=args.wave_keys,
        own_cap=args.own_cap,
    )
    if CORRECTED_HEAD_DEPLOYMENT is None:
        print("corrected head deployment not configured; recovery holds")
        return 0
    if not ENABLE_MARKER.is_file():
        print("affected head admissions paused: missing heads_dispatch_enabled")
        return 0
    lock = SUBMISSION_LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("submission lock busy; aborting")
        return 0
    entries = latest_registry_entries()
    pending = {k: e for k, e in entries.items() if needs_new_attempt(e)}
    states, reason = reconcile_states(
        [
            str(e[field])
            for e in pending.values()
            for field in ("extract_job_id", "classifier_job_id")
            if str(e.get(field) or "").isdigit()
        ]
    )
    if states is None:
        record({"at_utc": now(), "event": "reconcile_failed", "reason": reason, "pending_keys": sorted(pending)})
        print(f"reconcile failed closed: {reason}")
        return 1
    record({"at_utc": now(), "event": "reconciled_enforced", "states": states})
    appended = update_confound_registry(entries, states)
    if appended:
        record({"at_utc": now(), "event": "confound_registry_updated", "appended": appended})
    authorized, held = select_authorized(pending, states, confounded_records())
    print(f"pending={len(pending)} authorized={len(authorized)} held={len(held)}")
    if held:
        record({"at_utc": now(), "event": "replacement_held", "held": held})
    if not authorized:
        return 0
    own = own_nonterminal()
    if not wave_headroom_ok(own, len(authorized)):
        record(
            {
                "at_utc": now(),
                "event": "admission_hold",
                "own_nonterminal": own,
                "authorized_keys": authorized,
                "reason": "no headroom for the next wave; no submission attempted",
            }
        )
        print(f"admission hold: own={own}; nothing submitted")
        return 0
    wave.configure(campaign_dir=LANE)
    rc = wave.rebuild_plan()
    if rc != 0:
        record({"at_utc": now(), "event": "plan_rebuild_failed", "rc": rc})
        return 1
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    age = wave.plan_age_seconds(plan)
    if not plan.get("build_token") or age is None or age > wave.PLAN_MAX_AGE_SECONDS:
        record({"at_utc": now(), "event": "plan_stale_refused", "age": age, "token": plan.get("build_token")})
        print(f"plan stale or untokened (age={age}); refusing")
        return 1
    record({"at_utc": now(), "event": "plan_rebuilt", "build_token": plan["build_token"], "age_seconds": age})
    rc_all = 0
    wave_index = 0
    for start in range(0, len(authorized), WAVE_KEYS):
        keys = authorized[start : start + WAVE_KEYS]
        wave_index += 1
        own = own_nonterminal()
        if not wave_headroom_ok(own, len(keys)):
            record({"at_utc": now(), "event": "admission_hold", "own_nonterminal": own, "authorized_keys": authorized[start:]})
            print(f"hold before wave {wave_index}: own={own}")
            break
        subset = wave.plan_subset_for_keys(plan, set(keys))
        wave_plan = LANE / f"head_resubmit_wave_{wave_index}.json"
        wave_plan.write_text(json.dumps(subset, indent=1) + "\n", encoding="utf-8")
        command = guard_command(wave_plan, keys)
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        record(
            {
                "at_utc": now(),
                "event": "resubmit_wave",
                "wave": wave_index,
                "keys": keys,
                "wave_plan": str(wave_plan),
                "build_token": subset["build_token"],
                "deployment_id": CORRECTED_HEAD_DEPLOYMENT,
                "rc": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-3000:],
                "stderr_tail": (proc.stderr or "")[-1000:],
            }
        )
        print(f"wave {wave_index}: keys={len(keys)} rc={proc.returncode}")
        if proc.returncode != 0:
            rc_all = 1
            break
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
