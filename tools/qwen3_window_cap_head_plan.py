#!/usr/bin/env python3
"""Build the production masked-head dispatch plan from completed parents.

Tracked implementation of the window-cap production planner. The lane runtime
shim ``outputs/qwen3_train_window_cap_20261008/prepare_production_head_plan.py``
calls this tool with ``--campaign-dir <lane dir>`` so the running watcher keeps
its exact behavior.

Identity and safety rules enforced here:

* head identities are arm-explicit: ``route_id`` becomes ``<base_route>_<arm>``
  so ``(route_id, seed, fold)`` keys, logical run names, attempt ids and cache
  directories never collide across cap25/50/75; historical smoke registry keys
  are untouched;
* the full 378-row production matrix must yield 378 distinct expected head
  keys or the build fails closed;
* readiness requires the exact numeric train and best_eval Slurm job ids (from
  the fold's ``jobs.jsonl`` SUBMITTED events) to be confirmed ``COMPLETED``
  with exit ``0:0`` by the scheduler (``sacct``). Only exact top-level
  allocation records count: ``.batch``/``.extern``/step lines are ignored, a
  step-only result leaves the parent UNKNOWN, and contradictory duplicate
  top-level records are refused;
* the mask block must agree with the run_config window_cap block, the parent
  arm must match its nominal fraction, the split fingerprint must be non-null
  and non-empty, and run_config/metadata attempt ids must equal the submitted
  attempt;
* the plan file is rewritten on every build (empty plan included) and carries a
  fresh ``build_token`` so a stale plan can never be dispatched after
  readiness is withdrawn.

Usage:
  python tools/qwen3_window_cap_head_plan.py --campaign-dir <dir> [--no-fetch]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "qwen3_train_window_cap_20261008"
DEFAULT_RUN_ROOT = (
    "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/"
    "qwen3_train_window_cap_20261008"
)
DEFAULT_RUNTIME_ROOT = (
    "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
    "feat-qwen3-train-window-cap-20261008"
)
DEFAULT_TRANSFER_HOST = "ozu647717@transfer1.bsc.es"
DEFAULT_SCHEDULER_HOST = "ozu647717@alogin2.bsc.es"
HEAD_SEED = 1337
VARIANTS = ["logreg_raw", "xgb_raw"]
CAMPAIGN = "qwen3_train_window_cap_20261008"
GROUP_ID = "qwen3-train-window-cap-20261008"
EXPERIMENT_ID = "feat-qwen3-train-window-cap-20261008"
EXPECTED_TREATMENTS = 378
ARM_FRACTION = {"cap25": 0.25, "cap50": 0.5, "cap75": 0.75}

# Configured per campaign dir by configure(); defaults keep the operator path.
LANE = DEFAULT_CAMPAIGN_DIR
MATRIX = LANE / "production_matrix.json"
SUBMISSIONS = LANE / "production_submissions.jsonl"
RAW = LANE / "effective_fractions_raw"
OUT = LANE / "heads_production_plan.json"
SCHED_EVIDENCE = LANE / "heads_readiness_scheduler.json"
RUN_ROOT = DEFAULT_RUN_ROOT
RUNTIME = DEFAULT_RUNTIME_ROOT
CACHE_ROOT = f"{RUNTIME}/heads_cache_prod_20261009"
TRANSFER = DEFAULT_TRANSFER_HOST
SCHEDULER = DEFAULT_SCHEDULER_HOST


def configure(
    campaign_dir: Path | str | None = None,
    *,
    run_root: str | None = None,
    runtime_root: str | None = None,
    transfer_host: str | None = None,
    scheduler_host: str | None = None,
) -> None:
    """Point the planner at one campaign dir and optional remote roots."""
    global LANE, MATRIX, SUBMISSIONS, RAW, OUT, SCHED_EVIDENCE
    global RUN_ROOT, RUNTIME, CACHE_ROOT, TRANSFER, SCHEDULER
    LANE = Path(campaign_dir or DEFAULT_CAMPAIGN_DIR).resolve()
    MATRIX = LANE / "production_matrix.json"
    SUBMISSIONS = LANE / "production_submissions.jsonl"
    RAW = LANE / "effective_fractions_raw"
    OUT = LANE / "heads_production_plan.json"
    SCHED_EVIDENCE = LANE / "heads_readiness_scheduler.json"
    if run_root is not None:
        RUN_ROOT = str(run_root)
    if runtime_root is not None:
        RUNTIME = str(runtime_root)
    CACHE_ROOT = f"{RUNTIME}/heads_cache_prod_20261009"
    if transfer_host is not None:
        TRANSFER = str(transfer_host)
    if scheduler_host is not None:
        SCHEDULER = str(scheduler_host)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# arm-explicit identities
# ---------------------------------------------------------------------------


def head_route_id(row: dict) -> str:
    return f"{row['route_id']}_{row['arm']}"


def head_key(row: dict) -> str:
    return f"{head_route_id(row)}|{int(row['seed'])}|{int(row['fold'])}"


def logical_run_name(row: dict) -> str:
    return f"windowcap_head_{head_route_id(row)}_s{row['seed']}_f{row['fold']}"


def cache_dir(row: dict) -> str:
    return (
        f"{CACHE_ROOT}/{row['dataset']}/{row['modality']}/"
        f"{row['run_name']}_fold_{row['fold']}_pseed{row['seed']}_hseed{HEAD_SEED}"
    )


def expected_keys(matrix: dict) -> set[str]:
    return {head_key(row) for row in matrix.get("treatments") or []}


def assert_matrix_keys(matrix: dict) -> set[str]:
    rows = matrix.get("treatments") or []
    keys = expected_keys(matrix)
    if len(rows) != EXPECTED_TREATMENTS or len(keys) != EXPECTED_TREATMENTS:
        raise SystemExit(
            "production matrix key audit failed: "
            f"rows={len(rows)} distinct_keys={len(keys)} expected={EXPECTED_TREATMENTS}"
        )
    return keys


# ---------------------------------------------------------------------------
# evidence fetch and submission index
# ---------------------------------------------------------------------------


def fetch_evidence() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "rsync",
            "-a",
            "--no-motd",
            "--include=*/",
            "--include=status.json",
            "--include=jobs.jsonl",
            "--include=window_cap_mask.json",
            "--include=run_config.yaml",
            "--include=metadata.json",
            "--include=split_used.json",
            "--exclude=*",
            f"{TRANSFER}:{RUN_ROOT}/",
            str(RAW) + "/",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"rsync failed: {proc.stderr.strip()[:300]}", file=sys.stderr)
        return proc.returncode
    return 0


def load_rows() -> list[dict]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_name = {row["run_name"]: row for row in matrix["treatments"]}
    rows = []
    if SUBMISSIONS.is_file():
        for line in SUBMISSIONS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("rc") != 0:
                continue
            row = dict(by_name.get(entry["run_name"], {}))
            row["attempt_id"] = entry.get("attempt_id")
            rows.append(row)
    return rows


def fold_dir(row: dict, raw_root: Path | None = None) -> Path:
    root = raw_root if raw_root is not None else RAW
    return (
        root / row["modality"] / row["dataset"] / row["run_name"] / f"fold_{row['fold']}"
    )


def submitted_job_ids(row: dict, raw_root: Path | None = None) -> dict[str, str]:
    """Last numeric SUBMITTED job id per job key from the fold's jobs.jsonl."""
    ids: dict[str, str] = {}
    jobs_path = fold_dir(row, raw_root) / "jobs.jsonl"
    if not jobs_path.is_file():
        return ids
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") != "SUBMITTED":
            continue
        job_id = str(event.get("slurm_job_id") or "").strip()
        if job_id.isdigit():
            ids[str(event.get("job_key"))] = job_id
    return ids


def terminal_events(row: dict, raw_root: Path | None = None) -> dict[str, dict]:
    """Last terminal event per job key."""
    terminal: dict[str, dict] = {}
    jobs_path = fold_dir(row, raw_root) / "jobs.jsonl"
    if not jobs_path.is_file():
        return terminal
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") in {"COMPLETED", "FAILED", "CANCELLED"}:
            terminal[str(event.get("job_key"))] = event
    return terminal


# ---------------------------------------------------------------------------
# readiness
# ---------------------------------------------------------------------------


def readiness_local(row: dict, raw_root: Path | None = None) -> tuple[bool, str]:
    """Local evidence gate (no scheduler): identity, mask, arm, contradiction."""
    fold = fold_dir(row, raw_root)
    paths = {
        "jobs": fold / "jobs.jsonl",
        "mask": fold / "window_cap_mask.json",
        "run_config": fold / "run_config.yaml",
        "metadata": fold / "metadata.json",
        "split_used": fold / "logs" / "split_used.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return False, "missing evidence: " + ",".join(sorted(missing))
    import yaml

    run_config = yaml.safe_load(paths["run_config"].read_text(encoding="utf-8"))
    recorded = run_config.get("config") or {}
    window_cap = (recorded.get("training") or {}).get("window_cap") or {}
    mask = json.loads(paths["mask"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not bool(window_cap.get("enabled", False)):
        return False, "recorded config has no enabled window_cap"
    if (run_config.get("tracking") or {}).get("attempt_id") != row["attempt_id"]:
        return False, "run_config attempt id mismatch"
    if metadata.get("attempt_id") != row["attempt_id"]:
        return False, "metadata attempt id mismatch"
    for field in (
        "selection_sha256",
        "baseline_input_sha256",
        "fraction",
        "sampling_seed",
    ):
        if mask.get(field) != window_cap.get(field):
            return False, f"mask {field} disagrees with run_config"
    nominal = ARM_FRACTION.get(str(row.get("arm")))
    if nominal is None:
        return False, f"unknown arm {row.get('arm')!r}"
    if row.get("fraction") != nominal:
        return False, f"matrix fraction {row.get('fraction')!r} != arm nominal {nominal}"
    if mask.get("fraction") != nominal:
        return False, f"mask fraction {mask.get('fraction')!r} != arm nominal {nominal}"
    fingerprint = split_fingerprint(paths["split_used"])
    if fingerprint is None:
        return False, "missing split fingerprint"
    if not fingerprint.get("sha256") or not fingerprint.get("train_subject_ids"):
        return False, "empty split fingerprint"
    terminal = terminal_events(row, raw_root)
    for job_key in ("train", "best_eval"):
        event = terminal.get(job_key)
        if event is None:
            continue
        if event.get("event_type") != "COMPLETED":
            return False, f"local {job_key} terminal event {event.get('event_type')}"
        exit_code = event.get("exit_code")
        if exit_code not in (None, "0:0"):
            return False, f"local {job_key} exit_code {exit_code}"
    return True, "local-ready"


def parse_sacct(output: str, wanted: set[str]) -> dict[str, dict]:
    """Parse sacct -P output for exact top-level allocation records only.

    Only ``raw == <exact wanted id>`` lines are accepted. Step records
    (``<id>.batch``, ``<id>.extern``, ``<id>.0``) are ignored: a completed
    step can never prove the parent allocation's state. Contradictory
    duplicate top-level records are refused as a contradiction instead of
    last-wins.
    """
    parsed: dict[str, dict] = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        raw = parts[0].strip()
        if raw not in wanted:
            continue
        state = parts[1].strip().split()[0] if parts[1].strip() else "UNKNOWN"
        info = {"state": state, "exit": parts[2].strip()}
        if raw in parsed and parsed[raw] != info:
            previous = parsed[raw]
            parsed[raw] = {
                "state": "CONTRADICTION",
                "exit": (
                    f"{previous['state']}:{previous['exit']} vs "
                    f"{info['state']}:{info['exit']}"
                ),
            }
        elif raw not in parsed:
            parsed[raw] = info
    return parsed


def scheduler_confirm(
    rows: list[dict],
    *,
    host: str | None = None,
    runner=None,
    raw_root: Path | None = None,
) -> tuple[dict[str, dict], dict]:
    """Confirm exact job ids with sacct; batched per dataset."""
    host = host or SCHEDULER
    by_dataset: dict[str, set[str]] = {}
    for row in rows:
        ids = submitted_job_ids(row, raw_root)
        for job_id in ids.values():
            by_dataset.setdefault(row["dataset"], set()).add(job_id)
    mapping: dict[str, dict] = {}
    evidence = {"host": host, "generated_at_utc": now(), "datasets": {}}
    for dataset in sorted(by_dataset):
        wanted = sorted(by_dataset[dataset])
        if not wanted:
            continue
        command = f"sacct -j {','.join(wanted)} -n -P -o JobIDRaw,State,ExitCode"
        if runner is None:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=20",
                    host,
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=900,
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        else:
            rc, stdout, stderr = runner(command)
        evidence["datasets"][dataset] = {
            "command": command,
            "rc": rc,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-500:],
        }
        if rc == 0:
            mapping.update(parse_sacct(stdout, set(wanted)))
    return mapping, evidence


def readiness_scheduler(
    row: dict, scheduler_map: dict[str, dict], raw_root: Path | None = None
) -> tuple[bool, str]:
    ids = submitted_job_ids(row, raw_root)
    for job_key in ("train", "best_eval"):
        job_id = ids.get(job_key)
        if not job_id:
            return False, f"{job_key} has no numeric submitted job id"
        info = scheduler_map.get(job_id)
        if info is None:
            return False, f"scheduler UNKNOWN for {job_key} {job_id}"
        if info["state"] != "COMPLETED":
            return False, f"scheduler state {info['state']} for {job_key} {job_id}"
        if info["exit"] != "0:0":
            return False, f"scheduler exit {info['exit']} for {job_key} {job_id}"
    return True, "scheduler-confirmed 0:0"


# ---------------------------------------------------------------------------
# plan building
# ---------------------------------------------------------------------------


def adapter_hashes(rows: list[dict]) -> dict[str, dict[str, str]]:
    """One batched SSH per dataset for adapter file hashes."""
    result: dict[str, dict[str, str]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        command_parts = []
        for row in dataset_rows:
            remote = (
                f"{RUN_ROOT}/{row['modality']}/{dataset}/{row['run_name']}/"
                f"fold_{row['fold']}"
            )
            command_parts.append(
                f"sha256sum {remote}/best_model/adapter_config.json "
                f"{remote}/best_model/adapter_model.safetensors 2>/dev/null"
            )
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                TRANSFER,
                " ; ".join(command_parts),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            print(f"adapter hash batch failed for {dataset}", file=sys.stderr)
            continue
        by_path = {}
        for line in proc.stdout.splitlines():
            digest, _, path = line.partition("  ")
            if digest and path:
                by_path[path.strip()] = digest.strip()
        for row in dataset_rows:
            remote = (
                f"{RUN_ROOT}/{row['modality']}/{dataset}/{row['run_name']}/"
                f"fold_{row['fold']}"
            )
            config_sha = by_path.get(f"{remote}/best_model/adapter_config.json")
            model_sha = by_path.get(f"{remote}/best_model/adapter_model.safetensors")
            if config_sha and model_sha:
                result[row["run_name"]] = {
                    "adapter_config_sha256": config_sha,
                    "adapter_sha256": model_sha,
                }
    return result


def split_fingerprint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = {
        key: sorted(str(item) for item in payload.get(key) or [])
        for key in ("train_subject_ids", "selection_subject_ids", "final_eval_subject_ids")
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**normalized, "sha256": hashlib.sha256(canonical).hexdigest()}


def build_route(row: dict, hashes: dict[str, str]) -> dict:
    import yaml

    fold = fold_dir(row)
    remote_fold = (
        f"{RUN_ROOT}/{row['modality']}/{row['dataset']}/{row['run_name']}/"
        f"fold_{row['fold']}"
    )
    run_config = yaml.safe_load((fold / "run_config.yaml").read_text(encoding="utf-8"))
    recorded = run_config.get("config") or {}
    window_cap = (recorded.get("training") or {}).get("window_cap") or {}
    config_path = str(row["config"])
    if Path(config_path).is_absolute():
        config_path = str(Path(config_path).relative_to(PROJECT_ROOT.resolve()))
    evaluation = recorded.get("evaluation") or {}
    resources = recorded.get("resources") or {}
    fingerprint = split_fingerprint(fold / "logs" / "split_used.json")
    if fingerprint is None or not fingerprint.get("train_subject_ids"):
        raise SystemExit(f"{row['run_name']}: missing or empty split fingerprint")
    job = {
        "route_id": head_route_id(row),
        "seed": row["seed"],
        "fold": row["fold"],
        "parent_status": "resolved",
        "head_seed": HEAD_SEED,
        "variants": VARIANTS,
        "logical_run_name": logical_run_name(row),
        "parent": {
            "attempt_id": str(row["attempt_id"]),
            "run_name": row["run_name"],
            "fold_dir": remote_fold,
            "checkpoint_dir": f"{remote_fold}/best_model",
            "adapter_config_sha256": hashes["adapter_config_sha256"],
            "adapter_sha256": hashes["adapter_sha256"],
            "split_fingerprint": fingerprint,
            "manifest_hash": run_config.get("manifest_hash"),
            "selection": "explicit_parent_map",
            "selection_reason": (
                "production fit resolved by exact run name, arm and attempt id"
            ),
            "excluded_attempts": [],
            "state": "COMPLETED_ON_MN5",
        },
        "cache_dir": cache_dir(row),
        "extract_gpus": int(resources.get("eval_gpus_per_node", 4) or 4),
        "train_mask": {
            "mask_path": f"{remote_fold}/window_cap_mask.json",
            "expected_selection_sha256": window_cap["selection_sha256"],
            "baseline_input_sha256": window_cap["baseline_input_sha256"],
            "fraction": window_cap["fraction"],
            "sampling_seed": window_cap["sampling_seed"],
            "algorithm_version": window_cap["algorithm_version"],
        },
        "aggregation": evaluation.get("aggregation_level") or "subject",
    }
    return {
        "route_id": head_route_id(row),
        "config": config_path,
        "dataset": row["dataset"],
        "modality": row["modality"],
        "language": "native",
        "backend": "qwen3omni",
        "dataset_variant": recorded.get("dataset_variant"),
        "aggregation": evaluation.get("aggregation_level") or "subject",
        "subject_score_aggregation": evaluation.get("subject_score_aggregation"),
        "hierarchical_score_aggregation": evaluation.get("hierarchical_score_aggregation"),
        "arm": row["arm"],
        "nominal_fraction": row["fraction"],
        "jobs": [job],
    }


def build_plan(
    routes: list[dict],
    *,
    reasons: dict[str, int],
    waiting: int,
    scheduler_evidence_path: Path,
    token: str | None = None,
) -> dict:
    if token is None:
        digest = hashlib.sha256(
            json.dumps(routes, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        token = f"{now()}-{digest}"
    seeds = sorted({int(job["seed"]) for route in routes for job in route["jobs"]})
    return {
        "schema_version": "audiollm.qwen3_multiseed_head_dispatch_plan.v1",
        "campaign": CAMPAIGN,
        "group_id": GROUP_ID,
        "experiment_id": EXPERIMENT_ID,
        "language": "native",
        "tracking_kind": None,
        "run_schema": None,
        "evidence_dir": str(LANE),
        "runtime_root": RUNTIME,
        "seeds": seeds,
        "head_seed": HEAD_SEED,
        "variants": VARIANTS,
        "cache_root": CACHE_ROOT,
        "source_matrix": str(MATRIX),
        "source_matrix_sha256": sha256_file(MATRIX),
        "created_at_utc": now(),
        "build_token": token,
        "scheduler_evidence": str(scheduler_evidence_path),
        "scheduler_evidence_sha256": (
            sha256_file(scheduler_evidence_path)
            if scheduler_evidence_path.is_file()
            else None
        ),
        "routes": routes,
        "summary": {
            "resolved": len(routes),
            "waiting_for_checkpoint": waiting,
            "blocked_failed_parent": 0,
            "expected_keys": EXPECTED_TREATMENTS,
            "distinct_expected_keys": EXPECTED_TREATMENTS,
            "readiness_reasons": reasons,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--transfer-host", default=None)
    parser.add_argument("--scheduler-host", default=None)
    args = parser.parse_args(argv)
    configure(
        args.campaign_dir,
        run_root=args.run_root,
        runtime_root=args.runtime_root,
        transfer_host=args.transfer_host,
        scheduler_host=args.scheduler_host,
    )
    out = args.out or OUT
    if not args.no_fetch:
        rc = fetch_evidence()
        if rc != 0:
            return rc
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    keys = assert_matrix_keys(matrix)
    rows = load_rows()
    for row in rows:
        if head_key(row) not in keys:
            raise SystemExit(f"submitted row outside expected keys: {head_key(row)}")
    reasons: dict[str, int] = {}
    local_pass: list[dict] = []
    for row in rows:
        ok, reason = readiness_local(row)
        if ok:
            local_pass.append(row)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    scheduler_map, scheduler_evidence = scheduler_confirm(local_pass)
    SCHED_EVIDENCE.write_text(
        json.dumps(scheduler_evidence, indent=1, sort_keys=True), encoding="utf-8"
    )
    ready: list[dict] = []
    for row in local_pass:
        ok, reason = readiness_scheduler(row, scheduler_map)
        if ok:
            ready.append(row)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    if not ready:
        plan = build_plan(
            [],
            reasons=reasons,
            waiting=len(rows),
            scheduler_evidence_path=SCHED_EVIDENCE,
        )
        out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
        print("no completed production parents ready for heads yet")
        for reason, count in sorted(reasons.items()):
            print(f"  {count} x {reason}")
        print(f"wrote fresh empty plan {out} token={plan['build_token']}")
        return 0
    hashes = adapter_hashes(ready)
    routes = []
    for row in ready:
        if row["run_name"] not in hashes:
            reasons["adapter hashes unavailable"] = (
                reasons.get("adapter hashes unavailable", 0) + 1
            )
            continue
        routes.append(build_route(row, hashes[row["run_name"]]))
    plan_keys = [
        f"{route['route_id']}|{job['seed']}|{job['fold']}"
        for route in routes
        for job in route["jobs"]
    ]
    if len(set(plan_keys)) != len(plan_keys):
        raise SystemExit("duplicate head keys inside the plan")
    plan = build_plan(
        routes,
        reasons=reasons,
        waiting=len(rows) - len(routes),
        scheduler_evidence_path=SCHED_EVIDENCE,
    )
    out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} token={plan['build_token']}")
    print("ready parents:", len(routes), "plan chains:", len(routes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
