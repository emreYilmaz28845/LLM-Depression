#!/usr/bin/env python3
"""Managed production dispatch for Qwen3 standalone fixed-head attempts.

This tool consumes the validated planner output (``tools/qwen3_heads_matrix.py``)
and runs the full-cohort extraction + fixed-classifier chain for every resolved
``(route, parent_training_seed, fold)`` parent on MN5. It is the production
counterpart of the smoke-only ``submit_qwen3_hidden_smoke.sh``: no smoke subject
limits, no Optuna, no PCA/control variants, and no legacy campaign routing.

Design rules enforced here:

* one head attempt per parent key; the attempt directory is lane-owned under
  the experiment runtime and the local mirror is lane-owned under the lane
  evidence directory;
* extraction runs with ``SKIP_CLASSIFIERS=1`` and the explicit classifier
  worker follows with ``--dependency=afterok`` and ``SEED=1337`` (the extractor's
  inline classifier call does not forward SEED);
* classifier variants are exactly ``logreg_raw:xgb_raw``;
* backend activation (``ENV_ACTIVATE``) and the project-local hidden dependency
  path (``QWEN_HIDDEN_DEPS``) are explicit; ignored ``.deps`` directories are
  never deployed with tracked source;
* attempt sidecars use the repository's official lifecycle and evidence
  formats through ``src/native_en_text_heads_tracking.py``; the attempt is
  initialized write-once, transitions PLANNED -> DEPLOYED -> SUBMITTED ->
  RUNNING on the scheduler login, and the classifier worker materializes
  COMPLETED_ON_MN5 evidence after the fits;
* collection is a plain no-``--delete`` rsync of the compact attempt directory
  plus a remote/local SHA-256 manifest check; validation and finishing call the
  official head lifecycle adapter on the local mirror.

Commands: ``plan``, ``submit``, ``status``, ``collect``, ``validate``,
``finish``, ``coverage``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import sha256_file  # noqa: E402
from src.experiment_tracking.deployment import (  # noqa: E402
    DEFAULT_TRANSFER_HOST,
    REMOTE_BASE,
    RemoteRunner,
    verify_deployment,
)
from src.experiment_tracking.identity import new_attempt_id  # noqa: E402
from src.experiment_tracking.submit import DEFAULT_SCHEDULER_HOST  # noqa: E402
from src.utils import load_yaml_with_overrides  # noqa: E402

SCHEMA_VERSION = "audiollm.qwen3_multiseed_head_dispatch_plan.v1"
REGISTRY_SCHEMA = "audiollm.qwen3_multiseed_head_registry.v1"
CAMPAIGN = "qwen3_multiseed_native_20261002"
HEAD_SEED = 1337
HEAD_VARIANTS = ("logreg_raw", "xgb_raw")
TRACKING_KIND = "qwen3_multiseed_native_head"
RUN_SCHEMA = "audiollm.qwen3_multiseed_head_run.v1"
REMOTE_PROJECT_ROOT = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression")
REMOTE_RUNTIME_BASE = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime")
QWEN_HIDDEN_DEPS_DEFAULT = (
    "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden"
)
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
METRIC_NAMESPACE = "headline/binary_strict"
LANE_EVIDENCE = PROJECT_ROOT / "outputs" / CAMPAIGN
DEFAULT_PLAN = LANE_EVIDENCE / "head_dispatch_plan_v1.json"
DEFAULT_REGISTRY = LANE_EVIDENCE / "head_submissions.jsonl"
TERMINAL_STATES = {
    "COMPLETED": ("COMPLETED", "COMPLETED"),
    "FAILED": ("FAILED", "FAILED"),
    "CANCELLED": ("CANCELLED", "CANCELLED"),
    "TIMEOUT": ("FAILED", "TIMEOUT"),
    "OUT_OF_MEMORY": ("FAILED", "OUT_OF_MEMORY"),
    "NODE_FAIL": ("FAILED", "NODE_FAIL"),
    "PREEMPTED": ("FAILED", "PREEMPTED"),
    "BOOT_FAIL": ("FAILED", "NODE_FAIL"),
}


class DispatchError(RuntimeError):
    """A dispatch contract or remote operation failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lane_pin() -> dict[str, Any]:
    pin_path = PROJECT_ROOT / ".agent-pin.json"
    if not pin_path.is_file():
        raise DispatchError(f"lane pin is missing: {pin_path}")
    return _load_json(pin_path)


def lane_group_id() -> str:
    pin = _lane_pin()
    definition_rel = pin.get("definition_path")
    if not isinstance(definition_rel, str) or not definition_rel:
        raise DispatchError("lane pin has no definition_path")
    import yaml

    definition = yaml.safe_load((PROJECT_ROOT / definition_rel).read_text(encoding="utf-8"))
    group_rel = definition.get("experiment_group_path")
    if not group_rel:
        raise DispatchError(
            "lane definition has no experiment_group_path; link a complete "
            "audiollm.experiment_group.v1 definition before dispatch"
        )
    group = yaml.safe_load((PROJECT_ROOT / group_rel).read_text(encoding="utf-8"))
    group_id = group.get("group_id")
    if not group_id:
        raise DispatchError(f"linked experiment group has no group_id: {group_rel}")
    return str(group_id)


def _lane_runtime_root() -> Path:
    return REMOTE_RUNTIME_BASE / str(_lane_pin().get("experiment_id") or PROJECT_ROOT.name)


def _load_deployment(deployment_id: str | None) -> dict[str, Any]:
    root = PROJECT_ROOT / "outputs" / "exp_deploy"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for record_path in sorted(root.glob("*/deployment.json")):
        try:
            record = _load_json(record_path)
        except Exception:
            continue
        if deployment_id and record.get("deployment_id") != deployment_id:
            continue
        candidates.append((record_path, record))
    if not candidates:
        raise DispatchError(
            f"no local deployment record found{f' for {deployment_id}' if deployment_id else ''}"
        )
    return candidates[-1][1]


def _route_aggregation(config: dict[str, Any]) -> str:
    evaluation = config.get("evaluation") or {}
    return str(evaluation.get("aggregation_level") or "subject")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def command_plan(args: argparse.Namespace) -> int:
    matrix = _load_json(args.matrix)
    if matrix.get("schema_version") != "audiollm.qwen3_heads_matrix.v1":
        raise DispatchError(f"unexpected planner schema: {matrix.get('schema_version')!r}")
    seeds = [int(seed) for seed in (args.seed or matrix.get("planned_seeds") or [])]
    routes_out: list[dict[str, Any]] = []
    resolved = waiting = blocked = 0
    for route in matrix.get("routes") or []:
        if route.get("language") != args.language:
            continue
        config = load_yaml_with_overrides(PROJECT_ROOT / str(route["config"]), [])
        route_out = {
            "route_id": route["route_id"],
            "config": route["config"],
            "dataset": route["dataset"],
            "modality": route["modality"],
            "language": route["language"],
            "backend": route["backend"],
            "dataset_variant": config.get("dataset_variant"),
            "aggregation": _route_aggregation(config),
            "subject_score_aggregation": (config.get("evaluation") or {}).get(
                "subject_score_aggregation"
            ),
            "hierarchical_score_aggregation": (config.get("evaluation") or {}).get(
                "hierarchical_score_aggregation"
            ),
            "jobs": [],
        }
        for job in route.get("jobs") or []:
            if int(job["seed"]) not in seeds:
                continue
            entry: dict[str, Any] = {
                "route_id": route["route_id"],
                "seed": int(job["seed"]),
                "fold": int(job["fold"]),
                "parent_status": job["parent_status"],
                "head_seed": HEAD_SEED,
                "variants": list(HEAD_VARIANTS),
                "logical_run_name": f"q3ms_head_{route['route_id']}_s{job['seed']}_f{job['fold']}",
            }
            if job["parent_status"] == "resolved":
                parent = job["parent"]
                entry.update(
                    {
                        "parent": {
                            "attempt_id": parent["attempt_id"],
                            "run_name": parent["run_name"],
                            "fold_dir": parent["fold_dir"],
                            "checkpoint_dir": parent["checkpoint_dir"],
                            "adapter_config_sha256": parent["checkpoint_adapter_config_sha256"],
                            "adapter_sha256": parent["checkpoint_adapter_model_sha256"],
                            "split_fingerprint": parent.get("split_fingerprint"),
                            "manifest_hash": parent.get("manifest_hash_recorded"),
                            "selection": parent.get("selection"),
                            "selection_reason": parent.get("selection_reason"),
                            "excluded_attempts": parent.get("excluded_attempts") or [],
                            "state": parent.get("state"),
                        },
                        "cache_dir": job["extract"]["cache_dir"],
                        "extract_gpus": int(job["extract"]["gpus"]),
                    }
                )
                resolved += 1
            else:
                entry["reason"] = job.get("reason")
                if job["parent_status"] == "blocked_failed_parent":
                    blocked += 1
                else:
                    waiting += 1
            route_out["jobs"].append(entry)
        routes_out.append(route_out)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "language": args.language,
        "seeds": seeds,
        "head_seed": HEAD_SEED,
        "variants": list(HEAD_VARIANTS),
        "source_matrix": str(args.matrix),
        "source_matrix_sha256": sha256_file(args.matrix),
        "created_at_utc": _now(),
        "routes": routes_out,
        "summary": {
            "resolved": resolved,
            "waiting_for_checkpoint": waiting,
            "blocked_failed_parent": blocked,
            "jobs": resolved + waiting + blocked,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("dispatch plan:", json.dumps(payload["summary"], sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def _read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _append_registry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _key(route_id: str, seed: int, fold: int) -> str:
    return f"{route_id}|{seed}|{fold}"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _remote_attempt_dir(runtime_root: Path, route_id: str, attempt_id: str) -> str:
    return str(runtime_root / "heads" / route_id / attempt_id)


def _local_mirror(attempt_id: str) -> Path:
    return LANE_EVIDENCE / "head_attempts" / attempt_id


def _backend_env(config_path: str, code_root: str, host: str) -> dict[str, str]:
    """Resolve ENV_ACTIVATE and MODEL_PATH for one config on the scheduler login."""

    from src.experiment_tracking.submit import SshSubmitRunner

    qwen_env = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
    script = "\n".join(
        [
            "set -uo pipefail",
            f"source {shlex.quote(qwen_env)} >/dev/null 2>&1 || true",
            "bash "
            + shlex.quote(f"{code_root}/scripts/harmonized_backend_env.sh")
            + " "
            + shlex.quote(config_path)
            + " "
            + shlex.quote(code_root),
        ]
    )
    proc = SshSubmitRunner(host=host).run_script(script, timeout=300)
    if proc.returncode != 0:
        raise DispatchError(f"backend env resolution failed: {proc.stderr.strip()}")
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    if not values.get("ENV_ACTIVATE"):
        raise DispatchError("backend env resolution returned no ENV_ACTIVATE")
    return values


def _init_payload(
    *,
    job: dict[str, Any],
    route: dict[str, Any],
    attempt_id: str,
    remote_attempt_dir: str,
    deployment: dict[str, Any],
    group_id: str,
) -> dict[str, Any]:
    parent = job["parent"]
    context = {
        "attempt_id": attempt_id,
        "logical_run_name": job["logical_run_name"],
        "fold": int(job["fold"]),
        "seed": int(job["seed"]),
        "group_id": group_id,
        "tracking_kind": TRACKING_KIND,
        "run_schema_version": RUN_SCHEMA,
        "created_at_utc": _now(),
        "required_jobs": ["extract", "classifier"],
        "source": {
            "git_commit": deployment["git_commit"],
            "git_branch": deployment.get("git_branch_at_deploy"),
            "git_dirty": bool(deployment.get("git_dirty", False)),
            "deployed_source_sha256": deployment.get("source_manifest_sha256"),
            "deployment_id": deployment["deployment_id"],
        },
        "research": {
            "github_issue": None,
            "github_pr": None,
        },
        "hashes": {
            "manifest_sha256": parent.get("manifest_hash"),
            "split_sha256": (parent.get("split_fingerprint") or {}).get("sha256"),
        },
    }
    scientific = {
        "campaign": CAMPAIGN,
        "route_id": route["route_id"],
        "dataset": route["dataset"],
        "modality": route["modality"],
        "language": route["language"],
        "config": route["config"],
        "stage": "standalone_head",
        "parent_training_seed": int(job["seed"]),
        "head_seed": HEAD_SEED,
        "split_seed": int(matrix_split_seed()),
        "dataset_variant": route.get("dataset_variant"),
        "classifier": {
            "method": "fixed_logreg_xgb",
            "variants": list(HEAD_VARIANTS),
            "prediction_backend": "likelihood",
            "seed": HEAD_SEED,
            "sampling_mode": "legacy",
        },
        "evaluation": {
            "evaluation_view": EVALUATION_VIEW,
            "aggregation": route.get("aggregation") or "subject",
            "subject_score_aggregation": route.get("subject_score_aggregation"),
            "hierarchical_score_aggregation": route.get("hierarchical_score_aggregation"),
            "split_name": "outer_holdout",
            "split_protocol": "saved_split",
        },
    }
    parent_payload = {
        "parent_attempt_id": parent["attempt_id"],
        "parent_checkpoint_role": "best_model",
        "parent_checkpoint_path": parent["checkpoint_dir"],
        "adapter_config_sha256": parent["adapter_config_sha256"],
        "adapter_sha256": parent["adapter_sha256"],
    }
    return {
        "attempt_dir": remote_attempt_dir,
        "context": context,
        "config": scientific,
        "parent": parent_payload,
    }


def matrix_split_seed() -> int:
    return 1337


def _remote_python_block(code: str) -> str:
    return "python - <<'PY'\n" + code + "\nPY\n"


def _build_submit_script(
    *,
    code_root: str,
    jobs: list[dict[str, Any]],
    scheduler_env: dict[str, dict[str, str]],
    qwen_hidden_deps: str,
    log_root: str,
) -> str:
    lines = [
        "set -uo pipefail",
        f"cd {shlex.quote(code_root)}",
        f"export PROJECT_ROOT={shlex.quote(code_root)}",
    ]
    init_python = _remote_python_block(
        """
import base64, json, os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from src.native_en_text_heads_tracking import initialize_head_attempt, transition_head_attempt
payload = json.loads(base64.b64decode(os.environ["Q3MS_PAYLOAD_B64"]).decode("utf-8"))
result = initialize_head_attempt(
    payload["attempt_dir"], context=payload["context"], config=payload["config"], parent=payload["parent"]
)
transition_head_attempt(payload["attempt_dir"], "DEPLOYED", reason="managed production head dispatch")
print(json.dumps(result))
"""
    )
    events_python = _remote_python_block(
        """
import json, os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from src.native_en_text_heads_tracking import record_head_job, transition_head_attempt
attempt_dir = os.environ["Q3MS_ATTEMPT_DIR"]
transition_head_attempt(attempt_dir, "SUBMITTED", reason="slurm jobs submitted")
transition_head_attempt(attempt_dir, "RUNNING", reason="head jobs queued or running")
record_head_job(
    attempt_dir, job_key="extract", job_type="hidden_extraction", event_type="SUBMITTED",
    slurm_job_id=os.environ["Q3MS_EXTRACT_ID"], status="PENDING",
)
record_head_job(
    attempt_dir, job_key="classifier", job_type="hidden_classifier", event_type="SUBMITTED",
    slurm_job_id=os.environ["Q3MS_CLASSIFIER_ID"], status="PENDING",
    dependency_job_ids=[os.environ["Q3MS_EXTRACT_ID"]],
)
print("events-recorded")
"""
    )
    for job in jobs:
        key = job["registry_key"]
        env = scheduler_env[job["attempt_id"]]
        payload_b64 = base64.b64encode(
            json.dumps(job["payload"], sort_keys=True).encode("utf-8")
        ).decode("ascii")
        attempt_dir = job["remote_attempt_dir"]
        cache_dir = job["cache_dir"]
        classifier_dir = job["classifier_dir"]
        job_log_root = job["log_root"]
        extract_export = ",".join(
            [
                "ALL",
                f"PROJECT_ROOT={code_root}",
                f"CHECKPOINT_DIR={job['parent']['checkpoint_dir']}",
                f"CACHE_DIR={cache_dir}",
                f"CLASSIFIER_DIR={classifier_dir}",
                f"CONDITION={job['condition']}",
                "SKIP_CLASSIFIERS=1",
                f"ENV_ACTIVATE={env['ENV_ACTIVATE']}",
                f"QWEN_HIDDEN_DEPS={qwen_hidden_deps}",
                f"LOG_ROOT={job_log_root}",
            ]
        )
        classifier_export = ",".join(
            [
                "ALL",
                f"PROJECT_ROOT={code_root}",
                f"CACHE_DIR={cache_dir}",
                f"CLASSIFIER_DIR={classifier_dir}",
                f"CLASSIFIER_VARIANTS={':'.join(HEAD_VARIANTS)}",
                f"SEED={HEAD_SEED}",
                f"ENV_ACTIVATE={env['ENV_ACTIVATE']}",
                f"QWEN_HIDDEN_DEPS={qwen_hidden_deps}",
                f"LOG_ROOT={job_log_root}",
                f"ATTEMPT_DIR={attempt_dir}",
                f"MATERIALIZE_VARIANTS={':'.join(HEAD_VARIANTS)}",
            ]
        )
        short = re.sub(r"[^a-z0-9]", "", job["attempt_id"].lower())[:16]
        lines.extend(
            [
                f"echo '=== JOB {key} ==='",
                f"export Q3MS_PAYLOAD_B64={shlex.quote(payload_b64)}",
                f"deactivate >/dev/null 2>&1 || true",
                f"source {shlex.quote(env['ENV_ACTIVATE'])} >/dev/null 2>&1 || true",
                init_python.rstrip("\n"),
                f"unset Q3MS_PAYLOAD_B64",
                f"export Q3MS_ATTEMPT_DIR={shlex.quote(attempt_dir)}",
                (
                    f"Q3MS_EXTRACT_ID=\"$(sbatch --parsable --chdir={shlex.quote(code_root)} "
                    f"--job-name=q3mshx-{short} --gres=gpu:{job['extract_gpus']} "
                    f"--cpus-per-task={20 * job['extract_gpus']} "
                    f"--export={shlex.quote(extract_export)} "
                    f"{shlex.quote(code_root + '/scripts/run_qwen_hidden_extract_slurm.sh')})\""
                ),
                "if [ -z \"${Q3MS_EXTRACT_ID:-}\" ]; then echo 'ERROR=extract sbatch returned no id'; continue; fi",
                f"export Q3MS_EXTRACT_ID",
                (
                    f"Q3MS_CLASSIFIER_ID=\"$(sbatch --parsable --chdir={shlex.quote(code_root)} "
                    f"--job-name=q3mshc-{short} --dependency=afterok:$Q3MS_EXTRACT_ID "
                    f"--export={shlex.quote(classifier_export)} "
                    f"{shlex.quote(code_root + '/scripts/run_qwen_hidden_classifier_slurm.sh')})\""
                ),
                "if [ -z \"${Q3MS_CLASSIFIER_ID:-}\" ]; then echo 'ERROR=classifier sbatch returned no id'; continue; fi",
                f"export Q3MS_CLASSIFIER_ID",
                f"export LOG_ROOT={shlex.quote(job_log_root)}",
                events_python.rstrip("\n"),
                f"echo \"EXTRACT_ID=$Q3MS_EXTRACT_ID\"",
                f"echo \"CLASSIFIER_ID=$Q3MS_CLASSIFIER_ID\"",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_submit_output(output: str) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in output.splitlines():
        line = line.strip()
        match = re.match(r"^=== JOB (.+) ===$", line)
        if match:
            current = match.group(1)
            results[current] = {}
            continue
        if current is None:
            continue
        if line.startswith("ERROR="):
            results[current]["error"] = line.split("=", 1)[1]
        elif line.startswith("EXTRACT_ID="):
            results[current]["extract_job_id"] = line.split("=", 1)[1]
        elif line.startswith("CLASSIFIER_ID="):
            results[current]["classifier_job_id"] = line.split("=", 1)[1]
    return results


def command_submit(args: argparse.Namespace) -> int:
    plan = _load_json(args.plan)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise DispatchError(f"unexpected dispatch plan schema: {plan.get('schema_version')!r}")
    if args.execute and args.dry_run:
        raise DispatchError("specify either --dry-run or --execute, not both")
    deployment = _load_deployment(args.deployment_id)
    code_root = str(deployment["deployed_code_path"])
    runtime_root = _lane_runtime_root()
    group_id = lane_group_id()
    registry_path = Path(args.registry)
    registry = _read_registry(registry_path)
    submitted_keys = {entry["registry_key"] for entry in registry}
    seeds = {int(seed) for seed in (args.seed or plan.get("seeds") or [])}

    jobs: list[dict[str, Any]] = []
    for route in plan.get("routes") or []:
        if args.route and route["route_id"] not in set(args.route):
            continue
        config = load_yaml_with_overrides(PROJECT_ROOT / str(route["config"]), [])
        for job in route.get("jobs") or []:
            if job["parent_status"] != "resolved":
                continue
            if int(job["seed"]) not in seeds:
                continue
            key = _key(route["route_id"], int(job["seed"]), int(job["fold"]))
            if key in submitted_keys:
                continue
            if args.key and key not in set(args.key):
                continue
            jobs.append({"route": route, "job": job, "registry_key": key, "config": config})
            if args.limit and len(jobs) >= int(args.limit):
                break
        if args.limit and len(jobs) >= int(args.limit):
            break
    if not jobs:
        print("no dispatchable jobs selected")
        return 0

    runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    scheduler_env: dict[str, dict[str, str]] = {}
    prepared: list[dict[str, Any]] = []
    for item in jobs:
        route = item["route"]
        job = item["job"]
        attempt_id = new_attempt_id(job["logical_run_name"], deployment["git_commit"])
        remote_attempt_dir = _remote_attempt_dir(runtime_root, route["route_id"], attempt_id)
        classifier_dir = f"{remote_attempt_dir}/classifier"
        job_log_root = str(runtime_root / "logs" / "heads" / route["route_id"] / attempt_id)
        env = _backend_env(
            str(PROJECT_ROOT / route["config"]),
            code_root,
            args.scheduler_host or DEFAULT_SCHEDULER_HOST,
        )
        scheduler_env[attempt_id] = env
        payload = _init_payload(
            job=job,
            route=route,
            attempt_id=attempt_id,
            remote_attempt_dir=remote_attempt_dir,
            deployment=deployment,
            group_id=group_id,
        )
        prepared.append(
            {
                "registry_key": item["registry_key"],
                "route": route,
                "job": job,
                "attempt_id": attempt_id,
                "remote_attempt_dir": remote_attempt_dir,
                "classifier_dir": classifier_dir,
                "log_root": job_log_root,
                "cache_dir": job["cache_dir"],
                "parent": job["parent"],
                "condition": route["modality"],
                "extract_gpus": int(job["extract_gpus"]),
                "payload": payload,
            }
        )

    script = _build_submit_script(
        code_root=code_root,
        jobs=prepared,
        scheduler_env=scheduler_env,
        qwen_hidden_deps=str(args.qwen_hidden_deps or QWEN_HIDDEN_DEPS_DEFAULT),
        log_root=str(runtime_root / "logs" / "heads"),
    )
    print(f"=== head dispatch submit ({'execute' if args.execute else 'dry-run'}) ===")
    print(f"deployment_id: {deployment['deployment_id']}")
    print(f"git_commit: {deployment['git_commit']}")
    print(f"group_id: {group_id}")
    print(f"runtime_root: {runtime_root}")
    print(f"jobs: {len(prepared)}")
    for item in prepared:
        print(
            f"  {item['registry_key']} -> {item['attempt_id']} "
            f"(extract gpus={item['extract_gpus']}, cache={item['cache_dir']})"
        )
    evidence_dir = LANE_EVIDENCE / "head_submit_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "submit_script.sh").write_text(script, encoding="utf-8")
    print(f"submit script: {evidence_dir / 'submit_script.sh'}")

    if not args.execute:
        print(script)
        print("dry-run complete; no mutation performed")
        return 0

    try:
        verification = verify_deployment(
            runner,
            deployment["deployment_id"],
            remote_base=REMOTE_BASE,
            expected_git_commit=deployment.get("git_commit"),
            expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
        )
        print(
            f"deployment verified: {verification['deployment_id']} "
            f"({verification['tree_verification']['verified_files']}/"
            f"{verification['tree_verification']['expected_files']} files)"
        )
    except Exception as exc:  # DeploymentError or transport
        raise DispatchError(f"deployment verification failed: {exc}") from exc

    # Read-only collision preflight before any mutation.
    for item in prepared:
        for path in (item["remote_attempt_dir"], item["cache_dir"]):
            proc = runner.run(f"test -e {shlex.quote(path)} && echo exists || echo absent")
            if proc.stdout.strip() != "absent":
                raise DispatchError(f"collision: {path} already exists")

    from src.experiment_tracking.submit import SshSubmitRunner

    submit_runner = SshSubmitRunner(host=args.scheduler_host or DEFAULT_SCHEDULER_HOST)
    proc = submit_runner.run_script(script, timeout=3600)
    (evidence_dir / "submit_output.log").write_text(
        proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8"
    )
    parsed = _parse_submit_output(proc.stdout)
    recorded = 0
    for item in prepared:
        result = parsed.get(item["registry_key"]) or {}
        entry = {
            "schema_version": REGISTRY_SCHEMA,
            "registry_key": item["registry_key"],
            "attempt_id": item["attempt_id"],
            "logical_run_name": item["job"]["logical_run_name"],
            "route_id": item["route"]["route_id"],
            "dataset": item["route"]["dataset"],
            "modality": item["route"]["modality"],
            "config": item["route"]["config"],
            "parent_training_seed": int(item["job"]["seed"]),
            "head_seed": HEAD_SEED,
            "fold": int(item["job"]["fold"]),
            "parent_attempt_id": item["parent"]["attempt_id"],
            "parent_checkpoint_dir": item["parent"]["checkpoint_dir"],
            "remote_attempt_dir": item["remote_attempt_dir"],
            "local_mirror": str(_local_mirror(item["attempt_id"])),
            "cache_dir": item["cache_dir"],
            "classifier_dir": item["classifier_dir"],
            "extract_job_id": result.get("extract_job_id"),
            "classifier_job_id": result.get("classifier_job_id"),
            "error": result.get("error"),
            "deployment_id": deployment["deployment_id"],
            "git_commit": deployment["git_commit"],
            "submitted_at_utc": _now(),
        }
        _append_registry(registry_path, entry)
        context_path = LANE_EVIDENCE / "head_contexts" / f"{item['attempt_id']}.json"
        _write_json(context_path, item["payload"]["context"])
        recorded += 1
        status = "submitted" if not result.get("error") else f"ERROR {result['error']}"
        print(f"  {item['registry_key']}: {status}")
    print(f"registry updated: {recorded} entries -> {registry_path}")
    return 0 if recorded else 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _scheduler_states(
    job_ids: list[str], host: str
) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    states: dict[str, dict[str, str]] = {}
    batch = ",".join(job_ids)
    script = "\n".join(
        [
            "set -uo pipefail",
            f"sacct -j {shlex.quote(batch)} --parsable2 "
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList 2>/dev/null || true",
        ]
    )
    from src.experiment_tracking.submit import SshSubmitRunner

    proc = SshSubmitRunner(host=host).run_script(script, timeout=300)
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4 or not parts[0]:
            continue
        job_id, job_name, state, exit_code = parts[0], parts[1], parts[2], parts[3]
        if job_id in job_ids:
            states[job_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "state": state.split()[0] if state else "",
                "exit_code": exit_code,
                "elapsed": parts[4] if len(parts) > 4 else "",
                "node": parts[5] if len(parts) > 5 else "",
            }
    return states


def _remote_record_terminal(
    *,
    code_root: str,
    attempt_dir: str,
    events: list[dict[str, str]],
    host: str,
) -> None:
    if not events:
        return
    payload = base64.b64encode(json.dumps(events).encode("utf-8")).decode("ascii")
    python = _remote_python_block(
        """
import base64, json, os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from src.native_en_text_heads_tracking import record_head_job, transition_head_attempt
events = json.loads(base64.b64decode(os.environ["Q3MS_TERMINAL_B64"]).decode("utf-8"))
attempt_dir = os.environ["Q3MS_ATTEMPT_DIR"]
for event in events:
    record_head_job(
        attempt_dir,
        job_key=event["job_key"],
        job_type=event["job_type"],
        event_type=event["event_type"],
        slurm_job_id=event.get("slurm_job_id"),
        status=event.get("status"),
        exit_code=event.get("exit_code"),
        reason=event.get("reason"),
    )
print("terminal-events-recorded")
"""
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(code_root)}",
            f"export PROJECT_ROOT={shlex.quote(code_root)}",
            f"export Q3MS_ATTEMPT_DIR={shlex.quote(attempt_dir)}",
            f"export Q3MS_TERMINAL_B64={shlex.quote(payload)}",
            python.rstrip("\n"),
        ]
    )
    from src.experiment_tracking.submit import SshSubmitRunner

    proc = SshSubmitRunner(host=host).run_script(script, timeout=300)
    if proc.returncode != 0:
        raise DispatchError(f"terminal event recording failed: {proc.stderr.strip()}")


def command_status(args: argparse.Namespace) -> int:
    registry = _read_registry(Path(args.registry))
    deployment = _load_deployment(args.deployment_id) if args.deployment_id else _load_deployment(None)
    code_root = str(deployment["deployed_code_path"])
    host = args.scheduler_host or DEFAULT_SCHEDULER_HOST
    job_ids: list[str] = []
    for entry in registry:
        for key in ("extract_job_id", "classifier_job_id"):
            if entry.get(key):
                job_ids.append(str(entry[key]))
    states = _scheduler_states(sorted(set(job_ids)), host)
    report: dict[str, Any] = {
        "schema_version": "audiollm.qwen3_multiseed_head_status.v1",
        "checked_at_utc": _now(),
        "attempts": [],
    }
    failures: list[str] = []
    for entry in registry:
        attempt: dict[str, Any] = {
            "registry_key": entry["registry_key"],
            "attempt_id": entry["attempt_id"],
            "jobs": {},
        }
        events: list[dict[str, str]] = []
        terminal = True
        for key, job_type in (("extract_job_id", "hidden_extraction"), ("classifier_job_id", "hidden_classifier")):
            job_id = entry.get(key)
            state = states.get(str(job_id)) if job_id else None
            attempt["jobs"][key] = state
            if state is None:
                terminal = False
                continue
            raw = state["state"]
            if raw not in TERMINAL_STATES:
                terminal = False
                continue
            event_type, status = TERMINAL_STATES[raw]
            events.append(
                {
                    "job_key": "extract" if key.startswith("extract") else "classifier",
                    "job_type": job_type,
                    "event_type": event_type,
                    "status": status,
                    "exit_code": state.get("exit_code"),
                    "slurm_job_id": state.get("job_id"),
                    "reason": f"sacct {raw}",
                }
            )
            if event_type != "COMPLETED" or not str(state.get("exit_code", "")).startswith("0:0"):
                failures.append(f"{entry['registry_key']}: {key} {raw} {state.get('exit_code')}")
        attempt["terminal"] = terminal
        if args.execute and events:
            _remote_record_terminal(
                code_root=code_root,
                attempt_dir=entry["remote_attempt_dir"],
                events=events,
                host=host,
            )
            attempt["recorded_terminal_events"] = len(events)
        report["attempts"].append(attempt)
    report["summary"] = {
        "attempts": len(registry),
        "terminal": sum(1 for attempt in report["attempts"] if attempt["terminal"]),
        "failures": failures,
    }
    output = Path(args.output)
    _write_json(output, report)
    print(f"wrote {output}")
    print("status summary:", json.dumps(report["summary"], sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# collect / validate / finish
# ---------------------------------------------------------------------------


def _remote_manifest(runner: RemoteRunner, remote_dir: str) -> dict[str, str]:
    script = "\n".join(
        [
            "set -uo pipefail",
            f"find {shlex.quote(remote_dir)} -type f -not -name '*.tmp' -print0 "
            "| xargs -0 -r sha256sum",
        ]
    )
    proc = runner.run(script)
    if proc.returncode != 0:
        raise DispatchError(f"remote manifest failed: {proc.stderr.strip()}")
    manifest: dict[str, str] = {}
    prefix = remote_dir.rstrip("/") + "/"
    for line in proc.stdout.splitlines():
        if "  " not in line:
            continue
        digest, _, path = line.partition("  ")
        if path.startswith(prefix):
            manifest[path[len(prefix):]] = digest.strip()
    return manifest


def command_collect(args: argparse.Namespace) -> int:
    registry = _read_registry(Path(args.registry))
    selected = [entry for entry in registry if not args.attempt_id or entry["attempt_id"] in set(args.attempt_id)]
    if not selected:
        print("no registry entries selected")
        return 0
    runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    results = []
    for entry in selected:
        remote_dir = str(entry["remote_attempt_dir"])
        local_dir = Path(entry["local_mirror"])
        manifest = _remote_manifest(runner, remote_dir)
        argv = [
            "rsync", "-avh", "--itemize-changes", "--partial",
            "--exclude=*.tmp", "--exclude=__pycache__",
        ]
        if args.dry_run:
            argv.append("-n")
        argv.append(f"{DEFAULT_TRANSFER_HOST}:{remote_dir.rstrip('/')}/")
        argv.append(str(local_dir) + "/")
        local_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DispatchError(f"rsync failed for {entry['attempt_id']}: {proc.stderr.strip()}")
        mismatches = []
        missing = []
        for relative, digest in manifest.items():
            local_path = local_dir / relative
            if not local_path.is_file():
                missing.append(relative)
            elif sha256_file(local_path) != digest:
                mismatches.append(relative)
        result = {
            "attempt_id": entry["attempt_id"],
            "remote_files": len(manifest),
            "missing": missing,
            "mismatches": mismatches,
            "dry_run": bool(args.dry_run),
        }
        results.append(result)
        print(
            f"{entry['registry_key']}: files={len(manifest)} missing={len(missing)} "
            f"mismatches={len(mismatches)}"
        )
        if missing or mismatches:
            raise DispatchError(
                f"collection hash verification failed for {entry['attempt_id']}: "
                f"missing={missing} mismatches={mismatches}"
            )
    _write_json(LANE_EVIDENCE / "head_collection.json", {"schema_version": "audiollm.qwen3_multiseed_head_collection.v1", "results": results})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    registry = _read_registry(Path(args.registry))
    selected = [entry for entry in registry if not args.attempt_id or entry["attempt_id"] in set(args.attempt_id)]
    failures = 0
    for entry in selected:
        from src.native_en_text_heads_tracking import HeadTrackingError, validate_head_attempt

        try:
            result = validate_head_attempt(entry["local_mirror"])
        except (HeadTrackingError, OSError, ValueError) as exc:
            print(f"{entry['registry_key']}: VALIDATE FAILED {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{entry['registry_key']}: {result}")
        if not result.get("ok"):
            failures += 1
    return 1 if failures else 0


def command_finish(args: argparse.Namespace) -> int:
    registry = _read_registry(Path(args.registry))
    selected = [entry for entry in registry if not args.attempt_id or entry["attempt_id"] in set(args.attempt_id)]
    failures = 0
    for entry in selected:
        from src.native_en_text_heads_tracking import HeadTrackingError, finish_head_attempt

        try:
            result = finish_head_attempt(entry["local_mirror"])
        except (HeadTrackingError, OSError, ValueError) as exc:
            print(f"{entry['registry_key']}: FINISH FAILED {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{entry['registry_key']}: {result}")
        if not result.get("ok"):
            failures += 1
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def command_coverage(args: argparse.Namespace) -> int:
    plan = _load_json(args.plan)
    registry = _read_registry(Path(args.registry))
    by_key = {entry["registry_key"]: entry for entry in registry}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for route in plan.get("routes") or []:
        for job in route.get("jobs") or []:
            key = _key(route["route_id"], int(job["seed"]), int(job["fold"]))
            entry = by_key.get(key)
            state = None
            if entry is not None:
                status_path = Path(entry["local_mirror"]) / "status.json"
                if status_path.is_file():
                    state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
            if job["parent_status"] != "resolved":
                status = job["parent_status"]
            elif entry is None:
                status = "parent_resolved_head_not_submitted"
            elif entry.get("error"):
                status = "head_submit_error"
            elif state == "REPORTABLE":
                status = "reportable"
            else:
                status = f"head_{state or 'submitted'}"
            counts[status] = counts.get(status, 0) + 1
            rows.append(
                {
                    "route_id": route["route_id"],
                    "dataset": route["dataset"],
                    "modality": route["modality"],
                    "parent_training_seed": int(job["seed"]),
                    "head_seed": int(job["head_seed"]),
                    "fold": int(job["fold"]),
                    "parent_status": job["parent_status"],
                    "parent_attempt_id": (job.get("parent") or {}).get("attempt_id"),
                    "head_attempt_id": (entry or {}).get("attempt_id"),
                    "head_state": state,
                    "coverage_status": status,
                }
            )
    payload = {
        "schema_version": "audiollm.qwen3_multiseed_head_coverage.v1",
        "created_at_utc": _now(),
        "counts": counts,
        "rows": rows,
    }
    output = Path(args.output)
    _write_json(output, payload)
    print(f"wrote {output}")
    print("coverage:", json.dumps(counts, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="filter a validated planner matrix into a dispatch plan")
    plan_parser.add_argument("--matrix", type=Path, required=True)
    plan_parser.add_argument("--language", default="native")
    plan_parser.add_argument("--seed", action="append", type=int, default=None)
    plan_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    plan_parser.set_defaults(func=command_plan)

    submit_parser = sub.add_parser("submit", help="submit head extract+classifier chains (dry-run first)")
    submit_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    submit_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    submit_parser.add_argument("--deployment-id", default=None)
    submit_parser.add_argument("--seed", action="append", type=int, default=None)
    submit_parser.add_argument("--route", action="append", default=None)
    submit_parser.add_argument("--key", action="append", default=None)
    submit_parser.add_argument("--limit", type=int, default=None)
    submit_parser.add_argument("--scheduler-host", default=None)
    submit_parser.add_argument("--qwen-hidden-deps", default=None)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--execute", action="store_true")
    submit_parser.set_defaults(func=command_submit)

    status_parser = sub.add_parser("status", help="reconcile recorded jobs against squeue/sacct")
    status_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    status_parser.add_argument("--deployment-id", default=None)
    status_parser.add_argument("--scheduler-host", default=None)
    status_parser.add_argument("--output", type=Path, default=LANE_EVIDENCE / "head_status.json")
    status_parser.add_argument("--execute", action="store_true", help="record terminal events on MN5")
    status_parser.set_defaults(func=command_status)

    collect_parser = sub.add_parser("collect", help="collect compact attempt evidence (dry-run first)")
    collect_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    collect_parser.add_argument("--attempt-id", action="append", default=None)
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.set_defaults(func=command_collect)

    validate_parser = sub.add_parser("validate", help="official head lifecycle validation on local mirrors")
    validate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    validate_parser.add_argument("--attempt-id", action="append", default=None)
    validate_parser.set_defaults(func=command_validate)

    finish_parser = sub.add_parser("finish", help="advance validated head attempts to REPORTABLE")
    finish_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    finish_parser.add_argument("--attempt-id", action="append", default=None)
    finish_parser.set_defaults(func=command_finish)

    coverage_parser = sub.add_parser("coverage", help="audit every planned key against the registry")
    coverage_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    coverage_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    coverage_parser.add_argument("--output", type=Path, default=LANE_EVIDENCE / "head_coverage.json")
    coverage_parser.set_defaults(func=command_coverage)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except DispatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
