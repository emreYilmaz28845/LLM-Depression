"""Submission planning and execution for isolated lane experiments.

Builds one lossless common override representation (a JSON token array,
transported base64-encoded so shell/sbatch quoting cannot mangle it), resolves
the full writable-path contract, validates production qualifiers, and drives
the remote submission wrapper over SSH.
"""
from __future__ import annotations

import base64
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from src.experiment_tracking.identity import new_attempt_id, validate_attempt_id

DEFAULT_SCHEDULER_HOST = "ozu647717@alogin2.bsc.es"
REMOTE_PROJECT_BASE = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression")
REMOTE_RUNTIME_BASE = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime")

TRAIN_JOB_KEY = "train"
EVAL_JOB_KEY = "best_eval"


class SubmissionError(RuntimeError):
    """Raised when a submission contract is invalid or submission must fail."""


def encode_overrides(tokens: list[str]) -> str:
    return base64.b64encode(json.dumps(tokens).encode("utf-8")).decode("ascii")


def decode_overrides(payload: str) -> list[str]:
    return json.loads(base64.b64decode(payload).decode("utf-8"))


def build_common_overrides(
    *,
    manifest_dir: str,
    split_dir: str,
    run_root: str,
    extra_overrides: list[str],
) -> list[str]:
    """One common override token array shared by train, eval, manifests, sidecars."""
    for token in extra_overrides:
        if not isinstance(token, str) or not token:
            raise SubmissionError(f"override tokens must be non-empty strings: {token!r}")
    overrides = [
        f"--set=output_dirs.manifest_dir={manifest_dir}",
        f"--set=output_dirs.split_dir={split_dir}",
        f"--set=output_dirs.run_root={run_root}",
    ]
    overrides.extend(extra_overrides)
    return overrides


def _validate_override_token(token: str) -> tuple[str, str]:
    if not token.startswith("--set"):
        raise SubmissionError(f"only --set overrides are accepted, got: {token!r}")
    body = token[len("--set"):]
    if body.startswith("="):
        body = body[1:]
    elif body == "":
        raise SubmissionError("bare --set without key=value")
    else:
        raise SubmissionError(f"malformed --set token: {token!r}")
    if "=" not in body:
        raise SubmissionError(f"--set requires key=value, got: {token!r}")
    key, _, value = body.partition("=")
    return key, value


def resolve_contract(
    *,
    experiment_id: str,
    deployment: dict[str, Any],
    config_path_remote: str,
    config_dict: dict[str, Any],
    fold: int,
    seed: int | None,
    run_name: str,
    campaign: str,
    modality: str,
    dataset: str,
    extra_overrides: list[str],
    scheduler_host: str = DEFAULT_SCHEDULER_HOST,
    supersedes_attempt_id: str | None = None,
    group_id: str | None = None,
    github_issue: str | None = None,
    github_pr: str | None = None,
    attempt_id: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the complete submission contract without touching the network."""
    if dataset != config_dict.get("dataset"):
        raise SubmissionError(
            f"dataset qualifier {dataset!r} does not match resolved config dataset {config_dict.get('dataset')!r}"
        )
    runtime_root = REMOTE_RUNTIME_BASE / experiment_id
    permanent_output_base = REMOTE_PROJECT_BASE / "output_model"
    run_root = str(permanent_output_base / campaign / modality / dataset)
    manifest_dir = str(runtime_root / "manifests" / dataset)
    split_dir = str(runtime_root / "splits" / dataset)

    normalized_extra_overrides = list(extra_overrides)
    top_level_seed = None
    for token in normalized_extra_overrides:
        key, value = _validate_override_token(token)
        if key == "seed":
            top_level_seed = value
    if seed is None and top_level_seed is not None:
        raise SubmissionError("top-level seed override requires matching --seed for provenance")
    if seed is not None:
        if top_level_seed is not None and top_level_seed != str(seed):
            raise SubmissionError(
                f"--seed {seed} conflicts with top-level seed override {top_level_seed!r}"
            )
        if top_level_seed is None:
            normalized_extra_overrides.append(f"--set=seed={seed}")

    overrides = build_common_overrides(
        manifest_dir=manifest_dir,
        split_dir=split_dir,
        run_root=run_root,
        extra_overrides=normalized_extra_overrides,
    )
    for token in overrides[3:]:
        _validate_override_token(token)

    evaluation = config_dict.get("evaluation", {}) or {}
    evaluation_view = evaluation.get("evaluation_view")
    backend = evaluation.get("sample_prediction_mode")
    aggregation = evaluation.get("aggregation_level", "subject")
    if not evaluation_view:
        raise SubmissionError(
            "evaluation.evaluation_view is missing after overrides; production submission fails closed"
        )
    if not backend:
        raise SubmissionError("evaluation.sample_prediction_mode missing from resolved config")

    fold_dir = f"{run_root}/{run_name}/fold_{fold}"
    local_fold_rel = f"output_model/{campaign}/{modality}/{dataset}/{run_name}/fold_{fold}"
    checkpoint_dir = f"{fold_dir}/best_model"
    standalone_eval_dir = f"{checkpoint_dir}/standalone_eval"
    log_root_train = str(runtime_root / "logs" / "slurm_train" / dataset)
    log_root_eval = str(runtime_root / "logs" / "slurm_eval" / dataset)

    commit = deployment["git_commit"]
    resolved_attempt_id = attempt_id or new_attempt_id(run_name, commit)
    if not validate_attempt_id(resolved_attempt_id):
        raise SubmissionError(f"generated attempt id is invalid: {resolved_attempt_id}")
    context_path = str(runtime_root / "contexts" / resolved_attempt_id / f"fold_{fold}" / "context.json")

    context = {
        "schema_version": "audiollm.tracking_context.v1",
        "group_id": group_id or experiment_id,
        "logical_run_name": run_name,
        "attempt_id": resolved_attempt_id,
        "fold": fold,
        "seed": seed,
        "source": {
            "git_commit": commit,
            "git_branch": deployment.get("git_branch_at_deploy"),
            "git_dirty": deployment.get("git_dirty", False),
            "deployed_source_sha256": deployment.get("source_manifest_sha256"),
            "deployment_id": deployment.get("deployment_id"),
        },
        "research": {
            "github_issue": github_issue,
            "github_pr": github_pr,
        },
    }
    if supersedes_attempt_id:
        context["supersedes_attempt_id"] = supersedes_attempt_id

    return {
        "experiment_id": experiment_id,
        "deployment_id": deployment["deployment_id"],
        "deployed_code_path": deployment["deployed_code_path"],
        "scheduler_host": scheduler_host,
        "config_path_remote": config_path_remote,
        "fold": fold,
        "seed": seed,
        "run_name": run_name,
        "campaign": campaign,
        "modality": modality,
        "dataset": dataset,
        "attempt_id": resolved_attempt_id,
        "supersedes_attempt_id": supersedes_attempt_id,
        "context_path": context_path,
        "context": context,
        "runtime_root": str(runtime_root),
        "manifest_dir": manifest_dir,
        "split_dir": split_dir,
        "run_root": run_root,
        "fold_dir": fold_dir,
        "local_fold_rel": local_fold_rel,
        "checkpoint_dir": checkpoint_dir,
        "standalone_eval_dir": standalone_eval_dir,
        "log_root_train": log_root_train,
        "log_root_eval": log_root_eval,
        "overrides": overrides,
        "overrides_b64": encode_overrides(overrides),
        "env_exports": dict(extra_env or {}),
        "qualifiers": {
            "evaluation_view": evaluation_view,
            "backend": backend,
            "aggregation": aggregation,
            "namespace": "headline/binary_strict",
            "checkpoint_role": "best_model",
        },
        "job_graph": [
            {
                "job_key": TRAIN_JOB_KEY,
                "job_type": "train",
                "shape": "1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4 (DDP)",
                "depends_on": [],
                "script": "scripts/run_train_slurm.sh",
            },
            {
                "job_key": EVAL_JOB_KEY,
                "job_type": "evaluation",
                "shape": "1 node, 1 task, 1 H100",
                "depends_on": [TRAIN_JOB_KEY],
                "script": "scripts/run_eval_slurm.sh",
                "checkpoint_dir": checkpoint_dir,
                "output_dir": standalone_eval_dir,
            },
        ],
    }


def check_collisions(contract: dict[str, Any], remote_exists: Callable[[str], bool]) -> None:
    """Fail before sbatch when any target already exists."""
    conflicts = []
    if remote_exists(contract["fold_dir"]):
        conflicts.append(f"fold dir exists: {contract['fold_dir']}")
    if remote_exists(contract["context_path"]):
        conflicts.append(f"context exists (attempt id reuse): {contract['context_path']}")
    if remote_exists(contract["standalone_eval_dir"]):
        conflicts.append(f"standalone eval dir exists: {contract['standalone_eval_dir']}")
    if conflicts:
        raise SubmissionError("submission collisions: " + "; ".join(conflicts))


def build_remote_submit_script(contract: dict[str, Any]) -> str:
    """Shell script executed on the scheduler login; every value shlex-quoted."""
    q = shlex.quote
    train_args = " ".join(contract["overrides"])
    eval_args = " ".join(contract["overrides"])
    lines = [
        "set -euo pipefail",
        "export PYTHONDONTWRITEBYTECODE=1",
    ]
    for key, value in sorted((contract.get("env_exports") or {}).items()):
        lines.append(f"export {key}={q(str(value))}")
    lines += [
        f"cd {q(contract['deployed_code_path'])}",
        f"export PROJECT_ROOT={q(contract['deployed_code_path'])}",
        f"export CONFIG={q(contract['config_path_remote'])}",
        f"export FOLD={contract['fold']}",
        f"export RUN_NAME={q(contract['run_name'])}",
        f"export EXTRA_TRAIN_ARGS={q(train_args)}",
        f"export EXTRA_EVAL_ARGS={q(eval_args)}",
        f"export OVERRIDES_JSON_B64={q(contract['overrides_b64'])}",
        f"export LOG_ROOT={q(contract['log_root_train'])}",
        f"export EXPERIMENT_CONTEXT={q(contract['context_path'])}",
        "bash scripts/submit_train_and_eval.sh",
    ]
    return "\n".join(lines) + "\n"


class SshSubmitRunner:
    """Runs the submit script on the scheduler login over SSH."""

    def __init__(
        self,
        host: str = DEFAULT_SCHEDULER_HOST,
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.host = host
        self._runner = runner

    def run_script(self, script: str, timeout: int = 600) -> subprocess.CompletedProcess:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.host, "bash -s"]
        if self._runner is not None:
            return self._runner(argv)
        return subprocess.run(argv, input=script, capture_output=True, text=True, timeout=timeout)


def parse_submitted_job_ids(output: str) -> dict[str, str]:
    ids: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Submitted training job:"):
            ids[TRAIN_JOB_KEY] = line.split(":", 1)[1].strip()
        elif line.startswith("Submitted best-checkpoint eval job:"):
            ids[EVAL_JOB_KEY] = line.split(":", 1)[1].strip()
    return ids


def require_complete_job_ids(
    job_ids: dict[str, str], job_graph: list[dict[str, Any]]
) -> dict[str, str]:
    expected = {str(job["job_key"]) for job in job_graph}
    missing = sorted(expected - set(job_ids))
    invalid = sorted(key for key in expected if key in job_ids and not str(job_ids[key]).isdigit())
    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing job IDs for {missing}")
        if invalid:
            details.append(f"non-numeric job IDs for {invalid}")
        raise SubmissionError("incomplete submission output: " + "; ".join(details))
    return {key: str(job_ids[key]) for key in sorted(expected)}
