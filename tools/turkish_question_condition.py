#!/usr/bin/env python3
"""Managed orchestration for the Turkish mixed/negative-only campaign."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.identity import new_attempt_id
from src.experiment_tracking.submit import encode_overrides
from src.turkish_question_condition import (
    EVALUATION_BACKEND,
    EVALUATION_VIEW,
    EXPERIMENT_ID,
    GROUP_ID,
    METRIC_NAMESPACE,
    REMOTE_OUTPUT_ROOT,
    REMOTE_PROJECT_ROOT,
    REMOTE_RUNTIME_ROOT,
    build_plan,
    load_cells,
    write_plan,
)


DEFAULT_SLURM_HOST = "ozu647717@alogin2.bsc.es"
TRANSFER_HOST = "ozu647717@transfer1.bsc.es"
QWEN_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
GEMMA_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate"
GEMMA_MODEL_PATH = "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SUBMISSION_SCHEMA = "audiollm.turkish_question_condition_submission.v1"
LOCAL_ROOT = PROJECT_ROOT / "outputs" / "turkish_question_condition" / EXPERIMENT_ID


class CampaignError(RuntimeError):
    """Raised when the locked campaign cannot proceed safely."""


def q(value: Any) -> str:
    return shlex.quote(str(value))


def ssh_bash(host: str, script: str, *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_cat(host: str, path: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "cat", path],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise CampaignError(f"could not read {path} on {host}: {result.stderr.strip()}")
    return result.stdout


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _lane_and_deployment(slug: str, deployment_id: str | None, *, execute: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    import tools.exp as exp

    worktree, pin = exp._resolve_lane(slug)
    if worktree is None or pin is None:
        raise CampaignError(f"managed lane not found: {slug}")
    ok, message = exp._check_pin(worktree)
    if not ok:
        raise CampaignError(f"worktree pin failed: {message}")
    if pin.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError(f"lane experiment id mismatch: {pin.get('experiment_id')!r}")
    if pin.get("parent_sha") != "e176da5e0595464bc44320d32e04f7fe0a7adf5e":
        raise CampaignError(f"lane parent SHA changed: {pin.get('parent_sha')}")
    group = exp._load_linked_experiment_group(worktree, pin)
    if group.get("group_id") != GROUP_ID:
        raise CampaignError("linked group is not the locked campaign group")
    found = exp._find_deployment_record(EXPERIMENT_ID, deployment_id, allow_plan=not execute)
    if not isinstance(found, tuple) or len(found) != 2:
        raise CampaignError(str(found))
    _, deployment = found
    if deployment.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("deployment experiment id does not match the lane")
    if execute and deployment.get("git_dirty"):
        raise CampaignError("production execution requires a clean immutable deployment")
    return worktree, pin, deployment


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise CampaignError(f"refusing to overwrite incompatible evidence: {path}")
    path.write_text(text, encoding="utf-8")


def _source(deployment: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_commit": deployment.get("git_commit"),
        "git_branch": deployment.get("git_branch_at_deploy"),
        "git_dirty": bool(deployment.get("git_dirty", False)),
        "deployed_source_sha256": deployment.get("source_manifest_sha256"),
        "deployment_id": deployment.get("deployment_id"),
    }


def _preflight_pairs(audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if audit.get("status") != "passed":
        raise CampaignError(f"preflight is not passed: {audit.get('failures')}")
    if audit.get("group_id") != GROUP_ID:
        raise CampaignError("preflight group id mismatch")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in audit.get("pairs") or []:
        key = (str(pair.get("condition")), str(pair.get("language")))
        if key in result:
            raise CampaignError(f"duplicate preflight pair: {key}")
        result[key] = pair
    required = {(c, l) for c in ("mixed", "negative_only") for l in ("native", "english")}
    if set(result) != required:
        raise CampaignError(f"preflight pair set mismatch: {sorted(result)}")
    return result


def _hashes(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": pair.get("metadata_manifest_hash"),
        "split_sha256": pair.get("metadata_sha256"),
        "fold_hash": (pair.get("folds") or {}).get("fold_hash"),
    }


def _qualifiers() -> dict[str, str]:
    return {
        "evaluation_view": EVALUATION_VIEW,
        "evaluation_backend": EVALUATION_BACKEND,
        "metric_namespace": METRIC_NAMESPACE,
        "aggregation": "subject_level",
        "checkpoint_role": "best_model",
    }


def _head_context(*, attempt_id: str, logical: str, fold_job: dict[str, Any], deployment: dict[str, Any], hashes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "audiollm.tracking_context.v1",
        "group_id": GROUP_ID,
        "logical_run_name": logical,
        "attempt_id": attempt_id,
        "fold": int(fold_job["fold"]),
        "seed": int(fold_job["seed"]),
        "source": _source(deployment),
        "hashes": hashes,
        "tracking_kind": "turkish_question_condition_v1_head",
        "required_jobs": ["head"],
        "qualifiers": _qualifiers(),
        "research": {},
    }


def _head_config(*, fold_job: dict[str, Any], method: str, backend: str, trials: int) -> dict[str, Any]:
    language = str(fold_job["transcript_condition"])
    recording = "negative_only" if fold_job["recording_condition"] == "negative_only" else "mixed"
    return {
        "schema_version": "audiollm.turkish_question_condition_head_config.v1",
        "dataset": "turkish",
        "dataset_variant": "negative_only_t17" if recording == "negative_only" else "mixed_t17",
        "recording_condition": recording,
        "transcript_condition": language,
        "modality": fold_job["modality"],
        "backbone": fold_job["backbone"],
        "model_backend": "gemma4" if fold_job["backbone"] == "gemma4" else "qwen",
        "seed": int(fold_job["seed"]),
        "fold": int(fold_job["fold"]),
        "stage": fold_job["stage"],
        "evaluation": {
            "evaluation_view": EVALUATION_VIEW,
            "sample_prediction_mode": EVALUATION_BACKEND,
            "aggregation": "subject_level",
            "split_name": "outer_holdout",
            "split_protocol": "saved_split",
        },
        "classifier": {
            "method": method,
            "prediction_backend": backend,
            "head_seed": 1337,
            "protocol": "turkish_question_condition_v1",
            "optuna_trials": int(trials),
            "sampling_mode": "none",
        },
        "qualifiers": _qualifiers(),
    }


def _make_submission_plan(*, matrix: dict[str, Any], deployment: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    pairs = _preflight_pairs(preflight)
    cells = {cell.cell_id: cell for cell in load_cells(PROJECT_ROOT)}
    train_jobs = [job for job in matrix["jobs"] if job["job_type"] == "train"]
    source_sha = str(deployment["git_commit"])
    backbones: list[dict[str, Any]] = []
    head_jobs: list[dict[str, Any]] = []
    used_attempts: set[str] = set()
    for index, train_job in enumerate(train_jobs):
        cell = cells[str(train_job["cell_id"])]
        language = "native" if cell.transcript_condition == "not_applicable" else cell.transcript_condition
        condition = "negative_only" if cell.recording_condition == "negative_only" else "mixed"
        hashes = _hashes(pairs[(condition, language)])
        logical = str(train_job["run_name"])
        backbone_id = new_attempt_id(logical, source_sha)
        logreg_id = new_attempt_id(f"{logical}_logreg", source_sha)
        xgb_id = new_attempt_id(f"{logical}_xgb_optuna100", source_sha)
        for attempt_id in (backbone_id, logreg_id, xgb_id):
            if attempt_id in used_attempts:
                raise CampaignError(f"duplicate generated attempt id: {attempt_id}")
            used_attempts.add(attempt_id)
        fold_dir = Path(str(train_job["fold_dir"]))
        config_remote = str(Path(str(deployment["deployed_code_path"])) / str(train_job["config"]))
        backbone_context = {
            "schema_version": "audiollm.tracking_context.v1",
            "group_id": GROUP_ID,
            "logical_run_name": logical,
            "attempt_id": backbone_id,
            "fold": int(train_job["fold"]),
            "seed": int(train_job["seed"]),
            "source": _source(deployment),
            "hashes": hashes,
            "qualifiers": _qualifiers(),
            "research": {},
        }
        base = {
            "index": index,
            "cell_id": train_job["cell_id"],
            "recording_condition": train_job["recording_condition"],
            "transcript_condition": train_job["transcript_condition"],
            "modality": train_job["modality"],
            "backbone": train_job["backbone"],
            "seed": int(train_job["seed"]),
            "fold": int(train_job["fold"]),
            "stage": train_job["stage"],
            "run_name": logical,
            "campaign": train_job["campaign"],
            "run_root": train_job["run_root"],
            "fold_dir": str(fold_dir),
            "config": train_job["config"],
            "config_remote": config_remote,
            "overrides": list(train_job["overrides"]),
            "overrides_b64": encode_overrides(list(train_job["overrides"])),
            "manifest_dir": train_job["manifest_dir"],
            "split_dir": train_job["split_dir"],
            "backbone_attempt_id": backbone_id,
            "backbone_context": backbone_context,
            "backbone_context_path": str(REMOTE_RUNTIME_ROOT / "contexts" / backbone_id / f"fold_{train_job['fold']}" / "context.json"),
            "train_job_key": f"{logical}:train",
            "eval_job_key": f"{logical}:best_eval",
            "hashes": hashes,
            "qualifiers": _qualifiers(),
            "job_ids": {},
        }
        backbones.append(base)
        logreg_backend = "gemma4_hidden_logreg_raw" if train_job["backbone"] == "gemma4" else "qwen_hidden_logreg_raw"
        xgb_backend = "gemma4_hidden_xgb_optuna100" if train_job["backbone"] == "gemma4" else "qwen_hidden_xgb_optuna100"
        for method, attempt_id, backend, trials, parent_id, dependency_key, job_type in (
            ("logreg", logreg_id, logreg_backend, 0, backbone_id, base["eval_job_key"], "hidden_extraction"),
            ("xgb_optuna100", xgb_id, xgb_backend, matrix["protocol"]["xgb_completed_trials"], logreg_id, f"{logical}:logreg", "hidden_classifier"),
        ):
            logical_head = f"{logical}_{method}"
            context = _head_context(attempt_id=attempt_id, logical=logical_head, fold_job=train_job, deployment=deployment, hashes=hashes)
            config = _head_config(fold_job=train_job, method=method, backend=backend, trials=trials)
            parent = {"parent_attempt_id": parent_id, "parent_checkpoint_path": str(fold_dir / "best_model")}
            context_base = REMOTE_RUNTIME_ROOT / "contexts" / attempt_id / f"fold_{train_job['fold']}"
            head_jobs.append(
                {
                    "index": len(head_jobs),
                    "backbone_index": index,
                    "job_key": f"{logical}:{method}",
                    "job_type": job_type,
                    "method": method,
                    "attempt_id": attempt_id,
                    "attempt_dir": str(fold_dir / attempt_id),
                    "context": context,
                    "context_path": str(context_base / "context.json"),
                    "config": config,
                    "config_path": str(context_base / "config.json"),
                    "parent": parent,
                    "parent_path": str(context_base / "parent.json"),
                    "parent_attempt_id": parent_id,
                    "checkpoint_dir": str(fold_dir / "best_model"),
                    "cache_dir": str(fold_dir / logreg_id / "hidden_cache"),
                    "condition": language,
                    "backbone": train_job["backbone"],
                    "config_remote": config_remote,
                    "model_path": GEMMA_MODEL_PATH if train_job["backbone"] == "gemma4" else "",
                    "dependency_key": dependency_key,
                    "trials": trials,
                    "stage": train_job["stage"],
                    "fold": int(train_job["fold"]),
                    "seed": int(train_job["seed"]),
                    "recording_condition": train_job["recording_condition"],
                    "log_root": str(REMOTE_RUNTIME_ROOT / "logs" / job_type / train_job["recording_condition"] / train_job["backbone"]),
                    "job_id": None,
                }
            )
        backbones[-1]["logreg_attempt_id"] = logreg_id
        backbones[-1]["xgb_attempt_id"] = xgb_id
    plan: dict[str, Any] = {
        "schema_version": SUBMISSION_SCHEMA,
        "group_id": GROUP_ID,
        "experiment_id": EXPERIMENT_ID,
        "deployment_id": deployment.get("deployment_id"),
        "source_git_sha": deployment.get("git_commit"),
        "source_manifest_sha256": deployment.get("source_manifest_sha256"),
        "stage": matrix["stage"],
        "runtime_root": str(REMOTE_RUNTIME_ROOT),
        "output_root": str(REMOTE_OUTPUT_ROOT),
        "preflight": preflight,
        "matrix_plan_sha256": matrix["plan_sha256"],
        "backbones": backbones,
        "head_jobs": head_jobs,
        "expected_counts": matrix["expected_counts"],
        "submission_state": "PLANNED",
    }
    plan["submission_plan_sha256"] = _canonical_sha(plan)
    return plan


def _write_matrix_plan(args: argparse.Namespace) -> int:
    source_sha = args.source_sha or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    plan = build_plan(stage=args.stage, source_sha=source_sha, deployment_id=args.deployment_id, repo_root=PROJECT_ROOT)
    target = args.output or (LOCAL_ROOT / f"{args.stage}_matrix.json")
    write_plan(plan, target)
    print(json.dumps({"plan": str(target), "plan_sha256": plan["plan_sha256"], "expected_counts": plan["expected_counts"]}, indent=2, sort_keys=True))
    return 0


def _preflight_script(deployment: dict[str, Any]) -> str:
    code = str(deployment["deployed_code_path"])
    audit = REMOTE_RUNTIME_ROOT / "preflight" / "audit.json"
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"cd {q(code)}",
        f"test ! -e {q(REMOTE_RUNTIME_ROOT)} || test -z \"$(find {q(REMOTE_RUNTIME_ROOT)} -type f -print -quit)\"",
        "python scripts/turkish_question_condition_preflight.py"
        f" --stage production --output-root {q(REMOTE_RUNTIME_ROOT)}"
        f" --output {q(audit)} --require-models --require-environment",
    ]
    return "\n".join(lines) + "\n"


def _run_preflight(args: argparse.Namespace) -> int:
    if args.local:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/turkish_question_condition_preflight.py"),
            "--stage", args.stage,
            "--output-root", str(args.output_root),
            "--output", str(args.output),
        ]
        for flag, value in (("--mixed-root", args.mixed_root), ("--negative-root", args.negative_root), ("--translation-root", args.translation_root)):
            if value:
                command.extend((flag, str(value)))
        if args.require_models:
            command.append("--require-models")
        if args.require_environment:
            command.append("--require-environment")
        return subprocess.run(command, cwd=PROJECT_ROOT).returncode
    _, _, deployment = _lane_and_deployment(args.slug, args.deployment_id, execute=args.execute)
    if not args.execute:
        print(_preflight_script(deployment))
        return 0
    result = ssh_bash(args.scheduler_host, _preflight_script(deployment), timeout=8 * 60 * 60)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    audit = json.loads(ssh_cat(args.scheduler_host, str(REMOTE_RUNTIME_ROOT / "preflight/audit.json")))
    if audit.get("status") != "passed":
        raise CampaignError(f"remote preflight did not pass: {audit.get('failures')}")
    print(json.dumps({"status": audit["status"], "audit_sha256": audit["audit_sha256"]}, indent=2, sort_keys=True))
    return 0


def _payload_b64(value: Any) -> str:
    return base64.b64encode((json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")).decode("ascii")


def _remote_write_once() -> str:
    return """write_once() {
  target=$1
  payload=$2
  python - "$target" "$payload" <<'PY'
import base64, pathlib, sys
target = pathlib.Path(sys.argv[1])
data = base64.b64decode(sys.argv[2])
if target.exists():
    if target.is_file() and target.read_bytes() == data:
        raise SystemExit(0)
    raise SystemExit(f"collision or incompatible existing target: {target}")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(data)
PY
}"""


def _head_export(job: dict[str, Any], deployment: dict[str, Any], *, logreg: bool) -> str:
    values = {
        "PROJECT_ROOT": deployment["deployed_code_path"],
        "CONFIG": job["config_remote"],
        "ATTEMPT_DIR": job["attempt_dir"],
        "CONTEXT_JSON": job["context_path"],
        "CONFIG_JSON": job["config_path"],
        "PARENT_JSON": job["parent_path"],
        "CHECKPOINT_DIR": job["checkpoint_dir"],
        "CACHE_DIR": job["cache_dir"],
        "CONDITION": job["condition"],
        "BACKBONE": job["backbone"],
        "MODEL_PATH": job["model_path"],
        "LOG_ROOT": job["log_root"],
    }
    if not logreg:
        values.update({"TRIALS": job["trials"], "STAGE": job["stage"]})
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def _remote_submission_script(plan: dict[str, Any], deployment: dict[str, Any]) -> str:
    code = str(deployment["deployed_code_path"])
    by_backbone = {int(item["index"]): item for item in plan["backbones"]}
    by_head = {int(item["index"]): item for item in plan["head_jobs"]}
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"cd {q(code)}",
        _remote_write_once(),
        f"test -f {q(REMOTE_RUNTIME_ROOT / 'preflight/audit.json')}",
        f"python - {q(str(REMOTE_RUNTIME_ROOT / 'preflight/audit.json'))} <<'PY'",
        "import json, sys",
        "payload=json.load(open(sys.argv[1], encoding='utf-8'))",
        f"assert payload.get('status') == 'passed', payload.get('failures')",
        f"assert payload.get('group_id') == {GROUP_ID!r}",
        "PY",
    ]
    seen_manifests: set[tuple[str, str]] = set()
    for item in plan["backbones"]:
        lines.append(f"test ! -e {q(item['fold_dir'])}")
        key = (str(item["manifest_dir"]), str(item["split_dir"]))
        if key not in seen_manifests:
            seen_manifests.add(key)
            lines.append(f"test -f {q(Path(key[0]) / 'turkish_manifest.jsonl')}")
            lines.append(f"test -f {q(Path(key[1]) / 'turkish_manifest_metadata.json')}")
    for item in plan["backbones"]:
        lines.append(f"write_once {q(item['backbone_context_path'])} {q(_payload_b64(item['backbone_context']))}")
    for item in plan["head_jobs"]:
        lines.append(f"write_once {q(item['context_path'])} {q(_payload_b64(item['context']))}")
        lines.append(f"write_once {q(item['config_path'])} {q(_payload_b64(item['config']))}")
        lines.append(f"write_once {q(item['parent_path'])} {q(_payload_b64(item['parent']))}")
    # Creating the head leaves creates only empty fold ancestry; the training
    # wrapper accepts that state and still refuses an existing run_config.yaml.
    for item in plan["head_jobs"]:
        lines.append(
            "python tools/turkish_question_condition_worker.py init"
            f" --attempt-dir {q(item['attempt_dir'])}"
            f" --context {q(item['context_path'])}"
            f" --config {q(item['config_path'])}"
            f" --parent {q(item['parent_path'])}"
        )
    for index, item in by_backbone.items():
        env_activate = GEMMA_ENV_ACTIVATE if item["backbone"] == "gemma4" else QWEN_ENV_ACTIVATE
        model_path = GEMMA_MODEL_PATH if item["backbone"] == "gemma4" else ""
        lines.append(
            f"out_{index}=$(CONFIG={q(item['config_remote'])} FOLD={int(item['fold'])}"
            f" RUN_NAME={q(item['run_name'])} OVERRIDES_JSON_B64={q(item['overrides_b64'])}"
            f" EXPERIMENT_CONTEXT={q(item['backbone_context_path'])}"
            f" LOG_ROOT={q(str(REMOTE_RUNTIME_ROOT / 'logs/train' / item['recording_condition'] / item['backbone']))}"
            f" ENV_ACTIVATE={q(env_activate)} MODEL_PATH={q(model_path)}"
            f" SKIP_MANIFEST_BUILD=1 PROJECT_ROOT={q(code)} bash scripts/submit_train_and_eval.sh)"
        )
        lines.append(f"train_{index}=$(printf '%s\\n' \"$out_{index}\" | sed -n 's/^Submitted training job: //p' | tail -1)")
        lines.append(f"eval_{index}=$(printf '%s\\n' \"$out_{index}\" | sed -n 's/^Submitted best-checkpoint eval job: //p' | tail -1)")
        lines.append(f"test -n \"$train_{index}\"; test -n \"$eval_{index}\"")
        lines.append(f"echo '__BACKBONE__ {index}' \"$train_{index}\" \"$eval_{index}\"")
        logreg = by_head[2 * index]
        xgb = by_head[2 * index + 1]
        lines.append(
            f"logreg_{index}=$({q('sbatch')} --parsable --chdir={q(code)}"
            f" --dependency=afterok:$eval_{index} --export={q(_head_export(logreg, deployment, logreg=True))}"
            " scripts/run_turkish_question_logreg_slurm.sh)"
        )
        lines.append(f"logreg_{index}=$(printf '%s\\n' \"$logreg_{index}\" | sed 's/;.*//'); test -n \"$logreg_{index}\"")
        lines.append(
            "python tools/turkish_question_condition_worker.py record"
            f" --attempt-dir {q(logreg['attempt_dir'])} --job-key head"
            " --job-type hidden_extraction --event-type SUBMITTED"
            f" --slurm-job-id \"$logreg_{index}\" --status PENDING"
            f" --dependency-job-id \"$eval_{index}\""
        )
        lines.append(
            "python tools/turkish_question_condition_worker.py transition"
            f" --attempt-dir {q(logreg['attempt_dir'])} --to-state SUBMITTED"
            f" --reason {q('Turkish question-condition LogReg submitted')}"
        )
        lines.append(f"echo '__HEAD__ {2 * index}' \"$logreg_{index}\"")
        lines.append(
            f"xgb_{index}=$({q('sbatch')} --parsable --chdir={q(code)}"
            f" --dependency=afterok:$logreg_{index} --export={q(_head_export(xgb, deployment, logreg=False))}"
            " scripts/run_turkish_question_xgb_slurm.sh)"
        )
        lines.append(f"xgb_{index}=$(printf '%s\\n' \"$xgb_{index}\" | sed 's/;.*//'); test -n \"$xgb_{index}\"")
        lines.append(
            "python tools/turkish_question_condition_worker.py record"
            f" --attempt-dir {q(xgb['attempt_dir'])} --job-key head"
            " --job-type hidden_classifier --event-type SUBMITTED"
            f" --slurm-job-id \"$xgb_{index}\" --status PENDING"
            f" --dependency-job-id \"$logreg_{index}\""
        )
        lines.append(
            "python tools/turkish_question_condition_worker.py transition"
            f" --attempt-dir {q(xgb['attempt_dir'])} --to-state SUBMITTED"
            f" --reason {q('Turkish question-condition Optuna submitted')}"
        )
        lines.append(f"echo '__HEAD__ {2 * index + 1}' \"$xgb_{index}\"")
    lines.append(f"echo '__SUBMISSION_COMPLETE__ {len(by_backbone)} {len(by_head)}'")
    return "\n".join(lines) + "\n"


def _parse_submission_markers(plan: dict[str, Any], output: str) -> None:
    backbones = {int(item["index"]): item for item in plan["backbones"]}
    heads = {int(item["index"]): item for item in plan["head_jobs"]}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == "__BACKBONE__":
            index = int(fields[1])
            if index not in backbones or backbones[index].get("job_ids"):
                raise CampaignError(f"duplicate or unknown backbone marker: {line}")
            backbones[index]["job_ids"] = {"train": fields[2], "best_eval": fields[3]}
        elif len(fields) == 3 and fields[0] == "__HEAD__":
            index = int(fields[1])
            if index not in heads or heads[index].get("job_id"):
                raise CampaignError(f"duplicate or unknown head marker: {line}")
            heads[index]["job_id"] = fields[2]
    missing_backbones = sorted(index for index, item in backbones.items() if not item.get("job_ids"))
    missing_heads = sorted(index for index, item in heads.items() if not item.get("job_id"))
    if missing_backbones or missing_heads:
        raise CampaignError(
            f"submission output is missing markers: backbones={missing_backbones[:10]} heads={missing_heads[:10]}"
        )


def _submit(args: argparse.Namespace) -> int:
    _, _, deployment = _lane_and_deployment(args.slug, args.deployment_id, execute=args.execute)
    matrix = build_plan(
        stage=args.stage,
        source_sha=str(deployment["git_commit"]),
        deployment_id=deployment.get("deployment_id"),
        repo_root=PROJECT_ROOT,
    )
    if args.execute:
        preflight = json.loads(ssh_cat(args.scheduler_host, str(REMOTE_RUNTIME_ROOT / "preflight/audit.json")))
    elif args.preflight:
        preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    else:
        preflight = {
            "status": "passed",
            "group_id": GROUP_ID,
            "pairs": [],
        }
        raise CampaignError("--preflight is required for a dry-run submission")
    submission = _make_submission_plan(matrix=matrix, deployment=deployment, preflight=preflight)
    print(json.dumps({
        "stage": args.stage,
        "deployment_id": deployment.get("deployment_id"),
        "expected_counts": submission["expected_counts"],
        "submission_plan_sha256": submission["submission_plan_sha256"],
    }, indent=2, sort_keys=True))
    script = _remote_submission_script(submission, deployment)
    if not args.execute:
        print(script)
        return 0
    result = ssh_bash(args.scheduler_host, script, timeout=8 * 60 * 60)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise CampaignError("remote Slurm submission failed; no local submission plan was recorded")
    _parse_submission_markers(submission, result.stdout)
    submission["submission_state"] = "SUBMITTED"
    submission["submission_stdout_sha256"] = _canonical_sha(result.stdout)
    submission.pop("submission_plan_sha256", None)
    submission["submission_plan_sha256"] = _canonical_sha(submission)
    target = LOCAL_ROOT / f"{args.stage}_submission.json"
    _write_json_once(target, submission)
    print(json.dumps({"submission": str(target), "state": submission["submission_state"], "submission_plan_sha256": submission["submission_plan_sha256"]}, indent=2, sort_keys=True))
    return 0


def _sacct(host: str, job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host,
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
            "--format=JobIDRaw,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise CampaignError(f"sacct failed on {host}: {result.stderr.strip()}")
    records: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        records[fields[0].split(".", 1)[0]] = {"state": fields[1], "exit_code": fields[2]}
    return records


def _terminal_update_script(terminal: list[dict[str, Any]], deployment: dict[str, Any]) -> str:
    code = str(deployment["deployed_code_path"])
    payload = _payload_b64(terminal)
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"cd {q(code)}",
        f"python - {q(payload)} <<'PY'",
        "import base64, json, pathlib, sys",
        "from src.experiment_tracking import lifecycle",
        "items = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))",
        "for item in items:",
        "    target = pathlib.Path(item['attempt_dir'])",
        "    jobs_path = target / 'jobs.jsonl'",
        "    if not jobs_path.is_file(): continue",
        "    events = lifecycle.read_job_events(jobs_path)",
        "    jid = str(item['slurm_job_id'])",
        "    if not any(str(event.get('slurm_job_id')) == jid and event.get('event_type') in {'COMPLETED','FAILED','CANCELLED'} for event in events):",
        "        if item['state'] == 'COMPLETED' and str(item['exit_code']).startswith('0:0'): event_type = 'COMPLETED'",
        "        elif 'CANCEL' in item['state']: event_type = 'CANCELLED'",
        "        else: event_type = 'FAILED'",
        "        event = lifecycle.new_job_event(job_key=item['job_key'], job_type=item['job_type'], event_type=event_type, attempt_id=str(item['attempt_id']), fold=int(item['fold']), slurm_job_id=jid, status=item['state'])",
        "        event['exit_code'] = item['exit_code']",
        "        lifecycle.append_job_event(jobs_path, event)",
        "    status_path = target / 'status.json'",
        "    if status_path.is_file() and item['state'] != 'COMPLETED':",
        "        status = lifecycle.StatusRecord.from_dict(lifecycle.read_status(status_path))",
        "        if status.state in {'SUBMITTED','RUNNING'}:",
        "            status.transition('CANCELLED' if 'CANCEL' in item['state'] else 'FAILED', reason=f\"{item['job_key']} terminal {item['state']} {item['exit_code']}\")",
        "            lifecycle.write_status(status_path, status)",
        "fold_dirs = {str(item['backbone_fold_dir']) for item in items if item.get('backbone_fold_dir')}",
        "for fold_text in fold_dirs:",
        "    fold = pathlib.Path(fold_text); jobs_path = fold / 'jobs.jsonl'; status_path = fold / 'status.json'",
        "    if not jobs_path.is_file() or not status_path.is_file(): continue",
        "    events = lifecycle.read_job_events(jobs_path)",
        "    done = {str(event.get('job_key')) for event in events if event.get('event_type') == 'COMPLETED' and event.get('status') == 'COMPLETED' and str(event.get('exit_code','0:0')).startswith('0:0')}",
        "    status = lifecycle.StatusRecord.from_dict(lifecycle.read_status(status_path))",
        "    if {'train','best_eval'} <= done and status.state == 'RUNNING':",
        "        status.transition('COMPLETED_ON_MN5', reason='train and best evaluation completed 0:0')",
        "        lifecycle.write_status(status_path, status)",
        "PY",
    ]
    return "\n".join(lines) + "\n"


def _status(args: argparse.Namespace) -> int:
    target = Path(args.plan or (LOCAL_ROOT / f"{args.stage}_submission.json"))
    if not target.is_file():
        raise CampaignError(f"submission plan is missing: {target}")
    plan = json.loads(target.read_text(encoding="utf-8"))
    _, _, deployment = _lane_and_deployment(args.slug, plan.get("deployment_id"), execute=True)
    jobs: list[dict[str, Any]] = []
    for backbone in plan["backbones"]:
        for key, job_key, job_type in (
            ("train", "train", "train"),
            ("best_eval", "best_eval", "evaluation"),
        ):
            jobs.append({
                "slurm_job_id": backbone["job_ids"][key],
                "job_key": job_key,
                "job_type": job_type,
                "attempt_id": backbone["backbone_attempt_id"],
                "attempt_dir": backbone["fold_dir"],
                "backbone_fold_dir": backbone["fold_dir"],
                "fold": backbone["fold"],
            })
    for head in plan["head_jobs"]:
        jobs.append({
            "slurm_job_id": head["job_id"],
            "job_key": "head",
            "job_type": head["job_type"],
            "attempt_id": head["attempt_id"],
            "attempt_dir": head["attempt_dir"],
            "fold": head["fold"],
        })
    accounting = _sacct(args.scheduler_host, [str(item["slurm_job_id"]) for item in jobs])
    counts: dict[str, int] = {}
    terminal: list[dict[str, Any]] = []
    for item in jobs:
        record = accounting.get(str(item["slurm_job_id"]), {"state": "UNKNOWN", "exit_code": "-"})
        item.update(record)
        counts[record["state"]] = counts.get(record["state"], 0) + 1
        if record["state"] not in {"PENDING", "RUNNING", "CONFIGURING", "UNKNOWN"}:
            terminal.append(item)
    if terminal:
        result = ssh_bash(args.scheduler_host, _terminal_update_script(terminal, deployment), timeout=1800)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise CampaignError("remote terminal evidence update failed")
    status_path = target.with_name(target.name + ".status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"schema_version": "audiollm.turkish_question_condition_status.v1", "counts": counts, "jobs": jobs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(target), "counts": counts, "terminal_jobs": len(terminal)}, indent=2, sort_keys=True))
    return 0


def _collect(args: argparse.Namespace) -> int:
    target = Path(args.plan or (LOCAL_ROOT / f"{args.stage}_submission.json"))
    if not target.is_file():
        raise CampaignError(f"submission plan is missing: {target}")
    plan = json.loads(target.read_text(encoding="utf-8"))
    targets: list[tuple[str, Path]] = []
    for backbone in plan["backbones"]:
        remote = Path(backbone["fold_dir"])
        try:
            relative = remote.relative_to(REMOTE_PROJECT_ROOT)
        except ValueError as exc:
            raise CampaignError(f"collection path is outside the canonical output root: {remote}") from exc
        targets.append((str(remote), PROJECT_ROOT / relative))
    if args.dry_run:
        print(json.dumps({"targets": [{"remote": remote, "local": str(local)} for remote, local in targets]}, indent=2, sort_keys=True))
        return 0
    for _, local in targets:
        if local.exists():
            raise CampaignError(f"refusing to overwrite existing local collection target: {local}")
    for remote, local in targets:
        local.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "rsync", "-avh", "--itemize-changes",
            "--exclude=best_model/adapter_model*",
            "--exclude=best_model/*.safetensors",
            "--exclude=last_model/adapter_model*",
            "--exclude=last_model/*.safetensors",
            "--exclude=**/*.npz", "--exclude=**/*.joblib", "--exclude=**/*.pkl",
            "--exclude=**/*.safetensors", "--exclude=**/*.bin", "--exclude=**/*.pt", "--exclude=**/*.pth",
            f"{TRANSFER_HOST}:{remote}/", str(local) + "/",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=8 * 60 * 60)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise CampaignError(f"collection failed for {remote}")
    print(json.dumps({"collected_folds": len(targets), "root": str(PROJECT_ROOT / "output_model")}, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("plan")
    matrix.add_argument("--stage", choices=("smoke", "production"), default="production")
    matrix.add_argument("--source-sha")
    matrix.add_argument("--deployment-id")
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(function=_write_matrix_plan)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("slug", nargs="?", default="turkish-full-negonly-multimodal")
    preflight.add_argument("--stage", choices=("smoke", "production"), default="production")
    preflight.add_argument("--deployment-id")
    preflight.add_argument("--scheduler-host", default=DEFAULT_SLURM_HOST)
    preflight.add_argument("--local", action="store_true")
    preflight.add_argument("--execute", action="store_true")
    preflight.add_argument("--mixed-root", type=Path)
    preflight.add_argument("--negative-root", type=Path)
    preflight.add_argument("--translation-root", type=Path)
    preflight.add_argument("--output-root", type=Path, default=LOCAL_ROOT / "local_preflight_runtime")
    preflight.add_argument("--output", type=Path, default=LOCAL_ROOT / "local_preflight.json")
    preflight.add_argument("--require-models", action="store_true")
    preflight.add_argument("--require-environment", action="store_true")
    preflight.set_defaults(function=_run_preflight)

    submit = sub.add_parser("submit")
    submit.add_argument("slug")
    submit.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    submit.add_argument("--deployment-id")
    submit.add_argument("--preflight")
    submit.add_argument("--scheduler-host", default=DEFAULT_SLURM_HOST)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--execute", action="store_true")
    submit.set_defaults(function=_submit)

    status = sub.add_parser("status")
    status.add_argument("slug")
    status.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    status.add_argument("--plan")
    status.add_argument("--scheduler-host", default=DEFAULT_SLURM_HOST)
    status.set_defaults(function=_status)

    collect = sub.add_parser("collect")
    collect.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    collect.add_argument("--plan")
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("--execute", action="store_true")
    collect.set_defaults(function=_collect)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.function(args))
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
