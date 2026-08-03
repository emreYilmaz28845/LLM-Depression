#!/usr/bin/env python3
"""Plan or submit the smoke, CV, and final symmetric merged job chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merged.protocol import canonical_sha256
from src.merged.provenance import source_commits_match
from src.merged.runtime import load_merged_config, load_protocol_artifact
from src.utils import read_json, resolve_project_path, save_json, sha256_file


CONFIG_BY_MODALITY = {
    "audio_text": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_audio_text.yaml",
    "audio_only": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_audio_only.yaml",
    "text_only": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_text_only.yaml",
}


def _source_commit() -> str:
    explicit = str(os.environ.get("SYMMETRIC_MERGED_SOURCE_COMMIT", "")).strip()
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _reservation() -> str:
    return str(os.environ.get("SYMMETRIC_MERGED_RESERVATION", "")).strip()


def _job_id(run_id: str, modality: str, stage: str, fold: int, kind: str) -> str:
    value = f"{run_id}|{modality}|{stage}|{fold}|{kind}"
    return "dry_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _run_roots(config: dict[str, Any], run_id: str, stage: str, fold: int) -> dict[str, Path]:
    return {
        "train": Path(config["output_dirs"]["run_root"]) / run_id / stage / f"fold_{fold}",
        "post": Path(config["output_dirs"]["merged_root"]) / run_id / stage / f"fold_{fold}",
    }


def _expected_protocol_identity(
    config: dict[str, Any], config_path: str | Path, fold: int
) -> dict[str, str] | None:
    """Resolve the hashes required before a completed artifact may be reused."""

    try:
        protocol = load_protocol_artifact(config)
        fold_payload = protocol["protocol"]["folds"][str(int(fold))]
        expected = {
            "merged_config_sha256": sha256_file(config_path),
            "manifest_hash": str(protocol["manifest"]["manifest_hash"]),
            "split_hash": str(protocol["protocol"]["split_hash"]),
            "fold_hash": str(fold_payload["fold_hash"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        # A dry-run may be planned before the generated protocol artifact has
        # been materialized. In that case there is no evidence strong enough
        # to skip an existing output, so force the normal compatibility gate.
        return None
    if any(not value or value == "None" for value in expected.values()):
        return None
    return expected


def _provenance_matches(path: Path) -> bool:
    """Require the artifact's recorded source commit to match this submission."""

    if not path.is_file():
        return False
    current = _source_commit()
    if not current or current == "unknown":
        return False
    try:
        return source_commits_match(read_json(path).get("source_commit"), current)
    except (OSError, TypeError, ValueError):
        return False


def _identity_hashes_match(
    identity: dict[str, Any], expected: dict[str, str], *, split_key: str
) -> bool:
    return (
        identity.get("merged_config_sha256") == expected["merged_config_sha256"]
        and identity.get("manifest_hash") == expected["manifest_hash"]
        and identity.get(split_key) == expected["split_hash"]
        and identity.get("fold_hash") == expected["fold_hash"]
    )


def _completed(
    config: dict[str, Any],
    config_path: str | Path,
    run_id: str,
    stage: str,
    fold: int,
    kind: str,
    *,
    epochs: int | None = None,
    subjects_per_class: int | None = None,
    trials: int | None = None,
) -> bool:
    roots = _run_roots(config, run_id, stage, fold)
    expected_identity = _expected_protocol_identity(config, config_path, fold)
    if expected_identity is None:
        return False
    if kind == "train":
        complete = roots["train"] / "training_complete.json"
        if not complete.is_file() or not (roots["train"] / "best_model").is_dir():
            return False
        if not _provenance_matches(roots["train"] / "slurm_provenance.json"):
            return False
        payload = read_json(complete)
        identity = payload.get("identity", {})
        expected_epochs = int(epochs if epochs is not None else config["training"].get("num_train_epochs", 20))
        return (
            payload.get("status") == "completed"
            and identity.get("config_name") == config.get("name")
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and int(identity.get("epochs", -1)) == expected_epochs
            and identity.get("subjects_per_class") == subjects_per_class
            and _identity_hashes_match(
                identity, expected_identity, split_key="protocol_split_hash"
            )
        )
    if kind == "postprocess":
        complete = roots["post"] / "postprocess_complete.json"
        identity_path = roots["post"] / "postprocess_identity.json"
        if not complete.is_file() or not identity_path.is_file() or not (roots["post"] / "features" / "feature_metadata.json").is_file():
            return False
        if not _provenance_matches(roots["post"] / "slurm_provenance.json"):
            return False
        identity = read_json(identity_path)
        return (
            read_json(complete).get("status") == "completed"
            and identity.get("config_name") == config.get("name")
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and identity.get("modality") == config.get("modality")
            and identity.get("checkpoint_dir") == str((roots["train"] / "best_model").resolve())
            and identity.get("subjects_per_class") == (subjects_per_class if stage == "smoke" else None)
            and _identity_hashes_match(identity, expected_identity, split_key="split_hash")
        )
    if kind == "head":
        complete = roots["post"] / "heads" / "heads_complete.json"
        identity_path = roots["post"] / "heads" / "heads_identity.json"
        if not complete.is_file() or not identity_path.is_file():
            return False
        if not _provenance_matches(roots["post"] / "heads" / "slurm_provenance.json"):
            return False
        identity = read_json(identity_path)
        expected_trials = int(trials if trials is not None else 150)
        return (
            read_json(complete).get("status") == "completed"
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and identity.get("feature_metadata") == str((roots["post"] / "features" / "feature_metadata.json").resolve())
            and int(identity.get("optuna_trials", -1)) == expected_trials
            and _identity_hashes_match(identity, expected_identity, split_key="split_hash")
        )
    raise ValueError(kind)


def _final_epoch(config: dict[str, Any], run_id: str, modality: str) -> int:
    values: list[int] = []
    for fold in range(5):
        path = _run_roots(config, run_id, "cv", fold)["train"] / "logs" / "selected_checkpoint.json"
        if not path.is_file():
            raise FileNotFoundError(f"CV selection artifact is missing for final {modality} fold {fold}: {path}")
        values.append(int(read_json(path)["selected_epoch"]))
    result = int(math.floor(float(median(values)) + 0.5))
    if result < 1 or result > 20:
        raise ValueError(f"Invalid rounded median selected epoch for {modality}: {values} -> {result}")
    return result


def _final_epoch_for_dry_run(
    config: dict[str, Any], run_id: str, modality: str
) -> int | None:
    """Use the frozen CV epoch when a post-CV dry-run can resolve it.

    Dry-runs remain useful before CV exists, so missing gate artifacts do not
    fail planning. Once the passed CV audit and all five selections exist,
    however, using the real median epoch lets restart checks recognize a
    completed final training job instead of comparing it with the default 20.
    """

    audit_path = Path(config["output_dirs"]["merged_root"]) / run_id / "cv" / "acceptance_audit.json"
    if not audit_path.is_file() or read_json(audit_path).get("status") != "passed":
        return None
    selection_paths = [
        _run_roots(config, run_id, "cv", fold)["train"]
        / "logs"
        / "selected_checkpoint.json"
        for fold in range(5)
    ]
    if not all(path.is_file() for path in selection_paths):
        return None
    return _final_epoch(config, run_id, modality)


def _check_final_gate(config: dict[str, Any], run_id: str, modality: str) -> Path:
    path = _run_roots(config, run_id, "cv", 0)["post"] / "acceptance_audit.json"
    # The audit is written once per modality at the stage root, not per fold.
    path = Path(config["output_dirs"]["merged_root"]) / run_id / "cv" / "acceptance_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Final stage requires a passed CV audit for {modality}: {path}")
    payload = read_json(path)
    if payload.get("status") != "passed":
        raise ValueError(f"Final stage refused because the CV audit did not pass: {path}")
    return path


def build_job_specs(
    configs: list[Path], *, stage: str, run_id: str, dry_run: bool, smoke_subjects: int, smoke_epochs: int, smoke_trials: int
) -> dict[str, Any]:
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(stage)
    if stage == "smoke":
        configs = [CONFIG_BY_MODALITY["audio_text"]]
        folds = [0]
    elif stage == "cv":
        folds = list(range(5))
    else:
        folds = [0]
    jobs: list[dict[str, Any]] = []
    config_identities: list[dict[str, str]] = []
    for config_path in configs:
        config = load_merged_config(config_path)
        modality = str(config["modality"])
        config_identities.append(
            {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "modality": modality,
            }
        )
        if stage == "final":
            # This gate is evaluated by the real submit path. A dry-run still
            # reports the exact deterministic epoch input when available.
            final_epochs = (
                _final_epoch_for_dry_run(config, run_id, modality)
                if dry_run
                else None
            )
            if not dry_run:
                _check_final_gate(config, run_id, modality)
                final_epochs = _final_epoch(config, run_id, modality)
        else:
            final_epochs = None
        for fold in folds:
            roots = _run_roots(config, run_id, stage, fold)
            chain = [
                {
                    "kind": "train",
                    "config": str(config_path),
                    "modality": modality,
                    "stage": stage,
                    "fold": fold,
                    "run_id": run_id,
                    "run_root": str(roots["train"]),
                    "resource": {"gpus": 4, "cpus": 80, "time": config["execution"]["qwen_time"]},
                    "epochs": final_epochs if stage == "final" else (smoke_epochs if stage == "smoke" else None),
                    "subjects_per_class": smoke_subjects if stage == "smoke" else None,
                },
                {
                    "kind": "postprocess",
                    "config": str(config_path),
                    "modality": modality,
                    "stage": stage,
                    "fold": fold,
                    "run_id": run_id,
                    "run_root": str(roots["post"]),
                    "checkpoint_dir": str(roots["train"] / "best_model"),
                    "subjects_per_class": smoke_subjects if stage == "smoke" else None,
                    "resource": {"gpus": 1, "cpus": 20, "time": config["execution"]["postprocess_time"]},
                },
                {
                    "kind": "head",
                    "config": str(config_path),
                    "modality": modality,
                    "stage": stage,
                    "fold": fold,
                    "run_id": run_id,
                    "run_root": str(roots["post"] / "heads"),
                    "features_dir": str(roots["post"] / "features"),
                    "resource": {"gpus": 0, "cpus": 20, "time": config["execution"]["head_time"]},
                    "trials": smoke_trials if stage == "smoke" else 150,
                },
            ]
            previous_id: str | None = None
            for job in chain:
                job_key = f"{modality}:{stage}:fold_{fold}:{job['kind']}"
                job["job_key"] = job_key
                job["expected_job_id"] = _job_id(run_id, modality, stage, fold, job["kind"])
                job["dependency_job_key"] = (
                    f"{modality}:{stage}:fold_{fold}:{'train' if job['kind'] == 'postprocess' else 'postprocess'}"
                    if job["kind"] != "train" else None
                )
                job["completed_before_submission"] = _completed(
                    config,
                    config_path,
                    run_id,
                    stage,
                    fold,
                    job["kind"],
                    epochs=job.get("epochs"),
                    subjects_per_class=job.get("subjects_per_class"),
                    trials=job.get("trials"),
                )
                if job["completed_before_submission"]:
                    job["state"] = "skipped_compatible_complete"
                    previous_id = None
                else:
                    job["state"] = "planned"
                    if not dry_run and previous_id:
                        job["dependency_job_id"] = previous_id
                    previous_id = job["expected_job_id"]
                jobs.append(job)
    # The default production invocation has three modalities (45 CV or 9
    # final jobs), while targeted retries may intentionally pass one or more
    # configs. Count the actual planned chain so retry registries remain
    # truthful without changing the default protocol scope.
    expected = len(configs) * len(folds) * 3
    plan_identity = {
        "stage": stage,
        "configs": config_identities,
        "smoke_subjects": int(smoke_subjects),
        "smoke_epochs": int(smoke_epochs),
        "smoke_trials": int(smoke_trials),
    }
    stage_plan = {
        "stage": stage,
        "plan_identity": plan_identity,
        "plan_hash": canonical_sha256(plan_identity),
        "expected_fresh_job_count": expected,
    }
    return {
        "schema_version": "symmetric_merged_job_registry.v2",
        "run_id": run_id,
        "stage": stage,
        "source_commit": _source_commit(),
        "reservation": _reservation() or None,
        "plan_identity": plan_identity,
        "plan_hash": stage_plan["plan_hash"],
        "stages": [stage],
        "stage_plans": {stage: stage_plan},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_fresh_job_count": expected,
        "planned_job_count": sum(job["state"] == "planned" for job in jobs),
        "skipped_job_count": sum(job["state"] != "planned" for job in jobs),
        "jobs": jobs,
    }


def _submit_job(job: dict[str, Any], *, worker: Path, dependency_id: str | None) -> str:
    export_values = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CONFIG": job["config"],
        "STAGE": job["stage"],
        "FOLD": str(job["fold"]),
        "RUN_ID": job["run_id"],
        "SOURCE_COMMIT": _source_commit(),
    }
    for key in ("epochs", "subjects_per_class", "trials", "checkpoint_dir", "features_dir"):
        if job.get(key) is not None:
            export_values[key.upper()] = str(job[key])
    export_text = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    arguments = ["sbatch", "--parsable", f"--job-name=sym-{job['modality'][:4]}-{job['stage'][:4]}-{job['fold']}-{job['kind'][:4]}"]
    if dependency_id:
        arguments.append(f"--dependency=afterok:{dependency_id}")
    reservation = _reservation()
    if reservation:
        arguments.append(f"--reservation={reservation}")
    arguments.extend([f"--export={export_text}", str(worker)])
    output = subprocess.check_output(arguments, cwd=PROJECT_ROOT, text=True).strip()
    return output.split(";", 1)[0]


def submit_registry(registry: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    def _successful_slurm_job(job: dict[str, Any]) -> bool:
        state_tokens = str(job.get("observed_state", "")).upper().split(None, 1)
        return bool(state_tokens) and state_tokens[0] == "COMPLETED" and str(job.get("exit_code", "")) == "0:0"

    previous_ids: dict[str, str] = {
        str(job["job_key"]): str(job["job_id"])
        for job in registry.get("jobs", [])
        if job.get("job_id") and not str(job.get("job_id")).startswith("dry_")
        and not _successful_slurm_job(job)
    }
    worker_by_kind = {
        "train": PROJECT_ROOT / "scripts/run_symmetric_merged_train_slurm.sh",
        "postprocess": PROJECT_ROOT / "scripts/run_symmetric_merged_postprocess_slurm.sh",
        "head": PROJECT_ROOT / "scripts/run_symmetric_merged_head_slurm.sh",
    }
    active_before = sum(job.get("state") == "planned" for job in registry.get("jobs", []))
    for job in registry["jobs"]:
        if job["state"] != "planned":
            continue
        dependency_job_key = job.get("dependency_job_key")
        dependency_id = previous_ids.get(dependency_job_key) if dependency_job_key else None
        if dry_run:
            submitted_id = job["expected_job_id"]
        else:
            submitted_id = _submit_job(job, worker=worker_by_kind[job["kind"]], dependency_id=dependency_id)
        job["job_id"] = submitted_id
        if dependency_id:
            job["dependency_job_id"] = dependency_id
        job["submission_time_utc"] = datetime.now(timezone.utc).isoformat()
        job["state"] = "planned_dry_run" if dry_run else "submitted"
        previous_ids[job["job_key"]] = submitted_id
    # A dry-run or a restart can carry already-submitted jobs forward without
    # visiting them in the loop above.  Refresh their recorded dependency IDs
    # from the authoritative job-key map so the registry cannot retain a
    # stale dry-run ID even though Slurm received the real dependency.
    for job in registry["jobs"]:
        dependency_key = job.get("dependency_job_key")
        dependency_id = previous_ids.get(str(dependency_key)) if dependency_key else None
        if dependency_id and not str(dependency_id).startswith("dry_"):
            job["dependency_job_id"] = dependency_id
        elif not dependency_id:
            job.pop("dependency_job_id", None)
    registry["submission_mode"] = "dry_run" if dry_run else "sbatch"
    registry["terminal"] = False
    registry["planned_job_count"] = active_before
    registry["skipped_job_count"] = len(registry.get("jobs", [])) - active_before
    return registry


def merge_existing_registry(registry: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Carry forward submitted/terminal jobs so reruns are restart-safe."""

    old_jobs = {str(job.get("job_key")): job for job in existing.get("jobs", [])}
    # Slurm can expose dependency failures with several spellings (for
    # example, ``DependencyNeverSatisfied``).  Normalize the state before
    # deciding whether an old job can be carried forward.  A failed train
    # also invalidates any already-submitted descendants: leaving those old
    # jobs in place would keep them attached to the failed job ID forever.
    failed_states = {
        "failed",
        "cancelled",
        "canceled",
        "timeout",
        "oom",
        "outofmemory",
        "nodefail",
        "preempted",
        "dependencyneversatisfied",
    }

    def state_token(value: Any) -> str:
        return "".join(character for character in str(value).lower() if character.isalnum())

    retry_job_keys: set[str] = set()
    for job in registry.get("jobs", []):
        old = old_jobs.get(str(job["job_key"]))
        if not old:
            continue
        # Preserve retry metadata and the last known dependency while a
        # planned dry-run registry is promoted to a real submission.  The
        # dependency is refreshed from current job IDs in submit_registry.
        if "retry" in old:
            job["retry"] = old["retry"]
        if "dependency_job_id" in old:
            job["dependency_job_id"] = old["dependency_job_id"]
        old_state = state_token(old.get("observed_state", old.get("state", "")))
        old_job_id = old.get("job_id")
        terminal_success_without_artifact = (
            old_state == "completed" and not job.get("completed_before_submission")
        )
        if (
            old_job_id
            and not str(old_job_id).startswith("dry_")
            and old_state not in failed_states
            and not terminal_success_without_artifact
        ):
            job["state"] = old.get("state", "submitted")
            job["job_id"] = old_job_id
            for key in ("submission_time_utc", "observed_state", "exit_code"):
                if key in old:
                    job[key] = old[key]
        elif old_state in failed_states or terminal_success_without_artifact:
            job["retry"] = int(old.get("retry", 0)) + 1
            retry_job_keys.add(str(job["job_key"]))

    def reset_for_retry(job: dict[str, Any]) -> None:
        job["state"] = "planned"
        for key in (
            "job_id",
            "dependency_job_id",
            "submission_time_utc",
            "observed_state",
            "exit_code",
            "elapsed",
            "node_list",
            "allocated_cpus",
        ):
            job.pop(key, None)

    # Propagate a retry through the dependency chain.  The loop is
    # intentionally order-independent so a registry loaded from an older
    # run cannot submit a postprocess/head job before its replacement train
    # job.  Compatible artifacts remain reusable if they already exist.
    changed = True
    while changed:
        changed = False
        for job in registry.get("jobs", []):
            job_key = str(job["job_key"])
            dependency_key = job.get("dependency_job_key")
            if not dependency_key or str(dependency_key) not in retry_job_keys:
                continue
            if job.get("completed_before_submission"):
                continue
            old = old_jobs.get(job_key)
            if old:
                job["retry"] = int(old.get("retry", 0)) + 1
            reset_for_retry(job)
            if job_key not in retry_job_keys:
                retry_job_keys.add(job_key)
                changed = True
    current_keys = {str(job.get("job_key")) for job in registry.get("jobs", [])}
    for old in existing.get("jobs", []):
        if str(old.get("job_key")) not in current_keys:
            registry.setdefault("jobs", []).append(old)
    dependency_order = {"train": 0, "postprocess": 1, "head": 2}
    registry["jobs"].sort(
        key=lambda job: (
            str(job.get("stage", "")),
            str(job.get("modality", "")),
            int(job.get("fold", 0)),
            dependency_order.get(str(job.get("kind", "")), 99),
            str(job.get("kind", "")),
        )
    )
    return registry


def _stage_plans(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read stage plans from v2 registries and legacy single-stage files."""

    plans = registry.get("stage_plans")
    if isinstance(plans, dict) and plans:
        return {
            str(stage): dict(payload)
            for stage, payload in plans.items()
            if isinstance(payload, dict)
        }
    stage = str(registry.get("stage", "")).strip()
    if stage and stage != "multi":
        return {
            stage: {
                "stage": stage,
                "plan_identity": registry.get("plan_identity", {}),
                "plan_hash": registry.get("plan_hash"),
                "expected_fresh_job_count": registry.get("expected_fresh_job_count"),
            }
        }
    return {}


def _set_combined_registry_metadata(
    registry: dict[str, Any], stage_plans: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Make one registry authoritative for all stages sharing a run ID."""

    ordered = {stage: stage_plans[stage] for stage in sorted(stage_plans)}
    stages = list(ordered)
    registry["schema_version"] = "symmetric_merged_job_registry.v2"
    registry["stages"] = stages
    registry["stage"] = stages[0] if len(stages) == 1 else "multi"
    registry["stage_plans"] = ordered
    registry["plan_identity"] = {
        "stages": {
            stage: ordered[stage].get("plan_identity", {})
            for stage in stages
        }
    }
    registry["plan_hash"] = canonical_sha256(
        {stage: ordered[stage].get("plan_hash") for stage in stages}
    )
    registry["expected_job_count"] = sum(
        int(ordered[stage].get("expected_fresh_job_count", 0) or 0)
        for stage in stages
    )
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--config", action="append", type=Path, dest="configs")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--smoke-subjects", type=int, default=2)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-trials", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [resolve_project_path(value) for value in (args.configs or list(CONFIG_BY_MODALITY.values()))]
    if args.stage == "smoke":
        configs = [CONFIG_BY_MODALITY["audio_text"]]
    run_id = args.run_id
    if not run_id:
        identity = {
            "stage": args.stage,
            "configs": [str(path) + ":" + sha256_file(path) for path in configs],
            "source_commit": _source_commit(),
        }
        run_id = f"symmetric_merged_{args.stage}_{canonical_sha256(identity)[:12]}"
    registry = build_job_specs(
        configs,
        stage=args.stage,
        run_id=run_id,
        dry_run=args.dry_run,
        smoke_subjects=args.smoke_subjects,
        smoke_epochs=args.smoke_epochs,
        smoke_trials=args.smoke_trials,
    )
    registry_path = resolve_project_path(args.registry) if args.registry else PROJECT_ROOT / "outputs/symmetric_merged_jobs" / f"{run_id}.json"
    if registry_path.exists():
        existing = read_json(registry_path)
        if existing.get("run_id") != run_id:
            raise ValueError(f"Refusing colliding symmetric merged registry: {registry_path}")
        # A rerun may reuse a stage only when it is the same protocol plan.
        # A new stage may be appended to the same run ID: final needs the CV
        # artifacts and epoch selections under that shared run root.
        existing_stage_plans = _stage_plans(existing)
        current_stage_plan = _stage_plans(registry)[args.stage]
        if args.stage in existing_stage_plans:
            if existing_stage_plans[args.stage].get("plan_hash") != current_stage_plan.get("plan_hash"):
                raise ValueError(f"Existing registry has an incompatible protocol plan: {registry_path}")
        existing_configs = {
            str(job.get("job_key")): str(job.get("config"))
            for job in existing.get("jobs", [])
            if job.get("job_key") and str(job.get("stage")) == args.stage
        }
        current_configs = {
            str(job.get("job_key")): str(job.get("config"))
            for job in registry.get("jobs", [])
            if job.get("job_key")
        }
        if existing_configs and existing_configs != current_configs:
            raise ValueError(f"Existing registry has incompatible job/config identities: {registry_path}")
        registry = merge_existing_registry(registry, existing)
        existing_stage_plans[args.stage] = current_stage_plan
        registry = _set_combined_registry_metadata(registry, existing_stage_plans)
    else:
        registry = _set_combined_registry_metadata(registry, _stage_plans(registry))
    registry = submit_registry(registry, dry_run=args.dry_run)
    save_json(registry, registry_path)
    print(json.dumps({
        "status": "dry_run_complete" if args.dry_run else "submitted",
        "registry": str(registry_path),
        "run_id": run_id,
        "stage": args.stage,
        "expected_fresh_job_count": registry["expected_fresh_job_count"],
        "planned_or_submitted_job_count": registry["planned_job_count"],
        "skipped_job_count": registry["skipped_job_count"],
        "job_ids": [job.get("job_id") for job in registry["jobs"] if job.get("job_id")],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
