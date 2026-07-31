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
from src.merged.runtime import load_merged_config
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


def _job_id(run_id: str, modality: str, stage: str, fold: int, kind: str) -> str:
    value = f"{run_id}|{modality}|{stage}|{fold}|{kind}"
    return "dry_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _run_roots(config: dict[str, Any], run_id: str, stage: str, fold: int) -> dict[str, Path]:
    return {
        "train": Path(config["output_dirs"]["run_root"]) / run_id / stage / f"fold_{fold}",
        "post": Path(config["output_dirs"]["merged_root"]) / run_id / stage / f"fold_{fold}",
    }


def _completed(config: dict[str, Any], run_id: str, stage: str, fold: int, kind: str) -> bool:
    roots = _run_roots(config, run_id, stage, fold)
    if kind == "train":
        return (roots["train"] / "training_complete.json").is_file() and (roots["train"] / "best_model").is_dir()
    if kind == "postprocess":
        return (roots["post"] / "postprocess_complete.json").is_file() and (roots["post"] / "features" / "feature_metadata.json").is_file()
    if kind == "head":
        return (roots["post"] / "heads" / "heads_complete.json").is_file()
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
    for config_path in configs:
        config = load_merged_config(config_path)
        modality = str(config["modality"])
        if stage == "final":
            # This gate is evaluated by the real submit path. A dry-run still
            # reports the exact deterministic epoch input when available.
            final_epochs = None
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
                job["completed_before_submission"] = _completed(config, run_id, stage, fold, job["kind"])
                if job["completed_before_submission"]:
                    job["state"] = "skipped_compatible_complete"
                    previous_id = None
                else:
                    job["state"] = "planned"
                    if not dry_run and previous_id:
                        job["dependency_job_id"] = previous_id
                    previous_id = job["expected_job_id"]
                jobs.append(job)
    expected = 3 if stage == "smoke" else 45 if stage == "cv" else 9
    return {
        "schema_version": "symmetric_merged_job_registry.v1",
        "run_id": run_id,
        "stage": stage,
        "source_commit": _source_commit(),
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
    arguments.extend([f"--export={export_text}", str(worker)])
    output = subprocess.check_output(arguments, cwd=PROJECT_ROOT, text=True).strip()
    return output.split(";", 1)[0]


def submit_registry(registry: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    previous_ids: dict[str, str] = {
        str(job["job_key"]): str(job["job_id"])
        for job in registry.get("jobs", [])
        if job.get("job_id") and not str(job.get("job_id")).startswith("dry_")
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
        job["submission_time_utc"] = datetime.now(timezone.utc).isoformat()
        job["state"] = "planned_dry_run" if dry_run else "submitted"
        previous_ids[job["job_key"]] = submitted_id
    registry["submission_mode"] = "dry_run" if dry_run else "sbatch"
    registry["terminal"] = False
    registry["planned_job_count"] = active_before
    registry["skipped_job_count"] = len(registry.get("jobs", [])) - active_before
    return registry


def merge_existing_registry(registry: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Carry forward submitted/terminal jobs so reruns are restart-safe."""

    old_jobs = {str(job.get("job_key")): job for job in existing.get("jobs", [])}
    failed_states = {"failed", "cancelled", "timeout", "oom", "node_fail", "preempted"}
    for job in registry.get("jobs", []):
        old = old_jobs.get(str(job["job_key"]))
        if not old:
            continue
        old_state = str(old.get("observed_state", old.get("state", ""))).lower()
        old_job_id = old.get("job_id")
        if old_job_id and not str(old_job_id).startswith("dry_") and old_state not in failed_states:
            job["state"] = old.get("state", "submitted")
            job["job_id"] = old_job_id
            for key in ("submission_time_utc", "observed_state", "exit_code"):
                if key in old:
                    job[key] = old[key]
        elif old_state in failed_states:
            job["retry"] = int(old.get("retry", 0)) + 1
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
        if existing.get("run_id") != run_id or existing.get("stage") != args.stage:
            raise ValueError(f"Refusing colliding symmetric merged registry: {registry_path}")
        # A rerun may reuse a registry only when it is the same protocol plan.
        if existing.get("expected_fresh_job_count") != registry.get("expected_fresh_job_count"):
            raise ValueError(f"Existing registry is incompatible: {registry_path}")
        registry = merge_existing_registry(registry, existing)
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
