#!/usr/bin/env python3
"""Managed orchestration for the v2 native-versus-English text-head study."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import sha256_file
from src.experiment_tracking.deployment import (
    DEFAULT_TRANSFER_HOST,
    REMOTE_BASE,
    RemoteRunner,
    verify_deployment,
)
from src.experiment_tracking.identity import new_attempt_id
from src.experiment_tracking.submit import DEFAULT_SCHEDULER_HOST, encode_overrides
from src.native_en_text_heads import (
    BACKBONES,
    CONDITIONS,
    GROUP_ID,
    MERGED_CONFIGS,
    MERGED_DATASETS,
    STANDALONE_CONFIGS,
    STANDALONE_DATASETS,
    SMOKE_STANDALONE_DATASET,
    matrix_counts,
    matrix_payload,
    validate_configs,
)
from src.utils import load_yaml_with_overrides

REMOTE_PROJECT_ROOT = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression")
REMOTE_RUNTIME_BASE = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime")
REMOTE_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "output_model"
QWEN_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
GEMMA_ENV_ACTIVATE = "/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate"
GEMMA_MODEL_PATH = "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DATASET_MANIFEST_BASENAMES = {
    "daic": "daic_manifest.jsonl",
    "d3tec": "d3tec_manifest.jsonl",
    "androids_interview": "androids_interview_manifest.jsonl",
    "cmdc": "cmdc_manifest.jsonl",
    "turkish": "turkish_manifest.jsonl",
}


class OrchestrationError(RuntimeError):
    """A fail-closed orchestration or collision error."""


def q(value: Any) -> str:
    return shlex.quote(str(value))


def ssh_script(host: str, script: str, *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def rel_config(relative: str, deployed_code_path: str) -> str:
    return f"{deployed_code_path}/{relative}"


def load_deployment(slug: str, deployment_id: str | None, *, execute: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    import tools.exp as exp

    worktree, pin = exp._resolve_lane(slug)
    if worktree is None or pin is None:
        raise OrchestrationError(f"no managed lane found for {slug!r}")
    ok, message = exp._check_pin(worktree)
    if not ok:
        raise OrchestrationError(f"worktree pin failed: {message}")
    group = exp._load_linked_experiment_group(worktree, pin)
    experiment_id = str(pin.get("experiment_id") or slug)
    found = exp._find_deployment_record(experiment_id, deployment_id, allow_plan=not execute)
    if not isinstance(found, tuple) or len(found) != 2 or not isinstance(found[0], Path):
        raise OrchestrationError(str(found))
    _, deployment = found
    for key in ("experiment_group_id", "experiment_group_path", "experiment_group_sha256"):
        if deployment.get(key) != group[key]:
            raise OrchestrationError(
                f"deployment {key}={deployment.get(key)!r} does not match linked group {group[key]!r}"
            )
    return worktree, pin, deployment


def stage_root(experiment_id: str, stage: str) -> Path:
    return REMOTE_RUNTIME_BASE / experiment_id / stage


def manifest_paths(root: Path, condition: str, backbone: str, dataset: str) -> tuple[Path, Path]:
    manifest_dir = root / "manifests" / condition / backbone / dataset
    split_dir = root / "splits" / condition / backbone / dataset
    return manifest_dir / DATASET_MANIFEST_BASENAMES[dataset], split_dir / f"{dataset}_manifest_metadata.json"


def merged_root(
    root: Path, condition: str, backbone: str, output_suffix: str | None = None
) -> Path:
    suffix = _normalize_output_suffix(output_suffix)
    base = root / "merged"
    if suffix:
        base = base / suffix
    return base / condition / backbone


def _normalize_output_suffix(output_suffix: str | None) -> str:
    if output_suffix is None or not str(output_suffix).strip():
        return ""
    value = str(output_suffix).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", value):
        raise OrchestrationError(
            "output suffix must be 1-32 characters of letters, digits, '_' or '-'; "
            f"got {value!r}"
        )
    return value


def campaign(stage: str, condition: str, backbone: str, output_suffix: str | None = None) -> str:
    prefix = "native_en_text_heads_v2" if stage == "production" else "native_en_text_heads_v2_smoke"
    suffix = _normalize_output_suffix(output_suffix)
    if suffix:
        prefix = f"{prefix}_{suffix}"
    return f"{prefix}_{condition}_{backbone}"


def standalone_overrides(
    *, root: Path, stage: str, condition: str, backbone: str, dataset: str, seed: int,
    output_suffix: str | None = None,
) -> list[str]:
    manifest, metadata = manifest_paths(root, condition, backbone, dataset)
    run_root = REMOTE_OUTPUT_ROOT / campaign(stage, condition, backbone, output_suffix) / "text_only" / dataset
    values = [
        f"--set=output_dirs.manifest_dir={manifest.parent}",
        f"--set=output_dirs.split_dir={metadata.parent}",
        f"--set=output_dirs.run_root={run_root}",
        f"--set=seed={int(seed)}",
        "--set=evaluation.evaluation_view=harmonized_all_windows_full_coverage",
    ]
    if stage == "smoke":
        values.extend(("--set=training.num_train_epochs=1", "--set=split.smoke_subject_limit=6"))
    return values


def merged_overrides(
    *, root: Path, stage: str, condition: str, backbone: str, seed: int,
    output_suffix: str | None = None,
) -> list[str]:
    config = load_yaml_with_overrides(PROJECT_ROOT / MERGED_CONFIGS[(condition, backbone)], [])
    values = [
        f"--set=output_dirs.run_root={REMOTE_OUTPUT_ROOT / campaign(stage, condition, backbone, output_suffix) / 'text_only' / 'merged'}",
        f"--set=output_dirs.merged_root={merged_root(root, condition, backbone, output_suffix)}",
        f"--set=seed={int(seed)}",
    ]
    for index, item in enumerate(config.get("components") or []):
        dataset = str(item["name"]).lower()
        manifest, metadata = manifest_paths(root, condition, backbone, dataset)
        values.extend((
            f"--set=components.{index}.manifest_path={manifest}",
            f"--set=components.{index}.metadata_path={metadata}",
        ))
    return values


def resolved_config(relative: str, overrides: list[str], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_yaml_with_overrides(PROJECT_ROOT / relative, overrides)
    if extra:
        config.update(extra)
    return config


def source_fields(deployment: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_commit": deployment.get("git_commit"),
        "git_branch": deployment.get("git_branch_at_deploy"),
        "git_dirty": bool(deployment.get("git_dirty", False)),
        "deployed_source_sha256": deployment.get("source_manifest_sha256"),
        "deployment_id": deployment.get("deployment_id"),
    }


def pair_hashes(preflight: dict[str, Any], condition: str, backbone: str, dataset: str) -> dict[str, Any]:
    for pair in preflight.get("paired_manifests", []):
        if pair.get("backbone") == backbone and pair.get("dataset") == dataset:
            return dict(pair.get(condition) or {})
    return {}


def head_payload(
    *,
    deployment: dict[str, Any],
    logical: str,
    attempt_id: str,
    fold: int,
    seed: int,
    stage: str,
    condition: str,
    backbone: str,
    method: str,
    backend: str,
    config: dict[str, Any],
    manifest_hash: str | None,
    split_hash: str | None,
    parent_attempt_id: str,
    parent_checkpoint_path: str,
    optuna_trials: int | None = None,
    tracking_kind: str = "native_en_text_heads_v2_head",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(json.dumps(config))
    config["stage"] = stage
    config["condition"] = condition
    config["backbone"] = backbone
    evaluation = config.setdefault("evaluation", {})
    evaluation["evaluation_view"] = "harmonized_all_windows_full_coverage"
    evaluation["aggregation"] = "subject_level"
    evaluation["split_name"] = "outer_holdout" if stage != "final" else "daic_official_test"
    evaluation["split_protocol"] = "saved_split"
    config["classifier"] = {
        "method": method,
        "prediction_backend": backend,
        "head_seed": 1337,
        "protocol": "native_en_text_heads_v2",
        "optuna_trials": int(optuna_trials or 0),
    }
    context = {
        "schema_version": "audiollm.tracking_context.v1",
        "group_id": GROUP_ID,
        "logical_run_name": logical,
        "attempt_id": attempt_id,
        "fold": int(fold),
        "seed": int(seed),
        "source": source_fields(deployment),
        "hashes": {"manifest_sha256": manifest_hash, "split_sha256": split_hash},
        "tracking_kind": tracking_kind,
        "required_jobs": ["head"],
        "research": {},
    }
    parent = {
        "parent_attempt_id": parent_attempt_id,
        "parent_checkpoint_path": parent_checkpoint_path,
    }
    return context, config, parent


def job_payload(
    *,
    deployment: dict[str, Any],
    logical: str,
    attempt_id: str,
    fold: int,
    seed: int,
    stage: str,
    config: dict[str, Any],
    required_job: str,
    tracking_kind: str,
    manifest_hash: str | None,
    split_hash: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(json.dumps(config))
    config["stage"] = stage
    config["dataset"] = config.get("dataset", "merged")
    context = {
        "schema_version": "audiollm.tracking_context.v1",
        "group_id": GROUP_ID,
        "logical_run_name": logical,
        "attempt_id": attempt_id,
        "fold": int(fold),
        "seed": int(seed),
        "source": source_fields(deployment),
        "hashes": {"manifest_sha256": manifest_hash, "split_sha256": split_hash},
        "tracking_kind": tracking_kind,
        "required_jobs": [required_job],
        "research": {},
    }
    return context, config


def entry(
    *,
    attempt_id: str,
    attempt_dir: Path,
    context_path: Path,
    config_path: Path,
    parent_path: Path,
    logical: str,
    job_key: str,
    job_type: str,
    dependencies: list[str],
    script: str,
    seed: int,
    fold: int,
    condition: str,
    backbone: str,
    endpoint: str,
    method: str | None,
    stage: str,
    config_remote: str,
    overrides: list[str],
    checkpoint_dir: Path | None = None,
    features_dir: Path | None = None,
    cache_dir: Path | None = None,
    trials: int | None = None,
    run_id: str | None = None,
    kind: str = "custom",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_dir),
        "context_path": str(context_path),
        "config_json_path": str(config_path),
        "parent_json_path": str(parent_path),
        "logical_run_name": logical,
        "job_key": job_key,
        "job_type": job_type,
        "dependencies": list(dependencies),
        "script": script,
        "seed": int(seed),
        "fold": int(fold),
        "condition": condition,
        "backbone": backbone,
        "endpoint": endpoint,
        "method": method,
        "stage": stage,
        "config_remote": config_remote,
        "overrides": list(overrides),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "features_dir": str(features_dir) if features_dir else None,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "trials": trials,
        "run_id": run_id,
    }


def build_plan(
    *,
    stage: str,
    deployment: dict[str, Any],
    experiment_id: str,
    preflight: dict[str, Any] | None = None,
    output_suffix: str | None = None,
    retry_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_configs(PROJECT_ROOT)
    if stage not in {"smoke", "production"}:
        raise OrchestrationError(f"unsupported stage {stage!r}")
    output_suffix = _normalize_output_suffix(output_suffix)
    if retry_from is not None:
        if retry_from.get("experiment_id") != experiment_id:
            raise OrchestrationError(
                "retry source plan experiment_id does not match the linked lane: "
                f"{retry_from.get('experiment_id')!r} != {experiment_id!r}"
            )
        if retry_from.get("stage") != stage:
            raise OrchestrationError(
                f"retry source plan stage {retry_from.get('stage')!r} does not match {stage!r}"
            )
        if not retry_from.get("submission_complete"):
            raise OrchestrationError("retry source plan is not a completed submission plan")
        if not output_suffix:
            raise OrchestrationError("retries require a fresh --output-suffix")
    root = stage_root(experiment_id, stage)
    preflight = preflight or {}
    jobs: list[dict[str, Any]] = []
    collision_paths: set[str] = set()
    by_key: dict[str, dict[str, Any]] = {}
    seeds = (1337,) if stage == "smoke" else (7, 1337, 2024)

    supersedes_by_job_key: dict[str, str] = {}
    if retry_from is not None:
        for old_job in retry_from.get("jobs", []):
            old_attempt_id = old_job.get("attempt_id")
            if not old_attempt_id:
                continue
            for key in (old_job.get("job_key"), old_job.get("eval_job_key")):
                if key:
                    supersedes_by_job_key[str(key)] = str(old_attempt_id)

    def register(job: dict[str, Any]) -> None:
        if job["job_key"] in by_key:
            raise OrchestrationError(f"duplicate job key: {job['job_key']}")
        supersedes_attempt_id = supersedes_by_job_key.get(str(job["job_key"]))
        if supersedes_attempt_id:
            job["supersedes_attempt_id"] = supersedes_attempt_id
            if job.get("kind") == "standalone_backbone":
                job.setdefault("context", {})["supersedes_attempt_id"] = supersedes_attempt_id
            else:
                job.setdefault("context_payload", {})["supersedes_attempt_id"] = supersedes_attempt_id
        by_key[job["job_key"]] = job
        jobs.append(job)

    for condition in CONDITIONS:
        for backbone in BACKBONES:
            for dataset in STANDALONE_DATASETS:
                if stage == "smoke" and dataset != SMOKE_STANDALONE_DATASET[(condition, backbone)]:
                    continue
                relative = STANDALONE_CONFIGS[(condition, backbone, dataset)]
                config_remote = rel_config(relative, str(deployment["deployed_code_path"]))
                for seed in seeds:
                    logical = f"native_en_text_heads_v2_{condition}_{backbone}_{dataset}_s{seed}"
                    for fold in ((0,) if stage == "smoke" else (0, 1, 2, 3, 4)):
                        overrides = standalone_overrides(
                            root=root,
                            stage=stage,
                            condition=condition,
                            backbone=backbone,
                            dataset=dataset,
                            seed=seed,
                            output_suffix=output_suffix,
                        )
                        attempt_id = new_attempt_id(logical, str(deployment["git_commit"]))
                        run_root = REMOTE_OUTPUT_ROOT / campaign(stage, condition, backbone, output_suffix) / "text_only" / dataset
                        fold_dir = run_root / logical / f"fold_{fold}"
                        context_path = root / "contexts" / attempt_id / f"fold_{fold}" / "context.json"
                        pair = pair_hashes(preflight, condition, backbone, dataset)
                        context = {
                            "schema_version": "audiollm.tracking_context.v1",
                            "group_id": GROUP_ID,
                            "logical_run_name": logical,
                            "attempt_id": attempt_id,
                            "fold": fold,
                            "seed": seed,
                            "source": source_fields(deployment),
                            "hashes": {
                                "manifest_sha256": pair.get("manifest_hash"),
                                "split_sha256": pair.get("metadata_sha256"),
                            },
                            "research": {},
                        }
                        config = resolved_config(relative, overrides)
                        contract = {
                            "kind": "standalone_backbone",
                            "attempt_id": attempt_id,
                            "logical_run_name": logical,
                            "fold": fold,
                            "seed": seed,
                            "condition": condition,
                            "backbone": backbone,
                            "dataset": dataset,
                            "endpoint": "standalone",
                            "job_key": f"{logical}:standalone:{fold}:train:none",
                            "job_type": "train",
                            "config": relative,
                            "config_remote": config_remote,
                            "run_root": str(run_root),
                            "fold_dir": str(fold_dir),
                            "checkpoint_dir": str(fold_dir / "best_model"),
                            "standalone_eval_dir": str(fold_dir / "best_model" / "standalone_eval"),
                            "context_path": str(context_path),
                            "overrides": overrides,
                            "overrides_b64": encode_overrides(overrides),
                            "context": context,
                            "config_payload": config,
                            "parent_payload": {},
                            "qualifiers": {
                                "evaluation_view": "harmonized_all_windows_full_coverage",
                                "backend": "original_teacher_forced",
                                "aggregation": "subject_level",
                                "namespace": "headline/binary_strict",
                                "checkpoint_role": "best_model",
                            },
                            "job_ids": {},
                            "local_fold_rel": str(fold_dir).replace(str(REMOTE_PROJECT_ROOT), "output_model"),
                        }
                        register(contract)
                        by_key[f"{logical}:standalone:{fold}:train:none"] = contract
                        by_key[f"{logical}:standalone:{fold}:best_eval:none"] = contract
                        collision_paths.update((str(fold_dir), str(context_path)))
                        cache_root = root / "hidden_features"
                        if output_suffix:
                            cache_root = cache_root / output_suffix
                        cache_dir = cache_root / condition / backbone / dataset / logical / f"fold_{fold}"
                        for method, backend, script, deps, trials in (
                            (
                                "logreg",
                                f"{'gemma4' if backbone == 'gemma4' else 'qwen'}_hidden_logreg_raw",
                                "scripts/run_native_en_logreg_slurm.sh",
                                [f"{logical}:standalone:{fold}:best_eval:none"],
                                None,
                            ),
                            (
                                "xgb_optuna100",
                                f"{'gemma4' if backbone == 'gemma4' else 'qwen'}_hidden_xgb_optuna100",
                                "scripts/run_native_en_xgb_slurm.sh",
                                [f"{logical}:standalone:{fold}:head:logreg"],
                                2 if stage == "smoke" else 100,
                            ),
                        ):
                            method_name = "xgb_optuna" if method == "xgb_optuna100" else method
                            head_id = new_attempt_id(f"{logical}_{method}", str(deployment["git_commit"]))
                            head_dir = fold_dir / head_id
                            hp = root / "contexts" / head_id / f"fold_{fold}" / "context.json"
                            config_path = hp.with_name("config.json")
                            parent_path = hp.with_name("parent.json")
                            hctx, hcfg, hparent = head_payload(
                                deployment=deployment,
                                logical=f"{logical}_{method}",
                                attempt_id=head_id,
                                fold=fold,
                                seed=seed,
                                stage=stage,
                                condition=condition,
                                backbone=backbone,
                                method=method_name,
                                backend=backend,
                                config=config,
                                manifest_hash=pair.get("manifest_hash"),
                                split_hash=pair.get("metadata_sha256"),
                                parent_attempt_id=attempt_id,
                                parent_checkpoint_path=str(fold_dir / "best_model"),
                                optuna_trials=trials,
                            )
                            head = entry(
                                attempt_id=head_id,
                                attempt_dir=head_dir,
                                context_path=hp,
                                config_path=config_path,
                                parent_path=parent_path,
                                logical=f"{logical}_{method}",
                                job_key=f"{logical}:standalone:{fold}:head:{method}",
                                job_type="hidden_classifier",
                                dependencies=deps,
                                script=script,
                                seed=seed,
                                fold=fold,
                                condition=condition,
                                backbone=backbone,
                                endpoint="standalone",
                                method=method,
                                stage=stage,
                                config_remote=config_remote,
                                overrides=overrides,
                                checkpoint_dir=fold_dir / "best_model",
                                cache_dir=cache_dir,
                                trials=trials,
                            )
                            head.update({
                                "context_payload": hctx,
                                "config_payload": hcfg,
                                "parent_payload": hparent,
                                "parent_attempt_id": attempt_id,
                            })
                            register(head)
                            collision_paths.update((str(head_dir), str(hp), str(cache_dir)))

    for condition in CONDITIONS:
        for backbone in BACKBONES:
            relative = MERGED_CONFIGS[(condition, backbone)]
            config_remote = rel_config(relative, str(deployment["deployed_code_path"]))
            for seed in seeds:
                stages = ("cv",) if stage == "smoke" else ("cv", "final")
                for merged_stage in stages:
                    logical = f"native_en_text_heads_v2_{condition}_{backbone}_merged_{merged_stage}_s{seed}"
                    folds = (0,) if stage == "smoke" or merged_stage == "final" else (0, 1, 2, 3, 4)
                    overrides = merged_overrides(
                        root=root,
                        stage=stage,
                        condition=condition,
                        backbone=backbone,
                        seed=seed,
                        output_suffix=output_suffix,
                    )
                    base_config = resolved_config(relative, overrides, extra={"dataset": "merged"})
                    merged_info = next(
                        (
                            item for item in preflight.get("merged_configs", [])
                            if item.get("condition") == condition and item.get("backbone") == backbone
                        ),
                        {},
                    )
                    protocol_manifest_hash = (merged_info.get("protocol") or {}).get("manifest_hash")
                    protocol_split_hash = (merged_info.get("protocol") or {}).get("split_hash")
                    for fold in folds:
                        run_root = REMOTE_OUTPUT_ROOT / campaign(stage, condition, backbone, output_suffix) / "text_only" / "merged"
                        train_dir = run_root / logical / f"fold_{fold}"
                        train_id = new_attempt_id(logical, str(deployment["git_commit"]))
                        tp = root / "contexts" / train_id / f"fold_{fold}" / "context.json"
                        train_config_path = tp.with_name("config.json")
                        train_parent_path = tp.with_name("parent.json")
                        tctx, tcfg = job_payload(
                            deployment=deployment,
                            logical=logical,
                            attempt_id=train_id,
                            fold=fold,
                            seed=seed,
                            stage=merged_stage,
                            config=base_config,
                            required_job="train",
                            tracking_kind="native_en_text_heads_v2_merged_train",
                            manifest_hash=protocol_manifest_hash,
                            split_hash=protocol_split_hash,
                        )
                        train = entry(
                            attempt_id=train_id,
                            attempt_dir=train_dir,
                            context_path=tp,
                            config_path=train_config_path,
                            parent_path=train_parent_path,
                            logical=logical,
                            job_key=f"{logical}:merged{'_final' if merged_stage == 'final' else ''}:{fold}:train:none",
                            job_type="train",
                            dependencies=[],
                            script="scripts/run_native_en_merged_train_slurm.sh",
                            seed=seed,
                            fold=fold,
                            condition=condition,
                            backbone=backbone,
                            endpoint="merged_final" if merged_stage == "final" else "merged_cv",
                            method=None,
                            stage=merged_stage,
                            config_remote=config_remote,
                            overrides=overrides,
                            run_id=logical,
                        )
                        train.update({"context_payload": tctx, "config_payload": tcfg, "parent_payload": {}, "parent_attempt_id": None})
                        register(train)
                        collision_paths.update((str(train_dir), str(tp)))
                        aux = merged_root(root, condition, backbone, output_suffix) / logical / f"fold_{fold}"
                        post_id = new_attempt_id(f"{logical}_postprocess", str(deployment["git_commit"]))
                        pp = root / "contexts" / post_id / f"fold_{fold}" / "context.json"
                        post_config_path = pp.with_name("config.json")
                        post_parent_path = pp.with_name("parent.json")
                        pctx, pcfg = job_payload(
                            deployment=deployment,
                            logical=f"{logical}_postprocess",
                            attempt_id=post_id,
                            fold=fold,
                            seed=seed,
                            stage=merged_stage,
                            config=base_config,
                            required_job="postprocess",
                            tracking_kind="native_en_text_heads_v2_merged_postprocess",
                            manifest_hash=protocol_manifest_hash,
                            split_hash=protocol_split_hash,
                        )
                        pparent = {"parent_attempt_id": train_id, "parent_checkpoint_path": str(train_dir / "best_model")}
                        post = entry(
                            attempt_id=post_id,
                            attempt_dir=aux,
                            context_path=pp,
                            config_path=post_config_path,
                            parent_path=post_parent_path,
                            logical=f"{logical}_postprocess",
                            job_key=f"{logical}:merged{'_final' if merged_stage == 'final' else ''}:{fold}:postprocess:none",
                            job_type="evaluation",
                            dependencies=[train["job_key"]],
                            script="scripts/run_native_en_merged_postprocess_slurm.sh",
                            seed=seed,
                            fold=fold,
                            condition=condition,
                            backbone=backbone,
                            endpoint=train["endpoint"],
                            method=None,
                            stage=merged_stage,
                            config_remote=config_remote,
                            overrides=overrides,
                            checkpoint_dir=train_dir / "best_model",
                            features_dir=aux / "features",
                            run_id=logical,
                        )
                        post.update({"context_payload": pctx, "config_payload": pcfg, "parent_payload": pparent, "parent_attempt_id": train_id})
                        register(post)
                        collision_paths.update((str(aux), str(pp)))
                        head_keys: dict[str, str] = {}
                        for method, backend, trials in (
                            (
                                "logreg",
                                f"{'gemma4' if backbone == 'gemma4' else 'qwen'}_hidden_logreg_raw_symmetric_merged",
                                None,
                            ),
                            (
                                "xgb_optuna100",
                                f"{'gemma4' if backbone == 'gemma4' else 'qwen'}_hidden_xgb_optuna100_symmetric_merged",
                                2 if stage == "smoke" else 100,
                            ),
                        ):
                            method_name = "xgb_optuna" if method == "xgb_optuna100" else method
                            head_id = new_attempt_id(f"{logical}_{method}", str(deployment["git_commit"]))
                            hp = root / "contexts" / head_id / f"fold_{fold}" / "context.json"
                            head_config_path = hp.with_name("config.json")
                            head_parent_path = hp.with_name("parent.json")
                            hctx, hcfg, hparent = head_payload(
                                deployment=deployment,
                                logical=f"{logical}_{method}",
                                attempt_id=head_id,
                                fold=fold,
                                seed=seed,
                                stage=merged_stage,
                                condition=condition,
                                backbone=backbone,
                                method=method_name,
                                backend=backend,
                                config=base_config,
                                manifest_hash=protocol_manifest_hash,
                                split_hash=protocol_split_hash,
                                parent_attempt_id=post_id,
                                parent_checkpoint_path=str(train_dir / "best_model"),
                                optuna_trials=trials,
                            )
                            deps = [post["job_key"]] if method == "logreg" else [head_keys["logreg"]]
                            head = entry(
                                attempt_id=head_id,
                                attempt_dir=train_dir / head_id,
                                context_path=hp,
                                config_path=head_config_path,
                                parent_path=head_parent_path,
                                logical=f"{logical}_{method}",
                                job_key=f"{logical}:merged{'_final' if merged_stage == 'final' else ''}:{fold}:head:{method}",
                                job_type="hidden_classifier",
                                dependencies=deps,
                                script="scripts/run_native_en_merged_head_slurm.sh",
                                seed=seed,
                                fold=fold,
                                condition=condition,
                                backbone=backbone,
                                endpoint=train["endpoint"],
                                method=method,
                                stage=merged_stage,
                                config_remote=config_remote,
                                overrides=overrides,
                                checkpoint_dir=train_dir / "best_model",
                                features_dir=aux / "features",
                                trials=trials,
                                run_id=logical,
                            )
                            head.update({"context_payload": hctx, "config_payload": hcfg, "parent_payload": hparent, "parent_attempt_id": post_id})
                            register(head)
                            head_keys[method] = head["job_key"]
                            collision_paths.update((str(head["attempt_dir"]), str(hp)))

    standalone_count = sum(job.get("kind") == "standalone_backbone" for job in jobs)
    actual = {
        "total": len(jobs) + standalone_count,
        "train": standalone_count + sum(job["job_type"] == "train" for job in jobs if job.get("kind") != "standalone_backbone"),
        "best_eval": standalone_count,
        "postprocess": sum(
            job.get("job_type") == "evaluation"
            for job in jobs
            if job.get("kind") != "standalone_backbone"
        ),
        "logreg": sum(job.get("method") == "logreg" for job in jobs),
        "xgb_optuna100": sum(job.get("method") == "xgb_optuna100" for job in jobs),
    }
    expected = matrix_counts(stage)
    if actual != expected:
        raise OrchestrationError(f"expanded counts differ: expected={expected} actual={actual}")
    return {
        "schema_version": "native_en_text_heads_v2_submission_plan.v1",
        "group_id": GROUP_ID,
        "experiment_id": experiment_id,
        "deployment_id": deployment["deployment_id"],
        "source_commit": deployment.get("git_commit"),
        "stage": stage,
        "output_suffix": output_suffix or None,
        "retry_from_deployment_id": retry_from.get("deployment_id") if retry_from else None,
        "stage_root": str(root),
        "counts": expected,
        "matrix": matrix_payload(stage),
        "jobs": jobs,
        "collision_paths": sorted(collision_paths),
        "preflight": preflight,
    }


def payload_b64(value: Any) -> str:
    return base64.b64encode((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()).decode()


def save_plan(plan: dict[str, Any], stage: str) -> Path:
    path = plan_path(stage, str(plan.get("deployment_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("deployment_id") != plan.get("deployment_id") or old.get("source_commit") != plan.get("source_commit"):
            raise OrchestrationError(f"submission plan collision: {path}")
        # Do not replace an existing submitted plan.
        if old.get("submission_complete"):
            raise OrchestrationError(f"stage already submitted; refusing to reuse {path}")
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_local_contracts(plan: dict[str, Any]) -> None:
    for job in plan["jobs"]:
        if job.get("kind") != "standalone_backbone":
            continue
        target = PROJECT_ROOT / "outputs" / "exp_submit" / job["attempt_id"]
        target.mkdir(parents=True, exist_ok=False)
        contract = dict(job)
        contract["context"] = job["context"]
        (target / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "context.json").write_text(json.dumps(job["context"], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_once_function() -> str:
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


def manifest_map(
    experiment_id: str, stage: str, output_suffix: str | None = None
) -> dict[str, Any]:
    root = stage_root(experiment_id, stage)
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        for backbone in BACKBONES:
            for dataset in MERGED_DATASETS:
                manifest, metadata = manifest_paths(root, condition, backbone, dataset)
                result[f"{condition}/{backbone}/{dataset}"] = {
                    "manifest": str(manifest),
                    "metadata": str(metadata),
                }
            result[f"merged/{condition}/{backbone}"] = {
                "merged_root": str(merged_root(root, condition, backbone, output_suffix))
            }
    return result


def remote_prepare_script(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    manifest_map_payload: dict[str, Any],
    preflight_path: Path,
) -> str:
    code = str(deployment["deployed_code_path"])
    root = Path(plan["stage_root"])
    map_path = root / f"manifest_map_{deployment['deployment_id']}.json"
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"source {q(code)}/scripts/native_en_text_heads_env.sh",
        f"export NATIVE_EN_TEXT_HEADS_SOURCE_COMMIT={q(deployment.get('git_commit') or '')}",
        "export TRANSLATION_ROOT=" + "$" + "{TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}",
        f"cd {q(code)}",
        write_once_function(),
        f"write_once {q(map_path)} {q(payload_b64(manifest_map_payload))}",
        f"if [ -f {q(preflight_path)} ]; then",
        f"  python - {q(preflight_path)} <<'PY'",
        "import json, sys",
        "payload = json.load(open(sys.argv[1], encoding='utf-8'))",
        "if payload.get('status') != 'passed': raise SystemExit('existing preflight is not passed')",
        f"if payload.get('deployment_id') != {plan['deployment_id']!r}: raise SystemExit('existing preflight deployment mismatch')",
        "print('existing preflight passed')",
        "PY",
        "  exit 0",
        "fi",
    ]
    seen: set[tuple[str, str, str]] = set()
    for condition in CONDITIONS:
        for backbone in BACKBONES:
            relative = MERGED_CONFIGS[(condition, backbone)]
            config = load_yaml_with_overrides(PROJECT_ROOT / relative, [])
            overrides = merged_overrides(
                root=root,
                stage=plan["stage"],
                condition=condition,
                backbone=backbone,
                seed=1337,
                output_suffix=plan.get("output_suffix"),
            )
            component_targets: list[Path] = []
            for item in config.get("components") or []:
                dataset = str(item["name"]).lower()
                key = (condition, backbone, dataset)
                if key in seen:
                    continue
                seen.add(key)
                manifest, metadata = manifest_paths(root, condition, backbone, dataset)
                component_targets.extend((manifest, metadata))
            config_remote = rel_config(relative, code)
            merged_outputs = (
                merged_root(root, condition, backbone, plan.get("output_suffix")) / "merged_manifest.jsonl",
                merged_root(root, condition, backbone, plan.get("output_suffix")) / "merged_protocol.json",
            )
            lines.append(
                f"if [ -e {q(merged_outputs[0])} ] || [ -e {q(merged_outputs[1])} ]; then"
            )
            for target in (*component_targets, *merged_outputs):
                lines.append(f"  test -f {q(target)}")
            lines.append("  echo 'reusing complete existing merged preparation artifacts'")
            lines.append("else")
            command = [
                "python",
                "scripts/build_symmetric_merged_manifest.py",
                "--config",
                config_remote,
                "--build-components",
                "--skip-existing-components",
            ]
            for token in overrides:
                # Override values are themselves `--set=...` tokens.  Passing
                # them as the next argv item makes argparse treat the value as
                # another option; use the equals form so the value remains
                # attached to `--override` losslessly.
                command.append(f"--override={token}")
            lines.append("  " + " ".join(q(token) for token in command))
            lines.append("fi")
    lines.extend(
        [
            "python scripts/native_en_text_heads_preflight.py"
            f" --run-id {q(plan['experiment_id'] + '-' + plan['stage'])}"
            f" --stage {q(plan['stage'])}"
            f" --deployment-id {q(plan['deployment_id'])}"
            f" --source-manifest-sha256 {q(deployment.get('source_manifest_sha256') or '')}"
            f" --manifest-map {q(map_path)}"
            f" --output {q(preflight_path)}",
            f"cat {q(preflight_path)}",
        ]
    )
    return "\n".join(lines) + "\n"


def local_contracts(plan: dict[str, Any]) -> None:
    for job in plan["jobs"]:
        target = PROJECT_ROOT / "outputs" / "exp_submit" / job["attempt_id"]
        target.mkdir(parents=True, exist_ok=False)
        (target / "contract.json").write_text(
            json.dumps(
                {
                    **job,
                    "experiment_id": plan["experiment_id"],
                    "deployment_id": plan["deployment_id"],
                    "source_commit": plan.get("source_commit"),
                    "remote_evidence_root": (
                        job.get("fold_dir")
                        if job.get("kind") == "standalone_backbone"
                        else job["attempt_dir"]
                    ),
                    "local_evidence_rel": (
                        job.get("local_fold_rel")
                        if job.get("kind") == "standalone_backbone"
                        else (
                            str(Path(job["attempt_dir"]).relative_to(REMOTE_PROJECT_ROOT))
                            if str(job["attempt_dir"]).startswith(str(REMOTE_PROJECT_ROOT) + "/")
                            else f"outputs/native_en_text_heads_v2/{plan['stage']}/attempts/{job['attempt_id']}"
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        context = job.get("context") or job.get("context_payload")
        if context is not None:
            (target / "context.json").write_text(
                json.dumps(context, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def add_plan_indexes(plan: dict[str, Any]) -> None:
    aliases: dict[str, int] = {}
    for index, job in enumerate(plan["jobs"]):
        job["plan_index"] = index
        job["job_var"] = f"jid_{index}"
        job["log_root"] = str(
            Path(plan["stage_root"]) / "logs" / job["job_type"] / job["condition"] / job["backbone"]
        )
        if job.get("kind") == "standalone_backbone":
            job["job_key"] = f"{job['logical_run_name']}:standalone:{job['fold']}:train:none"
            job["eval_job_key"] = f"{job['logical_run_name']}:standalone:{job['fold']}:best_eval:none"
            aliases[job["job_key"]] = index
            aliases[job["eval_job_key"]] = index
        else:
            job["tracking_job_key"] = (
                "head"
                if job.get("method")
                else "train"
                if job.get("job_type") == "train"
                else "postprocess"
            )
            aliases[job["job_key"]] = index
    plan["dependency_index"] = aliases


def job_export(entry: dict[str, Any], deployment: dict[str, Any]) -> str:
    # The locked v2 matrix calls the method ``xgb_optuna100`` so that the
    # attempt identity and qualifier name carry the scientific trial-count
    # contract.  The merged-head implementation itself retains its historical
    # CLI spelling ``xgb_optuna``.  Keep that translation at the Slurm boundary
    # rather than weakening either identity check.
    exported_method = entry.get("method") or ""
    if entry.get("script", "").endswith("run_native_en_merged_head_slurm.sh"):
        exported_method = "xgb_optuna" if exported_method == "xgb_optuna100" else exported_method
    values: dict[str, Any] = {
        "PROJECT_ROOT": deployment["deployed_code_path"],
        "CONFIG": entry["config_remote"],
        "ATTEMPT_DIR": entry["attempt_dir"],
        "CONTEXT_JSON": entry["context_path"],
        "CONFIG_JSON": entry["config_json_path"],
        "PARENT_JSON": entry["parent_json_path"],
        "STAGE": entry["stage"],
        "FOLD": entry["fold"],
        "RUN_ID": entry.get("run_id") or "",
        "METHOD": exported_method,
        "CACHE_DIR": entry.get("cache_dir") or "",
        "FEATURES_DIR": entry.get("features_dir") or "",
        "CHECKPOINT_DIR": entry.get("checkpoint_dir") or "",
        "TRIALS": entry.get("trials") or "",
        "CONDITION": entry.get("condition") or "",
        "BACKBONE": entry.get("backbone") or "",
        "OVERRIDES_JSON_B64": encode_overrides(entry.get("overrides") or []),
        "LOG_ROOT": entry.get("log_root") or str(Path(entry["attempt_dir"]).parent / "logs"),
    }
    if entry["job_type"] == "train":
        values.update({
            "NPROC_PER_NODE": 4,
            "EPOCHS": (
                1
                if entry["stage"] == "smoke"
                else (entry.get("epochs") or "")
            ),
            "SUBJECTS_PER_CLASS": 2 if entry["stage"] == "smoke" else "",
            "ENV_ACTIVATE": GEMMA_ENV_ACTIVATE if entry["backbone"] == "gemma4" else QWEN_ENV_ACTIVATE,
        })
    elif entry["job_type"] == "evaluation":
        values.update({
            "SUBJECTS_PER_CLASS": 2 if entry["stage"] == "smoke" else "",
            "ENV_ACTIVATE": GEMMA_ENV_ACTIVATE if entry["backbone"] == "gemma4" else QWEN_ENV_ACTIVATE,
        })
    elif entry.get("script", "").endswith("logreg_slurm.sh"):
        values.update({
            "ENV_ACTIVATE": GEMMA_ENV_ACTIVATE if entry["backbone"] == "gemma4" else QWEN_ENV_ACTIVATE,
            "MODEL_PATH": GEMMA_MODEL_PATH if entry["backbone"] == "gemma4" else "",
        })
    else:
        values["ENV_ACTIVATE"] = QWEN_ENV_ACTIVATE
    return ",".join(f"{key}={str(value)}" for key, value in values.items())


def custom_init_lines(lines: list[str], job: dict[str, Any]) -> None:
    for key, payload_key in (
        ("context_path", "context_payload"),
        ("config_json_path", "config_payload"),
        ("parent_json_path", "parent_payload"),
    ):
        lines.append(f"write_once {q(job[key])} {q(payload_b64(job[payload_key]))}")
    python = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/python"
    lines.append(
        f"{q(python)} tools/native_en_text_heads_worker.py init"
        f" --attempt-dir {q(job['attempt_dir'])}"
        f" --context {q(job['context_path'])}"
        f" --config {q(job['config_json_path'])}"
        f" --parent {q(job['parent_json_path'])}"
    )
    lines.append(
        f"{q(python)} tools/native_en_text_heads_worker.py transition"
        f" --attempt-dir {q(job['attempt_dir'])}"
        " --to-state DEPLOYED"
        f" --reason {q('managed v2 deployment prepared')}"
    )


def remote_submission_script(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    preflight_path: Path,
    *,
    phase: str = "all",
) -> str:
    code = str(deployment["deployed_code_path"])
    by_index = {int(job["plan_index"]): job for job in plan["jobs"]}
    by_key = dict(plan["dependency_index"])
    lines = [
        "set -euo pipefail",
        "module purge",
        "module load bsc/1.0",
        "module load miniforge/24.3.0-0",
        f"source {q(QWEN_ENV_ACTIVATE)}",
        f"export PROJECT_ROOT={q(code)}",
        f"source {q(code)}/scripts/native_en_text_heads_env.sh",
        f"cd {q(code)}",
        write_once_function(),
        f"test -f {q(preflight_path)}",
        f"python - {q(preflight_path)} <<'PY'",
        "import json, sys",
        "payload = json.load(open(sys.argv[1], encoding='utf-8'))",
        "if payload.get('status') != 'passed': raise SystemExit('preflight is not passed')",
        "PY",
    ]
    selected_jobs = [
        job
        for job in plan["jobs"]
        if phase == "all"
        or (phase == "cv" and job.get("endpoint") != "merged_final")
        or (phase == "final" and job.get("endpoint") == "merged_final")
    ]
    selected_indexes = {int(job["plan_index"]) for job in selected_jobs}
    selected_collision_paths: set[str] = set()
    for job in selected_jobs:
        for key in ("attempt_dir", "context_path", "cache_dir", "features_dir"):
            value = job.get(key)
            if value:
                selected_collision_paths.add(str(value))
        if job.get("kind") == "standalone_backbone":
            selected_collision_paths.add(str(job["fold_dir"]))
    for path in sorted(selected_collision_paths):
        lines.append(f"test ! -e {q(path)}")
    for job in selected_jobs:
        if job.get("kind") != "standalone_backbone":
            custom_init_lines(lines, job)
    for job in selected_jobs:
        index = int(job["plan_index"])
        if job.get("kind") == "standalone_backbone":
            out_var = f"standalone_{index}"
            train_var = f"train_{index}"
            eval_var = f"eval_{index}"
            lines.append(f"write_once {q(job['context_path'])} {q(payload_b64(job['context']))}")
            lines.append(
                f"{out_var}=$(CONFIG={q(job['config_remote'])}"
                f" FOLD={job['fold']}"
                f" RUN_NAME={q(job['logical_run_name'])}"
                f" OVERRIDES_JSON_B64={q(job['overrides_b64'])}"
                f" EXPERIMENT_CONTEXT={q(job['context_path'])}"
                f" LOG_ROOT={q(job['log_root'])}"
                f" ENV_ACTIVATE={q(GEMMA_ENV_ACTIVATE if job['backbone'] == 'gemma4' else QWEN_ENV_ACTIVATE)}"
                f" MODEL_PATH={q(GEMMA_MODEL_PATH if job['backbone'] == 'gemma4' else '')}"
                " SKIP_MANIFEST_BUILD=1"
                f" PROJECT_ROOT={q(code)}"
                " bash scripts/submit_train_and_eval.sh)"
            )
            ref = "$" + out_var
            lines.append(
                f"{train_var}=$(printf '%s\\n' \"{ref}\" | sed -n 's/^Submitted training job: //p' | tail -1)"
            )
            lines.append(
                f"{eval_var}=$(printf '%s\\n' \"{ref}\" | sed -n 's/^Submitted best-checkpoint eval job: //p' | tail -1)"
            )
            lines.append(f'test -n "$' + train_var + '"')
            lines.append(f'test -n "$' + eval_var + '"')
            lines.append(
                "printf '%s %s %s %s\\n' __STANDALONE__ "
                + str(index)
                + ' "$'
                + train_var
                + '" "$'
                + eval_var
                + '"'
            )
        else:
            dep_vars: list[str] = []
            for dependency in job.get("dependencies", []):
                dep_index = by_key.get(dependency)
                if dep_index is None:
                    raise OrchestrationError(
                        f"unknown dependency {dependency} for {job['job_key']}"
                    )
                dep_job = by_index[dep_index]
                dep_vars.append(
                    f"eval_{dep_index}"
                    if dep_job.get("kind") == "standalone_backbone"
                    else f"jid_{dep_index}"
                )
            dependency_arg = ""
            if dep_vars:
                dependency_arg = " --dependency=afterok:" + ":".join("$" + name for name in dep_vars)
            if job.get("endpoint") == "merged_final" and not job.get("epochs"):
                raise OrchestrationError(
                    f"final merged job {job['job_key']} has no CV-derived epoch count"
                )
            command = (
                f"sbatch --parsable --chdir={q(code)}{dependency_arg}"
                f" --export={q('ALL,' + job_export(job, deployment))} {q(job['script'])}"
            )
            job_var = f"jid_{index}"
            lines.append(f"{job_var}=$({command})")
            lines.append(
                f"{job_var}=$(printf '%s\\n' \"$" + job_var + "\" | sed 's/;.*//')"
            )
            lines.append(f'test -n "$' + job_var + '"')
            lines.append(f'echo \'__JOB__ {index} \' "$' + job_var + '"')
            dep_args: list[str] = []
            for dependency in job.get("dependencies", []):
                dep_index = by_key[dependency]
                dep_job = by_index[dep_index]
                var = f"eval_{dep_index}" if dep_job.get("kind") == "standalone_backbone" else f"jid_{dep_index}"
                dep_args.append("--dependency-job-id " + '"' + "$" + var + '"')
            python = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/python"
            lines.append(
                f"{q(python)} tools/native_en_text_heads_worker.py record"
                f" --attempt-dir {q(job['attempt_dir'])}"
                f" --job-key {q(job['tracking_job_key'])}"
                f" --job-type {q('hidden_classifier' if job.get('method') else job['job_type'])}"
                " --event-type SUBMITTED"
                + ' --slurm-job-id "$'
                + job_var
                + '" --status PENDING '
                + " ".join(dep_args)
            )
            lines.append(
                f"{q(python)} tools/native_en_text_heads_worker.py transition"
                f" --attempt-dir {q(job['attempt_dir'])}"
                " --to-state SUBMITTED"
                f" --reason {q('managed v2 Slurm submission')}"
            )
    lines.append(f"echo '__SUBMISSION_COMPLETE__ {len(selected_indexes)}'")
    return "\n".join(lines) + "\n"


def load_remote_preflight(host: str, path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "cat", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise OrchestrationError(f"could not read remote preflight {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def parse_submission_markers(
    plan: dict[str, Any], stdout: str, expected_indexes: set[int] | None = None
) -> None:
    by_index = {int(job["plan_index"]): job for job in plan["jobs"]}
    expected = set(by_index) if expected_indexes is None else set(expected_indexes)
    seen: set[int] = set()
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "__STANDALONE__":
            index = int(fields[1])
            if index not in by_index or index in seen:
                raise OrchestrationError(f"duplicate or unknown standalone submission marker: {line}")
            by_index[index]["job_ids"] = {"train": fields[2], "best_eval": fields[3]}
            seen.add(index)
        elif len(fields) >= 3 and fields[0] == "__JOB__":
            index = int(fields[1])
            if index not in by_index or index in seen:
                raise OrchestrationError(f"duplicate or unknown custom submission marker: {line}")
            by_index[index]["job_ids"] = {"job": fields[2]}
            seen.add(index)
    missing = sorted(index for index in expected if index not in seen)
    if missing:
        raise OrchestrationError(f"submission output is missing job markers for plan indexes: {missing[:20]}")


def _local_source_path(remote_path: str) -> Path:
    target = Path(remote_path)
    try:
        relative = target.relative_to(REMOTE_PROJECT_ROOT)
    except ValueError as exc:
        raise OrchestrationError(
            f"CV evidence path is outside the canonical output root: {remote_path}"
        ) from exc
    return PROJECT_ROOT / relative


def _replace_or_append_set(overrides: list[str], key: str, value: Any) -> list[str]:
    prefix = f"--set={key}="
    result = list(overrides)
    for index, token in enumerate(result):
        if str(token).startswith(prefix):
            result[index] = prefix + str(value)
            return result
    result.append(prefix + str(value))
    return result


def derive_final_epochs(plan: dict[str, Any]) -> Path:
    """Freeze one rounded-median CV epoch per final condition/backbone/seed."""

    if plan.get("stage") != "production":
        raise OrchestrationError("final epoch derivation is only defined for production")
    final_train_jobs = [
        job
        for job in plan.get("jobs", [])
        if job.get("endpoint") == "merged_final" and job.get("job_type") == "train"
    ]
    if not final_train_jobs:
        raise OrchestrationError("production plan has no final merged train jobs")
    audit_rows: list[dict[str, Any]] = []
    epochs_by_panel: dict[tuple[str, str, int], int] = {}
    for final in final_train_jobs:
        panel = (str(final["condition"]), str(final["backbone"]), int(final["seed"]))
        if panel in epochs_by_panel:
            continue
        cv_jobs = [
            job
            for job in plan.get("jobs", [])
            if job.get("endpoint") == "merged_cv"
            and job.get("job_type") == "train"
            and str(job.get("condition")) == panel[0]
            and str(job.get("backbone")) == panel[1]
            and int(job.get("seed")) == panel[2]
        ]
        if {int(job["fold"]) for job in cv_jobs} != {0, 1, 2, 3, 4}:
            raise OrchestrationError(f"CV epoch evidence is incomplete for panel {panel}")
        selected: list[int] = []
        evidence: list[dict[str, Any]] = []
        for job in sorted(cv_jobs, key=lambda item: int(item["fold"])):
            local_train = _local_source_path(str(job["attempt_dir"]))
            selection_path = local_train / "logs" / "selected_checkpoint.json"
            if not selection_path.is_file():
                raise OrchestrationError(
                    f"CV selection artifact is missing for {panel} fold {job['fold']}: {selection_path}"
                )
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            epoch = int(payload.get("selected_epoch", 0))
            if not 1 <= epoch <= 20:
                raise OrchestrationError(
                    f"invalid CV selected_epoch={epoch} for {panel} fold {job['fold']}"
                )
            selected.append(epoch)
            evidence.append(
                {
                    "fold": int(job["fold"]),
                    "attempt_id": job["attempt_id"],
                    "path": str(selection_path),
                    "sha256": sha256_file(selection_path),
                    "selected_epoch": epoch,
                }
            )
        resolved = int(math.floor(float(statistics.median(selected)) + 0.5))
        if not 1 <= resolved <= 20:
            raise OrchestrationError(f"invalid rounded median epoch for {panel}: {selected} -> {resolved}")
        epochs_by_panel[panel] = resolved
        audit_rows.append(
            {
                "condition": panel[0],
                "backbone": panel[1],
                "seed": panel[2],
                "selected_epochs": selected,
                "final_epoch_count": resolved,
                "cv_evidence": evidence,
            }
        )

    for job in plan["jobs"]:
        if job.get("endpoint") != "merged_final":
            continue
        panel = (str(job["condition"]), str(job["backbone"]), int(job["seed"]))
        epoch = epochs_by_panel[panel]
        job["epochs"] = epoch
        job["overrides"] = _replace_or_append_set(
            list(job.get("overrides") or []), "training.final_epoch_count", epoch
        )
        config_payload = job.get("config_payload")
        if isinstance(config_payload, dict):
            config_payload.setdefault("training", {})["final_epoch_count"] = epoch

    audit = {
        "schema_version": "native_en_text_heads_v2_final_epoch_audit.v1",
        "status": "passed",
        "deployment_id": plan["deployment_id"],
        "source_commit": plan.get("source_commit"),
        "group_id": plan["group_id"],
        "policy": "rounded_median_selected_epoch",
        "panels": sorted(audit_rows, key=lambda row: (row["condition"], row["backbone"], row["seed"])),
    }
    audit_path = PROJECT_ROOT / "outputs" / "native_en_text_heads_v2" / "production" / "final_epoch_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing != audit:
            raise OrchestrationError(f"refusing to overwrite incompatible final epoch audit: {audit_path}")
    else:
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan["final_epoch_audit_path"] = str(audit_path)
    plan["final_epochs"] = {
        f"{condition}/{backbone}/s{seed}": epoch
        for (condition, backbone, seed), epoch in sorted(epochs_by_panel.items())
    }
    return audit_path


def plan_path(stage: str, deployment_id: str | None = None) -> Path:
    root = PROJECT_ROOT / "outputs" / "native_en_text_heads_v2" / stage
    if deployment_id:
        return root / f"submission_plan_{deployment_id}.json"
    standard = root / "submission_plan.json"
    if standard.is_file():
        return standard
    candidates = sorted(root.glob("submission_plan_*.json"))
    return candidates[-1] if candidates else standard


def preflight_path_for(experiment_id: str, stage: str, deployment_id: str) -> Path:
    return stage_root(experiment_id, stage) / f"preflight_{deployment_id}.json"


def write_local_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise OrchestrationError(f"refusing to overwrite incompatible local orchestration file: {path}")
    path.write_text(text, encoding="utf-8")


def print_plan_summary(plan: dict[str, Any], path: Path) -> None:
    print("=== native_en_text_heads_v2 plan ===")
    print(f"stage: {plan['stage']}")
    print(f"experiment_id: {plan['experiment_id']}")
    print(f"deployment_id: {plan['deployment_id']}")
    print(f"source_commit: {plan['source_commit']}")
    print(f"output_suffix: {plan.get('output_suffix') or '<canonical>'}")
    if plan.get("retry_from_plan"):
        print(f"retry_from_plan: {plan['retry_from_plan']}")
    print(f"stage_root: {plan['stage_root']}")
    print(f"counts: {json.dumps(plan['counts'], sort_keys=True)}")
    print(f"plan: {path}")


def _load_plan_for_status(slug: str | None, stage: str, deployment_id: str | None = None) -> dict[str, Any]:
    path = plan_path(stage, deployment_id)
    if not path.is_file():
        raise OrchestrationError(f"no local v2 submission plan exists for stage {stage}: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if slug:
        import tools.exp as exp

        _worktree, pin = exp._resolve_lane(slug)
        expected = str((pin or {}).get("experiment_id") or slug)
        if plan.get("experiment_id") != expected:
            raise OrchestrationError(
                f"plan experiment_id {plan.get('experiment_id')!r} does not match lane {expected!r}"
            )
    return plan


def _load_retry_source(
    path_value: str | None, *, stage: str, experiment_id: str
) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise OrchestrationError(f"retry source plan does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("experiment_id") != experiment_id:
        raise OrchestrationError(
            "retry source plan does not match the requested lane/stage: "
            f"experiment_id={payload.get('experiment_id')!r}, stage={payload.get('stage')!r}"
        )
    payload["_source_plan_path"] = str(path)
    return payload


def command_plan(args: argparse.Namespace) -> int:
    try:
        _worktree, _pin, deployment = load_deployment(args.slug, args.deployment_id, execute=False)
        experiment_id = str(deployment.get("experiment_id") or args.slug)
        retry_from = _load_retry_source(
            getattr(args, "retry_from", None), stage=args.stage, experiment_id=experiment_id
        )
        plan = build_plan(
            stage=args.stage,
            deployment=deployment,
            experiment_id=experiment_id,
            output_suffix=getattr(args, "output_suffix", None),
            retry_from=retry_from,
        )
        if retry_from:
            plan["retry_from_plan"] = retry_from["_source_plan_path"]
        add_plan_indexes(plan)
        path = save_plan(plan, args.stage)
    except (OrchestrationError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_plan_summary(plan, path)
    print("planning complete; no remote mutation performed")
    return 0


def command_submit(args: argparse.Namespace) -> int:
    if bool(args.dry_run) == bool(args.execute):
        print("ERROR: specify exactly one of --dry-run or --execute", file=sys.stderr)
        return 1
    phase = str(getattr(args, "phase", "all"))
    if args.stage == "smoke" and phase != "all":
        print("ERROR: smoke submission has no deferred final phase", file=sys.stderr)
        return 1
    if args.stage == "production" and phase not in {"cv", "final", "all"}:
        print(f"ERROR: unsupported production phase {phase!r}", file=sys.stderr)
        return 1
    execute = bool(args.execute)
    try:
        _worktree, _pin, deployment = load_deployment(args.slug, args.deployment_id, execute=execute)
        experiment_id = str(deployment.get("experiment_id") or args.slug)
        retry_from = _load_retry_source(
            getattr(args, "retry_from", None), stage=args.stage, experiment_id=experiment_id
        )
        if retry_from and args.stage == "production" and phase == "final":
            raise OrchestrationError("retry source plans are supported for a new CV/all submission, not final phase")
        if execute:
            transfer = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
            verified = verify_deployment(
                transfer,
                str(deployment["deployment_id"]),
                expected_git_commit=deployment.get("git_commit"),
                expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
            )
            print(
                "deployment verified: "
                f"{verified['tree_verification']['verified_files']}/"
                f"{verified['tree_verification']['expected_files']} files"
            )

        if not execute:
            plan = build_plan(
                stage=args.stage,
                deployment=deployment,
                experiment_id=experiment_id,
                output_suffix=getattr(args, "output_suffix", None),
                retry_from=retry_from,
            )
            if retry_from:
                plan["retry_from_plan"] = retry_from["_source_plan_path"]
            add_plan_indexes(plan)
            path = save_plan(plan, args.stage)
            print_plan_summary(plan, path)
            print(
                "dry-run complete; no manifest build, preflight, context transfer, or sbatch mutation performed"
            )
            return 0

        if args.stage == "production" and phase == "final":
            plan = _load_plan_for_status(args.slug, args.stage, str(deployment.get("deployment_id")))
            if plan.get("deployment_id") != deployment.get("deployment_id"):
                raise OrchestrationError("existing production plan deployment does not match requested deployment")
            if plan.get("submission_phase") != "cv":
                raise OrchestrationError(
                    "final production submission requires a completed CV submission phase"
                )
            derive_final_epochs(plan)
            preflight_path = Path(
                plan.get("preflight_path")
                or preflight_path_for(experiment_id, args.stage, str(deployment["deployment_id"]))
            )
            preflight = load_remote_preflight(DEFAULT_TRANSFER_HOST, preflight_path)
            if preflight.get("status") != "passed":
                raise OrchestrationError("final production submission requires a passed remote preflight")
            add_plan_indexes(plan)
            submit_plan = plan
        else:
            plan = build_plan(
                stage=args.stage,
                deployment=deployment,
                experiment_id=experiment_id,
                output_suffix=getattr(args, "output_suffix", None),
                retry_from=retry_from,
            )
            if retry_from:
                plan["retry_from_plan"] = retry_from["_source_plan_path"]
            add_plan_indexes(plan)
            preflight_path = preflight_path_for(experiment_id, args.stage, str(deployment["deployment_id"]))
            prepare_script = remote_prepare_script(
                plan,
                deployment,
                manifest_map(experiment_id, args.stage, plan.get("output_suffix")),
                preflight_path,
            )
            evidence_root = PROJECT_ROOT / "outputs" / "native_en_text_heads_v2" / args.stage
            write_local_once(
                evidence_root / f"prepare_{deployment['deployment_id']}.sh",
                prepare_script,
            )
            print(f"running model-free manifest/preflight preparation on {DEFAULT_SCHEDULER_HOST}")
            prep = ssh_script(DEFAULT_SCHEDULER_HOST, prepare_script, timeout=12 * 60 * 60)
            write_local_once(
                evidence_root / f"prepare_{deployment['deployment_id']}.log",
                "$ ssh " + DEFAULT_SCHEDULER_HOST + " bash -s\n"
                + prep.stdout
                + ("\n[stderr]\n" + prep.stderr if prep.stderr else ""),
            )
            if prep.returncode != 0:
                raise OrchestrationError(
                    f"remote manifest/preflight preparation failed rc={prep.returncode}: {prep.stderr.strip()}"
                )
            preflight = load_remote_preflight(DEFAULT_TRANSFER_HOST, preflight_path)
            if preflight.get("status") != "passed":
                raise OrchestrationError(f"remote preflight did not pass: {preflight.get('failures')}")
            if preflight.get("deployment_id") != deployment.get("deployment_id"):
                raise OrchestrationError("remote preflight deployment identity mismatch")

            plan = build_plan(
                stage=args.stage,
                deployment=deployment,
                experiment_id=experiment_id,
                preflight=preflight,
                output_suffix=getattr(args, "output_suffix", None),
                retry_from=retry_from,
            )
            if retry_from:
                plan["retry_from_plan"] = retry_from["_source_plan_path"]
            add_plan_indexes(plan)
            plan["preflight_path"] = str(preflight_path)
            plan["preflight_audit_sha256"] = preflight.get("audit_sha256")
            submit_plan = plan

        evidence_root = PROJECT_ROOT / "outputs" / "native_en_text_heads_v2" / args.stage
        if args.stage == "production" and phase == "final":
            # The contracts were created during the CV phase.  Requiring them
            # here prevents a final-only submission from silently minting a
            # second set of identities.
            for job in submit_plan["jobs"]:
                if job.get("endpoint") == "merged_final":
                    contract_path = PROJECT_ROOT / "outputs" / "exp_submit" / job["attempt_id"] / "contract.json"
                    if not contract_path.is_file():
                        raise OrchestrationError(f"final attempt contract is missing: {contract_path}")
        else:
            local_contracts(submit_plan)

        submit_script = remote_submission_script(
            submit_plan,
            deployment,
            Path(submit_plan.get("preflight_path") or preflight_path),
            phase=phase,
        )
        evidence_root = PROJECT_ROOT / "outputs" / "native_en_text_heads_v2" / args.stage
        write_local_once(
            evidence_root / f"submit_{deployment['deployment_id']}_{phase}.sh",
            submit_script,
        )
        print(f"submitting v2 {phase} job graph on {args.scheduler_host}")
        submitted = ssh_script(args.scheduler_host, submit_script, timeout=15 * 60)
        write_local_once(
            evidence_root / f"submit_{deployment['deployment_id']}_{phase}.log",
            "$ ssh " + args.scheduler_host + " bash -s\n"
            + submitted.stdout
            + ("\n[stderr]\n" + submitted.stderr if submitted.stderr else ""),
        )
        if submitted.returncode != 0:
            raise OrchestrationError(
                f"remote submission failed rc={submitted.returncode}: {submitted.stderr.strip()}"
            )
        if not any(line.startswith("__SUBMISSION_COMPLETE__") for line in submitted.stdout.splitlines()):
            raise OrchestrationError("remote submission did not emit the completion marker")
        expected_indexes = {
            int(job["plan_index"])
            for job in submit_plan["jobs"]
            if phase == "all"
            or (phase == "cv" and job.get("endpoint") != "merged_final")
            or (phase == "final" and job.get("endpoint") == "merged_final")
        }
        parse_submission_markers(submit_plan, submitted.stdout, expected_indexes)
        plan = submit_plan
        plan["submission_phase"] = "complete" if phase == "all" else phase
        plan["submission_complete"] = phase == "all"
        plan["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        plan["submission_output_path"] = str(evidence_root / f"submit_{deployment['deployment_id']}_{phase}.log")
        path = save_plan(plan, args.stage)
        print_plan_summary(plan, path)
        if args.stage == "production" and phase == "cv":
            print("CV submission complete; final merged submission is gated on collected CV selection evidence")
        else:
            print("submission complete; all jobs in this phase have recorded Slurm IDs")
        return 0
    except Exception as exc:
        if isinstance(exc, (OrchestrationError, ValueError, OSError)):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        raise


def command_status(args: argparse.Namespace) -> int:
    try:
        plan = _load_plan_for_status(args.slug, args.stage, getattr(args, "deployment_id", None))
        from src.experiment_tracking.monitor import MonitorError, SchedulerClient, reconcile_job

        by_job_id: dict[str, dict[str, Any]] = {}
        for job in plan.get("jobs", []):
            for role, job_id in (job.get("job_ids") or {}).items():
                if job_id:
                    by_job_id[str(job_id)] = {
                        "job": job,
                        "role": role,
                        "job_key": job.get("job_key") if role == "job" else role,
                        "job_type": job.get("job_type"),
                        "dependency_job_ids": [],
                    }
        if not by_job_id:
            raise OrchestrationError("submission plan contains no recorded Slurm IDs")
        scheduler = SchedulerClient(host=args.scheduler_host)
        ids = sorted(by_job_id)
        queue = scheduler.squeue(ids)
        accounting = scheduler.sacct(ids)
        missing = sorted(set(ids) - set(queue) - set(accounting))
        if missing:
            raise OrchestrationError(f"jobs missing from both squeue and sacct: {missing}")
        print(f"=== native_en_text_heads_v2 status ({args.stage}) ===")
        print(f"plan: {plan_path(args.stage, plan.get('deployment_id'))}")
        print(f"jobs: {len(ids)}")
        failures = 0
        for job_id in ids:
            record = by_job_id[job_id]
            try:
                rec = reconcile_job(
                    {
                        "slurm_job_id": job_id,
                        "job_key": record["job_key"],
                        "dependency_job_ids": record["dependency_job_ids"],
                    },
                    queue,
                    accounting,
                    artifacts_ok=None,
                )
            except MonitorError as exc:
                raise OrchestrationError(str(exc)) from exc
            state = rec.account_state or rec.queue_state or "UNKNOWN"
            classification = rec.classification or ("running" if rec.queue_state else "pending")
            if rec.terminal_failure:
                failures += 1
            print(
                f"{job_id}\t{record['job_type']}\t{record['job_key']}\t"
                f"{state}\t{rec.exit_code or '-'}\t{classification}"
            )
        if failures:
            print(f"terminal failures: {failures}; retry eligibility is limited to one unchanged transient retry per job")
            return 1
        return 0
    except (OrchestrationError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def command_derive_final_epochs(args: argparse.Namespace) -> int:
    try:
        plan = _load_plan_for_status(args.slug, "production", getattr(args, "deployment_id", None))
        audit_path = derive_final_epochs(plan)
        add_plan_indexes(plan)
        save_plan(plan, "production")
    except (OrchestrationError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"final epoch audit passed: {audit_path}")
    print(json.dumps(plan.get("final_epochs", {}), indent=2, sort_keys=True))
    return 0


def command_report(args: argparse.Namespace) -> int:
    try:
        from tools.native_en_text_heads_report import build_report, render_markdown

        plan = Path(args.plan) if args.plan else plan_path("production")
        output_json = Path(args.output_json)
        output_md = Path(args.output_md)
        report = build_report(plan, {item.strip() for item in args.attempts.split(",") if item.strip()} if args.attempts else None)
        if args.with_timestamp:
            report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(f"summary_rows={len(report['summary'])} seed_detail_rows={len(report['seed_details'])}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="expand and save the locked local matrix plan")
    plan_parser.add_argument("slug")
    plan_parser.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    plan_parser.add_argument("--deployment-id")
    plan_parser.add_argument("--output-suffix", default=None)
    plan_parser.add_argument("--retry-from", default=None, help="completed local submission plan to supersede")
    plan_parser.set_defaults(function=command_plan)

    submit_parser = sub.add_parser("submit", help="prepare, preflight, and submit the locked v2 graph")
    submit_parser.add_argument("slug")
    submit_parser.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    submit_parser.add_argument("--deployment-id")
    submit_parser.add_argument("--output-suffix", default=None)
    submit_parser.add_argument("--retry-from", default=None, help="completed local submission plan to supersede")
    submit_parser.add_argument("--scheduler-host", default=DEFAULT_SCHEDULER_HOST)
    submit_parser.add_argument(
        "--phase",
        choices=("all", "cv", "final"),
        default="all",
        help="production: submit CV first, then final after derive-final-epochs; smoke uses all",
    )
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--execute", action="store_true")
    submit_parser.set_defaults(function=command_submit)

    status_parser = sub.add_parser("status", help="reconcile the plan's recorded jobs with MN5")
    status_parser.add_argument("slug", nargs="?")
    status_parser.add_argument("--deployment-id", default=None)
    status_parser.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    status_parser.add_argument("--scheduler-host", default=DEFAULT_SCHEDULER_HOST)
    status_parser.set_defaults(function=command_status)

    epoch_parser = sub.add_parser(
        "derive-final-epochs",
        help="freeze rounded-median CV selected epochs before final production submission",
    )
    epoch_parser.add_argument("slug")
    epoch_parser.add_argument("--deployment-id", default=None)
    epoch_parser.set_defaults(function=command_derive_final_epochs)

    report_parser = sub.add_parser(
        "report",
        help="aggregate locally REPORTABLE native/English head evidence into JSON and Markdown",
    )
    report_parser.add_argument("--plan", default=None)
    report_parser.add_argument("--attempts", default=None, help="optional comma-separated explicit head attempt IDs")
    report_parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "outputs/native_en_text_heads_v2/reports/native_en_text_heads_v2_report.json"),
    )
    report_parser.add_argument(
        "--output-md",
        default=str(PROJECT_ROOT / "outputs/native_en_text_heads_v2/reports/native_en_text_heads_v2_report.md"),
    )
    report_parser.add_argument("--with-timestamp", action="store_true")
    report_parser.set_defaults(function=command_report)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
