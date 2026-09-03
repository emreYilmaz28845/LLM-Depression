"""Locked matrix definition for Turkish pooled question-conditioned training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


GROUP_ID = "turkish-pooled-qcond-clean-v1-20260903"
EXPERIMENT_ID = "exp-turkish-pooled-qcond-clean-v1-20260903"
PLAN_SCHEMA_VERSION = "audiollm.turkish_pooled_qcond_plan.v1"
PLAN_HASH_SCHEMA_VERSION = "audiollm.turkish_pooled_qcond_plan_hash.v1"
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
EVALUATION_BACKEND = "original_teacher_forced"
METRIC_NAMESPACE = "headline/binary_strict"
PRIMARY_METRIC = "macro_f1"
SECONDARY_METRIC = "positive_f1"
PAIR_POLICY = "turkish_pooled_text_pair_mean_margin_strict_v1"
TRAINING_SEEDS = (7, 1337, 2024)
FOLDS = (0, 1, 2, 3, 4)
SMOKE_SEED = 1337
SMOKE_FOLD = 0
SMOKE_CELL_IDS = ("Q04", "Q02")
PRODUCTION_XGB_TRIALS = 100
SMOKE_XGB_TRIALS = 2
REMOTE_PROJECT_ROOT = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression")
REMOTE_RUNTIME_ROOT = Path("/gpfs/projects/etur92/ozu647717/experiment_runtime") / EXPERIMENT_ID
REMOTE_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "output_model"
GROUP_RELATIVE_PATH = "experiments/definitions/turkish-pooled-qcond-clean-v1-20260903.yaml"


class MatrixError(ValueError):
    """Raised when the locked pooled matrix cannot be represented safely."""


@dataclass(frozen=True)
class BackboneCell:
    cell_id: str
    modality: str
    transcript_condition: str
    backbone: str
    config: str

    @property
    def language_token(self) -> str:
        return "native" if self.transcript_condition == "not_applicable" else self.transcript_condition

    @property
    def campaign(self) -> str:
        return f"turkish_pooled_qcond_clean_v1_{self.backbone}_{self.language_token}_{self.modality}"


@dataclass(frozen=True)
class FoldIdentity:
    cell: BackboneCell
    seed: int
    fold: int
    run_name: str
    campaign: str
    run_root: str
    fold_dir: str
    manifest_dir: str
    split_dir: str
    backbone_attempt_key: str
    logreg_attempt_key: str
    xgb_attempt_key: str
    hidden_cache_dir: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_group(root: Path) -> dict[str, Any]:
    path = root / GROUP_RELATIVE_PATH
    try:
        group = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MatrixError(f"could not read pooled group definition: {path}") from exc
    if not isinstance(group, dict) or group.get("group_id") != GROUP_ID:
        raise MatrixError(f"pooled group definition does not declare {GROUP_ID}: {path}")
    return group


def load_cells(repo_root: Path | None = None) -> tuple[BackboneCell, ...]:
    root = (repo_root or _repo_root()).resolve()
    records = _load_group(root).get("scope", {}).get("backbone_cells")
    if not isinstance(records, list) or len(records) != 10:
        raise MatrixError(f"pooled group must contain exactly ten cells, got {records!r}")
    cells: list[BackboneCell] = []
    seen_ids: set[str] = set()
    seen_configs: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise MatrixError("pooled cell is not an object")
        required = ("id", "modality", "transcript_condition", "backbone", "config")
        if any(not raw.get(key) for key in required):
            raise MatrixError(f"pooled cell is missing a required field: {raw!r}")
        cell = BackboneCell(
            cell_id=str(raw["id"]), modality=str(raw["modality"]),
            transcript_condition=str(raw["transcript_condition"]),
            backbone=str(raw["backbone"]), config=str(raw["config"]),
        )
        if cell.cell_id in seen_ids or cell.config in seen_configs:
            raise MatrixError(f"duplicate pooled cell id/config: {cell}")
        if cell.modality not in {"audio_only", "text_only", "audio_text"}:
            raise MatrixError(f"unsupported pooled modality: {cell.modality}")
        if cell.backbone not in {"qwen", "gemma4"}:
            raise MatrixError(f"unsupported pooled backbone: {cell.backbone}")
        if cell.modality == "audio_only" and cell.transcript_condition != "not_applicable":
            raise MatrixError(f"pooled audio-only cell must be not_applicable: {cell.cell_id}")
        if cell.modality != "audio_only" and cell.transcript_condition not in {"native", "english"}:
            raise MatrixError(f"pooled text-bearing cell has invalid transcript condition: {cell.cell_id}")
        if not (root / cell.config).is_file():
            raise MatrixError(f"pooled cell config is missing: {root / cell.config}")
        seen_ids.add(cell.cell_id)
        seen_configs.add(cell.config)
        cells.append(cell)
    expected = {f"Q{i:02d}" for i in range(1, 6)} | {f"G{i:02d}" for i in range(1, 6)}
    if {cell.cell_id for cell in cells} != expected:
        raise MatrixError("pooled cell IDs are not the locked Q01-Q05/G01-G05 set")
    if sum(cell.backbone == "qwen" for cell in cells) != 5 or sum(cell.backbone == "gemma4" for cell in cells) != 5:
        raise MatrixError("pooled matrix must contain five Qwen and five Gemma cells")
    if sum(cell.modality == "audio_only" for cell in cells) != 2:
        raise MatrixError("pooled matrix must contain exactly one audio-only cell per backbone")
    return tuple(cells)


def _attempt_key(logical_name: str, source_sha: str) -> str:
    return "planned-" + _canonical_sha({"logical_run_name": logical_name, "source_sha": source_sha})[:24]


def _fold_identity(cell: BackboneCell, seed: int, fold: int, source_sha: str, stage: str) -> FoldIdentity:
    if seed not in TRAINING_SEEDS or fold not in FOLDS:
        raise MatrixError(f"seed/fold outside pooled matrix: {seed}/{fold}")
    if stage not in {"smoke", "production"}:
        raise MatrixError(f"unsupported plan stage {stage!r}")
    suffix = _canonical_sha({"group": GROUP_ID, "cell": asdict(cell), "seed": seed, "fold": fold, "source_sha": source_sha, "stage": stage})[:8]
    stage_token = "smoke" if stage == "smoke" else "prod"
    run_name = f"tpq_{stage_token}_v1_{cell.backbone}_{cell.language_token}_{cell.modality}_s{seed}_f{fold}_{suffix}"
    run_root = REMOTE_OUTPUT_ROOT / cell.campaign / cell.modality / "turkish"
    fold_dir = run_root / run_name / f"fold_{fold}"
    language = cell.language_token
    manifest_dir = REMOTE_RUNTIME_ROOT / "manifests" / language
    split_dir = REMOTE_RUNTIME_ROOT / "splits" / language
    backbone_key = _attempt_key(run_name, source_sha)
    logreg_key = _attempt_key(f"{run_name}_logreg", source_sha)
    xgb_key = _attempt_key(f"{run_name}_xgb_optuna100", source_sha)
    return FoldIdentity(
        cell=cell, seed=seed, fold=fold, run_name=run_name, campaign=cell.campaign,
        run_root=str(run_root), fold_dir=str(fold_dir), manifest_dir=str(manifest_dir),
        split_dir=str(split_dir), backbone_attempt_key=backbone_key,
        logreg_attempt_key=logreg_key, xgb_attempt_key=xgb_key,
        hidden_cache_dir=str(fold_dir / logreg_key / "hidden_cache"),
    )


def _overrides(identity: FoldIdentity, stage: str) -> list[str]:
    if stage not in {"smoke", "production"}:
        raise MatrixError(f"unsupported plan stage {stage!r}")
    result = [
        f"--set=output_dirs.manifest_dir={identity.manifest_dir}",
        f"--set=output_dirs.split_dir={identity.split_dir}",
        f"--set=output_dirs.run_root={identity.run_root}",
        f"--set=seed={identity.seed}",
        f"--set=evaluation.evaluation_view={EVALUATION_VIEW}",
    ]
    if stage == "smoke":
        result.extend(("--set=training.num_train_epochs=1", "--set=split.smoke_subject_limit=6"))
    return result


def _config_summary(root: Path, cell: BackboneCell) -> dict[str, Any]:
    config = yaml.safe_load((root / cell.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise MatrixError(f"pooled config is not an object: {cell.config}")
    evaluation = config.get("evaluation") or {}
    split = config.get("split") or {}
    data = config.get("data") or {}
    if config.get("dataset") != "turkish" or config.get("dataset_variant") != "pooled_t17":
        raise MatrixError(f"pooled config has the wrong dataset identity: {cell.config}")
    if not str(config.get("recipe_id", "")).endswith("_qcond_v1"):
        raise MatrixError(f"pooled config recipe does not end in _qcond_v1: {cell.config}")
    if evaluation.get("evaluation_view") != EVALUATION_VIEW or evaluation.get("sample_prediction_mode") != EVALUATION_BACKEND:
        raise MatrixError(f"pooled config lacks locked evaluation qualifiers: {cell.config}")
    if int(split.get("outer_folds", -1)) != 5 or split.get("cv_protocol") != "train_val":
        raise MatrixError(f"pooled config lacks locked five-fold train_val split: {cell.config}")
    expected_audio = cell.modality in {"audio_only", "audio_text"}
    expected_text = cell.modality in {"text_only", "audio_text"}
    if bool(data.get("use_audio")) != expected_audio or bool(data.get("use_text")) != expected_text:
        raise MatrixError(f"pooled config modality flags disagree with cell {cell.cell_id}")
    if expected_audio and ((config.get("audio_adapter") or {}).get("enabled") or (config.get("audio_adapter") or {}).get("train_projector")):
        raise MatrixError(f"pooled config enables an audio adapter: {cell.config}")
    if expected_text and cell.modality == "text_only" and evaluation.get("subject_score_aggregation") != PAIR_POLICY:
        raise MatrixError(f"pooled text config lacks the exact pair policy: {cell.config}")
    return {
        "dataset": config["dataset"], "dataset_variant": config["dataset_variant"],
        "model_backend": config.get("model_backend", "qwen"), "use_audio": expected_audio,
        "use_text": expected_text, "transcript_condition": cell.transcript_condition,
        "pair_policy": evaluation.get("subject_score_aggregation"),
        "evaluation_view": evaluation["evaluation_view"],
        "sample_prediction_mode": evaluation["sample_prediction_mode"],
        "config_sha256": hashlib.sha256((root / cell.config).read_bytes()).hexdigest(),
    }


def _job(*, identity: FoldIdentity, stage: str, route: str, job_type: str, job_key: str,
         dependencies: Iterable[str], attempt_key: str, resource_shape: str,
         expected_artifacts: list[str], overrides: list[str] | None = None) -> dict[str, Any]:
    return {
        "job_id": job_key, "job_key": job_key, "job_type": job_type, "route": route,
        "stage": stage, "cell_id": identity.cell.cell_id, "modality": identity.cell.modality,
        "transcript_condition": identity.cell.transcript_condition, "backbone": identity.cell.backbone,
        "seed": identity.seed, "fold": identity.fold, "config": identity.cell.config,
        "run_name": identity.run_name, "campaign": identity.campaign, "run_root": identity.run_root,
        "fold_dir": identity.fold_dir, "manifest_dir": identity.manifest_dir, "split_dir": identity.split_dir,
        "attempt_key": attempt_key, "hidden_cache_dir": identity.hidden_cache_dir,
        "dependencies": list(dependencies), "resource_shape": resource_shape,
        "evaluation_view": EVALUATION_VIEW, "evaluation_backend": EVALUATION_BACKEND,
        "metric_namespace": METRIC_NAMESPACE, "checkpoint_role": "best_model",
        "aggregation": "subject_level", "expected_artifacts": expected_artifacts,
        "overrides": list(overrides or []),
    }


def build_plan(*, stage: str = "production", source_sha: str | None = None,
               deployment_id: str | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or _repo_root()).resolve()
    source = source_sha or _git_sha(root)
    if len(source) != 40 or any(char not in "0123456789abcdef" for char in source):
        raise MatrixError(f"source_sha must be a full lowercase Git SHA: {source!r}")
    cells = load_cells(root)
    if stage not in {"smoke", "production"}:
        raise MatrixError(f"unsupported plan stage {stage!r}")
    selected = cells if stage == "production" else tuple(cell for cell in cells if cell.cell_id in SMOKE_CELL_IDS)
    seeds = TRAINING_SEEDS if stage == "production" else (SMOKE_SEED,)
    folds = FOLDS if stage == "production" else (SMOKE_FOLD,)
    trials = PRODUCTION_XGB_TRIALS if stage == "production" else SMOKE_XGB_TRIALS
    jobs: list[dict[str, Any]] = []
    identities: list[FoldIdentity] = []
    summaries = {cell.cell_id: _config_summary(root, cell) for cell in selected}
    for cell in selected:
        for seed in seeds:
            for fold in folds:
                identity = _fold_identity(cell, seed, fold, source, stage)
                identities.append(identity)
                overrides = _overrides(identity, stage)
                train_key = f"{identity.run_name}:train"
                eval_key = f"{identity.run_name}:best_eval"
                logreg_key = f"{identity.run_name}:logreg"
                xgb_key = f"{identity.run_name}:xgb_optuna100"
                jobs.extend((
                    _job(identity=identity, stage=stage, route="backbone_train", job_type="train",
                         job_key=train_key, dependencies=(), attempt_key=identity.backbone_attempt_key,
                         resource_shape="1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4 (DDP)",
                         expected_artifacts=["run_config.yaml", "metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "best_model"], overrides=overrides),
                    _job(identity=identity, stage=stage, route="teacher_forced", job_type="evaluation",
                         job_key=eval_key, dependencies=(train_key,), attempt_key=identity.backbone_attempt_key,
                         resource_shape="1 node, 1 task, 1 H100", expected_artifacts=["best_model/standalone_eval/final_and_best_validation_metrics.json", "best_model/standalone_eval/predictions_subject_level.csv", "best_model/standalone_eval/metrics_original_teacher_forced.json"], overrides=overrides),
                    _job(identity=identity, stage=stage, route="logreg", job_type="hidden_extraction_logreg",
                         job_key=logreg_key, dependencies=(eval_key,), attempt_key=identity.logreg_attempt_key,
                         resource_shape="1 node, 1 task, 1 H100", expected_artifacts=["metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "hidden_cache/extraction_metadata.json", "classifier/logreg_raw/metrics.json", "classifier/logreg_raw/predictions_subject_level.csv"]),
                    _job(identity=identity, stage=stage, route="xgb_optuna100", job_type="xgb_optuna100",
                         job_key=xgb_key, dependencies=(logreg_key,), attempt_key=identity.xgb_attempt_key,
                         resource_shape="1 node, 1 task, 0 GPU, 20 CPUs", expected_artifacts=["metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "xgb_optuna100_harmonized_v1/trials.csv", "xgb_optuna100_harmonized_v1/study.sqlite3", "xgb_optuna100_harmonized_v1/classifier_metadata.json", "xgb_optuna100_harmonized_v1/metrics.json", "xgb_optuna100_harmonized_v1/predictions_subject_level.csv"]),
                ))
    expected_units = len(identities)
    expected_counts = {
        "cells": len(selected), "seeds": len(seeds), "folds": len(folds),
        "backbone_fold_runs": expected_units, "teacher_forced_evaluations": expected_units,
        "logreg_routes": expected_units, "xgb_routes": expected_units,
        "slurm_jobs": len(jobs), "xgb_completed_trials": trials,
    }
    if expected_counts["slurm_jobs"] != expected_units * 4:
        raise MatrixError(f"pooled job cardinality is not four per fold unit: {expected_counts}")
    keys = [str(job["job_key"]) for job in jobs]
    if len(keys) != len(set(keys)):
        raise MatrixError("pooled plan has duplicate job keys")
    attempts = [str(job["attempt_key"]) for job in jobs if job["job_type"] != "evaluation"]
    if len(attempts) != len(set(attempts)):
        raise MatrixError("pooled plan has duplicate planned attempt keys")
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION, "group_id": GROUP_ID, "experiment_id": EXPERIMENT_ID,
        "stage": stage, "source_git_sha": source, "deployment_id": deployment_id,
        "group_definition": GROUP_RELATIVE_PATH, "runtime_root": str(REMOTE_RUNTIME_ROOT),
        "output_root": str(REMOTE_OUTPUT_ROOT),
        "evaluation": {"backend": EVALUATION_BACKEND, "view": EVALUATION_VIEW, "namespace": METRIC_NAMESPACE,
                        "primary_metric": PRIMARY_METRIC, "secondary_metric": SECONDARY_METRIC,
                        "aggregation": "fold_mean_subject_level", "checkpoint_role": "best_model"},
        "protocol": {"dataset_variant": "pooled_t17", "threshold": 17, "split_protocol": "train_val",
                      "split_seed": 1337, "backbone_selection_metric": "inner_val_macro_f1",
                      "backbone_selection_mode": "max", "audio_encoder": "frozen",
                      "text_pair_policy": PAIR_POLICY, "xgb_protocol_profile": "harmonized_optuna100_v1",
                      "xgb_completed_trials": trials, "xgb_inner_folds": 3, "xgb_sampler_seed": 1337,
                      "xgb_model_seed": 1337, "xgb_inner_split_seed": 1337, "xgb_sampling_mode": "none"},
        "selected_cell_ids": [cell.cell_id for cell in selected], "config_summaries": summaries,
        "expected_counts": expected_counts, "jobs": jobs,
    }
    plan["plan_sha256"] = _canonical_sha(plan)
    return plan


def write_plan(plan: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    target = Path(output)
    without_hash = dict(plan)
    without_hash.pop("plan_sha256", None)
    expected_hash = _canonical_sha(without_hash)
    if plan.get("plan_sha256") != expected_hash:
        raise MatrixError("pooled plan hash does not match its contents")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise MatrixError(f"refusing to overwrite incompatible pooled plan: {target}")
    target.write_text(text, encoding="utf-8")
    sidecar = target.with_name(target.name + ".sha256")
    sidecar_text = json.dumps({"schema_version": PLAN_HASH_SCHEMA_VERSION, "plan_path": target.name, "plan_sha256": expected_hash}, indent=2, sort_keys=True) + "\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != sidecar_text:
        raise MatrixError(f"refusing to overwrite incompatible pooled plan hash: {sidecar}")
    sidecar.write_text(sidecar_text, encoding="utf-8")
    return target, sidecar


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "production"), default="production")
    parser.add_argument("--source-sha")
    parser.add_argument("--deployment-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(stage=args.stage, source_sha=args.source_sha, deployment_id=args.deployment_id)
    if args.output:
        write_plan(plan, args.output)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "expected_counts": plan["expected_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
