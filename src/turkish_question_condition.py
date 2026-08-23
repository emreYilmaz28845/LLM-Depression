"""Typed matrix and deterministic plan builder for the Turkish question study.

The existing native-versus-English campaign has a different scope and remains
isolated.  This module is the single source of truth for the paired
mixed-question versus negative-only study: twenty backbone cells, three
training seeds, five folds, and one hidden-state LogReg plus one Optuna-100
route per fresh backbone fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


GROUP_ID = "turkish-mixed-vs-negonly-native-en-multimodal-heads-v1-20260823"
EXPERIMENT_ID = "exp-turkish-full-negonly-multimodal-20260823"
PLAN_SCHEMA_VERSION = "audiollm.turkish_question_condition_plan.v1"
PLAN_HASH_SCHEMA_VERSION = "audiollm.turkish_question_condition_plan_hash.v1"
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
EVALUATION_BACKEND = "original_teacher_forced"
METRIC_NAMESPACE = "headline/binary_strict"
PRIMARY_METRIC = "macro_f1"
SECONDARY_METRIC = "positive_f1"
TRAINING_SEEDS = (7, 1337, 2024)
FOLDS = (0, 1, 2, 3, 4)
SMOKE_SEED = 1337
SMOKE_FOLD = 0
SMOKE_CELL_IDS = ("N01", "N02", "N05", "N06", "N08", "N09")
PRODUCTION_XGB_TRIALS = 100
SMOKE_XGB_TRIALS = 2
REMOTE_PROJECT_ROOT = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression")
REMOTE_RUNTIME_ROOT = Path("/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime") / EXPERIMENT_ID
REMOTE_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "output_model"
GROUP_RELATIVE_PATH = (
    "experiments/definitions/"
    "turkish-mixed-vs-negonly-native-en-multimodal-heads-v1-20260823.yaml"
)


class MatrixError(ValueError):
    """Raised when a locked matrix invariant cannot be represented."""


@dataclass(frozen=True)
class BackboneCell:
    cell_id: str
    recording_condition: str
    transcript_condition: str
    modality: str
    backbone: str
    config: str

    @property
    def recording_token(self) -> str:
        return "negonly" if self.recording_condition == "negative_only" else "mixed"

    @property
    def transcript_token(self) -> str:
        return "native" if self.transcript_condition == "not_applicable" else self.transcript_condition

    @property
    def campaign(self) -> str:
        return f"turkish_qcond_v1_{self.backbone}_{self.recording_token}_{self.transcript_token}"


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


def _git_sha(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_token(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if not token or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in token):
        raise MatrixError(f"unsafe matrix token: {value!r}")
    return token


def _load_group(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    path = root / GROUP_RELATIVE_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MatrixError(f"unable to read campaign group: {path}") from exc
    if not isinstance(payload, dict) or payload.get("group_id") != GROUP_ID:
        raise MatrixError(f"invalid or mismatched campaign group: {path}")
    return payload


def load_cells(repo_root: Path | None = None) -> tuple[BackboneCell, ...]:
    """Load and validate the exact twenty public group cells."""

    group = _load_group(repo_root)
    records = group.get("scope", {}).get("backbone_cells")
    if not isinstance(records, list) or len(records) != 20:
        raise MatrixError(f"campaign group must contain exactly 20 backbone cells, got {records!r}")
    cells: list[BackboneCell] = []
    seen_ids: set[str] = set()
    seen_configs: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise MatrixError("backbone cell must be an object")
        required = ("id", "recording_condition", "transcript_condition", "modality", "backbone", "config")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise MatrixError(f"backbone cell is missing {missing}: {raw!r}")
        cell = BackboneCell(
            cell_id=str(raw["id"]),
            recording_condition=str(raw["recording_condition"]),
            transcript_condition=str(raw["transcript_condition"]),
            modality=str(raw["modality"]),
            backbone=str(raw["backbone"]),
            config=str(raw["config"]),
        )
        if cell.cell_id in seen_ids or cell.config in seen_configs:
            raise MatrixError(f"duplicate cell id or config: {cell}")
        if cell.recording_condition not in {"mixed", "negative_only"}:
            raise MatrixError(f"unsupported recording condition: {cell.recording_condition}")
        if cell.transcript_condition not in {"not_applicable", "native", "english"}:
            raise MatrixError(f"unsupported transcript condition: {cell.transcript_condition}")
        if cell.modality not in {"audio_only", "text_only", "audio_text"}:
            raise MatrixError(f"unsupported modality: {cell.modality}")
        if cell.backbone not in {"qwen", "gemma4"}:
            raise MatrixError(f"unsupported backbone: {cell.backbone}")
        if cell.modality == "audio_only" and cell.transcript_condition != "not_applicable":
            raise MatrixError(f"audio-only cell must be not_applicable: {cell.cell_id}")
        if cell.modality != "audio_only" and cell.transcript_condition == "not_applicable":
            raise MatrixError(f"text-bearing cell must declare a transcript condition: {cell.cell_id}")
        if cell.transcript_condition == "english" and cell.modality == "audio_only":
            raise MatrixError(f"English audio-only cell is forbidden: {cell.cell_id}")
        config_path = (repo_root or _repo_root()) / cell.config
        if not config_path.is_file():
            raise MatrixError(f"cell config is missing: {config_path}")
        seen_ids.add(cell.cell_id)
        seen_configs.add(cell.config)
        cells.append(cell)
    if {cell.cell_id for cell in cells} != {f"M{i:02d}" for i in range(1, 11)} | {f"N{i:02d}" for i in range(1, 11)}:
        raise MatrixError("campaign cell IDs are not the locked M01-M10/N01-N10 set")
    if sum(cell.recording_condition == "mixed" for cell in cells) != 10:
        raise MatrixError("locked matrix must contain ten mixed cells")
    if sum(cell.recording_condition == "negative_only" for cell in cells) != 10:
        raise MatrixError("locked matrix must contain ten negative-only cells")
    return tuple(cells)


def _attempt_key(logical_name: str, source_sha: str) -> str:
    return "planned-" + _canonical_sha({"logical_run_name": logical_name, "source_sha": source_sha})[:24]


def _fold_identity(cell: BackboneCell, seed: int, fold: int, source_sha: str) -> FoldIdentity:
    if seed not in TRAINING_SEEDS or fold not in FOLDS:
        raise MatrixError(f"seed/fold outside locked matrix: seed={seed}, fold={fold}")
    language = cell.transcript_token
    stable_prefix = _canonical_sha(
        {"group_id": GROUP_ID, "cell": cell.cell_id, "seed": seed, "fold": fold, "source_sha": source_sha}
    )[:8]
    run_name = (
        f"tqcond_v1_{cell.backbone}_{cell.recording_token}_{language}_{cell.modality}"
        f"_s{seed}_f{fold}_{stable_prefix}"
    )
    campaign = cell.campaign
    run_root = REMOTE_OUTPUT_ROOT / campaign / cell.modality / "turkish"
    fold_dir = run_root / run_name / f"fold_{fold}"
    # Manifest and split evidence are shared across modalities within the same
    # recording/transcript condition.  The model-specific run root remains
    # isolated; this avoids four concurrent builders racing to materialize the
    # same subject/fold contract.
    path_token = f"{cell.recording_token}/{language}"
    manifest_dir = REMOTE_RUNTIME_ROOT / "manifests" / path_token
    split_dir = REMOTE_RUNTIME_ROOT / "splits" / path_token
    backbone_key = f"{run_name}:backbone"
    logreg_key = f"{run_name}:logreg"
    xgb_key = f"{run_name}:xgb_optuna100"
    logreg_attempt_key = _attempt_key(f"{run_name}_logreg", source_sha)
    xgb_attempt_key = _attempt_key(f"{run_name}_xgb_optuna100", source_sha)
    backbone_attempt_key = _attempt_key(run_name, source_sha)
    hidden_cache_dir = fold_dir / logreg_attempt_key / "hidden_cache"
    return FoldIdentity(
        cell=cell,
        seed=seed,
        fold=fold,
        run_name=run_name,
        campaign=campaign,
        run_root=str(run_root),
        fold_dir=str(fold_dir),
        manifest_dir=str(manifest_dir),
        split_dir=str(split_dir),
        backbone_attempt_key=backbone_attempt_key,
        logreg_attempt_key=logreg_attempt_key,
        xgb_attempt_key=xgb_attempt_key,
        hidden_cache_dir=str(hidden_cache_dir),
    )


def _overrides(identity: FoldIdentity, stage: str) -> list[str]:
    if stage not in {"smoke", "production"}:
        raise MatrixError(f"unsupported plan stage: {stage!r}")
    values = [
        f"--set=output_dirs.manifest_dir={identity.manifest_dir}",
        f"--set=output_dirs.split_dir={identity.split_dir}",
        f"--set=output_dirs.run_root={identity.run_root}",
        f"--set=seed={identity.seed}",
        f"--set=evaluation.evaluation_view={EVALUATION_VIEW}",
    ]
    if stage == "smoke":
        values.extend(("--set=training.num_train_epochs=1", "--set=split.smoke_subject_limit=6"))
    return values


def _config_summary(repo_root: Path, cell: BackboneCell) -> dict[str, Any]:
    config = yaml.safe_load((repo_root / cell.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise MatrixError(f"config is not an object: {cell.config}")
    evaluation = config.get("evaluation") or {}
    split = config.get("split") or {}
    if evaluation.get("evaluation_view") != EVALUATION_VIEW:
        raise MatrixError(f"config lacks the locked evaluation view: {cell.config}")
    if evaluation.get("sample_prediction_mode") != EVALUATION_BACKEND:
        raise MatrixError(f"config lacks teacher-forced evaluation: {cell.config}")
    if int(split.get("outer_folds", -1)) != len(FOLDS) or int(split.get("seed", -1)) != 1337:
        raise MatrixError(f"config has an unlocked split contract: {cell.config}")
    return {
        "dataset": config.get("dataset"),
        "dataset_variant": config.get("dataset_variant", "mixed"),
        "model_backend": config.get("model_backend", "qwen"),
        "use_audio": bool((config.get("data") or {}).get("use_audio")),
        "use_text": bool((config.get("data") or {}).get("use_text")),
        "evaluation_view": evaluation.get("evaluation_view"),
        "sample_prediction_mode": evaluation.get("sample_prediction_mode"),
        "config_sha256": hashlib.sha256((repo_root / cell.config).read_bytes()).hexdigest(),
    }


def _job(
    *,
    identity: FoldIdentity,
    stage: str,
    route: str,
    job_type: str,
    job_key: str,
    dependencies: Iterable[str],
    attempt_key: str,
    resource_shape: str,
    expected_artifacts: list[str],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_key,
        "job_key": job_key,
        "job_type": job_type,
        "route": route,
        "stage": stage,
        "cell_id": identity.cell.cell_id,
        "recording_condition": identity.cell.recording_condition,
        "transcript_condition": identity.cell.transcript_condition,
        "modality": identity.cell.modality,
        "backbone": identity.cell.backbone,
        "seed": identity.seed,
        "fold": identity.fold,
        "config": identity.cell.config,
        "run_name": identity.run_name,
        "campaign": identity.campaign,
        "run_root": identity.run_root,
        "fold_dir": identity.fold_dir,
        "manifest_dir": identity.manifest_dir,
        "split_dir": identity.split_dir,
        "attempt_key": attempt_key,
        "hidden_cache_dir": identity.hidden_cache_dir,
        "dependencies": list(dependencies),
        "resource_shape": resource_shape,
        "evaluation_view": EVALUATION_VIEW,
        "evaluation_backend": EVALUATION_BACKEND,
        "metric_namespace": METRIC_NAMESPACE,
        "checkpoint_role": "best_model",
        "aggregation": "subject_level",
        "expected_artifacts": expected_artifacts,
        "overrides": list(overrides or []),
    }


def build_plan(
    *,
    stage: str = "production",
    source_sha: str | None = None,
    deployment_id: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Expand the locked matrix into a deterministic, hashable plan."""

    root = (repo_root or _repo_root()).resolve()
    source = source_sha or _git_sha(root)
    if len(source) != 40 or any(char not in "0123456789abcdef" for char in source):
        raise MatrixError(f"source_sha must be a full lowercase Git SHA, got {source!r}")
    cells = load_cells(root)
    selected_cells = cells if stage == "production" else tuple(cell for cell in cells if cell.cell_id in SMOKE_CELL_IDS)
    if stage not in {"smoke", "production"}:
        raise MatrixError(f"unsupported plan stage: {stage!r}")
    seeds = TRAINING_SEEDS if stage == "production" else (SMOKE_SEED,)
    folds = FOLDS if stage == "production" else (SMOKE_FOLD,)
    xgb_trials = PRODUCTION_XGB_TRIALS if stage == "production" else SMOKE_XGB_TRIALS
    jobs: list[dict[str, Any]] = []
    identities: list[FoldIdentity] = []
    config_summaries: dict[str, dict[str, Any]] = {}
    for cell in selected_cells:
        config_summaries[cell.cell_id] = _config_summary(root, cell)
        for seed in seeds:
            for fold in folds:
                identity = _fold_identity(cell, seed, fold, source)
                identities.append(identity)
                common = _overrides(identity, stage)
                train_key = f"{identity.run_name}:train"
                eval_key = f"{identity.run_name}:best_eval"
                logreg_key = f"{identity.run_name}:logreg"
                xgb_key = f"{identity.run_name}:xgb_optuna100"
                jobs.append(
                    _job(
                        identity=identity,
                        stage=stage,
                        route="backbone_train",
                        job_type="train",
                        job_key=train_key,
                        dependencies=(),
                        attempt_key=identity.backbone_attempt_key,
                        resource_shape="1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4 (DDP)",
                        expected_artifacts=["run_config.yaml", "metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "best_model"],
                        overrides=common,
                    )
                )
                jobs.append(
                    _job(
                        identity=identity,
                        stage=stage,
                        route="teacher_forced",
                        job_type="evaluation",
                        job_key=eval_key,
                        dependencies=(train_key,),
                        attempt_key=identity.backbone_attempt_key,
                        resource_shape="1 node, 1 task, 1 H100",
                        expected_artifacts=["best_model/standalone_eval/final_summary.json", "best_model/standalone_eval/predictions_subject_level.csv", "best_model/standalone_eval/metrics_subject_level.json"],
                        overrides=common,
                    )
                )
                jobs.append(
                    _job(
                        identity=identity,
                        stage=stage,
                        route="logreg",
                        job_type="hidden_extraction_logreg",
                        job_key=logreg_key,
                        dependencies=(eval_key,),
                        attempt_key=identity.logreg_attempt_key,
                        resource_shape="1 node, 1 task, 1 H100",
                        expected_artifacts=["metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "hidden_cache/extraction_metadata.json", "classifier/logreg_raw/metrics.json", "classifier/logreg_raw/predictions_subject_level.csv"],
                    )
                )
                jobs.append(
                    _job(
                        identity=identity,
                        stage=stage,
                        route="xgb_optuna100",
                        job_type="xgb_optuna100",
                        job_key=xgb_key,
                        dependencies=(logreg_key,),
                        attempt_key=identity.xgb_attempt_key,
                        resource_shape="1 node, 1 task, 0 GPU, 20 CPUs",
                        expected_artifacts=["metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json", "xgb_optuna100_harmonized_v1/trials.csv", "xgb_optuna100_harmonized_v1/study.sqlite3", "xgb_optuna100_harmonized_v1/classifier_metadata.json", "xgb_optuna100_harmonized_v1/metrics.json", "xgb_optuna100_harmonized_v1/predictions_subject_level.csv"],
                    )
                )
    backbone_count = len(identities)
    expected = {
        "cells": len(selected_cells),
        "seeds": len(seeds),
        "folds": len(folds),
        "backbone_fold_runs": backbone_count,
        "teacher_forced_evaluations": backbone_count,
        "logreg_routes": backbone_count,
        "xgb_routes": backbone_count,
        "slurm_jobs": len(jobs),
        "xgb_completed_trials": xgb_trials,
    }
    expected_total = expected["cells"] * expected["seeds"] * expected["folds"]
    if backbone_count != expected_total or len(jobs) != 4 * expected_total:
        raise MatrixError(f"plan cardinality mismatch: expected {expected_total} fold units and {4 * expected_total} jobs")
    all_keys = [str(job["job_key"]) for job in jobs]
    if len(all_keys) != len(set(all_keys)):
        raise MatrixError("duplicate job key in campaign plan")
    all_attempt_keys = [str(job["attempt_key"]) for job in jobs if job["job_type"] != "evaluation"]
    if len(all_attempt_keys) != len(set(all_attempt_keys)):
        raise MatrixError("duplicate planned attempt key in campaign plan")
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "group_id": GROUP_ID,
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "source_git_sha": source,
        "deployment_id": deployment_id,
        "group_definition": GROUP_RELATIVE_PATH,
        "runtime_root": str(REMOTE_RUNTIME_ROOT),
        "output_root": str(REMOTE_OUTPUT_ROOT),
        "evaluation": {
            "backend": EVALUATION_BACKEND,
            "view": EVALUATION_VIEW,
            "namespace": METRIC_NAMESPACE,
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "aggregation": "fold_mean_subject_level",
            "checkpoint_role": "best_model",
        },
        "protocol": {
            "threshold": 17,
            "split_protocol": "train_val",
            "split_seed": 1337,
            "backbone_selection_metric": "inner_val_macro_f1",
            "backbone_selection_mode": "max",
            "audio_encoder": "frozen",
            "xgb_protocol_profile": "harmonized_optuna100_v1",
            "xgb_completed_trials": xgb_trials,
            "xgb_inner_folds": 3,
            "xgb_sampler_seed": 1337,
            "xgb_model_seed": 1337,
            "xgb_inner_split_seed": 1337,
            "xgb_sampling_mode": "none",
        },
        "selected_cell_ids": [cell.cell_id for cell in selected_cells],
        "config_summaries": config_summaries,
        "expected_counts": expected,
        "jobs": jobs,
    }
    plan["plan_sha256"] = _canonical_sha(plan)
    return plan


def write_plan(plan: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    """Write a deterministic JSON plan and a separate hash sidecar."""

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    without_hash = dict(plan)
    without_hash.pop("plan_sha256", None)
    expected_hash = _canonical_sha(without_hash)
    if str(plan.get("plan_sha256")) != expected_hash:
        raise MatrixError("plan_sha256 does not match canonical plan content")
    text = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise MatrixError(f"refusing to overwrite an incompatible plan: {target}")
    target.write_text(text, encoding="utf-8")
    sidecar = target.with_name(target.name + ".sha256")
    sidecar_text = json.dumps(
        {"schema_version": PLAN_HASH_SCHEMA_VERSION, "plan_path": target.name, "plan_sha256": expected_hash},
        indent=2,
        sort_keys=True,
    ) + "\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != sidecar_text:
        raise MatrixError(f"refusing to overwrite an incompatible plan hash sidecar: {sidecar}")
    sidecar.write_text(sidecar_text, encoding="utf-8")
    return target, sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "production"), default="production")
    parser.add_argument("--source-sha")
    parser.add_argument("--deployment-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(stage=args.stage, source_sha=args.source_sha, deployment_id=args.deployment_id)
    if args.output:
        write_plan(plan, args.output)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "expected_counts": plan["expected_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
