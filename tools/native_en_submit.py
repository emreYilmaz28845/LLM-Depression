"""Managed submission builders for the native-versus-English text-only study.

Pure, offline-testable helpers used by ``tools/exp.py`` subcommands
``submit-hidden``, ``submit-optuna100``, and ``submit-merged``. No network,
filesystem-mutation, or Git access happens here: every function either
derives paths/specs from explicit inputs or renders strings for the caller
to upload and execute remotely.
"""

from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

REMOTE_PROJECT_BASE = "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression"

STUDY_SEEDS = (7, 1337, 2024)
STUDY_SPLIT_SEED = 1337
STUDY_HEAD_SEED = 1337
STUDY_OPTUNA_SMOKE_TRIALS = 2

CONDITIONS = ("native", "english")
BACKBONES = ("qwen", "gemma4")
STANDALONE_DATASETS = ("d3tec", "androids_interview", "cmdc", "turkish")
MERGED_DATASETS = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")

CAMPAIGN_BY_CONDITION_BACKBONE = {
    ("native", "qwen"): "text_heads_native_v1",
    ("native", "gemma4"): "text_heads_native_gemma4_v1",
    ("english", "qwen"): "text_heads_en_v1",
    ("english", "gemma4"): "text_heads_en_gemma4_v1",
}

MERGED_RUN_ROOT_TEMPLATE = (
    REMOTE_PROJECT_BASE
    + "/output_model/symmetric_merged/native_en_text_heads_v1/{variant}_text_only"
)
MERGED_MERGED_ROOT_TEMPLATE = (
    REMOTE_PROJECT_BASE
    + "/outputs/symmetric_merged/native_en_text_heads_v1/{variant}_text_only"
)

HIDDEN_CLASSIFIERS_ROOT = REMOTE_PROJECT_BASE + "/outputs/hidden_classifiers"
HEADS_ATTEMPT_ROOT_TEMPLATE = REMOTE_PROJECT_BASE + "/output_model/{campaign}_heads"
OPTUNA_ATTEMPT_ROOT_TEMPLATE = REMOTE_PROJECT_BASE + "/output_model/{campaign}_optuna100"

LOGREG_EXPERIMENT_ID = "logreg_raw_harmonized_v1"
OPTUNA_EXPERIMENT_ID = "xgb_optuna100_harmonized_v1"
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
AGGREGATION = "subject_level"
METRIC_NAMESPACE = "headline/binary_strict"

QWEN_ENV_DEFAULT = "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
GEMMA_ENV_DEFAULT = "/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate"


def campaign_for(condition: str, backbone: str) -> str:
    key = (str(condition), str(backbone))
    if key not in CAMPAIGN_BY_CONDITION_BACKBONE:
        raise ValueError(f"unknown condition/backbone pair: {key}")
    return CAMPAIGN_BY_CONDITION_BACKBONE[key]


def merged_variant(condition: str, backbone: str) -> str:
    return f"{condition}_{backbone}"


def merged_run_root(condition: str, backbone: str) -> str:
    return MERGED_RUN_ROOT_TEMPLATE.format(
        variant=merged_variant(condition, backbone)
    )


def merged_merged_root(condition: str, backbone: str) -> str:
    return MERGED_MERGED_ROOT_TEMPLATE.format(
        variant=merged_variant(condition, backbone)
    )


def materialize_merged_config(config_dict: dict[str, Any], *, seed: int) -> str:
    """Render the per-seed derived merged config text.

    Only the top-level training seed changes. The split seed and fixed head
    seed must already be locked to the study values. Every ``${PROJECT_ROOT}``
    placeholder is rewritten to the permanent remote project base so runtime
    writes never land inside the immutable deployment code tree.
    """

    if int(seed) not in STUDY_SEEDS:
        raise ValueError(f"training seed must be one of {STUDY_SEEDS}, got {seed!r}")
    protocol_settings = config_dict.get("protocol_settings") or {}
    resolved_split_seed = protocol_settings.get("split_seed")
    if resolved_split_seed is None:
        raise ValueError(
            "study merged configs must declare protocol_settings.split_seed"
        )
    if int(resolved_split_seed) != STUDY_SPLIT_SEED:
        raise ValueError(
            f"split seed must be {STUDY_SPLIT_SEED}, got {resolved_split_seed!r}"
        )
    heads_cfg = config_dict.get("heads") or {}
    fixed_seed = heads_cfg.get("fixed_seed")
    if fixed_seed is None or int(fixed_seed) != STUDY_HEAD_SEED:
        raise ValueError(
            f"heads.fixed_seed must be {STUDY_HEAD_SEED}, got {fixed_seed!r}"
        )
    derived = copy.deepcopy(config_dict)
    derived["seed"] = int(seed)
    text = yaml.safe_dump(derived, sort_keys=False)
    return text.replace("${PROJECT_ROOT}", REMOTE_PROJECT_BASE)


def merged_stage_folds(stage: str) -> list[int]:
    if stage == "cv":
        return [0, 1, 2, 3, 4]
    if stage in {"final", "smoke"}:
        return [0]
    raise ValueError(f"unsupported merged stage: {stage!r}")


def merged_fold_paths(
    *,
    condition: str,
    backbone: str,
    run_id: str,
    stage: str,
    fold: int,
) -> dict[str, str]:
    run_root = merged_run_root(condition, backbone)
    merged_root = merged_merged_root(condition, backbone)
    fold_dir = f"{run_root}/{run_id}/{stage}/fold_{int(fold)}"
    return {
        "fold_dir": fold_dir,
        "checkpoint_dir": f"{fold_dir}/best_model",
        "features_dir": f"{merged_root}/{run_id}/{stage}/fold_{int(fold)}/features",
        "heads_dir": f"{merged_root}/{run_id}/{stage}/fold_{int(fold)}/heads",
        "derived_config_dir": (
            f"{REMOTE_PROJECT_BASE}/experiment_runtime/configs/{run_id}"
        ),
    }


def rounded_median_epoch(values: list[int], *, low: int = 1, high: int = 20) -> int:
    """The locked final-stage epoch rule: rounded median of CV selections."""

    if not values:
        raise ValueError("cannot derive the final epoch from an empty selection list")
    result = int(math.floor(float(statistics.median([int(v) for v in values])) + 0.5))
    return max(low, min(high, result))


def standalone_cache_paths(
    *,
    dataset: str,
    condition: str,
    run_name: str,
    fold: int,
) -> dict[str, str]:
    base = f"{HIDDEN_CLASSIFIERS_ROOT}/{dataset}/{condition}/{run_name}/fold_{int(fold)}"
    return {
        "cache_base": base,
        "cache_dir": f"{base}/hidden_features",
    }


def standalone_attempt_path(
    *,
    campaign: str,
    dataset: str,
    run_name: str,
    fold: int,
    experiment_id: str,
) -> str:
    family_root = OPTUNA_ATTEMPT_ROOT_TEMPLATE.format(campaign=campaign)
    if experiment_id == LOGREG_EXPERIMENT_ID:
        family_root = HEADS_ATTEMPT_ROOT_TEMPLATE.format(campaign=campaign)
    return (
        f"{family_root}/text_only/{dataset}/{run_name}/fold_{int(fold)}/{experiment_id}"
    )


def merged_attempt_path(
    *,
    campaign: str,
    run_id: str,
    stage: str,
    fold: int,
    experiment_id: str,
) -> str:
    family_root = (
        HEADS_ATTEMPT_ROOT_TEMPLATE.format(campaign=campaign)
        if experiment_id == LOGREG_EXPERIMENT_ID
        else OPTUNA_ATTEMPT_ROOT_TEMPLATE.format(campaign=campaign)
    )
    return (
        f"{family_root}/text_only/merged/{run_id}_{stage}/fold_{int(fold)}/{experiment_id}"
    )


def build_logreg_task_spec(
    *,
    family: str,
    backend: str,
    dataset: str,
    modality: str,
    condition: str,
    fold: int,
    seed: int,
    stage: str | None,
    cache_dir: str,
    group_id: str,
    run_name: str,
    branch: str,
    merged_sha: str,
    parent_checkpoint_path: str | None,
    github_issue: int | None = None,
    github_pr: int | None = None,
) -> dict[str, Any]:
    merged = family == "merged"
    if merged:
        qualifiers = {
            "dataset": "daic" if stage == "final" else "merged",
            "split_name": "test" if stage == "final" else "outer_holdout",
            "split_protocol": (
                "daic_official_train_fit_locked_test_evaluation"
                if stage == "final"
                else "symmetric_merged_cv_outer_holdout"
            ),
            "evaluation_view": EVALUATION_VIEW,
            "aggregation": AGGREGATION,
            "metric_namespace": METRIC_NAMESPACE,
        }
    else:
        qualifiers = {
            "dataset": dataset,
            "split_name": "final_eval",
            "split_protocol": "harmonized_subject_partitions",
            "evaluation_view": EVALUATION_VIEW,
            "aggregation": AGGREGATION,
            "metric_namespace": METRIC_NAMESPACE,
        }
    spec: dict[str, Any] = {
        "schema_version": "audiollm.logreg_head_task.v1",
        "dataset": dataset,
        "modality": modality,
        "condition": condition,
        "fold": int(fold),
        "seed": int(seed),
        "head_seed": STUDY_HEAD_SEED,
        "family": family,
        "backend": backend,
        "stage": stage,
        "cache_dir": cache_dir,
        "experiment_id": LOGREG_EXPERIMENT_ID,
        "group_id": group_id,
        "run_name": run_name,
        "branch": branch,
        "merged_sha": merged_sha,
        "parent": {
            "parent_attempt_id": None,
            "parent_fold_dir": (
                str(parent_checkpoint_path).rsplit("/best_model", 1)[0]
                if parent_checkpoint_path
                else None
            ),
            "parent_checkpoint_path": parent_checkpoint_path,
        },
        "evaluation_qualifiers": qualifiers,
        "github_issue": github_issue,
        "pr": github_pr,
    }
    return spec


def build_optuna_task_spec(
    *,
    family: str,
    backend: str,
    dataset: str,
    modality: str,
    condition: str,
    fold: int,
    seed: int,
    stage: str,
    cache_dir: str,
    group_id: str,
    run_name: str,
    branch: str,
    merged_sha: str,
    parent_checkpoint_path: str | None,
    checkpoint_hashes: dict[str, Any] | None = None,
    feature_metadata_sha256: str | None = None,
    merged_config: str | None = None,
    objective: str | None = None,
    target_trials: int = 100,
    github_issue: int | None = None,
    github_pr: int | None = None,
) -> dict[str, Any]:
    from src.features.optuna100_policy import OBJECTIVE_MERGED, OBJECTIVE_STANDALONE

    if family == "merged":
        qualifiers = {
            "dataset": "daic" if stage == "final" else "merged",
            "split_name": "test" if stage == "final" else "outer_holdout",
            "split_protocol": (
                "daic_official_train_fit_locked_test_evaluation"
                if stage == "final"
                else "symmetric_merged_cv_outer_holdout"
            ),
            "evaluation_view": EVALUATION_VIEW,
            "aggregation": AGGREGATION,
            "metric_namespace": METRIC_NAMESPACE,
        }
    else:
        qualifiers = {
            "dataset": dataset,
            "split_name": "final_eval",
            "split_protocol": "harmonized_subject_partitions",
            "evaluation_view": EVALUATION_VIEW,
            "aggregation": AGGREGATION,
            "metric_namespace": METRIC_NAMESPACE,
        }
    spec: dict[str, Any] = {
        "schema_version": "audiollm.posthoc_head_task.v1",
        "dataset": dataset,
        "modality": modality,
        "condition": condition,
        "fold": int(fold),
        "seed": int(seed),
        "family": family,
        "backend": backend,
        "stage": stage,
        "cache_dir": cache_dir,
        "experiment_id": OPTUNA_EXPERIMENT_ID,
        "objective": objective or (OBJECTIVE_MERGED if family == "merged" else OBJECTIVE_STANDALONE),
        "target_trials": int(target_trials),
        "group_id": group_id,
        "run_name": run_name,
        "branch": branch,
        "merged_sha": merged_sha,
        "parent": {
            "parent_attempt_id": None,
            "parent_fold_dir": (
                str(parent_checkpoint_path).rsplit("/best_model", 1)[0]
                if parent_checkpoint_path
                else None
            ),
            "parent_checkpoint_path": parent_checkpoint_path,
            "adapter_config_sha256": (checkpoint_hashes or {}).get("adapter_config_sha256"),
            "adapter_sha256": (checkpoint_hashes or {}).get("adapter_model_sha256"),
        },
        "evaluation_qualifiers": qualifiers,
        "github_issue": github_issue,
        "pr": github_pr,
    }
    if feature_metadata_sha256:
        spec["feature_metadata_sha256"] = feature_metadata_sha256
    if merged_config:
        spec["merged_config"] = merged_config
    return spec




def _q(value: Any) -> str:
    import shlex

    return shlex.quote(str(value))


GEMMA4_MODEL_DEFAULT = (
    "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/"
    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
)

_SIDECAR_WRITER = """python - <<'SIDEcarPY'
import os, sys
from pathlib import Path
code = os.environ["CODE"]
sys.path.insert(0, code)
from src.experiment_tracking.canonical import read_json, write_json_atomic
from src.experiment_tracking.lifecycle import (
    StatusRecord, new_job_event, append_job_event, write_status,
)
fold_dir = Path(os.environ["STUDY_FOLD_DIR"])
context = read_json(os.environ["EXPERIMENT_CONTEXT"])
attempt_id = str(context["attempt_id"])
fold = int(context["fold"])
metadata = {
    "schema_version": "audiollm.metadata.v1",
    "group_id": str(context["group_id"]),
    "logical_run_name": str(context["logical_run_name"]),
    "attempt_id": attempt_id,
    "fold": fold,
    "seed": int(context["seed"]),
    "created_at_utc": context["created_at_utc"],
    "source": dict(context["source"]),
    "research": dict(context.get("research") or {}),
    "hashes": {
        "resolved_config_sha256": os.environ.get("STUDY_DERIVED_CONFIG_SHA256") or None,
        "manifest_sha256": None,
        "split_sha256": None,
    },
    "paths": {"run_config": None, "best_model": None, "local_evidence_root": None},
    "parent": {
        "parent_attempt_id": context.get("supersedes_attempt_id"),
        "parent_checkpoint_role": "best_model",
        "parent_checkpoint_path": None,
        "adapter_config_sha256": None,
        "adapter_sha256": None,
    },
    "wandb": {
        "project": "audiollm-depression",
        "entity": None,
        "run_id": attempt_id + "-fold" + str(fold),
        "url": None,
        "sync_status": "NOT_EXPORTED",
    },
}
if metadata["parent"]["parent_attempt_id"] is None:
    metadata.pop("parent")
write_json_atomic(fold_dir / "metadata.json", metadata, indent=2)
status = StatusRecord(attempt_id=attempt_id, fold=fold, state="PLANNED")
status.transition("DEPLOYED", reason="managed merged submission on verified deployment")
write_status(fold_dir / "status.json", status)
status.transition("SUBMITTED", reason="sbatch chain accepted by scheduler")
write_status(fold_dir / "status.json", status)
jobs_path = fold_dir / "jobs.jsonl"
jobs_path.write_text("", encoding="utf-8")
chain = [
    ("train", int(os.environ["STUDY_TRAIN_ID"]), "merged_train", []),
    ("postprocess", int(os.environ["STUDY_POST_ID"]), "merged_postprocess", [int(os.environ["STUDY_TRAIN_ID"])]),
    ("head", int(os.environ["STUDY_HEAD_ID"]), "merged_head", [int(os.environ["STUDY_POST_ID"])]),
]
for job_key, job_id, job_type, deps in chain:
    append_job_event(
        jobs_path,
        new_job_event(
            job_key=job_key,
            job_type=job_type,
            event_type="SUBMITTED",
            attempt_id=attempt_id,
            fold=fold,
            slurm_job_id=str(job_id),
            status="PENDING",
            dependency_job_ids=[str(d) for d in deps],
        ),
    )
print("sidecars written:", fold_dir)
SIDEcarPY"""


def render_merged_chain_script(
    *,
    code_path: str,
    derived_config_path: str,
    derived_config_sha256: str,
    condition: str,
    backbone: str,
    run_id: str,
    stage: str,
    fold: int,
    fold_dir: str,
    checkpoint_dir: str,
    features_dir: str,
    source_commit: str,
    context_path: str,
    epochs: int | None = None,
    subjects_per_class: int | None = None,
    head_trials: int | None = 0,
) -> str:
    """Render the scheduler-login script for one merged train->post->head chain.

    Shapes stay exactly as the shared workers declare them: train is
    1 node x 4 tasks x 4 H100 (NPROC_PER_NODE=4), postprocess uses 1 GPU,
    heads run CPU-only. The Optuna-100 study is a separate managed submission.
    """

    env_activate = GEMMA_ENV_DEFAULT if backbone == "gemma4" else QWEN_ENV_DEFAULT
    gemma_exports = ""
    if backbone == "gemma4":
        gemma_exports = (
            f"export ENV_ACTIVATE={_q(env_activate)}\n"
            f'export MODEL_PATH="${{GEMMA4_MODEL_PATH:-{GEMMA4_MODEL_DEFAULT}}}"\n'
        )
    epoch_export = f"\nexport EPOCHS={int(epochs)}" if epochs is not None else ""
    smoke_export = (
        f"\nexport SUBJECTS_PER_CLASS={int(subjects_per_class)}"
        if subjects_per_class is not None
        else ""
    )
    head_trials_value = int(0 if head_trials is None else head_trials)
    return f"""#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE={_q(code_path)}
cd "$CODE"
module purge
module load bsc/1.0 miniforge/24.3.0-0
source {_q(QWEN_ENV_DEFAULT)}

export PROJECT_ROOT="$CODE"
export CONFIG={_q(derived_config_path)}
export STAGE={_q(stage)}
export FOLD={int(fold)}
export RUN_ID={_q(run_id)}
export SOURCE_COMMIT={_q(source_commit)}
{gemma_exports}

# --- train (1 node x 4 tasks x 4 H100, NPROC_PER_NODE=4 DDP) ---
export NPROC_PER_NODE=4{epoch_export}{smoke_export}
TRAIN_ID=$(sbatch --parsable --chdir="$CODE" scripts/run_symmetric_merged_train_slurm.sh)
echo "Submitted training job: $TRAIN_ID"
unset EPOCHS SUBJECTS_PER_CLASS || true

# --- postprocess: best-model evaluation + features (1 GPU) ---
export CHECKPOINT_DIR={_q(checkpoint_dir)}
POST_ID=$(sbatch --parsable --chdir="$CODE" --dependency=afterok:$TRAIN_ID scripts/run_symmetric_merged_postprocess_slurm.sh)
echo "Submitted postprocess job: $POST_ID"

# --- heads logreg+xgb_fixed with Optuna disabled (CPU-only) ---
export FEATURES_DIR={_q(features_dir)}
export TRIALS={head_trials_value}
HEAD_ID=$(sbatch --parsable --chdir="$CODE" --dependency=afterok:$POST_ID scripts/run_symmetric_merged_head_slurm.sh)
echo "Submitted head job: $HEAD_ID"

# --- tracking sidecars beside the resolved config ---
mkdir -p {_q(fold_dir)}
export CODE STUDY_FOLD_DIR={_q(fold_dir)}
export STUDY_TRAIN_ID=$TRAIN_ID STUDY_POST_ID=$POST_ID STUDY_HEAD_ID=$HEAD_ID
export EXPERIMENT_CONTEXT={_q(context_path)}
export STUDY_DERIVED_CONFIG_SHA256={_q(derived_config_sha256)}
{_SIDECAR_WRITER}
"""


def render_study_job_script(
    *,
    code_path: str,
    worker_relpath: str,
    job_name: str,
    exports: list[tuple[str, str]],
    after_job_ids: list[str],
    echo_label: str,
) -> str:
    """Render the scheduler-login script submitting ONE sbatch job."""

    export_lines = "\n".join(f"export {key}={_q(value)}" for key, value in exports)
    dep_args = ""
    if after_job_ids:
        dep_args = f" --dependency=afterok:{':'.join(after_job_ids)}"
    return f"""#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE={_q(code_path)}
cd "$CODE"
{export_lines}
JOB_ID=$(sbatch --parsable --chdir="$CODE"{dep_args} {worker_relpath})
echo "Submitted {echo_label} job: $JOB_ID"
"""
