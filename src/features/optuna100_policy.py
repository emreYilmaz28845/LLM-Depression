"""Shared Optuna-100 XGBoost protocol policy for the harmonized campaign.

Single source of truth for the fixed 100-trial protocol (runbook Section 7)
so ordinary standalone, English, official-development, and merged head paths
cannot drift apart. Historical matrices and the legacy
``baselines/qwen_hidden_xgb_optuna.py`` defaults are untouched; this policy
is opt-in through the ``harmonized_optuna100_v1`` profile.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_PROFILE = "harmonized_optuna100_v1"
EXPERIMENT_ID = "xgb_optuna100_harmonized_v1"

PRODUCTION_TARGET_TRIALS = 100
SMOKE_TARGET_TRIALS = 2
SAMPLER = "TPESampler"
SAMPLER_SEED = 1337
MODEL_SEED = 1337
INNER_SPLIT_SEED = 1337
INNER_FOLDS = 3
THRESHOLD = 0.5
SAMPLING_MODE = "none"
XGB_THREADS_PER_STUDY = 20
STUDY_OPTIMIZATION_JOBS = 1
OBJECTIVE_STANDALONE = "macro_f1"
OBJECTIVE_MERGED = "mean_per_dataset_inner_macro_f1"

OPTUNA_VERSION = "4.4.0"
XGBOOST_VERSION = "2.1.4"
PACKAGE_PINS = {"optuna": OPTUNA_VERSION, "xgboost": XGBOOST_VERSION}

QWEN_PREDICTION_BACKEND = "qwen_hidden_xgb_optuna100"
GEMMA4_PREDICTION_BACKEND = "gemma4_hidden_xgb_optuna100"
MERGED_BACKEND_SUFFIX = "symmetric_merged"

# The exact fixed search space from runbook Section 7.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"kind": "int", "low": 100, "high": 1000, "step": 50},
    "learning_rate": {"kind": "float", "low": 0.005, "high": 0.2, "log": True},
    "max_depth": {"kind": "int", "low": 1, "high": 6},
    "min_child_weight": {"kind": "float", "low": 0.5, "high": 20.0, "log": True},
    "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"kind": "float", "low": 0.1, "high": 1.0},
    "gamma": {"kind": "float", "low": 1e-8, "high": 5.0, "log": True},
    "reg_alpha": {"kind": "float", "low": 1e-8, "high": 20.0, "log": True},
    "reg_lambda": {"kind": "float", "low": 1e-3, "high": 50.0, "log": True},
    "scale_pos_weight": {"kind": "float", "low": 0.25, "high": 4.0, "log": True},
}


def resolved_search_space() -> dict[str, dict[str, Any]]:
    return {name: dict(spec) for name, spec in SEARCH_SPACE.items()}


def assert_production_target(trials: int) -> None:
    """Refuse any production study target other than 100."""
    assert_target(trials, stage="production")


def assert_target(trials: int, *, stage: str = "production") -> None:
    """Validate the only two supported execution targets.

    Smoke is deliberately resumability-only and is never reportable.  Every
    production/headline invocation still goes through the fixed 100-trial
    gate.
    """

    expected = SMOKE_TARGET_TRIALS if str(stage).lower() == "smoke" else PRODUCTION_TARGET_TRIALS
    if int(trials) != expected:
        raise ValueError(
            f"harmonized_optuna100_v1 {stage} studies require exactly "
            f"{expected} completed trials, got {int(trials)}."
        )


def assert_protocol_settings(
    *,
    inner_folds: int,
    seed: int,
    inner_seed: int,
    sampling_mode: str,
    objective_name: str,
) -> None:
    if int(inner_folds) != INNER_FOLDS:
        raise ValueError(
            f"harmonized_optuna100_v1 requires {INNER_FOLDS} inner folds, got {inner_folds}."
        )
    if int(seed) != MODEL_SEED or int(inner_seed) != INNER_SPLIT_SEED:
        raise ValueError(
            "harmonized_optuna100_v1 requires model seed 1337 and inner split "
            f"seed 1337; got seed={seed}, inner_seed={inner_seed}."
        )
    if str(sampling_mode) != SAMPLING_MODE:
        raise ValueError(
            f"harmonized_optuna100_v1 requires sampling mode {SAMPLING_MODE!r}, "
            f"got {sampling_mode!r}."
        )
    if objective_name != OBJECTIVE_STANDALONE:
        raise ValueError(
            "harmonized_optuna100_v1 standalone studies optimize pooled inner "
            f"subject-level macro-F1, got objective {objective_name!r}."
        )


def prediction_backend(model_backend: str | None, *, merged: bool = False) -> str:
    if str(model_backend or "").strip().lower() == "gemma4":
        backend = GEMMA4_PREDICTION_BACKEND
    else:
        backend = QWEN_PREDICTION_BACKEND
    if merged:
        backend = f"{backend}_{MERGED_BACKEND_SUFFIX}"
    return backend


def protocol_block(
    *,
    dataset: str,
    condition: str,
    modality: str,
    fold: int,
    seed: int,
    objective: str,
    merged: bool = False,
    model_backend: str | None = None,
) -> dict[str, Any]:
    """Canonical protocol record written into study configs and metadata."""
    return {
        "protocol_profile": PROTOCOL_PROFILE,
        "experiment_id": EXPERIMENT_ID,
        "classifier_family": "xgb_optuna_raw",
        "dataset": dataset,
        "condition": condition,
        "modality": modality,
        "outer_fold": int(fold),
        "seed": int(seed),
        "sampler": SAMPLER,
        "sampler_seed": SAMPLER_SEED,
        "model_seed": MODEL_SEED,
        "inner_split_seed": INNER_SPLIT_SEED,
        "inner_folds": INNER_FOLDS,
        "threshold": THRESHOLD,
        "sampling_mode": SAMPLING_MODE,
        "target_completed_trials": PRODUCTION_TARGET_TRIALS,
        "objective": objective,
        "search_space": resolved_search_space(),
        "packages": dict(PACKAGE_PINS),
        "xgb_threads_per_study": XGB_THREADS_PER_STUDY,
        "study_optimization_jobs": STUDY_OPTIMIZATION_JOBS,
        "prediction_backend": prediction_backend(model_backend, merged=merged),
        "model_backend": model_backend,
        "merged": bool(merged),
    }
