#!/usr/bin/env python3
"""Run the protocol-frozen Androids 150-trial raw-XGBoost head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from baselines.androids_hidden_classifier import _sample_rows
from src.features.androids_hidden_policy import (
    ANDROID_DATASET,
    ANDROID_HIDDEN_OPTUNA_SCHEMA,
    ANDROID_HEADS,
    ANDROID_THRESHOLD,
    aggregate_androids_hidden_predictions,
    androids_training_weights,
    cache_identity,
    canonical_sha256,
    load_androids_cache,
    read_json,
    write_csv,
    write_jsonl,
    write_sha256_manifest,
)
from src.utils import save_json


EXPERIMENT_ID = "xgb_optuna_150t_d6"
TARGET_TRIALS = 150
INNER_FOLDS = 3
SEED = 1337
SEARCH_PROFILE = "standard_d6"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("audio_only", "audio_text", "text_only"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-trials", type=int, default=TARGET_TRIALS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inner-seed", type=int, default=SEED)
    parser.add_argument("--xgb-threads", type=int, default=20)
    return parser.parse_args()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    tmp.replace(path)


def _study_config(
    *,
    cache_dir: Path,
    metadata: dict[str, Any],
    modality: str,
    fold: int,
    run_id: str,
    source_commit: str,
    target_trials: int,
    inner_folds: int,
    seed: int,
    inner_seed: int,
    xgb_threads: int,
) -> tuple[dict[str, Any], str]:
    try:
        import optuna
        import sklearn
        import xgboost
    except ImportError as exc:
        raise RuntimeError("Androids Optuna requires optuna==4.4.0, xgboost==2.1.4, and scikit-learn==1.7.0.") from exc
    from baselines.qwen_hidden_xgb_optuna import fixed_xgb_params, resolved_search_space

    search_space = resolved_search_space(SEARCH_PROFILE)
    fixed_params = fixed_xgb_params(seed, xgb_threads)
    config = {
        "schema_version": ANDROID_HIDDEN_OPTUNA_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": modality,
        "fold": int(fold),
        "run_id": run_id,
        "source_commit": source_commit,
        "classifier_family": "xgb_optuna_raw",
        "classifier_variant": EXPERIMENT_ID,
        "cache_dir": str(cache_dir),
        "cache_identity": cache_identity(cache_dir),
        "objective": "macro_f1",
        "target_trials": int(target_trials),
        "inner_fold_count": int(inner_folds),
        "seed": int(seed),
        "sampler_seed": int(seed),
        "model_seed": int(seed),
        "inner_seed": int(inner_seed),
        "xgb_threads": int(xgb_threads),
        "search_profile": SEARCH_PROFILE,
        "search_space": search_space,
        "fixed_xgb_params": fixed_params,
        "threshold": ANDROID_THRESHOLD,
        "aggregation_policy": metadata["aggregation_policy"],
        "manifest_sha256": metadata["manifest_sha256"],
        "split_metadata_sha256": metadata["split_metadata_sha256"],
        "checkpoint_hashes": {
            "adapter_config_sha256": metadata["adapter_config_sha256"],
            "adapter_sha256": metadata["adapter_sha256"],
        },
        "packages": {
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    config_hash = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return config, config_hash


def _load_or_create_study(output_dir: Path, config: dict[str, Any], config_hash: str):
    import optuna

    config_path = output_dir / "study_config.json"
    payload = {"canonical_config": config, "config_sha256": config_hash}
    if config_path.exists() and read_json(config_path) != payload:
        raise ValueError(f"Existing Androids Optuna configuration differs: {output_dir}")
    if not config_path.exists():
        _write_json_atomic(config_path, payload)
    study_name = (
        f"androids_hidden_{config['run_id']}_{config['modality']}_fold{config['fold']}_"
        f"{EXPERIMENT_ID}"
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"])),
        study_name=study_name,
        storage=f"sqlite:///{output_dir / 'study.sqlite3'}",
        load_if_exists=True,
    )
    existing_hash = study.user_attrs.get("config_sha256")
    existing_config = study.user_attrs.get("canonical_config")
    if existing_hash is not None and existing_hash != config_hash:
        raise ValueError("Existing Androids Optuna study config hash differs; refusing to resume.")
    if existing_config is not None and existing_config != config:
        raise ValueError("Existing Androids Optuna study config differs; refusing to resume.")
    if any(str(trial.state.name) in {"FAIL", "RUNNING", "WAITING"} for trial in study.trials):
        raise ValueError("Existing Androids Optuna study has non-terminal/failed trials; refusing to resume.")
    study.set_user_attr("config_sha256", config_hash)
    study.set_user_attr("canonical_config", config)
    return study


def _inner_assignments(rows: list[dict[str, Any]], inner_folds: int, seed: int) -> dict[str, Any]:
    from baselines.qwen_hidden_xgb_optuna import build_inner_subject_assignments

    return build_inner_subject_assignments(rows, inner_folds=inner_folds, seed=seed)


def _classifier(params: dict[str, Any], fixed_params: dict[str, Any]):
    from baselines.qwen_hidden_xgb_optuna import _classifier

    return _classifier(params, fixed_params)


def _suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from baselines.qwen_hidden_xgb_optuna import _suggest_params

    return _suggest_params(trial, search_space)


def _write_trials(study: Any, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": int(trial.number),
            "state": trial.state.name,
            "value": trial.value,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            row[f"user_{key}"] = value
        rows.append(row)
    _write_csv_atomic(rows, path)


def _objective(
    *,
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    modality: str,
    assignments: dict[str, Any],
    fixed_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    experiment_id: str,
):
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    outer_subjects = {str(row["subject_id"]) for row in train_rows}

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, search_space)
        oof_samples: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        weight_audits: list[dict[str, Any]] = []
        for inner_fold in assignments["folds"]:
            train_idx = np.asarray(inner_fold["train_row_indices"], dtype=np.int64)
            val_idx = np.asarray(inner_fold["validation_row_indices"], dtype=np.int64)
            fit_rows = [train_rows[index] for index in train_idx.tolist()]
            weights, weight_audit = androids_training_weights(fit_rows, modality)
            model = _classifier(params, fixed_params)
            model.fit(train_x[train_idx], labels[train_idx], sample_weight=weights)
            probabilities = np.asarray(model.predict_proba(train_x[val_idx])[:, 1], dtype=np.float64)
            val_rows = [train_rows[index] for index in val_idx.tolist()]
            sample_rows = _sample_rows(
                val_rows,
                probabilities,
                metadata,
                modality,
                int(metadata["fold"]),
                experiment_id,
            )
            _, _, fold_metric = aggregate_androids_hidden_predictions(sample_rows, modality)
            fold_metrics.append({"inner_fold": int(inner_fold["fold"]), **fold_metric})
            weight_audits.append({"inner_fold": int(inner_fold["fold"]), **weight_audit})
            oof_samples.extend(sample_rows)
        subject_ids = [str(row["subject_id"]) for row in oof_samples]
        if Counter(subject_ids) != Counter(outer_subjects):
            raise ValueError("Androids inner OOF rows do not cover each subject exactly once.")
        _, oof_subjects, pooled_metrics = aggregate_androids_hidden_predictions(oof_samples, modality)
        if len(oof_subjects) != len(outer_subjects):
            raise ValueError("Androids inner OOF aggregation changed the subject coverage.")
        trial.set_user_attr("inner_fold_metrics", fold_metrics)
        trial.set_user_attr("inner_oof_metrics", pooled_metrics)
        trial.set_user_attr("inner_weight_audits", weight_audits)
        return float(pooled_metrics["macro_f1"])

    return objective


def _required_final_artifacts() -> tuple[str, ...]:
    return (
        "study_config.json",
        "inner_subject_assignments.json",
        "inner_fold_metrics.json",
        "inner_oof_metrics.json",
        "inner_weight_audits.json",
        "trials.csv",
        "best_params.json",
        "final_fit_weight_audit.json",
        "pipeline.joblib",
        "predictions_sample_level.jsonl",
        "predictions_sample_level.csv",
        "predictions_turn_level.jsonl",
        "predictions_turn_level.csv",
        "predictions_subject_level.jsonl",
        "predictions_subject_level.csv",
        "metrics.json",
        "classifier_metadata.json",
    )


def _completed_result_if_compatible(
    output_dir: Path,
    *,
    config_hash: str,
    target_trials: int,
) -> dict[str, Any] | None:
    metadata_path = output_dir / "classifier_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    if metadata.get("search_config_sha256") != config_hash:
        raise ValueError("Existing Androids Optuna result has a different search config hash.")
    if int(metadata.get("completed_trials", -1)) != int(target_trials):
        raise ValueError("Existing Androids Optuna result has a different trial count.")
    if any(not (output_dir / name).is_file() for name in _required_final_artifacts()):
        return None
    return {"head": EXPERIMENT_ID, **read_json(output_dir / "metrics.json")}


def run_optuna(
    *,
    cache_dir: Path,
    output_dir: Path,
    modality: str,
    fold: int,
    run_id: str,
    source_commit: str,
    target_trials: int = TARGET_TRIALS,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
    inner_seed: int = SEED,
    xgb_threads: int = 20,
) -> dict[str, Any]:
    if target_trials < 1 or inner_folds < 2 or xgb_threads < 1:
        raise ValueError("Invalid Androids Optuna trial/fold/thread count.")
    train_x, train_rows, eval_x, eval_rows, metadata = load_androids_cache(
        cache_dir,
        modality=modality,
        fold=fold,
        source_commit=source_commit,
        require_production=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config, config_hash = _study_config(
        cache_dir=cache_dir,
        metadata=metadata,
        modality=modality,
        fold=fold,
        run_id=run_id,
        source_commit=source_commit,
        target_trials=target_trials,
        inner_folds=inner_folds,
        seed=seed,
        inner_seed=inner_seed,
        xgb_threads=xgb_threads,
    )
    assignments = _inner_assignments(train_rows, inner_folds, inner_seed)
    assignment_path = output_dir / "inner_subject_assignments.json"
    if assignment_path.exists() and read_json(assignment_path) != assignments:
        raise ValueError("Existing Androids inner subject assignments differ; refusing to resume.")
    if not assignment_path.exists():
        _write_json_atomic(assignment_path, assignments)
    study = _load_or_create_study(output_dir, config, config_hash)
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if len(completed) > target_trials:
        raise ValueError("Androids Optuna study already exceeds the requested trial count.")
    if len(completed) < target_trials:
        objective = _objective(
            train_x=train_x,
            train_rows=train_rows,
            metadata=metadata,
            modality=modality,
            assignments=assignments,
            fixed_params=config["fixed_xgb_params"],
            search_space=config["search_space"],
            experiment_id=EXPERIMENT_ID,
        )
        study.optimize(objective, n_trials=target_trials - len(completed), n_jobs=1)
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if len(completed) != target_trials or any(trial.state.name != "COMPLETE" for trial in study.trials):
        raise ValueError("Androids Optuna did not finish with exactly the requested completed trials.")
    _write_trials(study, output_dir / "trials.csv")
    best_trial = study.best_trial
    _write_json_atomic(
        output_dir / "best_params.json",
        {
            "best_trial_number": int(best_trial.number),
            "objective": "macro_f1",
            "objective_value": float(best_trial.value),
            "suggested_params": dict(best_trial.params),
            "fixed_params": config["fixed_xgb_params"],
            "completed_trial_count": len(completed),
        },
    )
    _write_json_atomic(output_dir / "inner_fold_metrics.json", best_trial.user_attrs["inner_fold_metrics"])
    _write_json_atomic(output_dir / "inner_oof_metrics.json", best_trial.user_attrs["inner_oof_metrics"])
    _write_json_atomic(output_dir / "inner_weight_audits.json", best_trial.user_attrs["inner_weight_audits"])
    existing = _completed_result_if_compatible(
        output_dir,
        config_hash=config_hash,
        target_trials=target_trials,
    )
    if existing is not None:
        return existing
    if any((output_dir / name).exists() for name in _required_final_artifacts() if name != "classifier_metadata.json"):
        # The study artifacts above are expected, but model/result artifacts
        # must never be silently mixed with a new search configuration.
        if (output_dir / "classifier_metadata.json").exists():
            raise ValueError("Androids Optuna output is an incompatible partial result.")

    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    weights, final_weight_audit = androids_training_weights(train_rows, modality)
    _write_json_atomic(output_dir / "final_fit_weight_audit.json", final_weight_audit)
    model = _classifier(dict(best_trial.params), config["fixed_xgb_params"])
    model.fit(train_x, labels, sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(eval_x)[:, 1], dtype=np.float64)
    sample_rows = _sample_rows(
        eval_rows,
        probabilities,
        metadata,
        modality,
        fold,
        EXPERIMENT_ID,
    )
    turn_rows, subject_rows, metrics = aggregate_androids_hidden_predictions(sample_rows, modality)
    metrics.update(
        {
            "dataset": ANDROID_DATASET,
            "modality": modality,
            "fold": int(fold),
            "head": EXPERIMENT_ID,
            "schema_version": ANDROID_HIDDEN_OPTUNA_SCHEMA,
        }
    )
    import joblib

    joblib.dump(model, output_dir / "pipeline.joblib")
    write_jsonl(sample_rows, output_dir / "predictions_sample_level.jsonl")
    write_csv(sample_rows, output_dir / "predictions_sample_level.csv")
    write_jsonl(turn_rows, output_dir / "predictions_turn_level.jsonl")
    write_csv(turn_rows, output_dir / "predictions_turn_level.csv")
    write_jsonl(subject_rows, output_dir / "predictions_subject_level.jsonl")
    write_csv(subject_rows, output_dir / "predictions_subject_level.csv")
    save_json(metrics, output_dir / "metrics.json")
    classifier_metadata = {
        "schema_version": ANDROID_HIDDEN_OPTUNA_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": modality,
        "fold": int(fold),
        "head": EXPERIMENT_ID,
        "run_id": run_id,
        "source_commit": source_commit,
        "objective": "macro_f1",
        "search_profile": SEARCH_PROFILE,
        "target_trials": int(target_trials),
        "completed_trials": len(completed),
        "failed_trials": 0,
        "trial_states": Counter(trial.state.name for trial in study.trials),
        "inner_fold_count": int(inner_folds),
        "seed": int(seed),
        "inner_seed": int(inner_seed),
        "threshold": ANDROID_THRESHOLD,
        "aggregation_policy": metadata["aggregation_policy"],
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "search_config_sha256": config_hash,
        "cache_dir": str(cache_dir),
        "cache_identity": config["cache_identity"],
        "cache_identity_sha256": canonical_sha256(config["cache_identity"]),
        "checkpoint_hashes": config["checkpoint_hashes"],
        "manifest_sha256": config["manifest_sha256"],
        "split_metadata_sha256": config["split_metadata_sha256"],
        "training_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
        "heldout_subject_ids": sorted({str(row["subject_id"]) for row in eval_rows}),
        "training_row_count": len(train_rows),
        "heldout_row_count": len(eval_rows),
        "input_dimension": int(train_x.shape[1]),
        "inner_subject_assignments": assignments,
        "inner_subject_leakage_count": 0,
        "final_fit_weight_audit": final_weight_audit,
        "no_pca": True,
        "no_oversampling": True,
        "no_controls": True,
    }
    save_json(classifier_metadata, output_dir / "classifier_metadata.json")
    write_sha256_manifest(
        output_dir,
        output_dir / "artifact_sha256.tsv",
        exclude_suffixes=(".joblib", ".sqlite3"),
    )
    return {"head": EXPERIMENT_ID, **metrics}


def main() -> None:
    args = _parse_args()
    result = run_optuna(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        modality=args.modality,
        fold=args.fold,
        run_id=args.run_id,
        source_commit=args.source_commit,
        target_trials=args.target_trials,
        inner_folds=args.inner_folds,
        seed=args.seed,
        inner_seed=args.inner_seed,
        xgb_threads=args.xgb_threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
