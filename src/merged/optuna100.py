"""Optuna-100 studies over symmetric-merged hidden features.

One post-hoc attempt per study, exactly 100 completed trials under the
shared ``harmonized_optuna100_v1`` policy, with the merged objective kept as
the unweighted mean of the five per-dataset inner macro-F1 values. Inner
folds are grouped and stratified by subject; fit and holdout are the merged
outer_train / outer_holdout partitions, so the final-stage holdout is the
untouched DAIC official test only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.features import optuna100_policy as policy
from src.merged.heads import (
    _load_features,
    _new_xgb,
    _predict_probability,
    _write_csv,
    aggregate_head_predictions,
)
from src.merged.protocol import DATASETS, compute_hierarchical_example_weights
from src.merged.runtime import load_merged_config
from src.utils import read_json, read_jsonl, save_json, write_jsonl


def _negative_f1(metrics: dict[str, Any]) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _inner_folds(train_rows: list[dict[str, Any]], *, inner_folds: int, seed: int) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    labels_by_subject: dict[str, int] = {}
    for row in train_rows:
        subject = str(row["subject_id"])
        label = int(row["label"])
        if subject in labels_by_subject and labels_by_subject[subject] != label:
            raise ValueError(f"Subject {subject} has inconsistent merged labels.")
        labels_by_subject[subject] = label
    if set(labels_by_subject.values()) != {0, 1}:
        raise ValueError("Merged outer-train subjects must contain both classes.")
    subject_ids = sorted(labels_by_subject)
    counts = Counter(labels_by_subject.values())
    if min(counts.values()) < inner_folds:
        raise ValueError(
            f"Each class needs at least {inner_folds} merged outer-train subjects; found {dict(counts)}."
        )
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    y = np.asarray([labels_by_subject[subject_id] for subject_id in subject_ids], dtype=np.int64)
    row_indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        row_indices_by_subject[str(row["subject_id"])].append(index)
    folds: list[dict[str, Any]] = []
    validation_coverage: list[str] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(subject_ids, y)):
        train_subjects = [subject_ids[index] for index in train_idx.tolist()]
        val_subjects = [subject_ids[index] for index in val_idx.tolist()]
        if set(train_subjects) & set(val_subjects):
            raise ValueError(f"Merged inner fold {fold_index} has subject overlap.")
        validation_coverage.extend(val_subjects)
        folds.append(
            {
                "fold": fold_index,
                "train_subject_ids": train_subjects,
                "validation_subject_ids": val_subjects,
                "train_row_indices": [
                    index for subject_id in train_subjects for index in row_indices_by_subject[subject_id]
                ],
                "validation_row_indices": [
                    index for subject_id in val_subjects for index in row_indices_by_subject[subject_id]
                ],
            }
        )
    if Counter(validation_coverage) != Counter(subject_ids):
        raise ValueError("Merged inner folds do not cover each subject exactly once.")
    return {
        "schema_version": "inner_subject_assignments.v1",
        "splitter": "StratifiedKFold",
        "n_splits": inner_folds,
        "shuffle": True,
        "random_state": seed,
        "subjects": [
            {"subject_id": subject_id, "label": labels_by_subject[subject_id]}
            for subject_id in subject_ids
        ],
        "folds": folds,
    }


def _inner_objective(
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    assignments: dict[str, Any],
    params: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    fold_metrics: list[dict[str, Any]] = []
    for fold in assignments["folds"]:
        fit_indices = list(fold["train_row_indices"])
        validation_indices = list(fold["validation_row_indices"])
        estimator = _new_xgb(params)
        fit_rows = [train_rows[index] for index in fit_indices]
        weighted_fit_rows, weight_audit = compute_hierarchical_example_weights(
            fit_rows, expected_datasets=DATASETS
        )
        weights = np.asarray([row["loss_weight"] for row in weighted_fit_rows], dtype=np.float64)
        estimator.fit(
            train_x[fit_indices],
            np.asarray([int(row["label"]) for row in fit_rows], dtype=np.int64),
            sample_weight=weights,
        )
        validation_rows = [train_rows[index] for index in validation_indices]
        probabilities = _predict_probability(estimator, train_x[validation_indices])
        _, metrics = aggregate_head_predictions(validation_rows, probabilities)
        values = [float(metrics[dataset]["macro_f1"]) for dataset in DATASETS if dataset in metrics]
        if len(values) != len(DATASETS):
            raise ValueError("Merged inner objective lost a dataset.")
        fold_metrics.append(
            {
                "fold": int(fold["fold"]),
                "dataset_macro_f1": {dataset: float(metrics[dataset]["macro_f1"]) for dataset in DATASETS},
                "mean_dataset_macro_f1": float(sum(values) / len(values)),
                "weight_audit": weight_audit,
            }
        )
    value = float(sum(item["mean_dataset_macro_f1"] for item in fold_metrics) / len(fold_metrics))
    return value, fold_metrics


def _suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": policy.MODEL_SEED,
        "n_jobs": policy.XGB_THREADS_PER_STUDY,
    }
    for name, spec in search_space.items():
        if spec["kind"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
    return params


def run_merged_optuna100(
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    merged_config_path: str | Path,
    stage: str,
    fold: int,
    run_id: str,
) -> dict[str, Any]:
    policy.assert_production_target(policy.PRODUCTION_TARGET_TRIALS)
    merged_config = load_merged_config(merged_config_path)
    modality = str(merged_config.get("modality") or "")
    features = Path(features_dir)
    feature_metadata = read_json(features / "feature_metadata.json")
    if str(feature_metadata.get("stage")) != stage or int(feature_metadata.get("fold", -1)) != int(fold):
        raise ValueError("Merged Optuna features have a stage/fold identity mismatch.")
    if str(feature_metadata.get("modality")) != modality:
        raise ValueError("Merged Optuna features have a modality identity mismatch.")
    model_backend = str(feature_metadata.get("model_backend") or "")
    prediction_backend = policy.prediction_backend(model_backend, merged=True)

    train_x, train_rows = _load_features(features, "outer_train")
    holdout_x, holdout_rows = _load_features(features, "outer_holdout")
    if train_x.shape[1] != holdout_x.shape[1]:
        raise ValueError("Merged Optuna train/holdout feature dimensions do not match.")
    if stage == "final" and {str(row["dataset"]) for row in holdout_rows} != {"daic"}:
        raise ValueError("Final merged Optuna may evaluate only the untouched DAIC official test.")

    output = Path(output_dir)
    if output.name != policy.EXPERIMENT_ID:
        raise ValueError(f"output dir must end with {policy.EXPERIMENT_ID!r}: {output}")
    output.mkdir(parents=True, exist_ok=True)

    assignments = _inner_folds(train_rows, inner_folds=policy.INNER_FOLDS, seed=policy.INNER_SPLIT_SEED)
    protocol = policy.protocol_block(
        dataset="merged",
        condition=modality,
        modality=modality,
        fold=int(fold),
        seed=policy.MODEL_SEED,
        objective=policy.OBJECTIVE_MERGED,
        merged=True,
        model_backend=model_backend or None,
    )
    config: dict[str, Any] = {
        "schema_version": "symmetric_merged_optuna100.v1",
        "stage": stage,
        "fold": int(fold),
        "run_id": run_id,
        "modality": modality,
        "merged_config": str(merged_config_path),
        "merged_config_sha256": feature_metadata.get("merged_config_sha256"),
        "manifest_hash": feature_metadata.get("manifest_hash"),
        "split_hash": feature_metadata.get("split_hash"),
        "features_dir": str(features),
        "feature_dimension": int(train_x.shape[1]),
        "protocol": protocol,
    }
    import hashlib

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    config_path = output / "study_config.json"
    if config_path.is_file():
        existing = read_json(config_path)
        if existing != {"canonical_config": config, "config_sha256": config_hash}:
            raise ValueError(f"Existing merged study_config differs: {output}. Refusing to resume.")
    save_json({"canonical_config": config, "config_sha256": config_hash}, config_path)
    save_json(assignments, output / "inner_subject_assignments.json")

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Merged Optuna-100 requires optuna==4.4.0.") from exc
    study_name = f"merged_{modality}_fold{int(fold)}_{policy.EXPERIMENT_ID}"
    storage = f"sqlite:///{output / 'study.sqlite3'}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=policy.SAMPLER_SEED),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    existing_hash = study.user_attrs.get("config_sha256")
    if existing_hash is not None and existing_hash != config_hash:
        raise ValueError("Existing merged study config hash differs. Refusing to resume.")
    study.set_user_attr("config_sha256", config_hash)
    study.set_user_attr("canonical_config", config)

    search_space = policy.resolved_search_space()

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, search_space)
        value, fold_metrics = _inner_objective(train_x, train_rows, assignments, params)
        trial.set_user_attr("inner_fold_metrics", fold_metrics)
        trial.set_user_attr("objective", policy.OBJECTIVE_MERGED)
        return value

    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    remaining = policy.PRODUCTION_TARGET_TRIALS - len(completed)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if len(completed) != policy.PRODUCTION_TARGET_TRIALS:
        raise RuntimeError(
            f"Merged study has {len(completed)} completed trials; expected "
            f"{policy.PRODUCTION_TARGET_TRIALS}."
        )

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        rows.append(row)
    _write_csv(rows, output / "trials.csv")

    best = study.best_trial
    save_json(
        {
            "best_trial_number": int(best.number),
            "objective": policy.OBJECTIVE_MERGED,
            "objective_value": float(best.value),
            "suggested_params": dict(best.params),
            "completed_trial_count": len(completed),
        },
        output / "best_params.json",
    )
    save_json(best.user_attrs["inner_fold_metrics"], output / "inner_fold_metrics.json")

    params = _suggest_params(best, search_space)
    estimator = _new_xgb(params)
    weighted_fit_rows, final_fit_weight_audit = compute_hierarchical_example_weights(
        train_rows, expected_datasets=DATASETS
    )
    weights = np.asarray([row["loss_weight"] for row in weighted_fit_rows], dtype=np.float64)
    estimator.fit(
        train_x,
        np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64),
        sample_weight=weights,
    )
    probabilities = _predict_probability(estimator, holdout_x)
    prediction_groups, metrics_by_dataset = aggregate_head_predictions(holdout_rows, probabilities)
    all_prediction_rows = [
        row for dataset in sorted(prediction_groups) for row in prediction_groups[dataset]
    ]
    for row in all_prediction_rows:
        row["prediction_backend"] = prediction_backend
        row["model_backend"] = model_backend or None
        row["classifier_variant"] = policy.EXPERIMENT_ID
    write_jsonl(all_prediction_rows, output / "predictions_subject_level.jsonl")
    _write_csv(all_prediction_rows, output / "predictions_subject_level.csv")

    pooled_true = [int(row["label"]) for row in all_prediction_rows]
    pooled_pred = [int(row["prediction"]) for row in all_prediction_rows]
    from src.metrics import classification_metrics

    pooled = classification_metrics(pooled_true, pooled_pred)
    pooled["negative_f1"] = _negative_f1(pooled)
    save_json(
        {
            "dataset_metrics": metrics_by_dataset,
            "pooled_subject_metrics": pooled,
        },
        output / "metrics.json",
    )
    import joblib

    joblib.dump(estimator, output / "pipeline.joblib")
    save_json(
        {
            "schema_version": "symmetric_merged_optuna100_classifier.v1",
            "stage": stage,
            "fold": int(fold),
            "modality": modality,
            "run_id": run_id,
            "classifier_family": "xgb_optuna_raw",
            "classifier_variant": policy.EXPERIMENT_ID,
            "protocol_profile": policy.PROTOCOL_PROFILE,
            "prediction_backend": prediction_backend,
            "model_backend": model_backend or None,
            "objective": policy.OBJECTIVE_MERGED,
            "target_trials": policy.PRODUCTION_TARGET_TRIALS,
            "completed_trials": len(completed),
            "best_value": float(best.value),
            "best_trial_number": int(best.number),
            "search_config_sha256": config_hash,
            "feature_dimension": int(train_x.shape[1]),
            "manifest_hash": feature_metadata.get("manifest_hash"),
            "split_hash": feature_metadata.get("split_hash"),
            "holdout_subject_counts": {
                dataset: len({str(row["subject_id"]) for row in rows_group})
                for dataset, rows_group in prediction_groups.items()
            },
            "training_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
            "holdout_subject_ids": sorted({str(row["subject_id"]) for row in holdout_rows}),
            "final_fit_weight_audit": final_fit_weight_audit,
        },
        output / "classifier_metadata.json",
    )
    return {
        "status": "completed",
        "prediction_backend": prediction_backend,
        "completed_trials": len(completed),
        "objective_value": float(best.value),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--merged-config", required=True)
    parser.add_argument("--stage", choices=("cv", "final"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_merged_optuna100(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        merged_config_path=args.merged_config,
        stage=args.stage,
        fold=args.fold,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
