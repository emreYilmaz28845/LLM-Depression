from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.merged.protocol import (
    DATASETS,
    build_grouped_inner_folds,
    canonical_sha256,
    compute_hierarchical_example_weights,
    resolve_head_inner_folds,
)
from src.merged.configuration import validate_shared_backend
from src.merged.runtime import load_merged_config, load_records_and_protocol
from src.metrics import binary_auroc, classification_metrics
from src.merged.provenance import write_slurm_provenance
from src.utils import (
    configure_logging,
    ensure_dir,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
    write_jsonl,
)


FIXED_HEAD = "xgb_fixed"
HEADS = ("logreg", FIXED_HEAD, "xgb_optuna")


def _load_features(features_dir: Path, partition: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix_path = features_dir / f"{partition}.npz"
    rows_path = features_dir / f"{partition}_rows.jsonl"
    if not matrix_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"Missing merged {partition} feature artifacts under {features_dir}.")
    with np.load(matrix_path) as payload:
        matrix = np.asarray(payload["vectors"], dtype=np.float32)
    rows = read_jsonl(rows_path)
    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise ValueError(f"Feature shape {matrix.shape} does not match {len(rows)} rows for {partition}.")
    if matrix.shape[0] and not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite merged features in {partition}.")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate merged feature sample identities in {partition}.")
    subject_labels: dict[str, int] = {}
    for row in rows:
        subject = str(row["subject_id"])
        label = int(row["label"])
        if subject in subject_labels and subject_labels[subject] != label:
            raise ValueError(f"Feature subject {subject} has inconsistent labels.")
        subject_labels[subject] = label
    return matrix, rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _negative_f1(metrics: dict[str, Any]) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _prediction_response_id(row: dict[str, Any]) -> str:
    """Use the same response hierarchy as merged training weights."""

    for field in ("response_id", "turn_key", "question_id", "sample_id"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Head prediction row has no response identity: {row}")


def aggregate_head_predictions(
    rows: list[dict[str, Any]], probabilities: np.ndarray, *, threshold: float = 0.5
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Reduce feature predictions using the component's response hierarchy."""

    if len(rows) != len(probabilities):
        raise ValueError("Head prediction rows and probabilities have different lengths.")
    sample_rows: list[dict[str, Any]] = []
    for row, probability in zip(rows, np.asarray(probabilities, dtype=np.float64).tolist()):
        sample_rows.append(
            {
                "dataset": str(row["dataset"]).lower(),
                "sample_id": str(row["sample_id"]),
                "subject_id": str(row["subject_id"]),
                "response_id": _prediction_response_id(row),
                "label": int(row["label"]),
                "probability": float(probability),
                "prediction": int(float(probability) >= float(threshold)),
            }
        )
    subject_rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset in sorted({str(row["dataset"]).lower() for row in sample_rows}):
        dataset_rows = [row for row in sample_rows if row["dataset"] == dataset]
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset_rows:
            by_subject[row["subject_id"]].append(row)
        for subject_id, subject_rows in sorted(by_subject.items()):
            # D3TEC and Androids have response/window rows; a response is
            # averaged first so windows cannot dominate a subject. The same
            # rule is harmless for one-vector or one-response datasets.
            by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in subject_rows:
                by_response[row["response_id"]].append(row)
            response_probabilities = [
                float(np.mean([item["probability"] for item in response_rows]))
                for response_rows in by_response.values()
            ]
            probability = float(np.mean(response_probabilities))
            label_values = {int(item["label"]) for item in subject_rows}
            if len(label_values) != 1:
                raise ValueError(f"Subject {subject_id} has inconsistent head labels.")
            subject_rows_by_dataset[dataset].append(
                {
                    "dataset": dataset,
                    "subject_id": subject_id,
                    "label": next(iter(label_values)),
                    "probability": probability,
                    "prediction": int(probability >= float(threshold)),
                    "response_count": len(by_response),
                    "sample_count": len(subject_rows),
                    "invalid_qwen_outputs": 0,
                }
            )
    metrics_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset, subject_rows in sorted(subject_rows_by_dataset.items()):
        y_true = [int(row["label"]) for row in subject_rows]
        y_pred = [int(row["prediction"]) for row in subject_rows]
        probabilities_subject = [float(row["probability"]) for row in subject_rows]
        metrics = classification_metrics(y_true, y_pred)
        metrics.update(
            {
                "dataset": dataset,
                "aggregation_level": "subject",
                "negative_f1": _negative_f1(metrics),
                "auroc": binary_auroc(y_true, probabilities_subject),
                "class_supports": {"non_depressed": int(sum(label == 0 for label in y_true)), "depressed": int(sum(label == 1 for label in y_true))},
                "invalid_qwen_outputs": 0,
                "subject_count": len(subject_rows),
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
        metrics_by_dataset[dataset] = metrics
    return dict(subject_rows_by_dataset), metrics_by_dataset


def _new_logreg(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def fixed_xgb_params(config: dict[str, Any], seed: int, threads: int) -> dict[str, Any]:
    values = dict((config.get("heads") or {}).get("fixed_xgb") or {})
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": int(values.get("n_estimators", 300)),
        "learning_rate": float(values.get("learning_rate", 0.03)),
        "max_depth": int(values.get("max_depth", 2)),
        "min_child_weight": float(values.get("min_child_weight", 5)),
        "subsample": float(values.get("subsample", 0.8)),
        "colsample_bytree": float(values.get("colsample_bytree", 0.25)),
        "reg_alpha": float(values.get("reg_alpha", 1.0)),
        "reg_lambda": float(values.get("reg_lambda", 10.0)),
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": int(threads),
    }


def _new_xgb(params: dict[str, Any]):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("The merged XGBoost heads require xgboost==2.1.4.") from exc
    return XGBClassifier(**params)


def _fit_weighted(estimator, x: np.ndarray, y: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_rows, audit = compute_hierarchical_example_weights(rows, expected_datasets=DATASETS)
    weights = np.asarray([row["loss_weight"] for row in weighted_rows], dtype=np.float64)
    estimator.fit(x, y, **({"sample_weight": weights} if estimator.__class__.__name__ != "Pipeline" else {"classifier__sample_weight": weights}))
    return audit


def _predict_probability(estimator, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier emitted non-finite probabilities.")
    return probabilities


def _objective_value(
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
        weighted_fit_rows, weight_audit = compute_hierarchical_example_weights(fit_rows, expected_datasets=DATASETS)
        weights = np.asarray([row["loss_weight"] for row in weighted_fit_rows], dtype=np.float64)
        estimator.fit(
            train_x[fit_indices],
            np.asarray([int(row["label"]) for row in fit_rows], dtype=np.int64),
            sample_weight=np.asarray(weights, dtype=np.float64),
        )
        validation_rows = [train_rows[index] for index in validation_indices]
        probabilities = _predict_probability(estimator, train_x[validation_indices])
        _, metrics = aggregate_head_predictions(validation_rows, probabilities)
        values = [float(metrics[dataset]["macro_f1"]) for dataset in DATASETS if dataset in metrics]
        if len(values) != len(DATASETS):
            raise ValueError("Head inner-fold objective lost a dataset.")
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


def _optuna_params(trial, config: dict[str, Any], seed: int, threads: int) -> dict[str, Any]:
    from src.features import optuna100_policy as policy

    optuna_cfg = (config.get("heads") or {}).get("optuna") or {}
    if optuna_cfg.get("protocol_profile") == policy.PROTOCOL_PROFILE:
        search_space = policy.resolved_search_space()
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": int(seed),
            "n_jobs": int(threads),
        }
        for name, spec in search_space.items():
            if spec["kind"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
            else:
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        return params
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": int(threads),
    }


def _run_optuna(
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    config: dict[str, Any],
    assignments: dict[str, Any],
    output_dir: Path,
    *,
    trials: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("The merged Optuna head requires optuna==4.4.0.") from exc
    optuna_cfg = (config.get("heads") or {}).get("optuna") or {}
    seed = int(optuna_cfg.get("seed", 1337))
    threads = int(optuna_cfg.get("xgb_threads", 20))
    ensure_dir(output_dir)
    identity = {
        "schema_version": "symmetric_merged_optuna_identity.v1",
        "objective": "unweighted_mean_per_dataset_macro_f1",
        "target_trials": int(trials),
        "inner_folds": int(len(assignments["folds"])),
        "seed": seed,
        "inner_assignments_hash": assignments["assignments_hash"],
        "feature_dimension": int(train_x.shape[1]),
    }
    identity_path = output_dir / "study_config.json"
    if identity_path.is_file() and read_json(identity_path) != identity:
        raise ValueError(f"Existing Optuna study has incompatible identity: {output_dir}")
    save_json(identity, identity_path)
    storage = f"sqlite:///{(output_dir / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=str(optuna_cfg.get("study_name", "symmetric_merged_xgb")),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        params = _optuna_params(trial, config, seed, threads)
        value, fold_metrics = _objective_value(train_x, train_rows, assignments, params)
        trial.set_user_attr("inner_fold_metrics", fold_metrics)
        trial.set_user_attr("objective", "unweighted_mean_per_dataset_macro_f1")
        return value

    completed = len([trial for trial in study.trials if trial.state.name == "COMPLETE"])
    if completed < int(trials):
        study.optimize(objective, n_trials=int(trials) - completed)
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if len(completed_trials) < int(trials):
        raise RuntimeError(f"Optuna study has only {len(completed_trials)} completed trials; expected {trials}.")
    best = dict(study.best_params)
    params = {
        **best,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": 1.0,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": threads,
    }
    summary = {
        "study_name": study.study_name,
        "study_db": str(output_dir / "study.db"),
        "target_trials": int(trials),
        "completed_trials": len(completed_trials),
        "best_value": float(study.best_value),
        "best_params": params,
        "objective": "unweighted_mean_per_dataset_macro_f1",
        "inner_assignments_hash": assignments["assignments_hash"],
    }
    save_json(summary, output_dir / "study_summary.json")
    return params, summary


def resolve_optuna_trials(
    merged_config: dict[str, Any], stage: str, trials: int | None
) -> int:
    """Resolve and enforce the merged Optuna trial count from the config
    identity. New harmonized configs declare the harmonized_optuna100_v1
    profile and require exactly 100 production trials; historical configs
    keep the fixed 150-trial rule; trials=0 disables Optuna."""
    expected_trial_count = int(
        trials
        if trials is not None
        else ((merged_config.get("heads") or {}).get("optuna") or {}).get("target_trials", 150)
    )
    from src.features import optuna100_policy as policy

    optuna_cfg = (merged_config.get("heads") or {}).get("optuna") or {}
    protocol_profile = str(optuna_cfg.get("protocol_profile") or "")
    if expected_trial_count == 0:
        return 0
    if protocol_profile == policy.PROTOCOL_PROFILE:
        policy.assert_production_target(expected_trial_count)
        if stage == "smoke":
            raise ValueError("Smoke merged Optuna studies must not use the production 100-trial profile.")
        return expected_trial_count
    if stage != "smoke" and expected_trial_count != 150:
        raise ValueError("Historical production merged Optuna is fixed to 150 trials.")
    return expected_trial_count


def run_merged_heads(
    config_path: str | Path,
    *,
    stage: str,
    fold: int,
    run_id: str,
    features_dir: str | Path,
    trials: int | None = None,
) -> dict[str, Any]:
    merged_config = load_merged_config(config_path)
    records, protocol = load_records_and_protocol(merged_config)
    model_backend = validate_shared_backend(merged_config, records)
    del records
    resolved_config_path = resolve_project_path(config_path)
    feature_dir = Path(features_dir).resolve()
    feature_metadata = read_json(feature_dir / "feature_metadata.json")
    if str(feature_metadata.get("stage")) != stage or int(feature_metadata.get("fold", -1)) != int(fold):
        raise ValueError("Merged head features have a stage/fold identity mismatch.")
    if str(feature_metadata.get("modality")) != str(merged_config.get("modality")):
        raise ValueError("Merged head features have a modality identity mismatch.")
    if feature_metadata.get("manifest_hash") != protocol["manifest"]["manifest_hash"]:
        raise ValueError("Merged head features have a manifest hash mismatch.")
    if feature_metadata.get("merged_config_sha256") and feature_metadata.get("merged_config_sha256") != sha256_file(resolved_config_path):
        raise ValueError("Merged head features have a merged-config hash mismatch.")
    if stage != "final" and feature_metadata.get("split_hash") != protocol["protocol"]["split_hash"]:
        raise ValueError("Merged head features have a split hash mismatch.")
    expected_fold_hash = protocol["protocol"].get("folds", {}).get(str(int(fold)), {}).get("fold_hash")
    if stage != "final" and feature_metadata.get("fold_hash") != expected_fold_hash:
        raise ValueError("Merged head features have a fold identity mismatch.")
    train_x, train_rows = _load_features(feature_dir, "outer_train")
    holdout_x, holdout_rows = _load_features(feature_dir, "outer_holdout")
    if train_x.shape[1] != holdout_x.shape[1]:
        raise ValueError("Merged train/holdout feature dimensions do not match.")
    if stage == "final" and {str(row["dataset"]) for row in holdout_rows} != {"daic"}:
        raise ValueError("Final merged heads may evaluate only the untouched DAIC official test.")
    expected_trial_count = resolve_optuna_trials(merged_config, stage, trials)
    from src.features import optuna100_policy as policy

    optuna_cfg = (merged_config.get("heads") or {}).get("optuna") or {}
    protocol_profile = str(optuna_cfg.get("protocol_profile") or "")
    if expected_trial_count == 0:
        print("Optuna disabled (trials=0); fitting logreg and fixed XGBoost only.", flush=True)
    prediction_backend = (
        policy.prediction_backend(model_backend, merged=True)
        if protocol_profile == policy.PROTOCOL_PROFILE
        else None
    )
    output_root = feature_dir.parent / "heads"
    identity = {
        "schema_version": "symmetric_merged_heads_identity.v1",
        "stage": stage,
        "fold": int(fold),
        "run_id": run_id,
        "feature_metadata": str(feature_dir / "feature_metadata.json"),
        "feature_dimension": int(train_x.shape[1]),
        "manifest_hash": feature_metadata.get("manifest_hash"),
        "split_hash": feature_metadata.get("split_hash"),
        "merged_config_sha256": sha256_file(resolved_config_path),
        "fold_hash": feature_metadata.get("fold_hash"),
        "optuna_trials": expected_trial_count,
        "inner_folds": resolve_head_inner_folds(merged_config, stage),
        "threshold": float((merged_config.get("protocol_settings") or {}).get("threshold", 0.5)),
        "model_backend": model_backend,
        "prediction_backend": prediction_backend,
    }
    identity_path = output_root / "heads_identity.json"
    complete_path = output_root / "heads_complete.json"
    if complete_path.is_file() and identity_path.is_file():
        existing_identity = read_json(identity_path)
        existing_identity.setdefault("model_backend", "")
        existing_identity.setdefault("prediction_backend", None)
        if existing_identity != identity:
            raise ValueError(f"Incompatible completed merged heads: {output_root}")
        return {"status": "skipped_compatible_complete", "output_root": str(output_root)}
    if output_root.exists() and any(output_root.iterdir()) and not identity_path.is_file():
        raise ValueError(f"Refusing to overwrite incomplete merged heads: {output_root}")
    ensure_dir(output_root)
    save_json(identity, identity_path)
    save_json(merged_config, output_root / "resolved_merged_config.json")
    write_slurm_provenance(
        output_root / "slurm_provenance.json",
        worker="src.merged.heads",
        stage=stage,
        fold=int(fold),
        run_id=run_id,
        feature_dimension=int(train_x.shape[1]),
        manifest_hash=feature_metadata.get("manifest_hash"),
        split_hash=feature_metadata.get("split_hash"),
    )
    inner_folds = int(identity["inner_folds"])
    assignments = build_grouped_inner_folds(
        train_rows,
        inner_folds=inner_folds,
        seed=int(((merged_config.get("heads") or {}).get("optuna") or {}).get("inner_seed", 1337)),
    )
    save_json(assignments, output_root / "inner_folds.json")
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    seed = int(merged_config.get("seed", 1337))
    method_summaries: dict[str, Any] = {}
    for method in HEADS:
        if expected_trial_count == 0 and method == "xgb_optuna":
            print("Skipping xgb_optuna (trials=0).", flush=True)
            continue
        method_dir = ensure_dir(output_root / method)
        if method == "logreg":
            estimator = _new_logreg(seed)
            weight_audit = _fit_weighted(estimator, train_x, y_train, train_rows)
            params = {"C": 1.0, "standardized": True}
        elif method == FIXED_HEAD:
            params = fixed_xgb_params(merged_config, seed, int(((merged_config.get("heads") or {}).get("optuna") or {}).get("xgb_threads", 20)))
            estimator = _new_xgb(params)
            weight_audit = _fit_weighted(estimator, train_x, y_train, train_rows)
        else:
            params, optuna_summary = _run_optuna(
                train_x,
                train_rows,
                merged_config,
                assignments,
                method_dir / "optuna",
                trials=expected_trial_count,
            )
            estimator = _new_xgb(params)
            weight_audit = _fit_weighted(estimator, train_x, y_train, train_rows)
        probabilities = _predict_probability(estimator, holdout_x)
        prediction_groups, metrics = aggregate_head_predictions(
            holdout_rows,
            probabilities,
            threshold=float((merged_config.get("protocol_settings") or {}).get("threshold", 0.5)),
        )
        all_prediction_rows = [row for dataset in sorted(prediction_groups) for row in prediction_groups[dataset]]
        write_jsonl(all_prediction_rows, method_dir / "predictions_subject_level.jsonl")
        _write_csv(all_prediction_rows, method_dir / "predictions_subject_level.csv")
        save_json(metrics, method_dir / "metrics_by_dataset.json")
        save_json(
            {
                "method": method,
                "params": params,
                "weight_audit": weight_audit,
                "threshold": float((merged_config.get("protocol_settings") or {}).get("threshold", 0.5)),
                "input_dimension": int(train_x.shape[1]),
                "training_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
                "holdout_subject_ids": sorted({str(row["subject_id"]) for row in holdout_rows}),
                "manifest_hash": feature_metadata.get("manifest_hash"),
                "split_hash": feature_metadata.get("split_hash"),
                "model_backend": model_backend,
                "prediction_backend": prediction_backend,
            },
            method_dir / "classifier_metadata.json",
        )
        import joblib

        joblib.dump(estimator, method_dir / "classifier.joblib")
        method_summaries[method] = {"metrics": metrics, "output_dir": str(method_dir)}
    save_json(method_summaries, output_root / "summary.json")
    save_json(
        {
            "status": "completed",
            "identity": identity,
            "methods": list(HEADS),
            "optuna_trials": expected_trial_count,
            "method_summaries": method_summaries,
        },
        complete_path,
    )
    return {"status": "completed", "output_root": str(output_root), "methods": list(HEADS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit merged Logistic Regression, fixed XGBoost, and Optuna XGBoost heads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = run_merged_heads(
        args.config,
        stage=args.stage,
        fold=args.fold,
        run_id=args.run_id,
        features_dir=args.features_dir,
        trials=args.trials,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
