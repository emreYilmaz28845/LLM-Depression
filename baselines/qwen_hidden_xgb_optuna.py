from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.aggregate import aggregate_binary_classifier_predictions
from src.metrics import classification_metrics
from src.utils import read_json, read_jsonl, save_json_atomic, write_jsonl


CLASSIFIER_VARIANT = "xgb_optuna_raw"
THRESHOLD = 0.5
CONFIG_SCHEMA_VERSION = "qwen_hidden_xgb_optuna.v1"
SUPPORTED_OBJECTIVES = ("positive_f1", "macro_f1")
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
FINAL_ARTIFACT_NAMES = (
    "pipeline.joblib",
    "predictions_sample_level.jsonl",
    "predictions_sample_level.csv",
    "predictions_subject_level.jsonl",
    "predictions_subject_level.csv",
    "metrics.json",
    "classifier_metadata.json",
)


def _load_partition(cache_dir: Path, name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(cache_dir / f"{name}.npz") as payload:
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    rows = read_jsonl(cache_dir / f"{name}_rows.jsonl")
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError(f"Invalid {name} cache shape {vectors.shape} for {len(rows)} rows.")
    if not bool(np.isfinite(vectors).all()):
        raise ValueError(f"{name} cache contains non-finite values.")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {name} cache.")
    return vectors, rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    tmp_path.replace(path)


def _write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    write_jsonl(rows, tmp_path)
    tmp_path.replace(path)


def _dump_joblib_atomic(value: Any, path: Path) -> None:
    import joblib

    tmp_path = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, tmp_path)
    tmp_path.replace(path)


def _metrics_with_negative_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    output = dict(metrics)
    output["negative_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return output


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _subject_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Subject {subject_id} has non-binary label {label}.")
        if subject_id in labels and labels[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        labels[subject_id] = label
    if set(labels.values()) != {0, 1}:
        raise ValueError("Outer-training subjects must contain both classes.")
    return labels


def build_inner_subject_assignments(
    rows: list[dict[str, Any]],
    *,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    labels_by_subject = _subject_labels(rows)
    subject_ids = sorted(labels_by_subject)
    counts = Counter(labels_by_subject.values())
    if min(counts.values()) < inner_folds:
        raise ValueError(
            f"Each class needs at least {inner_folds} outer-training subjects; found {dict(counts)}."
        )
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    y = np.asarray([labels_by_subject[subject_id] for subject_id in subject_ids], dtype=np.int64)
    row_indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_indices_by_subject[str(row["subject_id"])].append(index)

    folds: list[dict[str, Any]] = []
    validation_coverage: list[str] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(subject_ids, y)):
        train_subjects = [subject_ids[index] for index in train_idx.tolist()]
        val_subjects = [subject_ids[index] for index in val_idx.tolist()]
        if set(train_subjects) & set(val_subjects):
            raise ValueError(f"Inner fold {fold_index} has train/validation subject overlap.")
        validation_coverage.extend(val_subjects)
        folds.append(
            {
                "fold": fold_index,
                "train_subject_ids": train_subjects,
                "validation_subject_ids": val_subjects,
                "train_row_indices": [
                    row_index for subject_id in train_subjects for row_index in row_indices_by_subject[subject_id]
                ],
                "validation_row_indices": [
                    row_index for subject_id in val_subjects for row_index in row_indices_by_subject[subject_id]
                ],
            }
        )
    if Counter(validation_coverage) != Counter(subject_ids):
        raise ValueError("Inner validation folds do not cover each subject exactly once.")
    return {
        "schema_version": "inner_subject_assignments.v1",
        "splitter": "StratifiedKFold",
        "n_splits": inner_folds,
        "shuffle": True,
        "random_state": seed,
        "subjects": [{"subject_id": subject_id, "label": labels_by_subject[subject_id]} for subject_id in subject_ids],
        "folds": folds,
    }


def _suggest_params(trial: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["kind"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["kind"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        else:
            raise ValueError(f"Unsupported search space kind for {name}: {spec['kind']}")
    return params


def fixed_xgb_params(seed: int, xgb_threads: int) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": xgb_threads,
    }


def _classifier(params: dict[str, Any], fixed_params: dict[str, Any]):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Optuna raw XGBoost requires xgboost-cpu==2.1.4.") from exc
    return XGBClassifier(**fixed_params, **params)


def _sample_rows_for_predictions(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    condition = str(metadata.get("condition") or metadata["input_modality"])
    return [
        {
            "dataset": metadata["dataset"],
            "modality": metadata["input_modality"],
            "condition": condition,
            "fold": int(metadata["fold"]),
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "probability": float(probability),
            "predicted_class": int(prediction),
            "checkpoint": metadata.get("checkpoint_dir"),
            "classifier_variant": CLASSIFIER_VARIANT,
        }
        for row, probability, prediction in zip(rows, probabilities.tolist(), predictions.tolist())
    ]


def make_objective(
    *,
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    assignments: dict[str, Any],
    objective_name: str,
    fixed_params: dict[str, Any],
) -> Callable[[Any], float]:
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    outer_subjects = {str(row["subject_id"]) for row in train_rows}

    def objective(trial: Any) -> float:
        params = _suggest_params(trial)
        oof_subject_rows: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold in assignments["folds"]:
            train_idx = np.asarray(fold["train_row_indices"], dtype=np.int64)
            val_idx = np.asarray(fold["validation_row_indices"], dtype=np.int64)
            model = _classifier(params, fixed_params)
            model.fit(train_x[train_idx], train_y[train_idx])
            probabilities = np.asarray(model.predict_proba(train_x[val_idx])[:, 1], dtype=np.float64)
            predictions = (probabilities >= THRESHOLD).astype(np.int64)
            val_rows = [train_rows[index] for index in val_idx.tolist()]
            sample_rows = _sample_rows_for_predictions(val_rows, probabilities, predictions, metadata)
            subject_rows, metrics = aggregate_binary_classifier_predictions(sample_rows)
            fold_metrics.append({"inner_fold": int(fold["fold"]), **_metrics_with_negative_f1(metrics)})
            oof_subject_rows.extend(subject_rows)
        subject_ids = [str(row["subject_id"]) for row in oof_subject_rows]
        if Counter(subject_ids) != Counter(outer_subjects):
            raise ValueError("Trial OOF predictions do not cover each outer-training subject exactly once.")
        y_true = [int(row["label"]) for row in oof_subject_rows]
        y_pred = [int(row["prediction"]) for row in oof_subject_rows]
        pooled_metrics = _metrics_with_negative_f1(classification_metrics(y_true, y_pred))
        trial.set_user_attr("inner_fold_metrics", fold_metrics)
        trial.set_user_attr("inner_oof_metrics", pooled_metrics)
        return float(pooled_metrics[objective_name])

    return objective


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_study_config(
    *,
    cache_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    objective_name: str,
    target_trials: int,
    inner_folds: int,
    seed: int,
    xgb_threads: int,
) -> tuple[dict[str, Any], str]:
    try:
        import optuna
        import xgboost
    except ImportError as exc:
        raise RuntimeError("Optuna raw XGBoost requires optuna==4.4.0 and xgboost-cpu==2.1.4.") from exc
    import sklearn

    condition = str(metadata.get("condition") or metadata["input_modality"])
    study_name = f"{metadata['dataset']}_{condition}_fold{int(metadata['fold'])}_{CLASSIFIER_VARIANT}_{objective_name}"
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "condition": condition,
        "outer_fold": int(metadata["fold"]),
        "run_name": output_dir.parent.parent.name if output_dir.name == CLASSIFIER_VARIANT else output_dir.parent.name,
        "classifier_variant": CLASSIFIER_VARIANT,
        "cache_dir": str(cache_dir),
        "cache_identity": {
            "outer_train_npz": _file_identity(cache_dir / "outer_train.npz"),
            "outer_train_rows": _file_identity(cache_dir / "outer_train_rows.jsonl"),
            "extraction_metadata": _file_identity(cache_dir / "extraction_metadata.json"),
        },
        "extraction_metadata_identity": _file_identity(cache_dir / "extraction_metadata.json"),
        "objective": objective_name,
        "target_trials": target_trials,
        "inner_fold_count": inner_folds,
        "seed": seed,
        "aggregation_method": "aggregate_binary_classifier_predictions",
        "threshold": THRESHOLD,
        "fixed_xgb_params": fixed_xgb_params(seed, xgb_threads),
        "search_space": SEARCH_SPACE,
        "study_name": study_name,
        "packages": {
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    config_hash = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return config, config_hash


def _write_or_validate_study_config(output_dir: Path, config: dict[str, Any], config_hash: str) -> None:
    path = output_dir / "study_config.json"
    payload = {"canonical_config": config, "config_sha256": config_hash}
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise ValueError(f"Existing study_config.json differs for {output_dir}. Refusing to resume.")
        return
    save_json_atomic(payload, path)


def _write_or_validate_json(data: Any, path: Path, artifact_name: str) -> None:
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Existing {artifact_name} differs for {path.parent}. Refusing to resume.")
        return
    save_json_atomic(data, path)


def _study(output_dir: Path, study_name: str, config: dict[str, Any], config_hash: str):
    import optuna

    storage = f"sqlite:///{output_dir / 'study.sqlite3'}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"])),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    existing_hash = study.user_attrs.get("config_sha256")
    existing_config = study.user_attrs.get("canonical_config")
    if existing_hash is not None and existing_hash != config_hash:
        raise ValueError("Existing Optuna study config hash differs. Refusing to resume.")
    if existing_config is not None and existing_config != config:
        raise ValueError("Existing Optuna study config differs. Refusing to resume.")
    study.set_user_attr("config_sha256", config_hash)
    study.set_user_attr("canonical_config", config)
    return study


def _completed_trials(study: Any) -> list[Any]:
    import optuna

    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def _write_trials_csv(study: Any, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
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
    _write_csv(rows, path)


def _best_payload(study: Any, objective_name: str, fixed_params: dict[str, Any]) -> dict[str, Any]:
    best = study.best_trial
    return {
        "best_trial_number": int(best.number),
        "objective": objective_name,
        "objective_value": float(best.value),
        "suggested_params": dict(best.params),
        "fixed_params": dict(fixed_params),
        "completed_trial_count": len(_completed_trials(study)),
    }


def _inner_coverage(assignments: dict[str, Any]) -> dict[str, Any]:
    subjects = [str(item["subject_id"]) for item in assignments["subjects"]]
    validation_subjects = [
        str(subject_id) for fold in assignments["folds"] for subject_id in fold["validation_subject_ids"]
    ]
    return {
        "subject_count": len(subjects),
        "validation_subject_count": len(validation_subjects),
        "validation_covers_each_subject_once": Counter(subjects) == Counter(validation_subjects),
        "fold_validation_subject_counts": [
            len(fold["validation_subject_ids"]) for fold in assignments["folds"]
        ],
    }


def _completed_final_result(
    output_dir: Path,
    *,
    config_hash: str,
    target_trials: int,
) -> dict[str, Any] | None:
    metadata_path = output_dir / "classifier_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = read_json(metadata_path)
    if metadata.get("search_config_sha256") != config_hash:
        raise ValueError("Existing final classifier metadata has a different search config hash.")
    if int(metadata.get("completed_trials", -1)) != target_trials:
        raise ValueError("Existing final classifier metadata has a different completed-trial count.")
    if not all((output_dir / name).is_file() for name in FINAL_ARTIFACT_NAMES):
        return None
    return {"variant": CLASSIFIER_VARIANT, **read_json(output_dir / "metrics.json")}


def run_optuna_raw_xgb(
    *,
    cache_dir: Path,
    output_dir: Path,
    objective_name: str,
    target_trials: int,
    inner_folds: int,
    seed: int,
    xgb_threads: int,
) -> dict[str, Any]:
    if objective_name not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective {objective_name!r}; expected one of {SUPPORTED_OBJECTIVES}.")
    if target_trials < 1:
        raise ValueError("target_trials must be at least 1.")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2.")
    if xgb_threads < 1:
        raise ValueError("xgb_threads must be at least 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_rows = _load_partition(cache_dir, "outer_train")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    assignments = build_inner_subject_assignments(train_rows, inner_folds=inner_folds, seed=seed)
    config, config_hash = build_study_config(
        cache_dir=cache_dir,
        output_dir=output_dir,
        metadata=metadata,
        objective_name=objective_name,
        target_trials=target_trials,
        inner_folds=inner_folds,
        seed=seed,
        xgb_threads=xgb_threads,
    )
    _write_or_validate_study_config(output_dir, config, config_hash)
    _write_or_validate_json(
        assignments,
        output_dir / "inner_subject_assignments.json",
        "inner_subject_assignments.json",
    )
    fixed_params = fixed_xgb_params(seed, xgb_threads)
    study = _study(output_dir, config["study_name"], config, config_hash)
    completed = len(_completed_trials(study))
    if target_trials < completed:
        raise ValueError(f"target_trials={target_trials} is lower than already completed trials={completed}.")
    remaining = target_trials - completed
    if remaining:
        objective = make_objective(
            train_x=train_x,
            train_rows=train_rows,
            metadata=metadata,
            assignments=assignments,
            objective_name=objective_name,
            fixed_params=fixed_params,
        )
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    completed = len(_completed_trials(study))
    if completed != target_trials:
        raise ValueError(f"Expected {target_trials} completed trials, found {completed}.")

    _write_trials_csv(study, output_dir / "trials.csv")
    best = _best_payload(study, objective_name, fixed_params)
    save_json_atomic(best, output_dir / "best_params.json")
    best_trial = study.best_trial
    save_json_atomic(best_trial.user_attrs["inner_fold_metrics"], output_dir / "inner_fold_metrics.json")
    save_json_atomic(best_trial.user_attrs["inner_oof_metrics"], output_dir / "inner_oof_metrics.json")

    completed_result = _completed_final_result(
        output_dir,
        config_hash=config_hash,
        target_trials=target_trials,
    )
    if completed_result is not None:
        return completed_result

    final_x, final_rows = _load_partition(cache_dir, "final_eval")
    if final_x.shape[1] != train_x.shape[1]:
        raise ValueError(
            f"Training/final feature dimensions differ: {train_x.shape[1]} != {final_x.shape[1]}."
        )
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    final_subjects = {str(row["subject_id"]) for row in final_rows}
    overlap = sorted(train_subjects & final_subjects)
    if overlap:
        raise ValueError(f"Training/held-out subject leakage: {overlap[:10]}")
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    model = _classifier(dict(best_trial.params), fixed_params)
    model.fit(train_x, train_y)
    probabilities = np.asarray(model.predict_proba(final_x)[:, 1], dtype=np.float64)
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    sample_rows = _sample_rows_for_predictions(final_rows, probabilities, predictions, metadata)
    subject_rows, metrics = aggregate_binary_classifier_predictions(sample_rows)
    metrics = _metrics_with_negative_f1(metrics)
    condition = str(metadata.get("condition") or metadata["input_modality"])
    for row in subject_rows:
        row.update(
            {
                "dataset": metadata["dataset"],
                "modality": metadata["input_modality"],
                "condition": condition,
                "fold": int(metadata["fold"]),
                "classifier_variant": CLASSIFIER_VARIANT,
            }
        )
    _dump_joblib_atomic(model, output_dir / "pipeline.joblib")
    _write_jsonl_atomic(sample_rows, output_dir / "predictions_sample_level.jsonl")
    _write_jsonl_atomic(subject_rows, output_dir / "predictions_subject_level.jsonl")
    _write_csv(sample_rows, output_dir / "predictions_sample_level.csv")
    _write_csv(subject_rows, output_dir / "predictions_subject_level.csv")
    save_json_atomic(metrics, output_dir / "metrics.json")
    artifact_metadata = {
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "condition": condition,
        "fold": int(metadata["fold"]),
        "run_name": config["run_name"],
        "classifier_variant": CLASSIFIER_VARIANT,
        "seed": seed,
        "threshold": THRESHOLD,
        "inner_fold_count": inner_folds,
        "target_trials": target_trials,
        "completed_trials": completed,
        "objective": objective_name,
        "best_value": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "search_config_sha256": config_hash,
        "optuna_version": config["packages"]["optuna"],
        "xgboost_version": config["packages"]["xgboost"],
        "input_dimension": int(train_x.shape[1]),
        "post_pca_dimension": int(train_x.shape[1]),
        "requested_pca_components": None,
        "effective_pca_components": None,
        "training_row_ids": [str(row["sample_id"]) for row in train_rows],
        "training_subject_ids": sorted(train_subjects),
        "heldout_row_ids": [str(row["sample_id"]) for row in final_rows],
        "heldout_subject_ids": sorted(final_subjects),
        "outer_subject_overlap_count": len(overlap),
        "inner_subject_coverage": _inner_coverage(assignments),
        "inner_subject_assignments": assignments,
        "cache_dir": str(cache_dir),
        "extraction_metadata": str(cache_dir / "extraction_metadata.json"),
        "original_extraction_metadata": metadata,
        "evaluation_protocol": metadata.get("evaluation_protocol") or metadata.get("evaluation_provenance"),
        "turkish_table_aligned_warning": (
            "Turkish hidden-state classifier results are table-aligned outer-validation estimates, "
            "not unseen-test estimates; the underlying Qwen checkpoints were selected on those validation folds."
            if str(metadata["dataset"]).lower() == "turkish"
            else None
        ),
    }
    save_json_atomic(artifact_metadata, output_dir / "classifier_metadata.json")
    return {"variant": CLASSIFIER_VARIANT, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-tuned raw XGBoost on cached Qwen hidden vectors.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objective", required=True, choices=SUPPORTED_OBJECTIVES)
    parser.add_argument("--target-trials", type=int, default=50)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--xgb-threads", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_optuna_raw_xgb(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        objective_name=args.objective,
        target_trials=args.target_trials,
        inner_folds=args.inner_folds,
        seed=args.seed,
        xgb_threads=args.xgb_threads,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
