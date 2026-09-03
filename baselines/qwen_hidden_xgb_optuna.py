from __future__ import annotations

import argparse
import csv
import hashlib
import os
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.aggregate import (
    aggregate_binary_classifier_predictions,
    aggregate_binary_classifier_response_rows,
)
from src.features.hidden_classifier_policy import (
    TURKISH_POOLED_TEXT_PAIR_POLICY,
    cache_identity,
    canonical_sha256,
    classifier_aggregation_policy,
    response_normalized_sample_weights,
)
from src.metrics import classification_metrics
from src.sampling import (
    SAMPLING_MODE_NONE,
    SAMPLING_MODE_SUBJECT_OVERSAMPLE,
    build_no_sampling_audit,
    build_subject_oversampling,
)
from src.utils import read_json, read_jsonl, save_json_atomic, write_jsonl


CLASSIFIER_FAMILY = "xgb_optuna_raw"
CLASSIFIER_VARIANT = CLASSIFIER_FAMILY
DEFAULT_EXPERIMENT_ID = CLASSIFIER_FAMILY
THRESHOLD = 0.5
CONFIG_SCHEMA_VERSION = "qwen_hidden_xgb_optuna.v2"
SUPPORTED_OBJECTIVES = ("positive_f1", "macro_f1")
SUPPORTED_SEARCH_PROFILES = ("standard_d6", "depth8")
LEGACY_SAMPLING_MODE = "legacy"
SUPPORTED_SAMPLING_MODES = (SAMPLING_MODE_NONE, SAMPLING_MODE_SUBJECT_OVERSAMPLE)
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
STANDARD_SEARCH_SPACE: dict[str, dict[str, Any]] = {
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
# Backward-compatible public constant used by existing tests and callers.
SEARCH_SPACE = STANDARD_SEARCH_SPACE
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


def resolved_search_space(search_profile: str) -> dict[str, dict[str, Any]]:
    if search_profile not in SUPPORTED_SEARCH_PROFILES:
        raise ValueError(
            f"Unsupported search profile {search_profile!r}; expected one of "
            f"{SUPPORTED_SEARCH_PROFILES}."
        )
    search_space = {name: dict(spec) for name, spec in STANDARD_SEARCH_SPACE.items()}
    if search_profile == "depth8":
        search_space["max_depth"]["high"] = 8
    return search_space


def resolved_oversampling_search_space(
    search_profile: str,
    sampling_mode: str,
) -> dict[str, dict[str, Any]]:
    search_space = resolved_search_space(search_profile)
    if sampling_mode in SUPPORTED_SAMPLING_MODES:
        search_space.pop("scale_pos_weight", None)
    return search_space


def _suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec["kind"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["kind"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        else:
            raise ValueError(f"Unsupported search space kind for {name}: {spec['kind']}")
    return params


def fixed_xgb_params(
    seed: int,
    xgb_threads: int,
    *,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
) -> dict[str, Any]:
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": xgb_threads,
    }
    if sampling_mode in SUPPORTED_SAMPLING_MODES:
        params["scale_pos_weight"] = 1.0
    return params


def _fit_indices_and_audit(
    rows: list[dict[str, Any]],
    source_indices: list[int],
    validation_indices: list[int],
    *,
    sampling_mode: str,
    oversampling_ratio: float | None,
    oversampling_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_rows = [rows[index] for index in source_indices]
    validation_rows = [rows[index] for index in validation_indices]
    if sampling_mode == SAMPLING_MODE_SUBJECT_OVERSAMPLE:
        result = build_subject_oversampling(
            source_rows,
            ratio=oversampling_ratio,
            seed=oversampling_seed,
            expected_minority_label=0,
            validation_rows=validation_rows,
        )
    elif sampling_mode in {SAMPLING_MODE_NONE, LEGACY_SAMPLING_MODE}:
        result = build_no_sampling_audit(
            source_rows,
            seed=oversampling_seed,
            validation_rows=validation_rows,
        )
    else:
        raise ValueError(f"Unsupported sampling_mode {sampling_mode!r}.")
    fit_indices = np.asarray(
        [source_indices[index] for index in result.indices],
        dtype=np.int64,
    )
    return fit_indices, result.audit


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
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
    oversampling_ratio: float | None = None,
    oversampling_seed: int = 1337,
    prediction_backend: str | None = None,
) -> list[dict[str, Any]]:
    condition = str(metadata.get("condition") or metadata["input_modality"])
    output: list[dict[str, Any]] = []
    for row, probability, prediction in zip(rows, probabilities.tolist(), predictions.tolist()):
        record: dict[str, Any] = {
            "dataset": metadata["dataset"],
            "modality": metadata["input_modality"],
            "condition": condition,
            "dataset_variant": metadata.get("dataset_variant", ""),
            "fold": int(metadata["fold"]),
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            **{
                key: row[key]
                for key in ("response_id", "prompt_id", "segment_index", "num_segments")
                if key in row
            },
            "label": int(row["label"]),
            "probability": float(probability),
            "predicted_class": int(prediction),
            "checkpoint": metadata.get("checkpoint_dir"),
            "classifier_family": CLASSIFIER_FAMILY,
            "classifier_variant": experiment_id,
            "sampling_mode": sampling_mode,
            "oversampling_ratio": oversampling_ratio,
            "oversampling_seed": int(oversampling_seed),
        }
        if metadata.get("dataset_variant") == "pooled_t17":
            record["question_condition"] = str(row.get("question_condition", ""))
        if metadata.get("dataset_variant") == "pooled_t17" and metadata.get("input_modality") == "text_only":
            record["aggregation_policy"] = TURKISH_POOLED_TEXT_PAIR_POLICY
        else:
            record["aggregation_policy"] = classifier_aggregation_policy(metadata)
        if prediction_backend is not None:
            record["prediction_backend"] = prediction_backend
            record["model_backend"] = metadata.get("model_backend")
        output.append(record)
    return output


def make_objective(
    *,
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    assignments: dict[str, Any],
    objective_name: str,
    fixed_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]] | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
    oversampling_ratio: float | None = None,
    oversampling_seed: int = 1337,
    prediction_backend: str | None = None,
) -> Callable[[Any], float]:
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    outer_subjects = {str(row["subject_id"]) for row in train_rows}
    resolved_space = search_space or resolved_oversampling_search_space(
        "standard_d6", sampling_mode
    )

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, resolved_space)
        oof_subject_rows: list[dict[str, Any]] = []
        oof_sample_rows: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold in assignments["folds"]:
            if sampling_mode == LEGACY_SAMPLING_MODE:
                train_idx = np.asarray(fold["train_row_indices"], dtype=np.int64)
            else:
                train_idx, _ = _fit_indices_and_audit(
                    train_rows,
                    list(fold["train_row_indices"]),
                    list(fold["validation_row_indices"]),
                    sampling_mode=sampling_mode,
                    oversampling_ratio=oversampling_ratio,
                    oversampling_seed=oversampling_seed,
                )
            val_idx = np.asarray(fold["validation_row_indices"], dtype=np.int64)
            model = _classifier(params, fixed_params)
            fit_rows = [train_rows[index] for index in train_idx.tolist()]
            fit_weights, weight_audit = response_normalized_sample_weights(
                fit_rows,
                metadata,
            )
            model.fit(
                train_x[train_idx],
                train_y[train_idx],
                sample_weight=fit_weights,
            )
            probabilities = np.asarray(model.predict_proba(train_x[val_idx])[:, 1], dtype=np.float64)
            predictions = (probabilities >= THRESHOLD).astype(np.int64)
            val_rows = [train_rows[index] for index in val_idx.tolist()]
            sample_rows = _sample_rows_for_predictions(
                val_rows,
                probabilities,
                predictions,
                metadata,
                experiment_id,
                sampling_mode,
                oversampling_ratio,
                oversampling_seed,
                prediction_backend,
            )
            subject_rows, metrics = aggregate_binary_classifier_predictions(
                sample_rows,
                prediction_backend=prediction_backend or "qwen_hidden_classifier",
            )
            fold_metrics.append(
                {
                    "inner_fold": int(fold["fold"]),
                    "weight_audit": weight_audit,
                    **_metrics_with_negative_f1(metrics),
                }
            )
            oof_subject_rows.extend(subject_rows)
            oof_sample_rows.extend(sample_rows)
        subject_ids = [str(row["subject_id"]) for row in oof_subject_rows]
        if Counter(subject_ids) != Counter(outer_subjects):
            raise ValueError("Trial OOF predictions do not cover each outer-training subject exactly once.")
        y_true = [int(row["label"]) for row in oof_subject_rows]
        y_pred = [int(row["prediction"]) for row in oof_subject_rows]
        # The pooled text contract reduces the two condition rows to one
        # subject decision before scoring.  That decision may be INVALID on a
        # zero margin, so use the central strict-metric result rather than
        # passing INVALID=-1 into the binary metrics helper.
        if (
            str(metadata.get("dataset", "")).lower() == "turkish"
            and str(metadata.get("dataset_variant", "")).strip() == "pooled_t17"
            and str(metadata.get("input_modality", "")).strip() == "text_only"
        ):
            _, pooled_raw_metrics = aggregate_binary_classifier_predictions(
                oof_sample_rows,
                prediction_backend=prediction_backend or "qwen_hidden_classifier",
            )
            pooled_metrics = _metrics_with_negative_f1(pooled_raw_metrics)
        else:
            pooled_metrics = _metrics_with_negative_f1(classification_metrics(y_true, y_pred))
        trial.set_user_attr("inner_fold_metrics", fold_metrics)
        trial.set_user_attr("inner_oof_metrics", pooled_metrics)
        return float(pooled_metrics[objective_name])

    return objective


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_experiment_output(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    experiment_id: str,
) -> str:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id must be a lowercase slug containing only letters, digits, "
            "and single underscore separators."
        )
    if output_dir.name != experiment_id:
        raise ValueError(
            f"Output directory basename {output_dir.name!r} must equal experiment_id "
            f"{experiment_id!r}."
        )
    expected_fold = f"fold_{int(metadata['fold'])}"
    if output_dir.parent.name == expected_fold:
        return output_dir.parent.parent.name
    # The v2 managed head is a post-hoc attempt nested below the parent fold.
    # Keep the canonical experiment-id directory required by the Optuna
    # artifact contract, while placing it inside the unique attempt directory
    # that carries modern sidecars and the retry identity.
    attempt_env = (
        "TURKISH_POOLED_QCOND_ATTEMPT_DIR"
        if metadata.get("dataset_variant") == "pooled_t17"
        else "NATIVE_EN_TEXT_HEADS_V2_ATTEMPT_DIR"
    )
    if os.environ.get(attempt_env):
        attempt_dir = Path(os.environ[attempt_env]).resolve()
        if output_dir.parent.resolve() != attempt_dir or not (attempt_dir / "metadata.json").is_file():
            raise ValueError(
                "v2 Optuna output must be directly below its tracked attempt directory"
            )
        return attempt_dir.name
    raise ValueError(
        f"Output directory must be directly below {expected_fold}, found "
        f"{output_dir.parent.name!r}."
    )


def build_study_config(
    *,
    cache_dir: Path,
    metadata: dict[str, Any],
    objective_name: str,
    target_trials: int,
    inner_folds: int,
    seed: int,
    inner_seed: int,
    xgb_threads: int,
    experiment_id: str,
    search_profile: str,
    run_name: str,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
    oversampling_ratio: float | None = None,
    oversampling_seed: int = 1337,
    protocol_profile: str | None = None,
    prediction_backend: str | None = None,
) -> tuple[dict[str, Any], str]:
    try:
        import optuna
        import xgboost
    except ImportError as exc:
        raise RuntimeError("Optuna raw XGBoost requires optuna==4.4.0 and xgboost-cpu==2.1.4.") from exc
    import sklearn

    condition = str(metadata.get("condition") or metadata["input_modality"])
    if protocol_profile is not None:
        from src.features import optuna100_policy as policy

        search_space = policy.resolved_search_space()
        stage_env = (
            "TURKISH_POOLED_QCOND_STAGE"
            if metadata.get("dataset_variant") == "pooled_t17"
            else "NATIVE_EN_TEXT_HEADS_STAGE"
        )
        stage = str(os.environ.get(stage_env, "production")).lower()
        policy.assert_target(target_trials, stage=stage)
    else:
        search_space = resolved_oversampling_search_space(search_profile, sampling_mode)
    if protocol_profile is not None or sampling_mode == LEGACY_SAMPLING_MODE:
        study_name = (
            f"{metadata['dataset']}_{condition}_fold{int(metadata['fold'])}_"
            f"{experiment_id}_{objective_name}"
        )
    else:
        ratio_token = (
            "na"
            if oversampling_ratio is None
            else f"{int(round(float(oversampling_ratio) * 100)):03d}"
        )
        study_name = (
            f"{metadata['dataset']}_{condition}_fold{int(metadata['fold'])}_"
            f"{experiment_id}_{objective_name}_{sampling_mode}_r{ratio_token}_os{oversampling_seed}"
        )
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "condition": condition,
        "outer_fold": int(metadata["fold"]),
        "run_name": run_name,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_variant": experiment_id,
        "experiment_id": experiment_id,
        "cache_dir": str(cache_dir),
        "cache_identity": cache_identity(cache_dir),
        "extraction_metadata_identity": _file_identity(cache_dir / "extraction_metadata.json"),
        "objective": objective_name,
        "target_trials": target_trials,
        "inner_fold_count": inner_folds,
        "seed": seed,
        "sampler_seed": seed,
        "model_seed": seed,
        "inner_seed": inner_seed,
        "aggregation_method": classifier_aggregation_policy(metadata),
        "threshold": THRESHOLD,
        "fixed_xgb_params": fixed_xgb_params(
            seed, xgb_threads, sampling_mode=sampling_mode
        ),
        "search_profile": search_profile,
        "search_space": search_space,
        "study_name": study_name,
        "packages": {
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    if protocol_profile is not None:
        config.update(
            {
                "protocol_profile": protocol_profile,
                "prediction_backend": prediction_backend,
                "model_backend": metadata.get("model_backend"),
                "cache_parent": {
                    "checkpoint_dir": metadata.get("cache_config", {}).get("checkpoint_dir"),
                    "parent_attempt_id": metadata.get("cache_config", {}).get("parent_attempt_id"),
                    "adapter_config_sha256": metadata.get("cache_config", {}).get(
                        "adapter_config_sha256"
                    ),
                    "adapter_sha256": metadata.get("cache_config", {}).get("adapter_sha256"),
                    "saved_run_config_sha256": metadata.get("cache_config", {}).get(
                        "saved_run_config_sha256"
                    ),
                    "saved_split_sha256": metadata.get("cache_config", {}).get(
                        "saved_split_sha256"
                    ),
                    "manifest_sha256": metadata.get("cache_config", {}).get("manifest_sha256"),
                    "split_metadata_sha256": metadata.get("cache_config", {}).get(
                        "split_metadata_sha256"
                    ),
                },
            }
        )
    if sampling_mode != LEGACY_SAMPLING_MODE:
        config.update(
            {
                "sampling_mode": sampling_mode,
                "oversampling_ratio": oversampling_ratio,
                "oversampling_seed": int(oversampling_seed),
            }
        )
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
    experiment_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    metadata_path = output_dir / "classifier_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = read_json(metadata_path)
    if metadata.get("search_config_sha256") != config_hash:
        raise ValueError("Existing final classifier metadata has a different search config hash.")
    if int(metadata.get("completed_trials", -1)) != target_trials:
        raise ValueError("Existing final classifier metadata has a different completed-trial count.")
    if metadata.get("experiment_id", metadata.get("classifier_variant")) != experiment_id:
        raise ValueError("Existing final classifier metadata has a different experiment ID.")
    required = list(FINAL_ARTIFACT_NAMES)
    if str(metadata.get("dataset", "")).lower() == "d3tec":
        required.extend(
            (
                "inner_weight_audits.json",
                "final_fit_weight_audit.json",
            )
        )
        if metadata.get("input_modality") != "text_only":
            required.extend(
                (
                    "predictions_response_level.jsonl",
                    "predictions_response_level.csv",
                    "metrics_response_level.json",
                )
            )
    if not all((output_dir / name).is_file() for name in required):
        return None
    return {"variant": experiment_id, **read_json(output_dir / "metrics.json")}


def run_optuna_raw_xgb(
    *,
    cache_dir: Path,
    output_dir: Path,
    objective_name: str,
    target_trials: int,
    inner_folds: int,
    seed: int,
    xgb_threads: int,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    search_profile: str = "standard_d6",
    inner_seed: int | None = None,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
    oversampling_ratio: float | None = None,
    oversampling_seed: int = 1337,
    protocol_profile: str | None = None,
) -> dict[str, Any]:
    if objective_name not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective {objective_name!r}; expected one of {SUPPORTED_OBJECTIVES}.")
    if target_trials < 1:
        raise ValueError("target_trials must be at least 1.")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2.")
    if xgb_threads < 1:
        raise ValueError("xgb_threads must be at least 1.")
    if sampling_mode not in (LEGACY_SAMPLING_MODE, *SUPPORTED_SAMPLING_MODES):
        raise ValueError(f"Unsupported sampling_mode {sampling_mode!r}.")
    if sampling_mode != SAMPLING_MODE_SUBJECT_OVERSAMPLE and oversampling_ratio is not None:
        raise ValueError("oversampling_ratio is only valid for minority_subject_oversample.")
    if sampling_mode == SAMPLING_MODE_SUBJECT_OVERSAMPLE and oversampling_ratio is None:
        raise ValueError("oversampling_ratio is required for minority_subject_oversample.")
    resolved_inner_seed = seed if inner_seed is None else inner_seed

    from src.features import optuna100_policy as policy

    metadata = read_json(cache_dir / "extraction_metadata.json")
    prediction_backend: str | None = None
    protocol_profile_value: str | None = None
    if protocol_profile == policy.PROTOCOL_PROFILE:
        policy.assert_target(
            target_trials,
            stage=str(
                os.environ.get(
                    "TURKISH_POOLED_QCOND_STAGE"
                    if metadata.get("dataset_variant") == "pooled_t17"
                    else "NATIVE_EN_TEXT_HEADS_STAGE",
                    "production",
                )
            ),
        )
        policy.assert_protocol_settings(
            inner_folds=inner_folds,
            seed=seed,
            inner_seed=resolved_inner_seed,
            sampling_mode=sampling_mode,
            objective_name=objective_name,
        )
        search_profile = policy.PROTOCOL_PROFILE
        protocol_profile_value = policy.PROTOCOL_PROFILE
    elif protocol_profile is not None:
        raise ValueError(f"Unsupported protocol_profile {protocol_profile!r}.")

    train_x, train_rows = _load_partition(cache_dir, "outer_train")
    if protocol_profile_value is not None:
        prediction_backend = policy.prediction_backend(metadata.get("model_backend"))
    search_space = (
        policy.resolved_search_space()
        if protocol_profile_value is not None
        else resolved_oversampling_search_space(search_profile, sampling_mode)
    )
    run_name = validate_experiment_output(
        output_dir,
        metadata=metadata,
        experiment_id=experiment_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = build_inner_subject_assignments(
        train_rows,
        inner_folds=inner_folds,
        seed=resolved_inner_seed,
    )
    config, config_hash = build_study_config(
        cache_dir=cache_dir,
        metadata=metadata,
        objective_name=objective_name,
        target_trials=target_trials,
        inner_folds=inner_folds,
        seed=seed,
        inner_seed=resolved_inner_seed,
        xgb_threads=xgb_threads,
        experiment_id=experiment_id,
        search_profile=search_profile,
        run_name=run_name,
        sampling_mode=sampling_mode,
        oversampling_ratio=oversampling_ratio,
        oversampling_seed=oversampling_seed,
        protocol_profile=protocol_profile_value,
        prediction_backend=prediction_backend,
    )
    _write_or_validate_study_config(output_dir, config, config_hash)
    _write_or_validate_json(
        assignments,
        output_dir / "inner_subject_assignments.json",
        "inner_subject_assignments.json",
    )
    inner_sampling_audits = []
    inner_weight_audits = []
    for fold in assignments["folds"]:
        fit_indices, audit = _fit_indices_and_audit(
            train_rows,
            list(fold["train_row_indices"]),
            list(fold["validation_row_indices"]),
            sampling_mode=sampling_mode,
            oversampling_ratio=oversampling_ratio,
            oversampling_seed=oversampling_seed,
        )
        inner_sampling_audits.append(
            {
                **audit,
                "inner_fold": int(fold["fold"]),
                "applies_to_all_trials": True,
                "target_trial_count": int(target_trials),
            }
        )
        _, weight_audit = response_normalized_sample_weights(
            [train_rows[index] for index in fit_indices.tolist()],
            metadata,
        )
        inner_weight_audits.append(
            {
                **weight_audit,
                "inner_fold": int(fold["fold"]),
                "applies_to_all_trials": True,
                "target_trial_count": int(target_trials),
            }
        )
    _write_or_validate_json(
        inner_sampling_audits,
        output_dir / "inner_sampling_audits.json",
        "inner_sampling_audits.json",
    )
    _write_or_validate_json(
        inner_weight_audits,
        output_dir / "inner_weight_audits.json",
        "inner_weight_audits.json",
    )
    fixed_params = fixed_xgb_params(
        seed, xgb_threads, sampling_mode=sampling_mode
    )
    if protocol_profile_value is not None:
        # The harmonized_optuna100_v1 search space owns scale_pos_weight;
        # a fixed value would collide with the suggested parameter.
        fixed_params.pop("scale_pos_weight", None)
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
            search_space=search_space,
            experiment_id=experiment_id,
            sampling_mode=sampling_mode,
            oversampling_ratio=oversampling_ratio,
            oversampling_seed=oversampling_seed,
            prediction_backend=prediction_backend,
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
        experiment_id=experiment_id,
        metadata=metadata,
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
    full_train_indices, final_fit_sampling_audit = _fit_indices_and_audit(
        train_rows,
        list(range(len(train_rows))),
        [],
        sampling_mode=sampling_mode,
        oversampling_ratio=oversampling_ratio,
        oversampling_seed=oversampling_seed,
    )
    final_fit_sampling_audit["validation_indices_untouched"] = True
    final_fit_sampling_audit["evaluation_indices_untouched"] = True
    save_json_atomic(final_fit_sampling_audit, output_dir / "final_fit_sampling_audit.json")
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    model = _classifier(dict(best_trial.params), fixed_params)
    final_fit_rows = [train_rows[index] for index in full_train_indices.tolist()]
    final_fit_weights, final_fit_weight_audit = response_normalized_sample_weights(
        final_fit_rows,
        metadata,
    )
    save_json_atomic(
        final_fit_weight_audit,
        output_dir / "final_fit_weight_audit.json",
    )
    model.fit(
        train_x[full_train_indices],
        train_y[full_train_indices],
        sample_weight=final_fit_weights,
    )
    probabilities = np.asarray(model.predict_proba(final_x)[:, 1], dtype=np.float64)
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    sample_rows = _sample_rows_for_predictions(
        final_rows,
        probabilities,
        predictions,
        metadata,
        experiment_id,
        sampling_mode,
        oversampling_ratio,
        oversampling_seed,
        prediction_backend,
    )
    subject_rows, metrics = aggregate_binary_classifier_predictions(
        sample_rows,
        prediction_backend=prediction_backend or "qwen_hidden_classifier",
    )
    metrics = _metrics_with_negative_f1(metrics)
    condition = str(metadata.get("condition") or metadata["input_modality"])
    for row in subject_rows:
        row.update(
            {
                "dataset": metadata["dataset"],
                "modality": metadata["input_modality"],
                "condition": condition,
                "fold": int(metadata["fold"]),
                "classifier_family": CLASSIFIER_FAMILY,
                "classifier_variant": experiment_id,
            }
        )
    _dump_joblib_atomic(model, output_dir / "pipeline.joblib")
    _write_jsonl_atomic(sample_rows, output_dir / "predictions_sample_level.jsonl")
    _write_jsonl_atomic(subject_rows, output_dir / "predictions_subject_level.jsonl")
    _write_csv(sample_rows, output_dir / "predictions_sample_level.csv")
    _write_csv(subject_rows, output_dir / "predictions_subject_level.csv")
    response_rows = []
    if str(metadata["dataset"]).lower() == "d3tec" and metadata["input_modality"] != "text_only":
        response_rows, response_metrics = aggregate_binary_classifier_response_rows(sample_rows)
        for row in response_rows:
            row.update(
                {
                    "dataset": metadata["dataset"],
                    "modality": metadata["input_modality"],
                    "condition": condition,
                    "fold": int(metadata["fold"]),
                    "classifier_family": CLASSIFIER_FAMILY,
                    "classifier_variant": experiment_id,
                }
            )
        _write_jsonl_atomic(
            response_rows,
            output_dir / "predictions_response_level.jsonl",
        )
        _write_csv(response_rows, output_dir / "predictions_response_level.csv")
        save_json_atomic(response_metrics, output_dir / "metrics_response_level.json")
    save_json_atomic(metrics, output_dir / "metrics.json")
    artifact_metadata = {
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "condition": condition,
        "fold": int(metadata["fold"]),
        "run_name": config["run_name"],
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_variant": experiment_id,
        "experiment_id": experiment_id,
        "prediction_backend": prediction_backend,
        "model_backend": metadata.get("model_backend"),
        "protocol_profile": protocol_profile_value,
        "seed": seed,
        "sampler_seed": seed,
        "model_seed": seed,
        "inner_seed": resolved_inner_seed,
        "sampling_mode": sampling_mode,
        "oversampling_ratio": oversampling_ratio,
        "oversampling_seed": int(oversampling_seed),
        "search_profile": search_profile,
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
        "inner_sampling_audits": inner_sampling_audits,
        "inner_weight_audits": inner_weight_audits,
        "final_fit_sampling_audit": final_fit_sampling_audit,
        "final_fit_weight_audit": final_fit_weight_audit,
        "weight_policy": final_fit_weight_audit["policy"],
        "aggregation_policy": classifier_aggregation_policy(metadata),
        "cache_identity": config["cache_identity"],
        "cache_identity_sha256": canonical_sha256(config["cache_identity"]),
        "checkpoint_hashes": {
            "adapter_config_sha256": metadata.get("adapter_config_sha256"),
            "adapter_sha256": metadata.get("adapter_sha256"),
        },
        "split_hashes": {
            "saved_split_sha256": metadata.get("saved_split_sha256"),
            "split_metadata_sha256": metadata.get("split_metadata_sha256"),
            "manifest_sha256": metadata.get("manifest_sha256"),
        },
        "response_prediction_count": len(response_rows) if response_rows else None,
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
    return {"variant": experiment_id, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-tuned raw XGBoost on cached Qwen hidden vectors.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objective", required=True, choices=SUPPORTED_OBJECTIVES)
    parser.add_argument("--target-trials", type=int, default=50)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--inner-seed", type=int)
    parser.add_argument(
        "--sampling-mode",
        choices=SUPPORTED_SAMPLING_MODES,
        default=LEGACY_SAMPLING_MODE,
    )
    parser.add_argument("--oversampling-ratio", type=float)
    parser.add_argument("--oversampling-seed", type=int, default=1337)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--search-profile",
        choices=SUPPORTED_SEARCH_PROFILES,
        default="standard_d6",
    )
    parser.add_argument("--xgb-threads", type=int, default=20)
    parser.add_argument(
        "--protocol-profile",
        choices=("", "harmonized_optuna100_v1"),
        default="",
        help="Fixed 100-trial harmonized Optuna-100 protocol (production only).",
    )
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
        inner_seed=args.inner_seed,
        xgb_threads=args.xgb_threads,
        experiment_id=args.experiment_id,
        search_profile=args.search_profile,
        sampling_mode=args.sampling_mode,
        oversampling_ratio=args.oversampling_ratio,
        oversampling_seed=args.oversampling_seed,
        protocol_profile=args.protocol_profile or None,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
