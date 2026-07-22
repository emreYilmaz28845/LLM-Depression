from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.aggregate import aggregate_binary_classifier_predictions
from src.utils import read_json, read_jsonl, save_json, write_jsonl


PRIMARY_VARIANTS = ("logreg_raw", "xgb_raw", "xgb_pca32", "logreg_pca32", "xgb_pca64")
CONTROL_VARIANTS = ("majority_class", "xgb_raw_shuffled_labels")


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


def _variant_pipeline(variant: str, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pca_components = 32 if "pca32" in variant else 64 if "pca64" in variant else None
    if variant.startswith("xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost variants require xgboost==2.1.4.") from exc
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.25,
            reg_alpha=1.0,
            reg_lambda=10.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        steps = []
        if pca_components:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
        steps.append(("classifier", estimator))
        return Pipeline(steps), pca_components
    if variant.startswith("logreg"):
        steps = []
        if pca_components:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
        steps.extend(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        return Pipeline(steps), pca_components
    raise ValueError(f"Unsupported fitted variant: {variant}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _metrics_with_negative_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    output = dict(metrics)
    output["negative_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return output


def shuffled_subject_labels(rows: list[dict[str, Any]], seed: int) -> np.ndarray:
    """Permute labels between subjects while preserving response groups."""
    labels_by_subject: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in labels_by_subject and labels_by_subject[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent training labels.")
        labels_by_subject[subject_id] = label
    subject_ids = sorted(labels_by_subject)
    shuffled = np.random.default_rng(seed).permutation(
        [labels_by_subject[subject_id] for subject_id in subject_ids]
    )
    shuffled_by_subject = dict(zip(subject_ids, shuffled.tolist()))
    return np.asarray(
        [shuffled_by_subject[str(row["subject_id"])] for row in rows],
        dtype=np.int64,
    )


def run_variant(
    cache_dir: Path,
    output_root: Path,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    train_x, train_rows = _load_partition(cache_dir, "outer_train")
    test_x, test_rows = _load_partition(cache_dir, "final_eval")
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    test_subjects = {str(row["subject_id"]) for row in test_rows}
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Training/held-out subject leakage: {overlap[:10]}")
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    test_y = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    if set(train_y.tolist()) != {0, 1}:
        raise ValueError("Training cache must contain both classes.")
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    requested_components = None
    effective_components = None
    fit_y = train_y.copy()
    if variant == "majority_class":
        prediction = int(np.bincount(train_y, minlength=2).argmax())
        probability = float(train_y.mean())
        probabilities = np.full(test_y.shape, probability, dtype=np.float64)
        predictions = np.full(test_y.shape, prediction, dtype=np.int64)
    else:
        fitted_variant = "xgb_raw" if variant == "xgb_raw_shuffled_labels" else variant
        fitted, requested_components = _variant_pipeline(fitted_variant, seed)
        if requested_components:
            matrix_limit = min(train_x.shape)
            if requested_components > matrix_limit:
                raise ValueError(
                    f"PCA-{requested_components} exceeds training matrix limit {matrix_limit}."
                )
            effective_components = requested_components
        if variant == "xgb_raw_shuffled_labels":
            fit_y = shuffled_subject_labels(train_rows, seed)
        fitted.fit(train_x, fit_y)
        probabilities = np.asarray(fitted.predict_proba(test_x)[:, 1], dtype=np.float64)
        predictions = (probabilities >= 0.5).astype(np.int64)
        import joblib

        joblib.dump(fitted, variant_dir / "pipeline.joblib")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    sample_rows = []
    for row, probability, prediction in zip(test_rows, probabilities.tolist(), predictions.tolist()):
        sample_rows.append(
            {
                "dataset": metadata["dataset"],
                "modality": metadata["input_modality"],
                "fold": int(metadata["fold"]),
                "sample_id": str(row["sample_id"]),
                "subject_id": str(row["subject_id"]),
                "label": int(row["label"]),
                "probability": float(probability),
                "predicted_class": int(prediction),
                "checkpoint": metadata["checkpoint_dir"],
                "classifier_variant": variant,
            }
        )
    subject_rows, metrics = aggregate_binary_classifier_predictions(sample_rows)
    for row in subject_rows:
        row.update(
            {
                "dataset": metadata["dataset"],
                "modality": metadata["input_modality"],
                "fold": int(metadata["fold"]),
                "classifier_variant": variant,
            }
        )
    metrics = _metrics_with_negative_f1(metrics)
    artifact_metadata = {
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "fold": int(metadata["fold"]),
        "classifier_variant": variant,
        "seed": seed,
        "threshold": 0.5,
        "input_dimension": int(train_x.shape[1]),
        "requested_pca_components": requested_components,
        "effective_pca_components": effective_components,
        "training_row_ids": [str(row["sample_id"]) for row in train_rows],
        "training_subject_ids": sorted(train_subjects),
        "heldout_subject_ids": sorted(test_subjects),
        "shuffled_training_labels": variant == "xgb_raw_shuffled_labels",
        "label_shuffle_unit": "subject" if variant == "xgb_raw_shuffled_labels" else None,
        "extraction_metadata": str(cache_dir / "extraction_metadata.json"),
    }
    write_jsonl(sample_rows, variant_dir / "predictions_sample_level.jsonl")
    write_jsonl(subject_rows, variant_dir / "predictions_subject_level.jsonl")
    _write_csv(sample_rows, variant_dir / "predictions_sample_level.csv")
    _write_csv(subject_rows, variant_dir / "predictions_subject_level.csv")
    save_json(metrics, variant_dir / "metrics.json")
    save_json(artifact_metadata, variant_dir / "classifier_metadata.json")
    return {"variant": variant, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifiers on cached Qwen hidden vectors.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS + CONTROL_VARIANTS))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_variant(args.cache_dir, args.output_dir, variant, args.seed) for variant in args.variants]
    save_json(summaries, args.output_dir / "variant_summary.json")
    _write_csv(summaries, args.output_dir / "variant_summary.csv")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
