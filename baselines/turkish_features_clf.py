#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_utils import assign_stratified_group_folds
from src.metrics import binary_auroc, classification_metrics


SUFFIXED_SCIENTIFIC_NUMBER = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+)\.\d+$"
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _parse_feature_value(value: str) -> tuple[float, bool]:
    try:
        return float(value), False
    except ValueError:
        match = SUFFIXED_SCIENTIFIC_NUMBER.match(value.strip())
        if match is None:
            raise
        return float(match.group(1)), True


def _load_rows(
    root: Path,
    metadata_csv: str,
    feature_source: str,
) -> tuple[list[dict[str, Any]], int]:
    csv.field_size_limit(max(csv.field_size_limit(), 10**9))
    rows: list[dict[str, Any]] = []
    repaired_feature_values = 0
    with (root / metadata_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["depresyon_skoru"])
            label = int(score >= 25.0)
            if label != int(float(row["label_t25"])):
                raise ValueError(f"Label mismatch for {row['file_name']}")
            if feature_source == "features":
                parsed_features = []
                for value in row["features"].split(","):
                    if not value.strip():
                        continue
                    parsed, repaired = _parse_feature_value(value)
                    parsed_features.append(parsed)
                    repaired_feature_values += int(repaired)
                features = np.asarray(parsed_features, dtype=np.float64)
            else:
                features = np.asarray([float(row["w2v2_predicted_score"])], dtype=np.float64)
            rows.append(
                {
                    "subject_id": row["patient_id"].strip(),
                    "label": label,
                    "features": features,
                }
            )
    dimensions = {int(row["features"].shape[0]) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent Turkish feature dimensions: {sorted(dimensions)}")
    return rows, repaired_feature_values


def _standardize(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_train = np.where(np.isfinite(train), train, np.nan)
    means = np.nanmean(finite_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train = np.where(np.isfinite(train), train, means)
    test = np.where(np.isfinite(test), test, means)
    std = train.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (train - means) / std, (test - means) / std


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    bias_first = 0.0
    bias_second = 0.0
    positive_count = max(1, int(labels.sum()))
    negative_count = max(1, int(labels.shape[0] - labels.sum()))
    sample_weights = np.where(
        labels == 1,
        labels.shape[0] / (2.0 * positive_count),
        labels.shape[0] / (2.0 * negative_count),
    )

    for step in range(1, steps + 1):
        probabilities = _sigmoid(features @ weights + bias)
        residual = (probabilities - labels) * sample_weights
        gradient = features.T @ residual / labels.shape[0] + l2 * weights
        bias_gradient = float(residual.mean())

        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
        bias_first = 0.9 * bias_first + 0.1 * bias_gradient
        bias_second = 0.999 * bias_second + 0.001 * (bias_gradient * bias_gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        corrected_bias_first = bias_first / (1.0 - 0.9**step)
        corrected_bias_second = bias_second / (1.0 - 0.999**step)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        bias -= learning_rate * corrected_bias_first / (corrected_bias_second**0.5 + 1e-8)
    return weights, bias


def run_cv(args: argparse.Namespace) -> dict[str, Any]:
    rows, repaired_feature_values = _load_rows(
        Path(args.root),
        args.metadata_csv,
        args.feature_source,
    )
    subject_labels: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            raise ValueError(f"Mixed labels for subject {subject_id}")
        subject_labels[subject_id] = label

    fold_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        folds = assign_stratified_group_folds(subject_labels, args.folds, seed)
        for fold_idx, payload in sorted(folds.items()):
            train_subjects = set(payload["outer_train_subject_ids"])
            test_subjects = set(payload["final_eval_subject_ids"])
            train_rows = [row for row in rows if row["subject_id"] in train_subjects]
            test_rows = [row for row in rows if row["subject_id"] in test_subjects]
            train_x = np.stack([row["features"] for row in train_rows])
            train_y = np.asarray([row["label"] for row in train_rows], dtype=np.float64)
            test_x = np.stack([row["features"] for row in test_rows])
            train_x, test_x = _standardize(train_x, test_x)
            weights, bias = _fit_logistic(
                train_x,
                train_y,
                steps=args.steps,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            sample_probabilities = _sigmoid(test_x @ weights + bias)
            probabilities_by_subject: dict[str, list[float]] = defaultdict(list)
            for row, probability in zip(test_rows, sample_probabilities):
                probabilities_by_subject[str(row["subject_id"])].append(float(probability))
            subject_ids = sorted(probabilities_by_subject)
            y_true = [subject_labels[subject_id] for subject_id in subject_ids]
            scores = [
                float(statistics.mean(probabilities_by_subject[subject_id]))
                for subject_id in subject_ids
            ]
            y_pred = [int(score >= 0.5) for score in scores]
            metrics = classification_metrics(y_true, y_pred)
            metrics["auroc"] = binary_auroc(y_true, scores)
            fold_rows.append({"seed": seed, "fold": fold_idx, **metrics})

    metric_keys = ["accuracy", "positive_f1", "macro_f1", "auroc"]
    summary = {}
    for key in metric_keys:
        values = [float(row[key]) for row in fold_rows]
        summary[key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        }
    return {
        "feature_source": args.feature_source,
        "folds": args.folds,
        "seeds": args.seeds,
        "num_subjects": len(subject_labels),
        "num_samples": len(rows),
        "repaired_suffixed_feature_values": repaired_feature_values,
        "summary": summary,
        "fold_rows": fold_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turkish acoustic-feature logistic baseline.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--metadata-csv",
        default="metadata_turkish_t25_binary_merged.csv",
    )
    parser.add_argument(
        "--feature-source",
        choices=("features", "w2v2_predicted_score"),
        default="features",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 7, 2024])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cv(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
