#!/usr/bin/env python3
"""Paired subject-level comparison for E0 modality perturbation conditions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

SCORE_MODE_LEGACY = "legacy_alias"
SCORE_MODE_FIRST_TOKEN = "first_token"
SCORE_MODE_CANDIDATE = "candidate_likelihood"
SCORE_MODE_ALIASES = {
    "legacy": SCORE_MODE_LEGACY,
    "legacy_alias": SCORE_MODE_LEGACY,
    "first-token": SCORE_MODE_FIRST_TOKEN,
    "first_token": SCORE_MODE_FIRST_TOKEN,
    "candidate": SCORE_MODE_CANDIDATE,
    "candidate-likelihood": SCORE_MODE_CANDIDATE,
    "candidate_likelihood": SCORE_MODE_CANDIDATE,
}


def _normalize_score_mode(score_mode: str) -> str:
    normalized = str(score_mode).strip().lower()
    try:
        return SCORE_MODE_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(
            (SCORE_MODE_FIRST_TOKEN, SCORE_MODE_CANDIDATE, SCORE_MODE_LEGACY)
        )
        raise ValueError(
            f"Unsupported score mode {score_mode!r}; choose one of: {choices}"
        ) from error


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subject_id = str(row["subject_id"])
            if subject_id in rows:
                raise ValueError(f"Duplicate subject_id={subject_id!r} in {path}")
            rows[subject_id] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _condition_arrays(
    rows: dict[str, dict[str, Any]],
    subject_ids: list[str],
    *,
    score_mode: str = SCORE_MODE_LEGACY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_mode = _normalize_score_mode(score_mode)
    labels = np.asarray([int(rows[s]["label"]) for s in subject_ids], dtype=np.int64)
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("Condition labels must be binary integers 0 or 1")

    if score_mode == SCORE_MODE_FIRST_TOKEN:
        margin_field = "first_token_margin"
        prediction_field = "first_token_prediction"
        missing = [
            subject_id
            for subject_id in subject_ids
            if margin_field not in rows[subject_id]
            or prediction_field not in rows[subject_id]
        ]
        if missing:
            raise ValueError(
                "first_token mode requires canonical first_token_margin and "
                f"first_token_prediction fields; missing for subjects: {missing[:10]}"
            )
        scores = np.asarray(
            [float(rows[s][margin_field]) for s in subject_ids], dtype=np.float64
        )
        stored = np.asarray(
            [int(rows[s][prediction_field]) for s in subject_ids], dtype=np.int64
        )
    elif score_mode == SCORE_MODE_CANDIDATE:
        margin_field = "candidate_likelihood_margin"
        prediction_field = "candidate_likelihood_prediction"
        canonical = [
            margin_field in rows[subject_id] and prediction_field in rows[subject_id]
            for subject_id in subject_ids
        ]
        if all(canonical):
            scores = np.asarray(
                [float(rows[s][margin_field]) for s in subject_ids], dtype=np.float64
            )
            stored = np.asarray(
                [int(rows[s][prediction_field]) for s in subject_ids], dtype=np.int64
            )
        elif not any(canonical) and all(
            "dep_score" in rows[s]
            and "non_score" in rows[s]
            and "likelihood_prediction" in rows[s]
            and "first_token_margin" not in rows[s]
            for s in subject_ids
        ):
            # Pre-E0 standalone evaluation rows used these names for candidate
            # label likelihood.  Canonical E0 rows also expose the names as
            # first-token compatibility aliases, so fallback is deliberately
            # limited to rows with no canonical first-token fields.
            scores = np.asarray(
                [
                    float(rows[s]["dep_score"]) - float(rows[s]["non_score"])
                    for s in subject_ids
                ],
                dtype=np.float64,
            )
            stored = np.asarray(
                [int(rows[s]["likelihood_prediction"]) for s in subject_ids],
                dtype=np.int64,
            )
        else:
            missing = [
                subject_id
                for subject_id, present in zip(subject_ids, canonical)
                if not present
            ]
            raise ValueError(
                "candidate_likelihood mode requires canonical candidate_likelihood_margin and "
                "candidate_likelihood_prediction fields. Legacy dep_score/non_score fallback is "
                "accepted only when canonical first-token fields are absent; invalid subjects: "
                f"{missing[:10]}"
            )
    else:
        required = ("dep_score", "non_score", "likelihood_prediction")
        missing = [
            subject_id
            for subject_id in subject_ids
            if any(field not in rows[subject_id] for field in required)
        ]
        if missing:
            raise ValueError(
                f"legacy_alias mode requires {required}; missing for subjects: {missing[:10]}"
            )
        scores = np.asarray(
            [
                float(rows[s]["dep_score"]) - float(rows[s]["non_score"])
                for s in subject_ids
            ],
            dtype=np.float64,
        )
        stored = np.asarray(
            [int(rows[s]["likelihood_prediction"]) for s in subject_ids],
            dtype=np.int64,
        )

    if not np.all(np.isfinite(scores)):
        invalid = [
            subject_id
            for subject_id, score in zip(subject_ids, scores)
            if not np.isfinite(score)
        ]
        raise ValueError(f"Non-finite scores for subjects: {invalid[:10]}")
    predictions = (scores > 0.0).astype(np.int64)
    if not np.array_equal(predictions, stored):
        mismatches = [s for s, a, b in zip(subject_ids, predictions, stored) if a != b]
        raise ValueError(
            f"Score-sign/stored-prediction mismatch in {score_mode} mode for subjects: "
            f"{mismatches[:10]}"
        )
    return labels, scores, predictions


def _safe_auroc(labels: np.ndarray, _predictions: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_auprc(labels: np.ndarray, _predictions: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


METRICS: dict[str, Metric] = {
    "accuracy": lambda y, p, _s: float(accuracy_score(y, p)),
    "balanced_accuracy": lambda y, p, _s: float(balanced_accuracy_score(y, p)),
    "positive_f1": lambda y, p, _s: float(f1_score(y, p, pos_label=1, zero_division=0)),
    "macro_f1": lambda y, p, _s: float(f1_score(y, p, average="macro", zero_division=0)),
    "auroc": _safe_auroc,
    "auprc": _safe_auprc,
}


def _metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {name: metric(labels, predictions, scores) for name, metric in METRICS.items()}


def _percentile_interval(values: Iterable[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.float64,
    )
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "estimate": float("nan"),
            "ci_95_low": float("nan"),
            "ci_95_high": float("nan"),
            "valid_replicates": 0,
        }
    return {
        "estimate": float(np.mean(finite)),
        "ci_95_low": float(np.percentile(finite, 2.5)),
        "ci_95_high": float(np.percentile(finite, 97.5)),
        "valid_replicates": int(finite.size),
    }


def _classification_metrics_batched(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    """Match the sklearn classification metrics for bootstrap rows."""
    positive = labels == 1
    predicted_positive = predictions == 1
    tp = np.sum(positive & predicted_positive, axis=1, dtype=np.int64)
    fp = np.sum(~positive & predicted_positive, axis=1, dtype=np.int64)
    fn = np.sum(positive & ~predicted_positive, axis=1, dtype=np.int64)
    tn = labels.shape[1] - tp - fp - fn

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
        return result

    positive_f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    positive_in_union = (tp + fp + fn) > 0
    negative_in_union = (tn + fp + fn) > 0
    macro_f1 = divide(
        positive_f1 + negative_f1,
        positive_in_union.astype(np.int64) + negative_in_union.astype(np.int64),
    )

    positive_support = tp + fn
    negative_support = tn + fp
    positive_recall = divide(tp, positive_support)
    negative_recall = divide(tn, negative_support)
    balanced_accuracy = divide(
        positive_recall + negative_recall,
        (positive_support > 0).astype(np.int64) + (negative_support > 0).astype(np.int64),
    )

    return {
        "accuracy": (tp + tn).astype(np.float64) / labels.shape[1],
        "balanced_accuracy": balanced_accuracy,
        "positive_f1": positive_f1,
        "macro_f1": macro_f1,
    }


def _rank_metrics_batched(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary AUROC/AP exactly, including average handling of score ties."""
    positive = labels == 1
    negative = ~positive
    positive_count = np.sum(positive, axis=1, dtype=np.int64)
    negative_count = labels.shape[1] - positive_count
    both_classes = (positive_count > 0) & (negative_count > 0)

    # [replicate, candidate, comparator].  Bootstrap batches keep this bounded
    # (47 subjects and 512 replicates are only about 1.1M pairwise entries).
    candidate_scores = scores[:, :, None]
    comparator_scores = scores[:, None, :]
    positive_negative_pairs = positive[:, :, None] & negative[:, None, :]
    wins = np.count_nonzero(
        (candidate_scores > comparator_scores) & positive_negative_pairs, axis=(1, 2)
    )
    ties = np.count_nonzero(
        (candidate_scores == comparator_scores) & positive_negative_pairs, axis=(1, 2)
    )
    auroc = np.full(labels.shape[0], np.nan, dtype=np.float64)
    denominator = positive_count * negative_count
    auroc[both_classes] = (
        wins[both_classes] + 0.5 * ties[both_classes]
    ) / denominator[both_classes]

    # average_precision_score integrates the non-interpolated PR step curve.
    # Equivalently, each positive contributes the precision of the complete
    # (tie-inclusive) score threshold at which it is retrieved.
    at_or_above = comparator_scores >= candidate_scores
    retrieved = np.count_nonzero(at_or_above, axis=2)
    retrieved_positive = np.count_nonzero(
        at_or_above & positive[:, None, :], axis=2
    )
    threshold_precision = retrieved_positive / retrieved
    auprc = np.full(labels.shape[0], np.nan, dtype=np.float64)
    auprc[both_classes] = (
        np.sum(threshold_precision * positive, axis=1)[both_classes]
        / positive_count[both_classes]
    )
    return auroc, auprc


def _all_metrics_batched(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    metrics = _classification_metrics_batched(labels, predictions)
    auroc, auprc = _rank_metrics_batched(labels, scores)
    metrics["auroc"] = auroc
    metrics["auprc"] = auprc
    return metrics


def _paired_bootstrap_fast(
    labels: np.ndarray,
    real_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    real_predictions: np.ndarray,
    perturbed_predictions: np.ndarray,
    signed_margin_delta: np.ndarray,
    *,
    bootstrap_reps: int,
    seed: int,
    batch_size: int = 512,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run paired subject bootstrap with sklearn-equivalent vectorized metrics."""
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_subjects = labels.size
    if n_subjects < 1:
        raise ValueError("At least one subject is required")

    metric_deltas = {
        name: np.empty(bootstrap_reps, dtype=np.float64) for name in METRICS
    }
    margin_means = np.empty(bootstrap_reps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, bootstrap_reps, batch_size):
        stop = min(start + batch_size, bootstrap_reps)
        indices = rng.integers(0, n_subjects, size=(stop - start, n_subjects))
        boot_labels = labels[indices]
        real_metrics = _all_metrics_batched(
            boot_labels, real_scores[indices], real_predictions[indices]
        )
        perturbed_metrics = _all_metrics_batched(
            boot_labels, perturbed_scores[indices], perturbed_predictions[indices]
        )
        for name in METRICS:
            metric_deltas[name][start:stop] = real_metrics[name] - perturbed_metrics[name]
        margin_means[start:stop] = np.mean(signed_margin_delta[indices], axis=1)
    return metric_deltas, margin_means


def compare(
    real_path: Path,
    perturbed_path: Path,
    output_dir: Path,
    *,
    perturbed_name: str,
    bootstrap_reps: int,
    seed: int,
    score_mode: str = SCORE_MODE_LEGACY,
) -> dict[str, Any]:
    score_mode = _normalize_score_mode(score_mode)
    real_rows = _read_rows(real_path)
    perturbed_rows = _read_rows(perturbed_path)
    if set(real_rows) != set(perturbed_rows):
        raise ValueError("Condition files do not contain the same subject IDs")

    subject_ids = sorted(real_rows)
    labels, real_scores, real_predictions = _condition_arrays(
        real_rows, subject_ids, score_mode=score_mode
    )
    other_labels, perturbed_scores, perturbed_predictions = _condition_arrays(
        perturbed_rows, subject_ids, score_mode=score_mode
    )
    if not np.array_equal(labels, other_labels):
        raise ValueError("Condition labels differ")

    real_metrics = _metrics(labels, real_scores, real_predictions)
    perturbed_metrics = _metrics(labels, perturbed_scores, perturbed_predictions)
    point_metric_deltas = {
        name: real_metrics[name] - perturbed_metrics[name] for name in METRICS
    }

    label_sign = labels * 2 - 1
    real_correct_class_margins = label_sign * real_scores
    perturbed_correct_class_margins = label_sign * perturbed_scores
    signed_margin_delta = real_correct_class_margins - perturbed_correct_class_margins

    n_subjects = len(subject_ids)
    metric_delta_replicates, signed_margin_replicates = _paired_bootstrap_fast(
        labels,
        real_scores,
        perturbed_scores,
        real_predictions,
        perturbed_predictions,
        signed_margin_delta,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )

    metric_delta_intervals = {
        name: {
            "point_estimate": float(point_metric_deltas[name]),
            **{
                key: value
                for key, value in _percentile_interval(values).items()
                if key != "estimate"
            },
        }
        for name, values in metric_delta_replicates.items()
    }
    signed_margin_interval = _percentile_interval(signed_margin_replicates)
    signed_margin_interval["point_estimate"] = float(np.mean(signed_margin_delta))
    signed_margin_interval.pop("estimate", None)

    real_correct = real_predictions == labels
    perturbed_correct = perturbed_predictions == labels
    all_negative = np.zeros_like(labels)
    all_positive = np.ones_like(labels)
    report: dict[str, Any] = {
        "schema_version": 2,
        "comparison": f"real_minus_{perturbed_name}",
        "score_mode": score_mode,
        "real_predictions_path": str(real_path),
        "perturbed_predictions_path": str(perturbed_path),
        "n_subjects": n_subjects,
        "support_negative": int(np.sum(labels == 0)),
        "support_positive": int(np.sum(labels == 1)),
        "bootstrap": {
            "repetitions": bootstrap_reps,
            "seed": seed,
            "method": "paired subject resampling with replacement",
        },
        "conditions": {
            "real": real_metrics,
            perturbed_name: perturbed_metrics,
            "all_negative_baseline": _metrics(labels, np.zeros_like(real_scores), all_negative),
            "all_positive_baseline": _metrics(labels, np.ones_like(real_scores), all_positive),
        },
        "paired": {
            "prediction_disagreements": int(np.sum(real_predictions != perturbed_predictions)),
            "real_correct_perturbed_wrong": int(np.sum(real_correct & ~perturbed_correct)),
            "real_wrong_perturbed_correct": int(np.sum(~real_correct & perturbed_correct)),
            "mean_absolute_raw_margin_change": float(
                np.mean(np.abs(real_scores - perturbed_scores))
            ),
            "correct_class_margin_delta": signed_margin_interval,
            "metric_differences": metric_delta_intervals,
        },
        "interpretation_guardrail": (
            "This comparison alone can establish sensitivity to real versus perturbed audio, "
            "but the final E0 audio-use classification also requires across-subject and "
            "same-class shuffle controls."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"paired_real_vs_{perturbed_name}"
    if score_mode != SCORE_MODE_LEGACY:
        # Keep historical default artifact names unchanged while preventing
        # explicit primary and secondary comparisons from overwriting one another.
        artifact_stem = f"{artifact_stem}_{score_mode}"
    (output_dir / f"{artifact_stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / f"{artifact_stem}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "subject_id",
            "label",
            "real_score",
            f"{perturbed_name}_score",
            "real_prediction",
            f"{perturbed_name}_prediction",
            "real_correct",
            f"{perturbed_name}_correct",
            "correct_class_margin_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, subject_id in enumerate(subject_ids):
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "label": int(labels[index]),
                    "real_score": float(real_scores[index]),
                    f"{perturbed_name}_score": float(perturbed_scores[index]),
                    "real_prediction": int(real_predictions[index]),
                    f"{perturbed_name}_prediction": int(perturbed_predictions[index]),
                    "real_correct": bool(real_correct[index]),
                    f"{perturbed_name}_correct": bool(perturbed_correct[index]),
                    "correct_class_margin_delta": float(signed_margin_delta[index]),
                }
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--perturbed", type=Path, required=True)
    parser.add_argument("--perturbed-name", default="silence")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--score-mode",
        default=SCORE_MODE_LEGACY,
        help=(
            "Score schema: first_token (canonical primary), candidate_likelihood "
            "(canonical secondary), or legacy_alias (dep_score/non_score compatibility; default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    report = compare(
        args.real,
        args.perturbed,
        args.output_dir,
        perturbed_name=str(args.perturbed_name),
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        score_mode=str(args.score_mode),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
