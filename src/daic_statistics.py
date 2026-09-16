from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from src.metrics import classification_metrics


def _rows_by_key(
    rows: Sequence[dict[str, Any]], *, context: str
) -> dict[tuple[str, int], tuple[int, int]]:
    """One ``(gold, strict prediction)`` pair per ``(subject_id, seed)`` key.

    An INVALID output (not 0/1) counts as wrong for its true class, matching the
    strict convention used everywhere else in the repository.
    """
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for row in rows:
        key = (str(row["subject_id"]), int(row.get("seed", 0)))
        if key in out:
            raise ValueError(f"{context} must contain one row per subject/seed key.")
        gold = int(row["label"])
        if gold not in (0, 1):
            raise ValueError(f"{context} requires binary gold labels, got {gold!r}.")
        prediction = int(row["prediction"])
        if prediction not in (0, 1):
            prediction = 1 - gold
        out[key] = (gold, prediction)
    return out


def _validate_paired_rows(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *, context: str
) -> tuple[dict[tuple[str, int], tuple[int, int]], dict[tuple[str, int], tuple[int, int]]]:
    left = _rows_by_key(baseline, context=f"{context} baseline")
    right = _rows_by_key(comparison, context=f"{context} comparison")
    if set(left) != set(right):
        raise ValueError(f"{context} requires identical subject/seed keys.")
    for key in left:
        if left[key][0] != right[key][0]:
            raise ValueError(f"{context} label mismatch for subject {key[0]}.")
    labels_by_subject: dict[str, set[int]] = defaultdict(set)
    seeds_by_subject: dict[str, set[int]] = defaultdict(set)
    for (subject, run_seed), (gold, _) in left.items():
        labels_by_subject[subject].add(gold)
        seeds_by_subject[subject].add(run_seed)
    if any(len(labels) != 1 for labels in labels_by_subject.values()):
        raise ValueError(f"{context} requires one label per subject across seeds.")
    expected_seeds = next(iter(seeds_by_subject.values()), set())
    if any(seeds != expected_seeds for seeds in seeds_by_subject.values()):
        raise ValueError(f"{context} requires the same seed set for every subject.")
    return left, right


def stratified_paired_bootstrap(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metric: str = "macro_f1", iterations: int = 10000, seed: int = 1337,
) -> dict[str, float]:
    if int(iterations) < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    left, right = _validate_paired_rows(baseline, comparison, context="Paired bootstrap")
    seeds = sorted({key[1] for key in left})
    by_label: dict[int, list[str]] = defaultdict(list)
    for subject in sorted({key[0] for key in left}):
        by_label[left[(subject, seeds[0])][0]].append(subject)
    if not by_label[0] or not by_label[1]:
        raise ValueError("Stratified bootstrap requires both labels.")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled_subjects: list[str] = []
        for label in (0, 1):
            subjects = by_label[label]
            sampled_subjects.extend(rng.choice(subjects) for _ in subjects)
        per_seed = []
        for run_seed in seeds:
            keys = [(subject, run_seed) for subject in sampled_subjects]
            y = [left[key][0] for key in keys]
            left_metric = classification_metrics(y, [left[key][1] for key in keys])
            right_metric = classification_metrics(y, [right[key][1] for key in keys])
            if metric not in left_metric or metric not in right_metric:
                raise ValueError(f"Unsupported bootstrap metric: {metric!r}")
            lm = left_metric[metric]
            rm = right_metric[metric]
            per_seed.append(rm - lm)
        deltas.append(sum(per_seed) / len(per_seed))
    deltas.sort()
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "ci_low": deltas[int(0.025 * (len(deltas) - 1))],
        "ci_high": deltas[int(0.975 * (len(deltas) - 1))],
        "iterations": iterations, "seed": seed,
    }


def stratified_paired_bootstrap_many(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metrics: Sequence[str] = ("macro_f1", "positive_f1", "macro_recall"),
    iterations: int = 100_000, seed: int = 1337, chunk_size: int = 2_000,
) -> dict[str, dict[str, float]]:
    """Vectorized label-stratified bootstrap clustered by subject."""
    if int(iterations) < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    left, right = _validate_paired_rows(baseline, comparison, context="Paired bootstrap")
    subjects = sorted({key[0] for key in left})
    seeds = sorted({key[1] for key in left})
    gold = np.asarray([left[(subject, seeds[0])][0] for subject in subjects], dtype=np.int8)
    groups = [np.flatnonzero(gold == label) for label in (0, 1)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Stratified bootstrap requires both labels.")
    left_predictions = np.asarray(
        [[left[(subject, run_seed)][1] for subject in subjects] for run_seed in seeds], dtype=np.int8
    )
    right_predictions = np.asarray(
        [[right[(subject, run_seed)][1] for subject in subjects] for run_seed in seeds], dtype=np.int8
    )
    requested = tuple(metrics)
    unknown = [metric for metric in requested if metric not in classification_metrics([0, 1], [0, 1])]
    if unknown:
        raise ValueError(f"Unsupported bootstrap metric(s): {unknown!r}")

    def metrics_for(predictions: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
        tp = np.sum((truth == 1) & (predictions == 1), axis=2, dtype=np.int32)
        fn = np.sum((truth == 1) & (predictions == 0), axis=2, dtype=np.int32)
        tn = np.sum((truth == 0) & (predictions == 0), axis=2, dtype=np.int32)
        fp = np.sum((truth == 0) & (predictions == 1), axis=2, dtype=np.int32)
        divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        p_pos, r_pos = divide(tp, tp + fp), divide(tp, tp + fn)
        p_neg, r_neg = divide(tn, tn + fn), divide(tn, tn + fp)
        f_pos = divide(2 * p_pos * r_pos, p_pos + r_pos)
        f_neg = divide(2 * p_neg * r_neg, p_neg + r_neg)
        return {
            "positive_f1": np.mean(f_pos, axis=1),
            "macro_f1": np.mean((f_pos + f_neg) / 2.0, axis=1),
            "macro_recall": np.mean((r_pos + r_neg) / 2.0, axis=1),
        }

    rng = np.random.default_rng(seed)
    deltas = {metric: np.empty(int(iterations), dtype=float) for metric in requested}
    completed = 0
    while completed < int(iterations):
        batch = min(int(chunk_size), int(iterations) - completed)
        sampled = np.concatenate([
            rng.choice(group, size=(batch, len(group)), replace=True) for group in groups
        ], axis=1)
        truth = gold[sampled][:, None, :]
        left_values = metrics_for(left_predictions[:, sampled].transpose(1, 0, 2), truth)
        right_values = metrics_for(right_predictions[:, sampled].transpose(1, 0, 2), truth)
        for metric in requested:
            deltas[metric][completed:completed + batch] = right_values[metric] - left_values[metric]
        completed += batch
    return {
        metric: {
            "mean_delta": float(np.mean(deltas[metric])),
            "ci_low": float(np.quantile(deltas[metric], 0.025)),
            "ci_high": float(np.quantile(deltas[metric], 0.975)),
            "iterations": int(iterations),
            "seed": seed,
            "method": "subject_clustered_label_stratified_percentile",
        }
        for metric in requested
    }


def exact_mcnemar(baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]]) -> dict[str, Any]:
    left_keys = [str(row["subject_id"]) for row in baseline]
    right_keys = [str(row["subject_id"]) for row in comparison]
    if len(left_keys) != len(set(left_keys)) or len(right_keys) != len(set(right_keys)):
        raise ValueError("McNemar inputs must contain one row per subject.")
    left = dict(zip(left_keys, baseline))
    right = dict(zip(right_keys, comparison))
    if set(left) != set(right):
        raise ValueError("McNemar requires identical subject keys.")
    b = c = baseline_correct = comparison_correct = 0
    for subject_id, row in left.items():
        gold = int(row["label"])
        if int(right[subject_id]["label"]) != gold:
            raise ValueError(f"McNemar label mismatch for subject {subject_id}.")
        left_correct = int(row["prediction"]) == gold
        right_correct = int(right[subject_id]["prediction"]) == gold
        baseline_correct += int(left_correct)
        comparison_correct += int(right_correct)
        b += int(left_correct and not right_correct)
        c += int(not left_correct and right_correct)
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n) if n else 1.0
    total = len(left)
    return {
        "baseline_only_correct": b,
        "comparison_only_correct": c,
        "discordant_subjects": n,
        "subjects": total,
        "baseline_accuracy": baseline_correct / total if total else 0.0,
        "comparison_accuracy": comparison_correct / total if total else 0.0,
        "accuracy_delta": (comparison_correct - baseline_correct) / total if total else 0.0,
        "net_correct_gain": c - b,
        "p_value": min(1.0, 2.0 * tail),
    }


def paired_prediction_swap_permutation(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metric: str = "macro_f1", iterations: int = 10000, seed: int = 1337,
) -> dict[str, Any]:
    """Subject-level paired permutation test by swapping the two predictions.

    Each iteration swaps the two sides' predictions for every ``(subject_id,
    seed)`` key independently with probability 0.5, recomputes both pooled
    metrics on the swapped assignment, and keeps their difference. The two-sided
    p-value is ``(1 + #{|delta*| >= |delta_observed|}) / (1 + iterations)``.

    No class stratification is needed: a subject keeps its gold label, so only
    the two methods trade predictions and the class balance is preserved.
    Multi-seed inputs average the per-seed differences, matching
    ``stratified_paired_bootstrap``. Invalid outputs count as wrong for their
    true class before scoring.
    """
    if int(iterations) < 1:
        raise ValueError("Permutation iterations must be positive.")
    left, right = _validate_paired_rows(baseline, comparison, context="Permutation")
    keys = sorted(left)
    if metric not in classification_metrics([left[key][0] for key in keys], [left[key][1] for key in keys]):
        raise ValueError(f"Unsupported permutation metric: {metric!r}")

    by_seed: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        by_seed[key[1]].append(key)

    def delta(assignment: dict[tuple[str, int], int], opposite: dict[tuple[str, int], int]) -> float:
        per_seed: list[float] = []
        for run_seed in sorted(by_seed):
            group = by_seed[run_seed]
            gold = [left[key][0] for key in group]
            right_metric = classification_metrics(gold, [assignment[key] for key in group])[metric]
            left_metric = classification_metrics(gold, [opposite[key] for key in group])[metric]
            per_seed.append(right_metric - left_metric)
        return sum(per_seed) / len(per_seed)

    observed = delta({key: right[key][1] for key in keys}, {key: left[key][1] for key in keys})
    rng = random.Random(seed)
    null_values: list[float] = []
    hits = 0
    subjects = sorted({key[0] for key in keys})
    for _ in range(int(iterations)):
        # Subject is the inferential unit. All seeds for one subject move
        # together, preventing repeated seeds from becoming pseudo-replicates.
        subject_swap = {subject: rng.random() < 0.5 for subject in subjects}
        assignment = {key: (right[key][1] if subject_swap[key[0]] else left[key][1]) for key in keys}
        opposite = {key: (left[key][1] if subject_swap[key[0]] else right[key][1]) for key in keys}
        value = delta(assignment, opposite)
        null_values.append(value)
        if abs(value) >= abs(observed) - 1e-12:
            hits += 1
    return {
        "metric": metric,
        "observed_delta": observed,
        "null_mean": sum(null_values) / len(null_values),
        "p_value": (1 + hits) / (1 + int(iterations)),
        "iterations": int(iterations),
        "seed": seed,
        "subjects": len({key[0] for key in keys}),
        "keys": len(keys),
        "method": "monte_carlo_subject_clustered",
        "exceedances": hits,
    }


def paired_prediction_swap_permutation_many(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metrics: Sequence[str] = ("macro_f1", "positive_f1", "macro_recall"),
    iterations: int = 1_000_000, seed: int = 1337, chunk_size: int = 10_000,
) -> dict[str, dict[str, Any]]:
    """Vectorized subject-clustered Monte Carlo swaps for several metrics.

    One random swap mask is shared across seeds and metrics. This both preserves
    the subject as the inferential unit and avoids three independent expensive
    permutation streams for the co-primary metrics.
    """
    if int(iterations) < 1:
        raise ValueError("Permutation iterations must be positive.")
    left, right = _validate_paired_rows(baseline, comparison, context="Permutation")
    subjects = sorted({key[0] for key in left})
    seeds = sorted({key[1] for key in left})
    requested = tuple(metrics)
    supported = classification_metrics([0, 1], [0, 1])
    unknown = [metric for metric in requested if metric not in supported]
    if unknown:
        raise ValueError(f"Unsupported permutation metric(s): {unknown!r}")

    gold = np.asarray([left[(subject, seeds[0])][0] for subject in subjects], dtype=np.int8)
    left_by_seed = np.asarray([[left[(subject, run_seed)][1] for subject in subjects] for run_seed in seeds], dtype=np.int8)
    right_by_seed = np.asarray([[right[(subject, run_seed)][1] for subject in subjects] for run_seed in seeds], dtype=np.int8)

    def vector_metrics(predictions: np.ndarray) -> dict[str, np.ndarray]:
        # predictions shape: iterations x seeds x subjects
        truth = gold[None, None, :]
        tp = np.sum((truth == 1) & (predictions == 1), axis=2, dtype=np.int32)
        fn = np.sum((truth == 1) & (predictions == 0), axis=2, dtype=np.int32)
        tn = np.sum((truth == 0) & (predictions == 0), axis=2, dtype=np.int32)
        fp = np.sum((truth == 0) & (predictions == 1), axis=2, dtype=np.int32)
        divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        p_pos, r_pos = divide(tp, tp + fp), divide(tp, tp + fn)
        p_neg, r_neg = divide(tn, tn + fn), divide(tn, tn + fp)
        f_pos = divide(2 * p_pos * r_pos, p_pos + r_pos)
        f_neg = divide(2 * p_neg * r_neg, p_neg + r_neg)
        return {
            "positive_f1": np.mean(f_pos, axis=1),
            "macro_f1": np.mean((f_pos + f_neg) / 2.0, axis=1),
            "macro_recall": np.mean((r_pos + r_neg) / 2.0, axis=1),
        }

    observed_right = vector_metrics(right_by_seed[None, :, :])
    observed_left = vector_metrics(left_by_seed[None, :, :])
    observed = {metric: float(observed_right[metric][0] - observed_left[metric][0]) for metric in requested}
    hits = {metric: 0 for metric in requested}
    sums = {metric: 0.0 for metric in requested}
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < int(iterations):
        batch = min(int(chunk_size), int(iterations) - completed)
        swap = rng.integers(0, 2, size=(batch, 1, len(subjects)), dtype=np.int8).astype(bool)
        right_assignment = np.where(swap, right_by_seed[None, :, :], left_by_seed[None, :, :])
        left_assignment = np.where(swap, left_by_seed[None, :, :], right_by_seed[None, :, :])
        right_metrics = vector_metrics(right_assignment)
        left_metrics = vector_metrics(left_assignment)
        for metric in requested:
            values = right_metrics[metric] - left_metrics[metric]
            hits[metric] += int(np.sum(np.abs(values) >= abs(observed[metric]) - 1e-12))
            sums[metric] += float(np.sum(values))
        completed += batch

    def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
        z = 1.959963984540054
        proportion = successes / trials
        denominator = 1.0 + z * z / trials
        center = (proportion + z * z / (2 * trials)) / denominator
        radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    return {
        metric: {
            "metric": metric,
            "observed_delta": observed[metric],
            "null_mean": sums[metric] / int(iterations),
            "p_value": (1 + hits[metric]) / (1 + int(iterations)),
            "method": "monte_carlo_subject_clustered",
            "iterations": int(iterations),
            "seed": seed,
            "exceedances": hits[metric],
            "monte_carlo_probability_ci95": list(wilson_interval(hits[metric], int(iterations))),
            "subjects": len(subjects),
            "keys": len(left),
        }
        for metric in requested
    }


def exact_paired_prediction_swap(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metrics: Sequence[str] = ("macro_f1", "positive_f1", "macro_recall"),
) -> dict[str, dict[str, Any]]:
    """Exact paired prediction-swap tests for a binary, single-seed result.

    Concordant predictions never affect a swap. Within each gold-label group,
    every discordant pair contains one 0 and one 1, so the exact null can be
    summed over two binomial counts instead of enumerating ``2**n`` bit masks.
    """
    left, right = _validate_paired_rows(baseline, comparison, context="Exact permutation")
    seeds = {key[1] for key in left}
    if len(seeds) != 1:
        raise ValueError("Exact permutation currently requires a single seed.")
    requested = tuple(metrics)
    supported = classification_metrics([0, 1], [0, 1])
    unknown = [metric for metric in requested if metric not in supported]
    if unknown:
        raise ValueError(f"Unsupported permutation metric(s): {unknown!r}")

    fixed_prediction_counts = {0: [0, 0], 1: [0, 0]}
    discordant = {0: 0, 1: 0}
    for key, (gold, left_prediction) in left.items():
        right_prediction = right[key][1]
        if left_prediction == right_prediction:
            fixed_prediction_counts[gold][left_prediction] += 1
        else:
            discordant[gold] += 1

    def metric_values(k0: int, k1: int) -> tuple[dict[str, Any], dict[str, Any]]:
        # k_g is the number of discordant gold-g subjects assigned prediction 1
        # on the comparison side; the baseline side receives the complement.
        comparison_y = (
            [0] * (sum(fixed_prediction_counts[0]) + discordant[0])
            + [1] * (sum(fixed_prediction_counts[1]) + discordant[1])
        )
        comparison_pred = (
            [0] * fixed_prediction_counts[0][0]
            + [1] * fixed_prediction_counts[0][1]
            + [1] * k0 + [0] * (discordant[0] - k0)
            + [0] * fixed_prediction_counts[1][0]
            + [1] * fixed_prediction_counts[1][1]
            + [1] * k1 + [0] * (discordant[1] - k1)
        )
        baseline_pred = (
            [0] * fixed_prediction_counts[0][0]
            + [1] * fixed_prediction_counts[0][1]
            + [1] * (discordant[0] - k0) + [0] * k0
            + [0] * fixed_prediction_counts[1][0]
            + [1] * fixed_prediction_counts[1][1]
            + [1] * (discordant[1] - k1) + [0] * k1
        )
        return (
            classification_metrics(comparison_y, comparison_pred),
            classification_metrics(comparison_y, baseline_pred),
        )

    gold = [left[key][0] for key in sorted(left)]
    observed_left = classification_metrics(gold, [left[key][1] for key in sorted(left)])
    observed_right = classification_metrics(gold, [right[key][1] for key in sorted(left)])
    observed = {metric: observed_right[metric] - observed_left[metric] for metric in requested}
    tail_probability = {metric: 0.0 for metric in requested}
    null_mean = {metric: 0.0 for metric in requested}
    total_states = 0
    denominator = float(2 ** (discordant[0] + discordant[1]))
    for k0 in range(discordant[0] + 1):
        for k1 in range(discordant[1] + 1):
            weight = math.comb(discordant[0], k0) * math.comb(discordant[1], k1) / denominator
            comparison_metrics, baseline_metrics = metric_values(k0, k1)
            total_states += 1
            for metric in requested:
                value = comparison_metrics[metric] - baseline_metrics[metric]
                null_mean[metric] += weight * value
                if abs(value) >= abs(observed[metric]) - 1e-12:
                    tail_probability[metric] += weight
    return {
        metric: {
            "metric": metric,
            "observed_delta": observed[metric],
            "null_mean": null_mean[metric],
            "p_value": min(1.0, tail_probability[metric]),
            "method": "exact_subject_paired",
            "subjects": len({key[0] for key in left}),
            "keys": len(left),
            "discordant_subjects": discordant[0] + discordant[1],
            "compressed_states": total_states,
        }
        for metric in requested
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("Holm adjustment requires p-values in [0, 1].")
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted
