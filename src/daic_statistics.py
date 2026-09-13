from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Sequence

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
        prediction = int(row["prediction"])
        if prediction not in (0, 1):
            prediction = 1 - gold
        out[key] = (gold, prediction)
    return out


def stratified_paired_bootstrap(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metric: str = "macro_f1", iterations: int = 10000, seed: int = 1337,
) -> dict[str, float]:
    if int(iterations) < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    left = _rows_by_key(baseline, context="Paired bootstrap baseline")
    right = _rows_by_key(comparison, context="Paired bootstrap comparison")
    if set(left) != set(right):
        raise ValueError("Paired bootstrap requires identical subject/seed keys.")
    for key in left:
        if left[key][0] != right[key][0]:
            raise ValueError(f"Paired bootstrap label mismatch for subject {key[0]}.")
    by_seed_label: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for key in left:
        by_seed_label[(key[1], left[key][0])].append(key)
    for run_seed in sorted({key[1] for key in left}):
        if not by_seed_label[(run_seed, 0)] or not by_seed_label[(run_seed, 1)]:
            raise ValueError("Stratified bootstrap requires both labels for every seed.")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        per_seed = []
        for run_seed in sorted({key[1] for key in left}):
            sampled: list[tuple[str, int]] = []
            for label in (0, 1):
                pool = by_seed_label[(run_seed, label)]
                sampled.extend(rng.choice(pool) for _ in pool)
            y = [left[key][0] for key in sampled]
            left_metric = classification_metrics(y, [left[key][1] for key in sampled])
            right_metric = classification_metrics(y, [right[key][1] for key in sampled])
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


def exact_mcnemar(baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]]) -> dict[str, Any]:
    left_keys = [str(row["subject_id"]) for row in baseline]
    right_keys = [str(row["subject_id"]) for row in comparison]
    if len(left_keys) != len(set(left_keys)) or len(right_keys) != len(set(right_keys)):
        raise ValueError("McNemar inputs must contain one row per subject.")
    left = dict(zip(left_keys, baseline))
    right = dict(zip(right_keys, comparison))
    if set(left) != set(right):
        raise ValueError("McNemar requires identical subject keys.")
    b = c = 0
    for subject_id, row in left.items():
        gold = int(row["label"])
        if int(right[subject_id]["label"]) != gold:
            raise ValueError(f"McNemar label mismatch for subject {subject_id}.")
        b += int(int(row["prediction"]) == gold and int(right[subject_id]["prediction"]) != gold)
        c += int(int(row["prediction"]) != gold and int(right[subject_id]["prediction"]) == gold)
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n) if n else 1.0
    return {"baseline_only_correct": b, "comparison_only_correct": c, "p_value": min(1.0, 2.0 * tail)}


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
    left = _rows_by_key(baseline, context="Permutation baseline")
    right = _rows_by_key(comparison, context="Permutation comparison")
    if set(left) != set(right):
        raise ValueError("Permutation requires identical subject/seed keys.")
    keys = sorted(left)
    for key in keys:
        if left[key][0] != right[key][0]:
            raise ValueError(f"Permutation label mismatch for subject {key[0]}.")
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
    for _ in range(int(iterations)):
        swap = {key: rng.random() < 0.5 for key in keys}
        assignment = {key: (right[key][1] if swap[key] else left[key][1]) for key in keys}
        opposite = {key: (left[key][1] if swap[key] else right[key][1]) for key in keys}
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
