from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Sequence

from src.metrics import classification_metrics


def stratified_paired_bootstrap(
    baseline: Sequence[dict[str, Any]], comparison: Sequence[dict[str, Any]], *,
    metric: str = "macro_f1", iterations: int = 10000, seed: int = 1337,
) -> dict[str, float]:
    if int(iterations) < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    left_keys = [(str(row["subject_id"]), int(row.get("seed", 0))) for row in baseline]
    right_keys = [(str(row["subject_id"]), int(row.get("seed", 0))) for row in comparison]
    if len(left_keys) != len(set(left_keys)) or len(right_keys) != len(set(right_keys)):
        raise ValueError("Paired bootstrap inputs must contain one row per subject/seed key.")
    left = dict(zip(left_keys, baseline))
    right = dict(zip(right_keys, comparison))
    if set(left) != set(right):
        raise ValueError("Paired bootstrap requires identical subject/seed keys.")
    by_seed_label: dict[tuple[int, int], list[str]] = defaultdict(list)
    for subject_id, run_seed in left:
        by_seed_label[(run_seed, int(left[(subject_id, run_seed)]["label"]))].append(subject_id)
    for run_seed in sorted({key[1] for key in left}):
        if not by_seed_label[(run_seed, 0)] or not by_seed_label[(run_seed, 1)]:
            raise ValueError("Stratified bootstrap requires both labels for every seed.")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        per_seed = []
        for run_seed in sorted({key[1] for key in left}):
            sampled: list[str] = []
            for label in (0, 1):
                pool = by_seed_label[(run_seed, label)]
                sampled.extend(rng.choice(pool) for _ in pool)
            y = [int(left[(subject, run_seed)]["label"]) for subject in sampled]
            left_metric = classification_metrics(
                y, [int(left[(subject, run_seed)]["prediction"]) for subject in sampled]
            )
            right_metric = classification_metrics(
                y, [int(right[(subject, run_seed)]["prediction"]) for subject in sampled]
            )
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
