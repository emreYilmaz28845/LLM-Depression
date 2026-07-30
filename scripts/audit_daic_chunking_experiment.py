from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import classification_metrics
from src.utils import read_json, read_jsonl, save_json


CONDITIONS = ("c1", "c2", "c3", "c4")
HEADS = ("qwen", "logreg_raw", "xgb_raw")


def _qwen_dir(root: Path, condition: str) -> Path:
    return root / "qwen" / condition


def _head_dir(root: Path, condition: str, head: str) -> Path:
    return root / "classical" / condition / head


def _load_cell(root: Path, condition: str, head: str) -> tuple[dict, list[dict], list[dict]]:
    directory = _qwen_dir(root, condition) if head == "qwen" else _head_dir(root, condition, head)
    metrics_name = "metrics_original_teacher_forced.json" if head == "qwen" else "metrics.json"
    metrics = read_json(directory / metrics_name)
    samples = read_jsonl(directory / "predictions_sample_level.jsonl")
    if head == "qwen":
        with (directory / "predictions_subject_level.csv").open(newline="", encoding="utf-8") as handle:
            subjects = list(csv.DictReader(handle))
    else:
        subjects = read_jsonl(directory / "predictions_subject_level.jsonl")
    return metrics, samples, subjects


def _prediction(row: dict[str, Any]) -> int:
    return int(row.get("prediction", row.get("aggregated_prediction")))


def _paired_bootstrap(
    baseline: list[dict], comparison: list[dict], *, iterations: int = 10000, seed: int = 1337
) -> dict[str, Any]:
    base = {str(row["subject_id"]): row for row in baseline}
    comp = {str(row["subject_id"]): row for row in comparison}
    ids = sorted(base)
    if ids != sorted(comp):
        raise ValueError("Paired comparison subject sets differ.")
    rng = random.Random(seed)
    deltas = {"positive_f1": [], "macro_f1": []}
    for _ in range(iterations):
        sampled = [rng.choice(ids) for _ in ids]
        y = [int(base[sid]["label"]) for sid in sampled]
        base_metrics = classification_metrics(y, [_prediction(base[sid]) for sid in sampled])
        comp_metrics = classification_metrics(y, [_prediction(comp[sid]) for sid in sampled])
        for metric in deltas:
            deltas[metric].append(comp_metrics[metric] - base_metrics[metric])
    output = {}
    for metric, values in deltas.items():
        values.sort()
        output[metric] = {
            "low": values[int(0.025 * (len(values) - 1))],
            "high": values[int(0.975 * (len(values) - 1))],
        }
    return output


def _mcnemar(baseline: list[dict], comparison: list[dict]) -> dict[str, Any]:
    base = {str(row["subject_id"]): row for row in baseline}
    comp = {str(row["subject_id"]): row for row in comparison}
    if set(base) != set(comp):
        raise ValueError("McNemar subject sets differ.")
    b = c = 0
    for subject_id in base:
        gold = int(base[subject_id]["label"])
        base_correct = _prediction(base[subject_id]) == gold
        comp_correct = _prediction(comp[subject_id]) == gold
        b += int(base_correct and not comp_correct)
        c += int(not base_correct and comp_correct)
    n = b + c
    tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1)) / (2**n) if n else 1.0
    return {"baseline_only_correct": b, "comparison_only_correct": c, "exact_two_sided_p": min(1.0, 2 * tail)}


def audit(root: Path, sacct_path: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    cells: list[dict[str, Any]] = []
    subject_rows: dict[tuple[str, str], list[dict]] = {}
    for condition in CONDITIONS:
        for head in HEADS:
            try:
                metrics, samples, subjects = _load_cell(root, condition, head)
                if len(subjects) != len({str(row["subject_id"]) for row in subjects}):
                    failures.append(f"{condition}/{head}: not exactly one prediction per subject")
                required = {"accuracy", "positive_f1", "macro_f1", "precision", "recall", "confusion_matrix", "auroc", "predicted_positive_rate"}
                missing = sorted(required - set(metrics))
                if missing:
                    failures.append(f"{condition}/{head}: missing metrics {missing}")
                subject_rows[(condition, head)] = subjects
                cells.append(
                    {
                        "condition": condition,
                        "head": head,
                        **{key: metrics.get(key) for key in sorted(required)},
                        "num_samples": len(samples),
                        "num_subjects": len(subjects),
                    }
                )
                if condition in {"c3", "c4"}:
                    sample_ids = [str(row["sample_id"]) for row in samples]
                    if len(sample_ids) != len(set(sample_ids)):
                        failures.append(f"{condition}/{head}: held-out chunk duplicated")
            except Exception as exc:
                failures.append(f"{condition}/{head}: {exc}")

    try:
        c1_meta = read_json(root / "hidden/joint/c1_fixed/extraction_metadata.json")
        c2_meta = read_json(root / "hidden/joint/c2_balanced/extraction_metadata.json")
        for key in ("adapter_sha256", "adapter_config_sha256", "saved_split_sha256", "manifest_sha256"):
            if c1_meta.get(key) != c2_meta.get(key):
                failures.append(f"C1/C2 hidden metadata mismatch: {key}")
        for variant in ("logreg_raw", "xgb_raw"):
            c1_cfg = read_json(root / f"classical/c1/{variant}/result_config.json")
            c2_cfg = read_json(root / f"classical/c2/{variant}/result_config.json")
            c2_head = read_json(root / f"classical/c2/{variant}/classifier_metadata.json")
            if c1_cfg["fit_cache_identity"] != c2_cfg["fit_cache_identity"]:
                failures.append(f"C1/C2 {variant} fit cache was not reused")
            if c2_head.get("fit_action") != "reused":
                failures.append(f"C2 {variant} fitted model was not reused")
    except Exception as exc:
        failures.append(f"C1/C2 reuse audit: {exc}")

    sacct = None
    if sacct_path:
        sacct = sacct_path.read_text(encoding="utf-8")
        bad = [
            state
            for state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
            if state in sacct
        ]
        if bad:
            failures.append(f"Slurm accounting contains terminal failures: {bad}")

    comparisons = []
    for head in HEADS:
        baseline = subject_rows.get(("c1", head))
        if not baseline:
            continue
        for condition in ("c2", "c3", "c4"):
            comparison = subject_rows.get((condition, head))
            if not comparison:
                continue
            baseline_cell = next(row for row in cells if row["condition"] == "c1" and row["head"] == head)
            comparison_cell = next(row for row in cells if row["condition"] == condition and row["head"] == head)
            comparisons.append(
                {
                    "head": head,
                    "condition": condition,
                    "delta_positive_f1": comparison_cell["positive_f1"] - baseline_cell["positive_f1"],
                    "delta_macro_f1": comparison_cell["macro_f1"] - baseline_cell["macro_f1"],
                    "paired_bootstrap_95ci": _paired_bootstrap(baseline, comparison),
                    "mcnemar": _mcnemar(baseline, comparison),
                }
            )
    payload = {
        "schema_version": "daic_chunking_experiment_audit.v1",
        "root": str(root),
        "passed": not failures and len(cells) == 12,
        "failures": failures,
        "cells": cells,
        "comparisons_against_c1": comparisons,
        "slurm_accounting_path": str(sacct_path) if sacct_path else None,
        "slurm_accounting_captured": sacct is not None,
    }
    save_json(payload, root / "audit.json")
    with (root / "comparison_12_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]) if cells else ["condition", "head"])
        writer.writeheader()
        writer.writerows(cells)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sacct", type=Path)
    args = parser.parse_args()
    payload = audit(args.root, args.sacct)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
