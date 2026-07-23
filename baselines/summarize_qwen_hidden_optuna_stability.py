from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.build_qwen_hidden_optuna_followup_matrix import (
    PILOT_CONDITIONS,
    SEED_IDS,
    STAGE1_ID,
)
from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, read_jsonl, save_json


EXPERIMENT_IDS = [STAGE1_ID, SEED_IDS[7], SEED_IDS[2024]]
EXPECTED_INNER_SEEDS = {
    STAGE1_ID: 1337,
    SEED_IDS[7]: 7,
    SEED_IDS[2024]: 2024,
}
METRIC_KEYS = ("accuracy", "positive_f1", "macro_f1", "precision", "recall", "auroc")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _negative_f1(metrics: dict[str, Any]) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _per_experiment_rows(root: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    for experiment_id in EXPERIMENT_IDS:
        for result_dir in root.glob(f"*/*/*/fold_*/{experiment_id}"):
            if (result_dir / "metrics.json").is_file():
                fold_dir = result_dir.parent
                groups[
                    (
                        fold_dir.parent.parent.parent.name,
                        fold_dir.parent.parent.name,
                        fold_dir.parent.name,
                        experiment_id,
                    )
                ].append(result_dir)
    rows: list[dict[str, Any]] = []
    for (dataset, condition, run_name, experiment_id), result_dirs in sorted(groups.items()):
        expected_folds = 1 if dataset == "daic" else 5
        if len(result_dirs) != expected_folds:
            raise ValueError(
                f"Expected {expected_folds} folds for {dataset}/{condition}/{experiment_id}, "
                f"found {len(result_dirs)}."
            )
        fold_metrics: list[dict[str, Any]] = []
        subject_rows: list[dict[str, Any]] = []
        for result_dir in sorted(result_dirs):
            metadata = read_json(result_dir / "classifier_metadata.json")
            if metadata.get("experiment_id") != experiment_id:
                raise ValueError(f"Experiment identity mismatch: {result_dir}")
            if int(metadata.get("inner_seed", -1)) != EXPECTED_INNER_SEEDS[experiment_id]:
                raise ValueError(f"Inner seed mismatch: {result_dir}")
            if int(metadata.get("completed_trials", -1)) != 150:
                raise ValueError(f"Study is not complete at 150 trials: {result_dir}")
            fold_metrics.append(read_json(result_dir / "metrics.json"))
            subject_rows.extend(read_jsonl(result_dir / "predictions_subject_level.jsonl"))
        subject_ids = [str(row["subject_id"]) for row in subject_rows]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError(
                f"Outer-fold subject overlap for {dataset}/{condition}/{experiment_id}."
            )
        y_true = [int(row["label"]) for row in subject_rows]
        y_pred = [int(row["prediction"]) for row in subject_rows]
        probabilities = [float(row["probability"]) for row in subject_rows]
        pooled = classification_metrics(y_true, y_pred)
        pooled["negative_f1"] = _negative_f1(pooled)
        pooled["auroc"] = binary_auroc(y_true, probabilities)
        row: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "inner_seed": EXPECTED_INNER_SEEDS[experiment_id],
            "folds": len(result_dirs),
            "pooled_subjects": len(subject_rows),
            "primary_metric": "macro_f1" if dataset == "turkish" else "positive_f1",
            "pooled_confusion_matrix": json.dumps(pooled["confusion_matrix"]),
        }
        for key in (*METRIC_KEYS, "negative_f1"):
            values = [float(metrics[key]) for metrics in fold_metrics]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"pooled_{key}"] = float(pooled[key])
        rows.append(row)
    return rows


def summarize_stability(root: Path, gate_threshold: float = 0.03) -> dict[str, Any]:
    per_seed_rows = _per_experiment_rows(root)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        groups[(row["dataset"], row["condition"], row["run_name"])].append(row)
    available_conditions = {(dataset, condition) for dataset, condition, _ in groups}
    if not PILOT_CONDITIONS.issubset(available_conditions):
        raise ValueError("The stability panel does not contain all three representative conditions.")
    stability_rows: list[dict[str, Any]] = []
    for (dataset, condition, run_name), rows in sorted(groups.items()):
        if {int(row["inner_seed"]) for row in rows} != {7, 1337, 2024}:
            raise ValueError(f"Incomplete seed panel for {dataset}/{condition}.")
        primary_metric = "macro_f1" if dataset == "turkish" else "positive_f1"
        values = [float(row[f"pooled_{primary_metric}"]) for row in rows]
        stability_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "run_name": run_name,
                "primary_metric": primary_metric,
                "seed_count": len(rows),
                "primary_mean": float(np.mean(values)),
                "primary_std": float(np.std(values, ddof=1)),
                "primary_min": float(np.min(values)),
                "primary_max": float(np.max(values)),
                "primary_range": float(np.max(values) - np.min(values)),
                "pilot_condition": (dataset, condition) in PILOT_CONDITIONS,
            }
        )
    pilot_stability_rows = [row for row in stability_rows if row["pilot_condition"]]
    observed_max = max(float(row["primary_range"]) for row in pilot_stability_rows)
    return {
        "schema_version": "qwen_hidden_optuna_stability.v1",
        "source_experiment_ids": EXPERIMENT_IDS,
        "gate_threshold": gate_threshold,
        "observed_max_primary_range": observed_max,
        "expand_all": observed_max >= gate_threshold,
        "per_seed_rows": per_seed_rows,
        "stability_rows": stability_rows,
        "pilot_stability_rows": pilot_stability_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Optuna inner-fold seed stability.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hidden_classifiers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hidden_classifiers" / "optuna_stability",
    )
    parser.add_argument("--gate-threshold", type=float, default=0.03)
    args = parser.parse_args()
    if args.gate_threshold < 0:
        raise ValueError("gate-threshold must be non-negative.")
    payload = summarize_stability(args.root, args.gate_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "stability_summary.json")
    _write_csv(payload["per_seed_rows"], args.output_dir / "stability_per_seed.csv")
    _write_csv(payload["stability_rows"], args.output_dir / "stability_ranges.csv")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
