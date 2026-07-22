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

from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, read_jsonl, save_json


METRIC_KEYS = ("accuracy", "positive_f1", "negative_f1", "macro_f1", "precision", "recall", "auroc")


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


def summarize(root: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    for metrics_path in root.glob("*/*/*/fold_*/*/metrics.json"):
        variant_dir = metrics_path.parent
        fold_dir = variant_dir.parent
        run_name = fold_dir.parent.name
        modality = fold_dir.parent.parent.name
        dataset = fold_dir.parent.parent.parent.name
        groups[(dataset, modality, run_name, variant_dir.name)].append(variant_dir)
    summaries: list[dict[str, Any]] = []
    for (dataset, modality, run_name, variant), variant_dirs in sorted(groups.items()):
        fold_metrics = [read_json(path / "metrics.json") for path in sorted(variant_dirs)]
        subject_rows = []
        for path in sorted(variant_dirs):
            subject_rows.extend(read_jsonl(path / "predictions_subject_level.jsonl"))
        subject_ids = [str(row["subject_id"]) for row in subject_rows]
        if dataset == "cmdc" and len(subject_ids) != len(set(subject_ids)):
            raise ValueError(f"CMDC pooled held-out subjects overlap for {modality}/{run_name}/{variant}.")
        y_true = [int(row["label"]) for row in subject_rows]
        y_pred = [
            int(row["prediction"])
            if int(row["prediction"]) in (0, 1)
            else 1 - int(row["label"])
            for row in subject_rows
        ]
        probabilities = [float(row["probability"]) for row in subject_rows]
        pooled = classification_metrics(y_true, y_pred)
        pooled["negative_f1"] = _negative_f1(pooled)
        pooled["auroc"] = binary_auroc(y_true, probabilities)
        summary: dict[str, Any] = {
            "dataset": dataset,
            "modality": modality,
            "run_name": run_name,
            "classifier_variant": variant,
            "folds": len(fold_metrics),
            "pooled_subjects": len(subject_rows),
            "input_dimension": 3584,
            "post_pca_dimension": 32 if "pca32" in variant else 64 if "pca64" in variant else 3584,
            "pooling": "last_valid_prompt_token",
        }
        for key in METRIC_KEYS:
            values = [float(metrics[key]) for metrics in fold_metrics]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"pooled_{key}"] = float(pooled[key])
        summary["pooled_confusion_matrix"] = json.dumps(pooled["confusion_matrix"])
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Qwen hidden classifier folds and pooled predictions.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs" / "hidden_classifiers")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "hidden_classifiers")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(args.root)
    save_json(rows, args.output_dir / "summary.json")
    _write_csv(rows, args.output_dir / "summary.csv")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
