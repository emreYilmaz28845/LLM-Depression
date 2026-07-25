from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.metrics import classification_metrics
from src.utils import read_jsonl, save_json


def _negative_metrics(metrics: dict[str, Any]) -> tuple[float, float]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, recall


def summarize(matrix_path: Path, results_root: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    job_lookup = {}
    for job in matrix["jobs"]:
        result = (
            results_root
            / job["dataset"]
            / job["condition"]
            / Path(job["run_dir"]).name
            / f"fold_{job['fold']}"
            / job["experiment_id"]
        )
        rows = read_jsonl(result / "predictions_subject_level.jsonl")
        grouped[(job["condition"], job["experiment_id"])].extend(rows)
        job_lookup[(job["condition"], job["experiment_id"])] = job

    result_rows = []
    for key, rows in sorted(grouped.items()):
        condition, experiment_id = key
        subject_ids = [str(row["subject_id"]) for row in rows]
        if len(subject_ids) != 120 or len(set(subject_ids)) != 120:
            raise ValueError(f"Incomplete pooled Turkish coverage for {key}: {len(subject_ids)}")
        metrics = classification_metrics(
            [int(row["label"]) for row in rows],
            [int(row["prediction"]) for row in rows],
        )
        negative_f1, negative_recall = _negative_metrics(metrics)
        job = job_lookup[key]
        result_rows.append(
            {
                "condition": condition,
                "experiment_id": experiment_id,
                "sampling_mode": job["sampling_mode"],
                "oversampling_ratio": job.get("oversampling_ratio"),
                "oversampling_seed": int(job["oversampling_seed"]),
                "macro_f1": metrics["macro_f1"],
                "negative_f1": negative_f1,
                "negative_recall": negative_recall,
                "positive_f1": metrics["positive_f1"],
                "positive_recall": metrics["recall"],
                "balanced_accuracy": metrics["macro_recall"],
                "accuracy": metrics["accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    decisions = []
    for condition in ("audio_text", "audio_only", "text_only"):
        control = next(
            row
            for row in result_rows
            if row["condition"] == condition and row["sampling_mode"] == "none"
        )
        sampled = [
            row
            for row in result_rows
            if row["condition"] == condition
            and row["sampling_mode"] == "minority_subject_oversample"
        ]
        macro_gains = [row["macro_f1"] - control["macro_f1"] for row in sampled]
        negative_recall_gains = [
            row["negative_recall"] - control["negative_recall"] for row in sampled
        ]
        positive_recall_losses = [
            control["positive_recall"] - row["positive_recall"] for row in sampled
        ]
        decisions.append(
            {
                "condition": condition,
                "control_macro_f1": control["macro_f1"],
                "mean_sampled_macro_f1": statistics.mean(
                    row["macro_f1"] for row in sampled
                ),
                "mean_macro_f1_gain": statistics.mean(macro_gains),
                "seeds_beating_control": sum(gain > 0 for gain in macro_gains),
                "mean_negative_recall_gain": statistics.mean(negative_recall_gains),
                "mean_positive_recall_loss": statistics.mean(positive_recall_losses),
                "qualifies_modality_gate": (
                    statistics.mean(macro_gains) >= 0.02
                    and sum(gain > 0 for gain in macro_gains) >= 2
                    and statistics.mean(negative_recall_gains) >= 0.05
                    and statistics.mean(positive_recall_losses) <= 0.10
                ),
            }
        )
    control_overall = statistics.mean(
        row["macro_f1"] for row in result_rows if row["sampling_mode"] == "none"
    )
    sampled_overall = statistics.mean(
        row["macro_f1"]
        for row in result_rows
        if row["sampling_mode"] == "minority_subject_oversample"
    )
    global_gate = sampled_overall >= control_overall - 0.01
    qualifying = [
        row["condition"]
        for row in decisions
        if row["qualifies_modality_gate"] and global_gate
    ]
    return {
        "schema_version": "turkish_oversampling_optuna_summary.v1",
        "selected_ratio": matrix["selected_ratio"],
        "expected_studies": 60,
        "observed_studies": len(matrix["jobs"]),
        "control_all_modality_macro_f1": control_overall,
        "sampled_all_modality_macro_f1": sampled_overall,
        "all_modality_noninferiority_gate": global_gate,
        "qualifying_modalities": qualifying,
        "proceed_to_qwen": bool(qualifying),
        "decisions": decisions,
        "results": result_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.matrix, args.results_root)
    save_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
