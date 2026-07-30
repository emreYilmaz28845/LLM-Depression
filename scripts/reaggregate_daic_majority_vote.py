from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, read_jsonl, save_json


CONDITIONS = ("c3", "c4")
HEADS = ("qwen", "logreg_raw", "xgb_raw")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(row: dict[str, Any], head: str) -> float:
    return float(row["teacher_forced_margin"] if head == "qwen" else row["probability"])


def _positive(score: float, head: str) -> bool:
    return score > 0.0 if head == "qwen" else score >= 0.5


def aggregate_majority_rows(rows: list[dict[str, Any]], head: str) -> list[dict[str, Any]]:
    if head not in HEADS:
        raise ValueError(f"Unsupported head: {head}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    output: list[dict[str, Any]] = []
    for subject_id, subject_rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in subject_rows}
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels: {sorted(labels)}")
        chunk_ids = [str(row["chunk_id"]) for row in subject_rows]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"Subject {subject_id} contains duplicate chunks.")
        scores = [_score(row, head) for row in subject_rows]
        positive_votes = sum(_positive(score, head) for score in scores)
        negative_votes = len(scores) - positive_votes
        vote_fraction = positive_votes / len(scores)
        mean_score = sum(scores) / len(scores)
        tied = positive_votes == negative_votes
        if positive_votes > negative_votes:
            prediction = 1
            decision = "strict_positive_majority"
        elif positive_votes < negative_votes:
            prediction = 0
            decision = "strict_negative_majority"
        else:
            prediction = int(_positive(mean_score, head))
            decision = "mean_continuous_score_tiebreak"
        output.append(
            {
                "subject_id": subject_id,
                "label": labels.pop(),
                "prediction": prediction,
                "num_chunks": len(scores),
                "positive_votes": positive_votes,
                "negative_votes": negative_votes,
                "positive_vote_fraction": vote_fraction,
                "mean_continuous_score": mean_score,
                "tie": tied,
                "decision": decision,
            }
        )
    return output


def _cell_paths(root: Path, condition: str, head: str) -> tuple[Path, Path]:
    if head == "qwen":
        directory = root / "qwen" / condition
        metrics = directory / "metrics_original_teacher_forced.json"
    else:
        directory = root / "classical" / condition / head
        metrics = directory / "metrics.json"
    return directory / "predictions_sample_level.jsonl", metrics


def run(root: Path, output_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for head in HEADS:
            predictions_path, original_metrics_path = _cell_paths(root, condition, head)
            sample_rows = read_jsonl(predictions_path)
            subject_rows = aggregate_majority_rows(sample_rows, head)
            labels = [int(row["label"]) for row in subject_rows]
            predictions = [int(row["prediction"]) for row in subject_rows]
            vote_fractions = [float(row["positive_vote_fraction"]) for row in subject_rows]
            metrics = classification_metrics(labels, predictions)
            metrics.update(
                {
                    "auroc": binary_auroc(labels, vote_fractions),
                    "predicted_positive_rate": sum(predictions) / len(predictions),
                    "aggregation_method": "strict_chunk_majority_with_mean_score_tiebreak",
                    "auroc_score": "positive_chunk_vote_fraction",
                    "post_hoc_exploratory": True,
                    "num_subjects": len(subject_rows),
                    "num_chunks": len(sample_rows),
                }
            )
            original = read_json(original_metrics_path)
            cell_dir = output_root / condition / head
            cell_dir.mkdir(parents=True, exist_ok=False)
            save_json(metrics, cell_dir / "metrics.json")
            with (cell_dir / "predictions_subject_level.jsonl").open("w", encoding="utf-8") as handle:
                for row in subject_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            provenance = {
                "schema_version": "daic_majority_reaggregation.v1",
                "condition": condition,
                "head": head,
                "post_hoc_exploratory": True,
                "source_predictions": str(predictions_path),
                "source_predictions_sha256": _sha256(predictions_path),
                "source_metrics": str(original_metrics_path),
                "source_metrics_sha256": _sha256(original_metrics_path),
                "chunk_vote_threshold": 0.0 if head == "qwen" else 0.5,
                "tie_rule": "mean continuous score with the original head threshold",
                "model_inference_rerun": False,
            }
            save_json(provenance, cell_dir / "provenance.json")
            cells.append(
                {
                    "condition": condition,
                    "head": head,
                    "num_subjects": len(subject_rows),
                    "num_chunks": len(sample_rows),
                    "positive_f1": metrics["positive_f1"],
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "auroc": metrics["auroc"],
                    "predicted_positive_rate": metrics["predicted_positive_rate"],
                    "original_positive_f1": original["positive_f1"],
                    "original_macro_f1": original["macro_f1"],
                    "delta_positive_f1": metrics["positive_f1"] - original["positive_f1"],
                    "delta_macro_f1": metrics["macro_f1"] - original["macro_f1"],
                    "tie_subjects": sum(int(row["tie"]) for row in subject_rows),
                }
            )
    payload = {
        "schema_version": "daic_majority_reaggregation_summary.v1",
        "source_experiment": str(root),
        "post_hoc_exploratory": True,
        "model_inference_rerun": False,
        "aggregation_method": "strict chunk majority; exact ties use mean continuous score",
        "cells": cells,
    }
    save_json(payload, output_root / "summary.json")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root or args.root / "exploratory_majority_vote"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    payload = run(args.root, output_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
