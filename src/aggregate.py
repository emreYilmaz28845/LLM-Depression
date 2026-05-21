from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.metrics import classification_metrics
from src.utils import LABEL_DEPRESSED, LABEL_NON_DEPRESSED, label_text_from_int


def aggregate_likelihood_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for subject_id, rows in sorted(grouped.items()):
        dep_scores = [float(row["dep_score"]) for row in rows]
        non_scores = [float(row["non_score"]) for row in rows]
        mean_dep = sum(dep_scores) / len(dep_scores)
        mean_non = sum(non_scores) / len(non_scores)
        pred = int(mean_dep > mean_non)
        gold = int(rows[0]["label"])
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction": pred,
                "prediction_text": label_text_from_int(pred),
                "dep_score": mean_dep,
                "non_score": mean_non,
                "num_samples": len(rows),
            }
        )
        y_true.append(gold)
        y_pred.append(pred)
    metrics = classification_metrics(y_true, y_pred)
    metrics["num_subjects"] = len(subject_rows)
    return subject_rows, metrics


def aggregate_generation_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    invalid_subjects = 0
    total_invalid_generations = 0
    for subject_id, rows in sorted(grouped.items()):
        valid_predictions = [int(row["parsed_prediction"]) for row in rows if row["parsed_prediction"] in (0, 1)]
        total_invalid_generations += sum(1 for row in rows if row["parsed_prediction"] not in (0, 1))
        gold = int(rows[0]["label"])
        if not valid_predictions:
            invalid_subjects += 1
            pred = -1
            pred_text = "INVALID"
        else:
            counts = Counter(valid_predictions)
            if counts[1] > counts[0]:
                pred = 1
            elif counts[0] > counts[1]:
                pred = 0
            else:
                pred = -1
            pred_text = label_text_from_int(pred) if pred in (0, 1) else "INVALID"
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction": pred,
                "prediction_text": pred_text,
                "num_samples": len(rows),
                "num_valid_predictions": len(valid_predictions),
            }
        )
        if pred in (0, 1):
            y_true.append(gold)
            y_pred.append(pred)

    metrics = classification_metrics(y_true, y_pred) if y_true else {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "positive_f1": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "support_negative": 0,
        "support_positive": 0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }
    metrics["num_subjects"] = len(subject_rows)
    metrics["invalid_subjects"] = invalid_subjects
    metrics["invalid_generations"] = total_invalid_generations
    return subject_rows, metrics


def parse_generation_label(text: str) -> int | None:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return None
    depressed_markers = [
        "depressed",
        "depression",
        "speaker is depressed",
        "subject is depressed",
    ]
    non_markers = [
        "non-depressed",
        "non depressed",
        "not depressed",
        "healthy",
        "nondepressed",
    ]
    has_dep = any(marker in normalized for marker in depressed_markers)
    has_non = any(marker in normalized for marker in non_markers)
    if has_dep and not has_non:
        return 1
    if has_non and not has_dep:
        return 0
    return None

