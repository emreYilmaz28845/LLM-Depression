from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.metrics import classification_metrics
from src.utils import (
    PREDICTION_MODE_GENERATION,
    PREDICTION_MODE_LIKELIHOOD,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    evaluation_protocol_name,
    label_text_from_int,
)


def _prediction_count_payload(subject_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["prediction"]) for row in subject_rows)
    return {
        "predicted_non_depressed_subjects": int(counts[0]),
        "predicted_depressed_subjects": int(counts[1]),
        "predicted_invalid_subjects": int(counts[-1]),
    }


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
                "prediction_backend": PREDICTION_MODE_LIKELIHOOD,
                "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_LIKELIHOOD),
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
    metrics["invalid_subjects"] = 0
    metrics["prediction_backend"] = PREDICTION_MODE_LIKELIHOOD
    metrics["evaluation_protocol_name"] = evaluation_protocol_name(PREDICTION_MODE_LIKELIHOOD)
    metrics["aggregation_level"] = "subject"
    metrics.update(_prediction_count_payload(subject_rows))
    return subject_rows, metrics


def _aggregate_majority_vote_predictions(
    sample_rows: list[dict[str, Any]],
    *,
    prediction_field: str,
    backend_name: str,
    invalid_metric_name: str,
    valid_count_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    invalid_subjects = 0
    total_invalid_predictions = 0
    for subject_id, rows in sorted(grouped.items()):
        valid_predictions = [int(row[prediction_field]) for row in rows if row[prediction_field] in (0, 1)]
        total_invalid_predictions += sum(1 for row in rows if row[prediction_field] not in (0, 1))
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
                "prediction_backend": backend_name,
                "evaluation_protocol_name": evaluation_protocol_name(backend_name),
                "prediction": pred,
                "prediction_text": pred_text,
                "num_samples": len(rows),
                valid_count_field: len(valid_predictions),
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
    metrics[invalid_metric_name] = total_invalid_predictions
    metrics["prediction_backend"] = backend_name
    metrics["evaluation_protocol_name"] = evaluation_protocol_name(backend_name)
    metrics["aggregation_level"] = "subject"
    metrics.update(_prediction_count_payload(subject_rows))
    return subject_rows, metrics


def aggregate_generation_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _aggregate_majority_vote_predictions(
        sample_rows,
        prediction_field="parsed_prediction",
        backend_name=PREDICTION_MODE_GENERATION,
        invalid_metric_name="invalid_generations",
        valid_count_field="generation_num_valid_predictions",
    )


def aggregate_original_teacher_forced_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _aggregate_majority_vote_predictions(
        sample_rows,
        prediction_field="teacher_forced_prediction",
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        invalid_metric_name="invalid_teacher_forced_predictions",
        valid_count_field="teacher_forced_num_valid_predictions",
    )


def parse_generation_label(text: str) -> int | None:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return None
    non_markers = [
        "non-depressed",
        "non depressed",
        "not depressed",
        "healthy",
        "nondepressed",
    ]
    # Check negative-class markers first so "non-depressed" does not
    # accidentally trigger the positive "depressed" substring match.
    if any(marker in normalized for marker in non_markers):
        return 0
    depressed_markers = [
        "depression",
        "speaker is depressed",
        "subject is depressed",
    ]
    if any(marker in normalized for marker in depressed_markers):
        return 1
    if normalized == "depressed" or normalized.endswith(" depressed"):
        return 1
    return None
