from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.metrics import classification_metrics, multiclass_macro_f1
from src.utils import (
    PREDICTION_MODE_GENERATION,
    PREDICTION_MODE_LIKELIHOOD,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    evaluation_protocol_name,
    label_text_from_int,
)


INVALID_PREDICTION = -1
DIAGNOSTIC_LABELS = [0, 1, INVALID_PREDICTION]
DIAGNOSTIC_LABEL_NAMES = {
    0: "Non-depressed",
    1: "Depressed",
    INVALID_PREDICTION: "INVALID",
}


def _prediction_count_payload(subject_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["prediction"]) for row in subject_rows)
    return {
        "predicted_non_depressed_subjects": int(counts[0]),
        "predicted_depressed_subjects": int(counts[1]),
        "predicted_invalid_subjects": int(counts[INVALID_PREDICTION]),
    }


def _true_count_payload(subject_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in subject_rows)
    return {
        "true_non_depressed_subjects": int(counts[0]),
        "true_depressed_subjects": int(counts[1]),
    }


def _strict_binary_prediction(gold: int, pred: int) -> int:
    if pred in (0, 1):
        return pred
    return 1 - int(gold)


def _diagnostic_payload(subject_rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in subject_rows]
    y_pred = [int(row["prediction"]) for row in subject_rows]
    diagnostic = multiclass_macro_f1(y_true, y_pred, DIAGNOSTIC_LABELS)
    return {
        "diagnostic_three_class_labels": [DIAGNOSTIC_LABEL_NAMES[label] for label in DIAGNOSTIC_LABELS],
        "diagnostic_three_class_confusion_matrix": diagnostic["confusion_matrix"],
        "diagnostic_three_class_macro_f1": diagnostic["macro_f1"],
        "diagnostic_three_class_per_class": {
            DIAGNOSTIC_LABEL_NAMES[label]: stats for label, stats in diagnostic["per_class"].items()
        },
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
    binary_metrics = classification_metrics(y_true, y_pred)
    metrics = {
        **binary_metrics,
        "binary_strict_accuracy": binary_metrics["accuracy"],
        "binary_strict_precision": binary_metrics["precision"],
        "binary_strict_recall": binary_metrics["recall"],
        "binary_strict_positive_f1": binary_metrics["positive_f1"],
        "binary_strict_macro_f1": binary_metrics["macro_f1"],
        "binary_strict_weighted_f1": binary_metrics["weighted_f1"],
        "binary_strict_confusion_matrix": binary_metrics["confusion_matrix"],
    }
    metrics["num_subjects"] = len(subject_rows)
    metrics["invalid_subjects"] = 0
    metrics["prediction_backend"] = PREDICTION_MODE_LIKELIHOOD
    metrics["evaluation_protocol_name"] = evaluation_protocol_name(PREDICTION_MODE_LIKELIHOOD)
    metrics["aggregation_level"] = "subject"
    metrics["num_valid_subject_predictions"] = len(subject_rows)
    metrics["valid_only_accuracy"] = binary_metrics["accuracy"]
    metrics["valid_only_precision"] = binary_metrics["precision"]
    metrics["valid_only_recall"] = binary_metrics["recall"]
    metrics["valid_only_positive_f1"] = binary_metrics["positive_f1"]
    metrics["valid_only_macro_f1"] = binary_metrics["macro_f1"]
    metrics["valid_only_weighted_f1"] = binary_metrics["weighted_f1"]
    metrics.update(_prediction_count_payload(subject_rows))
    metrics.update(_true_count_payload(subject_rows))
    metrics.update(_diagnostic_payload(subject_rows))
    return subject_rows, metrics


def _aggregate_majority_vote_predictions(
    sample_rows: list[dict[str, Any]],
    *,
    prediction_field: str,
    backend_name: str,
    invalid_metric_name: str,
    valid_count_field: str,
    tie_break_positive_score_field: str | None = None,
    tie_break_negative_score_field: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    y_true_valid: list[int] = []
    y_pred_valid: list[int] = []
    invalid_subjects = 0
    total_invalid_predictions = 0
    for subject_id, rows in sorted(grouped.items()):
        valid_predictions = [int(row[prediction_field]) for row in rows if row[prediction_field] in (0, 1)]
        total_invalid_predictions += sum(1 for row in rows if row[prediction_field] not in (0, 1))
        gold = int(rows[0]["label"])
        if not valid_predictions:
            invalid_subjects += 1
            pred = INVALID_PREDICTION
        else:
            counts = Counter(valid_predictions)
            if counts[1] > counts[0]:
                pred = 1
            elif counts[0] > counts[1]:
                pred = 0
            else:
                if tie_break_positive_score_field and tie_break_negative_score_field:
                    dep_margin = sum(
                        float(row.get(tie_break_positive_score_field, 0.0))
                        - float(row.get(tie_break_negative_score_field, 0.0))
                        for row in rows
                    )
                    if dep_margin > 0:
                        pred = 1
                    elif dep_margin < 0:
                        pred = 0
                    else:
                        pred = INVALID_PREDICTION
                else:
                    pred = INVALID_PREDICTION
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
            y_true_valid.append(gold)
            y_pred_valid.append(pred)

    valid_only_metrics = classification_metrics(y_true_valid, y_pred_valid) if y_true_valid else {
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
    strict_y_true = [int(row["label"]) for row in subject_rows]
    strict_y_pred = [_strict_binary_prediction(int(row["label"]), int(row["prediction"])) for row in subject_rows]
    strict_metrics = classification_metrics(strict_y_true, strict_y_pred) if strict_y_true else {
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
    metrics = {
        **strict_metrics,
        "binary_strict_accuracy": strict_metrics["accuracy"],
        "binary_strict_precision": strict_metrics["precision"],
        "binary_strict_recall": strict_metrics["recall"],
        "binary_strict_positive_f1": strict_metrics["positive_f1"],
        "binary_strict_macro_f1": strict_metrics["macro_f1"],
        "binary_strict_weighted_f1": strict_metrics["weighted_f1"],
        "binary_strict_confusion_matrix": strict_metrics["confusion_matrix"],
    }
    metrics["num_subjects"] = len(subject_rows)
    metrics["invalid_subjects"] = invalid_subjects
    metrics["num_valid_subject_predictions"] = len(y_true_valid)
    metrics[invalid_metric_name] = total_invalid_predictions
    metrics["prediction_backend"] = backend_name
    metrics["evaluation_protocol_name"] = evaluation_protocol_name(backend_name)
    metrics["aggregation_level"] = "subject"
    metrics["valid_only_accuracy"] = valid_only_metrics["accuracy"]
    metrics["valid_only_precision"] = valid_only_metrics["precision"]
    metrics["valid_only_recall"] = valid_only_metrics["recall"]
    metrics["valid_only_positive_f1"] = valid_only_metrics["positive_f1"]
    metrics["valid_only_macro_f1"] = valid_only_metrics["macro_f1"]
    metrics["valid_only_weighted_f1"] = valid_only_metrics["weighted_f1"]
    metrics.update(_prediction_count_payload(subject_rows))
    metrics.update(_true_count_payload(subject_rows))
    metrics.update(_diagnostic_payload(subject_rows))
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
        tie_break_positive_score_field="dep_score",
        tie_break_negative_score_field="non_score",
    )
