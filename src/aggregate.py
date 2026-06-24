from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.metrics import binary_auroc, classification_metrics, multiclass_macro_f1
from src.utils import (
    AGGREGATION_LEVEL_SEGMENT,
    AGGREGATION_LEVEL_SUBJECT,
    PREDICTION_MODE_GENERATION,
    PREDICTION_MODE_LIKELIHOOD,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    evaluation_protocol_name,
    label_text_from_int,
    normalize_aggregation_level,
    normalize_prediction_mode,
)


INVALID_PREDICTION = -1
DIAGNOSTIC_LABELS = [0, 1, INVALID_PREDICTION]
DIAGNOSTIC_LABEL_NAMES = {
    0: "Non-depressed",
    1: "Depressed",
    INVALID_PREDICTION: "INVALID",
}


def _strict_binary_prediction(gold: int, pred: int) -> int:
    if pred in (0, 1):
        return pred
    return 1 - int(gold)


def _wrong_vote_for_gold(gold: int) -> int:
    return 1 - int(gold)


def _unit_suffix(aggregation_level: str) -> str:
    return "subjects" if aggregation_level == AGGREGATION_LEVEL_SUBJECT else "segments"


def _count_payload(rows: list[dict[str, Any]], aggregation_level: str) -> dict[str, int]:
    suffix = _unit_suffix(aggregation_level)
    prediction_counts = Counter(int(row["prediction"]) for row in rows)
    label_counts = Counter(int(row["label"]) for row in rows)
    return {
        f"predicted_non_depressed_{suffix}": int(prediction_counts[0]),
        f"predicted_depressed_{suffix}": int(prediction_counts[1]),
        f"predicted_invalid_{suffix}": int(prediction_counts[INVALID_PREDICTION]),
        f"true_non_depressed_{suffix}": int(label_counts[0]),
        f"true_depressed_{suffix}": int(label_counts[1]),
    }


def _diagnostic_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    diagnostic = multiclass_macro_f1(y_true, y_pred, DIAGNOSTIC_LABELS)
    return {
        "diagnostic_three_class_labels": [DIAGNOSTIC_LABEL_NAMES[label] for label in DIAGNOSTIC_LABELS],
        "diagnostic_three_class_confusion_matrix": diagnostic["confusion_matrix"],
        "diagnostic_three_class_macro_f1": diagnostic["macro_f1"],
        "diagnostic_three_class_per_class": {
            DIAGNOSTIC_LABEL_NAMES[label]: stats for label, stats in diagnostic["per_class"].items()
        },
    }


def _zero_binary_metrics() -> dict[str, Any]:
    return {
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


def _metrics_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    backend_name: str,
    aggregation_level: str,
    invalid_metric_name: str | None = None,
    invalid_prediction_count: int = 0,
) -> dict[str, Any]:
    aggregation_level = normalize_aggregation_level(aggregation_level)
    valid_rows = [row for row in rows if int(row["prediction"]) in (0, 1)]
    valid_y_true = [int(row["label"]) for row in valid_rows]
    valid_y_pred = [int(row["prediction"]) for row in valid_rows]
    valid_only_metrics = classification_metrics(valid_y_true, valid_y_pred) if valid_y_true else _zero_binary_metrics()

    strict_y_true = [int(row["label"]) for row in rows]
    strict_y_pred = [_strict_binary_prediction(int(row["label"]), int(row["prediction"])) for row in rows]
    strict_metrics = classification_metrics(strict_y_true, strict_y_pred) if strict_y_true else _zero_binary_metrics()

    unit_suffix = _unit_suffix(aggregation_level)
    invalid_units = sum(1 for row in rows if int(row["prediction"]) not in (0, 1))
    metrics = {
        **strict_metrics,
        "binary_strict_accuracy": strict_metrics["accuracy"],
        "binary_strict_precision": strict_metrics["precision"],
        "binary_strict_recall": strict_metrics["recall"],
        "binary_strict_positive_f1": strict_metrics["positive_f1"],
        "binary_strict_macro_f1": strict_metrics["macro_f1"],
        "binary_strict_weighted_f1": strict_metrics["weighted_f1"],
        "binary_strict_confusion_matrix": strict_metrics["confusion_matrix"],
        "prediction_backend": backend_name,
        "evaluation_protocol_name": evaluation_protocol_name(backend_name),
        "aggregation_level": aggregation_level,
        "unit_label": aggregation_level,
        "num_units": len(rows),
        f"num_{unit_suffix}": len(rows),
        f"invalid_{unit_suffix}": invalid_units,
        f"num_valid_{aggregation_level}_predictions": len(valid_rows),
        "valid_only_accuracy": valid_only_metrics["accuracy"],
        "valid_only_precision": valid_only_metrics["precision"],
        "valid_only_recall": valid_only_metrics["recall"],
        "valid_only_positive_f1": valid_only_metrics["positive_f1"],
        "valid_only_macro_f1": valid_only_metrics["macro_f1"],
        "valid_only_weighted_f1": valid_only_metrics["weighted_f1"],
    }
    if invalid_metric_name:
        metrics[invalid_metric_name] = int(invalid_prediction_count)
    metrics.update(_count_payload(rows, aggregation_level))
    metrics.update(_diagnostic_payload(rows))
    return metrics


def _subject_level_alias_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_subjects": metrics["num_subjects"],
        "invalid_subjects": metrics["invalid_subjects"],
        "num_valid_subject_predictions": metrics["num_valid_subject_predictions"],
        "predicted_non_depressed_subjects": metrics["predicted_non_depressed_subjects"],
        "predicted_depressed_subjects": metrics["predicted_depressed_subjects"],
        "predicted_invalid_subjects": metrics["predicted_invalid_subjects"],
        "true_non_depressed_subjects": metrics["true_non_depressed_subjects"],
        "true_depressed_subjects": metrics["true_depressed_subjects"],
    }


def aggregate_likelihood_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    score_margins: list[float] = []
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
        score_margins.append(mean_dep - mean_non)

    binary_metrics = classification_metrics(y_true, y_pred)
    metrics = {
        **binary_metrics,
        "auroc": binary_auroc(y_true, score_margins),
        "binary_strict_accuracy": binary_metrics["accuracy"],
        "binary_strict_precision": binary_metrics["precision"],
        "binary_strict_recall": binary_metrics["recall"],
        "binary_strict_positive_f1": binary_metrics["positive_f1"],
        "binary_strict_macro_f1": binary_metrics["macro_f1"],
        "binary_strict_weighted_f1": binary_metrics["weighted_f1"],
        "binary_strict_confusion_matrix": binary_metrics["confusion_matrix"],
        "prediction_backend": PREDICTION_MODE_LIKELIHOOD,
        "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_LIKELIHOOD),
        "aggregation_level": AGGREGATION_LEVEL_SUBJECT,
        "unit_label": AGGREGATION_LEVEL_SUBJECT,
        "num_units": len(subject_rows),
        "num_subjects": len(subject_rows),
        "invalid_subjects": 0,
        "num_valid_subject_predictions": len(subject_rows),
        "valid_only_accuracy": binary_metrics["accuracy"],
        "valid_only_precision": binary_metrics["precision"],
        "valid_only_recall": binary_metrics["recall"],
        "valid_only_positive_f1": binary_metrics["positive_f1"],
        "valid_only_macro_f1": binary_metrics["macro_f1"],
        "valid_only_weighted_f1": binary_metrics["weighted_f1"],
    }
    metrics.update(_count_payload(subject_rows, AGGREGATION_LEVEL_SUBJECT))
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
    count_invalid_as_wrong_vote: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["subject_id"]].append(row)

    subject_rows: list[dict[str, Any]] = []
    invalid_subjects = 0
    total_invalid_predictions = 0
    for subject_id, rows in sorted(grouped.items()):
        gold = int(rows[0]["label"])
        valid_predictions = [int(row[prediction_field]) for row in rows if row[prediction_field] in (0, 1)]
        invalid_prediction_count = sum(1 for row in rows if row[prediction_field] not in (0, 1))
        total_invalid_predictions += invalid_prediction_count
        majority_predictions = list(valid_predictions)
        if count_invalid_as_wrong_vote and invalid_prediction_count:
            majority_predictions.extend([_wrong_vote_for_gold(gold)] * invalid_prediction_count)
        if not majority_predictions:
            pred = INVALID_PREDICTION
        else:
            counts = Counter(majority_predictions)
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
        if pred not in (0, 1):
            invalid_subjects += 1
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": backend_name,
                "evaluation_protocol_name": evaluation_protocol_name(backend_name),
                "prediction": pred,
                "prediction_text": label_text_from_int(pred) if pred in (0, 1) else "INVALID",
                "num_samples": len(rows),
                valid_count_field: len(valid_predictions),
            }
        )

    metrics = _metrics_from_prediction_rows(
        subject_rows,
        backend_name=backend_name,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name=invalid_metric_name,
        invalid_prediction_count=total_invalid_predictions,
    )
    metrics["invalid_subjects"] = invalid_subjects
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
        count_invalid_as_wrong_vote=True,
    )


def aggregate_likelihood_predictions_segment_level(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segment_rows = [
        {
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "label_text": row["label_text"],
            "prediction_backend": PREDICTION_MODE_LIKELIHOOD,
            "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_LIKELIHOOD),
            "prediction": int(row["likelihood_prediction"]),
            "prediction_text": row["likelihood_prediction_text"],
            "dep_score": float(row["dep_score"]),
            "non_score": float(row["non_score"]),
        }
        for row in sample_rows
    ]
    metrics = _metrics_from_prediction_rows(
        segment_rows,
        backend_name=PREDICTION_MODE_LIKELIHOOD,
        aggregation_level=AGGREGATION_LEVEL_SEGMENT,
    )
    metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in segment_rows],
        [float(row["dep_score"]) - float(row["non_score"]) for row in segment_rows],
    )
    return segment_rows, metrics


def aggregate_generation_predictions_segment_level(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segment_rows = []
    invalid_generations = 0
    for row in sample_rows:
        prediction = int(row["parsed_prediction"]) if row["parsed_prediction"] in (0, 1) else INVALID_PREDICTION
        if prediction == INVALID_PREDICTION:
            invalid_generations += 1
        segment_rows.append(
            {
                "subject_id": row["subject_id"],
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "label_text": row["label_text"],
                "prediction_backend": PREDICTION_MODE_GENERATION,
                "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_GENERATION),
                "prediction": prediction,
                "prediction_text": label_text_from_int(prediction) if prediction in (0, 1) else "INVALID",
                "generation_text": row["generation_text"],
            }
        )
    metrics = _metrics_from_prediction_rows(
        segment_rows,
        backend_name=PREDICTION_MODE_GENERATION,
        aggregation_level=AGGREGATION_LEVEL_SEGMENT,
        invalid_metric_name="invalid_generations",
        invalid_prediction_count=invalid_generations,
    )
    return segment_rows, metrics


def aggregate_original_teacher_forced_predictions_segment_level(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segment_rows = []
    invalid_predictions = 0
    for row in sample_rows:
        prediction = (
            int(row["teacher_forced_prediction"])
            if row["teacher_forced_prediction"] in (0, 1)
            else INVALID_PREDICTION
        )
        if prediction == INVALID_PREDICTION:
            invalid_predictions += 1
        segment_rows.append(
            {
                "subject_id": row["subject_id"],
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "label_text": row["label_text"],
                "prediction_backend": PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_ORIGINAL_TEACHER_FORCED),
                "prediction": prediction,
                "prediction_text": label_text_from_int(prediction) if prediction in (0, 1) else "INVALID",
                "teacher_forced_decoded_text": row["teacher_forced_decoded_text"],
                "dep_score": float(row["dep_score"]),
                "non_score": float(row["non_score"]),
            }
        )
    metrics = _metrics_from_prediction_rows(
        segment_rows,
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SEGMENT,
        invalid_metric_name="invalid_teacher_forced_predictions",
        invalid_prediction_count=invalid_predictions,
    )
    return segment_rows, metrics


def aggregate_predictions(
    sample_rows: list[dict[str, Any]],
    *,
    mode: str,
    aggregation_level: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    mode = normalize_prediction_mode(mode)
    aggregation_level = normalize_aggregation_level(aggregation_level)

    if mode == PREDICTION_MODE_LIKELIHOOD:
        subject_rows, subject_metrics = aggregate_likelihood_predictions(sample_rows)
        if aggregation_level == AGGREGATION_LEVEL_SEGMENT:
            headline_rows, headline_metrics = aggregate_likelihood_predictions_segment_level(sample_rows)
        else:
            headline_rows, headline_metrics = subject_rows, subject_metrics
    elif mode == PREDICTION_MODE_GENERATION:
        subject_rows, subject_metrics = aggregate_generation_predictions(sample_rows)
        if aggregation_level == AGGREGATION_LEVEL_SEGMENT:
            headline_rows, headline_metrics = aggregate_generation_predictions_segment_level(sample_rows)
        else:
            headline_rows, headline_metrics = subject_rows, subject_metrics
    elif mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        subject_rows, subject_metrics = aggregate_original_teacher_forced_predictions(sample_rows)
        if aggregation_level == AGGREGATION_LEVEL_SEGMENT:
            headline_rows, headline_metrics = aggregate_original_teacher_forced_predictions_segment_level(sample_rows)
        else:
            headline_rows, headline_metrics = subject_rows, subject_metrics
    else:
        raise ValueError(f"Unsupported prediction backend: {mode}")

    if aggregation_level == AGGREGATION_LEVEL_SUBJECT:
        subject_metrics.update(_subject_level_alias_metrics(subject_metrics))

    return headline_rows, headline_metrics, subject_rows, subject_metrics
