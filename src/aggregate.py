from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import math
import statistics

from src.metrics import binary_auroc, classification_metrics, multiclass_macro_f1
from src.utils import (
    AGGREGATION_LEVEL_SEGMENT,
    AGGREGATION_LEVEL_SUBJECT,
    AGGREGATION_LEVEL_RESPONSE_SUBJECT,
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
TURKISH_POOLED_TEXT_PAIR_POLICY = "turkish_pooled_text_pair_mean_margin_strict_v1"
TURKISH_POOLED_TEXT_CONDITIONS = ("pos_only_t17", "negative_only_t17")
TURKISH_POOLED_AUDIO_AGGREGATION_POLICY = "turkish_pooled_audio_response_subject_mean_score_margin_v1"


def _strict_binary_prediction(gold: int, pred: int) -> int:
    if pred in (0, 1):
        return pred
    return 1 - int(gold)


def _wrong_vote_for_gold(gold: int) -> int:
    return 1 - int(gold)


def _unit_suffix(aggregation_level: str) -> str:
    if aggregation_level in {AGGREGATION_LEVEL_SUBJECT, AGGREGATION_LEVEL_RESPONSE_SUBJECT}:
        return "subjects"
    return "segments"


def _majority_with_margin(
    rows: list[dict[str, Any]],
    *,
    prediction_field: str,
    gold: int,
    invalid_as_wrong: bool,
) -> tuple[int, int, float]:
    valid = [int(row[prediction_field]) for row in rows if row.get(prediction_field) in (0, 1)]
    invalid_count = len(rows) - len(valid)
    votes = list(valid)
    if invalid_as_wrong:
        votes.extend([_wrong_vote_for_gold(gold)] * invalid_count)
    margin = sum(
        float(row.get("score_margin", row.get("dep_score", 0.0) - row.get("non_score", 0.0)))
        for row in rows
    )
    if not votes:
        return INVALID_PREDICTION, invalid_count, margin
    counts = Counter(votes)
    if counts[1] > counts[0]:
        return 1, invalid_count, margin
    if counts[0] > counts[1]:
        return 0, invalid_count, margin
    if margin > 0:
        return 1, invalid_count, margin
    if margin < 0:
        return 0, invalid_count, margin
    return INVALID_PREDICTION, invalid_count, margin


def aggregate_response_subject_predictions(
    sample_rows: list[dict[str, Any]],
    *,
    prediction_field: str,
    backend_name: str,
    invalid_as_wrong: bool = True,
    score_average: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Predeclared sample -> response -> subject hierarchy.

    Sample counts never directly influence the subject decision: each response
    is reduced first and each response then contributes exactly one subject
    value. ``score_average`` uses equal-weight score margins at both levels;
    the default retains the historical majority-vote behavior.
    """
    grouped_responses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        rid = str(row.get("response_id", "")).strip()
        if not rid:
            raise ValueError("response_subject aggregation requires response_id on every sample.")
        grouped_responses[rid].append(row)

    response_rows: list[dict[str, Any]] = []
    invalid_segments = 0
    for rid, rows in sorted(grouped_responses.items()):
        gold = int(rows[0]["label"])
        if len({str(row["subject_id"]) for row in rows}) != 1:
            raise ValueError(f"Response {rid} spans multiple subjects.")
        if len({int(row["label"]) for row in rows}) != 1:
            raise ValueError(f"Response {rid} has inconsistent labels.")
        declared = {
            int(row["num_segments"])
            for row in rows
            if row.get("num_segments") not in (None, "")
        }
        if declared and (len(declared) != 1 or next(iter(declared)) != len(rows)):
            raise ValueError(
                f"Response {rid} segment metadata disagrees with observed rows."
            )
        pred, invalid_count, margin_sum = _majority_with_margin(
            rows,
            prediction_field=prediction_field,
            gold=gold,
            invalid_as_wrong=invalid_as_wrong,
        )
        if score_average:
            pred = (
                1
                if margin_sum > 0
                else 0 if margin_sum < 0 else INVALID_PREDICTION
            )
        invalid_segments += invalid_count
        response_rows.append(
            {
                "subject_id": rows[0]["subject_id"],
                "response_id": rid,
                "prompt_id": rows[0].get("prompt_id", ""),
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": backend_name,
                "evaluation_protocol_name": evaluation_protocol_name(backend_name),
                "prediction": pred,
                "prediction_text": label_text_from_int(pred) if pred in (0, 1) else "INVALID",
                "num_segments": len(rows),
                "num_valid_segments": len(rows) - invalid_count,
                "invalid_segments": invalid_count,
                "score_margin_sum": margin_sum,
                "score_margin": margin_sum / len(rows),
                "dep_score": sum(float(row.get("dep_score", 0.0)) for row in rows) / len(rows),
                "non_score": sum(float(row.get("non_score", 0.0)) for row in rows) / len(rows),
            }
        )

    response_metrics = _metrics_from_prediction_rows(
        response_rows,
        backend_name=backend_name,
        aggregation_level=AGGREGATION_LEVEL_SEGMENT,
        invalid_metric_name="invalid_segment_predictions",
        invalid_prediction_count=invalid_segments,
    )
    response_metrics["aggregation_level"] = "response"
    response_metrics["unit_label"] = "response"
    response_metrics["num_responses"] = len(response_rows)
    response_metrics["invalid_responses"] = sum(
        int(row["prediction"]) not in (0, 1) for row in response_rows
    )
    response_metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in response_rows],
        [float(row["score_margin"]) for row in response_rows],
    )

    grouped_subjects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        grouped_subjects[str(row["subject_id"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    invalid_responses = 0
    for subject_id, rows in sorted(grouped_subjects.items()):
        gold = int(rows[0]["label"])
        pred, invalid_count, margin_sum = _majority_with_margin(
            rows,
            prediction_field="prediction",
            gold=gold,
            invalid_as_wrong=invalid_as_wrong,
        )
        if score_average:
            pred = (
                1
                if margin_sum > 0
                else 0 if margin_sum < 0 else INVALID_PREDICTION
            )
        invalid_responses += invalid_count
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": backend_name,
                "evaluation_protocol_name": evaluation_protocol_name(backend_name),
                "prediction": pred,
                "prediction_text": label_text_from_int(pred) if pred in (0, 1) else "INVALID",
                "num_responses": len(rows),
                "num_valid_responses": len(rows) - invalid_count,
                "invalid_responses": invalid_count,
                "score_margin_sum": margin_sum,
                "score_margin": margin_sum / len(rows),
                "dep_score": sum(float(row["dep_score"]) for row in rows) / len(rows),
                "non_score": sum(float(row["non_score"]) for row in rows) / len(rows),
            }
        )
    subject_metrics = _metrics_from_prediction_rows(
        subject_rows,
        backend_name=backend_name,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name="invalid_response_predictions",
        invalid_prediction_count=invalid_responses,
    )
    subject_metrics["aggregation_level"] = AGGREGATION_LEVEL_RESPONSE_SUBJECT
    subject_metrics["num_valid_response_subject_predictions"] = subject_metrics[
        "num_valid_subject_predictions"
    ]
    subject_metrics["invalid_segments"] = invalid_segments
    subject_metrics["invalid_responses"] = invalid_responses
    subject_metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in subject_rows],
        [float(row["score_margin"]) for row in subject_rows],
    )
    return response_rows, response_metrics, subject_rows, subject_metrics


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
        "predicted_positive_rate": (
            sum(y_pred) / len(y_pred) if y_pred else 0.0
        ),
    }
    metrics.update(_count_payload(subject_rows, AGGREGATION_LEVEL_SUBJECT))
    metrics.update(_diagnostic_payload(subject_rows))
    return subject_rows, metrics


def aggregate_mean_probability_predictions(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["subject_id"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for subject {subject_id}.")
        probability = sum(float(row["probability"]) for row in rows) / len(rows)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": next(iter(labels)),
                "prediction": int(probability >= 0.5),
                "probability": probability,
                "num_samples": len(rows),
                "aggregation_method": "mean_depressed_probability_threshold_0.5",
            }
        )
    y_true = [int(row["label"]) for row in subject_rows]
    y_pred = [int(row["prediction"]) for row in subject_rows]
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(
        {
            "auroc": binary_auroc(
                y_true, [float(row["probability"]) for row in subject_rows]
            ),
            "num_subjects": len(subject_rows),
            "predicted_positive_rate": sum(y_pred) / len(y_pred) if y_pred else 0.0,
            "aggregation_method": "mean_depressed_probability_threshold_0.5",
        }
    )
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
        subject_row = {
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
        if tie_break_positive_score_field and tie_break_negative_score_field:
            mean_positive = sum(
                float(row.get(tie_break_positive_score_field, 0.0)) for row in rows
            ) / len(rows)
            mean_negative = sum(
                float(row.get(tie_break_negative_score_field, 0.0)) for row in rows
            ) / len(rows)
            subject_row.update(
                {
                    "dep_score": mean_positive,
                    "non_score": mean_negative,
                    "score_margin": mean_positive - mean_negative,
                }
            )
        subject_rows.append(subject_row)

    metrics = _metrics_from_prediction_rows(
        subject_rows,
        backend_name=backend_name,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name=invalid_metric_name,
        invalid_prediction_count=total_invalid_predictions,
    )
    metrics["invalid_subjects"] = invalid_subjects
    if tie_break_positive_score_field and tie_break_negative_score_field:
        metrics["auroc"] = binary_auroc(
            [int(row["label"]) for row in subject_rows],
            [float(row["score_margin"]) for row in subject_rows],
        )
    return subject_rows, metrics


def aggregate_generation_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _aggregate_majority_vote_predictions(
        sample_rows,
        prediction_field="parsed_prediction",
        backend_name=PREDICTION_MODE_GENERATION,
        invalid_metric_name="invalid_generations",
        valid_count_field="generation_num_valid_predictions",
    )


PACKED30_PROTOCOL_ID = "daic_participant_speech_packed30_v1"


def aggregate_mean_score_margin_subject_predictions(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Locked v1 mean teacher-forced score-margin subject aggregation.

    The arithmetic mean of per-chunk ``dep_score - non_score`` is computed for
    every subject. Depressed when the mean margin is positive, Non-depressed
    when negative, and INVALID on an exact finite zero. Any non-finite
    candidate score or margin MUST fail evaluation. AUROC is intentionally not
    reported for the hard teacher-forced verdict.
    """
    if not sample_rows:
        raise ValueError("Mean score-margin aggregation requires at least one sample row.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["subject_id"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        gold = next(iter(labels))
        margins: list[float] = []
        for row in rows:
            dep_score = float(row["dep_score"])
            non_score = float(row["non_score"])
            if not (math.isfinite(dep_score) and math.isfinite(non_score)):
                raise ValueError(
                    f"Non-finite teacher-forced candidate score for subject_id={subject_id} "
                    f"sample_id={row.get('sample_id', '')}: dep_score={dep_score} non_score={non_score}"
                )
            margin = dep_score - non_score
            if not math.isfinite(margin):
                raise ValueError(
                    f"Non-finite teacher-forced margin for subject_id={subject_id} "
                    f"sample_id={row.get('sample_id', '')}: {margin}"
                )
            margins.append(margin)
        mean_margin = sum(margins) / len(margins)
        if mean_margin > 0.0:
            pred = 1
        elif mean_margin < 0.0:
            pred = 0
        else:
            pred = INVALID_PREDICTION
        mean_dep = sum(float(row["dep_score"]) for row in rows) / len(rows)
        mean_non = sum(float(row["non_score"]) for row in rows) / len(rows)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                "evaluation_protocol_name": evaluation_protocol_name(
                    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
                ),
                "prediction": pred,
                "prediction_text": (
                    label_text_from_int(pred) if pred in (0, 1) else "INVALID"
                ),
                "dep_score": mean_dep,
                "non_score": mean_non,
                "score_margin": mean_margin,
                "num_samples": len(rows),
                "aggregation_method": "mean_teacher_forced_score_margin",
            }
        )
    metrics = _metrics_from_prediction_rows(
        subject_rows,
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    metrics["aggregation_method"] = "mean_teacher_forced_score_margin"
    return subject_rows, metrics


def aggregate_original_teacher_forced_predictions(sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The DAIC chunking protocol predeclares mean score-margin aggregation.
    # This path also produces a valid prediction when argmax-decoded label
    # tokens are malformed, because the two teacher-forced candidate scores are
    # the scientifically authoritative signal for this experiment.
    if sample_rows and all(
        str(row.get("protocol_id", "")) == PACKED30_PROTOCOL_ID for row in sample_rows
    ):
        subject_rows, metrics = aggregate_mean_score_margin_subject_predictions(sample_rows)
        return subject_rows, metrics
    if sample_rows and all(
        str(row.get("subject_score_aggregation", "")).lower() == "mean_score"
        for row in sample_rows
    ):
        rows = [dict(row) for row in sample_rows]
        for row in rows:
            row["likelihood_prediction"] = int(
                float(row["dep_score"]) - float(row["non_score"]) > 0.0
            )
        subject_rows, metrics = aggregate_likelihood_predictions(rows)
        for row in subject_rows:
            row["prediction_backend"] = PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
            row["evaluation_protocol_name"] = evaluation_protocol_name(
                PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
            )
            row["aggregation_method"] = "mean_teacher_forced_score_margin"
        metrics["prediction_backend"] = PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
        metrics["evaluation_protocol_name"] = evaluation_protocol_name(
            PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
        )
        metrics["aggregation_method"] = "mean_teacher_forced_score_margin"
        return subject_rows, metrics
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


def _declared_pair_policy(row: dict[str, Any]) -> str:
    value = row.get("aggregation_policy")
    if value in (None, ""):
        value = row.get("subject_score_aggregation")
    return str(value or "").strip()


def _require_pooled_pair_rows(
    sample_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not sample_rows:
        raise ValueError("Turkish pooled text pair aggregation requires sample rows.")
    policies = {_declared_pair_policy(row) for row in sample_rows}
    if policies != {TURKISH_POOLED_TEXT_PAIR_POLICY}:
        raise ValueError(
            "Turkish pooled text pair aggregation requires every row to declare "
            f"{TURKISH_POOLED_TEXT_PAIR_POLICY!r}; found {sorted(policies)}."
        )
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sample_rows:
        subject_id = str(row.get("subject_id", "")).strip()
        if not subject_id:
            raise ValueError("Turkish pooled text pair rows require subject_id.")
        condition = str(row.get("question_condition", "")).strip()
        if condition not in TURKISH_POOLED_TEXT_CONDITIONS:
            raise ValueError(
                f"Unexpected Turkish pooled question condition {condition!r} for subject {subject_id!r}."
            )
        if condition in grouped[subject_id]:
            raise ValueError(
                f"Duplicate Turkish pooled question condition {condition!r} for subject {subject_id!r}."
            )
        grouped[subject_id][condition] = row
    expected = set(TURKISH_POOLED_TEXT_CONDITIONS)
    for subject_id, rows_by_condition in grouped.items():
        if set(rows_by_condition) != expected or len(rows_by_condition) != 2:
            raise ValueError(
                f"Turkish pooled text pair for subject {subject_id!r} must contain "
                f"exactly {sorted(expected)}; found {sorted(rows_by_condition)}."
            )
        labels = {int(row.get("label", -1)) for row in rows_by_condition.values()}
        if len(labels) != 1 or labels - {0, 1}:
            raise ValueError(
                f"Turkish pooled text pair for subject {subject_id!r} has inconsistent labels."
            )
    return grouped


def _pair_prediction(margin: float, *, decoded_valid: bool = True) -> int:
    if not decoded_valid:
        return INVALID_PREDICTION
    if margin > 0.0:
        return 1
    if margin < 0.0:
        return 0
    return INVALID_PREDICTION


def aggregate_turkish_pooled_text_teacher_forced_predictions(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the locked two-condition teacher-forced pair rule."""
    grouped = _require_pooled_pair_rows(sample_rows)
    subject_rows: list[dict[str, Any]] = []
    invalid_components = 0
    for subject_id, rows_by_condition in sorted(grouped.items()):
        components: dict[str, dict[str, Any]] = {}
        for condition in TURKISH_POOLED_TEXT_CONDITIONS:
            row = rows_by_condition[condition]
            dep_score = float(row["dep_score"])
            non_score = float(row["non_score"])
            if not (math.isfinite(dep_score) and math.isfinite(non_score)):
                raise ValueError(
                    f"Non-finite Turkish pooled teacher-forced score for subject {subject_id!r} "
                    f"condition {condition!r}."
                )
            margin = dep_score - non_score
            if not math.isfinite(margin):
                raise ValueError(
                    f"Non-finite Turkish pooled teacher-forced margin for subject {subject_id!r} "
                    f"condition {condition!r}."
                )
            valid_value = row.get("teacher_forced_valid")
            if not isinstance(valid_value, bool):
                raise ValueError(
                    "Turkish pooled teacher-forced pair rows require a boolean "
                    "teacher_forced_valid field for both conditions."
                )
            decoded = row.get("teacher_forced_prediction")
            decoded_prediction = int(decoded) if decoded in (0, 1) else INVALID_PREDICTION
            component_valid = bool(valid_value) and decoded_prediction in (0, 1)
            invalid_components += int(not component_valid)
            components[condition] = {
                "condition": condition,
                "margin": margin,
                "dep_score": dep_score,
                "non_score": non_score,
                "decoded_prediction": decoded_prediction,
                "decoded_prediction_text": (
                    label_text_from_int(decoded_prediction)
                    if decoded_prediction in (0, 1)
                    else "INVALID"
                ),
                "decoded_valid": component_valid,
                "sample_id": str(row.get("sample_id", "")),
            }
        positive = components["pos_only_t17"]
        negative = components["negative_only_t17"]
        pair_margin = (positive["margin"] + negative["margin"]) / 2.0
        if not math.isfinite(pair_margin):
            raise ValueError(f"Non-finite Turkish pooled pair margin for subject {subject_id!r}.")
        gold = int(rows_by_condition["pos_only_t17"]["label"])
        prediction = _pair_prediction(
            pair_margin,
            decoded_valid=bool(positive["decoded_valid"] and negative["decoded_valid"]),
        )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                "evaluation_protocol_name": evaluation_protocol_name(
                    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED
                ),
                "prediction": prediction,
                "prediction_text": label_text_from_int(prediction) if prediction in (0, 1) else "INVALID",
                "positive_condition": "pos_only_t17",
                "negative_condition": "negative_only_t17",
                "question_conditions": list(TURKISH_POOLED_TEXT_CONDITIONS),
                "positive_margin": positive["margin"],
                "negative_margin": negative["margin"],
                "positive_condition_margin": positive["margin"],
                "negative_condition_margin": negative["margin"],
                "pair_margin": pair_margin,
                "score_margin": pair_margin,
                "positive_decoded_prediction": positive["decoded_prediction"],
                "negative_decoded_prediction": negative["decoded_prediction"],
                "positive_decoded_prediction_text": positive["decoded_prediction_text"],
                "negative_decoded_prediction_text": negative["decoded_prediction_text"],
                "positive_decoded_valid": positive["decoded_valid"],
                "negative_decoded_valid": negative["decoded_valid"],
                "positive_sample_id": positive["sample_id"],
                "negative_sample_id": negative["sample_id"],
                "num_samples": 2,
                "invalid_component_predictions": int(
                    not positive["decoded_valid"]
                ) + int(not negative["decoded_valid"]),
                "aggregation_method": TURKISH_POOLED_TEXT_PAIR_POLICY,
            }
        )
    metrics = _metrics_from_prediction_rows(
        subject_rows,
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name="invalid_paired_subject_predictions",
        invalid_prediction_count=sum(
            int(row["prediction"]) not in (0, 1) for row in subject_rows
        ),
    )
    metrics.update(
        {
            "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
            "aggregation_method": TURKISH_POOLED_TEXT_PAIR_POLICY,
            "invalid_component_predictions": int(invalid_components),
            "invalid_paired_subject_predictions": sum(
                int(row["prediction"]) not in (0, 1) for row in subject_rows
            ),
        }
    )
    return subject_rows, metrics


def aggregate_turkish_pooled_text_condition_predictions(
    sample_rows: list[dict[str, Any]],
    condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score one pooled text condition without applying pair aggregation."""
    if condition not in TURKISH_POOLED_TEXT_CONDITIONS:
        raise ValueError(f"Unsupported Turkish pooled question condition: {condition!r}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        if _declared_pair_policy(row) != TURKISH_POOLED_TEXT_PAIR_POLICY:
            raise ValueError("Pooled condition rows must declare the pooled pair policy.")
        if str(row.get("question_condition", "")).strip() != condition:
            raise ValueError("Condition breakdown received a row for another condition.")
        grouped[str(row["subject_id"])].append(row)
    rows: list[dict[str, Any]] = []
    invalid = 0
    for subject_id, components in sorted(grouped.items()):
        if len(components) != 1:
            raise ValueError(f"Condition {condition!r} has duplicate rows for subject {subject_id!r}.")
        source = components[0]
        label = int(source["label"])
        valid_value = source.get("teacher_forced_valid")
        if not isinstance(valid_value, bool):
            raise ValueError("Condition rows require boolean teacher_forced_valid.")
        prediction = int(source["teacher_forced_prediction"]) if source.get("teacher_forced_prediction") in (0, 1) else INVALID_PREDICTION
        if not valid_value or prediction not in (0, 1):
            prediction = INVALID_PREDICTION
        invalid += int(prediction not in (0, 1))
        dep_score = float(source["dep_score"])
        non_score = float(source["non_score"])
        if not (math.isfinite(dep_score) and math.isfinite(non_score)):
            raise ValueError(f"Non-finite condition score for subject {subject_id!r}.")
        margin = dep_score - non_score
        rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "label_text": label_text_from_int(label),
                "prediction_backend": PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                "evaluation_protocol_name": evaluation_protocol_name(PREDICTION_MODE_ORIGINAL_TEACHER_FORCED),
                "prediction": prediction,
                "prediction_text": label_text_from_int(prediction) if prediction in (0, 1) else "INVALID",
                "question_condition": condition,
                "dep_score": dep_score,
                "non_score": non_score,
                "score_margin": margin,
                "teacher_forced_decoded_text": source.get("teacher_forced_decoded_text", ""),
                "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
                "aggregation_method": "condition_specific_decoded_prediction",
                "num_samples": 1,
            }
        )
    metrics = _metrics_from_prediction_rows(
        rows,
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name="invalid_condition_predictions",
        invalid_prediction_count=invalid,
    )
    metrics.update(
        {
            "question_condition": condition,
            "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
            "aggregation_method": "condition_specific_decoded_prediction",
        }
    )
    return rows, metrics


def aggregate_turkish_pooled_text_classifier_predictions(
    sample_rows: list[dict[str, Any]],
    *,
    prediction_backend: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the locked two-condition classifier probability pair rule."""
    grouped = _require_pooled_pair_rows(sample_rows)
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows_by_condition in sorted(grouped.items()):
        components: dict[str, dict[str, Any]] = {}
        for condition in TURKISH_POOLED_TEXT_CONDITIONS:
            row = rows_by_condition[condition]
            probability = float(row["probability"])
            if not math.isfinite(probability):
                raise ValueError(
                    f"Non-finite Turkish pooled classifier probability for subject {subject_id!r} "
                    f"condition {condition!r}."
                )
            margin = probability - 0.5
            if not math.isfinite(margin):
                raise ValueError(f"Non-finite Turkish pooled classifier margin for subject {subject_id!r}.")
            components[condition] = {
                "condition": condition,
                "probability": probability,
                "margin": margin,
                "prediction": int(row.get("predicted_class", int(probability >= 0.5))),
                "sample_id": str(row.get("sample_id", "")),
            }
        positive = components["pos_only_t17"]
        negative = components["negative_only_t17"]
        pair_margin = (positive["margin"] + negative["margin"]) / 2.0
        gold = int(rows_by_condition["pos_only_t17"]["label"])
        prediction = _pair_prediction(pair_margin)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": gold,
                "label_text": label_text_from_int(gold),
                "prediction_backend": prediction_backend,
                "evaluation_protocol_name": "turkish_pooled_text_pair_classifier",
                "prediction": prediction,
                "prediction_text": label_text_from_int(prediction) if prediction in (0, 1) else "INVALID",
                "positive_condition": "pos_only_t17",
                "negative_condition": "negative_only_t17",
                "question_conditions": list(TURKISH_POOLED_TEXT_CONDITIONS),
                "positive_probability": positive["probability"],
                "negative_probability": negative["probability"],
                "positive_margin": positive["margin"],
                "negative_margin": negative["margin"],
                "positive_condition_margin": positive["margin"],
                "negative_condition_margin": negative["margin"],
                "pair_margin": pair_margin,
                "score_margin": pair_margin,
                "positive_component_prediction": positive["prediction"],
                "negative_component_prediction": negative["prediction"],
                "positive_sample_id": positive["sample_id"],
                "negative_sample_id": negative["sample_id"],
                "num_samples": 2,
                "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
                "aggregation_method": TURKISH_POOLED_TEXT_PAIR_POLICY,
            }
        )
    metrics = _metrics_from_prediction_rows(
        subject_rows,
        # The shared metric helper also resolves the generic evaluation
        # protocol name, which accepts the three model-evaluation modes only.
        # The pooled classifier-specific backend is written explicitly below.
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
        invalid_metric_name="invalid_paired_subject_predictions",
        invalid_prediction_count=sum(
            int(row["prediction"]) not in (0, 1) for row in subject_rows
        ),
    )
    metrics.update(
        {
            "prediction_backend": prediction_backend,
            "evaluation_protocol_name": "turkish_pooled_text_pair_classifier",
            "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
            "aggregation_method": TURKISH_POOLED_TEXT_PAIR_POLICY,
        }
    )
    metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in subject_rows],
        [float(row["pair_margin"]) for row in subject_rows],
    )
    return subject_rows, metrics


def aggregate_margin_predictions(
    sample_rows: list[dict[str, Any]], method: str = "mean_score"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a declared chunk/bundle aggregation with one output per subject."""
    supported = {"mean_score", "median_score", "trimmed_mean_10", "majority_margin_tiebreak", "max_score"}
    if method not in supported:
        raise ValueError(f"Unsupported subject score aggregation {method!r}.")
    if not sample_rows:
        raise ValueError("Subject score aggregation requires at least one sample row.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["subject_id"])].append(row)
    subjects: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        margins = [float(row["dep_score"]) - float(row["non_score"]) for row in rows]
        mean_margin = sum(margins) / len(margins)
        prediction = int(mean_margin > 0.0)
        if method == "mean_score":
            margin = mean_margin
        elif method == "median_score":
            margin = float(statistics.median(margins))
        elif method == "trimmed_mean_10":
            trim = math.floor(0.1 * len(margins))
            ordered = sorted(margins)
            kept = ordered[trim:len(ordered) - trim] if trim else ordered
            margin = sum(kept) / len(kept)
        elif method == "max_score":
            margin = max(margins)
        else:
            votes = Counter(int(value > 0.0) for value in margins)
            # The vote decides the binary prediction, while the continuous
            # score remains the mean margin so AUROC and downstream reports do
            # not invent a magnitude of exactly +/-1.
            margin = mean_margin
            prediction = (
                1 if votes[1] > votes[0]
                else 0 if votes[0] > votes[1]
                else int(mean_margin > 0.0)
            )
        if method != "majority_margin_tiebreak":
            prediction = int(margin > 0.0)
        subjects.append({
            "subject_id": subject_id, "label": next(iter(labels)),
            "prediction": prediction, "score_margin": margin,
            "num_samples": len(rows), "aggregation_method": method,
        })
    metrics = _metrics_from_prediction_rows(
        subjects, backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    metrics["aggregation_method"] = method
    metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in subjects],
        [float(row["score_margin"]) for row in subjects],
    )
    return subjects, metrics


def aggregate_binary_classifier_predictions(
    sample_rows: list[dict[str, Any]],
    prediction_backend: str = "qwen_hidden_classifier",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the baseline majority vote and score-margin tie rule to classifiers.

    ``prediction_backend`` defaults to ``qwen_hidden_classifier`` so existing
    Qwen callers and tests are unchanged; the Gemma fixed-head campaign passes
    its exact backend value explicitly (``gemma4_hidden_logreg_raw`` /
    ``gemma4_hidden_xgb_raw``).
    """
    declared_policies = {_declared_pair_policy(row) for row in sample_rows}
    if TURKISH_POOLED_TEXT_PAIR_POLICY in declared_policies:
        if declared_policies != {TURKISH_POOLED_TEXT_PAIR_POLICY}:
            raise ValueError(
                "Pooled text classifier rows may not be mixed with another "
                f"aggregation policy: {sorted(declared_policies)}."
            )
        return aggregate_turkish_pooled_text_classifier_predictions(
            sample_rows,
            prediction_backend=prediction_backend,
        )
    if sample_rows and all(
        str(row.get("dataset", "")).lower() == "turkish"
        and str(row.get("dataset_variant", "")).strip() == "pooled_t17"
        and str(row.get("modality", "")).strip() in {"audio_only", "audio_text"}
        for row in sample_rows
    ):
        normalized = []
        for row in sample_rows:
            probability = float(row["probability"])
            normalized.append(
                {
                    **row,
                    "classifier_prediction": int(row["predicted_class"]),
                    "dep_score": probability,
                    "non_score": 1.0 - probability,
                    "score_margin": 2.0 * probability - 1.0,
                }
            )
        if not all(str(row.get("response_id", "")).strip() for row in normalized):
            raise ValueError(
                "Pooled Turkish audio classifier aggregation requires a response/source-unit id."
            )
        response_rows, _, subject_rows, metrics = aggregate_response_subject_predictions(
            normalized,
            prediction_field="classifier_prediction",
            backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
            invalid_as_wrong=False,
            score_average=True,
        )
        for row in subject_rows:
            row.update(
                {
                    "prediction_backend": prediction_backend,
                    "evaluation_protocol_name": "turkish_pooled_audio_hidden_response_subject",
                    "aggregation_policy": TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
                    "aggregation_method": TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
                    "response_count": len([item for item in response_rows if str(item["subject_id"]) == str(row["subject_id"])]),
                }
            )
        metrics.update(
            {
                "prediction_backend": prediction_backend,
                "evaluation_protocol_name": "turkish_pooled_audio_hidden_response_subject",
                "aggregation_policy": TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
                "aggregation_method": TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
            }
        )
        return subject_rows, metrics
    if sample_rows and all(
        str(row.get("classifier_aggregation", "")).lower()
        == "mean_depressed_probability_threshold_0_5"
        for row in sample_rows
    ):
        subject_rows, metrics = aggregate_mean_probability_predictions(sample_rows)
        for row in subject_rows:
            row.update(
                {
                    "prediction_backend": prediction_backend,
                    "evaluation_protocol_name": "daic_participant_packed30_hidden_mean_probability",
                    "aggregated_prediction": int(row["prediction"]),
                    "aggregation_method": "mean_depressed_probability_threshold_0_5",
                }
            )
        metrics.update(
            {
                "prediction_backend": prediction_backend,
                "evaluation_protocol_name": "daic_participant_packed30_hidden_mean_probability",
                "aggregation_method": "mean_depressed_probability_threshold_0_5",
                "predicted_positive_rate": (
                    sum(int(row["prediction"]) == 1 for row in subject_rows)
                    / len(subject_rows)
                    if subject_rows
                    else 0.0
                ),
            }
        )
        return subject_rows, metrics
    normalized = []
    for row in sample_rows:
        probability = float(row["probability"])
        normalized.append(
            {
                **row,
                "classifier_prediction": int(row["predicted_class"]),
                "dep_score": probability,
                "non_score": 1.0 - probability,
            }
        )
    if normalized and all(str(row.get("response_id", "")).strip() for row in normalized):
        response_rows, _, subject_rows, metrics = aggregate_response_subject_predictions(
            normalized,
            prediction_field="classifier_prediction",
            backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
            invalid_as_wrong=False,
        )
        responses_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        samples_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in response_rows:
            responses_by_subject[str(row["subject_id"])].append(row)
        for row in normalized:
            samples_by_subject[str(row["subject_id"])].append(row)
        for subject_row in subject_rows:
            subject_id = str(subject_row["subject_id"])
            samples = samples_by_subject[subject_id]
            responses = responses_by_subject[subject_id]
            subject_row.update(
                {
                    "prediction_backend": prediction_backend,
                    "evaluation_protocol_name": "d3tec_hidden_response_subject",
                    "sample_count": len(samples),
                    "response_count": len(responses),
                    "response_ids": [str(row["response_id"]) for row in responses],
                    "response_predictions": [int(row["prediction"]) for row in responses],
                    "response_probabilities": [float(row["dep_score"]) for row in responses],
                    "aggregated_prediction": int(subject_row["prediction"]),
                    "aggregation_method": (
                        "segment_majority_probability_margin_tie_break_then_"
                        "equal_response_majority_probability_margin_tie_break"
                    ),
                    "probability": float(subject_row["dep_score"]),
                }
            )
        metrics["prediction_backend"] = prediction_backend
        metrics["evaluation_protocol_name"] = "d3tec_hidden_response_subject"
        metrics["aggregation_method"] = (
            "segment_majority_probability_margin_tie_break_then_"
            "equal_response_majority_probability_margin_tie_break"
        )
        metrics["predicted_positive_rate"] = (
            sum(int(row["prediction"]) == 1 for row in subject_rows) / len(subject_rows)
            if subject_rows
            else 0.0
        )
        return subject_rows, metrics
    subject_rows, metrics = _aggregate_majority_vote_predictions(
        normalized,
        prediction_field="classifier_prediction",
        # Reuse the teacher-forced majority-vote path because its tie rule is
        # exactly the required summed positive-versus-negative score margin.
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        invalid_metric_name="invalid_classifier_predictions",
        valid_count_field="classifier_num_valid_predictions",
        tie_break_positive_score_field="dep_score",
        tie_break_negative_score_field="non_score",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["subject_id"])].append(row)
    for subject_row in subject_rows:
        rows = grouped[str(subject_row["subject_id"])]
        probabilities = [float(row["probability"]) for row in rows]
        subject_row.update(
            {
                "prediction_backend": prediction_backend,
                "evaluation_protocol_name": "qwen_hidden_majority_vote",
                "sample_count": len(rows),
                "sample_predictions": [int(row["predicted_class"]) for row in rows],
                "sample_probabilities": probabilities,
                "aggregated_prediction": int(subject_row["prediction"]),
                "aggregation_method": "majority_vote_probability_margin_tie_break",
                "probability": float(sum(probabilities) / len(probabilities)),
            }
        )
    metrics["prediction_backend"] = prediction_backend
    metrics["evaluation_protocol_name"] = "qwen_hidden_majority_vote"
    metrics["auroc"] = binary_auroc(
        [int(row["label"]) for row in subject_rows],
        [float(row["probability"]) for row in subject_rows],
    )
    metrics["predicted_positive_rate"] = (
        sum(int(row["prediction"]) == 1 for row in subject_rows) / len(subject_rows)
        if subject_rows
        else 0.0
    )
    return subject_rows, metrics


def aggregate_binary_classifier_response_rows(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return D3TEC response-level classifier predictions and metrics."""
    normalized = []
    for row in sample_rows:
        probability = float(row["probability"])
        normalized.append(
            {
                **row,
                "classifier_prediction": int(row["predicted_class"]),
                "dep_score": probability,
                "non_score": 1.0 - probability,
            }
        )
    response_rows, response_metrics, _, _ = aggregate_response_subject_predictions(
        normalized,
        prediction_field="classifier_prediction",
        backend_name=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        invalid_as_wrong=False,
    )
    for row in response_rows:
        row["prediction_backend"] = "qwen_hidden_classifier"
        row["evaluation_protocol_name"] = "d3tec_hidden_response"
    response_metrics["prediction_backend"] = "qwen_hidden_classifier"
    response_metrics["evaluation_protocol_name"] = "d3tec_hidden_response"
    return response_rows, response_metrics


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
    if aggregation_level == AGGREGATION_LEVEL_RESPONSE_SUBJECT:
        prediction_field = {
            PREDICTION_MODE_LIKELIHOOD: "likelihood_prediction",
            PREDICTION_MODE_GENERATION: "parsed_prediction",
            PREDICTION_MODE_ORIGINAL_TEACHER_FORCED: "teacher_forced_prediction",
        }[mode]
        _, _, subject_rows, subject_metrics = aggregate_response_subject_predictions(
            sample_rows,
            prediction_field=prediction_field,
            backend_name=mode,
            invalid_as_wrong=mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        )
        return subject_rows, subject_metrics, subject_rows, subject_metrics

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
        declared_policies = {_declared_pair_policy(row) for row in sample_rows}
        if TURKISH_POOLED_TEXT_PAIR_POLICY in declared_policies:
            if declared_policies != {TURKISH_POOLED_TEXT_PAIR_POLICY}:
                raise ValueError(
                    "Pooled text teacher-forced rows may not be mixed with another "
                    f"aggregation policy: {sorted(declared_policies)}."
                )
            subject_rows, subject_metrics = aggregate_turkish_pooled_text_teacher_forced_predictions(
                sample_rows
            )
        else:
            subject_rows, subject_metrics = aggregate_original_teacher_forced_predictions(sample_rows)
        if aggregation_level == AGGREGATION_LEVEL_SEGMENT:
            if TURKISH_POOLED_TEXT_PAIR_POLICY in declared_policies:
                headline_rows, headline_metrics = subject_rows, subject_metrics
            else:
                headline_rows, headline_metrics = aggregate_original_teacher_forced_predictions_segment_level(sample_rows)
        else:
            headline_rows, headline_metrics = subject_rows, subject_metrics
    else:
        raise ValueError(f"Unsupported prediction backend: {mode}")

    if aggregation_level == AGGREGATION_LEVEL_SUBJECT:
        subject_metrics.update(_subject_level_alias_metrics(subject_metrics))

    return headline_rows, headline_metrics, subject_rows, subject_metrics
