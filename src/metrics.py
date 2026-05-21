from __future__ import annotations

from typing import Any


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _binary_confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    tn = fp = fn = tp = 0
    for gold, pred in zip(y_true, y_pred):
        if gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        elif gold == 1 and pred == 1:
            tp += 1
    return [[tn, fp], [fn, tp]]


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        raise ValueError("Cannot compute metrics on an empty target list.")
    matrix = _binary_confusion(y_true, y_pred)
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    precision_neg = _safe_divide(tn, tn + fn)
    recall_neg = _safe_divide(tn, tn + fp)
    f1_neg = _safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    precision_pos = _safe_divide(tp, tp + fp)
    recall_pos = _safe_divide(tp, tp + fn)
    f1_pos = _safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    support_neg = tn + fp
    support_pos = tp + fn
    total = support_neg + support_pos
    accuracy = _safe_divide(tp + tn, total)

    macro_precision = (precision_neg + precision_pos) / 2.0
    macro_recall = (recall_neg + recall_pos) / 2.0
    macro_f1 = (f1_neg + f1_pos) / 2.0
    weighted_precision = _safe_divide(precision_neg * support_neg + precision_pos * support_pos, total)
    weighted_recall = _safe_divide(recall_neg * support_neg + recall_pos * support_pos, total)
    weighted_f1 = _safe_divide(f1_neg * support_neg + f1_pos * support_pos, total)

    return {
        "accuracy": accuracy,
        "precision": precision_pos,
        "recall": recall_pos,
        "positive_f1": f1_pos,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "support_negative": int(support_neg),
        "support_positive": int(support_pos),
        "confusion_matrix": matrix,
    }
