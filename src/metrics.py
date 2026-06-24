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


def binary_auroc(y_true: list[int], scores: list[float]) -> float:
    if len(y_true) != len(scores):
        raise ValueError("AUROC targets and scores must have the same length.")
    positives = sum(int(label) == 1 for label in y_true)
    negatives = sum(int(label) == 0 for label in y_true)
    if positives == 0 or negatives == 0:
        return 0.0

    ordered_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered_indices):
        tie_end = index
        score = float(scores[ordered_indices[index]])
        while (
            tie_end + 1 < len(ordered_indices)
            and float(scores[ordered_indices[tie_end + 1]]) == score
        ):
            tie_end += 1
        average_rank = (index + tie_end) / 2.0 + 1.0
        for rank_index in range(index, tie_end + 1):
            ranks[ordered_indices[rank_index]] = average_rank
        index = tie_end + 1

    positive_rank_sum = sum(
        ranks[index] for index, label in enumerate(y_true) if int(label) == 1
    )
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


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


def multiclass_confusion(y_true: list[int], y_pred: list[int], labels: list[int]) -> list[list[int]]:
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(y_true, y_pred):
        if gold not in index_by_label or pred not in index_by_label:
            continue
        matrix[index_by_label[gold]][index_by_label[pred]] += 1
    return matrix


def multiclass_macro_f1(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, Any]:
    matrix = multiclass_confusion(y_true, y_pred, labels)
    per_class: dict[int, dict[str, float]] = {}
    f1_values: list[float] = []
    for idx, label in enumerate(labels):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(labels)) if row != idx)
        fn = sum(matrix[idx][col] for col in range(len(labels)) if col != idx)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        support = sum(matrix[idx])
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
        f1_values.append(f1)
    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "macro_f1": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }
