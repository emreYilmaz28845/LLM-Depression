#!/usr/bin/env python3
"""Score one compact Turkish pooled prediction artifact.

The command is deliberately model-free.  It reads prediction rows emitted by
evaluation or a hidden-state head, applies the locked pooled pair rule, and
prints only aggregate metrics and artifact identity.  It never prints prompt,
transcript, audio-path, or subject-level data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import (
    TURKISH_POOLED_TEXT_CONDITIONS,
    TURKISH_POOLED_TEXT_PAIR_POLICY,
    aggregate_binary_classifier_predictions,
    aggregate_response_subject_predictions,
    aggregate_turkish_pooled_text_condition_predictions,
    aggregate_turkish_pooled_text_teacher_forced_predictions,
)
from src.metrics import classification_metrics


SCHEMA_VERSION = "audiollm.turkish_pooled_score.v1"
FORBIDDEN_KEYS = {
    "transcript", "transcript_original", "prompt_text", "prompt_user_text",
    "prompt_system_text", "training_text", "audio_path", "audio_paths",
}


class ScoreError(ValueError):
    """Raised when a prediction artifact cannot support a locked score."""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ScoreError(f"prediction artifact is missing: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ScoreError(f"prediction JSONL row is not an object: {path}")
            rows.append(value)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ScoreError(f"prediction artifact is empty: {path}")
    for row in rows:
        leaked = sorted(FORBIDDEN_KEYS.intersection(row))
        if leaked:
            raise ScoreError(f"prediction artifact contains privacy-sensitive fields: {leaked}")
    return rows


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    raise ScoreError(f"expected a boolean teacher_forced_valid value, got {value!r}")


def _int_or_invalid(value: Any) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return -1
    return candidate if candidate in (0, 1) else -1


def _normalise_rows(rows: list[dict[str, Any]], *, route: str) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if "label" not in row or "subject_id" not in row:
            raise ScoreError("every prediction row requires label and subject_id")
        try:
            row["label"] = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ScoreError(f"invalid binary label: {row.get('label')!r}") from exc
        if row["label"] not in (0, 1):
            raise ScoreError(f"invalid binary label: {row['label']!r}")
        if route == "teacher_forced":
            if "teacher_forced_valid" not in row:
                raise ScoreError("teacher-forced rows require teacher_forced_valid")
            row["teacher_forced_valid"] = _bool(row["teacher_forced_valid"])
            row["teacher_forced_prediction"] = _int_or_invalid(row.get("teacher_forced_prediction"))
            for key in ("dep_score", "non_score"):
                try:
                    row[key] = float(row[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ScoreError(f"teacher-forced rows require numeric {key}") from exc
        else:
            try:
                row["probability"] = float(row["probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ScoreError("classifier rows require numeric probability") from exc
            row["predicted_class"] = _int_or_invalid(row.get("predicted_class"))
        normalised.append(row)
    return normalised


def _strict_metrics(subject_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    if not subject_rows:
        raise ScoreError("aggregation produced no subject rows")
    labels = [int(row["label"]) for row in subject_rows]
    raw_predictions = [int(row.get("prediction", -1)) for row in subject_rows]
    invalid = sum(prediction not in (0, 1) for prediction in raw_predictions)
    strict_predictions = [prediction if prediction in (0, 1) else 1 - label for label, prediction in zip(labels, raw_predictions)]
    metrics = classification_metrics(labels, strict_predictions)
    return {
        "accuracy": float(metrics["accuracy"]),
        "positive_f1": float(metrics["positive_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "binary_strict_accuracy": float(metrics["accuracy"]),
        "binary_strict_positive_f1": float(metrics["positive_f1"]),
        "binary_strict_macro_f1": float(metrics["macro_f1"]),
        "binary_strict_weighted_f1": float(metrics["weighted_f1"]),
        "num_subjects": len(subject_rows),
        "invalid_subjects": invalid,
        "aggregation_policy": TURKISH_POOLED_TEXT_PAIR_POLICY,
    }, invalid


def _classifier_text_condition(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("question_condition", "")).strip() != condition:
            raise ScoreError(f"condition artifact contains a different question condition: {condition}")
        grouped[str(row["subject_id"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject_id, values in sorted(grouped.items()):
        if len(values) != 1:
            raise ScoreError(f"classifier condition view has duplicate subject rows: {subject_id}")
        row = values[0]
        probability = float(row["probability"])
        if not math.isfinite(probability):
            raise ScoreError(f"non-finite classifier probability for {subject_id}")
        subject_rows.append({
            "subject_id": subject_id,
            "label": int(row["label"]),
            "prediction": int(probability >= 0.5),
            "score_margin": probability - 0.5,
        })
    return subject_rows


def score_rows(
    rows: list[dict[str, Any]],
    *,
    route: str,
    modality: str,
    view: str,
    backend: str,
) -> dict[str, Any]:
    if route not in {"teacher_forced", "logreg", "xgb_optuna100"}:
        raise ScoreError(f"unsupported route: {route}")
    if modality not in {"text_only", "audio_only", "audio_text"}:
        raise ScoreError(f"unsupported modality: {modality}")
    if view not in {"positive", "negative", "combined"}:
        raise ScoreError(f"unsupported view: {view}")
    rows = _normalise_rows(rows, route=route)
    is_text = modality == "text_only"
    if view == "positive" or view == "negative":
        condition = TURKISH_POOLED_TEXT_CONDITIONS[0 if view == "positive" else 1]
        rows = [row for row in rows if str(row.get("question_condition", "")).strip() == condition]
        if not rows:
            raise ScoreError(f"no rows for {condition}")
        if route == "teacher_forced" and is_text:
            subject_rows, _ = aggregate_turkish_pooled_text_condition_predictions(rows, condition)
        elif route != "teacher_forced" and is_text:
            subject_rows = _classifier_text_condition(rows, condition)
        elif route == "teacher_forced":
            _, _, subject_rows, _ = aggregate_response_subject_predictions(
                rows, prediction_field="teacher_forced_prediction", backend_name=backend,
                invalid_as_wrong=True, score_average=True,
            )
        else:
            subject_rows, _ = aggregate_binary_classifier_predictions(rows, prediction_backend=backend)
        aggregation_policy = (
            TURKISH_POOLED_TEXT_PAIR_POLICY
            if is_text
            else str(subject_rows[0].get("aggregation_policy", "response_subject"))
        )
    elif route == "teacher_forced" and is_text:
        subject_rows, _ = aggregate_turkish_pooled_text_teacher_forced_predictions(rows)
        aggregation_policy = TURKISH_POOLED_TEXT_PAIR_POLICY
    elif route == "teacher_forced":
        _, _, subject_rows, _ = aggregate_response_subject_predictions(
            rows, prediction_field="teacher_forced_prediction", backend_name=backend,
            invalid_as_wrong=True, score_average=True,
        )
        aggregation_policy = "response_subject_hierarchical_mean"
    else:
        subject_rows, _ = aggregate_binary_classifier_predictions(rows, prediction_backend=backend)
        aggregation_policy = (
            TURKISH_POOLED_TEXT_PAIR_POLICY
            if is_text
            else str(subject_rows[0].get("aggregation_policy", "response_subject"))
        )
    metrics, invalid = _strict_metrics(subject_rows)
    metrics["aggregation_policy"] = aggregation_policy
    return {
        "schema_version": SCHEMA_VERSION,
        "route": route,
        "modality": modality,
        "view": view,
        "backend": backend,
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "metric_namespace": "headline/binary_strict",
        "aggregation": "subject_level",
        "metrics": metrics,
        "support": len(subject_rows),
        "invalid_count": invalid,
    }


def score_prediction_file(
    path: str | Path,
    *,
    route: str,
    modality: str,
    view: str,
    backend: str,
) -> dict[str, Any]:
    target = Path(path).resolve()
    result = score_rows(_read_rows(target), route=route, modality=modality, view=view, backend=backend)
    result["evidence_path"] = str(target)
    result["evidence_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("teacher_forced", "logreg", "xgb_optuna100"))
    parser.add_argument("--modality", required=True, choices=("audio_only", "text_only", "audio_text"))
    parser.add_argument("--view", required=True, choices=("positive", "negative", "combined"))
    parser.add_argument("--backend", required=True)
    args = parser.parse_args(argv)
    try:
        result = score_prediction_file(
            args.predictions, route=args.route, modality=args.modality,
            view=args.view, backend=args.backend,
        )
    except (OSError, ScoreError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
