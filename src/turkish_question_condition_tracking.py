"""Tracking guardrails for Turkish question-condition hidden-head attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiment_tracking.canonical import read_json, write_json_atomic
from src.experiment_tracking.sidecars import ARTIFACTS_FILE
from src.native_en_text_heads_tracking import (
    HeadTrackingError,
    finish_head_attempt,
    initialize_head_attempt,
    materialize_head_evidence,
    materialize_job_evidence,
    record_head_job,
    transition_head_attempt,
    validate_head_attempt,
    validate_job_attempt,
)
from src.turkish_question_condition import (
    EVALUATION_BACKEND,
    EVALUATION_VIEW,
    GROUP_ID,
    METRIC_NAMESPACE,
)


TRACKING_KIND = "turkish_question_condition_v1_head"
_HEAVY_SUFFIXES = {
    ".npz",
    ".joblib",
    ".pkl",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}


def _guard_context(context: dict[str, Any]) -> None:
    if context.get("group_id") != GROUP_ID:
        raise HeadTrackingError(
            f"head context group_id {context.get('group_id')!r} != {GROUP_ID!r}"
        )
    if context.get("tracking_kind") != TRACKING_KIND:
        raise HeadTrackingError(
            f"head context tracking_kind {context.get('tracking_kind')!r} != {TRACKING_KIND!r}"
        )
    qualifiers = context.get("qualifiers") or {}
    if qualifiers.get("evaluation_view") != EVALUATION_VIEW:
        raise HeadTrackingError("head context is missing the locked evaluation view")
    if qualifiers.get("evaluation_backend") != EVALUATION_BACKEND:
        raise HeadTrackingError("head context is missing the locked teacher-forced backend")
    if qualifiers.get("metric_namespace") != METRIC_NAMESPACE:
        raise HeadTrackingError("head context is missing the locked metric namespace")


def initialize_turkish_head_attempt(
    attempt_dir: str | Path,
    *,
    context: dict[str, Any],
    config: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    _guard_context(context)
    result = initialize_head_attempt(
        attempt_dir,
        context=context,
        config=config,
        parent=parent,
    )
    transition_head_attempt(
        attempt_dir,
        "DEPLOYED",
        reason="Turkish question-condition managed head deployment prepared",
    )
    return {**result, "state": "DEPLOYED"}


def _remove_heavy_artifacts(attempt_dir: str | Path) -> int:
    """Keep compact evidence authoritative while leaving runtime files usable."""

    target = Path(attempt_dir)
    path = target / ARTIFACTS_FILE
    document = read_json(path)
    before = list(document.get("artifacts") or [])
    kept = [
        item
        for item in before
        if not str(item.get("path", "")).startswith("hidden_cache/")
        or str(item.get("path", "")) == "hidden_cache/extraction_metadata.json"
    ]
    kept = [
        item
        for item in kept
        if Path(str(item.get("path", ""))).suffix.lower() not in _HEAVY_SUFFIXES
        and not str(item.get("path", "")).endswith("/pipeline.joblib")
    ]
    if len(kept) != len(before):
        document["artifacts"] = kept
        write_json_atomic(path, document)
    return len(before) - len(kept)


def materialize_turkish_head_evidence(
    attempt_dir: str | Path,
    *,
    predictions_path: str | Path,
    metrics_path: str | Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    result = materialize_head_evidence(
        attempt_dir,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
    )
    result["heavy_artifacts_excluded"] = _remove_heavy_artifacts(attempt_dir)
    return result


__all__ = [
    "HeadTrackingError",
    "TRACKING_KIND",
    "finish_head_attempt",
    "initialize_turkish_head_attempt",
    "materialize_job_evidence",
    "materialize_turkish_head_evidence",
    "record_head_job",
    "transition_head_attempt",
    "validate_head_attempt",
    "validate_job_attempt",
]
