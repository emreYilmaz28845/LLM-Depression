"""Tracking guards and compact-evidence helpers for pooled Turkish heads."""

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
from src.turkish_pooled_qcond import (
    EVALUATION_BACKEND,
    EVALUATION_VIEW,
    GROUP_ID,
    METRIC_NAMESPACE,
    PAIR_POLICY,
)


TRACKING_KIND = "turkish_pooled_qcond_v1_head"
_HEAVY_SUFFIXES = {".npz", ".joblib", ".pkl", ".safetensors", ".bin", ".pt", ".pth"}


def _guard_context(context: dict[str, Any]) -> None:
    if context.get("group_id") != GROUP_ID:
        raise HeadTrackingError(f"pooled head context group_id mismatch: {context.get('group_id')!r}")
    if context.get("tracking_kind") != TRACKING_KIND:
        raise HeadTrackingError(f"pooled head context tracking_kind mismatch: {context.get('tracking_kind')!r}")
    qualifiers = context.get("qualifiers") or {}
    if qualifiers.get("evaluation_view") != EVALUATION_VIEW or qualifiers.get("evaluation_backend") != EVALUATION_BACKEND:
        raise HeadTrackingError("pooled head context lacks locked evaluation qualifiers")
    if qualifiers.get("metric_namespace") != METRIC_NAMESPACE:
        raise HeadTrackingError("pooled head context lacks the locked metric namespace")
    if context.get("pair_policy") not in (None, PAIR_POLICY):
        raise HeadTrackingError("pooled head context has an unexpected pair policy")


def initialize_turkish_pooled_head_attempt(
    attempt_dir: str | Path, *, context: dict[str, Any], config: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    _guard_context(context)
    result = initialize_head_attempt(attempt_dir, context=context, config=config, parent=parent)
    transition_head_attempt(attempt_dir, "DEPLOYED", reason="Turkish pooled managed head deployment prepared")
    return {**result, "state": "DEPLOYED"}


def _remove_heavy_artifacts(attempt_dir: str | Path) -> int:
    path = Path(attempt_dir) / ARTIFACTS_FILE
    document = read_json(path)
    before = list(document.get("artifacts") or [])
    kept = [
        item for item in before
        if (not str(item.get("path", "")).startswith("hidden_cache/")
            or str(item.get("path", "")) == "hidden_cache/extraction_metadata.json")
        and Path(str(item.get("path", ""))).suffix.lower() not in _HEAVY_SUFFIXES
        and not str(item.get("path", "")).endswith("/pipeline.joblib")
    ]
    if len(kept) != len(before):
        document["artifacts"] = kept
        write_json_atomic(path, document)
    return len(before) - len(kept)


def materialize_turkish_pooled_head_evidence(
    attempt_dir: str | Path, *, predictions_path: str | Path, metrics_path: str | Path, checkpoint_path: str
) -> dict[str, Any]:
    result = materialize_head_evidence(
        attempt_dir, predictions_path=predictions_path, metrics_path=metrics_path, checkpoint_path=checkpoint_path
    )
    result["heavy_artifacts_excluded"] = _remove_heavy_artifacts(attempt_dir)
    return result


__all__ = [
    "HeadTrackingError", "TRACKING_KIND", "finish_head_attempt",
    "initialize_turkish_pooled_head_attempt", "materialize_job_evidence",
    "materialize_turkish_pooled_head_evidence", "record_head_job",
    "transition_head_attempt", "validate_head_attempt", "validate_job_attempt",
]
