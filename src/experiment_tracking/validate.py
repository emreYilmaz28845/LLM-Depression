"""Local validation and finish gates for collected attempts.

Verifies identity (artifact/attempt/deployment/config/checkpoint/backend/
view/aggregation/namespace), recomputes strict headline metrics from local
subject predictions, enforces evaluation idempotency, and advances lifecycle
only through official single-step transitions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiment_tracking import lifecycle
from src.experiment_tracking.sidecars import (
    ModernSidecars,
    read_modern_sidecars,
    verify_modern_evidence_locally,
)


class ValidationError(RuntimeError):
    """Raised when local validation must fail closed."""


def recompute_strict_headline(subject_predictions_csv: Path) -> dict[str, float]:
    """Recompute binary_strict headline metrics from subject-level predictions.

    INVALID predictions count as wrong (strict view).
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    with subject_predictions_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError):
                raise ValidationError(f"subject predictions row missing label: {row}")
            pred_raw = (row.get("prediction_text") or "").strip().lower()
            if pred_raw == "depressed":
                pred = 1
            elif pred_raw == "non-depressed":
                pred = 0
            else:
                pred = -1  # INVALID counts as wrong under binary_strict
            y_true.append(label)
            y_pred.append(pred)
    if not y_true:
        raise ValidationError("subject predictions file has no rows")
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p != 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    positive_f1 = (
        2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    )
    f1_neg = (
        2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    )
    macro_f1 = (positive_f1 + f1_neg) / 2
    accuracy = (tp + tn) / n
    return {
        "binary_strict_macro_f1": macro_f1,
        "binary_strict_positive_f1": positive_f1,
        "binary_strict_accuracy": accuracy,
        "support": float(n),
    }


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def validate_attempt(
    fold_dir: str | Path,
    *,
    expected_attempt_id: str | None = None,
    expected_dataset: str | None = None,
    expected_evaluation_view: str | None = None,
    expected_backend: str | None = None,
    expected_aggregation: str | None = None,
    require_standalone_eval: bool = True,
) -> dict[str, Any]:
    """Full local verification. Returns a structured result; raises on hard errors."""
    fold = Path(fold_dir)
    issues: list[str] = []
    result: dict[str, Any] = {"fold_dir": str(fold), "issues": issues}

    required = [
        "run_config.yaml", "metadata.json", "status.json", "jobs.jsonl",
        "artifacts.json", "evaluations.json",
    ]
    missing = [f for f in required if not (fold / f).is_file()]
    if missing:
        raise ValidationError(f"required evidence files missing in {fold}: {missing}")

    sidecars: ModernSidecars | None = read_modern_sidecars(fold)
    if sidecars is None:
        raise ValidationError(f"sidecars in {fold} are malformed or contradictory")
    result["attempt_id"] = sidecars.attempt_id
    result["state"] = sidecars.state

    issues.extend(verify_modern_evidence_locally(sidecars))

    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    if expected_attempt_id and metadata.get("attempt_id") != expected_attempt_id:
        issues.append(
            f"metadata attempt_id {metadata.get('attempt_id')!r} != expected {expected_attempt_id!r}"
        )
    source = metadata.get("source", {}) or {}
    if not source.get("deployed_source_sha256"):
        issues.append("metadata.source.deployed_source_sha256 missing")

    # Resolved config identity vs expected qualifiers.
    import yaml as _yaml
    run_config = _yaml.safe_load((fold / "run_config.yaml").read_text(encoding="utf-8"))
    evaluation_cfg = (run_config or {}).get("evaluation", {}) or {}
    if expected_dataset and (run_config or {}).get("dataset") != expected_dataset:
        issues.append(
            f"run_config dataset {(run_config or {}).get('dataset')!r} != expected {expected_dataset!r}"
        )
    if expected_evaluation_view and evaluation_cfg.get("evaluation_view") != expected_evaluation_view:
        issues.append(
            f"evaluation_view {evaluation_cfg.get('evaluation_view')!r} != expected {expected_evaluation_view!r}"
        )
    if expected_backend and evaluation_cfg.get("sample_prediction_mode") != expected_backend:
        issues.append(
            f"backend {evaluation_cfg.get('sample_prediction_mode')!r} != expected {expected_backend!r}"
        )
    agg = evaluation_cfg.get("aggregation_level", "subject")
    if expected_aggregation and agg != expected_aggregation:
        issues.append(f"aggregation {agg!r} != expected {expected_aggregation!r}")

    # Standalone evaluation requirement: train-time-only evidence cannot pass.
    standalone_dir = fold / "best_model" / "standalone_eval"
    standalone_metrics = standalone_dir / "metrics_original_teacher_forced.json"
    standalone_preds = standalone_dir / "predictions_subject_level.csv"
    if require_standalone_eval:
        if not standalone_metrics.is_file() or not standalone_preds.is_file():
            issues.append(
                "standalone evaluation evidence missing under best_model/standalone_eval "
                "(train-time eval/best_checkpoint is not an allowed substitute)"
            )
    last_only = (fold / "last_model" / "standalone_eval").exists()
    if last_only:
        issues.append("standalone evaluation found under last_model; checkpoint role must be best_model")

    # Recompute headline from local subject predictions and compare.
    recomputed: dict[str, float] | None = None
    if standalone_metrics.is_file() and standalone_preds.is_file():
        metrics_record = json.loads(standalone_metrics.read_text(encoding="utf-8"))
        recomputed = recompute_strict_headline(standalone_preds)
        for key, value in recomputed.items():
            if key not in metrics_record:
                continue  # support and derived keys are compared only when recorded
            recorded = metrics_record.get(key)
            if recorded is None or not _close(float(recorded), float(value)):
                issues.append(
                    f"recomputed {key}={value:.6f} differs from recorded {recorded!r} "
                    f"in {standalone_metrics.name}"
                )
        for key in ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_accuracy"):
            if key not in metrics_record:
                issues.append(f"standalone metrics file missing required key {key}")
        result["recomputed"] = recomputed

        # Recorded evaluations must agree with the artifact values.
        evaluations = json.loads((fold / "evaluations.json").read_text(encoding="utf-8"))
        records = evaluations.get("evaluations", []) if isinstance(evaluations, dict) else evaluations
        seen_ids: dict[str, Any] = {}
        for record in records:
            eid = record.get("evaluation_id")
            if eid in seen_ids and seen_ids[eid] != record:
                issues.append(f"evaluation idempotency violated: {eid} reused with changed content")
            seen_ids[eid] = record
            if record.get("metrics_artifact_path") and "standalone_eval" in str(record.get("metrics_artifact_path")):
                metrics_by_name = {
                    m.get("name"): m.get("value")
                    for m in record.get("metrics", [])
                    if isinstance(m, dict)
                }
                for key in ("macro_f1", "positive_f1"):
                    got = metrics_by_name.get(key)
                    want = recomputed.get(f"binary_strict_{key}")
                    if got is not None and want is not None and not _close(float(got), float(want)):
                        issues.append(
                            f"evaluations.json {key}={got} disagrees with recomputation {want}"
                        )

    result["ok"] = not issues
    return result


def read_state(fold_dir: str | Path) -> tuple[str, list[dict[str, Any]]]:
    status = json.loads((Path(fold_dir) / "status.json").read_text(encoding="utf-8"))
    return status.get("state", "PLANNED"), status.get("history", [])


def advance_lifecycle(fold_dir: str | Path, target: str) -> str:
    """Single-step official transition; refuses skips."""
    fold = Path(fold_dir)
    current, _history = read_state(fold)
    from src.experiment_tracking.monitor import MonitorError, validate_lifecycle_advancement
    try:
        validate_lifecycle_advancement(current, target)
    except MonitorError as e:
        raise ValidationError(str(e))
    status_path = fold / "status.json"
    record = lifecycle.StatusRecord.from_dict(lifecycle.read_status(status_path))
    try:
        record.transition(target)
    except lifecycle.InvalidTransitionError as e:
        raise ValidationError(str(e))
    lifecycle.write_status(status_path, record)
    return target


def _attempt_id_of(fold: Path) -> str:
    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    return str(metadata.get("attempt_id"))


def _fold_of(fold: Path) -> int:
    name = fold.name
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def finish_gates(
    fold_dir: str | Path,
    *,
    expected_attempt_id: str | None = None,
    expected_dataset: str | None = None,
    expected_evaluation_view: str | None = None,
    expected_backend: str | None = None,
    expected_aggregation: str | None = None,
    required_jobs_terminal_success: bool = True,
) -> dict[str, Any]:
    """Gate orchestrator: validates everything, then advances stepwise toward
    REPORTABLE. Never skips states; returns the exact blocking gate otherwise."""
    fold = Path(fold_dir)
    state, _history = read_state(fold)

    if required_jobs_terminal_success:
        jobs_path = fold / "jobs.jsonl"
        events = [json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_key: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_key.setdefault(str(event.get("job_key")), []).append(event)
        for key in ("train", "best_eval"):
            terminal = [
                e for e in by_key.get(key, [])
                if e.get("event_type") == "COMPLETED" and e.get("status") == "COMPLETED"
                and (not e.get("exit_code") or str(e["exit_code"]).startswith("0:0"))
            ]
            if not terminal:
                return {
                    "ok": False,
                    "state": state,
                    "next_action": (
                        f"job '{key}' lacks a COMPLETED 0:0 job event in {jobs_path}; "
                        "run exp status to reconcile scheduler accounting first"
                    ),
                }

    validation = validate_attempt(
        fold,
        expected_attempt_id=expected_attempt_id,
        expected_dataset=expected_dataset,
        expected_evaluation_view=expected_evaluation_view,
        expected_backend=expected_backend,
        expected_aggregation=expected_aggregation,
        require_standalone_eval=True,
    )
    if not validation["ok"]:
        return {
            "ok": False,
            "state": state,
            "next_action": "fix validation issues: " + "; ".join(validation["issues"][:5]),
            "issues": validation["issues"],
        }

    # Stepwise advancement through official transitions only.
    if state == "COMPLETED_ON_MN5":
        advance_lifecycle(fold, "SYNCED_LOCALLY")
        state = "SYNCED_LOCALLY"
    if state == "SYNCED_LOCALLY":
        advance_lifecycle(fold, "LOCALLY_VALIDATED")
        state = "LOCALLY_VALIDATED"
    if state == "LOCALLY_VALIDATED":
        advance_lifecycle(fold, "REPORTABLE")
        state = "REPORTABLE"
    if state == "REPORTABLE":
        return {"ok": True, "state": state, "next_action": "generate deterministic reports"}
    return {
        "ok": False,
        "state": state,
        "next_action": f"lifecycle state {state} cannot reach REPORTABLE without its preceding gates",
    }
