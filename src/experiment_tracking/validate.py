"""Local validation and finish gates for collected attempts.

Verifies identity (artifact/attempt/deployment/config/checkpoint/backend/
view/aggregation/namespace), recomputes strict headline metrics from local
subject predictions, enforces evaluation idempotency, and advances lifecycle
only through official single-step transitions.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiment_tracking import lifecycle
from src.experiment_tracking.canonical import parse_utc_timestamp
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
                # Keep the invalid output in the wrong binary class so it is
                # counted as a false positive or false negative by strict
                # metrics, matching the training/evaluation aggregation rule.
                pred = 1 - int(row["label"])
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
    positive_recall = tp / (tp + fn) if (tp + fn) else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) else 0.0
    # UAR (unweighted average recall / balanced accuracy); an invalid output is
    # already mapped to the wrong class above, so it lowers the true class recall.
    uar = (positive_recall + negative_recall) / 2
    return {
        "binary_strict_macro_f1": macro_f1,
        "binary_strict_positive_f1": positive_f1,
        "binary_strict_accuracy": accuracy,
        "binary_strict_uar": uar,
        "support": float(n),
    }


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _canonical_aggregation(value: str | None) -> str | None:
    """Map the config spelling to the evidence qualifier spelling."""
    if value == "subject_level":
        return "subject"
    return value


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
    raw_config = _yaml.safe_load((fold / "run_config.yaml").read_text(encoding="utf-8")) or {}
    # src/train.py writes the fully resolved config under the "config" key;
    # older layouts keep values at the top level.
    run_config = raw_config.get("config") if isinstance(raw_config.get("config"), dict) else raw_config
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
    if expected_aggregation and _canonical_aggregation(agg) != _canonical_aggregation(expected_aggregation):
        issues.append(f"aggregation {agg!r} != expected {expected_aggregation!r}")

    # Standalone evaluation requirement: train-time-only evidence cannot pass.
    standalone_dir = fold / "best_model" / "standalone_eval"
    backend = evaluation_cfg.get("sample_prediction_mode")
    if backend not in {"original_teacher_forced", "likelihood", "generation"}:
        issues.append(f"unsupported standalone evaluation backend {backend!r}")
    standalone_metrics = standalone_dir / f"metrics_{backend}.json"
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
        for key in ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_accuracy", "binary_strict_uar"):
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


TERMINAL_EVENT_TYPES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
FAILED_TERMINAL_EVENT_TYPES = frozenset({"FAILED", "CANCELLED", "TIMEOUT"})
EVAL_PARENT_SUBMIT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "exp_submit"


def _clean_completed(event: dict[str, Any]) -> bool:
    """Lenient terminal-success predicate for the ordinary finish gate."""
    if event.get("event_type") != "COMPLETED" or event.get("status") != "COMPLETED":
        return False
    code = event.get("exit_code")
    return not code or str(code).startswith("0:0")


def _strict_clean_completed(event: dict[str, Any]) -> bool:
    """Recovery predicate: an explicit exact Slurm ``0:0`` exit code is required."""
    return (
        event.get("event_type") == "COMPLETED"
        and event.get("status") == "COMPLETED"
        and str(event.get("exit_code") or "") == "0:0"
    )


def _event_time(event: dict[str, Any]) -> datetime | None:
    raw = event.get("at_utc")
    if not isinstance(raw, str):
        return None
    try:
        return parse_utc_timestamp(raw)
    except ValueError:
        return None


def _verify_eval_parent_contract(fold: Path, attempt_id: str, fold_number: int) -> dict[str, Any]:
    """Verify eval parent linkage through the attempt's submitted contract.

    Used only when the best_eval retry events carry no scheduler dependency
    ids. The contract must describe this exact attempt and fold and must point
    at this fold's ``best_model`` checkpoint and its standalone eval dir.
    """
    contract_path = EVAL_PARENT_SUBMIT_ROOT / attempt_id / "contract.json"
    if not contract_path.is_file():
        return {"ok": False, "reason": f"attempt submission contract not found at {contract_path}"}
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"ok": False, "reason": f"unreadable attempt contract: {error}"}
    if not isinstance(contract, dict):
        return {"ok": False, "reason": "attempt contract is not a JSON object"}
    if contract.get("attempt_id") != attempt_id:
        return {"ok": False, "reason": "contract attempt_id does not match the fold"}
    try:
        contract_fold = int(contract.get("fold", -1))
    except (TypeError, ValueError):
        return {"ok": False, "reason": "contract fold is not an integer"}
    if contract_fold != fold_number:
        return {"ok": False, "reason": "contract fold does not match the fold"}
    if contract.get("kind", "standalone_backbone") != "standalone_backbone":
        return {"ok": False, "reason": "contract is not a standalone backbone submission"}
    repo_root = EVAL_PARENT_SUBMIT_ROOT.parent.parent
    local_fold_rel = contract.get("local_fold_rel")
    if not local_fold_rel or (repo_root / str(local_fold_rel)).resolve() != fold.resolve():
        return {"ok": False, "reason": "contract local fold path does not match the fold"}
    parts = fold.resolve().parts
    if "output_model" not in parts:
        return {"ok": False, "reason": "fold path lacks the output_model root"}
    rel = Path(*parts[parts.index("output_model") :]).as_posix()
    checkpoint_dir = str(contract.get("checkpoint_dir") or "").replace("\\", "/").rstrip("/")
    standalone_dir = str(contract.get("standalone_eval_dir") or "").replace("\\", "/").rstrip("/")
    if not checkpoint_dir.endswith(f"/{rel}/best_model"):
        return {"ok": False, "reason": "contract checkpoint_dir is not this fold's best_model checkpoint"}
    if not standalone_dir.endswith(f"/{rel}/best_model/standalone_eval"):
        return {"ok": False, "reason": "contract standalone_eval_dir is not this fold's best_model standalone eval"}
    qualifiers = contract.get("qualifiers")
    if not isinstance(qualifiers, dict) or qualifiers.get("checkpoint_role") != "best_model":
        return {"ok": False, "reason": "contract qualifiers do not record checkpoint_role=best_model"}
    return {"ok": True, "contract_path": str(contract_path), "checkpoint_dir": checkpoint_dir}


def _verify_retry_chains_for_leg(
    key: str,
    key_events: list[dict[str, Any]],
    *,
    attempt_id: str,
    fold_number: int,
    fold: Path,
    completed_train_id: str | None,
) -> dict[str, Any]:
    """Verify one required leg: a strict completion plus superseded failures.

    Returns ``{"ok": True, "chains": [...], "completed_id": <strict id>}`` or
    ``{"ok": False, "reason": ...}``. Every failed/cancelled/timeout terminal
    must be superseded by a retry linked through ``resubmission_of_job_id``
    whose own clean COMPLETED 0:0 event has strictly increasing immutable
    timestamps: predecessor submitted < retry submitted < retry completed.
    A leg with no failures only needs at least one clean COMPLETED 0:0
    terminal. Failures are resolved per terminal event, never by append order,
    so late-mirrored failures cannot ride on an earlier success.
    """
    for event in key_events:
        try:
            event_fold = int(event.get("fold", -1))
        except (TypeError, ValueError):
            return {"ok": False, "reason": f"{key}: event fold is not an integer"}
        if str(event.get("attempt_id")) != attempt_id or event_fold != fold_number:
            return {"ok": False, "reason": f"{key}: event identity does not match the fold metadata"}
    terminals = [event for event in key_events if event.get("event_type") in TERMINAL_EVENT_TYPES]
    if not terminals:
        return {"ok": False, "reason": f"{key}: no terminal job event"}
    for event in terminals:
        if _event_time(event) is None:
            return {"ok": False, "reason": f"{key}: terminal event without a parseable at_utc timestamp"}
    if not any(_strict_clean_completed(event) for event in terminals):
        return {"ok": False, "reason": f"{key}: no COMPLETED 0:0 terminal event"}
    completed_id = next(
        (
            str(event.get("slurm_job_id") or "")
            for event in reversed(terminals)
            if _strict_clean_completed(event) and str(event.get("slurm_job_id") or "")
        ),
        None,
    )
    chains: list[dict[str, str]] = []
    failures = [event for event in terminals if event.get("event_type") in FAILED_TERMINAL_EVENT_TYPES]
    for failed in failures:
        failed_id = str(failed.get("slurm_job_id") or "")
        if not failed_id:
            return {"ok": False, "reason": f"{key}: failed terminal event without a Slurm job id cannot be linked"}
        predecessor_submitted = next(
            (
                event
                for event in key_events
                if event.get("event_type") == "SUBMITTED"
                and str(event.get("slurm_job_id") or "") == failed_id
            ),
            None,
        )
        if predecessor_submitted is None:
            return {"ok": False, "reason": f"{key}: failed job {failed_id} has no SUBMITTED event to anchor chronology"}
        submitted_retry = next(
            (
                event
                for event in key_events
                if event.get("event_type") == "SUBMITTED"
                and str(event.get("resubmission_of_job_id") or "") == failed_id
                and str(event.get("slurm_job_id") or "")
            ),
            None,
        )
        if submitted_retry is None:
            return {"ok": False, "reason": f"{key}: failed job {failed_id} has no linked submitted retry"}
        retry_id = str(submitted_retry["slurm_job_id"])
        retry_completed = next(
            (
                event
                for event in key_events
                if event.get("event_type") == "COMPLETED"
                and str(event.get("slurm_job_id") or "") == retry_id
                and _strict_clean_completed(event)
            ),
            None,
        )
        if retry_completed is None:
            return {"ok": False, "reason": f"{key}: retry job {retry_id} has no COMPLETED 0:0 event"}
        predecessor_time = _event_time(predecessor_submitted)
        submitted_time = _event_time(submitted_retry)
        completed_time = _event_time(retry_completed)
        if predecessor_time is None or submitted_time is None or completed_time is None:
            return {"ok": False, "reason": f"{key}: retry chain events need parseable at_utc timestamps"}
        if not (predecessor_time < submitted_time < completed_time):
            return {
                "ok": False,
                "reason": (
                    f"{key}: retry {retry_id} chronology invalid: predecessor submitted "
                    f"{predecessor_submitted.get('at_utc')}, retry submitted {submitted_retry.get('at_utc')}, "
                    f"retry completed {retry_completed.get('at_utc')}"
                ),
            }
        if key == "best_eval":
            if completed_train_id is None:
                return {
                    "ok": False,
                    "reason": f"{key}: completed train job has no Slurm job id; cannot verify parent linkage",
                }
            dependencies = {
                str(item)
                for item in (retry_completed.get("dependency_job_ids") or [])
                + (submitted_retry.get("dependency_job_ids") or [])
            }
            if dependencies:
                if completed_train_id not in dependencies:
                    return {
                        "ok": False,
                        "reason": (
                            f"{key}: retry {retry_id} dependency link does not include the completed train job "
                            f"{completed_train_id}"
                        ),
                    }
            else:
                parent = _verify_eval_parent_contract(fold, attempt_id, fold_number)
                if not parent["ok"]:
                    return {
                        "ok": False,
                        "reason": (
                            f"{key}: retry {retry_id} has no scheduler dependency and no verified parent "
                            f"contract: {parent['reason']}"
                        ),
                    }
        chains.append(
            {
                "job_key": key,
                "failed_job_id": failed_id,
                "retry_job_id": retry_id,
                "predecessor_submitted_at_utc": str(predecessor_submitted.get("at_utc")),
                "retry_submitted_at_utc": str(submitted_retry.get("at_utc")),
                "retry_completed_at_utc": str(retry_completed.get("at_utc")),
            }
        )
    return {"ok": True, "chains": chains, "completed_id": completed_id}


def recover_failed_attempt_from_verified_retry(fold_dir: str | Path) -> dict[str, Any]:
    """Recover a FAILED fold only when every failed required leg has a verified retry.

    The intended protocol for a lost evaluation leg is an append-only retry
    within the same attempt: a SUBMITTED event that links to the failed job via
    ``resubmission_of_job_id`` and a later cleanly COMPLETED terminal event for
    that retry. This helper is the single verification gate for the
    ``FAILED -> COMPLETED_ON_MN5`` recovery. It fails closed for unlinked
    retries, wrong attempt/fold events, missing parent links, failed retries
    without their own successor, missing or inexact exit codes and a FAILED
    state with no retry evidence at all.

    Chronology is verified from the immutable event ``at_utc`` timestamps, not
    from append order: the retry must be submitted after its failed
    predecessor's own submission and before its own clean completion. Mirrored
    terminal events record the reconciliation time, so the failed predecessor's
    SUBMITTED time is the only truthful anchor for "the retry follows the
    failure"; late-appended official evidence stays valid when its timestamps
    are the real submission times. Every failure terminal is resolved
    individually, so a failure mirrored after the retry success still cannot
    ride on that success without its own linked chain.

    Recovery requires an explicit exact Slurm ``0:0`` exit code. When the
    best_eval retry carries no scheduler dependency ids, the attempt's
    submission contract must verify the parent: this exact attempt and fold,
    this fold's ``best_model`` checkpoint and its standalone eval dir.

    The original FAILED/CANCELLED events stay in the append-only job history and
    the verification payload is recorded in the status history.
    """
    fold = Path(fold_dir)
    status_path = fold / "status.json"
    status = lifecycle.read_status(status_path)
    state = status.get("state")
    if state not in {"FAILED", "CANCELLED"}:
        return {
            "recovered": False,
            "state": state,
            "reason": f"state is {state!r}, not 'FAILED' or 'CANCELLED'",
        }
    try:
        metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"recovered": False, "state": state, "reason": f"unreadable fold metadata: {error}"}
    if not isinstance(metadata, dict):
        return {"recovered": False, "state": state, "reason": "fold metadata is not an object"}
    try:
        fold_number = int(metadata.get("fold", 0) or 0)
    except (TypeError, ValueError):
        return {"recovered": False, "state": state, "reason": "fold metadata fold is not an integer"}
    attempt_id = str(metadata.get("attempt_id") or "")
    events = lifecycle.read_job_events(fold / "jobs.jsonl")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_key.setdefault(str(event.get("job_key")), []).append(event)

    def blocked(reason: str) -> dict[str, Any]:
        return {"recovered": False, "state": state, "reason": reason}

    verified: list[dict[str, str]] = []
    completed_train_id: str | None = None
    for key in ("train", "best_eval"):
        result = _verify_retry_chains_for_leg(
            key,
            by_key.get(key, []),
            attempt_id=attempt_id,
            fold_number=fold_number,
            fold=fold,
            completed_train_id=completed_train_id,
        )
        if not result["ok"]:
            return blocked(result["reason"])
        verified.extend(result["chains"])
        if key == "train":
            completed_train_id = result.get("completed_id")
    if not verified:
        return blocked("FAILED state has no linked retry evidence")
    record = lifecycle.StatusRecord.from_dict(status)
    try:
        new_state = record.recover_failed_to_completed(
            reason=(
                "verified linked retry completed cleanly for every failed required leg"
            ),
            verification={"verified_retry_jobs": verified},
        )
    except lifecycle.InvalidTransitionError as exc:
        return blocked(str(exc))
    lifecycle.write_status(status_path, record)
    return {
        "recovered": True,
        "state": new_state,
        "retry_jobs": verified,
        "train_retry_id": next(
            (chain["retry_job_id"] for chain in verified if chain["job_key"] == "train"), None
        ),
    }


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

    if state in {"FAILED", "CANCELLED"}:
        recovery = recover_failed_attempt_from_verified_retry(fold)
        if not recovery["recovered"]:
            return {
                "ok": False,
                "state": state,
                "next_action": recovery.get("reason", "failed state has no verified linked retry"),
            }
        state = recovery["state"]

    if required_jobs_terminal_success:
        jobs_path = fold / "jobs.jsonl"
        events = [json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_key: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_key.setdefault(str(event.get("job_key")), []).append(event)
        chain_attempt_id: str | None = None
        chain_fold_number = 0
        completed_train_id: str | None = None
        for key in ("train", "best_eval"):
            key_events = by_key.get(key, [])
            terminals = [e for e in key_events if e.get("event_type") in TERMINAL_EVENT_TYPES]
            has_failure = any(e.get("event_type") in FAILED_TERMINAL_EVENT_TYPES for e in terminals)
            if has_failure:
                if chain_attempt_id is None:
                    try:
                        metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
                        chain_attempt_id = str(metadata.get("attempt_id") or "")
                        chain_fold_number = int(metadata.get("fold", 0) or 0)
                    except (OSError, ValueError, TypeError):
                        return {
                            "ok": False,
                            "state": state,
                            "next_action": (
                                f"job '{key}' has a failure but the fold metadata is unreadable; "
                                "cannot verify the linked retry chain"
                            ),
                        }
                result = _verify_retry_chains_for_leg(
                    key,
                    key_events,
                    attempt_id=chain_attempt_id,
                    fold_number=chain_fold_number,
                    fold=fold,
                    completed_train_id=completed_train_id,
                )
                if not result["ok"]:
                    return {
                        "ok": False,
                        "state": state,
                        "next_action": f"job '{key}' has an unresolved failure: {result['reason']}",
                    }
                if key == "train":
                    completed_train_id = result.get("completed_id")
            else:
                clean = [e for e in terminals if _clean_completed(e)]
                if not clean:
                    return {
                        "ok": False,
                        "state": state,
                        "next_action": (
                            f"job '{key}' lacks a COMPLETED 0:0 job event in {jobs_path}; "
                            "run exp status to reconcile scheduler accounting first"
                        ),
                    }
                if key == "train":
                    completed_train_id = str(clean[-1].get("slurm_job_id") or "") or None

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
