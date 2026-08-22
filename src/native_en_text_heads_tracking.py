"""Tracking and evidence helpers for the v2 hidden-head attempts.

Standalone backbone jobs use the repository's existing train/evaluate sidecar
writer.  LogReg, Optuna, and merged-head jobs need the same modern sidecars,
but their outputs are not model-training runs.  This module keeps those head
attempts on the official lifecycle and evidence formats without importing W&B
or writing the SQLite registry from a worker.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.experiment_tracking.canonical import (
    format_utc_timestamp,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.experiment_tracking.identity import artifact_id, evaluation_id
from src.experiment_tracking.lifecycle import (
    StatusRecord,
    append_job_event,
    new_job_event,
    read_status,
    write_status,
)
from src.experiment_tracking.sidecars import (
    ARTIFACTS_FILE,
    EVALUATIONS_FILE,
    JOBS_FILE,
    METADATA_FILE,
    STATUS_FILE,
    read_modern_sidecars,
    verify_modern_evidence_locally,
)
from src.metrics import classification_metrics


HEAD_TRACKING_KIND = "native_en_text_heads_v2_head"
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
METRIC_NAMESPACE = "headline/binary_strict"


class HeadTrackingError(ValueError):
    """A head attempt is malformed, contradictory, or unsafe to resume."""


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing != payload:
            raise HeadTrackingError(f"refusing to overwrite incompatible {path}")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    return sha256_file(target) if target.is_file() else None


def _context_source(context: dict[str, Any]) -> dict[str, Any]:
    source = dict(context.get("source") or {})
    return {
        "git_commit": source.get("git_commit"),
        "git_branch": source.get("git_branch"),
        "git_dirty": bool(source.get("git_dirty", False)),
        "deployed_source_sha256": source.get("deployed_source_sha256"),
        "deployment_id": source.get("deployment_id"),
    }


def initialize_head_attempt(
    attempt_dir: str | Path,
    *,
    context: dict[str, Any],
    config: dict[str, Any],
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create a new head attempt and refuse every destination collision."""

    target = Path(attempt_dir)
    if target.exists():
        raise HeadTrackingError(f"head attempt destination already exists: {target}")
    required = ("attempt_id", "logical_run_name", "fold", "seed")
    missing = [key for key in required if key not in context]
    if missing:
        raise HeadTrackingError(f"head context is missing {missing}")
    if parent and not parent.get("parent_attempt_id"):
        raise HeadTrackingError("parent evidence requires parent_attempt_id")
    target.mkdir(parents=True, exist_ok=False)
    attempt_id = str(context["attempt_id"])
    fold = int(context["fold"])
    tracking = {
        "kind": context.get("tracking_kind") or HEAD_TRACKING_KIND,
        "group_id": context.get("group_id"),
        "logical_run_name": context["logical_run_name"],
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": int(context["seed"]),
        "method": config.get("classifier", {}).get("method"),
        "stage": config.get("stage"),
        "required_jobs": list(context.get("required_jobs") or ["head"]),
    }
    run_config = {
        "schema_version": "native_en_text_heads_v2_run.v1",
        "config": config,
        "tracking": tracking,
    }
    _write_yaml(target / "run_config.yaml", run_config)
    context_hashes = context.get("hashes") if isinstance(context.get("hashes"), dict) else {}
    metadata = {
        "schema_version": "audiollm.metadata.v1",
        "group_id": context.get("group_id"),
        "logical_run_name": str(context["logical_run_name"]),
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": int(context["seed"]),
        "created_at_utc": str(
            context.get("created_at_utc") or format_utc_timestamp(utc_now())
        ),
        "source": _context_source(context),
        "research": {
            "github_issue": context.get("research", {}).get("github_issue"),
            "github_pr": context.get("research", {}).get("github_pr"),
        },
        "hashes": {
            "resolved_config_sha256": sha256_file(target / "run_config.yaml"),
            "manifest_sha256": context.get("manifest_sha256") or context_hashes.get("manifest_sha256"),
            "split_sha256": context.get("split_sha256") or context_hashes.get("split_sha256"),
        },
        "paths": {
            "run_config": "run_config.yaml",
            "best_model": (parent or {}).get("parent_checkpoint_path"),
            "local_evidence_root": None,
        },
        "wandb": {
            "project": "audiollm-depression",
            "entity": None,
            "run_id": f"{attempt_id}-fold{fold}",
            "url": None,
            "sync_status": "NOT_EXPORTED",
        },
    }
    if parent:
        metadata["parent"] = {
            "parent_attempt_id": parent.get("parent_attempt_id"),
            "parent_checkpoint_role": "best_model",
            "parent_checkpoint_path": parent.get("parent_checkpoint_path"),
            "adapter_config_sha256": parent.get("adapter_config_sha256"),
            "adapter_sha256": parent.get("adapter_sha256"),
        }
    if context.get("supersedes_attempt_id"):
        metadata["supersedes_attempt_id"] = context["supersedes_attempt_id"]
    write_json_atomic(target / METADATA_FILE, metadata)
    write_status(target / STATUS_FILE, StatusRecord(attempt_id, fold))
    (target / JOBS_FILE).write_text("", encoding="utf-8")
    write_json_atomic(
        target / ARTIFACTS_FILE,
        {
            "schema_version": "audiollm.artifacts.v1",
            "attempt_id": attempt_id,
            "fold": fold,
            "artifacts": [],
        },
    )
    write_json_atomic(
        target / EVALUATIONS_FILE,
        {
            "schema_version": "audiollm.evaluations.v1",
            "attempt_id": attempt_id,
            "fold": fold,
            "evaluations": [],
        },
    )
    return {"attempt_id": attempt_id, "attempt_dir": str(target), "state": "PLANNED"}


def _sidecar(target: str | Path):
    sidecars = read_modern_sidecars(target)
    if sidecars is None:
        raise HeadTrackingError(f"missing modern sidecars under {target}")
    return sidecars


def transition_head_attempt(
    attempt_dir: str | Path, to_state: str, *, reason: str
) -> dict[str, Any]:
    target = Path(attempt_dir)
    current = read_status(target / STATUS_FILE)
    record = StatusRecord.from_dict(current)
    if record.state != to_state:
        record.transition(to_state, reason=reason)
        write_status(target / STATUS_FILE, record)
    return {"attempt_id": record.attempt_id, "state": record.state}


def record_head_job(
    attempt_dir: str | Path,
    *,
    job_key: str,
    job_type: str,
    event_type: str,
    slurm_job_id: str | None,
    status: str | None,
    reason: str | None = None,
    dependency_job_ids: list[str] | None = None,
    resubmission_of_job_id: str | None = None,
    exit_code: str | None = None,
) -> dict[str, Any]:
    target = Path(attempt_dir)
    metadata = read_json(target / METADATA_FILE)
    event = new_job_event(
        job_key=job_key,
        job_type=job_type,
        event_type=event_type,
        attempt_id=str(metadata["attempt_id"]),
        fold=int(metadata["fold"]),
        slurm_job_id=slurm_job_id,
        status=status,
        reason=reason,
        dependency_job_ids=dependency_job_ids or [],
        resubmission_of_job_id=resubmission_of_job_id,
    )
    if exit_code is not None:
        event["exit_code"] = exit_code
    append_job_event(target / JOBS_FILE, event)
    return event


def _artifact_type(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "prediction" in name:
        return "predictions", name.rsplit(".", 1)[0]
    if "metric" in name:
        return "metrics", name.rsplit(".", 1)[0]
    if name.endswith((".joblib", ".pkl")):
        return "checkpoint", name.rsplit(".", 1)[0]
    if name.endswith((".json", ".jsonl", ".csv", ".db")):
        return "audit", name.rsplit(".", 1)[0]
    return "summary", name.rsplit(".", 1)[0]


def _iter_evidence_files(target: Path) -> Iterable[Path]:
    excluded = {
        METADATA_FILE,
        STATUS_FILE,
        JOBS_FILE,
        ARTIFACTS_FILE,
        EVALUATIONS_FILE,
        "run_config.yaml",
    }
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name.endswith(".tmp"):
            continue
        yield path


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise HeadTrackingError("head evaluation has no subject prediction rows")
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row.get("prediction", row.get("predicted_class"))) for row in rows]
    metrics = classification_metrics(labels, predictions)
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    metrics["negative_f1"] = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    metrics["invalid_qwen_outputs"] = int(sum(int(row.get("invalid_qwen_outputs", 0)) for row in rows))
    return metrics


def _build_evaluations(
    target: Path,
    *,
    config: dict[str, Any],
    predictions_path: Path,
    metrics_path: Path,
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    scientific = config.get("config") if isinstance(config.get("config"), dict) else config
    tracking = config.get("tracking") if isinstance(config.get("tracking"), dict) else scientific.get("tracking", {})
    rows = _prediction_rows(predictions_path)
    datasets = sorted({str(row.get("dataset", scientific.get("dataset", ""))).lower() for row in rows})
    eval_cfg = scientific.get("evaluation") or {}
    classifier = scientific.get("classifier") or {}
    backend = str(classifier.get("prediction_backend") or scientific.get("backend") or "")
    view = str(eval_cfg.get("evaluation_view") or EVALUATION_VIEW)
    aggregation = str(eval_cfg.get("aggregation") or "subject_level")
    split_name = str(eval_cfg.get("split_name") or scientific.get("stage") or "outer_holdout")
    split_protocol = str(eval_cfg.get("split_protocol") or "saved_split")
    result: list[dict[str, Any]] = []
    metrics_sha = sha256_file(metrics_path)
    for dataset in datasets:
        dataset_rows = [
            row for row in rows
            if str(row.get("dataset", scientific.get("dataset", ""))).lower() == dataset
        ]
        metrics = _metric_payload(dataset_rows)
        support = len({str(row["subject_id"]) for row in dataset_rows})
        eid = evaluation_id(
            attempt_id=str(tracking["attempt_id"]),
            fold=int(tracking["fold"]),
            dataset=dataset,
            split_name=split_name,
            split_protocol=split_protocol,
            checkpoint_role="best_model",
            checkpoint_path=checkpoint_path,
            backend=backend,
            evaluation_view=view,
            aggregation=aggregation,
            metric_namespace=METRIC_NAMESPACE,
            metrics_artifact_sha256=metrics_sha,
        )
        result.append(
            {
                "evaluation_id": eid,
                "dataset": dataset,
                "split_name": split_name,
                "split_protocol": split_protocol,
                "checkpoint_role": "best_model",
                "checkpoint_path": checkpoint_path,
                "backend": backend,
                "evaluation_view": view,
                "aggregation": aggregation,
                "metric_namespace": METRIC_NAMESPACE,
                "metrics_artifact_path": str(metrics_path.relative_to(target)),
                "predictions_artifact_path": str(predictions_path.relative_to(target)),
                "metrics": [
                    {"name": name, "value": float(metrics.get(name, 0.0)), "support": support}
                    for name in ("macro_f1", "positive_f1", "accuracy", "negative_f1")
                ],
                "locally_verified": False,
                "reportable": False,
                "warnings": [],
            }
        )
    return result


def materialize_head_evidence(
    attempt_dir: str | Path,
    *,
    predictions_path: str | Path,
    metrics_path: str | Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    run_config = yaml.safe_load((target / "run_config.yaml").read_text(encoding="utf-8")) or {}
    records: list[dict[str, Any]] = []
    for path in _iter_evidence_files(target):
        relative = path.relative_to(target).as_posix()
        artifact_type, role = _artifact_type(path)
        records.append(
            {
                "artifact_id": artifact_id(
                    attempt_id=sidecars.attempt_id,
                    fold=sidecars.fold,
                    role=role,
                    relative_path=relative,
                    artifact_sha256=sha256_file(path),
                ),
                "artifact_type": artifact_type,
                "role": role,
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "exists_on_mn5": True,
                "exists_locally": False,
                "locally_verified": False,
            }
        )
    artifact_doc = read_json(target / ARTIFACTS_FILE)
    existing_by_path = {str(item["path"]): item for item in artifact_doc.get("artifacts", [])}
    for item in records:
        old = existing_by_path.get(item["path"])
        if old is not None and old != item:
            raise HeadTrackingError(f"artifact identity changed for {item['path']}")
    artifact_doc["artifacts"] = [existing_by_path.get(item["path"], item) for item in records]
    write_json_atomic(target / ARTIFACTS_FILE, artifact_doc)

    predictions = Path(predictions_path).resolve()
    metrics = Path(metrics_path).resolve()
    if not predictions.is_file() or not metrics.is_file():
        raise HeadTrackingError("head metrics/predictions evidence is incomplete")
    evaluations = _build_evaluations(
        target,
        config=run_config,
        predictions_path=predictions,
        metrics_path=metrics,
        checkpoint_path=checkpoint_path,
    )
    evaluation_doc = read_json(target / EVALUATIONS_FILE)
    previous = {str(item["evaluation_id"]): item for item in evaluation_doc.get("evaluations", [])}
    for item in evaluations:
        old = previous.get(item["evaluation_id"])
        if old is not None and old != item:
            raise HeadTrackingError(f"evaluation identity changed for {item['evaluation_id']}")
    evaluation_doc["evaluations"] = [previous.get(item["evaluation_id"], item) for item in evaluations]
    write_json_atomic(target / EVALUATIONS_FILE, evaluation_doc)
    state = read_status(target / STATUS_FILE)
    if state["state"] == "RUNNING":
        record = StatusRecord.from_dict(state)
        record.transition("COMPLETED_ON_MN5", reason="head evidence materialized")
        write_status(target / STATUS_FILE, record)
    return {
        "attempt_id": sidecars.attempt_id,
        "artifacts": len(records),
        "evaluations": len(evaluations),
        "state": read_status(target / STATUS_FILE)["state"],
    }


def materialize_job_evidence(
    attempt_dir: str | Path,
    *,
    artifact_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Register compact non-evaluation evidence for train/postprocess jobs."""

    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    artifact_doc = read_json(target / ARTIFACTS_FILE)
    existing = {str(item["path"]): item for item in artifact_doc.get("artifacts", [])}
    selected: list[dict[str, Any]] = []
    for raw in artifact_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = target / path
        path = path.resolve()
        if not path.is_file() or target.resolve() not in path.parents:
            raise HeadTrackingError(f"job evidence path is not a file below the attempt: {raw}")
        relative = path.relative_to(target.resolve()).as_posix()
        digest = sha256_file(path)
        item = {
            "artifact_id": artifact_id(
                attempt_id=sidecars.attempt_id,
                fold=sidecars.fold,
                role=relative.rsplit("/", 1)[-1].rsplit(".", 1)[0],
                relative_path=relative,
                artifact_sha256=digest,
            ),
            "artifact_type": "audit" if path.suffix in {".json", ".jsonl", ".yaml"} else "summary",
            "role": relative.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            "path": relative,
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
            "exists_on_mn5": True,
            "exists_locally": False,
            "locally_verified": False,
        }
        old = existing.get(relative)
        if old is not None and old != item:
            raise HeadTrackingError(f"job artifact identity changed for {relative}")
        selected.append(old or item)
    if not selected:
        raise HeadTrackingError(
            "job completion cannot be materialized without compact evidence artifacts"
        )
    artifact_doc["artifacts"] = selected
    write_json_atomic(target / ARTIFACTS_FILE, artifact_doc)
    state = read_status(target / STATUS_FILE)
    if state["state"] == "RUNNING":
        record = StatusRecord.from_dict(state)
        record.transition("COMPLETED_ON_MN5", reason="job evidence materialized")
        write_status(target / STATUS_FILE, record)
    return {
        "attempt_id": sidecars.attempt_id,
        "artifacts": len(selected),
        "evaluations": 0,
        "state": read_status(target / STATUS_FILE)["state"],
    }


def _close(a: Any, b: Any) -> bool:
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def validate_head_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    """Recompute strict subject metrics and advance through local validation."""

    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    if sidecars.state not in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"}:
        raise HeadTrackingError(f"head validation requires completed evidence, got {sidecars.state}")
    issues = verify_modern_evidence_locally(sidecars)
    run_config = yaml.safe_load((target / "run_config.yaml").read_text(encoding="utf-8")) or {}
    expected_backend = str((run_config.get("config") or {}).get("classifier", {}).get("prediction_backend") or "")
    evaluations_doc = read_json(target / EVALUATIONS_FILE)
    artifacts_by_path = {str(item["path"]): item for item in read_json(target / ARTIFACTS_FILE).get("artifacts", [])}
    for evaluation in evaluations_doc.get("evaluations", []):
        prediction_path = target / str(evaluation["predictions_artifact_path"])
        rows = _prediction_rows(prediction_path)
        dataset_rows = [row for row in rows if str(row.get("dataset", "")).lower() == str(evaluation["dataset"]).lower()]
        computed = _metric_payload(dataset_rows)
        if str(evaluation.get("backend")) != expected_backend:
            issues.append(f"backend qualifier mismatch: {evaluation.get('backend')} != {expected_backend}")
        for metric in evaluation.get("metrics", []):
            name = str(metric.get("name"))
            if name in computed and not _close(metric.get("value"), computed[name]):
                issues.append(f"{evaluation['dataset']} {name} does not match subject-level recomputation")
        if str(evaluation.get("metric_namespace")) != METRIC_NAMESPACE:
            issues.append(f"wrong metric namespace: {evaluation.get('metric_namespace')}")
        if str(evaluation.get("evaluation_view")) != EVALUATION_VIEW:
            issues.append(f"wrong evaluation view: {evaluation.get('evaluation_view')}")
        for path in (evaluation.get("metrics_artifact_path"), evaluation.get("predictions_artifact_path")):
            if path not in artifacts_by_path:
                issues.append(f"evaluation artifact is not registered: {path}")
    if issues:
        return {"ok": False, "attempt_id": sidecars.attempt_id, "state": sidecars.state, "issues": issues}

    # Official evidence helpers only upgrade verification flags after hashes
    # match; they never alter metric content.
    from src.experiment_tracking.evidence import verify_artifacts_locally, verify_evaluations_locally

    verify_artifacts_locally(target)
    verify_evaluations_locally(target)
    record = StatusRecord.from_dict(read_status(target / STATUS_FILE))
    if record.state == "COMPLETED_ON_MN5":
        record.transition("SYNCED_LOCALLY", reason="head compact evidence synced locally")
    if record.state == "SYNCED_LOCALLY":
        record.transition("LOCALLY_VALIDATED", reason="head strict metrics recomputed locally")
    write_status(target / STATUS_FILE, record)
    return {"ok": True, "attempt_id": sidecars.attempt_id, "state": record.state, "issues": []}


def finish_head_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    events = read_jsonl(target / JOBS_FILE)
    successful = {
        str(event.get("job_key"))
        for event in events
        if event.get("event_type") == "COMPLETED"
        and event.get("status") == "COMPLETED"
        and str(event.get("exit_code", "0:0")).startswith("0:0")
    }
    required = set((yaml.safe_load((target / "run_config.yaml").read_text(encoding="utf-8")) or {}).get("tracking", {}).get("required_jobs", ["head"]))
    missing = sorted(required - successful)
    if missing:
        return {"ok": False, "state": sidecars.state, "next_action": f"missing successful jobs: {missing}"}
    record = StatusRecord.from_dict(read_status(target / STATUS_FILE))
    if record.state == "LOCALLY_VALIDATED":
        record.transition("REPORTABLE", reason="head evidence and job gates passed")
        write_status(target / STATUS_FILE, record)
    return {"ok": record.state == "REPORTABLE", "state": record.state}


def _successful_required_jobs(target: Path) -> tuple[set[str], set[str]]:
    run_config = yaml.safe_load((target / "run_config.yaml").read_text(encoding="utf-8")) or {}
    tracking = run_config.get("tracking") if isinstance(run_config.get("tracking"), dict) else {}
    required = set(tracking.get("required_jobs") or [])
    events = read_jsonl(target / JOBS_FILE)
    successful = {
        str(event.get("job_key"))
        for event in events
        if event.get("event_type") == "COMPLETED"
        and event.get("status") == "COMPLETED"
        and str(event.get("exit_code", "0:0")).startswith("0:0")
    }
    return required, successful


def validate_job_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    """Validate a compact train/postprocess attempt without inventing metrics."""

    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    if sidecars.state not in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"}:
        raise HeadTrackingError(
            f"job validation requires completed evidence, got {sidecars.state}"
        )
    required, successful = _successful_required_jobs(target)
    missing_jobs = sorted(required - successful)
    if missing_jobs:
        return {
            "ok": False,
            "attempt_id": sidecars.attempt_id,
            "state": sidecars.state,
            "issues": [f"missing successful required jobs: {missing_jobs}"],
        }
    if not sidecars.artifacts:
        return {
            "ok": False,
            "attempt_id": sidecars.attempt_id,
            "state": sidecars.state,
            "issues": ["attempt has no registered compact evidence artifacts"],
        }
    issues = verify_modern_evidence_locally(sidecars)
    if issues:
        return {"ok": False, "attempt_id": sidecars.attempt_id, "state": sidecars.state, "issues": issues}
    from src.experiment_tracking.evidence import verify_artifacts_locally

    verified = verify_artifacts_locally(target)
    if verified["verified_artifacts"] != verified["total_artifacts"]:
        return {
            "ok": False,
            "attempt_id": sidecars.attempt_id,
            "state": sidecars.state,
            "issues": [
                f"only {verified['verified_artifacts']}/{verified['total_artifacts']} compact artifacts verified locally"
            ],
        }
    record = StatusRecord.from_dict(read_status(target / STATUS_FILE))
    if record.state == "COMPLETED_ON_MN5":
        record.transition("SYNCED_LOCALLY", reason="compact evidence synced locally")
    if record.state == "SYNCED_LOCALLY":
        record.transition("LOCALLY_VALIDATED", reason="compact evidence hashes verified locally")
    write_status(target / STATUS_FILE, record)
    return {"ok": True, "attempt_id": sidecars.attempt_id, "state": record.state, "issues": []}


def finish_job_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    """Advance a non-evaluation managed attempt to REPORTABLE after its job gate."""

    target = Path(attempt_dir)
    sidecars = _sidecar(target)
    required, successful = _successful_required_jobs(target)
    missing = sorted(required - successful)
    if missing:
        return {"ok": False, "state": sidecars.state, "next_action": f"missing successful jobs: {missing}"}
    record = StatusRecord.from_dict(read_status(target / STATUS_FILE))
    if record.state == "LOCALLY_VALIDATED":
        record.transition("REPORTABLE", reason="compact evidence and job gates passed")
        write_status(target / STATUS_FILE, record)
    return {"ok": record.state == "REPORTABLE", "state": record.state}
