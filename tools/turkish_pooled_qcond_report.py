#!/usr/bin/env python3
"""Build the deterministic provenance-complete Turkish pooled report.

The report consumes only locally collected, locally verified REPORTABLE
evidence.  It recomputes all three views from compact sample predictions so
condition breakdowns cannot be made by filtering already aggregated rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import (
    TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
    TURKISH_POOLED_TEXT_PAIR_POLICY,
)
from src.turkish_pooled_qcond import (
    EVALUATION_BACKEND,
    EVALUATION_VIEW,
    EXPERIMENT_ID,
    FOLDS,
    GROUP_ID,
    METRIC_NAMESPACE,
    PAIR_POLICY,
    REMOTE_PROJECT_ROOT,
    TRAINING_SEEDS,
    load_cells,
)
from tools.score_turkish_pooled import ScoreError, score_rows


REPORT_SCHEMA = "audiollm.turkish_pooled_qcond_report.v1"
BASELINE_STATUS = (
    "baseline unavailable or not reportable: separate-condition baseline "
    "provenance is incomplete or ambiguous; no archival value was copied"
)
ROUTES = ("teacher_forced", "logreg", "xgb_optuna100")
EXPECTED_FOLD_ROWS = 10 * 3 * 5 * 3
EXPECTED_SEED_ROWS = 10 * 3 * 3
EXPECTED_SUMMARY_ROWS = 10 * 3
FORBIDDEN_KEYS = {
    "transcript", "transcript_original", "prompt_text", "prompt_user_text",
    "prompt_system_text", "training_text", "audio_path", "audio_paths",
}


class ReportError(ValueError):
    """Raised when compact evidence cannot support the locked report."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON evidence {path}: {exc}") from exc


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReportError(f"prediction artifact is missing: {path}")
    try:
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                values = list(csv.DictReader(handle))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        raise ReportError(f"cannot read prediction artifact {path}: {exc}") from exc
    if not values or not all(isinstance(value, dict) for value in values):
        raise ReportError(f"prediction artifact is empty or malformed: {path}")
    result = [dict(value) for value in values]
    for row in result:
        leaked = sorted(FORBIDDEN_KEYS.intersection(row))
        if leaked:
            raise ReportError(f"privacy-sensitive prediction fields found in {path}: {leaked}")
    return result


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReportError(f"cannot hash evidence {path}: {exc}") from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _local_from_remote(remote: str | Path) -> Path:
    path = Path(remote)
    try:
        return PROJECT_ROOT / path.relative_to(REMOTE_PROJECT_ROOT)
    except ValueError as exc:
        raise ReportError(f"remote evidence path is outside the canonical project root: {path}") from exc


def _local_runtime_path(plan: dict[str, Any], remote: str | Path) -> Path:
    runtime = Path(str(plan.get("runtime_root", "")))
    path = Path(remote)
    try:
        relative = path.relative_to(runtime)
    except ValueError as exc:
        raise ReportError(f"runtime evidence is outside the locked runtime root: {path}") from exc
    return PROJECT_ROOT / "outputs" / "turkish_pooled_qcond" / EXPERIMENT_ID / str(plan["stage"]) / "runtime" / relative


def _sidecar(path: Path, name: str) -> dict[str, Any]:
    value = _json(path / name)
    if not isinstance(value, dict):
        raise ReportError(f"sidecar is not an object: {path / name}")
    return value


def _events(path: Path) -> list[dict[str, Any]]:
    target = path / "jobs.jsonl"
    if not target.is_file():
        raise ReportError(f"missing job history: {target}")
    result: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(f"malformed job event: {target}")
            result.append(value)
    if not result:
        raise ReportError(f"empty job history: {target}")
    return result


_SLURM_JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


def _slurm_job_ids(events: Iterable[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract canonical scheduler IDs while preserving malformed evidence by hash.

    A submit-side correction can leave an append-only raw field containing both
    wait diagnostics and the real numeric ID. Keep the original sidecar intact,
    report the numeric ID once, and expose a non-sensitive correction record so
    the anomaly is not silently discarded.
    """
    canonical: set[str] = set()
    noncanonical: list[dict[str, Any]] = []
    for event in events:
        raw = event.get("slurm_job_id")
        if raw is None:
            continue
        raw_text = str(raw)
        candidates = [
            value.strip()
            for value in raw_text.splitlines()
            if _SLURM_JOB_ID_PATTERN.fullmatch(value.strip())
        ]
        canonical.update(candidates)
        if not _SLURM_JOB_ID_PATTERN.fullmatch(raw_text.strip()):
            noncanonical.append(
                {
                    "attempt_id": event.get("attempt_id"),
                    "fold": event.get("fold"),
                    "job_key": event.get("job_key"),
                    "job_type": event.get("job_type"),
                    "event_id": event.get("event_id"),
                    "raw_value_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "resolved_job_ids": sorted(set(candidates)),
                }
            )
    return sorted(canonical), noncanonical


def _terminal_job(events: Iterable[dict[str, Any]], *, job_type: str, job_key: str | None = None) -> dict[str, Any]:
    matches = [
        event for event in events
        if str(event.get("job_type")) == job_type
        and (job_key is None or str(event.get("job_key")) == job_key)
        and event.get("event_type") == "COMPLETED"
        and str(event.get("status")) == "COMPLETED"
        and str(event.get("exit_code", "0:0")).startswith("0:0")
    ]
    if not matches:
        raise ReportError(f"no successful {job_type} terminal event")
    return matches[-1]


def _failure_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: event.get(key)
            for key in ("event_type", "job_type", "job_key", "slurm_job_id", "status", "reason", "exit_code", "resubmission_of_job_id")
        }
        for event in events
        if event.get("event_type") in {"FAILED", "CANCELLED"}
    ]


def _artifact(attempt: Path, relative: str) -> dict[str, Any]:
    target = attempt / relative
    if not target.is_file():
        raise ReportError(f"locally collected artifact is missing: {target}")
    artifacts = _sidecar(attempt, "artifacts.json").get("artifacts") or []
    matches = [item for item in artifacts if str(item.get("path")) == relative]
    if len(matches) != 1:
        raise ReportError(f"artifact is not uniquely registered in artifacts.json: {attempt / relative}")
    item = matches[0]
    digest = _sha256(target)
    if str(item.get("sha256")) != digest:
        raise ReportError(f"artifact hash mismatch: {target}")
    if item.get("exists_locally") is not True or item.get("locally_verified") is not True:
        raise ReportError(f"artifact is not locally verified: {target}")
    return {"path": str(target), "sha256": digest, "artifact_id": item.get("artifact_id")}


def _evaluation(attempt: Path, *, backend: str, aggregation: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _sidecar(attempt, "status.json")
    if status.get("state") != "REPORTABLE":
        raise ReportError(f"attempt is not REPORTABLE: {attempt} ({status.get('state')})")
    document = _sidecar(attempt, "evaluations.json")
    records = document.get("evaluations") or []
    matches = [
        record for record in records
        if record.get("backend") == backend
        and record.get("evaluation_view") == EVALUATION_VIEW
        and record.get("metric_namespace") == METRIC_NAMESPACE
        and record.get("locally_verified") is True
        and record.get("reportable") is True
        and (aggregation is None or record.get("aggregation") == aggregation)
    ]
    if len(matches) != 1:
        raise ReportError(f"expected exactly one qualified evaluation in {attempt}, got {len(matches)}")
    record = matches[0]
    metrics_rel = str(record.get("metrics_artifact_path", ""))
    predictions_rel = str(record.get("predictions_artifact_path", ""))
    if not metrics_rel or not predictions_rel:
        raise ReportError(f"evaluation paths are incomplete: {attempt}")
    metrics_artifact = _artifact(attempt, metrics_rel)
    predictions_artifact = _artifact(attempt, predictions_rel)
    metrics_payload = _json(attempt / metrics_rel)
    if not isinstance(metrics_payload, dict):
        raise ReportError(f"evaluation metrics are not an object: {attempt / metrics_rel}")
    return record, {
        "metrics": metrics_artifact,
        "predictions": predictions_artifact,
        "metrics_payload": metrics_payload,
    }


def _sample_artifact(attempt: Path, predictions_rel: str) -> dict[str, Any]:
    subject_path = attempt / predictions_rel
    parent = subject_path.parent
    for name in ("predictions_sample_level.jsonl", "predictions_sample_level.csv"):
        candidate = parent / name
        if candidate.is_file():
            relative = candidate.relative_to(attempt).as_posix()
            return _artifact(attempt, relative)
    raise ReportError(f"sample-level prediction artifact is missing beside {subject_path}")


def _condition_coverage(rows: list[dict[str, Any]], *, modality: str) -> None:
    expected = {"pos_only_t17", "negative_only_t17"}
    by_subject: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, int] = {}
    for row in rows:
        subject = str(row.get("subject_id", "")).strip()
        condition = str(row.get("question_condition", "")).strip()
        if not subject or condition not in expected:
            raise ReportError("pooled prediction rows must carry both exact question conditions")
        by_subject[subject].add(condition)
        label = int(row["label"])
        if label not in (0, 1):
            raise ReportError("pooled prediction rows contain a non-binary label")
        if subject in labels and labels[subject] != label:
            raise ReportError(f"subject label differs across question conditions: {subject}")
        labels[subject] = label
    if not by_subject or any(conditions != expected for conditions in by_subject.values()):
        raise ReportError("every pooled subject must have both question conditions in the raw artifact")
    if modality == "text_only":
        for subject in sorted(by_subject):
            counts = {condition: 0 for condition in expected}
            for row in rows:
                if str(row["subject_id"]) == subject:
                    counts[str(row["question_condition"]).strip()] += 1
            if counts != {"pos_only_t17": 1, "negative_only_t17": 1}:
                raise ReportError(f"text pooled subject does not have exactly one row per condition: {subject}")


def _view_score(rows: list[dict[str, Any]], *, route: str, modality: str, view: str, backend: str) -> dict[str, Any]:
    try:
        return score_rows(rows, route=route, modality=modality, view=view, backend=backend)
    except (ScoreError, ValueError, KeyError, TypeError) as exc:
        raise ReportError(f"cannot recompute {route}/{modality}/{view}: {exc}") from exc


def _view_payload(score: dict[str, Any], *, rows: list[dict[str, Any]], route: str, view: str) -> dict[str, Any]:
    metrics = score.get("metrics") or {}
    for key in ("binary_strict_macro_f1", "binary_strict_positive_f1", "num_subjects"):
        if key not in metrics:
            raise ReportError(f"recomputed score lacks {key}: {route}/{view}")
    invalid_components = 0
    if route == "teacher_forced":
        invalid_components = sum(
            int(
                not (
                    bool(row.get("teacher_forced_valid"))
                    and int(row.get("teacher_forced_prediction", -1)) in (0, 1)
                )
            )
            for row in rows
        )
    return {
        "macro_f1": float(metrics["binary_strict_macro_f1"]),
        "positive_f1": float(metrics["binary_strict_positive_f1"]),
        "num_subjects": int(metrics["num_subjects"]),
        "invalid_subjects": int(metrics.get("invalid_subjects", metrics.get("invalid_count", 0))),
        "invalid_component_predictions": int(invalid_components),
        "aggregation_policy": metrics.get("aggregation_policy"),
        "metric_namespace": "headline/binary_strict",
        "view": view,
    }


def _source_metadata(attempt: Path, *, plan: dict[str, Any], expected_attempt_id: str, seed: int, fold: int) -> tuple[dict[str, Any], str, str]:
    metadata = _sidecar(attempt, "metadata.json")
    if str(metadata.get("attempt_id")) != expected_attempt_id:
        raise ReportError(f"attempt identity mismatch: {attempt}")
    if str(metadata.get("group_id")) != GROUP_ID:
        raise ReportError(f"group identity mismatch: {attempt}")
    if int(metadata.get("seed", -1)) != seed or int(metadata.get("fold", -1)) != fold:
        raise ReportError(f"seed/fold mismatch: {attempt}")
    source = metadata.get("source") or {}
    if source.get("git_commit") != plan.get("source_git_sha"):
        raise ReportError(f"source Git SHA mismatch: {attempt}")
    if source.get("deployment_id") != plan.get("deployment_id"):
        raise ReportError(f"deployment identity mismatch: {attempt}")
    config_path = attempt / "run_config.yaml"
    if not config_path.is_file():
        raise ReportError(f"run_config.yaml is missing: {attempt}")
    return metadata, str(config_path), _sha256(config_path)


def _language_for_cell(cell: Any) -> str:
    return cell.language_token


def _runtime_audit(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("stage") != "production":
        raise ReportError("only a production submission can produce the report")
    audit_remote = str(plan.get("preflight_audit_path") or (Path(str(plan["runtime_root"])) / "preflight" / "production.json"))
    audit_path = _local_runtime_path(plan, audit_remote)
    audit = _json(audit_path)
    if audit.get("status") != "passed" or audit.get("group_id") != GROUP_ID or audit.get("stage") != "production":
        raise ReportError(f"production preflight audit is not passed: {audit_path}")
    outputs = audit.get("outputs") or {}
    if set(outputs) != {"native", "english"}:
        raise ReportError("production preflight does not contain native and English outputs")
    runtime_records: dict[str, Any] = {"audit": {"path": str(audit_path), "sha256": _sha256(audit_path)}}
    for language in ("native", "english"):
        output = outputs[language]
        manifest = audit_path.parents[1] / "manifests" / language / "turkish_manifest.jsonl"
        split = audit_path.parents[1] / "splits" / language / "turkish_folds.json"
        if _sha256(manifest) != str(output.get("manifest_sha256")):
            raise ReportError(f"local production manifest hash mismatch: {manifest}")
        if _sha256(split) != str(output.get("folds_sha256")):
            raise ReportError(f"local production split hash mismatch: {split}")
        if int(output.get("row_count", -1)) != 2221 or int(output.get("subject_count", -1)) != 120:
            raise ReportError(f"production preflight count mismatch: {language}")
        if output.get("condition_counts") != {"negative_only_t17": 1170, "pos_only_t17": 1051}:
            raise ReportError(f"production preflight condition mismatch: {language}")
        runtime_records[language] = {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "manifest_canonical_hash": output.get("manifest_hash"),
            "split_path": str(split),
            "split_sha256": _sha256(split),
            "canonical_fold_hash": output.get("fold_hash"),
            "row_count": int(output["row_count"]),
            "subject_count": int(output["subject_count"]),
            "condition_counts": dict(output["condition_counts"]),
        }
    return audit, runtime_records


def _route_backend(cell: Any, route: str) -> str:
    if route == "teacher_forced":
        return EVALUATION_BACKEND
    if route == "logreg":
        return "gemma4_hidden_logreg_raw" if cell.backbone == "gemma4" else "qwen_hidden_logreg_raw"
    if route == "xgb_optuna100":
        return "gemma4_hidden_xgb_optuna100" if cell.backbone == "gemma4" else "qwen_hidden_xgb_optuna100"
    raise ReportError(f"unknown route: {route}")


def _attempt_route(
    *, plan: dict[str, Any], backbone: dict[str, Any], cell: Any, route: str,
    runtime: dict[str, Any], provenance: dict[str, Any], execution: dict[str, Any],
) -> dict[str, Any]:
    fold = int(backbone["fold"])
    seed = int(backbone["seed"])
    fold_dir = _local_from_remote(str(backbone["fold_dir"]))
    if route == "teacher_forced":
        attempt = fold_dir
        attempt_id = str(backbone["backbone_attempt_id"])
        job_types = (("train", "train"), ("evaluation", "best_eval"))
        expected_aggregation = "response_subject" if cell.modality in {"audio_only", "audio_text"} else "subject_level"
    else:
        attempt_id = str(backbone["logreg_attempt_id"] if route == "logreg" else backbone["xgb_attempt_id"])
        attempt = fold_dir / attempt_id
        job_types = (("hidden_extraction", "head"),) if route == "logreg" else (("hidden_classifier", "head"),)
        expected_aggregation = "subject_level"
    metadata, config_path, config_sha = _source_metadata(
        attempt, plan=plan, expected_attempt_id=attempt_id, seed=seed, fold=fold,
    )
    events = _events(attempt)
    successful_jobs = [_terminal_job(events, job_type=job_type, job_key=job_key) for job_type, job_key in job_types]
    execution["planned"] += 1
    execution["locally_validated"] += 1
    execution["reportable"] += 1
    for event in successful_jobs:
        execution["successful"] += 1
    failures = _failure_events(events)
    execution["failed"] += sum(item["event_type"] == "FAILED" for item in failures)
    execution["cancelled"] += sum(item["event_type"] == "CANCELLED" for item in failures)
    execution["retried"] += sum(bool(item.get("resubmission_of_job_id")) for item in failures)

    backend = _route_backend(cell, route)
    record, artifacts = _evaluation(attempt, backend=backend, aggregation=expected_aggregation)
    sample_artifact = _sample_artifact(attempt, str(record["predictions_artifact_path"]))
    rows = _rows(Path(sample_artifact["path"]))
    _condition_coverage(rows, modality=cell.modality)
    views: dict[str, dict[str, Any]] = {}
    for view in ("positive", "negative", "combined"):
        view_rows = rows if view == "combined" else [
            row for row in rows
            if str(row.get("question_condition", "")).strip() == ("pos_only_t17" if view == "positive" else "negative_only_t17")
        ]
        views[view] = _view_payload(
            _view_score(view_rows, route=route, modality=cell.modality, view=view, backend=backend),
            rows=view_rows, route=route, view=view,
        )
    combined_record_metrics = {
        str(item.get("name")): float(item.get("value"))
        for item in (record.get("metrics") or [])
        if item.get("name") in {"macro_f1", "positive_f1", "binary_strict_macro_f1", "binary_strict_positive_f1"}
    }
    if combined_record_metrics:
        for record_name, score_name in (("macro_f1", "macro_f1"), ("positive_f1", "positive_f1"), ("binary_strict_macro_f1", "macro_f1"), ("binary_strict_positive_f1", "positive_f1")):
            if record_name in combined_record_metrics and abs(combined_record_metrics[record_name] - views["combined"][score_name]) > 1e-9:
                raise ReportError(f"evaluation record disagrees with local strict recomputation: {attempt}")

    language = _language_for_cell(cell)
    hashes = metadata.get("hashes") or {}
    if hashes.get("manifest_sha256") not in {runtime[language]["manifest_canonical_hash"], runtime[language]["manifest_sha256"]}:
        raise ReportError(f"manifest hash qualifier mismatch: {attempt}")
    if hashes.get("split_sha256") not in {runtime[language]["split_sha256"]}:
        raise ReportError(f"split hash qualifier mismatch: {attempt}")
    source = metadata.get("source") or {}
    jobs, noncanonical_job_events = _slurm_job_ids(events)
    provenance_payload = {
        "group_id": GROUP_ID,
        "logical_run_name": metadata.get("logical_run_name"),
        "attempt_id": attempt_id,
        "seed": seed,
        "fold": fold,
        "cell_id": cell.cell_id,
        "backbone": cell.backbone,
        "modality": cell.modality,
        "transcript_condition": cell.transcript_condition,
        "branch": source.get("git_branch"),
        "git_sha": source.get("git_commit"),
        "deployment_id": source.get("deployment_id"),
        "deployed_source_sha256": source.get("deployed_source_sha256"),
        "config_path": config_path,
        "config_sha256": config_sha,
        "manifest_path": runtime[language]["manifest_path"],
        "manifest_sha256": runtime[language]["manifest_sha256"],
        "manifest_canonical_hash": runtime[language]["manifest_canonical_hash"],
        "split_path": runtime[language]["split_path"],
        "split_sha256": runtime[language]["split_sha256"],
        "canonical_fold_hash": runtime[language]["canonical_fold_hash"],
        "checkpoint_role": "best_model",
        "checkpoint_path": record.get("checkpoint_path"),
        "metric_name": "binary_strict_macro_f1",
        "metric_namespace": METRIC_NAMESPACE,
        "backend": record.get("backend"),
        "evaluation_view": record.get("evaluation_view"),
        "aggregation": record.get("aggregation"),
        "split_protocol": (metadata.get("config") or {}).get("evaluation", {}).get("split_protocol", "saved_split"),
        "evaluation_id": record.get("evaluation_id"),
        "evaluation_metrics_artifact": artifacts["metrics"],
        "evaluation_predictions_artifact": artifacts["predictions"],
        "sample_predictions_artifact": sample_artifact,
        "slurm_job_ids": jobs,
        "noncanonical_slurm_job_events": noncanonical_job_events,
        "failure_events": failures,
        "locally_verified": True,
        "reportable": True,
        "views": {name: {"aggregation_policy": value["aggregation_policy"]} for name, value in views.items()},
    }
    provenance_key = "prov-" + _canonical_sha(provenance_payload)[:24]
    provenance[provenance_key] = provenance_payload

    return {
        "group_id": GROUP_ID,
        "cell_id": cell.cell_id,
        "model": "Gemma 4" if cell.backbone == "gemma4" else "Qwen",
        "model_token": cell.backbone,
        "modality": cell.modality,
        "transcript_condition": cell.transcript_condition,
        "input_cell": f"{cell.modality}/{cell.language_token}",
        "route": route,
        "backend": backend,
        "seed": seed,
        "fold": fold,
        "positive_question_macro_f1": views["positive"]["macro_f1"],
        "negative_question_macro_f1": views["negative"]["macro_f1"],
        "combined_macro_f1": views["combined"]["macro_f1"],
        "positive_question_positive_f1": views["positive"]["positive_f1"],
        "negative_question_positive_f1": views["negative"]["positive_f1"],
        "combined_positive_f1": views["combined"]["positive_f1"],
        "positive_question_invalid_subjects": views["positive"]["invalid_subjects"],
        "negative_question_invalid_subjects": views["negative"]["invalid_subjects"],
        "combined_invalid_subjects": views["combined"]["invalid_subjects"],
        "positive_question_invalid_components": views["positive"]["invalid_component_predictions"],
        "negative_question_invalid_components": views["negative"]["invalid_component_predictions"],
        "combined_invalid_components": views["combined"]["invalid_component_predictions"],
        "positive_question_support": views["positive"]["num_subjects"],
        "negative_question_support": views["negative"]["num_subjects"],
        "combined_support": views["combined"]["num_subjects"],
        "aggregation_positive": views["positive"]["aggregation_policy"],
        "aggregation_negative": views["negative"]["aggregation_policy"],
        "aggregation_combined": views["combined"]["aggregation_policy"],
        "baseline_positive_macro_f1": None,
        "baseline_negative_macro_f1": None,
        "pooled_minus_positive_baseline_macro_f1": None,
        "pooled_minus_negative_baseline_macro_f1": None,
        "baseline_positive_status": BASELINE_STATUS,
        "baseline_negative_status": BASELINE_STATUS,
        "attempt_id": attempt_id,
        "logical_run_name": metadata.get("logical_run_name"),
        "evaluation_id": record.get("evaluation_id"),
        "run_config_path": config_path,
        "run_config_sha256": config_sha,
        "provenance_key": provenance_key,
        "locally_verified": True,
        "reportable": True,
    }


def _mean_sd(rows: list[dict[str, Any]], prefix: str) -> tuple[float, float]:
    values = [float(row[prefix]) for row in rows]
    if len(values) != 3:
        raise ReportError(f"expected three seed values for {prefix}")
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _aggregate_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 5 or {int(row["fold"]) for row in rows} != set(FOLDS):
        raise ReportError("each seed result must contain exactly five folds")
    first = rows[0]
    result = {key: first.get(key) for key in ("group_id", "cell_id", "model", "model_token", "modality", "transcript_condition", "input_cell", "route", "backend", "seed")}
    result.update({
        "fold_count": 5,
        "positive_question_macro_f1_fold_mean": statistics.mean(float(row["positive_question_macro_f1"]) for row in rows),
        "negative_question_macro_f1_fold_mean": statistics.mean(float(row["negative_question_macro_f1"]) for row in rows),
        "combined_macro_f1_fold_mean": statistics.mean(float(row["combined_macro_f1"]) for row in rows),
        "positive_question_positive_f1_fold_mean": statistics.mean(float(row["positive_question_positive_f1"]) for row in rows),
        "negative_question_positive_f1_fold_mean": statistics.mean(float(row["negative_question_positive_f1"]) for row in rows),
        "combined_positive_f1_fold_mean": statistics.mean(float(row["combined_positive_f1"]) for row in rows),
        "invalid_subjects": sum(int(row["combined_invalid_subjects"]) for row in rows),
        "invalid_components": sum(int(row["combined_invalid_components"]) for row in rows),
        "baseline_positive_macro_f1": None,
        "baseline_negative_macro_f1": None,
        "pooled_minus_positive_baseline_macro_f1": None,
        "pooled_minus_negative_baseline_macro_f1": None,
        "baseline_positive_status": BASELINE_STATUS,
        "baseline_negative_status": BASELINE_STATUS,
        "provenance_keys": ",".join(sorted(str(row["provenance_key"]) for row in rows)),
        "pooled_oof_status": "not reported; fold-mean is the locked headline",
    })
    return result


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3 or {int(row["seed"]) for row in rows} != set(TRAINING_SEEDS):
        raise ReportError("each main summary must contain exactly three seeds")
    first = rows[0]
    result = {key: first.get(key) for key in ("group_id", "cell_id", "model", "model_token", "modality", "transcript_condition", "input_cell", "route", "backend")}
    for metric in (
        "positive_question_macro_f1_fold_mean", "negative_question_macro_f1_fold_mean", "combined_macro_f1_fold_mean",
        "positive_question_positive_f1_fold_mean", "negative_question_positive_f1_fold_mean", "combined_positive_f1_fold_mean",
    ):
        mean, sd = _mean_sd(rows, metric)
        result[f"{metric.replace('_fold_mean', '_seed_mean')}"] = mean
        result[f"{metric.replace('_fold_mean', '_seed_sample_sd')}"] = sd
    result.update({
        "seed_count": 3,
        "fold_count": 15,
        "baseline_positive_macro_f1": None,
        "baseline_negative_macro_f1": None,
        "pooled_minus_positive_baseline_macro_f1": None,
        "pooled_minus_negative_baseline_macro_f1": None,
        "baseline_positive_status": BASELINE_STATUS,
        "baseline_negative_status": BASELINE_STATUS,
        "provenance_keys": ",".join(sorted({key for row in rows for key in str(row["provenance_keys"]).split(",") if key})),
        "researcher_conclusion": "",
        "pooled_oof_status": "not reported; fold-mean is the locked headline",
    })
    return result


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise ReportError(f"refusing to overwrite incompatible report artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    _write_once(path, buffer.getvalue())


def generate_report(plan_path: Path, output_dir: Path) -> dict[str, Any]:
    plan = _json(plan_path)
    if plan.get("group_id") != GROUP_ID or plan.get("experiment_id") != EXPERIMENT_ID:
        raise ReportError("submission plan identity does not match the locked pooled campaign")
    if plan.get("stage") != "production":
        raise ReportError("report generation requires the production submission plan")
    audit, runtime = _runtime_audit(plan)
    cells = {cell.cell_id: cell for cell in load_cells(PROJECT_ROOT)}
    if len(plan.get("backbones", [])) != 150:
        raise ReportError(f"production submission plan has {len(plan.get('backbones', []))} backbones; expected 150")
    fold_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    execution = {
        route: {"planned": 0, "successful": 0, "failed": 0, "cancelled": 0, "retried": 0, "locally_validated": 0, "reportable": 0}
        for route in ROUTES
    }
    for backbone in plan["backbones"]:
        cell = cells.get(str(backbone.get("cell_id")))
        if cell is None:
            raise ReportError(f"unknown pooled cell in submission plan: {backbone.get('cell_id')}")
        for route in ROUTES:
            fold_rows.append(
                _attempt_route(
                    plan=plan, backbone=backbone, cell=cell, route=route,
                    runtime=runtime, provenance=provenance, execution=execution[route],
                )
            )
    fold_rows.sort(key=lambda row: (row["model_token"], row["cell_id"], row["modality"], row["transcript_condition"], row["route"], int(row["seed"]), int(row["fold"])))
    if len(fold_rows) != EXPECTED_FOLD_ROWS:
        raise ReportError(f"expected {EXPECTED_FOLD_ROWS} fold rows, got {len(fold_rows)}")
    by_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_seed[(row["cell_id"], row["route"], int(row["seed"]))].append(row)
    seed_rows = [_aggregate_seed(rows) for _, rows in sorted(by_seed.items(), key=lambda item: tuple(str(value) for value in item[0]))]
    if len(seed_rows) != EXPECTED_SEED_ROWS:
        raise ReportError(f"expected {EXPECTED_SEED_ROWS} seed rows, got {len(seed_rows)}")
    by_summary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_summary[(str(row["cell_id"]), str(row["route"]))].append(row)
    summary_rows = [_aggregate_summary(rows) for _, rows in sorted(by_summary.items())]
    if len(summary_rows) != EXPECTED_SUMMARY_ROWS:
        raise ReportError(f"expected {EXPECTED_SUMMARY_ROWS} summary rows, got {len(summary_rows)}")

    terminal_counts = {"COMPLETED": 0, "FAILED": 0, "CANCELLED": 0}
    slurm_ids: list[str] = []
    noncanonical_job_events: list[dict[str, Any]] = []
    for item in provenance.values():
        slurm_ids.extend(str(value) for value in item.get("slurm_job_ids", []))
        noncanonical_job_events.extend(item.get("noncanonical_slurm_job_events", []))
        for failure in item.get("failure_events", []):
            terminal_counts[str(failure.get("event_type"))] = terminal_counts.get(str(failure.get("event_type")), 0) + 1
    terminal_counts["COMPLETED"] = sum(item["successful"] for item in execution.values())
    unique_slurm_ids = sorted(set(slurm_ids))
    if terminal_counts["COMPLETED"] != 600:
        raise ReportError(f"production terminal successful job count is {terminal_counts['COMPLETED']}; expected 600")
    job_audit = {
        "schema_version": "audiollm.turkish_pooled_qcond_job_audit.v1",
        "group_id": GROUP_ID,
        "experiment_id": EXPERIMENT_ID,
        "deployment_id": plan.get("deployment_id"),
        "source_git_sha": plan.get("source_git_sha"),
        "expected_counts": plan.get("expected_counts"),
        "execution_by_route": execution,
        "terminal_counts": terminal_counts,
        "unique_slurm_job_count": len(unique_slurm_ids),
        "slurm_job_ids": unique_slurm_ids,
        "noncanonical_slurm_job_events": sorted(
            noncanonical_job_events,
            key=lambda item: (
                str(item.get("attempt_id")),
                int(item.get("fold", -1)),
                str(item.get("event_id")),
            ),
        ),
        "resubmission_events": [failure for item in provenance.values() for failure in item.get("failure_events", []) if failure.get("resubmission_of_job_id")],
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "group_id": GROUP_ID,
        "experiment_id": EXPERIMENT_ID,
        "deployment_id": plan.get("deployment_id"),
        "source_git_sha": plan.get("source_git_sha"),
        "evaluation": {
            "backend": EVALUATION_BACKEND,
            "view": EVALUATION_VIEW,
            "metric_namespace": METRIC_NAMESPACE,
            "headline_metric": "binary_strict_macro_f1",
            "secondary_metric": "binary_strict_positive_f1",
            "aggregation": "fold_mean_subject_level",
            "checkpoint_role": "best_model",
            "text_pair_policy": PAIR_POLICY,
            "audio_policy": TURKISH_POOLED_AUDIO_AGGREGATION_POLICY,
        },
        "baseline_comparison": {
            "status": BASELINE_STATUS,
            "positive_only_values": None,
            "negative_only_values": None,
            "matched_deltas": None,
        },
        "limitations": [
            "positive transcripts are unreviewed while negative transcripts are reviewed",
            "train_val is not an untouched test protocol",
            "there is no tag-ablation control",
            "audio combined aggregation is source-unit weighted, while text combines one full-transcript example per condition",
            "post-hoc head fitting weights differ from backbone hierarchical loss weights",
            BASELINE_STATUS,
        ],
        "researcher_conclusion": "",
        "tables": {"summary": summary_rows, "seed_results": seed_rows, "fold_results": fold_rows},
        "provenance_index": provenance,
        "runtime_evidence": runtime,
        "preflight_audit": {"path": runtime["audit"]["path"], "sha256": runtime["audit"]["sha256"], "audit_sha256_field": audit.get("audit_sha256")},
        "job_audit": job_audit,
        "validation": {
            "status": "passed",
            "summary_rows": len(summary_rows),
            "seed_rows": len(seed_rows),
            "fold_rows": len(fold_rows),
            "provenance_rows": len(provenance),
            "local_reportable_routes": len(provenance),
        },
    }
    report["report_sha256"] = _canonical_sha(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "seed_results.csv", seed_rows)
    _write_csv(output_dir / "fold_results.csv", fold_rows)
    _write_once(output_dir / "provenance_index.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "job_audit.json", json.dumps(job_audit, indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "report_validation.json", json.dumps(report["validation"], indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = [
        "# Turkish pooled question-conditioned report",
        "",
        f"Group: `{GROUP_ID}`  ",
        f"Deployment: `{plan.get('deployment_id')}`  ",
        f"Source SHA: `{plan.get('source_git_sha')}`  ",
        "",
        "All displayed metrics are recomputed from locally collected REPORTABLE compact evidence. The headline aggregation is a five-fold subject-level mean within each seed, followed by the mean and sample standard deviation across the three seeds.",
        "",
        "Text-only uses `turkish_pooled_text_pair_mean_margin_strict_v1`: one positive-question and one negative-question example per subject, with the mean of the two component margins and INVALID counted as wrong. Audio-bearing views filter raw rows by question condition first and use the unchanged response/source-unit hierarchy; the combined score weights source units rather than forcing equal condition-pile weight.",
        "",
        f"Separate-condition baseline comparison: `{BASELINE_STATUS}`. Baseline metrics and matched deltas are blank.",
        "",
        "The researcher conclusion is intentionally blank.",
        "",
        "## Report tables",
        "",
        "- [summary.csv](summary.csv) — 30 cell/route summaries",
        "- [seed_results.csv](seed_results.csv) — 90 three-fold-mean seed records",
        "- [fold_results.csv](fold_results.csv) — 450 route/fold records with positive, negative, and combined views",
        "- [provenance_index.json](provenance_index.json)",
        "- [job_audit.json](job_audit.json)",
        "",
        "## Limitations",
        "",
    ] + [f"- {item}" for item in report["limitations"]]
    _write_once(output_dir / "report.md", "\n".join(markdown) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = generate_report(args.plan, args.output_dir)
    except (OSError, ReportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["validation"]["status"], "report": str(args.output_dir / "report.json"), "report_sha256": report["report_sha256"], "row_counts": report["validation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
