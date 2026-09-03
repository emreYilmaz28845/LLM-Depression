#!/usr/bin/env python3
"""Generate the deterministic, provenance-complete Turkish campaign report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.turkish_question_condition import (
    EVALUATION_BACKEND,
    EVALUATION_VIEW,
    EXPERIMENT_ID,
    GROUP_ID,
    METRIC_NAMESPACE,
    REMOTE_PROJECT_ROOT,
    TRAINING_SEEDS,
)
LOCAL_CAMPAIGN_ROOT = PROJECT_ROOT / "outputs" / "turkish_question_condition" / EXPERIMENT_ID


REPORT_SCHEMA = "audiollm.turkish_question_condition_report.v1"
EXPECTED_FOLD_ROWS = 900
EXPECTED_SEED_ROWS = 180
ROUTES = ("teacher_forced", "logreg", "xgb_optuna100")
ROUTE_BACKENDS = {
    "teacher_forced": "original_teacher_forced",
    "logreg": {"qwen": "qwen_hidden_logreg_raw", "gemma4": "gemma4_hidden_logreg_raw"},
    "xgb_optuna100": {"qwen": "qwen_hidden_xgb_optuna100", "gemma4": "gemma4_hidden_xgb_optuna100"},
}
TEXT_MODALITIES = ("text_only", "audio_text")
INPUT_ORDER = (
    ("audio_only", "not_applicable"),
    ("text_only", "native"),
    ("text_only", "english"),
    ("audio_text", "native"),
    ("audio_text", "english"),
)


class ReportError(ValueError):
    """Raised when evidence cannot support a complete qualified report."""


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON evidence {path}: {exc}") from exc


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReportError(f"prediction artifact is missing: {path}")
    if path.suffix == ".jsonl":
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReportError(f"malformed JSONL row in {path}")
                result.append(value)
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metric(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, float], int, int]:
    values = list(rows)
    if not values:
        raise ReportError("empty subject prediction evidence")
    labels: list[int] = []
    predictions: list[int] = []
    invalid = 0
    subjects: set[str] = set()
    for row in values:
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReportError(f"non-binary prediction row: {row}") from exc
        raw_prediction = row.get("prediction", row.get("predicted_class"))
        if raw_prediction is None:
            raw_prediction = row.get("prediction_text")
        if raw_prediction is None:
            raw_prediction = row.get("teacher_forced_prediction_text")
        if isinstance(raw_prediction, str):
            normalized = raw_prediction.strip().lower()
            if normalized in {"1", "depressed"}:
                prediction = 1
            elif normalized in {"0", "non-depressed", "non_depressed", "nondepressed"}:
                prediction = 0
            else:
                prediction = -1
        else:
            try:
                prediction = int(raw_prediction)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"non-binary prediction row: {row}") from exc
        if label not in (0, 1):
            raise ReportError(f"non-binary prediction row: {row}")
        labels.append(label)
        predictions.append(prediction)
        subjects.add(str(row.get("subject_id", row.get("participant_id", ""))))
        invalid += max(int(row.get("invalid_qwen_outputs", 0) or 0), int(prediction not in (0, 1)))
    if len(subjects) != len(values) or "" in subjects:
        raise ReportError("subject prediction evidence is not exactly one row per subject")
    # Match the repository's strict headline convention: an INVALID prediction
    # is wrong.  In particular, an invalid positive prediction contributes to
    # FN, while an invalid negative prediction is not counted as TN.
    tp = sum(1 for truth, prediction in zip(labels, predictions) if truth == 1 and prediction == 1)
    fp = sum(1 for truth, prediction in zip(labels, predictions) if truth == 0 and prediction == 1)
    fn = sum(1 for truth, prediction in zip(labels, predictions) if truth == 1 and prediction != 1)
    tn = sum(1 for truth, prediction in zip(labels, predictions) if truth == 0 and prediction == 0)
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    negative_f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {
        "macro_f1": float((negative_f1 + positive_f1) / 2.0),
        "positive_f1": float(positive_f1),
    }, len(subjects), invalid


def _local_fold(remote_fold: str) -> Path:
    path = Path(remote_fold)
    try:
        return PROJECT_ROOT / path.relative_to(REMOTE_PROJECT_ROOT)
    except ValueError as exc:
        raise ReportError(f"evidence path is outside canonical output root: {remote_fold}") from exc


def _sidecar(path: Path, name: str) -> Any:
    target = path / name
    if not target.is_file():
        raise ReportError(f"missing {name}: {path}")
    return _json(target)


def _job_history(path: Path) -> dict[str, Any]:
    events = []
    for line in (path / "jobs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    if not events:
        raise ReportError(f"empty job history: {path / 'jobs.jsonl'}")
    return {
        "events": events,
        "slurm_job_ids": sorted({str(event.get("slurm_job_id")) for event in events if event.get("slurm_job_id")}),
        "failures": [
            {key: event.get(key) for key in ("event_type", "slurm_job_id", "status", "reason", "exit_code", "resubmission_of_job_id")}
            for event in events if event.get("event_type") in {"FAILED", "CANCELLED"}
        ],
    }


def _local_runtime_path(plan: dict[str, Any], remote_path: str | Path) -> Path:
    runtime_value = plan.get("runtime_root")
    if not runtime_value:
        raise ReportError("submission plan has no runtime root")
    runtime = Path(str(runtime_value))
    remote = Path(remote_path)
    try:
        relative = remote.relative_to(runtime)
    except ValueError as exc:
        raise ReportError(f"runtime evidence is outside the locked runtime root: {remote}") from exc
    local = LOCAL_CAMPAIGN_ROOT / str(plan.get("stage")) / "runtime" / relative
    if not local.is_file():
        raise ReportError(f"runtime evidence is not locally collected: {local}")
    return local


def _runtime_pair(plan: dict[str, Any], backbone: dict[str, Any]) -> dict[str, Any]:
    condition = "negative_only" if backbone["recording_condition"] == "negative_only" else "pos_only"
    language = "native" if backbone["transcript_condition"] == "not_applicable" else str(backbone["transcript_condition"])
    pair = next(
        (
            item for item in plan.get("preflight", {}).get("pairs", [])
            if item.get("condition") == condition and item.get("language") == language
        ),
        None,
    )
    if not isinstance(pair, dict):
        raise ReportError(f"missing preflight pair for {condition}/{language}")
    remote_manifest = Path(str(backbone["manifest_dir"])) / "turkish_manifest.jsonl"
    remote_metadata = Path(str(backbone["split_dir"])) / "turkish_manifest_metadata.json"
    manifest = _local_runtime_path(plan, remote_manifest)
    metadata = _local_runtime_path(plan, remote_metadata)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != str(pair.get("manifest_sha256")):
        raise ReportError(f"local manifest hash does not match preflight evidence: {manifest}")
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != str(pair.get("metadata_sha256")):
        raise ReportError(f"local split metadata hash does not match preflight evidence: {metadata}")
    audit_remote = Path(str(plan.get("preflight_audit_path") or (Path(str(plan["runtime_root"])) / "preflight" / "audit.json")))
    audit = _local_runtime_path(plan, audit_remote)
    return {
        "condition": condition,
        "language": language,
        "remote_manifest": str(remote_manifest),
        "remote_split": str(remote_metadata),
        "manifest": str(manifest),
        "split": str(metadata),
        "preflight_audit": str(audit),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "preflight_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }


def _execution_counts() -> dict[str, dict[str, int]]:
    fields = ("planned", "submitted", "successful", "failed", "cancelled", "retried", "superseded", "locally_validated", "reportable")
    return {
        job_type: {field: 0 for field in fields}
        for job_type in ("train", "evaluation", "hidden_extraction", "hidden_classifier")
    }


def _record_execution_attempt(
    counts: dict[str, dict[str, int]],
    attempt_dir: Path,
    job_type: str,
    *,
    planned: int = 1,
) -> None:
    if job_type not in counts:
        raise ReportError(f"unknown execution job type: {job_type}")
    counts[job_type]["planned"] += planned
    status = _sidecar(attempt_dir, "status.json")
    if status.get("state") in {"LOCALLY_VALIDATED", "REPORTABLE"}:
        counts[job_type]["locally_validated"] += 1
    if status.get("state") == "REPORTABLE":
        counts[job_type]["reportable"] += 1
    history = _job_history(attempt_dir)
    for event in history["events"]:
        if str(event.get("job_type")) != job_type:
            continue
        event_type = str(event.get("event_type"))
        if event_type == "SUBMITTED":
            counts[job_type]["submitted"] += 1
        elif event_type == "COMPLETED" and str(event.get("status")) == "COMPLETED" and str(event.get("exit_code", "0:0")).startswith("0:0"):
            counts[job_type]["successful"] += 1
        elif event_type == "FAILED":
            counts[job_type]["failed"] += 1
        elif event_type == "CANCELLED":
            counts[job_type]["cancelled"] += 1
        if event.get("resubmission_of_job_id"):
            counts[job_type]["retried"] += 1
    if _sidecar(attempt_dir, "metadata.json").get("supersedes_attempt_id"):
        counts[job_type]["superseded"] += 1


def _evaluation(attempt_dir: Path, *, route: str, expected_backend: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    status = _sidecar(attempt_dir, "status.json")
    if status.get("state") != "REPORTABLE":
        raise ReportError(f"attempt is not REPORTABLE: {attempt_dir} ({status.get('state')})")
    evaluations = _sidecar(attempt_dir, "evaluations.json")
    records = evaluations.get("evaluations", []) if isinstance(evaluations, dict) else evaluations
    matches = [
        record for record in records
        if record.get("evaluation_view") == EVALUATION_VIEW
        and record.get("metric_namespace") == METRIC_NAMESPACE
        and record.get("backend") == expected_backend
        and record.get("reportable") is True
        and record.get("locally_verified") is True
    ]
    if len(matches) != 1:
        raise ReportError(f"expected one qualified REPORTABLE evaluation for {route} in {attempt_dir}, got {len(matches)}")
    record = matches[0]
    prediction_path = attempt_dir / str(record.get("predictions_artifact_path"))
    metrics, support, invalid = _metric(_rows(prediction_path))
    return record, {"metrics": metrics, "support": support, "invalid_count": invalid}, prediction_path


def _provenance_key(payload: dict[str, Any]) -> str:
    return "prov-" + _sha(payload)[:24]


def collect_fold_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    if plan.get("stage") != "production":
        raise ReportError("only the production submission plan is reportable")
    fold_rows: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    execution_counts = _execution_counts()
    for backbone in plan.get("backbones", []):
        fold_dir = _local_fold(str(backbone["fold_dir"]))
        metadata = _sidecar(fold_dir, "metadata.json")
        run_config_path = fold_dir / "run_config.yaml"
        if not run_config_path.is_file():
            raise ReportError(f"missing run_config.yaml: {fold_dir}")
        run_config_hash = hashlib.sha256(run_config_path.read_bytes()).hexdigest()
        source = metadata.get("source") or {}
        if source.get("git_commit") != plan.get("source_git_sha"):
            raise ReportError(f"source commit mismatch: {fold_dir}")
        jobs = _job_history(fold_dir)
        runtime = _runtime_pair(plan, backbone)
        route_attempts = {
            "teacher_forced": (fold_dir, ROUTE_BACKENDS["teacher_forced"]),
            "logreg": (fold_dir / str(backbone["logreg_attempt_id"]), ROUTE_BACKENDS["logreg"][str(backbone["backbone"])]),
            "xgb_optuna100": (fold_dir / str(backbone["xgb_attempt_id"]), ROUTE_BACKENDS["xgb_optuna100"][str(backbone["backbone"])]),
        }
        for route in ROUTES:
            attempt_dir, expected_backend = route_attempts[route]
            record, values, prediction_path = _evaluation(attempt_dir, route=route, expected_backend=expected_backend)
            attempt_meta = _sidecar(attempt_dir, "metadata.json")
            attempt_jobs = _job_history(attempt_dir)
            attempt_run_config = attempt_dir / "run_config.yaml"
            if not attempt_run_config.is_file():
                raise ReportError(f"missing head run_config.yaml: {attempt_dir}")
            if str(attempt_meta.get("group_id")) != GROUP_ID:
                raise ReportError(f"group id mismatch: {attempt_dir}")
            if route == "teacher_forced":
                _record_execution_attempt(execution_counts, fold_dir, "train")
                _record_execution_attempt(execution_counts, fold_dir, "evaluation")
            elif route == "logreg":
                _record_execution_attempt(execution_counts, attempt_dir, "hidden_extraction")
            else:
                _record_execution_attempt(execution_counts, attempt_dir, "hidden_classifier")
            logical = str(attempt_meta.get("logical_run_name"))
            parent = attempt_meta.get("parent") or {}
            provenance_payload = {
                "group_id": GROUP_ID,
                "logical_run_name": logical,
                "attempt_id": str(attempt_meta.get("attempt_id")),
                "fold": int(backbone["fold"]),
                "git_commit": source.get("git_commit"),
                "deployment_id": source.get("deployment_id") or plan.get("deployment_id"),
                "source_manifest_sha256": source.get("deployed_source_sha256") or plan.get("source_manifest_sha256"),
                "run_config": str(attempt_run_config),
                "run_config_sha256": hashlib.sha256(attempt_run_config.read_bytes()).hexdigest(),
                "manifest_sha256": (attempt_meta.get("hashes") or {}).get("manifest_sha256") or metadata.get("hashes", {}).get("manifest_sha256"),
                "split_sha256": (attempt_meta.get("hashes") or {}).get("split_sha256") or metadata.get("hashes", {}).get("split_sha256"),
                "manifest_path": runtime["manifest"],
                "split_path": runtime["split"],
                "remote_manifest_path": runtime["remote_manifest"],
                "remote_split_path": runtime["remote_split"],
                "preflight_audit_path": runtime["preflight_audit"],
                "manifest_artifact_sha256": runtime["manifest_sha256"],
                "split_artifact_sha256": runtime["split_sha256"],
                "preflight_audit_sha256": runtime["preflight_audit_sha256"],
                "checkpoint_role": "best_model",
                "checkpoint_path": record.get("checkpoint_path") or parent.get("parent_checkpoint_path") or "best_model",
                "metric_namespace": METRIC_NAMESPACE,
                "metric_name": "macro_f1",
                "backend": record.get("backend"),
                "evaluation_view": record.get("evaluation_view"),
                "aggregation": record.get("aggregation"),
                "metrics_path": str(attempt_dir / str(record["metrics_artifact_path"])),
                "predictions_path": str(prediction_path),
                "evaluation_id": record.get("evaluation_id"),
                "slurm_job_ids": sorted(set(jobs["slurm_job_ids"] + attempt_jobs["slurm_job_ids"])),
                "job_failures": jobs["failures"] + attempt_jobs["failures"],
                "locally_verified": True,
            }
            provenance_key = _provenance_key(provenance_payload)
            provenance[provenance_key] = provenance_payload
            row = {
                "condition": "negative_only" if backbone["recording_condition"] == "negative_only" else "pos_only",
                "model": "Gemma 4" if backbone["backbone"] == "gemma4" else "Qwen",
                "model_token": str(backbone["backbone"]),
                "modality": backbone["modality"],
                "transcript_condition": backbone["transcript_condition"],
                "route": route,
                "backend": record["backend"],
                "seed": int(backbone["seed"]),
                "fold": int(backbone["fold"]),
                "macro_f1": values["metrics"]["macro_f1"],
                "positive_f1": values["metrics"]["positive_f1"],
                "support": values["support"],
                "invalid_count": values["invalid_count"],
                "attempt_id": attempt_meta.get("attempt_id"),
                "evaluation_id": record.get("evaluation_id"),
                "provenance_key": provenance_key,
                "run_name": backbone["run_name"],
                "fold_dir": str(attempt_dir),
            }
            fold_rows.append(row)
    if len(fold_rows) != EXPECTED_FOLD_ROWS:
        raise ReportError(f"expected {EXPECTED_FOLD_ROWS} qualified fold rows, got {len(fold_rows)}")
    fields = ("planned", "submitted", "successful", "failed", "cancelled", "retried", "superseded", "locally_validated", "reportable")
    execution = {
        field: {job_type: execution_counts[job_type][field] for job_type in sorted(execution_counts)}
        for field in fields
    }
    execution["by_job_type"] = execution_counts
    return fold_rows, provenance, execution


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise ReportError(f"expected three seed means, got {len(values)}")
    return float(statistics.mean(values)), float(statistics.stdev(values))


def build_tables(fold_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        key = (row["condition"], row["model_token"], row["modality"], row["transcript_condition"], row["route"], int(row["seed"]))
        by_seed[key].append(row)
    seed_details: list[dict[str, Any]] = []
    for key, rows in sorted(by_seed.items(), key=lambda item: tuple(str(value) for value in item[0])):
        if len(rows) != 5 or {int(row["fold"]) for row in rows} != {0, 1, 2, 3, 4}:
            raise ReportError(f"incomplete five-fold seed group: {key}")
        seed_details.append({
            "condition": key[0], "model": "Gemma 4" if key[1] == "gemma4" else "Qwen", "model_token": key[1],
            "modality": key[2], "transcript_condition": key[3], "route": key[4], "seed": key[5],
            "fold_count": len(rows), "macro_f1_mean": statistics.mean(float(row["macro_f1"]) for row in rows),
            "positive_f1_mean": statistics.mean(float(row["positive_f1"]) for row in rows),
            "provenance_keys": ",".join(sorted({str(row["provenance_key"]) for row in rows})),
        })
    if len(seed_details) != EXPECTED_SEED_ROWS:
        raise ReportError(f"expected {EXPECTED_SEED_ROWS} seed rows, got {len(seed_details)}")

    def aggregate(condition: str, model: str, modality: str, transcript: str, route: str) -> dict[str, Any]:
        selected = [row for row in seed_details if row["condition"] == condition and row["model_token"] == model and row["modality"] == modality and row["transcript_condition"] == transcript and row["route"] == route]
        if len(selected) != 3:
            raise ReportError(f"incomplete aggregate group: {condition}/{model}/{modality}/{transcript}/{route}")
        macro_mean, macro_sd = _mean_sd([float(row["macro_f1_mean"]) for row in selected])
        pos_mean, pos_sd = _mean_sd([float(row["positive_f1_mean"]) for row in selected])
        return {"macro_f1_mean": macro_mean, "macro_f1_sd": macro_sd, "positive_f1_mean": pos_mean, "positive_f1_sd": pos_sd, "seed_count": 3, "fold_count": 15, "provenance_keys": ",".join(sorted({key for row in selected for key in str(row["provenance_keys"]).split(",") if key}))}

    table1: list[dict[str, Any]] = []
    for model in ("qwen", "gemma4"):
        for modality, transcript in INPUT_ORDER:
            for route in ROUTES:
                pos_only = aggregate("pos_only", model, modality, transcript, route)
                negative = aggregate("negative_only", model, modality, transcript, route)
                table1.append({
                    "model": "Gemma 4" if model == "gemma4" else "Qwen", "model_token": model, "modality": modality, "transcript_condition": transcript, "route": route, "backend": ROUTE_BACKENDS[route] if route == "teacher_forced" else ROUTE_BACKENDS[route][model],
                    "pos_only_macro_f1_mean": pos_only["macro_f1_mean"], "pos_only_macro_f1_sd": pos_only["macro_f1_sd"], "negative_only_macro_f1_mean": negative["macro_f1_mean"], "negative_only_macro_f1_sd": negative["macro_f1_sd"], "paired_macro_f1_delta": negative["macro_f1_mean"] - pos_only["macro_f1_mean"],
                    "pos_only_positive_f1_mean": pos_only["positive_f1_mean"], "pos_only_positive_f1_sd": pos_only["positive_f1_sd"], "negative_only_positive_f1_mean": negative["positive_f1_mean"], "negative_only_positive_f1_sd": negative["positive_f1_sd"], "paired_positive_f1_delta": negative["positive_f1_mean"] - pos_only["positive_f1_mean"], "complete_seed_fold_count": 15, "provenance_key": _sha({"pos_only": pos_only["provenance_keys"], "negative_only": negative["provenance_keys"]}),
                })

    table2: list[dict[str, Any]] = []
    for condition in ("pos_only", "negative_only"):
        for model in ("qwen", "gemma4"):
            for modality in TEXT_MODALITIES:
                for route in ROUTES:
                    native = aggregate(condition, model, modality, "native", route)
                    english = aggregate(condition, model, modality, "english", route)
                    table2.append({
                        "condition": condition, "model": "Gemma 4" if model == "gemma4" else "Qwen", "model_token": model, "modality": modality, "route": route, "native_macro_f1_mean": native["macro_f1_mean"], "native_macro_f1_sd": native["macro_f1_sd"], "english_macro_f1_mean": english["macro_f1_mean"], "english_macro_f1_sd": english["macro_f1_sd"], "english_minus_native_macro_f1": english["macro_f1_mean"] - native["macro_f1_mean"], "native_positive_f1_mean": native["positive_f1_mean"], "native_positive_f1_sd": native["positive_f1_sd"], "english_positive_f1_mean": english["positive_f1_mean"], "english_positive_f1_sd": english["positive_f1_sd"], "english_minus_native_positive_f1": english["positive_f1_mean"] - native["positive_f1_mean"], "provenance_key": _sha({"native": native["provenance_keys"], "english": english["provenance_keys"]}),
                    })

    table3: list[dict[str, Any]] = []
    for model in ("qwen", "gemma4"):
        for modality in TEXT_MODALITIES:
            for route in ROUTES:
                pos_only_native = aggregate("pos_only", model, modality, "native", route)
                pos_only_english = aggregate("pos_only", model, modality, "english", route)
                neg_native = aggregate("negative_only", model, modality, "native", route)
                neg_english = aggregate("negative_only", model, modality, "english", route)
                table3.append({"model": "Gemma 4" if model == "gemma4" else "Qwen", "model_token": model, "modality": modality, "route": route, "pos_only_translation_macro_f1": pos_only_english["macro_f1_mean"] - pos_only_native["macro_f1_mean"], "negative_only_translation_macro_f1": neg_english["macro_f1_mean"] - neg_native["macro_f1_mean"], "interaction_macro_f1": (neg_english["macro_f1_mean"] - neg_native["macro_f1_mean"]) - (pos_only_english["macro_f1_mean"] - pos_only_native["macro_f1_mean"]), "pos_only_translation_positive_f1": pos_only_english["positive_f1_mean"] - pos_only_native["positive_f1_mean"], "negative_only_translation_positive_f1": neg_english["positive_f1_mean"] - neg_native["positive_f1_mean"], "interaction_positive_f1": (neg_english["positive_f1_mean"] - neg_native["positive_f1_mean"]) - (pos_only_english["positive_f1_mean"] - pos_only_native["positive_f1_mean"]), "provenance_key": _sha({"model": model, "modality": modality, "route": route})})

    table4: list[dict[str, Any]] = []
    for condition in ("pos_only", "negative_only"):
        for modality, transcript in INPUT_ORDER:
            for route in ROUTES:
                qwen = aggregate(condition, "qwen", modality, transcript, route)
                gemma = aggregate(condition, "gemma4", modality, transcript, route)
                table4.append({"condition": condition, "modality": modality, "transcript_condition": transcript, "route": route, "qwen_macro_f1_mean": qwen["macro_f1_mean"], "gemma4_macro_f1_mean": gemma["macro_f1_mean"], "gemma4_minus_qwen_macro_f1": gemma["macro_f1_mean"] - qwen["macro_f1_mean"], "qwen_positive_f1_mean": qwen["positive_f1_mean"], "gemma4_positive_f1_mean": gemma["positive_f1_mean"], "gemma4_minus_qwen_positive_f1": gemma["positive_f1_mean"] - qwen["positive_f1_mean"], "provenance_key": _sha({"qwen": qwen["provenance_keys"], "gemma4": gemma["provenance_keys"]})})
    return {"table1_dataset_condition": table1, "table2_translation": table2, "table3_translation_interaction": table3, "table4_model_comparison": table4, "table5_seed_details": seed_details, "table6_fold_details": fold_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    text_lines: list[str] = []
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    _write_once(path, buffer.getvalue())


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise ReportError(f"refusing to overwrite incompatible report artifact: {path}")
    path.write_text(text, encoding="utf-8")


def generate_report(plan_path: Path, output_dir: Path) -> dict[str, Any]:
    plan = _json(plan_path)
    if plan.get("group_id") != GROUP_ID or plan.get("experiment_id") != EXPERIMENT_ID:
        raise ReportError("submission plan group or experiment id mismatch")
    fold_rows, provenance, execution = collect_fold_rows(plan)
    tables = build_tables(fold_rows)
    if [len(tables[key]) for key in ("table1_dataset_condition", "table2_translation", "table3_translation_interaction", "table4_model_comparison", "table5_seed_details", "table6_fold_details")] != [30, 24, 12, 30, 180, 900]:
        raise ReportError("report table cardinality does not match the locked runbook")
    job_audit = {"schema_version": "audiollm.turkish_question_condition_job_audit.v1", "group_id": GROUP_ID, "deployment_id": plan.get("deployment_id"), "source_git_sha": plan.get("source_git_sha"), "expected_counts": plan.get("expected_counts"), "execution": execution}
    validation = {"schema_version": "audiollm.turkish_question_condition_report_validation.v1", "status": "passed", "group_id": GROUP_ID, "row_counts": {key: len(value) for key, value in tables.items()}, "provenance_count": len(provenance), "qualified_fold_rows": len(fold_rows)}
    report = {"schema_version": REPORT_SCHEMA, "group_id": GROUP_ID, "experiment_id": EXPERIMENT_ID, "deployment_id": plan.get("deployment_id"), "source_git_sha": plan.get("source_git_sha"), "evaluation": {"backend": EVALUATION_BACKEND, "view": EVALUATION_VIEW, "namespace": METRIC_NAMESPACE, "aggregation": "fold_mean_subject_level", "checkpoint_role": "best_model"}, "tables": tables, "provenance_index": provenance, "job_audit": job_audit, "validation": validation}
    report["report_sha256"] = _sha(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary_cells.csv", tables["table1_dataset_condition"])
    _write_csv(output_dir / "dataset_condition_deltas.csv", tables["table1_dataset_condition"])
    _write_csv(output_dir / "translation_deltas.csv", tables["table2_translation"])
    _write_csv(output_dir / "translation_interactions.csv", tables["table3_translation_interaction"])
    _write_csv(output_dir / "model_deltas.csv", tables["table4_model_comparison"])
    _write_csv(output_dir / "seed_details.csv", tables["table5_seed_details"])
    _write_csv(output_dir / "fold_details.csv", tables["table6_fold_details"])
    _write_once(output_dir / "provenance_index.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "job_audit.json", json.dumps(job_audit, indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "report_validation.json", json.dumps(validation, indent=2, sort_keys=True) + "\n")
    _write_once(output_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = [
        f"# Turkish positive-only versus negative-only campaign",
        "",
        f"Group: `{GROUP_ID}`  ",
        f"Deployment: `{plan.get('deployment_id')}`  ",
        f"Source SHA: `{plan.get('source_git_sha')}`  ",
        "",
        "All values below come from locally REPORTABLE evidence and strict subject-level recomputation. The aggregation is five-fold mean within each seed, followed by the mean and sample standard deviation across seeds.",
        "",
        "## Compact tables",
        "",
        "- Table 1: [dataset-condition comparison](dataset_condition_deltas.csv)",
        "- Table 2: [translation comparison](translation_deltas.csv)",
        "- Table 3: [translation interaction](translation_interactions.csv)",
        "- Table 4: [model comparison](model_deltas.csv)",
        "- Table 5: [seed details](seed_details.csv)",
        "- Table 6: [fold details](fold_details.csv)",
        "- [provenance index](provenance_index.json)",
        "- [execution audit](job_audit.json)",
    ]
    _write_once(output_dir / "report.md", "\n".join(markdown) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = generate_report(args.plan, args.output_dir)
    except ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["validation"]["status"], "report": str(args.output_dir / 'report.json'), "report_sha256": report["report_sha256"], "row_counts": report["validation"]["row_counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
