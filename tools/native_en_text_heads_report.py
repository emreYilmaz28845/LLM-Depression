#!/usr/bin/env python3
"""Build the deterministic native-versus-English text-head report.

The report is evidence-first: it reads only explicit managed head attempts,
recomputes strict metrics from local subject predictions, and refuses missing,
duplicate, non-reportable, or qualifier-incompatible cells.  No metric value is
stored in this source file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import sha256_file
from src.experiment_tracking.sidecars import read_modern_sidecars, verify_modern_evidence_locally
from src.metrics import classification_metrics
from src.native_en_text_heads import (
    BACKBONES,
    CONDITIONS,
    GROUP_ID,
    HEADS,
    SPLIT_SEED,
    TRAINING_SEEDS,
)


EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
NAMESPACE = "headline/binary_strict"
HEAD_SEED = 1337
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
METHOD_TO_CONFIG = {"logreg": "logreg", "xgb_optuna100": "xgb_optuna"}
DATASET_ORDER = ("d3tec", "androids_interview", "cmdc", "turkish")
ENDPOINT_ORDER = ("standalone", "merged_cv", "merged_final")


class ReportError(ValueError):
    """The evidence cannot support a complete, qualified report."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"expected a JSON object at {path}")
    return value


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReportError(f"prediction artifact is missing: {path}")
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        raise ReportError(f"cannot read prediction artifact {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows) or not rows:
        raise ReportError(f"prediction artifact is empty or malformed: {path}")
    return [dict(row) for row in rows]


def _subject_id(row: dict[str, Any]) -> str:
    for key in ("subject_id", "participant_id", "subject"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    raise ReportError("subject-level prediction is missing subject_id")


def _label_prediction(row: dict[str, Any]) -> tuple[int, int]:
    try:
        label = int(row["label"])
        prediction = int(row.get("prediction", row.get("predicted_class")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportError(f"prediction row has no strict integer label/prediction: {row}") from exc
    if label not in (0, 1) or prediction not in (0, 1):
        raise ReportError(f"prediction row has a non-binary label/prediction: {row}")
    return label, prediction


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        raise ReportError("cannot aggregate an empty prediction set")
    labels: list[int] = []
    predictions: list[int] = []
    for row in values:
        label, prediction = _label_prediction(row)
        labels.append(label)
        predictions.append(prediction)
    result = classification_metrics(labels, predictions)
    return {
        "macro_f1": float(result["macro_f1"]),
        "positive_f1": float(result["positive_f1"]),
    }


def _sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        raise ReportError("sample standard deviation requires at least two seeds")
    return float(statistics.stdev(values))


def _path_from_rel(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _contract_for(job: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    attempt_id = str(job.get("attempt_id") or "")
    if not attempt_id:
        raise ReportError("head job is missing attempt_id")
    contract_path = PROJECT_ROOT / "outputs" / "exp_submit" / attempt_id / "contract.json"
    if not contract_path.is_file():
        raise ReportError(f"submission contract is missing for {attempt_id}: {contract_path}")
    contract = _read_json(contract_path)
    local_rel = contract.get("local_evidence_rel")
    if not local_rel:
        raise ReportError(f"submission contract has no local evidence path: {contract_path}")
    return contract, _path_from_rel(PROJECT_ROOT, str(local_rel)).resolve()


def _source_provenance(metadata: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    if source.get("git_commit") != plan.get("source_commit"):
        raise ReportError(
            f"source commit mismatch for {metadata.get('attempt_id')}: "
            f"{source.get('git_commit')} != {plan.get('source_commit')}"
        )
    deployment_id = source.get("deployment_id") or plan.get("deployment_id")
    return {
        "branch": source.get("git_branch"),
        "git_commit": source.get("git_commit"),
        "deployment_id": deployment_id,
        "source_manifest_sha256": source.get("deployed_source_sha256"),
    }


def _job_events(root: Path) -> dict[str, Any]:
    jobs_path = root / "jobs.jsonl"
    events: list[dict[str, Any]] = []
    if jobs_path.is_file():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReportError(f"malformed job event in {jobs_path}")
                events.append(value)
    completed = [
        event
        for event in events
        if event.get("event_type") == "COMPLETED"
        and event.get("status") == "COMPLETED"
        and str(event.get("exit_code", "0:0")).startswith("0:0")
    ]
    if not completed:
        raise ReportError(f"no successful completion event in {jobs_path}")
    job_ids = sorted({str(event["slurm_job_id"]) for event in events if event.get("slurm_job_id")})
    failures = [
        {
            "event_type": event.get("event_type"),
            "slurm_job_id": event.get("slurm_job_id"),
            "status": event.get("status"),
            "reason": event.get("reason"),
            "exit_code": event.get("exit_code"),
            "resubmission_of_job_id": event.get("resubmission_of_job_id"),
        }
        for event in events
        if event.get("event_type") in {"FAILED", "CANCELLED"}
    ]
    return {"slurm_job_ids": job_ids, "failures": failures, "events": events}


def _qualify_attempt(
    *,
    job: dict[str, Any],
    contract: dict[str, Any],
    root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    sidecars = read_modern_sidecars(root)
    if sidecars is None:
        raise ReportError(f"missing modern sidecars for {job['attempt_id']}: {root}")
    if sidecars.state != "REPORTABLE":
        raise ReportError(f"attempt {job['attempt_id']} is not REPORTABLE: {sidecars.state}")
    issues = verify_modern_evidence_locally(sidecars)
    if issues:
        raise ReportError(f"local evidence verification failed for {job['attempt_id']}: {issues}")
    run_config = _read_yaml(root / "run_config.yaml")
    scientific = run_config.get("config") if isinstance(run_config.get("config"), dict) else run_config
    tracking = run_config.get("tracking") if isinstance(run_config.get("tracking"), dict) else {}
    if str(tracking.get("attempt_id")) != str(job["attempt_id"]):
        raise ReportError(f"tracking attempt ID mismatch at {root}")
    source = _source_provenance(_read_json(root / "metadata.json"), plan)
    classifier = scientific.get("classifier") if isinstance(scientific.get("classifier"), dict) else {}
    method = str(job.get("method"))
    expected_method = METHOD_TO_CONFIG[method]
    if classifier.get("method") != expected_method:
        raise ReportError(f"method mismatch for {job['attempt_id']}: {classifier.get('method')} != {expected_method}")
    if classifier.get("head_seed") != HEAD_SEED or classifier.get("protocol") != "native_en_text_heads_v2":
        raise ReportError(f"head seed/protocol mismatch for {job['attempt_id']}")
    expected_trials = int(
        job["trials"]
        if job.get("trials") is not None
        else classifier.get("optuna_trials", -1)
    )
    if int(classifier.get("optuna_trials", -1)) != expected_trials:
        raise ReportError(f"Optuna target mismatch for {job['attempt_id']}")
    eval_doc = _read_json(root / "evaluations.json")
    evaluations = eval_doc.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ReportError(f"no evaluations recorded for {job['attempt_id']}")
    # Older v2 submission plans omitted the backend/trials convenience fields
    # from their serialized job entries even though the immutable attempt
    # config records them.  Prefer explicit plan fields, then fall back to the
    # attempt config so preserved reportable smoke evidence remains usable.
    expected_backend = str(job.get("backend") or classifier.get("prediction_backend") or "")
    expected_view = EVALUATION_VIEW
    selected: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ReportError(f"malformed evaluation record for {job['attempt_id']}")
        if evaluation.get("backend") != expected_backend:
            raise ReportError(f"backend mismatch for {job['attempt_id']}: {evaluation.get('backend')} != {expected_backend}")
        for key, expected in (
            ("evaluation_view", expected_view),
            ("metric_namespace", NAMESPACE),
            ("checkpoint_role", "best_model"),
            ("aggregation", "subject_level"),
        ):
            if evaluation.get(key) != expected:
                raise ReportError(f"{key} mismatch for {job['attempt_id']}: {evaluation.get(key)!r} != {expected!r}")
        if evaluation.get("locally_verified") is not True or evaluation.get("reportable") is not True:
            raise ReportError(f"evaluation is not locally verified/reportable for {job['attempt_id']}")
        prediction_path = root / str(evaluation.get("predictions_artifact_path", ""))
        rows = _read_predictions(prediction_path)
        dataset = str(evaluation.get("dataset") or "").lower()
        if dataset == "":
            raise ReportError(f"evaluation has no dataset for {job['attempt_id']}")
        dataset_rows = [
            row for row in rows
            if str(row.get("dataset", dataset)).lower() == dataset
        ]
        if not dataset_rows:
            raise ReportError(f"evaluation has no prediction rows for {job['attempt_id']} dataset={dataset}")
        selected.append(
            {
                "evaluation_id": evaluation.get("evaluation_id"),
                "dataset": dataset,
                "rows": dataset_rows,
                "metrics": _metrics(dataset_rows),
                "evaluation": evaluation,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
                "metrics_path": str(root / str(evaluation.get("metrics_artifact_path", ""))),
                "metrics_sha256": sha256_file(root / str(evaluation.get("metrics_artifact_path", ""))),
            }
        )
    return {
        "job": job,
        "contract": contract,
        "root": str(root),
        "attempt_id": str(job["attempt_id"]),
        "logical_run_name": job.get("logical_run_name"),
        "fold": int(job["fold"]),
        "seed": int(job["seed"]),
        "condition": str(job["condition"]),
        "backbone": str(job["backbone"]),
        "endpoint": str(job["endpoint"]),
        "method": method,
        "backend": expected_backend,
        "config_path": str(root / "run_config.yaml"),
        "config_sha256": sha256_file(root / "run_config.yaml"),
        "manifest_sha256": (_read_json(root / "metadata.json").get("hashes") or {}).get("manifest_sha256"),
        "split_sha256": (_read_json(root / "metadata.json").get("hashes") or {}).get("split_sha256"),
        "source": source,
        "head_seed": HEAD_SEED,
        "split_seed": _split_seed(scientific),
        "checkpoint_path": str(((_read_json(root / "metadata.json").get("paths") or {}).get("best_model")) or ""),
        "evaluations": selected,
        "jobs": _job_events(root),
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ReportError(f"cannot read YAML evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"expected a YAML object at {path}")
    return value


def _split_seed(scientific: dict[str, Any]) -> int | None:
    if str(scientific.get("protocol")) == "symmetric_merged":
        return int((scientific.get("protocol_settings") or {}).get("split_seed", -1))
    return int((scientific.get("split") or {}).get("seed", -1))


def _load_jobs(plan_path: Path, attempts: set[str] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _read_json(plan_path)
    if plan.get("group_id") != GROUP_ID or plan.get("schema_version") != "native_en_text_heads_v2_submission_plan.v1":
        raise ReportError(f"not a validated v2 submission plan: {plan_path}")
    jobs = [
        job for job in plan.get("jobs", [])
        if isinstance(job, dict)
        and job.get("method") in HEADS
        and job.get("endpoint") in {"standalone", "merged_cv", "merged_final"}
    ]
    if attempts is not None:
        unknown = attempts - {str(job.get("attempt_id")) for job in jobs}
        if unknown:
            raise ReportError(f"attempt list contains IDs absent from the plan: {sorted(unknown)}")
        jobs = [job for job in jobs if str(job.get("attempt_id")) in attempts]
    if not jobs:
        raise ReportError("the plan contains no selected head jobs")
    return plan, jobs


def _validate_matrix(records: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    expected: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {}
    for record in records:
        job = record["job"]
        for evaluation in record["evaluations"]:
            key = (
                str(job["endpoint"]),
                str(job["condition"]),
                str(job["backbone"]),
                str(job["method"]),
                int(job["seed"]),
                int(job["fold"]),
            )
            if key in expected:
                raise ReportError(f"duplicate attempt for matrix cell {key}")
            expected[key] = record
            break
    for endpoint in ENDPOINT_ORDER:
        for condition in CONDITIONS:
            for backbone in BACKBONES:
                for method in HEADS:
                    if endpoint == "standalone":
                        datasets = DATASET_ORDER
                        folds = EXPECTED_FOLDS
                    elif endpoint == "merged_cv":
                        datasets = ("merged",)
                        folds = EXPECTED_FOLDS
                    else:
                        datasets = ("merged",)
                        folds = (0,)
                    for dataset in datasets:
                        for seed in TRAINING_SEEDS:
                            for fold in folds:
                                matching = [
                                    record
                                    for record in records
                                    if record["endpoint"] == endpoint
                                    and record["condition"] == condition
                                    and record["backbone"] == backbone
                                    and record["method"] == method
                                    and record["seed"] == seed
                                    and record["fold"] == fold
                                ]
                                if len(matching) != 1:
                                    raise ReportError(
                                        f"expected exactly one {endpoint}/{condition}/{backbone}/{method}/"
                                        f"{dataset}/seed{seed}/fold{fold}, found {len(matching)}"
                                    )
                                if endpoint == "standalone":
                                    evaluation_datasets = {item["dataset"] for item in matching[0]["evaluations"]}
                                    if evaluation_datasets != {dataset}:
                                        raise ReportError(
                                            f"standalone evaluation dataset mismatch: {evaluation_datasets} != {dataset}"
                                        )
                                elif endpoint == "merged_cv":
                                    evaluation_datasets = {item["dataset"] for item in matching[0]["evaluations"]}
                                    expected_datasets = {"daic", "d3tec", "androids_interview", "cmdc", "turkish"}
                                    if evaluation_datasets != expected_datasets:
                                        raise ReportError(
                                            f"merged CV evaluation dataset mismatch: {evaluation_datasets} != {expected_datasets}"
                                        )
                                else:
                                    evaluation_datasets = {item["dataset"] for item in matching[0]["evaluations"]}
                                    if evaluation_datasets != {"daic"}:
                                        raise ReportError(
                                            f"merged final evaluation dataset mismatch: {evaluation_datasets} != {{'daic'}}"
                                        )


def _record_provenance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": record["attempt_id"],
        "logical_run_name": record["logical_run_name"],
        "fold": record["fold"],
        "seed": record["seed"],
        "config_path": record["config_path"],
        "config_sha256": record["config_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "split_sha256": record["split_sha256"],
        "checkpoint_path": record["checkpoint_path"],
        "evaluation_ids": [item["evaluation_id"] for item in record["evaluations"]],
        "metrics_artifacts": [
            {
                "path": item["metrics_path"],
                "sha256": item["metrics_sha256"],
                "prediction_path": item["prediction_path"],
                "prediction_sha256": item["prediction_sha256"],
            }
            for item in record["evaluations"]
        ],
        "slurm_job_ids": record["jobs"]["slurm_job_ids"],
        "failures": record["jobs"]["failures"],
        "source": record["source"],
        "evaluation_view": EVALUATION_VIEW,
        "namespace": NAMESPACE,
        "backend": record["backend"],
        "split_seed": record["split_seed"],
        "head_seed": record["head_seed"],
    }


def _fold_metrics(record: dict[str, Any], dataset: str | None = None) -> dict[str, float]:
    evaluations = record["evaluations"]
    if dataset is not None:
        evaluations = [item for item in evaluations if item["dataset"] == dataset]
    if not evaluations:
        raise ReportError(f"no matching evaluation for {record['attempt_id']} dataset={dataset}")
    if len(evaluations) > 1 and dataset is not None:
        raise ReportError(f"duplicate dataset evaluations for {record['attempt_id']} dataset={dataset}")
    if len(evaluations) == 1:
        return dict(evaluations[0]["metrics"])
    return {
        metric: float(statistics.mean(item["metrics"][metric] for item in evaluations))
        for metric in ("macro_f1", "positive_f1")
    }


def _pooled_metrics(records: list[dict[str, Any]], dataset: str) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    subject_ids: set[str] = set()
    for record in records:
        for evaluation in record["evaluations"]:
            if evaluation["dataset"] != dataset:
                continue
            for row in evaluation["rows"]:
                subject = _subject_id(row)
                if subject in subject_ids:
                    raise ReportError(f"duplicate subject across pooled folds for {dataset}: {subject}")
                subject_ids.add(subject)
                rows.append(row)
    return _metrics(rows)


def _aggregate_cell(
    records: list[dict[str, Any]],
    *,
    endpoint: str,
    condition: str,
    backbone: str,
    method: str,
    dataset: str,
) -> dict[str, Any]:
    by_seed = {int(record["seed"]): record for record in records}
    seed_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        seed_records = [record for record in records if int(record["seed"]) == seed]
        if endpoint == "standalone":
            if dataset in {"d3tec", "androids_interview"}:
                metrics = _pooled_metrics(seed_records, dataset)
                aggregation = "pooled subject-level across five outer folds"
            else:
                fold_metrics = [_fold_metrics(record, dataset) for record in seed_records]
                metrics = {
                    metric: float(statistics.mean(item[metric] for item in fold_metrics))
                    for metric in ("macro_f1", "positive_f1")
                }
                aggregation = "unweighted mean of five outer-fold subject-level scores"
        elif endpoint == "merged_cv":
            fold_metrics = [_fold_metrics(record) for record in seed_records]
            metrics = {
                metric: float(statistics.mean(item[metric] for item in fold_metrics))
                for metric in ("macro_f1", "positive_f1")
            }
            aggregation = "unweighted dataset mean within fold, then unweighted five-fold mean"
        else:
            metrics = _fold_metrics(seed_records[0], "daic")
            aggregation = "DAIC subject-level final evaluation"
        seed_rows.append(
            {
                "seed": seed,
                "native_or_english": condition,
                "macro_f1": metrics["macro_f1"],
                "positive_f1": metrics["positive_f1"],
                "provenance": [
                    _record_provenance(record)
                    for record in sorted(seed_records, key=lambda item: int(item["fold"]))
                ],
            }
        )
    return {
        "endpoint": endpoint,
        "dataset": dataset,
        "backbone": backbone,
        "head": method,
        "condition": condition,
        "aggregation": aggregation,
        "seed_count": len(seed_rows),
        "seed_rows": seed_rows,
        "provenance_key": "|".join((GROUP_ID, endpoint, dataset, backbone, method)),
        "provenance_status": "reportable_local_evidence",
    }


def _summary_pair(native: dict[str, Any], english: dict[str, Any]) -> dict[str, Any]:
    if native["aggregation"] != english["aggregation"]:
        raise ReportError("native and English aggregation conventions differ")
    details: list[dict[str, Any]] = []
    for native_seed, english_seed in zip(native["seed_rows"], english["seed_rows"]):
        if native_seed["seed"] != english_seed["seed"]:
            raise ReportError("native and English seed order differs")
        details.append(
            {
                "endpoint": native["endpoint"],
                "dataset": native["dataset"],
                "backbone": native["backbone"],
                "head": native["head"],
                "seed": native_seed["seed"],
                "native_macro_f1": native_seed["macro_f1"],
                "english_macro_f1": english_seed["macro_f1"],
                "delta_macro_f1": english_seed["macro_f1"] - native_seed["macro_f1"],
                "native_positive_f1": native_seed["positive_f1"],
                "english_positive_f1": english_seed["positive_f1"],
                "delta_positive_f1": english_seed["positive_f1"] - native_seed["positive_f1"],
                "split_seed": SPLIT_SEED,
                "head_seed": HEAD_SEED,
                "head_protocol": "native_en_text_heads_v2",
                "evaluation_view": EVALUATION_VIEW,
                "aggregation": native["aggregation"],
                "native_provenance": native_seed["provenance"],
                "english_provenance": english_seed["provenance"],
                "status": "reportable_local_evidence",
            }
        )
    macro_native = [row["native_macro_f1"] for row in details]
    macro_english = [row["english_macro_f1"] for row in details]
    macro_delta = [row["delta_macro_f1"] for row in details]
    pos_native = [row["native_positive_f1"] for row in details]
    pos_english = [row["english_positive_f1"] for row in details]
    pos_delta = [row["delta_positive_f1"] for row in details]
    return {
        "endpoint": native["endpoint"],
        "dataset": native["dataset"],
        "backbone": native["backbone"],
        "head": native["head"],
        "native_macro_mean": float(statistics.mean(macro_native)),
        "native_macro_sd": _sample_sd(macro_native),
        "english_macro_mean": float(statistics.mean(macro_english)),
        "english_macro_sd": _sample_sd(macro_english),
        "delta_macro_mean": float(statistics.mean(macro_delta)),
        "delta_macro_sd": _sample_sd(macro_delta),
        "native_positive_mean": float(statistics.mean(pos_native)),
        "native_positive_sd": _sample_sd(pos_native),
        "english_positive_mean": float(statistics.mean(pos_english)),
        "english_positive_sd": _sample_sd(pos_english),
        "delta_positive_mean": float(statistics.mean(pos_delta)),
        "delta_positive_sd": _sample_sd(pos_delta),
        "aggregation": native["aggregation"],
        "seed_count": len(details),
        "provenance_key": native["provenance_key"],
        "provenance_status": "reportable_local_evidence",
        "seed_details": details,
    }


def build_report(plan_path: str | Path, attempts: set[str] | None = None) -> dict[str, Any]:
    plan, jobs = _load_jobs(Path(plan_path).resolve(), attempts)
    records = []
    seen_attempts: set[str] = set()
    for job in jobs:
        attempt_id = str(job["attempt_id"])
        if attempt_id in seen_attempts:
            raise ReportError(f"duplicate attempt ID in plan: {attempt_id}")
        seen_attempts.add(attempt_id)
        contract, root = _contract_for(job)
        records.append(_qualify_attempt(job=job, contract=contract, root=root, plan=plan))
    _validate_matrix(records, plan)

    by_cell: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        job = record["job"]
        datasets = {item["dataset"] for item in record["evaluations"]}
        dataset = next(iter(datasets)) if job["endpoint"] == "standalone" else "merged"
        by_cell.setdefault(
            (str(job["endpoint"]), str(job["condition"]), str(job["backbone"]), str(job["method"]), dataset),
            [],
        ).append(record)

    summaries: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_ORDER:
        datasets = DATASET_ORDER if endpoint == "standalone" else ("merged",)
        report_dataset = "daic" if endpoint == "merged_final" else None
        for dataset in datasets:
            target = report_dataset or dataset
            for backbone in BACKBONES:
                for method in HEADS:
                    native_records = by_cell[(endpoint, "native", backbone, method, dataset)]
                    english_records = by_cell[(endpoint, "english", backbone, method, dataset)]
                    native = _aggregate_cell(
                        native_records,
                        endpoint=endpoint,
                        condition="native",
                        backbone=backbone,
                        method=method,
                        dataset=target,
                    )
                    english = _aggregate_cell(
                        english_records,
                        endpoint=endpoint,
                        condition="english",
                        backbone=backbone,
                        method=method,
                        dataset=target,
                    )
                    summaries.append(_summary_pair(native, english))
    details = [detail for summary in summaries for detail in summary.pop("seed_details")]
    if len(summaries) != 24 or len(details) != 72:
        raise ReportError(f"report cardinality mismatch: summaries={len(summaries)} details={len(details)}")
    return {
        "schema_version": "native_en_text_heads_v2_report.v1",
        "status": "passed",
        "group_id": GROUP_ID,
        "plan_path": str(Path(plan_path).resolve()),
        "deployment_id": plan.get("deployment_id"),
        "source_commit": plan.get("source_commit"),
        "aggregation": {
            "d3tec_androids_interview": "pooled subject-level across five outer folds",
            "cmdc_turkish": "unweighted mean of five outer-fold subject-level scores",
            "merged_cv": "unweighted dataset mean within fold, then unweighted five-fold mean",
            "merged_final": "DAIC subject-level final evaluation",
        },
        "summary": summaries,
        "seed_details": details,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native versus English text-only head report",
        "",
        f"Group: `{report['group_id']}`  ",
        f"Source commit: `{report['source_commit']}`  ",
        f"Deployment: `{report['deployment_id']}`  ",
        "",
        "All values below were recomputed from locally verified subject-level prediction artifacts. "
        "Delta means English minus native.",
        "",
        "## Summary",
        "",
        "| Endpoint | Dataset | Backbone | Head | Native macro mean | English macro mean | Δ macro mean | Native positive mean | English positive mean | Δ positive mean | Aggregation | Seeds | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in report["summary"]:
        lines.append(
            "| "
            + " | ".join(
                _fmt(row[key])
                for key in (
                    "endpoint", "dataset", "backbone", "head", "native_macro_mean",
                    "english_macro_mean", "delta_macro_mean", "native_positive_mean",
                    "english_positive_mean", "delta_positive_mean", "aggregation",
                    "seed_count", "provenance_status",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Seed details",
            "",
            "| Endpoint | Dataset | Backbone | Head | Seed | Native macro | English macro | Δ macro | Native positive | English positive | Δ positive | Split seed | Head seed | View | Aggregation | Status |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in report["seed_details"]:
        lines.append(
            "| "
            + " | ".join(
                _fmt(row[key])
                for key in (
                    "endpoint", "dataset", "backbone", "head", "seed", "native_macro_f1",
                    "english_macro_f1", "delta_macro_f1", "native_positive_f1",
                    "english_positive_f1", "delta_positive_f1", "split_seed", "head_seed",
                    "evaluation_view", "aggregation", "status",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="validated v2 submission plan JSON")
    parser.add_argument("--attempts", default=None, help="optional comma-separated explicit head attempt IDs")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--with-timestamp", action="store_true")
    args = parser.parse_args()
    attempts = None
    if args.attempts:
        attempts = {item.strip() for item in str(args.attempts).split(",") if item.strip()}
        if len(attempts) != len([item for item in str(args.attempts).split(",") if item.strip()]):
            print("ERROR: --attempts contains duplicate IDs", file=sys.stderr)
            return 1
    try:
        report = build_report(args.plan, attempts)
        if args.with_timestamp:
            from datetime import datetime, timezone

            report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        json_path = Path(args.output_json)
        md_path = Path(args.output_md)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ReportError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"summary_rows={len(report['summary'])} seed_detail_rows={len(report['seed_details'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
