from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .constants import SCHEMA_VERSION_EVALUATIONS
from .discovery import (
    BEST_EVAL_LOCATIONS,
    LAST_EVAL_LOCATION,
    RUN_ROOT,
    DiscoveredRun,
)
from .identity import evaluation_id, legacy_attempt_id

STATUS_QUALIFIED = "QUALIFIED"
STATUS_QUARANTINED_AMBIGUOUS = "QUARANTINED_AMBIGUOUS"
STATUS_REJECTED = "REJECTED"

_HEADLINE_NAMESPACE = "headline/binary_strict"
_VALID_ONLY_NAMESPACE = "valid_only"

_BACKEND_TOKENS = (
    "original_teacher_forced",
    "candidate_label_likelihood",
    "teacher_forced",
    "likelihood",
    "generation",
)

_AGGREGATION_MODIFIERS = ("subject_level", "sample_level")
_VIEW_MARKERS = ("k4", "k2", "coverage")

_HEADLINE_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "positive_f1",
    "macro_f1",
    "weighted_f1",
)


@dataclasses.dataclass(frozen=True)
class QualifiedMetric:
    name: str
    value: float | int | None
    support: int | None


@dataclasses.dataclass(frozen=True)
class QualifiedEvaluation:
    attempt_id: str
    evaluation_id: str
    dataset: str | None
    split_name: str | None
    split_protocol: str | None
    checkpoint_role: str | None
    checkpoint_path: str | None
    backend: str | None
    evaluation_view: str | None
    aggregation: str | None
    metric_namespace: str | None
    metrics_artifact_path: str | None
    predictions_artifact_path: str | None
    metrics: tuple[QualifiedMetric, ...]
    locally_verified: bool
    reportable: bool
    reportability_issues: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "dataset": self.dataset,
            "split_name": self.split_name,
            "split_protocol": self.split_protocol,
            "checkpoint_role": self.checkpoint_role,
            "checkpoint_path": self.checkpoint_path,
            "backend": self.backend,
            "evaluation_view": self.evaluation_view,
            "aggregation": self.aggregation,
            "metric_namespace": self.metric_namespace,
            "metrics_artifact_path": self.metrics_artifact_path,
            "predictions_artifact_path": self.predictions_artifact_path,
            "metrics": [
                {"name": metric.name, "value": metric.value, "support": metric.support}
                for metric in self.metrics
            ],
            "locally_verified": self.locally_verified,
            "reportable": self.reportable,
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class QualificationResult:
    fold_dir: str
    fold: int
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    evaluations: tuple[QualifiedEvaluation, ...]

    @property
    def attempt_id(self) -> str | None:
        return self.evaluations[0].attempt_id if self.evaluations else None


def _is_float_or_int(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _content_aggregation(content: Any) -> str | None:
    if not isinstance(content, dict) or not isinstance(content.get("aggregation_level"), str):
        return None
    level = content["aggregation_level"]
    if level == "subject":
        return "subject_level"
    if level == "sample":
        return "sample_level"
    return level


def _filename_backend_and_modifiers(stem: str) -> tuple[str | None, tuple[str, ...]]:
    if not stem.startswith("metrics_"):
        return None, ()
    remainder = stem[len("metrics_"):]
    best: str | None = None
    for token in _BACKEND_TOKENS:
        if remainder.endswith(token) and (best is None or len(token) > len(best)):
            best = token
    if best is None:
        return None, ()
    rest = remainder[: -len(best)].strip("_")
    return best, tuple(part for part in rest.split("_") if part)


def _filename_view_and_aggregation(stem: str) -> tuple[str | None, str | None]:
    _, modifiers = _filename_backend_and_modifiers(stem)
    if not modifiers:
        return None, None
    joined = "_".join(modifiers)
    if joined in _AGGREGATION_MODIFIERS:
        return None, joined
    if any(marker in joined for marker in _VIEW_MARKERS):
        return joined, None
    return None, None


def _namespace_of(content: Any) -> str | None:
    if not isinstance(content, dict):
        return None
    if any(str(key).startswith("binary_strict_") for key in content):
        return _HEADLINE_NAMESPACE
    if any(str(key).startswith("valid_only_") for key in content):
        return _VALID_ONLY_NAMESPACE
    return None


def _headline_metrics(content: Any) -> tuple[QualifiedMetric, ...]:
    if not isinstance(content, dict):
        return ()
    support = content.get("num_units") if _is_int(content.get("num_units")) else None
    metrics: list[QualifiedMetric] = []
    for name in _HEADLINE_METRIC_NAMES:
        value = content.get(f"binary_strict_{name}")
        if _is_float_or_int(value):
            metrics.append(QualifiedMetric(name=name, value=value, support=support))
    return tuple(metrics)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _split_identity(
    config: dict[str, Any], protocol: dict[str, Any]
) -> tuple[str | None, str | None, bool]:
    final_eval_protocol = protocol.get("final_eval_protocol")
    if not isinstance(final_eval_protocol, dict):
        final_eval_protocol = {}
    split_config = config.get("split") if isinstance(config.get("split"), dict) else {}
    cv_protocol = protocol.get("cv_protocol")
    if not isinstance(cv_protocol, str):
        cv_protocol = split_config.get("cv_protocol") if isinstance(split_config.get("cv_protocol"), str) else None
    split_name: str | None = None
    if isinstance(final_eval_protocol.get("final_eval_split_name"), str):
        split_name = final_eval_protocol["final_eval_split_name"]
    if split_name is None and isinstance(split_config.get("final_eval_partition"), str):
        split_name = split_config["final_eval_partition"]
    no_held_out_test = cv_protocol == "train_val"
    if no_held_out_test:
        if split_name is None and isinstance(split_config.get("selection_partition"), str):
            split_name = split_config["selection_partition"]
        if split_name is None and isinstance(split_config.get("train_partition"), str):
            split_name = split_config["train_partition"]
    if isinstance(cv_protocol, str):
        split_protocol = cv_protocol
    elif split_config.get("mode") == "fixed":
        split_protocol = "fixed_train_val_test"
    else:
        split_protocol = None
    return split_name, split_protocol, no_held_out_test


def _resolve_identity(
    artifact: Any, config: dict[str, Any], content: Any
) -> tuple[str | None, str | None, str | None]:
    evaluation_config = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    backend: str | None = None
    if isinstance(content, dict) and isinstance(content.get("prediction_backend"), str):
        backend = content["prediction_backend"]
    if backend is None and isinstance(evaluation_config.get("sample_prediction_mode"), str):
        backend = evaluation_config["sample_prediction_mode"]
    filename_backend, _ = _filename_backend_and_modifiers(Path(artifact.relative_path).name)
    if backend is None:
        backend = filename_backend

    view: str | None = None
    if isinstance(content, dict) and isinstance(content.get("evaluation_view"), str):
        view = content["evaluation_view"]
    if view is None:
        view, _ = _filename_view_and_aggregation(Path(artifact.relative_path).name)
    if view is None and isinstance(evaluation_config.get("evaluation_view"), str):
        view = evaluation_config["evaluation_view"]

    aggregation = _content_aggregation(content)
    if aggregation is None:
        _, aggregation = _filename_view_and_aggregation(Path(artifact.relative_path).name)
    return backend, view, aggregation


def _legacy_attempt_input(
    discovered: DiscoveredRun, protocol: dict[str, Any], artifact: Any
) -> dict[str, Any]:
    try:
        relative_run_dir = str(Path(discovered.fold_dir).relative_to(discovered.scan_root))
    except ValueError:
        relative_run_dir = discovered.fold_dir
    manifest_hash = protocol.get("manifest_hash")
    split_hash = protocol.get("split_metadata_hash")
    return {
        "relative_run_dir": relative_run_dir,
        "fold": discovered.fold,
        "resolved_config_sha256": discovered.resolved_config_sha256,
        "manifest_sha256": manifest_hash if isinstance(manifest_hash, str) else None,
        "split_sha256": split_hash if isinstance(split_hash, str) else None,
        "checkpoint_role": "best_model",
        "checkpoint_path": "best_model",
        "evaluation_artifact_sha256": artifact.sha256,
    }


def legacy_identity_payload(
    discovered: DiscoveredRun, artifact: Any
) -> dict[str, Any]:
    return _legacy_attempt_input(discovered, discovered.protocol or {}, artifact)


def reportability_issues(
    evaluation: QualifiedEvaluation,
    *,
    hashes_present: dict[str, bool] | None = None,
    include_job_history_check: bool = False,
    legacy_exception: bool = False,
) -> list[str]:
    issues: list[str] = []
    hashes_present = hashes_present or {}
    if not evaluation.dataset:
        issues.append("empty dataset")
    if not evaluation.split_name:
        issues.append("missing split name")
    if not evaluation.split_protocol:
        issues.append("missing split protocol")
    if not evaluation.checkpoint_role:
        issues.append("missing checkpoint role")
    if not evaluation.checkpoint_path:
        issues.append("missing checkpoint path")
    if not evaluation.backend:
        issues.append("missing prediction backend")
    if not evaluation.evaluation_view:
        issues.append("missing evaluation view")
    if not evaluation.aggregation:
        issues.append("missing aggregation")
    if not evaluation.metric_namespace:
        issues.append("missing metric namespace")
    if not hashes_present.get("resolved_config"):
        issues.append("missing resolved config hash")
    if not hashes_present.get("manifest"):
        issues.append("missing manifest hash")
    if not hashes_present.get("split"):
        issues.append("missing split hash")
    if not evaluation.predictions_artifact_path:
        issues.append("missing predictions artifact")
    if not evaluation.metrics:
        issues.append("no qualified metrics")
    if evaluation.metrics and any(metric.value is None for metric in evaluation.metrics):
        issues.append("null metric values present")
    if evaluation.warnings:
        issues.append("evaluation warnings present: " + "; ".join(evaluation.warnings))
    if include_job_history_check and not legacy_exception:
        issues.append("job/resubmit history not recorded")
    return issues


def is_reportable(
    evaluation: QualifiedEvaluation,
    *,
    hashes_present: dict[str, bool] | None = None,
    include_job_history_check: bool = False,
    legacy_exception: bool = False,
) -> tuple[bool, list[str]]:
    issues = reportability_issues(
        evaluation,
        hashes_present=hashes_present,
        include_job_history_check=include_job_history_check,
        legacy_exception=legacy_exception,
    )
    return (not issues, issues)


def _metrics_artifacts_at(discovered: DiscoveredRun, location: str) -> list[Any]:
    return [artifact for artifact in discovered.artifacts if artifact.location == location and artifact.kind == "metrics"]


def _predictions_path_at(discovered: DiscoveredRun, location: str) -> str | None:
    for kind in ("subject_predictions", "predictions"):
        for artifact in discovered.artifacts:
            if artifact.location == location and artifact.kind == kind:
                return artifact.relative_path
    return None


def _build_fold_evaluation(
    discovered: DiscoveredRun,
    config: dict[str, Any],
    protocol: dict[str, Any],
    hashes_present: dict[str, bool],
    artifact: Any,
    content: Any,
    backend: str | None,
    view: str | None,
    aggregation: str | None,
    namespace: str | None,
) -> QualifiedEvaluation:
    warnings: list[str] = []
    dataset: str | None = None
    if isinstance(config.get("dataset"), str):
        dataset = config["dataset"]
    else:
        warnings.append("dataset not recorded in resolved config")
    split_name, split_protocol, no_held_out_test = _split_identity(config, protocol)
    if no_held_out_test:
        warnings.append("cv protocol train_val has no held-out test split")
    if split_name is None:
        warnings.append("final split name not recorded")
    if split_protocol is None:
        warnings.append("split protocol not recorded")
    if backend is None:
        warnings.append("prediction backend not recorded")
    if view is None:
        warnings.append("evaluation view not recorded in evidence")
    if aggregation is None:
        warnings.append("aggregation not recorded")
    metrics = _headline_metrics(content)
    predictions_path = _predictions_path_at(discovered, artifact.location)
    if predictions_path is None:
        warnings.append("missing subject-level predictions artifact")
    attempt_id = legacy_attempt_id(_legacy_attempt_input(discovered, protocol, artifact))
    evaluation_id_value = evaluation_id(
        attempt_id=attempt_id,
        fold=discovered.fold,
        dataset=dataset,
        split_name=split_name,
        split_protocol=split_protocol,
        checkpoint_role="best_model",
        checkpoint_path="best_model",
        backend=backend,
        evaluation_view=view,
        aggregation=aggregation,
        metric_namespace=namespace,
        metrics_artifact_sha256=artifact.sha256,
    )
    evaluation = QualifiedEvaluation(
        attempt_id=attempt_id,
        evaluation_id=evaluation_id_value,
        dataset=dataset,
        split_name=split_name,
        split_protocol=split_protocol,
        checkpoint_role="best_model",
        checkpoint_path="best_model",
        backend=backend,
        evaluation_view=view,
        aggregation=aggregation,
        metric_namespace=namespace,
        metrics_artifact_path=artifact.relative_path,
        predictions_artifact_path=predictions_path,
        metrics=metrics,
        locally_verified=False,
        reportable=False,
        reportability_issues=(),
        warnings=tuple(warnings),
    )
    issues = reportability_issues(
        evaluation, hashes_present=hashes_present, include_job_history_check=False
    )
    evaluation = dataclasses.replace(evaluation, reportable=not issues, reportability_issues=tuple(issues))
    return evaluation


def _build_summary_evaluations(
    discovered: DiscoveredRun,
    config: dict[str, Any],
    protocol: dict[str, Any],
    hashes_present: dict[str, bool],
) -> tuple[tuple[QualifiedEvaluation, ...], tuple[str, ...]]:
    warnings: list[str] = []
    evaluations: list[QualifiedEvaluation] = []
    summaries = [
        artifact
        for artifact in discovered.artifacts
        if artifact.kind == "final_summary" and artifact.location == RUN_ROOT
    ]
    for artifact in summaries:
        if artifact.parse_ok is False:
            warnings.append("final_summary unreadable")
            continue
        content = artifact.json_content
        if not isinstance(content, dict):
            warnings.append("final_summary unrecognized structure")
            continue
        row = content.get("active_backend_summary_row")
        if not isinstance(row, dict):
            row = {}
        backend = content.get("active_backend")
        if not isinstance(backend, str):
            backend = row.get("active_backend")
        pooled = content.get("active_backend_pooled_metrics")
        mean_summary = content.get("active_backend_metric_summary")
        if not isinstance(pooled, dict) and not isinstance(mean_summary, dict):
            warnings.append("final_summary unrecognized structure")
            continue
        split_name, split_protocol, no_held_out_test = _split_identity(config, protocol)
        common = {
            "dataset": config.get("dataset") if isinstance(config.get("dataset"), str) else None,
            "split_name": split_name,
            "split_protocol": split_protocol,
            "checkpoint_role": "best_model",
            "checkpoint_path": "best_model",
            "backend": backend,
            "evaluation_view": None,
            "metric_namespace": _HEADLINE_NAMESPACE,
            "metrics_artifact_path": artifact.relative_path,
            "predictions_artifact_path": None,
            "locally_verified": False,
        }
        if no_held_out_test:
            warnings.append("cv protocol train_val has no held-out test split")
        if isinstance(pooled, dict) and pooled:
            support: int | None = None
            if _is_int(row.get("pooled_support_negative")) and _is_int(row.get("pooled_support_positive")):
                support = row["pooled_support_negative"] + row["pooled_support_positive"]
            metrics = tuple(
                QualifiedMetric(name=name, value=pooled.get(name), support=support)
                for name in _HEADLINE_METRIC_NAMES
                if _is_float_or_int(pooled.get(name))
            )
            ev_warnings = [
                "evaluation view not recorded in aggregate summary",
                "aggregate summary has no single predictions artifact",
            ]
            evaluation = _finish_evaluation(discovered, protocol, artifact, common, metrics, "pooled_subject_level", ev_warnings, hashes_present)
            evaluations.append(evaluation)
        if isinstance(mean_summary, dict) and mean_summary:
            metrics: list[QualifiedMetric] = []
            for name in _HEADLINE_METRIC_NAMES:
                entry = mean_summary.get(name)
                if isinstance(entry, dict) and _is_float_or_int(entry.get("mean")):
                    metrics.append(QualifiedMetric(name=name, value=entry["mean"], support=None))
                if isinstance(entry, dict) and _is_float_or_int(entry.get("std")):
                    metrics.append(QualifiedMetric(name=f"{name}_std", value=entry["std"], support=None))
            ev_warnings = [
                "evaluation view not recorded in aggregate summary",
                "aggregate summary has no single predictions artifact",
            ]
            evaluation = _finish_evaluation(discovered, protocol, artifact, common, tuple(metrics), "fold_mean", ev_warnings, hashes_present)
            evaluations.append(evaluation)
    return tuple(evaluations), tuple(warnings)


def _finish_evaluation(
    discovered: DiscoveredRun,
    protocol: dict[str, Any],
    artifact: Any,
    common: dict[str, Any],
    metrics: tuple[QualifiedMetric, ...],
    aggregation: str,
    warnings: list[str],
    hashes_present: dict[str, bool],
) -> QualifiedEvaluation:
    attempt_id = legacy_attempt_id(_legacy_attempt_input(discovered, protocol, artifact))
    evaluation_id_value = evaluation_id(
        attempt_id=attempt_id,
        fold=discovered.fold,
        dataset=common["dataset"],
        split_name=common["split_name"],
        split_protocol=common["split_protocol"],
        checkpoint_role=common["checkpoint_role"],
        checkpoint_path=common["checkpoint_path"],
        backend=common["backend"],
        evaluation_view=common["evaluation_view"],
        aggregation=aggregation,
        metric_namespace=common["metric_namespace"],
        metrics_artifact_sha256=artifact.sha256,
    )
    evaluation = QualifiedEvaluation(
        attempt_id=attempt_id,
        evaluation_id=evaluation_id_value,
        dataset=common["dataset"],
        split_name=common["split_name"],
        split_protocol=common["split_protocol"],
        checkpoint_role=common["checkpoint_role"],
        checkpoint_path=common["checkpoint_path"],
        backend=common["backend"],
        evaluation_view=common["evaluation_view"],
        aggregation=aggregation,
        metric_namespace=common["metric_namespace"],
        metrics_artifact_path=common["metrics_artifact_path"],
        predictions_artifact_path=common["predictions_artifact_path"],
        metrics=metrics,
        locally_verified=False,
        reportable=False,
        reportability_issues=(),
        warnings=tuple(warnings),
    )
    issues = reportability_issues(
        evaluation, hashes_present=hashes_present, include_job_history_check=False
    )
    evaluation = dataclasses.replace(evaluation, reportable=not issues, reportability_issues=tuple(issues))
    return evaluation


def qualify_run(discovered: DiscoveredRun) -> QualificationResult:
    warnings = list(discovered.warnings)
    if not discovered.run_config_parse_ok:
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_REJECTED,
            reasons=("run_config_unparseable",),
            warnings=tuple(warnings),
            evaluations=(),
        )
    if discovered.resolved_config is None:
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_REJECTED,
            reasons=("run_config_missing_resolved_config",),
            warnings=tuple(warnings),
            evaluations=(),
        )
    config = discovered.resolved_config
    protocol = discovered.protocol or {}
    manifest_hash = protocol.get("manifest_hash")
    split_hash = protocol.get("split_metadata_hash")
    hashes_present = {
        "resolved_config": discovered.resolved_config_sha256 is not None,
        "manifest": isinstance(manifest_hash, str),
        "split": isinstance(split_hash, str),
    }
    if not hashes_present["manifest"]:
        warnings.append("missing manifest hash")
    if not hashes_present["split"]:
        warnings.append("missing split hash")

    location_metrics = {
        location: _metrics_artifacts_at(discovered, location) for location in BEST_EVAL_LOCATIONS
    }
    readable_at = {
        location: [artifact for artifact in artifacts if artifact.parse_ok is not False]
        for location, artifacts in location_metrics.items()
    }
    best_locations = [location for location in BEST_EVAL_LOCATIONS if readable_at[location]]
    last_metrics = _metrics_artifacts_at(discovered, LAST_EVAL_LOCATION)
    last_has_metrics = any(artifact.parse_ok is not False for artifact in last_metrics)
    if not best_locations:
        if last_has_metrics:
            warnings.append("no best_model evaluation evidence; last_model evidence never substituted")
            return QualificationResult(
                fold_dir=discovered.fold_dir,
                fold=discovered.fold,
                status=STATUS_QUALIFIED,
                reasons=(),
                warnings=tuple(warnings),
                evaluations=(),
            )
        if any(location_metrics[location] for location in BEST_EVAL_LOCATIONS):
            return QualificationResult(
                fold_dir=discovered.fold_dir,
                fold=discovered.fold,
                status=STATUS_REJECTED,
                reasons=("metrics_unreadable",),
                warnings=tuple(warnings),
                evaluations=(),
            )
        warnings.append("no evaluation evidence in any known location")
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_QUALIFIED,
            reasons=(),
            warnings=tuple(warnings),
            evaluations=(),
        )
    if len(best_locations) > 1:
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_QUARANTINED_AMBIGUOUS,
            reasons=("multiple_eval_locations:" + ",".join(best_locations),),
            warnings=tuple(warnings),
            evaluations=(),
        )
    location = best_locations[0]
    if last_has_metrics:
        warnings.append("last_model evaluation evidence ignored because best_model exists")

    metrics_artifacts = location_metrics[location]
    unreadable = [artifact for artifact in metrics_artifacts if artifact.parse_ok is False]
    readable = readable_at[location]
    for artifact in unreadable:
        warnings.append(f"unreadable metrics artifact ignored: {artifact.relative_path}")
    if not readable:
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_REJECTED,
            reasons=("metrics_unreadable",),
            warnings=tuple(warnings),
            evaluations=(),
        )

    groups: dict[tuple[str | None, str | None, str | None], list[tuple[Any, Any, str | None]]] = {}
    for artifact in readable:
        content = artifact.json_content
        if not isinstance(content, dict):
            warnings.append(f"metrics artifact without object content ignored: {artifact.relative_path}")
            continue
        backend, view, aggregation = _resolve_identity(artifact, config, content)
        namespace = _namespace_of(content)
        if namespace == _VALID_ONLY_NAMESPACE:
            warnings.append(
                f"rejected valid_only as headline namespace: {artifact.relative_path}"
            )
            continue
        if namespace is None:
            warnings.append(f"no headline metric namespace in: {artifact.relative_path}")
            continue
        groups.setdefault((backend, view, aggregation), []).append((artifact, content, namespace))

    if not groups:
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_REJECTED,
            reasons=("no_headline_namespace",),
            warnings=tuple(warnings),
            evaluations=(),
        )
    duplicates = {key: items for key, items in groups.items() if len(items) > 1}
    if duplicates:
        descriptions = [
            f"{artifact.relative_path} ({','.join(str(part) for part in key)})"
            for key, items in sorted(duplicates.items())
            for artifact, _, _ in items
        ]
        return QualificationResult(
            fold_dir=discovered.fold_dir,
            fold=discovered.fold,
            status=STATUS_QUARANTINED_AMBIGUOUS,
            reasons=("duplicate_metrics_same_identity:" + ";".join(descriptions),),
            warnings=tuple(warnings),
            evaluations=(),
        )

    evaluations: list[QualifiedEvaluation] = []
    for (backend, view, aggregation), [(artifact, content, namespace)] in sorted(groups.items()):
        evaluations.append(
            _build_fold_evaluation(
                discovered,
                config,
                protocol,
                hashes_present,
                artifact,
                content,
                backend,
                view,
                aggregation,
                namespace,
            )
        )
    summary_evaluations, summary_warnings = _build_summary_evaluations(
        discovered, config, protocol, hashes_present
    )
    evaluations.extend(summary_evaluations)
    warnings.extend(summary_warnings)

    return QualificationResult(
        fold_dir=discovered.fold_dir,
        fold=discovered.fold,
        status=STATUS_QUALIFIED,
        reasons=(),
        warnings=tuple(warnings),
        evaluations=tuple(evaluations),
    )


def build_evaluations_record(result: QualificationResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_EVALUATIONS,
        "attempt_id": result.attempt_id,
        "fold": result.fold,
        "evaluations": [evaluation.to_record() for evaluation in result.evaluations],
    }
