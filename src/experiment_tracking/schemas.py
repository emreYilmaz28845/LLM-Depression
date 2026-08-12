from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .canonical import parse_utc_timestamp
from .constants import (
    ARTIFACT_TYPES,
    CHECKPOINT_ROLES,
    JOB_EVENT_TYPES,
    JOB_STATUS_VALUES,
    JOB_TYPES,
    LIFECYCLE_STATES,
    SCHEMA_VERSION_ARTIFACTS,
    SCHEMA_VERSION_EVALUATIONS,
    SCHEMA_VERSION_EXPERIMENT_GROUP,
    SCHEMA_VERSION_JOB_EVENT,
    SCHEMA_VERSION_METADATA,
    SCHEMA_VERSION_REPORT,
    SCHEMA_VERSION_STATUS,
    WANDB_SYNC_STATUSES,
)
from .lifecycle import is_allowed_transition

_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[a-z0-9._-]+$")
_ATTEMPT_ID_PATTERN = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]+-[0-9a-f]{8}-[0-9a-f]{8}$"
)


class _FieldErrors:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def result(self) -> tuple[bool, list[str]]:
        return (not self.errors, list(self.errors))


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_timestamp(errors: _FieldErrors, value: Any, field: str) -> None:
    if not isinstance(value, str):
        errors.require(False, f"{field} must be a string")
        return
    try:
        parse_utc_timestamp(value)
    except ValueError:
        errors.require(False, f"{field} must match YYYY-MM-DDTHH:MM:SS.ffffffZ UTC format")


def _check_safe_id(errors: _FieldErrors, value: Any, field: str, allow_legacy: bool = False) -> None:
    if not isinstance(value, str):
        errors.require(False, f"{field} must be a string")
        return
    legacy_ok = allow_legacy and value.startswith("legacy-") and _SAFE_ID_PATTERN.fullmatch(value)
    modern_ok = _ATTEMPT_ID_PATTERN.fullmatch(value) is not None
    if not (legacy_ok or modern_ok):
        errors.require(False, f"{field} must be a valid attempt id or legacy attempt id")


def validate_metadata(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "metadata record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_METADATA,
        f"schema_version must be {SCHEMA_VERSION_METADATA}",
    )
    errors.require(
        record.get("group_id") is None or isinstance(record["group_id"], str),
        "group_id must be a string or null",
    )
    errors.require(
        isinstance(record.get("logical_run_name"), str) and record["logical_run_name"],
        "logical_run_name must be a non-empty string",
    )
    _check_safe_id(errors, record.get("attempt_id"), "attempt_id", allow_legacy=True)
    if record.get("supersedes_attempt_id") is not None:
        _check_safe_id(errors, record["supersedes_attempt_id"], "supersedes_attempt_id")
    errors.require(_is_integer(record.get("fold")), "fold must be an integer")
    errors.require(
        record.get("seed") is None or _is_integer(record["seed"]),
        "seed must be an integer or null",
    )
    _check_timestamp(errors, record.get("created_at_utc"), "created_at_utc")
    source = record.get("source")
    if not isinstance(source, dict):
        errors.require(False, "source must be an object")
    else:
        commit = source.get("git_commit")
        errors.require(
            commit is None or (isinstance(commit, str) and _GIT_COMMIT_PATTERN.fullmatch(commit)),
            "source.git_commit must be a 40-character hex SHA or null",
        )
        errors.require(
            source.get("git_branch") is None or isinstance(source["git_branch"], str),
            "source.git_branch must be a string or null",
        )
        errors.require(
            isinstance(source.get("git_dirty"), bool),
            "source.git_dirty must be a boolean",
        )
        deployed = source.get("deployed_source_sha256")
        errors.require(
            deployed is None or (isinstance(deployed, str) and _SHA256_PATTERN.fullmatch(deployed)),
            "source.deployed_source_sha256 must be a 64-character hex SHA or null",
        )
    research = record.get("research")
    if not isinstance(research, dict):
        errors.require(False, "research must be an object")
    else:
        errors.require(
            research.get("github_issue") is None or _is_integer(research["github_issue"]),
            "research.github_issue must be an integer or null",
        )
        errors.require(
            research.get("github_pr") is None or _is_integer(research["github_pr"]),
            "research.github_pr must be an integer or null",
        )
    hashes = record.get("hashes")
    if not isinstance(hashes, dict):
        errors.require(False, "hashes must be an object")
    else:
        for key in ("resolved_config_sha256", "manifest_sha256", "split_sha256"):
            value = hashes.get(key)
            errors.require(
                value is None or (isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)),
                f"hashes.{key} must be a 64-character hex SHA or null",
            )
    paths = record.get("paths")
    if not isinstance(paths, dict):
        errors.require(False, "paths must be an object")
    else:
        errors.require(
            isinstance(paths.get("run_config"), str) and paths["run_config"],
            "paths.run_config must be a non-empty string",
        )
        errors.require(
            paths.get("best_model") is None or isinstance(paths["best_model"], str),
            "paths.best_model must be a string or null",
        )
        errors.require(
            paths.get("local_evidence_root") is None or isinstance(paths["local_evidence_root"], str),
            "paths.local_evidence_root must be a string or null",
        )
    wandb = record.get("wandb")
    if not isinstance(wandb, dict):
        errors.require(False, "wandb must be an object")
    else:
        errors.require(
            wandb.get("project") is None or isinstance(wandb["project"], str),
            "wandb.project must be a string or null",
        )
        errors.require(
            wandb.get("entity") is None or isinstance(wandb["entity"], str),
            "wandb.entity must be a string or null",
        )
        errors.require(
            wandb.get("run_id") is None or isinstance(wandb["run_id"], str),
            "wandb.run_id must be a string or null",
        )
        errors.require(
            wandb.get("url") is None or isinstance(wandb["url"], str),
            "wandb.url must be a string or null",
        )
        errors.require(
            isinstance(wandb.get("sync_status"), str)
            and wandb["sync_status"] in WANDB_SYNC_STATUSES,
            f"wandb.sync_status must be one of {WANDB_SYNC_STATUSES}",
        )
    if "legacy_import" in record:
        errors.require(
            isinstance(record["legacy_import"], bool),
            "legacy_import must be a boolean when present",
        )
    return errors.result()


def validate_status(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "status record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_STATUS,
        f"schema_version must be {SCHEMA_VERSION_STATUS}",
    )
    _check_safe_id(errors, record.get("attempt_id"), "attempt_id", allow_legacy=True)
    errors.require(_is_integer(record.get("fold")), "fold must be an integer")
    errors.require(
        isinstance(record.get("state"), str) and record["state"] in LIFECYCLE_STATES,
        f"state must be one of {LIFECYCLE_STATES}",
    )
    _check_timestamp(errors, record.get("updated_at_utc"), "updated_at_utc")
    history = record.get("history")
    if not isinstance(history, list):
        errors.require(False, "history must be an array")
        return errors.result()
    for index, entry in enumerate(history):
        field = f"history[{index}]"
        if not isinstance(entry, dict):
            errors.require(False, f"{field} must be an object")
            continue
        from_state = entry.get("from")
        to_state = entry.get("to")
        errors.require(
            isinstance(from_state, str) and from_state in LIFECYCLE_STATES,
            f"{field}.from must be one of {LIFECYCLE_STATES}",
        )
        errors.require(
            isinstance(to_state, str) and to_state in LIFECYCLE_STATES,
            f"{field}.to must be one of {LIFECYCLE_STATES}",
        )
        if (
            isinstance(from_state, str)
            and isinstance(to_state, str)
            and from_state in LIFECYCLE_STATES
            and to_state in LIFECYCLE_STATES
        ):
            errors.require(
                is_allowed_transition(from_state, to_state),
                f"{field} transition {from_state} -> {to_state} is not allowed",
            )
        _check_timestamp(errors, entry.get("at_utc"), f"{field}.at_utc")
        errors.require(
            entry.get("reason") is None or isinstance(entry["reason"], str),
            f"{field}.reason must be a string or null",
        )
    if isinstance(record.get("state"), str) and history:
        last = history[-1]
        if isinstance(last, dict):
            errors.require(
                last.get("to") == record["state"],
                "state must equal the last history entry's target state",
            )
    return errors.result()


def validate_job_event(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "job event must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_JOB_EVENT,
        f"schema_version must be {SCHEMA_VERSION_JOB_EVENT}",
    )
    event_id = record.get("event_id")
    errors.require(
        isinstance(event_id, str) and _UUID4_PATTERN.fullmatch(event_id) is not None,
        "event_id must be a UUID4 string",
    )
    _check_safe_id(errors, record.get("attempt_id"), "attempt_id", allow_legacy=True)
    errors.require(_is_integer(record.get("fold")), "fold must be an integer")
    errors.require(
        isinstance(record.get("job_key"), str) and record["job_key"],
        "job_key must be a non-empty string",
    )
    errors.require(
        isinstance(record.get("job_type"), str) and record["job_type"] in JOB_TYPES,
        f"job_type must be one of {JOB_TYPES}",
    )
    errors.require(
        isinstance(record.get("event_type"), str) and record["event_type"] in JOB_EVENT_TYPES,
        f"event_type must be one of {JOB_EVENT_TYPES}",
    )
    for key in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "reason", "resubmission_of_job_id"):
        value = record.get(key)
        errors.require(value is None or isinstance(value, str), f"{key} must be a string or null")
    dependencies = record.get("dependency_job_ids")
    errors.require(
        isinstance(dependencies, list) and all(isinstance(item, str) for item in dependencies),
        "dependency_job_ids must be an array of strings",
    )
    errors.require(
        record.get("status") is None
        or (isinstance(record["status"], str) and record["status"] in JOB_STATUS_VALUES),
        f"status must be one of {JOB_STATUS_VALUES} or null",
    )
    _check_timestamp(errors, record.get("at_utc"), "at_utc")
    return errors.result()


def validate_artifacts(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "artifacts record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_ARTIFACTS,
        f"schema_version must be {SCHEMA_VERSION_ARTIFACTS}",
    )
    _check_safe_id(errors, record.get("attempt_id"), "attempt_id", allow_legacy=True)
    errors.require(_is_integer(record.get("fold")), "fold must be an integer")
    items = record.get("artifacts")
    if not isinstance(items, list):
        errors.require(False, "artifacts must be an array")
        return errors.result()
    for index, item in enumerate(items):
        field = f"artifacts[{index}]"
        if not isinstance(item, dict):
            errors.require(False, f"{field} must be an object")
            continue
        artifact_id = item.get("artifact_id")
        errors.require(
            isinstance(artifact_id, str) and re.fullmatch(r"art-[0-9a-f]{24}", artifact_id) is not None,
            f"{field}.artifact_id must match art-<24 hex>",
        )
        errors.require(
            isinstance(item.get("artifact_type"), str) and item["artifact_type"] in ARTIFACT_TYPES,
            f"{field}.artifact_type must be one of {ARTIFACT_TYPES}",
        )
        errors.require(
            isinstance(item.get("role"), str) and item["role"],
            f"{field}.role must be a non-empty string",
        )
        errors.require(
            isinstance(item.get("path"), str) and item["path"],
            f"{field}.path must be a non-empty string",
        )
        sha = item.get("sha256")
        errors.require(
            sha is None or (isinstance(sha, str) and _SHA256_PATTERN.fullmatch(sha)),
            f"{field}.sha256 must be a 64-character hex SHA or null",
        )
        errors.require(
            item.get("size_bytes") is None or _is_integer(item["size_bytes"]),
            f"{field}.size_bytes must be an integer or null",
        )
        for key in ("exists_on_mn5", "exists_locally"):
            value = item.get(key)
            errors.require(value is None or isinstance(value, bool), f"{field}.{key} must be a boolean or null")
        errors.require(
            isinstance(item.get("locally_verified"), bool),
            f"{field}.locally_verified must be a boolean",
        )
    return errors.result()


def validate_evaluations(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "evaluations record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_EVALUATIONS,
        f"schema_version must be {SCHEMA_VERSION_EVALUATIONS}",
    )
    _check_safe_id(errors, record.get("attempt_id"), "attempt_id", allow_legacy=True)
    errors.require(_is_integer(record.get("fold")), "fold must be an integer")
    items = record.get("evaluations")
    if not isinstance(items, list):
        errors.require(False, "evaluations must be an array")
        return errors.result()
    for index, item in enumerate(items):
        field = f"evaluations[{index}]"
        if not isinstance(item, dict):
            errors.require(False, f"{field} must be an object")
            continue
        evaluation_id = item.get("evaluation_id")
        errors.require(
            isinstance(evaluation_id, str) and re.fullmatch(r"eval-[0-9a-f]{24}", evaluation_id) is not None,
            f"{field}.evaluation_id must match eval-<24 hex>",
        )
        for key in (
            "dataset",
            "split_name",
            "split_protocol",
            "checkpoint_path",
            "backend",
            "aggregation",
            "metric_namespace",
            "metrics_artifact_path",
        ):
            errors.require(
                isinstance(item.get(key), str) and item[key],
                f"{field}.{key} must be a non-empty string",
            )
        errors.require(
            item.get("evaluation_view") is None
            or (isinstance(item["evaluation_view"], str) and item["evaluation_view"]),
            f"{field}.evaluation_view must be a non-empty string or null",
        )
        errors.require(
            item.get("predictions_artifact_path") is None
            or (isinstance(item["predictions_artifact_path"], str) and item["predictions_artifact_path"]),
            f"{field}.predictions_artifact_path must be a non-empty string or null",
        )
        errors.require(
            isinstance(item.get("checkpoint_role"), str)
            and item["checkpoint_role"] in CHECKPOINT_ROLES,
            f"{field}.checkpoint_role must be one of {CHECKPOINT_ROLES}",
        )
        metrics = item.get("metrics")
        if not isinstance(metrics, list):
            errors.require(False, f"{field}.metrics must be an array")
        else:
            for metric_index, metric in enumerate(metrics):
                metric_field = f"{field}.metrics[{metric_index}]"
                if not isinstance(metric, dict):
                    errors.require(False, f"{metric_field} must be an object")
                    continue
                errors.require(
                    isinstance(metric.get("name"), str) and metric["name"],
                    f"{metric_field}.name must be a non-empty string",
                )
                value = metric.get("value")
                errors.require(
                    value is None
                    or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)),
                    f"{metric_field}.value must be a finite number or null",
                )
                errors.require(
                    metric.get("support") is None or _is_integer(metric["support"]),
                    f"{metric_field}.support must be an integer or null",
                )
        errors.require(
            isinstance(item.get("locally_verified"), bool),
            f"{field}.locally_verified must be a boolean",
        )
        errors.require(
            isinstance(item.get("reportable"), bool),
            f"{field}.reportable must be a boolean",
        )
        errors.require(
            isinstance(item.get("warnings"), list)
            and all(isinstance(warning, str) for warning in item["warnings"]),
            f"{field}.warnings must be an array of strings",
        )
    return errors.result()


def validate_experiment_group(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "experiment group record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_EXPERIMENT_GROUP,
        f"schema_version must be {SCHEMA_VERSION_EXPERIMENT_GROUP}",
    )
    for key in ("group_id", "title", "research_question", "dataset", "baseline", "treatment"):
        errors.require(
            isinstance(record.get(key), str) and record[key],
            f"{key} must be a non-empty string",
        )
    errors.require(
        isinstance(record.get("expected_seeds"), list)
        and all(_is_integer(item) for item in record["expected_seeds"]),
        "expected_seeds must be an array of integers",
    )
    errors.require(
        isinstance(record.get("expected_folds"), list)
        and all(_is_integer(item) for item in record["expected_folds"]),
        "expected_folds must be an array of integers",
    )
    primary_metric = record.get("primary_metric")
    if not isinstance(primary_metric, dict):
        errors.require(False, "primary_metric must be an object")
    else:
        for key in ("namespace", "name", "backend", "aggregation", "evaluation_view"):
            errors.require(
                isinstance(primary_metric.get(key), str) and primary_metric[key],
                f"primary_metric.{key} must be a non-empty string",
            )
    errors.require(
        record.get("github_issue") is None or _is_integer(record["github_issue"]),
        "github_issue must be an integer or null",
    )
    errors.require(
        record.get("github_pr") is None or _is_integer(record["github_pr"]),
        "github_pr must be an integer or null",
    )
    return errors.result()


def validate_report(record: Any) -> tuple[bool, list[str]]:
    errors = _FieldErrors()
    if not isinstance(record, dict):
        errors.require(False, "report record must be an object")
        return errors.result()
    errors.require(
        record.get("schema_version") == SCHEMA_VERSION_REPORT,
        f"schema_version must be {SCHEMA_VERSION_REPORT}",
    )
    for key in ("logical_run_name", "attempt_id", "group_id", "dataset", "modality"):
        if key in record:
            errors.require(
                record[key] is None or isinstance(record[key], str),
                f"{key} must be a string or null",
            )
    if "status" in record:
        errors.require(
            record["status"] is None or isinstance(record["status"], str),
            "status must be a string or null",
        )
    for key in ("git", "hashes", "paths", "wandb"):
        if key in record:
            errors.require(
                record[key] is None or isinstance(record[key], dict),
                f"{key} must be an object or null",
            )
    for key in ("jobs", "metrics", "warnings", "exclusions"):
        if key in record:
            errors.require(
                record[key] is None or isinstance(record[key], list),
                f"{key} must be an array or null",
            )
    if "conclusion" in record:
        errors.require(
            record["conclusion"] is None or isinstance(record["conclusion"], str),
            "conclusion must be a string or null",
        )
    return errors.result()


_VALIDATORS: dict[str, Any] = {
    SCHEMA_VERSION_METADATA: validate_metadata,
    SCHEMA_VERSION_STATUS: validate_status,
    SCHEMA_VERSION_JOB_EVENT: validate_job_event,
    SCHEMA_VERSION_ARTIFACTS: validate_artifacts,
    SCHEMA_VERSION_EVALUATIONS: validate_evaluations,
    SCHEMA_VERSION_EXPERIMENT_GROUP: validate_experiment_group,
    SCHEMA_VERSION_REPORT: validate_report,
}


def validate_record(schema_version: str, record: Any) -> tuple[bool, list[str]]:
    validator = _VALIDATORS.get(schema_version)
    if validator is None:
        return False, [f"unknown schema version: {schema_version!r}"]
    return validator(record)
