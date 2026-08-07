from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import yaml

from .canonical import canonical_sha256
from .constants import WANDB_PROJECT
from .discovery import DiscoveredRun, discover_run_at
from .qualification import (
    STATUS_QUALIFIED,
    QualificationResult,
    qualify_run,
)

EXPORT_SCHEMA_VERSION = "audiollm.wandb_export.v1"
LEGACY_WANDB_ID_ALGORITHM_VERSION = "legacy-wandb-v1"

_SENSITIVE_KEY_FRAGMENTS = (
    "audio",
    "waveform",
    "transcript",
    "prompt",
    "subject_id",
    "patient_id",
    "participant_id",
    "api_key",
    "token",
    "password",
    "secret",
    "credential",
)

_SENSITIVE_VALUE_FRAGMENTS = (
    "gpfs/projects/etur92",
    "/media/emre",
    "api_key=",
    "bearer ",
)

_SUBJECT_DATA_VALUE_FRAGMENTS = (
    "transcript",
    "prompt",
    "subject_id",
    "patient_id",
    "participant_id",
    "api_key",
    "password",
    "secret",
    "credential",
)

_ABSOLUTE_PATH_PREFIXES = ("/", "~", "c:\\")

_DROP = object()


def is_sensitive_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _has_sensitive_value(value: str) -> bool:
    lowered = value.lower()
    return any(
        fragment in lowered
        for fragment in (*_SENSITIVE_VALUE_FRAGMENTS, *_SUBJECT_DATA_VALUE_FRAGMENTS)
    )


def is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_ABSOLUTE_PATH_PREFIXES)


def _filter(value: Any, path: str, exclusions: list[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(key):
                exclusions.append(f"{path}.{key} (sensitive key)")
                continue
            filtered = _filter(item, f"{path}.{key}", exclusions)
            if filtered is not _DROP:
                out[str(key)] = filtered
        if not out and value:
            return _DROP
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            filtered = _filter(item, f"{path}[{index}]", exclusions)
            if filtered is not _DROP:
                out.append(filtered)
        return out
    if isinstance(value, str):
        if is_absolute_path(value):
            exclusions.append(f"{path} (absolute path)")
            return _DROP
        if _has_sensitive_value(value):
            exclusions.append(f"{path} (sensitive value)")
            return _DROP
    return value


def filter_safe(value: Any) -> tuple[Any, list[str]]:
    exclusions: list[str] = []
    return _filter(value, "root", exclusions), exclusions


def legacy_wandb_id(attempt_id: str, fold: int, evaluation_id: str) -> str:
    payload = {
        "algorithm_version": LEGACY_WANDB_ID_ALGORITHM_VERSION,
        "attempt_id": attempt_id,
        "fold": fold,
        "evaluation_id": evaluation_id,
    }
    return f"wandb-{canonical_sha256(payload)[:24]}"


def _run_display_name(
    attempt_id: str, fold: int, logical_run_name: str | None, evaluation_id: str | None
) -> str:
    if logical_run_name is None:
        return f"attempt-{attempt_id[-8:]}-fold{fold}"
    return f"{logical_run_name}-attempt{attempt_id[-8:]}-fold{fold}"


def _history_curves(training_history: Any) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    curves: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(training_history, list) or not training_history:
        return curves, False
    names = (
        "train_loss",
        "selection_loss",
        "inner_val_loss",
        "selection_positive_f1",
        "inner_val_positive_f1",
        "selection_macro_f1",
        "selection_accuracy",
    )
    for row in training_history:
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch")
        for name in names:
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            curves.setdefault(f"train/{name}" if name == "train_loss" else f"selection/{name}", []).append(
                {"epoch": epoch, "value": value}
            )
    return curves, bool(curves)


def _summary_metric_name(evaluation: dict[str, Any], metric_name: str) -> str:
    namespace = evaluation.get("metric_namespace") or "unknown"
    backend = evaluation.get("backend") or "unknown"
    aggregation = evaluation.get("aggregation") or "unknown"
    view = evaluation.get("evaluation_view")
    split_name = evaluation.get("split_name")
    if aggregation in ("pooled_subject_level", "fold_mean"):
        prefix = f"aggregate/{aggregation}"
    elif split_name == "val":
        prefix = "selection"
    elif split_name == "test":
        prefix = f"test/{view}" if view else "test"
    else:
        prefix = str(split_name or "unknown")
    return f"{prefix}/{backend}/{aggregation}/{namespace}/{metric_name}"


def _build_plan(
    *,
    attempt_id: str,
    fold: int,
    logical_run: dict[str, Any],
    attempt: dict[str, Any],
    evaluations: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    run_dir: str | None,
    project: str,
) -> dict[str, Any]:
    resolved_config: Any = None
    run_config_candidates = [
        artifact for artifact in artifacts if artifact.get("artifact_type") == "run_config"
    ]
    run_config_path = next(
        (artifact["path"] for artifact in run_config_candidates if artifact.get("role") == "run_config"),
        None,
    )
    if run_config_path is None and run_config_candidates:
        run_config_path = run_config_candidates[0]["path"]
    config_path: Path | None = None
    if run_dir and run_config_path:
        candidate = Path(run_dir) / run_config_path
        if candidate.is_file():
            config_path = candidate
    if config_path is not None:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("config"), dict):
                resolved_config = data["config"]
        except (OSError, yaml.YAMLError):
            resolved_config = None
    safe_config, config_exclusions = filter_safe(resolved_config) if resolved_config is not None else (None, [])

    curves: dict[str, list[dict[str, Any]]] = {}
    history_ok = False
    history_artifact = next(
        (artifact for artifact in artifacts if artifact.get("artifact_type") == "training_history"),
        None,
    )
    if history_artifact and run_dir:
        history_path = Path(run_dir) / history_artifact["path"]
        if history_path.is_file():
            try:
                curves, history_ok = _history_curves(json.loads(history_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                curves, history_ok = {}, False

    summary_metrics: dict[str, Any] = {}
    incomplete_reasons: list[str] = []
    first_evaluation_id: str | None = None
    for entry in sorted(evaluations, key=lambda item: item["evaluation"]["evaluation_id"]):
        evaluation = entry["evaluation"]
        if first_evaluation_id is None:
            first_evaluation_id = evaluation["evaluation_id"]
        if evaluation.get("reportable") != 1:
            incomplete_reasons.append(f"evaluation not reportable: {evaluation['evaluation_id']}")
        for metric in sorted(entry["metrics"], key=lambda item: item["metric_name"]):
            if metric.get("metric_value") is None:
                continue
            summary_metrics[_summary_metric_name(evaluation, metric["metric_name"])] = metric["metric_value"]
    if not summary_metrics:
        incomplete_reasons.append("no qualified headline metrics")
    if not history_ok:
        incomplete_reasons.append("training history missing or unreadable")

    if str(attempt_id).startswith("legacy-"):
        run_id = legacy_wandb_id(attempt_id, fold, first_evaluation_id or "")
    else:
        run_id = f"{attempt_id}-fold{fold}"

    slurm_ids = sorted(
        {job["slurm_job_id"] for job in jobs if job.get("slurm_job_id")}
    )
    tags = ["legacy"] if str(attempt_id).startswith("legacy-") else []
    status = "incomplete" if incomplete_reasons else "complete"
    if status == "incomplete":
        tags.append("incomplete")

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "name": _run_display_name(attempt_id, fold, logical_run.get("logical_run_name"), first_evaluation_id),
        "project": project,
        "entity": None,
        "group": None,
        "job_type": "train_eval",
        "identity": {
            "attempt_id": attempt_id,
            "fold": fold,
            "logical_run_name": logical_run.get("logical_run_name"),
            "dataset": logical_run.get("dataset"),
            "modality": logical_run.get("modality"),
            "method": None,
            "seed": logical_run.get("seed"),
            "evaluation_id": first_evaluation_id,
        },
        "safe_config": safe_config,
        "hashes": {
            "resolved_config_sha256": attempt.get("resolved_config_sha256"),
            "manifest_sha256": attempt.get("manifest_sha256"),
            "split_sha256": attempt.get("split_sha256"),
        },
        "git": {
            "git_commit": attempt.get("git_commit"),
            "github_issue": attempt.get("github_issue"),
            "github_pr": attempt.get("github_pr"),
        },
        "slurm_ids": slurm_ids,
        "epoch_curves": curves,
        "summary_metrics": summary_metrics,
        "status": status,
        "tags": sorted(tags),
        "incomplete_reasons": incomplete_reasons,
        "exclusions": sorted(set(config_exclusions)),
        "safe": True,
    }


def build_export_plan_from_result(
    discovered: DiscoveredRun,
    result: QualificationResult,
    *,
    project: str = WANDB_PROJECT,
) -> dict[str, Any] | None:
    if result.status != STATUS_QUALIFIED or not result.evaluations:
        return None
    config = discovered.resolved_config or {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), str) else None
    seed = config.get("seed") if isinstance(config.get("seed"), int) else None
    evaluations = [
        {
            "evaluation": evaluation.to_record(),
            "metrics": [
                {"metric_name": metric.name, "metric_value": metric.value, "support": metric.support}
                for metric in evaluation.metrics
            ],
        }
        for evaluation in sorted(result.evaluations, key=lambda item: item.evaluation_id)
    ]
    attempt = {
        "resolved_config_sha256": discovered.resolved_config_sha256,
        "manifest_sha256": discovered.protocol.get("manifest_hash"),
        "split_sha256": discovered.protocol.get("split_metadata_hash"),
        "git_commit": None,
        "github_issue": None,
        "github_pr": None,
    }
    logical = {
        "logical_run_name": discovered.run_name,
        "dataset": dataset,
        "modality": discovered.modality,
        "seed": seed,
    }
    artifacts = [
        {
            "artifact_type": artifact.artifact_type,
            "role": artifact.kind,
            "path": artifact.relative_path,
        }
        for artifact in discovered.artifacts
    ]
    return _build_plan(
        attempt_id=result.attempt_id,
        fold=discovered.fold,
        logical_run=logical,
        attempt=attempt,
        evaluations=evaluations,
        artifacts=artifacts,
        jobs=[],
        run_dir=discovered.fold_dir,
        project=project,
    )


def build_export_plan(
    connection: Any,
    attempt_id: str,
    fold: int | None = None,
    *,
    project: str = WANDB_PROJECT,
) -> dict[str, Any] | None:
    from .registry import RegistryError, show_attempt

    try:
        payload = show_attempt(connection, attempt_id, fold)
    except RegistryError:
        return None
    run_dir = payload["folds"][0]["run_dir"] if payload["folds"] else None
    return _build_plan(
        attempt_id=attempt_id,
        fold=payload["folds"][0]["fold"] if payload["folds"] else fold,
        logical_run=payload["logical_run"] or {},
        attempt=payload["attempt"],
        evaluations=payload["evaluations"],
        artifacts=payload["artifacts"],
        jobs=payload["jobs"],
        run_dir=run_dir,
        project=project,
    )


class WandbAdapter(Protocol):
    def create_run(
        self,
        *,
        run_id: str,
        name: str | None,
        config: Any,
        project: str,
        entity: str | None,
        mode: str,
        tags: list[str],
    ) -> None: ...
    def log_curves(self, *, run_id: str, curves: dict[str, list[dict[str, Any]]]) -> None: ...
    def log_summary(self, *, run_id: str, summary: dict[str, Any]) -> None: ...
    def set_status(self, *, run_id: str, status: str, tags: list[str]) -> None: ...


def _normalize_mode(mode: str) -> str:
    if mode == "cloud":
        return "online"
    if mode == "dry_run":
        return "dryrun"
    return mode


class RealWandbAdapter:
    def __init__(self, entity: str | None = None) -> None:
        self._entity = entity

    @staticmethod
    def _wandb() -> Any:
        try:
            import wandb  # type: ignore

            return wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is not installed; install it in the llmdep4090 environment before "
                "performing a real export"
            ) from error

    def create_run(self, *, run_id: str, name: str | None, config: Any, project: str, entity: str | None, mode: str, tags: list[str]) -> None:
        wandb = self._wandb()
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            id=run_id,
            name=name,
            project=project,
            entity=entity if entity is not None else self._entity,
            config=config,
            tags=tags,
            mode=_normalize_mode(mode),
        )

    def log_curves(self, *, run_id: str, curves: dict[str, list[dict[str, Any]]]) -> None:
        wandb = self._wandb()
        for name, points in curves.items():
            for point in points:
                wandb.log({name: point["value"], "system/epoch": point["epoch"]})

    def log_summary(self, *, run_id: str, summary: dict[str, Any]) -> None:
        self._wandb().summary.update(summary)

    def set_status(self, *, run_id: str, status: str, tags: list[str]) -> None:
        self._wandb().summary.update({"lifecycle/status": status})


def _export_config(plan: dict[str, Any]) -> dict[str, Any]:
    config = dict(plan["safe_config"] or {})
    identity = plan["identity"]
    config["tracking/logical_run_name"] = identity.get("logical_run_name")
    config["tracking/attempt_id"] = identity.get("attempt_id")
    config["tracking/fold"] = identity.get("fold")
    config["tracking/evaluation_id"] = identity.get("evaluation_id")
    return config


def execute_export(
    plan: dict[str, Any],
    adapter: WandbAdapter | None,
    *,
    mode: str = "offline",
    entity: str | None = None,
) -> dict[str, Any]:
    if mode == "dry_run":
        return {"mode": "dry_run", "run_id": plan["run_id"], "status": plan["status"]}
    if adapter is None:
        raise ValueError("a WandbAdapter is required for non-dry-run export")
    run_id = plan["run_id"]
    adapter.create_run(
        run_id=run_id,
        name=plan.get("name"),
        config=_export_config(plan),
        project=plan["project"],
        entity=entity,
        mode=mode,
        tags=plan["tags"],
    )
    if plan["epoch_curves"]:
        adapter.log_curves(run_id=run_id, curves=plan["epoch_curves"])
    if plan["summary_metrics"]:
        adapter.log_summary(run_id=run_id, summary=plan["summary_metrics"])
    adapter.set_status(run_id=run_id, status=plan["status"], tags=plan["tags"])
    return {"mode": mode, "run_id": run_id, "status": plan["status"]}
