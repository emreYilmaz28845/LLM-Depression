"""Tracked attempt campaign for raw hidden-state LogReg head cells.

Mirrors ``posthoc_head_campaign.py`` but for the locked LogReg protocol:
one modern attempt per (parent fold x dataset x condition x backbone x
training seed), created around a qualified hidden-feature cache, with the
fixed head seed 1337 and method-specific prediction backends:

- standalone: ``qwen_hidden_logreg_raw`` / ``gemma4_hidden_logreg_raw``;
- merged: the same names suffixed ``_symmetric_merged``.

Merged attempts carry per-dataset evaluation records built from
``metrics_by_dataset.json`` and the dataset-tagged subject-level predictions.
Never writes the SQLite registry; MN5 jobs write sidecars only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiment_tracking.canonical import (
    canonical_sha256,
    format_utc_timestamp,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.experiment_tracking.identity import (
    artifact_id,
    evaluation_id,
    new_attempt_id,
    sanitize_logical_run_name,
)
from src.experiment_tracking.lifecycle import (
    StatusRecord,
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
)
from src.features.hidden_classifier_policy import (
    cache_identity,
    file_identity,
)
from src.features.posthoc_head_campaign import (
    EVALUATION_METRIC_NAMES,
    PosthocError,
    deployed_source_sha256,
    mark_deployed,
    record_job,
    transition,
)

TASK_SCHEMA_VERSION = "audiollm.logreg_head_task.v1"
RUN_CONFIG_SCHEMA_VERSION = "audiollm.logreg_head_run.v1"
LOGREG_RAW_EXPERIMENT_ID = "logreg_raw_harmonized_v1"
HEAD_SEED = 1337
QWEN_LOGREG_RAW_BACKEND = "qwen_hidden_logreg_raw"
GEMMA4_LOGREG_RAW_BACKEND = "gemma4_hidden_logreg_raw"
MERGED_BACKEND_SUFFIX = "_symmetric_merged"

STANDALONE_ARTIFACT_FILES = (
    ("pipeline.joblib", "checkpoint", "classifier_pipeline"),
    ("classifier_metadata.json", "audit", "classifier_metadata"),
    ("sampling_audit.json", "audit", "sampling_audit"),
    ("result_config.json", "audit", "result_config"),
    ("metrics.json", "metrics", "classifier_metrics"),
    ("predictions_sample_level.jsonl", "predictions", "predictions_sample_level"),
    ("predictions_sample_level.csv", "predictions", "predictions_sample_level_csv"),
    ("predictions_subject_level.jsonl", "predictions", "predictions_subject_level"),
    ("predictions_subject_level.csv", "predictions", "predictions_subject_level_csv"),
)
MERGED_ARTIFACT_FILES = (
    ("pipeline.joblib", "checkpoint", "classifier_pipeline"),
    ("classifier_metadata.json", "audit", "classifier_metadata"),
    ("metrics_by_dataset.json", "metrics", "classifier_metrics_by_dataset"),
    ("predictions_subject_level.jsonl", "predictions", "predictions_subject_level"),
    ("predictions_subject_level.csv", "predictions", "predictions_subject_level_csv"),
)


def prediction_backend(backend: str, *, merged: bool) -> str:
    base = (
        GEMMA4_LOGREG_RAW_BACKEND
        if str(backend or "").strip().lower() == "gemma4"
        else QWEN_LOGREG_RAW_BACKEND
    )
    return f"{base}{MERGED_BACKEND_SUFFIX}" if merged else base


def load_task_spec(path: str | Path) -> dict[str, Any]:
    spec = read_json(path)
    if str(spec.get("schema_version")) != TASK_SCHEMA_VERSION:
        raise PosthocError(
            f"task spec has unsupported schema_version {spec.get('schema_version')!r}"
        )
    for key in (
        "dataset",
        "modality",
        "condition",
        "fold",
        "seed",
        "family",
        "backend",
        "cache_dir",
        "experiment_id",
        "group_id",
        "run_name",
        "branch",
        "merged_sha",
    ):
        if key not in spec:
            raise PosthocError(f"task spec missing required key {key!r}")
    family = str(spec["family"])
    if family not in {"standalone", "merged"}:
        raise PosthocError(f"unsupported logreg task family {family!r}")
    backend = str(spec["backend"])
    if backend not in {"qwen", "gemma4"}:
        raise PosthocError(f"unsupported logreg backbone {backend!r}")
    if str(spec["experiment_id"]) != LOGREG_RAW_EXPERIMENT_ID:
        raise PosthocError(
            f"logreg task experiment_id must be {LOGREG_RAW_EXPERIMENT_ID!r}, "
            f"got {spec['experiment_id']!r}"
        )
    if int(spec.get("head_seed", HEAD_SEED)) != HEAD_SEED:
        raise PosthocError(
            f"LogReg head seed is fixed to {HEAD_SEED}, got {spec.get('head_seed')!r}"
        )
    if family == "merged":
        stage = str(spec.get("stage") or "")
        if stage not in {"cv", "final"}:
            raise PosthocError("merged logreg task requires stage cv|final")
    return spec


def _cache_identity_for_spec(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    if str(spec.get("family")) == "merged":
        names = (
            "feature_metadata.json",
            "outer_train.npz",
            "outer_train_rows.jsonl",
            "outer_holdout.npz",
            "outer_holdout_rows.jsonl",
        )
        return {name: file_identity(cache_dir / name) for name in names}
    return cache_identity(cache_dir)


def _verify_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(spec["cache_dir"])
    metadata_path = (
        cache_dir / "feature_metadata.json"
        if str(spec.get("family")) == "merged"
        else cache_dir / "extraction_metadata.json"
    )
    if not metadata_path.is_file():
        raise PosthocError(f"logreg cache metadata missing: {metadata_path}")
    metadata = read_json(metadata_path)
    expected = spec.get("cache_identity_sha256")
    if expected:
        actual = canonical_sha256(metadata)
        if actual != str(expected):
            raise PosthocError(
                f"cache identity changed for {cache_dir}: {actual} != {expected}"
            )
    return metadata


def _require_clean_production_source(repo_root: Path) -> None:
    provenance = repo_root / ".provenance"
    if provenance.is_dir():
        return
    import subprocess

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise PosthocError("cannot determine repository cleanliness")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        raise PosthocError(
            f"refusing to create an attempt from a dirty production source: {dirty[:5]}"
        )


def _git_commit(repo_root: Path) -> str:
    provenance = repo_root / ".provenance" / "git_commit.txt"
    if provenance.exists():
        return provenance.read_text(encoding="utf-8").strip()
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:  # pragma: no cover - local development fallback
        return "unknown"


def _source_manifest_records(repo_root: Path) -> list[dict[str, Any]]:
    manifest_path = repo_root / ".provenance" / "source_manifest.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        return list(payload.get("files", payload if isinstance(payload, list) else []))
    import subprocess

    try:
        listing = subprocess.check_output(
            ["git", "ls-files"], cwd=str(repo_root), text=True
        ).splitlines()
    except Exception:  # pragma: no cover - local development fallback
        return []
    records: list[dict[str, Any]] = []
    for relative in sorted(listing):
        full = repo_root / relative
        if not full.is_file():
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
            }
        )
    return records


def build_run_config(spec: dict[str, Any], attempt_id: str, source_sha: str) -> dict[str, Any]:
    cache_dir = Path(spec["cache_dir"])
    cache_metadata = _verify_cache_identity(spec)
    cache_config = cache_metadata.get("cache_config") or {}
    parent = spec.get("parent") or {}
    qualifiers = spec.get("evaluation_qualifiers") or {}
    merged = str(spec.get("family")) == "merged"
    cache_identity_payload = _cache_identity_for_spec(spec, cache_dir)
    scientific = {
        "dataset": spec["dataset"],
        "modality": spec["modality"],
        "method": LOGREG_RAW_EXPERIMENT_ID,
        "fold": int(spec["fold"]),
        "seed": int(spec["seed"]),
        "stage": spec.get("stage"),
        "condition": str(spec["condition"]),
        "parent": {
            "parent_attempt_id": parent.get("parent_attempt_id"),
            "parent_fold_dir": parent.get("parent_fold_dir"),
            "parent_checkpoint_role": "best_model",
            "parent_checkpoint_path": parent.get("parent_checkpoint_path"),
            "adapter_config_sha256": parent.get("adapter_config_sha256"),
            "adapter_sha256": parent.get("adapter_sha256"),
        },
        "hashes": {
            "manifest_sha256": cache_metadata.get("manifest_sha256"),
            "split_sha256": cache_metadata.get("split_metadata_sha256"),
        },
        "base_model": {
            "id": str(cache_config.get("model_name_or_path") or ""),
            "revision": cache_config.get("model_revision"),
            "path": str(cache_config.get("checkpoint_dir") or ""),
        },
        "hidden_state": {
            "layer": "final",
            "pooling": "last_valid_prompt_token",
            "dimension": cache_config.get("hidden_dimension"),
            "dtype": "float32",
            "cache_schema": cache_config.get("schema_version"),
        },
        "cache": {
            "cache_dir": str(cache_dir),
            "cache_identity": cache_identity_payload,
            "cache_identity_sha256": canonical_sha256(cache_identity_payload),
            "extraction_metadata_sha256": sha256_file(
                cache_dir
                / (
                    "feature_metadata.json"
                    if merged
                    else "extraction_metadata.json"
                )
            ),
            "adapter_config_sha256": cache_config.get("adapter_config_sha256"),
            "adapter_sha256": cache_config.get("adapter_sha256"),
        },
        "classifier": {
            "experiment_id": LOGREG_RAW_EXPERIMENT_ID,
            "classifier_family": "logreg_raw",
            "variants": ["logreg_raw"],
            "head_seed": HEAD_SEED,
            "seed": int(spec["seed"]),
            "sampling_mode": "none",
            "threshold": 0.5,
            "backend_policy": "harmonized_hidden_logreg_raw_v1",
            "prediction_backend": prediction_backend(spec["backend"], merged=merged),
            "model_backend": spec["backend"],
        },
        "evaluation": {
            "dataset": qualifiers.get("dataset", spec["dataset"]),
            "split_name": qualifiers.get("split_name"),
            "split_protocol": qualifiers.get("split_protocol"),
            "checkpoint_role": "best_model",
            "checkpoint_path": str(parent.get("parent_checkpoint_path") or ""),
            "evaluation_view": qualifiers.get(
                "evaluation_view", "harmonized_all_windows_full_coverage"
            ),
            "aggregation": qualifiers.get("aggregation", "subject_level"),
            "metric_namespace": qualifiers.get("metric_namespace", "headline/binary_strict"),
            "support": qualifiers.get("support"),
            "metric_names": list(EVALUATION_METRIC_NAMES),
        },
        "family": spec.get("family"),
        "implementation": {
            "branch": spec["branch"],
            "merged_sha": spec["merged_sha"],
            "pr": spec.get("pr"),
            "deployed_source_sha256": source_sha,
        },
    }
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "config": scientific,
        "manifest_sha256": scientific["hashes"]["manifest_sha256"],
        "split_metadata_hash": scientific["hashes"]["split_sha256"],
        "tracking": {
            "schema_version": "audiollm.tracking.v1",
            "group_id": spec["group_id"],
            "logical_run_name": sanitize_logical_run_name(str(spec["run_name"])),
            "attempt_id": attempt_id,
            "fold": int(spec["fold"]),
        },
    }


def create_attempt(
    *,
    repo_root: str | Path,
    attempt_dir: str | Path,
    task_spec: str | Path,
) -> dict[str, Any]:
    spec = load_task_spec(task_spec)
    root = Path(repo_root)
    _require_clean_production_source(root)
    _verify_cache_identity(spec)
    attempt_path = Path(attempt_dir)
    if attempt_path.name != LOGREG_RAW_EXPERIMENT_ID:
        raise PosthocError(
            f"attempt dir must end with {LOGREG_RAW_EXPERIMENT_ID!r}: {attempt_path}"
        )
    if attempt_path.parent.name != f"fold_{int(spec['fold'])}":
        raise PosthocError(
            f"attempt dir must sit directly below fold_{int(spec['fold'])}: {attempt_path}"
        )
    attempt_path.mkdir(parents=True, exist_ok=False)

    commit = _git_commit(root)
    if commit != str(spec["merged_sha"]):
        raise PosthocError(
            f"repository HEAD {commit} is not the deployed SHA {spec['merged_sha']}"
        )
    source_records = _source_manifest_records(root)
    source_sha = deployed_source_sha256(source_records)
    attempt_id = new_attempt_id(str(spec["run_name"]), commit)
    run_config_doc = build_run_config(spec, attempt_id, source_sha)
    write_json_atomic(attempt_path / "run_config.yaml", run_config_doc, indent=2)
    metadata_cache = _verify_cache_identity(spec)
    parent_spec = spec.get("parent") or {}
    # The sidecar schema requires the parent block to be absent unless it
    # carries a valid (or legacy) parent attempt id.
    metadata_parent: dict[str, Any] | None = None
    if parent_spec.get("parent_attempt_id"):
        metadata_parent = {
            "parent_attempt_id": str(parent_spec["parent_attempt_id"]),
            "parent_checkpoint_role": "best_model",
            "parent_checkpoint_path": parent_spec.get("parent_checkpoint_path"),
            "adapter_config_sha256": parent_spec.get("adapter_config_sha256"),
            "adapter_sha256": parent_spec.get("adapter_sha256"),
        }
    metadata = {
        "schema_version": "audiollm.metadata.v1",
        "group_id": spec["group_id"],
        "logical_run_name": sanitize_logical_run_name(str(spec["run_name"])),
        "attempt_id": attempt_id,
        "fold": int(spec["fold"]),
        "seed": int(spec["seed"]),
        "created_at_utc": format_utc_timestamp(utc_now()),
        "source": {
            "git_commit": commit,
            "git_branch": spec["branch"],
            "git_dirty": False,
            "deployed_source_sha256": source_sha,
        },
        "research": {
            "github_issue": spec.get("github_issue"),
            "github_pr": spec.get("pr"),
        },
        "hashes": {
            "resolved_config_sha256": canonical_sha256(run_config_doc),
            "manifest_sha256": metadata_cache.get("manifest_sha256"),
            "split_sha256": metadata_cache.get("split_metadata_sha256"),
        },
        "paths": {
            "run_config": "run_config.yaml",
            "best_model": None,
            "local_evidence_root": None,
        },
        "parent": metadata_parent,
        "wandb": {
            "project": "audiollm-depression",
            "entity": None,
            "run_id": f"{attempt_id}-fold{int(spec['fold'])}",
            "url": None,
            "sync_status": "NOT_EXPORTED",
        },
    }
    write_json_atomic(attempt_path / METADATA_FILE, metadata, indent=2)
    status = StatusRecord(attempt_id=attempt_id, fold=int(spec["fold"]), state="PLANNED")
    write_status(attempt_path / STATUS_FILE, status)
    (attempt_path / JOBS_FILE).write_text("", encoding="utf-8")
    write_json_atomic(
        attempt_path / ARTIFACTS_FILE,
        {
            "schema_version": "audiollm.artifacts.v1",
            "attempt_id": attempt_id,
            "fold": int(spec["fold"]),
            "artifacts": [],
        },
    )
    write_json_atomic(
        attempt_path / EVALUATIONS_FILE,
        {
            "schema_version": "audiollm.evaluations.v1",
            "attempt_id": attempt_id,
            "fold": int(spec["fold"]),
            "evaluations": [],
        },
    )
    write_json_atomic(
        attempt_path / "source_manifest.json", {"files": source_records}, indent=2
    )
    return {
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_path),
        "state": "PLANNED",
        "task_spec": str(task_spec),
    }


def _required_artifact_files(family: str) -> tuple[tuple[str, str, str], ...]:
    return MERGED_ARTIFACT_FILES if family == "merged" else STANDALONE_ARTIFACT_FILES


def _artifact_records(attempt_dir: Path, family: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, artifact_type, role in _required_artifact_files(family):
        full = attempt_dir / name
        if not full.is_file():
            raise PosthocError(f"required logreg artifact missing: {full}")
        records.append(
            {
                "artifact_type": artifact_type,
                "role": role,
                "path": name,
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
            }
        )
    return records


def _positive_f1(metrics: dict[str, Any]) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def strict_subject_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from src.metrics import classification_metrics

    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    metrics = classification_metrics(y_true, y_pred)
    metrics["positive_f1"] = _positive_f1(metrics)
    metrics["subject_count"] = len({str(row["subject_id"]) for row in rows})
    return metrics


def _dataset_rows(
    rows: list[dict[str, Any]], dataset: str | None
) -> list[dict[str, Any]]:
    if dataset is None:
        return rows
    selected = [row for row in rows if str(row.get("dataset")) == str(dataset)]
    if not selected:
        raise PosthocError(f"predictions contain no rows for dataset {dataset!r}")
    return selected


def _evaluation_records(attempt_dir: Path, family: str) -> list[dict[str, Any]]:
    sidecars = read_modern_sidecars(attempt_dir)
    run_config = read_json(attempt_dir / "run_config.yaml")
    config = run_config.get("config") or {}
    evaluation = config.get("evaluation") or {}
    classifier = config.get("classifier") or {}
    subject_rows = read_jsonl(attempt_dir / "predictions_subject_level.jsonl")
    if family == "merged":
        metrics_by_dataset = read_json(attempt_dir / "metrics_by_dataset.json")
        scopes: list[tuple[str | None, dict[str, Any]]] = sorted(metrics_by_dataset.items())
        metrics_artifact = "metrics_by_dataset.json"
    else:
        scopes = [(None, read_json(attempt_dir / "metrics.json"))]
        metrics_artifact = "metrics.json"
    metrics_sha = sha256_file(attempt_dir / metrics_artifact)
    records: list[dict[str, Any]] = []
    for dataset_scope, metrics in scopes:
        dataset = (
            str(dataset_scope)
            if dataset_scope is not None
            else str(evaluation.get("dataset") or config.get("dataset"))
        )
        scope_rows = _dataset_rows(subject_rows, dataset if family == "merged" else None)
        support = len(scope_rows)
        eval_id = evaluation_id(
            attempt_id=sidecars.attempt_id,
            fold=sidecars.fold,
            dataset=dataset,
            split_name=str(evaluation.get("split_name") or ""),
            split_protocol=str(evaluation.get("split_protocol") or ""),
            checkpoint_role="best_model",
            checkpoint_path=str(evaluation.get("checkpoint_path") or ""),
            backend=str(classifier.get("prediction_backend") or ""),
            evaluation_view=str(evaluation.get("evaluation_view") or ""),
            aggregation=str(evaluation.get("aggregation") or ""),
            metric_namespace=str(evaluation.get("metric_namespace") or ""),
            metrics_artifact_sha256=metrics_sha,
        )
        records.append(
            {
                "evaluation_id": eval_id,
                "dataset": dataset,
                "split_name": evaluation.get("split_name"),
                "split_protocol": evaluation.get("split_protocol"),
                "checkpoint_role": "best_model",
                "checkpoint_path": evaluation.get("checkpoint_path"),
                "backend": classifier.get("prediction_backend"),
                "evaluation_view": evaluation.get("evaluation_view"),
                "aggregation": evaluation.get("aggregation"),
                "metric_namespace": evaluation.get("metric_namespace"),
                "metrics_artifact_path": metrics_artifact,
                "predictions_artifact_path": "predictions_subject_level.csv",
                "metrics": [
                    {"name": name, "value": metrics.get(name), "support": support}
                    for name in EVALUATION_METRIC_NAMES
                    if name in {"macro_f1", "positive_f1"}
                ],
                "locally_verified": False,
                "reportable": False,
                "warnings": [],
            }
        )
    return records


def materialize_mn5_evidence(
    attempt_dir: str | Path,
    *,
    transition_to_completed: bool = True,
) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = read_modern_sidecars(attempt_path)
    run_config = read_json(attempt_path / "run_config.yaml")
    family = str((run_config.get("config") or {}).get("family") or "")
    records = _artifact_records(attempt_path, family)

    existing = read_json(attempt_path / ARTIFACTS_FILE)
    known = {record["path"] for record in existing.get("artifacts", [])}
    additions: list[dict[str, Any]] = []
    for record in records:
        if record["path"] in known:
            continue
        additions.append(
            {
                "artifact_id": artifact_id(
                    attempt_id=sidecars.attempt_id,
                    fold=sidecars.fold,
                    role=record["role"],
                    relative_path=record["path"],
                    artifact_sha256=record["sha256"],
                ),
                "artifact_type": record["artifact_type"],
                "role": record["role"],
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "exists_on_mn5": True,
                "exists_locally": False,
                "locally_verified": False,
            }
        )
    if additions:
        existing.setdefault("artifacts", []).extend(additions)
        write_json_atomic(attempt_path / ARTIFACTS_FILE, existing)

    new_evaluations = _evaluation_records(attempt_path, family)
    evaluations_record = read_json(attempt_path / EVALUATIONS_FILE)
    prior_by_id = {
        entry["evaluation_id"]: entry for entry in evaluations_record.get("evaluations", [])
    }
    for entry in new_evaluations:
        prior = prior_by_id.get(entry["evaluation_id"])
        if prior is not None:
            if prior != entry:
                raise PosthocError(
                    "refusing to change evaluation record content: "
                    f"{entry['evaluation_id']}"
                )
            continue
        evaluations_record.setdefault("evaluations", []).append(entry)
    write_json_atomic(attempt_path / EVALUATIONS_FILE, evaluations_record)

    result = {
        "status": "materialized",
        "attempt_id": sidecars.attempt_id,
        "artifacts": len(records),
        "evaluations": len(new_evaluations),
    }
    if transition_to_completed:
        record = StatusRecord.from_dict(read_status(attempt_path / STATUS_FILE))
        if record.state == "RUNNING":
            record.transition("COMPLETED_ON_MN5", reason="LogReg head completed")
            write_status(attempt_path / STATUS_FILE, record)
        result["state"] = record.state
    return result


def verify_local(attempt_dir: str | Path) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = read_modern_sidecars(attempt_path)
    if sidecars.state not in {"SYNCED_LOCALLY", "COMPLETED_ON_MN5", "LOCALLY_VALIDATED"}:
        raise PosthocError(
            f"verify-local requires SYNCED_LOCALLY or later state, got {sidecars.state}"
        )

    artifact_record = read_json(attempt_path / ARTIFACTS_FILE)
    for artifact in artifact_record.get("artifacts", []):
        if artifact.get("sha256") is None:
            continue
        full = attempt_path / artifact["path"]
        if not full.is_file():
            raise PosthocError(f"artifact missing locally: {artifact['path']}")
        if sha256_file(full) != artifact["sha256"]:
            raise PosthocError(f"artifact hash mismatch: {artifact['path']}")

    run_config = read_json(attempt_path / "run_config.yaml")
    config = run_config.get("config") or {}
    family = str(config.get("family") or "")
    subject_rows = read_jsonl(attempt_path / "predictions_subject_level.jsonl")
    recomputed_by_dataset: dict[str | None, dict[str, float]] = {}
    scopes: list[str | None] = (
        sorted({str(row["dataset"]) for row in subject_rows})
        if family == "merged"
        else [None]
    )
    for dataset in scopes:
        metrics = strict_subject_metrics(_dataset_rows(subject_rows, dataset))
        recomputed_by_dataset[dataset] = {
            "macro_f1": float(metrics["macro_f1"]),
            "positive_f1": float(metrics["positive_f1"]),
        }

    evaluations_record = read_json(attempt_path / EVALUATIONS_FILE)
    for entry in evaluations_record.get("evaluations", []):
        expected_map = recomputed_by_dataset[
            str(entry["dataset"]) if family == "merged" else None
        ]
        for metric in entry.get("metrics", []):
            name = str(metric.get("name"))
            if name not in {"macro_f1", "positive_f1"}:
                continue
            expected = expected_map[name]
            actual = metric.get("value")
            if actual is None or abs(float(expected) - float(actual)) > 1e-9:
                raise PosthocError(
                    f"recomputed {entry['dataset']} {name} ({expected}) does not match "
                    f"evaluation record ({actual})"
                )
        entry["locally_verified"] = True
        entry["reportable"] = True
        entry["warnings"] = []
    write_json_atomic(attempt_path / EVALUATIONS_FILE, evaluations_record)

    artifact_record = read_json(attempt_path / ARTIFACTS_FILE)
    for artifact in artifact_record.get("artifacts", []):
        if artifact.get("sha256") is not None:
            artifact["exists_locally"] = True
            artifact["locally_verified"] = True
    write_json_atomic(attempt_path / ARTIFACTS_FILE, artifact_record)

    record = StatusRecord.from_dict(read_status(attempt_path / STATUS_FILE))
    if record.state in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED"}:
        if record.state == "COMPLETED_ON_MN5":
            record.transition("SYNCED_LOCALLY", reason="compact evidence synced locally")
        if record.state == "SYNCED_LOCALLY":
            record.transition("LOCALLY_VALIDATED", reason="local recomputation matched")
        if record.state == "LOCALLY_VALIDATED":
            record.transition("REPORTABLE", reason="local verification passed")
        write_status(attempt_path / STATUS_FILE, record)
    return {
        "attempt_id": sidecars.attempt_id,
        "state": record.state,
        "subject_rows": len(subject_rows),
        "evaluations": len(evaluations_record.get("evaluations", [])),
    }
