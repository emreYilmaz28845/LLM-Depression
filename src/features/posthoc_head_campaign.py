"""General post-hoc head-attempt workflow over arbitrary qualified caches.

The DAIC-specific ``gemma4_hidden_campaign.py`` remains the historical
DAIC-contract implementation. This module implements the runbook Section 9.3
generalization: one modern post-hoc head attempt per Optuna study, created
around any qualified cache identity (Qwen or Gemma, native / English /
official-development / merged), with the exact parent/checkpoint/cache
relationship, modern sidecars, idempotent evaluation materialization, and
local recomputation before reportability.

Never writes the SQLite registry; MN5 jobs write sidecars only.
"""

from __future__ import annotations

import json
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
)
from src.features.hidden_classifier_policy import (
    cache_identity,
    canonical_sha256 as policy_canonical_sha256,
    classifier_aggregation_policy,
)

TASK_SCHEMA_VERSION = "audiollm.posthoc_head_task.v1"
METHOD = "xgb_optuna_posthoc"
HIDDEN_LAYER = "final"
POOLING = "last_valid_prompt_token"
AGGREGATION_POLICY = "mean_depressed_probability_threshold_0_5"
EVALUATION_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "positive_f1",
    "negative_f1",
    "macro_f1",
)

STUDY_ARTIFACT_FILES = (
    "study.sqlite3",
    "trials.csv",
    "inner_subject_assignments.json",
    "inner_fold_metrics.json",
    "inner_oof_metrics.json",
    "inner_sampling_audits.json",
    "inner_weight_audits.json",
    "best_params.json",
    "final_fit_sampling_audit.json",
    "final_fit_weight_audit.json",
    "pipeline.joblib",
    "classifier_metadata.json",
    "metrics.json",
    "predictions_sample_level.jsonl",
    "predictions_sample_level.csv",
    "predictions_subject_level.jsonl",
    "predictions_subject_level.csv",
)


MERGED_STUDY_ARTIFACT_FILES = (
    "study_config.json",
    "study.sqlite3",
    "trials.csv",
    "inner_subject_assignments.json",
    "inner_fold_metrics.json",
    "best_params.json",
    "pipeline.joblib",
    "classifier_metadata.json",
    "metrics.json",
    "predictions_subject_level.jsonl",
    "predictions_subject_level.csv",
)


def _attempt_family(attempt_dir: Path) -> str:
    run_config = read_json(attempt_dir / "run_config.yaml")
    return str((run_config.get("config") or {}).get("family") or "")


class PosthocError(ValueError):
    pass


def _git_commit(repo_root: str | Path) -> str:
    provenance = Path(repo_root) / ".provenance" / "git_commit.txt"
    if provenance.exists():
        return provenance.read_text(encoding="utf-8").strip()
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _source_manifest_records(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    manifest_path = root / ".provenance" / "source_manifest.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        return list(payload.get("files", payload if isinstance(payload, list) else []))
    import subprocess

    try:
        listing = subprocess.check_output(
            ["git", "ls-files"], cwd=str(root), text=True
        ).splitlines()
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for relative in sorted(listing):
        full = root / relative
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


def deployed_source_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_clean_production_source(repo_root: str | Path) -> None:
    root = Path(repo_root)
    provenance = root / ".provenance"
    if not provenance.is_dir():
        # Local development runs outside .provenance: allow when git is clean.
        import subprocess

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            raise PosthocError("cannot determine repository cleanliness")
        dirty = [line for line in status.stdout.splitlines() if line.strip()]
        if dirty:
            raise PosthocError(
                f"refusing to create an attempt from a dirty production source: "
                f"{dirty[:5]}"
            )


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
        "objective",
        "target_trials",
        "group_id",
        "run_name",
        "branch",
        "merged_sha",
    ):
        if key not in spec:
            raise PosthocError(f"task spec missing required key {key!r}")
    from src.features import optuna100_policy as policy

    if int(spec["target_trials"]) != policy.PRODUCTION_TARGET_TRIALS:
        raise PosthocError(
            f"post-hoc task target_trials must be "
            f"{policy.PRODUCTION_TARGET_TRIALS}, got {spec['target_trials']}"
        )
    return spec


def _cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir)
    metadata_path = path / "extraction_metadata.json"
    if metadata_path.is_file():
        return read_json(metadata_path)
    # Symmetric-merged feature sets carry feature_metadata.json instead.
    merged_metadata_path = path / "feature_metadata.json"
    if merged_metadata_path.is_file():
        return read_json(merged_metadata_path)
    raise PosthocError(
        f"cache has neither extraction_metadata.json nor feature_metadata.json: {path}"
    )


def _cache_identity_for_spec(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    """Cache identity honoring the merged feature layout (no final_eval
    partition; the postprocess writes outer_train/outer_holdout only)."""
    if str(spec.get("family", "")) == "merged":
        return {
            "feature_metadata.json": file_identity(cache_dir / "feature_metadata.json"),
            "outer_train.npz": file_identity(cache_dir / "outer_train.npz"),
            "outer_train_rows.jsonl": file_identity(cache_dir / "outer_train_rows.jsonl"),
            "outer_holdout.npz": file_identity(cache_dir / "outer_holdout.npz"),
            "outer_holdout_rows.jsonl": file_identity(cache_dir / "outer_holdout_rows.jsonl"),
        }
    return cache_identity(cache_dir)


def _verify_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    """Verify the cache identity against the task spec; never overwrite."""
    metadata = _cache_metadata(spec["cache_dir"])
    merged = str(spec.get("family", "")) == "merged"
    if merged:
        if str(metadata.get("modality", "")) != str(spec["modality"]):
            raise PosthocError(
                f"merged features modality {metadata.get('modality')} != task modality {spec['modality']}"
            )
        if int(metadata.get("fold", -1)) != int(spec["fold"]):
            raise PosthocError(
                f"merged features fold {metadata.get('fold')} != task fold {spec['fold']}"
            )
        backend = str(metadata.get("model_backend") or "").strip().lower()
        spec_backend = str(spec.get("backend") or "").strip().lower()
        if spec_backend == "qwen":
            if backend not in {"", "qwen2audio", "qwen_text", "text", "qwen3omni"}:
                raise PosthocError(
                    f"merged features model_backend {backend!r} is not a Qwen backend"
                )
        elif spec_backend and backend != spec_backend:
            raise PosthocError(
                f"merged features model_backend {backend!r} != task backend {spec_backend!r}"
            )
        return metadata
    if str(metadata.get("dataset", "")).lower() != str(spec["dataset"]).lower():
        raise PosthocError(
            f"cache dataset {metadata.get('dataset')} != task dataset {spec['dataset']}"
        )
    if str(metadata.get("input_modality", "")) != str(spec["modality"]):
        raise PosthocError(
            f"cache modality {metadata.get('input_modality')} != task modality {spec['modality']}"
        )
    if int(metadata.get("fold", -1)) != int(spec["fold"]):
        raise PosthocError(
            f"cache fold {metadata.get('fold')} != task fold {spec['fold']}"
        )
    condition = str(metadata.get("condition") or metadata.get("input_modality") or "")
    if condition != str(spec["condition"]):
        raise PosthocError(
            f"cache condition {condition!r} != task condition {spec['condition']!r}"
        )
    backend = str(metadata.get("model_backend") or "").strip().lower()
    spec_backend = str(spec.get("backend") or "").strip().lower()
    if spec_backend == "qwen":
        qwen_ok = backend in {"", "qwen2audio", "qwen_text", "text", "qwen3omni"}
        if not qwen_ok:
            raise PosthocError(
                f"cache model_backend {backend!r} is not a Qwen backend for task "
                f"backend {spec_backend!r}"
            )
    elif spec_backend and backend != spec_backend:
        raise PosthocError(
            f"cache model_backend {backend!r} != task backend {spec_backend!r}"
        )
    cache_config = metadata.get("cache_config") or {}
    if cache_config.get("subject_selection_sha256"):
        raise PosthocError("refusing a smoke cache in production: cache_config.subject_selection_sha256")
    return metadata


def build_run_config(spec: dict[str, Any], attempt_id: str, source_sha: str) -> dict[str, Any]:
    from src.features import optuna100_policy as policy

    cache_dir = Path(spec["cache_dir"])
    metadata = _cache_metadata(cache_dir)
    cache_config = metadata.get("cache_config") or {}
    parent = spec.get("parent") or {}
    evaluation = spec.get("evaluation_qualifiers") or {}
    protocol_block = policy.protocol_block(
        dataset=spec["dataset"],
        condition=spec["condition"],
        modality=spec["modality"],
        fold=int(spec["fold"]),
        seed=int(spec["seed"]),
        objective=spec["objective"],
        merged=bool(spec.get("merged", False)),
        model_backend=metadata.get("model_backend"),
    )
    scientific = {
        "dataset": spec["dataset"],
        "modality": spec["modality"],
        "method": METHOD,
        "fold": int(spec["fold"]),
        "seed": int(spec["seed"]),
        "parent": {
            "parent_attempt_id": parent.get("parent_attempt_id"),
            "parent_fold_dir": parent.get("parent_fold_dir"),
            "parent_checkpoint_role": "best_model",
            "parent_checkpoint_path": parent.get("parent_checkpoint_path"),
            "adapter_config_sha256": parent.get("adapter_config_sha256"),
            "adapter_sha256": parent.get("adapter_sha256"),
        },
        "hashes": {
            "manifest_sha256": metadata.get("manifest_sha256"),
            "split_sha256": metadata.get("split_metadata_sha256"),
        },
        "base_model": {
            "id": (
                "google/gemma-4-12B-it"
                if str(metadata.get("model_backend", "")).lower() == "gemma4"
                else str(metadata.get("model_name_or_path") or cache_config.get("model_name_or_path") or "")
            ),
            "revision": cache_config.get("model_revision"),
            "path": str(cache_config.get("checkpoint_dir") or ""),
        },
        "hidden_state": {
            "layer": HIDDEN_LAYER,
            "pooling": POOLING,
            "dimension": int(metadata.get("hidden_dimension") or 0) or None,
            "dtype": "float32",
            "cache_schema": cache_config.get("schema_version"),
        },
        "cache": {
            "cache_dir": str(cache_dir),
            "cache_identity": _cache_identity_for_spec(spec, cache_dir),
            "cache_identity_sha256": policy_canonical_sha256(_cache_identity_for_spec(spec, cache_dir)),
            "extraction_metadata_sha256": sha256_file(
                cache_dir / ("feature_metadata.json" if str(spec.get("family", "")) == "merged" else "extraction_metadata.json")
            ),
            "parent_attempt_id": cache_config.get("parent_attempt_id"),
            "adapter_config_sha256": cache_config.get("adapter_config_sha256"),
            "adapter_sha256": cache_config.get("adapter_sha256"),
            "saved_run_config_sha256": cache_config.get("saved_run_config_sha256"),
            "saved_split_sha256": cache_config.get("saved_split_sha256"),
            "manifest_sha256": cache_config.get("manifest_sha256"),
            "split_metadata_sha256": cache_config.get("split_metadata_sha256"),
        },
        "classifier": {
            "experiment_id": spec["experiment_id"],
            "classifier_family": "xgb_optuna_raw",
            "variants": [spec["experiment_id"]],
            "objective": spec["objective"],
            "target_trials": int(spec["target_trials"]),
            "seed": int(spec["seed"]),
            "sampling_mode": policy.SAMPLING_MODE,
            "threshold": policy.THRESHOLD,
            "aggregation_policy": classifier_aggregation_policy(metadata),
            "weight_policy": "response_normalized_subject_equalizing",
            "prediction_backend": protocol_block["prediction_backend"],
            "model_backend": metadata.get("model_backend"),
            "protocol": protocol_block,
        },
        "evaluation": {
            "dataset": evaluation.get("dataset", spec["dataset"]),
            "split_name": evaluation.get("split_name"),
            "split_protocol": evaluation.get("split_protocol"),
            "checkpoint_role": "best_model",
            "checkpoint_path": str(parent.get("parent_checkpoint_path") or ""),
            "evaluation_view": evaluation.get("evaluation_view", "harmonized_all_windows_full_coverage"),
            "aggregation": evaluation.get("aggregation", "subject_level"),
            "metric_namespace": evaluation.get("metric_namespace", "headline/binary_strict"),
            "support": evaluation.get("support"),
            "metric_names": list(EVALUATION_METRIC_NAMES),
        },
        "family": spec["family"],
        "implementation": {
            "branch": spec["branch"],
            "merged_sha": spec["merged_sha"],
            "pr": spec.get("pr"),
            "deployed_source_sha256": source_sha,
        },
    }
    return {
        "schema_version": "audiollm.posthoc_head_run.v1",
        "config": scientific,
        "manifest_sha256": metadata.get("manifest_sha256"),
        "split_metadata_hash": metadata.get("split_metadata_sha256"),
        "tracking": {
            "schema_version": "audiollm.tracking.v1",
            "group_id": spec["group_id"],
            "logical_run_name": sanitize_logical_run_name(spec["run_name"]),
            "attempt_id": attempt_id,
            "fold": int(spec["fold"]),
        },
    }


def create_attempt(*, repo_root: str | Path, attempt_dir: str | Path, task_spec: str | Path) -> dict[str, Any]:
    spec = load_task_spec(task_spec)
    _require_clean_production_source(repo_root)
    metadata_cache = _verify_cache_identity(spec)
    attempt_path = Path(attempt_dir)
    if attempt_path.name != spec["experiment_id"]:
        raise PosthocError(
            f"attempt dir must end with {spec['experiment_id']!r}: {attempt_path}"
        )
    if attempt_path.parent.name != f"fold_{int(spec['fold'])}":
        raise PosthocError(
            f"attempt dir must sit directly below fold_{int(spec['fold'])}: {attempt_path}"
        )
    attempt_path.mkdir(parents=True, exist_ok=False)

    parent_metadata_path = Path(spec.get("parent", {}).get("parent_fold_dir") or "") / METADATA_FILE
    parent_attempt_id = spec.get("parent", {}).get("parent_attempt_id")
    if parent_metadata_path.is_file():
        actual = str(read_json(parent_metadata_path).get("attempt_id") or "")
        if parent_attempt_id and actual and actual != parent_attempt_id:
            raise PosthocError(
                f"parent metadata attempt_id {actual} does not match task parent "
                f"{parent_attempt_id}"
            )

    source_records = _source_manifest_records(repo_root)
    source_sha = deployed_source_sha256(source_records)
    actual_commit = _git_commit(repo_root)
    merged_sha = str(spec["merged_sha"])
    if actual_commit != merged_sha:
        raise PosthocError(
            f"repository HEAD {actual_commit} is not the merged SHA {merged_sha}"
        )
    attempt_id = new_attempt_id(spec["run_name"], merged_sha)
    try:
        run_config_doc = build_run_config(spec, attempt_id, source_sha)
        write_json_atomic(attempt_path / "run_config.yaml", run_config_doc, indent=2)

        metadata = {
            "schema_version": "audiollm.metadata.v1",
            "group_id": spec["group_id"],
            "logical_run_name": sanitize_logical_run_name(spec["run_name"]),
            "attempt_id": attempt_id,
            "fold": int(spec["fold"]),
            "seed": int(spec["seed"]),
            "created_at_utc": format_utc_timestamp(utc_now()),
            "source": {
                "git_commit": merged_sha,
                "git_branch": spec["branch"],
                "git_dirty": False,
                "deployed_source_sha256": source_sha,
            },
            "research": {"github_issue": spec.get("github_issue"), "github_pr": spec.get("pr")},
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
            "parent": {
                "parent_attempt_id": spec.get("parent", {}).get("parent_attempt_id"),
                "parent_checkpoint_role": "best_model",
                "parent_checkpoint_path": spec.get("parent", {}).get("parent_checkpoint_path"),
                "adapter_config_sha256": spec.get("parent", {}).get("adapter_config_sha256"),
                "adapter_sha256": spec.get("parent", {}).get("adapter_sha256"),
            },
            "wandb": {
                "project": "audiollm-depression",
                "entity": None,
                "run_id": f"{attempt_id}-fold{int(spec['fold'])}",
                "url": None,
                "sync_status": "NOT_EXPORTED",
            },
        }
        write_json_atomic(attempt_path / METADATA_FILE, metadata, indent=2)

        status = StatusRecord(
            attempt_id=attempt_id,
            fold=int(spec["fold"]),
            state="PLANNED",
        )
        write_status(attempt_path / STATUS_FILE, status)

        (attempt_path / JOBS_FILE).write_text("", encoding="utf-8")
        write_json_atomic(attempt_path / ARTIFACTS_FILE, {"schema_version": "audiollm.artifacts.v1", "attempt_id": attempt_id, "fold": int(spec["fold"]), "artifacts": []})
        write_json_atomic(attempt_path / EVALUATIONS_FILE, {"schema_version": "audiollm.evaluations.v1", "attempt_id": attempt_id, "fold": int(spec["fold"]), "evaluations": []})
        write_json_atomic(attempt_path / "source_manifest.json", {"files": source_records}, indent=2)
    except Exception:
        # The destination is exclusively task-owned; a failed creation leaves
        # a partial dir that a retry must refuse (exist_ok=False).
        raise
    return {
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_path),
        "state": "PLANNED",
        "task_spec": str(task_spec),
    }


def _read_lenient_sidecars(attempt_dir: Path) -> Any:
    """Read sidecars for lifecycle bookkeeping before jobs exist.

    ``read_modern_sidecars`` requires a non-empty jobs.jsonl, which does not
    hold between create-attempt and the first SUBMITTED event. The strict
    reader is used by materialize-mn5-evidence and verify-local, where job
    history must exist.
    """
    from src.experiment_tracking.sidecars import ModernSidecars

    metadata = read_json(attempt_dir / METADATA_FILE)
    status = read_json(attempt_dir / STATUS_FILE)
    jobs_path = attempt_dir / JOBS_FILE
    jobs = tuple(read_jsonl(jobs_path)) if jobs_path.is_file() and jobs_path.stat().st_size else ()
    artifacts_path = attempt_dir / ARTIFACTS_FILE
    artifacts = read_json(artifacts_path) if artifacts_path.is_file() else {}
    evaluations_path = attempt_dir / EVALUATIONS_FILE
    evaluations = read_json(evaluations_path) if evaluations_path.is_file() else {}
    return ModernSidecars(
        fold_dir=str(attempt_dir),
        metadata=metadata,
        status=status,
        jobs=jobs,
        artifacts=tuple(artifacts.get("artifacts") or []),
        evaluations=tuple(evaluations.get("evaluations") or []),
        file_sha256={},
    )


def mark_deployed(attempt_dir: str | Path, reason: str | None = None) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = _read_lenient_sidecars(attempt_path)
    record = StatusRecord.from_dict(sidecars.status)
    record.transition("DEPLOYED", reason=reason or "attempt deployed to MN5")
    write_status(attempt_path / STATUS_FILE, record)
    return {"attempt_id": sidecars.attempt_id, "state": record.state}


def record_job(
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
) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = _read_lenient_sidecars(attempt_path)
    event = new_job_event(
        job_key=job_key,
        job_type=job_type,
        event_type=event_type,
        attempt_id=sidecars.attempt_id,
        fold=sidecars.fold,
        slurm_job_id=slurm_job_id,
        status=status,
        reason=reason,
        dependency_job_ids=dependency_job_ids or [],
        resubmission_of_job_id=resubmission_of_job_id,
    )
    append_job_event(attempt_path / JOBS_FILE, event)
    return {"attempt_id": sidecars.attempt_id, "event_type": event_type}


def transition(attempt_dir: str | Path, to_state: str, reason: str | None = None) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = _read_lenient_sidecars(attempt_path)
    record = StatusRecord.from_dict(sidecars.status)
    record.transition(to_state, reason=reason)
    write_status(attempt_path / STATUS_FILE, record)
    return {"attempt_id": sidecars.attempt_id, "state": record.state}


def _study_artifacts(attempt_dir: Path) -> list[dict[str, Any]]:
    family = _attempt_family(attempt_dir)
    required_names = MERGED_STUDY_ARTIFACT_FILES if family == "merged" else STUDY_ARTIFACT_FILES
    records: list[dict[str, Any]] = []
    for name, artifact_type, role in (
        ("run_config.yaml", "run_config", "run_config"),
        ("source_manifest.json", "source_manifest", "source_manifest"),
        ("study_config.json", "audit", "study_config"),
        ("study.sqlite3", "audit", "study_sqlite"),
        ("trials.csv", "audit", "trials_csv"),
        ("inner_subject_assignments.json", "audit", "inner_subject_assignments"),
        ("inner_fold_metrics.json", "audit", "inner_fold_metrics"),
        ("inner_oof_metrics.json", "audit", "inner_oof_metrics"),
        ("inner_sampling_audits.json", "audit", "inner_sampling_audits"),
        ("inner_weight_audits.json", "audit", "inner_weight_audits"),
        ("best_params.json", "audit", "best_params"),
        ("final_fit_sampling_audit.json", "audit", "final_fit_sampling_audit"),
        ("final_fit_weight_audit.json", "audit", "final_fit_weight_audit"),
        ("pipeline.joblib", "checkpoint", "classifier_pipeline"),
        ("classifier_metadata.json", "audit", "classifier_metadata"),
        ("metrics.json", "metrics", "classifier_metrics"),
        ("predictions_sample_level.jsonl", "predictions", "predictions_sample_level"),
        ("predictions_sample_level.csv", "predictions", "predictions_sample_level_csv"),
        ("predictions_subject_level.jsonl", "predictions", "predictions_subject_level"),
        ("predictions_subject_level.csv", "predictions", "predictions_subject_level_csv"),
    ):
        full = attempt_dir / name
        if name not in required_names:
            continue
        if not full.is_file():
            raise PosthocError(f"required study artifact missing: {full}")
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


def _evaluation_records(attempt_dir: Path) -> list[dict[str, Any]]:
    sidecars = read_modern_sidecars(attempt_dir)
    run_config = read_json(attempt_dir / "run_config.yaml")
    config = run_config.get("config") or {}
    evaluation = config.get("evaluation") or {}
    classifier = config.get("classifier") or {}
    metrics_doc = read_json(attempt_dir / "metrics.json")
    # Merged studies store per-dataset metrics plus a pooled subject summary.
    if "pooled_subject_metrics" in metrics_doc:
        metrics = metrics_doc["pooled_subject_metrics"]
        merged = True
    else:
        metrics = metrics_doc
        merged = False
    metrics_sha = sha256_file(attempt_dir / "metrics.json")
    support = evaluation.get("support")
    if support is None:
        subject_rows = read_jsonl(attempt_dir / "predictions_subject_level.jsonl")
        support = len({str(row["subject_id"]) for row in subject_rows})
    eval_id = evaluation_id(
        attempt_id=sidecars.attempt_id,
        fold=sidecars.fold,
        dataset=str(evaluation.get("dataset") or config.get("dataset")),
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
    return [
        {
            "evaluation_id": eval_id,
            "dataset": evaluation.get("dataset") or config.get("dataset"),
            "split_name": evaluation.get("split_name"),
            "split_protocol": evaluation.get("split_protocol"),
            "checkpoint_role": "best_model",
            "checkpoint_path": evaluation.get("checkpoint_path"),
            "backend": classifier.get("prediction_backend"),
            "evaluation_view": evaluation.get("evaluation_view"),
            "aggregation": evaluation.get("aggregation"),
            "metric_namespace": evaluation.get("metric_namespace"),
            "metrics_artifact_path": "metrics.json",
            "predictions_artifact_path": "predictions_subject_level.csv",
            "metrics": [
                {"name": name, "value": metrics.get(name), "support": support}
                for name in EVALUATION_METRIC_NAMES
            ],
            "locally_verified": False,
            "reportable": False,
            "warnings": [],
        }
    ]


def materialize_mn5_evidence(
    attempt_dir: str | Path,
    *,
    transition_to_completed: bool = True,
) -> dict[str, Any]:
    attempt_path = Path(attempt_dir)
    sidecars = read_modern_sidecars(attempt_path)
    records = _study_artifacts(attempt_path)

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

    new_evaluations = _evaluation_records(attempt_path)
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
            record.transition("COMPLETED_ON_MN5", reason="Optuna study completed")
            write_status(attempt_path / STATUS_FILE, record)
        result["state"] = record.state
    return result


def _recompute_metrics(attempt_dir: Path) -> dict[str, Any]:
    from src.metrics import classification_metrics

    sample_path = attempt_dir / "predictions_sample_level.jsonl"
    if sample_path.is_file():
        sample_rows = read_jsonl(sample_path)
        from src.aggregate import aggregate_binary_classifier_predictions

        subject_rows, metrics = aggregate_binary_classifier_predictions(sample_rows)
    else:
        # Merged studies store only subject-level predictions; recompute the
        # six headline metrics directly from the subject rows.
        subject_rows = read_jsonl(attempt_dir / "predictions_subject_level.jsonl")
        y_true = [int(row["label"]) for row in subject_rows]
        y_pred = [int(row["prediction"]) for row in subject_rows]
        metrics = classification_metrics(y_true, y_pred)
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    metrics["negative_f1"] = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if precision_neg + recall_neg
        else 0.0
    )
    return {"subject_rows": subject_rows, "metrics": metrics}


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

    recomputed = _recompute_metrics(attempt_path)
    stored = read_json(attempt_path / "metrics.json")
    for name in EVALUATION_METRIC_NAMES:
        expected = recomputed["metrics"].get(name)
        actual = stored.get(name)
        if expected is None or actual is None or abs(float(expected) - float(actual)) > 1e-9:
            raise PosthocError(
                f"recomputed metric {name} ({expected}) does not match metrics.json ({actual})"
            )

    evaluations_record = read_json(attempt_path / EVALUATIONS_FILE)
    for entry in evaluations_record.get("evaluations", []):
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
        "subject_rows": len(recomputed["subject_rows"]),
        "evaluations": len(evaluations_record.get("evaluations", [])),
    }
