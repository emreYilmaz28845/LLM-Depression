from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

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
    read_job_events,
    read_status,
    write_status,
)
from src.experiment_tracking.schemas import validate_metadata
from src.experiment_tracking.sidecars import (
    ARTIFACTS_FILE,
    EVALUATIONS_FILE,
    JOBS_FILE,
    METADATA_FILE,
    STATUS_FILE,
    read_modern_sidecars,
)

FIXED_HEAD_METHOD = "gemma4_hidden_fixed_heads"
GEMMA4_BASE_MODEL_ID = "google/gemma-4-12B-it"
GEMMA4_BASE_MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA4_HIDDEN_DIMENSION = 3840
GEMMA4_CACHE_SCHEMA = "gemma4_hidden_cache.v1"
HIDDEN_LAYER = "final"
POOLING = "last_valid_prompt_token"
FIT_WEIGHT_POLICY = "inverse_chunks_per_subject_rescaled_to_mean_one"
AGGREGATION_POLICY = "mean_depressed_probability_threshold_0_5"
THRESHOLD = 0.5
SEED = 1337
VARIANTS = ("logreg_raw", "xgb_raw")
LOGGREG_PREDICTION_BACKEND = "gemma4_hidden_logreg_raw"
XGB_PREDICTION_BACKEND = "gemma4_hidden_xgb_raw"

MANIFEST_SHA256 = "72e2dd204b915ccba3ebf922f030531fe5678b3ea8c9c52b81b41242fe9dda17"
SPLIT_SHA256 = "441333e0c88845eeacba9ea5355a8920cdd1f70e8cf7a7c15b9547b46da51473"

EXPECTED_ADAPTER_HASHES: dict[str, dict[str, str]] = {
    "audio_text": {
        "adapter_config_sha256": "7dd0c1ebfb3269bea23751384ef4e38276978d0ec0f775fcdd876b8b911ba68d",
        "adapter_sha256": "5544d535af7efd6b2f551de0e75ed6551ca897240287c3aafbd4c154690d6a0a",
    },
    "audio_only": {
        "adapter_config_sha256": "7dd0c1ebfb3269bea23751384ef4e38276978d0ec0f775fcdd876b8b911ba68d",
        "adapter_sha256": "10ce803ce3b8700b955bade952ea4f677ec7142b7fabf0cd9a61b7293e2dd624",
    },
    "text_only": {
        "adapter_config_sha256": "7dd0c1ebfb3269bea23751384ef4e38276978d0ec0f775fcdd876b8b911ba68d",
        "adapter_sha256": "47379e9ba61e83df06a2d518843b723037134bb7ffa6f37bff576a220c461f3d",
    },
}

PARENT_ATTEMPTS: dict[str, str] = {
    "audio_text": "20260812T031624Z-gemma4_daic_audio_text_seed1337-a6749b05-146c8805",
    "audio_only": "20260812T020449Z-gemma4_daic_audio_only_seed1337-cca3f4ae-8789edf2",
    "text_only": "20260812T020449Z-gemma4_daic_text_only_seed1337-cca3f4ae-ed58a7a3",
}

PARENT_RUN_DIRS: dict[str, str] = {
    "audio_text": "output_model/harmonized_v1_gemma4/audio_text/daic/"
    "gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_audio_text_r2/fold_0",
    "audio_only": "output_model/harmonized_v1_gemma4/audio_only/daic/"
    "gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_audio_only/fold_0",
    "text_only": "output_model/harmonized_v1_gemma4/text_only/daic/"
    "gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_text_only/fold_0",
}

EXPECTED_ROW_COUNTS: dict[str, dict[str, int]] = {
    "audio_text": {"fit_rows": 1598, "fit_subjects": 107, "test_rows": 820, "test_subjects": 47},
    "audio_only": {"fit_rows": 1598, "fit_subjects": 107, "test_rows": 820, "test_subjects": 47},
    "text_only": {"fit_rows": 107, "fit_subjects": 107, "test_rows": 47, "test_subjects": 47},
}

EVALUATION_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "positive_f1",
    "negative_f1",
    "macro_f1",
)


class CampaignError(ValueError):
    pass


def _require_clean_production_source(repo_root: str | Path) -> None:
    root = Path(repo_root)
    provenance = root / ".provenance" / "git_commit.txt"
    if provenance.is_file():
        return
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CampaignError(f"cannot inspect git status in {root}")
    if result.stdout.strip():
        raise CampaignError(
            "production source is dirty; deployment and submission require a clean "
            "tree at the merged implementation SHA"
        )


def _git_commit(repo_root: str | Path) -> str:
    root = Path(repo_root)
    provenance = root / ".provenance" / "git_commit.txt"
    if provenance.is_file():
        return provenance.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CampaignError(f"cannot resolve git HEAD in {root}")
    return result.stdout.strip()


def _source_manifest_records(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    records: list[dict[str, Any]] = []
    for relative in sorted(line for line in result.stdout.splitlines() if line.strip()):
        path = root / relative
        if not path.is_file():
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def deployed_source_sha256(records: list[dict[str, Any]]) -> str:
    return canonical_sha256(sorted(records, key=lambda record: record["path"]))


def _read_run_config(fold_dir: str | Path) -> dict[str, Any]:
    path = Path(fold_dir) / "run_config.yaml"
    if not path.is_file():
        raise CampaignError(f"run_config.yaml not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"invalid run_config.yaml: {path}")
    return payload


def _verify_parent_identity(modality: str, parent_fold_dir: str | Path) -> dict[str, Any]:
    expected = EXPECTED_ADAPTER_HASHES[modality]
    parent_dir = Path(parent_fold_dir)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (parent_dir / "best_model" / name).is_file():
            raise CampaignError(
                f"parent best_model missing {name} in {parent_dir / 'best_model'}"
            )
    actual_config = sha256_file(parent_dir / "best_model" / "adapter_config.json")
    actual_adapter = sha256_file(parent_dir / "best_model" / "adapter_model.safetensors")
    if actual_config != expected["adapter_config_sha256"]:
        raise CampaignError(
            f"parent adapter_config.json hash mismatch for {modality}: "
            f"{actual_config} != {expected['adapter_config_sha256']}"
        )
    if actual_adapter != expected["adapter_sha256"]:
        raise CampaignError(
            f"parent adapter_model.safetensors hash mismatch for {modality}: "
            f"{actual_adapter} != {expected['adapter_sha256']}"
        )
    return {"adapter_config_sha256": actual_config, "adapter_sha256": actual_adapter}


def _verify_parent_config_hashes(parent_fold_dir: str | Path) -> dict[str, Any]:
    run_config = _read_run_config(parent_fold_dir)
    manifest_hash = run_config.get("manifest_hash")
    split_hash = run_config.get("split_metadata_hash")
    if manifest_hash != MANIFEST_SHA256 or split_hash != SPLIT_SHA256:
        raise CampaignError(
            "parent run_config hashes do not match the fixed contract: "
            f"manifest={manifest_hash} split={split_hash}"
        )
    return {"manifest_sha256": manifest_hash, "split_sha256": split_hash}


def _evaluation_qualifiers(
    parent_fold_dir: str | Path,
    backend: str,
    metrics_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset": "daic",
        "split_name": "test",
        "split_protocol": "daic_official_train_fit_locked_test_evaluation",
        "checkpoint_role": "best_model",
        "checkpoint_path": str(Path(parent_fold_dir) / "best_model"),
        "backend": backend,
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "subject_level",
        "metric_namespace": "headline/binary_strict",
        "metrics_artifact_sha256": metrics_artifact_sha256,
    }


def _attempt_dir_to_layout(attempt_dir: str | Path) -> Path:
    """The attempt layout is fixed: <root>/<modality>/daic/<run>/fold_0."""
    path = Path(attempt_dir)
    if path.name != "fold_0":
        raise CampaignError(f"attempt dir must end in fold_0: {path}")
    return path


def build_run_config(
    *,
    modality: str,
    run_name: str,
    attempt_id: str,
    parent_fold_dir: str | Path,
    parent_attempt_id: str,
    adapter_hashes: dict[str, str],
    merged_sha: str,
    branch: str,
    pr_number: int | None,
    deployed_source_sha256_value: str,
    group_id: str,
    fold: int = 0,
) -> dict[str, Any]:
    if modality not in EXPECTED_ROW_COUNTS:
        raise CampaignError(f"unsupported modality {modality!r}")
    run_config = _read_run_config(parent_fold_dir)
    base_model_path = run_config.get("resolved_model_name_or_path") or (
        run_config.get("config", {}) or {}
    ).get("model_name_or_path")
    run_config_doc = {
        "schema_version": "audiollm.fixed_head_run.v1",
        "dataset": "daic",
        "modality": modality,
        "method": FIXED_HEAD_METHOD,
        "fold": fold,
        "seed": SEED,
        "parent": {
            "parent_attempt_id": parent_attempt_id,
            "parent_fold_dir": str(parent_fold_dir),
            "parent_checkpoint_role": "best_model",
            "parent_checkpoint_path": str(Path(parent_fold_dir) / "best_model"),
            "adapter_config_sha256": adapter_hashes["adapter_config_sha256"],
            "adapter_sha256": adapter_hashes["adapter_sha256"],
        },
        "hashes": {
            "manifest_sha256": MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256,
            "parent_run_config_sha256": sha256_file(Path(parent_fold_dir) / "run_config.yaml"),
        },
        "base_model": {
            "id": GEMMA4_BASE_MODEL_ID,
            "revision": GEMMA4_BASE_MODEL_REVISION,
            "path": str(base_model_path or ""),
        },
        "hidden_state": {
            "layer": HIDDEN_LAYER,
            "pooling": POOLING,
            "dimension": GEMMA4_HIDDEN_DIMENSION,
            "dtype": "float32",
            "cache_schema": GEMMA4_CACHE_SCHEMA,
        },
        "classifiers": {
            "variants": list(VARIANTS),
            "seed": SEED,
            "sampling_mode": "legacy",
            "weight_policy": FIT_WEIGHT_POLICY,
            "aggregation_policy": AGGREGATION_POLICY,
            "threshold": THRESHOLD,
            "library_versions": {"scikit_learn": "1.7.0", "xgboost": "2.1.4"},
            "logreg": {
                "backend": LOGGREG_PREDICTION_BACKEND,
                "params": {
                    "scaler": "StandardScaler",
                    "class_weight": "balanced",
                    "C": 1.0,
                    "solver": "liblinear",
                    "max_iter": 5000,
                    "seed": SEED,
                },
            },
            "xgb": {
                "backend": XGB_PREDICTION_BACKEND,
                "params": {
                    "objective": "binary:logistic",
                    "n_estimators": 300,
                    "learning_rate": 0.03,
                    "max_depth": 2,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                    "colsample_bytree": 0.25,
                    "reg_alpha": 1.0,
                    "reg_lambda": 10.0,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "seed": SEED,
                },
            },
        },
        "evaluation": {
            "dataset": "daic",
            "split_name": "test",
            "split_protocol": "daic_official_train_fit_locked_test_evaluation",
            "checkpoint_role": "best_model",
            "evaluation_view": "harmonized_all_windows_full_coverage",
            "aggregation": "subject_level",
            "metric_namespace": "headline/binary_strict",
            "support": 47,
            "metric_names": list(EVALUATION_METRIC_NAMES),
        },
        "expected_counts": EXPECTED_ROW_COUNTS[modality],
        "implementation": {
            "branch": branch,
            "merged_sha": merged_sha,
            "pr": pr_number,
            "deployed_source_sha256": deployed_source_sha256_value,
        },
        "tracking": {
            "schema_version": "audiollm.tracking.v1",
            "group_id": group_id,
            "logical_run_name": sanitize_logical_run_name(run_name),
            "attempt_id": attempt_id,
            "fold": fold,
        },
    }
    return run_config_doc


def create_attempt(
    *,
    repo_root: str | Path,
    attempt_dir: str | Path,
    modality: str,
    run_name: str,
    group_id: str,
    parent_fold_dir: str | Path,
    parent_attempt_id: str,
    merged_sha: str,
    branch: str,
    pr_number: int | None,
    fold: int = 0,
    supersedes_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Create a new post-hoc fixed-head attempt destination.

    Refuses a dirty production source and any parent/checkpoint/hash/split
    mismatch. Writes run_config.yaml, metadata.json, status.json, jobs.jsonl,
    artifacts.json, evaluations.json, and source_manifest.json.
    ``supersedes_attempt_id`` links a retry attempt to a failed/cancelled one.
    """
    if modality not in EXPECTED_ROW_COUNTS:
        raise CampaignError(f"unsupported modality {modality!r}")
    _require_clean_production_source(repo_root)
    adapter_hashes = _verify_parent_identity(modality, parent_fold_dir)
    _verify_parent_config_hashes(parent_fold_dir)
    attempt_path = Path(attempt_dir)
    attempt_path.mkdir(parents=True, exist_ok=False)

    parent_metadata_path = Path(parent_fold_dir) / METADATA_FILE
    parent_metadata = (
        read_json(parent_metadata_path) if parent_metadata_path.is_file() else {}
    )
    actual_parent_attempt = str(parent_metadata.get("attempt_id") or "")
    if actual_parent_attempt and actual_parent_attempt != parent_attempt_id:
        raise CampaignError(
            f"parent metadata attempt_id {actual_parent_attempt} does not match "
            f"the requested parent {parent_attempt_id}"
        )

    source_records = _source_manifest_records(repo_root)
    source_sha = deployed_source_sha256(source_records)
    actual_commit = _git_commit(repo_root)
    if actual_commit != merged_sha:
        raise CampaignError(
            f"repository HEAD {actual_commit} is not the merged implementation "
            f"SHA {merged_sha}"
        )
    attempt_id = new_attempt_id(run_name, merged_sha)
    try:
        run_config_doc = build_run_config(
            modality=modality,
            run_name=run_name,
            attempt_id=attempt_id,
            parent_fold_dir=parent_fold_dir,
            parent_attempt_id=parent_attempt_id,
            adapter_hashes=adapter_hashes,
            merged_sha=merged_sha,
            branch=branch,
            pr_number=pr_number,
            deployed_source_sha256_value=source_sha,
            group_id=group_id,
            fold=fold,
        )
        write_json_atomic(attempt_path / "run_config.yaml", run_config_doc, indent=2)

        metadata = {
            "schema_version": "audiollm.metadata.v1",
            "group_id": group_id,
            "logical_run_name": sanitize_logical_run_name(run_name),
            "attempt_id": attempt_id,
            "fold": fold,
            "seed": SEED,
            "created_at_utc": format_utc_timestamp(utc_now()),
            "source": {
                "git_commit": merged_sha,
                "git_branch": branch,
                "git_dirty": False,
                "deployed_source_sha256": source_sha,
            },
            "research": {"github_issue": None, "github_pr": pr_number},
            "hashes": {
                "resolved_config_sha256": canonical_sha256(run_config_doc),
                "manifest_sha256": MANIFEST_SHA256,
                "split_sha256": SPLIT_SHA256,
            },
            "paths": {
                "run_config": "run_config.yaml",
                "best_model": None,
                "local_evidence_root": None,
            },
            "parent": {
                "parent_attempt_id": parent_attempt_id,
                "parent_checkpoint_role": "best_model",
                "parent_checkpoint_path": str(Path(parent_fold_dir) / "best_model"),
                "adapter_config_sha256": adapter_hashes["adapter_config_sha256"],
                "adapter_sha256": adapter_hashes["adapter_sha256"],
            },
            "wandb": {
                "project": "audiollm-depression",
                "entity": None,
                "run_id": f"{attempt_id}-fold{fold}",
                "url": None,
                "sync_status": "NOT_EXPORTED",
            },
        }
        if supersedes_attempt_id is not None:
            metadata["supersedes_attempt_id"] = supersedes_attempt_id
        ok, errors = validate_metadata(metadata)
        if not ok:
            raise CampaignError("invalid metadata: " + "; ".join(errors))
        write_json_atomic(attempt_path / METADATA_FILE, metadata)

        write_status(
            attempt_path / STATUS_FILE,
            StatusRecord(attempt_id=attempt_id, fold=fold, state="PLANNED"),
        )
        (attempt_path / JOBS_FILE).write_text("", encoding="utf-8")
        write_json_atomic(
            attempt_path / ARTIFACTS_FILE,
            {
                "schema_version": "audiollm.artifacts.v1",
                "attempt_id": attempt_id,
                "fold": fold,
                "artifacts": [],
            },
        )
        write_json_atomic(
            attempt_path / EVALUATIONS_FILE,
            {
                "schema_version": "audiollm.evaluations.v1",
                "attempt_id": attempt_id,
                "fold": fold,
                "evaluations": [],
            },
        )
        write_json_atomic(
            attempt_path / "source_manifest.json",
            {
                "schema_version": "audiollm.source_manifest.v1",
                "git_commit": merged_sha,
                "git_branch": branch,
                "file_count": len(source_records),
                "deployed_source_sha256": source_sha,
                "files": source_records,
            },
        )
    except Exception:
        import shutil

        shutil.rmtree(attempt_path, ignore_errors=True)
        raise
    return {
        "status": "created",
        "attempt_dir": str(attempt_path),
        "attempt_id": attempt_id,
        "run_name": run_name,
        "group_id": group_id,
        "deployed_source_sha256": source_sha,
    }


def _read_sidecars(attempt_dir: str | Path):
    """Read sidecars tolerating an empty jobs file before any job is recorded.

    The import-time reader refuses an empty jobs.jsonl; a PLANNED attempt
    legitimately has no job events yet, so the campaign helper validates the
    other sidecars itself and only requires jobs when a job is being recorded.
    """
    attempt_path = Path(attempt_dir)
    metadata = read_json(attempt_path / METADATA_FILE)
    ok, errors = validate_metadata(metadata)
    if not ok:
        raise CampaignError("invalid metadata: " + "; ".join(errors))
    status = read_status(attempt_path / STATUS_FILE)
    from src.experiment_tracking.sidecars import (
        ARTIFACTS_FILE,
        EVALUATIONS_FILE,
        METADATA_FILE as META,
        STATUS_FILE as ST,
        ModernSidecars,
    )
    from src.experiment_tracking.schemas import validate_record

    artifacts = read_json(attempt_path / ARTIFACTS_FILE)
    evaluations_path = attempt_path / EVALUATIONS_FILE
    evaluations = (
        read_json(evaluations_path) if evaluations_path.is_file() else {"evaluations": []}
    )
    for name, version, record in (
        (ST, "audiollm.status.v1", status),
        (ARTIFACTS_FILE, "audiollm.artifacts.v1", artifacts),
        (EVALUATIONS_FILE, "audiollm.evaluations.v1", evaluations),
    ):
        ok, errors = validate_record(version, record)
        if not ok:
            raise CampaignError(f"invalid {name}: " + "; ".join(errors))
    for label, expected in (
        (META, str(metadata.get("attempt_id"))),
        (ST, str(metadata.get("attempt_id"))),
        (ARTIFACTS_FILE, str(metadata.get("attempt_id"))),
        (EVALUATIONS_FILE, str(metadata.get("attempt_id"))),
    ):
        record = metadata if label == META else (
            status if label == ST else (
                artifacts if label == ARTIFACTS_FILE else evaluations
            )
        )
        if str(record.get("attempt_id")) != expected:
            raise CampaignError(f"{label} attempt_id differs from metadata.json")
    jobs = read_job_events(attempt_path / JOBS_FILE)
    file_sha256 = {
        META: sha256_file(attempt_path / META),
        ST: sha256_file(attempt_path / ST),
        JOBS_FILE: sha256_file(attempt_path / JOBS_FILE),
        ARTIFACTS_FILE: sha256_file(attempt_path / ARTIFACTS_FILE),
    }
    if (attempt_path / EVALUATIONS_FILE).is_file():
        file_sha256[EVALUATIONS_FILE] = sha256_file(attempt_path / EVALUATIONS_FILE)
    return ModernSidecars(
        fold_dir=str(attempt_path),
        metadata=metadata,
        status=status,
        jobs=tuple(jobs),
        artifacts=tuple(artifacts.get("artifacts") or []),
        evaluations=tuple(evaluations.get("evaluations") or []),
        file_sha256=file_sha256,
    )


def mark_deployed(attempt_dir: str | Path, reason: str | None = None) -> dict[str, Any]:
    sidecars = _read_sidecars(attempt_dir)
    status_path = Path(attempt_dir) / STATUS_FILE
    record = StatusRecord.from_dict(read_json(status_path))
    record.transition("DEPLOYED", reason=reason or "source deployed to MN5")
    write_status(status_path, record)
    return {"status": "deployed", "attempt_id": sidecars.attempt_id, "state": record.state}


def transition(
    attempt_dir: str | Path, to_state: str, reason: str | None = None
) -> dict[str, Any]:
    sidecars = _read_sidecars(attempt_dir)
    status_path = Path(attempt_dir) / STATUS_FILE
    record = StatusRecord.from_dict(read_json(status_path))
    record.transition(to_state, reason=reason)
    write_status(status_path, record)
    return {"status": "transitioned", "attempt_id": sidecars.attempt_id, "state": record.state}


def record_job(
    attempt_dir: str | Path,
    *,
    job_key: str,
    job_type: str,
    event_type: str,
    slurm_job_id: str | None = None,
    dependency_job_ids: list[str] | None = None,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    sidecars = _read_sidecars(attempt_dir)
    event = new_job_event(
        job_key=job_key,
        job_type=job_type,
        event_type=event_type,
        attempt_id=sidecars.attempt_id,
        fold=sidecars.fold,
        slurm_job_id=slurm_job_id,
        dependency_job_ids=dependency_job_ids,
        status=status,
        reason=reason,
    )
    append_job_event(Path(attempt_dir) / JOBS_FILE, event)
    return {"status": "recorded", "event_id": event["event_id"]}


def _hidden_feature_artifacts(attempt_dir: Path, fold_dir: Path) -> list[dict[str, Any]]:
    hidden_dir = fold_dir / "hidden_features"
    names = (
        "outer_train.npz",
        "outer_train_rows.jsonl",
        "final_eval.npz",
        "final_eval_rows.jsonl",
        "extraction_metadata.json",
    )
    records = []
    for name in names:
        full = hidden_dir / name
        if not full.is_file():
            raise CampaignError(f"hidden feature artifact missing: {full}")
        records.append(
            {
                "artifact_type": "summary" if name.endswith(".npz") else "audit",
                "role": name.removesuffix(".npz").removesuffix(".jsonl").removesuffix(".json"),
                "path": f"hidden_features/{name}",
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
            }
        )
    return records


def _classifier_artifacts(attempt_dir: Path, fold_dir: Path) -> list[dict[str, Any]]:
    classifiers_dir = fold_dir / "hidden_classifiers"
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_dir = classifiers_dir / variant
        for name, artifact_type, role in (
            ("result_config.json", "audit", "result_config"),
            ("classifier_metadata.json", "audit", "classifier_metadata"),
            ("sampling_audit.json", "audit", "sampling_audit"),
            ("pipeline.joblib", "checkpoint", "pipeline"),
            ("metrics.json", "metrics", "metrics"),
            ("predictions_sample_level.jsonl", "predictions", "predictions_sample"),
            ("predictions_sample_level.csv", "predictions", "predictions_sample_csv"),
            ("predictions_subject_level.jsonl", "predictions", "predictions_subject"),
            ("predictions_subject_level.csv", "predictions", "predictions_subject_csv"),
        ):
            full = variant_dir / name
            if not full.is_file():
                raise CampaignError(f"classifier artifact missing: {full}")
            records.append(
                {
                    "artifact_type": artifact_type,
                    "role": f"{variant}_{role}",
                    "path": f"hidden_classifiers/{variant}/{name}",
                    "sha256": sha256_file(full),
                    "size_bytes": full.stat().st_size,
                }
            )
    for name in ("variant_summary.json", "variant_summary.csv"):
        full = classifiers_dir / name
        if not full.is_file():
            raise CampaignError(f"variant summary missing: {full}")
        records.append(
            {
                "artifact_type": "summary",
                "role": name.removesuffix(".json").removesuffix(".csv"),
                "path": f"hidden_classifiers/{name}",
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
            }
        )
    return records


def _materialize_evaluations(
    attempt_dir: Path,
    fold_dir: Path,
    parent_fold_dir: Path,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    attempt_id = str(read_json(attempt_dir / METADATA_FILE)["attempt_id"])
    for variant, backend in zip(VARIANTS, (LOGGREG_PREDICTION_BACKEND, XGB_PREDICTION_BACKEND)):
        metrics_path = fold_dir / "hidden_classifiers" / variant / "metrics.json"
        predictions_path = fold_dir / "hidden_classifiers" / variant / "predictions_subject_level.csv"
        metrics = read_json(metrics_path)
        metrics_sha = sha256_file(metrics_path)
        qualifiers = _evaluation_qualifiers(parent_fold_dir, backend, metrics_sha)
        eval_id = evaluation_id(
            attempt_id=attempt_id,
            fold=0,
            **qualifiers,
        )
        evaluations.append(
            {
                "evaluation_id": eval_id,
                "dataset": "daic",
                "split_name": "test",
                "split_protocol": "daic_official_train_fit_locked_test_evaluation",
                "checkpoint_role": "best_model",
                "checkpoint_path": str(parent_fold_dir / "best_model"),
                "backend": backend,
                "evaluation_view": "harmonized_all_windows_full_coverage",
                "aggregation": "subject_level",
                "metric_namespace": "headline/binary_strict",
                "metrics_artifact_path": f"hidden_classifiers/{variant}/metrics.json",
                "predictions_artifact_path": f"hidden_classifiers/{variant}/predictions_subject_level.csv",
                "metrics": [
                    {"name": name, "value": metrics.get(name), "support": 47}
                    for name in EVALUATION_METRIC_NAMES
                ],
                "locally_verified": False,
                "reportable": False,
                "warnings": [],
            }
        )
    return evaluations


def materialize_mn5_evidence(
    attempt_dir: str | Path,
    parent_fold_dir: str | Path,
    *,
    transition_to_completed: bool = True,
) -> dict[str, Any]:
    """Materialize compact artifacts and evaluation records on MN5 after the
    head job completes. Never writes the SQLite registry."""
    sidecars = _read_sidecars(attempt_dir)
    attempt_path = Path(attempt_dir)
    fold_dir = attempt_path
    artifact_records = _hidden_feature_artifacts(attempt_path, fold_dir)
    artifact_records += _classifier_artifacts(attempt_path, fold_dir)

    attempt_id = sidecars.attempt_id
    existing = read_json(attempt_path / ARTIFACTS_FILE)
    known = {record["path"] for record in existing.get("artifacts", [])}
    additions: list[dict[str, Any]] = []
    for record in artifact_records:
        if record["path"] in known:
            continue
        additions.append(
            {
                "artifact_id": artifact_id(
                    attempt_id=attempt_id,
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

    new_evaluations = _materialize_evaluations(attempt_path, fold_dir, Path(parent_fold_dir))
    evaluations_record = read_json(attempt_path / EVALUATIONS_FILE)
    prior_by_id = {
        entry["evaluation_id"]: entry for entry in evaluations_record.get("evaluations", [])
    }
    prior_by_backend = {
        entry["backend"]: entry for entry in evaluations_record.get("evaluations", [])
    }
    for entry in new_evaluations:
        prior = prior_by_id.get(entry["evaluation_id"])
        if prior is not None:
            if prior != entry:
                raise CampaignError(
                    "refusing to change evaluation record content: "
                    f"{entry['evaluation_id']}"
                )
            continue
        same_backend = prior_by_backend.get(entry["backend"])
        if same_backend is not None:
            raise CampaignError(
                "refusing to add a new evaluation record for an existing backend: "
                f"{entry['backend']} already has {same_backend['evaluation_id']}, "
                f"materialized {entry['evaluation_id']}"
            )
        evaluations_record.setdefault("evaluations", []).append(entry)
    write_json_atomic(attempt_path / EVALUATIONS_FILE, evaluations_record)

    result = {
        "status": "materialized",
        "attempt_id": attempt_id,
        "artifacts": len(artifact_records),
        "evaluations": len(new_evaluations),
    }
    if transition_to_completed:
        status_path = attempt_path / STATUS_FILE
        record = StatusRecord.from_dict(read_json(status_path))
        if record.state == "RUNNING":
            record.transition("COMPLETED_ON_MN5", reason="extraction and head jobs COMPLETED")
            write_status(status_path, record)
        result["state"] = record.state
    return result


def _recompute_subject_metrics(fold_dir: Path, variant: str) -> dict[str, Any]:
    """Recompute subject predictions from sample probabilities and the six
    headline metrics from the subject rows."""
    from src.aggregate import aggregate_binary_classifier_predictions

    variant_dir = fold_dir / "hidden_classifiers" / variant
    sample_rows = read_jsonl(variant_dir / "predictions_sample_level.jsonl")
    subject_rows, metrics = aggregate_binary_classifier_predictions(sample_rows)
    return {
        "subject_rows": subject_rows,
        "metrics": metrics,
        "subject_row_count": len(subject_rows),
    }


def verify_local(attempt_dir: str | Path) -> dict[str, Any]:
    """Verify every compact artifact hash locally, recompute subject
    predictions and all six metrics, match them to metrics.json and the
    evaluation records, then mark artifacts/evaluations locally verified and
    transition through SYNCED_LOCALLY, LOCALLY_VALIDATED to REPORTABLE."""
    sidecars = _read_sidecars(attempt_dir)
    attempt_path = Path(attempt_dir)
    fold_dir = attempt_path

    # 1. Sidecar schemas and lifecycle history were validated by _read_sidecars;
    #    require the state to be at least SYNCED_LOCALLY to become reportable.
    if sidecars.state not in {"SYNCED_LOCALLY", "COMPLETED_ON_MN5", "LOCALLY_VALIDATED"}:
        raise CampaignError(
            f"verify-local requires SYNCED_LOCALLY or later state, got {sidecars.state}"
        )

    # 2. Verify every hashed artifact exists locally with the recorded hash.
    artifact_record = read_json(attempt_path / ARTIFACTS_FILE)
    for artifact in artifact_record.get("artifacts", []):
        if artifact.get("sha256") is None:
            continue
        full = fold_dir / artifact["path"]
        if not full.is_file():
            raise CampaignError(
                f"artifact missing locally: {artifact['path']}"
            )
        if sha256_file(full) != artifact["sha256"]:
            raise CampaignError(
                f"artifact hash mismatch: {artifact['path']}"
            )

    # 3. Recompute subject predictions and metrics per variant.
    for variant in VARIANTS:
        recomputed = _recompute_subject_metrics(fold_dir, variant)
        variant_dir = fold_dir / "hidden_classifiers" / variant
        saved_metrics = read_json(variant_dir / "metrics.json")
        for name in EVALUATION_METRIC_NAMES:
            recomputed_value = recomputed["metrics"].get(name)
            saved_value = saved_metrics.get(name)
            if recomputed_value is None or saved_value is None:
                raise CampaignError(
                    f"{variant} metrics.json missing {name}"
                )
            if abs(float(recomputed_value) - float(saved_value)) > 1e-9:
                raise CampaignError(
                    f"{variant} recomputed {name}={recomputed_value} does not "
                    f"match metrics.json {saved_value}"
                )
        subject_rows = recomputed["subject_rows"]
        if len(subject_rows) != 47:
            raise CampaignError(
                f"{variant} recomputed subject rows {len(subject_rows)} != 47"
            )
        saved_subject_rows = read_jsonl(variant_dir / "predictions_subject_level.jsonl")
        if len(saved_subject_rows) != len(subject_rows):
            raise CampaignError(
                f"{variant} saved subject rows {len(saved_subject_rows)} != "
                f"recomputed {len(subject_rows)}"
            )
        for saved_row, recomputed_row in zip(saved_subject_rows, subject_rows):
            if (
                str(saved_row["subject_id"]) != str(recomputed_row["subject_id"])
                or int(saved_row["prediction"]) != int(recomputed_row["prediction"])
            ):
                raise CampaignError(
                    f"{variant} saved subject prediction differs from recomputed "
                    f"for subject {saved_row['subject_id']}"
                )

    # 4. Mark artifacts and evaluations locally verified.
    for artifact in artifact_record.get("artifacts", []):
        artifact["exists_locally"] = True
        artifact["locally_verified"] = True
    write_json_atomic(attempt_path / ARTIFACTS_FILE, artifact_record)

    evaluations_record = read_json(attempt_path / EVALUATIONS_FILE)
    for entry in evaluations_record.get("evaluations", []):
        entry["locally_verified"] = True
        entry["reportable"] = True
        entry["warnings"] = []
    write_json_atomic(attempt_path / EVALUATIONS_FILE, evaluations_record)

    # 5. Transition through the remaining local lifecycle.
    status_path = attempt_path / STATUS_FILE
    record = StatusRecord.from_dict(read_json(status_path))
    if record.state == "COMPLETED_ON_MN5":
        record.transition("SYNCED_LOCALLY", reason="compact evidence synced locally")
        write_status(status_path, record)
    if record.state == "SYNCED_LOCALLY":
        record.transition("LOCALLY_VALIDATED", reason="local hash and metric verification passed")
        write_status(status_path, record)
    if record.state == "LOCALLY_VALIDATED":
        record.transition("REPORTABLE", reason="all evidence locally verified and reportable")
        write_status(status_path, record)
    return {
        "status": "verified",
        "attempt_id": sidecars.attempt_id,
        "state": record.state,
        "verified_artifacts": len(artifact_record.get("artifacts", [])),
        "verified_evaluations": len(evaluations_record.get("evaluations", [])),
    }
