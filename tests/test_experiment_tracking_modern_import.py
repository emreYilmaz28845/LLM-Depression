from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.experiment_tracking import registry
from src.experiment_tracking.canonical import write_json_atomic
from src.experiment_tracking.discovery import discover_runs
from src.experiment_tracking.evidence import (
    set_metadata_supersedes,
    verify_artifacts_locally,
    verify_evaluations_locally,
)
from src.experiment_tracking.identity import evaluation_id
from src.experiment_tracking.lifecycle import append_job_event, new_job_event
from src.experiment_tracking.qualification import qualify_run
from src.experiment_tracking.schemas import validate_metadata
from src.experiment_tracking.sidecars import (
    SidecarValidationError,
    read_modern_sidecars,
)

from test_experiment_tracking_discovery import (
    build_standard_run,
    write_standalone_eval,
)

ATTEMPT_ID = "20260812T020449Z-gemma4_daic_text_only_seed1337-cca3f4ae-ed58a7a3"
GIT_COMMIT = "cca3f4aed1545e6dc8f6db48051a85fc424c4173"
SUPERSEDED_ATTEMPT_ID = "20260812T020449Z-gemma4_daic_audio_text_seed1337-cca3f4ae-5704f1f7"
RETRY_ATTEMPT_ID = "20260812T031624Z-gemma4_daic_audio_text_seed1337-a6749b05-146c8805"
MANIFEST_HASH = "7" * 64
SPLIT_HASH = "8" * 64


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _attempt_id(name: str) -> str:
    return f"20260812T020449Z-{name}-cca3f4ae-{uuid.uuid4().hex[:8]}"


def write_metadata(
    fold_dir: Path,
    *,
    attempt_id: str = ATTEMPT_ID,
    fold: int = 0,
    git_commit: str = GIT_COMMIT,
    supersedes: str | None = None,
) -> None:
    metadata = {
        "schema_version": "audiollm.metadata.v1",
        "group_id": "gemma4-daic-v1-cca3f4ae",
        "logical_run_name": "gemma4_daic_text_only_seed1337",
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": 1337,
        "created_at_utc": _ts(),
        "source": {
            "git_commit": git_commit,
            "git_branch": "main",
            "git_dirty": False,
            "deployed_source_sha256": None,
        },
        "research": {"github_issue": None, "github_pr": 31},
        "hashes": {
            "resolved_config_sha256": "a" * 64,
            "manifest_sha256": MANIFEST_HASH,
            "split_sha256": SPLIT_HASH,
        },
        "paths": {"run_config": "run_config.yaml", "best_model": "best_model", "local_evidence_root": None},
        "wandb": {"project": "audiollm-depression", "entity": None, "run_id": f"{attempt_id}-fold{fold}", "url": None, "sync_status": "NOT_EXPORTED"},
    }
    if supersedes is not None:
        metadata["supersedes_attempt_id"] = supersedes
    write_json_atomic(fold_dir / "metadata.json", metadata)


def write_status(fold_dir: Path, *, attempt_id: str = ATTEMPT_ID, fold: int = 0, state: str = "LOCALLY_VALIDATED") -> None:
    if state == "FAILED":
        history = [
            {"from": "SUBMITTED", "to": "RUNNING", "at_utc": _ts(), "reason": "training job started"},
            {"from": "RUNNING", "to": "FAILED", "at_utc": _ts(), "reason": "train job FAILED"},
        ]
    else:
        history = [
            {"from": "SUBMITTED", "to": "RUNNING", "at_utc": _ts(), "reason": "training job started"},
            {"from": "RUNNING", "to": "COMPLETED_ON_MN5", "at_utc": _ts(), "reason": "train + eval COMPLETED"},
            {"from": "COMPLETED_ON_MN5", "to": "SYNCED_LOCALLY", "at_utc": _ts(), "reason": "evidence synced"},
            {"from": "SYNCED_LOCALLY", "to": "LOCALLY_VALIDATED", "at_utc": _ts(), "reason": "audit passed"},
        ]
        if state == "REPORTABLE":
            history.append({"from": "LOCALLY_VALIDATED", "to": "REPORTABLE", "at_utc": _ts(), "reason": "reportable"})
    write_json_atomic(
        fold_dir / "status.json",
        {
            "schema_version": "audiollm.status.v1",
            "attempt_id": attempt_id,
            "fold": fold,
            "state": state,
            "updated_at_utc": _ts(),
            "history": history,
        },
    )


def write_jobs(fold_dir: Path, *, attempt_id: str = ATTEMPT_ID, fold: int = 0) -> None:
    for job_key, job_type, slurm_job_id, event_type, status in (
        ("train", "train", "44517484", "SUBMITTED", "PENDING"),
        ("best_eval", "evaluation", "44517485", "SUBMITTED", "PENDING"),
        ("train", "train", None, "STARTED", "RUNNING"),
        ("train", "train", None, "COMPLETED", "COMPLETED"),
        ("best_eval", "evaluation", "44517485", "STARTED", "RUNNING"),
        ("best_eval", "evaluation", "44517485", "COMPLETED", "COMPLETED"),
    ):
        append_job_event(
            fold_dir / "jobs.jsonl",
            new_job_event(
                job_key=job_key,
                job_type=job_type,
                event_type=event_type,
                attempt_id=attempt_id,
                fold=fold,
                slurm_job_id=slurm_job_id,
                status=status,
            ),
        )


def write_artifacts(fold_dir: Path, *, attempt_id: str = ATTEMPT_ID, fold: int = 0, verified: bool = True) -> None:
    from src.experiment_tracking.identity import artifact_id

    records = []
    for role, path, artifact_type in (
        ("run_config", "run_config.yaml", "run_config"),
        ("standalone_eval_metrics", "best_model/standalone_eval/metrics_original_teacher_forced.json", "metrics"),
        ("standalone_eval_predictions", "best_model/standalone_eval/predictions_subject_level.csv", "predictions"),
    ):
        full = fold_dir / path
        content = path.endswith(".yaml") or path.endswith(".csv")
        if not content:
            full.write_text(json.dumps({"aggregation_level": "subject", "num_units": 47, "binary_strict_macro_f1": 0.75}), encoding="utf-8")
        else:
            full.write_text("x\n" if path.endswith(".csv") else "fold: 0\n", encoding="utf-8")
        from src.experiment_tracking.canonical import sha256_file

        sha = sha256_file(full)
        records.append(
            {
                "artifact_id": artifact_id(attempt_id=attempt_id, fold=fold, role=role, relative_path=path, artifact_sha256=sha),
                "artifact_type": artifact_type,
                "role": role,
                "path": path,
                "sha256": sha,
                "size_bytes": full.stat().st_size,
                "exists_on_mn5": True,
                "exists_locally": False,
                "locally_verified": False,
            }
        )
    (fold_dir / "best_model" / "standalone_eval").mkdir(parents=True, exist_ok=True)
    if not verified:
        (fold_dir / "best_model" / "standalone_eval" / "metrics_original_teacher_forced.json").unlink()
    write_json_atomic(
        fold_dir / "artifacts.json",
        {
            "schema_version": "audiollm.artifacts.v1",
            "attempt_id": attempt_id,
            "fold": fold,
            "artifacts": records,
        },
    )


def write_evaluations(fold_dir: Path, *, attempt_id: str = ATTEMPT_ID, fold: int = 0, reportable: bool = False) -> None:
    eval_id = evaluation_id(
        attempt_id=attempt_id,
        fold=fold,
        dataset="daic",
        split_name="test",
        split_protocol="fixed_train_val_test",
        checkpoint_role="best_model",
        checkpoint_path="best_model",
        backend="original_teacher_forced",
        evaluation_view="harmonized_all_windows_full_coverage",
        aggregation="subject_level",
        metric_namespace="headline/binary_strict",
        metrics_artifact_sha256="3fe944ed0635e515d2a3ec2ec1ae8ac342f173ba9559f12b2e26e1ffaf26eb83",
    )
    write_json_atomic(
        fold_dir / "evaluations.json",
        {
            "schema_version": "audiollm.evaluations.v1",
            "attempt_id": attempt_id,
            "fold": fold,
            "evaluations": [
                {
                    "evaluation_id": eval_id,
                    "dataset": "daic",
                    "split_name": "test",
                    "split_protocol": "fixed_train_val_test",
                    "checkpoint_role": "best_model",
                    "checkpoint_path": "best_model",
                    "backend": "original_teacher_forced",
                    "evaluation_view": "harmonized_all_windows_full_coverage",
                    "aggregation": "subject_level",
                    "metric_namespace": "headline/binary_strict",
                    "metrics_artifact_path": "best_model/standalone_eval/metrics_original_teacher_forced.json",
                    "predictions_artifact_path": "best_model/standalone_eval/predictions_subject_level.csv",
                    "metrics": [
                        {"name": "accuracy", "value": 0.7872340425531915, "support": 47},
                        {"name": "positive_f1", "value": 0.6666666666666666, "support": 47},
                        {"name": "macro_f1", "value": 0.7552083333333333, "support": 47},
                    ],
                    "locally_verified": reportable,
                    "reportable": reportable,
                    "warnings": [],
                }
            ],
        },
    )


def build_modern_run(tmp_path: Path, *, run_name: str = "gemma_run", state: str = "LOCALLY_VALIDATED") -> Path:
    fold_dir = build_standard_run(tmp_path, run_name=run_name)
    write_metadata(fold_dir)
    write_status(fold_dir, state=state)
    write_jobs(fold_dir)
    write_artifacts(fold_dir)
    write_evaluations(fold_dir)
    return fold_dir


def _import(tmp_path: Path, db_path: Path) -> dict:
    runs = discover_runs(tmp_path)
    connection = registry.ensure_registry(db_path)
    try:
        return registry.import_run(connection, runs[0], qualify_run(runs[0]))
    finally:
        connection.close()


def test_modern_run_imported_under_real_attempt_id(tmp_path: Path) -> None:
    build_modern_run(tmp_path)
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    outcome = registry.import_run(connection, run, qualify_run(run))
    assert outcome["status"] == "IMPORTED"
    assert outcome["attempt_id"] == ATTEMPT_ID
    attempt = connection.execute(
        "SELECT * FROM run_attempts WHERE attempt_id = ?", (ATTEMPT_ID,)
    ).fetchone()
    assert attempt is not None
    assert attempt["legacy_import"] == 0
    assert attempt["current_state"] == "LOCALLY_VALIDATED"
    assert attempt["git_commit"] == GIT_COMMIT
    assert attempt["git_branch"] == "main"
    assert attempt["git_dirty"] == 0
    assert attempt["github_pr"] == 31
    assert attempt["manifest_sha256"] == MANIFEST_HASH
    assert attempt["split_sha256"] == SPLIT_HASH
    assert attempt["metadata_path"] == "metadata.json"
    legacy_rows = connection.execute(
        "SELECT COUNT(*) FROM run_attempts WHERE attempt_id LIKE 'legacy-%'"
    ).fetchone()[0]
    assert legacy_rows == 0
    assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 6
    assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3
    connection.close()


def test_modern_import_is_idempotent(tmp_path: Path) -> None:
    build_modern_run(tmp_path)
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    first = registry.import_run(connection, run, qualify_run(run))
    second = registry.import_run(connection, run, qualify_run(run))
    assert first["status"] == "IMPORTED"
    assert second["status"] == "SKIPPED_DUPLICATE"
    assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
    connection.close()


def test_rebuild_preserves_modern_provenance_and_lifecycle(tmp_path: Path) -> None:
    build_modern_run(tmp_path, state="LOCALLY_VALIDATED")
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == 1
    assert summary["modern_rejected_runs"] == 0
    connection = registry.connect(db_path)
    try:
        attempt = connection.execute(
            "SELECT * FROM run_attempts WHERE attempt_id = ?", (ATTEMPT_ID,)
        ).fetchone()
        assert attempt is not None
        assert attempt["git_commit"] == GIT_COMMIT
        assert attempt["current_state"] == "LOCALLY_VALIDATED"
        assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 6
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM run_attempts WHERE attempt_id LIKE 'legacy-%'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_reportable_state_requires_verified_evidence(tmp_path: Path) -> None:
    build_modern_run(tmp_path, state="REPORTABLE")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    outcome = registry.import_run(connection, run, qualify_run(run))
    assert outcome["status"] == "REJECTED"
    assert any(
        "REPORTABLE" in reason or "not reportable" in reason for reason in outcome["reasons"]
    )
    assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    audit = connection.execute("SELECT status, details_json FROM registry_imports").fetchone()
    assert audit["status"] == "REJECTED"
    connection.close()


def test_verified_evidence_can_become_reportable(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["modern_rejected_runs"] == 0
    connection = registry.connect(db_path)
    try:
        attempt = connection.execute(
            "SELECT current_state FROM run_attempts WHERE attempt_id = ?", (ATTEMPT_ID,)
        ).fetchone()
        assert attempt["current_state"] == "REPORTABLE"
        evaluation = connection.execute("SELECT locally_verified, reportable FROM evaluations").fetchone()
        assert evaluation["locally_verified"] == 1
        assert evaluation["reportable"] == 1
    finally:
        connection.close()


def test_malformed_modern_sidecars_fail_closed(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    (fold_dir / "status.json").write_text("{ not json", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    outcome = registry.import_run(connection, run, qualify_run(run))
    assert outcome["status"] == "REJECTED"
    assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM run_attempts WHERE attempt_id LIKE 'legacy-%'"
    ).fetchone()[0] == 0
    connection.close()


def test_rejected_modern_import_is_idempotent(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    (fold_dir / "status.json").write_text("{ not json", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    first = registry.import_run(connection, run, qualify_run(run))
    second = registry.import_run(connection, run, qualify_run(run))
    assert first["status"] == "REJECTED"
    assert second["status"] == "REJECTED"
    assert connection.execute(
        "SELECT COUNT(*) FROM registry_imports WHERE status = 'REJECTED'"
    ).fetchone()[0] == 1
    connection.close()


def test_contradictory_attempt_ids_fail_closed(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    write_metadata(fold_dir, attempt_id=_attempt_id("gemma4_daic_audio_only_seed1337"))
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    outcome = registry.import_run(connection, run, qualify_run(run))
    assert outcome["status"] == "REJECTED"
    assert any("attempt_id" in reason for reason in outcome["reasons"])
    connection.close()


def test_supersedes_relationship_survives_rebuild(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    write_metadata(fold_dir, supersedes=SUPERSEDED_ATTEMPT_ID)
    write_status(fold_dir)
    write_jobs(fold_dir)
    write_artifacts(fold_dir)
    write_evaluations(fold_dir)

    failed_dir = build_standard_run(tmp_path, run_name="gemma_failed")
    write_metadata(failed_dir, attempt_id=SUPERSEDED_ATTEMPT_ID)
    write_status(failed_dir, attempt_id=SUPERSEDED_ATTEMPT_ID, state="FAILED")
    for event in (
        ("train", "train", "44517567", "SUBMITTED", "PENDING"),
        ("best_eval", "evaluation", "44517568", "SUBMITTED", "PENDING"),
        ("train", "train", None, "STARTED", "RUNNING"),
        ("train", "train", "44517567", "FAILED", "FAILED"),
    ):
        append_job_event(
            failed_dir / "jobs.jsonl",
            new_job_event(
                job_key=event[0],
                job_type=event[1],
                event_type=event[3],
                attempt_id=SUPERSEDED_ATTEMPT_ID,
                fold=0,
                slurm_job_id=event[2],
                status=event[4],
            ),
        )
    write_artifacts(failed_dir, attempt_id=SUPERSEDED_ATTEMPT_ID)

    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == 2
    assert summary["modern_rejected_runs"] == 0
    connection = registry.connect(db_path)
    try:
        attempt = connection.execute(
            "SELECT supersedes_attempt_id FROM run_attempts WHERE attempt_id = ?", (ATTEMPT_ID,)
        ).fetchone()
        assert attempt["supersedes_attempt_id"] == SUPERSEDED_ATTEMPT_ID
        failed = connection.execute(
            "SELECT current_state FROM run_attempts WHERE attempt_id = ?", (SUPERSEDED_ATTEMPT_ID,)
        ).fetchone()
        assert failed["current_state"] == "FAILED"
        assert connection.execute(
            "SELECT COUNT(*) FROM provenance WHERE key = 'pending_supersedes_attempt_id'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_metadata_supersedes_field_validates() -> None:
    ok, errors = validate_metadata(
        {
            "schema_version": "audiollm.metadata.v1",
            "group_id": None,
            "logical_run_name": "run",
            "attempt_id": RETRY_ATTEMPT_ID,
            "supersedes_attempt_id": SUPERSEDED_ATTEMPT_ID,
            "fold": 0,
            "seed": None,
            "created_at_utc": "2026-08-12T03:20:50.685523Z",
            "source": {"git_commit": None, "git_branch": None, "git_dirty": False, "deployed_source_sha256": None},
            "research": {"github_issue": None, "github_pr": None},
            "hashes": {"resolved_config_sha256": None, "manifest_sha256": None, "split_sha256": None},
            "paths": {"run_config": "run_config.yaml", "best_model": None, "local_evidence_root": None},
            "wandb": {"project": None, "entity": None, "run_id": None, "url": None, "sync_status": "NOT_EXPORTED"},
        }
    )
    assert ok, errors
    ok, errors = validate_metadata(
        {
            "schema_version": "audiollm.metadata.v1",
            "group_id": None,
            "logical_run_name": "run",
            "attempt_id": RETRY_ATTEMPT_ID,
            "supersedes_attempt_id": "not-an-attempt-id",
            "fold": 0,
            "seed": None,
            "created_at_utc": "2026-08-12T03:20:50.685523Z",
            "source": {"git_commit": None, "git_branch": None, "git_dirty": False, "deployed_source_sha256": None},
            "research": {"github_issue": None, "github_pr": None},
            "hashes": {"resolved_config_sha256": None, "manifest_sha256": None, "split_sha256": None},
            "paths": {"run_config": "run_config.yaml", "best_model": None, "local_evidence_root": None},
            "wandb": {"project": None, "entity": None, "run_id": None, "url": None, "sync_status": "NOT_EXPORTED"},
        }
    )
    assert not ok
    assert any("supersedes_attempt_id" in error for error in errors)


def test_legacy_only_runs_still_import_normally(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="legacy_run")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    outcome = registry.import_run(connection, run, qualify_run(run))
    assert outcome["status"] == "IMPORTED"
    assert outcome["attempts"] == 1
    attempt = connection.execute("SELECT * FROM run_attempts").fetchone()
    assert attempt["attempt_id"].startswith("legacy-")
    assert attempt["legacy_import"] == 1
    connection.close()


def test_evidence_verification_flips_flags_from_local_hashes(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    sidecars = read_modern_sidecars(fold_dir)
    assert all(artifact["locally_verified"] is False for artifact in sidecars.artifacts)
    artifacts_result = verify_artifacts_locally(fold_dir)
    assert artifacts_result["verified_artifacts"] == 3
    evaluations_result = verify_evaluations_locally(fold_dir)
    assert evaluations_result["verified_evaluations"] == 1
    assert evaluations_result["reportable_evaluations"] == 1
    sidecars = read_modern_sidecars(fold_dir)
    assert all(artifact["locally_verified"] is True for artifact in sidecars.artifacts)
    assert sidecars.evaluations[0]["locally_verified"] is True
    assert sidecars.evaluations[0]["reportable"] is True


def test_evidence_verification_refuses_missing_artifacts(tmp_path: Path) -> None:
    from src.experiment_tracking.evidence import EvidenceVerificationError

    fold_dir = build_modern_run(tmp_path)
    write_artifacts(fold_dir, verified=False)
    write_evaluations(fold_dir, reportable=False)
    verify_artifacts_locally(fold_dir)
    with pytest.raises(EvidenceVerificationError):
        verify_evaluations_locally(fold_dir)


def test_set_metadata_supersedes_only_when_absent(tmp_path: Path) -> None:
    from src.experiment_tracking.evidence import EvidenceVerificationError

    fold_dir = build_modern_run(tmp_path)
    result = set_metadata_supersedes(fold_dir, SUPERSEDED_ATTEMPT_ID)
    assert result["changed"] is True
    with pytest.raises(EvidenceVerificationError):
        set_metadata_supersedes(fold_dir, RETRY_ATTEMPT_ID)


def test_failed_modern_run_imports_without_evaluations(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path)
    (fold_dir / "evaluations.json").unlink()
    write_status(fold_dir, state="FAILED")
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == 1
    connection = registry.connect(db_path)
    try:
        attempt = connection.execute(
            "SELECT current_state FROM run_attempts WHERE attempt_id = ?", (ATTEMPT_ID,)
        ).fetchone()
        assert attempt["current_state"] == "FAILED"
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 6
    finally:
        connection.close()


def test_tampered_predictions_after_verification_reject_reportable_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    predictions = fold_dir / "best_model/standalone_eval/predictions_subject_level.csv"
    predictions.write_text("subject_id,prediction\ntampered\n", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["modern_rejected_runs"] == 1
    assert summary["imported_runs"] == 0
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM folds").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM run_attempts WHERE attempt_id LIKE 'legacy-%'"
        ).fetchone()[0] == 0
        rejection = connection.execute(
            "SELECT details_json FROM registry_imports WHERE status = 'REJECTED'"
        ).fetchone()
        assert rejection is not None
        details = json.loads(rejection["details_json"])
        joined = "\n".join(details["reasons"])
        assert "predictions_subject_level.csv" in joined
        assert "SHA-256" in joined
        assert "differs" in joined
    finally:
        connection.close()


def test_tampered_metrics_after_verification_reject_reportable_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    metrics = fold_dir / "best_model/standalone_eval/metrics_original_teacher_forced.json"
    metrics.write_text('{"binary_strict_macro_f1": 0.999}', encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    outcome = _import(tmp_path, db_path)
    assert outcome["status"] == "REJECTED"
    assert any(
        "metrics_original_teacher_forced.json" in reason and "SHA-256" in reason
        for reason in outcome["reasons"]
    )
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_deleted_evaluation_artifact_rejects_reportable_run(tmp_path: Path) -> None:
    for index, missing in enumerate(
        (
            "best_model/standalone_eval/metrics_original_teacher_forced.json",
            "best_model/standalone_eval/predictions_subject_level.csv",
        )
    ):
        root = tmp_path / f"case_{index}"
        fold_dir = build_modern_run(root, state="REPORTABLE")
        verify_artifacts_locally(fold_dir)
        verify_evaluations_locally(fold_dir)
        write_status(fold_dir, state="REPORTABLE")
        (fold_dir / missing).unlink()
        db_path = root / "registry.sqlite"
        outcome = _import(root, db_path)
        assert outcome["status"] == "REJECTED"
        assert any(missing in reason for reason in outcome["reasons"])
        connection = registry.connect(db_path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
        finally:
            connection.close()


def test_evaluation_artifact_absent_from_artifacts_json_rejects_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    record = json.loads((fold_dir / "artifacts.json").read_text(encoding="utf-8"))
    record["artifacts"] = [
        artifact
        for artifact in record["artifacts"]
        if artifact["path"] != "best_model/standalone_eval/predictions_subject_level.csv"
    ]
    write_json_atomic(fold_dir / "artifacts.json", record)
    db_path = tmp_path / "registry.sqlite"
    outcome = _import(tmp_path, db_path)
    assert outcome["status"] == "REJECTED"
    assert any(
        "predictions_subject_level.csv" in reason and "artifacts.json record" in reason
        for reason in outcome["reasons"]
    )
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_reportable_evaluation_without_local_verification_rejects_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    db_path = tmp_path / "registry.sqlite"
    outcome = _import(tmp_path, db_path)
    assert outcome["status"] == "REJECTED"
    assert any("not locally verified" in reason for reason in outcome["reasons"])
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_reportable_evaluation_with_warnings_rejects_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    record = json.loads((fold_dir / "evaluations.json").read_text(encoding="utf-8"))
    record["evaluations"][0]["warnings"] = ["legacy ambiguity"]
    write_json_atomic(fold_dir / "evaluations.json", record)
    write_status(fold_dir, state="REPORTABLE")
    db_path = tmp_path / "registry.sqlite"
    outcome = _import(tmp_path, db_path)
    assert outcome["status"] == "REJECTED"
    assert any("warnings" in reason for reason in outcome["reasons"])
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_reportable_attempt_without_evaluations_rejects_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    write_evaluations(fold_dir, reportable=True)
    (fold_dir / "evaluations.json").unlink()
    write_status(fold_dir, state="REPORTABLE")
    db_path = tmp_path / "registry.sqlite"
    outcome = _import(tmp_path, db_path)
    assert outcome["status"] == "REJECTED"
    assert any("no evaluations" in reason for reason in outcome["reasons"])
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_missing_last_model_does_not_reject_reportable_run(tmp_path: Path) -> None:
    fold_dir = build_modern_run(tmp_path, state="REPORTABLE")
    verify_artifacts_locally(fold_dir)
    verify_evaluations_locally(fold_dir)
    write_status(fold_dir, state="REPORTABLE")
    last_model = fold_dir / "last_model"
    assert last_model.is_dir()
    last_model.rmdir()
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == 1
    assert summary["modern_rejected_runs"] == 0
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
    finally:
        connection.close()
