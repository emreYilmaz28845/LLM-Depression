from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_sha256, format_utc_timestamp, sha256_file, utc_now
from .constants import SCHEMA_VERSION_METADATA
from .discovery import DiscoveredRun, discover_runs
from .identity import artifact_id as make_artifact_id
from .identity import logical_run_id, sanitize_logical_run_name
from .qualification import (
    STATUS_QUALIFIED,
    STATUS_QUARANTINED_AMBIGUOUS,
    STATUS_REJECTED,
    QualificationResult,
    legacy_identity_payload,
    qualify_run,
)
from .sidecars import (
    METADATA_FILE,
    STATUS_FILE,
    ModernSidecars,
    SidecarValidationError,
    is_modern_tracked,
    read_modern_sidecars,
    reportable_state_issues,
)

DEFAULT_DB_PATH = "outputs/experiment_registry/experiments.sqlite"
IMPORTER_VERSION = "audiollm.registry_importer.v1"

_QUARANTINE_STATUSES = (STATUS_QUARANTINED_AMBIGUOUS, STATUS_REJECTED)
_FAILED_JOB_STATUSES = (
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
)


class RegistryError(Exception):
    pass


class SchemaMismatchError(RegistryError):
    pass


def migration_path() -> Path:
    return Path(__file__).parent / "migrations" / "001_initial.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 1:
        return
    if version != 0:
        raise SchemaMismatchError(
            f"database schema version is {version}, expected 1; "
            "do not delete or rewrite the database; resolve the migration first"
        )
    connection.executescript(migration_path().read_text(encoding="utf-8"))
    if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise RegistryError("migration did not set PRAGMA user_version = 1")


def ensure_registry(db_path: str | Path) -> sqlite3.Connection:
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(target)
    initialize(connection)
    return connection


def _evidence_manifest_hash(discovered: DiscoveredRun) -> str:
    records = [
        {"relative_path": artifact.relative_path, "sha256": artifact.sha256}
        for artifact in discovered.artifacts
        if artifact.sha256 is not None
    ]
    records.append(
        {"relative_path": "run_config.yaml", "sha256": discovered.run_config_file_sha256}
    )
    return canonical_sha256(sorted(records, key=lambda record: record["relative_path"]))


def _insert_logical_run(
    cursor: sqlite3.Cursor, discovered: DiscoveredRun
) -> str:
    config = discovered.resolved_config or {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), str) else None
    seed = config.get("seed") if isinstance(config.get("seed"), int) else None
    logical_id = logical_run_id(
        group_id=None,
        logical_run_name=discovered.run_name,
        dataset=dataset,
        modality=discovered.modality,
        method=None,
        seed=seed,
    )
    cursor.execute(
        "INSERT OR IGNORE INTO logical_runs "
        "(logical_run_id, group_id, logical_run_name, dataset, modality, method, seed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (logical_id, None, discovered.run_name, dataset, discovered.modality, None, seed),
    )
    return logical_id


def _insert_attempt_and_fold(
    cursor: sqlite3.Cursor,
    logical_run_id_value: str,
    discovered: DiscoveredRun,
    attempt_id_value: str,
) -> int:
    config = discovered.resolved_config or {}
    manifest_hash = discovered.protocol.get("manifest_hash") if discovered.protocol else None
    split_hash = (
        discovered.protocol.get("split_metadata_hash") if discovered.protocol else None
    )
    cursor.execute(
        "INSERT OR IGNORE INTO run_attempts "
        "(attempt_id, logical_run_id, schema_version, legacy_import, created_at_utc, "
        "git_commit, git_branch, git_dirty, deployed_source_sha256, resolved_config_sha256, "
        "manifest_sha256, split_sha256, github_issue, github_pr, supersedes_attempt_id, "
        "metadata_path, current_state) VALUES (?, ?, ?, 1, NULL, NULL, NULL, NULL, NULL, "
        "?, ?, ?, NULL, NULL, NULL, NULL, ?)",
        (
            attempt_id_value,
            logical_run_id_value,
            SCHEMA_VERSION_METADATA,
            discovered.resolved_config_sha256,
            manifest_hash if isinstance(manifest_hash, str) else None,
            split_hash if isinstance(split_hash, str) else None,
            "IMPORTED_LEGACY",
        ),
    )
    cursor.execute(
        "INSERT OR IGNORE INTO folds (attempt_id, fold, run_dir, run_config_path, status_path, locally_verified) "
        "VALUES (?, ?, ?, ?, NULL, 0)",
        (attempt_id_value, discovered.fold, discovered.fold_dir, discovered.run_config_path),
    )
    row = cursor.execute(
        "SELECT fold_id FROM folds WHERE attempt_id = ? AND fold = ?",
        (attempt_id_value, discovered.fold),
    ).fetchone()
    return row[0]


def _insert_artifacts(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    attempt_id_value: str,
    fold_id: int,
) -> dict[str, int]:
    artifact_ids: dict[str, int] = {}
    for artifact in discovered.artifacts:
        if artifact.sha256 is None:
            continue
        artifact_key = make_artifact_id(
            attempt_id=attempt_id_value,
            fold=discovered.fold,
            role=artifact.kind,
            relative_path=artifact.relative_path,
            artifact_sha256=artifact.sha256,
        )
        cursor.execute(
            "INSERT OR IGNORE INTO artifacts "
            "(artifact_id, fold_id, artifact_type, role, path, sha256, size_bytes, "
            "exists_on_mn5, exists_locally, locally_verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1, 0)",
            (
                artifact_key,
                fold_id,
                artifact.artifact_type,
                artifact.kind,
                artifact.relative_path,
                artifact.sha256,
                artifact.size_bytes,
            ),
        )
        artifact_ids[artifact.relative_path] = artifact_key
    return artifact_ids


def _import_evaluations(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    result: QualificationResult,
    fold_ids: dict[str, int],
    artifact_ids: dict[str, int],
) -> None:
    config = discovered.resolved_config or {}
    protocol = discovered.protocol or {}
    for evaluation in result.evaluations:
        fold_id = fold_ids[evaluation.attempt_id]
        metrics_artifact_id = artifact_ids.get(evaluation.metrics_artifact_path or "")
        predictions_artifact_id = artifact_ids.get(evaluation.predictions_artifact_path or "")
        cursor.execute(
            "INSERT OR IGNORE INTO evaluations "
            "(evaluation_id, fold_id, dataset, split_name, split_protocol, checkpoint_role, "
            "checkpoint_path, backend, evaluation_view, aggregation, metric_namespace, "
            "metrics_artifact_id, predictions_artifact_id, locally_verified, reportable, "
            "warnings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                evaluation.evaluation_id,
                fold_id,
                evaluation.dataset,
                evaluation.split_name,
                evaluation.split_protocol,
                evaluation.checkpoint_role,
                evaluation.checkpoint_path,
                evaluation.backend,
                evaluation.evaluation_view,
                evaluation.aggregation,
                evaluation.metric_namespace,
                metrics_artifact_id,
                predictions_artifact_id,
                1 if evaluation.reportable else 0,
                json.dumps(list(evaluation.warnings), ensure_ascii=False),
            ),
        )
        for metric in evaluation.metrics:
            cursor.execute(
                "INSERT OR IGNORE INTO metrics "
                "(evaluation_id, namespace, metric_name, metric_value, support, aggregation, "
                "backend, evaluation_view, split_name, checkpoint_role, evidence_artifact_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation.evaluation_id,
                    evaluation.metric_namespace,
                    metric.name,
                    metric.value,
                    metric.support,
                    evaluation.aggregation,
                    evaluation.backend,
                    evaluation.evaluation_view,
                    evaluation.split_name,
                    evaluation.checkpoint_role,
                    metrics_artifact_id,
                ),
            )
        payload = legacy_identity_payload(
            discovered,
            _artifact_by_path(discovered, evaluation.metrics_artifact_path),
        )
        provenance_rows = [
            ("legacy_identity_input", json.dumps(payload, sort_keys=True, ensure_ascii=False)),
            ("relative_run_dir", payload.get("relative_run_dir")),
            ("evidence_manifest_sha256", _evidence_manifest_hash(discovered)),
        ]
        for key, value in provenance_rows:
            cursor.execute(
                "INSERT OR REPLACE INTO provenance "
                "(attempt_id, fold_id, key, value_json, source_artifact_id) VALUES (?, ?, ?, ?, ?)",
                (evaluation.attempt_id, fold_id, key, json.dumps(value, ensure_ascii=False), metrics_artifact_id),
            )


def _artifact_by_path(discovered: DiscoveredRun, relative_path: str | None):
    for artifact in discovered.artifacts:
        if artifact.relative_path == relative_path:
            return artifact
    raise RegistryError(
        f"metrics artifact not found among discovered artifacts: {relative_path!r}"
    )


def _insert_attempts_and_folds(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    result: QualificationResult,
) -> dict[str, int]:
    fold_ids: dict[str, int] = {}
    for evaluation in result.evaluations:
        if evaluation.attempt_id not in fold_ids:
            logical_id = _insert_logical_run(cursor, discovered)
            fold_ids[evaluation.attempt_id] = _insert_attempt_and_fold(
                cursor, logical_id, discovered, evaluation.attempt_id
            )
    return fold_ids


def _import_attempt_only(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    result: QualificationResult,
) -> None:
    config = discovered.resolved_config or {}
    seed = config.get("seed") if isinstance(config.get("seed"), int) else None
    payload = {
        "relative_run_dir": result.fold_dir,
        "fold": discovered.fold,
        "resolved_config_sha256": discovered.resolved_config_sha256,
        "manifest_sha256": None,
        "split_sha256": None,
        "checkpoint_role": None,
        "checkpoint_path": None,
        "evaluation_artifact_sha256": None,
    }
    attempt_id_value = (
        "legacy-"
        + sanitize_logical_run_name(discovered.run_name)
        + "-"
        + canonical_sha256(payload)[:24]
    )
    logical_id = _insert_logical_run(cursor, discovered)
    fold_id = _insert_attempt_and_fold(cursor, logical_id, discovered, attempt_id_value)
    cursor.execute(
        "INSERT INTO provenance (attempt_id, fold_id, key, value_json, source_artifact_id) "
        "VALUES (?, ?, ?, ?, NULL)",
        (
            attempt_id_value,
            fold_id,
            "attempt_identity_input",
            json.dumps(payload, sort_keys=True, ensure_ascii=False),
        ),
    )


# --------------------------------------------------------------------------- modern tracked runs

_LOCALLY_VALIDATED_STATES = ("LOCALLY_VALIDATED", "REPORTABLE")


def _insert_modern_logical_run(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    sidecars: ModernSidecars,
) -> str:
    metadata = sidecars.metadata
    config = discovered.resolved_config or {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), str) else None
    group_id = metadata.get("group_id")
    if isinstance(group_id, str) and group_id:
        cursor.execute(
            "INSERT OR IGNORE INTO experiment_groups "
            "(group_id, schema_version, title, research_question, github_issue, github_pr, "
            "status, source_path, source_sha256) VALUES (?, ?, ?, NULL, ?, ?, NULL, ?, ?)",
            (
                group_id,
                SCHEMA_VERSION_METADATA,
                group_id,
                metadata.get("research", {}).get("github_issue")
                if isinstance(metadata.get("research"), dict)
                else None,
                metadata.get("research", {}).get("github_pr")
                if isinstance(metadata.get("research"), dict)
                else None,
                f"{discovered.fold_dir}/metadata.json",
                sidecars.file_sha256[METADATA_FILE],
            ),
        )
    logical_id = logical_run_id(
        group_id=group_id,
        logical_run_name=metadata.get("logical_run_name") or discovered.run_name,
        dataset=dataset,
        modality=discovered.modality,
        method=None,
        seed=metadata.get("seed"),
    )
    cursor.execute(
        "INSERT OR IGNORE INTO logical_runs "
        "(logical_run_id, group_id, logical_run_name, dataset, modality, method, seed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            logical_id,
            group_id,
            metadata.get("logical_run_name") or discovered.run_name,
            dataset,
            discovered.modality,
            None,
            metadata.get("seed"),
        ),
    )
    return logical_id


def _insert_modern_attempt_and_fold(
    cursor: sqlite3.Cursor,
    logical_run_id_value: str,
    discovered: DiscoveredRun,
    sidecars: ModernSidecars,
) -> int:
    metadata = sidecars.metadata
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    research = metadata.get("research") if isinstance(metadata.get("research"), dict) else {}
    hashes = metadata.get("hashes") if isinstance(metadata.get("hashes"), dict) else {}
    attempt_id_value = sidecars.attempt_id
    supersedes = metadata.get("supersedes_attempt_id")
    cursor.execute(
        "INSERT OR IGNORE INTO run_attempts "
        "(attempt_id, logical_run_id, schema_version, legacy_import, created_at_utc, "
        "git_commit, git_branch, git_dirty, deployed_source_sha256, resolved_config_sha256, "
        "manifest_sha256, split_sha256, github_issue, github_pr, supersedes_attempt_id, "
        "metadata_path, current_state) VALUES (?, ?, ?, 0, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
        (
            attempt_id_value,
            logical_run_id_value,
            SCHEMA_VERSION_METADATA,
            metadata.get("created_at_utc"),
            source.get("git_commit"),
            source.get("git_branch"),
            source.get("git_dirty"),
            source.get("deployed_source_sha256"),
            hashes.get("resolved_config_sha256") or discovered.resolved_config_sha256,
            hashes.get("manifest_sha256"),
            hashes.get("split_sha256"),
            research.get("github_issue"),
            research.get("github_pr"),
            METADATA_FILE,
            sidecars.state,
        ),
    )
    if isinstance(supersedes, str):
        target_exists = cursor.execute(
            "SELECT 1 FROM run_attempts WHERE attempt_id = ?", (supersedes,)
        ).fetchone()
        if target_exists is not None:
            cursor.execute(
                "UPDATE run_attempts SET supersedes_attempt_id = ? WHERE attempt_id = ?",
                (supersedes, attempt_id_value),
            )
    locally_verified = (
        1 if sidecars.state in _LOCALLY_VALIDATED_STATES else 0
    )
    cursor.execute(
        "INSERT OR IGNORE INTO folds (attempt_id, fold, run_dir, run_config_path, status_path, locally_verified) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            attempt_id_value,
            sidecars.fold,
            discovered.fold_dir,
            discovered.run_config_path,
            STATUS_FILE,
            locally_verified,
        ),
    )
    row = cursor.execute(
        "SELECT fold_id FROM folds WHERE attempt_id = ? AND fold = ?",
        (attempt_id_value, sidecars.fold),
    ).fetchone()
    if isinstance(supersedes, str) and not target_exists:
        cursor.execute(
            "INSERT OR REPLACE INTO provenance (attempt_id, fold_id, key, value_json, source_artifact_id) "
            "VALUES (?, ?, ?, ?, NULL)",
            (attempt_id_value, row[0], "pending_supersedes_attempt_id", json.dumps(supersedes)),
        )
    return row[0]


def _insert_modern_jobs(
    cursor: sqlite3.Cursor,
    sidecars: ModernSidecars,
    fold_id: int,
) -> None:
    for event in sidecars.jobs:
        cursor.execute(
            "INSERT OR IGNORE INTO job_events "
            "(event_id, fold_id, job_key, job_type, event_type, slurm_job_id, "
            "slurm_array_job_id, slurm_array_task_id, dependency_job_ids_json, status, "
            "at_utc, reason, resubmission_of_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["event_id"],
                fold_id,
                event["job_key"],
                event["job_type"],
                event["event_type"],
                event.get("slurm_job_id"),
                event.get("slurm_array_job_id"),
                event.get("slurm_array_task_id"),
                json.dumps(list(event.get("dependency_job_ids") or [])),
                event.get("status"),
                event["at_utc"],
                event.get("reason"),
                event.get("resubmission_of_job_id"),
            ),
        )


def _insert_modern_artifacts(
    cursor: sqlite3.Cursor,
    fold_dir: Path,
    sidecars: ModernSidecars,
    fold_id: int,
) -> dict[str, int]:
    artifact_ids: dict[str, int] = {}
    for artifact in sidecars.artifacts:
        relative_path = artifact["path"]
        sha256 = artifact.get("sha256")
        full = fold_dir / relative_path
        exists_locally = 1 if full.exists() else 0
        locally_verified = 0
        if sha256 is not None and full.is_file():
            locally_verified = 1 if sha256_file(full) == sha256 else 0
        elif sha256 is None and full.is_dir():
            locally_verified = 1
        cursor.execute(
            "INSERT OR IGNORE INTO artifacts "
            "(artifact_id, fold_id, artifact_type, role, path, sha256, size_bytes, "
            "exists_on_mn5, exists_locally, locally_verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact["artifact_id"],
                fold_id,
                artifact["artifact_type"],
                artifact["role"],
                relative_path,
                sha256,
                artifact.get("size_bytes"),
                artifact.get("exists_on_mn5"),
                exists_locally,
                locally_verified,
            ),
        )
        artifact_ids[relative_path] = artifact["artifact_id"]
    return artifact_ids


def _insert_modern_evaluations(
    cursor: sqlite3.Cursor,
    sidecars: ModernSidecars,
    fold_id: int,
    artifact_ids: dict[str, int],
) -> None:
    for evaluation in sidecars.evaluations:
        metrics_artifact_id = artifact_ids.get(evaluation.get("metrics_artifact_path") or "")
        predictions_artifact_id = artifact_ids.get(evaluation.get("predictions_artifact_path") or "")
        cursor.execute(
            "INSERT OR IGNORE INTO evaluations "
            "(evaluation_id, fold_id, dataset, split_name, split_protocol, checkpoint_role, "
            "checkpoint_path, backend, evaluation_view, aggregation, metric_namespace, "
            "metrics_artifact_id, predictions_artifact_id, locally_verified, reportable, "
            "warnings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evaluation["evaluation_id"],
                fold_id,
                evaluation.get("dataset"),
                evaluation.get("split_name"),
                evaluation.get("split_protocol"),
                evaluation.get("checkpoint_role"),
                evaluation.get("checkpoint_path"),
                evaluation.get("backend"),
                evaluation.get("evaluation_view"),
                evaluation.get("aggregation"),
                evaluation.get("metric_namespace"),
                metrics_artifact_id,
                predictions_artifact_id,
                1 if evaluation.get("locally_verified") else 0,
                1 if evaluation.get("reportable") else 0,
                json.dumps(list(evaluation.get("warnings") or []), ensure_ascii=False),
            ),
        )
        for metric in evaluation.get("metrics") or []:
            cursor.execute(
                "INSERT OR IGNORE INTO metrics "
                "(evaluation_id, namespace, metric_name, metric_value, support, aggregation, "
                "backend, evaluation_view, split_name, checkpoint_role, evidence_artifact_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation["evaluation_id"],
                    evaluation.get("metric_namespace"),
                    metric["name"],
                    metric.get("value"),
                    metric.get("support"),
                    evaluation.get("aggregation"),
                    evaluation.get("backend"),
                    evaluation.get("evaluation_view"),
                    evaluation.get("split_name"),
                    evaluation.get("checkpoint_role"),
                    metrics_artifact_id,
                ),
            )


def _insert_modern_provenance(
    cursor: sqlite3.Cursor,
    discovered: DiscoveredRun,
    sidecars: ModernSidecars,
    fold_id: int,
    source_sha256: str,
) -> None:
    try:
        relative_run_dir = str(Path(discovered.fold_dir).relative_to(discovered.scan_root))
    except ValueError:
        relative_run_dir = discovered.fold_dir
    rows = [
        ("relative_run_dir", relative_run_dir),
        ("evidence_manifest_sha256", _evidence_manifest_hash(discovered)),
        ("modern_sidecar_sha256", source_sha256),
    ]
    for key, value in rows:
        cursor.execute(
            "INSERT OR REPLACE INTO provenance (attempt_id, fold_id, key, value_json, source_artifact_id) "
            "VALUES (?, ?, ?, ?, NULL)",
            (sidecars.attempt_id, fold_id, key, json.dumps(value, ensure_ascii=False)),
        )


def _backfill_pending_supersedes(connection: sqlite3.Connection) -> None:
    """Resolve supersede links that were deferred because the superseded
    attempt had not been imported yet (incremental single-run imports)."""
    pending = connection.execute(
        "SELECT p.attempt_id, p.fold_id, p.value_json FROM provenance p "
        "WHERE p.key = 'pending_supersedes_attempt_id'"
    ).fetchall()
    for row in pending:
        try:
            supersedes = json.loads(row["value_json"])
        except (ValueError, TypeError):
            continue
        if not isinstance(supersedes, str):
            continue
        target = connection.execute(
            "SELECT 1 FROM run_attempts WHERE attempt_id = ?", (supersedes,)
        ).fetchone()
        if target is None:
            continue
        connection.execute(
            "UPDATE run_attempts SET supersedes_attempt_id = ? WHERE attempt_id = ?",
            (supersedes, row["attempt_id"]),
        )
        connection.execute(
            "DELETE FROM provenance WHERE attempt_id = ? AND fold_id = ? AND key = "
            "'pending_supersedes_attempt_id'",
            (row["attempt_id"], row["fold_id"]),
        )


def _modern_import_outcome(
    connection: sqlite3.Connection,
    discovered: DiscoveredRun,
    sidecars: ModernSidecars,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    source_sha256 = sidecars.evidence_sha256(discovered.run_config_file_sha256)
    if dry_run:
        return {
            "status": "DRY_RUN",
            "fold_dir": discovered.fold_dir,
            "attempt_id": sidecars.attempt_id,
            "state": sidecars.state,
            "evaluations": len(sidecars.evaluations),
            "import_manifest_sha256": source_sha256,
        }
    cursor = connection.cursor()
    existing = cursor.execute(
        "SELECT status FROM registry_imports WHERE source_path = ? AND source_sha256 = ? "
        "AND importer_version = ?",
        (discovered.run_config_path, source_sha256, IMPORTER_VERSION),
    ).fetchone()
    if existing is not None:
        return {
            "status": "SKIPPED_DUPLICATE",
            "fold_dir": discovered.fold_dir,
            "attempt_id": sidecars.attempt_id,
            "attempts": 0,
            "evaluations": 0,
        }
    try:
        with connection:
            logical_id = _insert_modern_logical_run(cursor, discovered, sidecars)
            fold_id = _insert_modern_attempt_and_fold(
                cursor, logical_id, discovered, sidecars
            )
            _insert_modern_jobs(cursor, sidecars, fold_id)
            artifact_ids = _insert_modern_artifacts(
                cursor, Path(discovered.fold_dir), sidecars, fold_id
            )
            _insert_modern_evaluations(cursor, sidecars, fold_id, artifact_ids)
            _insert_modern_provenance(cursor, discovered, sidecars, fold_id, source_sha256)
            cursor.execute(
                "INSERT INTO registry_imports "
                "(source_path, source_sha256, importer_version, imported_at_utc, status, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    discovered.run_config_path,
                    source_sha256,
                    IMPORTER_VERSION,
                    format_utc_timestamp(utc_now()),
                    "IMPORTED",
                    json.dumps(
                        {
                            "kind": "modern",
                            "attempt_id": sidecars.attempt_id,
                            "state": sidecars.state,
                            "evaluations": len(sidecars.evaluations),
                            "warnings": [],
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise RegistryError(
            f"integrity violation while importing {discovered.fold_dir}: {error}"
        ) from error
    return {
        "status": "IMPORTED",
        "fold_dir": discovered.fold_dir,
        "attempt_id": sidecars.attempt_id,
        "attempts": 1,
        "evaluations": len(sidecars.evaluations),
    }


def _rejected_modern_outcome(
    connection: sqlite3.Connection,
    discovered: DiscoveredRun,
    reasons: list[str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "REJECTED",
            "fold_dir": discovered.fold_dir,
            "attempts": 0,
            "evaluations": 0,
            "reasons": reasons,
        }
    cursor = connection.cursor()
    source_sha256 = canonical_sha256(
        {
            "run_dir": discovered.fold_dir,
            "reasons": sorted(reasons),
        }
    )
    existing = cursor.execute(
        "SELECT status FROM registry_imports WHERE source_path = ? AND source_sha256 = ? "
        "AND importer_version = ?",
        (discovered.run_config_path, source_sha256, IMPORTER_VERSION),
    ).fetchone()
    if existing is not None:
        return {
            "status": "REJECTED",
            "fold_dir": discovered.fold_dir,
            "attempts": 0,
            "evaluations": 0,
            "reasons": reasons,
        }
    try:
        with connection:
            cursor.execute(
                "INSERT INTO registry_imports "
                "(source_path, source_sha256, importer_version, imported_at_utc, status, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    discovered.run_config_path,
                    source_sha256,
                    IMPORTER_VERSION,
                    format_utc_timestamp(utc_now()),
                    "REJECTED",
                    json.dumps(
                        {"kind": "modern", "reasons": sorted(reasons)},
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise RegistryError(
            f"integrity violation while rejecting {discovered.fold_dir}: {error}"
        ) from error
    return {
        "status": "REJECTED",
        "fold_dir": discovered.fold_dir,
        "attempts": 0,
        "evaluations": 0,
        "reasons": reasons,
    }


def import_run(
    connection: sqlite3.Connection,
    discovered: DiscoveredRun,
    result: QualificationResult,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if discovered.run_config_file_sha256 is None:
        raise RegistryError(
            f"cannot import {discovered.fold_dir}: run_config file hash unavailable"
        )
    if is_modern_tracked(discovered.fold_dir):
        try:
            sidecars = read_modern_sidecars(discovered.fold_dir)
        except SidecarValidationError as error:
            return _rejected_modern_outcome(
                connection,
                discovered,
                [str(error)],
                dry_run=dry_run,
            )
        if sidecars is None:
            raise RegistryError(
                f"cannot import {discovered.fold_dir}: modern sidecars disappeared during import"
            )
        reportable_issues = reportable_state_issues(sidecars)
        if reportable_issues:
            return _rejected_modern_outcome(
                connection,
                discovered,
                reportable_issues,
                dry_run=dry_run,
            )
        return _modern_import_outcome(connection, discovered, sidecars, dry_run=dry_run)
    manifest_hash = _evidence_manifest_hash(discovered)
    if dry_run:
        return {
            "status": "DRY_RUN",
            "fold_dir": discovered.fold_dir,
            "attempts": 0,
            "evaluations": len(result.evaluations),
            "import_manifest_sha256": manifest_hash,
        }
    cursor = connection.cursor()
    existing = cursor.execute(
        "SELECT status FROM registry_imports WHERE source_path = ? AND source_sha256 = ? "
        "AND importer_version = ?",
        (discovered.run_config_path, manifest_hash, IMPORTER_VERSION),
    ).fetchone()
    if existing is not None:
        return {
            "status": "SKIPPED_DUPLICATE",
            "fold_dir": discovered.fold_dir,
            "attempts": 0,
            "evaluations": 0,
        }
    try:
        with connection:
            if result.status == STATUS_QUALIFIED:
                if result.evaluations:
                    fold_ids = _insert_attempts_and_folds(cursor, discovered, result)
                    first_attempt = next(iter(fold_ids))
                    artifact_ids = _insert_artifacts(
                        cursor, discovered, first_attempt, fold_ids[first_attempt]
                    )
                    _import_evaluations(cursor, discovered, result, fold_ids, artifact_ids)
                    audit_status = "IMPORTED"
                    details = {
                        "attempts": len(fold_ids),
                        "evaluations": len(result.evaluations),
                        "warnings": list(result.warnings),
                    }
                else:
                    _import_attempt_only(cursor, discovered, result)
                    audit_status = "IMPORTED"
                    details = {
                        "attempts": 1,
                        "evaluations": 0,
                        "reasons": list(result.reasons),
                        "warnings": list(result.warnings),
                    }
            else:
                _import_attempt_only(cursor, discovered, result)
                audit_status = (
                    "QUARANTINED" if result.status == STATUS_QUARANTINED_AMBIGUOUS else "REJECTED"
                )
                details = {"reasons": list(result.reasons), "warnings": list(result.warnings)}
            cursor.execute(
                "INSERT INTO registry_imports "
                "(source_path, source_sha256, importer_version, imported_at_utc, status, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    discovered.run_config_path,
                    manifest_hash,
                    IMPORTER_VERSION,
                    format_utc_timestamp(utc_now()),
                    audit_status,
                    json.dumps(details, sort_keys=True, ensure_ascii=False),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise RegistryError(
            f"integrity violation while importing {discovered.fold_dir}: {error}"
        ) from error
    return {
        "status": "IMPORTED",
        "fold_dir": discovered.fold_dir,
        "attempts": len({evaluation.attempt_id for evaluation in result.evaluations})
        if result.status == STATUS_QUALIFIED
        else 0,
        "evaluations": len(result.evaluations),
    }


def rebuild_registry(
    scan_root: str | Path, db_path: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    runs = discover_runs(scan_root)
    results = [qualify_run(run) for run in runs]
    summary: dict[str, Any] = {
        "scan_root": str(scan_root),
        "db_path": str(db_path),
        "discovered_runs": len(runs),
        "qualified_runs": sum(1 for result in results if result.status == STATUS_QUALIFIED),
        "quarantined_runs": sum(
            1 for result in results if result.status == STATUS_QUARANTINED_AMBIGUOUS
        ),
        "rejected_runs": sum(1 for result in results if result.status == STATUS_REJECTED),
        "qualified_evaluations": sum(len(result.evaluations) for result in results),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    os.close(fd)
    os.unlink(tmp_name)
    try:
        connection = connect(tmp_name)
        initialize(connection)
        imported = 0
        skipped = 0
        modern_rejected = 0
        for run, result in zip(runs, results):
            outcome = import_run(connection, run, result)
            if outcome["status"] == "SKIPPED_DUPLICATE":
                skipped += 1
            elif outcome["status"] == "REJECTED":
                modern_rejected += 1
            else:
                imported += 1
        _backfill_pending_supersedes(connection)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        connection.close()
        if violations:
            raise RegistryError(f"foreign key violations in rebuilt registry: {violations}")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    summary["imported_runs"] = imported
    summary["skipped_duplicates"] = skipped
    summary["modern_rejected_runs"] = modern_rejected
    return summary


def list_runs(
    connection: sqlite3.Connection,
    *,
    dataset: str | None = None,
    status: str | None = None,
) -> list[sqlite3.Row]:
    rows = connection.execute(
        "SELECT a.attempt_id, lr.logical_run_name, lr.dataset, lr.modality, "
        "f.fold, a.current_state, e.backend, e.evaluation_view, e.aggregation, "
        "e.metric_namespace, m.metric_value AS headline_positive_f1 "
        "FROM run_attempts a "
        "JOIN logical_runs lr ON lr.logical_run_id = a.logical_run_id "
        "JOIN folds f ON f.attempt_id = a.attempt_id "
        "JOIN evaluations e ON e.fold_id = f.fold_id "
        "JOIN metrics m ON m.evaluation_id = e.evaluation_id "
        "AND m.metric_name = 'positive_f1' "
        "WHERE (? IS NULL OR lr.dataset = ?) AND (? IS NULL OR a.current_state = ?) "
        "ORDER BY lr.logical_run_name, f.fold, e.evaluation_view, e.aggregation, e.backend",
        (dataset, dataset, status, status),
    ).fetchall()
    return rows


def show_attempt(
    connection: sqlite3.Connection, attempt_id: str, fold: int | None = None
) -> dict[str, Any]:
    attempt = connection.execute(
        "SELECT * FROM run_attempts WHERE attempt_id = ?", (attempt_id,)
    ).fetchone()
    if attempt is None:
        raise RegistryError(f"unknown attempt: {attempt_id}")
    logical = connection.execute(
        "SELECT * FROM logical_runs WHERE logical_run_id = ?", (attempt["logical_run_id"],)
    ).fetchone()
    fold_rows = connection.execute(
        "SELECT * FROM folds WHERE attempt_id = ? AND (? IS NULL OR fold = ?)",
        (attempt_id, fold, fold),
    ).fetchall()
    evaluations: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for fold_row in fold_rows:
        for row in connection.execute(
            "SELECT * FROM evaluations WHERE fold_id = ?", (fold_row["fold_id"],)
        ).fetchall():
            metrics = connection.execute(
                "SELECT metric_name, metric_value, support FROM metrics WHERE evaluation_id = ?",
                (row["evaluation_id"],),
            ).fetchall()
            evaluations.append({"evaluation": dict(row), "metrics": [dict(m) for m in metrics]})
        artifacts.extend(
            dict(row) for row in connection.execute(
                "SELECT * FROM artifacts WHERE fold_id = ?", (fold_row["fold_id"],)
            ).fetchall()
        )
        jobs.extend(
            dict(row) for row in connection.execute(
                "SELECT * FROM job_events WHERE fold_id = ?", (fold_row["fold_id"],)
            ).fetchall()
        )
    return {
        "attempt": dict(attempt),
        "logical_run": dict(logical) if logical else None,
        "folds": [dict(row) for row in fold_rows],
        "evaluations": evaluations,
        "artifacts": artifacts,
        "jobs": jobs,
    }


def provenance_of_metric(connection: sqlite3.Connection, metric_id: int) -> dict[str, Any]:
    metric = connection.execute(
        "SELECT * FROM metrics WHERE metric_id = ?", (metric_id,)
    ).fetchone()
    if metric is None:
        raise RegistryError(f"unknown metric id: {metric_id}")
    evaluation = connection.execute(
        "SELECT * FROM evaluations WHERE evaluation_id = ?", (metric["evaluation_id"],)
    ).fetchone()
    fold = connection.execute(
        "SELECT * FROM folds WHERE fold_id = ?", (evaluation["fold_id"],)
    ).fetchone()
    attempt = connection.execute(
        "SELECT * FROM run_attempts WHERE attempt_id = ?", (fold["attempt_id"],)
    ).fetchone()
    logical = connection.execute(
        "SELECT * FROM logical_runs WHERE logical_run_id = ?", (attempt["logical_run_id"],)
    ).fetchone()
    artifacts = connection.execute(
        "SELECT * FROM artifacts WHERE fold_id = ?", (fold["fold_id"],)
    ).fetchall()
    provenance = connection.execute(
        "SELECT key, value_json FROM provenance WHERE attempt_id = ? AND fold_id = ?",
        (fold["attempt_id"], fold["fold_id"]),
    ).fetchall()
    return {
        "metric": dict(metric),
        "evaluation": dict(evaluation),
        "fold": dict(fold),
        "attempt": dict(attempt),
        "logical_run": dict(logical),
        "artifacts": [dict(row) for row in artifacts],
        "provenance": [dict(row) for row in provenance],
    }


def list_jobs(
    connection: sqlite3.Connection, *, failed_only: bool = False
) -> list[sqlite3.Row]:
    query = (
        "SELECT je.*, a.attempt_id, lr.logical_run_name, f.fold "
        "FROM job_events je "
        "JOIN folds f ON f.fold_id = je.fold_id "
        "JOIN run_attempts a ON a.attempt_id = f.attempt_id "
        "JOIN logical_runs lr ON lr.logical_run_id = a.logical_run_id "
    )
    if failed_only:
        placeholders = ",".join("?" for _ in _FAILED_JOB_STATUSES)
        query += f"WHERE je.status IN ({placeholders}) "
        query += "ORDER BY je.at_utc"
        return connection.execute(query, _FAILED_JOB_STATUSES).fetchall()
    query += "ORDER BY je.at_utc"
    return connection.execute(query).fetchall()


def best_runs(
    connection: sqlite3.Connection,
    *,
    dataset: str,
    metric: str,
    namespace: str,
    backend: str,
    view: str,
    aggregation: str,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    query = (
        "SELECT a.attempt_id, lr.logical_run_name, f.fold, m.metric_id, m.metric_value, "
        "m.support, e.backend, e.evaluation_view, e.aggregation, m.namespace, "
        "e.split_name, e.checkpoint_role, m.evidence_artifact_id "
        "FROM metrics m "
        "JOIN evaluations e ON e.evaluation_id = m.evaluation_id "
        "JOIN folds f ON f.fold_id = e.fold_id "
        "JOIN run_attempts a ON a.attempt_id = f.attempt_id "
        "JOIN logical_runs lr ON lr.logical_run_id = a.logical_run_id "
        "WHERE lr.dataset = ? AND m.metric_name = ? AND m.namespace = ? "
        "AND e.backend = ? AND e.evaluation_view = ? AND e.aggregation = ? "
        "ORDER BY m.metric_value DESC"
    )
    rows = connection.execute(
        query, (dataset, metric, namespace, backend, view, aggregation)
    ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    return rows
