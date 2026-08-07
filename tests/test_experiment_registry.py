from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.experiment_tracking import registry
from src.experiment_tracking.discovery import discover_runs
from src.experiment_tracking.qualification import (
    STATUS_QUARANTINED_AMBIGUOUS,
    QualificationResult,
    qualify_run,
)

from test_experiment_tracking_discovery import (
    build_standard_run,
    metrics_content,
    write_standalone_eval,
)


def _import_tree(tmp_path: Path) -> tuple[sqlite3.Connection, Path]:
    build_standard_run(tmp_path, run_name="daic_run")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    runs = discover_runs(tmp_path)
    outcomes = []
    for run in runs:
        outcomes.append(registry.import_run(connection, run, qualify_run(run)))
    return connection, db_path


def _counts(connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "logical_runs",
            "run_attempts",
            "folds",
            "artifacts",
            "evaluations",
            "metrics",
            "provenance",
            "registry_imports",
        )
    }


def test_two_identical_imports_create_no_duplicate_rows(tmp_path: Path) -> None:
    connection, _ = _import_tree(tmp_path)
    before = _counts(connection)
    runs = discover_runs(tmp_path)
    outcome = registry.import_run(connection, runs[0], qualify_run(runs[0]))
    assert outcome["status"] == "SKIPPED_DUPLICATE"
    assert _counts(connection) == before
    connection.close()


def test_changed_evidence_creates_new_identity_and_never_replaces_old_metric(tmp_path: Path) -> None:
    connection, _ = _import_tree(tmp_path)
    run = discover_runs(tmp_path)[0]
    old_evaluation_id = qualify_run(run).evaluations[0].evaluation_id
    old_metric_row = connection.execute(
        "SELECT metric_value FROM metrics WHERE metric_name = 'positive_f1'"
    ).fetchone()
    old_value = old_metric_row["metric_value"]
    metrics_path = (
        Path(run.fold_dir)
        / "best_model"
        / "standalone_eval"
        / "metrics_original_teacher_forced_full_coverage_k4.json"
    )
    changed = metrics_content(view="full_coverage_k4")
    changed["binary_strict_positive_f1"] = 0.999
    metrics_path.write_text(json.dumps(changed), encoding="utf-8")
    rediscovered = discover_runs(tmp_path)[0]
    result = qualify_run(rediscovered)
    outcome = registry.import_run(connection, rediscovered, result)
    assert outcome["status"] == "IMPORTED"
    new_evaluation_id = result.evaluations[0].evaluation_id
    assert new_evaluation_id != old_evaluation_id
    rows = connection.execute(
        "SELECT evaluation_id, metric_value FROM metrics WHERE metric_name = 'positive_f1'"
    ).fetchall()
    assert {row["evaluation_id"]: row["metric_value"] for row in rows}[old_evaluation_id] == old_value
    assert connection.execute(
        "SELECT COUNT(*) FROM run_attempts"
    ).fetchone()[0] == 2
    connection.close()


def test_failed_import_rolls_back_fully(tmp_path: Path, monkeypatch) -> None:
    connection, _ = _import_tree(tmp_path)
    build_standard_run(tmp_path, run_name="second_run")
    run = discover_runs(tmp_path)[1]
    result = qualify_run(run)

    def boom(cursor, discovered, result, fold_ids, artifact_ids):
        raise registry.RegistryError("injected mid-import failure")

    monkeypatch.setattr(registry, "_import_evaluations", boom)
    with pytest.raises(registry.RegistryError, match="injected mid-import failure"):
        registry.import_run(connection, run, result)
    assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM registry_imports").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM logical_runs").fetchone()[0] == 1
    connection.close()


def test_ambiguous_evidence_is_stored_as_quarantined_with_no_headline_metric(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path, run_name="daic_run")
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics.json"],
        contents=[metrics_content(view="full_coverage_k4")],
        location="eval/best_checkpoint",
    )
    db_path = tmp_path / "registry.sqlite"
    connection = registry.ensure_registry(db_path)
    run = discover_runs(tmp_path)[0]
    result = qualify_run(run)
    assert result.status == STATUS_QUARANTINED_AMBIGUOUS
    outcome = registry.import_run(connection, run, result)
    assert outcome["status"] == "IMPORTED"
    audit = connection.execute(
        "SELECT status, details_json FROM registry_imports"
    ).fetchone()
    assert audit["status"] == "QUARANTINED"
    assert "multiple_eval_locations" in audit["details_json"]
    assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
    connection.close()


def test_database_can_be_deleted_and_rebuilt_from_fixtures(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    first_counts = _counts(connection)
    connection.close()
    db_path.unlink()
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == 1
    assert summary["discovered_runs"] == 1
    assert summary["qualified_runs"] == 1
    rebuilt = registry.connect(db_path)
    try:
        assert _counts(rebuilt) == first_counts
    finally:
        rebuilt.close()


def test_full_rebuild_equals_incremental_import_logically(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="daic_run")
    fold_dir = tmp_path / "text_only" / "cmdc" / "cmdc_run" / "fold_0"
    fold_dir.mkdir(parents=True)
    from test_experiment_tracking_discovery import write_run_config

    write_run_config(fold_dir, dataset="cmdc")
    write_standalone_eval(fold_dir)

    rebuild_db = tmp_path / "rebuild.sqlite"
    incremental_db = tmp_path / "incremental.sqlite"
    registry.rebuild_registry(tmp_path, rebuild_db)

    incremental = registry.ensure_registry(incremental_db)
    for run in discover_runs(tmp_path):
        registry.import_run(incremental, run, qualify_run(run))
    incremental.close()

    rebuilt = registry.connect(rebuild_db)
    incremental = registry.connect(incremental_db)
    try:
        assert _counts(rebuilt) == _counts(incremental)
        rebuilt_metrics = sorted(
            (
                row["evaluation_id"],
                row["metric_name"],
                row["metric_value"],
            )
            for row in rebuilt.execute("SELECT evaluation_id, metric_name, metric_value FROM metrics")
        )
        incremental_metrics = sorted(
            (
                row["evaluation_id"],
                row["metric_name"],
                row["metric_value"],
            )
            for row in incremental.execute("SELECT evaluation_id, metric_name, metric_value FROM metrics")
        )
        assert rebuilt_metrics == incremental_metrics
    finally:
        rebuilt.close()
        incremental.close()


def test_foreign_key_and_uniqueness_violations_are_enforced(tmp_path: Path) -> None:
    connection, _ = _import_tree(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO folds (attempt_id, fold, run_dir) VALUES ('missing-attempt', 9, '/x')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO run_attempts (attempt_id, logical_run_id, schema_version, legacy_import, current_state) "
            "VALUES ('a', 'missing-logical', 'v1', 1, 'PLANNED')"
        )
    connection.close()


def test_rebuild_dry_run_never_creates_the_database(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="daic_run")
    db_path = tmp_path / "never_created.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["discovered_runs"] == 1
    assert summary["qualified_runs"] == 1
    assert not db_path.exists()


def test_rebuild_preserves_existing_db_when_rebuild_fails(tmp_path: Path, monkeypatch) -> None:
    build_standard_run(tmp_path, run_name="daic_run")
    db_path = tmp_path / "existing.sqlite"
    registry.rebuild_registry(tmp_path, db_path)
    original_bytes = db_path.read_bytes()

    def boom(connection, discovered, result, *, dry_run=False):
        raise registry.RegistryError("injected rebuild failure")

    monkeypatch.setattr(registry, "import_run", boom)
    with pytest.raises(registry.RegistryError):
        registry.rebuild_registry(tmp_path, db_path)
    assert db_path.read_bytes() == original_bytes
    leftovers = [
        path.name
        for path in db_path.parent.iterdir()
        if path.name.startswith(db_path.name + ".") and path.name.endswith(".tmp")
    ]
    assert leftovers == []
