from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.experiment_tracking import registry

from test_experiment_tracking_discovery import build_standard_run
from test_experiment_registry import _import_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_tool(tool: str, *args: str, expected_code: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / tool), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == expected_code, result.stderr
    return result


def test_rebuild_dry_run_cli_creates_nothing(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="daic_run")
    db_path = tmp_path / "cli.sqlite"
    result = _run_tool("rebuild_experiment_registry.py", "--scan-root", str(tmp_path), "--db", str(db_path), "--dry-run")
    assert "discovered_runs" in result.stdout
    assert not db_path.exists()


def test_rebuild_and_list_cli_round_trip(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="daic_run")
    db_path = tmp_path / "cli.sqlite"
    _run_tool("rebuild_experiment_registry.py", "--scan-root", str(tmp_path), "--db", str(db_path))
    listed = _run_tool("exp.py", "--db", str(db_path), "list")
    assert "daic_run" in listed.stdout
    assert "original_teacher_forced" in listed.stdout
    assert "full_coverage_k4" in listed.stdout
    assert "headline/binary_strict" in listed.stdout
    assert "subject_level" in listed.stdout


def test_import_cli_dry_run_and_real(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path, run_name="daic_run")
    run_dir = fold_dir.parent
    db_path = tmp_path / "cli.sqlite"
    result = _run_tool("import_experiment.py", "--run-dir", str(run_dir), "--db", str(db_path), "--dry-run")
    assert "DRY_RUN" in result.stdout
    assert not db_path.exists()
    _run_tool("import_experiment.py", "--run-dir", str(run_dir), "--db", str(db_path))
    assert db_path.exists()
    connection = registry.connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
    finally:
        connection.close()


def test_show_cli_prints_attempt_sections(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    result = _run_tool("exp.py", "--db", str(db_path), "show", attempt_id, "--fold", "0")
    assert "attempt" in result.stdout
    assert "evaluations" in result.stdout
    assert "artifacts" in result.stdout
    assert attempt_id in result.stdout


def test_provenance_cli_traces_metric_chain(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    metric_id = connection.execute("SELECT metric_id FROM metrics WHERE metric_name = 'positive_f1'").fetchone()[0]
    connection.close()
    result = _run_tool("exp.py", "--db", str(db_path), "provenance", str(metric_id))
    assert "metric" in result.stdout
    assert "evaluation" in result.stdout
    assert "attempt" in result.stdout
    assert "legacy_identity_input" in result.stdout
    assert "evidence_manifest_sha256" in result.stdout


def test_best_cli_refuses_underqualified_queries(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    connection.close()
    _run_tool(
        "exp.py",
        "--db",
        str(db_path),
        "best",
        "--dataset",
        "daic",
        "--metric",
        "positive_f1",
        expected_code=2,
    )


def test_best_cli_fully_qualified_query(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    connection.close()
    result = _run_tool(
        "exp.py",
        "--db",
        str(db_path),
        "best",
        "--dataset",
        "daic",
        "--metric",
        "positive_f1",
        "--namespace",
        "headline/binary_strict",
        "--backend",
        "original_teacher_forced",
        "--view",
        "full_coverage_k4",
        "--aggregation",
        "subject_level",
    )
    assert "daic_run" in result.stdout
    assert "0.718" in result.stdout


def test_jobs_cli_with_and_without_failed_filter(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    fold_id = connection.execute("SELECT fold_id FROM folds LIMIT 1").fetchone()[0]
    from src.experiment_tracking import lifecycle

    connection.execute(
        "INSERT INTO job_events (event_id, fold_id, job_key, job_type, event_type, "
        "dependency_job_ids_json, status, at_utc, reason, resubmission_of_job_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "00000000-0000-4000-8000-000000000001",
            fold_id,
            "train",
            "train",
            "SUBMITTED",
            "[]",
            "FAILED",
            "2026-08-07T08:35:22.000000Z",
            "node failure",
            None,
        ),
    )
    connection.execute(
        "INSERT INTO job_events (event_id, fold_id, job_key, job_type, event_type, "
        "dependency_job_ids_json, status, at_utc, reason, resubmission_of_job_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "00000000-0000-4000-8000-000000000002",
            fold_id,
            "evaluation",
            "evaluation",
            "SUBMITTED",
            "[]",
            "COMPLETED",
            "2026-08-07T09:35:22.000000Z",
            None,
            None,
        ),
    )
    connection.commit()
    connection.close()
    all_jobs = _run_tool("exp.py", "--db", str(db_path), "jobs")
    assert "train" in all_jobs.stdout
    failed = _run_tool("exp.py", "--db", str(db_path), "jobs", "--failed")
    assert "FAILED" in failed.stdout
    assert "evaluation" not in failed.stdout.split("\n")[1]
