from __future__ import annotations

import json
from pathlib import Path

from src.experiment_tracking import registry, reporting
from src.experiment_tracking.qualification import qualify_run

from test_experiment_registry import _import_tree
from test_experiment_tracking_discovery import build_standard_run


def _attempt_and_db(tmp_path: Path) -> tuple[Path, str]:
    connection, db_path = _import_tree(tmp_path)
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    return db_path, attempt_id


def test_run_report_is_byte_identical_across_runs(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        first = json.dumps(
            reporting.build_run_report(connection, attempt_id, fold=0),
            indent=2,
            sort_keys=True,
        )
        second = json.dumps(
            reporting.build_run_report(connection, attempt_id, fold=0),
            indent=2,
            sort_keys=True,
        )
    finally:
        connection.close()
    assert first == second


def test_run_report_timestamp_is_opt_in(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        plain = reporting.build_run_report(connection, attempt_id)
        stamped = reporting.build_run_report(connection, attempt_id, generated_at_utc="2026-08-07T08:35:22.000000Z")
    finally:
        connection.close()
    assert plain["generated_at_utc"] is None
    assert stamped["generated_at_utc"] == "2026-08-07T08:35:22.000000Z"
    assert plain["evaluations"] == stamped["evaluations"]


def test_run_report_markdown_shows_evaluation_id_and_evidence_path(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        payload = reporting.build_run_report(connection, attempt_id, fold=0)
    finally:
        connection.close()
    markdown = reporting.render_run_report_markdown(payload)
    evaluation = payload["evaluations"][0]
    assert evaluation["evaluation_id"] in markdown
    assert f"`{evaluation['metrics_artifact_path']}`" in markdown
    assert "headline/binary_strict" in markdown
    assert "original_teacher_forced" in markdown
    assert "full_coverage_k4" in markdown
    assert "subject_level" in markdown
    assert "Researcher interpretation" in markdown
    assert "generated_at" not in markdown


def test_run_report_marks_mn5_only_evidence_explicitly(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        connection.execute(
            "UPDATE artifacts SET exists_locally = 0, exists_on_mn5 = 1 WHERE artifact_type = 'metrics'"
        )
        connection.commit()
        payload = reporting.build_run_report(connection, attempt_id)
    finally:
        connection.close()
    assert payload["mn5_only"]
    markdown = reporting.render_run_report_markdown(payload)
    assert "MN5-only, not locally verifiable" in markdown


def test_run_report_shows_failed_and_resubmitted_jobs(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        fold_id = connection.execute("SELECT fold_id FROM folds LIMIT 1").fetchone()[0]
        connection.execute(
            "INSERT INTO job_events (event_id, fold_id, job_key, job_type, event_type, "
            "slurm_job_id, dependency_job_ids_json, status, at_utc, reason, resubmission_of_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "10000000-0000-4000-8000-000000000001",
                fold_id,
                "train",
                "train",
                "FAILED",
                "1843920",
                "[]",
                "FAILED",
                "2026-08-07T08:35:22.000000Z",
                "node failure",
                None,
            ),
        )
        connection.execute(
            "INSERT INTO job_events (event_id, fold_id, job_key, job_type, event_type, "
            "slurm_job_id, dependency_job_ids_json, status, at_utc, reason, resubmission_of_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "10000000-0000-4000-8000-000000000002",
                fold_id,
                "train",
                "train",
                "SUBMITTED",
                "1843921",
                "[]",
                "COMPLETED",
                "2026-08-07T09:35:22.000000Z",
                None,
                "10000000-0000-4000-8000-000000000001",
            ),
        )
        connection.commit()
        payload = reporting.build_run_report(connection, attempt_id)
    finally:
        connection.close()
    assert any(job["slurm_job_id"] == "1843920" for job in payload["failed_jobs"])
    assert payload["resubmitted_jobs"]
    markdown = reporting.render_run_report_markdown(payload)
    assert "node failure" in markdown
    assert "resubmission of" in markdown


def test_run_report_conclusion_blank_unless_supplied(tmp_path: Path) -> None:
    db_path, attempt_id = _attempt_and_db(tmp_path)
    connection = registry.connect(db_path)
    try:
        blank = reporting.build_run_report(connection, attempt_id)
        authored = reporting.build_run_report(connection, attempt_id, conclusion="evidence supports claim")
    finally:
        connection.close()
    assert blank["conclusion"] is None
    assert authored["conclusion"] == "evidence supports claim"


def test_group_report_labels_pooled_and_fold_mean_separately(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    fold_dir = tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
    summary = {
        "active_backend": "original_teacher_forced",
        "active_backend_pooled_metrics": {"positive_f1": 0.71},
        "active_backend_metric_summary": {"positive_f1": {"mean": 0.70, "std": 0.05}},
        "active_backend_summary_row": {
            "folds": 1,
            "active_backend": "original_teacher_forced",
            "pooled_support_negative": 33,
            "pooled_support_positive": 14,
        },
    }
    (fold_dir / "final_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[0]
    registry.import_run(connection, run, qualify_run(run))
    connection.close()
    connection = registry.connect(db_path)
    try:
        attempt_ids = [row["attempt_id"] for row in connection.execute("SELECT attempt_id FROM run_attempts")]
        payload = reporting.build_group_report(
            connection,
            attempt_ids,
            metric_name="positive_f1",
            namespace="headline/binary_strict",
            backend="original_teacher_forced",
            view="full_coverage_k4",
            aggregation="subject_level",
        )
    finally:
        connection.close()
    markdown = reporting.render_group_report_markdown(payload)
    assert "pooled" in markdown.lower()
    assert "fold_mean" in markdown
    assert payload["compatibility"]["ok"]


def test_group_report_refuses_incompatible_aggregation(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    fold_dir = tmp_path / "text_only" / "cmdc" / "cmdc_run" / "fold_0"
    fold_dir.mkdir(parents=True)
    from test_experiment_tracking_discovery import write_run_config, write_standalone_eval

    write_run_config(fold_dir, dataset="cmdc")
    write_standalone_eval(fold_dir)
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[1]
    registry.import_run(connection, run, qualify_run(run))
    connection.close()
    connection = registry.connect(db_path)
    try:
        attempt_ids = [row["attempt_id"] for row in connection.execute("SELECT attempt_id FROM run_attempts")]
        payload = reporting.build_group_report(
            connection,
            attempt_ids,
            metric_name="positive_f1",
            namespace="headline/binary_strict",
            backend="original_teacher_forced",
            view="full_coverage_k4",
            aggregation="subject_level",
        )
    finally:
        connection.close()
    assert not payload["compatibility"]["ok"]
    assert any("incompatible dataset" in issue for issue in payload["compatibility"]["issues"])
    markdown = reporting.render_group_report_markdown(payload)
    assert "incompatible dataset" in markdown


def test_group_report_paired_deltas_and_matrix(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    seed_a = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    fold_dir = tmp_path / "audio_text" / "daic" / "second_run" / "fold_0"
    fold_dir.mkdir(parents=True)
    from test_experiment_tracking_discovery import write_run_config, write_standalone_eval

    write_run_config(fold_dir)
    write_standalone_eval(fold_dir)
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[1]
    registry.import_run(connection, run, qualify_run(run))
    seed_b = connection.execute("SELECT attempt_id FROM run_attempts ORDER BY attempt_id").fetchall()[-1][0]
    connection.close()
    connection = registry.connect(db_path)
    try:
        payload = reporting.build_group_report(
            connection,
            [seed_a, seed_b],
            metric_name="positive_f1",
            namespace="headline/binary_strict",
            backend="original_teacher_forced",
            view="full_coverage_k4",
            aggregation="subject_level",
            compare_a=[seed_a],
            compare_b=[seed_b],
            expected_seeds=[1337],
            expected_folds=[0],
        )
    finally:
        connection.close()
    assert payload["completed_matrix"] == [(1337, 0)]
    assert payload["missing_matrix"] == []
    assert payload["paired_deltas"] is not None
    assert payload["aggregate"]["n"] == 2
    markdown = reporting.render_group_report_markdown(payload)
    assert "Paired deltas" in markdown
