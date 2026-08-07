from __future__ import annotations

import json
from pathlib import Path

from src.experiment_tracking import registry, wandb_export
from src.experiment_tracking.qualification import qualify_run

from test_experiment_registry import _import_tree
from test_experiment_tracking_discovery import (
    build_standard_run,
    metrics_content,
    write_standalone_eval,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_run(self, **kwargs) -> None:
        self.calls.append(("create_run", kwargs))

    def log_curves(self, **kwargs) -> None:
        self.calls.append(("log_curves", kwargs))

    def log_summary(self, **kwargs) -> None:
        self.calls.append(("log_summary", kwargs))

    def set_status(self, **kwargs) -> None:
        self.calls.append(("set_status", kwargs))


def _plan_from_tree(tmp_path: Path) -> tuple[Path, dict]:
    connection, db_path = _import_tree(tmp_path)
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    connection = registry.connect(db_path)
    try:
        plan = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    return db_path, plan


def _plan_from_attempt(db_path: Path) -> dict:
    connection = registry.connect(db_path)
    try:
        attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
        return wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()


def test_export_plan_summary_metrics_use_qualified_names(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    names = set(plan["summary_metrics"])
    assert "test/full_coverage_k4/original_teacher_forced/subject_level/headline/binary_strict/positive_f1" in names
    assert "selection" not in " ".join(names)
    assert plan["status"] == "complete"


def test_export_plan_curves_from_training_history(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    fold_dir = tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
    history = [
        {"epoch": 0, "train_loss": 0.5, "selection_positive_f1": 0.4},
        {"epoch": 1, "train_loss": 0.3, "selection_positive_f1": 0.6},
    ]
    (fold_dir / "logs" / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[0]
    registry.import_run(connection, run, qualify_run(run))
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    connection = registry.connect(db_path)
    try:
        plan = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    assert plan["epoch_curves"]["train/train_loss"][0]["value"] == 0.5
    assert plan["epoch_curves"]["selection/selection_positive_f1"][1]["value"] == 0.6
    assert plan["status"] == "complete"


def test_safe_filtering_is_recursive_and_drops_sensitive_material(tmp_path: Path) -> None:
    payload = {
        "config": {
            "prompt": {"system": "you are a psychologist", "user_template": "audio here"},
            "data": {"use_audio": True, "transcript_max_chars": 4000},
            "dataset_root": "/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/DAIC-WOZ",
            "model": {"path": "/media/emre/Backup/AudioLLM/models/Qwen2-7B"},
            "subject_id": "302",
            "labels": {"Depressed": 1},
            "nested": {"list": ["/absolute/path", "safe", {"api_key": "sk-123"}]},
        }
    }
    filtered, exclusions = wandb_export.filter_safe(payload)
    serialized = json.dumps(filtered)
    for forbidden in (
        "prompt",
        "use_audio",
        "transcript_max_chars",
        "subject_id",
        "api_key",
        "gpfs",
        "/media/emre",
        "/absolute/path",
        "302",
    ):
        assert forbidden not in serialized, forbidden
    assert filtered["config"]["labels"]["Depressed"] == 1
    assert filtered["config"]["nested"]["list"] == ["safe"]
    assert exclusions
    assert any("dataset_root" in item for item in exclusions)
    assert any("api_key" in item for item in exclusions)


def test_export_plan_config_is_safe_filtered(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    serialized = json.dumps(plan)
    for forbidden in ("/gpfs", "prompt", "transcript", "subject_id"):
        assert forbidden not in serialized
    assert plan["safe_config"]["dataset"] == "daic"
    assert plan["safe_config"]["seed"] == 1337


def test_export_config_carries_tracking_identity(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    adapter = FakeAdapter()
    wandb_export.execute_export(plan, adapter, mode="offline")
    config = adapter.calls[0][1]["config"]
    assert config["tracking/attempt_id"] == plan["identity"]["attempt_id"]
    assert config["tracking/logical_run_name"] == "daic_run"
    assert config["tracking/fold"] == 0
    assert config["tracking/evaluation_id"] == plan["identity"]["evaluation_id"]
    assert plan["name"].endswith("-fold0")
    assert adapter.calls[0][1]["name"] == plan["name"]


def test_legacy_wandb_ids_are_deterministic_and_versioned() -> None:
    first = wandb_export.legacy_wandb_id("legacy-legacy-attempt-v1-abc", 0, "eval-xyz")
    second = wandb_export.legacy_wandb_id("legacy-legacy-attempt-v1-abc", 0, "eval-xyz")
    assert first == second
    assert first.startswith("wandb-")
    assert first != wandb_export.legacy_wandb_id("legacy-legacy-attempt-v1-abc", 1, "eval-xyz")
    assert first != wandb_export.legacy_wandb_id("legacy-legacy-attempt-v1-abc", 0, "eval-other")


def test_incomplete_history_marks_status_incomplete(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    connection = registry.connect(db_path)
    try:
        complete = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    assert complete["status"] == "complete"
    assert "incomplete" not in complete["tags"]
    fold_dir = tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
    (fold_dir / "logs" / "training_history.json").unlink()
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[0]
    connection = registry.connect(db_path)
    try:
        registry.import_run(connection, run, qualify_run(run))
        plan = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    assert plan["status"] == "incomplete"
    assert "incomplete" in plan["tags"]
    assert "training history missing or unreadable" in plan["incomplete_reasons"]
    assert plan["summary_metrics"]


def test_missing_history_still_allows_safe_final_scalars(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    (fold_dir / "logs" / "training_history.json").unlink()
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[0]
    plan = wandb_export.build_export_plan_from_result(run, qualify_run(run))
    assert plan is not None
    assert plan["epoch_curves"] == {}
    assert plan["summary_metrics"]


def test_ambiguous_evaluation_exports_no_headline_metric(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics.json"],
        contents=[metrics_content(view="full_coverage_k4")],
        location="eval/best_checkpoint",
    )
    run = __import__("src.experiment_tracking.discovery", fromlist=["discover_runs"]).discover_runs(tmp_path)[0]
    plan = wandb_export.build_export_plan_from_result(run, qualify_run(run))
    assert plan is None


def test_rerun_produces_identical_plan_and_idempotent_execution(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    attempt_id = connection.execute("SELECT attempt_id FROM run_attempts").fetchone()[0]
    connection.close()
    connection = registry.connect(db_path)
    try:
        first = wandb_export.build_export_plan(connection, attempt_id)
        second = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    adapter = FakeAdapter()
    outcome = wandb_export.execute_export(first, adapter, mode="offline")
    assert outcome["run_id"] == first["run_id"]
    assert adapter.calls[0][0] == "create_run"
    assert adapter.calls[0][1]["run_id"] == first["run_id"]
    assert any(call[0] == "log_summary" for call in adapter.calls)
    assert any(call[0] == "set_status" for call in adapter.calls)
    adapter2 = FakeAdapter()
    wandb_export.execute_export(second, adapter2, mode="offline")
    assert [call[0] for call in adapter2.calls] == [call[0] for call in adapter.calls]


def test_exporter_never_touches_lifecycle_status(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"state": "COMPLETED_ON_MN5"}), encoding="utf-8")
    before = status_path.read_text(encoding="utf-8")
    wandb_export.execute_export(plan, FakeAdapter(), mode="offline")
    assert status_path.read_text(encoding="utf-8") == before


def test_execute_export_requires_adapter_for_real_mode(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    outcome = wandb_export.execute_export(plan, None, mode="dry_run")
    assert outcome["mode"] == "dry_run"
    import pytest

    with pytest.raises(ValueError, match="WandbAdapter"):
        wandb_export.execute_export(plan, None, mode="offline")


def test_backfill_report_written_only_with_explicit_output(tmp_path: Path) -> None:
    _, plan = _plan_from_tree(tmp_path)
    assert plan["schema_version"] == "audiollm.wandb_export.v1"
