from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.experiment_tracking import canonical, schemas
from src.experiment_tracking.constants import (
    SCHEMA_VERSION_JOB_EVENT,
    SCHEMA_VERSION_METADATA,
    SCHEMA_VERSION_STATUS,
)

_MIGRATION = Path(__file__).parents[1] / "src" / "experiment_tracking" / "migrations" / "001_initial.sql"

_ATTEMPT_ID = "20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b"


def test_canonical_hash_is_dictionary_order_independent() -> None:
    first = {"dataset": "daic", "seed": 1337, "fold": 0}
    second = {"fold": 0, "seed": 1337, "dataset": "daic"}
    assert canonical.canonical_sha256(first) == canonical.canonical_sha256(second)


def test_canonical_json_uses_compact_sorted_serialization() -> None:
    text = canonical.canonical_json({"b": 1, "a": 2})
    assert text == '{"a":2,"b":1}'


def test_canonical_hash_rejects_nan_and_infinity() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical.canonical_json({"value": value})
    with pytest.raises(ValueError):
        canonical.canonical_sha256({"nested": {"value": float("nan")}})


def test_utc_timestamp_format_is_exact() -> None:
    value = datetime(2026, 8, 7, 8, 35, 22, tzinfo=timezone.utc)
    assert canonical.format_utc_timestamp(value) == "2026-08-07T08:35:22.000000Z"
    value = datetime(2026, 8, 7, 8, 35, 22, 123456, tzinfo=timezone.utc)
    assert canonical.format_utc_timestamp(value) == "2026-08-07T08:35:22.123456Z"


def test_utc_timestamp_round_trip_and_awareness() -> None:
    parsed = canonical.parse_utc_timestamp("2026-08-07T08:35:22.123456Z")
    assert parsed.tzinfo == timezone.utc
    assert parsed.year == 2026
    assert parsed.microsecond == 123456
    with pytest.raises(ValueError):
        canonical.format_utc_timestamp(datetime(2026, 8, 7, 8, 35, 22))


def test_utc_timestamp_parse_rejects_bad_formats() -> None:
    bad_values = (
        "2026-08-07T08:35:22Z",
        "2026-08-07 08:35:22.123456Z",
        "2026-08-07T08:35:22.123456+00:00",
        "garbage",
    )
    for bad in bad_values:
        with pytest.raises(ValueError):
            canonical.parse_utc_timestamp(bad)


def test_sha256_file_streams_file_bytes() -> None:
    payload = b"x" * (5 * 1024 * 1024 + 13)
    with tempfile.NamedTemporaryFile() as handle:
        handle.write(payload)
        handle.flush()
        assert canonical.sha256_file(handle.name) == hashlib.sha256(payload).hexdigest()


def test_normalize_relative_path_rules() -> None:
    assert canonical.normalize_relative_path("fold_0/run_config.yaml") == "fold_0/run_config.yaml"
    assert canonical.normalize_relative_path("./best_model/standalone_eval") == "best_model/standalone_eval"
    assert canonical.normalize_relative_path("best_model") == "best_model"
    with pytest.raises(ValueError):
        canonical.normalize_relative_path("/absolute/path")
    with pytest.raises(ValueError):
        canonical.normalize_relative_path("a/../b")
    with pytest.raises(ValueError):
        canonical.normalize_relative_path("")


def test_write_json_atomic_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "status.json"
    canonical.write_json_atomic(target, {"state": "PLANNED"})
    assert target.exists()
    assert canonical.read_json(target) == {"state": "PLANNED"}
    assert list(tmp_path.iterdir()) == [target]


def test_append_jsonl_atomic_appends_records(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    canonical.append_jsonl_atomic(path, {"event_type": "SUBMITTED"})
    canonical.append_jsonl_atomic(path, {"event_type": "STARTED"})
    assert [row["event_type"] for row in canonical.read_jsonl(path)] == ["SUBMITTED", "STARTED"]


def test_schema_validators_return_explicit_field_errors() -> None:
    ok, errors = schemas.validate_status(
        {
            "schema_version": SCHEMA_VERSION_STATUS,
            "attempt_id": _ATTEMPT_ID,
            "fold": 0,
            "state": "NOT_A_STATE",
            "updated_at_utc": "2026-08-07T08:35:22.000000Z",
            "history": [],
        }
    )
    assert not ok
    assert any("state" in error for error in errors)

    ok, errors = schemas.validate_metadata({"schema_version": SCHEMA_VERSION_METADATA})
    assert not ok
    assert any("logical_run_name" in error for error in errors)

    ok, errors = schemas.validate_job_event(
        {
            "schema_version": SCHEMA_VERSION_JOB_EVENT,
            "event_id": "not-a-uuid",
            "attempt_id": _ATTEMPT_ID,
            "fold": 0,
            "job_key": "train",
            "job_type": "train",
            "event_type": "SUBMITTED",
            "dependency_job_ids": [],
            "at_utc": "2026-08-07T08:35:22.000000Z",
        }
    )
    assert not ok
    assert any("event_id" in error for error in errors)

    ok, errors = schemas.validate_record("audiollm.does_not_exist.v9", {})
    assert not ok
    assert any("unknown schema version" in error for error in errors)


def test_valid_metadata_validates_clean() -> None:
    record = {
        "schema_version": SCHEMA_VERSION_METADATA,
        "group_id": None,
        "logical_run_name": "daic_rotary_k4_seed1337",
        "attempt_id": _ATTEMPT_ID,
        "fold": 0,
        "seed": 1337,
        "created_at_utc": "2026-08-07T08:35:22.000000Z",
        "source": {
            "git_commit": "1c2344f1d33e301978549748c5bf936319a43db6",
            "git_branch": "exp/86-daic-rotary-k",
            "git_dirty": False,
            "deployed_source_sha256": "d" * 64,
        },
        "research": {"github_issue": 86, "github_pr": 91},
        "hashes": {
            "resolved_config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "split_sha256": "c" * 64,
        },
        "paths": {
            "run_config": "run_config.yaml",
            "best_model": "best_model",
            "local_evidence_root": None,
        },
        "wandb": {
            "project": "audiollm-depression",
            "entity": None,
            "run_id": _ATTEMPT_ID + "-fold0",
            "url": None,
            "sync_status": "NOT_EXPORTED",
        },
    }
    ok, errors = schemas.validate_metadata(record)
    assert ok, errors


def test_initial_migration_applies_to_empty_db_with_foreign_keys(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "experiments.sqlite")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_MIGRATION.read_text(encoding="utf-8"))
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    expected = {
        "schema_meta",
        "experiment_groups",
        "logical_runs",
        "run_attempts",
        "folds",
        "job_events",
        "artifacts",
        "evaluations",
        "metrics",
        "provenance",
        "registry_imports",
    }
    assert expected <= tables
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO folds (attempt_id, fold, run_dir) VALUES ('missing-attempt', 0, '/tmp/x')"
        )
    connection.close()
