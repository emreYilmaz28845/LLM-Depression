from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from src.evaluate import _record_evaluation_sidecars, parse_args as eval_parse_args
from src.train import (
    _attach_tracking_block,
    _finalize_tracking_artifacts,
    _initialize_tracking_sidecars,
    _load_experiment_context,
    parse_args as train_parse_args,
)
from src.experiment_tracking import schemas
from src.experiment_tracking.constants import (
    SCHEMA_VERSION_ARTIFACTS,
    SCHEMA_VERSION_EVALUATIONS,
    SCHEMA_VERSION_METADATA,
    SCHEMA_VERSION_STATUS,
)
from src.experiment_tracking.lifecycle import read_job_events, read_status

GIT_COMMIT = "1c2344f1d33e301978549748c5bf936319a43db6"


def _context(tmp_path: Path, *, attempt_id: str | None = None, fold: int = 0) -> Path:
    payload = {
        "schema_version": "audiollm.experiment_context.v1",
        "group_id": None,
        "logical_run_name": "daic_rotary_k4_seed1337",
        "attempt_id": attempt_id or "20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b",
        "fold": fold,
        "seed": 1337,
        "source": {"git_commit": GIT_COMMIT, "git_branch": "exp/86-daic-rotary-k", "git_dirty": False},
        "research": {"github_issue": 86, "github_pr": 91},
        "hashes": {
            "manifest_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        "slurm": {"train_job_id": "1843921", "eval_job_ids": ["1843922"]},
    }
    path = tmp_path / "context.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_args(tmp_path: Path, *, context_path: str | None = None, fold: int = 0):
    args = train_parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--fold",
            str(fold),
            "--run_name",
            "daic_rotary_k4_seed1337",
        ]
        + (["--experiment-context", str(context_path)] if context_path else [])
    )
    return args


def test_no_context_preserves_absent_behavior() -> None:
    args = _fake_args(Path("/tmp"))
    assert args.experiment_context is None
    assert _load_experiment_context(args) is None


def test_context_loading_validates_identity(tmp_path: Path) -> None:
    context_path = _context(tmp_path)
    args = _fake_args(tmp_path, context_path=str(context_path), fold=0)
    context = _load_experiment_context(args)
    assert context["attempt_id"].startswith("20260807T113522Z-")


def test_context_fold_mismatch_is_rejected(tmp_path: Path) -> None:
    context_path = _context(tmp_path, fold=0)
    args = _fake_args(tmp_path, context_path=str(context_path), fold=1)
    with pytest.raises(ValueError, match="fold"):
        _load_experiment_context(args)


def test_context_invalid_attempt_id_is_rejected(tmp_path: Path) -> None:
    context_path = _context(tmp_path, attempt_id="not-an-attempt-id")
    args = _fake_args(tmp_path, context_path=str(context_path))
    with pytest.raises(ValueError, match="attempt_id"):
        _load_experiment_context(args)


def test_run_config_gains_tracking_block_without_losing_fields() -> None:
    run_config = {"config": {"dataset": "daic"}, "fold": 0}
    context = {"group_id": None, "logical_run_name": "daic_rotary_k4_seed1337", "attempt_id": "20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b"}
    _attach_tracking_block(run_config, context, 0)
    assert run_config["config"]["dataset"] == "daic"
    assert run_config["tracking"]["attempt_id"] == context["attempt_id"]
    assert run_config["tracking"]["schema_version"] == "audiollm.tracking.v1"


def test_training_initializes_sidecars_on_rank0(tmp_path: Path) -> None:
    context_path = _context(tmp_path)
    run_root = tmp_path / "run" / "fold_0"
    run_root.mkdir(parents=True)
    (run_root / "run_config.yaml").write_text("config:\n  dataset: daic\n", encoding="utf-8")
    args = _fake_args(tmp_path, context_path=str(context_path))
    context = _load_experiment_context(args)
    run_config = {"config": {"dataset": "daic"}, "manifest_hash": "a" * 64, "split_metadata_hash": "b" * 64}
    _initialize_tracking_sidecars(args, context, run_root, run_config)
    metadata = json.loads((run_root / "metadata.json").read_text(encoding="utf-8"))
    ok, errors = schemas.validate_metadata(metadata)
    assert ok, errors
    assert metadata["attempt_id"] == context["attempt_id"]
    assert metadata["hashes"]["manifest_sha256"] == "a" * 64
    status = read_status(run_root / "status.json")
    ok, errors = schemas.validate_status(status)
    assert ok, errors
    assert status["state"] == "RUNNING"
    assert status["history"][-1]["from"] == "SUBMITTED"
    events = read_job_events(run_root / "jobs.jsonl")
    assert events[0]["event_type"] == "STARTED"
    assert events[0]["slurm_job_id"] == "1843921"
    artifacts = json.loads((run_root / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts["schema_version"] == SCHEMA_VERSION_ARTIFACTS
    assert any(artifact["path"] == "run_config.yaml" for artifact in artifacts["artifacts"])


def test_training_finalize_appends_artifacts_and_completed_event(tmp_path: Path) -> None:
    context_path = _context(tmp_path)
    run_root = tmp_path / "run" / "fold_0"
    run_root.mkdir(parents=True)
    logs = run_root / "logs"
    logs.mkdir()
    (logs / "training_history.json").write_text("[]", encoding="utf-8")
    (logs / "split_used.json").write_text("{}", encoding="utf-8")
    (run_root / "best_model").mkdir()
    args = _fake_args(tmp_path, context_path=str(context_path))
    context = _load_experiment_context(args)
    _initialize_tracking_sidecars(args, context, run_root, {"config": {"dataset": "daic"}})
    _finalize_tracking_artifacts(args, context, run_root, {"config": {"dataset": "daic"}})
    artifacts = json.loads((run_root / "artifacts.json").read_text(encoding="utf-8"))
    paths = {artifact["path"] for artifact in artifacts["artifacts"]}
    assert "logs/training_history.json" in paths
    assert "logs/split_used.json" in paths
    assert "best_model" in paths
    events = read_job_events(run_root / "jobs.jsonl")
    assert events[-1]["event_type"] == "COMPLETED"
    assert events[-1]["status"] == "COMPLETED"
    status = read_status(run_root / "status.json")
    assert status["state"] == "RUNNING"


def _eval_context(tmp_path: Path) -> tuple[Path, dict]:
    context_path = _context(tmp_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    return context_path, context


def test_evaluation_records_sidecars_idempotently(tmp_path: Path) -> None:
    context_path, context = _eval_context(tmp_path)
    fold_dir = tmp_path / "run" / "fold_0"
    checkpoint_dir = fold_dir / "best_model"
    output_dir = checkpoint_dir / "standalone_eval"
    output_dir.mkdir(parents=True)
    metrics_payload = {
        "prediction_backend": "original_teacher_forced",
        "aggregation_level": "subject",
        "num_units": 47,
        "backend_results": {
            "original_teacher_forced": {
                "headline_metrics": {
                    "accuracy": 0.76,
                    "precision": 0.56,
                    "recall": 1.0,
                    "positive_f1": 0.718,
                    "macro_f1": 0.759,
                    "weighted_f1": 0.776,
                }
            }
        },
    }
    (output_dir / "metrics_original_teacher_forced.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (output_dir / "predictions_subject_level.csv").write_text("subject_id,prediction\n", encoding="utf-8")
    (output_dir / "confusion_matrix.json").write_text("{}", encoding="utf-8")
    (output_dir / "eval_config.yaml").write_text("sample_prediction_mode: original_teacher_forced\n", encoding="utf-8")
    (output_dir / "final_and_best_validation_metrics.json").write_text("{}", encoding="utf-8")
    config = {"dataset": "daic", "split": {"final_eval_partition": "test"}}
    metrics = {"active_backend": "original_teacher_forced", "backend_results": metrics_payload["backend_results"]}

    args = SimpleNamespace(
        experiment_context=str(context_path),
        checkpoint_dir=str(checkpoint_dir),
        fold=0,
    )
    for _ in range(2):
        _record_evaluation_sidecars(
            args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None
        )
    evaluations = json.loads((fold_dir / "evaluations.json").read_text(encoding="utf-8"))
    assert evaluations["schema_version"] == SCHEMA_VERSION_EVALUATIONS
    assert len(evaluations["evaluations"]) == 1
    ok, errors = schemas.validate_evaluations(evaluations)
    assert ok, errors
    record = evaluations["evaluations"][0]
    assert record["evaluation_id"].startswith("eval-")
    assert record["backend"] == "original_teacher_forced"
    assert record["aggregation"] == "subject_level"
    assert record["metrics_artifact_path"] == "best_model/standalone_eval/metrics_original_teacher_forced.json"
    assert record["predictions_artifact_path"] == "best_model/standalone_eval/predictions_subject_level.csv"
    artifacts = json.loads((fold_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts["schema_version"] == SCHEMA_VERSION_ARTIFACTS
    assert len(artifacts["artifacts"]) == 5


def test_evaluation_different_evidence_creates_new_evaluation_id(tmp_path: Path) -> None:
    context_path, context = _eval_context(tmp_path)
    fold_dir = tmp_path / "run" / "fold_0"
    checkpoint_dir = fold_dir / "best_model"
    output_dir = checkpoint_dir / "standalone_eval"
    output_dir.mkdir(parents=True)
    base = {
        "prediction_backend": "original_teacher_forced",
        "aggregation_level": "subject",
        "num_units": 47,
        "backend_results": {
            "original_teacher_forced": {
                "headline_metrics": {
                    "accuracy": 0.76,
                    "precision": 0.56,
                    "recall": 1.0,
                    "positive_f1": 0.718,
                    "macro_f1": 0.759,
                    "weighted_f1": 0.776,
                }
            }
        },
    }
    (output_dir / "metrics_original_teacher_forced.json").write_text(json.dumps(base), encoding="utf-8")
    (output_dir / "predictions_subject_level.csv").write_text("subject_id,prediction\n", encoding="utf-8")
    config = {"dataset": "daic", "split": {"final_eval_partition": "test"}}
    metrics = {"active_backend": "original_teacher_forced", "backend_results": base["backend_results"]}

    args = SimpleNamespace(
        experiment_context=str(context_path),
        checkpoint_dir=str(checkpoint_dir),
        fold=0,
    )
    _record_evaluation_sidecars(args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None)
    changed = dict(base)
    changed["num_units"] = 46
    (output_dir / "metrics_original_teacher_forced.json").write_text(json.dumps(changed), encoding="utf-8")
    metrics = {"active_backend": "original_teacher_forced", "backend_results": changed["backend_results"]}
    _record_evaluation_sidecars(args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None)
    evaluations = json.loads((fold_dir / "evaluations.json").read_text(encoding="utf-8"))
    assert len(evaluations["evaluations"]) == 2
    assert evaluations["evaluations"][0]["evaluation_id"] != evaluations["evaluations"][1]["evaluation_id"]


def test_evaluation_refuses_overwrite_with_different_content(tmp_path: Path) -> None:
    context_path, context = _eval_context(tmp_path)
    fold_dir = tmp_path / "run" / "fold_0"
    checkpoint_dir = fold_dir / "best_model"
    output_dir = checkpoint_dir / "standalone_eval"
    output_dir.mkdir(parents=True)
    base = {
        "prediction_backend": "original_teacher_forced",
        "aggregation_level": "subject",
        "num_units": 47,
        "backend_results": {
            "original_teacher_forced": {
                "headline_metrics": {
                    "accuracy": 0.76,
                    "precision": 0.56,
                    "recall": 1.0,
                    "positive_f1": 0.718,
                    "macro_f1": 0.759,
                    "weighted_f1": 0.776,
                }
            }
        },
    }
    (output_dir / "metrics_original_teacher_forced.json").write_text(json.dumps(base), encoding="utf-8")
    (output_dir / "predictions_subject_level.csv").write_text("subject_id,prediction\n", encoding="utf-8")
    config = {"dataset": "daic", "split": {"final_eval_partition": "test"}}
    metrics = {"active_backend": "original_teacher_forced", "backend_results": base["backend_results"]}

    args = SimpleNamespace(
        experiment_context=str(context_path),
        checkpoint_dir=str(checkpoint_dir),
        fold=0,
    )
    _record_evaluation_sidecars(args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None)
    evaluations = json.loads((fold_dir / "evaluations.json").read_text(encoding="utf-8"))
    record = evaluations["evaluations"][0]
    record["metrics"][0]["value"] = 0.999
    (fold_dir / "evaluations.json").write_text(json.dumps(evaluations), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _record_evaluation_sidecars(args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None)


def test_evaluation_sidecars_accept_str_output_dir(tmp_path: Path) -> None:
    context_path, context = _eval_context(tmp_path)
    fold_dir = tmp_path / "run" / "fold_0"
    checkpoint_dir = fold_dir / "best_model"
    output_dir = checkpoint_dir / "standalone_eval"
    output_dir.mkdir(parents=True)
    headline = {"positive_f1": 0.6, "macro_f1": 0.5}
    metrics_payload = {
        "prediction_backend": "original_teacher_forced",
        "aggregation_level": "subject",
        "num_units": 6,
        "backend_results": {"original_teacher_forced": {"headline_metrics": headline}},
    }
    (output_dir / "metrics_original_teacher_forced.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (output_dir / "predictions_subject_level.csv").write_text("a\n", encoding="utf-8")
    config = {"dataset": "daic", "split": {"final_eval_partition": "test"}}
    metrics = {"active_backend": "original_teacher_forced", "backend_results": metrics_payload["backend_results"]}
    args = SimpleNamespace(experiment_context=str(context_path), checkpoint_dir=str(checkpoint_dir), fold=0)
    _record_evaluation_sidecars(args, config, metrics, str(output_dir), aggregation_level="subject", split_mode="fixed", cv_protocol=None)
    evaluations = json.loads((fold_dir / "evaluations.json").read_text(encoding="utf-8"))
    assert len(evaluations["evaluations"]) == 1


def test_no_context_evaluation_writes_no_sidecars(tmp_path: Path) -> None:
    fold_dir = tmp_path / "run" / "fold_0"
    checkpoint_dir = fold_dir / "best_model"
    output_dir = checkpoint_dir / "standalone_eval"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics_original_teacher_forced.json").write_text("{}", encoding="utf-8")
    config = {"dataset": "daic", "split": {"final_eval_partition": "test"}}
    metrics = {"active_backend": "original_teacher_forced", "backend_results": {"headline_metrics": {}}}

    args = SimpleNamespace(experiment_context=None, checkpoint_dir=str(checkpoint_dir), fold=0)
    _record_evaluation_sidecars(args, config, metrics, output_dir, aggregation_level="subject", split_mode="fixed", cv_protocol=None)
    assert not (fold_dir / "evaluations.json").exists()
    assert not (fold_dir / "artifacts.json").exists()
