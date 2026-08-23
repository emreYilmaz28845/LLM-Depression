from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.turkish_question_condition import EVALUATION_BACKEND, EVALUATION_VIEW, GROUP_ID, METRIC_NAMESPACE
from src.turkish_question_condition_tracking import (
    HeadTrackingError,
    TRACKING_KIND,
    finish_head_attempt,
    initialize_turkish_head_attempt,
    materialize_turkish_head_evidence,
    record_head_job,
    transition_head_attempt,
    validate_head_attempt,
)


def _context(attempt_id: str) -> dict[str, object]:
    return {
        "group_id": GROUP_ID,
        "tracking_kind": TRACKING_KIND,
        "logical_run_name": "test-head",
        "attempt_id": attempt_id,
        "fold": 0,
        "seed": 1337,
        "source": {
            "git_commit": "e176da5e0595464bc44320d32e04f7fe0a7adf5e",
            "deployment_id": "test-deployment",
            "deployed_source_sha256": "a" * 64,
        },
        "hashes": {"manifest_sha256": "b" * 64, "split_sha256": "c" * 64},
        "qualifiers": {
            "evaluation_view": EVALUATION_VIEW,
            "evaluation_backend": EVALUATION_BACKEND,
            "metric_namespace": METRIC_NAMESPACE,
        },
        "required_jobs": ["head"],
    }


def _config() -> dict[str, object]:
    return {
        "dataset": "turkish",
        "evaluation": {
            "evaluation_view": EVALUATION_VIEW,
            "sample_prediction_mode": EVALUATION_BACKEND,
            "aggregation": "subject_level",
            "split_name": "outer_holdout",
            "split_protocol": "saved_split",
        },
        "classifier": {
            "method": "logreg",
            "prediction_backend": "qwen_hidden_logreg_raw",
        },
    }


def test_head_attempt_lifecycle_and_strict_recomputation(tmp_path: Path) -> None:
    attempt_id = "20260823T000000Z-test-head-e176da5e-abcdef12"
    attempt_dir = tmp_path / attempt_id
    parent = {"parent_attempt_id": "20260823T000000Z-parent-e176da5e-abcdef12", "parent_checkpoint_path": str(tmp_path / "best_model")}
    initialize_turkish_head_attempt(attempt_dir, context=_context(attempt_id), config=_config(), parent=parent)
    record_head_job(attempt_dir, job_key="head", job_type="hidden_extraction", event_type="SUBMITTED", slurm_job_id="1", status="PENDING")
    transition_head_attempt(attempt_dir, "SUBMITTED", reason="test submission")
    transition_head_attempt(attempt_dir, "RUNNING", reason="test start")
    record_head_job(attempt_dir, job_key="head", job_type="hidden_extraction", event_type="STARTED", slurm_job_id="1", status="RUNNING")
    predictions = attempt_dir / "classifier" / "logreg_raw" / "predictions_subject_level.jsonl"
    metrics = attempt_dir / "classifier" / "logreg_raw" / "metrics.json"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        "\n".join(
            json.dumps({"dataset": "turkish", "subject_id": subject, "label": label, "prediction": prediction})
            for subject, label, prediction in (("s1", 1, 1), ("s2", 0, 1))
        )
        + "\n",
        encoding="utf-8",
    )
    metrics.write_text("{}\n", encoding="utf-8")
    result = materialize_turkish_head_evidence(
        attempt_dir,
        predictions_path=predictions,
        metrics_path=metrics,
        checkpoint_path=parent["parent_checkpoint_path"],
    )
    assert result["state"] == "COMPLETED_ON_MN5"
    record_head_job(attempt_dir, job_key="head", job_type="hidden_extraction", event_type="COMPLETED", slurm_job_id="1", status="COMPLETED", exit_code="0:0")
    assert validate_head_attempt(attempt_dir)["ok"]
    assert finish_head_attempt(attempt_dir)["ok"]


def test_head_context_group_is_fail_closed(tmp_path: Path) -> None:
    context = _context("20260823T000000Z-test-head-e176da5e-abcdef12")
    context["group_id"] = "wrong-group"
    with pytest.raises(HeadTrackingError, match="group_id"):
        initialize_turkish_head_attempt(
            tmp_path / "attempt",
            context=context,
            config=_config(),
            parent={"parent_attempt_id": "20260823T000000Z-parent-e176da5e-abcdef12", "parent_checkpoint_path": "/tmp/best"},
        )
