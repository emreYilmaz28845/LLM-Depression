"""Multi-variant head evidence materialization.

The head classifier worker fits two fixed variants (logreg_raw, xgb_raw) and
materializes their evidence into one attempt. Variant-qualified evaluation
identity, union retention of previously materialized records, idempotent
repeats and fail-closed conflict rejection are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiment_tracking.canonical import canonical_sha256, read_json
from src.experiment_tracking.identity import evaluation_id, new_attempt_id
from src.native_en_text_heads_tracking import (
    REPAIR_JOB_KEY,
    HeadTrackingError,
    _successful_required_jobs,
    initialize_head_attempt,
    materialize_head_evidence,
    prevalidate_head_fit_outputs,
    record_head_job,
    record_head_repair_event,
    transition_head_attempt,
)

PREDICTIONS = [
    {"dataset": "d3tec", "subject_id": "s1", "label": 1, "predicted_class": 1},
    {"dataset": "d3tec", "subject_id": "s2", "label": 0, "predicted_class": 0},
]
PARENT_ADAPTER_SHA = "a" * 64


def _attempt(tmp_path: Path) -> Path:
    attempt_dir = tmp_path / "attempt"
    initialize_head_attempt(
        attempt_dir,
        context={
            "attempt_id": new_attempt_id("q3ms_head_d3tec_text_only_native_s1337_f0", "a" * 40),
            "logical_run_name": "q3ms_head_d3tec_text_only_native_s1337_f0",
            "fold": 0,
            "seed": 1337,
            "tracking_kind": "qwen3_multiseed_native_head",
            "run_schema_version": "audiollm.qwen3_multiseed_head_run.v1",
            "group_id": "group-x",
        },
        config={
            "dataset": "d3tec",
            "dataset_variant": None,
            "classifier": {"prediction_backend": "likelihood"},
            "evaluation": {
                "evaluation_view": "harmonized_all_windows_full_coverage",
                "aggregation": "subject",
            },
        },
        parent={
            "parent_attempt_id": new_attempt_id("run_parent", "b" * 40),
            "parent_checkpoint_path": "/gpfs/parent/best_model",
            "adapter_sha256": PARENT_ADAPTER_SHA,
        },
    )
    for state in ("DEPLOYED", "SUBMITTED", "RUNNING"):
        transition_head_attempt(attempt_dir, state, reason="test")
    record_head_job(
        attempt_dir,
        job_key="extract",
        job_type="hidden_extraction",
        event_type="SUBMITTED",
        slurm_job_id="1",
        status="PENDING",
    )
    record_head_job(
        attempt_dir,
        job_key="classifier",
        job_type="hidden_classifier",
        event_type="SUBMITTED",
        slurm_job_id="2",
        status="PENDING",
    )
    return attempt_dir


def _write_variant(attempt_dir: Path, variant: str, metrics: bytes) -> tuple[Path, Path]:
    variant_dir = attempt_dir / "classifier" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    predictions = variant_dir / "predictions_subject_level.jsonl"
    metrics_path = variant_dir / "metrics.json"
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in PREDICTIONS), encoding="utf-8"
    )
    metrics_path.write_bytes(metrics)
    return predictions, metrics_path


def _materialize(attempt_dir: Path, variant: str, metrics: bytes) -> dict:
    predictions, metrics_path = _write_variant(attempt_dir, variant, metrics)
    return materialize_head_evidence(
        attempt_dir,
        predictions_path=predictions,
        metrics_path=metrics_path,
        checkpoint_path="/gpfs/parent/best_model",
        variant=variant,
    )


def _write_classifier_metadata(attempt_dir: Path, variant: str, adapter_sha: str = PARENT_ADAPTER_SHA) -> None:
    variant_dir = attempt_dir / "classifier" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "classifier_metadata.json").write_text(
        json.dumps(
            {
                "seed": 1337,
                "checkpoint_hashes": {"adapter_sha256": adapter_sha},
            }
        ),
        encoding="utf-8",
    )


def test_identical_metrics_across_variants_are_both_retained(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    identical = b'{"macro_f1": 0.5}\n'
    _materialize(attempt_dir, "logreg_raw", identical)
    _materialize(attempt_dir, "xgb_raw", identical)
    evaluations = read_json(attempt_dir / "evaluations.json")["evaluations"]
    assert len(evaluations) == 2
    ids = {item["evaluation_id"] for item in evaluations}
    assert len(ids) == 2
    assert {item["head_variant"] for item in evaluations} == {"logreg_raw", "xgb_raw"}
    assert {item["metrics_artifact_path"] for item in evaluations} == {
        "classifier/logreg_raw/metrics.json",
        "classifier/xgb_raw/metrics.json",
    }


def test_differing_metrics_keep_two_evaluations(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    _materialize(attempt_dir, "logreg_raw", b'{"macro_f1": 0.4}\n')
    _materialize(attempt_dir, "xgb_raw", b'{"macro_f1": 0.6}\n')
    evaluations = read_json(attempt_dir / "evaluations.json")["evaluations"]
    assert len(evaluations) == 2


def test_materialize_repeat_is_idempotent(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    metrics = b'{"macro_f1": 0.5}\n'
    _materialize(attempt_dir, "logreg_raw", metrics)
    evaluations_before = read_json(attempt_dir / "evaluations.json")["evaluations"]
    artifacts_before = read_json(attempt_dir / "artifacts.json")["artifacts"]
    _materialize(attempt_dir, "logreg_raw", metrics)
    assert read_json(attempt_dir / "evaluations.json")["evaluations"] == evaluations_before
    assert read_json(attempt_dir / "artifacts.json")["artifacts"] == artifacts_before
    assert read_json(attempt_dir / "status.json")["state"] == "COMPLETED_ON_MN5"


def test_materialize_rejects_same_identity_with_different_content(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    _materialize(attempt_dir, "logreg_raw", b'{"macro_f1": 0.5}\n')
    evaluation_doc = read_json(attempt_dir / "evaluations.json")
    evaluation_doc["evaluations"][0]["aggregation"] = "tampered"
    (attempt_dir / "evaluations.json").write_text(
        json.dumps(evaluation_doc), encoding="utf-8"
    )
    with pytest.raises(HeadTrackingError, match="evaluation identity changed"):
        _materialize(attempt_dir, "logreg_raw", b'{"macro_f1": 0.5}\n')


def test_materialize_union_keeps_previous_artifacts(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    _materialize(attempt_dir, "logreg_raw", b'{"macro_f1": 0.5}\n')
    extra = attempt_dir / "extra_evidence.json"
    extra.write_text('{"note": "kept"}\n', encoding="utf-8")
    _materialize(attempt_dir, "logreg_raw", b'{"macro_f1": 0.5}\n')
    paths = {item["path"] for item in read_json(attempt_dir / "artifacts.json")["artifacts"]}
    assert "extra_evidence.json" in paths
    extra.unlink()
    _materialize(attempt_dir, "xgb_raw", b'{"macro_f1": 0.5}\n')
    paths = {item["path"] for item in read_json(attempt_dir / "artifacts.json")["artifacts"]}
    assert "extra_evidence.json" in paths


def test_evaluation_id_qualifier_is_additive() -> None:
    base = {
        "attempt_id": "20261002T000000Z-run-aaaaaaaa-12345678",
        "fold": 0,
        "dataset": "d3tec",
        "split_name": "outer_holdout",
        "split_protocol": "saved_split",
        "checkpoint_role": "best_model",
        "checkpoint_path": "/gpfs/parent/best_model",
        "backend": "likelihood",
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "subject",
        "metric_namespace": "headline/binary_strict",
        "metrics_artifact_sha256": "a" * 64,
    }
    legacy_payload = dict(base)
    expected_legacy = "eval-" + canonical_sha256(legacy_payload)[:24]
    assert evaluation_id(**base) == expected_legacy
    assert evaluation_id(**base, qualifier=None) == expected_legacy
    qualified = evaluation_id(**base, qualifier="head_variant:logreg_raw")
    assert qualified != expected_legacy
    assert evaluation_id(**base, qualifier="head_variant:xgb_raw") != qualified


def test_prevalidate_requires_all_variants_before_writes(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    _write_variant(attempt_dir, "logreg_raw", b'{"macro_f1": 0.5}\n')
    _write_classifier_metadata(attempt_dir, "logreg_raw")
    evaluations_before = read_json(attempt_dir / "evaluations.json")
    with pytest.raises(HeadTrackingError, match="no fitted evidence"):
        prevalidate_head_fit_outputs(
            attempt_dir, attempt_dir / "classifier", ["logreg_raw", "xgb_raw"]
        )
    assert read_json(attempt_dir / "evaluations.json") == evaluations_before


def test_prevalidate_rejects_adapter_mismatch(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    for variant in ("logreg_raw", "xgb_raw"):
        _write_variant(attempt_dir, variant, b'{"macro_f1": 0.5}\n')
        _write_classifier_metadata(attempt_dir, variant, adapter_sha="b" * 64)
    with pytest.raises(HeadTrackingError, match="parent adapter"):
        prevalidate_head_fit_outputs(
            attempt_dir, attempt_dir / "classifier", ["logreg_raw", "xgb_raw"]
        )


def test_repair_event_is_observed_idempotent_and_satisfies_gate(tmp_path: Path) -> None:
    attempt_dir = _attempt(tmp_path)
    record_head_repair_event(
        attempt_dir, original_job_id="46955648", reason="non-scheduler repair"
    )
    record_head_repair_event(
        attempt_dir, original_job_id="46955648", reason="non-scheduler repair"
    )
    events = [
        json.loads(line)
        for line in (attempt_dir / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repairs = [event for event in events if event.get("job_key") == REPAIR_JOB_KEY]
    assert len(repairs) == 1
    assert repairs[0]["event_type"] == "OBSERVED"
    assert repairs[0]["slurm_job_id"] is None
    assert repairs[0].get("exit_code") is None
    assert repairs[0]["resubmission_of_job_id"] == "46955648"
    _required, successful = _successful_required_jobs(attempt_dir)
    assert "classifier" in successful
