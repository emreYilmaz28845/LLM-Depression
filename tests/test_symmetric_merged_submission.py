from __future__ import annotations

from scripts.submit_symmetric_merged import (
    _set_combined_registry_metadata,
    _stage_plans,
    merge_existing_registry,
)


def _stage_plan(stage: str, plan_hash: str) -> dict:
    identity = {"stage": stage, "configs": [{"modality": "audio_text"}]}
    return {
        "stage": stage,
        "plan_identity": identity,
        "plan_hash": plan_hash,
        "expected_fresh_job_count": 45 if stage == "cv" else 9,
    }


def _job(stage: str, key: str, state: str, job_id: str) -> dict:
    return {
        "job_key": key,
        "stage": stage,
        "modality": "audio_text",
        "fold": 0,
        "kind": "train",
        "config": f"configs/{stage}.yaml",
        "state": state,
        "job_id": job_id,
    }


def test_registry_can_append_final_stage_to_completed_cv_run() -> None:
    cv_plan = _stage_plan("cv", "cv-hash")
    final_plan = _stage_plan("final", "final-hash")
    existing = {
        "run_id": "shared-run",
        "stage": "cv",
        "stage_plans": {"cv": cv_plan},
        "jobs": [_job("cv", "audio_text:cv:fold_0:train", "submitted", "123")],
    }
    registry = {
        "run_id": "shared-run",
        "stage": "final",
        "stage_plans": {"final": final_plan},
        "jobs": [_job("final", "audio_text:final:fold_0:train", "planned", "dry-final")],
    }

    registry = merge_existing_registry(registry, existing)
    registry = _set_combined_registry_metadata(
        registry,
        {**_stage_plans(existing), **_stage_plans(registry)},
    )

    assert registry["stage"] == "multi"
    assert registry["stages"] == ["cv", "final"]
    assert registry["expected_job_count"] == 54
    assert {job["stage"] for job in registry["jobs"]} == {"cv", "final"}
    assert {job["job_id"] for job in registry["jobs"]} == {"123", "dry-final"}


def test_registry_reuses_legacy_single_stage_plan() -> None:
    legacy = {
        "stage": "cv",
        "plan_identity": {"stage": "cv"},
        "plan_hash": "legacy-hash",
        "expected_fresh_job_count": 45,
    }
    assert _stage_plans(legacy)["cv"]["plan_hash"] == "legacy-hash"
