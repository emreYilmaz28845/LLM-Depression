from __future__ import annotations

from pathlib import Path

from scripts.submit_symmetric_merged import (
    CONFIG_BY_MODALITY,
    build_job_specs,
    _set_combined_registry_metadata,
    _final_epoch_for_dry_run,
    _stage_plans,
    merge_existing_registry,
)
from src.merged.runtime import load_merged_config


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


def test_final_dry_run_resolves_frozen_median_epoch_when_cv_is_accepted(tmp_path: Path) -> None:
    config = {
        "output_dirs": {
            "merged_root": str(tmp_path / "merged"),
            "run_root": str(tmp_path / "models"),
        }
    }
    run_id = "shared-run"
    audit_path = tmp_path / "merged" / run_id / "cv" / "acceptance_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"status": "passed"}\n', encoding="utf-8")
    for fold, epoch in enumerate((1, 4, 3, 2, 3)):
        path = (
            tmp_path
            / "models"
            / run_id
            / "cv"
            / f"fold_{fold}"
            / "logs"
            / "selected_checkpoint.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(f'{{"selected_epoch": {epoch}}}\n', encoding="utf-8")

    assert _final_epoch_for_dry_run(config, run_id, "audio_text") == 3

    (tmp_path / "merged" / run_id / "cv" / "acceptance_audit.json").write_text(
        '{"status": "failed"}\n', encoding="utf-8"
    )
    assert _final_epoch_for_dry_run(config, run_id, "audio_text") is None


def test_targeted_retry_job_count_matches_selected_configs() -> None:
    registry = build_job_specs(
        [CONFIG_BY_MODALITY["audio_text"]],
        stage="cv",
        run_id="targeted_retry_count",
        dry_run=True,
        smoke_subjects=2,
        smoke_epochs=1,
        smoke_trials=2,
    )
    assert registry["expected_fresh_job_count"] == 15
    assert len(registry["jobs"]) == 15


def test_qwen_worker_uses_all_allocated_gpus() -> None:
    worker = Path("scripts/run_symmetric_merged_train_slurm.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:4" in worker
    assert "torchrun --standalone" in worker
    assert "--nproc_per_node=\"$NPROC_PER_NODE\"" in worker
    assert "python -m src.merged.train" not in worker


def test_merged_train_preflight_does_not_self_create_an_incomplete_run() -> None:
    source = Path("src/merged/train.py").read_text(encoding="utf-8")
    assert 'logs_dir = run_root / "logs"' in source
    assert 'logs_dir = ensure_dir(run_root / "logs")' not in source
    assert "is_local_main_process = accelerator.is_main_process" in source
    assert source.index("accelerator = Accelerator(") < source.index("if complete_path.is_file()")
    assert source.count("accelerator.wait_for_everyone()") >= 2
    assert "if is_local_main_process:" in source


def test_merged_text_only_uses_the_dense_text_backbone() -> None:
    config = load_merged_config(CONFIG_BY_MODALITY["text_only"])
    assert config["modality"] == "text_only"
    assert config["model_name_or_path"].endswith("/Qwen2-7B-Instruct")
