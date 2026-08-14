from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from scripts.submit_symmetric_merged import _head_trials, build_job_specs
from src.merged.audit import _expected_head_methods, _job_registry_path
from src.merged.runtime import load_merged_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/experiments/harmonized/standalone_matrix.yaml"
MERGED = {
    modality: ROOT
    / "configs/experiments/merged"
    / f"symmetric_merged_harmonized_{modality}.yaml"
    for modality in ("audio_text", "audio_only", "text_only")
}


def test_standalone_matrix_is_the_complete_fixed_head_recipe() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert matrix["fixed_heads"] == ["logreg_raw", "xgb_raw"]
    assert matrix["max_epochs"] == 20
    assert matrix["checkpoint_selection"] == "inner_val_macro_f1"
    assert len(matrix["experiments"]) == 15
    assert sum(len(item["folds"]) for item in matrix["experiments"]) == 63
    assert sum(len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"]) == 33
    for item in matrix["experiments"]:
        config = yaml.safe_load((ROOT / item["config"]).read_text(encoding="utf-8"))
        assert config["training"]["num_train_epochs"] == 20
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["early_stopping"]["patience"] == 3


def test_harmonized_merged_configs_use_only_harmonized_components() -> None:
    expected = {"daic", "cmdc", "turkish", "d3tec", "androids_interview"}
    for modality, path in MERGED.items():
        config = load_merged_config(path)
        assert config["modality"] == modality
        assert config["recipe_id"] == "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1"
        assert config["training"]["num_train_epochs"] == 20
        assert config["training"]["final_epoch_policy"] == "rounded_median_selected_epoch"
        assert config["protocol_settings"]["selection_metric"] == "mean_dataset_macro_f1"
        assert config["heads"]["optuna"]["enabled"] is False
        assert config["heads"]["optuna"]["target_trials"] == 0
        assert {item["name"] for item in config["components"]} == expected
        for item in config["components"]:
            assert "harmonized" in item["config"]
            assert "manifests_harmonized" in item["manifest_path"]
            assert "splits_harmonized" in item["metadata_path"]
            assert (ROOT / item["config"]).is_file()


def test_merged_planner_disables_optuna_and_applies_64_gpu_lanes() -> None:
    configs = list(MERGED.values())
    registry = build_job_specs(
        configs,
        stage="cv",
        run_id="harmonized_test",
        dry_run=True,
        smoke_subjects=2,
        smoke_epochs=1,
        smoke_trials=99,
        max_concurrent_trains=15,
        max_concurrent_postprocess=4,
        github_issue=12,
        github_pr=10,
    )
    jobs = registry["jobs"]
    assert len(jobs) == 45
    assert {job["trials"] for job in jobs if job["kind"] == "head"} == {0}
    trains = [job for job in jobs if job["kind"] == "train"]
    posts = [job for job in jobs if job["kind"] == "postprocess"]
    assert {job["concurrency_lane"] for job in trains} == set(range(15))
    assert {job["concurrency_lane"] for job in posts} == set(range(4))
    assert sum("throttle_dependency_job_key" not in job for job in trains) == 15
    assert sum("throttle_dependency_job_key" not in job for job in posts) == 4
    assert registry["plan_identity"]["max_concurrent_trains"] == 15
    assert registry["plan_identity"]["max_concurrent_postprocess"] == 4
    assert registry["research"] == {"github_issue": 12, "github_pr": 10}
    assert registry["plan_identity"]["research"] == registry["research"]


def test_harmonized_smoke_keeps_custom_config_and_zero_trials() -> None:
    config = load_merged_config(MERGED["audio_text"])
    assert _head_trials(config, stage="smoke", smoke_trials=50) == 0
    registry = build_job_specs(
        list(MERGED.values()),
        stage="smoke",
        run_id="harmonized_smoke_test",
        dry_run=True,
        smoke_subjects=2,
        smoke_epochs=1,
        smoke_trials=50,
        github_issue=12,
        github_pr=10,
    )
    # The smoke stage now covers all three modalities (runbook Section 13:
    # one merged smoke per modality): 3 modalities x train/postprocess/head.
    assert len(registry["jobs"]) == 9
    assert {job["config"] for job in registry["jobs"]} == {str(path) for path in MERGED.values()}
    assert {job["kind"] for job in registry["jobs"]} == {"train", "postprocess", "head"}
    assert all(next(job for job in registry["jobs"] if job["kind"] == "head" and job["modality"] == modality)["trials"] == 0 for modality in ("audio_text", "audio_only", "text_only"))


def test_harmonized_auditor_requires_only_enabled_heads_and_global_registry() -> None:
    config = load_merged_config(MERGED["audio_text"])
    assert _expected_head_methods(config) == ("logreg", "xgb_fixed")
    historical = {"heads": {"optuna": {"target_trials": 150}}}
    assert _expected_head_methods(historical) == ("logreg", "xgb_fixed", "xgb_optuna")
    assert _job_registry_path("run-1") == ROOT / "outputs/symmetric_merged_jobs/run-1.json"


def test_workers_export_every_harmonized_dataset_root() -> None:
    required = {
        "DAIC_UNPROCESSED_ROOT",
        "DAIC_LABEL_ROOT",
        "D3TEC_DATASET_ROOT",
        "D3TEC_FULL_TRANSCRIPTS",
        "D3TEC_SEGMENT_TRANSCRIPTS",
        "ANDROIDS_DATASET_ROOT",
        "ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS",
        "ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS",
    }
    workers = (
        "scripts/run_train_slurm.sh",
        "scripts/run_eval_slurm.sh",
        "scripts/run_symmetric_merged_train_slurm.sh",
        "scripts/run_symmetric_merged_postprocess_slurm.sh",
        "scripts/run_symmetric_merged_head_slurm.sh",
    )
    for worker in workers:
        text = (ROOT / worker).read_text(encoding="utf-8")
        assert all(name in text for name in required), worker


def test_standalone_dry_run_has_63_trains_33_evals_and_63_fixed_heads() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_standalone.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "1",
            "GITHUB_ISSUE": "12",
            "GITHUB_PR": "10",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    commands = [line for line in result.stderr.splitlines() if line.startswith("DRY_RUN sbatch")]
    assert len(commands) == 159
    assert sum("run_train_slurm.sh" in line for line in commands) == 63
    assert sum("run_eval_slurm.sh" in line for line in commands) == 33
    assert sum("run_qwen_hidden_extract_slurm.sh" in line for line in commands) == 63
    assert "max_gpus=64" in result.stdout
    assert "xgb_optuna" not in result.stdout + result.stderr
    assert "github_issue=12 github_pr=10" in result.stdout


def test_harmonized_launchers_refuse_more_than_64_gpus() -> None:
    for launcher, extra_env in (
        ("scripts/submit_harmonized_standalone.sh", {"MAX_CONCURRENT_AUX": "1"}),
        ("scripts/submit_harmonized_standalone_retry.sh", {"MAX_CONCURRENT_AUX": "1", "CELLS": "unused"}),
        ("scripts/submit_harmonized_merged.sh", {"MAX_CONCURRENT_POSTPROCESS": "1", "STAGE": "smoke"}),
    ):
        result = subprocess.run(
            ["bash", str(ROOT / launcher)],
            cwd=ROOT,
            env={
                **os.environ,
                "PROJECT_ROOT": str(ROOT),
                "RUN_ID": "unit",
                "DRY_RUN": "1",
                "GITHUB_ISSUE": "12",
                "GITHUB_PR": "10",
                "MAX_CONCURRENT_TRAINS": "16",
                **extra_env,
            },
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert "64-GPU project ceiling" in result.stderr


def test_harmonized_launchers_require_campaign_provenance() -> None:
    for launcher, extra_env in (
        ("scripts/submit_harmonized_standalone.sh", {}),
        ("scripts/submit_harmonized_merged.sh", {"STAGE": "smoke"}),
    ):
        env = {
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "1",
            **extra_env,
        }
        env.pop("GITHUB_ISSUE", None)
        env.pop("GITHUB_PR", None)
        result = subprocess.run(
            ["bash", str(ROOT / launcher)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert "GITHUB_ISSUE" in result.stderr


def test_harmonized_merged_wrapper_records_campaign_provenance(tmp_path: Path) -> None:
    registry_path = tmp_path / "merged_registry.json"
    subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_merged.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "provenance_unit",
            "STAGE": "smoke",
            "DRY_RUN": "1",
            "GITHUB_ISSUE": "12",
            "GITHUB_PR": "10",
            "REGISTRY": str(registry_path),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["research"] == {"github_issue": 12, "github_pr": 10}
    assert registry["stage_plans"]["smoke"]["plan_identity"]["research"] == registry["research"]
