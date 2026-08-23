from __future__ import annotations

from pathlib import Path

from src.native_en_text_heads import (
    BACKBONES,
    CONDITIONS,
    HEAD_SEED,
    SPLIT_SEED,
    TRAINING_SEEDS,
    build_matrix,
    matrix_counts,
    validate_configs,
)
from tools.native_en_text_heads import (
    add_plan_indexes,
    build_plan,
    job_export,
    parse_submission_markers,
    stage_root,
    remote_prepare_script,
    remote_submission_script,
)


def test_locked_smoke_and_production_counts() -> None:
    assert matrix_counts("smoke") == {
        "total": 32,
        "train": 8,
        "best_eval": 4,
        "postprocess": 4,
        "logreg": 8,
        "xgb_optuna100": 8,
    }
    assert matrix_counts("production") == {
        "total": 1248,
        "train": 312,
        "best_eval": 240,
        "postprocess": 72,
        "logreg": 312,
        "xgb_optuna100": 312,
    }


def test_production_matrix_covers_three_seeds_and_four_backbone_condition_panels() -> None:
    jobs = build_matrix("production")
    assert {job.seed for job in jobs} == set(TRAINING_SEEDS)
    assert {(job.condition, job.backbone) for job in jobs} == {
        (condition, backbone) for condition in CONDITIONS for backbone in BACKBONES
    }
    assert {job.fold for job in jobs if job.endpoint == "standalone"} == {0, 1, 2, 3, 4}
    assert {job.fold for job in jobs if job.endpoint == "merged_cv"} == {0, 1, 2, 3, 4}
    assert {job.fold for job in jobs if job.endpoint == "merged_final"} == {0}
    assert all(job.evaluation_view == "harmonized_all_windows_full_coverage" for job in jobs if job.job_type != "train")
    assert all(job.backend for job in jobs if job.method in {"logreg", "xgb_optuna100"})
    assert all(job.optuna_trials == 100 for job in jobs if job.method == "xgb_optuna100")


def test_matrix_config_and_fixed_seed_contract() -> None:
    identities = validate_configs(".")
    assert len(identities) == 20
    assert SPLIT_SEED == 1337
    assert HEAD_SEED == 1337


def _fake_deployment() -> dict[str, object]:
    return {
        "deployment_id": "native-en-text-heads-v2-test-deployment",
        "experiment_id": "exp-native-en-text-heads-v2-20260822",
        "git_commit": "d0b9a0074a7da1a2402855e00cada5b21ea518f7",
        "git_branch_at_deploy": "agent/exp-native-en-text-heads-v2",
        "git_dirty": False,
        "source_manifest_sha256": "a" * 64,
        "deployed_code_path": "/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/test/code",
    }


def test_managed_smoke_plan_has_four_job_chains_and_parseable_markers() -> None:
    plan = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(plan)
    assert plan["counts"]["total"] == 32
    assert all(
        job.get("dependencies") is not None
        for job in plan["jobs"]
        if job.get("kind") != "standalone_backbone"
    )
    script = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
    )
    assert "__SUBMISSION_COMPLETE__" in script
    assert "scripts/submit_train_and_eval.sh" in script
    assert "--dependency=afterok:" in script
    assert "SBATCH_EXTRA_ARGS=--exclude=as01r2b12" in script
    assert "--exclude=as01r2b12" in script
    standalone = next(job for job in plan["jobs"] if job.get("kind") == "standalone_backbone")
    assert "--set=evaluation.evaluation_view=harmonized_all_windows_full_coverage" in standalone["overrides"]
    head_jobs = [job for job in plan["jobs"] if job.get("method")]
    assert all(job.get("backend") for job in head_jobs)
    assert {job["trials"] for job in head_jobs if job["method"] == "xgb_optuna100"} == {2}

    marker_lines = []
    for job in plan["jobs"]:
        index = int(job["plan_index"])
        if job.get("kind") == "standalone_backbone":
            marker_lines.append(f"__STANDALONE__ {index} {1000 + index} {2000 + index}")
        else:
            marker_lines.append(f"__JOB__ {index} {3000 + index}")
    parse_submission_markers(plan, "\n".join(marker_lines))
    assert all(job["job_ids"] for job in plan["jobs"])


def test_submission_initializes_each_custom_attempt_near_its_sbatch() -> None:
    plan = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(plan)
    script = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
    )
    init_pos = script.index("tools/native_en_text_heads_worker.py init")
    sbatch_pos = script.index("sbatch --parsable", init_pos)
    assert init_pos < sbatch_pos


def test_retry_plan_uses_fresh_output_identity_and_links_superseded_attempts() -> None:
    original = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(original)
    original["submission_complete"] = True
    retry_deployment = {
        **_fake_deployment(),
        "deployment_id": "native-en-text-heads-v2-retry-deployment",
        "git_commit": "47d163f44f72586a2f905798af885b15955be519",
    }
    retry = build_plan(
        stage="smoke",
        deployment=retry_deployment,
        experiment_id="exp-native-en-text-heads-v2-20260822",
        output_suffix="retry1",
        retry_from=original,
    )
    add_plan_indexes(retry)

    assert retry["output_suffix"] == "retry1"
    assert retry["retry_from_deployment_id"] == original["deployment_id"]
    old_paths = {
        job.get("fold_dir") if job.get("kind") == "standalone_backbone" else job.get("attempt_dir")
        for job in original["jobs"]
    }
    old_paths.update(job.get("cache_dir") for job in original["jobs"] if job.get("cache_dir"))
    retry_paths = {
        job.get("fold_dir") if job.get("kind") == "standalone_backbone" else job.get("attempt_dir")
        for job in retry["jobs"]
    }
    retry_paths.update(job.get("cache_dir") for job in retry["jobs"] if job.get("cache_dir"))
    assert old_paths.isdisjoint(retry_paths)
    assert all(job.get("supersedes_attempt_id") for job in retry["jobs"])
    assert all(
        (job.get("context") or job.get("context_payload") or {}).get("supersedes_attempt_id")
        for job in retry["jobs"]
    )
    standalone = next(job for job in retry["jobs"] if job.get("kind") == "standalone_backbone")
    assert "native_en_text_heads_v2_smoke_retry1" in standalone["run_root"]
    head = next(job for job in retry["jobs"] if job.get("method") == "logreg")
    assert "/hidden_features/retry1/" in head["cache_dir"]


def test_retry_output_suffix_isolates_runtime_namespace() -> None:
    canonical = stage_root("exp-native-en-text-heads-v2-20260822", "production")
    retry = stage_root("exp-native-en-text-heads-v2-20260822", "production", "splitfix1")
    assert retry == canonical / "splitfix1"
    assert retry != canonical


def test_selective_head_retry_reuses_completed_parent_artifacts() -> None:
    original = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
        output_suffix="retry1",
    )
    add_plan_indexes(original)
    original["submission_complete"] = True
    for index, job in enumerate(original["jobs"]):
        job["job_ids"] = {"job": str(5000 + index)}
    failed_keys = [
        job["job_key"]
        for job in original["jobs"]
        if job.get("method") == "xgb_optuna100"
    ][:2]
    retry_deployment = {
        **_fake_deployment(),
        "deployment_id": "native-en-text-heads-v2-selective-retry",
        "git_commit": "047ec597fcafd7ab730b06c8001412f792741494",
    }
    retry = build_plan(
        stage="smoke",
        deployment=retry_deployment,
        experiment_id="exp-native-en-text-heads-v2-20260822",
        output_suffix="retry1",
        retry_from=original,
        retry_job_keys=failed_keys,
    )
    add_plan_indexes(retry)

    assert len(retry["jobs"]) == 2
    assert all(job["method"] == "xgb_optuna100" for job in retry["jobs"])
    assert all(job["dependencies"] == [] for job in retry["jobs"])
    assert all(job["reuse_parent_artifacts"] for job in retry["jobs"])
    assert len(retry["retry_reused_dependencies"]) == 2
    assert retry["counts"]["xgb_optuna100"] == 2
    assert all("features" not in path for path in retry["collision_paths"])


def test_remote_workers_source_mn5_dataset_environment() -> None:
    plan = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(plan)
    expected = "source /gpfs/projects/etur92/ozu647717/AudioLLM/deployments/test/code/scripts/native_en_text_heads_env.sh"
    prepare = remote_prepare_script(
        plan,
        _fake_deployment(),
        {},
        Path(plan["stage_root"]) / "preflight.json",
    )
    submission = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
    )
    assert expected in prepare
    assert expected in submission
    assert "--skip-existing-components" in prepare
    assert "reusing complete existing merged preparation artifacts" in prepare
    assert "manifest_map_native-en-text-heads-v2-test-deployment.json" in prepare


def test_remote_prepare_attaches_set_tokens_to_override_options() -> None:
    plan = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    script = remote_prepare_script(
        plan,
        _fake_deployment(),
        {},
        Path(plan["stage_root"]) / "preflight.json",
    )
    assert "--override=--set=output_dirs.run_root=" in script
    assert "--override --set=output_dirs.run_root=" not in script


def test_production_cv_submission_script_defers_final_jobs() -> None:
    plan = build_plan(
        stage="production",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(plan)
    script = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
        phase="cv",
    )
    assert "merged_final" not in script
    assert "__SUBMISSION_COMPLETE__ 960" in script
    assert sum(job.get("endpoint") == "merged_final" for job in plan["jobs"]) == 48


def test_merged_optuna100_identity_uses_historical_cli_spelling_at_boundary() -> None:
    entry = {
        "script": "scripts/run_native_en_merged_head_slurm.sh",
        "method": "xgb_optuna100",
        "job_type": "hidden_classifier",
        "attempt_dir": "/gpfs/attempt",
        "context_path": "/gpfs/context.json",
        "config_json_path": "/gpfs/config.json",
        "parent_json_path": "/gpfs/parent.json",
        "config_remote": "/gpfs/config.yaml",
        "stage": "production",
        "fold": 0,
        "run_id": "run",
        "features_dir": "/gpfs/features",
        "checkpoint_dir": "/gpfs/best_model",
        "condition": "native",
        "backbone": "qwen",
        "overrides": [],
    }
    exported = job_export(entry, _fake_deployment())
    assert "METHOD=xgb_optuna" in exported
    assert "METHOD=xgb_optuna100" not in exported


def test_smoke_merged_optuna_head_exports_smoke_policy_stage() -> None:
    entry = {
        "script": "scripts/run_native_en_merged_head_slurm.sh",
        "method": "xgb_optuna100",
        "job_type": "hidden_classifier",
        "attempt_dir": "/gpfs/attempt",
        "context_path": "/gpfs/context.json",
        "config_json_path": "/gpfs/config.json",
        "parent_json_path": "/gpfs/parent.json",
        "config_remote": "/gpfs/config.yaml",
        "stage": "cv",
        "fold": 0,
        "run_id": "run",
        "features_dir": "/gpfs/features",
        "checkpoint_dir": "/gpfs/best_model",
        "condition": "native",
        "backbone": "qwen",
        "trials": 2,
        "overrides": [],
    }
    exported = job_export(entry, _fake_deployment())
    assert "OPTUNA_STAGE=smoke" in exported
