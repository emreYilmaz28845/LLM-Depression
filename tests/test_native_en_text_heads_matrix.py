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
    standalone = next(job for job in plan["jobs"] if job.get("kind") == "standalone_backbone")
    assert "--set=evaluation.evaluation_view=harmonized_all_windows_full_coverage" in standalone["overrides"]

    marker_lines = []
    for job in plan["jobs"]:
        index = int(job["plan_index"])
        if job.get("kind") == "standalone_backbone":
            marker_lines.append(f"__STANDALONE__ {index} {1000 + index} {2000 + index}")
        else:
            marker_lines.append(f"__JOB__ {index} {3000 + index}")
    parse_submission_markers(plan, "\n".join(marker_lines))
    assert all(job["job_ids"] for job in plan["jobs"])


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
    retry_paths = {
        job.get("fold_dir") if job.get("kind") == "standalone_backbone" else job.get("attempt_dir")
        for job in retry["jobs"]
    }
    assert old_paths.isdisjoint(retry_paths)
    assert all(job.get("supersedes_attempt_id") for job in retry["jobs"])
    assert all(
        (job.get("context") or job.get("context_payload") or {}).get("supersedes_attempt_id")
        for job in retry["jobs"]
    )
    standalone = next(job for job in retry["jobs"] if job.get("kind") == "standalone_backbone")
    assert "native_en_text_heads_v2_smoke_retry1" in standalone["run_root"]


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
