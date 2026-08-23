from __future__ import annotations

import json
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
    derive_final_epochs,
    job_export,
    parse_submission_markers,
    _reconcile_collected_standalone_sidecars,
    resume_job_ids,
    stage_root,
    remote_prepare_script,
    remote_submission_script,
)
from src.native_en_text_heads_tracking import initialize_head_attempt_batch
from src.experiment_tracking.canonical import read_jsonl
from src.experiment_tracking.lifecycle import (
    StatusRecord,
    append_job_event,
    new_job_event,
    write_status,
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
    assert standalone["local_fold_rel"].startswith("output_model/")
    assert not standalone["local_fold_rel"].startswith("output_model/output_model/")
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


def test_submission_batches_custom_attempt_initialization_before_sbatch() -> None:
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
    init_pos = script.index("tools/native_en_text_heads_worker.py init-batch")
    sbatch_pos = script.index("sbatch --parsable", init_pos)
    assert init_pos < sbatch_pos
    assert "write_once_stdin" in script
    assert "write_once_stdin" in script[:init_pos]


def test_resume_submission_reuses_prefix_ids_without_reinitializing_custom_attempts() -> None:
    plan = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(plan)
    existing = resume_job_ids(
        plan,
        "__STANDALONE__ 0 1000 2000\n__JOB__ 1 3001\n__JOB__ 2 3002\n",
        2,
        phase="all",
    )
    script = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
        resume_after=2,
        existing_job_ids=existing,
    )
    assert "train_0=1000" in script
    assert "eval_0=2000" in script
    assert "jid_1=3001" in script
    assert "jid_2=3002" in script
    assert "init-batch" not in script
    assert "__STANDALONE__ 3" in script
    future_standalone = next(
        job
        for job in plan["jobs"]
        if job.get("kind") == "standalone_backbone" and int(job["plan_index"]) > 2
    )
    assert f"test ! -e {future_standalone['fold_dir']}" not in script

    prefix_markers = []
    for job in plan["jobs"]:
        index = int(job["plan_index"])
        if index > 3:
            break
        if job.get("kind") == "standalone_backbone":
            prefix_markers.append(f"__STANDALONE__ {index} {4000 + index} {5000 + index}")
        else:
            prefix_markers.append(f"__JOB__ {index} {6000 + index}")
    prefix_ids = resume_job_ids(plan, "\n".join(prefix_markers), 3, phase="all")
    completed_parent_script = remote_submission_script(
        plan,
        _fake_deployment(),
        Path(plan["stage_root"]) / "preflight.json",
        resume_after=3,
        existing_job_ids=prefix_ids,
        completed_existing_indexes={3},
    )
    assert "--dependency=afterok:$eval_3" not in completed_parent_script
    assert '--dependency-job-id "$eval_3"' in completed_parent_script


def test_batch_head_initialization_writes_sidecars_and_deploys(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    context_path = tmp_path / "context.json"
    config_path = tmp_path / "config.json"
    parent_path = tmp_path / "parent.json"
    context = {
        "attempt_id": "20260823T000000Z-test-head",
        "logical_run_name": "test-head",
        "fold": 0,
        "seed": 1337,
    }
    config = {"classifier": {"method": "logreg"}}
    parent = {"parent_attempt_id": "parent-1", "parent_checkpoint_path": "/tmp/best_model"}
    result = initialize_head_attempt_batch(
        [
            {
                "attempt_dir": str(attempt),
                "context_path": str(context_path),
                "config_path": str(config_path),
                "parent_path": str(parent_path),
                "context": context,
                "config": config,
                "parent": parent,
            }
        ]
    )
    assert result[0]["state"] == "DEPLOYED"
    assert json.loads(context_path.read_text()) == context
    assert json.loads(config_path.read_text()) == config
    assert json.loads(parent_path.read_text()) == parent
    assert json.loads((attempt / "status.json").read_text())["state"] == "DEPLOYED"


def test_batch_head_initialization_allows_shared_fold_ancestry(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    parent_attempt = fold_dir
    child_attempt = fold_dir / "child_attempt"
    items = []
    for index, attempt_dir in enumerate((parent_attempt, child_attempt)):
        context_path = tmp_path / f"context_{index}.json"
        config_path = tmp_path / f"config_{index}.json"
        parent_path = tmp_path / f"parent_{index}.json"
        context = {
            "attempt_id": f"20260823T00000{index}Z-test-head",
            "logical_run_name": f"test-head-{index}",
            "fold": 0,
            "seed": 1337,
        }
        parent = (
            None
            if index == 0
            else {
                "parent_attempt_id": "parent-1",
                "parent_checkpoint_path": "/tmp/best_model",
            }
        )
        items.append(
            {
                "attempt_dir": str(attempt_dir),
                "context_path": str(context_path),
                "config_path": str(config_path),
                "parent_path": str(parent_path),
                "context": context,
                "config": {"classifier": {"method": "logreg"}},
                "parent": parent,
            }
        )
    results = initialize_head_attempt_batch(items)
    assert [result["state"] for result in results] == ["DEPLOYED", "DEPLOYED"]
    assert (parent_attempt / "status.json").is_file()
    assert (child_attempt / "status.json").is_file()


def test_retry_plan_uses_fresh_output_identity_and_links_superseded_attempts() -> None:
    original = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(original)
    original["submission_phase"] = "cv"
    original["submission_complete"] = False
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


def test_training_chain_retry_selects_fresh_standalone_descendants() -> None:
    original = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(original)
    original["submission_complete"] = True
    for index, job in enumerate(original["jobs"]):
        job["job_ids"] = (
            {"train": str(5000 + index), "best_eval": str(6000 + index)}
            if job.get("kind") == "standalone_backbone"
            else {"job": str(7000 + index)}
        )
    failed_parent = next(job for job in original["jobs"] if job.get("kind") == "standalone_backbone")
    retry_deployment = {
        **_fake_deployment(),
        "deployment_id": "native-en-text-heads-v2-chain-retry",
        "git_commit": "47d163f44f72586a2f905798af885b15955be519",
    }
    retry = build_plan(
        stage="smoke",
        deployment=retry_deployment,
        experiment_id="exp-native-en-text-heads-v2-20260822",
        output_suffix="chainretry1",
        retry_from=original,
        retry_chain_job_keys=[failed_parent["job_key"]],
    )
    add_plan_indexes(retry)

    assert len(retry["jobs"]) == 3
    assert retry["counts"] == {
        "total": 4,
        "train": 1,
        "best_eval": 1,
        "postprocess": 0,
        "logreg": 1,
        "xgb_optuna100": 1,
    }
    assert retry["retry_chain_source_job_ids"] == {failed_parent["job_key"]: "5000"}
    assert all(job.get("supersedes_attempt_id") for job in retry["jobs"])
    assert all(not job.get("reuse_parent_artifacts") for job in retry["jobs"])
    assert all("chainretry1" in str(job.get("attempt_dir") or job.get("run_root")) for job in retry["jobs"])


def test_training_chain_retry_closes_over_merged_dependencies() -> None:
    original = build_plan(
        stage="smoke",
        deployment=_fake_deployment(),
        experiment_id="exp-native-en-text-heads-v2-20260822",
    )
    add_plan_indexes(original)
    original["submission_complete"] = True
    for index, job in enumerate(original["jobs"]):
        job["job_ids"] = (
            {"train": str(5000 + index), "best_eval": str(6000 + index)}
            if job.get("kind") == "standalone_backbone"
            else {"job": str(7000 + index)}
        )
    failed_parent = next(
        job
        for job in original["jobs"]
        if job.get("job_type") == "train" and job.get("kind") != "standalone_backbone"
    )
    retry_deployment = {
        **_fake_deployment(),
        "deployment_id": "native-en-text-heads-v2-merged-chain-retry",
        "git_commit": "47d163f44f72586a2f905798af885b15955be519",
    }
    retry = build_plan(
        stage="smoke",
        deployment=retry_deployment,
        experiment_id="exp-native-en-text-heads-v2-20260822",
        output_suffix="mergedchainretry1",
        retry_from=original,
        retry_chain_job_keys=[failed_parent["job_key"]],
    )

    assert [job["job_type"] for job in retry["jobs"]] == ["train", "evaluation", "hidden_classifier", "hidden_classifier"]
    assert retry["jobs"][1]["dependencies"] == [failed_parent["job_key"]]
    assert retry["counts"]["train"] == 1
    assert retry["counts"]["postprocess"] == 1
    assert retry["counts"]["logreg"] == 1
    assert retry["counts"]["xgb_optuna100"] == 1


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


def test_native_worker_environment_sets_cuda_reproducibility_defaults() -> None:
    env_script = (Path(__file__).resolve().parents[1] / "scripts/native_en_text_heads_env.sh").read_text(encoding="utf-8")
    assert 'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in env_script
    assert 'export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"' in env_script
    assert 'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in env_script


def test_final_epoch_derivation_uses_merged_epochs_launcher_without_config_override(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.native_en_text_heads as orchestration

    monkeypatch.setattr(orchestration, "PROJECT_ROOT", tmp_path)
    remote_root = orchestration.REMOTE_PROJECT_ROOT / "output_model" / "finalfix" / "text_only" / "merged"
    cv_jobs = []
    for fold, selected_epoch in enumerate((1, 2, 3, 4, 5)):
        attempt_dir = remote_root / f"cv_fold_{fold}"
        local_selection = tmp_path / attempt_dir.relative_to(orchestration.REMOTE_PROJECT_ROOT) / "logs"
        local_selection.mkdir(parents=True)
        (local_selection / "selected_checkpoint.json").write_text(
            json.dumps({"selected_epoch": selected_epoch}), encoding="utf-8"
        )
        cv_jobs.append(
            {
                "endpoint": "merged_cv",
                "job_type": "train",
                "condition": "native",
                "backbone": "qwen",
                "seed": 7,
                "fold": fold,
                "attempt_id": f"cv-{fold}",
                "attempt_dir": str(attempt_dir),
            }
        )
    final_job = {
        "endpoint": "merged_final",
        "job_type": "train",
        "condition": "native",
        "backbone": "qwen",
        "seed": 7,
        "fold": 0,
        "attempt_id": "final",
        "attempt_dir": str(remote_root / "final"),
        "overrides": ["--set=seed=7"],
        "config_payload": {"training": {"num_train_epochs": 20}},
    }
    plan = {
        "stage": "production",
        "deployment_id": "dep",
        "source_commit": "commit",
        "group_id": "group",
        "jobs": cv_jobs + [final_job],
    }

    derive_final_epochs(plan)

    assert final_job["epochs"] == 3
    assert not any("training.final_epoch_count" in token for token in final_job["overrides"])
    assert "final_epoch_count" not in final_job["config_payload"]["training"]
    assert plan["final_epoch_audit_path"].endswith("final_epoch_audit_dep.json")


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
def test_status_reconciles_collected_standalone_best_eval_sidecar(tmp_path: Path, monkeypatch) -> None:
    import tools.native_en_text_heads as orchestration

    monkeypatch.setattr(orchestration, "PROJECT_ROOT", tmp_path)
    remote_fold = orchestration.REMOTE_PROJECT_ROOT / "output_model" / "campaign" / "text_only" / "d3tec" / "run" / "fold_0"
    local_fold = tmp_path / "output_model" / "campaign" / "text_only" / "d3tec" / "run" / "fold_0"
    local_fold.mkdir(parents=True)
    attempt_id = "20260823T000000Z-test-standalone-sidecar-a1b2c3d4-e5f60718"
    (local_fold / "metadata.json").write_text(
        json.dumps({"attempt_id": attempt_id, "fold": 0}) + "\n"
    )
    write_status(local_fold / "status.json", StatusRecord(attempt_id, 0, state="RUNNING"))
    jobs_path = local_fold / "jobs.jsonl"
    for key, job_type, slurm_id in (
        ("train", "train", "101"),
        ("best_eval", "evaluation", "102"),
    ):
        append_job_event(
            jobs_path,
            new_job_event(
                job_key=key,
                job_type=job_type,
                event_type="SUBMITTED",
                attempt_id=attempt_id,
                fold=0,
                slurm_job_id=slurm_id,
                status="PENDING",
            ),
        )
    plan = {
        "jobs": [
            {
                "kind": "standalone_backbone",
                "fold_dir": str(remote_fold),
                "job_ids": {"train": "101", "best_eval": "102"},
            }
        ]
    }
    accounting = {
        "101": {"State": "COMPLETED", "ExitCode": "0:0"},
        "102": {"State": "COMPLETED", "ExitCode": "0:0"},
    }
    assert _reconcile_collected_standalone_sidecars(plan, accounting) == 2
    events = read_jsonl(jobs_path)
    assert {event["job_key"] for event in events if event["event_type"] == "COMPLETED"} == {
        "train",
        "best_eval",
    }
    assert json.loads((local_fold / "status.json").read_text())["state"] == "COMPLETED_ON_MN5"
