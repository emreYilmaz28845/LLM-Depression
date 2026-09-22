from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.experiment_tracking.manifest_policy import (
    ManifestPolicyError,
    prebuilt_manifest_files,
    resolve_manifest_policy,
    validate_manifest_policy,
)
from src.experiment_tracking.submit import (
    SubmissionError,
    build_remote_submit_script,
    resolve_contract,
    verify_prebuilt_manifest_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOLED_CONFIG = yaml.safe_load(
    (
        PROJECT_ROOT
        / "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml"
    ).read_text(encoding="utf-8")
)
DAIC_CONFIG = yaml.safe_load(
    (
        PROJECT_ROOT
        / "configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml"
    ).read_text(encoding="utf-8")
)


def _deployment(commit: str = "a" * 40) -> dict:
    return {
        "schema_version": "audiollm.deployment.v1",
        "deployment_id": "feat-x-20260922T000000Z-abcdef01-12345678",
        "experiment_id": "feat-x-20260922",
        "git_commit": commit,
        "git_branch_at_deploy": "agent/feat-x",
        "git_dirty": False,
        "source_manifest_sha256": "b" * 64,
        "uncommitted_patch_sha256": "c" * 64,
        "deployed_code_path": "/gpfs/AudioLLM/deployments/feat-x-20260922T000000Z-abcdef01-12345678/code",
        "created_at_utc": "2026-09-22T00:00:00Z",
    }


def _config(config: dict, **overrides) -> dict:
    payload = {key: value for key, value in config.items()}
    payload.update(overrides)
    return payload


def _base_kwargs(dataset: str, modality: str, run_name: str) -> dict:
    return dict(
        experiment_id="feat-x-20260922",
        config_path_remote="/gpfs/AudioLLM/deployments/d1/code/configs/main/x.yaml",
        fold=0,
        seed=1337,
        run_name=run_name,
        campaign="promptcontext_v1_qwen38_likelihood",
        modality=modality,
        dataset=dataset,
        extra_overrides=[],
    )


def test_policy_defaults_to_build_and_rejects_unknown_values() -> None:
    assert resolve_manifest_policy(DAIC_CONFIG) == "build"
    assert resolve_manifest_policy({}) == "build"
    assert resolve_manifest_policy(POOLED_CONFIG) == "prebuilt"
    assert resolve_manifest_policy(DAIC_CONFIG, "prebuilt") == "prebuilt"
    with pytest.raises(ManifestPolicyError, match="Unsupported manifest policy"):
        resolve_manifest_policy({**DAIC_CONFIG, "manifest_policy": "later"})


def test_pooled_requires_prebuilt_and_ordinary_datasets_keep_build() -> None:
    with pytest.raises(ManifestPolicyError, match="requires manifest_policy=prebuilt"):
        validate_manifest_policy({**POOLED_CONFIG, "manifest_policy": "build"})
    with pytest.raises(ManifestPolicyError, match="requires manifest_policy=prebuilt"):
        validate_manifest_policy(_config(POOLED_CONFIG, manifest_policy=None))
    assert validate_manifest_policy(POOLED_CONFIG) == "prebuilt"
    assert validate_manifest_policy(DAIC_CONFIG) == "build"
    assert validate_manifest_policy(DAIC_CONFIG, "build") == "build"


def test_prebuilt_files_cover_manifest_and_split_inputs() -> None:
    files = prebuilt_manifest_files(
        manifest_dir="/rt/manifests/turkish", split_dir="/rt/splits/turkish", dataset="turkish"
    )
    assert files == {
        "manifest": "/rt/manifests/turkish/turkish_manifest.jsonl",
        "manifest_csv": "/rt/manifests/turkish/turkish_manifest.csv",
        "folds": "/rt/splits/turkish/turkish_folds.json",
        "split_metadata": "/rt/splits/turkish/turkish_manifest_metadata.json",
    }


def test_prebuilt_verification_fails_closed_on_missing_files() -> None:
    contract = {"manifest_policy": "prebuilt", "prebuilt_manifest_files": {"manifest": "/rt/a.jsonl"}}
    verify_prebuilt_manifest_files(contract, lambda path: True)
    with pytest.raises(SubmissionError, match="prebuilt manifest files are missing"):
        verify_prebuilt_manifest_files(contract, lambda path: False)
    # A build submission never requires prebuilt files.
    verify_prebuilt_manifest_files(
        {"manifest_policy": "build", "prebuilt_manifest_files": {}}, lambda path: False
    )


def test_contract_marks_prebuilt_and_exports_the_skip_flag() -> None:
    pooled = resolve_contract(
        deployment=_deployment(),
        config_dict=POOLED_CONFIG,
        **_base_kwargs("turkish", "text_only", "pooled_fold0"),
    )
    assert pooled["manifest_policy"] == "prebuilt"
    assert pooled["skip_manifest_build"] is True
    assert set(pooled["prebuilt_manifest_files"]) == {
        "manifest",
        "manifest_csv",
        "folds",
        "split_metadata",
    }
    assert all(
        path.startswith(pooled["manifest_dir"]) or path.startswith(pooled["split_dir"])
        for path in pooled["prebuilt_manifest_files"].values()
    )
    pooled_script = build_remote_submit_script(pooled)
    assert "export SKIP_MANIFEST_BUILD=1" in pooled_script

    daic = resolve_contract(
        deployment=_deployment(),
        config_dict=DAIC_CONFIG,
        **_base_kwargs("daic", "text_only", "daic_fold0"),
    )
    assert daic["manifest_policy"] == "build"
    assert daic["skip_manifest_build"] is False
    assert daic["prebuilt_manifest_files"] == {}
    assert "export SKIP_MANIFEST_BUILD=0" in build_remote_submit_script(daic)


def test_pooled_contract_refuses_the_build_route() -> None:
    with pytest.raises(SubmissionError, match="requires manifest_policy=prebuilt"):
        resolve_contract(
            deployment=_deployment(),
            config_dict={**POOLED_CONFIG, "manifest_policy": "build"},
            **_base_kwargs("turkish", "text_only", "pooled_fold0"),
        )
    with pytest.raises(SubmissionError, match="Unsupported manifest policy"):
        resolve_contract(
            deployment=_deployment(),
            config_dict=DAIC_CONFIG,
            manifest_policy="sometimes",
            **_base_kwargs("daic", "text_only", "daic_fold0"),
        )


def test_pr259_training_shape_arguments_still_work() -> None:
    four_gpu = resolve_contract(
        deployment=_deployment(),
        config_dict=DAIC_CONFIG,
        train_nodes=1,
        train_gpus_per_node=4,
        env_activate="/gpfs/venvs/qwen38/bin/activate",
        **_base_kwargs("daic", "text_only", "daic_fold0"),
    )
    assert four_gpu["training_shape"] == {
        "strategy": "fsdp",
        "nodes": 1,
        "gpus_per_node": 4,
        "world_size": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "effective_global_batch_size": 128,
    }
    assert four_gpu["env_activate"].endswith("bin/activate")
    assert "export ENV_ACTIVATE=" in build_remote_submit_script(four_gpu)

    # The caller resolves the config with the overrides applied, and passes the
    # same tokens through for the workers (exp.py does both).
    eight_gpu_config = {
        **DAIC_CONFIG,
        "training": {**DAIC_CONFIG["training"], "gradient_accumulation_steps": 16},
    }
    eight_gpu_kwargs = _base_kwargs("daic", "text_only", "daic_fold0")
    eight_gpu_kwargs["extra_overrides"] = ["--set=training.gradient_accumulation_steps=16"]
    eight_gpu = resolve_contract(
        deployment=_deployment(),
        config_dict=eight_gpu_config,
        train_nodes=2,
        train_gpus_per_node=4,
        **eight_gpu_kwargs,
    )
    assert eight_gpu["training_shape"]["world_size"] == 8
    assert eight_gpu["training_shape"]["effective_global_batch_size"] == 128
    with pytest.raises(SubmissionError, match="effective global batch of 128"):
        resolve_contract(
            deployment=_deployment(),
            config_dict=DAIC_CONFIG,
            train_nodes=2,
            train_gpus_per_node=4,
            **_base_kwargs("daic", "text_only", "daic_fold0"),
        )


def test_train_and_evaluation_jobs_keep_distinct_shapes() -> None:
    contract = resolve_contract(
        deployment=_deployment(),
        config_dict=DAIC_CONFIG,
        **_base_kwargs("daic", "text_only", "daic_fold0"),
    )
    train_job, eval_job = contract["job_graph"]
    assert train_job["script"] == "scripts/run_train_slurm.sh"
    assert "4 task(s)/node" in train_job["shape"]
    assert eval_job["script"] == "scripts/run_eval_slurm.sh"
    assert eval_job["shape"] == "1 node, 1 task, 1 H100"
    assert eval_job["depends_on"] == [train_job["job_key"]]
    assert eval_job["checkpoint_dir"] == contract["checkpoint_dir"]


def test_submit_cli_keeps_the_pr259_flags_and_adds_manifest_policy() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/exp.py", "submit", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in ("--train-nodes", "--train-gpus-per-node", "--env-activate", "--manifest-policy"):
        assert flag in proc.stdout
    for value in ("build", "prebuilt"):
        assert value in proc.stdout
