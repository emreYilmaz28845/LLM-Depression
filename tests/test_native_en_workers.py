"""Shell-contract tests for native-versus-English study workers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGREG_SCRIPT = PROJECT_ROOT / "scripts/run_native_en_logreg_attempt_slurm.sh"
OPTUNA_SCRIPT = PROJECT_ROOT / "scripts/run_native_en_optuna100_attempt_slurm.sh"


def _bash_syntax(path: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class TestLogregAttemptWorker:
    def test_syntax(self) -> None:
        _bash_syntax(LOGREG_SCRIPT)

    def test_gpu_shape_and_required_env(self) -> None:
        text = LOGREG_SCRIPT.read_text()
        assert "--gres=gpu:1" in text
        assert 'PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?Set PARENT_FOLD_DIR}"' in text
        assert 'CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"' in text
        assert 'ATTEMPT_DIR="${ATTEMPT_DIR:?Set ATTEMPT_DIR}"' in text
        assert 'TASK_SPEC_PATH="${TASK_SPEC_PATH:?Set TASK_SPEC_PATH}"' in text

    def test_gemma_backbone_requires_dedicated_env_and_model_path(self) -> None:
        text = LOGREG_SCRIPT.read_text()
        assert 'BACKBONE=gemma4 requires MODEL_PATH' in text
        assert "gemma4_12b_tf5_14_1" in text

    def test_classifier_stage_runs_in_qwen_env_with_policy(self) -> None:
        text = LOGREG_SCRIPT.read_text()
        assert "--variants logreg_raw" in text
        assert "--seed 1337" in text
        assert "--sampling-mode none" in text
        assert "--head-backend-policy harmonized_hidden_logreg_raw_v1" in text
        # The classifier stage sources the Qwen environment after extraction.
        classifier_pos = text.find("baselines/qwen_hidden_classifier.py")
        qwen_source_pos = text.rfind("source \"$QWEN_ENV\"")
        extract_pos = text.find("extract_qwen_hidden.py")
        assert -1 < extract_pos < qwen_source_pos < classifier_pos

    def test_attempt_lifecycle_commands_present(self) -> None:
        text = LOGREG_SCRIPT.read_text()
        for command in (
            "create-attempt --task-spec",
            "mark-deployed",
            "record-job",
            "transition --to-state SUBMITTED",
            "transition --to-state RUNNING",
            "materialize-mn5-evidence",
        ):
            assert command in text, command

    def test_no_destructive_operations(self) -> None:
        for path in (LOGREG_SCRIPT, OPTUNA_SCRIPT):
            text = path.read_text()
            assert "--delete" not in text
            assert "rm -rf" not in text


class TestOptunaAttemptWorker:
    def test_syntax(self) -> None:
        _bash_syntax(OPTUNA_SCRIPT)

    def test_cpu_only_shape(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert "--gres" not in text
        assert "--cpus-per-task=20" in text

    def test_production_default_is_exactly_hundred_trials(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert 'TARGET_TRIALS="${TARGET_TRIALS:-100}"' in text

    def test_standalone_branch_pins_locked_protocol(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert "--seed 1337" in text
        assert "--inner-seed 1337" in text
        assert "--protocol-profile harmonized_optuna100_v1" in text
        assert "--experiment-id xgb_optuna100_harmonized_v1" in text
        assert "--target-trials \"$TARGET_TRIALS\"" in text

    def test_merged_branch_requires_config_and_run_id(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert 'MERGED_CONFIG:?MODE=merged requires MERGED_CONFIG' in text
        assert 'RUN_ID:?MODE=merged requires RUN_ID' in text
        assert "src/merged/optuna100.py" in text

    def test_uses_posthoc_campaign_for_optuna_attempts(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert "tools/posthoc_head_campaign.py" in text


class TestVendoredDepsResolution:
    """The deployed code tree has no .deps (gitignored); pinned optuna/
    xgboost/sklearn must resolve from the PERMANENT tree's vendored deps."""

    PERMANENT_DEPS = (
        "QWEN_HIDDEN_DEPS=\"${QWEN_HIDDEN_DEPS:-"
        "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}\""
    )

    def test_optuna_worker_defaults_to_permanent_deps(self) -> None:
        text = OPTUNA_SCRIPT.read_text()
        assert self.PERMANENT_DEPS in text
        assert "$PROJECT_ROOT/.deps/qwen_hidden" not in text

    def test_preflight_worker_defaults_to_permanent_deps(self) -> None:
        text = (PROJECT_ROOT / "scripts/run_native_en_preflight_slurm.sh").read_text()
        assert "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden" in text
        assert "$CODE/.deps/" not in text

    def test_logreg_worker_defaults_to_permanent_deps(self) -> None:
        text = LOGREG_SCRIPT.read_text()
        assert "$PROJECT_ROOT/.deps/qwen_hidden" not in text
        assert "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden" in text


class TestMergedSidecarJobTypes:
    def test_sidecar_writer_uses_valid_job_type_enum(self) -> None:
        import tools.native_en_submit as ns2

        script = ns2.render_merged_chain_script(
            code_path="/code", derived_config_path="/cfg.yaml",
            derived_config_sha256="c" * 64, condition="native", backbone="qwen",
            run_id="r", stage="smoke", fold=0,
            fold_dir="/out/fold_0", checkpoint_dir="/out/fold_0/best_model",
            features_dir="/outs/fold_0/features", source_commit="a" * 40,
            context_path="/ctx.json",
        )
        assert '"train"' in script
        assert '"evaluation"' in script
        assert '"hidden_classifier"' in script
        for invalid in ("merged_train", "merged_postprocess", "merged_head"):
            assert f'"{invalid}"' not in script


class TestGemmaEnvForwarding:
    def test_builder_emits_env_exports(self) -> None:
        from src.experiment_tracking.submit import build_remote_submit_script

        contract = {
            "deployed_code_path": "/code",
            "config_path_remote": "/code/configs/x.yaml",
            "fold": 0,
            "run_name": "r",
            "overrides": [],
            "overrides_b64": "",
            "log_root_train": "/logs/train",
            "context_path": "/ctx.json",
            "env_exports": {
                "ENV_ACTIVATE": "/gpfs/venvs/gemma4_12b_tf5_14_1/bin/activate",
                "MODEL_PATH": "/gpfs/models/gemma-4-12B-it/707f0a3b8a3c7ad5",
            },
        }
        script = build_remote_submit_script(contract)
        assert "export ENV_ACTIVATE=/gpfs/venvs/gemma4_12b_tf5_14_1/bin/activate" in script
        assert "export MODEL_PATH=/gpfs/models/gemma-4-12B-it/707f0a3b8a3c7ad5" in script

    def test_resolve_contract_carries_extra_env(self, tmp_path) -> None:
        from src.experiment_tracking.submit import resolve_contract

        deployment = {
            "deployment_id": "dep", "git_commit": "a" * 40,
            "git_branch_at_deploy": "b", "git_dirty": False,
            "source_manifest_sha256": "s" * 64,
            "deployed_code_path": "/code",
            "experiment_id": "exp",
        }
        contract = resolve_contract(
            experiment_id="exp", deployment=deployment,
            config_path_remote="/code/configs/main/x.yaml",
            config_dict={"dataset": "d3tec",
                         "evaluation": {"evaluation_view": "v",
                                        "sample_prediction_mode": "original_teacher_forced"}},
            fold=0, seed=1337, run_name="r", campaign="c", modality="text_only",
            dataset="d3tec", extra_overrides=[],
            extra_env={"MODEL_PATH": "/m"},
        )
        assert contract["env_exports"] == {"MODEL_PATH": "/m"}


class TestWorkerManifestOverrideForms:
    """b64 token arrays carry single-token --set=k=v; the whitespace fallback
    carries two tokens. Both must reach build_manifest."""

    @pytest.mark.parametrize("script", ["scripts/run_train_slurm.sh", "scripts/run_eval_slurm.sh"])
    def test_loop_handles_both_override_forms(self, script: str) -> None:
        text = (PROJECT_ROOT / script).read_text()
        assert '= "--set" ]' in text
        assert "elif [[ \"${OVERRIDE_ARGS[$i]}\" == --set=* ]]; then" in text

    def test_behavioral_pairing(self) -> None:
        import subprocess

        snippet = r'''
OVERRIDE_ARGS=("--set=a.b=1" "--set" "c.d=2" "ignored")
MANIFEST_CMD=("python" "build.py")
i=0
while [ $i -lt ${#OVERRIDE_ARGS[@]} ]; do
    if [ "${OVERRIDE_ARGS[$i]}" = "--set" ] && [ $((i + 1)) -lt ${#OVERRIDE_ARGS[@]} ]; then
        MANIFEST_CMD+=("${OVERRIDE_ARGS[$i]}" "${OVERRIDE_ARGS[$((i + 1))]}")
        i=$((i + 2))
    elif [[ "${OVERRIDE_ARGS[$i]}" == --set=* ]]; then
        MANIFEST_CMD+=("${OVERRIDE_ARGS[$i]}")
        i=$((i + 1))
    else
        i=$((i + 1))
    fi
done
printf '%s\n' "${MANIFEST_CMD[@]}"
'''
        out = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True).stdout.split()
        assert out == ["python", "build.py", "--set=a.b=1", "--set", "c.d=2"]
