"""Shell-contract tests for native-versus-English study workers."""

from __future__ import annotations

import subprocess
from pathlib import Path

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
