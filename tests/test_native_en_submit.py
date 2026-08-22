"""Offline tests for native-versus-English submission builders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import tools.native_en_submit as ns
from src.features import logreg_head_campaign
from src.features.posthoc_head_campaign import load_task_spec as load_posthoc_spec


def _merged_config() -> dict:
    return yaml.safe_load(
        (ns_path_config()).read_text(encoding="utf-8")
    )


def ns_path_config() -> Path:
    root = Path(__file__).resolve().parents[1]
    return (
        root
        / "configs/experiments/merged/symmetric_merged_text_heads_native_qwen.yaml"
    )


class TestDerivedConfig:
    def test_seed_replaced_and_project_root_rewritten(self) -> None:
        config = _merged_config()
        text = ns.materialize_merged_config(config, seed=7)
        doc = yaml.safe_load(text)
        assert doc["seed"] == 7
        assert "${PROJECT_ROOT}" not in text
        assert text.count(ns.REMOTE_PROJECT_BASE) >= 2

    @pytest.mark.parametrize("seed", [1, 99999])
    def test_out_of_scope_seeds_refused(self, seed: int) -> None:
        with pytest.raises(ValueError, match="training seed"):
            ns.materialize_merged_config(_merged_config(), seed=seed)

    def test_missing_split_seed_refused(self) -> None:
        config = _merged_config()
        del config["protocol_settings"]["split_seed"]
        with pytest.raises(ValueError, match="split_seed"):
            ns.materialize_merged_config(config, seed=1337)

    def test_wrong_fixed_head_seed_refused(self) -> None:
        config = _merged_config()
        config["heads"]["fixed_seed"] = 7
        with pytest.raises(ValueError, match="fixed_seed"):
            ns.materialize_merged_config(config, seed=1337)

    def test_derived_configs_differ_only_in_top_level_seed(self) -> None:
        base = ns.materialize_merged_config(_merged_config(), seed=7)
        other = ns.materialize_merged_config(_merged_config(), seed=2024)
        import difflib

        delta = [
            line
            for line in difflib.unified_diff(base.splitlines(), other.splitlines())
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        assert delta == ["-seed: 7", "+seed: 2024"]


class TestPathBuilders:
    def test_campaign_names_match_group_definition(self) -> None:
        expected = {
            ("native", "qwen"): "text_heads_native_v1",
            ("native", "gemma4"): "text_heads_native_gemma4_v1",
            ("english", "qwen"): "text_heads_en_v1",
            ("english", "gemma4"): "text_heads_en_gemma4_v1",
        }
        for (condition, backbone), campaign in expected.items():
            assert ns.campaign_for(condition, backbone) == campaign
        with pytest.raises(ValueError):
            ns.campaign_for("klingon", "qwen")

    def test_merged_fold_paths_shape(self) -> None:
        paths = ns.merged_fold_paths(
            condition="english", backbone="gemma4", run_id="tmh-s7", stage="cv", fold=3
        )
        assert paths["fold_dir"].endswith(
            "/output_model/symmetric_merged/native_en_text_heads_v1/"
            "english_gemma4_text_only/tmh-s7/cv/fold_3"
        )
        assert paths["checkpoint_dir"].endswith("/fold_3/best_model")
        assert "/outputs/symmetric_merged/" in paths["features_dir"]
        assert paths["features_dir"].endswith("/fold_3/features")

    def test_stage_folds(self) -> None:
        assert ns.merged_stage_folds("cv") == [0, 1, 2, 3, 4]
        assert ns.merged_stage_folds("final") == [0]
        assert ns.merged_stage_folds("smoke") == [0]
        with pytest.raises(ValueError):
            ns.merged_stage_folds("nope")

    def test_rounded_median_epoch_matches_locked_rule(self) -> None:
        assert ns.rounded_median_epoch([1, 4, 3, 2, 3]) == 3
        assert ns.rounded_median_epoch([5]) == 5
        assert ns.rounded_median_epoch([20, 21]) == 20  # clamped high
        assert ns.rounded_median_epoch([0]) == 1  # clamped low
        with pytest.raises(ValueError):
            ns.rounded_median_epoch([])

    def test_standalone_attempt_layout_keeps_importable_position(self) -> None:
        path = ns.standalone_attempt_path(
            campaign="text_heads_en_gemma4_v1",
            dataset="turkish",
            run_name="tnh-en-gemma4-turkish-s2024",
            fold=2,
            experiment_id=ns.OPTUNA_EXPERIMENT_ID,
        )
        parts = path.split("/")
        idx = parts.index("output_model")
        assert parts[idx + 1] == "text_heads_en_gemma4_v1_optuna100"
        assert parts[idx + 2] == "text_only"
        assert parts[idx + 3] == "turkish"
        assert path.endswith(
            "fold_2/xgb_optuna100_harmonized_v1"
        )


class TestSpecBuilders:
    def test_logreg_spec_passes_campaign_validator(self, tmp_path: Path) -> None:
        spec = ns.build_logreg_task_spec(
            family="standalone",
            backend="gemma4",
            dataset="d3tec",
            modality="text_only",
            condition="native_gemma4",
            fold=1,
            seed=2024,
            stage=None,
            cache_dir=str(tmp_path / "cache"),
            group_id="g",
            run_name="r",
            branch="b",
            merged_sha="a" * 40,
            parent_checkpoint_path="/gpfs/x/run/fold_1/best_model",
        )
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        loaded = logreg_head_campaign.load_task_spec(path)
        assert loaded["head_seed"] == 1337
        parent_dir = spec["parent"]["parent_fold_dir"]
        assert parent_dir.endswith("/fold_1")

    def test_optuna_production_spec_passes_posthoc_validator(self, tmp_path: Path) -> None:
        spec = ns.build_optuna_task_spec(
            family="merged",
            backend="qwen",
            dataset="merged",
            modality="text_only",
            condition="english_qwen",
            fold=0,
            seed=7,
            stage="cv",
            cache_dir=str(tmp_path / "features"),
            group_id="g",
            run_name="tmh-en-qwen-s7-cv",
            branch="b",
            merged_sha="a" * 40,
            parent_checkpoint_path="/gpfs/x/tmh-s7/cv/fold_0/best_model",
            target_trials=100,
        )
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        loaded = load_posthoc_spec(path)
        assert loaded["target_trials"] == 100
        assert spec["evaluation_qualifiers"]["split_protocol"] == (
            "symmetric_merged_cv_outer_holdout"
        )

    def test_optuna_smoke_spec_accepted_only_at_stage_smoke(self, tmp_path: Path) -> None:
        kwargs = dict(
            family="merged",
            backend="qwen",
            dataset="merged",
            modality="text_only",
            condition="english_qwen",
            fold=0,
            seed=1337,
            stage="smoke",
            cache_dir=str(tmp_path / "features"),
            group_id="g",
            run_name="tmh-smoke",
            branch="b",
            merged_sha="a" * 40,
            parent_checkpoint_path=None,
            target_trials=2,
        )
        spec = ns.build_optuna_task_spec(**kwargs)
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        assert load_posthoc_spec(path)["target_trials"] == 2
        bad = dict(kwargs, stage="cv")
        (tmp_path / "bad.json").write_text(json.dumps(ns.build_optuna_task_spec(**bad)))
        from src.features.posthoc_head_campaign import PosthocError

        with pytest.raises(PosthocError, match="stage=smoke"):
            load_posthoc_spec(tmp_path / "bad.json")


class TestScriptRenderers:
    def test_merged_chain_script_contract(self) -> None:
        script = ns.render_merged_chain_script(
            code_path="/gpfs/deployments/d1/code",
            derived_config_path="/gpfs/runtime/configs/r/seed_7/merged.yaml",
            derived_config_sha256="c" * 64,
            condition="native",
            backbone="qwen",
            run_id="tmh-native-qwen-s7",
            stage="cv",
            fold=2,
            fold_dir="/gpfs/out/run/cv/fold_2",
            checkpoint_dir="/gpfs/out/run/cv/fold_2/best_model",
            features_dir="/gpfs/outs/run/cv/fold_2/features",
            source_commit="a" * 40,
            context_path="/gpfs/runtime/contexts/att/fold_2/context.json",
        )
        assert "--delete" not in script
        assert script.count("sbatch --parsable --chdir=\"$CODE\"") == 3
        assert "--dependency=afterok:$TRAIN_ID" in script
        assert "--dependency=afterok:$POST_ID" in script
        assert "export NPROC_PER_NODE=4" in script
        assert "export TRIALS=0" in script
        assert "run_symmetric_merged_train_slurm.sh" in script
        assert "run_symmetric_merged_postprocess_slurm.sh" in script
        assert "run_symmetric_merged_head_slurm.sh" in script
        # sidecar skeleton writes all three SUBMITTED events
        assert '("train"' in script and '"postprocess"' in script and '"head"' in script
        # no EPOCHS export on cv chains
        assert "\nexport EPOCHS=" not in script

    def test_final_chain_exports_derived_epochs(self) -> None:
        script = ns.render_merged_chain_script(
            code_path="/code",
            derived_config_path="/cfg.yaml",
            derived_config_sha256="c" * 64,
            condition="native",
            backbone="gemma4",
            run_id="r",
            stage="final",
            fold=0,
            fold_dir="/out/final/fold_0",
            checkpoint_dir="/out/final/fold_0/best_model",
            features_dir="/outs/final/fold_0/features",
            source_commit="a" * 40,
            context_path="/runtime/ctx.json",
            epochs=9,
        )
        assert "\nexport EPOCHS=9\n" in script
        assert "gemma4_12b_tf5_14_1" in script
        assert "GEMMA4_MODEL_PATH:-" in script

    def test_study_job_script_dependencies_and_echo(self) -> None:
        script = ns.render_study_job_script(
            code_path="/code",
            worker_relpath="scripts/run_native_en_optuna100_attempt_slurm.sh",
            job_name="nmq-optuna-x",
            exports=[("TARGET_TRIALS", "100"), ("MODE", "standalone")],
            after_job_ids=["12345"],
            echo_label="optuna attempt",
        )
        assert "--dependency=afterok:12345" in script
        assert "TARGET_TRIALS" in script and "100" in script
        assert 'echo "Submitted optuna attempt job: $JOB_ID"' in script
        assert "--delete" not in script
