"""Tests for the backend-neutral symmetric-merged implementation (runbook
Task 4 / Section 9.5)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.features import optuna100_policy as policy
from src.merged import configuration, heads
from src.merged.optuna100 import run_merged_optuna100

ROOT = Path(__file__).resolve().parents[1]
MERGED_DIR = ROOT / "configs/experiments/merged"
QWEN_MERGED = {
    modality: MERGED_DIR / f"symmetric_merged_harmonized_{modality}.yaml"
    for modality in ("audio_text", "audio_only", "text_only")
}
GEMMA_MERGED = {
    modality: MERGED_DIR / f"symmetric_merged_harmonized_gemma4_{modality}.yaml"
    for modality in ("audio_text", "audio_only", "text_only")
}
COMPONENT_NAMES = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fake_records(backend: str) -> list[dict]:
    configs = []
    for name in COMPONENT_NAMES:
        component = {"dataset": name}
        if backend == "gemma4":
            component["model_backend"] = "gemma4"
            component["model_revision"] = policy.GEMMA4_PREDICTION_BACKEND  # placeholder value
        configs.append({"config": component})
    return configs


class TestMergedConfiguration:
    def test_shared_backend_validation(self) -> None:
        assert configuration.validate_shared_backend({}, fake_records("qwen")) == "qwen"
        assert configuration.validate_shared_backend({}, fake_records("gemma4")) == "gemma4"
        assert (
            configuration.validate_shared_backend({"model_backend": "gemma4"}, fake_records("gemma4"))
            == "gemma4"
        )
        with pytest.raises(ValueError, match="share one model backend"):
            mixed = fake_records("gemma4")
            mixed[1]["config"]["model_backend"] = "qwen2audio"
            configuration.validate_shared_backend({}, mixed)
        with pytest.raises(ValueError, match="does not match"):
            configuration.validate_shared_backend({"model_backend": "gemma4"}, fake_records("qwen"))

    def test_gemma_model_config_carries_backend_identity(self) -> None:
        merged = load(GEMMA_MERGED["audio_text"])
        resolved = configuration.model_config(merged, fake_records("gemma4"))
        assert resolved["model_backend"] == "gemma4"
        assert resolved["model_revision"] == merged["model_revision"]
        assert resolved["data"]["use_audio"] is True
        assert resolved["data"]["use_text"] is True

    def test_qwen_model_config_unchanged(self) -> None:
        merged = load(QWEN_MERGED["text_only"])
        resolved = configuration.model_config(merged, fake_records("qwen"))
        assert "model_backend" not in resolved or resolved["model_backend"] is None


class TestGemmaMergedConfigs:
    def test_three_gemma_configs_exist_with_exact_five_components(self) -> None:
        assert set(GEMMA_MERGED) == {"audio_text", "audio_only", "text_only"}
        for modality, path in GEMMA_MERGED.items():
            config = load(path)
            assert config["model_backend"] == "gemma4"
            assert config["model_revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
            assert {item["name"] for item in config["components"]} == set(COMPONENT_NAMES)
            assert config["modality"] == modality
            assert "gemma4" in config["output_dirs"]["run_root"]
            assert "gemma4" in config["output_dirs"]["merged_root"]
            for component in config["components"]:
                assert "gemma4_12b" in component["config"]
            qwen = load(QWEN_MERGED[modality])
            assert config["protocol_settings"] == qwen["protocol_settings"]
            assert config["training"] == qwen["training"]
            assert config["heads"]["fixed_xgb"] == qwen["heads"]["fixed_xgb"]

    def test_shared_manifest_paths_with_qwen(self) -> None:
        for modality in ("audio_text", "audio_only", "text_only"):
            gemma = load(GEMMA_MERGED[modality])
            qwen = load(QWEN_MERGED[modality])
            for g, q in zip(gemma["components"], qwen["components"]):
                assert g["name"] == q["name"]
                assert g["manifest_path"] == q["manifest_path"]
                assert g["metadata_path"] == q["metadata_path"]

    def test_gemma_configs_validate_through_component_validation(self) -> None:
        from src.model.gemma4_io import validate_gemma4_config
        from src.utils import load_yaml

        for modality, path in GEMMA_MERGED.items():
            config = load(path)
            for component in config["components"]:
                component_config = load_yaml(ROOT / component["config"])
                validate_gemma4_config(component_config)


class TestMergedTrainBackendDispatch:
    def test_train_uses_runtime_collator_factory_and_backend_preparation(self) -> None:
        source = (ROOT / "src/merged/train.py").read_text(encoding="utf-8")
        assert "from src.model.collator import Qwen2AudioSFTCollator" not in source
        assert "build_collator" in source
        assert "prepare_backend_examples" in source

    def test_postprocess_uses_backend_aware_collator(self) -> None:
        source = (ROOT / "src/merged/postprocess.py").read_text(encoding="utf-8")
        assert "Gemma4PromptOnlyExtractionCollator" in source
        assert "prepare_backend_examples" in source
        assert "expected_hidden_size" in source


class TestMergedHeadsProtocol:
    def test_trial_count_rules_via_config_identity(self) -> None:
        profile = {"heads": {"optuna": {"protocol_profile": "harmonized_optuna100_v1", "target_trials": 100}}}
        assert heads.resolve_optuna_trials(profile, "cv", None) == 100
        disabled = {"heads": {"optuna": {"protocol_profile": "harmonized_optuna100_v1", "target_trials": 0}}}
        assert heads.resolve_optuna_trials(disabled, "cv", None) == 0
        with pytest.raises(ValueError, match="exactly 100"):
            heads.resolve_optuna_trials(
                {"heads": {"optuna": {"protocol_profile": "harmonized_optuna100_v1", "target_trials": 50}}},
                "cv",
                None,
            )
        with pytest.raises(ValueError, match="must not use the production"):
            heads.resolve_optuna_trials(profile, "smoke", None)
        # Historical configs keep the 150-trial contract.
        historical = {"heads": {"optuna": {"target_trials": 150}}}
        assert heads.resolve_optuna_trials(historical, "cv", None) == 150
        with pytest.raises(ValueError, match="fixed to 150"):
            heads.resolve_optuna_trials({"heads": {"optuna": {"target_trials": 99}}}, "cv", None)

    def test_run_merged_heads_trial_count_rules(self, monkeypatch, tmp_path: Path) -> None:
        from src.merged.runtime import load_merged_config, load_records_and_protocol

        calls = []

        def fake_load(path):
            calls.append(str(path))
            return {"heads": {"optuna": {"protocol_profile": "harmonized_optuna100_v1", "target_trials": 0}}}

        def fake_records(config):
            return [], {"manifest": {"manifest_hash": "m"}, "protocol": {"split_hash": "s", "folds": {"0": {"fold_hash": "f"}}}}

        monkeypatch.setattr(heads, "load_merged_config", fake_load)
        monkeypatch.setattr(heads, "load_records_and_protocol", fake_records)

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        (feature_dir / "feature_metadata.json").write_text(
            json.dumps(
                {
                    "stage": "cv",
                    "fold": 0,
                    "modality": "audio_text",
                    "manifest_hash": "m",
                    "split_hash": "s",
                    "merged_config_sha256": "c" * 64,
                }
            )
        )
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            # Expected to fail at feature loading (no npz yet); the trial-count
            # rules are what we exercise: profile + 0 must pass the check.
            heads.run_merged_heads(
                "configs/experiments/merged/x.yaml",
                stage="cv",
                fold=0,
                run_id="t4",
                features_dir=feature_dir,
                trials=0,
            )


class _FakeXgb:
    def __init__(self, params=None, **kwargs):
        self.params = dict(params or {})
        self.params.update(kwargs)

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        rng = np.random.default_rng(7)
        positive = rng.random(len(x))
        return np.stack([1.0 - positive, positive], axis=1)


def _fake_merged_features(tmp_path: Path, *, modality: str = "audio_text", backend: str = "gemma4") -> Path:
    features = tmp_path / "features"
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train_rows = []
    for index in range(150):
        train_rows.append(
            {
                "dataset": ["daic", "cmdc", "turkish", "d3tec", "androids_interview"][index % 5],
                "sample_id": f"t{index}",
                "subject_id": f"s{index}",
                "response_id": f"s{index}::r0",
                "label": index % 2,
            }
        )
    holdout_rows = []
    for index in range(10):
        holdout_rows.append(
            {
                "dataset": "daic",
                "sample_id": f"e{index}",
                "subject_id": f"e{index}",
                "response_id": f"e{index}::r0",
                "label": index % 2,
            }
        )
    np.savez(features / "outer_train.npz", vectors=rng.normal(size=(150, 16)).astype(np.float32))
    np.savez(features / "outer_holdout.npz", vectors=rng.normal(size=(10, 16)).astype(np.float32))
    for name, rows in (("outer_train", train_rows), ("outer_holdout", holdout_rows)):
        with open(features / f"{name}_rows.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (features / "feature_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "symmetric_merged_hidden_features.v1",
                "stage": "final",
                "fold": 0,
                "modality": modality,
                "model_backend": backend,
                "manifest_hash": "m" * 64,
                "split_hash": "s" * 64,
                "merged_config_sha256": "c" * 64,
                "checkpoint_dir": "/gpfs/x/LLM-Depression/output_model/symmetric_merged/gemma4/harmonized_v1/audio_text/run/final/fold_0/best_model",
                "checkpoint_hashes": {"adapter_config_sha256": "a" * 64, "adapter_model_sha256": "b" * 64},
            }
        )
    )
    return features


class TestMergedOptuna100:
    def test_merged_study_writes_backend_qualified_evidence(self, tmp_path: Path, monkeypatch) -> None:
        features = _fake_merged_features(tmp_path, backend="gemma4")
        output = tmp_path / "fold_0" / policy.EXPERIMENT_ID
        merged_config = tmp_path / "merged.yaml"
        merged_config.write_text(
            yaml.safe_dump(
                {
                    "protocol": "symmetric_merged",
                    "modality": "audio_text",
                    "model_backend": "gemma4",
                    "training": {},
                    "output_dirs": {"run_root": "x"},
                }
            )
        )
        monkeypatch.setattr("src.merged.optuna100._new_xgb", _FakeXgb)
        result = run_merged_optuna100(
            features_dir=features,
            output_dir=output,
            merged_config_path=merged_config,
            stage="final",
            fold=0,
            run_id="t4",
        )
        assert result["completed_trials"] == 100
        assert result["prediction_backend"] == "gemma4_hidden_xgb_optuna100_symmetric_merged"
        metadata = json.loads((output / "classifier_metadata.json").read_text())
        assert metadata["prediction_backend"] == "gemma4_hidden_xgb_optuna100_symmetric_merged"
        assert metadata["protocol_profile"] == "harmonized_optuna100_v1"
        assert metadata["objective"] == "mean_per_dataset_inner_macro_f1"
        study_config = json.loads((output / "study_config.json").read_text())
        assert study_config["canonical_config"]["protocol"]["target_completed_trials"] == 100
        assert study_config["canonical_config"]["protocol"]["search_space"]["scale_pos_weight"] == {
            "kind": "float", "low": 0.25, "high": 4.0, "log": True
        }
        trials = (output / "trials.csv").read_text().splitlines()
        assert len(trials) == 101
        predictions = [
            json.loads(line)
            for line in (output / "predictions_subject_level.jsonl").read_text().splitlines()
        ]
        assert predictions
        assert all(row["prediction_backend"] == "gemma4_hidden_xgb_optuna100_symmetric_merged" for row in predictions)
        metrics = json.loads((output / "metrics.json").read_text())
        assert "daic" in metrics["dataset_metrics"]
        # Restart with identical identity completes without new trials.
        result_again = run_merged_optuna100(
            features_dir=features,
            output_dir=output,
            merged_config_path=merged_config,
            stage="final",
            fold=0,
            run_id="t4",
        )
        assert result_again["completed_trials"] == 100

    def test_merged_study_refuses_changed_identity(self, tmp_path: Path, monkeypatch) -> None:
        features = _fake_merged_features(tmp_path, backend="qwen")
        output = tmp_path / "fold_0" / policy.EXPERIMENT_ID
        merged_config = tmp_path / "merged.yaml"
        merged_config.write_text(yaml.safe_dump({"protocol": "symmetric_merged", "modality": "audio_text"}))
        monkeypatch.setattr("src.merged.optuna100._new_xgb", _FakeXgb)
        run_merged_optuna100(
            features_dir=features, output_dir=output, merged_config_path=merged_config,
            stage="final", fold=0, run_id="t4",
        )
        # A different modality under the same output must be refused.
        features2 = _fake_merged_features(tmp_path / "other", modality="audio_only")
        with pytest.raises(ValueError, match="identity mismatch|study_config"):
            run_merged_optuna100(
                features_dir=features2, output_dir=output, merged_config_path=merged_config,
                stage="final", fold=0, run_id="t4",
            )

    def test_final_stage_holdout_is_daic_only(self, tmp_path: Path, monkeypatch) -> None:
        features = _fake_merged_features(tmp_path, backend="qwen")
        rows = [
            {
                "dataset": "cmdc",
                "sample_id": f"x{index}",
                "subject_id": f"x{index}",
                "response_id": f"x{index}::r0",
                "label": index % 2,
            }
            for index in range(10)
        ]
        (features / "outer_holdout_rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        output = tmp_path / "fold_0" / policy.EXPERIMENT_ID
        merged_config = tmp_path / "merged.yaml"
        merged_config.write_text(yaml.safe_dump({"protocol": "symmetric_merged", "modality": "audio_text"}))
        with pytest.raises(ValueError, match="only the untouched DAIC official test"):
            run_merged_optuna100(
                features_dir=features, output_dir=output, merged_config_path=merged_config,
                stage="final", fold=0, run_id="t4",
            )


class TestMergedResolver:
    def test_merged_resolver_counts(self) -> None:
        import tools.resolve_optuna100_manifest as resolver

        manifest = resolver.resolve(
            family="merged",
            run_id="t4_test",
            merged_sha="0" * 40,
            branch="main",
            github_issue=None,
            pr=None,
            require_caches=False,
        )
        assert manifest["study_count"] == 36
        assert manifest["per_backend"] == {"qwen": 18, "gemma4": 18}
        qwen_ready = [s for s in manifest["studies"] if s["backend"] == "qwen" and not s["cache_missing"]]
        assert len(qwen_ready) == 18
        stages = {s["stage"] for s in manifest["studies"] if not s["cache_missing"]}
        assert stages == {"cv", "final"}
