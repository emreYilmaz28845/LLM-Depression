"""Contracts for the native-versus-English text-only head study.

Covers merged split-seed separation, fixed-head seed resolution, per-method
merged backend qualifiers, and the four study merged configs.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.merged.heads import (
    GEMMA4_LOGREG_RAW_MERGED_BACKEND,
    QWEN_LOGREG_RAW_MERGED_BACKEND,
    method_prediction_backend,
    resolve_fixed_head_seed,
)
from src.merged.protocol import (
    DATASETS,
    build_protocol_splits,
    resolve_split_seed,
)
from src.features import optuna100_policy as policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUDY_MERGED_CONFIGS = {
    "native_qwen": PROJECT_ROOT
    / "configs/experiments/merged/symmetric_merged_text_heads_native_qwen.yaml",
    "english_qwen": PROJECT_ROOT
    / "configs/experiments/merged/symmetric_merged_text_heads_english_qwen.yaml",
    "native_gemma4": PROJECT_ROOT
    / "configs/experiments/merged/symmetric_merged_text_heads_native_gemma4.yaml",
    "english_gemma4": PROJECT_ROOT
    / "configs/experiments/merged/symmetric_merged_text_heads_english_gemma4.yaml",
}


def _config(**overrides):
    config = {
        "protocol": "symmetric_merged",
        "modality": "text_only",
        "seed": 1337,
        "heads": {},
        "protocol_settings": {},
    }
    config.update(copy.deepcopy(overrides))
    return config


def test_resolve_split_seed_prefers_protocol_settings_key() -> None:
    config = _config(seed=7)
    assert resolve_split_seed(config) == 7
    config["protocol_settings"]["split_seed"] = 1337
    assert resolve_split_seed(config) == 1337


def test_resolve_split_seed_falls_back_to_top_level_then_default() -> None:
    assert resolve_split_seed(_config()) == 1337
    config = _config()
    del config["seed"]
    assert resolve_split_seed(config) == 1337
    config["seed"] = 2024
    assert resolve_split_seed(config) == 2024


def test_resolve_fixed_head_seed_prefers_heads_fixed_seed() -> None:
    config = _config(seed=2024)
    assert resolve_fixed_head_seed(config) == 2024
    config["heads"]["fixed_seed"] = 1337
    assert resolve_fixed_head_seed(config) == 1337


def _records():
    records = []
    for dataset in DATASETS:
        labels = {}
        rows = []
        for subject_index in range(12):
            subject = f"{dataset[:3]}-{subject_index:02d}"
            label = subject_index % 2
            labels[subject] = label
            for response in range(2):
                rows.append(
                    {
                        "dataset": dataset,
                        "subject_id": subject,
                        "sample_id": f"{subject}-r{response}",
                        "response_id": f"{subject}-r{response}",
                        "num_segments": 1,
                        "label": label,
                    }
                )
        record = {
            "dataset": dataset,
            "config_path": f"configs/{dataset}.yaml",
            "manifest_hash": f"manifest-{dataset}",
            "labels": labels,
            "rows": rows,
            "folds": {},
            "official_test_subject_ids": [],
        }
        if dataset == "daic":
            record["official_test_subject_ids"] = [f"mai-10", f"mai-11"]
            record["labels"].pop("dai-10")
            record["labels"].pop("dai-11")
            record["rows"] = [
                row for row in rows if row["subject_id"] not in {"dai-10", "dai-11"}
            ]
        records.append(record)
    return records


def test_split_hash_is_stable_across_training_seeds_when_split_seed_set() -> None:
    base_config = {
        "protocol_settings": {"split_seed": 1337},
        "seed": 7,
    }
    other_config = {
        "protocol_settings": {"split_seed": 1337},
        "seed": 2024,
    }
    first = build_protocol_splits(_records(), seed=resolve_split_seed(base_config), inner_val_ratio=0.2)
    second = build_protocol_splits(_records(), seed=resolve_split_seed(other_config), inner_val_ratio=0.2)
    assert first["split_hash"] == second["split_hash"]


def test_top_level_seed_still_changes_splits_without_the_key() -> None:
    first = build_protocol_splits(_records(), seed=7, inner_val_ratio=0.2)
    second = build_protocol_splits(_records(), seed=2024, inner_val_ratio=0.2)
    assert first["split_hash"] != second["split_hash"]


class TestMethodPredictionBackends:
    def test_logreg_backends_are_method_specific(self) -> None:
        assert (
            method_prediction_backend("", "logreg", profile_matches=True)
            == QWEN_LOGREG_RAW_MERGED_BACKEND
        )
        assert method_prediction_backend(None, "logreg", profile_matches=True) == (
            QWEN_LOGREG_RAW_MERGED_BACKEND
        )
        assert method_prediction_backend("qwen2audio", "logreg", profile_matches=True) == (
            QWEN_LOGREG_RAW_MERGED_BACKEND
        )
        assert method_prediction_backend("gemma4", "logreg", profile_matches=True) == (
            GEMMA4_LOGREG_RAW_MERGED_BACKEND
        )

    def test_optuna_uses_policy_string_and_fixed_xgb_stays_unqualified(self) -> None:
        assert method_prediction_backend("gemma4", "xgb_optuna", profile_matches=True) == (
            policy.prediction_backend("gemma4", merged=True)
        )
        assert method_prediction_backend("", "xgb_optuna", profile_matches=True) == (
            "qwen_hidden_xgb_optuna100_symmetric_merged"
        )
        assert method_prediction_backend("gemma4", "xgb_fixed", profile_matches=True) is None
        assert method_prediction_backend("gemma4", "logreg", profile_matches=False) is None
        assert method_prediction_backend("gemma4", "xgb_optuna", profile_matches=False) is None


@pytest.mark.parametrize("variant", sorted(STUDY_MERGED_CONFIGS))
class TestStudyMergedConfigContract:
    def test_file_exists_and_declares_study_identity(self, variant: str) -> None:
        path = STUDY_MERGED_CONFIGS[variant]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert str(config["protocol"]) == "symmetric_merged"
        assert str(config["modality"]) == "text_only"
        assert config["name"] == f"symmetric_merged_text_heads_{variant}"
        assert int(config["protocol_settings"]["split_seed"]) == 1337
        assert int(config["heads"]["fixed_seed"]) == 1337
        optuna = config["heads"]["optuna"]
        assert bool(optuna["enabled"]) is False
        assert int(optuna["target_trials"]) == 0
        assert optuna["protocol_profile"] == policy.PROTOCOL_PROFILE

    def test_exactly_five_components_with_expected_names(self, variant: str) -> None:
        config = yaml.safe_load(STUDY_MERGED_CONFIGS[variant].read_text(encoding="utf-8"))
        names = [str(item["name"]) for item in config["components"]]
        assert names == ["daic", "cmdc", "turkish", "d3tec", "androids_interview"]

    def test_transcript_condition_matches_variant(self, variant: str) -> None:
        config = yaml.safe_load(STUDY_MERGED_CONFIGS[variant].read_text(encoding="utf-8"))
        english = variant.startswith("english")
        gemma = variant.endswith("gemma4")
        for component in config["components"]:
            source = str(component["config"])
            if component["name"] == "daic":
                # DAIC stays original English in every condition.
                assert "_en_" not in source.replace("_gemma4_12b", "")
                continue
            if english:
                assert "_en" in source, source
                assert "manifests_harmonized_en" in str(component["manifest_path"])
                assert "splits_harmonized_en" in str(component["metadata_path"])
            else:
                assert "_en_" not in source, source
                assert "manifests_harmonized_en" not in str(component["manifest_path"])
            if gemma:
                assert "gemma4_12b" in source, source
            else:
                assert "gemma4_12b" not in source, source

    def test_gemma_variants_pin_backbone_and_revision(self, variant: str) -> None:
        config = yaml.safe_load(STUDY_MERGED_CONFIGS[variant].read_text(encoding="utf-8"))
        if variant.endswith("gemma4"):
            assert str(config["model_backend"]) == "gemma4"
            assert str(config["model_revision"]).startswith("707f0a3b8a3c7ad5")
        else:
            assert "model_backend" not in config


def test_study_merged_output_roots_are_distinct_and_new() -> None:
    roots: set[str] = set()
    legacy_roots: set[str] = set()
    legacy_dir = PROJECT_ROOT / "configs/experiments/merged"
    for path in sorted(legacy_dir.glob("symmetric_merged_*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        run_root = str(config.get("output_dirs", {}).get("run_root", ""))
        if not run_root:
            continue
        if path.name.startswith("symmetric_merged_text_heads_"):
            roots.add(run_root)
        else:
            legacy_roots.add(run_root)
    assert len(roots) == 4
    assert roots.isdisjoint(legacy_roots)


def test_manifest_builder_uses_resolve_split_seed() -> None:
    import inspect

    source = Path(
        PROJECT_ROOT / "scripts/build_symmetric_merged_manifest.py"
    ).read_text(encoding="utf-8")
    assert "resolve_split_seed(config)" in source
    assert 'args.seed if args.seed is not None else resolve_split_seed(config)' in source


class TestMergedOptunaSmokeGate:
    def _features(self, tmp_path: Path) -> Path:
        return tmp_path / "missing_features"

    def test_production_refuses_non_100_target(self, tmp_path: Path) -> None:
        from src.merged.optuna100 import run_merged_optuna100

        with pytest.raises(ValueError, match="exactly"):
            run_merged_optuna100(
                features_dir=self._features(tmp_path),
                output_dir=tmp_path / "xgb_optuna100_harmonized_v1",
                merged_config_path=tmp_path / "merged.yaml",
                stage="cv",
                fold=0,
                run_id="r",
                target_trials=50,
            )

    def test_smoke_allows_exactly_two_trials_and_passes_gate(self, tmp_path: Path) -> None:
        from src.merged.optuna100 import run_merged_optuna100

        # The gate must pass (no ValueError) and fail later on missing
        # feature metadata, proving the two-trial smoke route opens.
        with pytest.raises(FileNotFoundError):
            run_merged_optuna100(
                features_dir=self._features(tmp_path),
                output_dir=tmp_path / "xgb_optuna100_harmonized_v1",
                merged_config_path=tmp_path / "merged.yaml",
                stage="smoke",
                fold=0,
                run_id="r",
                target_trials=2,
            )

    def test_two_trials_refused_outside_smoke_stage(self, tmp_path: Path) -> None:
        from src.merged.optuna100 import run_merged_optuna100

        with pytest.raises(ValueError, match="exactly"):
            run_merged_optuna100(
                features_dir=self._features(tmp_path),
                output_dir=tmp_path / "xgb_optuna100_harmonized_v1",
                merged_config_path=tmp_path / "merged.yaml",
                stage="final",
                fold=0,
                run_id="r",
                target_trials=2,
            )
