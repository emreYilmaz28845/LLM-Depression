"""Contract tests for the prompt-context Qwen/Gemma config family.

The family must represent a prompt change only: every scientific field is
inherited byte-for-byte from the canonical source config, and the Gemma configs
differ from their Qwen counterparts only where the backend requires it. The
tests also pin the recording-context text published in
``docs/PROMPT_PROPOSAL_DEPRESSION_20260920.md`` so it cannot drift silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_promptcontext_configs import (
    COMMON_INSTRUCTION,
    DATASET_CONTEXT,
    EVALUATION_VIEW,
    MODALITIES,
    QUESTION_CONTEXT_VERSION,
    RECIPE_ID,
    STANDALONE_DATASET_ORDER,
    STANDALONE_DATASETS,
    build_merged,
    build_standalone,
    merged_source_name,
    merged_target_name,
    source_name,
    system_prompt,
    target_name,
)
from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
MERGED = ROOT / "configs/experiments/merged"
MATRIX = ROOT / "configs/experiments/promptcontext/matrix.yaml"

GEMMA_ALLOWED_DIFF = {
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "lora.target_modules",
    "output_dirs.run_root",
}
# The pre-prompt-context system instruction every canonical source still carries.
LEGACY_SYSTEM_PREFIXES = (
    "You are a psychologist analyzing speech and transcript information for depression screening.",
    "You are a psychologist analyzing transcript information for depression screening.",
    "You are a psychologist analyzing speech audio for depression screening.",
)


def _flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        flat: dict = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            # Keep scalar lists (LoRA targets, folds, ...) as one comparable value
            # so a list/scalar type change shows up as a single differing key.
            return {prefix: list(value)}
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: value}


def _diff(source: dict, target: dict) -> set[str]:
    left = _flatten(source)
    right = _flatten(target)
    return {
        key
        for key in set(left) | set(right)
        if left.get(key, "<missing>") != right.get(key, "<missing>")
    }


def _standalone_pairs() -> list[tuple[str, str, bool]]:
    return [
        (dataset, modality, gemma)
        for dataset in STANDALONE_DATASET_ORDER
        for modality in MODALITIES
        for gemma in (False, True)
    ]


def test_matrix_declares_every_cell_and_exact_counts() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert matrix["recipe_id"] == RECIPE_ID
    assert matrix["evaluation_view"] == EVALUATION_VIEW
    assert matrix["counts"] == {
        "standalone_cells": 30,
        "merged_cells": 6,
        "standalone_training_folds": 126,
        "merged_cv_folds": 30,
        "merged_final_fits": 6,
        "total_training_fits": 162,
    }
    standalone = matrix["standalone"]
    merged = matrix["merged"]
    assert len(standalone) == 30
    assert len(merged) == 6
    assert len({item["config"] for item in standalone}) == 30
    assert len({item["config"] for item in merged}) == 6
    assert sum(cell["backbone"] == "qwen" for cell in standalone) == 15
    assert sum(cell["backbone"] == "gemma4" for cell in standalone) == 15
    assert sum(len(cell["folds"]) for cell in standalone) == 126
    assert sum(len(cell["stages"]) for cell in merged) == 12
    daic = [cell for cell in standalone if cell["dataset"] == "daic"]
    assert len(daic) == 6
    assert all(cell["folds"] == [0] for cell in daic)
    assert all(cell["folds"] == [0, 1, 2, 3, 4] for cell in standalone if cell["dataset"] != "daic")


def test_matrix_cells_match_the_generator_and_exist_on_disk() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    expected = set()
    for dataset, modality, gemma in _standalone_pairs():
        expected.add(
            (
                dataset,
                modality,
                "gemma4" if gemma else "qwen",
                f"configs/main/{target_name(dataset, modality, gemma=gemma)}",
            )
        )
    declared = {
        (cell["dataset"], cell["modality"], cell["backbone"], cell["config"])
        for cell in matrix["standalone"]
    }
    assert declared == expected
    merged_expected = {
        (
            modality,
            "gemma4" if gemma else "qwen",
            f"configs/experiments/merged/{merged_target_name(modality, gemma=gemma)}",
        )
        for modality in MODALITIES
        for gemma in (False, True)
    }
    merged_declared = {
        (cell["modality"], cell["backbone"], cell["config"]) for cell in matrix["merged"]
    }
    assert merged_declared == merged_expected
    for cell in matrix["standalone"] + matrix["merged"]:
        assert (ROOT / cell["config"]).is_file(), cell["config"]


def test_standalone_configs_change_only_the_prompt_recipe() -> None:
    for dataset, modality, gemma in _standalone_pairs():
        source = load_yaml(MAIN / source_name(dataset, modality))
        target = load_yaml(MAIN / target_name(dataset, modality, gemma=gemma))
        expected_diff = {"recipe_id", "prompt.system", "output_dirs.run_root"}
        if "evaluation_view" not in source["evaluation"]:
            expected_diff.add("evaluation.evaluation_view")
        if dataset == "turkish":
            expected_diff.add("prompt.question_context_version")
        if gemma:
            expected_diff |= {
                "model_backend",
                "model_name_or_path",
                "model_revision",
                "lora.target_modules",
            }
        assert _diff(source, target) == expected_diff, (dataset, modality, gemma)
        assert target["recipe_id"] == RECIPE_ID
        assert target["evaluation"]["evaluation_view"] == EVALUATION_VIEW
        assert target["evaluation"]["sample_prediction_mode"] == "likelihood"
        assert target["training"]["selection_metric"] == "inner_val_macro_f1"
        assert target["training"]["selection_metric_mode"] == "max"
        assert target["prompt"]["prompt_language"] == "english"
        assert target["prompt"]["system"] == system_prompt(dataset)
        assert target["prompt"]["user_template"] == source["prompt"]["user_template"]
        assert target["labels"] == source["labels"]
        assert target["data"] == source["data"]
        assert target["split"] == source["split"]


def test_prompt_context_system_text_matches_the_published_proposal() -> None:
    assert COMMON_INSTRUCTION.startswith(
        "You are classifying a participant's depression study label from the provided "
        "speech audio and/or transcript."
    )
    assert "Make a research label prediction, not a clinical diagnosis." in COMMON_INSTRUCTION
    assert set(DATASET_CONTEXT) == set(STANDALONE_DATASETS)
    assert "psychiatrist's DSM-5 diagnosis" in DATASET_CONTEXT["androids_interview"]
    assert "PHQ-9 \u2265 10" in DATASET_CONTEXT["d3tec"]
    assert "PHQ-8 binary label" in DATASET_CONTEXT["daic"]
    assert "confirmed with MINI" in DATASET_CONTEXT["cmdc"]
    assert "BDI \u2265 17" in DATASET_CONTEXT["turkish"]
    for dataset, modality, gemma in _standalone_pairs():
        config = load_yaml(MAIN / target_name(dataset, modality, gemma=gemma))
        system = config["prompt"]["system"]
        assert system.startswith(COMMON_INSTRUCTION)
        assert DATASET_CONTEXT[dataset] in system
        for other, block in DATASET_CONTEXT.items():
            if other != dataset:
                assert block not in system, (dataset, other)


def test_gemma_and_qwen_configs_differ_only_in_the_backend_switch() -> None:
    for dataset, modality, _gemma in _standalone_pairs():
        qwen = load_yaml(MAIN / target_name(dataset, modality, gemma=False))
        gemma = load_yaml(MAIN / target_name(dataset, modality, gemma=True))
        assert _diff(qwen, gemma) == GEMMA_ALLOWED_DIFF, (dataset, modality)
        for key in ("dataset", "seed", "recipe_id", "labels", "prompt", "data", "split", "training", "evaluation"):
            assert qwen[key] == gemma[key], f"{dataset}/{modality}: {key} changed"
        assert gemma["model_backend"] == "gemma4"
        assert gemma["lora"]["target_modules"].startswith("^model\\.language_model\\.layers")
        assert qwen["output_dirs"]["manifest_dir"] == gemma["output_dirs"]["manifest_dir"]
        assert qwen["output_dirs"]["split_dir"] == gemma["output_dirs"]["split_dir"]
        assert qwen["output_dirs"]["run_root"] != gemma["output_dirs"]["run_root"]


def test_turkish_pooled_cells_carry_the_versioned_condition_contract() -> None:
    for modality in MODALITIES:
        for gemma in (False, True):
            config = load_yaml(MAIN / target_name("turkish", modality, gemma=gemma))
            assert config["dataset"] == "turkish"
            assert config["dataset_variant"] == "pooled_t17"
            assert config["threshold"] == 17
            assert config["prompt"]["question_context_version"] == QUESTION_CONTEXT_VERSION
            assert "{question_context}" in config["prompt"]["user_template"]
            assert config["output_dirs"]["manifest_dir"].endswith(
                "manifests_harmonized/turkish_pooled_t17_qwen3asr"
            )
            if modality == "text_only":
                assert (
                    config["evaluation"]["subject_score_aggregation"]
                    == "turkish_pooled_text_pair_mean_margin_strict_v1"
                )
    for dataset in STANDALONE_DATASET_ORDER:
        if dataset == "turkish":
            continue
        for modality in MODALITIES:
            config = load_yaml(MAIN / target_name(dataset, modality, gemma=False))
            assert "question_context_version" not in config["prompt"]


def test_standalone_configs_use_new_campaign_run_roots() -> None:
    roots = set()
    for dataset, modality, gemma in _standalone_pairs():
        config = load_yaml(MAIN / target_name(dataset, modality, gemma=gemma))
        run_root = config["output_dirs"]["run_root"]
        campaign = "promptcontext_v1_gemma4_likelihood" if gemma else "promptcontext_v1_likelihood"
        # load_yaml expands ${PROJECT_ROOT} to this checkout.
        assert run_root == f"{ROOT}/output_model/{campaign}/{modality}/{dataset}"
        roots.add(run_root)
    assert len(roots) == 30


def test_merged_configs_use_pooled_turkish_and_new_roots() -> None:
    for modality in MODALITIES:
        for gemma in (False, True):
            source = load_yaml(MERGED / merged_source_name(modality, gemma=gemma))
            config = load_yaml(MERGED / merged_target_name(modality, gemma=gemma))
            assert config["recipe_id"] == RECIPE_ID
            assert config["modality"] == modality
            assert config["protocol"] == "symmetric_merged"
            assert config["model_name_or_path"] == source["model_name_or_path"]
            assert config["protocol_settings"] == source["protocol_settings"]
            assert config["heads"] == source["heads"]
            assert config["heads"]["optuna"]["enabled"] is False
            assert int(config["training"]["dist_timeout_minutes"]) >= 120
            assert {item["name"] for item in config["components"]} == {
                "daic",
                "cmdc",
                "turkish",
                "d3tec",
                "androids_interview",
            }
            for component in config["components"]:
                target = ROOT / component["config"]
                assert target.is_file(), component["config"]
                if component["name"] == "turkish":
                    assert target.name == target_name("turkish", modality, gemma=gemma)
                    assert component["manifest_path"] == (
                        "outputs/manifests_harmonized/turkish_pooled_t17_qwen3asr/"
                        "turkish_manifest.jsonl"
                    )
                    assert component["metadata_path"] == (
                        "outputs/splits_harmonized/turkish_pooled_t17_qwen3asr/"
                        "turkish_manifest_metadata.json"
                    )
                else:
                    assert target.name == target_name(component["name"], modality, gemma=gemma)
            expected_diff = {"name", "recipe_id", "output_dirs.merged_root", "output_dirs.run_root"}
            for index, component in enumerate(source["components"]):
                expected_diff.add(f"components[{index}].config")
                if component["name"] == "turkish":
                    expected_diff.add(f"components[{index}].manifest_path")
                    expected_diff.add(f"components[{index}].metadata_path")
            assert _diff(source, config) == expected_diff, (modality, gemma)
            output_campaign = "promptcontext_v1_gemma4" if gemma else "promptcontext_v1"
            run_campaign = (
                "symmetric_merged/promptcontext_v1_gemma4_likelihood"
                if gemma
                else "symmetric_merged/promptcontext_v1_likelihood"
            )
            assert config["output_dirs"]["merged_root"] == (
                f"{ROOT}/outputs/symmetric_merged/{output_campaign}/{modality}"
            )
            assert config["output_dirs"]["run_root"] == (
                f"{ROOT}/output_model/{run_campaign}/{modality}"
            )


def test_generated_configs_match_the_generator() -> None:
    """Every file on disk must equal its derived content (run --check to fix)."""
    for config, path in build_standalone() + build_merged():
        assert path.is_file(), path
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == config, path


def test_canonical_source_configs_are_untouched() -> None:
    for dataset, modality, _gemma in _standalone_pairs():
        source = load_yaml(MAIN / source_name(dataset, modality))
        assert source["recipe_id"] != RECIPE_ID
        assert source["prompt"]["system"].startswith(LEGACY_SYSTEM_PREFIXES)
        assert "question_context_version" not in source["prompt"]
        assert "promptcontext" not in source["output_dirs"]["run_root"]
    for modality in MODALITIES:
        for gemma in (False, True):
            source = load_yaml(MERGED / merged_source_name(modality, gemma=gemma))
            assert source["recipe_id"] != RECIPE_ID
            assert "promptcontext" not in source["output_dirs"]["merged_root"]
            turkish = next(item for item in source["components"] if item["name"] == "turkish")
            assert "turkish_pooled" not in turkish["config"]
