"""Contract tests for the Qwen3 DAIC label-vocabulary configs and submission matrix."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.build_qwen3_daic_label_configs import (
    ALLOWED_DIFF_KEYS,
    ARM_ORDER,
    ARMS,
    MAIN,
    LABELS,
    SOURCES,
    build as build_configs,
    diff_keys,
    generated_name,
)
from src.model.qwen38_lora import validate_qwen38_config
from src.model.qwen3omni_lora import validate_qwen3omni_config
from src.utils import prompt_label_instruction, prompt_label_descriptor
from tools.qwen3_daic_label_vocab_matrix import (
    CAMPAIGN,
    PRODUCTION_SEEDS,
    SMOKE_CAMPAIGN,
    SMOKE_EPOCHS,
    SMOKE_SEED,
    build_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = tuple(SOURCES)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical(modality: str) -> dict:
    return _load(MAIN / SOURCES[modality][0])


def _generated(modality: str, arm: str) -> dict:
    return _load(LABELS / generated_name(modality, arm))


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("arm", ARM_ORDER)
def test_generated_config_equals_canonical_source_plus_the_allowlist(modality: str, arm: str) -> None:
    source = _canonical(modality)
    candidate = _generated(modality, arm)

    expected = deepcopy(source)
    expected["recipe_id"] = f"{source['recipe_id']}_{arm}"
    vocab, positive, negative = ARMS[arm]
    expected["labels"] = {
        "label_vocab_version": vocab,
        "internal_positive_label": positive,
        "internal_negative_label": negative,
        "external_positive_label": "Depressed",
        "external_negative_label": "Non-depressed",
    }
    expected["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{CAMPAIGN}/{modality}/daic"
    )

    assert candidate == expected
    differences = set(diff_keys(source, candidate))
    assert differences <= set(ALLOWED_DIFF_KEYS), sorted(differences)
    assert "output_dirs.run_root" in differences
    assert "recipe_id" in differences


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("arm", ARM_ORDER)
def test_file_name_carries_model_and_arm_without_colliding_with_other_families(modality: str, arm: str) -> None:
    name = generated_name(modality, arm)
    model = SOURCES[modality][1]
    assert name.startswith(f"daic_{modality}_harmonized_selmacrof1_likelihood_{model}_{arm}")
    # Older families glob for names that end with likelihood_<tag>_v1.yaml.
    for tag in ("ab", "01", "truefalse", "yesno"):
        assert not name.endswith(f"likelihood_{tag}_v1.yaml"), name
    assert (LABELS / name).is_file()


@pytest.mark.parametrize("modality", MODALITIES)
def test_legend_renders_for_short_arms_and_en_keeps_the_answer_only_instruction(modality: str) -> None:
    for arm in ("ab", "01", "truefalse", "yesno"):
        _, positive, negative = ARMS[arm]
        instruction = prompt_label_instruction(_generated(modality, arm))
        assert instruction == (
            "Use this label legend:\n"
            f"{positive} = Depressed\n"
            f"{negative} = Non-depressed\n"
            f"Answer with exactly one label: {positive} or {negative}."
        )
    en_instruction = prompt_label_instruction(_generated(modality, "en"))
    assert en_instruction == "Answer with exactly one label: Depressed or Non-depressed."
    assert "legend" not in en_instruction
    assert prompt_label_descriptor(_generated(modality, "en")) == "Depressed or Non-depressed"


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("arm", ARM_ORDER)
def test_split_likelihood_and_evaluation_contract_is_frozen(modality: str, arm: str) -> None:
    config = _generated(modality, arm)
    split = config["split"]
    assert split["mode"] == "fixed"
    assert split["seed"] == 1337
    assert split["train_partition"] == "train"
    assert split["selection_partition"] == "val"
    assert split["final_eval_partition"] == "test"
    assert split["dev_pool_partitions"] == ["train"]
    assert "smoke_subject_limit" not in split
    assert config["seed"] == 1337

    evaluation = config["evaluation"]
    assert evaluation["sample_prediction_mode"] == "likelihood"
    assert evaluation["headline_mode"] == "likelihood"
    assert evaluation["evaluation_view"] == "harmonized_all_windows_full_coverage"
    assert evaluation["aggregation_level"] == "subject"
    assert evaluation["evaluate_last_checkpoint"] is False
    assert evaluation["inference_dtype"] == "bf16"

    training = config["training"]
    assert training["selection_metric"] == "inner_val_macro_f1"
    assert training["selection_metric_mode"] == "max"
    assert training["early_stopping"]["metric"] == "inner_val_macro_f1"
    assert training["early_stopping"]["mode"] == "max"
    assert training["strategy"] == "fsdp"
    assert training["activation_offload"] == "cpu"
    assert training["per_device_train_batch_size"] == 1
    # The audio arms inherit the canonical two-node audio lane; the text arm
    # keeps the one-node text lane. Both keep the effective global batch of 128.
    if modality == "text_only":
        assert training["gradient_accumulation_steps"] == 32
    else:
        assert training["gradient_accumulation_steps"] == 16
    assert config["prompt"]["version"] == "promptcontext_v1"
    assert config["prompt"]["dataset_context"] == "daic"
    assert config["quarantine_path"] == "${PROJECT_ROOT}/configs/quarantines.yaml"


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("arm", ARM_ORDER)
def test_model_identity_and_evaluation_shape_are_preserved(modality: str, arm: str) -> None:
    config = _generated(modality, arm)
    if modality == "text_only":
        assert config["model_backend"] == "qwen38"
        assert config["model_revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        assert config["model_name_or_path"].startswith("${QWEN38_MODEL_PATH:-")
        assert "resources" not in config
        validate_qwen38_config(config)
    else:
        assert config["model_backend"] == "qwen3omni"
        assert config["model_attn_implementation"] == "sdpa"
        assert config["model_name_or_path"].startswith("${QWEN3_OMNI_MODEL_PATH:-")
        assert config["resources"] == {
            "train_nodes": 2,
            "eval_nodes": 1,
            "eval_gpus_per_node": 4,
        }
        validate_qwen3omni_config(config)
    adapter = config.get("audio_adapter") or {}
    assert adapter.get("enabled", False) is False
    assert adapter.get("train_projector", False) is False


def test_generator_check_mode_reports_no_drift() -> None:
    assert build_configs(check=True) == 0


def test_matrix_expands_to_fifteen_smoke_and_fortyfive_production_chains() -> None:
    matrix = build_matrix()
    assert len(matrix["smoke"]) == 15
    assert len(matrix["production"]) == 45
    names = [entry["run_name"] for entry in matrix["smoke"] + matrix["production"]]
    assert len(set(names)) == len(names)
    assert {entry["seed"] for entry in matrix["smoke"]} == {SMOKE_SEED}
    assert {entry["seed"] for entry in matrix["production"]} == set(PRODUCTION_SEEDS)
    assert {entry["campaign"] for entry in matrix["smoke"]} == {SMOKE_CAMPAIGN}
    assert {entry["campaign"] for entry in matrix["production"]} == {CAMPAIGN}
    for entry in matrix["production"]:
        assert entry["extra_overrides"] == []
        assert f"_{entry['arm']}_s{entry['seed']}_f0" in entry["run_name"]
    for entry in matrix["smoke"]:
        assert entry["extra_overrides"] == [f"--set=training.num_train_epochs={SMOKE_EPOCHS}"]
    overrides = " ".join(
        token
        for entry in matrix["smoke"] + matrix["production"]
        for token in entry["extra_overrides"]
    )
    assert "split.seed" not in overrides
    assert "split.smoke_subject_limit" not in overrides


def test_matrix_points_at_real_configs_with_matching_arms() -> None:
    matrix = build_matrix()
    for entry in matrix["production"]:
        config_path = ROOT / entry["config"]
        assert config_path.is_file(), entry["config"]
        config = _load(config_path)
        vocab, positive, negative = ARMS[entry["arm"]]
        assert config["labels"]["label_vocab_version"] == vocab
        assert config["labels"]["internal_positive_label"] == positive
        assert config["labels"]["internal_negative_label"] == negative
        assert entry["modality"] in config["output_dirs"]["run_root"]
        assert entry["run_name"].startswith("q3dlv_")
    for entry in build_matrix()["smoke"]:
        assert entry["run_name"].startswith("q3dlvsmoke_")
        assert entry["env_activate"].endswith("/bin/activate")
