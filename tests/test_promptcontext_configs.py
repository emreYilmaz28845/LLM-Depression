from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.build_promptcontext_qwen38_configs import (
    ALLOWED_DIFF_PATHS,
    CELLS,
    QWEN38_LORA_TARGET_REGEX,
    QWEN38_MODEL_PATH,
    QWEN38_MODEL_REVISION,
    diff_paths,
    main as generator_main,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
MATRIX = ROOT / "configs/experiments/promptcontext_qwen38/matrix.yaml"
PR259_CONFIG = MAIN / "daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml"

NEW_CONFIG_NAMES = [cell[2] for cell in CELLS]
SOURCE_NAMES = {cell[0]: cell[1] for cell in CELLS}
TARGET_NAMES = {cell[0]: cell[2] for cell in CELLS}
FOLDS = {cell[0]: tuple(cell[5]) for cell in CELLS}

# Fields the recipe must inherit unchanged from its canonical source.
FROZEN_TOP_LEVEL = (
    "seed",
    "split",
    "labels",
    "data",
    "quarantine_path",
    "dataset_root",
)
FROZEN_TRAINING = (
    "num_train_epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "logging_steps",
    "bf16",
    "gradient_checkpointing",
    "dataloader_num_workers",
    "max_grad_norm",
    "class_balance",
    "selection_metric",
    "selection_metric_mode",
    "early_stopping",
    "dist_timeout_minutes",
)
FROZEN_LORA = ("rank", "alpha", "dropout", "bias")
FROZEN_EVALUATION = (
    "sample_prediction_mode",
    "headline_mode",
    "aggregation_level",
    "subject_score_aggregation",
    "generation_max_new_tokens",
    "num_beams",
    "do_sample",
    "evaluate_last_checkpoint",
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pairs() -> list[tuple[str, dict, dict]]:
    return [
        (cell[0], _load(MAIN / cell[1]), _load(MAIN / cell[2]))
        for cell in CELLS
    ]


def test_exactly_five_new_configs_and_21_fits() -> None:
    found = sorted(path.name for path in MAIN.glob("*_promptcontext_v1_qwen38_27b.yaml"))
    assert found == sorted(NEW_CONFIG_NAMES)
    assert sorted(path.name for path in MAIN.glob("*promptcontext*_qwen38_27b.yaml")) == sorted(
        NEW_CONFIG_NAMES
    )
    matrix = _load(MATRIX)
    assert len(matrix["experiments"]) == 5
    assert sum(len(cell["folds"]) for cell in matrix["experiments"]) == 21
    assert matrix["experiments"][0]["folds"] == [0]
    for cell in matrix["experiments"]:
        assert cell["modality"] == "text_only"
        assert cell["backbone"] == "qwen38"
        assert cell["separate_eval"] is True
        assert (ROOT / cell["config"]).is_file()


def test_every_cell_is_native_text_only_qwen38() -> None:
    expected_dataset = {
        "daic": "daic",
        "d3tec": "d3tec",
        "androids": "androids_interview",
        "cmdc": "cmdc",
        "turkish_pooled": "turkish",
    }
    for cell_id, source, config in _pairs():
        assert config["data"]["use_audio"] is False
        assert config["data"]["use_text"] is True
        assert config["model_backend"] == "qwen38"
        assert config["model_name_or_path"] == QWEN38_MODEL_PATH
        assert config["model_revision"] == QWEN38_MODEL_REVISION
        assert config["recipe_id"] == f"{source['recipe_id']}_promptcontext_v1"
        assert config["dataset"] == expected_dataset[cell_id]


def test_prompt_selection_is_explicit_and_centralized() -> None:
    for cell_id, _source, config in _pairs():
        assert config["prompt"]["version"] == "promptcontext_v1"
        assert config["prompt"]["dataset_context"]
        assert "system" not in config["prompt"], cell_id
        assert config["prompt"]["user_template"] == _load(
            MAIN / SOURCE_NAMES[cell_id]
        )["prompt"]["user_template"]


def test_evaluation_view_dtype_and_checkpoint_selection() -> None:
    for _cell_id, _source, config in _pairs():
        evaluation = config["evaluation"]
        assert evaluation["evaluation_view"] == "harmonized_all_windows_full_coverage"
        assert evaluation["inference_dtype"] == "bf16"
        assert evaluation["sample_prediction_mode"] == "likelihood"
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["selection_metric_mode"] == "max"
        assert config["training"]["run_final_eval_in_train"] is False


def test_qwen38_fsdp_recipe_matches_pr259() -> None:
    reference = _load(PR259_CONFIG)["training"]
    for cell_id, _source, config in _pairs():
        training = config["training"]
        assert training["strategy"] == reference["strategy"] == "fsdp"
        assert training["activation_offload"] == "cpu"
        for key in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "bf16",
        ):
            assert training[key] == reference[key], f"{cell_id}:{key}"
        assert (
            training["per_device_train_batch_size"]
            * training["gradient_accumulation_steps"]
            * 4
            == 128
        )


def test_lora_targets_are_the_pr259_set_and_the_rest_is_inherited() -> None:
    reference = _load(PR259_CONFIG)["lora"]
    assert reference["target_modules"] == QWEN38_LORA_TARGET_REGEX
    for cell_id, source, config in _pairs():
        assert config["lora"]["target_modules"] == QWEN38_LORA_TARGET_REGEX
        for key in FROZEN_LORA:
            assert config["lora"][key] == source["lora"][key], f"{cell_id}:lora.{key}"


def test_split_seed_label_windowing_and_epochs_are_unchanged() -> None:
    for cell_id, source, config in _pairs():
        for key in FROZEN_TOP_LEVEL:
            assert config[key] == source[key], f"{cell_id}:{key}"
        for key in FROZEN_TRAINING:
            assert config["training"][key] == source["training"][key], f"{cell_id}:training.{key}"
        for key in FROZEN_EVALUATION:
            if key in source["evaluation"]:
                assert (
                    config["evaluation"][key] == source["evaluation"][key]
                ), f"{cell_id}:evaluation.{key}"
        for key in ("protocol_id", "manifest_variant"):
            if key in source:
                assert config[key] == source[key], f"{cell_id}:{key}"


def test_run_roots_are_new_and_do_not_collide() -> None:
    reference = _load(PR259_CONFIG)
    roots = set()
    for _cell_id, source, config in _pairs():
        run_root = config["output_dirs"]["run_root"]
        assert run_root != source["output_dirs"]["run_root"]
        assert run_root != reference["output_dirs"]["run_root"]
        assert "/output_model/promptcontext_v1_qwen38_likelihood/text_only/" in run_root
        assert run_root not in roots
        roots.add(run_root)
        assert config["output_dirs"]["manifest_dir"] == source["output_dirs"]["manifest_dir"]
        assert config["output_dirs"]["split_dir"] == source["output_dirs"]["split_dir"]


def test_pooled_cell_keeps_the_pair_contract_and_asks_for_the_prebuilt_manifest() -> None:
    pooled = _load(MAIN / TARGET_NAMES["turkish_pooled"])
    assert pooled["dataset_variant"] == "pooled_t17"
    assert pooled["manifest_policy"] == "prebuilt"
    assert "{question_context}" in pooled["prompt"]["user_template"]
    assert pooled["prompt"]["question_context_version"] == "promptcontext_v1"
    assert (
        pooled["evaluation"]["subject_score_aggregation"]
        == "turkish_pooled_text_pair_mean_margin_strict_v1"
    )
    assert pooled["split"]["mode"] == "cv"
    assert pooled["split"]["cv_protocol"] == "train_val"
    assert pooled["split"]["outer_folds"] == 5
    for _cell_id, _source, config in _pairs():
        if config["dataset"] != "turkish":
            assert "manifest_policy" not in config


def test_structured_config_diffs_contain_only_allowed_fields() -> None:
    for cell_id, source, config in _pairs():
        changed = diff_paths(source, config)
        assert changed, cell_id
        disallowed = [path for path in changed if path not in ALLOWED_DIFF_PATHS]
        assert not disallowed, f"{cell_id}: {disallowed}"


def test_generator_check_mode_reports_no_drift(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    assert generator_main(["--check", "--audit-output", str(audit)]) == 0
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["training_fits"] == 21
    assert all(entry["allowed"] for entry in audit_payload["configs"])
