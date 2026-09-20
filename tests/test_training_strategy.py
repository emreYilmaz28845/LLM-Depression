from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.model import runtime
from src.model.qwen38_lora import fsdp_transformer_cls_names
from src.training_strategy import (
    TRAINING_STRATEGY_DDP,
    TRAINING_STRATEGY_FSDP,
    build_accelerator,
    build_fsdp_plugin,
    effective_global_batch_size,
    resolve_training_strategy,
)
from src.utils import (
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_TEXT,
    load_yaml,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
QWEN38_CONFIG = MAIN / "daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml"


class _Layer(torch.nn.Module):
    pass


class _LinearAttentionLayer(_Layer):
    pass


def _fake_decoder_model(layer_types=(_Layer, _LinearAttentionLayer)) -> SimpleNamespace:
    layers = torch.nn.ModuleList([layer_type() for layer_type in layer_types])
    language_model = SimpleNamespace(layers=layers)
    inner = SimpleNamespace(language_model=language_model)
    return SimpleNamespace(model=SimpleNamespace(model=inner, language_model=language_model))


def test_strategy_defaults_to_ddp() -> None:
    assert resolve_training_strategy({}) == TRAINING_STRATEGY_DDP
    assert resolve_training_strategy({"training": {}}) == TRAINING_STRATEGY_DDP
    assert resolve_training_strategy({"training": {"strategy": None}}) == TRAINING_STRATEGY_DDP


def test_strategy_accepts_fsdp_and_rejects_unknown() -> None:
    assert resolve_training_strategy({"training": {"strategy": "FSDP"}}) == TRAINING_STRATEGY_FSDP
    with pytest.raises(ValueError, match="Unsupported training.strategy"):
        resolve_training_strategy({"training": {"strategy": "deepspeed"}})


def test_effective_global_batch_size_keeps_128_across_shapes() -> None:
    recipe = {"training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 32}}
    assert effective_global_batch_size(recipe, 4) == 128
    eight_gpu = {
        "training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16}
    }
    assert effective_global_batch_size(eight_gpu, 8) == 128


def test_fsdp_plugin_uses_orig_params_and_wrap_policy() -> None:
    config = {"training": {"strategy": "fsdp"}}
    model = _fake_decoder_model()
    plugin = build_fsdp_plugin(config, model, fsdp_transformer_cls_names)
    assert plugin.use_orig_params is True
    assert plugin.sync_module_states is True
    assert plugin.transformer_cls_names_to_wrap == ["_Layer", "_LinearAttentionLayer"]


def test_fsdp_plugin_refuses_an_empty_wrap_policy() -> None:
    with pytest.raises(ValueError, match="no FSDP transformer class names"):
        build_fsdp_plugin({"training": {"strategy": "fsdp"}}, _fake_decoder_model(), lambda model: [])


def test_fsdp_plugin_allows_a_backend_without_a_wrap_policy() -> None:
    plugin = build_fsdp_plugin({"training": {"strategy": "fsdp"}}, _fake_decoder_model(), None)
    assert plugin.transformer_cls_names_to_wrap is None


def test_accelerator_refuses_in_train_held_out_eval_under_fsdp() -> None:
    config = {
        "training": {
            "strategy": "fsdp",
            "run_final_eval_in_train": True,
            "gradient_accumulation_steps": 32,
            "bf16": True,
        }
    }
    with pytest.raises(ValueError, match="run_final_eval_in_train must be false"):
        build_accelerator(config, model=_fake_decoder_model(), wrap_policy_names=fsdp_transformer_cls_names)


def test_accelerator_requires_the_model_under_fsdp() -> None:
    with pytest.raises(ValueError, match="needs the model"):
        build_accelerator({"training": {"strategy": "fsdp"}})


def test_accelerator_builds_for_ddp_by_default() -> None:
    config = {"training": {"gradient_accumulation_steps": 32, "bf16": True}}
    accelerator = build_accelerator(config, model=_fake_decoder_model())
    assert accelerator.gradient_accumulation_steps == 32
    assert accelerator.mixed_precision == "bf16"


def test_accelerator_builds_for_fsdp() -> None:
    config = {
        "training": {
            "strategy": "fsdp",
            "run_final_eval_in_train": False,
            "gradient_accumulation_steps": 32,
            "bf16": True,
        }
    }
    accelerator = build_accelerator(
        config, model=_fake_decoder_model(), wrap_policy_names=fsdp_transformer_cls_names
    )
    assert accelerator.gradient_accumulation_steps == 32
    assert accelerator.mixed_precision == "bf16"


def test_runtime_hook_dispatches_to_the_backend() -> None:
    config = {"model_backend": "qwen38", "data": {"use_audio": False, "use_text": True}}
    assert runtime.fsdp_wrap_policy_names(config, _fake_decoder_model()) == [
        "_Layer",
        "_LinearAttentionLayer",
    ]
    text_config = {"model_backend": MODEL_BACKEND_TEXT, "data": {"use_audio": False, "use_text": True}}
    assert runtime.fsdp_wrap_policy_names(text_config, _fake_decoder_model()) is None
    audio_config = {"model_backend": MODEL_BACKEND_QWEN2AUDIO, "data": {"use_audio": True, "use_text": True}}
    assert runtime.fsdp_wrap_policy_names(audio_config, _fake_decoder_model()) is None


def test_wrap_policy_reads_the_layers_from_a_peft_wrapped_model() -> None:
    layers = torch.nn.ModuleList([_Layer(), _Layer()])
    base = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    peft_like = SimpleNamespace(base_model=SimpleNamespace(model=base))
    assert fsdp_transformer_cls_names(peft_like) == ["_Layer"]


def test_qwen38_config_selects_fsdp_and_disables_in_train_eval() -> None:
    config = load_yaml(QWEN38_CONFIG)
    assert config["training"]["strategy"] == TRAINING_STRATEGY_FSDP
    assert config["training"]["run_final_eval_in_train"] is False
    assert config["training"]["gradient_accumulation_steps"] == 32
    assert effective_global_batch_size(config, 4) == 128
