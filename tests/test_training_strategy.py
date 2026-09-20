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


@pytest.fixture(autouse=True)
def _reset_accelerate_state():
    """Accelerate keeps one process-wide state; each test must start clean."""
    yield
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


class _Layer(torch.nn.Module):
    pass


class _LinearAttentionLayer(_Layer):
    pass


class _FakeDecoderModel(torch.nn.Module):
    """Minimal module tree with the Qwen3.8 decoder layout and one fp32 parameter."""

    def __init__(self, layer_types=(_Layer, _LinearAttentionLayer)) -> None:
        super().__init__()
        inner = torch.nn.Module()
        inner.language_model = torch.nn.Module()
        inner.language_model.layers = torch.nn.ModuleList([layer_type() for layer_type in layer_types])
        self.model = inner
        self.head = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))


def _fake_decoder_model(layer_types=(_Layer, _LinearAttentionLayer)) -> _FakeDecoderModel:
    return _FakeDecoderModel(layer_types)


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
    # Accelerate only builds the policy from the class names when this function is
    # the configured policy, so an unset policy would wrap the whole model.
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    assert plugin.auto_wrap_policy is transformer_auto_wrap_policy


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
    # FSDP keeps the parameters in bfloat16 through the plugin policy, so
    # Accelerator.mixed_precision stays off (otherwise Accelerate upcasts every
    # flat parameter to float32).
    assert accelerator.mixed_precision == "no"


def test_fsdp_plugin_pins_bf16_parameters() -> None:
    plugin = build_fsdp_plugin(
        {"training": {"strategy": "fsdp", "bf16": True}},
        _fake_decoder_model(),
        fsdp_transformer_cls_names,
    )
    policy = plugin.mixed_precision_policy
    assert policy is not None
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.bfloat16
    assert policy.buffer_dtype is torch.bfloat16


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


class _MainProcessAccelerator:
    def __init__(self, is_main: bool) -> None:
        self.is_main_process = is_main
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def test_broadcast_flag_returns_the_rank_0_decision() -> None:
    from src.training_strategy import broadcast_flag

    accelerator = _MainProcessAccelerator(True)
    assert broadcast_flag(accelerator, True) is True
    assert broadcast_flag(accelerator, False) is False


def test_selection_evaluation_runs_on_every_rank_and_only_main_writes(monkeypatch, tmp_path) -> None:
    from src import train as train_module

    calls: list[dict] = []
    monkeypatch.setattr(train_module, "prepare_model_for_evaluation", lambda model, config: None)
    monkeypatch.setattr(train_module, "_compute_dataset_loss", lambda model, loader: 0.25)

    def fake_evaluate_examples(
        model,
        processor,
        examples,
        config,
        output_dir,
        checkpoint_name,
        sample_prediction_mode=None,
        write_artifacts=True,
    ):
        calls.append({"examples": len(examples), "write_artifacts": write_artifacts})
        return {
            "backend_results": {
                "likelihood": {
                    "headline_metrics": {
                        "positive_f1": 0.5,
                        "macro_f1": 0.5,
                        "precision": 0.5,
                        "recall": 0.5,
                    }
                }
            }
        }

    monkeypatch.setattr(train_module, "evaluate_examples", fake_evaluate_examples)
    components = [
        {
            "name": "daic",
            "dataset": "daic",
            "split_name": "val",
            "config": {"evaluation": {}},
            "examples": [1, 2, 3],
            "loss_loader": [],
            "log_dir_prefix": "selection",
        }
    ]
    config = {"evaluation": {}}
    main = train_module._evaluate_selection_components(
        _MainProcessAccelerator(True), object(), object(), components, config, tmp_path, 1, "likelihood"
    )
    other = train_module._evaluate_selection_components(
        _MainProcessAccelerator(False), object(), object(), components, config, tmp_path, 1, "likelihood"
    )
    # Same examples, same order, same forward count on every rank; only rank 0 writes.
    assert [call["write_artifacts"] for call in calls] == [True, False]
    assert [call["examples"] for call in calls] == [3, 3]
    assert main["component_losses"] == other["component_losses"] == {"daic_loss": 0.25}
    assert main["primary_headline_metrics"] == other["primary_headline_metrics"]
    assert main["component_eval_dirs"] == other["component_eval_dirs"]


def test_save_training_checkpoint_ddp_writes_only_on_main(monkeypatch, tmp_path) -> None:
    from src import training_strategy as strategy_module

    saved: list[tuple] = []
    monkeypatch.setattr(
        strategy_module,
        "resolve_training_strategy",
        lambda config: TRAINING_STRATEGY_DDP,
    )
    monkeypatch.setattr(
        "src.model.runtime.save_adapter_and_processor",
        lambda model, processor, output_dir, config=None: saved.append(str(output_dir)),
    )
    accelerator = _MainProcessAccelerator(True)
    strategy_module.save_training_checkpoint(
        accelerator, "model", "processor", tmp_path / "best", {"training": {"strategy": "ddp"}}
    )
    assert saved == [str(tmp_path / "best")]

    strategy_module.save_training_checkpoint(
        _MainProcessAccelerator(False), "model", "processor", tmp_path / "best", {"training": {"strategy": "ddp"}}
    )
    assert saved == [str(tmp_path / "best")]


class _FsdpAccelerator:
    def __init__(self, is_main: bool) -> None:
        self.is_main_process = is_main
        self.events: list[str] = []

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        self.events.append("gather")
        return {"base_model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight": 1}


def test_save_training_checkpoint_fsdp_gathers_on_every_rank(monkeypatch, tmp_path) -> None:
    from src import training_strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "resolve_training_strategy",
        lambda config: TRAINING_STRATEGY_FSDP,
    )
    written: list[str] = []

    class FakeModel:
        def save_pretrained(self, target, state_dict=None, safe_serialization=False):
            written.append(f"adapter:{target}")

    class FakeProcessor:
        def save_pretrained(self, target):
            written.append(f"processor:{target}")

    config = {"training": {"strategy": "fsdp"}}
    main = _FsdpAccelerator(True)
    strategy_module.save_training_checkpoint(
        main, FakeModel(), FakeProcessor(), tmp_path / "best", config
    )
    assert main.events == ["gather"]
    assert written == [f"adapter:{tmp_path / 'best'}", f"processor:{tmp_path / 'best'}"]

    written.clear()
    other = _FsdpAccelerator(False)
    strategy_module.save_training_checkpoint(
        other, FakeModel(), FakeProcessor(), tmp_path / "best", config
    )
    # Every rank joins the gather; only rank 0 writes.
    assert other.events == ["gather"]
    assert written == []


def test_align_fsdp_model_dtypes_makes_parameters_uniform() -> None:
    from src.training_strategy import align_fsdp_model_dtypes

    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    model.adapter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    optimizer = torch.optim.AdamW([model.base, model.adapter], lr=1e-4)
    resolved = align_fsdp_model_dtypes(model, dtype=torch.bfloat16)
    assert resolved == "torch.bfloat16"
    assert model.base.dtype is torch.bfloat16
    assert model.adapter.dtype is torch.bfloat16
    # In-place data copies keep the optimizer bound to the same parameter objects.
    assert {id(group["params"][0]) for group in optimizer.param_groups} == {id(model.base)}


def test_accelerator_aligns_dtypes_for_fsdp(monkeypatch) -> None:
    from src import training_strategy as strategy_module

    seen: list[object] = []
    original = strategy_module.align_fsdp_model_dtypes

    def spy(model, dtype=None):
        seen.append(dtype)
        return original(model, dtype=dtype)

    monkeypatch.setattr(strategy_module, "align_fsdp_model_dtypes", spy)
    build_accelerator(
        {
            "training": {
                "strategy": "fsdp",
                "run_final_eval_in_train": False,
                "gradient_accumulation_steps": 32,
                "bf16": True,
            }
        },
        model=_fake_decoder_model(),
        wrap_policy_names=fsdp_transformer_cls_names,
    )
    assert len(seen) == 1


def test_qwen38_config_selects_fsdp_and_disables_in_train_eval() -> None:
    config = load_yaml(QWEN38_CONFIG)
    assert config["training"]["strategy"] == TRAINING_STRATEGY_FSDP
    assert config["training"]["run_final_eval_in_train"] is False
    assert config["training"]["gradient_accumulation_steps"] == 32
    assert effective_global_batch_size(config, 4) == 128
