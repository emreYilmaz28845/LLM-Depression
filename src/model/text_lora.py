from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.lora_common import build_lora_config
from src.utils import get_logger


LOGGER = get_logger(__name__)


class TextProcessorAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.feature_extractor = None
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

    def __call__(self, *args, **kwargs):
        kwargs.pop("audio", None)
        kwargs.pop("sampling_rate", None)
        return self.tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, output_dir: str | Path) -> None:
        self.tokenizer.save_pretrained(output_dir)


def load_processor(model_name_or_path: str, config: dict[str, Any] | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return TextProcessorAdapter(tokenizer)


def _set_use_cache(model, enabled: bool) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = bool(enabled)
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model") and hasattr(base_model.model, "config"):
        base_model.model.config.use_cache = bool(enabled)


def _summarize_trainable_parameters(model) -> dict[str, int]:
    lora_trainable_params = 0
    other_trainable_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_trainable_params += parameter.numel()
        else:
            other_trainable_params += parameter.numel()
    return {
        "total_trainable_params": int(lora_trainable_params + other_trainable_params),
        "lora_trainable_params": int(lora_trainable_params),
        "other_trainable_params": int(other_trainable_params),
    }


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    torch_dtype = torch.bfloat16 if bool(config["training"].get("bf16", False)) and torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )
    _set_use_cache(model, enabled=False)
    if bool(config["training"].get("gradient_checkpointing", False)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    lora_config, lora_layer_selection = build_lora_config(config, model)
    model = get_peft_model(model, lora_config)
    model._resolved_lora_layer_selection = dict(lora_layer_selection)
    model.print_trainable_parameters()
    trainable_summary = _summarize_trainable_parameters(model)
    LOGGER.info(
        "LoRA layer selection | requested_last_n_layers=%s decoder_hidden_layers=%s resolved_layers_to_transform=%s",
        lora_layer_selection["requested_last_n_layers"],
        lora_layer_selection["decoder_hidden_layer_count"],
        lora_layer_selection["layers_to_transform"],
    )
    LOGGER.info(
        "Trainable parameter summary | total=%s lora=%s other=%s",
        trainable_summary["total_trainable_params"],
        trainable_summary["lora_trainable_params"],
        trainable_summary["other_trainable_params"],
    )
    return model


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
):
    inference_dtype = str(
        (config or {}).get("evaluation", {}).get("inference_dtype", "fp32")
    ).strip().lower()
    if inference_dtype not in {"fp32", "bf16"}:
        raise ValueError(
            "evaluation.inference_dtype must be one of fp32 or bf16 for the text backend."
        )
    torch_dtype = torch.bfloat16 if inference_dtype == "bf16" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    _set_use_cache(model, enabled=True)
    model.eval()
    return model


def prepare_model_for_evaluation(model) -> None:
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    _set_use_cache(model, enabled=True)


def restore_model_for_training(model, config: dict[str, Any]) -> None:
    """Re-apply training-time memory config (gradient checkpointing, use_cache=False)
    after an evaluation pass, which otherwise leaves checkpointing disabled. Safe to
    call every epoch (disable-then-enable keeps the input-require-grads hook clean)."""
    _set_use_cache(model, enabled=False)
    if not bool(config["training"].get("gradient_checkpointing", False)):
        return
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def save_adapter_and_processor(model, processor, output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    LOGGER.info("Saved adapter and tokenizer to %s", output_dir)
