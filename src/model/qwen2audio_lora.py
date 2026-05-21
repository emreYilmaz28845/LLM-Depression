from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, GenerationConfig, Qwen2AudioForConditionalGeneration

from src.utils import get_logger


LOGGER = get_logger(__name__)


def load_processor(model_name_or_path: str):
    return AutoProcessor.from_pretrained(model_name_or_path)


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    lora_cfg = config["lora"]
    return LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg["bias"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    torch_dtype = torch.bfloat16 if bool(config["training"].get("bf16", False)) and torch.cuda.is_available() else None
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False
    if bool(config["training"].get("gradient_checkpointing", False)):
        # PEFT + gradient checkpointing needs gradient-carrying inputs, otherwise
        # checkpointed blocks can drop all grads and DDP reports unused params.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_inference(model_name_or_path: str, adapter_path: str | Path | None = None):
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    return model


def prepare_model_for_evaluation(model) -> None:
    base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    if hasattr(model, "config"):
        model.config.use_cache = True


def build_generation_config(config: dict[str, Any]) -> GenerationConfig:
    do_sample = bool(config["evaluation"]["do_sample"])
    generation_kwargs = {
        "max_new_tokens": int(config["evaluation"]["generation_max_new_tokens"]),
        "num_beams": int(config["evaluation"]["num_beams"]),
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(config["evaluation"].get("temperature", 1.0))
        generation_kwargs["top_p"] = float(config["evaluation"].get("top_p", 1.0))
        generation_kwargs["top_k"] = int(config["evaluation"].get("top_k", 50))
    return GenerationConfig(**generation_kwargs)


def save_adapter_and_processor(model, processor, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    LOGGER.info("Saved adapter and processor to %s", output_dir)
