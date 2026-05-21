from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

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
        model.gradient_checkpointing_enable()
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_inference(model_name_or_path: str, adapter_path: str | Path | None = None):
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def save_adapter_and_processor(model, processor, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    LOGGER.info("Saved adapter and processor to %s", output_dir)
