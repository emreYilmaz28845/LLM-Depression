from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.model.gemma4_io import (
    GEMMA4_EXPECTED_LORA_MODULES,
    GEMMA4_LORA_TARGET_REGEX,
    validate_gemma4_config,
)
from src.model.lora_common import (
    build_lora_config,
    resolve_lora_layer_selection,
)
from src.utils import get_logger


LOGGER = get_logger(__name__)

GEMMA4_PROCESSOR_CLASS_NAME = "Gemma4UnifiedProcessor"
GEMMA4_MODEL_CLASS_NAME = "Gemma4UnifiedForConditionalGeneration"

_PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")


def _import_gemma4_classes():
    """Lazily import the pinned Transformers Gemma4 classes.

    These classes only exist in the dedicated Gemma environment
    (transformers==5.14.1); importing them in the Qwen environment must never
    happen, so every Gemma import stays inside the Gemma code path.
    """
    from transformers import (  # noqa: PLC0415
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedProcessor,
    )

    return Gemma4UnifiedProcessor, Gemma4UnifiedForConditionalGeneration


def _processor_files_present(path: Path) -> bool:
    return any((path / name).is_file() for name in _PROCESSOR_FILES)


def _resolve_processor_path(model_name_or_path: str, config: dict[str, Any] | None) -> str:
    candidate = Path(model_name_or_path)
    if _processor_files_present(candidate):
        return str(candidate)
    from src.utils import resolve_model_name_or_path

    base = resolve_model_name_or_path(None, config or {})
    return str(base)


def load_processor(model_name_or_path: str, config: dict[str, Any] | None = None):
    Gemma4UnifiedProcessor, _ = _import_gemma4_classes()
    path = _resolve_processor_path(model_name_or_path, config)
    processor = Gemma4UnifiedProcessor.from_pretrained(path, local_files_only=True)
    if type(processor).__name__ != GEMMA4_PROCESSOR_CLASS_NAME:
        raise ValueError(
            f"Expected {GEMMA4_PROCESSOR_CLASS_NAME}, got {type(processor).__name__} "
            f"from {path}."
        )
    return processor


def _audit_lora_modules(model, matched_modules: set[str]) -> dict[str, Any]:
    """Run the runbook §4/§9.2 freeze and match audit for the Gemma backend."""
    violations: list[str] = []
    if len(matched_modules) != GEMMA4_EXPECTED_LORA_MODULES:
        violations.append(
            f"expected exactly {GEMMA4_EXPECTED_LORA_MODULES} adapted modules, "
            f"found {len(matched_modules)}"
        )
    vision_adapted = [name for name in matched_modules if "vision" in name or "embed_vision" in name]
    if vision_adapted:
        violations.append(f"vision modules are adapted: {sorted(vision_adapted)[:8]}")
    embedding_adapted = [
        name
        for name in matched_modules
        if "embed_tokens" in name or "lm_head" in name or "embed_audio" in name
    ]
    if embedding_adapted:
        violations.append(f"embedding/LM-head modules are adapted: {sorted(embedding_adapted)[:8]}")

    lora_trainable = 0
    non_lora_trainable: list[str] = []
    audio_projection_trainable = False
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            if parameter.requires_grad:
                lora_trainable += int(parameter.numel())
        elif parameter.requires_grad:
            non_lora_trainable.append(name)
        if "embed_audio.embedding_projection" in name and parameter.requires_grad:
            audio_projection_trainable = True
    if not matched_modules:
        violations.append("no LoRA modules were adapted")
    if lora_trainable <= 0:
        violations.append("no trainable LoRA parameter found")
    if non_lora_trainable:
        violations.append(
            f"non-LoRA parameters are trainable: {sorted(non_lora_trainable)[:8]}"
        )
    if audio_projection_trainable:
        violations.append("model.embed_audio.embedding_projection must be frozen")
    if violations:
        raise ValueError("Gemma LoRA audit failed:\n- " + "\n- ".join(violations))
    return {
        "matched_modules": len(matched_modules),
        "lora_trainable_params": lora_trainable,
        "audit_passed": True,
    }


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    validate_gemma4_config(config)
    _, Gemma4UnifiedForConditionalGeneration = _import_gemma4_classes()
    use_bf16 = bool(config["training"].get("bf16", False)) and torch.cuda.is_available()
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    if bool(config["training"].get("gradient_checkpointing", False)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    lora_config, lora_layer_selection = build_lora_config(config, model)
    from peft import get_peft_model  # noqa: PLC0415

    model = get_peft_model(model, lora_config)
    model._resolved_lora_layer_selection = dict(lora_layer_selection)
    from peft.utils import inspect_matched_modules  # noqa: PLC0415

    matched = {
        str(name) for name in inspect_matched_modules(model.base_model)["matched"]
    }
    audit = _audit_lora_modules(model, matched)
    model.print_trainable_parameters()
    LOGGER.info(
        "Gemma4 LoRA audit | matched_modules=%s lora_trainable_params=%s",
        audit["matched_modules"],
        audit["lora_trainable_params"],
    )
    LOGGER.info(
        "Gemma4 LoRA layer selection | requested_last_n_layers=%s decoder_hidden_layers=%s",
        lora_layer_selection["requested_last_n_layers"],
        lora_layer_selection["decoder_hidden_layer_count"],
    )
    return model


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
):
    validate_gemma4_config(config or {})
    _, Gemma4UnifiedForConditionalGeneration = _import_gemma4_classes()
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if adapter_path:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=False, local_files_only=True
        )
    base_model = _unwrap_base_model(model)
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


def _unwrap_base_model(model):
    return model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model


def prepare_model_for_evaluation(model) -> None:
    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    if hasattr(model, "config"):
        model.config.use_cache = True


def restore_model_for_training(model, config: dict[str, Any]) -> None:
    """Re-apply the training-time memory config after an evaluation pass."""
    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not bool(config["training"].get("gradient_checkpointing", False)):
        return
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(base_model, "enable_input_require_grads"):
        try:
            base_model.enable_input_require_grads()
        except Exception:
            pass
    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        base_model.gradient_checkpointing_enable()


def save_adapter_and_processor(model, processor, output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    LOGGER.info("Saved Gemma4 adapter and processor to %s", output_dir)
