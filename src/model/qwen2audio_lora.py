from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoProcessor, GenerationConfig, Qwen2AudioForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from src.model.lora_common import build_lora_config, resolved_lora_layer_selection, resolve_lora_layer_selection
from src.utils import get_logger


LOGGER = get_logger(__name__)

AUDIO_ADAPTER_METADATA_FILENAME = "audio_adapter_config.json"
AUDIO_ADAPTER_STATE_FILENAME = "audio_adapter_state.pt"
PROJECTOR_STATE_FILENAME = "multi_modal_projector_state.pt"


class DepAdapter(nn.Module):
    def __init__(self, audio_dim: int, adapter_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(audio_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, audio_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(audio_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        residual = audio_features
        hidden = self.down_proj(audio_features)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.up_proj(hidden)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + residual)


def load_processor(model_name_or_path: str, config: dict[str, Any] | None = None):
    return AutoProcessor.from_pretrained(model_name_or_path)


def resolve_audio_adapter_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_cfg = (config or {}).get("audio_adapter") or {}
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "adapter_dim": int(raw_cfg.get("adapter_dim", 512)),
        "dropout": float(raw_cfg.get("dropout", 0.1)),
        "train_projector": bool(raw_cfg.get("train_projector", False)),
    }


def _unwrap_base_model(model):
    return model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model


def _load_torch_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _reference_parameter(module: nn.Module) -> torch.nn.Parameter | None:
    for parameter in module.parameters():
        return parameter
    return None


def attach_dep_adapter(model, adapter_cfg: dict[str, Any], adapter_state_dict: dict[str, Any] | None = None):
    base_model = _unwrap_base_model(model)
    encoder = base_model.audio_tower
    if getattr(encoder, "_dep_adapter_attached", False):
        if adapter_state_dict is not None and hasattr(encoder, "audio_adapter"):
            encoder.audio_adapter.load_state_dict(adapter_state_dict)
            reference = _reference_parameter(encoder)
            if reference is not None:
                encoder.audio_adapter.to(device=reference.device, dtype=reference.dtype)
        return model

    adapter = DepAdapter(
        audio_dim=int(encoder.config.d_model),
        adapter_dim=int(adapter_cfg["adapter_dim"]),
        dropout=float(adapter_cfg["dropout"]),
    )
    if adapter_state_dict is not None:
        adapter.load_state_dict(adapter_state_dict)
    reference = _reference_parameter(encoder)
    if reference is not None:
        adapter.to(device=reference.device, dtype=reference.dtype)

    original_forward = encoder.forward

    def new_forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self._dep_adapter_original_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if hasattr(outputs, "last_hidden_state"):
            adapted_audio_features = self.audio_adapter(outputs.last_hidden_state)
            return BaseModelOutput(
                last_hidden_state=adapted_audio_features,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        adapted_audio_features = self.audio_adapter(outputs[0])
        return (adapted_audio_features,) + outputs[1:]

    encoder._dep_adapter_original_forward = original_forward
    encoder.forward = new_forward.__get__(encoder, type(encoder))
    encoder.audio_adapter = adapter
    encoder._dep_adapter_attached = True
    encoder._audio_adapter_config = dict(adapter_cfg)
    return model


def configure_trainable_audio_modules(model, audio_adapter_cfg: dict[str, Any]) -> None:
    base_model = _unwrap_base_model(model)
    encoder = getattr(base_model, "audio_tower", None)
    if encoder is not None and hasattr(encoder, "audio_adapter"):
        for parameter in encoder.audio_adapter.parameters():
            parameter.requires_grad = bool(audio_adapter_cfg["enabled"])

    if hasattr(base_model, "multi_modal_projector"):
        for parameter in base_model.multi_modal_projector.parameters():
            parameter.requires_grad = bool(audio_adapter_cfg["train_projector"])


def summarize_audio_module_state(model) -> dict[str, Any]:
    base_model = _unwrap_base_model(model)
    encoder = getattr(base_model, "audio_tower", None)
    adapter_module = getattr(encoder, "audio_adapter", None) if encoder is not None else None
    projector = getattr(base_model, "multi_modal_projector", None)
    return {
        "adapter_attached": adapter_module is not None,
        "adapter_trainable_params": sum(parameter.numel() for parameter in adapter_module.parameters() if parameter.requires_grad)
        if adapter_module is not None
        else 0,
        "projector_present": projector is not None,
        "projector_trainable_params": sum(parameter.numel() for parameter in projector.parameters() if parameter.requires_grad)
        if projector is not None
        else 0,
    }


def summarize_trainable_parameter_groups(model) -> dict[str, int]:
    lora_trainable_params = 0
    other_trainable_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_trainable_params += parameter.numel()
        else:
            other_trainable_params += parameter.numel()

    audio_state = summarize_audio_module_state(model)
    adapter_trainable_params = int(audio_state["adapter_trainable_params"])
    projector_trainable_params = int(audio_state["projector_trainable_params"])
    total_trainable_params = lora_trainable_params + other_trainable_params
    return {
        "total_trainable_params": int(total_trainable_params),
        "lora_trainable_params": int(lora_trainable_params),
        "adapter_trainable_params": adapter_trainable_params,
        "projector_trainable_params": projector_trainable_params,
        "other_trainable_params": int(
            other_trainable_params - adapter_trainable_params - projector_trainable_params
        ),
    }


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    torch_dtype = torch.bfloat16 if bool(config["training"].get("bf16", False)) and torch.cuda.is_available() else None
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )
    if audio_adapter_cfg["enabled"]:
        attach_dep_adapter(model, audio_adapter_cfg)
    model.config.use_cache = False
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
    configure_trainable_audio_modules(model, audio_adapter_cfg)
    model.print_trainable_parameters()
    audio_state = summarize_audio_module_state(model)
    trainable_summary = summarize_trainable_parameter_groups(model)
    LOGGER.info(
        "LoRA layer selection | requested_last_n_layers=%s decoder_hidden_layers=%s resolved_layers_to_transform=%s",
        lora_layer_selection["requested_last_n_layers"],
        lora_layer_selection["decoder_hidden_layer_count"],
        lora_layer_selection["layers_to_transform"],
    )
    LOGGER.info(
        "Trainable parameter summary | total=%s lora=%s adapter=%s projector=%s other=%s",
        trainable_summary["total_trainable_params"],
        trainable_summary["lora_trainable_params"],
        trainable_summary["adapter_trainable_params"],
        trainable_summary["projector_trainable_params"],
        trainable_summary["other_trainable_params"],
    )
    LOGGER.info(
        "Audio adaptation state | adapter_enabled=%s adapter_attached=%s adapter_trainable_params=%s "
        "train_projector=%s projector_present=%s projector_trainable_params=%s",
        audio_adapter_cfg["enabled"],
        audio_state["adapter_attached"],
        audio_state["adapter_trainable_params"],
        audio_adapter_cfg["train_projector"],
        audio_state["projector_present"],
        audio_state["projector_trainable_params"],
    )
    return model


def load_checkpoint_audio_adapter_config(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    metadata_path = Path(checkpoint_dir) / AUDIO_ADAPTER_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_additional_audio_modules(model, checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_cfg = load_checkpoint_audio_adapter_config(checkpoint_dir)
    if not checkpoint_cfg:
        return {
            "enabled": False,
            "train_projector": False,
            "adapter_state_loaded": False,
            "projector_state_loaded": False,
        }

    if bool(checkpoint_cfg.get("enabled", False)):
        adapter_state = None
        adapter_state_path = checkpoint_dir / AUDIO_ADAPTER_STATE_FILENAME
        if adapter_state_path.exists():
            adapter_state = _load_torch_state(adapter_state_path)
        attach_dep_adapter(model, checkpoint_cfg, adapter_state_dict=adapter_state)

    projector_state_loaded = False
    if bool(checkpoint_cfg.get("train_projector", False)):
        base_model = _unwrap_base_model(model)
        projector_state_path = checkpoint_dir / PROJECTOR_STATE_FILENAME
        if projector_state_path.exists() and hasattr(base_model, "multi_modal_projector"):
            base_model.multi_modal_projector.load_state_dict(_load_torch_state(projector_state_path))
            projector_state_loaded = True

    return {
        **checkpoint_cfg,
        "adapter_state_loaded": bool(checkpoint_cfg.get("enabled", False))
        and (checkpoint_dir / AUDIO_ADAPTER_STATE_FILENAME).exists(),
        "projector_state_loaded": projector_state_loaded,
    }


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
):
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)
    checkpoint_audio_cfg = None
    if adapter_path:
        checkpoint_audio_cfg = load_additional_audio_modules(model, adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    if checkpoint_audio_cfg:
        LOGGER.info(
            "Loaded checkpoint audio modules | adapter_enabled=%s train_projector=%s "
            "adapter_state_loaded=%s projector_state_loaded=%s",
            bool(checkpoint_audio_cfg.get("enabled", False)),
            bool(checkpoint_audio_cfg.get("train_projector", False)),
            bool(checkpoint_audio_cfg.get("adapter_state_loaded", False)),
            bool(checkpoint_audio_cfg.get("projector_state_loaded", False)),
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


def save_additional_audio_modules(model, output_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    base_model = _unwrap_base_model(model)
    encoder = getattr(base_model, "audio_tower", None)
    adapter_module = getattr(encoder, "audio_adapter", None) if encoder is not None else None
    projector = getattr(base_model, "multi_modal_projector", None)

    metadata = {
        "format_version": 1,
        "enabled": bool(audio_adapter_cfg["enabled"] and adapter_module is not None),
        "adapter_dim": int(audio_adapter_cfg["adapter_dim"]),
        "dropout": float(audio_adapter_cfg["dropout"]),
        "train_projector": bool(audio_adapter_cfg["train_projector"] and projector is not None),
        "adapter_state_filename": AUDIO_ADAPTER_STATE_FILENAME,
        "projector_state_filename": PROJECTOR_STATE_FILENAME,
    }
    (output_dir / AUDIO_ADAPTER_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if metadata["enabled"]:
        torch.save(adapter_module.state_dict(), output_dir / AUDIO_ADAPTER_STATE_FILENAME)
    if metadata["train_projector"]:
        torch.save(projector.state_dict(), output_dir / PROJECTOR_STATE_FILENAME)
    return metadata


def save_adapter_and_processor(model, processor, output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    aux_metadata = save_additional_audio_modules(model, output_dir, config=config)
    LOGGER.info("Saved adapter and processor to %s", output_dir)
    LOGGER.info("Saved additional audio module metadata: %s", aux_metadata)
