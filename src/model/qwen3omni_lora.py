"""Qwen3-Omni (Thinker-only) audio+text / audio-only / text-only backend.

Mirrors ``qwen2audio_lora`` but targets the **Thinker** of
``Qwen/Qwen3-Omni-30B-A3B-Instruct`` as the trainable CausalLM. See
QWEN3_OMNI_IMPLEMENTATION.md for the full plan. Key differences from Qwen2-Audio:

* The trainable model is ``Qwen3OmniMoeThinkerForConditionalGeneration`` (the
  talker is never built). If the standalone Thinker can't load its weights
  straight from the Instruct checkpoint, we fall back to the full omni model,
  ``disable_talker()``, and take ``.thinker``.
* The audio encoder is ``thinker.audio_tower`` and its projector lives *inside*
  it (``proj1``/``proj2``/``conv_out``) — there is no separate
  ``multi_modal_projector``. The encoder freeze-guard (the documented overfit
  trap) therefore keys on ``audio_tower`` exactly as before.
* Likelihood scoring is LM-head only, so ``src/evaluate.py`` is unchanged.

The pure, model-agnostic primitives (``DepAdapter``, the adapter-config resolver,
torch-state IO, the generation-config builder, and the adapter/projector filename
constants) are reused from ``qwen2audio_lora``; everything that touches the model
graph is repointed here so the Thinker submodule wiring stays explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, get_peft_model
from transformers.modeling_outputs import BaseModelOutput

from src.model.lora_common import build_lora_config
from src.model.qwen2audio_lora import (
    AUDIO_ADAPTER_METADATA_FILENAME,
    AUDIO_ADAPTER_STATE_FILENAME,
    PROJECTOR_STATE_FILENAME,
    DepAdapter,
    _load_torch_state,
    _reference_parameter,
    build_generation_config,
    load_checkpoint_audio_adapter_config,
    resolve_audio_adapter_config,
)
from src.utils import get_logger


LOGGER = get_logger(__name__)


def load_processor(model_name_or_path: str, config: dict[str, Any] | None = None):
    from transformers import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)


def _unwrap_base_model(model):
    """Strip the PEFT wrapper, then descend into ``.thinker`` if a full omni model
    slipped through. Our loaders return the Thinker directly, so the descent is a
    defensive no-op in the common path."""
    base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if hasattr(base, "thinker") and hasattr(base.thinker, "audio_tower"):
        return base.thinker
    return base


def _resolve_attn_implementation(config: dict[str, Any] | None) -> str | None:
    raw = (config or {}).get("model_attn_implementation", "flash_attention_2")
    if raw in (None, "", "none", "default"):
        return None
    return str(raw)


def _set_use_cache(model, enabled: bool) -> None:
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = bool(enabled)


def _encoder_hidden_size(encoder) -> int:
    """Hidden size of the audio encoder's emitted features.

    Qwen3-Omni's ``audio_tower`` projects internally (``proj2`` -> ``output_dim``),
    so the ``last_hidden_state`` the (optional) DepAdapter wraps is ``output_dim``-
    dimensional, falling back to ``d_model`` if a build lacks ``output_dim``."""
    cfg = encoder.config
    return int(getattr(cfg, "output_dim", None) or cfg.d_model)


def _load_thinker_base(model_name_or_path: str, *, torch_dtype, attn_implementation: str | None):
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    load_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        LOGGER.info("Loaded Qwen3-Omni Thinker directly (talker never built) from %s.", model_name_or_path)
        return model
    except Exception as exc:  # noqa: BLE001 - any load failure -> robust full-model fallback
        LOGGER.warning(
            "Direct Qwen3-Omni Thinker load failed (%s); falling back to the full omni model "
            "+ disable_talker() and taking .thinker.",
            exc,
        )

    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
    if hasattr(full, "disable_talker"):
        full.disable_talker()
    thinker = full.thinker
    LOGGER.info("Loaded Qwen3-Omni Thinker via the full omni model (talker disabled).")
    return thinker


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
        audio_dim=_encoder_hidden_size(encoder),
        adapter_dim=int(adapter_cfg["adapter_dim"]),
        dropout=float(adapter_cfg["dropout"]),
    )
    if adapter_state_dict is not None:
        adapter.load_state_dict(adapter_state_dict)
    reference = _reference_parameter(encoder)
    if reference is not None:
        adapter.to(device=reference.device, dtype=reference.dtype)

    original_forward = encoder.forward

    def new_forward(self, *args, **kwargs):
        outputs = self._dep_adapter_original_forward(*args, **kwargs)
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

    # Qwen3-Omni has no separate multi_modal_projector (projection is inside
    # audio_tower); the getattr stays for parity and is simply a no-op here.
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


def enforce_audio_encoder_freeze(model, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify-and-enforce that no trainable LoRA weights live in the audio encoder.

    Identical semantics to the Qwen2-Audio guard: ``exclude_modules`` should already
    keep LoRA out of ``audio_tower`` (which, for Qwen3-Omni, also contains the
    audio projector), but this scans the real parameters and freezes anything that
    still landed there. Because LoRA ``B`` is zero-initialised, a frozen-from-start
    encoder LoRA is a no-op, so the encoder's forward stays unchanged.
    """
    tune_audio_encoder = bool(((config or {}).get("lora") or {}).get("tune_audio_encoder", False))
    leaked_params = 0
    frozen_params = 0
    leaked_names: list[str] = []
    for name, parameter in model.named_parameters():
        if "audio_tower" not in name or "lora_" not in name:
            continue
        if not parameter.requires_grad:
            continue
        leaked_params += parameter.numel()
        if len(leaked_names) < 8:
            leaked_names.append(name)
        if not tune_audio_encoder:
            parameter.requires_grad = False
            frozen_params += parameter.numel()
    summary = {
        "tune_audio_encoder": tune_audio_encoder,
        "leaked_lora_params": int(leaked_params),
        "frozen_lora_params": int(frozen_params),
        "leaked_examples": leaked_names,
    }
    if leaked_params and not tune_audio_encoder:
        LOGGER.warning(
            "Audio-encoder freeze guard: found %s trainable LoRA params under audio_tower "
            "despite exclude_modules; froze them. Examples: %s",
            leaked_params,
            leaked_names,
        )
    elif leaked_params and tune_audio_encoder:
        LOGGER.info(
            "Audio-encoder freeze guard: %s trainable LoRA params under audio_tower (tune_audio_encoder=true, kept).",
            leaked_params,
        )
    else:
        LOGGER.info("Audio-encoder freeze guard: 0 trainable LoRA params under audio_tower (encoder frozen).")
    return summary


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    torch_dtype = torch.bfloat16 if bool(config["training"].get("bf16", False)) and torch.cuda.is_available() else None
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    model = _load_thinker_base(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=_resolve_attn_implementation(config),
    )
    if audio_adapter_cfg["enabled"]:
        attach_dep_adapter(model, audio_adapter_cfg)
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
    configure_trainable_audio_modules(model, audio_adapter_cfg)
    audio_freeze_summary = enforce_audio_encoder_freeze(model, config)
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
        "train_projector=%s projector_present=%s projector_trainable_params=%s "
        "tune_audio_encoder=%s encoder_lora_leaked_params=%s encoder_lora_frozen_params=%s",
        audio_adapter_cfg["enabled"],
        audio_state["adapter_attached"],
        audio_state["adapter_trainable_params"],
        audio_adapter_cfg["train_projector"],
        audio_state["projector_present"],
        audio_state["projector_trainable_params"],
        audio_freeze_summary["tune_audio_encoder"],
        audio_freeze_summary["leaked_lora_params"],
        audio_freeze_summary["frozen_lora_params"],
    )
    return model


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
    model = _load_thinker_base(
        model_name_or_path,
        torch_dtype=None,
        attn_implementation=_resolve_attn_implementation(config),
    )
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
    _set_use_cache(base_model, enabled=True)
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
    _set_use_cache(base_model, enabled=True)
    if hasattr(model, "config"):
        model.config.use_cache = True


def restore_model_for_training(model, config: dict[str, Any]) -> None:
    """Re-apply the training-time memory config after an evaluation pass.

    ``prepare_model_for_evaluation`` disables gradient checkpointing and turns on
    ``use_cache``; ``model.train()`` undoes neither. Disable-then-enable keeps the
    input-require-grads hook clean, so this is safe to call every epoch.
    """
    base_model = _unwrap_base_model(model)
    _set_use_cache(base_model, enabled=False)
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
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        base_model.gradient_checkpointing_enable()


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
