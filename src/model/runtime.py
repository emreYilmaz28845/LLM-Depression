from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import GenerationConfig

from src.model import qwen2audio_lora, qwen3omni_lora, text_lora
from src.model.lora_common import resolve_lora_layer_selection, resolved_lora_layer_selection
from src.utils import (
    INPUT_MODALITY_TEXT_ONLY,
    MODEL_BACKEND_GEMMA4,
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_QWEN3OMNI,
    MODEL_BACKEND_TEXT,
    resolve_input_modality,
    resolve_model_backend,
)


def _backend(config: dict[str, Any]):
    # An explicit model_backend wins over the modality default. This is what lets
    # the same-backbone text-only control (data.use_audio=false) route through the
    # omni Thinker instead of the dense text model (QWEN3_OMNI_IMPLEMENTATION.md §4.2)
    # and what selects the Gemma 4 unified backend.
    backend = resolve_model_backend(config)
    if backend == MODEL_BACKEND_GEMMA4:
        # Lazy: the Gemma classes only exist in the dedicated Gemma environment
        # (transformers==5.14.1). Importing this module in the Qwen environment
        # must never happen while Qwen is selected.
        from src.model import gemma4_lora  # noqa: PLC0415

        return gemma4_lora
    if backend == MODEL_BACKEND_QWEN3OMNI:
        return qwen3omni_lora
    if backend == MODEL_BACKEND_QWEN2AUDIO:
        return qwen2audio_lora
    if backend == MODEL_BACKEND_TEXT:
        return text_lora
    # No explicit backend -> today's behavior: modality picks the backend.
    if resolve_input_modality(config) == INPUT_MODALITY_TEXT_ONLY:
        return text_lora
    return qwen2audio_lora


def build_collator(config: dict[str, Any], processor, debug: bool = False):
    """Backend-dispatched collator factory for train/selection/final DataLoaders."""
    if resolve_model_backend(config) == MODEL_BACKEND_GEMMA4:
        from src.model.gemma4_io import Gemma4SFTCollator  # noqa: PLC0415

        return Gemma4SFTCollator(processor=processor, debug=debug)
    from src.model.collator import Qwen2AudioSFTCollator  # noqa: PLC0415

    return Qwen2AudioSFTCollator(processor=processor, debug=debug)


def prepare_backend_examples(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    processor,
) -> list[dict[str, Any]]:
    """Backend-dispatched example prompt preparation.

    Gemma re-renders ``prompt_text``/``training_text`` from the raw system/user
    fields through its pinned chat template; Qwen keeps the pre-rendered text
    (no-op).
    """
    if resolve_model_backend(config) == MODEL_BACKEND_GEMMA4:
        from src.model.gemma4_io import prepare_gemma4_examples  # noqa: PLC0415

        return prepare_gemma4_examples(examples, config, processor)
    return examples


def load_processor(model_name_or_path: str | Path, config: dict[str, Any]):
    return _backend(config).load_processor(str(model_name_or_path), config=config)


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    return _backend(config).load_model_for_training(model_name_or_path, config)


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
):
    if config is None:
        raise ValueError("load_model_for_inference requires the resolved config to select the model backend.")
    return _backend(config).load_model_for_inference(model_name_or_path, adapter_path=adapter_path, config=config)


def prepare_model_for_evaluation(model, config: dict[str, Any]) -> None:
    _backend(config).prepare_model_for_evaluation(model)


def restore_model_for_training(model, config: dict[str, Any]) -> None:
    _backend(config).restore_model_for_training(model, config)


def save_adapter_and_processor(model, processor, output_dir: str | Path, config: dict[str, Any]) -> None:
    _backend(config).save_adapter_and_processor(model, processor, output_dir, config=config)


def resolve_audio_adapter_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_cfg = (config or {}).get("audio_adapter") or {}
    resolved = {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "adapter_dim": int(raw_cfg.get("adapter_dim", 512)),
        "dropout": float(raw_cfg.get("dropout", 0.1)),
        "train_projector": bool(raw_cfg.get("train_projector", False)),
    }
    if config is not None and resolve_input_modality(config) == INPUT_MODALITY_TEXT_ONLY:
        resolved["enabled"] = False
        resolved["train_projector"] = False
    return resolved


def resolve_processor_sampling_rate(processor) -> int | None:
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None:
        return None
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    if sampling_rate is None:
        return None
    return int(sampling_rate)


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
