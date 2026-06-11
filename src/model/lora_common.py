from __future__ import annotations

from typing import Any

from peft import LoraConfig


def _resolve_decoder_hidden_layer_count(model_or_config) -> int:
    model_config = getattr(model_or_config, "config", model_or_config)
    candidate_values = [
        getattr(getattr(model_config, "text_config", None), "num_hidden_layers", None),
        getattr(model_config, "num_hidden_layers", None),
        getattr(getattr(model_config, "language_model", None), "num_hidden_layers", None),
    ]
    for value in candidate_values:
        if value is None:
            continue
        count = int(value)
        if count <= 0:
            break
        return count
    raise ValueError(
        "Could not resolve decoder hidden-layer count from model config. "
        "Expected config.text_config.num_hidden_layers or config.num_hidden_layers."
    )


def resolve_lora_layer_selection(config: dict[str, Any], model_or_config) -> dict[str, Any]:
    lora_cfg = config["lora"]
    raw_last_n_layers = lora_cfg.get("last_n_layers")
    if raw_last_n_layers in (None, "", False):
        return {
            "requested_last_n_layers": None,
            "decoder_hidden_layer_count": _resolve_decoder_hidden_layer_count(model_or_config),
            "layers_to_transform": None,
        }

    try:
        last_n_layers = int(raw_last_n_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid lora.last_n_layers={raw_last_n_layers!r}. Expected a positive integer."
        ) from exc
    if last_n_layers <= 0:
        raise ValueError(
            f"Invalid lora.last_n_layers={raw_last_n_layers!r}. Expected a positive integer."
        )

    decoder_hidden_layer_count = _resolve_decoder_hidden_layer_count(model_or_config)
    if last_n_layers > decoder_hidden_layer_count:
        raise ValueError(
            f"Invalid lora.last_n_layers={last_n_layers}. "
            f"Model only has {decoder_hidden_layer_count} decoder layers."
        )

    start_index = decoder_hidden_layer_count - last_n_layers
    return {
        "requested_last_n_layers": last_n_layers,
        "decoder_hidden_layer_count": decoder_hidden_layer_count,
        "layers_to_transform": list(range(start_index, decoder_hidden_layer_count)),
    }


def build_lora_config(config: dict[str, Any], model_or_config) -> tuple[LoraConfig, dict[str, Any]]:
    lora_cfg = config["lora"]
    layer_selection = resolve_lora_layer_selection(config, model_or_config)
    lora_kwargs: dict[str, Any] = {
        "r": int(lora_cfg["rank"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
        "bias": str(lora_cfg["bias"]),
        "target_modules": list(lora_cfg["target_modules"]),
        "task_type": "CAUSAL_LM",
    }
    if layer_selection["layers_to_transform"] is not None:
        lora_kwargs["layers_to_transform"] = list(layer_selection["layers_to_transform"])
    if "exclude_modules" in lora_cfg and lora_cfg["exclude_modules"]:
        lora_kwargs["exclude_modules"] = lora_cfg["exclude_modules"]
    return LoraConfig(**lora_kwargs), layer_selection


def resolved_lora_layer_selection(model) -> dict[str, Any] | None:
    selection = getattr(model, "_resolved_lora_layer_selection", None)
    if selection is None and hasattr(model, "base_model"):
        selection = getattr(model.base_model, "_resolved_lora_layer_selection", None)
    if selection is None:
        return None
    return dict(selection)
