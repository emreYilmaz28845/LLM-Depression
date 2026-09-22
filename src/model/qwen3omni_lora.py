"""Qwen3-Omni (Thinker-only) audio+text / audio-only / text-only backend.

Mirrors ``qwen2audio_lora`` but targets the **Thinker** of
``Qwen/Qwen3-Omni-30B-A3B-Instruct`` as the trainable CausalLM. See
``docs/QWEN3OMNI_DAIC_PILOT_PLAN.md`` for the pilot contract. Key differences from
Qwen2-Audio:

* The trainable model is ``Qwen3OmniMoeThinkerForConditionalGeneration`` (the
  talker is never built). If the standalone Thinker can't load its weights
  straight from the Instruct checkpoint, we fall back to the full omni model,
  ``disable_talker()``, and take ``.thinker``; the talker weights are released
  immediately and their absence is audited before training or evaluation.
* The audio encoder is ``thinker.audio_tower`` and its projector lives *inside*
  it (``proj1``/``proj2``/``conv_out``) — there is no separate
  ``multi_modal_projector``. The encoder freeze-guard (the documented overfit
  trap) therefore keys on ``audio_tower`` exactly as before.
* The audio front end does not cast float32 features the way Whisper does inside
  Qwen2-Audio, so the backend installs a dtype boundary on the model: floating
  ``input_features`` are cast to the audio front end's dtype at the forward call,
  while integer masks and indices (``feature_attention_mask``, ``input_ids``,
  ``labels``) are never touched.
* LoRA is restricted to the language-model decoder by an anchored regex: the four
  attention projections of every decoder layer, plus gate/up/down where a layer's
  MLP is dense. The checkpoint's 48 layers are all MoE, so the expert weights are
  fused 3-D parameters in practice; routers (``mlp.gate``) and expert internals
  always stay frozen.
* Likelihood scoring is LM-head only, so ``src/evaluate.py`` is unchanged.

The pure, model-agnostic primitives (``DepAdapter``, the adapter-config resolver,
torch-state IO, the generation-config builder, and the adapter/projector filename
constants) are reused from ``qwen2audio_lora``; everything that touches the model
graph is repointed here so the Thinker submodule wiring stays explicit.
"""

from __future__ import annotations

import gc
import hashlib
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
from src.utils import (
    MODEL_BACKEND_QWEN3OMNI,
    MODEL_LOAD_AUDIT_ATTR,
    get_logger,
    resolve_evaluation_device_map,
    resolve_evaluation_resource_shape,
    resolve_model_backend,
)


LOGGER = get_logger(__name__)

QWEN3OMNI_MODEL_CLASS_NAME = "Qwen3OmniMoeThinkerForConditionalGeneration"
QWEN3OMNI_FULL_MODEL_CLASS_NAME = "Qwen3OmniMoeForConditionalGeneration"
QWEN3OMNI_SUPPORTED_MODEL_CLASSES = (
    QWEN3OMNI_MODEL_CLASS_NAME,
    QWEN3OMNI_FULL_MODEL_CLASS_NAME,
)
# Anchored decoder regex, verified against the real Thinker tree. The checkpoint
# declares 48 decoder layers, every one of them MoE (decoder_sparse_step=1,
# mlp_only_layers=[]), so the only adaptable non-expert modules are the attention
# projections: the expert weights are fused 3-D parameters (gate_up_proj /
# down_proj) that LoRA cannot wrap, the router (mlp.gate) is a TopK router, and
# there is no dense or shared MLP. A dense layer, if a future checkpoint has one,
# would also expose mlp.gate_proj/up_proj/down_proj and is matched here.
QWEN3OMNI_LORA_TARGET_REGEX = (
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
QWEN3OMNI_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
QWEN3OMNI_ALLOWED_EVALUATION_MODES = ("likelihood", "original_teacher_forced")
QWEN3OMNI_ALLOWED_INFERENCE_DTYPES = ("bf16",)
# The offline MN5 wheelhouse has no flash-attn, so sdpa is the only supported
# attention implementation for this backend.
QWEN3OMNI_ALLOWED_ATTN_IMPLEMENTATIONS = ("sdpa",)
QWEN3OMNI_FORBIDDEN_LORA_MARKERS = (
    "audio_tower",
    "visual",
    "vision",
    "embed_tokens",
    "lm_head",
    "talker",
    "multi_modal_projector",
    "mlp.gate",
    ".experts.",
    "shared_expert_gate",
)
AUDIO_FEATURE_DTYPE_ATTR = "_qwen3omni_audio_feature_dtype"


def _is_qwen3omni_backend(config: dict[str, Any]) -> bool:
    return resolve_model_backend(config) == MODEL_BACKEND_QWEN3OMNI


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
    # The offline MN5 wheelhouse has no flash-attn, so sdpa is the only
    # attention implementation that can load here.
    raw = (config or {}).get("model_attn_implementation", "sdpa")
    if raw in (None, "", "none", "default"):
        return None
    return str(raw)


def _set_use_cache(model, enabled: bool) -> None:
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = bool(enabled)


def validate_qwen3omni_config(config: dict[str, Any]) -> None:
    """Fail unless every Qwen3-Omni-specific config invariant holds."""
    if not _is_qwen3omni_backend(config):
        return
    errors: list[str] = []

    def _require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    data_cfg = config.get("data", {})
    _require(bool(data_cfg.get("use_audio", False)), "data.use_audio must be true")
    training_cfg = config.get("training", {})
    _require(bool(training_cfg.get("bf16", False)), "training.bf16 must be true")
    _require(
        str(training_cfg.get("strategy", "")).strip().lower() == "fsdp",
        "training.strategy must be fsdp for the Qwen3-Omni Thinker",
    )
    _require(
        bool(training_cfg.get("gradient_checkpointing", False)),
        "training.gradient_checkpointing must be true",
    )
    _require(
        not bool(training_cfg.get("run_final_eval_in_train", False)),
        "training.run_final_eval_in_train must be false under the fsdp strategy",
    )
    _require(
        str(training_cfg.get("selection_metric", "")) == "inner_val_macro_f1",
        "training.selection_metric must be inner_val_macro_f1",
    )
    _require(
        str(training_cfg.get("selection_metric_mode", "")).lower() == "max",
        "training.selection_metric_mode must be max",
    )
    evaluation_cfg = config.get("evaluation", {})
    _require(
        str(evaluation_cfg.get("sample_prediction_mode", ""))
        in QWEN3OMNI_ALLOWED_EVALUATION_MODES,
        "evaluation.sample_prediction_mode must be likelihood or original_teacher_forced",
    )
    _require(
        str(evaluation_cfg.get("headline_mode", "")) in QWEN3OMNI_ALLOWED_EVALUATION_MODES,
        "evaluation.headline_mode must be likelihood or original_teacher_forced",
    )
    _require(
        evaluation_cfg.get("evaluation_view") == QWEN3OMNI_EVALUATION_VIEW,
        f"evaluation.evaluation_view must be {QWEN3OMNI_EVALUATION_VIEW}",
    )
    _require(
        str(evaluation_cfg.get("inference_dtype", "")).strip().lower()
        in QWEN3OMNI_ALLOWED_INFERENCE_DTYPES,
        "evaluation.inference_dtype must be bf16 (the 30B Thinker only fits that way)",
    )
    lora_cfg = config.get("lora", {})
    _require(
        lora_cfg.get("target_modules") == QWEN3OMNI_LORA_TARGET_REGEX,
        "lora.target_modules must be the exact Qwen3-Omni language-model decoder regex",
    )
    _require(
        not bool(lora_cfg.get("tune_audio_encoder", False)),
        "lora.tune_audio_encoder must be false (audio encoder frozen)",
    )
    audio_adapter_cfg = config.get("audio_adapter") or {}
    _require(
        not bool(audio_adapter_cfg.get("enabled", False)),
        "audio_adapter.enabled must be false (the canonical recipe trains no audio adapter)",
    )
    _require(
        not bool(audio_adapter_cfg.get("train_projector", False)),
        "audio_adapter.train_projector must be false",
    )
    attention_implementation = str(config.get("model_attn_implementation", "")).strip().lower()
    _require(
        attention_implementation in QWEN3OMNI_ALLOWED_ATTN_IMPLEMENTATIONS,
        "model_attn_implementation must be one of "
        f"{list(QWEN3OMNI_ALLOWED_ATTN_IMPLEMENTATIONS)}; the offline MN5 wheelhouse "
        "has no flash-attn",
    )
    _require(
        bool(str(config.get("model_name_or_path", "")).strip()),
        "model_name_or_path must point at the offline Qwen3-Omni snapshot",
    )
    if errors:
        raise ValueError("Invalid qwen3omni config:\n- " + "\n- ".join(errors))


def snapshot_identity(model_name_or_path: str | Path) -> dict[str, Any]:
    """Offline snapshot identity for provenance: config hash and declared classes."""
    root = Path(model_name_or_path)
    config_path = root / "config.json"
    identity: dict[str, Any] = {
        "snapshot_dir": str(root),
        "config_path": str(config_path),
        "config_sha256": None,
        "declared_architectures": [],
    }
    if config_path.is_file():
        raw = config_path.read_bytes()
        identity["config_sha256"] = hashlib.sha256(raw).hexdigest()
        identity["declared_architectures"] = list(
            json.loads(raw.decode("utf-8")).get("architectures") or []
        )
    return identity


SNAPSHOT_STRUCTURE_KEYS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "intermediate_size",
)


def snapshot_structure_metadata(model_name_or_path: str | Path) -> dict[str, Any]:
    """Depth and MoE metadata declared by the snapshot, for parameter reporting.

    The values stay ``None`` when the checkpoint does not declare the key, so a
    report never states a number the checkpoint does not carry.
    """
    metadata: dict[str, Any] = {key: None for key in SNAPSHOT_STRUCTURE_KEYS}
    config_path = Path(model_name_or_path) / "config.json"
    if not config_path.is_file():
        return metadata
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = [raw]
    for key in ("text_config", "thinker_config", "language_config"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
            for inner_key in ("text_config", "language_config"):
                inner = nested.get(inner_key)
                if isinstance(inner, dict):
                    candidates.append(inner)
    for key in SNAPSHOT_STRUCTURE_KEYS:
        for candidate in candidates:
            if candidate.get(key) is not None:
                metadata[key] = candidate[key]
                break
    return metadata


def parameter_inventory(model) -> dict[str, int]:
    """Total and trainable parameter counts of the loaded model as it stands."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def resolve_qwen3omni_model_class(model_name_or_path: str | Path) -> str:
    """Check the declared architectures against the allowlist and return the class name.

    The pilot loads the standalone Thinker first and falls back to the full omni
    model only to extract ``.thinker``. Anything else in ``config.json`` is a
    different checkpoint and fails closed.
    """
    declared = list(snapshot_identity(model_name_or_path)["declared_architectures"])
    unsupported = [name for name in declared if name not in QWEN3OMNI_SUPPORTED_MODEL_CLASSES]
    if not declared or unsupported:
        raise ValueError(
            f"Expected architectures {list(QWEN3OMNI_SUPPORTED_MODEL_CLASSES)} in "
            f"{Path(model_name_or_path) / 'config.json'}, got {declared!r}."
        )
    return QWEN3OMNI_MODEL_CLASS_NAME


def _qwen3omni_decoder_layers_path(model) -> str:
    """Dotted path of the decoder-layer ModuleList inside ``model``.

    Structure-based rather than name-based: the decoder is the ModuleList whose
    children carry both ``self_attn`` and ``mlp``. The audio encoder's layer list
    has ``self_attn`` but no ``mlp``, and the vision tower's blocks have ``mlp``
    but no ``self_attn``, so neither can be mistaken for the decoder.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                return name
    raise ValueError(
        "Could not locate the Qwen3-Omni Thinker decoder layers; expected a "
        "ModuleList of layers carrying self_attn and mlp."
    )


def _qwen3omni_decoder_layers(model):
    """The Thinker's decoder-layer ModuleList (duck-typed lookups allowed).

    Resolved against the PEFT base model, so a wrapped model and its base agree.
    """
    base = (
        model.base_model.model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model")
        else model
    )
    target = base
    for part in _qwen3omni_decoder_layers_path(base).split("."):
        target = getattr(target, part)
    return target


def _decoder_module_prefix(model) -> str:
    """Decoder-layer path relative to the PEFT base model (``model.layers``).

    ``inspect_matched_modules`` reports names relative to ``model.base_model``
    (the Thinker), so the audit must use the same frame of reference.
    """
    base = (
        model.base_model.model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model")
        else model
    )
    return _qwen3omni_decoder_layers_path(base)


def fsdp_transformer_cls_names(model) -> list[str]:
    """Decoder layer classes FSDP should wrap for the Qwen3-Omni Thinker."""
    layers = _qwen3omni_decoder_layers(model)
    if len(layers) == 0:
        raise ValueError("The Qwen3-Omni language-model decoder has no layers to wrap.")
    return sorted({type(layer).__name__ for layer in layers})


def expected_lora_module_names(model) -> set[str]:
    """Decoder module names the anchored regex must adapt in this model.

    Derived from the instantiated tree: every layer contributes its attention
    projections, and a layer whose MLP is dense (not MoE) also contributes its
    gate/up/down projections. Routers, routed experts and fused expert
    parameters are never part of the expected set.
    """
    layers = _qwen3omni_decoder_layers(model)
    prefix = _decoder_module_prefix(model)
    expected: set[str] = set()
    for layer_index, layer in enumerate(layers):
        suffixes = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
        mlp = getattr(layer, "mlp", None)
        if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            suffixes += [
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        for suffix in suffixes:
            expected.add(f"{prefix}.{layer_index}.{suffix}")
    return expected


def _audit_qwen3omni_lora_modules(model, matched_modules: set[str]) -> dict[str, Any]:
    """Fail unless LoRA covers exactly the expected language-model decoder modules."""
    violations: list[str] = []
    matched_modules = {str(name) for name in matched_modules}
    if not matched_modules:
        violations.append(
            "no LoRA modules were adapted; check lora.target_modules against the "
            "model's module names"
        )
    forbidden = sorted(
        name
        for name in matched_modules
        if any(marker in name for marker in QWEN3OMNI_FORBIDDEN_LORA_MARKERS)
    )
    if forbidden:
        violations.append(
            f"routers, experts or non-decoder modules are adapted: {forbidden[:8]}"
        )
    decoder_prefix = f"{_decoder_module_prefix(model)}."
    outside_decoder = sorted(
        name for name in matched_modules if decoder_prefix not in f"{name}."
    )
    if outside_decoder:
        violations.append(
            f"modules outside the language-model decoder are adapted: {outside_decoder[:8]}"
        )
    expected = expected_lora_module_names(model)
    missing = sorted(expected - matched_modules)
    unexpected = sorted(matched_modules - expected)
    if missing:
        violations.append(
            f"expected decoder modules were not adapted ({len(missing)}): {missing[:8]}"
        )
    if unexpected:
        violations.append(f"unexpected modules were adapted: {unexpected[:8]}")

    lora_trainable = 0
    non_lora_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_trainable += int(parameter.numel())
        else:
            non_lora_trainable.append(name)
    if lora_trainable <= 0:
        violations.append("no trainable LoRA parameter found")
    if non_lora_trainable:
        violations.append(
            f"non-LoRA parameters are trainable: {sorted(non_lora_trainable)[:8]}"
        )
    if violations:
        raise ValueError("Qwen3-Omni LoRA audit failed:\n- " + "\n- ".join(violations))
    return {
        "matched_modules": len(matched_modules),
        "expected_modules": len(expected),
        "lora_trainable_params": lora_trainable,
        "audit_passed": True,
    }


def audit_talker_absence(model, loaded_class: str | None = None) -> dict[str, Any]:
    """Fail unless no talker parameter, buffer or module is present."""
    talker_parameters = [name for name, _ in model.named_parameters() if "talker" in name]
    talker_buffers = [name for name, _ in model.named_buffers() if "talker" in name]
    if talker_parameters or talker_buffers:
        raise ValueError(
            "Qwen3-Omni talker tensors are present in the loaded model: "
            f"{sorted(talker_parameters + talker_buffers)[:8]}"
        )
    return {
        "loaded_class": loaded_class or type(model).__name__,
        "load_mode": getattr(model, "_qwen3omni_load_mode", "unknown"),
        "talker_parameters": 0,
    }


def _audio_front_end_dtype(model) -> torch.dtype | None:
    """dtype of the audio front end's parameters, falling back to the model dtype."""
    base_model = _unwrap_base_model(model)
    encoder = getattr(base_model, "audio_tower", None)
    if encoder is not None:
        for parameter in encoder.parameters():
            return parameter.dtype
    for parameter in model.parameters():
        return parameter.dtype
    return None


def _device_map_summary(model) -> dict[str, Any] | None:
    """Compact device-map summary (device -> module count) for a sharded model."""
    raw = getattr(model, "hf_device_map", None)
    if not raw:
        return None
    counts: dict[str, int] = {}
    for device in raw.values():
        key = str(device)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def normalize_audio_feature_dtype(
    tensors: dict[str, Any], target_dtype: torch.dtype | None
) -> dict[str, Any]:
    """Cast floating audio features to the audio front end's dtype.

    Integer tensors (``feature_attention_mask``, ``input_ids``, ``labels``) and
    every other entry pass through unchanged; the mapping is only copied when a
    cast is needed.
    """
    if target_dtype is None:
        return tensors
    features = tensors.get("input_features")
    if not torch.is_tensor(features) or not features.is_floating_point():
        return tensors
    if features.dtype == target_dtype:
        return tensors
    normalized = dict(tensors)
    normalized["input_features"] = features.to(target_dtype)
    return normalized


def _cast_audio_features_pre_forward_hook(module, args, kwargs):
    """Forward pre-hook: cast floating ``input_features`` to the model dtype."""
    if not isinstance(kwargs, dict):
        return None
    features = kwargs.get("input_features")
    if not torch.is_tensor(features) or not features.is_floating_point():
        return None
    target_dtype = _audio_front_end_dtype(module)
    if target_dtype is not None and features.dtype != target_dtype:
        kwargs["input_features"] = features.to(target_dtype)
    return None


def install_audio_feature_dtype_boundary(model) -> None:
    """Install the audio dtype boundary once per loaded model."""
    if getattr(model, "_qwen3omni_audio_dtype_boundary_installed", False):
        return
    model.register_forward_pre_hook(_cast_audio_features_pre_forward_hook, with_kwargs=True)
    model._qwen3omni_audio_dtype_boundary_installed = True
    LOGGER.info(
        "Qwen3-Omni audio dtype boundary installed | target_dtype=%s",
        _audio_front_end_dtype(model),
    )


def _encoder_hidden_size(encoder) -> int:
    """Hidden size of the audio encoder's emitted features.

    Qwen3-Omni's ``audio_tower`` projects internally (``proj2`` -> ``output_dim``),
    so the ``last_hidden_state`` the (optional) DepAdapter wraps is ``output_dim``-
    dimensional, falling back to ``d_model`` if a build lacks ``output_dim``."""
    cfg = encoder.config
    return int(getattr(cfg, "output_dim", None) or cfg.d_model)


def _load_thinker_base(
    model_name_or_path: str,
    *,
    torch_dtype,
    attn_implementation: str | None,
    device_map: str | None = None,
):
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    resolve_qwen3omni_model_class(model_name_or_path)
    load_kwargs: dict[str, Any] = {"local_files_only": True}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        load_kwargs["device_map"] = device_map

    try:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        LOGGER.info("Loaded Qwen3-Omni Thinker directly (talker never built) from %s.", model_name_or_path)
        return model, "thinker_direct"
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
    # The talker must never be trained, saved, evaluated or kept on GPU: drop it
    # before anything else looks at the model. The absence audit runs after load.
    try:
        full.talker = None
    except Exception:  # pragma: no cover - only if the attribute is read-only
        pass
    del full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("Loaded Qwen3-Omni Thinker via the full omni model (talker disabled and released).")
    return thinker, "talker_disabled"


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
    validate_qwen3omni_config(config)
    snapshot = snapshot_identity(model_name_or_path)
    torch_dtype = torch.bfloat16 if bool(config["training"].get("bf16", False)) and torch.cuda.is_available() else None
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    model, load_mode = _load_thinker_base(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=_resolve_attn_implementation(config),
    )
    loaded_class = type(model).__name__
    model._qwen3omni_load_mode = load_mode
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
    model._qwen3omni_load_mode = load_mode
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
    from peft.tuners.tuners_utils import inspect_matched_modules

    matched = {str(name) for name in inspect_matched_modules(model.base_model)["matched"]}
    lora_audit = _audit_qwen3omni_lora_modules(model, matched)
    talker_audit = audit_talker_absence(model, loaded_class)
    install_audio_feature_dtype_boundary(model)
    setattr(
        model,
        MODEL_LOAD_AUDIT_ATTR,
        {
            **snapshot,
            **snapshot_structure_metadata(model_name_or_path),
            **talker_audit,
            **lora_audit,
            **parameter_inventory(model),
            "training_dtype": str(torch_dtype),
            "audio_feature_dtype": str(_audio_front_end_dtype(model)),
            "audio_freeze_guard": audio_freeze_summary,
            "lora_layer_selection": lora_layer_selection,
        },
    )
    LOGGER.info(
        "Qwen3-Omni load audit | %s",
        json.dumps(getattr(model, MODEL_LOAD_AUDIT_ATTR), sort_keys=True),
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
    config = config or {}
    validate_qwen3omni_config(config)
    snapshot = snapshot_identity(model_name_or_path)
    inference_dtype = (
        str((config.get("evaluation") or {}).get("inference_dtype", "")).strip().lower()
    )
    if inference_dtype not in QWEN3OMNI_ALLOWED_INFERENCE_DTYPES:
        raise ValueError(
            "evaluation.inference_dtype must be bf16 for the Qwen3-Omni backend; "
            f"got {inference_dtype!r}. A 30B Thinker does not fit in fp32."
        )
    # The 30B Thinker does not fit on one H100 in bf16, so a config that declares
    # a sharded evaluation shape loads with a device map instead of one device.
    resource_shape = resolve_evaluation_resource_shape(config)
    device_map = resolve_evaluation_device_map(config)
    model, load_mode = _load_thinker_base(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=_resolve_attn_implementation(config),
        device_map=device_map,
    )
    loaded_class = type(model).__name__
    model._qwen3omni_load_mode = load_mode
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
    talker_audit = audit_talker_absence(model, loaded_class)
    install_audio_feature_dtype_boundary(model)
    setattr(
        model,
        MODEL_LOAD_AUDIT_ATTR,
        {
            **snapshot,
            **snapshot_structure_metadata(model_name_or_path),
            **talker_audit,
            **parameter_inventory(model),
            "inference_dtype": inference_dtype,
            "audio_feature_dtype": str(_audio_front_end_dtype(model)),
            "evaluation_resource_shape": resource_shape,
            "evaluation_device_map": _device_map_summary(model),
        },
    )
    LOGGER.info(
        "Qwen3-Omni inference load audit | %s",
        json.dumps(getattr(model, MODEL_LOAD_AUDIT_ATTR), sort_keys=True),
    )
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
