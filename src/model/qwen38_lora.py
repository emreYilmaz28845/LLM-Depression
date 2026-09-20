"""Qwen3.8-27B text-only backend for the harmonized depression recipe.

Qwen3.8 is a unified conditional-generation model: ``Qwen3_5ForConditionalGeneration``
(text decoder under ``model.language_model``, plus an unused vision tower). This
backend adds the text-only control path for that architecture:

* the pinned architecture class is imported lazily, so the Qwen2-Audio and Gemma
  environments never pull it in by accident;
* prompts, label spans and the assistant terminator are rendered with the model's
  own chat template with ``enable_thinking=False``. The template otherwise opens an
  unterminated ``<think>`` block and the label span would be scored against
  reasoning tokens instead of the answer;
* LoRA is restricted to the language-model decoder layers, so the vision tower,
  the embeddings and the LM head stay frozen. The decoder is a hybrid: 16 layers
  carry full ``self_attn`` and 48 carry ``linear_attn`` (gated delta net), and every
  layer has an MLP. This first experiment adapts each layer's MLP and the four
  full-attention projections (256 modules) and leaves the linear-attention block
  untouched, so the adapted capacity sits in modules whose behaviour matches the
  Qwen2 recipe exactly. Adding ``linear_attn.out_proj`` (304 modules) is a separate
  comparison experiment on the same split and evaluation, not part of this config;
* training, checkpoint reload and likelihood scoring all consume the same
  ``prompt_text`` / ``training_text`` produced by ``prepare_qwen38_examples``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.model.lora_common import build_lora_config
from src.utils import (
    INPUT_MODALITY_TEXT_ONLY,
    MODEL_BACKEND_QWEN38,
    get_logger,
    resolve_input_modality,
    resolve_model_backend,
)


LOGGER = get_logger(__name__)

QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN38_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN38_MODEL_CLASS_NAME = "Qwen3_5ForConditionalGeneration"
QWEN38_SUPPORTED_MODEL_CLASSES = (QWEN38_MODEL_CLASS_NAME,)
QWEN38_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
QWEN38_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
QWEN38_ALLOWED_EVALUATION_MODES = ("likelihood", "original_teacher_forced")
QWEN38_ALLOWED_INFERENCE_DTYPES = ("bf16",)
QWEN38_ASSISTANT_TURN_ANCHOR = "<|im_start|>assistant\n"
QWEN38_FORBIDDEN_LORA_MARKERS = (
    "vision",
    "visual",
    "embed_tokens",
    "lm_head",
    "audio_tower",
    "multi_modal_projector",
)


def _is_qwen38_backend(config: dict[str, Any]) -> bool:
    return resolve_model_backend(config) == MODEL_BACKEND_QWEN38


class Qwen38ProcessorAdapter:
    """Tokenizer adapter exposing the processor surface the pipeline expects.

    The Qwen3.8 training and evaluation paths are text-only, so the tokenizer is
    enough. ``feature_extractor`` stays ``None``: ``resolve_processor_sampling_rate``
    then reports no sampling rate and no audio key can reach the model.
    """

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

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, output_dir: str | Path) -> None:
        self.tokenizer.save_pretrained(output_dir)


def load_processor(model_name_or_path: str, config: dict[str, Any] | None = None):
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path), local_files_only=True
    )
    return Qwen38ProcessorAdapter(tokenizer)


def _qwen38_messages(system_text: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def render_qwen38_prompt(processor, system_text: str, user_text: str) -> str:
    """Render the generation prompt with thinking disabled.

    With ``enable_thinking=False`` the chat template closes an empty thinking
    block (``<think>\\n\\n</think>\\n\\n``) after the assistant header, which is the
    exact prefix the model answers from. With thinking enabled (or unset) the
    template leaves the block open and the training label span would sit inside
    a reasoning trace.
    """
    return processor.apply_chat_template(
        _qwen38_messages(system_text, user_text),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def render_qwen38_training_text(
    processor, system_text: str, user_text: str, label_text: str
) -> str:
    """Return ``prompt_text + label_text + assistant terminator``.

    The terminator is read back from the model's own template instead of being
    hard-coded: the full conversation is rendered with the assistant answer and
    the suffix after the label becomes the turn terminator. The training text
    then starts with the byte-identical generation prompt, so the prompt/label
    boundary used by the collator and by likelihood scoring is the same one the
    model was prompted with.
    """
    prompt_text = render_qwen38_prompt(processor, system_text, user_text)
    full_text = processor.apply_chat_template(
        _qwen38_messages(system_text, user_text)
        + [{"role": "assistant", "content": label_text}],
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=False,
    )
    head, anchor, tail = full_text.rpartition(QWEN38_ASSISTANT_TURN_ANCHOR)
    if not anchor:
        raise ValueError(
            "Qwen3.8 chat template did not render an assistant turn; cannot derive "
            "the label terminator."
        )
    if not tail.startswith(label_text):
        raise ValueError(
            "Qwen3.8 chat template did not place the label directly after the "
            "assistant header."
        )
    terminator = tail[len(label_text) :]
    if not terminator:
        raise ValueError(
            "Qwen3.8 chat template produced no assistant turn terminator after the label."
        )
    if not prompt_text.startswith(head + anchor):
        raise ValueError(
            "Qwen3.8 generation prompt and the rendered assistant turn diverge; "
            "training and scoring would use different tokenization."
        )
    return f"{prompt_text}{label_text}{terminator}"


def prepare_qwen38_example(
    example: dict[str, Any],
    config: dict[str, Any],
    processor,
) -> dict[str, Any]:
    """Render the Qwen3.8 prompt and training text for one example.

    Requires ``prompt_system_text`` and ``prompt_user_text`` on the example.
    Returns a shallow copy updated only in the backend-rendered prompt fields;
    subject IDs, labels, transcripts, weights and audio plans are untouched.
    """
    system_text = example.get("prompt_system_text")
    user_text = example.get("prompt_user_text")
    if not isinstance(system_text, str) or not system_text.strip():
        raise ValueError(
            f"Qwen3.8 example {example.get('sample_id', '')} is missing "
            "prompt_system_text."
        )
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError(
            f"Qwen3.8 example {example.get('sample_id', '')} is missing "
            "prompt_user_text."
        )
    label_text = example["internal_label_text"]
    prepared = dict(example)
    prepared["prompt_text"] = render_qwen38_prompt(processor, system_text, user_text)
    prepared["training_text"] = render_qwen38_training_text(
        processor, system_text, user_text, label_text
    )
    return prepared


def prepare_qwen38_examples(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    processor,
) -> list[dict[str, Any]]:
    return [
        prepare_qwen38_example(example, config, processor)
        if _is_qwen38_backend(config)
        else example
        for example in examples
    ]


def _import_qwen38_model_class(class_name: str):
    import transformers  # noqa: PLC0415

    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"{class_name} is not available in transformers {transformers.__version__}. "
            "The Qwen3.8 backend needs a transformers build that ships the qwen3_5 "
            "architecture (the pinned Qwen3.8 environment uses 5.8.0)."
        )
    return model_class


def resolve_qwen38_model_class(model_name_or_path: str | Path):
    """Resolve the pinned architecture class from the local model directory.

    The declared ``architectures`` entry is checked against the allowlist first,
    so a directory carrying a different backbone fails with the intended message
    instead of loading an unintended class.
    """
    config_path = Path(model_name_or_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "The Qwen3.8 backend loads a local model directory; no config.json "
            f"found under {str(model_name_or_path)!r}."
        )
    declared = list(
        json.loads(config_path.read_text(encoding="utf-8")).get("architectures") or []
    )
    unsupported = [name for name in declared if name not in QWEN38_SUPPORTED_MODEL_CLASSES]
    if not declared or unsupported:
        raise ValueError(
            f"Expected architectures {list(QWEN38_SUPPORTED_MODEL_CLASSES)} in "
            f"{config_path}, got {declared!r}."
        )
    return _import_qwen38_model_class(QWEN38_MODEL_CLASS_NAME)


def validate_qwen38_config(config: dict[str, Any]) -> None:
    """Fail unless every Qwen3.8-specific config invariant holds."""
    if not _is_qwen38_backend(config):
        return
    errors: list[str] = []

    def _require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    _require(
        resolve_input_modality(config) == INPUT_MODALITY_TEXT_ONLY,
        "the Qwen3.8 backend is introduced for the text-only control; "
        "data.use_audio must be false",
    )
    data_cfg = config.get("data", {})
    _require(
        not bool(data_cfg.get("use_audio", False)),
        "data.use_audio must be false",
    )
    training_cfg = config.get("training", {})
    _require(bool(training_cfg.get("bf16", False)), "training.bf16 must be true")
    _require(
        bool(training_cfg.get("gradient_checkpointing", False)),
        "training.gradient_checkpointing must be true",
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
        in QWEN38_ALLOWED_EVALUATION_MODES,
        "evaluation.sample_prediction_mode must be likelihood or original_teacher_forced",
    )
    _require(
        str(evaluation_cfg.get("headline_mode", "")) in QWEN38_ALLOWED_EVALUATION_MODES,
        "evaluation.headline_mode must be likelihood or original_teacher_forced",
    )
    _require(
        evaluation_cfg.get("evaluation_view") == QWEN38_EVALUATION_VIEW,
        f"evaluation.evaluation_view must be {QWEN38_EVALUATION_VIEW}",
    )
    _require(
        str(evaluation_cfg.get("inference_dtype", "")).strip().lower()
        in QWEN38_ALLOWED_INFERENCE_DTYPES,
        "evaluation.inference_dtype must be bf16 for the Qwen3.8 backend",
    )
    batching = str(evaluation_cfg.get("candidate_batching", "sequential")).strip().lower()
    _require(
        batching in {"", "sequential"},
        "candidate paired batching is not allowed in Qwen3.8 configs",
    )
    lora_cfg = config.get("lora", {})
    _require(
        lora_cfg.get("target_modules") == QWEN38_LORA_TARGET_REGEX,
        "lora.target_modules must be the exact Qwen3.8 language-model decoder regex",
    )
    _require(
        not bool(lora_cfg.get("tune_audio_encoder", False)),
        "lora.tune_audio_encoder must be false (audio encoder tuning disabled)",
    )
    audio_adapter_cfg = config.get("audio_adapter") or {}
    _require(
        not bool(audio_adapter_cfg.get("enabled", False)),
        "audio_adapter.enabled must be false",
    )
    _require(
        not bool(audio_adapter_cfg.get("train_projector", False)),
        "audio_adapter.train_projector must be false",
    )
    revision = config.get("model_revision")
    _require(
        isinstance(revision, str) and revision == QWEN38_MODEL_REVISION,
        f"model_revision must be the pinned revision {QWEN38_MODEL_REVISION}",
    )
    if errors:
        raise ValueError("Invalid qwen38 config:\n- " + "\n- ".join(errors))


def _audit_qwen38_lora_modules(model, matched_modules: set[str]) -> dict[str, Any]:
    """Fail unless LoRA covers only the language-model decoder layers."""
    violations: list[str] = []
    if not matched_modules:
        violations.append(
            "no LoRA modules were adapted; check lora.target_modules against the "
            "model's module names"
        )
    forbidden = sorted(
        name
        for name in matched_modules
        if any(marker in name for marker in QWEN38_FORBIDDEN_LORA_MARKERS)
    )
    if forbidden:
        violations.append(f"non-decoder modules are adapted: {forbidden[:8]}")
    outside_decoder = sorted(
        name for name in matched_modules if ".language_model.layers." not in name
    )
    if outside_decoder:
        violations.append(
            f"modules outside the language-model decoder are adapted: {outside_decoder[:8]}"
        )

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
        raise ValueError("Qwen3.8 LoRA audit failed:\n- " + "\n- ".join(violations))
    return {
        "matched_modules": len(matched_modules),
        "lora_trainable_params": lora_trainable,
        "audit_passed": True,
    }


def _unwrap_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def load_model_for_training(model_name_or_path: str, config: dict[str, Any]):
    validate_qwen38_config(config)
    model_class = resolve_qwen38_model_class(model_name_or_path)
    use_bf16 = bool(config["training"].get("bf16", False)) and torch.cuda.is_available()
    model = model_class.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    _set_use_cache(model, enabled=False)
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
    from peft.tuners.tuners_utils import inspect_matched_modules  # noqa: PLC0415

    matched = {str(name) for name in inspect_matched_modules(model.base_model)["matched"]}
    audit = _audit_qwen38_lora_modules(model, matched)
    model.print_trainable_parameters()
    LOGGER.info(
        "Qwen3.8 LoRA audit | matched_modules=%s lora_trainable_params=%s",
        audit["matched_modules"],
        audit["lora_trainable_params"],
    )
    LOGGER.info(
        "Qwen3.8 LoRA layer selection | requested_last_n_layers=%s decoder_hidden_layers=%s",
        lora_layer_selection["requested_last_n_layers"],
        lora_layer_selection["decoder_hidden_layer_count"],
    )
    return model


def load_model_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
):
    validate_qwen38_config(config or {})
    model_class = resolve_qwen38_model_class(model_name_or_path)
    model = model_class.from_pretrained(
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
    prepare_model_for_evaluation(model)
    return model


def _set_use_cache(model, enabled: bool) -> None:
    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "config"):
        base_model.config.use_cache = bool(enabled)
    if hasattr(model, "config"):
        model.config.use_cache = bool(enabled)


def prepare_model_for_evaluation(model) -> None:
    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    _set_use_cache(model, enabled=True)
    model.eval()


def restore_model_for_training(model, config: dict[str, Any]) -> None:
    """Re-apply the training-time memory config after an evaluation pass."""
    base_model = _unwrap_base_model(model)
    _set_use_cache(model, enabled=False)
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


def save_adapter_and_processor(
    model, processor, output_dir: str | Path, config: dict[str, Any] | None = None
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    LOGGER.info("Saved Qwen3.8 adapter and tokenizer to %s", output_dir)
