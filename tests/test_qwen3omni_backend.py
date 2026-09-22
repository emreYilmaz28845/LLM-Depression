from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import runtime as data_runtime
from src.data.prompt_context import (
    DATASET_CONTEXT_BLOCKS,
    PROMPT_CONTEXT_VERSION,
    SHARED_INSTRUCTION,
    prompt_context_record,
    resolve_system_prompt,
)
from src.experiment_tracking.submit import SubmissionError, resolve_evaluation_shape
from src.model import qwen3omni_lora, runtime
from src.model.collator import Qwen2AudioSFTCollator
from src.model.qwen3omni_lora import (
    QWEN3OMNI_EVALUATION_VIEW,
    QWEN3OMNI_FULL_MODEL_CLASS_NAME,
    QWEN3OMNI_LORA_TARGET_REGEX,
    QWEN3OMNI_MODEL_CLASS_NAME,
    _audit_qwen3omni_lora_modules,
    audit_talker_absence,
    enforce_audio_encoder_freeze,
    expected_lora_module_names,
    fsdp_transformer_cls_names,
    install_audio_feature_dtype_boundary,
    load_model_for_inference,
    normalize_audio_feature_dtype,
    resolve_qwen3omni_model_class,
    snapshot_identity,
    validate_qwen3omni_config,
)
from src.training_strategy import effective_global_batch_size
from src.utils import (
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_QWEN3OMNI,
    MODEL_BACKEND_TEXT,
    load_yaml,
    resolve_evaluation_device_map,
    resolve_evaluation_resource_shape,
    resolve_model_backend,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
OMNI_AUDIO_ONLY = MAIN / "daic_audio_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen3omni_30b_a3b.yaml"
OMNI_AUDIO_TEXT = MAIN / "daic_audio_text_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen3omni_30b_a3b.yaml"
QWEN_AUDIO_ONLY = MAIN / "daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
QWEN_AUDIO_TEXT = MAIN / "daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"

# Every difference between an Omni config and its canonical Qwen2-Audio source,
# grouped by the plan's allowed list: model backend/snapshot, attention
# implementation (the offline environment has no flash-attn), the prompt-context
# recipe replacing the inline system prompt, isolated run root and recipe id,
# anchored LoRA regex, FSDP/resource settings, explicit BF16 inference dtype, the
# explicit evaluation view and the declared sharded evaluation shape.
ALLOWED_CONFIG_DIFFERENCES = {
    "model_backend",
    "model_name_or_path",
    "model_attn_implementation",
    "recipe_id",
    "output_dirs.run_root",
    "prompt.system",
    "prompt.version",
    "prompt.dataset_context",
    "lora.target_modules",
    "training.strategy",
    "training.activation_offload",
    "training.run_final_eval_in_train",
    "evaluation.evaluation_view",
    "evaluation.inference_dtype",
    "resources",
    "resources.eval_nodes",
    "resources.eval_gpus_per_node",
}


def _flatten(node, prefix: str = "") -> dict:
    if isinstance(node, dict):
        flat: dict = {}
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: node}


def _config_diff(base: dict, candidate: dict) -> dict:
    base_flat = _flatten(base)
    candidate_flat = _flatten(candidate)
    keys = set(base_flat) | set(candidate_flat)
    return {
        key: (base_flat.get(key), candidate_flat.get(key))
        for key in sorted(keys)
        if base_flat.get(key) != candidate_flat.get(key)
    }


# --------------------------------------------------------------------------- #
# Fake module trees (ordinary nn.Module trees: the real classes only exist in
# the offline MN5 transformers build).
# --------------------------------------------------------------------------- #


def _frozen_linear() -> torch.nn.Linear:
    linear = torch.nn.Linear(2, 2, bias=False)
    linear.requires_grad_(False)
    return linear


class _FakeLoraLinear(torch.nn.Module):
    """Linear stand-in that carries PEFT-shaped adapter parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _FakeLoraLinear()
        self.k_proj = _FakeLoraLinear()
        self.v_proj = _FakeLoraLinear()
        self.o_proj = _FakeLoraLinear()
        self.q_norm = torch.nn.LayerNorm(2)
        self.q_norm.requires_grad_(False)


class _FakeMoEMlp(torch.nn.Module):
    """MoE MLP shaped like the real one: a router plus fused expert parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = _frozen_linear()
        self.experts = torch.nn.Module()
        self.experts.gate_up_proj = torch.nn.Parameter(
            torch.zeros(2, 4, 2), requires_grad=False
        )
        self.experts.down_proj = torch.nn.Parameter(torch.zeros(2, 2, 2), requires_grad=False)


class _FakeDenseMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _FakeLoraLinear()
        self.up_proj = _FakeLoraLinear()
        self.down_proj = _FakeLoraLinear()


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, *, dense_mlp: bool) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = _FakeDenseMlp() if dense_mlp else _FakeMoEMlp()


class _FakePeftWrapper(torch.nn.Module):
    """PEFT-shaped wrapper: ``base_model.model`` is a different module."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = base


class _FakeThinker(torch.nn.Module):
    """Thinker-shaped tree: ``audio_tower`` plus a text model at ``model``."""

    def __init__(self, *, layers: int = 2, dense_mlp: bool = False) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2, bias=False)
        self.audio_tower.requires_grad_(False)
        self.visual = torch.nn.Linear(2, 2, bias=False)
        self.visual.requires_grad_(False)
        text_model = torch.nn.Module()
        text_model.layers = torch.nn.ModuleList(
            [_FakeDecoderLayer(dense_mlp=dense_mlp) for _ in range(layers)]
        )
        self.model = text_model


def _matched_lora_names(model) -> set[str]:
    names = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_A" in name:
            names.add(name.rsplit(".lora_A", 1)[0])
    return names


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids) -> str:
        return "Depressed" if len(list(token_ids)) else ""


class _FakeOmniProcessor:
    """Collation surface of ``Qwen3OmniMoeProcessor`` with synthetic audio."""

    def __init__(self) -> None:
        self.feature_extractor = _FakeFeatureExtractor()
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, audio=None, sampling_rate=None, return_tensors=None, padding=False):
        if isinstance(text, list):
            text = text[0]
        token_count = len(str(text))
        payload = {
            "input_ids": list(range(token_count)),
            "attention_mask": [1] * token_count,
        }
        if audio is not None:
            payload["input_features"] = np.zeros((1, 2, 8), dtype=np.float64)
            payload["feature_attention_mask"] = np.ones((1, 8), dtype=np.int64)
        return payload


def _omni_config() -> dict:
    return load_yaml(OMNI_AUDIO_ONLY)


# --------------------------------------------------------------------------- #
# Dispatch and config contract
# --------------------------------------------------------------------------- #


def test_backend_constant_and_dispatch() -> None:
    assert MODEL_BACKEND_QWEN3OMNI == "qwen3omni"
    assert resolve_model_backend({"model_backend": "qwen3omni"}) == MODEL_BACKEND_QWEN3OMNI
    assert runtime._backend(_omni_config()) is qwen3omni_lora
    assert (
        runtime._backend({"model_backend": None, "data": {"use_audio": True, "use_text": False}})
        is not qwen3omni_lora
    )


@pytest.mark.parametrize("config_path", [OMNI_AUDIO_ONLY, OMNI_AUDIO_TEXT])
def test_config_carries_the_mandatory_fields(config_path: Path) -> None:
    config = load_yaml(config_path)
    assert config["model_backend"] == MODEL_BACKEND_QWEN3OMNI
    assert config["model_attn_implementation"] == "sdpa"
    assert config["training"]["strategy"] == "fsdp"
    assert config["training"]["run_final_eval_in_train"] is False
    assert config["training"]["activation_offload"] in {"none", "cpu"}
    assert config["lora"]["target_modules"] == QWEN3OMNI_LORA_TARGET_REGEX
    assert config["evaluation"]["evaluation_view"] == QWEN3OMNI_EVALUATION_VIEW
    assert config["evaluation"]["inference_dtype"] == "bf16"
    assert "promptcontext_v1_qwen3omni_likelihood" in config["output_dirs"]["run_root"]
    resources = config.get("resources")
    if resources is not None:
        assert set(resources) <= {"eval_nodes", "eval_gpus_per_node"}
        assert int(resources["eval_nodes"]) == 1
        assert int(resources["eval_gpus_per_node"]) in {1, 4}
    validate_qwen3omni_config(config)


@pytest.mark.parametrize(
    "candidate_path,base_path",
    [(OMNI_AUDIO_ONLY, QWEN_AUDIO_ONLY), (OMNI_AUDIO_TEXT, QWEN_AUDIO_TEXT)],
)
def test_config_diff_is_limited_to_the_documented_differences(
    candidate_path: Path, base_path: Path
) -> None:
    diff = _config_diff(load_yaml(base_path), load_yaml(candidate_path))
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFERENCES)
    assert not unexpected, {key: diff[key] for key in unexpected}
    assert diff["model_backend"] == (None, "qwen3omni")
    assert diff["model_attn_implementation"] == (None, "sdpa")
    assert diff["training.strategy"] == (None, "fsdp")
    assert diff["training.run_final_eval_in_train"] == (True, False)
    assert diff["evaluation.evaluation_view"] == (None, QWEN3OMNI_EVALUATION_VIEW)
    assert diff["evaluation.inference_dtype"] == (None, "bf16")
    assert diff["lora.target_modules"][1] == QWEN3OMNI_LORA_TARGET_REGEX
    assert diff["prompt.version"] == (None, "promptcontext_v1")
    assert diff["prompt.dataset_context"] == (None, "daic")
    assert diff["prompt.system"][1] is None
    # The prompt template itself is inherited unchanged.
    assert "prompt.user_template" not in diff
    assert "prompt.prompt_language" not in diff


def test_the_two_modalities_share_manifest_and_split_identity() -> None:
    audio_only = load_yaml(OMNI_AUDIO_ONLY)
    audio_text = load_yaml(OMNI_AUDIO_TEXT)
    for key in (
        "dataset",
        "seed",
        "recipe_id",
        "protocol_id",
        "manifest_variant",
        "split",
        "model_name_or_path",
        "model_backend",
        "dataset_root",
        "label_root",
        "quarantine_path",
    ):
        assert audio_only[key] == audio_text[key], key
    assert audio_only["output_dirs"]["manifest_dir"] == audio_text["output_dirs"]["manifest_dir"]
    assert audio_only["output_dirs"]["split_dir"] == audio_text["output_dirs"]["split_dir"]
    diff = _config_diff(audio_only, audio_text)
    assert sorted(diff) == [
        "data.audio_text_transcript_scope",
        "data.use_text",
        "output_dirs.run_root",
    ]
    # The prompt context is identical for both modalities (DAIC default block);
    # modality honesty is a rendering property, not a config difference.
    assert audio_only["prompt"] == audio_text["prompt"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["evaluation"].pop("evaluation_view"),
        lambda config: config["evaluation"].pop("inference_dtype"),
        lambda config: config["evaluation"].update({"inference_dtype": "fp32"}),
        lambda config: config["evaluation"].update({"headline_mode": "generation"}),
        lambda config: config["lora"].update({"target_modules": ["q_proj", "k_proj"]}),
        lambda config: config["lora"].update({"tune_audio_encoder": True}),
        lambda config: config["audio_adapter"].update({"enabled": True}),
        lambda config: config["training"].update({"strategy": "ddp"}),
        lambda config: config["training"].update({"run_final_eval_in_train": True}),
        lambda config: config["training"].update({"selection_metric": "inner_val_positive_f1"}),
        lambda config: config["data"].update({"use_audio": False}),
        lambda config: config.update({"model_attn_implementation": "flash_attention_2"}),
        lambda config: config.update({"model_name_or_path": ""}),
    ],
)
def test_validation_refuses_config_drift(mutate) -> None:
    config = _omni_config()
    mutate(config)
    with pytest.raises(ValueError):
        validate_qwen3omni_config(config)


def test_validation_is_noop_for_other_backends() -> None:
    validate_qwen3omni_config({"model_backend": MODEL_BACKEND_QWEN2AUDIO})
    validate_qwen3omni_config({"model_backend": MODEL_BACKEND_TEXT})
    validate_qwen3omni_config({})


def test_effective_batch_matches_the_canonical_recipe_for_both_shapes() -> None:
    config = _omni_config()
    training = config["training"]
    assert training["per_device_train_batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 32
    assert effective_global_batch_size(config, 4) == 128
    eight_rank = dict(config)
    eight_rank["training"] = dict(training, gradient_accumulation_steps=16)
    assert effective_global_batch_size(eight_rank, 8) == 128


# --------------------------------------------------------------------------- #
# Audio placeholder, transcript scope and prompt/label boundary
# --------------------------------------------------------------------------- #


def test_audio_placeholder_and_audio_only_transcript_exclusion() -> None:
    transcript = "I have been feeling low for a few weeks and sleeping is hard."
    audio_only = load_yaml(OMNI_AUDIO_ONLY)
    audio_text = load_yaml(OMNI_AUDIO_TEXT)
    placeholder = data_runtime.resolve_audio_placeholder(audio_only)
    assert placeholder == "<|audio_start|><|audio_pad|><|audio_end|>"
    assert data_runtime.resolve_audio_placeholder(audio_text) == placeholder

    audio_only_user = data_runtime.render_user_prompt_text(audio_only, transcript=transcript)
    audio_text_user = data_runtime.render_user_prompt_text(audio_text, transcript=transcript)
    assert transcript not in audio_only_user
    assert transcript in audio_text_user

    prompt_text = data_runtime.build_prompt_text(
        resolve_system_prompt(audio_only), audio_only_user, 1, True, audio_placeholder=placeholder
    )
    assert placeholder in prompt_text
    training_text = data_runtime.build_training_text(prompt_text, "Depressed")
    assert training_text.startswith(prompt_text)
    assert training_text[len(prompt_text) :] == "Depressed<|im_end|>\n"


def test_collator_emits_native_cpu_dtypes_and_masks_the_prompt() -> None:
    config = load_yaml(OMNI_AUDIO_TEXT)
    processor = _FakeOmniProcessor()
    collator = Qwen2AudioSFTCollator(processor=processor)
    example = {
        "sample_id": "daic-300-0",
        "subject_id": "300",
        "label": 1,
        "audio_arrays": [np.zeros(1600, dtype=np.float32)],
        "prompt_text": "SYS prompt text",
        "training_text": "SYS prompt textDepressed",
        "loss_weight": 0.5,
    }
    batch = collator([example])
    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "labels",
        "loss_weight",
        "input_features",
        "feature_attention_mask",
    }
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    # CPU preprocessing keeps its native float32 features; the model boundary
    # casts them, not the collator.
    assert batch["input_features"].dtype == torch.float32
    assert batch["feature_attention_mask"].dtype == torch.long
    assert batch["input_features"].shape == (1, 2, 8)
    assert batch["feature_attention_mask"].shape == (1, 8)
    assert batch["loss_weight"].dtype == torch.float32
    prompt_len = len(example["prompt_text"])
    assert int((batch["labels"][0, :prompt_len] == -100).sum()) == prompt_len
    assert int((batch["labels"][0, prompt_len:] == -100).sum()) == 0
    assert batch["input_ids"][0, prompt_len:].tolist() == batch["labels"][0, prompt_len:].tolist()


# --------------------------------------------------------------------------- #
# Audio dtype boundary
# --------------------------------------------------------------------------- #


class _FakeAudioFrontEnd(torch.nn.Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.audio_tower = torch.nn.Linear(2, 2, bias=False).to(dtype)
        self.received: dict | None = None

    def forward(self, **kwargs):
        self.received = dict(kwargs)
        return self.received


def test_audio_dtype_boundary_casts_only_floating_features() -> None:
    model = _FakeAudioFrontEnd(torch.bfloat16)
    install_audio_feature_dtype_boundary(model)
    features = torch.zeros(1, 2, 8, dtype=torch.float32)
    mask = torch.ones(1, 8, dtype=torch.long)
    model(input_ids=None, input_features=features, feature_attention_mask=mask, labels=None)
    assert model.received is not None
    assert model.received["input_features"].dtype == torch.bfloat16
    assert model.received["feature_attention_mask"].dtype == torch.long
    # The caller's tensor is not mutated in place.
    assert features.dtype == torch.float32


def test_audio_dtype_boundary_is_a_noop_without_audio_features() -> None:
    model = _FakeAudioFrontEnd(torch.bfloat16)
    install_audio_feature_dtype_boundary(model)
    input_ids = torch.ones(1, 4, dtype=torch.long)
    model(input_ids=input_ids)
    assert model.received is not None
    assert model.received["input_ids"].dtype == torch.long


def test_normalize_audio_feature_dtype_preserves_other_entries() -> None:
    tensors = {
        "input_features": torch.zeros(1, 2, 8, dtype=torch.float32),
        "feature_attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    normalized = normalize_audio_feature_dtype(tensors, torch.bfloat16)
    assert normalized["input_features"].dtype == torch.bfloat16
    assert normalized["feature_attention_mask"].dtype == torch.long
    assert tensors["input_features"].dtype == torch.float32
    # Already-correct features come back as the same object (no needless copy).
    assert normalize_audio_feature_dtype(normalized, torch.bfloat16) is normalized
    # A model without floating features and a None target are both no-ops.
    assert normalize_audio_feature_dtype(tensors, None) is tensors


# --------------------------------------------------------------------------- #
# LoRA targets, freeze guard, FSDP hook and Talker audit
# --------------------------------------------------------------------------- #


def test_lora_regex_matches_only_attention_and_dense_mlp() -> None:
    import re

    pattern = re.compile(QWEN3OMNI_LORA_TARGET_REGEX)
    matched = [
        f"model.layers.{layer}.{module}"
        for layer in (0, 47)
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # A dense layer would expose these; the checkpoint has none, but the
            # anchored pattern must cover them rather than accidentally skip them.
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    assert len(matched) == 14
    for name in matched:
        assert pattern.fullmatch(name), name
    for name in (
        # Router and fused expert parameters of the real MoE layers.
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.127.down_proj",
        # Norms, embeddings, heads and the other towers.
        "model.layers.0.self_attn.q_norm",
        "model.layers.0.input_layernorm",
        "model.embed_tokens",
        "model.norm",
        "lm_head",
        "audio_tower.layers.0.self_attn.q_proj",
        "visual.blocks.0.attn.qkv",
        "talker.model.layers.0.self_attn.q_proj",
    ):
        assert not pattern.fullmatch(name), name


def test_expected_lora_module_names_follow_the_module_tree() -> None:
    # The real checkpoint is MoE in every layer: attention only.
    moe = _FakeThinker(layers=3, dense_mlp=False)
    expected = sorted(expected_lora_module_names(moe))
    assert len(expected) == 3 * 4
    assert expected[0] == "model.layers.0.self_attn.k_proj"
    assert not any("mlp." in name for name in expected)
    # A dense layer also exposes its gate/up/down projections.
    dense = _FakeThinker(layers=3, dense_mlp=True)
    assert len(expected_lora_module_names(dense)) == 3 * 7
    # The PEFT-wrapped path form resolves to the same frame of reference, while an
    # unwrapped Thinker is not mistaken for a wrapper (transformers' base_model
    # property makes .base_model.model resolve to the text model itself).
    wrapped = _FakePeftWrapper(moe)
    assert sorted(expected_lora_module_names(wrapped)) == expected
    assert sorted(expected_lora_module_names(moe)) == expected


def test_resolved_lora_targets_are_exact_names_peft_can_consume() -> None:
    """PEFT 0.19.1 mangles a regex string on transformers-v5 MoE models.

    The backend must hand PEFT the exact names the anchored pattern matches, so
    the config keeps the audited regex while the injected targets are a set of
    module names.
    """
    import re

    from src.model.lora_common import build_lora_config

    model = _FakeThinker(layers=3, dense_mlp=False)
    pattern = re.compile(QWEN3OMNI_LORA_TARGET_REGEX)
    resolved = sorted(expected_lora_module_names(model))
    assert resolved
    assert all(pattern.fullmatch(name) for name in resolved)

    config = _omni_config()
    # build_lora_config resolves the decoder depth from the model config, so the
    # stand-in carries one (the real Thinker exposes it at config.text_config).
    config_holder = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=len(model.model.layers)))
    )
    lora_config, _ = build_lora_config(
        config, config_holder, resolved_target_modules=resolved
    )
    assert isinstance(lora_config.target_modules, set)
    assert lora_config.target_modules == set(resolved)
    # An exactly resolved scope behaves like the regex scope: no default excludes.
    assert getattr(lora_config, "exclude_modules", None) is None

    # Without the override the declared regex is passed through unchanged.
    regex_config, _ = build_lora_config(config, config_holder)
    assert regex_config.target_modules == QWEN3OMNI_LORA_TARGET_REGEX


def test_lora_audit_accepts_the_expected_set_and_rejects_drift() -> None:
    model = _FakeThinker(layers=2, dense_mlp=False)
    matched = _matched_lora_names(model)
    assert len(matched) == 8
    audit = _audit_qwen3omni_lora_modules(model, matched)
    assert audit["audit_passed"] is True
    assert audit["matched_modules"] == 8
    assert audit["expected_modules"] == 8
    assert audit["lora_trainable_params"] == 8 * (1 * 2 + 2 * 1)

    with pytest.raises(ValueError, match="routers, experts or non-decoder modules"):
        _audit_qwen3omni_lora_modules(model, matched | {"model.layers.0.mlp.gate"})
    with pytest.raises(ValueError, match="expected decoder modules were not adapted"):
        _audit_qwen3omni_lora_modules(
            model, matched - {"model.layers.0.self_attn.q_proj"}
        )
    with pytest.raises(ValueError, match="unexpected modules were adapted"):
        _audit_qwen3omni_lora_modules(model, matched | {"audio_tower.layers.0.fc1"})
    with pytest.raises(ValueError, match="no LoRA modules were adapted"):
        _audit_qwen3omni_lora_modules(model, set())

    drifted = _FakeThinker(layers=2)
    drifted.audio_tower.weight.requires_grad = True
    with pytest.raises(ValueError, match="non-LoRA parameters are trainable"):
        _audit_qwen3omni_lora_modules(drifted, _matched_lora_names(drifted))


def test_fsdp_wrap_policy_names_reads_the_decoder_layer_classes() -> None:
    model = _FakeThinker(layers=2)
    assert fsdp_transformer_cls_names(model) == ["_FakeDecoderLayer"]

    # A tree with no decoder layers is a lookup failure, not a silent empty wrap.
    empty = _FakeThinker(layers=0)
    with pytest.raises(ValueError, match="Could not locate the Qwen3-Omni Thinker decoder layers"):
        fsdp_transformer_cls_names(empty)

    with pytest.raises(ValueError, match="Could not locate the Qwen3-Omni Thinker decoder layers"):
        fsdp_transformer_cls_names(torch.nn.Linear(2, 2))


def test_audio_encoder_freeze_guard_freezes_leaked_lora_parameters() -> None:
    model = _FakeThinker(layers=1)
    leaked = torch.nn.Linear(2, 1, bias=False)
    model.audio_tower.add_module("lora_A_default", leaked)
    summary = enforce_audio_encoder_freeze(model, {"lora": {}})
    assert summary["frozen_lora_params"] > 0
    assert leaked.weight.requires_grad is False


def test_fsdp_wrap_hook_is_dispatched_through_the_runtime() -> None:
    model = _FakeThinker(layers=2)
    assert runtime.fsdp_wrap_policy_names(_omni_config(), model) == ["_FakeDecoderLayer"]
    assert runtime.fsdp_wrap_policy_names({"model_backend": MODEL_BACKEND_QWEN2AUDIO}, model) is None


def test_talker_absence_audit() -> None:
    clean = _FakeThinker(layers=1)
    audit = audit_talker_absence(clean, loaded_class=QWEN3OMNI_MODEL_CLASS_NAME)
    assert audit["talker_parameters"] == 0
    assert audit["loaded_class"] == QWEN3OMNI_MODEL_CLASS_NAME

    with_talker = _FakeThinker(layers=1)
    with_talker.talker = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="talker tensors are present"):
        audit_talker_absence(with_talker)


# --------------------------------------------------------------------------- #
# Snapshot identity and load guard rails
# --------------------------------------------------------------------------- #


def test_snapshot_identity_and_model_class_resolution(tmp_path: Path) -> None:
    config_bytes = json.dumps(
        {"architectures": [QWEN3OMNI_FULL_MODEL_CLASS_NAME], "model_type": "qwen3_omni_moe"}
    ).encode("utf-8")
    (tmp_path / "config.json").write_bytes(config_bytes)
    identity = snapshot_identity(tmp_path)
    assert identity["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert identity["declared_architectures"] == [QWEN3OMNI_FULL_MODEL_CLASS_NAME]
    assert resolve_qwen3omni_model_class(tmp_path) == QWEN3OMNI_MODEL_CLASS_NAME

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["SomethingElseForConditionalGeneration"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Expected architectures"):
        resolve_qwen3omni_model_class(tmp_path)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Expected architectures"):
        resolve_qwen3omni_model_class(empty)


def test_inference_loader_fails_closed_before_loading_weights() -> None:
    config = _omni_config()
    config["evaluation"].pop("inference_dtype")
    with pytest.raises(ValueError, match="inference_dtype"):
        load_model_for_inference("does-not-exist", config=config)

    drifted = _omni_config()
    drifted["training"]["strategy"] = "ddp"
    with pytest.raises(ValueError, match="training.strategy must be fsdp"):
        load_model_for_inference("does-not-exist", config=drifted)


def test_thinker_base_loader_prefers_the_direct_class_and_drops_the_talker(
    monkeypatch, tmp_path: Path
) -> None:
    import transformers

    calls: dict[str, int] = {"direct": 0, "full": 0}

    class FakeThinker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.audio_tower = torch.nn.Linear(2, 2, bias=False)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["direct"] += 1
            return cls()

    class FakeFull(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.thinker = FakeThinker()
            self.talker = torch.nn.Linear(2, 2, bias=False)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["full"] += 1
            return cls()

        def disable_talker(self) -> None:
            self.talker = None

    model_dir = tmp_path
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": [QWEN3OMNI_MODEL_CLASS_NAME]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        transformers, "Qwen3OmniMoeThinkerForConditionalGeneration", FakeThinker, raising=False
    )
    monkeypatch.setattr(
        transformers, "Qwen3OmniMoeForConditionalGeneration", FakeFull, raising=False
    )

    thinker, load_mode = qwen3omni_lora._load_thinker_base(
        str(model_dir), torch_dtype=None, attn_implementation="sdpa"
    )
    assert load_mode == "thinker_direct"
    assert calls == {"direct": 1, "full": 0}
    assert isinstance(thinker, FakeThinker)
    audit_talker_absence(thinker)

    class FailingThinker(FakeThinker):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise RuntimeError("no direct thinker weights")

    monkeypatch.setattr(
        transformers, "Qwen3OmniMoeThinkerForConditionalGeneration", FailingThinker, raising=False
    )
    thinker, load_mode = qwen3omni_lora._load_thinker_base(
        str(model_dir), torch_dtype=None, attn_implementation="sdpa"
    )
    assert load_mode == "talker_disabled"
    assert calls["full"] == 1
    assert isinstance(thinker, FakeThinker)
    audit_talker_absence(thinker)


# --------------------------------------------------------------------------- #
# prompt-context recipe, config generator and evaluation resource shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("config_path", [OMNI_AUDIO_ONLY, OMNI_AUDIO_TEXT])
def test_prompt_context_resolution_is_centralized_and_hashed(config_path: Path) -> None:
    config = load_yaml(config_path)
    record = prompt_context_record(config)
    assert record["version"] == PROMPT_CONTEXT_VERSION
    assert record["dataset_context"] == "daic"
    assert record["input_modality"] in {"audio_only", "audio_text"}
    system_prompt = resolve_system_prompt(config)
    assert system_prompt == record["system_prompt"]
    assert record["system_prompt_sha256"] == hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest()
    # Audio-bearing modalities keep the approved instruction and the DAIC default
    # recording context verbatim.
    assert SHARED_INSTRUCTION in system_prompt
    assert DATASET_CONTEXT_BLOCKS[PROMPT_CONTEXT_VERSION]["daic"]["default"] in system_prompt


def test_backend_module_carries_no_copy_of_the_shared_prompt_text() -> None:
    """The prompt text must live only in the central resolver."""
    source = (ROOT / "src/model/qwen3omni_lora.py").read_text(encoding="utf-8")
    assert SHARED_INSTRUCTION not in source
    assert "Recording context:" not in source
    for key, blocks in DATASET_CONTEXT_BLOCKS[PROMPT_CONTEXT_VERSION].items():
        for text in blocks.values():
            assert text[:40] not in source, key


def test_config_generator_is_idempotent_and_keeps_the_diff_allowlisted(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "config_diff_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_qwen3omni_daic_configs.py",
            "--check",
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["prompt_context_version"] == PROMPT_CONTEXT_VERSION
    assert len(audit["configs"]) == 2
    for entry in audit["configs"]:
        assert entry["allowed"] is True
        assert set(entry["changed_paths"]) <= ALLOWED_CONFIG_DIFFERENCES


@pytest.mark.parametrize(
    "candidate_path", [OMNI_AUDIO_ONLY, OMNI_AUDIO_TEXT]
)
def test_declared_evaluation_shape_matches_the_submission_contract(
    candidate_path: Path,
) -> None:
    """The loader's shape and the submit contract's shape must agree."""
    config = load_yaml(candidate_path)
    loader_shape = resolve_evaluation_resource_shape(config)
    contract_shape = resolve_evaluation_shape(config)
    assert loader_shape == {
        "nodes": contract_shape["nodes"],
        "gpus_per_node": contract_shape["gpus_per_node"],
        "sharded": contract_shape["sharded"],
    }
    assert loader_shape["gpus_per_node"] in {1, 4}
    device_map = resolve_evaluation_device_map(config)
    assert (device_map == "auto") is loader_shape["sharded"]

    # A config that does not declare the block keeps the one-GPU behavior.
    undeclared = dict(config)
    undeclared.pop("resources")
    assert resolve_evaluation_shape(undeclared)["gpus_per_node"] == 1
    assert resolve_evaluation_device_map(undeclared) is None
    # Both copies of the resolution refuse an out-of-range shape.
    with pytest.raises(ValueError):
        resolve_evaluation_resource_shape({**undeclared, "resources": {"eval_gpus_per_node": 0}})
    with pytest.raises(SubmissionError):
        resolve_evaluation_shape(undeclared, eval_gpus_per_node=0)
