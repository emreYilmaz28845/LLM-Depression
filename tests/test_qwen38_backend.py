from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.model import qwen38_lora, runtime
from src.model.lora_common import build_lora_config
from src.model.qwen38_lora import (
    QWEN38_EVALUATION_VIEW,
    QWEN38_LORA_TARGET_REGEX,
    QWEN38_MODEL_CLASS_NAME,
    QWEN38_MODEL_REVISION,
    _audit_qwen38_lora_modules,
    prepare_qwen38_examples,
    render_qwen38_prompt,
    render_qwen38_training_text,
    resolve_qwen38_model_class,
    validate_qwen38_config,
)
from src.utils import (
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_QWEN38,
    MODEL_BACKEND_TEXT,
    load_yaml,
    resolve_model_backend,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
QWEN38_CONFIG = MAIN / "daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml"
QWEN_CONFIG = MAIN / "daic_text_only_harmonized_selmacrof1_likelihood_v1.yaml"
CLOSED_THINKING_BLOCK = "<think>\n\n</think>\n\n"


class FakeQwen38Processor:
    """Mirror of the pinned Qwen3.8 template for the text-only path.

    The real template writes the closed thinking block after the assistant header
    for assistant messages as well as for the generation prompt (verified against
    the checkpoint's own ``chat_template.jinja``).
    """

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        tokenize: bool = False,
    ) -> str:
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            if message["role"] == "assistant" and enable_thinking is False:
                text += CLOSED_THINKING_BLOCK
            text += f"{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            text += CLOSED_THINKING_BLOCK if enable_thinking is False else "<think>\n"
        return text


def qwen38_config() -> dict:
    return load_yaml(QWEN38_CONFIG)


def make_example(**overrides) -> dict:
    example = {
        "sample_id": "daic-300-0",
        "subject_id": "300",
        "label": 1,
        "internal_label_text": "Depressed",
        "prompt_system_text": "You are a psychologist analyzing transcript information.",
        "prompt_user_text": "The transcript of the subject's speech is:\nexample\n\nBased on the transcript, determine whether the subject is Depressed or Non-depressed.\nAnswer with one label.",
        "prompt_text": "PRE-RENDERED",
        "training_text": "PRE-RENDERED",
    }
    example.update(overrides)
    return example


def test_backend_constant_registered() -> None:
    assert MODEL_BACKEND_QWEN38 == "qwen38"


def test_resolve_model_backend_qwen38_explicit() -> None:
    assert resolve_model_backend({"model_backend": "qwen38"}) == MODEL_BACKEND_QWEN38


def test_backend_dispatch_selects_qwen38_only_when_explicit() -> None:
    assert runtime._backend(qwen38_config()) is qwen38_lora
    assert (
        runtime._backend({"model_backend": None, "data": {"use_audio": False, "use_text": True}})
        is not qwen38_lora
    )
    assert (
        runtime._backend(
            {"model_backend": MODEL_BACKEND_QWEN2AUDIO, "data": {"use_audio": True, "use_text": True}}
        )
        is not qwen38_lora
    )


def test_backend_module_keeps_the_architecture_import_lazy() -> None:
    # The module must stay importable in the local Qwen2 environment: the
    # architecture class is only resolved inside the loaders.
    assert "Qwen3_5ForConditionalGeneration" not in vars(qwen38_lora)
    assert qwen38_lora.QWEN38_MODEL_CLASS_NAME == QWEN38_MODEL_CLASS_NAME
    assert callable(qwen38_lora.resolve_qwen38_model_class)


def test_config_carries_the_three_mandatory_fields() -> None:
    config = qwen38_config()
    assert config["model_backend"] == MODEL_BACKEND_QWEN38
    assert config["evaluation"]["evaluation_view"] == QWEN38_EVALUATION_VIEW
    assert config["evaluation"]["inference_dtype"] == "bf16"
    assert config["model_revision"] == QWEN38_MODEL_REVISION
    assert config["lora"]["target_modules"] == QWEN38_LORA_TARGET_REGEX
    assert "harmonized_v1_qwen38_likelihood" in config["output_dirs"]["run_root"]
    validate_qwen38_config(config)


def test_config_preserves_the_base_recipe_and_splits() -> None:
    config = qwen38_config()
    base = load_yaml(QWEN_CONFIG)
    for key in ("dataset", "seed", "recipe_id", "protocol_id", "manifest_variant", "labels", "prompt", "split"):
        assert config[key] == base[key], key
    # The training block deviates in exactly two documented ways: the FSDP
    # strategy, and the in-train held-out evaluation that a sharded run cannot do.
    training = dict(config["training"])
    assert training.pop("strategy") == "fsdp"
    assert training.pop("run_final_eval_in_train") is False
    base_training = dict(base["training"])
    assert base_training.pop("run_final_eval_in_train") is True
    assert training == base_training
    assert config["data"] == base["data"]
    assert config["output_dirs"]["manifest_dir"] == base["output_dirs"]["manifest_dir"]
    assert config["output_dirs"]["split_dir"] == base["output_dirs"]["split_dir"]
    for key in ("sample_prediction_mode", "headline_mode", "aggregation_level"):
        assert config["evaluation"][key] == base["evaluation"][key], key


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["evaluation"].pop("evaluation_view"),
        lambda config: config["evaluation"].pop("inference_dtype"),
        lambda config: config["evaluation"].update({"inference_dtype": "fp32"}),
        lambda config: config["lora"].update({"target_modules": ["q_proj", "k_proj"]}),
        lambda config: config.update({"model_revision": "deadbeef"}),
        lambda config: config["data"].update({"use_audio": True}),
        lambda config: config["training"].update({"selection_metric": "inner_val_positive_f1"}),
    ],
)
def test_validation_refuses_config_drift(mutate) -> None:
    config = qwen38_config()
    mutate(config)
    with pytest.raises(ValueError):
        validate_qwen38_config(config)


def test_validation_is_noop_for_other_backends() -> None:
    validate_qwen38_config({"model_backend": MODEL_BACKEND_QWEN2AUDIO})
    validate_qwen38_config({"model_backend": MODEL_BACKEND_TEXT})


def test_lora_regex_matches_the_first_experiment_target_set() -> None:
    # The decoder is a hybrid: every fourth layer (indices 3, 7, ... 63) carries
    # full attention and the other 48 carry linear attention (gated delta net);
    # every layer has an MLP. The 800 decoder submodule names behind this layout
    # were read from the pinned checkpoint's safetensors index. The first
    # experiment adapts the MLP of every layer plus the four full-attention
    # projections; the linear-attention block stays untouched.
    pattern = re.compile(QWEN38_LORA_TARGET_REGEX)
    self_attn_layers = tuple(range(3, 64, 4))
    linear_attn_layers = tuple(layer for layer in range(64) if layer not in self_attn_layers)
    assert len(self_attn_layers) == 16 and len(linear_attn_layers) == 48
    matched = []
    for layer in range(64):
        for module in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"model.language_model.layers.{layer}.{module}"
            assert pattern.fullmatch(name), name
            matched.append(name)
    for layer in self_attn_layers:
        for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            name = f"model.language_model.layers.{layer}.{module}"
            assert pattern.fullmatch(name), name
            matched.append(name)
    assert len(matched) == 16 * 7 + 48 * 3 == 256
    for name in (
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.language_model.embed_tokens",
        "lm_head",
        "model.language_model.layers.3.self_attn.q_norm",
        "model.language_model.layers.3.self_attn.k_norm",
        "model.language_model.layers.0.input_layernorm",
        "model.language_model.layers.0.post_attention_layernorm",
        # Linear-attention internals, including the out_proj reserved for the
        # separate 304-module comparison experiment.
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.4.linear_attn.out_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.conv1d",
    ):
        assert not pattern.fullmatch(name), name


def test_lora_config_keeps_the_regex_string() -> None:
    lora_config, _ = build_lora_config(
        qwen38_config(),
        SimpleNamespace(config=SimpleNamespace(num_hidden_layers=64)),
    )
    assert lora_config.target_modules == QWEN38_LORA_TARGET_REGEX
    assert isinstance(lora_config.target_modules, str)


def test_audit_rejects_vision_and_non_lora_trainables() -> None:
    model = SimpleNamespace(
        named_parameters=lambda: [
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
             torch.nn.Parameter(torch.zeros(4))),
        ]
    )
    audit = _audit_qwen38_lora_modules(
        model,
        {"model.language_model.layers.0.self_attn.q_proj"},
    )
    assert audit["audit_passed"] is True
    assert audit["matched_modules"] == 1

    with pytest.raises(ValueError, match="non-decoder modules are adapted"):
        _audit_qwen38_lora_modules(
            model,
            {
                "model.language_model.layers.0.self_attn.q_proj",
                "model.visual.blocks.0.attn.qkv",
            },
        )

    with pytest.raises(ValueError, match="no LoRA modules were adapted"):
        _audit_qwen38_lora_modules(model, set())

    drifting = SimpleNamespace(
        named_parameters=lambda: [
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
             torch.nn.Parameter(torch.zeros(4))),
            ("base_model.model.model.visual.blocks.0.attn.qkv.weight",
             torch.nn.Parameter(torch.zeros(4))),
        ]
    )
    with pytest.raises(ValueError, match="non-LoRA parameters are trainable"):
        _audit_qwen38_lora_modules(
            drifting, {"model.language_model.layers.0.self_attn.q_proj"}
        )


def test_render_prompt_closes_the_thinking_block() -> None:
    prompt = render_qwen38_prompt(FakeQwen38Processor(), "SYS", "USER")
    assert prompt.startswith("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n" + CLOSED_THINKING_BLOCK)


def test_training_text_is_prompt_plus_label_plus_template_terminator() -> None:
    processor = FakeQwen38Processor()
    prompt = render_qwen38_prompt(processor, "SYS", "USER")
    training_text = render_qwen38_training_text(processor, "SYS", "USER", "Depressed")
    # The training text must continue the generation prompt byte for byte: the
    # collator masks the prompt and the likelihood backend scores the same span.
    assert training_text.startswith(prompt)
    assert training_text[len(prompt) :] == "Depressed<|im_end|>\n"
    assert training_text == f"{prompt}Depressed<|im_end|>\n"


def test_prepare_examples_render_and_require_raw_fields() -> None:
    prepared = prepare_qwen38_examples([make_example()], qwen38_config(), FakeQwen38Processor())
    example = prepared[0]
    assert example["prompt_text"] != "PRE-RENDERED"
    assert example["training_text"].startswith(example["prompt_text"])
    assert example["training_text"].endswith("Depressed<|im_end|>\n")
    assert example["subject_id"] == "300" and example["label"] == 1
    assert "PRE-RENDERED" == make_example()["prompt_text"]

    with pytest.raises(ValueError, match="prompt_user_text"):
        prepare_qwen38_examples([make_example(prompt_user_text="")], qwen38_config(), FakeQwen38Processor())


def test_prepare_examples_is_noop_for_other_backends() -> None:
    source = [make_example()]
    config = {"model_backend": MODEL_BACKEND_TEXT, "data": {"use_audio": False, "use_text": True}}
    result = prepare_qwen38_examples(source, config, FakeQwen38Processor())
    assert result[0] is source[0]


def test_runtime_hook_dispatches_to_qwen38_rendering() -> None:
    prepared = runtime.prepare_backend_examples([make_example()], qwen38_config(), FakeQwen38Processor())
    assert prepared[0]["prompt_text"] != "PRE-RENDERED"


def test_resolve_model_class_reads_and_checks_architectures(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_qwen38_model_class(tmp_path)

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["SomethingElseForConditionalGeneration"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected architectures"):
        resolve_qwen38_model_class(tmp_path)

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": [QWEN38_MODEL_CLASS_NAME]}), encoding="utf-8"
    )
    sentinel = object()
    monkeypatch.setattr(qwen38_lora, "_import_qwen38_model_class", lambda name: sentinel)
    assert resolve_qwen38_model_class(tmp_path) is sentinel


def test_save_adapter_and_processor_calls() -> None:
    saved: list[tuple] = []

    class FakePEFTModel:
        def save_pretrained(self, output_dir, safe_serialization=False):
            saved.append((Path(output_dir), safe_serialization))

    class FakeProcessorSave:
        def save_pretrained(self, output_dir):
            saved.append((Path(output_dir), None))

    qwen38_lora.save_adapter_and_processor(
        FakePEFTModel(), FakeProcessorSave(), "/tmp/qwen38_out", config=qwen38_config()
    )
    assert saved == [(Path("/tmp/qwen38_out"), True), (Path("/tmp/qwen38_out"), None)]
