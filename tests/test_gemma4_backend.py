from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model import runtime
from src.model.gemma4_io import (
    GEMMA4_LORA_TARGET_REGEX,
    Gemma4SFTCollator,
)
from src.model.lora_common import build_lora_config
from src.utils import (
    MODEL_BACKEND_GEMMA4,
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_QWEN3OMNI,
    MODEL_BACKEND_TEXT,
    resolve_model_backend,
)

from test_gemma4_io import FakeProcessor, gemma_config, make_example


def test_backend_constant_registered() -> None:
    assert MODEL_BACKEND_GEMMA4 == "gemma4"


def test_resolve_model_backend_gemma4_explicit() -> None:
    assert resolve_model_backend({"model_backend": "gemma4"}) == MODEL_BACKEND_GEMMA4


def test_resolve_model_backend_absent_returns_none() -> None:
    assert resolve_model_backend({}) is None
    assert resolve_model_backend({"model_backend": ""}) is None


def test_resolve_model_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported model_backend"):
        resolve_model_backend({"model_backend": "gemma9"})


class FakeBackendModule:
    load_processor = staticmethod(lambda *a, **k: "fake_processor")
    load_model_for_training = staticmethod(lambda *a, **k: "fake_model")
    load_model_for_inference = staticmethod(lambda *a, **k: "fake_inference_model")
    prepare_model_for_evaluation = staticmethod(lambda model: None)
    restore_model_for_training = staticmethod(lambda model, config: None)
    save_adapter_and_processor = staticmethod(lambda *a, **k: None)


def test_lora_regex_string_is_preserved(monkeypatch) -> None:
    config = gemma_config()
    # build_lora_config must keep the regex string intact (not a char list).
    lora_config, _ = build_lora_config(config, SimpleNamespace(config=SimpleNamespace(num_hidden_layers=48)))
    assert lora_config.target_modules == GEMMA4_LORA_TARGET_REGEX
    assert isinstance(lora_config.target_modules, str)


def test_lora_list_targets_still_expand_to_list() -> None:
    config = {
        "model_backend": "qwen2audio",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
    }
    lora_config, _ = build_lora_config(config, SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4)))
    assert set(lora_config.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def _synthetic_gemma_module_tree(num_layers: int = 48, with_v_proj: bool = False) -> dict[str, SimpleNamespace]:
    """Build a fake module tree with the Gemma4 unified naming pattern."""
    modules: dict[str, SimpleNamespace] = {}
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}"
        attn: dict[str, SimpleNamespace] = {
            "q_proj": SimpleNamespace(),
            "k_proj": SimpleNamespace(),
            "o_proj": SimpleNamespace(),
        }
        if with_v_proj:
            attn["v_proj"] = SimpleNamespace()
        mlp = {
            "gate_proj": SimpleNamespace(),
            "up_proj": SimpleNamespace(),
            "down_proj": SimpleNamespace(),
        }
        for module_name, module in attn.items():
            modules[f"{prefix}.self_attn.{module_name}"] = module
        for module_name, module in mlp.items():
            modules[f"{prefix}.mlp.{module_name}"] = module
        modules[f"{prefix}.self_attn"] = SimpleNamespace(**attn)
        modules[f"{prefix}.mlp"] = SimpleNamespace(**mlp)
    return modules


def test_regex_matches_exactly_288_decoder_modules() -> None:
    modules = _synthetic_gemma_module_tree(48)
    pattern = re.compile(GEMMA4_LORA_TARGET_REGEX)
    matched = sorted(name for name in modules if pattern.fullmatch(name))
    assert len(matched) == 288
    assert all("vision" not in name for name in matched)
    assert all("embed_audio" not in name for name in matched)
    assert all("lm_head" not in name and "embed_tokens" not in name for name in matched)


def test_regex_rejects_v_proj_and_outside_decoder() -> None:
    modules = _synthetic_gemma_module_tree(48, with_v_proj=True)
    modules["model.embed_audio.embedding_projection"] = SimpleNamespace()
    modules["model.language_model.embed_tokens"] = SimpleNamespace()
    pattern = re.compile(GEMMA4_LORA_TARGET_REGEX)
    matched = sorted(name for name in modules if pattern.fullmatch(name))
    assert len(matched) == 288
    assert "model.language_model.layers.0.self_attn.v_proj" not in matched
    assert "model.embed_audio.embedding_projection" not in matched
    assert "model.language_model.embed_tokens" not in matched


def test_backend_dispatch_selects_gemma_only_when_explicit(monkeypatch) -> None:
    from src.model import gemma4_lora

    monkeypatch.setattr(runtime, "_backend", lambda config: gemma4_lora)
    assert runtime._backend({"model_backend": "gemma4"}) is gemma4_lora


def test_backend_defaults_unchanged_for_qwen() -> None:
    from src.model import qwen2audio_lora, qwen3omni_lora, text_lora

    text_config = {"model_backend": None, "data": {"use_audio": False, "use_text": True}}
    audio_config = {"model_backend": None, "data": {"use_audio": True, "use_text": True}}
    assert runtime._backend(text_config) is text_lora
    assert runtime._backend(audio_config) is qwen2audio_lora
    assert (
        runtime._backend({"model_backend": MODEL_BACKEND_QWEN3OMNI, "data": {"use_audio": False, "use_text": True}})
        is qwen3omni_lora
    )
    assert (
        runtime._backend({"model_backend": MODEL_BACKEND_TEXT, "data": {"use_audio": True, "use_text": True}})
        is text_lora
    )
    assert (
        runtime._backend({"model_backend": MODEL_BACKEND_QWEN2AUDIO, "data": {"use_audio": False, "use_text": True}})
        is qwen2audio_lora
    )


def test_collator_factory_dispatches_by_backend(monkeypatch) -> None:
    from src.model.collator import Qwen2AudioSFTCollator

    gemma_collator = runtime.build_collator(gemma_config(), FakeProcessor())
    assert isinstance(gemma_collator, Gemma4SFTCollator)
    qwen_collator = runtime.build_collator(
        {"model_backend": MODEL_BACKEND_QWEN2AUDIO, "data": {"use_audio": True, "use_text": True}},
        FakeProcessor(),
    )
    assert isinstance(qwen_collator, Qwen2AudioSFTCollator)


def test_prepare_backend_examples_noop_for_qwen() -> None:
    config = {"model_backend": MODEL_BACKEND_QWEN2AUDIO, "data": {"use_audio": True, "use_text": True}}
    source = [make_example()]
    result = runtime.prepare_backend_examples(source, config, FakeProcessor())
    assert result is source


def test_prepare_backend_examples_gemma_renders(monkeypatch) -> None:
    from src.model import gemma4_io

    calls = []

    def fake_prepare(examples, config, processor):
        calls.append((examples, config, processor))
        return [dict(example, prompt_text="rendered") for example in examples]

    monkeypatch.setattr(gemma4_io, "prepare_gemma4_examples", fake_prepare)
    monkeypatch.setattr(
        runtime,
        "prepare_backend_examples",
        lambda examples, config, processor: (
            fake_prepare(examples, config, processor)
            if resolve_model_backend(config) == MODEL_BACKEND_GEMMA4
            else examples
        ),
    )
    prepared = runtime.prepare_backend_examples([make_example()], gemma_config(), FakeProcessor())
    assert prepared[0]["prompt_text"] == "rendered"


def test_adapter_save_and_processor_save_calls() -> None:
    from src.model import gemma4_lora

    saved: list[tuple] = []

    class FakePEFTModel:
        def save_pretrained(self, output_dir, safe_serialization=False):
            saved.append((output_dir, safe_serialization))

    class FakeProcessorSave:
        def save_pretrained(self, output_dir):
            saved.append((output_dir, None))

    gemma4_lora.save_adapter_and_processor(FakePEFTModel(), FakeProcessorSave(), "/tmp/out", config=gemma_config())
    assert saved == [(Path("/tmp/out"), True), (Path("/tmp/out"), None)]
