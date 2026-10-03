"""Unit tests for the memory-reduced Qwen3.8 hidden-state extraction path.

The Qwen3.8 extraction on a single 64 GiB GPU out-of-memory during long
transcripts because the wrapper computed full-vocabulary logits and retained
every layer's hidden state. ``_forward_final_hidden_states`` now calls the
adapter-active base decoder directly for that backend. The real loader returns
a ``PeftModel`` whose ``.model`` attribute resolves to the full
conditional-generation model, so these tests pin the module-tree lookup that
finds the decoder instead. No model is loaded.
"""

from __future__ import annotations

from src.features.extract_qwen_hidden import _forward_final_hidden_states
from src.utils import MODEL_BACKEND_QWEN38, MODEL_BACKEND_QWEN3OMNI


class _Output:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _PositionalOutput:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, index):
        assert index == 0
        return self._value


class Qwen3_5Model:
    """Fake decoder; the class name matches the extractor's lookup."""

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.language_model = object()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output

    def modules(self):
        return [self]


class _FullConditionalModel:
    """Fake wrapper that computes logits and exposes the decoder in modules()."""

    def __init__(self, decoder, output):
        self.model = decoder
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output

    def modules(self):
        return [self] + self.model.modules()


class _PeftWrapper:
    """Fake PeftModel: .model delegates to the full conditional model."""

    def __init__(self, full):
        self.model = full
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.model(**kwargs)

    def modules(self):
        return [self] + self.model.modules()


def test_peft_wrapper_resolves_decoder_and_skips_full_model():
    hidden = object()
    decoder = Qwen3_5Model(_Output(last_hidden_state=hidden))
    full = _FullConditionalModel(decoder, _Output(logits=object()))
    peft = _PeftWrapper(full)
    returned, mask = _forward_final_hidden_states(
        peft, {"input_ids": object()}, MODEL_BACKEND_QWEN38
    )
    assert returned is hidden
    assert mask is None
    assert full.calls == []
    assert peft.calls == []
    assert len(decoder.calls) == 1
    call = decoder.calls[0]
    assert call["output_hidden_states"] is False
    assert call["use_cache"] is False
    assert call["return_dict"] is True
    assert "labels" not in call


def test_bare_qwen38_wrapper_resolves_decoder():
    hidden = object()
    decoder = Qwen3_5Model(_Output(last_hidden_state=hidden))
    full = _FullConditionalModel(decoder, _Output(logits=object()))
    returned, _ = _forward_final_hidden_states(full, {}, MODEL_BACKEND_QWEN38)
    assert returned is hidden
    assert full.calls == []


def test_qwen38_falls_back_to_first_output_element():
    hidden = object()
    decoder = Qwen3_5Model(_PositionalOutput(hidden))
    peft = _PeftWrapper(_FullConditionalModel(decoder, _Output(logits=object())))
    returned, _ = _forward_final_hidden_states(peft, {}, MODEL_BACKEND_QWEN38)
    assert returned is hidden


def test_other_backends_keep_wrapper_hidden_states():
    last = object()
    wrapper = _FullConditionalModel(
        Qwen3_5Model(_Output()),
        _Output(hidden_states=(object(), last), attention_mask="mask"),
    )
    returned, mask = _forward_final_hidden_states(wrapper, {}, MODEL_BACKEND_QWEN3OMNI)
    assert returned is last
    assert mask == "mask"
    assert len(wrapper.calls) == 1
    assert wrapper.calls[0]["output_hidden_states"] is True
    assert wrapper.calls[0]["labels"] is None
