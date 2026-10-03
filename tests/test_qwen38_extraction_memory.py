"""Unit tests for the memory-reduced Qwen3.8 hidden-state extraction path.

The Qwen3.8 extraction on a single 64 GiB GPU out-of-memory during long
transcripts because the wrapper computed full-vocabulary logits and retained
every layer's hidden state. ``_forward_final_hidden_states`` now calls the base
decoder directly for that backend. These tests pin the call contract without
loading any model.
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


class _FakeBase:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


class _FakeWrapper:
    def __init__(self, base, output):
        self.model = base
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_qwen38_uses_base_decoder_without_logits_or_layer_tuple():
    hidden = object()
    base = _FakeBase(_Output(last_hidden_state=hidden))
    wrapper = _FakeWrapper(base, _Output(hidden_states=(object(), object())))
    returned, mask = _forward_final_hidden_states(
        wrapper, {"input_ids": object()}, MODEL_BACKEND_QWEN38
    )
    assert returned is hidden
    assert mask is None
    assert wrapper.calls == []
    assert len(base.calls) == 1
    call = base.calls[0]
    assert call["output_hidden_states"] is False
    assert call["use_cache"] is False
    assert call["return_dict"] is True
    assert "labels" not in call


def test_qwen38_falls_back_to_first_output_element():
    hidden = object()
    base = _FakeBase(_PositionalOutput(hidden))
    wrapper = _FakeWrapper(base, _Output(hidden_states=(object(),)))
    returned, _ = _forward_final_hidden_states(wrapper, {}, MODEL_BACKEND_QWEN38)
    assert returned is hidden


def test_other_backends_keep_wrapper_hidden_states():
    last = object()
    wrapper = _FakeWrapper(
        _FakeBase(_Output()),
        _Output(hidden_states=(object(), last), attention_mask="mask"),
    )
    returned, mask = _forward_final_hidden_states(wrapper, {}, MODEL_BACKEND_QWEN3OMNI)
    assert returned is last
    assert mask == "mask"
    assert len(wrapper.calls) == 1
    assert wrapper.calls[0]["output_hidden_states"] is True
    assert wrapper.calls[0]["labels"] is None
