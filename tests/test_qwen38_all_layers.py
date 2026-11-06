"""All-layer tap of the Qwen3.8 hidden extractor, checked on a tiny stand-in decoder."""

import json

import numpy as np
import pytest
import torch

from baselines.layer_select_heads import layer_dirs, run
from src.features.extract_qwen_hidden import (
    LayerTap,
    _armed_index,
    _write_layer_partition,
    layer_dir,
    write_layer_metadata,
)


class _Block(torch.nn.Module):
    def __init__(self, size, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        return x + torch.tanh(self.linear(x))


class _TinyDecoder(torch.nn.Module):
    def __init__(self, vocab=11, size=6, layers=4):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(vocab, size)
        self.layers = torch.nn.ModuleList(_Block(size, seed) for seed in range(layers))
        self.norm = torch.nn.LayerNorm(size)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        states = [hidden]
        for block in self.layers:
            hidden = block(hidden)
            states.append(hidden)
        return self.norm(hidden), states


def test_tap_matches_reference_states_at_last_valid_token():
    model = _TinyDecoder().eval()
    tap = LayerTap(model.embed_tokens, list(model.layers))
    ids = torch.tensor([[1, 4, 2, 7, 3]])
    index = _armed_index({"attention_mask": torch.ones_like(ids)})
    assert index == 4
    tap.arm(index)
    with torch.no_grad():
        final, states = model(ids)
    final_vector = final[0, index].numpy()
    stack = tap.collect(final_vector)
    assert stack.shape == (5, 6)
    for layer in range(4):  # embedding + blocks 1..3; block 4 is replaced by the final state
        np.testing.assert_array_equal(stack[layer], states[layer][0, index].numpy())
    np.testing.assert_array_equal(stack[4], final_vector)
    tap.close()
    with torch.no_grad():
        model(ids)  # hooks removed: nothing captured, no error


def test_tap_is_inert_until_armed_and_refuses_missing_rows():
    model = _TinyDecoder().eval()
    tap = LayerTap(model.embed_tokens, list(model.layers))
    with torch.no_grad():
        model(torch.tensor([[1, 2]]))
    tap.arm(1)
    with pytest.raises(RuntimeError, match="did not run"):
        tap.collect(np.zeros(6, np.float32))


def test_armed_index_rejects_batches():
    with pytest.raises(ValueError, match="one prompt"):
        _armed_index({"attention_mask": torch.ones(2, 3)})


def _rows(subjects, labels, prefix):
    return [{"sample_id": f"{prefix}{s}", "subject_id": s, "label": y, "hidden_layer": "final"}
            for s, y in zip(subjects, labels)]


def test_layer_caches_are_readable_by_layer_select_heads(tmp_path):
    rng = np.random.default_rng(0)
    layers, size = 3, 5
    subjects = [f"s{i:02d}" for i in range(40)]
    labels = [i % 2 for i in range(40)]
    out = tmp_path / "cache"
    for partition, chunk in (("outer_train", slice(0, 30)), ("final_eval", slice(30, 40))):
        rows = _rows(subjects[chunk], labels[chunk], partition)
        stacks = []
        for y in labels[chunk]:
            stack = rng.normal(size=(layers + 1, size)).astype(np.float32)
            stack[2, 0] += 3.0 * (1 if y else -1)
            stacks.append(stack)
        _write_layer_partition(out, partition, stacks, rows, layers, size)
    metadata = {"weights_source": "base_model_no_adapter", "manifest_sha256": "m", "base_model": "q",
                "condition": "c", "input_modality": "text_only", "pooling": "p", "hidden_layer": "final"}
    write_layer_metadata(out, metadata, layers)
    assert sorted(layer_dirs(out / "layers")) == [0, 1, 2, 3]
    meta = json.loads((layer_dir(out, 2) / "extraction_metadata.json").read_text())
    assert meta["hidden_layer_index"] == 2 and meta["hidden_layer"] == 2
    row = json.loads((layer_dir(out, 2) / "outer_train_rows.jsonl").read_text().splitlines()[0])
    assert row["hidden_layer"] == 2
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subject_ids": subjects[:30], "selection_subject_ids": subjects[30:]}))
    summary = run(layers_root=out / "layers", dev_layers_root=None, split_files=[split],
                  protocol="train_val", output_dir=tmp_path / "heads", route={})
    assert summary["selected_layers"] == [2]


def test_write_layer_partition_checks_shape(tmp_path):
    with pytest.raises(ValueError, match="layer stack shape"):
        _write_layer_partition(tmp_path, "outer_train", [np.zeros((2, 5), np.float32)],
                               [{"sample_id": "a"}], 3, 5)
