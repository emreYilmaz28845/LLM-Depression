from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import train as train_module
from src.lpft_head import (
    HEAD_INIT_SCHEMA_VERSION,
    build_head_module,
    fold_standardization,
    last_prompt_hidden,
    load_head_init,
    load_head_state,
    save_head,
    subject_mean_logits,
)
from src.model.collator import Qwen2AudioSFTCollator


def _init(tmp_path, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    mean, scale = rng.normal(size=dim), rng.uniform(0.5, 2.0, size=dim)
    weight, bias = rng.normal(size=dim), 0.3
    folded, folded_bias = fold_standardization(mean, scale, weight, bias)
    path = tmp_path / "init.npz"
    np.savez(path, schema_version=np.asarray(HEAD_INIT_SCHEMA_VERSION), weight=folded,
             bias=np.asarray(folded_bias), mean=mean, scale=scale, l2_penalty=np.asarray(10.0))
    return path, mean, scale, weight, bias


def test_fold_standardization_matches_standardised_head(tmp_path):
    _, mean, scale, weight, bias = _init(tmp_path)
    x = np.random.default_rng(1).normal(size=(9, 6)) * 3 + 1
    folded, folded_bias = fold_standardization(mean, scale, weight, bias)
    np.testing.assert_allclose(((x - mean) / scale) @ weight + bias, x @ folded + folded_bias, atol=1e-12)


def test_fold_standardization_rejects_bad_scale():
    with pytest.raises(ValueError):
        fold_standardization(np.zeros(2), np.array([1.0, 0.0]), np.ones(2), 0.0)


def test_head_module_forward_and_l2(tmp_path):
    path, _, scale, weight, _ = _init(tmp_path)
    init = load_head_init(path)
    head = build_head_module(init, "cpu")
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    expected = x.float().numpy() @ init["weight"] + init["bias"]
    np.testing.assert_allclose(head(x).detach().numpy(), expected, rtol=1e-5, atol=1e-5)
    # The penalty is on the standardised weights, as in the frozen fit.
    assert float(head.l2_term()) == pytest.approx(0.5 * 10.0 * float(np.sum(weight**2)), rel=1e-5)


def test_load_head_init_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, schema_version=np.asarray("other"), weight=np.ones(2), bias=np.asarray(0.0),
             scale=np.ones(2), l2_penalty=np.asarray(1.0))
    with pytest.raises(ValueError, match="unexpected schema"):
        load_head_init(path)


def test_save_and_load_head_state_round_trip(tmp_path):
    path, *_ = _init(tmp_path)
    head = build_head_module(load_head_init(path), "cpu")
    sha = save_head(head, tmp_path / "head.pt", {"epoch": 2})
    state = load_head_state(tmp_path / "head.pt")
    assert state["sha256"] == sha
    assert state["meta"] == {"epoch": 2}
    np.testing.assert_allclose(state["weight"], head.weight.detach().double().numpy())


def test_last_prompt_hidden_picks_prompt_end():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    picked = last_prompt_hidden(hidden, torch.tensor([2, 5]))
    assert torch.equal(picked[0], hidden[0, 1])
    assert torch.equal(picked[1], hidden[1, 4])
    with pytest.raises(ValueError):
        last_prompt_hidden(hidden, torch.tensor([0, 1]))


def test_subject_mean_logits_equal_chunk_weight_and_label_check():
    rows = [{"subject_id": "b", "label": 1}, {"subject_id": "a", "label": 0},
            {"subject_id": "b", "label": 1}, {"subject_id": "b", "label": 1}]
    subjects, labels, logits = subject_mean_logits([1.0, -2.0, 2.0, 3.0], rows)
    assert subjects == ["a", "b"]
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(logits, [-2.0, 2.0])
    with pytest.raises(ValueError, match="conflicting"):
        subject_mean_logits([0.0, 0.0], [{"subject_id": "a", "label": 0}, {"subject_id": "a", "label": 1}])


class _FakeProcessor:
    def __init__(self, prefix_ok=True):
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.prefix_ok = prefix_ok

    def __call__(self, text, return_tensors=None, padding=False, **_):
        ids = [ord(c) % 50 + 1 for c in text]
        if not self.prefix_ok and text.endswith("X"):
            ids[0] = 99
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _example(label=1):
    return {"training_text": "promptX", "prompt_text": "prompt", "audio_arrays": [],
            "sample_id": "s1", "subject_id": "a", "label": label, "loss_weight": 1.5}


def test_collator_emits_head_targets():
    batch = Qwen2AudioSFTCollator(_FakeProcessor(), emit_head_targets=True)([_example(1)])
    assert batch["head_label"].tolist() == [1.0]
    assert batch["prompt_len"].tolist() == [len("prompt")]
    assert "head_label" not in Qwen2AudioSFTCollator(_FakeProcessor())([_example(1)])


def test_collator_refuses_non_prefix_prompt():
    with pytest.raises(ValueError, match="not a prefix"):
        Qwen2AudioSFTCollator(_FakeProcessor(prefix_ok=False), emit_head_targets=True)([_example(0)])


def test_lpft_model_inputs_strip_targets():
    batch = {"input_ids": 1, "labels": 2, "loss_weight": 3, "head_label": 4, "prompt_len": 5}
    inputs, label, prompt_len = train_module._lpft_model_inputs(batch)
    assert inputs == {"input_ids": 1}
    assert (label, prompt_len) == (4, 5)
    assert "labels" in batch  # the caller's batch is not mutated


def test_epoch_zero_parity_check():
    cfg = {"expected_selection": {"macro_f1": 0.75, "binary_cross_entropy": 0.5}, "parity_bce_tolerance": 0.01}
    good = {"primary_headline_metrics": {"macro_f1": 0.75}, "primary_selection_loss": 0.505}
    assert train_module._check_lpft_epoch_zero_parity(good, cfg)["passed"]
    bad_f1 = {"primary_headline_metrics": {"macro_f1": 0.70}, "primary_selection_loss": 0.5}
    assert not train_module._check_lpft_epoch_zero_parity(bad_f1, cfg)["passed"]
    bad_loss = {"primary_headline_metrics": {"macro_f1": 0.75}, "primary_selection_loss": 0.6}
    assert not train_module._check_lpft_epoch_zero_parity(bad_loss, cfg)["passed"]
    with pytest.raises(ValueError):
        train_module._check_lpft_epoch_zero_parity(good, {})


class _FakeModel:
    """Hidden state at every position equals the sample's feature vector."""

    def __init__(self, features):
        self.features = features
        self.calls = 0

    def __call__(self, input_ids, output_hidden_states, use_cache, return_dict, **_):
        assert output_hidden_states and not use_cache and return_dict
        vector = self.features[self.calls]
        self.calls += 1
        hidden = vector.reshape(1, 1, -1).expand(1, input_ids.shape[1], -1)
        return SimpleNamespace(hidden_states=(hidden * 0, hidden))


def test_lpft_selection_eval_scores_subject_means(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "prepare_model_for_evaluation", lambda model, config: None)
    path, *_ = _init(tmp_path, dim=3)
    init = load_head_init(path)
    head = build_head_module(init, "cpu")
    rng = np.random.default_rng(3)
    examples = [{"sample_id": f"s{i}", "subject_id": f"p{i // 2}", "label": int(i // 2 % 2)} for i in range(8)]
    features = torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float32)
    loader = [
        {"input_ids": torch.ones(1, 4, dtype=torch.long), "labels": torch.ones(1, 4, dtype=torch.long),
         "head_label": torch.tensor([float(e["label"])]), "prompt_len": torch.tensor([3])}
        for e in examples
    ]
    component = {"name": "daic_val", "dataset": "daic", "split_name": "val", "examples": examples,
                 "loss_loader": loader, "log_dir_prefix": "val"}
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    result = train_module._evaluate_lpft_head_selection(
        accelerator, _FakeModel(features), head, [component], {}, tmp_path, 0
    )
    sample_logits = features.numpy() @ init["weight"] + init["bias"]
    subject_logits = sample_logits.reshape(4, 2).mean(1)
    labels = np.array([0, 1, 0, 1])
    expected_bce = float(np.mean(np.logaddexp(0.0, subject_logits) - labels * subject_logits))
    assert result["primary_selection_loss"] == pytest.approx(expected_bce, rel=1e-5)
    assert (tmp_path / "val_epoch_0" / "predictions_subject_level.csv").exists()
    assert (tmp_path / "val_epoch_0" / "metrics_lpft_head.json").exists()
    assert set(result["primary_headline_metrics"]) >= {"macro_f1", "positive_f1", "accuracy", "precision", "recall"}


def test_lpft_selection_eval_avoids_inference_mode():
    # Epoch 0 is the run's first FSDP unshard; storage created under
    # inference_mode breaks the checkpoint state_dict afterwards (MN5 job 49050580).
    import inspect

    source = inspect.getsource(train_module._evaluate_lpft_head_selection)
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    assert "torch.no_grad()" in code
    assert "inference_mode" not in code
