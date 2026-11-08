from collections import Counter

import numpy as np
import torch

import baselines.dann_subject_head as dh
from baselines.run_merged_dann_matrix import lambda_name, variant_lambda


def _toy(seed=0, n=40, dim=16):
    rng = np.random.default_rng(seed)
    groups = [g for g in "abcd" for _ in range(n)]
    y = np.tile(np.r_[np.ones(n // 4), np.zeros(n - n // 4)], 4).astype(int)
    x = rng.normal(size=(len(groups), dim))
    x += np.repeat(rng.normal(scale=3.0, size=(4, dim)), n, axis=0)  # strong dataset offsets
    x[:, 0] += 2.0 * y
    return x, y, groups


def test_gradient_reversal_is_identity_forward_and_flips_gradient():
    x = torch.ones(3, requires_grad=True)
    out = dh.grad_reverse(x, 0.5)
    assert torch.equal(out, x)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_oversampling_equalizes_every_dataset_class_cell():
    _, y, groups = _toy()
    rows = dh.oversample_cells(groups, y, np.random.default_rng(0))
    cells = Counter((groups[i], int(y[i])) for i in rows)
    assert len(set(cells.values())) == 1 and len(cells) == 8


def test_training_is_deterministic_for_a_seed():
    x, y, groups = _toy()
    cfg = dh.DannConfig(adv_lambda=0.3, seed=3, epochs=3, hidden=8, dataset_hidden=8)
    a = dh.predict(dh.train_dann(x, y, groups, cfg)["model"], x)["logit"]
    b = dh.predict(dh.train_dann(x, y, groups, cfg)["model"], x)["logit"]
    assert np.array_equal(a, b)


def test_reversal_lowers_dataset_readout_and_keeps_depression():
    x, y, groups = _toy(n=60)
    out = {}
    for lam in (0.0, 1.0):
        cfg = dh.DannConfig(adv_lambda=lam, seed=0, epochs=60, hidden=16, dataset_hidden=16, dropout=0.0)
        model = dh.train_dann(x, y, groups, cfg)["model"]
        p = dh.predict(model, x)
        head_acc = np.mean(np.asarray(sorted(set(groups)))[p["dataset_logit"].argmax(1)] == np.asarray(groups))
        out[lam] = (head_acc, np.mean((p["logit"] > 0) == y))
    assert out[0.0][0] > 0.9
    assert out[1.0][0] < out[0.0][0] - 0.2
    assert out[1.0][1] > 0.75


def test_variant_names():
    assert variant_lambda("M") == 0.0 and variant_lambda("D0.3") == 0.3
    assert [lambda_name(x) for x in (0.0, 0.1, 1.0)] == ["M", "D0.1", "D1"]


def test_two_stage_selects_on_inner_folds_and_fits_l2_head_on_z():
    x, y, groups = _toy(n=24, dim=10)
    rng = np.random.default_rng(0)
    folds = [np.sort(f) for f in np.array_split(rng.permutation(len(y)), 3)]
    base = dh.DannConfig(hidden=8, dataset_hidden=8)
    out = dh.two_stage(x, y, groups, x[:5], folds, seed=0, lambdas=(0.0, 0.5), epoch_grid=(2, 4),
                       l2_grid=(0.1, 10.0), base=base)
    assert set(out["per_lambda"]) == {0.0, 0.5} and out["selected_lambda"] in (0.0, 0.5)
    for r in out["per_lambda"].values():
        assert r["selection"]["selected_epochs"] in (2, 4)
        assert r["selection"]["selected_l2_penalty"] in (0.1, 10.0)
        assert len(r["selection"]["candidates"]) == 4
        assert r["holdout_logits"].shape == (5,) and r["z_holdout"].shape == (5, 8)
