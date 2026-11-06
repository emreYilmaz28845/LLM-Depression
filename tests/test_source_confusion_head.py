from collections import Counter

import numpy as np
import torch

import baselines.source_confusion_head as sc


def _toy(seed=0, n=40, dim=20, shift=3.0):
    rng = np.random.default_rng(seed)
    groups = [g for g in "abcd" for _ in range(n)]
    y = np.tile(np.r_[np.ones(n // 4), np.zeros(n - n // 4)], 4).astype(int)
    x = rng.normal(size=(len(groups), dim))
    x += np.repeat(rng.normal(scale=shift, size=(4, dim)), n, axis=0)
    x[:, 0] += 2.0 * y
    return x, y, groups


def test_cell_sampler_draws_every_cell_equally():
    _, y, groups = _toy()
    d = np.asarray(["abcd".index(g) for g in groups])
    sampler = sc.CellSampler(d, y, 3, 0)
    ids = np.concatenate([sampler.draw() for _ in range(5)])
    cells = Counter((int(d[i]), int(y[i])) for i in ids)
    assert len(cells) == 8 and set(cells.values()) == {15}


def test_cell_weights_give_each_cell_equal_mass():
    _, y, groups = _toy()
    d = np.asarray(["abcd".index(g) for g in groups])
    w = sc.cell_weights(d, y)
    assert np.isclose(w.sum(), 1.0)
    for a in range(4):
        for b in (0, 1):
            assert np.isclose(w[(d == a) & (y == b)].sum(), 1 / 8)


def test_confusion_loss_is_zero_for_uniform_critics():
    cfg = sc.ConfusionConfig(width=4)
    model = sc.ConfusionNet(4, 3, cfg)
    for heads in model.critics:
        for head in heads:
            for p in head.parameters():
                torch.nn.init.zeros_(p)
    z = torch.randn(12, 4)
    d = torch.arange(12) % 3
    y = (torch.arange(12) // 6).long()
    assert abs(float(sc.critic_loss(model, z, d, y, confuse=True))) < 1e-6


def test_alignment_matrix_is_zero_for_identical_dataset_distributions():
    d = np.repeat(np.arange(3), 4)
    y = np.tile([0, 0, 1, 1], 3)
    matrix = sc.alignment_matrix(d, y)
    z = torch.randn(4, 5).repeat(3, 1)  # every dataset has the same points
    assert abs(float(sc.mmd_loss(z, matrix))) < 1e-5
    z2 = z.clone()
    z2[d == 0] += 5.0
    assert float(sc.mmd_loss(z2, matrix)) > 0.01


def test_training_is_deterministic_and_snapshots_requested_steps():
    x, y, groups = _toy(n=20)
    cfg = sc.ConfusionConfig(method="combined", strength=1.0, seed=1, steps=6, warmup_steps=2, per_cell=2,
                             critic_steps=1, width=8, hidden=16)
    a = sc.train_confusion(x, y, groups, cfg, (0, 6), (x,))
    b = sc.train_confusion(x, y, groups, cfg, (0, 6), (x,))
    assert set(a["snapshots"]) == {0, 6}
    assert np.array_equal(a["snapshots"][6][0], b["snapshots"][6][0])
    assert not np.array_equal(a["snapshots"][0][0], a["snapshots"][6][0])


def test_confusion_lowers_fresh_dataset_recovery_on_toy_data():
    x, y, groups = _toy(n=60, dim=12, shift=1.0)
    pca = sc.fit_pca(x, np.asarray(groups), y, 12)
    p = sc.apply_pca(pca, x)
    rng = np.random.default_rng(0)
    test = rng.permutation(len(y))[:80]
    train = np.setdiff1d(np.arange(len(y)), test)
    out = {}
    for method, strength in (("task", 0.0), ("conditional", 10.0)):
        cfg = sc.ConfusionConfig(method=method, strength=strength, seed=0, steps=300, warmup_steps=20, per_cell=6,
                                 width=8, hidden=32)
        z = sc.train_confusion(p[train], y[train], [groups[i] for i in train], cfg, (300,), (p,))["snapshots"][300][0]
        g = np.asarray(groups)
        out[method] = sc.fresh_attacks(z[train], list(g[train]), y[train], z[test], list(g[test]), y[test])["all_linear"]
    assert out["conditional"]["balanced_accuracy"] < out["task"]["balanced_accuracy"] - 0.05


def test_two_stage_selects_step_and_l2_on_inner_folds():
    x, y, groups = _toy(n=24, dim=10)
    rng = np.random.default_rng(0)
    folds = [np.sort(f) for f in np.array_split(rng.permutation(len(y)), 3)]
    base = sc.ConfusionConfig(width=4, hidden=8, per_cell=2, critic_steps=1, warmup_steps=2, pca_dims=6)
    out = sc.two_stage(x, y, groups, x[:5], folds, seed=0, variants=(("task", 0.0), ("combined", 1.0)),
                       step_grid=(0, 4), l2_grid=(0.1, 10.0), base=base)
    assert set(out) == {"task", "comb1"}
    for r in out.values():
        assert r["selection"]["selected_step"] in (0, 4)
        assert r["selection"]["selected_l2_penalty"] in (0.1, 10.0)
        assert len(r["selection"]["candidates"]) == 4
        assert r["holdout_logits"].shape == (5,) and r["z_holdout"].shape == (5, 4)
