import numpy as np
import pytest

from baselines.merge_diag_common import Head, Prep, best_threshold, head_weights, macro_f1
from baselines.merge_diagnostics import holm, strat_order


def _toy(seed=0, n=60, d=8):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.array(["a", "b"]), n // 2)
    y = np.r_[(rng.random(n // 2) < 0.2).astype(int), (rng.random(n // 2) < 0.8).astype(int)]
    y[0], y[1], y[n // 2], y[n // 2 + 1] = 0, 1, 0, 1
    x = rng.normal(size=(n, d)) + 2.0 * y[:, None] * (np.arange(d) == 0)
    x[g == "b"] += 5.0  # dataset shift
    return x, g, y


def test_head_weights_equal_dataset_and_class_mass():
    _, g, y = _toy()
    w = head_weights(g, y)
    for d in ("a", "b"):
        for c in (0, 1):
            assert w[(g == d) & (y == c)].sum() == pytest.approx(len(w) / 4)


def test_cb_center_ignores_prevalence_but_per_dataset_does_not():
    x, g, y = _toy()
    cb = Prep("cb_center").fit(x, g, y)
    pdz = Prep("per_dataset").fit(x, g, y)
    for d in ("a", "b"):
        m = g == d
        centre = 0.5 * (x[m & (y == 0)].mean(0) + x[m & (y == 1)].mean(0))
        np.testing.assert_allclose(cb.stats[d][0], centre)
        np.testing.assert_allclose(pdz.stats[d][0], x[m].mean(0))


def test_cb_center_falls_back_to_mean_without_both_classes():
    x, g, y = _toy()
    unl = {"c": np.ones((5, x.shape[1]))}
    p = Prep("cb_center").fit(x, g, y, unlabeled=unl)
    np.testing.assert_allclose(p.stats["c"][0], np.ones(x.shape[1]))


def test_combat_removes_dataset_shift_and_keeps_label_signal():
    x, g, y = _toy(n=200)
    p = Prep("combat").fit(x, g, y)
    z = p.transform(x, g)
    # class-balanced dataset centres are close after harmonization
    centres = [0.5 * (z[(g == d) & (y == 0)].mean(0) + z[(g == d) & (y == 1)].mean(0)) for d in ("a", "b")]
    assert np.abs(centres[0] - centres[1]).max() < 0.35
    assert z[y == 1, 0].mean() - z[y == 0, 0].mean() > 1.0


def test_consistent_keeps_same_sign_feature():
    x, g, y = _toy(n=200)
    p = Prep("consistent", k_consistent=1).fit(x, g, y)
    assert p.columns.tolist() == [0]


def test_head_learns_and_threshold_helps_shifted_logits():
    x, g, y = _toy(n=200)
    z = Prep("per_dataset").fit(x, g, y).transform(x, g)
    head = Head(z, g, y, 0.01)
    logits = head.logits(z)
    assert macro_f1(y, logits) > 0.7
    t = best_threshold(y, logits + 3.0)
    assert macro_f1(y, logits + 3.0 - t) >= macro_f1(y, logits + 3.0)


def test_strat_order_prefixes_keep_both_classes():
    y = np.array([0] * 30 + [1] * 10)
    order = strat_order(y, np.random.default_rng(0))
    assert sorted(order.tolist()) == list(range(40))
    assert set(y[order[:2]].tolist()) == {0, 1}
    assert abs(y[order[:20]].mean() - 0.25) <= 0.05


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
