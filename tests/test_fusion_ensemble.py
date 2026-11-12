import numpy as np
import pytest

from baselines.fusion_ensemble import GRID_WIDE, balanced_weights, members_for, platt, select_with_oof


def test_members_scored_vs_clinical():
    assert members_for("cmdc") == [("text", "logit_base"), ("audio", "logit_base")]
    assert len(members_for("daic")) == 6
    assert ("audio", "score_reg") in members_for("turkish")


def test_balanced_weights_equal_class_mass():
    y = np.array([0] * 30 + [1] * 10)
    w = balanced_weights(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert w.sum() == pytest.approx(len(y))


def test_platt_is_monotone_and_balanced():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    pred = 3.0 * (y - 0.5) + rng.normal(size=len(y)) + 10.0  # shifted, unbalanced
    slope, icpt = platt(pred, y)
    assert slope > 0
    cal = slope * pred + icpt
    # balanced calibration: the decision point sits between the class means
    assert cal[y == 0].mean() < 0 < cal[y == 1].mean()


def test_select_with_oof_returns_grid_value_and_full_oof():
    rng = np.random.default_rng(1)
    n = 60
    g = np.array(["daic"] * n)
    s = rng.uniform(0, 24, n)
    y = (s >= 10).astype(int)
    x = rng.normal(size=(n, 5)) + 0.3 * (s - 10)[:, None] * (np.arange(5) == 0)
    pen, oof = select_with_oof(x, g, y, s, "score_reg", seed=0)
    assert pen in GRID_WIDE and oof.shape == (n,) and np.isfinite(oof).all()
