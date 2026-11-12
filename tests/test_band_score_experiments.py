import numpy as np
import pytest

from baselines.band_score_experiments import (
    GRID, Linear, band_mask, score_target, select_penalty, train_multiplier,
)
from baselines.merge_diag_common import macro_f1


def _scored():
    g = np.array(["turkish"] * 4 + ["daic"] * 4 + ["cmdc"] * 2)
    s = np.array([5.0, 15.0, 18.0, 30.0, 2.0, 9.0, 11.0, 20.0, np.nan, np.nan])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    return g, s, y


def test_band_mask_uses_corpus_bands_only():
    g, s, _ = _scored()
    assert band_mask(g, s).tolist() == [False, True, True, False, False, True, True, False, False, False]


def test_drop_and_soft_multipliers():
    g, s, _ = _scored()
    drop = train_multiplier(g, s, "drop")
    assert drop.tolist() == [1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
    soft = train_multiplier(g, s, "soft")
    # turkish mid 16.5, ramp 3: |15-16.5|/3 = 0.5 ; daic mid 9.5, ramp 2: |9-9.5|/2 = 0.25
    assert soft[1] == pytest.approx(0.5) and soft[5] == pytest.approx(0.25)
    assert soft[0] == 1.0 and soft[8] == 1.0 and soft.min() >= 0.2
    assert train_multiplier(g, s, "base").tolist() == [1.0] * 10


def test_score_target_sign_matches_cut_and_clinical_is_pm1():
    g, s, y = _scored()
    t = score_target(g, y, s)
    for i in range(8):
        assert (t[i] > 0) == (y[i] == 1)
    assert t[8] == -1.0 and t[9] == 1.0


def test_ridge_recovers_linear_target_and_logistic_classifies():
    rng = np.random.default_rng(0)
    n, d = 120, 6
    z = rng.normal(size=(n, d))
    target = z[:, 0] - 0.5 * z[:, 1]
    y = (target > 0).astype(int)
    g = np.array(["a"] * n)
    reg = Linear(z, g, y, 1e-4, target=target)
    np.testing.assert_allclose(reg.predict(z), target, atol=0.05)
    clf = Linear(z, g, y, 1e-2)
    assert macro_f1(y, clf.predict(z)) > 0.9


def test_select_penalty_returns_grid_value():
    rng = np.random.default_rng(1)
    n = 60
    g = np.array(["daic"] * n)
    s = rng.uniform(0, 24, n)
    y = (s >= 10).astype(int)
    x = rng.normal(size=(n, 5)) + 0.3 * (s - 10)[:, None] * (np.arange(5) == 0)
    for arm in ("logit_base", "logit_drop", "logit_soft", "score_reg"):
        assert select_penalty(x, g, y, s, "global", arm, seed=0) in GRID
