import numpy as np

from baselines.text_replication import fit_scaler, metrics, oversample_draw


def test_oversample_draw_balances_and_keeps_every_speaker():
    labels = np.array([0] * 7 + [1] * 3)
    draw = oversample_draw(labels, seed=0)
    assert len(draw) == 14
    assert (labels[draw] == 0).sum() == (labels[draw] == 1).sum() == 7
    assert set(draw.tolist()) == set(range(10))
    assert np.array_equal(draw, oversample_draw(labels, seed=0))


def test_oversample_draw_matches_reference_sampler():
    labels = np.array([1, 0, 1, 1, 0, 1])
    rng = np.random.default_rng(2)
    groups = [np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)]
    n = max(len(g) for g in groups)
    ref = np.concatenate([np.concatenate([g, rng.choice(g, n - len(g), replace=True)]) for g in groups])
    rng.shuffle(ref)
    assert np.array_equal(oversample_draw(labels, seed=2), ref)


def test_fit_scaler_population_std_and_constant_guard():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, std = fit_scaler(x)
    assert np.allclose(mean, [2.0, 5.0])
    assert np.allclose(std, [1.0, 1.0])  # ddof=0 std of [1, 3] is 1; constant column -> 1


def test_metrics_macro_f1_zero_division():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))
    assert m["macro_f1"] == (0.0 + 2 * 0.5 / 1.5) / 2
    assert m["confusion_matrix"] == [[2, 0], [2, 0]]
