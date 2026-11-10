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


def test_swap_test_matches_brute_force_and_holm():
    from itertools import product

    from sklearn.metrics import f1_score

    from baselines.text_replication_significance import Pair, holm

    rng = np.random.default_rng(0)
    subjects = [f"d::{i}" for i in range(8)]
    labels = {s: int(i % 2) for i, s in enumerate(subjects)}
    a = {(f, s, u): int(rng.integers(0, 2)) for f in (0, 1) for s in (0, 1) for u in subjects[f * 4:(f + 1) * 4]}
    b = {k: int(rng.integers(0, 2)) for k in a}
    pair = Pair(a, b, labels)

    def score(pred):
        out = []
        for f in (0, 1):
            out.append(np.mean([f1_score([labels[u] for u in subjects[f * 4:(f + 1) * 4]],
                                         [pred[(f, s, u)] for u in subjects[f * 4:(f + 1) * 4]],
                                         labels=[0, 1], average="macro", zero_division=0) for s in (0, 1)]))
        return float(np.mean(out))

    sa, sb, d = pair.observed()
    assert np.isclose(sa, score(a)) and np.isclose(sb, score(b))
    p, k, method = pair.swap_test(0)
    disc = sorted({u for (f, s, u) in a if a[(f, s, u)] != b[(f, s, u)]})
    hits = 0
    for bits in product((0, 1), repeat=len(disc)):
        swap = {u for u, bit in zip(disc, bits) if bit}
        a2 = {key: (b[key] if key[2] in swap else v) for key, v in a.items()}
        b2 = {key: (a[key] if key[2] in swap else v) for key, v in b.items()}
        hits += abs(score(b2) - score(a2)) >= abs(d) - 1e-12
    assert method == "exact" and k == len(disc) and np.isclose(p, hits / 2 ** len(disc))
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
