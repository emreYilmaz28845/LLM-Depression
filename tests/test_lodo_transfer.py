import numpy as np

import baselines.lodo_transfer as lt


def test_lodo_runs_all_variants_and_scores_every_target():
    rng = np.random.default_rng(0)
    datasets = ["a", "b", "c", "d", "e"]
    g = np.repeat(datasets, 16)
    y = np.tile([0, 1], len(g) // 2)
    x = rng.normal(size=(len(g), 12)) + np.stack([rng.normal(scale=3, size=12) for _ in datasets]).repeat(16, 0)
    x[:, 0] += 3.0 * y
    for v in lt.VARIANTS:
        out = lt.lodo(x, g, y, v)
        assert sorted(out) == datasets
        assert all(out[t]["subjects"] == 16 for t in datasets)
    # Per-dataset standardization should transfer the shared label direction despite the offsets.
    assert np.mean([r["macro_f1"] for r in lt.lodo(x, g, y, "T").values()]) > 0.8


def test_oversample_fills_cells_and_lodo_runs_seeded():
    g = np.asarray(["a"] * 5 + ["b"] * 3)
    y = np.asarray([1, 0, 0, 0, 0, 1, 1, 0])
    index = lt._oversample(g, y, np.random.default_rng(0))
    cells = {}
    for i in index:
        cells[(g[i], y[i])] = cells.get((g[i], y[i]), 0) + 1
    assert set(cells.values()) == {4}
    rng = np.random.default_rng(1)
    g = np.repeat(["a", "b", "c", "d", "e"], 12)
    y = np.asarray(([1] * 3 + [0] * 9) * 5)
    x = rng.normal(size=(len(g), 6))
    x[:, 0] += 2.0 * y
    first, second = lt.lodo(x, g, y, "B", 0), lt.lodo(x, g, y, "B", 0)
    assert first == second
