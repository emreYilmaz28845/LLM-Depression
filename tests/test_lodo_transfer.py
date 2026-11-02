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
