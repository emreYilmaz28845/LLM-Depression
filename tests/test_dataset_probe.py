import numpy as np

import baselines.dataset_probe as dp


def _data(seed=0, per=24, dim=30, shift=3.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.asarray(dp.DATASETS), per)
    offsets = {d: rng.normal(scale=shift, size=dim) for d in dp.DATASETS}
    y = np.tile([0, 1], len(g) // 2)
    x = rng.normal(size=(len(g), dim)) + np.stack([offsets[d] for d in g])
    x[:, 0] += 2.0 * y
    return x, g, y


def test_probe_reads_datasets_and_erasure_removes_them():
    x, g, _ = _data()
    assert dp.probe_cv(x, g)["balanced_accuracy"] > 0.9
    assert dp.probe_cv(x, g, erase=True)["balanced_accuracy"] < 0.45
    assert dp.shuffled_control(x, g)["balanced_accuracy_mean"] < 0.45


def test_few_shot_window_probe_and_overlap_run():
    x, g, y = _data()
    fs = dp.few_shot(x, g)["per_dataset_train_subjects"]
    assert set(fs) == {"2", "5", "10", "20"}
    subjects = [f"s{i}" for i in range(len(g))]
    windows = (np.repeat(x, 3, axis=0), [s for s in subjects for _ in range(3)])
    out = dp.window_probe(windows, dict(zip(subjects, g)))
    assert out["windows"] == 3 * len(g) and out["balanced_accuracy"] > 0.9
    ov = dp.depression_overlap(x, g, y)
    assert 0.0 <= ov["cos2_depression_vs_dataset_subspace"] <= 1.0
    assert ov["dataset_subspace_dims"] == 4
