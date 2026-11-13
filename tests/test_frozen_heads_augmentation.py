import json
from pathlib import Path

import numpy as np
import pytest

from baselines.frozen_standalone_heads import (
    AUG_METHODS,
    PooledCache,
    _design,
    _prepare,
    _synthetic,
    run_route,
)


def _cache(path: Path, n=40, dim=6, units=3, windows=2, seed=0) -> list[str]:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    rows, vectors = [], []
    for i in range(n):
        subject, label = f"s{i:02d}", int(i % 3 == 0)
        for u in range(units):
            for w in range(windows):
                rows.append({"sample_id": f"{subject}_{u}_{w}", "subject_id": subject, "label": label,
                             "response_id": f"{subject}::q{u}"})
                vectors.append(rng.normal(size=dim) + 1.5 * label)
    half = len(rows) // 2
    for name, part in (("outer_train", slice(0, half)), ("final_eval", slice(half, None))):
        np.savez(path / f"{name}.npz", vectors=np.asarray(vectors[part], dtype=np.float64))
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[part]))
    meta = {"weights_source": "base_model_no_adapter", "base_model": "m", "manifest_sha256": "x",
            "condition": "c", "input_modality": "audio_only", "pooling": "p"}
    (path / "extraction_metadata.json").write_text(json.dumps(meta))
    return [f"s{i:02d}" for i in range(n)]


def _pooled(tmp_path):
    ids = _cache(tmp_path / "cache")
    from baselines.frozen_standalone_heads import _load_caches
    cache, _ = _load_caches(tmp_path / "cache", None, False)
    return cache, ids


@pytest.mark.parametrize("method", AUG_METHODS)
def test_synthetic_rows_keep_class_balance_and_size(tmp_path, method):
    cache, ids = _pooled(tmp_path)
    fit = ids[:30]
    prep = _prepare(cache, fit, None)
    x, y, w, info = _design(cache, prep, fit, None, {"method": method, "ratio": 2.0, "seed": 0},
                            np.random.default_rng(0))
    real = cache.y(fit)
    assert info["real_rows"] == 30
    # Total weight per class stays equal (balanced) with soft-label shares.
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum(), rtol=0.25 if method == "mixup" else 1e-9)
    if method != "mixup":
        assert info["synthetic_rows"] == 2 * len(real)
        assert np.isclose(info["synthetic_weight_share"], 2 / 3)


@pytest.mark.parametrize("method", ("smote", "gauss", "mixup"))
def test_span_methods_stay_in_real_subject_span(method):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 6))  # rank-2 subject means in 6-D
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    xs, _, _ = _synthetic(None, None, [], x, y, {"method": method, "ratio": 3.0}, np.random.default_rng(1))
    basis = np.linalg.svd(x, full_matrices=False)[2][:2]
    assert np.abs(xs - xs @ basis.T @ basis).max() < 1e-9


def test_wboot_uses_only_fit_subject_windows(tmp_path):
    cache, ids = _pooled(tmp_path)
    fit = ids[:6]
    prep = _prepare(cache, fit, None)
    z = (cache.x - prep["mean"]) / prep["scale"]
    from baselines.frozen_standalone_heads import _features
    x = _features(cache, prep, fit, None)
    y = cache.y(fit)
    xs, ys, _ = _synthetic(cache, prep, fit, x, y, {"method": "wboot", "ratio": 1.0}, np.random.default_rng(3))
    allowed = np.concatenate([z[rows] for s in fit for rows in cache.units[s].values()])
    lo, hi = allowed.min(0) - 1e-9, allowed.max(0) + 1e-9
    assert ((xs >= lo) & (xs <= hi)).all()
    assert sorted(ys.tolist()) == sorted(y.tolist())


def test_augmented_route_is_deterministic_and_records_config(tmp_path):
    ids = _cache(tmp_path / "cache")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:],
                                 "final_eval_subject_ids": []}))
    aug = {"method": "wboot", "ratio": 1.0, "seed": 4}
    a = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[split], protocol="train_val",
                  output_dir=tmp_path / "a", route={}, augment=aug)
    b = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[split], protocol="train_val",
                  output_dir=tmp_path / "b", route={}, augment=aug)
    assert a["folds"] == b["folds"]
    config = json.loads((tmp_path / "a" / "result_config.json").read_text())
    assert config["variant"] == "logreg_subjmean_aug_l2sel" and config["augmentation"]["method"] == "wboot"
    pa = (tmp_path / "a" / "fold_0" / "predictions_subject_level.jsonl").read_text()
    assert pa == (tmp_path / "b" / "fold_0" / "predictions_subject_level.jsonl").read_text()
    metrics = json.loads((tmp_path / "a" / "fold_0" / "metrics.json").read_text())
    assert metrics["fit_info"]["augmentation"]["synthetic_rows"] == 30


def test_no_augmentation_is_unchanged(tmp_path):
    ids = _cache(tmp_path / "cache")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:],
                                 "final_eval_subject_ids": []}))
    s = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[split], protocol="train_val",
                  output_dir=tmp_path / "a", route={})
    config = json.loads((tmp_path / "a" / "result_config.json").read_text())
    assert config["variant"] == "logreg_subjmean_l2sel" and "augmentation" not in config
    assert s["folds"][0]["report"] == 10


def test_rejects_pca_with_augmentation(tmp_path):
    ids = _cache(tmp_path / "cache")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:],
                                 "final_eval_subject_ids": []}))
    with pytest.raises(ValueError):
        run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[split], protocol="train_val",
                  output_dir=tmp_path / "a", route={}, pca_grid=(2,), augment={"method": "smote", "ratio": 1,
                                                                             "seed": 0})
