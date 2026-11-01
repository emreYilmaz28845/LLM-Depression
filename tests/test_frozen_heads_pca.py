import json
from pathlib import Path

import numpy as np
import pytest

from baselines.frozen_standalone_heads import (
    PooledCache,
    _prepare,
    fit_pca,
    fit_scaler,
    run_fold,
    run_route,
)


def _write(path: Path, rows, vectors) -> None:
    path.mkdir(parents=True)
    half = len(rows) // 2
    for name, part in (("outer_train", slice(0, half)), ("final_eval", slice(half, None))):
        np.savez(path / f"{name}.npz", vectors=np.asarray(vectors[part], dtype=np.float64))
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[part]))
    meta = {"weights_source": "base_model_no_adapter", "base_model": "m", "manifest_sha256": "x",
            "condition": "c", "input_modality": "audio_only", "pooling": "p"}
    (path / "extraction_metadata.json").write_text(json.dumps(meta))


def _data(n=40, dim=8, windows=4, seed=0, overrides=None):
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for i in range(n):
        subject, label = f"s{i:02d}", int(i % 3 == 0)
        count = (overrides or {}).get(subject, windows)
        for w in range(count):
            rows.append({"sample_id": f"{subject}_{w}", "subject_id": subject, "label": label,
                         "response_id": f"{subject}::q{w % 2}"})
            vectors.append(rng.normal(size=dim) * np.linspace(3, 0.3, dim) + 1.0 * label)
    return rows, vectors


def _cache(tmp_path, name="c", **kw) -> PooledCache:
    rows, vectors = _data(**kw)
    _write(tmp_path / name, rows, vectors)
    return PooledCache([(tmp_path / name, "outer_train"), (tmp_path / name, "final_eval")])


def _split(n=40):
    ids = [f"s{i:02d}" for i in range(n)]
    return {"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:], "final_eval_subject_ids": []}


def test_full_rank_pca_matches_no_pca(tmp_path):
    """K = dimension is a rotation; an L2 logistic head is rotation invariant."""
    cache = _cache(tmp_path)
    base = run_fold(cache, _split(), "train_val", (0.01, 1.0, 100.0))
    full = run_fold(cache, _split(), "train_val", (0.01, 1.0, 100.0), pca_grid=(8,))
    assert base["selection"]["selected_l2_penalty"] == full["selection"]["selected_l2_penalty"]
    a = np.asarray([p["mean_logit"] for p in base["predictions"]])
    b = np.asarray([p["mean_logit"] for p in full["predictions"]])
    assert np.allclose(a, b, atol=1e-6)
    assert np.isclose(full["fit_info"]["pca"]["explained_variance_ratio"], 1.0)


def test_pca_matches_weighted_covariance(tmp_path):
    cache = _cache(tmp_path, overrides={"s00": 12, "s01": 1})
    subjects = [f"s{i:02d}" for i in range(30)]
    mean, scale = fit_scaler(cache, subjects)
    pca = fit_pca(cache, subjects, mean, scale, 3)
    index, weight = cache.row_weights(subjects)
    w = weight / weight.sum()
    z = (cache.x[index] - mean) / scale
    cov = (z * w[:, None]).T @ z
    values, vectors = np.linalg.eigh(cov)
    top = vectors[:, ::-1][:, :3]
    assert np.allclose(np.abs(top.T @ pca["components"]), np.eye(3), atol=1e-6)
    assert np.isclose(pca["cumulative_explained"][-1], values[::-1][:3].sum() / values.sum())


def test_duplicating_a_subjects_windows_leaves_pca_unchanged(tmp_path):
    """Each subject has equal mass, so window count does not steer the components."""
    rows, vectors = _data()
    extra_rows = [dict(r, sample_id=r["sample_id"] + "_dup") for r in rows if r["subject_id"] == "s03"]
    extra_vecs = [v for r, v in zip(rows, vectors) if r["subject_id"] == "s03"]
    _write(tmp_path / "a", rows, vectors)
    _write(tmp_path / "b", rows + extra_rows, vectors + extra_vecs)
    subjects = [f"s{i:02d}" for i in range(30)]
    pa = _prepare(PooledCache([(tmp_path / "a", "outer_train"), (tmp_path / "a", "final_eval")]), subjects, 4)
    pb = _prepare(PooledCache([(tmp_path / "b", "outer_train"), (tmp_path / "b", "final_eval")]), subjects, 4)
    assert np.allclose(np.abs(pa["pca"]["components"].T @ pb["pca"]["components"]), np.eye(4), atol=1e-6)


def test_reported_subjects_never_enter_scaler_or_pca(tmp_path):
    rows, vectors = _data()
    held = {f"s{i:02d}" for i in range(30, 40)}
    noisy = [v * 50 + 7 if r["subject_id"] in held else v for r, v in zip(rows, vectors)]
    _write(tmp_path / "a", rows, vectors)
    _write(tmp_path / "b", rows, noisy)
    out = []
    for name in ("a", "b"):
        cache = PooledCache([(tmp_path / name, "outer_train"), (tmp_path / name, "final_eval")])
        out.append(run_fold(cache, _split(), "train_val", (0.01, 1.0), pca_grid=(2, 4)))
    assert out[0]["selection"]["candidates"] == out[1]["selection"]["candidates"]
    assert out[0]["fit_info"]["pca"] == out[1]["fit_info"]["pca"]


def test_select_grid_and_route_outputs(tmp_path):
    _cache(tmp_path, n=40)
    ids = [f"s{i:02d}" for i in range(40)]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subject_ids": ids[:24], "selection_subject_ids": ids[24:32],
                                 "final_eval_subject_ids": ids[32:]}))
    summary = run_route(cache_dirs=[tmp_path / "c"], dev_caches=None, split_files=[split],
                        protocol="train_val_test", output_dir=tmp_path / "out", route={}, pca_grid=(2, 5))
    fold = summary["folds"][0]
    assert summary["variant"] == "logreg_subjmean_pca_l2sel"
    assert fold["selected_pca_components"] in (2, 5)
    assert 0 < fold["explained_variance_ratio"] <= 1
    assert 0 < fold["between_subject_share_in_subspace"] <= 1
    config = json.loads((tmp_path / "out" / "result_config.json").read_text())
    assert config["pca"]["components_grid"] == [2, 5] and config["pca"]["whiten"] is False
    selection = json.loads((tmp_path / "out" / "fold_0" / "selection.json").read_text())
    assert {c["pca_components"] for c in selection["candidates"]} == {2, 5}


def test_no_pca_output_has_no_pca_fields(tmp_path):
    _cache(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(_split()))
    summary = run_route(cache_dirs=[tmp_path / "c"], dev_caches=None, split_files=[split],
                        protocol="train_val", output_dir=tmp_path / "out", route={})
    config = json.loads((tmp_path / "out" / "result_config.json").read_text())
    assert summary["variant"] == "logreg_subjmean_l2sel" and "pca" not in config
    assert "selected_pca_components" not in summary["folds"][0]


def test_too_many_components_fails(tmp_path):
    cache = _cache(tmp_path, n=40, dim=8)
    with pytest.raises(ValueError, match="rank"):
        _prepare(cache, ["s00", "s01"], 20)
