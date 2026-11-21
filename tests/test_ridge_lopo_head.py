"""Tests for the separate OZY-158 closed-form ridge + nested LOPO head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.ridge_lopo_head import (
    DEFAULT_CONFIG_PATH,
    fit_ridge_path,
    fit_ridge_single,
    fit_subject_znorm,
    load_head_config,
    oversample_minority,
    run_route,
    select_c_nested_lopo,
)
from src.utils import read_json, save_json, write_jsonl


def _write_synthetic_cache(
    cache_dir: Path,
    *,
    train_subjects: list[tuple[str, int, np.ndarray]],
    eval_subjects: list[tuple[str, int, np.ndarray]],
    weights_source: str = "base_model_no_adapter",
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for part, items in (("outer_train", train_subjects), ("final_eval", eval_subjects)):
        rows = []
        vecs = []
        for sid, label, chunks in items:
            for c_idx, vec in enumerate(chunks):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "sample_id": f"{sid}_c{c_idx}",
                        "subject_id": sid,
                        "label": int(label),
                    }
                )
                vecs.append(vec)
        np.savez_compressed(cache_dir / f"{part}.npz", vectors=np.stack(vecs).astype(np.float32))
        write_jsonl(rows, cache_dir / f"{part}_rows.jsonl")
    save_json(
        {
            "schema_version": "qwen_hidden_cache.v1",
            "model_backend": "qwen3_omni",
            "base_model": "/models/Qwen3-Omni-30B-A3B",
            "weights_source": weights_source,
            "manifest_sha256": "deadbeef" * 8,
            "condition": "text_only",
            "input_modality": "text_only",
            "pooling": "last_valid_prompt_token",
        },
        cache_dir / "extraction_metadata.json",
    )


def test_default_yaml_config_matches_ozy158_spec() -> None:
    cfg = load_head_config(DEFAULT_CONFIG_PATH)
    assert cfg["config_name"] == "ridge_lopo_ozy158_v1"
    assert cfg["c_grid"] == (0.001,)
    assert cfg["seed"] == 100
    assert cfg["oversample_minority"] is True
    assert cfg["threshold"] == 0.0


def test_closed_form_svd_ridge_matches_primal_normal_equations() -> None:
    rng = np.random.default_rng(7)
    # Test both d > n (high-dimensional) and n > d regimes.
    for n, d in ((16, 48), (40, 12)):
        x = rng.normal(loc=2.5, scale=1.7, size=(n, d))
        y = np.array([1] * (n // 3) + [0] * (n - n // 3), dtype=np.int64)
        t = np.where(y == 1, 1.0, -1.0)
        c_grid = (0.001, 0.01, 0.1, 1.0, 10.0)
        sols = fit_ridge_path(x, y, c_grid)

        x_mean = x.mean(axis=0)
        t_mean = float(t.mean())
        xc = x - x_mean
        tc = t - t_mean

        for c in c_grid:
            lam = 1.0 / c
            w_primal = np.linalg.solve(xc.T @ xc + lam * np.eye(d), xc.T @ tc)
            b_primal = t_mean - float(x_mean @ w_primal)
            w_svd, b_svd = sols[c]
            np.testing.assert_allclose(w_svd, w_primal, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(b_svd, b_primal, rtol=1e-10, atol=1e-10)
            w_single, b_single = fit_ridge_single(x, y, c)
            np.testing.assert_allclose(w_single, w_svd, rtol=1e-12, atol=1e-12)
            assert b_single == pytest.approx(b_svd, abs=1e-12)


def test_oversample_minority_balances_deterministically() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    x1, y1 = oversample_minority(x, y, np.random.default_rng(100))
    x2, y2 = oversample_minority(x, y, np.random.default_rng(100))
    assert int((y1 == 1).sum()) == 7
    assert int((y1 == 0).sum()) == 7
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_znorm_handles_constant_features() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mean, std = fit_subject_znorm(x)
    np.testing.assert_allclose(mean, [3.0, 5.0])
    assert std[1] == 1.0
    z = (x - mean) / std
    assert np.isfinite(z).all()


def test_nested_lopo_is_deterministic_and_separates_linear_signal() -> None:
    rng = np.random.default_rng(42)
    n_pos, n_neg, d = 8, 16, 24
    x_pos = rng.normal(loc=0.0, scale=0.5, size=(n_pos, d))
    x_pos[:, 0] += 3.0
    x_neg = rng.normal(loc=0.0, scale=0.5, size=(n_neg, d))
    x_neg[:, 0] -= 3.0
    x = np.vstack([x_pos, x_neg])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)

    s1 = select_c_nested_lopo(x, y, (0.001, 0.01, 0.1, 1.0, 10.0), seed=100, outer_fold=0)
    s2 = select_c_nested_lopo(x, y, (0.001, 0.01, 0.1, 1.0, 10.0), seed=100, outer_fold=0)
    assert s1 == s2
    assert s1["num_inner_folds"] == n_pos + n_neg
    assert s1["selected_inner_metrics"]["macro_f1"] >= 0.95


def test_run_route_split_protocol_and_outer_lopo_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    d = 16

    def make_subject(sid: str, label: int, n_chunks: int = 2) -> tuple[str, int, np.ndarray]:
        base = rng.normal(0.0, 0.4, size=(n_chunks, d))
        base[:, 0] += 2.5 if label == 1 else -2.5
        return sid, label, base

    train_items = [make_subject(f"tr_{i}", 1 if i < 6 else 0) for i in range(18)]
    eval_items = [make_subject(f"ev_{i}", 1 if i < 3 else 0) for i in range(8)]
    cache_dir = tmp_path / "cache"
    _write_synthetic_cache(cache_dir, train_subjects=train_items, eval_subjects=eval_items)

    split_path = tmp_path / "split_0.json"
    save_json(
        {
            "train_subject_ids": [s for s, _, _ in train_items],
            "selection_subject_ids": [],
            "final_eval_subject_ids": [s for s, _, _ in eval_items],
        },
        split_path,
    )

    out_split = tmp_path / "out_split"
    summary_split = run_route(
        cache_dirs=[cache_dir],
        split_files=[split_path],
        protocol="train_cv_test",
        output_dir=out_split,
        route={"name": "synthetic/text_only"},
    )
    assert summary_split["variant"] == "ridge_closedform_oversample_znorm_lopo"
    assert summary_split["fold_mean"]["macro_f1"] >= 0.85
    assert (out_split / "result_config.json").exists()
    assert (out_split / "summary.json").exists()
    assert (out_split / "fold_0" / "predictions_subject_level.jsonl").exists()
    assert (out_split / "fold_0" / "predictions_subject_level.csv").exists()
    cfg = read_json(out_split / "result_config.json")
    assert cfg["c_grid"] == [0.001, 0.01, 0.1, 1.0, 10.0]
    assert cfg["seed"] == 100
    assert cfg["oversample_minority"] is True

    out_lopo = tmp_path / "out_lopo"
    summary_lopo = run_route(
        cache_dirs=[cache_dir],
        protocol="outer_lopo",
        output_dir=out_lopo,
        route={"name": "synthetic/outer_lopo"},
    )
    assert summary_lopo["aggregation"] == "pooled_subject_level_outer_lopo"
    assert summary_lopo["counts"]["total_subjects"] == 26
    assert summary_lopo["pooled_metrics"]["macro_f1"] >= 0.85
    assert sum(summary_lopo["c_selection_counts"].values()) == 26
    assert (out_lopo / "selection.json").exists()
    assert (out_lopo / "predictions_subject_level.jsonl").exists()


def test_per_fold_manyshot_cache_accepted_without_allow_adapter(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    d = 8
    train_items = [(f"tr_{i}", 1 if i < 4 else 0, rng.normal(2.0 if i < 4 else -2.0, 0.3, (1, d))) for i in range(10)]
    eval_items = [(f"ev_{i}", 1 if i < 2 else 0, rng.normal(2.0 if i < 2 else -2.0, 0.3, (1, d))) for i in range(4)]
    cache_dir = tmp_path / "ms_fold_0"
    _write_synthetic_cache(
        cache_dir,
        train_subjects=train_items,
        eval_subjects=eval_items,
        weights_source="base_model_manyshot_context",
    )
    split_path = tmp_path / "split_0.json"
    save_json(
        {
            "train_subject_ids": [s for s, _, _ in train_items],
            "selection_subject_ids": [],
            "final_eval_subject_ids": [s for s, _, _ in eval_items],
        },
        split_path,
    )
    summary = run_route(
        cache_dirs=[cache_dir],
        split_files=[split_path],
        protocol="train_cv_test",
        output_dir=tmp_path / "out_ms",
        per_fold_caches=True,
    )
    assert summary["feature_source"] == "base_model_manyshot_context"
    assert summary["fold_mean"]["macro_f1"] >= 0.85


def test_daic_traindev_test_fits_train_plus_dev_and_reports_test(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    d = 10
    train_items = [(f"tr_{i}", 1 if i < 4 else 0, rng.normal(2.0 if i < 4 else -2.0, 0.3, (1, d))) for i in range(10)]
    dev_items = [(f"dv_{i}", 1 if i < 2 else 0, rng.normal(2.0 if i < 2 else -2.0, 0.3, (1, d))) for i in range(4)]
    test_items = [(f"te_{i}", 1 if i < 2 else 0, rng.normal(2.0 if i < 2 else -2.0, 0.3, (1, d))) for i in range(4)]

    # Case 1: many-shot daic_traindev_test cache (dev already inside outer_train, no --dev-caches).
    ms_cache = tmp_path / "ms_traindev"
    _write_synthetic_cache(
        ms_cache,
        train_subjects=train_items + dev_items,
        eval_subjects=test_items,
        weights_source="base_model_manyshot_context",
    )
    split_path = tmp_path / "split.json"
    save_json(
        {
            "train_subject_ids": [s for s, _, _ in train_items],
            "selection_subject_ids": [s for s, _, _ in dev_items],
            "final_eval_subject_ids": [s for s, _, _ in test_items],
        },
        split_path,
    )
    summary_ms = run_route(
        cache_dirs=[ms_cache],
        split_files=[split_path],
        protocol="daic_traindev_test",
        output_dir=tmp_path / "out_ms_traindev",
        per_fold_caches=True,
    )
    assert summary_ms["folds"][0]["fit_subjects"] == 14
    assert summary_ms["folds"][0]["report_subjects"] == 4
    assert summary_ms["fold_mean"]["macro_f1"] >= 0.85

    # Case 2: zero-shot frozen DAIC cache with separate --dev-caches.
    zs_main = tmp_path / "zs_main"
    zs_dev = tmp_path / "zs_dev"
    _write_synthetic_cache(zs_main, train_subjects=train_items, eval_subjects=test_items)
    _write_synthetic_cache(zs_dev, train_subjects=train_items, eval_subjects=dev_items)
    summary_zs = run_route(
        cache_dirs=[zs_main],
        dev_caches=[zs_dev],
        split_files=[split_path],
        protocol="daic_traindev_test",
        output_dir=tmp_path / "out_zs_traindev",
        c_grid=(0.001,),
    )
    assert summary_zs["folds"][0]["fit_subjects"] == 14
    assert summary_zs["folds"][0]["report_subjects"] == 4
    assert summary_zs["folds"][0]["selected_c"] == 0.001
    assert read_json(tmp_path / "out_zs_traindev" / "result_config.json")["selection_method"] == "fixed_c"
    assert summary_zs["fold_mean"]["macro_f1"] >= 0.85

