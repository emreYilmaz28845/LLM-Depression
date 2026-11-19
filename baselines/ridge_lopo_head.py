"""Closed-form Ridge head with minority oversampling, train-only z-norm, and nested LOPO.

Implements the OZY-158 standalone head configuration as a separate module and
config (``configs/experiments/heads/ridge_lopo_ozy158_v1.yaml``), leaving the
repository's default L2 logistic head (``baselines/frozen_standalone_heads.py``)
untouched.

Recipe (OZY-158):
* Unit: participant (subject). Segments/windows are hierarchically mean-pooled
  per subject (windows within a unit, then units within the subject).
* Standardization: train-only z-norm fit on training-fold subject vectors.
* Balancing: minority-class oversampling with replacement within each training
  fold (deterministic RNG seeded from ``seed``, default 100).
* Classifier: closed-form ridge regression on ``{-1, +1}`` labels with
  ``lambda = 1 / C`` and an unpenalized intercept, solved exactly in the row
  span via SVD. Threshold is ``0.0`` (``score > 0`` -> Depressed).
* Hyperparameter selection: nested leave-one-participant-out (LOPO) over the
  training fold choosing ``C`` (default ``[0.001, 0.01, 0.1, 1.0, 10.0]``) by
  pooled inner macro-F1 (tie-breakers: lower MSE on ``±1``, then smaller ``C``).

Supported evaluation protocols:
* ``daic_official``, ``train_val_test``, ``train_val``, ``train_cv_test``:
  read split files and report per-fold subject-level metrics and fold means
  (also supports ``--per-fold-caches`` for many-shot caches).
* ``outer_lopo``: leave-one-participant-out across all subjects in the cache,
  running nested inner LOPO inside each outer fold and reporting pooled
  subject-level metrics across all held-out participants.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from baselines.frozen_standalone_heads import (
    HANDCRAFTED_BACKENDS,
    PooledCache,
    _check_adapter_partition,
    _load_caches,
    _write_csv,
)
from src.features.hidden_classifier_policy import cache_identity, canonical_sha256
from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, save_json, write_jsonl


SCHEMA_VERSION = "ridge_lopo_head.v1"
VARIANT = "ridge_closedform_oversample_znorm_lopo"
PER_FOLD_WEIGHTS_SOURCES = ("base_model_manyshot_context",)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/heads/ridge_lopo_ozy158_v1.yaml"
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_SEED = 100
DEFAULT_THRESHOLD = 0.0
MIN_STD = 1e-6
FINAL_FIT_ID = -1
SPLIT_PROTOCOLS = ("daic_official", "daic_traindev_test", "train_val_test", "train_val", "train_cv_test")
PROTOCOLS = (*SPLIT_PROTOCOLS, "outer_lopo")


def load_head_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML head config and validate required OZY-158 fields."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid head config YAML: {cfg_path}")
    c_grid = tuple(float(c) for c in payload["head"]["c_grid"])
    if not c_grid or any(c <= 0 or not np.isfinite(c) for c in c_grid):
        raise ValueError(f"c_grid must contain positive finite floats, got {c_grid!r}.")
    return {
        "config_path": str(cfg_path),
        "config_name": str(payload.get("name") or cfg_path.stem),
        "c_grid": c_grid,
        "seed": int(payload["balancing"]["seed"]),
        "oversample_minority": bool(payload["balancing"]["oversample_minority"]),
        "threshold": float(payload["head"].get("threshold", DEFAULT_THRESHOLD)),
        "min_std": float(payload["preprocessing"].get("min_std", MIN_STD)),
    }


def fit_subject_znorm(x: np.ndarray, min_std: float = MIN_STD) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-normalization (mean, std) on training-fold subject vectors only."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < min_std, 1.0, std)
    return mean, std


def oversample_minority(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with replacement to match the majority count."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes are required in the training fold.")
    if len(pos_idx) == len(neg_idx):
        return x, y
    minority = pos_idx if len(pos_idx) < len(neg_idx) else neg_idx
    deficit = abs(len(pos_idx) - len(neg_idx))
    extra = rng.choice(minority, size=deficit, replace=True)
    order = np.concatenate([np.arange(len(y)), extra])
    return x[order], y[order]


def fit_ridge_path(
    x: np.ndarray,
    y: np.ndarray,
    c_grid: tuple[float, ...],
) -> dict[float, tuple[np.ndarray, float]]:
    """Closed-form ridge on ±1 labels with lambda = 1/C and unpenalized intercept.

    Computes one thin SVD of the centered design matrix ``X_c = U S V^T`` and
    evaluates the exact ridge solution for every ``C`` in ``c_grid``:
        w(lambda) = V diag(s / (s^2 + lambda)) U^T t_c
        b(lambda) = t_bar - x_bar^T w(lambda)
    """
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError(f"Mismatched x shape {x.shape} and y length {len(y)}.")
    if set(np.asarray(y, dtype=np.int64).tolist()) != {0, 1}:
        raise ValueError("Both classes (0 and 1) are required to fit ridge.")
    t = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    x_mean = x.mean(axis=0)
    t_mean = float(t.mean())
    x_centered = x - x_mean
    t_centered = t - t_mean

    u, singular, vt = np.linalg.svd(x_centered, full_matrices=False)
    ut_t = u.T @ t_centered
    s2 = singular ** 2

    solutions: dict[float, tuple[np.ndarray, float]] = {}
    for c in c_grid:
        if c <= 0 or not np.isfinite(c):
            raise ValueError(f"C must be positive and finite, got {c!r}.")
        lam = 1.0 / float(c)
        coeff = (singular / (s2 + lam)) * ut_t
        w = vt.T @ coeff
        b = t_mean - float(x_mean @ w)
        solutions[float(c)] = (w, b)
    return solutions


def fit_ridge_single(x: np.ndarray, y: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge solution for a single C value."""
    return fit_ridge_path(x, y, (float(c),))[float(c)]


def score_ridge_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compute canonical headline metrics (macro-F1, positive-F1, UAR) + MSE/AUROC."""
    predictions = (scores > threshold).astype(int).tolist()
    metrics = classification_metrics(labels.tolist(), predictions)
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    targets = np.where(labels == 1, 1.0, -1.0).astype(np.float64)
    mse = float(np.mean((scores - targets) ** 2))
    metrics.update(
        {
            "negative_f1": (
                2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
                if (precision_neg + recall_neg)
                else 0.0
            ),
            "binary_strict_uar": float(metrics["macro_recall"]),
            "mse": mse,
            "auroc": binary_auroc(labels.tolist(), scores.tolist()),
        }
    )
    return metrics


def _fold_rng(seed: int, outer_fold: int, fit_id: int) -> np.random.Generator:
    # SeedSequence accepts non-negative integers; map FINAL_FIT_ID (-1) to a distinct tag.
    tag = 0 if fit_id >= 0 else 1
    idx = fit_id if fit_id >= 0 else 0
    return np.random.default_rng([int(seed), int(outer_fold), tag, int(idx)])


def _prepare_and_fit_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: tuple[float, ...],
    *,
    oversample: bool,
    rng: np.random.Generator,
    min_std: float = MIN_STD,
) -> tuple[np.ndarray, np.ndarray, dict[float, tuple[np.ndarray, float]]]:
    mean, std = fit_subject_znorm(x_train, min_std=min_std)
    z_train = (x_train - mean) / std
    if oversample:
        z_fit, y_fit = oversample_minority(z_train, y_train, rng)
    else:
        z_fit, y_fit = z_train, y_train
    solutions = fit_ridge_path(z_fit, y_fit, c_grid)
    return mean, std, solutions


def select_c_nested_lopo(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: tuple[float, ...],
    *,
    seed: int = DEFAULT_SEED,
    outer_fold: int = 0,
    oversample: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    min_std: float = MIN_STD,
) -> dict[str, Any]:
    """Select C by nested LOPO over the training subjects (inner metric: macro-F1)."""
    n = len(y_train)
    if n < 3:
        raise ValueError(f"Nested LOPO needs at least 3 training subjects, got {n}.")
    if int((y_train == 1).sum()) < 2 or int((y_train == 0).sum()) < 2:
        raise ValueError("Nested LOPO requires at least 2 subjects of each class in the training fold.")

    oof_scores: dict[float, np.ndarray] = {float(c): np.empty(n, dtype=np.float64) for c in c_grid}
    all_idx = np.arange(n)
    for inner_i in range(n):
        keep = all_idx != inner_i
        rng = _fold_rng(seed, outer_fold, inner_i)
        mean, std, solutions = _prepare_and_fit_path(
            x_train[keep], y_train[keep], c_grid, oversample=oversample, rng=rng, min_std=min_std
        )
        z_val = (x_train[inner_i] - mean) / std
        for c, (w, b) in solutions.items():
            oof_scores[c][inner_i] = float(z_val @ w + b)

    candidates: list[dict[str, Any]] = []
    for c in c_grid:
        cf = float(c)
        m = score_ridge_metrics(y_train, oof_scores[cf], threshold=threshold)
        candidates.append(
            {
                "c": cf,
                "lambda": 1.0 / cf,
                "macro_f1": float(m["macro_f1"]),
                "positive_f1": float(m["positive_f1"]),
                "uar": float(m["binary_strict_uar"]),
                "mse": float(m["mse"]),
                "auroc": m["auroc"],
            }
        )
    best = max(candidates, key=lambda item: (item["macro_f1"], -item["mse"], -item["c"]))
    return {
        "method": "nested_lopo",
        "num_inner_folds": n,
        "selected_c": best["c"],
        "selected_lambda": best["lambda"],
        "selected_at_min_c": best["c"] == min(float(c) for c in c_grid),
        "selected_at_max_c": best["c"] == max(float(c) for c in c_grid),
        "selected_inner_metrics": best,
        "candidates": candidates,
    }


def _prediction_rows(
    subjects: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    cache: PooledCache,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for subject, label, score in zip(subjects, labels.tolist(), scores.tolist()):
        units = cache.units[subject]
        rows.append(
            {
                "subject_id": subject,
                "label": int(label),
                "prediction": int(score > threshold),
                "score": float(score),
                "num_units": len(units),
                "num_samples": sum(len(u) for u in units.values()),
            }
        )
    return rows


def run_split_fold(
    cache: PooledCache,
    split: dict[str, Any],
    protocol: str,
    c_grid: tuple[float, ...],
    *,
    seed: int = DEFAULT_SEED,
    oversample: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    min_std: float = MIN_STD,
    fold: int = 0,
) -> dict[str, Any]:
    """Run one outer split fold with nested LOPO C-selection on the fit subjects."""
    fit_subjects = sorted(str(s) for s in split["train_subject_ids"])
    sel_subjects = sorted(str(s) for s in split.get("selection_subject_ids") or [])
    fin_subjects = sorted(str(s) for s in split.get("final_eval_subject_ids") or [])
    if protocol == "daic_traindev_test":
        if not sel_subjects or not fin_subjects:
            raise ValueError("daic_traindev_test requires non-empty selection (dev) and final_eval (test) subjects.")
        if set(fit_subjects) & set(sel_subjects):
            raise ValueError("Train and selection (dev) subjects overlap.")
        fit_subjects = sorted(set(fit_subjects) | set(sel_subjects))
    report_subjects = sel_subjects if protocol == "train_val" else fin_subjects
    if not fit_subjects or not report_subjects:
        raise ValueError(f"Empty fit or report split in protocol {protocol!r}.")
    if set(fit_subjects) & set(report_subjects):
        raise ValueError("Fit and report subjects overlap.")
    cache.require(fit_subjects, "fit")
    cache.require(report_subjects, "report")

    x_fit = cache.subject_means(cache.x, fit_subjects)
    y_fit = cache.y(fit_subjects)
    x_rep = cache.subject_means(cache.x, report_subjects)
    y_rep = cache.y(report_subjects)

    selection = select_c_nested_lopo(
        x_fit,
        y_fit,
        c_grid,
        seed=seed,
        outer_fold=fold,
        oversample=oversample,
        threshold=threshold,
        min_std=min_std,
    )
    best_c = selection["selected_c"]
    rng_final = _fold_rng(seed, fold, FINAL_FIT_ID)
    mean, std, solutions = _prepare_and_fit_path(
        x_fit, y_fit, (best_c,), oversample=oversample, rng=rng_final, min_std=min_std
    )
    w, b = solutions[best_c]
    scores = ((x_rep - mean) / std) @ w + b
    metrics = score_ridge_metrics(y_rep, scores, threshold=threshold)
    predictions = _prediction_rows(report_subjects, y_rep, scores, cache, threshold)
    return {
        "selection": selection,
        "metrics": metrics,
        "predictions": predictions,
        "fit_info": {
            "selected_c": best_c,
            "selected_lambda": 1.0 / best_c,
            "oversample_minority": oversample,
            "seed": seed,
            "weight_l2_norm": float(np.linalg.norm(w)),
            "bias": float(b),
        },
        "counts": {
            "fit_subjects": len(fit_subjects),
            "fit_positives": int((y_fit == 1).sum()),
            "report_subjects": len(report_subjects),
            "report_positives": int((y_rep == 1).sum()),
        },
    }


def run_outer_lopo(
    cache: PooledCache,
    c_grid: tuple[float, ...],
    *,
    seed: int = DEFAULT_SEED,
    oversample: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    min_std: float = MIN_STD,
) -> dict[str, Any]:
    """Run full outer LOPO across every subject in ``cache`` with nested inner LOPO."""
    subjects = sorted(cache.units.keys())
    x_all = cache.subject_means(cache.x, subjects)
    y_all = cache.y(subjects)
    n = len(subjects)
    if n < 5:
        raise ValueError(f"outer_lopo needs at least 5 subjects, got {n}.")

    scores = np.empty(n, dtype=np.float64)
    per_subject_selection: list[dict[str, Any]] = []
    all_idx = np.arange(n)
    for outer_i in range(n):
        keep = all_idx != outer_i
        sel = select_c_nested_lopo(
            x_all[keep],
            y_all[keep],
            c_grid,
            seed=seed,
            outer_fold=outer_i,
            oversample=oversample,
            threshold=threshold,
            min_std=min_std,
        )
        best_c = sel["selected_c"]
        rng_final = _fold_rng(seed, outer_i, FINAL_FIT_ID)
        mean, std, solutions = _prepare_and_fit_path(
            x_all[keep], y_all[keep], (best_c,), oversample=oversample, rng=rng_final, min_std=min_std
        )
        w, b = solutions[best_c]
        scores[outer_i] = float(((x_all[outer_i] - mean) / std) @ w + b)
        per_subject_selection.append(
            {
                "subject_id": subjects[outer_i],
                "selected_c": best_c,
                "selected_inner_macro_f1": sel["selected_inner_metrics"]["macro_f1"],
                "selected_at_min_c": sel["selected_at_min_c"],
                "selected_at_max_c": sel["selected_at_max_c"],
            }
        )

    metrics = score_ridge_metrics(y_all, scores, threshold=threshold)
    predictions = _prediction_rows(subjects, y_all, scores, cache, threshold)
    c_counts = {
        str(float(c)): sum(1 for item in per_subject_selection if item["selected_c"] == float(c))
        for c in c_grid
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "selection_summary": {
            "method": "outer_lopo_with_nested_inner_lopo",
            "num_outer_folds": n,
            "c_selection_counts": c_counts,
            "selected_at_min_c_count": sum(1 for item in per_subject_selection if item["selected_at_min_c"]),
            "selected_at_max_c_count": sum(1 for item in per_subject_selection if item["selected_at_max_c"]),
            "per_subject": per_subject_selection,
        },
        "counts": {
            "total_subjects": n,
            "total_positives": int((y_all == 1).sum()),
        },
    }


def run_route(
    *,
    cache_dirs: list[Path],
    output_dir: Path,
    protocol: str,
    split_files: list[Path] | None = None,
    dev_caches: list[Path] | None = None,
    route: dict[str, Any] | None = None,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    seed: int = DEFAULT_SEED,
    oversample: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    min_std: float = MIN_STD,
    allow_adapter: bool = False,
    per_fold_caches: bool = False,
    config_name: str = "ridge_lopo_ozy158_v1",
) -> dict[str, Any]:
    """Run the OZY-158 ridge head on one route and write full provenance artifacts."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol {protocol!r}; expected one of {PROTOCOLS}.")
    route = route or {}
    split_files = split_files or []
    dev_caches = dev_caches or []

    if protocol == "outer_lopo":
        if len(cache_dirs) != 1:
            raise ValueError("outer_lopo expects exactly one cache directory.")
        if split_files:
            raise ValueError("outer_lopo does not take --split-files.")
    else:
        if not split_files:
            raise ValueError(f"Protocol {protocol!r} requires --split-files.")
        if protocol == "daic_official" and not dev_caches:
            raise ValueError("A dev cache is required for daic_official.")
        if protocol not in {"daic_official", "daic_traindev_test"} and dev_caches:
            raise ValueError("A dev cache is only allowed for daic_official or daic_traindev_test.")
        if len(cache_dirs) not in {1, len(split_files)}:
            raise ValueError("Give one cache for all folds or one cache per split file.")
        if (allow_adapter or per_fold_caches) and len(cache_dirs) != len(split_files):
            raise ValueError("Fold-specific caches require one --cache-dirs entry per split file.")
        if dev_caches and len(dev_caches) != len(cache_dirs):
            raise ValueError("Give one dev cache per main cache.")

    loaded = []
    for i, c_dir in enumerate(cache_dirs):
        dev_c = dev_caches[i] if dev_caches else None
        # Allow many-shot per-fold caches without requiring --allow-adapter.
        meta_path = c_dir / "extraction_metadata.json"
        is_per_fold_frozen = False
        if per_fold_caches and meta_path.exists():
            meta_preview = read_json(meta_path)
            is_per_fold_frozen = meta_preview.get("weights_source") in PER_FOLD_WEIGHTS_SOURCES
        loaded.append(_load_caches(c_dir, dev_c, allow_adapter or is_per_fold_frozen))

    meta = loaded[0][1]
    feature_source = (
        "lora_adapter"
        if allow_adapter
        else "handcrafted"
        if meta.get("model_backend") in HANDCRAFTED_BACKENDS
        else str(meta.get("weights_source") or "frozen_base")
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing output: {output_dir}")
    final_dir = output_dir
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise ValueError(f"A partial output from an earlier failed run exists: {partial_dir}")
    partial_dir.mkdir(parents=True)

    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "variant": VARIANT,
        "config_name": config_name,
        "protocol": protocol,
        "route": route,
        "c_grid": [float(c) for c in c_grid],
        "lambda_grid": [1.0 / float(c) for c in c_grid],
        "seed": int(seed),
        "oversample_minority": bool(oversample),
        "threshold": float(threshold),
        "min_std": float(min_std),
        "znorm": "train_only_subject_level",
        "selection_method": "nested_lopo",
        "feature_source": feature_source,
        "cache_identities": [cache_identity(c) for c in cache_dirs],
        "dev_cache_identities": [cache_identity(c) for c in dev_caches] if dev_caches else None,
        "split_files": (
            [{"path": str(p), "sha256": canonical_sha256(read_json(p))} for p in split_files]
            if split_files
            else None
        ),
        "base_model": meta.get("base_model"),
        "model_backend": meta.get("model_backend"),
        "manifest_sha256": meta.get("manifest_sha256"),
    }
    identity["config_sha256"] = canonical_sha256(identity)
    save_json(identity, partial_dir / "result_config.json")

    if protocol == "outer_lopo":
        result = run_outer_lopo(
            loaded[0][0],
            c_grid,
            seed=seed,
            oversample=oversample,
            threshold=threshold,
            min_std=min_std,
        )
        save_json(result["selection_summary"], partial_dir / "selection.json")
        save_json({**result["metrics"], "counts": result["counts"]}, partial_dir / "metrics.json")
        write_jsonl(result["predictions"], partial_dir / "predictions_subject_level.jsonl")
        _write_csv(result["predictions"], partial_dir / "predictions_subject_level.csv")
        m = result["metrics"]
        summary = {
            "route": route,
            "protocol": protocol,
            "variant": VARIANT,
            "config_name": config_name,
            "aggregation": "pooled_subject_level_outer_lopo",
            "pooled_metrics": {
                "macro_f1": float(m["macro_f1"]),
                "positive_f1": float(m["positive_f1"]),
                "uar": float(m["binary_strict_uar"]),
                "auroc": m["auroc"],
                "confusion_matrix": m["confusion_matrix"],
            },
            "c_selection_counts": result["selection_summary"]["c_selection_counts"],
            "selected_at_min_c_count": result["selection_summary"]["selected_at_min_c_count"],
            "selected_at_max_c_count": result["selection_summary"]["selected_at_max_c_count"],
            "counts": result["counts"],
            "feature_source": feature_source,
            "num_subjects_in_cache": len(loaded[0][0].units),
        }
    else:
        folds = []
        for fold, path in enumerate(split_files):
            split = read_json(path)
            cache = loaded[fold if len(loaded) > 1 else 0][0]
            if allow_adapter or per_fold_caches:
                _check_adapter_partition(cache_dirs[fold], split, protocol)
            result = run_split_fold(
                cache,
                split,
                protocol,
                c_grid,
                seed=seed,
                oversample=oversample,
                threshold=threshold,
                min_std=min_std,
                fold=fold,
            )
            fold_dir = partial_dir / f"fold_{fold}"
            fold_dir.mkdir()
            save_json(result["selection"], fold_dir / "selection.json")
            save_json(
                {**result["metrics"], "fit_info": result["fit_info"], "counts": result["counts"]},
                fold_dir / "metrics.json",
            )
            write_jsonl(result["predictions"], fold_dir / "predictions_subject_level.jsonl")
            _write_csv(result["predictions"], fold_dir / "predictions_subject_level.csv")
            m = result["metrics"]
            folds.append(
                {
                    "fold": fold,
                    "selected_c": result["selection"]["selected_c"],
                    "selected_at_min_c": result["selection"]["selected_at_min_c"],
                    "selected_at_max_c": result["selection"]["selected_at_max_c"],
                    "macro_f1": float(m["macro_f1"]),
                    "positive_f1": float(m["positive_f1"]),
                    "uar": float(m["binary_strict_uar"]),
                    "auroc": m["auroc"],
                    "confusion_matrix": m["confusion_matrix"],
                    **result["counts"],
                }
            )
        summary = {
            "route": route,
            "protocol": protocol,
            "variant": VARIANT,
            "config_name": config_name,
            "folds": folds,
            "aggregation": "unweighted_fold_mean_of_subject_level_metrics",
            "fold_mean": {
                k: float(np.mean([f[k] for f in folds])) for k in ("macro_f1", "positive_f1", "uar")
            },
            "feature_source": feature_source,
            "num_subjects_in_cache": max(len(c.units) for c, _ in loaded),
        }

    save_json(summary, partial_dir / "summary.json")
    if final_dir.exists():
        final_dir.rmdir()
    partial_dir.rename(final_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML head config file.")
    parser.add_argument("--cache-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--dev-caches", nargs="+", type=Path, help="DAIC only: one dev cache per main cache.")
    parser.add_argument("--split-files", nargs="+", type=Path, help="Required for split-based protocols.")
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", default="{}", help="JSON object recorded in the output identity.")
    parser.add_argument("--c-grid", type=float, nargs="+", help="Override c_grid from --config.")
    parser.add_argument("--seed", type=int, help="Override oversampling seed from --config.")
    parser.add_argument("--no-oversample", action="store_true", help="Disable minority oversampling.")
    parser.add_argument("--allow-adapter", action="store_true", help="Accept per-fold LoRA caches.")
    parser.add_argument("--per-fold-caches", action="store_true", help="Accept per-fold frozen/many-shot caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_head_config(args.config)
    c_grid = tuple(float(c) for c in args.c_grid) if args.c_grid else cfg["c_grid"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    oversample = False if args.no_oversample else cfg["oversample_minority"]
    summary = run_route(
        cache_dirs=args.cache_dirs,
        dev_caches=args.dev_caches,
        split_files=args.split_files,
        protocol=args.protocol,
        output_dir=args.output_dir,
        route=json.loads(args.route),
        c_grid=c_grid,
        seed=seed,
        oversample=oversample,
        threshold=cfg["threshold"],
        min_std=cfg["min_std"],
        allow_adapter=args.allow_adapter,
        per_fold_caches=args.per_fold_caches,
        config_name=cfg["config_name"],
    )
    key = "pooled_metrics" if args.protocol == "outer_lopo" else "fold_mean"
    print(json.dumps({"route": summary["route"], key: summary[key]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
