"""Subject-mean L2 logistic head for frozen-base standalone caches, all folds.

A frozen model gives every subject the same vectors in every fold, so one
extraction covers all folds: the fold-0 cache (``outer_train`` plus
``final_eval``) holds every subject of the dataset, and DAIC adds its official
dev subjects from a second cache. This script splits that pooled cache with
the saved ``split_used.json`` of each fold and fits the head per fold.

Head (same as ``baselines/subject_mean_logreg_head.py``):

* train-only scaler with equal weight per subject (and per unit inside it);
* one fit row per training subject: hierarchical mean of standardized vectors
  (windows within a unit, then units within the subject; the unit is
  ``response_id``, else ``question_condition``, else the sample itself);
* balanced class weights on training-subject counts; L2 grid 1e-4..1e3;
* subject score = hierarchical mean of window logits (= logit of the
  hierarchical mean vector, because the head is linear); threshold 0.5.

Optional PCA (``--pca-components K`` or ``--pca-select-grid K1 K2 ...``): after
the scaler, PCA is fit on the standardized window vectors of the fit subjects
with the scaler's weights (equal mass per subject, and per unit inside it), no
whitening; subject means are projected on the top K components. PCA is refit on
every fit set (each inner-CV fold and the final fit), so selection and reported
subjects never enter it. With a select grid, K is chosen with L2 on the same
inner validation. Without these options the head is unchanged.

With ``--allow-adapter`` the same head runs on LoRA caches instead. A LoRA
adapter differs per fold, so each fold then takes its own cache (one
``--cache-dirs`` entry per split file), and the cache's ``final_eval``
subjects must equal that fold's reported subjects.

L2 selection by protocol (the reported subjects never take part):

* ``daic_official``: fit on official train, select on official dev, report test;
* ``train_val_test``: fit on ``train_subject_ids``, select on the saved inner
  validation (``selection_subject_ids``), report ``final_eval_subject_ids``;
* ``train_val``: fit on ``train_subject_ids``, select by stratified inner
  5-fold CV on those subjects (pooled out-of-fold logits, seed 1337), refit on
  all of them, report ``selection_subject_ids`` (the outer validation fold);
* ``train_cv_test``: fit on ``train_subject_ids``, select by the same inner CV,
  report ``final_eval_subject_ids``; ``selection_subject_ids`` stay unused (DAIC
  LoRA caches have no dev partition, so both sides use this for a like-for-like
  comparison).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from baselines.subject_mean_logreg_head import (
    DEFAULT_L2_GRID,
    balanced_weights,
    fit_logistic,
    load_partition,
    score_metrics,
)
from src.features.hidden_classifier_policy import cache_identity, canonical_sha256
from src.utils import read_json, save_json, write_jsonl


SCHEMA_VERSION = "frozen_standalone_heads.v2"
VARIANT = "logreg_subjmean_l2sel"
PCA_VARIANT = "logreg_subjmean_pca_l2sel"
PROTOCOLS = ("daic_official", "train_val_test", "train_val", "train_cv_test")
INNER_CV_PROTOCOLS = ("train_val", "train_cv_test")
INNER_FOLDS = 5
INNER_SEED = 1337
HANDCRAFTED_BACKENDS = ("opensmile_mfcc",)


def unit_key(row: dict[str, Any]) -> str:
    for key in ("response_id", "question_condition"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(row["sample_id"])


class PooledCache:
    """All rows of one route, indexed by subject and unit."""

    def __init__(self, caches: list[tuple[Path, str]]):
        vectors, rows = [], []
        for cache_dir, partition in caches:
            x, r = load_partition(cache_dir, partition)
            vectors.append(x)
            rows.extend(r)
        self.x = np.concatenate(vectors)
        self.rows = rows
        if len({str(row["sample_id"]) for row in rows}) != len(rows):
            raise ValueError("A sample appears in more than one pooled partition.")
        self.units: dict[str, OrderedDict[str, list[int]]] = {}
        self.labels: dict[str, int] = {}
        for index, row in enumerate(rows):
            subject = str(row["subject_id"])
            label = int(row["label"])
            if self.labels.setdefault(subject, label) != label:
                raise ValueError(f"Subject {subject} has conflicting labels.")
            self.units.setdefault(subject, OrderedDict()).setdefault(unit_key(row), []).append(index)

    def require(self, subjects: list[str], what: str) -> None:
        missing = sorted(set(subjects) - set(self.units))
        if missing:
            raise ValueError(f"{what}: {len(missing)} subjects missing from cache, e.g. {missing[:5]}")

    def row_weights(self, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Row indices and weights giving each subject (and unit within it) equal mass."""
        index, weight = [], []
        for subject in subjects:
            units = self.units[subject]
            for rows in units.values():
                index.extend(rows)
                weight.extend([1.0 / (len(units) * len(rows))] * len(rows))
        return np.asarray(index), np.asarray(weight)

    def subject_means(self, z: np.ndarray, subjects: list[str]) -> np.ndarray:
        return np.stack(
            [np.mean([z[rows].mean(0) for rows in self.units[s].values()], axis=0) for s in subjects]
        )

    def y(self, subjects: list[str]) -> np.ndarray:
        return np.asarray([self.labels[s] for s in subjects], dtype=np.int64)


def fit_scaler(cache: PooledCache, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index, weight = cache.row_weights(subjects)
    w = weight / weight.sum()
    x = cache.x[index]
    mean = w @ x
    scale = np.sqrt(w @ (x - mean) ** 2)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # Ties: lower BCE, then stronger L2, then fewer PCA components.
    return max(candidates, key=lambda c: (c["macro_f1"], -c["binary_cross_entropy"], c["l2_penalty"],
                                          -(c.get("pca_components") or 0)))


def fit_pca(cache: PooledCache, subjects: list[str], mean: np.ndarray, scale: np.ndarray,
            max_components: int) -> dict[str, Any]:
    """Weighted PCA of standardized window vectors (scaler weights), no whitening."""
    index, weight = cache.row_weights(subjects)
    w = weight / weight.sum()
    z = (cache.x[index] - mean) / scale
    center = w @ z
    _, singular, vt = np.linalg.svd(np.sqrt(w)[:, None] * (z - center), full_matrices=False)
    if max_components > vt.shape[0]:
        raise ValueError(f"PCA needs {max_components} components but the fit set has rank <= {vt.shape[0]}.")
    variance = singular ** 2
    return {"center": center, "components": vt[:max_components].T,
            "cumulative_explained": np.cumsum(variance[:max_components]) / variance.sum()}


def _prepare(cache: PooledCache, subjects: list[str], max_components: int | None) -> dict[str, Any]:
    """Everything fit on the fit subjects before the classifier: scaler and (optional) PCA."""
    mean, scale = fit_scaler(cache, subjects)
    pca = fit_pca(cache, subjects, mean, scale, max_components) if max_components else None
    return {"mean": mean, "scale": scale, "pca": pca}


def _features(cache: PooledCache, prep: dict[str, Any], subjects: list[str], k: int | None) -> np.ndarray:
    x = cache.subject_means((cache.x - prep["mean"]) / prep["scale"], subjects)
    if k:
        pca = prep["pca"]
        x = (x - pca["center"]) @ pca["components"][:, :k]
    return x


def pca_diagnostics(cache: PooledCache, prep: dict[str, Any], subjects: list[str], k: int) -> dict[str, Any]:
    """Explained variance of the top k components and the between-subject share of variance.

    With the scaler weights every subject has equal mass, so total weighted
    variance = between-subject variance (of hierarchical subject means) + within.
    """
    index, weight = cache.row_weights(subjects)
    w = weight / weight.sum()
    pca = prep["pca"]
    z = (cache.x[index] - prep["mean"]) / prep["scale"] - pca["center"]
    means = cache.subject_means((cache.x - prep["mean"]) / prep["scale"], subjects) - pca["center"]

    def share(rows: np.ndarray, subject_means: np.ndarray) -> float:
        total = float(w @ (rows ** 2).sum(1))
        centered = subject_means - subject_means.mean(0)
        return float((centered ** 2).sum(1).mean() / total)

    p = pca["components"][:, :k]
    return {"pca_components": k, "explained_variance_ratio": float(pca["cumulative_explained"][k - 1]),
            "between_subject_share_in_subspace": share(z @ p, means @ p),
            "between_subject_share_full_space": share(z, means)}


def _fit(cache: PooledCache, subjects: list[str], penalty: float, k: int | None = None,
         prep: dict[str, Any] | None = None):
    prep = prep if prep is not None else _prepare(cache, subjects, k)
    x = _features(cache, prep, subjects, k)
    y = cache.y(subjects)
    weight, bias, info = fit_logistic(x, y, balanced_weights(y), penalty)
    return prep, k, weight, bias, info


def _logits(cache: PooledCache, head, subjects: list[str]) -> np.ndarray:
    prep, k, weight, bias, _ = head
    return _features(cache, prep, subjects, k) @ weight + bias


def inner_cv_folds(subjects: list[str], y: np.ndarray) -> list[list[str]]:
    """Stratified inner folds: shuffle each class with seed 1337, deal round-robin."""
    rng = np.random.default_rng(INNER_SEED)
    folds: list[list[str]] = [[] for _ in range(INNER_FOLDS)]
    position = 0
    for label in (1, 0):
        members = [s for s, v in zip(subjects, y.tolist()) if v == label]
        for subject in rng.permutation(members).tolist():
            folds[position % INNER_FOLDS].append(subject)
            position += 1
    return folds


def run_fold(cache: PooledCache, split: dict[str, Any], protocol: str, grid,
             pca_grid: tuple[int, ...] | None = None) -> dict[str, Any]:
    train = sorted(str(s) for s in split["train_subject_ids"])
    selection = sorted(str(s) for s in split.get("selection_subject_ids", []))
    final = sorted(str(s) for s in split.get("final_eval_subject_ids", []))
    if protocol == "train_val":
        report, select_on = selection, None
        if final:
            raise ValueError("train_val split unexpectedly has final_eval subjects.")
    elif protocol == "train_cv_test":
        report, select_on = final, None
    else:
        report, select_on = final, selection
    if not report or (protocol not in INNER_CV_PROTOCOLS and not select_on):
        raise ValueError(f"{protocol}: empty selection or report subjects.")
    groups = [set(train), set(select_on or []), set(report)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Train, selection and report subjects overlap.")
    cache.require(train + (select_on or []) + report, protocol)

    ks = tuple(int(k) for k in pca_grid) if pca_grid else (None,)
    max_k = max(ks) if pca_grid else None
    configs = [(k, float(penalty)) for k in ks for penalty in grid]

    def candidate(k, penalty, m, extra=None):
        out = {"l2_penalty": penalty, "macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"],
               "binary_cross_entropy": m["binary_cross_entropy"], **(extra or {})}
        return {"pca_components": k, **out} if k else out

    candidates = []
    inner = None
    if protocol in INNER_CV_PROTOCOLS:
        y_train = cache.y(train)
        inner = inner_cv_folds(train, y_train)
        oof: dict[tuple, dict[str, float]] = {c: {} for c in configs}
        for held in inner:
            fit_subjects = [s for s in train if s not in set(held)]
            prep = _prepare(cache, fit_subjects, max_k)
            for k, penalty in configs:
                head = _fit(cache, fit_subjects, penalty, k, prep)
                oof[(k, penalty)].update(zip(held, _logits(cache, head, held).tolist()))
        for k, penalty in configs:
            m = score_metrics(y_train, np.asarray([oof[(k, penalty)][s] for s in train]))
            candidates.append(candidate(k, penalty, m))
        best = _select(candidates)
        prep = _prepare(cache, train, max_k)
        head = _fit(cache, train, best["l2_penalty"], best.get("pca_components"), prep)
    else:
        heads = {}
        y_sel = cache.y(select_on)
        prep = _prepare(cache, train, max_k)
        for k, penalty in configs:
            heads[(k, penalty)] = _fit(cache, train, penalty, k, prep)
            m = score_metrics(y_sel, _logits(cache, heads[(k, penalty)], select_on))
            candidates.append(candidate(k, penalty, m, heads[(k, penalty)][4]))
        best = _select(candidates)
        head = heads[(best.get("pca_components"), best["l2_penalty"])]
    fit_info = dict(head[4])
    if pca_grid:
        fit_info["pca"] = pca_diagnostics(cache, prep, train, best["pca_components"])
    logits = _logits(cache, head, report)
    metrics = score_metrics(cache.y(report), logits)
    predictions = [
        {"subject_id": s, "label": int(cache.labels[s]), "prediction": int(z > 0),
         "probability": float(1.0 / (1.0 + math.exp(-z))), "mean_logit": float(z),
         "num_units": len(cache.units[s]), "num_samples": sum(len(v) for v in cache.units[s].values())}
        for s, z in zip(report, logits.tolist())
    ]
    selection = {"selected_l2_penalty": best["l2_penalty"]}
    if pca_grid:
        selection.update({"selected_pca_components": best["pca_components"], "pca_grid": list(ks)})
    return {
        "selection": {**selection, "candidates": candidates,
                      "selection_source": "inner_5fold_cv_stratified_seed1337" if inner else
                      ("official_dev" if protocol == "daic_official" else "saved_inner_validation"),
                      "inner_folds": inner},
        "fit_info": fit_info,
        "counts": {"train": len(train), "selection": len(select_on or []), "report": len(report)},
        "metrics": metrics,
        "predictions": predictions,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_fixed_feature_cache(meta: dict[str, Any]) -> bool:
    """Features that do not change with the fold: a frozen base model or handcrafted openSMILE MFCC."""
    return meta.get("weights_source") == "base_model_no_adapter" or meta.get("model_backend") in HANDCRAFTED_BACKENDS


def _load_caches(cache_dir: Path, dev_cache: Path | None, allow_adapter: bool) -> tuple[PooledCache, dict]:
    meta = read_json(cache_dir / "extraction_metadata.json")
    if not allow_adapter and not is_fixed_feature_cache(meta):
        raise ValueError("Cache is not a frozen base-model (no adapter) or handcrafted-feature extraction.")
    if allow_adapter and is_fixed_feature_cache(meta):
        raise ValueError("--allow-adapter was given but the cache is a frozen base or handcrafted extraction.")
    parts = [(cache_dir, "outer_train"), (cache_dir, "final_eval")]
    if dev_cache is not None:
        dev_meta = read_json(dev_cache / "extraction_metadata.json")
        keys = ("base_model", "weights_source", "manifest_sha256", "condition", "input_modality", "pooling")
        for key in keys + (("checkpoint_dir", "adapter_sha256") if allow_adapter else ()):
            if dev_meta.get(key) != meta.get(key):
                raise ValueError(f"Dev cache differs from main cache in {key}.")
        parts.append((dev_cache, "final_eval"))
    return PooledCache(parts), meta


def _check_adapter_partition(cache_dir: Path, split: dict[str, Any], protocol: str) -> None:
    """A per-fold LoRA cache must hold exactly that fold's reported subjects in final_eval."""
    _, rows = load_partition(cache_dir, "final_eval")
    held = {str(r["subject_id"]) for r in rows}
    key = "selection_subject_ids" if protocol == "train_val" else "final_eval_subject_ids"
    if held != {str(s) for s in split.get(key, [])}:
        raise ValueError(f"{cache_dir}: final_eval subjects do not match the fold's {key}.")


def run_route(
    *, cache_dirs: list[Path], dev_caches: list[Path] | None, split_files: list[Path], protocol: str,
    output_dir: Path, route: dict[str, Any], grid=DEFAULT_L2_GRID, allow_adapter: bool = False,
    pca_grid: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}.")
    if pca_grid is not None and (not pca_grid or min(pca_grid) < 1):
        raise ValueError("PCA components must be positive integers.")
    variant = PCA_VARIANT if pca_grid else VARIANT
    if (protocol == "daic_official") != bool(dev_caches):
        raise ValueError("A dev cache is required for, and only for, daic_official.")
    if len(cache_dirs) not in {1, len(split_files)}:
        raise ValueError("Give one cache for all folds or one cache per split file.")
    if allow_adapter and len(cache_dirs) != len(split_files):
        raise ValueError("Adapter caches are fold-specific: give one cache per split file.")
    if dev_caches and len(dev_caches) != len(cache_dirs):
        raise ValueError("Give one dev cache per main cache.")
    loaded = [_load_caches(c, dev_caches[i] if dev_caches else None, allow_adapter)
              for i, c in enumerate(cache_dirs)]
    meta = loaded[0][1]
    feature_source = ("lora_adapter" if allow_adapter else
                      "handcrafted" if meta.get("model_backend") in HANDCRAFTED_BACKENDS else "frozen_base")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing output: {output_dir}")
    final_dir = output_dir
    output_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists():
        raise ValueError(f"A partial output from an earlier failed run exists: {output_dir}")
    output_dir.mkdir(parents=True)
    identity = {
        "schema_version": SCHEMA_VERSION, "variant": variant, "protocol": protocol, "route": route,
        "l2_grid": list(grid), "inner_folds": INNER_FOLDS, "inner_seed": INNER_SEED, "threshold": 0.5,
        "feature_source": feature_source,
        "cache_identities": [cache_identity(c) for c in cache_dirs],
        "dev_cache_identities": [cache_identity(c) for c in dev_caches] if dev_caches else None,
        "checkpoints": [m.get("checkpoint_dir") for _, m in loaded] if allow_adapter else None,
        "split_files": [{"path": str(p), "sha256": canonical_sha256(read_json(p))} for p in split_files],
        "base_model": meta.get("base_model"), "model_backend": meta.get("model_backend"),
        "manifest_sha256": meta.get("manifest_sha256"),
    }
    if pca_grid:
        identity["pca"] = {"components_grid": sorted(set(int(k) for k in pca_grid)), "whiten": False,
                           "fit_rows": "standardized window vectors of the fit subjects, scaler weights "
                                       "(equal mass per subject and per unit within it)",
                           "selection": "fixed" if len(set(pca_grid)) == 1 else "inner_validation_with_l2",
                           "refit": "every fit set (each inner-CV fold and the final fit)"}
    identity["config_sha256"] = canonical_sha256(identity)
    save_json(identity, output_dir / "result_config.json")
    folds = []
    for fold, path in enumerate(split_files):
        split = read_json(path)
        cache = loaded[fold if len(loaded) > 1 else 0][0]
        if allow_adapter:
            _check_adapter_partition(cache_dirs[fold], split, protocol)
        result = run_fold(cache, split, protocol, grid, tuple(sorted(set(pca_grid))) if pca_grid else None)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir()
        save_json(result["selection"], fold_dir / "selection.json")
        save_json({**result["metrics"], "fit_info": result["fit_info"], "counts": result["counts"]},
                  fold_dir / "metrics.json")
        write_jsonl(result["predictions"], fold_dir / "predictions_subject_level.jsonl")
        _write_csv(result["predictions"], fold_dir / "predictions_subject_level.csv")
        m = result["metrics"]
        extra = {}
        if pca_grid:
            extra = {"selected_pca_components": result["selection"]["selected_pca_components"],
                     "explained_variance_ratio": result["fit_info"]["pca"]["explained_variance_ratio"],
                     "between_subject_share_in_subspace":
                         result["fit_info"]["pca"]["between_subject_share_in_subspace"],
                     "between_subject_share_full_space":
                         result["fit_info"]["pca"]["between_subject_share_full_space"]}
        folds.append({"fold": fold, "selected_l2_penalty": result["selection"]["selected_l2_penalty"], **extra,
                      "macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"],
                      "uar": m["binary_strict_uar"], "auroc": m["auroc"],
                      "confusion_matrix": m["confusion_matrix"], **result["counts"]})
    summary = {
        "route": route, "protocol": protocol, "variant": variant, "folds": folds,
        "aggregation": "unweighted_fold_mean_of_subject_level_metrics",
        "fold_mean": {k: float(np.mean([f[k] for f in folds])) for k in ("macro_f1", "positive_f1", "uar")},
        "feature_source": identity["feature_source"],
        "num_subjects_in_cache": max(len(c.units) for c, _ in loaded),
    }
    save_json(summary, output_dir / "summary.json")
    if final_dir.exists():
        final_dir.rmdir()  # empty, checked above
    output_dir.rename(final_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dirs", required=True, nargs="+", type=Path,
                        help="One cache for all folds (frozen base) or one per split file.")
    parser.add_argument("--dev-caches", nargs="+", type=Path, help="DAIC only: one dev cache per main cache.")
    parser.add_argument("--allow-adapter", action="store_true",
                        help="Accept per-fold LoRA caches instead of a frozen base cache.")
    parser.add_argument("--split-files", required=True, nargs="+", type=Path)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", default="{}", help="JSON object recorded in the output identity.")
    pca = parser.add_mutually_exclusive_group()
    pca.add_argument("--pca-components", type=int, help="Fixed number of PCA components before the head.")
    pca.add_argument("--pca-select-grid", type=int, nargs="+",
                     help="Choose the number of PCA components with L2 on the inner validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_route(cache_dirs=args.cache_dirs, dev_caches=args.dev_caches, split_files=args.split_files,
                        protocol=args.protocol, output_dir=args.output_dir, route=json.loads(args.route),
                        allow_adapter=args.allow_adapter,
                        pca_grid=(args.pca_components,) if args.pca_components else
                        (tuple(args.pca_select_grid) if args.pca_select_grid else None))
    print(json.dumps({"route": summary["route"], "fold_mean": summary["fold_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
