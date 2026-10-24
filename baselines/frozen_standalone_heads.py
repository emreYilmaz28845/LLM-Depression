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

L2 selection by protocol (the reported subjects never take part):

* ``daic_official``: fit on official train, select on official dev, report test;
* ``train_val_test``: fit on ``train_subject_ids``, select on the saved inner
  validation (``selection_subject_ids``), report ``final_eval_subject_ids``;
* ``train_val``: fit on ``train_subject_ids``, select by stratified inner
  5-fold CV on those subjects (pooled out-of-fold logits, seed 1337), refit on
  all of them, report ``selection_subject_ids`` (the outer validation fold).
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


SCHEMA_VERSION = "frozen_standalone_heads.v1"
VARIANT = "logreg_subjmean_l2sel"
PROTOCOLS = ("daic_official", "train_val_test", "train_val")
INNER_FOLDS = 5
INNER_SEED = 1337


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
    return max(candidates, key=lambda c: (c["macro_f1"], -c["binary_cross_entropy"], c["l2_penalty"]))


def _fit(cache: PooledCache, subjects: list[str], penalty: float):
    mean, scale = fit_scaler(cache, subjects)
    x = cache.subject_means((cache.x - mean) / scale, subjects)
    y = cache.y(subjects)
    weight, bias, info = fit_logistic(x, y, balanced_weights(y), penalty)
    return mean, scale, weight, bias, info


def _logits(cache: PooledCache, head, subjects: list[str]) -> np.ndarray:
    mean, scale, weight, bias, _ = head
    return cache.subject_means((cache.x - mean) / scale, subjects) @ weight + bias


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


def run_fold(cache: PooledCache, split: dict[str, Any], protocol: str, grid) -> dict[str, Any]:
    train = sorted(str(s) for s in split["train_subject_ids"])
    selection = sorted(str(s) for s in split.get("selection_subject_ids", []))
    final = sorted(str(s) for s in split.get("final_eval_subject_ids", []))
    if protocol == "train_val":
        report, select_on = selection, None
        if final:
            raise ValueError("train_val split unexpectedly has final_eval subjects.")
    else:
        report, select_on = final, selection
    if not report or (protocol != "train_val" and not select_on):
        raise ValueError(f"{protocol}: empty selection or report subjects.")
    groups = [set(train), set(select_on or []), set(report)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Train, selection and report subjects overlap.")
    cache.require(train + (select_on or []) + report, protocol)

    candidates = []
    inner = None
    if protocol == "train_val":
        y_train = cache.y(train)
        inner = inner_cv_folds(train, y_train)
        for penalty in grid:
            oof: dict[str, float] = {}
            for held in inner:
                fit_subjects = [s for s in train if s not in set(held)]
                head = _fit(cache, fit_subjects, float(penalty))
                oof.update(zip(held, _logits(cache, head, held).tolist()))
            logits = np.asarray([oof[s] for s in train])
            m = score_metrics(y_train, logits)
            candidates.append({"l2_penalty": float(penalty), "macro_f1": m["macro_f1"],
                               "positive_f1": m["positive_f1"], "binary_cross_entropy": m["binary_cross_entropy"]})
        best = _select(candidates)
        head = _fit(cache, train, best["l2_penalty"])
    else:
        heads = {}
        y_sel = cache.y(select_on)
        for penalty in grid:
            heads[float(penalty)] = _fit(cache, train, float(penalty))
            m = score_metrics(y_sel, _logits(cache, heads[float(penalty)], select_on))
            candidates.append({"l2_penalty": float(penalty), "macro_f1": m["macro_f1"],
                               "positive_f1": m["positive_f1"], "binary_cross_entropy": m["binary_cross_entropy"],
                               **heads[float(penalty)][4]})
        best = _select(candidates)
        head = heads[best["l2_penalty"]]
    logits = _logits(cache, head, report)
    metrics = score_metrics(cache.y(report), logits)
    predictions = [
        {"subject_id": s, "label": int(cache.labels[s]), "prediction": int(z > 0),
         "probability": float(1.0 / (1.0 + math.exp(-z))), "mean_logit": float(z),
         "num_units": len(cache.units[s]), "num_samples": sum(len(v) for v in cache.units[s].values())}
        for s, z in zip(report, logits.tolist())
    ]
    return {
        "selection": {"selected_l2_penalty": best["l2_penalty"], "candidates": candidates,
                      "selection_source": "inner_5fold_cv_stratified_seed1337" if inner else
                      ("official_dev" if protocol == "daic_official" else "saved_inner_validation"),
                      "inner_folds": inner},
        "fit_info": head[4],
        "counts": {"train": len(train), "selection": len(select_on or []), "report": len(report)},
        "metrics": metrics,
        "predictions": predictions,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_route(
    *, cache_dir: Path, dev_cache: Path | None, split_files: list[Path], protocol: str,
    output_dir: Path, route: dict[str, Any], grid=DEFAULT_L2_GRID,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}.")
    if (protocol == "daic_official") != (dev_cache is not None):
        raise ValueError("A dev cache is required for, and only for, daic_official.")
    meta = read_json(cache_dir / "extraction_metadata.json")
    if meta.get("weights_source") != "base_model_no_adapter":
        raise ValueError("Cache is not a frozen base-model (no adapter) extraction.")
    parts = [(cache_dir, "outer_train"), (cache_dir, "final_eval")]
    if dev_cache is not None:
        dev_meta = read_json(dev_cache / "extraction_metadata.json")
        for key in ("base_model", "weights_source", "manifest_sha256", "condition", "input_modality", "pooling"):
            if dev_meta.get(key) != meta.get(key):
                raise ValueError(f"Dev cache differs from main cache in {key}.")
        parts.append((dev_cache, "final_eval"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = PooledCache(parts)
    identity = {
        "schema_version": SCHEMA_VERSION, "variant": VARIANT, "protocol": protocol, "route": route,
        "l2_grid": list(grid), "inner_folds": INNER_FOLDS, "inner_seed": INNER_SEED, "threshold": 0.5,
        "cache_identity": cache_identity(cache_dir),
        "dev_cache_identity": cache_identity(dev_cache) if dev_cache else None,
        "split_files": [{"path": str(p), "sha256": canonical_sha256(read_json(p))} for p in split_files],
        "base_model": meta.get("base_model"), "model_backend": meta.get("model_backend"),
        "manifest_sha256": meta.get("manifest_sha256"),
    }
    identity["config_sha256"] = canonical_sha256(identity)
    save_json(identity, output_dir / "result_config.json")
    folds = []
    for fold, path in enumerate(split_files):
        result = run_fold(cache, read_json(path), protocol, grid)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir()
        save_json(result["selection"], fold_dir / "selection.json")
        save_json({**result["metrics"], "fit_info": result["fit_info"], "counts": result["counts"]},
                  fold_dir / "metrics.json")
        write_jsonl(result["predictions"], fold_dir / "predictions_subject_level.jsonl")
        _write_csv(result["predictions"], fold_dir / "predictions_subject_level.csv")
        m = result["metrics"]
        folds.append({"fold": fold, "selected_l2_penalty": result["selection"]["selected_l2_penalty"],
                      "macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"],
                      "uar": m["binary_strict_uar"], "auroc": m["auroc"],
                      "confusion_matrix": m["confusion_matrix"], **result["counts"]})
    summary = {
        "route": route, "protocol": protocol, "variant": VARIANT, "folds": folds,
        "aggregation": "unweighted_fold_mean_of_subject_level_metrics",
        "fold_mean": {k: float(np.mean([f[k] for f in folds])) for k in ("macro_f1", "positive_f1", "uar")},
        "num_subjects_in_cache": len(cache.units),
    }
    save_json(summary, output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--dev-cache", type=Path)
    parser.add_argument("--split-files", required=True, nargs="+", type=Path)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", default="{}", help="JSON object recorded in the output identity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_route(cache_dir=args.cache_dir, dev_cache=args.dev_cache, split_files=args.split_files,
                        protocol=args.protocol, output_dir=args.output_dir, route=json.loads(args.route))
    print(json.dumps({"route": summary["route"], "fold_mean": summary["fold_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
