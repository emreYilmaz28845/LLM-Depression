#!/usr/bin/env python3
"""Paper LogReg head and subject-mean L2 head on symmetric-merged hidden features.

One merged fold directory (``.../{cv,final}/fold_<n>/features``) supplies the
row index: ``outer_train`` and ``outer_holdout`` rows with their partition,
labels and prompt hashes. Two feature sources share that index:

- ``lora``: the merged run's own adapter features, as saved.
- ``frozen``: each row's vector is replaced by the frozen base-model vector of
  the same component sample, taken from standalone frozen caches. A row whose
  sample is missing or whose prompt hash differs fails the run.

Two heads are fitted on each source:

- ``paper_logreg``: the merged paper head from ``src.merged.heads`` (standardized
  LogReg, C=1, hierarchical dataset/subject/response/window weights, response
  then subject mean probability). On ``lora`` it reproduces the saved cell.
- ``subjmean_l2``: one row per training subject (mean over response units of
  the mean standardized vector), weighted so every dataset has equal mass and
  classes are balanced inside each dataset. The L2 penalty is chosen on the
  merged protocol's 3 grouped inner folds by mean per-dataset macro-F1, then
  lower BCE, then larger L2. A subject's score is its mean chunk logit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.subject_mean_logreg_head import DEFAULT_L2_GRID, fit_logistic, score_metrics  # noqa: E402
from src.merged.heads import (  # noqa: E402
    _fit_weighted,
    _load_features,
    _new_logreg,
    _predict_probability,
    _prediction_response_id,
    aggregate_head_predictions,
)
from src.merged.protocol import build_grouped_inner_folds  # noqa: E402
from src.utils import sha256_file  # noqa: E402

SCHEMA_VERSION = "merged_subject_mean_heads.v1"
HEAD_SEED = 1337
INNER_FOLDS = 3
INNER_SEED = 1337
SOURCES = ("lora", "frozen")
SELECTION_RULE = "max mean per-dataset inner macro-F1, then min BCE, then larger L2"
WEIGHT_POLICY = "equal_dataset_then_balanced_class_one_row_per_subject"


def load_frozen_index(cache_dirs: list[Path]) -> tuple[dict[str, tuple[np.ndarray, str]], dict[str, Any]]:
    """Map component sample id -> (vector, prompt hash) over frozen caches.

    A sample present in several caches (for example DAIC train rows in both the
    test and the dev extraction) keeps its first vector; the largest absolute
    difference between copies is recorded as a reproducibility check.
    """
    index: dict[str, tuple[np.ndarray, str]] = {}
    audit = {"caches": [], "duplicate_samples": 0, "duplicate_max_abs_diff": 0.0}
    for cache_dir in cache_dirs:
        meta = json.loads((cache_dir / "extraction_metadata.json").read_text())
        if meta.get("weights_source") != "base_model_no_adapter":
            raise ValueError(f"{cache_dir} is not a frozen base-model cache.")
        audit["caches"].append({"path": str(cache_dir), "base_model": meta.get("base_model"),
                                "metadata_sha256": sha256_file(cache_dir / "extraction_metadata.json")})
        for npz in sorted(cache_dir.glob("*.npz")):
            rows_path = npz.with_name(f"{npz.stem}_rows.jsonl")
            with np.load(npz) as payload:
                matrix = np.asarray(payload[payload.files[0]], dtype=np.float32)
            rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            if len(rows) != len(matrix):
                raise ValueError(f"{npz}: {len(matrix)} vectors for {len(rows)} rows.")
            for row, vector in zip(rows, matrix):
                key = str(row["sample_id"])
                if key in index:
                    audit["duplicate_samples"] += 1
                    diff = float(np.max(np.abs(index[key][0] - vector)))
                    audit["duplicate_max_abs_diff"] = max(audit["duplicate_max_abs_diff"], diff)
                    continue
                index[key] = (vector, str(row.get("prompt_sha256")))
    return index, audit


def frozen_matrix(rows: list[dict[str, Any]], index: dict[str, tuple[np.ndarray, str]]) -> np.ndarray:
    vectors, missing, mismatched = [], [], []
    for row in rows:
        key = str(row["component_sample_id"])
        if key not in index:
            missing.append(key)
            continue
        vector, prompt = index[key]
        if prompt != str(row.get("prompt_sha256")):
            mismatched.append(key)
        vectors.append(vector)
    if missing or mismatched:
        raise ValueError(f"Frozen caches do not cover the merged rows: {len(missing)} missing "
                         f"(e.g. {missing[:3]}), {len(mismatched)} prompt mismatches (e.g. {mismatched[:3]}).")
    return np.stack(vectors)


class SubjectUnits:
    """Row indices of every subject, grouped by the merged response unit."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.units: dict[str, OrderedDict[str, list[int]]] = {}
        self.labels: dict[str, int] = {}
        self.dataset: dict[str, str] = {}
        for i, row in enumerate(rows):
            subject = str(row["subject_id"])
            if self.labels.setdefault(subject, int(row["label"])) != int(row["label"]):
                raise ValueError(f"Subject {subject} has conflicting labels.")
            self.dataset[subject] = str(row["dataset"]).lower()
            self.units.setdefault(subject, OrderedDict()).setdefault(_prediction_response_id(row), []).append(i)

    def means(self, z: np.ndarray, subjects: list[str]) -> np.ndarray:
        return np.stack([np.mean([z[r].mean(0) for r in self.units[s].values()], axis=0) for s in subjects])

    def y(self, subjects: list[str]) -> np.ndarray:
        return np.asarray([self.labels[s] for s in subjects], dtype=np.int64)


def dataset_class_weights(subjects: list[str], units: SubjectUnits) -> np.ndarray:
    """Equal total weight per dataset, balanced classes inside each dataset."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for s in subjects:
        counts[(units.dataset[s], units.labels[s])] += 1
    datasets = {d for d, _ in counts}
    for d in datasets:
        if counts[(d, 0)] == 0 or counts[(d, 1)] == 0:
            raise ValueError(f"Dataset {d} lacks a class in this training set.")
    return np.asarray([1.0 / (len(datasets) * 2 * counts[(units.dataset[s], units.labels[s])]) for s in subjects])


def _fit_subject_head(x: np.ndarray, units: SubjectUnits, subjects: list[str], penalty: float):
    # Train-only scaler in which every subject has equal total weight.
    sub_means = [np.mean([x[r].mean(0) for r in units.units[s].values()], axis=0) for s in subjects]
    mean = np.mean(sub_means, axis=0)
    variance = np.mean([np.mean([((x[r] - mean) ** 2).mean(0) for r in units.units[s].values()], axis=0)
                        for s in subjects], axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    weight, bias, info = fit_logistic(units.means(z, subjects), units.y(subjects),
                                      dataset_class_weights(subjects, units), penalty)
    return mean, scale, weight, bias, info


def _subject_logits(x: np.ndarray, units: SubjectUnits, head, subjects: list[str]) -> np.ndarray:
    mean, scale, weight, bias, _ = head
    return units.means((x - mean) / scale, subjects) @ weight + bias


def _per_dataset(units: SubjectUnits, subjects: list[str], logits: np.ndarray) -> dict[str, dict[str, Any]]:
    out = {}
    for d in sorted({units.dataset[s] for s in subjects}):
        keep = [i for i, s in enumerate(subjects) if units.dataset[s] == d]
        out[d] = score_metrics(units.y([subjects[i] for i in keep]), logits[keep])
    return out


def fit_subjmean_l2(x_train, train_rows, x_holdout, holdout_rows, grid=DEFAULT_L2_GRID) -> dict[str, Any]:
    tr = SubjectUnits(train_rows)
    ho = SubjectUnits(holdout_rows)
    train = sorted(tr.units)
    inner = build_grouped_inner_folds(train_rows, inner_folds=INNER_FOLDS, seed=INNER_SEED)
    candidates = []
    for penalty in grid:
        oof: dict[str, float] = {}
        for fold in inner["folds"]:
            held = set(fold["validation_subject_ids"])
            fit_subjects = [s for s in train if s not in held]
            head = _fit_subject_head(x_train, tr, fit_subjects, float(penalty))
            held_list = [s for s in train if s in held]
            oof.update(zip(held_list, _subject_logits(x_train, tr, head, held_list).tolist()))
        logits = np.asarray([oof[s] for s in train])
        per = _per_dataset(tr, train, logits)
        candidates.append({
            "l2_penalty": float(penalty),
            "mean_dataset_macro_f1": float(np.mean([m["macro_f1"] for m in per.values()])),
            "binary_cross_entropy": float(np.mean(np.logaddexp(0.0, logits) - tr.y(train) * logits)),
            "per_dataset_macro_f1": {d: m["macro_f1"] for d, m in per.items()},
        })
    best = max(candidates, key=lambda c: (c["mean_dataset_macro_f1"], -c["binary_cross_entropy"], c["l2_penalty"]))
    head = _fit_subject_head(x_train, tr, train, best["l2_penalty"])
    holdout = sorted(ho.units)
    logits = _subject_logits(x_holdout, ho, head, holdout)
    predictions = [
        {"dataset": ho.dataset[s], "subject_id": s, "label": ho.labels[s], "prediction": int(z > 0),
         "probability": 1.0 / (1.0 + math.exp(-z)), "mean_logit": z, "num_units": len(ho.units[s])}
        for s, z in zip(holdout, logits.tolist())
    ]
    return {
        "selection": {"selected_l2_penalty": best["l2_penalty"], "rule": SELECTION_RULE,
                      "inner_folds": INNER_FOLDS, "inner_seed": INNER_SEED,
                      "inner_assignments_hash": inner.get("assignments_hash"), "candidates": candidates},
        "fit_info": head[4],
        "metrics_by_dataset": _per_dataset(ho, holdout, logits),
        "predictions": predictions,
    }


def fit_paper_logreg(x_train, train_rows, x_holdout, holdout_rows) -> dict[str, Any]:
    y = np.asarray([int(r["label"]) for r in train_rows], dtype=np.int64)
    estimator = _new_logreg(HEAD_SEED)
    _fit_weighted(estimator, x_train, y, train_rows)
    probability = _predict_probability(estimator, x_holdout)
    subjects, metrics = aggregate_head_predictions(holdout_rows, probability, threshold=0.5)
    predictions = [row for rows in subjects.values() for row in rows]
    for m in metrics.values():
        m["binary_strict_uar"] = m["macro_recall"]
    return {"metrics_by_dataset": metrics, "predictions": predictions}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(features_dir: Path, output_dir: Path, source: str, frozen_caches: list[Path] | None = None,
        heads: tuple[str, ...] = ("paper_logreg", "subjmean_l2")) -> dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite {output_dir}.")
    meta = json.loads((features_dir / "feature_metadata.json").read_text())
    x_train, train_rows = _load_features(features_dir, "outer_train")
    x_holdout, holdout_rows = _load_features(features_dir, "outer_holdout")
    frozen_audit = None
    if source == "frozen":
        if not frozen_caches:
            raise ValueError("--frozen-cache is required for the frozen source.")
        index, frozen_audit = load_frozen_index(frozen_caches)
        x_train, x_holdout = frozen_matrix(train_rows, index), frozen_matrix(holdout_rows, index)
    elif frozen_caches:
        raise ValueError("--frozen-cache applies only to the frozen source.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "features_dir": str(features_dir),
        "feature_metadata_sha256": sha256_file(features_dir / "feature_metadata.json"),
        "stage": meta.get("stage"), "fold": meta.get("fold"), "modality": meta.get("modality"),
        "lora_checkpoint_dir": meta.get("checkpoint_dir") if source == "lora" else None,
        "manifest_hash": meta.get("manifest_hash"), "split_hash": meta.get("split_hash"),
        "fold_hash": meta.get("fold_hash"), "row_hashes": meta.get("row_hashes"),
        "frozen_audit": frozen_audit,
        "feature_dimension": int(x_train.shape[1]),
        "counts": {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows)},
        "heads": {},
    }
    fitters = {"paper_logreg": fit_paper_logreg, "subjmean_l2": fit_subjmean_l2}
    for name in heads:
        result = fitters[name](x_train, train_rows, x_holdout, holdout_rows)
        head_dir = output_dir / name
        head_dir.mkdir()
        _write_csv(result["predictions"], head_dir / "predictions_subject_level.csv")
        (head_dir / "metrics_by_dataset.json").write_text(json.dumps(result["metrics_by_dataset"], indent=2))
        summary["heads"][name] = {k: v for k, v in result.items() if k != "predictions"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--frozen-cache", action="append", type=Path,
                        help="Frozen standalone cache directory; repeat for every dataset/partition.")
    parser.add_argument("--heads", default="paper_logreg,subjmean_l2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.features_dir, args.output_dir, args.source, args.frozen_cache,
                  tuple(h for h in args.heads.split(",") if h))
    for name, head in summary["heads"].items():
        print(name, {d: round(m["macro_f1"], 4) for d, m in head["metrics_by_dataset"].items()})


if __name__ == "__main__":
    main()
