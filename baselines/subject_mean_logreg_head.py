"""Subject-mean L2 logistic head on cached DAIC hidden vectors.

The head fits one row per training subject (the mean of that subject's
standardized chunk vectors), selects the L2 strength on the official DAIC
development split, and scores a subject as sigmoid(mean chunk logit), which
equals the logit of the subject's mean vector because the head is linear.

Inputs are two caches written by ``src/features/extract_qwen_hidden.py`` from
the same checkpoint and manifest: one with ``--final-eval-source selection``
(final_eval = official dev) and one with the default locked test. Both must
hold the identical ``outer_train`` partition. The test cache is read only
after the L2 strength is fixed.
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

from src.features.hidden_classifier_policy import cache_identity, canonical_sha256
from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, read_jsonl, save_json, write_jsonl


VARIANT = "logreg_subjmean_l2dev"
SCHEMA_VERSION = "subject_mean_logreg_head.v1"
PREDICTION_BACKEND = "qwen_hidden_logreg_subjmean_l2dev"
AGGREGATION_POLICY = "sigmoid_mean_chunk_logit_threshold_0_5"
FIT_POLICY = "subject_mean_standardized_vector_one_row_per_subject"
CLASS_WEIGHT_POLICY = "balanced_by_training_subject_counts"
SCALER_POLICY = "train_only_equal_subject_weight"
SELECTION_RULE = "max dev macro-F1, then min dev BCE, then larger L2"
DEV_PROTOCOL = "daic_official_train_fit_dev_evaluation"
TEST_PROTOCOL = "daic_official_train_fit_locked_test_evaluation"
PACKED30_PROTOCOL = "daic_participant_speech_packed30_v1"
DEFAULT_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
THRESHOLD = 0.5


def load_partition(cache_dir: Path, name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(cache_dir / f"{name}.npz") as payload:
        vectors = np.asarray(payload["vectors"], dtype=np.float64)
    rows = read_jsonl(cache_dir / f"{name}_rows.jsonl")
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError(f"Invalid {name} cache shape {vectors.shape} for {len(rows)} rows.")
    if not bool(np.isfinite(vectors).all()):
        raise ValueError(f"{name} cache contains non-finite values.")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {name} cache.")
    return vectors, rows


def subject_groups(rows: list[dict[str, Any]]) -> tuple["OrderedDict[str, list[int]]", np.ndarray]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    labels: dict[str, int] = {}
    for index, row in enumerate(rows):
        subject = str(row["subject_id"])
        label = int(row["label"])
        if labels.setdefault(subject, label) != label:
            raise ValueError(f"Subject {subject} has conflicting labels.")
        groups.setdefault(subject, []).append(index)
    return groups, np.asarray([labels[subject] for subject in groups], dtype=np.int64)


def fit_scaler(x: np.ndarray, groups: "OrderedDict[str, list[int]]") -> tuple[np.ndarray, np.ndarray]:
    """Train-only scaler in which every subject has equal total weight."""
    mean = np.mean([x[index].mean(0) for index in groups.values()], axis=0)
    variance = np.mean([((x[index] - mean) ** 2).mean(0) for index in groups.values()], axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """sklearn ``class_weight='balanced'`` on subject counts."""
    weights = np.empty(len(y), dtype=np.float64)
    for label in (0, 1):
        weights[y == label] = len(y) / (2.0 * int((y == label).sum()))
    return weights


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalty: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Newton/Armijo on weighted-mean BCE + 0.5*penalty*||w||^2, free intercept.

    The fit runs in the row span of ``x`` (exact for an L2 linear model), so a
    2048-D fit on ~100 subjects stays a small dense solve.
    """
    if penalty <= 0:
        raise ValueError("L2 penalty must be positive.")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes are required to fit the head.")
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[singular > np.finfo(float).eps * max(x.shape) * singular[0]].T
    design = np.column_stack([x @ basis, np.ones(len(x))])
    regularization = np.r_[np.full(basis.shape[1], penalty), 0.0]
    w = weights / weights.sum()
    yf = y.astype(np.float64)
    theta = np.zeros(design.shape[1])
    prior = float(np.sum(w * yf))
    theta[-1] = math.log(prior / (1.0 - prior))

    def objective(value: np.ndarray) -> float:
        z = design @ value
        return float(np.sum(w * (np.logaddexp(0.0, z) - yf * z)) + 0.5 * np.sum(regularization * value**2))

    gradient_norm = math.inf
    iteration = 0
    for iteration in range(max_iterations + 1):
        probability = 1.0 / (1.0 + np.exp(-(design @ theta)))
        gradient = design.T @ (w * (probability - yf)) + regularization * theta
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            break
        hessian = design.T @ (design * (w * probability * (1 - probability))[:, None]) + np.diag(regularization)
        direction = np.linalg.solve(hessian, gradient)
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope <= 0:
            raise ValueError("Invalid Newton direction.")
        loss, step = objective(theta), 1.0
        for _ in range(60):
            if objective(theta - step * direction) <= loss - 1e-4 * step * slope:
                theta = theta - step * direction
                break
            step /= 2.0
        else:
            raise ValueError("Line search failed.")
    converged = gradient_norm <= tolerance
    if not converged:
        raise ValueError(f"Head did not converge (gradient norm {gradient_norm:.3e}).")
    return basis @ theta[:-1], float(theta[-1]), {
        "iterations": int(iteration),
        "gradient_infinity_norm": gradient_norm,
        "converged": converged,
    }


def subject_scores(
    x: np.ndarray, rows: list[dict[str, Any]], weight: np.ndarray, bias: float
) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    groups, labels = subject_groups(rows)
    chunk_logits = x @ weight + bias
    logits = np.asarray([chunk_logits[index].mean() for index in groups.values()])
    return list(groups), labels, logits, [len(index) for index in groups.values()]


def score_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predictions = (logits > 0).astype(int).tolist()
    metrics = classification_metrics(labels.tolist(), predictions)
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    metrics.update(
        {
            "negative_f1": 2 * precision_neg * recall_neg / (precision_neg + recall_neg)
            if precision_neg + recall_neg
            else 0.0,
            "binary_strict_uar": metrics["macro_recall"],
            "auroc": binary_auroc(labels.tolist(), logits.tolist()),
            "binary_cross_entropy": float(np.mean(np.logaddexp(0.0, logits) - labels * logits)),
            "num_subjects": int(len(labels)),
            "invalid_count": 0,
            "aggregation": "subject_level",
            "aggregation_policy": AGGREGATION_POLICY,
            "prediction_backend": PREDICTION_BACKEND,
        }
    )
    return metrics


def _check_cache_pair(dev_cache: Path, test_cache: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dev_meta = read_json(dev_cache / "extraction_metadata.json")
    test_meta = read_json(test_cache / "extraction_metadata.json")
    for name, meta, protocol in (("dev", dev_meta, DEV_PROTOCOL), ("test", test_meta, TEST_PROTOCOL)):
        if str(meta.get("dataset", "")).lower() != "daic":
            raise ValueError(f"{name} cache is not DAIC.")
        observed = meta.get("evaluation_provenance", {}).get("evaluation_protocol")
        if observed != protocol:
            raise ValueError(f"{name} cache protocol is {observed!r}, expected {protocol!r}.")
        if meta.get("protocol_id") != PACKED30_PROTOCOL:
            raise ValueError(f"{name} cache is not DAIC packed30.")
    for key in (
        "model_backend",
        "base_model",
        "weights_source",
        "adapter_sha256",
        "condition",
        "input_modality",
        "manifest_sha256",
        "saved_split_sha256",
        "hidden_layer",
        "pooling",
        "fold",
    ):
        if dev_meta.get(key) != test_meta.get(key):
            raise ValueError(f"Dev and test caches differ in {key}.")
    dev_id, test_id = cache_identity(dev_cache), cache_identity(test_cache)
    for name in ("outer_train.npz", "outer_train_rows.jsonl"):
        if dev_id[name]["sha256"] != test_id[name]["sha256"]:
            raise ValueError(f"Dev and test caches hold different {name}.")
    return dev_meta, test_meta


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    dev_cache: Path,
    test_cache: Path,
    output_dir: Path,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
) -> dict[str, Any]:
    dev_meta, test_meta = _check_cache_pair(dev_cache, test_cache)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "variant": VARIANT,
        "prediction_backend": PREDICTION_BACKEND,
        "fit_policy": FIT_POLICY,
        "scaler_policy": SCALER_POLICY,
        "class_weight_policy": CLASS_WEIGHT_POLICY,
        "aggregation_policy": AGGREGATION_POLICY,
        "selection_rule": SELECTION_RULE,
        "l2_grid": list(l2_grid),
        "threshold": THRESHOLD,
        "dev_cache_identity": cache_identity(dev_cache),
        "test_cache_identity": cache_identity(test_cache),
    }
    identity["config_sha256"] = canonical_sha256(identity)
    variant_dir = output_dir / VARIANT
    if variant_dir.exists() and any(variant_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing head output: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_json(identity, variant_dir / "result_config.json")

    train_x, train_rows = load_partition(test_cache, "outer_train")
    train_groups, train_y = subject_groups(train_rows)
    mean, scale = fit_scaler(train_x, train_groups)
    standardized = (train_x - mean) / scale
    fit_x = np.stack([standardized[index].mean(0) for index in train_groups.values()])
    fit_weights = balanced_weights(train_y)

    dev_x, dev_rows = load_partition(dev_cache, "final_eval")
    dev_z = (dev_x - mean) / scale
    overlap = set(train_groups) & set(subject_groups(dev_rows)[0])
    if overlap:
        raise ValueError(f"Train/dev subject leakage: {sorted(overlap)[:10]}")
    candidates = []
    for penalty in l2_grid:
        weight, bias, fit_info = fit_logistic(fit_x, train_y, fit_weights, float(penalty))
        _, labels, logits, _ = subject_scores(dev_z, dev_rows, weight, bias)
        metrics = score_metrics(labels, logits)
        candidates.append(
            {
                "l2_penalty": float(penalty),
                **fit_info,
                "dev_macro_f1": metrics["macro_f1"],
                "dev_positive_f1": metrics["positive_f1"],
                "dev_binary_cross_entropy": metrics["binary_cross_entropy"],
                "dev_auroc": metrics["auroc"],
                "_head": (weight, bias),
            }
        )
    best = max(
        candidates,
        key=lambda c: (c["dev_macro_f1"], -c["dev_binary_cross_entropy"], c["l2_penalty"]),
    )
    weight, bias = best["_head"]
    selection = {
        "status": "fixed_before_test",
        "selection_rule": SELECTION_RULE,
        "selected_l2_penalty": best["l2_penalty"],
        "candidates": [{k: v for k, v in c.items() if k != "_head"} for c in candidates],
    }
    save_json(selection, variant_dir / "selection.json")
    with (variant_dir / "head.npz").open("xb") as handle:
        np.savez(
            handle,
            weight=weight,
            bias=np.asarray(bias),
            feature_mean=mean,
            feature_scale=scale,
            l2_penalty=np.asarray(best["l2_penalty"]),
            threshold=np.asarray(THRESHOLD),
        )

    test_x, test_rows = load_partition(test_cache, "final_eval")
    overlap = set(train_groups) & set(subject_groups(test_rows)[0])
    if overlap:
        raise ValueError(f"Train/test subject leakage: {sorted(overlap)[:10]}")
    summary: dict[str, Any] = {"variant": VARIANT, "selected_l2_penalty": best["l2_penalty"]}
    for split, x, rows, meta in (
        ("dev", dev_x, dev_rows, dev_meta),
        ("test", test_x, test_rows, test_meta),
    ):
        subjects, labels, logits, counts = subject_scores((x - mean) / scale, rows, weight, bias)
        metrics = score_metrics(labels, logits)
        metrics["evaluation_protocol"] = meta["evaluation_provenance"]["evaluation_protocol"]
        split_dir = variant_dir / split
        split_dir.mkdir()
        subject_rows = [
            {
                "subject_id": subject,
                "label": int(label),
                "prediction": int(logit > 0),
                "probability": float(1.0 / (1.0 + math.exp(-logit))),
                "mean_chunk_logit": float(logit),
                "num_samples": int(count),
                "aggregation_method": AGGREGATION_POLICY,
                "prediction_backend": PREDICTION_BACKEND,
                "evaluation_protocol_name": metrics["evaluation_protocol"],
                "dataset": meta["dataset"],
                "modality": meta["input_modality"],
                "condition": meta.get("condition") or meta["input_modality"],
                "fold": int(meta["fold"]),
                "classifier_variant": VARIANT,
                "protocol_id": meta["protocol_id"],
            }
            for subject, label, logit, count in zip(subjects, labels, logits, counts)
        ]
        write_jsonl(subject_rows, split_dir / "predictions_subject_level.jsonl")
        _write_csv(subject_rows, split_dir / "predictions_subject_level.csv")
        save_json(metrics, split_dir / "metrics.json")
        summary[split] = {
            key: metrics[key]
            for key in ("macro_f1", "positive_f1", "binary_strict_uar", "auroc", "confusion_matrix")
        }
    save_json(
        {
            **{k: identity[k] for k in identity if not k.endswith("_cache_identity")},
            "dev_cache": str(dev_cache),
            "test_cache": str(test_cache),
            "dataset": test_meta["dataset"],
            "condition": test_meta.get("condition"),
            "model_backend": test_meta.get("model_backend"),
            "weights_source": test_meta.get("weights_source"),
            "checkpoint_hashes": {
                "adapter_config_sha256": test_meta.get("adapter_config_sha256"),
                "adapter_sha256": test_meta.get("adapter_sha256"),
            },
            "split_hashes": {
                "saved_split_sha256": test_meta.get("saved_split_sha256"),
                "split_metadata_sha256": test_meta.get("split_metadata_sha256"),
                "manifest_sha256": test_meta.get("manifest_sha256"),
            },
            "training_subject_ids": list(train_groups),
            "training_class_counts": {"0": int((train_y == 0).sum()), "1": int((train_y == 1).sum())},
            "selected_l2_penalty": best["l2_penalty"],
        },
        variant_dir / "classifier_metadata.json",
    )
    save_json(summary, variant_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-cache", required=True, type=Path, help="Cache whose final_eval is the official dev split.")
    parser.add_argument("--test-cache", required=True, type=Path, help="Cache whose final_eval is the locked test split.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--l2-grid", nargs="+", type=float, default=list(DEFAULT_L2_GRID))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.dev_cache, args.test_cache, args.output_dir, tuple(args.l2_grid))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
