"""Summarize the local openSMILE MFCC + XGBoost campaign with provenance.

Reads only artifacts under the campaign root. Reports both aggregations:
fold-mean of per-fold subject metrics and pooled subject-level metrics over
all held-out folds. INVALID subject predictions count as wrong.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.metrics import classification_metrics
from src.utils import read_json, read_jsonl, save_json, sha256_file

HEADS = {
    "xgb_raw": ("fixed_heads", "xgb_raw"),
    "xgb_optuna100": ("xgb_optuna100",),
}
DATASETS = ("androids", "androids_standalone", "cmdc", "d3tec", "daic", "turkish_pooled")
METRICS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")


def strict_pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [
        int(row["prediction"]) if int(row["prediction"]) in (0, 1) else 1 - int(row["label"])
        for row in rows
    ]
    metrics = classification_metrics(y_true, y_pred)
    return {
        "binary_strict_macro_f1": metrics["macro_f1"],
        "binary_strict_positive_f1": metrics["positive_f1"],
        "binary_strict_uar": metrics["macro_recall"],
        "subjects": len(rows),
        "invalid_subjects": sum(int(row["prediction"]) not in (0, 1) for row in rows),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def summarize_head(dataset_dir: Path, head_parts: tuple[str, ...]) -> dict[str, Any] | None:
    folds = sorted(dataset_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[1]))
    per_fold = []
    pooled_rows: list[dict[str, Any]] = []
    for fold_dir in folds:
        head_dir = fold_dir.joinpath(*head_parts)
        metrics_path = head_dir / "metrics.json"
        predictions_path = head_dir / "predictions_subject_level.jsonl"
        if not metrics_path.is_file() or not predictions_path.is_file():
            return None
        metrics = read_json(metrics_path)
        rows = read_jsonl(predictions_path)
        pooled_rows.extend(rows)
        # Recompute from the subject predictions and cross-check the head's
        # own metrics file. The DAIC packed30 head writes macro_f1 /
        # positive_f1 / macro_recall instead of binary_strict_* names.
        recomputed = strict_pooled_metrics(rows)
        for name, fallback in zip(METRICS, ("macro_f1", "positive_f1", "macro_recall")):
            reported = metrics.get(name, metrics.get(fallback))
            if reported is None or abs(float(reported) - recomputed[name]) > 1e-9:
                raise ValueError(f"{metrics_path}: {name} {reported} != recomputed {recomputed[name]}")
        per_fold.append(
            {
                "fold": int(fold_dir.name.split("_")[1]),
                **{name: recomputed[name] for name in METRICS},
                "subjects": recomputed["subjects"],
                "invalid_subjects": recomputed["invalid_subjects"],
                "aggregation_policy": metrics.get("aggregation_policy") or metrics.get("aggregation_method"),
                "prediction_backend": metrics.get("prediction_backend"),
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_subject_level_path": str(predictions_path),
                "predictions_subject_level_sha256": sha256_file(predictions_path),
            }
        )
    if not per_fold:
        return None
    subjects = [str(row["subject_id"]) for row in pooled_rows]
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"{dataset_dir}: a subject is held out in more than one fold.")
    fold_mean = {
        name: statistics.fmean(item[name] for item in per_fold) for name in METRICS
    }
    fold_std = {
        name: statistics.pstdev(item[name] for item in per_fold) if len(per_fold) > 1 else 0.0
        for name in METRICS
    }
    return {
        "folds": per_fold,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "pooled_subject_level": strict_pooled_metrics(pooled_rows),
        "aggregation_policies": sorted({str(item["aggregation_policy"]) for item in per_fold}),
        "prediction_backends": sorted({str(item["prediction_backend"]) for item in per_fold}),
    }


def cache_provenance(dataset_dir: Path) -> dict[str, Any]:
    first = sorted(dataset_dir.glob("fold_*/cache/extraction_metadata.json"))[0]
    metadata = read_json(first)
    keys = (
        "config_path", "config_sha256", "manifest_path", "manifest_sha256", "folds_path", "folds_sha256",
        "subject_partition_path", "locked_folds_path", "locked_folds_sha256", "path_maps", "opensmile",
        "git_commit", "git_dirty", "evaluation_provenance", "classifier_aggregation_policy",
    )
    return {key: metadata.get(key) for key in keys}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# openSMILE MFCC + XGBoost (local CPU)",
        "",
        "Subject-level, binary_strict (INVALID counts as wrong). Window probabilities are averaged per",
        "response, then per subject (threshold 0.5). Cells are Macro-F1 / Positive-F1 / UAR.",
        "",
        "| Dataset | Head | Folds | Fold-mean | Pooled subject-level |",
        "|---|---|---|---|---|",
    ]
    for dataset, heads in summary["datasets"].items():
        for head, result in heads["heads"].items():
            if result is None:
                lines.append(f"| {dataset} | {head} | — | not finished | not finished |")
                continue
            mean = result["fold_mean"]
            pooled = result["pooled_subject_level"]
            lines.append(
                f"| {dataset} | {head} | {len(result['folds'])} | "
                f"{mean['binary_strict_macro_f1']:.3f} / {mean['binary_strict_positive_f1']:.3f} / {mean['binary_strict_uar']:.3f} | "
                f"{pooled['binary_strict_macro_f1']:.3f} / {pooled['binary_strict_positive_f1']:.3f} / {pooled['binary_strict_uar']:.3f} |"
            )
    lines += ["", "Provenance per dataset (config, manifest, folds, openSMILE config, git commit) is in summary.json."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("outputs/mfcc_xgb"))
    args = parser.parse_args()
    summary: dict[str, Any] = {"schema_version": "opensmile_mfcc_xgb_summary.v1", "root": str(args.root.resolve()), "datasets": {}}
    for dataset in DATASETS:
        dataset_dir = args.root / dataset
        if not any(dataset_dir.glob("fold_*/cache/extraction_metadata.json")):
            continue
        summary["datasets"][dataset] = {
            "provenance": cache_provenance(dataset_dir),
            "heads": {name: summarize_head(dataset_dir, parts) for name, parts in HEADS.items()},
        }
    save_json(summary, args.root / "summary.json")
    (args.root / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
