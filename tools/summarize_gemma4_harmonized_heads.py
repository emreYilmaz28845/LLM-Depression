#!/usr/bin/env python3
"""Deterministic LR group report for the Gemma harmonized native wave.

Aggregates the fixed-LogReg raw hidden-head outputs from
``outputs/hidden_classifiers/harmonized_v1_gemma4/<dataset>/<run>/fold_<n>/logreg_raw``
into per-dataset/modality qualified reports with pooled subject-level and
fold-mean macro-F1/positive-F1 views, matching the workbook conventions
(D3TEC/Androids pooled; CMDC/Turkish fold-mean). Every displayed value links
to the parent cache identity and its local artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import canonical_sha256, read_json  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402

HEADS_ROOT = PROJECT_ROOT / "outputs/hidden_classifiers/harmonized_v1_gemma4"
FEATURES_ROOT = PROJECT_ROOT / "outputs/hidden_features/harmonized_v1_gemma4"
RUN_FILTER = "gemma4_v1_prod_20260814T2030Z_1ab337d2_r2"
VARIANT = "logreg_raw"
BACKEND = "gemma4_hidden_logreg_raw"


def _negative_f1(metrics: dict) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    return (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if precision_neg + recall_neg
        else 0.0
    )


def collect() -> dict:
    """Map (dataset, modality, fold) -> per-subject predictions + metadata."""
    cells: dict[tuple[str, str, int], dict] = {}
    for variant_dir in HEADS_ROOT.rglob(f"*/{VARIANT}/predictions_subject_level.csv"):
        run_dir = variant_dir.parents[1]
        if RUN_FILTER not in str(run_dir):
            continue
        fold_dir = variant_dir.parents[2] if variant_dir.parents[2].name.startswith("fold_") else variant_dir.parents[1]
        if not fold_dir.name.startswith("fold_"):
            continue
        fold = int(fold_dir.name.split("_", 1)[1])
        cache_dir = FEATURES_ROOT / run_dir.relative_to(HEADS_ROOT).parent / fold_dir.name
        metadata = read_json(cache_dir / "extraction_metadata.json")
        dataset = str(metadata["dataset"]).lower()
        modality = str(metadata["input_modality"])
        classifier_meta = read_json(variant_dir.parent / "classifier_metadata.json")
        metrics = read_json(variant_dir.parent / "metrics.json")
        subject_rows = []
        with open(variant_dir, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                subject_rows.append(
                    {
                        "subject_id": row["subject_id"],
                        "label": int(float(row["label"])),
                        "prediction": int(float(row.get("prediction") or row.get("predicted_class"))),
                    }
                )
        cells[(dataset, modality, fold)] = {
            "fold": fold,
            "metadata": metadata,
            "classifier_metadata": classifier_meta,
            "metrics": metrics,
            "subject_rows": subject_rows,
            "predictions_path": str(variant_dir),
            "metrics_path": str(variant_dir.parent / "metrics.json"),
            "cache_dir": str(cache_dir),
            "run_dir": str(run_dir),
        }
    return cells


def group_report(cells: dict, dataset: str, modality: str) -> dict:
    members = {
        fold: info
        for (ds, mod, fold), info in cells.items()
        if ds == dataset and mod == modality
    }
    folds = sorted(members)
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    fold_metrics: list[dict] = []
    rows: list[dict] = []
    for fold in folds:
        info = members[fold]
        y_true = [row["label"] for row in info["subject_rows"]]
        y_pred = [row["prediction"] for row in info["subject_rows"]]
        metrics = classification_metrics(y_true, y_pred)
        metrics["negative_f1"] = _negative_f1(metrics)
        fold_metrics.append(metrics)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)
        rows.append(
            {
                "fold": fold,
                "macro_f1": metrics["macro_f1"],
                "positive_f1": metrics["positive_f1"],
                "accuracy": metrics["accuracy"],
                "support": len(y_true),
                "parent_cache": info["cache_dir"],
                "predictions_path": info["predictions_path"],
                "metrics_path": info["metrics_path"],
                "prediction_backend": info["classifier_metadata"].get("prediction_backend"),
                "manifest_sha256": info["metadata"].get("manifest_sha256"),
                "split_metadata_sha256": info["metadata"].get("split_metadata_sha256"),
            }
        )
    pooled = classification_metrics(pooled_true, pooled_pred)
    pooled["negative_f1"] = _negative_f1(pooled)
    fold_mean = {
        "macro_f1": sum(item["macro_f1"] for item in fold_metrics) / len(fold_metrics) if fold_metrics else None,
        "positive_f1": sum(item["positive_f1"] for item in fold_metrics) / len(fold_metrics) if fold_metrics else None,
    }
    return {
        "schema_version": "gemma4_harmonized_lr_group_report.v1",
        "family": "native",
        "dataset": dataset,
        "modality": modality,
        "backend": BACKEND,
        "variant": VARIANT,
        "aggregation_views": {
            "pooled_subject_level": {"macro_f1": pooled["macro_f1"], "positive_f1": pooled["positive_f1"]},
            "fold_mean": fold_mean,
        },
        "folds": rows,
        "fold_count": len(folds),
        "report_sha256": canonical_sha256(
            {
                "dataset": dataset,
                "modality": modality,
                "rows": rows,
                "pooled": {k: pooled[k] for k in ("macro_f1", "positive_f1")},
            }
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/experiment_reports/gemma4_harmonized")
    args = parser.parse_args()
    cells = collect()
    pairs = [
        ("d3tec", m) for m in ("audio_text", "audio_only", "text_only")
    ] + [
        ("androids_interview", m) for m in ("audio_text", "audio_only", "text_only")
    ] + [
        ("cmdc", m) for m in ("audio_text", "audio_only", "text_only")
    ] + [
        ("turkish", m) for m in ("audio_text", "audio_only", "text_only")
    ]
    reports = {}
    for dataset, modality in pairs:
        report = group_report(cells, dataset, modality)
        if report["fold_count"] != 5:
            print(f"WARNING {dataset} {modality}: expected 5 folds, got {report['fold_count']}", file=sys.stderr)
        reports[f"{dataset}/{modality}"] = report
        out = args.output_root / "native_lr" / f"{dataset}_{modality}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
    summary = {
        key: {
            "pooled": value["aggregation_views"]["pooled_subject_level"],
            "fold_mean": value["aggregation_views"]["fold_mean"],
        }
        for key, value in reports.items()
    }
    (args.output_root / "native_lr" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    for key, value in sorted(summary.items()):
        pooled = value["pooled"]
        fold = value["fold_mean"]
        print(f"{key:40s} pooled macro={pooled['macro_f1']:.4f} posF1={pooled['positive_f1']:.4f} | fold-mean macro={fold['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
