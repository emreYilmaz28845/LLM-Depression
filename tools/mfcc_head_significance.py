"""Paired significance of MFCC fixed XGBoost vs Optuna-100 XGBoost.

Both heads score the same held-out subjects, so each dataset is a paired
comparison. CV folds are pooled out-of-fold (every subject appears once), the
same view as tools/paired_significance.py. Tests per dataset:

* exact paired prediction-swap test (primary p-value),
* label-stratified paired bootstrap (delta and 95% CI),
* exact McNemar test on subject correctness.

Holm correction runs across the datasets, separately per metric and for
McNemar. Macro-F1 is the primary metric. Read-only on evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daic_statistics import (  # noqa: E402
    exact_mcnemar,
    exact_paired_prediction_swap,
    holm_adjust,
    stratified_paired_bootstrap_many,
)
from src.metrics import classification_metrics  # noqa: E402
from src.utils import read_jsonl, save_json, sha256_file  # noqa: E402

DATASETS = ("cmdc", "d3tec", "turkish_pooled", "androids_standalone", "androids", "daic")
HEADS = {"fixed": ("fixed_heads", "xgb_raw"), "optuna": ("xgb_optuna100",)}
METRICS = ("macro_f1", "positive_f1", "macro_recall")
LABELS = {"macro_f1": "Macro-F1", "positive_f1": "Positive-F1", "macro_recall": "UAR"}


def pooled_rows(dataset_dir: Path, head: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    files = []
    for fold_dir in sorted(dataset_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[1])):
        path = fold_dir.joinpath(*head, "predictions_subject_level.jsonl")
        rows.extend(
            {"subject_id": str(row["subject_id"]), "label": int(row["label"]), "prediction": int(row["prediction"])}
            for row in read_jsonl(path)
        )
        files.append({"path": str(path), "sha256": sha256_file(path)})
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError(f"{dataset_dir}: a subject is held out in more than one fold.")
    return rows, files


def strict(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = [row["label"] for row in rows]
    p = [row["prediction"] if row["prediction"] in (0, 1) else 1 - row["label"] for row in rows]
    metrics = classification_metrics(y, p)
    return {metric: metrics[metric] for metric in METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/mfcc_xgb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/significance/mfcc_fixed_vs_optuna")
    parser.add_argument("--bootstrap-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    comparisons = []
    for dataset in DATASETS:
        fixed, fixed_files = pooled_rows(args.root / dataset, HEADS["fixed"])
        optuna, optuna_files = pooled_rows(args.root / dataset, HEADS["optuna"])
        permutation = exact_paired_prediction_swap(fixed, optuna, metrics=METRICS)
        bootstrap = stratified_paired_bootstrap_many(
            fixed, optuna, metrics=METRICS, iterations=args.bootstrap_iterations, seed=args.seed
        )
        comparisons.append({
            "dataset": dataset,
            "baseline": "fixed xgb_raw",
            "comparison": "optuna xgb_optuna100",
            "aggregation": "pooled_out_of_fold_subject_level",
            "subjects": len(fixed),
            "fixed": strict(fixed),
            "optuna": strict(optuna),
            "metrics": {metric: {"permutation": permutation[metric], "bootstrap": bootstrap[metric]} for metric in METRICS},
            "mcnemar": exact_mcnemar(fixed, optuna),
            "evidence": {"fixed": fixed_files, "optuna": optuna_files},
        })
    for metric in METRICS:
        adjusted = holm_adjust([row["metrics"][metric]["permutation"]["p_value"] for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row["metrics"][metric]["permutation"]["p_value_holm"] = value
            row["metrics"][metric]["permutation"]["significant_holm"] = value <= args.alpha
    for row, value in zip(comparisons, holm_adjust([row["mcnemar"]["p_value"] for row in comparisons])):
        row["mcnemar"]["p_value_holm"] = value

    payload = {
        "schema_version": "mfcc_fixed_vs_optuna_significance.v1",
        "alpha": args.alpha,
        "primary_metric": "macro_f1",
        "correction": f"Holm across the {len(comparisons)} dataset comparisons, per metric",
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "report.json")

    lines = [
        "# MFCC fixed XGBoost vs Optuna-100 XGBoost: paired significance",
        "",
        "Pooled out-of-fold subject-level, binary_strict. Delta = Optuna - fixed. p = exact paired",
        f"prediction-swap test; Holm across {len(comparisons)} datasets per metric. CI = stratified paired bootstrap.",
        "",
        "| Dataset | N | Metric | Fixed | Optuna | Delta [95% CI] | p | p (Holm) | McNemar b/c, p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in comparisons:
        mc = row["mcnemar"]
        for metric in METRICS:
            perm = row["metrics"][metric]["permutation"]
            boot = row["metrics"][metric]["bootstrap"]
            mark = " *" if perm["significant_holm"] else ""
            mcnemar = (
                f"{mc['baseline_only_correct']}/{mc['comparison_only_correct']}, {mc['p_value']:.3f}"
                if metric == "macro_f1" else ""
            )
            lines.append(
                f"| {row['dataset']} | {row['subjects']} | {LABELS[metric]} | {row['fixed'][metric]:.3f} | "
                f"{row['optuna'][metric]:.3f} | {perm['observed_delta']:+.3f} "
                f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}] | {perm['p_value']:.3f} | "
                f"{perm['p_value_holm']:.3f}{mark} | {mcnemar} |"
            )
    lines += ["", "McNemar b = only fixed correct, c = only Optuna correct. * = significant after Holm at alpha 0.05."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
