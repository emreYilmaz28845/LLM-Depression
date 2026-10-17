"""Paired significance of the MFCC XGBoost heads vs Qwen3-Omni.

Each MFCC head (fixed xgb_raw, Optuna-100) is compared with the Qwen3-Omni
prompt-context runs (audio only and audio + text) on the same held-out
subjects. Androids uses the standalone folds (the split the Omni runs used).
CV folds are pooled out-of-fold, as in tools/mfcc_head_significance.py.

Tests per comparison: exact paired prediction-swap test (primary p), label-
stratified paired bootstrap (delta and 95% CI) and exact McNemar. Raw p values
are reported; Holm across all comparisons per metric is stored in the JSON.
Subject sets and labels must match exactly, or the run fails. Read-only on
evidence.
"""

from __future__ import annotations

import argparse
import csv
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
from src.utils import save_json, sha256_file  # noqa: E402
from tools.mfcc_head_significance import HEADS, LABELS, METRICS, pooled_rows, strict  # noqa: E402

AUDIOLLM = PROJECT_ROOT.parents[1] if PROJECT_ROOT.parent.name == "worktrees" else PROJECT_ROOT.parent
OMNI_CV_ROOT = (
    AUDIOLLM / "worktrees/LLM-Depression-feat-qwen3omni-standalone-promptcontext"
    / "output_model/promptcontext_v1_qwen3omni_likelihood"
)
OMNI_DAIC_ROOT = AUDIOLLM / "LLM-Depression/output_model/promptcontext_v1_qwen3omni_likelihood"
# MFCC dataset dir -> (display name, Omni dataset dir)
DATASETS = {
    "cmdc": ("CMDC", "cmdc"),
    "d3tec": ("D3TEC", "d3tec"),
    "turkish_pooled": ("Turkish pooled", "turkish"),
    "androids_standalone": ("Androids (standalone folds)", "androids_interview"),
    "daic": ("DAIC", "daic"),
}
OMNI_MODALITIES = {"audio_only": "Omni audio only", "audio_text": "Omni audio+text"}
MFCC_LABELS = {"fixed": "MFCC fixed", "optuna": "MFCC Optuna-100"}


def omni_rows(omni_dataset: str, modality: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = OMNI_DAIC_ROOT if omni_dataset == "daic" else OMNI_CV_ROOT
    paths = sorted((root / modality / omni_dataset).glob("*/fold_*/best_model/standalone_eval/predictions_subject_level.csv"))
    expected = 1 if omni_dataset == "daic" else 5
    if len(paths) != expected:
        raise ValueError(f"Omni {modality}/{omni_dataset}: expected {expected} prediction files, found {len(paths)}.")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    prediction = int(float(row["prediction"]))
                except (TypeError, ValueError):
                    prediction = -1
                rows.append({"subject_id": str(row["subject_id"]), "label": int(row["label"]), "prediction": prediction})
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError(f"Omni {modality}/{omni_dataset}: a subject appears more than once.")
    return rows, [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/mfcc_xgb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/significance/mfcc_vs_omni")
    parser.add_argument("--bootstrap-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    comparisons = []
    for mfcc_dataset, (display, omni_dataset) in DATASETS.items():
        for modality, omni_label in OMNI_MODALITIES.items():
            omni, omni_files = omni_rows(omni_dataset, modality)
            for head, head_parts in HEADS.items():
                mfcc, mfcc_files = pooled_rows(args.root / mfcc_dataset, head_parts)
                permutation = exact_paired_prediction_swap(mfcc, omni, metrics=METRICS)
                bootstrap = stratified_paired_bootstrap_many(
                    mfcc, omni, metrics=METRICS, iterations=args.bootstrap_iterations, seed=args.seed
                )
                comparisons.append({
                    "dataset": display,
                    "baseline": MFCC_LABELS[head],
                    "comparison": omni_label,
                    "aggregation": "pooled_out_of_fold_subject_level",
                    "subjects": len(mfcc),
                    "baseline_scores": strict(mfcc),
                    "comparison_scores": strict(omni),
                    "metrics": {
                        metric: {"permutation": permutation[metric], "bootstrap": bootstrap[metric]} for metric in METRICS
                    },
                    "mcnemar": exact_mcnemar(mfcc, omni),
                    "evidence": {"baseline": mfcc_files, "comparison": omni_files},
                })
    for metric in METRICS:
        adjusted = holm_adjust([row["metrics"][metric]["permutation"]["p_value"] for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row["metrics"][metric]["permutation"]["p_value_holm"] = value
    payload = {
        "schema_version": "mfcc_vs_omni_significance.v1",
        "alpha": args.alpha,
        "delta": "Omni minus MFCC",
        "holm": f"stored only; across all {len(comparisons)} comparisons per metric",
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "report.json")

    lines = [
        "# MFCC XGBoost vs Qwen3-Omni: paired significance",
        "",
        "Pooled out-of-fold subject-level, binary_strict. Delta = Omni - MFCC. p = exact paired",
        "prediction-swap test (uncorrected). CI = stratified paired bootstrap.",
        "",
        "| Dataset | N | MFCC head | Omni | Metric | MFCC | Omni | Delta [95% CI] | p | Significant? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in comparisons:
        for metric in METRICS:
            perm = row["metrics"][metric]["permutation"]
            boot = row["metrics"][metric]["bootstrap"]
            if perm["p_value"] < args.alpha:
                verdict = "Yes (Omni better)" if perm["observed_delta"] > 0 else "Yes (MFCC better)"
            else:
                verdict = "No"
            lines.append(
                f"| {row['dataset']} | {row['subjects']} | {row['baseline']} | {row['comparison']} | {LABELS[metric]} | "
                f"{row['baseline_scores'][metric]:.3f} | {row['comparison_scores'][metric]:.3f} | "
                f"{perm['observed_delta']:+.3f} [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}] | "
                f"{perm['p_value']:.3f} | {verdict} |"
            )
    lines += ["", "| Dataset | MFCC head | Omni | McNemar b / c | p |", "|---|---|---|---|---|"]
    for row in comparisons:
        mc = row["mcnemar"]
        lines.append(
            f"| {row['dataset']} | {row['baseline']} | {row['comparison']} | "
            f"{mc['baseline_only_correct']} / {mc['comparison_only_correct']} | {mc['p_value']:.3f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
