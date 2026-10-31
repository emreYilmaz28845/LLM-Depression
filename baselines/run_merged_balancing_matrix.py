#!/usr/bin/env python3
"""Compare class/dataset balancing of the merged heads on frozen features.

Frozen base features, one merged route (default native audio_only), 5 CV folds
plus the merged final fold. Variants (see ``merged_subject_mean_heads``):

- new head (``subjmean_l2``) with ``none``, ``dataset_weight``,
  ``dataset_class_weight`` (the current head): deterministic, one run per fold;
- new head with ``oversample_dataset_class`` and ``smote_dataset_class``: one
  run per fold and seed;
- paper head (``paper_logreg``) and the same head with class-balanced
  hierarchical weights (``paper_logreg_class_balanced``).

Writes every fold run, then ``summary.json`` and ``summary.md``: per-dataset
fold-mean macro-F1 / positive-F1 / UAR (subject level, binary strict, seed mean
for seeded variants), the DAIC merged-final official-test cell, the training
subject counts per fold, and paired t-tests of CV macro-F1 over the 5 datasets
(no correction): every new-head variant against ``dataset_class_weight``, the
class-balanced paper head against the paper head, and the current new head
against the paper head.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.head_ablation_stats import _metrics, _paired_t  # noqa: E402
from baselines.merged_subject_mean_heads import run  # noqa: E402
from baselines.run_merged_head_matrix import STAGES, features_dir, frozen_caches  # noqa: E402

# variant -> (head, balancing, seeded)
VARIANTS = {
    "none": ("subjmean_l2", "none", False),
    "dataset_weight": ("subjmean_l2", "dataset_weight", False),
    "dataset_class_weight": ("subjmean_l2", "dataset_class_weight", False),
    "oversample_dataset_class": ("subjmean_l2", "oversample_dataset_class", True),
    "smote_dataset_class": ("subjmean_l2", "smote_dataset_class", True),
    "paper": ("paper_logreg", "dataset_class_weight", False),
    "paper_class_balanced": ("paper_logreg_class_balanced", "dataset_class_weight", False),
}
REFERENCE = "dataset_class_weight"
COMPARISONS = [(v, REFERENCE) for v in ("none", "dataset_weight", "oversample_dataset_class", "smote_dataset_class")]
COMPARISONS += [("paper_class_balanced", "paper"), (REFERENCE, "paper")]
LABELS = {"none": "A: new head, no balancing", "dataset_weight": "C: new head, dataset weight only",
          "dataset_class_weight": "B: new head, dataset + class weight (current)",
          "oversample_dataset_class": "D1: new head, dataset x class oversampling",
          "smote_dataset_class": "E: new head, dataset x class SMOTE",
          "paper": "P: paper head (dataset weight, no class balance)",
          "paper_class_balanced": "P+: paper head + class balance"}
DATASET_NAMES = {"androids_interview": "Androids", "cmdc": "CMDC", "d3tec": "D3TEC", "daic": "DAIC", "turkish": "Turkish"}


def _job(args):
    features, output, caches, variant, seed = args
    head, balancing, _ = VARIANTS[variant]
    run(features, output, "frozen", caches, (head,), balancing, seed)
    return str(output)


def _variant_dir(root: Path, variant: str, seed: int | None) -> Path:
    return root / "runs" / (variant if seed is None else f"{variant}_s{seed}")


def _runs(seeds: list[int]) -> list[tuple[str, int | None]]:
    return [(v, s) for v, (_, _, seeded) in VARIANTS.items() for s in (seeds if seeded else [None])]


def train_counts(features: Path) -> dict[str, dict[str, int]]:
    """Training subjects (by class) and rows per dataset in one fold."""
    subjects: dict[str, dict[str, int]] = defaultdict(dict)
    rows: dict[str, int] = defaultdict(int)
    for line in (features / "outer_train_rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            subjects[r["dataset"]][r["subject_id"]] = int(r["label"])
            rows[r["dataset"]] += 1
    return {d: {"depressed": sum(v.values()), "non_depressed": len(v) - sum(v.values()),
                "subjects": len(v), "rows": rows[d]} for d, v in sorted(subjects.items())}


def _fold_scores(run_dir: Path, head: str) -> dict[str, dict[str, float]]:
    by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in csv.DictReader((run_dir / head / "predictions_subject_level.csv").open()):
        by_dataset[r["dataset"]].append((int(r["label"]), int(r["prediction"])))
    out = {}
    for d, pairs in by_dataset.items():
        y, p = (np.asarray(v) for v in zip(*pairs))
        m = _metrics(np.sum(y * p), np.sum((1 - y) * p), np.sum(y * (1 - p)), np.sum((1 - y) * (1 - p)))
        out[d] = {k: float(v) for k, v in m.items()}
    return out


def summarize(root: Path, seeds: list[int], features_of: dict[tuple[str, int], Path]) -> dict:
    cv: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    final: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    l2: dict[str, list] = defaultdict(list)
    for balancing, seed in _runs(seeds):
        head = VARIANTS[balancing][0]
        per_seed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for stage, fold in STAGES:
            run_dir = _variant_dir(root, balancing, seed) / f"{stage}_fold_{fold}"
            summary = json.loads((run_dir / "summary.json").read_text())
            if head == "subjmean_l2":
                l2[balancing].append({"seed": seed, "stage": stage, "fold": fold,
                                      "l2": summary["heads"][head]["selection"]["selected_l2_penalty"]})
            for d, m in _fold_scores(run_dir, head).items():
                if stage == "cv":
                    for k, v in m.items():
                        per_seed[d][k].append(v)
                else:
                    for k, v in m.items():
                        final[balancing][f"{d}:{k}"].append(v)
        for d, metrics in per_seed.items():
            for k, folds in metrics.items():
                if len(folds) != 5:
                    raise ValueError(f"{balancing} s{seed} {d}: {len(folds)} CV folds.")
                cv[balancing][d][k].append(float(np.mean(folds)))  # fold-mean, one value per seed
    table = {b: {d: {k: {"mean": float(np.mean(v)), "sd_over_seeds": float(np.std(v)) if len(v) > 1 else None,
                         "n_seeds": len(v)} for k, v in ms.items()} for d, ms in ds.items()} for b, ds in cv.items()}
    datasets = sorted(table[REFERENCE])
    tests = {}
    for b, ref in COMPARISONS:
        diffs = [table[b][d]["macro_f1"]["mean"] - table[ref][d]["macro_f1"]["mean"] for d in datasets]
        tests[f"{b}_vs_{ref}"] = {"variant": b, "versus": ref, "mean_diff": float(np.mean(diffs)),
                                  "per_dataset_diff": dict(zip(datasets, diffs)), **_paired_t(diffs)}
    return {
        "aggregation": "CV: per-fold subject-level binary-strict metrics, mean over 5 folds, then mean over "
                       "oversampling seeds; final: merged final fold, DAIC official test (47 subjects).",
        "cv": table,
        "cv_mean_over_datasets": {b: float(np.mean([table[b][d]["macro_f1"]["mean"] for d in datasets])) for b in table},
        "final": {b: {k: {"mean": float(np.mean(v)), "sd_over_seeds": float(np.std(v)) if len(v) > 1 else None}
                      for k, v in ms.items()} for b, ms in final.items()},
        "paired_t_vs_reference": tests,
        "selected_l2": l2,
        "train_counts": {f"{stage}_fold_{fold}": train_counts(features_of[(stage, fold)]) for stage, fold in STAGES},
    }


def render(summary: dict, meta: dict) -> str:
    cv = summary["cv"]
    datasets = sorted(next(iter(cv.values())))
    order = [b for b in VARIANTS if b in cv]
    lines = ["# Merged audio-only, frozen features, new head: balancing policies", "",
             f"Source commit `{meta['commit']}`. Route `{meta['route']}`. Oversampling seeds {meta['seeds']}.", "",
             "Cells: macro-F1 / positive-F1 / UAR, subject level, binary strict, CV fold-mean "
             "(seeded variants: mean over seeds, ± SD of macro-F1 over seeds).", "",
             "| Dataset | " + " | ".join(LABELS[b] for b in order) + " |",
             "|---|" + "---|" * len(order)]
    for d in datasets:
        cells = []
        for b in order:
            m = cv[b][d]
            sd = m["macro_f1"]["sd_over_seeds"]
            cells.append(f"{m['macro_f1']['mean']:.3f}{f' ±{sd:.3f}' if sd is not None else ''} / "
                         f"{m['positive_f1']['mean']:.3f} / {m['uar']['mean']:.3f}")
        lines.append(f"| {DATASET_NAMES.get(d, d)} | " + " | ".join(cells) + " |")
    lines.append("| Mean macro-F1 | " + " | ".join(f"{summary['cv_mean_over_datasets'][b]:.3f}" for b in order) + " |")
    daic = [summary["final"][b].get("daic:macro_f1") for b in order]
    lines.append("| DAIC merged final (official test) | "
                 + " | ".join("—" if v is None else f"{v['mean']:.3f}" for v in daic) + " |")
    lines += ["", "Paired t-tests of CV macro-F1 over 5 datasets (no correction):", ""]
    for t in summary["paired_t_vs_reference"].values():
        p = "—" if t["p"] is None else f"{t['p']:.4f}"
        lines.append(f"- {LABELS[t['variant']]} vs {LABELS[t['versus']]}: mean diff {t['mean_diff']:+.4f}, p = {p}")
    lines += ["", "Training subjects per fold (depressed / non-depressed, rows):", "",
              "| Fold | " + " | ".join(DATASET_NAMES.get(d, d) for d in datasets) + " |", "|---|" + "---|" * len(datasets)]
    for fold, counts in summary["train_counts"].items():
        lines.append(f"| {fold} | " + " | ".join(
            f"{counts[d]['depressed']} / {counts[d]['non_depressed']} ({counts[d]['rows']})" for d in datasets) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged-features-root", required=True, type=Path)
    parser.add_argument("--frozen-caches-root", required=True, type=Path)
    parser.add_argument("--frozen-daic-audio", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--language", default="native")
    parser.add_argument("--modality", default="audio_only")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Seeds of the oversampling and SMOTE variants.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_root}.")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    caches = frozen_caches(args.frozen_caches_root, args.frozen_daic_audio, args.language, args.modality)
    features_of = {(stage, fold): features_dir(args.merged_features_root, args.language, args.modality, 1337, stage, fold)
                   for stage, fold in STAGES}
    jobs = [(features_of[(stage, fold)], _variant_dir(args.output_root, v, s) / f"{stage}_fold_{fold}", caches, v, s or 0)
            for v, s in _runs(seeds) for stage, fold in STAGES]
    print(f"{len(jobs)} head runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(_job, j) for j in jobs]):
            print(future.result(), flush=True)
    meta = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=PROJECT_ROOT).stdout.strip(),
            "route": f"{args.language}_{args.modality}", "seeds": seeds,
            "merged_features_root": str(args.merged_features_root), "row_index_seed": 1337,
            "frozen_caches": [str(c) for c in caches]}
    summary = {"meta": meta, **summarize(args.output_root, seeds, features_of)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_root / "summary.md").write_text(render(summary, meta))
    print(render(summary, meta))


if __name__ == "__main__":
    main()
