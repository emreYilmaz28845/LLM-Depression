#!/usr/bin/env python3
"""Test whether removing per-dataset feature shift helps the merged new head.

Frozen base features, one merged route (default native audio_only), 5 CV folds
plus the merged final fold. The new head (``subjmean_l2``, balancing
``dataset_class_weight`` by default, ``--balancing`` to change it) is fitted in four variants (see
``merged_subject_mean_heads``):

- ``B``: one global scaler (the current head; a control that must reproduce
  ``--reference-root`` prediction for prediction when given);
- ``S``: one scaler per dataset, fit on that dataset's training subjects only;
- ``I``: global scaler plus a one-hot dataset indicator (same L2 penalty);
- ``SI``: both;
- ``E``: global scaler plus LEACE erasure of the dataset from the subject means
  (opt-in through ``--variants``; see ``baselines.concept_erasure``);
- ``X`` / ``XI``: the XGBoost head ``subjmean_xgb`` instead of the L2 head, without / with the
  one-hot dataset indicator (opt-in through ``--variants``).

Also loads the standalone frozen new head of the same route: the advisor export
test-split predictions per CV fold, and for DAIC the train-CV-selected head on
the official test (same sources as ``head_ablation_stats``).

Writes every fold run, then ``summary.json`` and ``summary.md``: per-dataset
fold-mean macro-F1 / positive-F1 / UAR (subject level, binary strict), the DAIC
merged-final official-test cell, and paired t-tests of macro-F1 over the 5
datasets (no correction): every variant against ``B`` on CV, and every variant
against standalone (DAIC: merged final vs standalone on the official test).
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
from baselines.merged_subject_mean_heads import BALANCING as BALANCING_CHOICES, run  # noqa: E402
from baselines.run_merged_balancing_matrix import DATASET_NAMES, _fold_scores, train_counts  # noqa: E402
from baselines.run_merged_head_matrix import STAGES, features_dir, frozen_caches  # noqa: E402

# variant -> (normalization, dataset_indicator, erasure)
VARIANTS = {"B": ("global", False, "none"), "S": ("per_dataset", False, "none"), "I": ("global", True, "none"),
            "SI": ("per_dataset", True, "none"), "E": ("global", False, "leace"),
            "X": ("global", False, "none"), "XI": ("global", True, "none")}
DEFAULT_VARIANTS = ("B", "S", "I", "SI")
REFERENCE = "B"
STANDALONE = "standalone"
LABELS = {"B": "B: global scaler (current)", "S": "S: per-dataset scaler", "I": "I: dataset indicator",
          "SI": "S+I: per-dataset scaler + indicator", "E": "E: LEACE dataset erasure",
          "X": "X: XGBoost", "XI": "X+I: XGBoost + dataset indicator",
          STANDALONE: "Standalone frozen + new head"}
HEAD = "subjmean_l2"
XGB_HEAD = "subjmean_xgb"


def head_of(variant: str) -> str:
    return XGB_HEAD if variant.startswith("X") else HEAD
BALANCING = "dataset_class_weight"
DAIC = "daic"


def _job(args):
    features, output, caches, variant, balancing, seed = args
    normalization, indicator, erasure = VARIANTS[variant]
    run(features, output, "frozen", caches, (head_of(variant),), balancing, seed, normalization, indicator, erasure)
    return str(output)


def _scores(pairs: list[tuple[int, int]]) -> dict[str, float]:
    y, p = (np.asarray(v) for v in zip(*pairs))
    m = _metrics(np.sum(y * p), np.sum((1 - y) * p), np.sum(y * (1 - p)), np.sum((1 - y) * (1 - p)))
    return {k: float(v) for k, v in m.items()}


def load_standalone(export_csv_root: Path, daic_traincv_root: Path, language: str, modality: str) -> dict:
    """{dataset: {fold: [(label, prediction)]}} of the standalone frozen new head, plus subject ids."""
    pairs: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    subjects: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted(export_csv_root.glob("*/predictions/*__predictions__frozen__new_head__1run.csv")):
        for r in csv.DictReader(path.open()):
            if r["split"] != "test" or r["dataset"] == DAIC or r["modality"] != modality or r["language"] != language:
                continue
            pairs[r["dataset"]][int(r["fold"])].append((int(r["label"]), int(r["predicted"])))
            subjects[r["dataset"]][int(r["fold"])].add(r["subject_id"])
    for r in csv.DictReader((daic_traincv_root / f"{modality}_native" / "fold_0" / "predictions_subject_level.csv").open()):
        pairs[DAIC][0].append((int(r["label"]), int(r["prediction"])))
        subjects[DAIC][0].add(str(r["subject_id"]).split("::", 1)[-1])
    return {"pairs": pairs, "subjects": subjects}


def holdout_subjects(features: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for line in (features / "outer_holdout_rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["dataset"]].add(str(r["subject_id"]).split("::", 1)[1])
    return out


def fold_identity(standalone: dict, features_of: dict) -> dict:
    """Do standalone test subjects equal the merged holdout subjects, fold by fold?"""
    out = {}
    for d, folds in sorted(standalone["subjects"].items()):
        rows = []
        for fold, subjects in sorted(folds.items()):
            merged = holdout_subjects(features_of[("final", 0) if d == DAIC else ("cv", fold)])[d]
            rows.append({"fold": fold, "merged_stage": "final" if d == DAIC else "cv", "standalone": len(subjects),
                         "merged": len(merged), "identical": subjects == merged})
        out[d] = {"identical": all(r["identical"] for r in rows), "folds": rows}
    return out


def compare_reference(root: Path, reference: Path) -> dict:
    """Prediction-for-prediction check of variant B against an earlier run of the current head."""
    out = {}
    for stage, fold in STAGES:
        name = f"{stage}_fold_{fold}"
        mine = list(csv.DictReader((root / "runs" / REFERENCE / name / HEAD / "predictions_subject_level.csv").open()))
        theirs = list(csv.DictReader((reference / name / HEAD / "predictions_subject_level.csv").open()))
        key = lambda r: (r["dataset"], r["subject_id"])  # noqa: E731
        a, b = {key(r): r for r in mine}, {key(r): r for r in theirs}
        diff = max(abs(float(a[k]["mean_logit"]) - float(b[k]["mean_logit"])) for k in a) if a.keys() == b.keys() else None
        out[name] = {"same_subjects": a.keys() == b.keys(),
                     "same_predictions": a.keys() == b.keys() and all(a[k]["prediction"] == b[k]["prediction"] for k in a),
                     "max_abs_logit_diff": diff}
    return {"reference": str(reference), "identical": all(v["same_predictions"] for v in out.values()), "folds": out}


def summarize(root: Path, standalone: dict, variants: tuple[str, ...] = DEFAULT_VARIANTS) -> dict:
    cv: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    final: dict[str, dict[str, dict[str, float]]] = {}
    l2: dict[str, list] = defaultdict(list)
    indicator: dict[str, list] = defaultdict(list)
    for variant in variants:
        for stage, fold in STAGES:
            run_dir = root / "runs" / variant / f"{stage}_fold_{fold}"
            head = json.loads((run_dir / "summary.json").read_text())["heads"][head_of(variant)]
            selection = head["selection"]
            l2[variant].append({"stage": stage, "fold": fold,
                                "l2": selection.get("selected_l2_penalty", selection.get("selected_params"))})
            weights = head["fit_info"].get("indicator_weights", head["fit_info"].get("indicator_gain"))
            if weights is not None:
                indicator[variant].append({"stage": stage, "fold": fold, "weights": weights})
            scores = _fold_scores(run_dir, head_of(variant))
            if stage == "final":
                final[variant] = scores
                continue
            for d, m in scores.items():
                for k, v in m.items():
                    cv[variant][d][k].append(v)
    for d, folds in standalone["pairs"].items():
        for fold in sorted(folds):
            for k, v in _scores(folds[fold]).items():
                cv[STANDALONE][d][k].append(v)
    table = {b: {d: {k: float(np.mean(v)) for k, v in ms.items()} for d, ms in ds.items()} for b, ds in cv.items()}
    n_folds = {b: {d: len(ms["macro_f1"]) for d, ms in ds.items()} for b, ds in cv.items()}
    datasets = sorted(table[REFERENCE])
    # "Headline" per dataset: CV fold-mean, except DAIC = official test (merged final / standalone).
    headline = {b: {d: (final[b][d] if d == DAIC else table[b][d]) for d in datasets} for b in variants}
    headline[STANDALONE] = {d: table[STANDALONE][d] for d in datasets}
    tests = {}
    for b in variants:
        if b != REFERENCE:
            diffs = [table[b][d]["macro_f1"] - table[REFERENCE][d]["macro_f1"] for d in datasets]
            tests[f"{b}_vs_{REFERENCE}_cv"] = {"variant": b, "versus": REFERENCE, "basis": "merged CV fold-mean, 5 datasets",
                                               "mean_diff": float(np.mean(diffs)),
                                               "per_dataset_diff": dict(zip(datasets, diffs)), **_paired_t(diffs)}
        diffs = [headline[b][d]["macro_f1"] - headline[STANDALONE][d]["macro_f1"] for d in datasets]
        tests[f"{b}_vs_{STANDALONE}"] = {"variant": b, "versus": STANDALONE,
                                         "basis": "CV fold-mean; DAIC official test (merged final vs standalone)",
                                         "mean_diff": float(np.mean(diffs)),
                                         "per_dataset_diff": dict(zip(datasets, diffs)), **_paired_t(diffs)}
    return {
        "aggregation": "Per-fold subject-level binary-strict metrics, mean over the 5 CV folds. Standalone DAIC and "
                       "every merged 'final' cell: official test (47 subjects), one fold.",
        "cv": table,
        "cv_n_folds": n_folds,
        "cv_mean_over_datasets": {b: float(np.mean([table[b][d]["macro_f1"] for d in datasets]))
                                  for b in variants},
        "variants": list(variants),
        "final": final,
        "headline": headline,
        "headline_mean_over_datasets": {b: float(np.mean([headline[b][d]["macro_f1"] for d in datasets]))
                                        for b in headline},
        "paired_t": tests,
        "selected_l2": l2,
        "indicator_weights": indicator,
    }


def render(summary: dict, meta: dict) -> str:
    cv, headline = summary["cv"], summary["headline"]
    variants = tuple(summary.get("variants", DEFAULT_VARIANTS))
    datasets = sorted(cv[REFERENCE])
    order = list(variants) + [STANDALONE]
    cell = lambda m: f"{m['macro_f1']:.3f} / {m['positive_f1']:.3f} / {m['uar']:.3f}"  # noqa: E731
    lines = [f"# Merged {meta['route']}, frozen features, new head: per-dataset shift removal", "",
             f"Source commit `{meta['commit']}`. Route `{meta['route']}`. Balancing `{BALANCING}`.", "",
             "Cells: macro-F1 / positive-F1 / UAR, subject level, binary strict. Merged rows: CV fold-mean. "
             "Standalone: CV fold-mean over the same holdout subjects; DAIC standalone is the official test.", "",
             "| Dataset | " + " | ".join(LABELS[b] for b in order) + " |", "|---|" + "---|" * len(order)]
    for d in datasets:
        lines.append(f"| {DATASET_NAMES.get(d, d)} | " + " | ".join(cell(cv[b][d]) for b in order) + " |")
    lines.append("| Mean macro-F1 (CV) | " + " | ".join(
        f"{summary['cv_mean_over_datasets'][b]:.3f}" for b in variants) + " | — |")
    lines.append("| DAIC official test | " + " | ".join(cell(summary["final"][b][DAIC]) for b in variants)
                 + f" | {cell(headline[STANDALONE][DAIC])} |")
    lines.append("| Mean macro-F1 (DAIC = official test) | " + " | ".join(
        f"{summary['headline_mean_over_datasets'][b]:.3f}" for b in order) + " |")
    lines += ["", "Paired t-tests of macro-F1 over 5 datasets (no correction):", ""]
    for t in summary["paired_t"].values():
        p = "—" if t["p"] is None else f"{t['p']:.4f}"
        lines.append(f"- {LABELS[t['variant']]} vs {LABELS[t['versus']]} ({t['basis']}): "
                     f"mean diff {t['mean_diff']:+.4f}, p = {p}")
    lines += ["", f"Fold identity, standalone test vs merged holdout subjects: "
              + ", ".join(f"{DATASET_NAMES.get(d, d)} {'identical' if v['identical'] else 'DIFFERENT'}"
                          for d, v in meta["fold_identity"].items())]
    if meta.get("reference_check"):
        lines.append(f"Control B vs `{meta['reference_check']['reference']}`: "
                     f"{'identical predictions' if meta['reference_check']['identical'] else 'DIFFERENT'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged-features-root", required=True, type=Path)
    parser.add_argument("--frozen-caches-root", required=True, type=Path)
    parser.add_argument("--frozen-daic-audio", required=True, type=Path)
    parser.add_argument("--standalone-export-csv", required=True, type=Path)
    parser.add_argument("--standalone-daic-traincv", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path,
                        help="Earlier frozen run of the current head for this route (…/frozen/<language>_<modality>).")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--language", default="native")
    parser.add_argument("--modality", default="audio_only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--balancing", default=BALANCING, choices=BALANCING_CHOICES)
    parser.add_argument("--oversample-seed", type=int, default=0)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                        help=f"Comma list from {','.join(VARIANTS)}; must include {REFERENCE}.")
    args = parser.parse_args()
    variants = tuple(v for v in args.variants.split(",") if v)
    if REFERENCE not in variants or any(v not in VARIANTS for v in variants):
        raise SystemExit(f"--variants must be drawn from {sorted(VARIANTS)} and include {REFERENCE}.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_root}.")
    caches = frozen_caches(args.frozen_caches_root, args.frozen_daic_audio, args.language, args.modality)
    features_of = {(stage, fold): features_dir(args.merged_features_root, args.language, args.modality, 1337, stage, fold)
                   for stage, fold in STAGES}
    standalone = load_standalone(args.standalone_export_csv, args.standalone_daic_traincv, args.language, args.modality)
    identity = fold_identity(standalone, features_of)
    jobs = [(features_of[(stage, fold)], args.output_root / "runs" / v / f"{stage}_fold_{fold}", caches, v,
             args.balancing, args.oversample_seed)
            for v in variants for stage, fold in STAGES]
    print(f"{len(jobs)} head runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(_job, j) for j in jobs]):
            print(future.result(), flush=True)
    meta = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=PROJECT_ROOT).stdout.strip(),
            "route": f"{args.language}_{args.modality}", "balancing": args.balancing, "oversample_seed": args.oversample_seed,
            "merged_features_root": str(args.merged_features_root), "row_index_seed": 1337,
            "frozen_caches": [str(c) for c in caches],
            "standalone_export_csv": str(args.standalone_export_csv),
            "standalone_daic_traincv": str(args.standalone_daic_traincv),
            "fold_identity": identity,
            "train_counts": {f"{s}_fold_{f}": train_counts(features_of[(s, f)]) for s, f in STAGES},
            "reference_check": compare_reference(args.output_root, args.reference_root) if args.reference_root else None}
    summary = {"meta": meta, **summarize(args.output_root, standalone, variants)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_root / "summary.md").write_text(render(summary, meta))
    print(render(summary, meta))


if __name__ == "__main__":
    main()
