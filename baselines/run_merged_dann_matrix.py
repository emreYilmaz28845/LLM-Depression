#!/usr/bin/env python3
"""Merged domain-adversarial (gradient reversal) head on frozen features, against the current L2 head.

One merged route (default native text_only), the 5 merged CV folds plus the
merged final fold (DAIC official test), the same row index and frozen
base-model vectors as ``merged_subject_mean_heads`` (source ``frozen``).
Two stages per fold (``baselines.dann_subject_head``): a DANN encoder trained
with dataset x class oversampling, then the usual L2 head on its frozen
representation ``z``; epochs and the L2 penalty are chosen jointly on the
grouped inner folds. Variants, one run per seed:

- ``M``: encoder trained without gradient reversal (lambda 0);
- ``D<lambda>``: encoder trained with gradient reversal, maximum lambda;
- ``Dsel``: per fold and seed, the lambda chosen on the inner folds.

The current head ``B`` (``subjmean_l2``, global scaler) is read from
``--reference-root``; the standalone frozen new head from the advisor export
(same sources as ``run_merged_domainshift_matrix``).

Per-dataset numbers: subject level, binary strict, CV fold-mean; DAIC is the
merged final fold on the official test. A seeded variant is first summarised
per seed, then averaged over seeds (the SD over seeds is reported). Paired
t-tests of macro-F1 over the 5 datasets, no correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dann_subject_head import LAMBDA_GRID, DannConfig, fit_merged_fold  # noqa: E402
from baselines.head_ablation_stats import _paired_t  # noqa: E402
from baselines.merged_subject_mean_heads import _load_features, _write_csv, frozen_matrix, load_frozen_index  # noqa: E402
from baselines.run_merged_balancing_matrix import DATASET_NAMES, _fold_scores  # noqa: E402
from baselines.run_merged_domainshift_matrix import fold_identity, load_standalone  # noqa: E402
from baselines.run_merged_head_matrix import STAGES, features_dir, frozen_caches  # noqa: E402

HEAD = "subjmean_dann"
REFERENCE_HEAD = "subjmean_l2"
B, STANDALONE, DAIC = "B", "standalone", "daic"
_DATA: dict = {}


SELECTED = "Dsel"


def lambda_name(lam: float) -> str:
    return "M" if lam == 0 else f"D{lam:g}"


def variant_lambda(name: str) -> float:
    if name == "M":
        return 0.0
    if name.startswith("D") and name != SELECTED:
        return float(name[1:])
    raise ValueError(f"Unknown variant {name!r}.")


def _write_run(output: Path, result: dict, meta: dict) -> None:
    head_dir = output / HEAD
    head_dir.mkdir(parents=True)
    _write_csv(result["predictions"], head_dir / "predictions_subject_level.csv")
    (head_dir / "metrics_by_dataset.json").write_text(json.dumps(result["metrics_by_dataset"], indent=2))
    (output / "summary.json").write_text(json.dumps(
        {**meta, **{k: v for k, v in result.items() if k != "predictions"}}, indent=2, default=float))


def _job(args):
    stage, fold, seed, root, lambdas = args
    import torch

    torch.set_num_threads(int(os.environ.get("DANN_TORCH_THREADS", "2")))
    x_train, train_rows, x_holdout, holdout_rows = _DATA[(stage, fold)]
    result = fit_merged_fold(x_train, train_rows, x_holdout, holdout_rows, seed, lambdas)
    meta = {"stage": stage, "fold": fold, "seed": seed, "selected_lambda": result["selected_lambda"],
            "selection_rule": result["selection_rule"], "inner_assignments_hash": result["inner_assignments_hash"]}
    for lam, r in result["per_lambda"].items():
        _write_run(Path(root) / lambda_name(lam) / f"s{seed}" / f"{stage}_fold_{fold}", r,
                   {**meta, "variant": lambda_name(lam), "adv_lambda": lam})
    lam = result["selected_lambda"]
    _write_run(Path(root) / SELECTED / f"s{seed}" / f"{stage}_fold_{fold}", result["per_lambda"][lam],
               {**meta, "variant": SELECTED, "adv_lambda": lam})
    return f"{stage}_fold_{fold} s{seed} selected lambda {lam}"


def _table(run_dir_of) -> dict:
    """{'cv': {dataset: metric means}, 'final': {dataset: metrics}, 'headline': {...}} for one run set."""
    cv: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    final = {}
    for stage, fold in STAGES:
        scores = _fold_scores(*run_dir_of(stage, fold))
        if stage == "final":
            final = scores
            continue
        for d, m in scores.items():
            for k, v in m.items():
                cv[d][k].append(v)
    cv_mean = {d: {k: float(np.mean(v)) for k, v in ms.items()} for d, ms in cv.items()}
    headline = {d: (final[d] if d == DAIC else cv_mean[d]) for d in sorted(cv_mean)}
    return {"cv": cv_mean, "final": final, "headline": headline}


def summarize(root: Path, reference: Path, standalone: dict, variants: list[str], seeds: list[int]) -> dict:
    tables = {B: _table(lambda s, f: (reference / f"{s}_fold_{f}", REFERENCE_HEAD))}
    per_seed: dict[str, list[dict]] = {}
    for v in variants:
        per_seed[v] = [_table(lambda s, f, v=v, seed=seed: (root / "runs" / v / f"s{seed}" / f"{s}_fold_{f}", HEAD))
                       for seed in seeds]
        datasets = sorted(per_seed[v][0]["headline"])
        tables[v] = {"headline": {d: {k: float(np.mean([t["headline"][d][k] for t in per_seed[v]]))
                                      for k in per_seed[v][0]["headline"][d]} for d in datasets},
                     "headline_sd_over_seeds": {d: float(np.std([t["headline"][d]["macro_f1"] for t in per_seed[v]],
                                                               ddof=1)) if len(seeds) > 1 else None
                                                for d in datasets},
                     "cv": {d: {k: float(np.mean([t["cv"][d][k] for t in per_seed[v]]))
                                for k in per_seed[v][0]["cv"][d]} for d in datasets}}
    std = defaultdict(dict)
    for d, folds in standalone["pairs"].items():
        from baselines.run_merged_domainshift_matrix import _scores
        per_fold = [_scores(folds[f]) for f in sorted(folds)]
        std[d] = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
    tables[STANDALONE] = {"headline": dict(std)}
    datasets = sorted(tables[B]["headline"])
    means = {name: float(np.mean([t["headline"][d]["macro_f1"] for d in datasets])) for name, t in tables.items()}
    seed_means = {v: [float(np.mean([t["headline"][d]["macro_f1"] for d in datasets])) for t in per_seed[v]]
                  for v in variants}
    tests = {}
    for v in variants:
        for ref in (B, STANDALONE) + (("M",) if v != "M" and "M" in variants else ()):
            diffs = [tables[v]["headline"][d]["macro_f1"] - tables[ref]["headline"][d]["macro_f1"] for d in datasets]
            tests[f"{v}_vs_{ref}"] = {"variant": v, "versus": ref, "mean_diff": float(np.mean(diffs)),
                                      "per_dataset_diff": dict(zip(datasets, diffs)), **_paired_t(diffs)}
    readout = {}
    for v in variants:
        vals = defaultdict(list)
        for seed in seeds:
            for stage, fold in STAGES:
                s = json.loads((root / "runs" / v / f"s{seed}" / f"{stage}_fold_{fold}" / "summary.json").read_text())
                vals["probe"].append(s["dataset_readout"]["linear_probe_z_holdout_balanced_accuracy"])
                vals["epochs"].append(s["selection"]["selected_epochs"])
                vals["l2"].append(s["selection"]["selected_l2_penalty"])
                vals["lambda"].append(s["adv_lambda"])
        readout[v] = {"linear_probe_z_holdout_balanced_accuracy": float(np.mean(vals["probe"])),
                      "selected_epochs": dict(sorted(Counter(vals["epochs"]).items())),
                      "selected_l2": dict(sorted(Counter(vals["l2"]).items())),
                      "adv_lambda": dict(sorted(Counter(vals["lambda"]).items()))}
    return {"aggregation": "Headline per dataset: CV fold-mean, DAIC = merged final on the official test (47). "
                           "Seeded variants: mean over seeds of the per-seed headline.",
            "datasets": datasets, "tables": tables, "mean_macro_f1": means, "seed_mean_macro_f1": seed_means,
            "paired_t": tests, "dataset_readout": readout}


def render(summary: dict, meta: dict, variants: list[str]) -> str:
    t, datasets = summary["tables"], summary["datasets"]
    order = [B] + variants + [STANDALONE]
    label = {B: "B (L2 head on raw embedding)", STANDALONE: "Standalone", "M": "M (encoder, no reversal)",
             SELECTED: "Dsel (inner-selected λ)"}
    cell = lambda m: f"{m['macro_f1']:.3f} / {m['positive_f1']:.3f}"  # noqa: E731
    lines = [f"# Merged {meta['route']}: gradient reversal (DANN) head vs the L2 head", "",
             f"Commit `{meta['commit']}`. Seeds {meta['seeds']}. Cells: macro-F1 / positive-F1, subject level, binary "
             "strict; CV fold-mean, DAIC = official test; seeded variants averaged over seeds.", "",
             "| Dataset | " + " | ".join(label.get(v, f"{v} (reversal, λ={variant_lambda(v)})") for v in order) + " |",
             "|---|" + "---|" * len(order)]
    for d in datasets:
        lines.append(f"| {DATASET_NAMES.get(d, d)} | " + " | ".join(cell(t[v]["headline"][d]) for v in order) + " |")
    lines.append("| Mean macro-F1 | " + " | ".join(f"{summary['mean_macro_f1'][v]:.3f}" for v in order) + " |")
    lines.append("| Seed SD of mean | — | " + " | ".join(
        f"{np.std(summary['seed_mean_macro_f1'][v], ddof=1):.3f}" if len(meta['seeds']) > 1 else "—"
        for v in variants) + " | — |")
    lines += ["", "Linear dataset probe on held-out z (balanced accuracy, chance 0.20) and selections "
              "(counts over folds x seeds):", ""]
    for v in variants:
        r = summary["dataset_readout"][v]
        lines.append(f"- {v}: probe {r['linear_probe_z_holdout_balanced_accuracy']:.3f}; epochs {r['selected_epochs']}; "
                     f"L2 {r['selected_l2']}; λ {r['adv_lambda']}")
    lines += ["", "Paired t-tests of macro-F1 over 5 datasets (no correction):", ""]
    for x in summary["paired_t"].values():
        p = "—" if x["p"] is None else f"{x['p']:.3f}"
        lines.append(f"- {x['variant']} vs {x['versus']}: {x['mean_diff']:+.3f}, p = {p}")
    lines += ["", "Fold identity, standalone test vs merged holdout: " + ", ".join(
        f"{DATASET_NAMES.get(d, d)} {'identical' if v['identical'] else 'DIFFERENT'}"
        for d, v in meta["fold_identity"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged-features-root", required=True, type=Path)
    parser.add_argument("--frozen-caches-root", required=True, type=Path)
    parser.add_argument("--frozen-daic-audio", required=True, type=Path)
    parser.add_argument("--standalone-export-csv", required=True, type=Path)
    parser.add_argument("--standalone-daic-traincv", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path,
                        help="Frozen run of the current L2 head for this route (…/frozen/<language>_<modality>).")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--language", default="native")
    parser.add_argument("--modality", default="text_only")
    parser.add_argument("--lambdas", default=",".join(f"{x:g}" for x in LAMBDA_GRID))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    lambdas = tuple(float(x) for x in args.lambdas.split(",") if x)
    variants = [lambda_name(x) for x in lambdas] + [SELECTED]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_root}.")
    caches = frozen_caches(args.frozen_caches_root, args.frozen_daic_audio, args.language, args.modality)
    index, audit = load_frozen_index(caches)
    features_of = {}
    for stage, fold in STAGES:
        fdir = features_dir(args.merged_features_root, args.language, args.modality, 1337, stage, fold)
        features_of[(stage, fold)] = fdir
        _, tr = _load_features(fdir, "outer_train")
        _, ho = _load_features(fdir, "outer_holdout")
        _DATA[(stage, fold)] = (frozen_matrix(tr, index).astype(np.float64), tr,
                                frozen_matrix(ho, index).astype(np.float64), ho)
    del index
    standalone = load_standalone(args.standalone_export_csv, args.standalone_daic_traincv, args.language, args.modality)
    identity = fold_identity(standalone, features_of)
    jobs = [(stage, fold, seed, str(args.output_root / "runs"), lambdas) for seed in seeds for stage, fold in STAGES]
    print(f"{len(jobs)} head runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:  # fork: workers inherit _DATA
        for future in as_completed([pool.submit(_job, j) for j in jobs]):
            print(future.result(), flush=True)
    meta = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=PROJECT_ROOT).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                             capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()),
            "route": f"{args.language}_{args.modality}", "seeds": seeds, "variants": variants,
            "merged_features_root": str(args.merged_features_root), "row_index_seed": 1337,
            "frozen_caches": [str(c) for c in caches], "frozen_audit": audit,
            "reference_root": str(args.reference_root),
            "standalone_export_csv": str(args.standalone_export_csv),
            "standalone_daic_traincv": str(args.standalone_daic_traincv),
            "dann_config_defaults": DannConfig().__dict__, "fold_identity": identity}
    summary = {"meta": meta, **summarize(args.output_root, args.reference_root, standalone, variants, seeds)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    text = render(summary, meta, variants)
    (args.output_root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
