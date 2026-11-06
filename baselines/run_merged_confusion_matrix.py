#!/usr/bin/env python3
"""Merged critic-confusion head (colleague's source-suppression recipe) on frozen features.

One merged route (default native text_only), the 5 merged CV folds plus the
merged final fold (DAIC official test), the same row index and frozen
base-model vectors as ``merged_subject_mean_heads``. Two stages per fold
(``baselines.source_confusion_head``): a critic-confusion encoder, then the
usual L2 head on its frozen representation; snapshot step and L2 penalty are
chosen on the grouped inner folds. Variants (one run per seed): ``task`` (no
source terms), ``cond<strength>`` (critic confusion), ``comb<strength>``
(confusion + MMD + cross-source contrastive).

References in the same table: ``B`` (current L2 head, ``--reference-root``),
optionally the DANN runs ``M`` and ``Dsel`` (``--dann-root``), and standalone.

Dataset recovery: fresh logistic / RBF-SVM / MLP attackers fit on outer-train
``z`` and scored on held-out ``z`` (balanced accuracy, chance 0.20), on the 5
CV folds only (the final fold holds out DAIC alone). ``max`` is the strongest
of the 9 attackers (3 families x all / y=0 / y=1).

Per-dataset numbers: subject level, binary strict, CV fold-mean; DAIC is the
merged final fold on the official test. Seeded variants: mean over seeds of the
per-seed headline. Paired t-tests of macro-F1 over the 5 datasets, no correction.
"""

from __future__ import annotations

import argparse
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

from baselines.head_ablation_stats import _paired_t  # noqa: E402
from baselines.merged_subject_mean_heads import _load_features, _write_csv, frozen_matrix, load_frozen_index  # noqa: E402
from baselines.run_merged_balancing_matrix import DATASET_NAMES  # noqa: E402
from baselines.run_merged_dann_matrix import _table  # noqa: E402
from baselines.run_merged_domainshift_matrix import fold_identity, load_standalone  # noqa: E402
from baselines.run_merged_head_matrix import STAGES, features_dir, frozen_caches  # noqa: E402
from baselines.source_confusion_head import VARIANT_GRID, ConfusionConfig, fit_merged_fold, variant_name  # noqa: E402

HEAD = "subjmean_confusion"
REFERENCE_HEAD = "subjmean_l2"
DANN_HEAD = "subjmean_dann"
B, STANDALONE, DAIC = "B", "standalone", "daic"
_DATA: dict = {}


def _write_run(output: Path, result: dict, meta: dict) -> None:
    head_dir = output / HEAD
    head_dir.mkdir(parents=True)
    _write_csv(result["predictions"], head_dir / "predictions_subject_level.csv")
    (head_dir / "metrics_by_dataset.json").write_text(json.dumps(result["metrics_by_dataset"], indent=2))
    (output / "summary.json").write_text(json.dumps(
        {**meta, **{k: v for k, v in result.items() if k != "predictions"}}, indent=2, default=float))


def _job(args):
    stage, fold, seed, root, variants = args
    import torch

    torch.set_num_threads(int(os.environ.get("DANN_TORCH_THREADS", "2")))
    x_train, train_rows, x_holdout, holdout_rows = _DATA[(stage, fold)]
    result = fit_merged_fold(x_train, train_rows, x_holdout, holdout_rows, seed, variants)
    meta = {"stage": stage, "fold": fold, "seed": seed, "inner_assignments_hash": result["inner_assignments_hash"],
            "reference_attacks": result["reference_attacks"]}
    for name, r in result["variants"].items():
        _write_run(Path(root) / name / f"s{seed}" / f"{stage}_fold_{fold}", r, {**meta, "variant": name})
    return f"{stage}_fold_{fold} s{seed} done"


def _seed_mean_table(per_seed: list[dict], seeds: list[int]) -> dict:
    datasets = sorted(per_seed[0]["headline"])
    return {"headline": {d: {k: float(np.mean([t["headline"][d][k] for t in per_seed]))
                             for k in per_seed[0]["headline"][d]} for d in datasets},
            "headline_sd_over_seeds": {d: float(np.std([t["headline"][d]["macro_f1"] for t in per_seed], ddof=1))
                                       if len(seeds) > 1 else None for d in datasets}}


def summarize(root: Path, reference: Path, dann_root: Path | None, standalone: dict, variants: list[str],
              seeds: list[int]) -> dict:
    tables = {B: _table(lambda s, f: (reference / f"{s}_fold_{f}", REFERENCE_HEAD))}
    seed_tables: dict[str, list[dict]] = {}
    sources = [(v, root / "runs" / v, HEAD) for v in variants]
    if dann_root is not None:
        sources = [(f"DANN-{v}", dann_root / "runs" / v, DANN_HEAD) for v in ("M", "Dsel")] + sources
    for name, base, head in sources:
        seed_tables[name] = [_table(lambda s, f, base=base, seed=seed, head=head:
                                    (base / f"s{seed}" / f"{s}_fold_{f}", head)) for seed in seeds]
        tables[name] = _seed_mean_table(seed_tables[name], seeds)
    from baselines.run_merged_domainshift_matrix import _scores
    std = {}
    for d, folds in standalone["pairs"].items():
        per_fold = [_scores(folds[f]) for f in sorted(folds)]
        std[d] = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
    tables[STANDALONE] = {"headline": std}
    datasets = sorted(tables[B]["headline"])
    means = {name: float(np.mean([t["headline"][d]["macro_f1"] for d in datasets])) for name, t in tables.items()}
    seed_means = {name: [float(np.mean([t["headline"][d]["macro_f1"] for d in datasets])) for t in ts]
                  for name, ts in seed_tables.items()}
    tests = {}
    for v in variants:
        refs = [B, STANDALONE] + (["task"] if v != "task" and "task" in variants else []) + \
            (["DANN-M"] if "DANN-M" in tables else [])
        for ref in refs:
            diffs = [tables[v]["headline"][d]["macro_f1"] - tables[ref]["headline"][d]["macro_f1"] for d in datasets]
            tests[f"{v}_vs_{ref}"] = {"variant": v, "versus": ref, "mean_diff": float(np.mean(diffs)),
                                      "per_dataset_diff": dict(zip(datasets, diffs)), **_paired_t(diffs)}
    attacks, selections = {}, {}
    reference_attacks = defaultdict(list)
    for v in variants:
        vals = defaultdict(list)
        steps, l2s = [], []
        for seed in seeds:
            for stage, fold in STAGES:
                s = json.loads((root / "runs" / v / f"s{seed}" / f"{stage}_fold_{fold}" / "summary.json").read_text())
                steps.append(s["selection"]["selected_step"])
                l2s.append(s["selection"]["selected_l2_penalty"])
                if s.get("dataset_attacks"):
                    for k, a in s["dataset_attacks"].items():
                        vals[k].append(a["balanced_accuracy"] if isinstance(a, dict) else a)
                if v == variants[0] and s.get("reference_attacks"):
                    for ref_name, ra in s["reference_attacks"].items():
                        reference_attacks[ref_name].append(ra)
        attacks[v] = {k: float(np.mean(x)) for k, x in vals.items()} | {"n_runs": len(vals.get("max", []))}
        selections[v] = {"selected_step": dict(sorted(Counter(steps).items())),
                         "selected_l2": dict(sorted(Counter(l2s).items()))}
    ref_summary = {}
    for name, runs in reference_attacks.items():
        keys = [k for k in runs[0] if k != "chance"]
        ref_summary[name] = {k: float(np.mean([r[k]["balanced_accuracy"] if isinstance(r[k], dict) else r[k]
                                               for r in runs])) for k in keys}
    return {"aggregation": "Headline per dataset: CV fold-mean, DAIC = merged final on the official test (47). "
                           "Seeded variants: mean over seeds of the per-seed headline. Attacks: mean over the "
                           "5 CV folds x seeds of held-out balanced accuracy.",
            "datasets": datasets, "tables": tables, "mean_macro_f1": means, "seed_mean_macro_f1": seed_means,
            "paired_t": tests, "dataset_attacks": attacks, "reference_attacks": ref_summary,
            "selections": selections}


LABELS = {B: "B (L2 head)", STANDALONE: "Standalone", "DANN-M": "DANN M (no reversal)",
          "DANN-Dsel": "DANN Dsel", "task": "task (no source loss)"}


def _label(v: str) -> str:
    if v in LABELS:
        return LABELS[v]
    if v.startswith("cond"):
        return f"confusion, s={v[4:]}"
    if v.startswith("comb"):
        return f"conf+MMD+SupCon, s={v[4:]}"
    return v


def render(summary: dict, meta: dict, variants: list[str]) -> str:
    t, datasets = summary["tables"], summary["datasets"]
    order = [B] + [v for v in ("DANN-M", "DANN-Dsel") if v in t] + variants + [STANDALONE]
    cell = lambda m: f"{m['macro_f1']:.3f} / {m['positive_f1']:.3f}"  # noqa: E731
    lines = [f"# Merged {meta['route']}: critic-confusion encoder + L2 head", "",
             f"Runs at commit `{meta.get('runs_commit') or meta['commit']}` (summary `{meta['commit']}`). "
             f"Seeds {meta['seeds']}. Cells: macro-F1 / positive-F1, subject level, binary strict; CV fold-mean, "
             "DAIC = official test; seeded variants averaged over seeds.", "",
             "| Dataset | " + " | ".join(_label(v) for v in order) + " |", "|---|" + "---|" * len(order)]
    for d in datasets:
        lines.append(f"| {DATASET_NAMES.get(d, d)} | " + " | ".join(cell(t[v]["headline"][d]) for v in order) + " |")
    lines.append("| Mean macro-F1 | " + " | ".join(f"{summary['mean_macro_f1'][v]:.3f}" for v in order) + " |")
    sd = summary["seed_mean_macro_f1"]
    lines.append("| Seed SD of mean | " + " | ".join(
        f"{np.std(sd[v], ddof=1):.3f}" if v in sd and len(sd[v]) > 1 else "—" for v in order) + " |")
    lines += ["", "Dataset recovery on held-out subjects (balanced accuracy, chance 0.20; mean over 5 CV folds x seeds). "
              "max = strongest of 9 fresh attackers:", "",
              "| Representation | linear | RBF | MLP | max of 9 |", "|---|---|---|---|---|"]
    for name, a in summary["reference_attacks"].items():
        lines.append(f"| input: {name} | {a['all_linear']:.3f} | {a['all_rbf']:.3f} | {a['all_mlp']:.3f} | {a['max']:.3f} |")
    for v in variants:
        a = summary["dataset_attacks"][v]
        lines.append(f"| z: {_label(v)} | {a['all_linear']:.3f} | {a['all_rbf']:.3f} | {a['all_mlp']:.3f} | {a['max']:.3f} |")
    lines += ["", "Selections (counts over folds x seeds):", ""]
    for v in variants:
        s = summary["selections"][v]
        lines.append(f"- {_label(v)}: step {s['selected_step']}; L2 {s['selected_l2']}")
    lines += ["", "Paired t-tests of macro-F1 over 5 datasets (no correction):", ""]
    for x in summary["paired_t"].values():
        p = "—" if x["p"] is None else f"{x['p']:.3f}"
        lines.append(f"- {_label(x['variant'])} vs {_label(x['versus'])}: {x['mean_diff']:+.3f}, p = {p}")
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
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--dann-root", type=Path, default=None, help="DANN output root for this route (optional).")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--language", default="native")
    parser.add_argument("--modality", default="text_only")
    parser.add_argument("--variants", default=",".join(f"{m}:{s:g}" for m, s in VARIANT_GRID),
                        help="method:strength list")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only-job", default=None, help="stage,fold,seed: run one job (timing smoke).")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    grid = tuple((m, float(s)) for m, s in (x.split(":") for x in args.variants.split(",") if x))
    variants = [variant_name(m, s) for m, s in grid]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.summarize_only and not (args.output_root / "runs").is_dir():
        raise SystemExit(f"--summarize-only needs existing runs under {args.output_root}.")
    if not args.summarize_only and args.output_root.exists() and any(args.output_root.iterdir()):
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
    jobs = [(stage, fold, seed, str(args.output_root / "runs"), grid) for seed in seeds for stage, fold in STAGES]
    if args.only_job:
        stage, fold, seed = args.only_job.split(",")
        jobs = [j for j in jobs if (j[0], j[1], j[2]) == (stage, int(fold), int(seed))]
    if args.summarize_only:
        jobs = []
    print(f"{len(jobs)} head runs", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:  # fork: workers inherit _DATA
        for future in as_completed([pool.submit(_job, j) for j in jobs]):
            print(future.result(), flush=True)
    if args.only_job:
        return
    standalone = load_standalone(args.standalone_export_csv, args.standalone_daic_traincv, args.language, args.modality)
    meta = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=PROJECT_ROOT).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                             capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()),
            "route": f"{args.language}_{args.modality}", "seeds": seeds, "variants": variants,
            "merged_features_root": str(args.merged_features_root), "row_index_seed": 1337,
            "frozen_caches": [str(c) for c in caches], "frozen_audit": audit,
            "reference_root": str(args.reference_root), "dann_root": str(args.dann_root) if args.dann_root else None,
            "standalone_export_csv": str(args.standalone_export_csv),
            "standalone_daic_traincv": str(args.standalone_daic_traincv),
            "config_defaults": ConfusionConfig().__dict__, "fold_identity": fold_identity(standalone, features_of)}
    previous = args.output_root / "summary.json"
    meta["runs_commit"] = (json.loads(previous.read_text())["meta"].get("runs_commit")
                           if args.summarize_only and previous.exists() else
                           None if args.summarize_only else meta["commit"])
    summary = {"meta": meta, **summarize(args.output_root, args.reference_root, args.dann_root, standalone,
                                         variants, seeds)}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    text = render(summary, meta, variants)
    (args.output_root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
