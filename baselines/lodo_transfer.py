#!/usr/bin/env python3
"""Leave-one-dataset-out transfer of the subject-mean depression head, with and without dataset erasure.

For every target dataset, the head is trained on the other four datasets and
scored on every subject of the target (never seen in any fit). One vector per
subject (``dataset_probe`` loaders). Variants:

- ``B``: one StandardScaler fit on the training subjects;
- ``E``: ``B`` plus LEACE erasure of the (training) dataset concept, fit on
  the standardized training subjects with the head's sample weights;
- ``T``: per-dataset standardization; the target dataset is standardized with
  its own (unlabeled) subject statistics. Transductive: it uses target
  features, never target labels.
- ``TE``: ``T`` followed by LEACE erasure of the (training) dataset concept,
  fit on the per-dataset standardized training subjects with the head's sample
  weights and applied to the standardized target.

Head: L2 logistic (``fit_logistic``), equal dataset mass and balanced classes
(sample weights by default; ``--balancing oversample`` instead copies every
training dataset x class cell up to the largest one, seeded remainder, copies
entered as integer weights; scaler and LEACE are still fit on the original training subjects),
penalty chosen by leave-one-training-dataset-out inner CV (mean inner macro-F1,
then lower BCE, then larger penalty) so selection also rewards transfer.
Threshold 0 on the logit; binary-strict subject-level metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.concept_erasure import fit_leace  # noqa: E402
from baselines.dataset_probe import parse_source  # noqa: E402
from baselines.subject_mean_logreg_head import DEFAULT_L2_GRID, fit_logistic, score_metrics  # noqa: E402

VARIANTS = ("B", "E", "T", "TE")
PAIRED_TESTS = (("E", "B"), ("T", "B"), ("TE", "B"), ("TE", "T"))


def _weights(g: np.ndarray, y: np.ndarray) -> np.ndarray:
    cells = {}
    for d, label in zip(g, y):
        cells[(d, int(label))] = cells.get((d, int(label)), 0) + 1
    datasets = sorted(set(g.tolist()))
    for d in datasets:
        if (d, 0) not in cells or (d, 1) not in cells:
            raise ValueError(f"{d} lacks a class.")
    return np.asarray([1.0 / (len(datasets) * 2 * cells[(d, int(label))]) for d, label in zip(g, y)])


def _standardize(x, ref):
    mean, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-8] = 1.0
    return (x - mean) / sd


def _prepare(variant: str, xtr, gtr, ytr, xte, gte):
    if variant in ("T", "TE"):
        ztr = np.empty_like(xtr)
        for d in set(gtr.tolist()):
            ztr[gtr == d] = _standardize(xtr[gtr == d], xtr[gtr == d])
        zte = np.empty_like(xte)
        for d in set(gte.tolist()):
            zte[gte == d] = _standardize(xte[gte == d], xte[gte == d])
    else:
        ztr, zte = _standardize(xtr, xtr), _standardize(xte, xtr)
    if variant in ("E", "TE"):
        eraser = fit_leace(ztr, gtr, _weights(gtr, ytr))
        ztr, zte = eraser(ztr), eraser(zte)
    return ztr, zte


def _oversample(g: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices that copy every dataset x class cell up to the largest cell (seeded remainder)."""
    _weights(g, y)  # same check: every dataset needs both classes
    cells = {}
    for i, key in enumerate(zip(g.tolist(), y.tolist())):
        cells.setdefault(key, []).append(i)
    target = max(len(v) for v in cells.values())
    out = []
    for key in sorted(cells):
        members = cells[key]
        whole, remainder = divmod(target, len(members))
        out += members * whole + [members[i] for i in sorted(rng.choice(len(members), size=remainder, replace=False))]
    return np.asarray(out)


def _fit_predict(variant, penalty, xtr, gtr, ytr, xte, gte, rng=None):
    ztr, zte = _prepare(variant, xtr, gtr, ytr, xte, gte)
    if rng is None:
        w = _weights(gtr, ytr)
        weight, bias, _ = fit_logistic(ztr, ytr, w / w.sum() * len(w), penalty)
    else:
        # Copies enter as integer weights on the original rows: the same loss, and the SVD inside
        # fit_logistic does not converge on matrices with many duplicated rows.
        copies = np.bincount(_oversample(gtr, ytr, rng), minlength=len(ytr)).astype(float)
        weight, bias, _ = fit_logistic(ztr, ytr, copies, penalty)
    return zte @ weight + bias


def lodo(x: np.ndarray, g: np.ndarray, y: np.ndarray, variant: str, oversample_seed: int | None = None) -> dict:
    # One stream per (target, inner fit), shared by every penalty; None = sample weights.
    rng = (lambda *key: None) if oversample_seed is None else \
        (lambda *key: np.random.default_rng([oversample_seed, *key]))
    out = {}
    for t_index, target in enumerate(sorted(set(g.tolist()))):
        tr, te = g != target, g == target
        xtr, gtr, ytr = x[tr], g[tr], y[tr]
        candidates = []
        for penalty in DEFAULT_L2_GRID:
            f1s, bces = [], []
            for i_index, inner in enumerate(sorted(set(gtr.tolist()))):
                a, b = gtr != inner, gtr == inner
                logits = _fit_predict(variant, penalty, xtr[a], gtr[a], ytr[a], xtr[b], gtr[b],
                                      rng(t_index, i_index))
                f1s.append(score_metrics(ytr[b], logits)["macro_f1"])
                bces.append(float(np.mean(np.logaddexp(0.0, logits) - ytr[b] * logits)))
            candidates.append((float(np.mean(f1s)), -float(np.mean(bces)), penalty))
        best = max(candidates)
        logits = _fit_predict(variant, best[2], xtr, gtr, ytr, x[te], g[te], rng(t_index, 99))
        m = score_metrics(y[te], logits)
        out[target] = {"macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"], "uar": m["binary_strict_uar"],
                       "predicted_positive_rate": float(np.mean(logits > 0)), "true_positive_rate": float(np.mean(y[te])),
                       "selected_l2": best[2], "inner_mean_macro_f1": best[0], "subjects": int(te.sum())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--merged-features-root", type=Path)
    parser.add_argument("--frozen-caches-root", type=Path)
    parser.add_argument("--frozen-daic-audio", type=Path)
    parser.add_argument("--w2v-cache-root", type=Path)
    parser.add_argument("--acoustic-npz", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--balancing", default="weight", choices=("weight", "oversample"))
    parser.add_argument("--oversample-seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.oversample_seed if args.balancing == "oversample" else None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_dir}.")
    result = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=PROJECT_ROOT).stdout.strip(),
              "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                               capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()),
              "l2_grid": list(DEFAULT_L2_GRID), "balancing": args.balancing, "oversample_seed": seed, "sources": {}}
    lines = ["# Leave-one-dataset-out transfer (train on 4, test on the unseen 5th)", "",
             "Cells: macro-F1 / positive-F1, subject level, binary strict, threshold 0. "
             "B = global scaler, E = + LEACE dataset erasure, T = per-dataset standardization (target uses its own "
             "unlabeled statistics), TE = T + LEACE dataset erasure.", ""]
    for spec in args.source:
        name, src = parse_source(spec, args)
        x, g, y = src["x"], np.asarray(src["dataset"]), np.asarray(src["label"])
        res = {v: lodo(x, g, y, v, seed) for v in VARIANTS}
        targets = sorted(res["B"])
        tests = {}
        for v, ref in PAIRED_TESTS:
            diffs = [res[v][t]["macro_f1"] - res[ref][t]["macro_f1"] for t in targets]
            p = float(stats.ttest_1samp(diffs, 0.0).pvalue) if np.std(diffs) > 0 else None
            tests[f"{v}_vs_{ref}"] = {"mean_diff": float(np.mean(diffs)), "p": p, "per_target": dict(zip(targets, diffs))}
        result["sources"][name] = {"subjects": len(y), "dimension": int(x.shape[1]), "provenance": src["provenance"],
                                   "lodo": res, "paired_t": tests,
                                   "mean_macro_f1": {v: float(np.mean([res[v][t]["macro_f1"] for t in targets]))
                                                     for v in VARIANTS}}
        lines += [f"## {name} ({len(y)} subjects, dim {x.shape[1]})", "",
                  "| Target | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
        for t in targets:
            lines.append(f"| {t} | " + " | ".join(f"{res[v][t]['macro_f1']:.3f} / {res[v][t]['positive_f1']:.3f}"
                                                  for v in VARIANTS) + " |")
        lines.append("| Mean macro-F1 | " + " | ".join(f"{result['sources'][name]['mean_macro_f1'][v]:.3f}"
                                                      for v in VARIANTS) + " |")
        for k, t in tests.items():
            lines.append(f"\n{k}: mean diff {t['mean_diff']:+.3f}, p = "
                         + ("—" if t["p"] is None else f"{t['p']:.3f}") + " (paired over 5 targets, no correction)")
        lines.append("")
        print("\n".join(lines[-(len(targets) + 6):]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lodo_results.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / "lodo_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
