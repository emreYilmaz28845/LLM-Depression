#!/usr/bin/env python3
"""Report for ``run_pca_heads_matrix.py``: PCA vs no PCA, Omni vs wav2vec2 under PCA.

A dataset's score is the fold-mean of subject-level binary-strict metrics
(DAIC: one official-test fold). Every pair is checked to report the same
subjects per fold. Tests: two-sided paired t-tests on macro-F1 differences, no
multiplicity correction; primary unit = dataset (n=5), secondary = dataset x
fold cell (n=21). K is never chosen on test: K=50 is fixed in advance, ``ksel``
picks K on the inner validation, other fixed K are a sensitivity table only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
NAMES = {"daic": "DAIC-WOZ (official test)", "d3tec": "D3TEC", "androids_interview": "Androids interview",
         "cmdc": "CMDC", "turkish": "Turkish pooled"}
FEATURES = {"omni": "Qwen3-Omni frozen", "w2v_meanvar": "wav2vec2 mean+var", "w2v_mean": "wav2vec2 mean"}
VARIANTS = ("nopca", "k10", "k20", "k50", "k100", "k200", "ksel")
METRICS = ("macro_f1", "positive_f1", "uar")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(root: Path, variant: str, feature: str, dataset: str) -> dict:
    d = root / variant / feature / dataset
    summary = json.loads((d / "summary.json").read_text())
    folds = []
    for f in summary["folds"]:
        fit = json.loads((d / f"fold_{f['fold']}" / "metrics.json").read_text())["fit_info"]
        folds.append({**f, "subjects": sorted(str(p["subject_id"]) for p in
                                              _jsonl(d / f"fold_{f['fold']}" / "predictions_subject_level.jsonl")),
                      "convergence": fit.get("convergence")})
    return {"folds": folds, "fold_mean": {m: float(np.mean([f[m] for f in folds])) for m in METRICS}}


def paired_t(diffs) -> dict:
    d = np.asarray(diffs, dtype=float)
    out = {"n": int(d.size), "mean_diff": float(d.mean()), "t": None, "p": None}
    if d.size >= 2 and np.ptp(d) > 0:
        r = stats.ttest_1samp(d, 0.0)
        out.update({"t": float(r.statistic), "p": float(r.pvalue)})
    return out


def compare(runs: dict, a: tuple[str, str], b: tuple[str, str]) -> dict:
    """b - a on macro-F1."""
    rows, cells = [], []
    for ds in DATASETS:
        ra, rb = runs[(*a, ds)], runs[(*b, ds)]
        for fa, fb in zip(ra["folds"], rb["folds"], strict=True):
            if fa["subjects"] != fb["subjects"]:
                raise ValueError(f"{ds} fold {fa['fold']}: {a} and {b} report different subjects.")
            cells.append(fb["macro_f1"] - fa["macro_f1"])
        rows.append(rb["fold_mean"]["macro_f1"] - ra["fold_mean"]["macro_f1"])
    return {"a": "/".join(a), "b": "/".join(b), "per_dataset_diff": dict(zip(DATASETS, rows)),
            "dataset_unit": paired_t(rows), "fold_cell_unit": paired_t(cells)}


def _p(x):
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    runs = {(v, f, d): load(args.root, v, f, d) for v in VARIANTS for f in FEATURES for d in DATASETS}

    table = {}
    for v in VARIANTS:
        for f in FEATURES:
            per = {d: runs[(v, f, d)]["fold_mean"] for d in DATASETS}
            table[f"{v}/{f}"] = {"datasets": per, "mean": {m: float(np.mean([per[d][m] for d in DATASETS]))
                                                            for m in METRICS}}
    tests = [compare(runs, ("nopca", f), (v, f)) for f in FEATURES for v in ("k50", "ksel")]
    tests += [compare(runs, (v, "omni"), (v, f)) for v in ("nopca", "k50", "ksel") for f in ("w2v_meanvar", "w2v_mean")]

    diag = {}
    for v in VARIANTS[1:]:
        for f in FEATURES:
            folds = [x for d in DATASETS for x in runs[(v, f, d)]["folds"]]
            diag[f"{v}/{f}"] = {
                "explained_variance_ratio_mean": float(np.mean([x["explained_variance_ratio"] for x in folds])),
                "explained_variance_ratio_min": float(np.min([x["explained_variance_ratio"] for x in folds])),
                "between_share_subspace_mean": float(np.mean([x["between_subject_share_in_subspace"] for x in folds])),
                "between_share_full_mean": float(np.mean([x["between_subject_share_full_space"] for x in folds])),
                "selected_k_counts": dict(Counter(str(x["selected_pca_components"]) for x in folds)),
                "selected_l2_counts": dict(Counter(str(x["selected_l2_penalty"]) for x in folds)),
                "convergence": dict(Counter(str(x["convergence"]) for x in folds)),
            }
    nopca_l2 = {f: dict(Counter(str(x["selected_l2_penalty"]) for d in DATASETS for x in runs[("nopca", f, d)]["folds"]))
                for f in FEATURES}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(
        {"table": table, "tests": tests, "pca_diagnostics": diag, "nopca_selected_l2": nopca_l2,
         "aggregation": "unweighted fold-mean of subject-level binary_strict metrics; mean of 5 datasets"}, indent=2))
    with (args.output_dir / "fold_cells.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["variant", "feature", "dataset", "fold", "macro_f1", "positive_f1", "uar", "selected_l2",
                    "selected_k", "explained_variance_ratio", "between_share_subspace", "n_report"])
        for (v, f, d), r in runs.items():
            for x in r["folds"]:
                w.writerow([v, f, d, x["fold"], x["macro_f1"], x["positive_f1"], x["uar"], x["selected_l2_penalty"],
                            x.get("selected_pca_components", ""), x.get("explained_variance_ratio", ""),
                            x.get("between_subject_share_in_subspace", ""), x["report"]])

    cell = lambda r: f"{r['macro_f1']:.3f} / {r['positive_f1']:.3f}"
    lines = ["## Headline: K=50 vs no PCA (Macro-F1 / Positive-F1)", "",
             "| Dataset | " + " | ".join(f"{FEATURES[f]} no PCA | {FEATURES[f]} PCA-50" for f in FEATURES) + " |",
             "|---|" + "---|" * (2 * len(FEATURES))]
    for d in DATASETS + ("mean",):
        get = (lambda k: table[k]["mean"]) if d == "mean" else (lambda k: table[k]["datasets"][d])
        lines.append(f"| {'Mean of datasets' if d == 'mean' else NAMES[d]} | " +
                     " | ".join(f"{cell(get('nopca/' + f))} | {cell(get('k50/' + f))}" for f in FEATURES) + " |")
    lines += ["", "## Sensitivity: mean Macro-F1 of 5 datasets by K (not used to pick K)", "",
              "| Features | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for f in FEATURES:
        lines.append(f"| {FEATURES[f]} | " + " | ".join(f"{table[f'{v}/{f}']['mean']['macro_f1']:.3f}" for v in VARIANTS) + " |")
    lines += ["", "## Paired t-tests on Macro-F1 (b - a), no correction", ""]
    for t in tests:
        a, b = t["dataset_unit"], t["fold_cell_unit"]
        lines.append(f"- {t['b']} - {t['a']}: {a['mean_diff']:+.3f}, p {_p(a['p'])} (datasets, n={a['n']}); "
                     f"cells {b['mean_diff']:+.3f}, p {_p(b['p'])} (n={b['n']})")
    lines += ["", "## PCA diagnostics (fold means over 21 folds)", "",
              "| Variant / features | explained var (mean, min) | between-subject share: subspace vs full | selected K | convergence |",
              "|---|---|---|---|---|"]
    for k, x in diag.items():
        lines.append(f"| {k} | {x['explained_variance_ratio_mean']:.3f}, {x['explained_variance_ratio_min']:.3f} | "
                     f"{x['between_share_subspace_mean']:.3f} vs {x['between_share_full_mean']:.3f} | "
                     f"{x['selected_k_counts']} | {x['convergence']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
