#!/usr/bin/env python3
"""Frozen/LoRA x paper/new head ablation: table and paired t-tests.

Conditions per route (``<modality>_<language>``):
  frozen_paper, frozen_new (one run), lora_paper, lora_new, lora_llm (3 LoRA seeds).

Scopes: ``standalone`` (one model per dataset), ``merged_cv`` (5-fold merged CV,
every dataset) and ``merged_final`` (merged final model, DAIC official test).

A route's score is the fold-mean of subject-level binary-strict metrics; LoRA
conditions are averaged over seeds. A group statistic is the mean over the
group's routes of the paired difference between two conditions.

Tests: two-sided paired t-tests on macro-F1 differences, no multiplicity
correction. Primary unit = route (the fold-mean score, so the tested mean equals
the reported group difference). Secondary unit = route x fold cell (LoRA scores
averaged over seeds within the fold). A test needs at least two units with
non-identical differences; otherwise p is None.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CONDITIONS = ("frozen_paper", "frozen_new", "lora_paper", "lora_new", "lora_llm")
EXPORT_CONDITIONS = {
    "frozen__paper_logreg__1run": "frozen_paper", "frozen__new_head__1run": "frozen_new",
    "lora__paper_logreg__3seeds": "lora_paper", "lora__new_head__3seeds": "lora_new",
    "lora__llm_output__3seeds": "lora_llm",
}
COMPARISONS = (
    ("head_effect_frozen", "frozen_new", "frozen_paper"),
    ("head_effect_lora", "lora_new", "lora_paper"),
    ("lora_effect_new_head", "lora_new", "frozen_new"),
    ("lora_effect_paper_head", "lora_paper", "frozen_paper"),
    ("lora_llm_vs_frozen_new", "lora_llm", "frozen_new"),
)
GROUPS = ("audio_only", "audio_text", "text_only", "all")
MERGED_ROUTES = (("native", "audio_only"), ("native", "audio_text"), ("native", "text_only"),
                 ("english", "audio_text"), ("english", "text_only"))
SEEDS = (7, 1337, 2024)


def _strip(subject: str) -> str:
    return subject.split("::")[-1]


def load_standalone(export_csv_root: Path, daic_traincv_root: Path) -> list[dict]:
    """Test-split predictions from the advisor export; DAIC frozen_new from train-CV selection.

    DAIC LoRA caches have no dev subjects, so DAIC LoRA new-head L2 was chosen by
    inner CV; the frozen side uses the same rule here to keep the pair symmetric.
    """
    records = []
    for path in sorted(export_csv_root.glob("*/predictions/*__predictions__*.csv")):
        if path.name.endswith("metrics_check.csv"):
            continue
        for r in csv.DictReader(path.open()):
            cond = EXPORT_CONDITIONS[r["condition"]]
            if r["split"] != "test" or (r["dataset"] == "daic" and cond == "frozen_new"):
                continue
            records.append({"scope": "standalone", "dataset": r["dataset"],
                            "route": f"{r['modality']}_{r['language']}", "condition": cond,
                            "seed": int(r.get("seed") or 0), "fold": int(r["fold"]),
                            "subject": r["subject_id"], "label": int(r["label"]), "pred": int(r["predicted"])})
    for route_dir in sorted(p for p in daic_traincv_root.iterdir() if p.is_dir()):
        for r in csv.DictReader((route_dir / "fold_0" / "predictions_subject_level.csv").open()):
            records.append({"scope": "standalone", "dataset": "daic", "route": route_dir.name,
                            "condition": "frozen_new", "seed": 0, "fold": 0, "subject": r["subject_id"],
                            "label": int(r["label"]), "pred": int(r["prediction"])})
    return records


def load_merged(heads_root: Path, features_root: Path) -> list[dict]:
    records = []
    heads = {"paper_logreg": "paper", "subjmean_l2": "new"}
    for language, modality in MERGED_ROUTES:
        route = f"{modality}_{language}"
        for stage, folds in (("cv", range(5)), ("final", (0,))):
            scope = f"merged_{stage}"
            for fold in folds:
                cell = f"{stage}_fold_{fold}"
                runs = [("frozen", 0, heads_root / "frozen" / f"{language}_{modality}" / cell)]
                runs += [("lora", s, heads_root / "lora" / f"{language}_{modality}" / f"s{s}" / cell) for s in SEEDS]
                for source, seed, run_dir in runs:
                    for head, tag in heads.items():
                        path = run_dir / head / "predictions_subject_level.csv"
                        if source == "lora" and head == "paper_logreg":
                            # The paper's own merged LogReg cell, as saved on MN5. The local
                            # refit matches it in 89/90 cells; one subject sits at p=0.495-0.501.
                            path = (features_root / f"qwen3_pooled_{language}" / modality
                                    / f"qmsm_{language}_{modality}_s{seed}" / stage / f"fold_{fold}"
                                    / "heads" / "logreg" / "predictions_subject_level.csv")
                        for r in csv.DictReader(path.open()):
                            records.append({"scope": scope, "dataset": r["dataset"], "route": route,
                                            "condition": f"{source}_{tag}", "seed": seed, "fold": fold,
                                            "subject": _strip(r["subject_id"]), "label": int(r["label"]),
                                            "pred": int(r["prediction"])})
                for seed in SEEDS:
                    qwen = (features_root / f"qwen3_pooled_{language}" / modality / f"qmsm_{language}_{modality}_s{seed}"
                            / stage / f"fold_{fold}" / "qwen")
                    for ds_dir in sorted(p for p in qwen.iterdir() if p.is_dir()):
                        for r in csv.DictReader((ds_dir / "predictions_subject_level.csv").open()):
                            pred = int(r["prediction"]) if r["prediction"] in ("0", "1") else 1 - int(r["label"])
                            records.append({"scope": scope, "dataset": ds_dir.name, "route": route,
                                            "condition": "lora_llm", "seed": seed, "fold": fold,
                                            "subject": _strip(r["subject_id"]), "label": int(r["label"]),
                                            "pred": pred})
    return records


def _metrics(tp, fp, fn, tn):
    """Vectorized binary-strict macro-F1, positive-F1 and UAR from (weighted) counts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        rec_pos = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        rec_neg = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
    return {"macro_f1": (f1_pos + f1_neg) / 2, "positive_f1": f1_pos, "uar": (rec_pos + rec_neg) / 2}


class Cells:
    """Per (scope, dataset, fold): subject universe; per cell: prediction vectors."""

    def __init__(self, records: list[dict]):
        self.universe: dict[tuple, dict[str, int]] = defaultdict(dict)
        self.cells: dict[tuple, dict[str, int]] = defaultdict(dict)
        for r in records:
            unit = (r["scope"], r["dataset"], r["fold"])
            if self.universe[unit].setdefault(r["subject"], r["label"]) != r["label"]:
                raise ValueError(f"Label conflict for {unit} {r['subject']}.")
            key = (r["scope"], r["dataset"], r["route"], r["condition"], r["seed"], r["fold"])
            if r["subject"] in self.cells[key]:
                raise ValueError(f"Duplicate subject {r['subject']} in {key}.")
            self.cells[key][r["subject"]] = r["pred"]
        self.order = {u: sorted(s) for u, s in self.universe.items()}
        self.vectors = {}
        for key, preds in self.cells.items():
            unit = (key[0], key[1], key[5])
            subjects = self.order[unit]
            if set(preds) != set(subjects):
                raise ValueError(f"{key}: {len(set(subjects) - set(preds))} test subjects missing.")
            y = np.asarray([self.universe[unit][s] for s in subjects])
            p = np.asarray([preds[s] for s in subjects])
            self.vectors[key] = (y, p)

    def route_scores(self) -> dict[tuple, dict]:
        """(scope, dataset, route, condition) -> fold-mean, seed-mean scores and per-fold macro-F1."""
        by_route: dict[tuple, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
        for key, (y, p) in self.vectors.items():
            scope, dataset, route, cond, seed, fold = key
            m = _metrics(np.sum(y * p), np.sum((1 - y) * p), np.sum(y * (1 - p)), np.sum((1 - y) * (1 - p)))
            by_route[(scope, dataset, route, cond)][seed][fold] = {k: float(v) for k, v in m.items()}
        out = {}
        for key, seeds in by_route.items():
            folds = {tuple(sorted(v)) for v in seeds.values()}
            if len(folds) != 1:
                raise ValueError(f"{key}: seeds have different folds {folds}.")
            folds = folds.pop()
            per_seed = {metric: [np.mean([v[f][metric] for f in folds]) for v in seeds.values()]
                        for metric in ("macro_f1", "positive_f1", "uar")}
            out[key] = {metric: float(np.mean(vals)) for metric, vals in per_seed.items()}
            out[key]["fold_macro_f1"] = {f: float(np.mean([v[f]["macro_f1"] for v in seeds.values()])) for f in folds}
            out[key]["seed_sd_macro_f1"] = float(np.std(per_seed["macro_f1"])) if len(seeds) > 1 else None
            out[key]["n_seeds"] = len(seeds)
            out[key]["n_folds"] = len(folds)
        return out


def _group_of(route: str) -> str:
    return route.rsplit("_", 1)[0]


def _paired_t(diffs: list[float]) -> dict:
    """Two-sided one-sample t-test of paired differences against 0, with a 95% CI."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "t": None, "df": None, "p": None, "ci95": [None, None]}
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:  # identical differences: t is undefined, not significant
        return {"n": n, "t": None, "df": n - 1, "p": 1.0 if d.mean() == 0 else None, "ci95": [None, None]}
    t = d.mean() / se
    half = stats.t.ppf(0.975, n - 1) * se
    return {"n": n, "t": float(t), "df": n - 1, "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "ci95": [float(d.mean() - half), float(d.mean() + half)]}


def analyse(records: list[dict]) -> dict:
    observed = Cells(records).route_scores()
    table = []
    for (scope, dataset, route, cond), m in sorted(observed.items()):
        table.append({"scope": scope, "dataset": dataset, "route": route, "condition": cond,
                      "n_folds": m["n_folds"], "n_seeds": m["n_seeds"], "macro_f1": m["macro_f1"],
                      "positive_f1": m["positive_f1"], "uar": m["uar"], "seed_sd_macro_f1": m["seed_sd_macro_f1"]})
    tests = []
    for scope in sorted({k[0] for k in observed}):
        routes = sorted({(k[1], k[2]) for k in observed if k[0] == scope})
        for group in GROUPS:
            members = [r for r in routes if group == "all" or _group_of(r[1]) == group]
            for name, a, b in COMPARISONS:
                pairs = [(d, r) for d, r in members
                         if (scope, d, r, a) in observed and (scope, d, r, b) in observed]
                if not pairs:
                    continue
                route_diffs = [observed[(scope, d, r, a)]["macro_f1"] - observed[(scope, d, r, b)]["macro_f1"]
                               for d, r in pairs]
                fold_diffs = []
                for d, r in pairs:
                    fa, fb = observed[(scope, d, r, a)]["fold_macro_f1"], observed[(scope, d, r, b)]["fold_macro_f1"]
                    if set(fa) != set(fb):
                        raise ValueError(f"{scope} {d} {r}: folds differ between {a} and {b}.")
                    fold_diffs += [fa[f] - fb[f] for f in sorted(fa)]
                by_route, by_fold = _paired_t(route_diffs), _paired_t(fold_diffs)
                tests.append({
                    "scope": scope, "group": group, "comparison": name, "a": a, "b": b,
                    "n_routes": len(pairs), "mean_diff_macro_f1": float(np.mean(route_diffs)),
                    "t_route": by_route["t"], "df_route": by_route["df"], "p_route": by_route["p"],
                    "ci95_route": by_route["ci95"],
                    "n_route_folds": len(fold_diffs), "mean_diff_fold_cells": float(np.mean(fold_diffs)),
                    "t_fold": by_fold["t"], "df_fold": by_fold["df"], "p_fold": by_fold["p"],
                    "routes_a_better": int(sum(x > 1e-9 for x in route_diffs)),
                    "routes_b_better": int(sum(x < -1e-9 for x in route_diffs)),
                })
    return {"table": table, "tests": tests,
            "settings": {"metric": "fold-mean subject-level macro-F1", "lora_seed_aggregation": "mean over seeds",
                         "test": "two-sided paired t-test on macro-F1 differences",
                         "units": "primary: routes (fold-mean); secondary: route x fold cells",
                         "correction": "none"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-csv-root", required=True, type=Path)
    parser.add_argument("--daic-traincv-root", required=True, type=Path)
    parser.add_argument("--merged-heads-root", required=True, type=Path)
    parser.add_argument("--merged-features-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite {args.output_dir}.")
    records = load_standalone(args.export_csv_root, args.daic_traincv_root)
    records += load_merged(args.merged_heads_root, args.merged_features_root)
    result = analyse(records)
    result["inputs"] = {k: str(v) for k, v in vars(args).items() if k.endswith("root")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_stats.json").write_text(json.dumps(result, indent=2))
    for name in ("table", "tests"):
        rows = result[name]
        with (args.output_dir / f"ablation_{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for t in result["tests"]:
        print(f"{t['scope']:13s} {t['group']:10s} {t['comparison']:24s} n={t['n_routes']:2d} "
              f"diff={t['mean_diff_macro_f1']:+.3f} p_route={t['p_route']} "
              f"n_cells={t['n_route_folds']:3d} p_fold={t['p_fold']}")


if __name__ == "__main__":
    main()
