#!/usr/bin/env python3
"""Build the report of the text replication of Kaan's protocol from the saved run outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LANGUAGES = ("English", "Turkish", "Italian", "Spanish", "Mandarin")
ARMS = ("mixed_train", "x_train_only")
S12_ARMS = ("base", "onehot_lr", "perlang_z_lr", "global_leace_lr", "perlang_leace_lr",
            "xgb_all", "xgb_onehot", "xgb_top50", "xgb_top100")


def _summary(path: Path) -> dict:
    return {(e["experiment"], e["source"], e["eval"]): e for e in json.loads((path / "summary.json").read_text())["summary"]}


def _fits(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "fits.jsonl").read_text().splitlines()]


def _f(v, sd=None) -> str:
    return "—" if v is None else (f"{v:.3f}" if sd is None else f"{v:.3f} ±{sd:.3f}")


def _sig(sig: dict, tl: str, label: str) -> str:
    c = sig.get((tl, label))
    if c is None:
        return ""
    mark = "**" if c["p_holm"] < 0.05 else ""
    return f"{mark}p={c['p_raw']:.3g}, Holm {c['p_holm']:.2g}{mark}"


def training_observations(tl: str, dirs: dict) -> list[str]:
    lines = []
    for name, path in dirs.items():
        fits = _fits(path)
        conv = sum(f["converged"] for f in fits)
        const = Counter()
        tot = Counter()
        for f in fits:
            cm = f["confusion_matrix"]
            pred_pos = cm[0][1] + cm[1][1]
            key = f.get("experiment", "")
            tot[key] += 1
            if pred_pos in (0, f["n"]):
                const[key] += 1
        lines.append(f"- {name}: {len(fits)} fit-evaluations, converged {conv}/{len(fits)}, "
                     f"max lbfgs iterations {max(f['n_iter'] for f in fits)}; constant-prediction evaluations: "
                     + ", ".join(f"{k} {const[k]}/{tot[k]}" for k in sorted(tot)) + ".")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--core-a", required=True)
    p.add_argument("--aug", required=True)
    p.add_argument("--s12", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--significance", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    sig = {}
    with (root / a.significance / "comparisons.csv").open() as fh:
        for r in csv.DictReader(fh):
            r["p_raw"], r["p_holm"] = float(r["p_raw"]), float(r["p_holm"])
            sig[(r["text"], r["label"])] = r
    sig_summary = json.loads((root / a.significance / "summary.json").read_text())
    git = {k: json.loads((root / d / "native" / ("geometry.json" if k == "geometry" else "summary.json")).read_text())["git_head"][:7]
           for k, d in (("A", a.core_a), ("aug", a.aug), ("s12", a.s12), ("geometry", a.geometry))}
    fa = json.loads((root / "study" / "fold_assignments_manifest.json").read_text())

    L = ["# Text replication of Kaan's multilingual logistic protocol", "",
         "Frozen Qwen3.8-27B text features (base model, no adapter, revision 1d4bf0f), final layer, "
         "last prompt token, D = 5120, one equal-weight mean vector per speaker. Two text versions: native "
         "transcripts and English translations (DAIC-WOZ is English in both). Five corpora, English = DAIC-WOZ. "
         "Folds: our merged-protocol `fold_assignments.csv` "
         f"(sha256 {fa['output_sha256'][:12]}…): English fixed 142 train+val -> 47 official test, the other four "
         "corpora five outer folds (Androids = merged folds). Cohort: "
         + ", ".join(f"{k} {v['total']} ({v['pos']}+/{v['neg']}-)" for k, v in fa["cohort"].items())
         + f", total {fa['cohort_total']}.", "",
         "Head: global TRAIN z-norm (population std), Kaan's class oversampler, seeds 0/1/2, "
         "`LogisticRegression(C, max_iter=10000, tol=1e-8, lbfgs)`, P >= threshold. C and thresholds are the "
         "fixed audio-derived settings of the handoff, not tuned on text. Metric: subject-level macro-F1, "
         "seed mean per fold, equal mean over folds (± sample SD of the fold means). Commits: "
         f"core A {git['A']}, B/C/D {git['aug']}, Section 12 {git['s12']}, geometry {git['geometry']}, "
         f"significance registry sha256 {sig_summary['registry_sha256'][:12]}… (frozen before any test).", "",
         "Not comparable with Kaan's audio numbers yet: his audio runs use E-DAIC and the official Androids "
         "folds. Text vs audio tests wait for his merged-fold DAIC-WOZ rerun.", ""]

    for tl in ("native", "english"):
        A = _summary(root / a.core_a / tl)
        G = _summary(root / a.aug / tl)
        S = _summary(root / a.s12 / tl)
        L += [f"## {tl.capitalize()} text", "", "### A1 same-language and A3 combined (held-out TEST, macro-F1)", "",
              "| Target | A1 X -> X | A3 Combined -> X | A3 vs A1 |", "|---|---|---|---|"]
        for lang in (*LANGUAGES, "Pooled"):
            e1, e3 = A.get(("same", lang, lang)), A.get(("combined", "Combined", lang))
            L.append(f"| {lang} | {_f(e1 and e1['macro_f1'], e1 and e1['macro_f1_fold_sd'])} | "
                     f"{_f(e3['macro_f1'], e3['macro_f1_fold_sd'])} | "
                     f"{_sig(sig, tl, f'Combined -> {lang} vs {lang} -> {lang}')} |")
        L += ["", "English A1 is one fixed split (mean of 3 seeds). A3 English repeats the same 47 test speakers "
              "in all five folds.", "", "### A2 cross-language (all X -> all Y, 3-seed mean macro-F1)", "",
              "Full-target scores; not comparable with held-out scores.", "",
              "| Source \\ Target | " + " | ".join(LANGUAGES) + " |", "|---" * 6 + "|"]
        for src in LANGUAGES:
            L.append(f"| {src} | " + " | ".join("—" if src == t else f"{A[('cross', src, t)]['macro_f1']:.3f}"
                                               for t in LANGUAGES) + " |")
        # Turkish-target diagnosis
        probs = defaultdict(list)
        with (root / a.core_a / tl / "predictions.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["experiment"] == "cross" and r["eval"] == "Turkish":
                    probs[(r["source"], r["seed"])].append((int(r["label"]), float(r["prob"]), int(r["pred"])))
        L += ["", "Turkish as A2 target (diagnostic; AUROC is not a headline metric):", "",
              "| Source | predicted positive (of 120, per seed) | median P | AUROC (seed mean) |", "|---|---|---|---|"]
        for src in LANGUAGES[0:1] + LANGUAGES[2:]:
            rows = [probs[(src, str(s))] for s in range(3)]
            L.append(f"| {src} | {' / '.join(str(sum(x[2] for x in r)) for r in rows)} | "
                     f"{np.median([x[1] for r in rows for x in r]):.3f} | "
                     f"{np.mean([roc_auc_score([x[0] for x in r], [x[1] for x in r]) for r in rows]):.3f} |")

        L += ["", "### B: X + limited Y -> X (macro-F1; Δ vs X-only)", "",
              "| Target X | X-only | + Y | mixed_train | x_train_only |", "|---|---|---|---|---|"]
        for x in LANGUAGES:
            base = G[("xonly", f"{x}|x_train_only", x)]["macro_f1"]
            for y in LANGUAGES:
                if y == x:
                    continue
                cells = []
                for arm in ARMS:
                    e = G[("pair", f"{x}+{y}|{arm}", x)]
                    cells.append(f"{e['macro_f1']:.3f} ({e['macro_f1'] - base:+.3f}) "
                                 f"{_sig(sig, tl, f'{x}+{y} [{arm}] vs {x}-only')}")
                L.append(f"| {x} | {base:.3f} | {y} | {cells[0]} | {cells[1]} |")
        L += ["", "### C: X + limited Y + limited Z -> X (macro-F1; Δ vs X-only)", "",
              "| Target X | X-only | + Y + Z | mixed_train | x_train_only |", "|---|---|---|---|---|"]
        for x in LANGUAGES:
            base = G[("xonly", f"{x}|x_train_only", x)]["macro_f1"]
            for y, z in combinations([o for o in LANGUAGES if o != x], 2):
                cells = []
                for arm in ARMS:
                    e = G[("triple", f"{x}+{y}+{z}|{arm}", x)]
                    cells.append(f"{e['macro_f1']:.3f} ({e['macro_f1'] - base:+.3f}) "
                                 f"{_sig(sig, tl, f'{x}+{y}+{z} [{arm}] vs {x}-only')}")
                L.append(f"| {x} | {base:.3f} | {y} + {z} | {cells[0]} | {cells[1]} |")
        sp = G[("xonly", "Spanish|x_train_only", "Spanish")]["macro_f1"]
        L += ["", "### D: Spanish + all 116 Italian -> Spanish TEST (macro-F1)", "",
              f"Spanish-only {sp:.3f}.", "", "| Arm | Spanish + all Italian | capped Spanish + Italian | vs Spanish-only | vs capped |",
              "|---|---|---|---|---|"]
        for arm in ARMS:
            d = G[("all_italian", f"Spanish+allItalian|{arm}", "Spanish")]
            c = G[("pair", f"Spanish+Italian|{arm}", "Spanish")]
            L.append(f"| {arm} | {d['macro_f1']:.3f} ±{d['macro_f1_fold_sd']:.3f} | {c['macro_f1']:.3f} | "
                     f"{_sig(sig, tl, f'Spanish+allItalian [{arm}] vs Spanish-only')} | "
                     f"{_sig(sig, tl, f'Spanish+allItalian [{arm}] vs capped Spanish+Italian')} |")
        L += ["", "### Section 12 source-control arms on the combined setup (macro-F1)", "",
              "| Arm | " + " | ".join((*LANGUAGES, "Pooled")) + " |", "|---" * 7 + "|"]
        for arm in S12_ARMS:
            cells = []
            for ev in (*LANGUAGES, "Pooled"):
                e = S[("s12", arm, ev)]
                s = "" if arm == "base" else " " + _sig(sig, tl, f"{arm} vs base on {ev}")
                cells.append(f"{e['macro_f1']:.3f}{s}")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L += ["", "`base` = A3 refit. `onehot_lr` equals `base`: with C=0.01 the head keeps the one-hot weights near 0. "
              "`perlang_leace_lr` matches `perlang_z_lr` (to within float noise): per-language centering already "
              "equalizes the language means, so LEACE has almost nothing left to erase. XGBoost arms use fixed settings (depth 3, 300 trees, lr 0.05, colsample 0.5) "
              "and threshold 0.5. Not run: 32D encoder, GRL, MMD, supervised contrastive (need Kaan's source).", ""]

        geo = json.loads((root / a.geometry / tl / "geometry.json").read_text())
        L += ["### E1: mean speaker-pair cosine (raw / mean of the 5 combined-TRAIN scaler views)", "",
              "| | " + " | ".join(LANGUAGES) + " |", "|---" * 6 + "|"]
        for x in LANGUAGES:
            L.append(f"| {x} | " + " | ".join(
                f"{geo['raw']['cosine']['matrix'][f'{x}|{y}']:.3f} / "
                f"{geo['normalized_mean_of_5_views']['cosine']['matrix'][f'{x}|{y}']:.3f}" for y in LANGUAGES) + " |")
        rc, nc = geo["raw"]["cosine"], geo["normalized_mean_of_5_views"]["cosine"]
        L += ["", f"Pooled over {rc['pairs']['all']} pairs ({rc['pairs']['within']} within-language, "
              f"{rc['pairs']['cross']} cross-language): all {rc['pooled_all']:.3f} / {nc['pooled_all']:.3f}, "
              f"within {rc['pooled_within']:.3f} / {nc['pooled_within']:.3f}, cross {rc['pooled_cross']:.3f} / "
              f"{nc['pooled_cross']:.3f}. Distance = 1 - cosine.", "",
              "### E2: pooled cosine kNN, share of neighbours from the same language (normalized views)", "",
              "| Language | N | K=30 | K=60 | K=90 | random | capacity binds at |", "|---|---|---|---|---|---|---|"]
        kn = geo["normalized_mean_of_5_views"]["pooled_knn"]
        for x in LANGUAGES:
            r = [kn[k]["per_language"][x] for k in ("30", "60", "90")]
            binds = ", ".join(f"K={k}" for k in ("30", "60", "90") if kn[k]["per_language"][x]["capacity_binding"]) or "—"
            L.append(f"| {x} | {int(r[0]['n'])} | {r[0]['same_pct']:.3f} | {r[1]['same_pct']:.3f} | {r[2]['same_pct']:.3f} | "
                     f"{r[0]['random_same_pct']:.3f} | {binds} |")
        L.append(f"| Overall | 565 | {kn['30']['overall_same_pct']:.3f} | {kn['60']['overall_same_pct']:.3f} | "
                 f"{kn['90']['overall_same_pct']:.3f} | | |")
        L += ["", "### E3: within-language cosine kNN (K=10), share of neighbours with the query's label (normalized views)", "",
              "| Language | all queries (random) | positive queries (random) | negative queries (random) | balanced (random) |",
              "|---|---|---|---|---|"]
        wk = geo["normalized_mean_of_5_views"]["within_knn"]
        for x in (*LANGUAGES, "Overall"):
            r = wk[x]
            L.append(f"| {x} | {r['same_label']:.3f} ({r['random_expectation']:.3f}) | {r['pos_query']:.3f}"
                     + (f" ({r['pos_expectation']:.3f})" if "pos_expectation" in r else "")
                     + f" | {r['neg_query']:.3f}" + (f" ({r['neg_expectation']:.3f})" if "neg_expectation" in r else "")
                     + f" | {r['balanced']:.3f} ({r['balanced_expectation']:.3f}) |")
        L += ["", "Neighbour agreement is a descriptive diagnostic, not classifier accuracy. Raw-view numbers are in "
              "`geometry.json`.", "", "### Training observations", ""]
        L += training_observations(tl, {"core A": root / a.core_a / tl, "B/C/D": root / a.aug / tl,
                                        "Section 12": root / a.s12 / tl})
        L.append("")

    rows = [r for r in sig.values()]
    holm_sig = [r for r in rows if r["p_holm"] < 0.05]
    L += ["## F: paired significance", "",
          f"{sig_summary['n_comparisons']} registered comparisons (both text versions), registry frozen before "
          "testing. Paired speaker-block swap test (exact for <= 16 discordant speakers, else 100000 swaps, plus-one), "
          "10000-sample stratified paired bootstrap for 95% CIs, Holm over the whole family.", "",
          f"Raw p < 0.05: {sig_summary['n_raw_below_0.05']}. Holm-adjusted p < 0.05: {sig_summary['n_holm_below_0.05']}.", ""]
    if holm_sig:
        L += ["| Text | Comparison | A | B | Δ (95% CI) | p raw | p Holm |", "|---|---|---|---|---|---|---|"]
        for r in sorted(holm_sig, key=lambda r: r["p_holm"]):
            L.append(f"| {r['text']} | {r['label']} | {float(r['score_a']):.3f} | {float(r['score_b']):.3f} | "
                     f"{float(r['delta']):+.3f} ({float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}) | "
                     f"{r['p_raw']:.3g} | {r['p_holm']:.3g} |")
    L += ["", "Bootstrap CIs condition on the fitted models. Seeds and folds are not independent subjects; shared "
          "training data across folds adds uncertainty the tests do not model. The comparisons are exploratory.", "",
          "## Notes", "",
          "- Prompt audit: the frozen-cache prompt asks whether the subject is depressed and lists the candidate "
          "labels; it never contains the subject's label, score or diagnosis. Turkish prompts add one sentence saying "
          "the answer belongs to the positive or negative question set (a corpus-specific cue, no label). No corpus "
          "name or language ID is in the prompt.",
          "- Speaker vectors: Turkish has two rows per speaker (positive and negative question sets), averaged with "
          "equal weight; every other corpus has one row per speaker. No chunks are counted as subjects.",
          "- English DEV: none unused here; DAIC-WOZ train+val form the 142-speaker TRAIN, as in our merged final stage.",
          "- Sampling plans are regenerated with the handoff's algorithm on our folds (not copied from Kaan's audits); "
          "all cap, balance, matched-draw and TEST-disjointness assertions passed.",
          f"- UMAP maps: `{a.maps}/<text>/` (PNG only + EXPERIMENTS.txt); coordinates in `umap_coords_*`.", ""]
    a.output.write_text("\n".join(L))
    print(a.output)


if __name__ == "__main__":
    main()
