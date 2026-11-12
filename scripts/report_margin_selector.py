"""Aggregate-only Markdown report for margin_selector_analysis.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NAMES = {"cmdc": "Mandarin (CMDC)", "androids_interview": "Italian (Androids)", "daic": "English (DAIC-WOZ)",
         "turkish": "Turkish", "d3tec": "Spanish (D3TEC)"}
METHODS = ["text", "audio_text", "max_conf", "advisor_gate", "oracle"]
LABEL = {"text": "Text", "audio_text": "A+T", "max_conf": "Max-conf", "margin_sum": "Margin sum",
         "advisor_gate": "Advisor gate", "oracle": "Oracle"}


def cell(r):
    return f"{r.macro_f1:.3f} / {r.positive_f1:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    a = p.parse_args(); d = a.analysis
    perf = pd.read_csv(d / "margin_selector_performance.csv")
    tests = pd.read_csv(d / "margin_selector_tests.csv")
    diag = pd.read_csv(d / "margin_confidence_diagnostic.csv")
    thr = pd.DataFrame(json.loads((d / "margin_threshold_selection.json").read_text()))
    cfg = json.loads((d / "margin_analysis_config.json").read_text())
    shares = perf[perf.method == "advisor_gate_share"].groupby("corpus")[["share_text", "share_audio_text", "share_combined"]].mean()
    perf = perf[perf.method != "advisor_gate_share"]
    m = perf.groupby(["corpus", "method"], sort=False)[["macro_f1", "positive_f1", "uar"]].mean()

    L = ["# Direct LoRA LLM margin selector: text vs audio+text", "",
         "Selects, per subject, between direct Qwen3.8-27B text and Qwen3-Omni audio+text likelihood predictions "
         "using each model's own candidate-label likelihood margin as confidence. No inference was run.", "",
         f"Source commit `{cfg['source_commit']}`, script sha256 `{cfg['source_script_sha256'][:16]}...`, "
         f"{cfg['reps']} bootstrap/permutation replicates (seed {cfg['rng_seed']}), runtime {cfg['runtime_seconds']} s.", "",
         "## Rules (fixed before looking at outer results)", ""]
    L += [f"- **{k}**: {v}" for k, v in cfg["rules"].items()]
    L += [f"- Scale: {cfg['scale']}.", "- Oracle: picks the correct model whenever either is correct (uses labels; ceiling only).",
          "- Threshold selection: DAIC, Androids, D3TEC use each fold's own inner validation. Turkish and CMDC have no "
          "independent inner set (checkpoints were selected on the outer validation fold), so tau for fold k is chosen on "
          "the other folds' predictions (leave-one-fold-out); fold k labels never choose its tau.", "",
          "## Performance", "",
          "Cells: Macro-F1 / Positive-F1. Equal-weight fold mean per seed, then mean over seeds 7, 1337, 2024. "
          "DAIC is one official test fold; Turkish and CMDC are outer validation (not locked test).", "",
          "| Corpus | " + " | ".join(LABEL[x] for x in METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(cell(m.loc[(ds, x)]) for x in METHODS) + " |")
    L += ["", "UAR (same aggregation):", "", "| Corpus | " + " | ".join(LABEL[x] for x in METHODS) + " |",
          "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(f"{m.loc[(ds, x)].uar:.3f}" for x in METHODS) + " |")
    L += ["", "## Paired tests (rule minus reference, seed-mean fold-mean Macro-F1)", "",
          "Bootstrap: subjects resampled class-stratified within fold, jointly for all seeds; conditions on fixed "
          "models and thresholds. Holm within each family of 10 (5 corpora x 2 rules).", "",
          "| Corpus | Rule | Reference | Delta | 95% CI | Perm. p | Holm p |", "|---|---|---|---:|---|---:|---:|"]
    for _, r in tests.iterrows():
        L.append(f"| {NAMES[r.corpus]} | {LABEL[r.rule]} | {LABEL[r.reference]} | {r.delta_macro_f1:+.3f} | "
                 f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.permutation_p:.3f} | {r.holm_p:.3f} |")
    L += ["", "## Is the margin a confidence signal?", "",
          "AUROC of |z| for predicting that model's own correctness on outer evaluation, mean over seeds "
          "(diagnostic only, not a headline metric; 0.5 = no signal).", "",
          "| Corpus | Text | A+T | Text errors (mean/seed) | A+T errors (mean/seed) |", "|---|---:|---:|---:|---:|"]
    g = diag.groupby(["corpus", "model"])[["correctness_auroc_abs_margin", "errors"]].mean()
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | {g.loc[(ds, 'text')].correctness_auroc_abs_margin:.2f} | "
                 f"{g.loc[(ds, 'audio_text')].correctness_auroc_abs_margin:.2f} | "
                 f"{g.loc[(ds, 'text')].errors:.1f} | {g.loc[(ds, 'audio_text')].errors:.1f} |")
    L += ["", "## Advisor gate routing (share of outer subjects, mean over seeds) and chosen tau", "",
          "| Corpus | Text | A+T only | Combined | Fold-seeds with tau=inf (text only) |", "|---|---:|---:|---:|---:|"]
    for ds in NAMES:
        t = thr[thr.corpus == ds]
        L.append(f"| {NAMES[ds]} | {shares.loc[ds].share_text:.2f} | {shares.loc[ds].share_audio_text:.2f} | "
                 f"{shares.loc[ds].share_combined:.2f} | {(t.tau.astype(str) == 'inf').sum()}/{len(t)} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in cfg["limitations"]]
    L += ["- Turkish/CMDC outer numbers were already optimistic for single models, because checkpoints were selected "
          "on the same fold; this is unchanged here.", ""]
    (d / "MARGIN_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
