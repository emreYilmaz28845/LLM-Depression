"""Aggregate-only Markdown report and English figure for seed_ensemble_analysis.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"cmdc": "Mandarin (CMDC)", "androids_interview": "Italian (Androids)", "daic": "English (DAIC-WOZ)",
         "turkish": "Turkish", "d3tec": "Spanish (D3TEC)"}
METHODS = ["text", "audio_text", "text_ens3", "at_ens3", "ens6", "seed_agree", "oracle"]
LABEL = {"text": "Text (1 seed)", "audio_text": "A+T (1 seed)", "text_ens3": "Text, 3-seed ensemble",
         "at_ens3": "A+T, 3-seed ensemble", "ens6": "Text + A+T, 6-model ensemble",
         "seed_agree": "Seed-agreement selector", "oracle": "Oracle (ceiling)"}
COLOR = {"text": "#3a7bd5", "audio_text": "#e07b39", "text_ens3": "#1d4e89", "at_ens3": "#a0522d",
         "ens6": "#2a9d8f", "seed_agree": "#9b5de5", "oracle": "#5b5b5b"}


def figure(perf, diag, path):
    m = perf.groupby(["corpus", "method"]).macro_f1.mean(); order = list(NAMES)
    shown = ["text", "audio_text", "text_ens3", "ens6", "seed_agree", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]; w = .8 / len(shown)
    for j, k in enumerate(shown):
        xs = [i + (j - (len(shown) - 1) / 2) * w for i in range(len(order))]
        ax.bar(xs, [m[(c, k)] for c in order], w, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[c].replace(" (", "\n(") for c in order], fontsize=9)
    ax.set_ylim(.5, 1.02); ax.set_ylabel("Macro-F1 (fold mean)")
    ax.set_title("Macro-F1 per language\n(single-seed bars = mean of 3 seeds)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.14))
    ax = axes[1]; w = .2
    for j, (mod, lv, col, lab) in enumerate([("text", 3, "#1d4e89", "Text, 3/3 seeds agree"),
                                             ("text", 2, "#9cc3f0", "Text, 2/3 seeds agree"),
                                             ("audio_text", 3, "#a0522d", "A+T, 3/3 seeds agree"),
                                             ("audio_text", 2, "#f2b48a", "A+T, 2/3 seeds agree")]):
        d = diag[(diag.model == mod) & (diag.seeds_agreeing == lv)].set_index("corpus")
        xs = [i + (j - 1.5) * w for i in range(len(order))]
        vals = [100 * d.loc[c].ensemble_accuracy if c in d.index and d.loc[c].n else 0 for c in order]
        ax.bar(xs, vals, w, color=col, label=lab)
        for x, c in zip(xs, order):
            if c in d.index and d.loc[c].n:
                ax.text(x, 2, str(int(d.loc[c].n)), ha="center", fontsize=6.5, color="white" if lv == 3 else "black")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[c].replace(" (", "\n(") for c in order], fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("Accuracy of the 3-seed vote (%)")
    ax.set_title("Is seed agreement a confidence signal?\n(number in bar = subjects)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.14))
    fig.suptitle("Seed ensembles and seed-agreement selection: direct text vs audio+text LLMs", fontsize=11.5)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    d = p.parse_args().analysis
    perf = pd.read_csv(d / "seed_ensemble_performance.csv"); tests = pd.read_csv(d / "seed_ensemble_tests.csv")
    diag = pd.read_csv(d / "seed_agreement_diagnostic.csv"); cfg = json.loads((d / "seed_ensemble_config.json").read_text())
    m = perf.groupby(["corpus", "method"])[["macro_f1", "positive_f1", "uar"]].mean()
    short = {"text": "Text", "audio_text": "A+T", "text_ens3": "Text ens3", "at_ens3": "A+T ens3", "ens6": "Ens6",
             "seed_agree": "Seed-agree", "oracle": "Oracle"}
    L = ["# Seed ensembles and seed-agreement selection: direct text vs audio+text LLMs", "",
         "Direct LoRA Qwen3.8-27B text and Qwen3-Omni audio+text likelihood predictions, seeds 7, 1337, 2024. "
         "No inference, no label-based tuning.", "",
         f"Source commit `{cfg['source_commit']}`, script sha256 `{cfg['source_script_sha256'][:16]}...`, "
         f"{cfg['reps']} bootstrap/permutation replicates (seed {cfg['rng_seed']}), runtime {cfg['runtime_seconds']} s.", "",
         "## Rules (fixed before looking at outer results)", ""]
    L += [f"- **{k}**: {v}" for k, v in cfg["rules"].items()] + [f"- Scale: {cfg['scale']}.", "",
          "## Performance", "", f"Cells: Macro-F1 / Positive-F1. {cfg['aggregation']}. DAIC is one official test fold; "
          "Turkish and CMDC are outer validation (not locked test).", "",
          "| Corpus | " + " | ".join(short[x] for x in METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(f"{m.loc[(ds, x)].macro_f1:.3f} / {m.loc[(ds, x)].positive_f1:.3f}"
                                                for x in METHODS) + " |")
    L += ["", "UAR (same aggregation):", "", "| Corpus | " + " | ".join(short[x] for x in METHODS) + " |",
          "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(f"{m.loc[(ds, x)].uar:.3f}" for x in METHODS) + " |")
    L += ["", "## Paired tests (method minus reference, fold-mean Macro-F1)", "", cfg["tests"] + ".", "",
          "| Corpus | Method | Reference | Delta | 95% CI | Perm. p | Holm p |", "|---|---|---|---:|---|---:|---:|"]
    for _, r in tests.iterrows():
        L.append(f"| {NAMES[r.corpus]} | {short[r.method]} | {short.get(r.reference, r.reference)} | {r.delta_macro_f1:+.3f} | "
                 f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.permutation_p:.3f} | {r.holm_p:.3f} |")
    L += ["", "## Seed agreement as confidence", "", "Accuracy of each modality's 3-seed vote by number of seeds agreeing.", "",
          "| Corpus | Model | 3/3 agree: n (share) | 3/3 accuracy | 2/3 agree: n | 2/3 accuracy |", "|---|---|---:|---:|---:|---:|"]
    for ds in NAMES:
        for mod in ("text", "audio_text"):
            a = diag[(diag.corpus == ds) & (diag.model == mod)].set_index("seeds_agreeing")
            acc = lambda lv: "-" if pd.isna(a.loc[lv].ensemble_accuracy) else f"{a.loc[lv].ensemble_accuracy:.2f}"  # noqa: E731
            L.append(f"| {NAMES[ds]} | {short[mod]} | {int(a.loc[3].n)} ({a.loc[3].share:.2f}) | {acc(3)} | "
                     f"{int(a.loc[2].n)} | {acc(2)} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in cfg["limitations"]] + [""]
    (d / "SEED_ENSEMBLE_REPORT.md").write_text("\n".join(L) + "\n")
    figure(perf, diag, d / "seed_ensemble_summary.png")


if __name__ == "__main__":
    main()
