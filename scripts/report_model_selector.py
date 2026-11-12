"""Aggregate-only Markdown report and English summary figure for model_selector_analysis.py."""
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
METHODS = ["text", "audio_text", "audio_only", "ola", "lca", "vote3", "stack", "oracle"]
LABEL = {"text": "Text", "audio_text": "A+T", "audio_only": "Audio-only", "ola": "OLA", "lca": "LCA",
         "vote3": "3-model vote", "stack": "Stacking", "oracle": "Oracle"}
COLOR = {"text": "#3a7bd5", "audio_text": "#e07b39", "ola": "#2a9d8f", "lca": "#8ab17d",
         "vote3": "#9b5de5", "stack": "#c44569", "oracle": "#5b5b5b"}


def figure(perf, dis, path):
    m = perf.groupby(["corpus", "method"]).macro_f1.mean()
    d = dis.groupby(["corpus", "method"])[["n_disagreements", "correct_picks"]].sum()
    shown = ["text", "audio_text", "ola", "lca", "vote3", "stack", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; order = list(NAMES); w = .8 / len(shown)
    for j, k in enumerate(shown):
        xs = [i + (j - (len(shown) - 1) / 2) * w for i in range(len(order))]
        ax.bar(xs, [m[(c, k)] for c in order], w, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[c].replace(" (", "\n(") for c in order], fontsize=9)
    ax.set_ylim(.5, 1.02); ax.set_ylabel("Macro-F1 (fold mean, mean of 3 seeds)")
    ax.set_title("Macro-F1 per language", fontsize=10.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    ax = axes[1]
    for j, k in enumerate(shown[:-1]):
        xs = [i + (j - (len(shown) - 2) / 2) * (.8 / (len(shown) - 1)) for i in range(len(order))]
        ax.bar(xs, [100 * d.loc[(c, k)].correct_picks / d.loc[(c, k)].n_disagreements for c in order],
               .8 / (len(shown) - 1), color=COLOR[k])
    ax.axhline(50, color="k", ls="--", lw=1)
    for i, c in enumerate(order):
        ax.text(i, 101, f"n={int(d.loc[(c, 'text')].n_disagreements)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[c].replace(" (", "\n(") for c in order], fontsize=9)
    ax.set_ylim(0, 108); ax.set_ylabel("% of disagreements resolved correctly")
    ax.set_title("When text and A+T disagree, how often is the right one picked?\n"
                 "(n = disagreement subjects summed over 3 seeds; oracle = 100%)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Choosing between direct text and audio+text LLMs per subject: no rule reliably beats text",
                 fontsize=11.5)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    d = p.parse_args().analysis
    perf = pd.read_csv(d / "selector_performance.csv"); tests = pd.read_csv(d / "selector_tests.csv")
    dis = pd.read_csv(d / "selector_disagreements.csv"); coef = pd.read_csv(d / "stack_coefficients.csv")
    reg = pd.DataFrame(json.loads((d / "selection_regions.json").read_text()))
    cfg = json.loads((d / "selector_config.json").read_text())
    m = perf.groupby(["corpus", "method"])[["macro_f1", "positive_f1", "uar"]].mean()
    dd = dis.groupby(["corpus", "method"])[["n_disagreements", "correct_picks"]].sum()

    L = ["# Which model to trust when direct text and audio+text LLMs disagree?", "",
         "Per-subject choice between direct LoRA Qwen3.8-27B text and Qwen3-Omni audio+text likelihood predictions. "
         "No inference was run.", "",
         f"Source commit `{cfg['source_commit']}`, script sha256 `{cfg['source_script_sha256'][:16]}...`, "
         f"K={cfg['k']}, {cfg['reps']} bootstrap/permutation replicates (seed {cfg['rng_seed']}), "
         f"runtime {cfg['runtime_seconds']} s.", "", "## Rules (fixed before looking at outer results)", ""]
    L += [f"- **{k}**: {v}" for k, v in cfg["rules"].items()]
    L += [f"- Embedding for neighbours: {cfg['embedding']}.", f"- Selection subjects: {cfg['selection']}.",
          "- Oracle: picks the correct model whenever either is correct (uses labels; ceiling only).", "",
          "## Performance", "", "Cells: Macro-F1 / Positive-F1. " + cfg["aggregation"] + ". DAIC is one official "
          "test fold; Turkish and CMDC are outer validation (not locked test).", "",
          "| Corpus | " + " | ".join(LABEL[x] for x in METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(f"{m.loc[(ds, x)].macro_f1:.3f} / {m.loc[(ds, x)].positive_f1:.3f}"
                                                for x in METHODS) + " |")
    L += ["", "UAR (same aggregation):", "", "| Corpus | " + " | ".join(LABEL[x] for x in METHODS) + " |",
          "|---|" + "---:|" * len(METHODS)]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | " + " | ".join(f"{m.loc[(ds, x)].uar:.3f}" for x in METHODS) + " |")
    dm = ["text", "audio_text", "ola", "lca", "vote3", "stack"]
    L += ["", "## Disagreements resolved correctly", "",
          "Subjects where text and A+T predictions differ, summed over seeds (a subject counts once per seed). "
          "Always-text and always-A+T are the reference rules.", "",
          "| Corpus | Disagreements | " + " | ".join(LABEL[x] for x in dm) + " |", "|---|---:|" + "---:|" * len(dm)]
    for ds in NAMES:
        n = int(dd.loc[(ds, "text")].n_disagreements)
        L.append(f"| {NAMES[ds]} | {n} | " + " | ".join(f"{int(dd.loc[(ds, x)].correct_picks)} ({100 * dd.loc[(ds, x)].correct_picks / n:.0f}%)"
                                                       for x in dm) + " |")
    L += ["", "## Paired tests (rule minus reference, seed-mean fold-mean Macro-F1)", "", cfg["tests"] + ".", "",
          "| Corpus | Rule | Reference | Delta | 95% CI | Perm. p | Holm p |", "|---|---|---|---:|---|---:|---:|"]
    for _, r in tests.iterrows():
        L.append(f"| {NAMES[r.corpus]} | {LABEL[r.rule]} | {LABEL[r.reference]} | {r.delta_macro_f1:+.3f} | "
                 f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.permutation_p:.3f} | {r.holm_p:.3f} |")
    L += ["", "## Selection sets", "", "Mean per fold-seed. Few disagreements in the selection set means a selector "
          "has little to learn from.", "", "| Corpus | Mode | Selection subjects | Selection disagreements |",
          "|---|---|---:|---:|"]
    for ds in NAMES:
        r = reg[reg.corpus == ds]
        L.append(f"| {NAMES[ds]} | {r['mode'].iloc[0]} | {r.n_selection.mean():.1f} | {r.selection_disagreements.mean():.1f} |")
    c = coef.groupby("corpus")[["coef_text", "coef_audio_text", "coef_audio_only"]].mean()
    L += ["", "## Stacking coefficients (mean over fold-seeds, standardized margins)", "",
          "| Corpus | Text | A+T | Audio-only |", "|---|---:|---:|---:|"]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | {c.loc[ds].coef_text:+.2f} | {c.loc[ds].coef_audio_text:+.2f} | {c.loc[ds].coef_audio_only:+.2f} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in cfg["limitations"]] + [""]
    (d / "SELECTOR_REPORT.md").write_text("\n".join(L) + "\n")
    figure(perf, dis, d / "model_selector_summary.png")


if __name__ == "__main__":
    main()
