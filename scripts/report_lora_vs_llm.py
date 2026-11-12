"""Side-by-side aggregate report and English figures: direct LLM likelihood vs LoRA-embedding heads.

Reads the outputs of overlap_oracle_analysis, margin_selector_analysis, model_selector_analysis and
seed_ensemble_analysis run with --backend llm and --backend lora_heads under one root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"cmdc": "Mandarin (CMDC)", "androids_interview": "Italian (Androids)", "daic": "English (DAIC-WOZ)",
         "turkish": "Turkish", "d3tec": "Spanish (D3TEC)"}
SHORT = {"cmdc": "Mandarin", "androids_interview": "Italian", "daic": "English", "turkish": "Turkish", "d3tec": "Spanish"}
BACK = {"llm": "Direct LLM (likelihood)", "lora_heads": "LoRA-embedding head"}
METHODS = [("margin", "max_conf", "Max-confidence"), ("margin", "advisor_gate", "Advisor gate"),
           ("selectors", "ola", "OLA"), ("selectors", "lca", "LCA"), ("selectors", "vote3", "3-model vote"),
           ("selectors", "stack", "Stacking"), ("seed_ensemble", "text_ens3", "Text 3-seed ensemble"),
           ("seed_ensemble", "ens6", "6-model ensemble"), ("seed_ensemble", "seed_agree", "Seed-agreement selector")]
PERF = {"margin": "margin_selector_performance.csv", "selectors": "selector_performance.csv",
        "seed_ensemble": "seed_ensemble_performance.csv"}
TESTS = {"margin": ("margin_selector_tests.csv", "rule"), "selectors": ("selector_tests.csv", "rule"),
         "seed_ensemble": ("seed_ensemble_tests.csv", "method")}


def load(root, b):
    perf, tests = {}, {}
    for k, f in PERF.items():
        p = pd.read_csv(root / b / k / f); perf[k] = p[p.method.isin(p.method.unique()) & ~p.method.str.endswith("_share")]
        t, col = TESTS[k]; t = pd.read_csv(root / b / k / t); tests[k] = t.rename(columns={col: "method"})
    return perf, tests


def mf1(perf, k, m, ds):
    p = perf[k]; return float(p[(p.corpus == ds) & (p.method == m)].macro_f1.mean())


def delta_table(perf, tests):
    rows = []
    for k, m, lab in METHODS:
        for ds in NAMES:
            base = mf1(perf, k, "text", ds)
            t = tests[k]; t = t[(t.corpus == ds) & (t.method == m) & (t.reference == "text")]
            rows.append({"method": lab, "corpus": ds, "macro_f1": mf1(perf, k, m, ds), "text": base,
                         "delta": mf1(perf, k, m, ds) - base,
                         "perm_p": float(t.permutation_p.iloc[0]) if len(t) else np.nan,
                         "holm_p": float(t.holm_p.iloc[0]) if len(t) else np.nan})
    return pd.DataFrame(rows)


def fig_overlap(root, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6)); order = list(NAMES); w = .38
    for j, (b, col) in enumerate((("llm", "#3a7bd5"), ("lora_heads", "#e07b39"))):
        e = pd.read_csv(root / b / "overlap/error_overlap.csv").groupby("corpus").shared_error_pct_of_union.mean()
        axes[0].bar([i + (j - .5) * w for i in range(len(order))], [e[c] for c in order], w, color=col, label=BACK[b])
    axes[0].set_ylabel("Shared errors (% of all errors)"); axes[0].set_ylim(0, 100)
    axes[0].set_title("Do text and audio+text make the same errors?\n(mean over 3 seeds)", fontsize=10)
    styles = [("llm", "text_vs_audio_text", "#3a7bd5", "", "LLM: text vs A+T"),
              ("llm", "text_only_seed_pair", "#3a7bd5", "//", "LLM: two text seeds"),
              ("lora_heads", "text_vs_audio_text", "#e07b39", "", "Head: text vs A+T"),
              ("lora_heads", "text_only_seed_pair", "#e07b39", "//", "Head: two text seeds")]
    w2 = .2
    for j, (b, comp, col, hatch, lab) in enumerate(styles):
        o = pd.read_csv(root / b / "overlap/oracle_seed_noise.csv"); o = o[o.comparison == comp].groupby("corpus").oracle_gain.mean()
        axes[1].bar([i + (j - 1.5) * w2 for i in range(len(order))], [o[c] for c in order], w2, color=col,
                    hatch=hatch, edgecolor="white" if not hatch else "black", linewidth=.4, label=lab)
    axes[1].set_ylabel("Oracle gain over the better model (Macro-F1)")
    axes[1].set_title("Oracle gain: text vs A+T compared with two seeds of the same text model\n"
                      "(similar bars = the gap is mostly seed noise)", fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[c] for c in order])
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def fig_deltas(tables, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    vmax = max(abs(t.delta).max() for t in tables.values())
    for ax, (b, t) in zip(axes, tables.items()):
        piv = t.pivot(index="method", columns="corpus", values="delta").loc[[m[2] for m in METHODS], list(NAMES)]
        im = ax.imshow(piv.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, m in enumerate(piv.index):
            for j, c in enumerate(piv.columns):
                r = t[(t.method == m) & (t.corpus == c)].iloc[0]
                mark = "**" if r.holm_p < .05 else ("*" if r.perm_p < .05 else "")
                ax.text(j, i, f"{r.delta:+.3f}{mark}", ha="center", va="center", fontsize=8,
                        color="white" if abs(r.delta) > .6 * vmax else "black")
        ax.set_xticks(range(len(NAMES))); ax.set_xticklabels([SHORT[c] for c in NAMES])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_title(f"{BACK[b]}\nMacro-F1 change vs single-seed text", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=.8, label="Macro-F1 difference")
    fig.suptitle("All selectors and ensembles, both decision rules  (* p<0.05 unadjusted, ** Holm p<0.05)", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def fig_agreement(root, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True); order = list(NAMES); w = .2
    for ax, b in zip(axes, ("llm", "lora_heads")):
        d = pd.read_csv(root / b / "seed_ensemble/seed_agreement_diagnostic.csv")
        for j, (mod, lv, col, lab) in enumerate([("text", 3, "#1d4e89", "Text, 3/3 agree"), ("text", 2, "#9cc3f0", "Text, 2/3 agree"),
                                                 ("audio_text", 3, "#a0522d", "A+T, 3/3 agree"), ("audio_text", 2, "#f2b48a", "A+T, 2/3 agree")]):
            s = d[(d.model == mod) & (d.seeds_agreeing == lv)].set_index("corpus")
            vals = [100 * s.loc[c].ensemble_accuracy if s.loc[c].n else 0 for c in order]
            xs = [i + (j - 1.5) * w for i in range(len(order))]
            ax.bar(xs, vals, w, color=col, label=lab)
            for x, c in zip(xs, order):
                ax.text(x, 2, str(int(s.loc[c].n)), ha="center", fontsize=6.5, color="white" if lv == 3 else "black")
        ax.set_title(f"{BACK[b]}: accuracy of the 3-seed vote\n(number in bar = subjects)", fontsize=10)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[c] for c in order]); ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    fig.suptitle("Is seed agreement a confidence signal under both decision rules?", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(path, dpi=160)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    tables = {}
    L = ["# Direct LLM likelihood vs LoRA-embedding heads: overlap, selectors and seed ensembles", "",
         "Same LoRA runs, seeds (7, 1337, 2024), folds and outer subjects; only the decision rule differs: the LLM's "
         "candidate-label likelihood vs an L2 logistic head on the LoRA model's subject-mean embeddings. "
         "All analyses are re-run here for both backends from one source commit. No inference.", ""]
    cfgs = {b: json.loads((root / b / "seed_ensemble/seed_ensemble_config.json").read_text()) for b in BACK}
    L += [f"- Source commit `{cfgs['llm']['source_commit']}`.",
          "- LoRA heads: every head refit from its cache with the saved L2 penalty reproduced the saved outer logits "
          "(tolerance 1e-6); every head's LoRA checkpoint is the same run as the matching LLM cell; outer and "
          "(Androids, D3TEC) inner subjects match the LLM cells exactly; no text LoRA vector equals the frozen vector.",
          "- Selection sets for heads: Androids/D3TEC use the LLM's inner-validation subjects; Turkish/CMDC use "
          "leave-one-fold-out as for the LLM; **DAIC heads use inner-CV out-of-fold logits on LoRA-trained "
          "subjects (no LoRA dev embeddings exist), so DAIC tuned-selector results for heads are optimistic and not "
          "comparable** (DAIC seed ensembles and votes need no selection set).", ""]
    for b in BACK:
        perf, tests = load(root, b); t = delta_table(perf, tests); tables[b] = t
    L += ["## Single models (fold-mean Macro-F1, mean of 3 seeds)", "", "| Corpus | LLM text | LLM A+T | Head text | Head A+T |",
          "|---|---:|---:|---:|---:|"]
    pl, ph = load(root, "llm")[0], load(root, "lora_heads")[0]
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | {mf1(pl, 'margin', 'text', ds):.3f} | {mf1(pl, 'margin', 'audio_text', ds):.3f} | "
                 f"{mf1(ph, 'margin', 'text', ds):.3f} | {mf1(ph, 'margin', 'audio_text', ds):.3f} |")
    L += ["", "## Error overlap and oracle vs seed noise", "",
          "| Corpus | Shared errors LLM | Shared errors head | Oracle gain LLM text vs A+T | LLM two text seeds | "
          "Oracle gain head text vs A+T | Head two text seeds |", "|---|---:|---:|---:|---:|---:|---:|"]
    ov = {b: pd.read_csv(root / b / "overlap/error_overlap.csv").groupby("corpus").shared_error_pct_of_union.mean() for b in BACK}
    oc = {b: pd.read_csv(root / b / "overlap/oracle_seed_noise.csv").groupby(["corpus", "comparison"]).oracle_gain.mean() for b in BACK}
    for ds in NAMES:
        L.append(f"| {NAMES[ds]} | {ov['llm'][ds]:.0f}% | {ov['lora_heads'][ds]:.0f}% | "
                 f"{oc['llm'][(ds, 'text_vs_audio_text')]:+.3f} | {oc['llm'][(ds, 'text_only_seed_pair')]:+.3f} | "
                 f"{oc['lora_heads'][(ds, 'text_vs_audio_text')]:+.3f} | {oc['lora_heads'][(ds, 'text_only_seed_pair')]:+.3f} |")
    for b in BACK:
        t = tables[b]
        L += ["", f"## {BACK[b]}: every method vs single-seed text", "",
              "Cells: Macro-F1 (change vs text; unadjusted permutation p; Holm p within that analysis's own family).", "",
              "| Method | " + " | ".join(SHORT[c] for c in NAMES) + " |", "|---|" + "---|" * len(NAMES)]
        for _, _, lab in METHODS:
            cells = []
            for ds in NAMES:
                r = t[(t.method == lab) & (t.corpus == ds)].iloc[0]
                cells.append(f"{r.macro_f1:.3f} ({r.delta:+.3f}; p {r.perm_p:.2f}; Holm {r.holm_p:.2f})")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append(f"| Text (reference) | " + " | ".join(f"{t[t.corpus == ds].text.iloc[0]:.3f}" for ds in NAMES) + " |")
    L += ["", "## Seed agreement as confidence", "", "| Corpus | Backend | Model | 3/3 accuracy (n) | 2/3 accuracy (n) |",
          "|---|---|---|---:|---:|"]
    for ds in NAMES:
        for b in BACK:
            d = pd.read_csv(root / b / "seed_ensemble/seed_agreement_diagnostic.csv")
            for mod in ("text", "audio_text"):
                a = d[(d.corpus == ds) & (d.model == mod)].set_index("seeds_agreeing")
                f = lambda lv: "-" if not a.loc[lv].n else f"{a.loc[lv].ensemble_accuracy:.2f} ({int(a.loc[lv].n)})"  # noqa: E731
                L.append(f"| {NAMES[ds]} | {BACK[b]} | {'Text' if mod == 'text' else 'A+T'} | {f(3)} | {f(2)} |")
    L += ["", "## Limitations", "",
          "- Holm families are per analysis (margin: 10, selectors: 20, seed ensembles: 15 primary); no correction across analyses or backends.",
          "- Ensembles use 3 or 6 trained models; single-seed text uses 1.",
          "- Turkish and CMDC are outer validation, not locked test; DAIC is one official test fold.",
          "- Bootstrap conditions on fixed models and selection sets.", ""]
    (root / "LORA_VS_LLM_REPORT.md").write_text("\n".join(L) + "\n")
    fig_overlap(root, root / "overlap_oracle_lora_vs_llm.png")
    fig_deltas(tables, root / "methods_vs_text_lora_vs_llm.png")
    fig_agreement(root, root / "seed_agreement_lora_vs_llm.png")


if __name__ == "__main__":
    main()
