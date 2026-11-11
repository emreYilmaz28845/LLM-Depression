#!/usr/bin/env python3
"""Build REPORT.md from ``merge_diagnostics.py`` outputs (``<dir>/<source>/<analysis>.json``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.merge_diag_common import DATASETS, SHORT  # noqa: E402

ARM_NAMES = {"standalone": "Standalone", "global": "Merged, global z", "per_dataset": "Merged, per-corpus z",
             "cb_center": "Merged, class-balanced centre", "combat": "Merged, ComBat",
             "consistent": "Merged, consistent dirs K=256", "per_dataset_thr": "Merged, per-corpus z + threshold"}
LABELS = {"androids_interview": "clinical diagnosis, patient vs control (64/52)",
          "cmdc": "clinical diagnosis, MDD vs healthy control (subject-id prefix)",
          "daic": "PHQ-8 >= 10 self-report", "d3tec": "PHQ-9 >= 10 self-report",
          "turkish": "BDI >= 17 self-report (pooled pos-only + negative-only conditions)"}


def load(d: Path, src: str, an: str):
    p = d / src / f"{an}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def bold_best(cells: list[tuple[float, str]]) -> list[str]:
    best = max(v for v, _ in cells)
    return [f"**{s}**" if v == best else s for v, s in cells]


def sec_cv(cv, lines):
    r = cv["result"]
    arms = list(ARM_NAMES)
    lines += ["One row = one corpus; cells = fold-mean macro-F1 / positive-F1 over the 5 CV folds "
              "(DAIC: 142 train+dev subjects). Best macro-F1 per row in bold.", "",
              "| Corpus | " + " | ".join(ARM_NAMES[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for d in DATASETS:
        cells = [(r["table"][d][a]["macro_f1"], f"{r['table'][d][a]['macro_f1']:.3f} / "
                                              f"{r['table'][d][a]['positive_f1']:.3f}") for a in arms]
        lines.append(f"| {SHORT[d]} | " + " | ".join(bold_best(cells)) + " |")
    lines.append("| Mean macro-F1 | " + " | ".join(f"{r['mean_macro_f1'][a]:.3f}" for a in arms) + " |")
    lines += ["", "Consistent-direction K sweep (mean macro-F1): " + ", ".join(
        f"K={k.split('_k')[1]} {r['mean_macro_f1'][k]:.3f}" for k in r["mean_macro_f1"] if k.startswith("consistent_k"))
        + f"; same-sign features per fold: {r['extra'].get('consistent_n_same_sign')}.", "",
        "Paired subject bootstrap of pooled macro-F1 (2000 resamples), Holm over all "
        f"{len(r['tests'])} tests in this table. Rows with raw p < 0.05:", ""]
    sig = [(k, v) for k, v in r["tests"].items() if v["p"] < 0.05]
    if not sig:
        lines.append("- none")
    for k, v in sorted(sig, key=lambda kv: kv[1]["p"]):
        d, cmp_ = k.split("|")
        lines.append(f"- {SHORT[d]}, {cmp_}: diff {v['diff']:+.3f} (95% CI {v['ci95'][0]:+.3f} to "
                     f"{v['ci95'][1]:+.3f}), p {v['p']:.3f}, Holm {v['p_holm']:.3f}")
    lines.append("")


def sec_mde(mde, lines):
    r = mde["result"]
    lines += ["| Corpus | n (CV) | SE macro-F1 | SE paired diff | MDE (alpha 0.05) | MDE (Holm, 552 tests) | "
              "Observed merged-global − standalone |", "|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        v = r[d]
        lines.append(f"| {SHORT[d]} | {v['n']} | {v['se_macro_f1']:.3f} | {v['se_paired_diff']:.3f} | "
                     f"{v['mde_alpha05']:.3f} | {v['mde_alpha_holm552']:.3f} | "
                     f"{v['observed_merged_global_minus_standalone']:+.3f} |")
    lines += ["", "MDE = (z_{1-α/2} + z_{0.8}) × SE of the paired difference, 80 % power, two-sided.", ""]


def sec_curve(curve, lines):
    r = curve["result"]
    fr = list(next(iter(r.values()))["macro_f1"])
    lines += ["Standalone head, fraction of each training fold (stratified; 10 draws below 100 %), fold-mean "
              "macro-F1. Slope = change in macro-F1 per doubling of training subjects (least squares on log2 n).", "",
              "| Corpus | " + " | ".join(f"{int(float(x) * 100)} %" for x in fr) + " | n train at 100 % | "
              "Slope / doubling | 50→100 % gain |"]
    lines.append("|---|" + "---|" * (len(fr) + 3))
    for d in DATASETS:
        v = r[d]
        lines.append(f"| {SHORT[d]} | " + " | ".join(f"{v['macro_f1'][x]:.3f}" for x in fr) +
                     f" | {v['n_train'][fr[-1]]:.0f} | {v['slope_per_doubling']:+.3f} | {v['half_to_full_gain']:+.3f} |")
    lines.append("")


def sec_direction(di, lines):
    r = di["result"]
    lines += ["Transfer AUROC: head trained on all subjects of the row corpus (corpus-standardized), applied to the "
              "column corpus (standardized with its own statistics). Threshold-free; 0.5 = no transfer.", "",
              "| Trained on \\ scored on | " + " | ".join(SHORT[d] for d in DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)]
    for a in DATASETS:
        lines.append(f"| {SHORT[a]} | " + " | ".join("—" if a == b else f"{r['transfer_auroc'][f'{a}->{b}']:.3f}"
                                                   for b in DATASETS) + " |")
    lines += ["", "Direction cosine on half samples vs the split-half ceiling (same corpus, two disjoint halves). "
              "Ratio = cross-corpus cosine / sqrt(ceiling_A × ceiling_B).", "",
              "| Corpus | Split-half cosine (ceiling) |", "|---|---|"]
    for d in DATASETS:
        lines.append(f"| {SHORT[d]} | {r['split_half'][d]['mean']:.3f} ± {r['split_half'][d]['sd']:.3f} |")
    lines += ["", "| Pair | Cross-corpus half cosine | Ceiling | Ratio |", "|---|---|---|---|"]
    for k, v in r["cosine_half"].items():
        a, b = k.split("|")
        lines.append(f"| {SHORT[a]} – {SHORT[b]} | {v['mean']:.3f} | {v['ceiling']:.3f} | {v['ratio']:.2f} |")
    lines.append("")


def sec_target_n(tn, lines):
    r = tn["result"]
    arms = ("target_only", "merged_global", "merged_per_dataset", "merged_cb_center")
    names = {"target_only": "Target only", "merged_global": "Merged global z",
             "merged_per_dataset": "Merged per-corpus z", "merged_cb_center": "Merged class-bal. centre"}
    lines += ["Leave-one-corpus-out with k labeled target subjects. Each of 20 draws holds out a stratified half of "
              "the target as test; labeled subjects come from the other half (target z-statistics use that half "
              "only). Cells = mean macro-F1 / positive-F1 over draws; best macro-F1 per row in bold. Target-only "
              "needs k ≥ 10.", ""]
    for d in DATASETS:
        v = r[d]
        ks = sorted({int(k) for a in arms for k in v["summary"][a]})
        lines += [f"**{SHORT[d]}** (test n = {v['test_n']}, pool n = {v['pool_n']})", "",
                  "| k | " + " | ".join(names[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
        for k in ks:
            cells = []
            for a in arms:
                s = v["summary"][a].get(str(k))
                cells.append((-1, "—") if s is None else (s["macro_f1"], f"{s['macro_f1']:.3f} / {s['positive_f1']:.3f}"))
            lines.append(f"| {k} | " + " | ".join(bold_best(cells)) + " |")
        lines.append("")


def sec_band(band, lines):
    r = band["result"]
    lines += ["Standalone out-of-fold logits. Band = subjects whose score lies next to the cut "
              "(Turkish BDI 14–19 around 17; PHQ 8–11 around 10). Cells = macro-F1 / AUROC.", "",
              "| Corpus | Cut | n / in band | Label = score ≥ cut | All | Inside band | Outside band | "
              "Spearman(logit, score) | Train without band: all / outside |", "|---|---|---|---|---|---|---|---|---|"]
    for d, v in r.items():
        lines.append(
            f"| {SHORT[d]} | {v['cut']:.0f} | {v['n']} / {v['n_band']} | {v['label_agrees_with_cut']:.3f} | "
            f"{v['all']['macro_f1']:.3f} / {v['all']['auroc']:.3f} | "
            f"{v['inside_band']['macro_f1']:.3f} / {v['inside_band']['auroc']:.3f} | "
            f"{v['outside_band']['macro_f1']:.3f} / {v['outside_band']['auroc']:.3f} | "
            f"{v['spearman_logit_score']['rho']:+.3f} (p {v['spearman_logit_score']['p']:.3f}) | "
            f"{v['train_without_band']['all']['macro_f1']:.3f} / {v['train_without_band']['outside_band']['macro_f1']:.3f} |")
    lines.append("")


def sec_tfidf(tf, lines):
    r = tf["result"]["per_dataset"]
    lines += ["Word 1–2 gram + char 2–5 gram TF-IDF, balanced logistic, C on 3 inner folds; native transcripts "
              "(see provenance note). Cells = fold-mean macro-F1 / positive-F1.", "",
              "| Corpus | TF-IDF | Qwen3.8 frozen standalone | Qwen − TF-IDF (pooled, 95% CI, p) |",
              "|---|---|---|---|"]
    for d in DATASETS:
        v = r[d]
        cells = bold_best([(v["tfidf"]["macro_f1"], f"{v['tfidf']['macro_f1']:.3f} / {v['tfidf']['positive_f1']:.3f}"),
                           (v["qwen_standalone"]["macro_f1"],
                            f"{v['qwen_standalone']['macro_f1']:.3f} / {v['qwen_standalone']['positive_f1']:.3f}")])
        b = v["qwen_minus_tfidf"]
        lines.append(f"| {SHORT[d]} | {cells[0]} | {cells[1]} | {b['diff']:+.3f} ({b['ci95'][0]:+.3f} to "
                     f"{b['ci95'][1]:+.3f}), p {b['p']:.3f} |")
    lines += ["", "Top word n-grams per class (fit on all CV subjects of the corpus, C = 1; local report only, "
              "do not copy into journal, PR or chat logs):", ""]
    for d in DATASETS:
        t = r[d]["top_word_ngrams"]
        lines.append(f"- {SHORT[d]}: depressed: {', '.join(t['depressed'])}; non-depressed: "
                     f"{', '.join(t['non_depressed'])}")
    lines.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    args = p.parse_args()
    lines = ["# Can the corpora be merged? Diagnostics and local normalization arms", ""]
    any_ = load(args.dir, "text_native", "cv") or load(args.dir, "audio_native", "cv")
    lines += [f"Lane `feat-merge-diagnostics-20261111`. Commit `{any_['git_head']}` (dirty: {any_['git_dirty']}). "
              "Local CPU. Frozen subject-mean vectors: Qwen3.8-27B text (native transcripts, 5120-d) and "
              "Qwen3-Omni-30B-A3B Thinker audio (2048-d); one vector per subject. Merged CV folds (= standalone "
              "folds); DAIC uses its 142 train+dev subjects, the 47 official test subjects are not used. "
              "Metrics: binary strict, subject level, threshold 0 on the logit. One deterministic run, seed "
              f"{any_['seed']}.", "",
              "Subjects in CV: " + ", ".join(f"{SHORT[d]} {n}" for d, n in any_["cv_subjects_per_dataset"].items())
              + ".", "", "## Label definitions (from the loaders in src/data)", "",
              "| Corpus | Label |", "|---|---|"] + [f"| {SHORT[d]} | {LABELS[d]} |" for d in DATASETS] + [""]
    for src in ("text_native", "audio_native"):
        title = {"text_native": "Text (Qwen3.8-27B, native)", "audio_native": "Audio (Qwen3-Omni, native)"}[src]
        parts = [("cv", "Merged CV: normalization arms", sec_cv), ("mde", "Minimal detectable effect", sec_mde),
                 ("curve", "Learning curves (standalone)", sec_curve),
                 ("direction", "Is the depression direction shared?", sec_direction),
                 ("target_n", "Leave-one-corpus-out with k target labels", sec_target_n),
                 ("band", "Label-threshold band", sec_band), ("tfidf", "Non-LLM TF-IDF baseline", sec_tfidf)]
        present = [(an, t, fn, load(args.dir, src, an)) for an, t, fn in parts]
        if not any(x[3] for x in present):
            continue
        lines += [f"# {title}", ""]
        for an, t, fn, data in present:
            if data:
                lines += [f"## {t}", "", f"_({an}.json, {data['seconds']} s)_", ""]
                fn(data, lines)
    (args.dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.dir / "REPORT.md")


if __name__ == "__main__":
    main()
