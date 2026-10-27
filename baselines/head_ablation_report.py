#!/usr/bin/env python3
"""Markdown report from ``head_ablation_stats`` output (ablation_stats.json)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("frozen_paper", "frozen_new", "lora_paper", "lora_new", "lora_llm")
HEADERS = ("Frozen + paper head", "Frozen + new head", "LoRA + paper head", "LoRA + new head", "LoRA LLM output")
SCOPES = (("standalone", "Standalone (one model per dataset)"),
          ("merged_cv", "Merged, 5-fold CV over all datasets"),
          ("merged_final", "Merged final model, DAIC official test"))
DATASET_NAMES = {"androids_interview": "Androids", "cmdc": "CMDC", "d3tec": "D3TEC", "daic": "DAIC", "turkish": "Turkish"}
ROUTE_NAMES = {"audio_only": "Audio", "audio_text": "Audio+text", "text_only": "Text"}
COMPARISON_NAMES = {
    "head_effect_frozen": "New head − paper head (frozen)",
    "head_effect_lora": "New head − paper head (LoRA)",
    "lora_effect_new_head": "LoRA − frozen (new head)",
    "lora_effect_paper_head": "LoRA − frozen (paper head)",
    "lora_llm_vs_frozen_new": "LoRA LLM output − frozen + new head",
}


def _route(route: str) -> str:
    modality, language = route.rsplit("_", 1)
    return f"{ROUTE_NAMES[modality]} ({'EN' if language == 'english' else 'native'})"


def render(stats: dict) -> str:
    lines = ["# Frozen vs LoRA × paper vs new head", "",
             "Cells: macro-F1 / positive-F1 (fold-mean, subject level, binary strict). LoRA cells are the mean of 3 seeds; "
             "frozen cells are one deterministic run. Green: highest macro-F1 in the row.", ""]
    rows = defaultdict(dict)
    for r in stats["table"]:
        rows[(r["scope"], r["dataset"], r["route"])][r["condition"]] = r
    for scope, title in SCOPES:
        keys = sorted(k for k in rows if k[0] == scope)
        if not keys:
            continue
        lines += [f"## {title}", "", "| Dataset | Route | " + " | ".join(HEADERS) + " |",
                  "|---|---|" + "---|" * len(HEADERS)]
        for key in keys:
            cells = rows[key]
            best = max((cells[c]["macro_f1"] for c in CONDITIONS if c in cells), default=None)
            out = []
            for c in CONDITIONS:
                if c not in cells:
                    out.append("—")
                    continue
                text = f"{cells[c]['macro_f1']:.3f} / {cells[c]['positive_f1']:.3f}"
                out.append(f"🟩 **{text}**" if abs(cells[c]["macro_f1"] - best) < 1e-12 else text)
            lines.append(f"| {DATASET_NAMES.get(key[1], key[1])} | {_route(key[2])} | " + " | ".join(out) + " |")
        lines.append("")
    lines += ["## Paired tests (macro-F1)", "",
              f"Subject-level paired bootstrap, {stats['settings']['replicates']} replicates; "
              "resampling inside each class per dataset and fold, shared across routes, conditions and seeds. "
              "Group difference = mean over routes. Holm correction within each scope.", "",
              "| Scope | Group | Comparison | Routes | Mean diff | 95% CI | p | p (Holm) | Routes up / down |",
              "|---|---|---|---|---|---|---|---|---|"]
    for t in stats["tests"]:
        mark = " **" if t["p_holm"] < 0.05 else " "
        lines.append(f"| {t['scope']} | {t['group']} | {COMPARISON_NAMES[t['comparison']]} | {t['n_routes']} |"
                     f"{mark}{t['mean_diff_macro_f1']:+.3f}{mark.strip() and '**'} | "
                     f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['p_bootstrap']:.4f} | {t['p_holm']:.4f} | "
                     f"{t['routes_a_better']} / {t['routes_b_better']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}.")
    args.output.write_text(render(json.loads(args.stats.read_text())))


if __name__ == "__main__":
    main()
