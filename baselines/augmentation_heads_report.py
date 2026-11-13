#!/usr/bin/env python3
"""Compare augmented frozen heads with the unchanged head (``none``).

Reads ``<root>/<variant>[/seed<k>]/<dataset>/<route>/summary.json`` written by
``run_augmentation_heads_matrix.py``. Per route the metric is the unweighted
fold mean of subject-level binary-strict metrics (DAIC: one official-test
fold). Augmented variants are averaged over their seeds first.

Test per variant: exact paired sign-flip permutation over the 15 route deltas
(macro-F1, seed mean minus ``none``), two-sided, Holm over variants. Routes
share subjects across modalities, so the p values are descriptive.
Writes ``report.json``, ``route_table.csv`` and ``REPORT.md`` into the root.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
ROUTES = ("audio_only_native", "text_only_native", "audio_text_native")
METRICS = ("macro_f1", "positive_f1", "uar")


def _summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def sign_flip_p(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = abs(deltas.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=len(deltas))))
    return float(np.mean(np.abs((signs * deltas).mean(1)) >= observed - 1e-12))


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[key]))
        out[key] = running
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    keys = [f"{d}/{r}" for d in DATASETS for r in ROUTES]
    base = {k: _summary(root / "none" / k) for k in keys}
    variants = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "none")
    table, report = [], {"variants": {}}
    for k in keys:
        row = {"route": k, "variant": "none", "seeds": 0,
               **{m: base[k]["fold_mean"][m] for m in METRICS},
               "selected_l2": ";".join(str(f["selected_l2_penalty"]) for f in base[k]["folds"])}
        table.append(row)
    deltas_all = {}
    for v in variants:
        seeds = sorted(p.name for p in (root / v).iterdir() if p.is_dir())
        deltas = []
        for k in keys:
            runs = [_summary(root / v / s / k) for s in seeds]
            vals = {m: float(np.mean([r["fold_mean"][m] for r in runs])) for m in METRICS}
            sd = float(np.std([r["fold_mean"]["macro_f1"] for r in runs]))
            table.append({"route": k, "variant": v, "seeds": len(seeds), **vals, "macro_f1_seed_sd": sd,
                          "selected_l2": ";".join(str(f["selected_l2_penalty"]) for r in runs for f in r["folds"])})
            deltas.append(vals["macro_f1"] - base[k]["fold_mean"]["macro_f1"])
        deltas = np.asarray(deltas)
        deltas_all[v] = deltas
        by_route = {r: float(np.mean([d for k, d in zip(keys, deltas) if k.endswith(r)])) for r in ROUTES}
        report["variants"][v] = {"seeds": seeds, "mean_delta_macro_f1": float(deltas.mean()),
                                 "wins": int((deltas > 1e-9).sum()), "losses": int((deltas < -1e-9).sum()),
                                 "ties": int((np.abs(deltas) <= 1e-9).sum()),
                                 "mean_delta_by_route": by_route, "p_sign_flip": sign_flip_p(deltas),
                                 "deltas": dict(zip(keys, deltas.tolist()))}
    adjusted = holm({v: r["p_sign_flip"] for v, r in report["variants"].items()})
    for v in report["variants"]:
        report["variants"][v]["p_holm"] = adjusted[v]
    report["none_mean_macro_f1"] = float(np.mean([base[k]["fold_mean"]["macro_f1"] for k in keys]))
    (root / "report.json").write_text(json.dumps(report, indent=2))
    with (root / "route_table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["route", "variant", "seeds", *METRICS, "macro_f1_seed_sd",
                                                "selected_l2"])
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Embedding-level augmentation for frozen subject-mean heads", "",
             "Macro-F1, subject level, binary strict, fold mean (DAIC: official test). "
             "Augmented = mean over seeds. Delta = augmented minus `none`.", "",
             "| Variant | Mean delta | audio | text | audio+text | W/L/T | p (sign-flip) | p (Holm) |",
             "|---|---|---|---|---|---|---|---|"]
    for v, r in sorted(report["variants"].items(), key=lambda kv: -kv[1]["mean_delta_macro_f1"]):
        b = r["mean_delta_by_route"]
        lines.append(f"| {v} | {r['mean_delta_macro_f1']:+.3f} | {b['audio_only_native']:+.3f} | "
                     f"{b['text_only_native']:+.3f} | {b['audio_text_native']:+.3f} | "
                     f"{r['wins']}/{r['losses']}/{r['ties']} | {r['p_sign_flip']:.3f} | {r['p_holm']:.3f} |")
    lines += ["", "## Per route (macro-F1)", "", "| Route | none | " + " | ".join(variants) + " |",
              "|---|---|" + "---|" * len(variants)]
    for k in keys:
        cells = [f"{base[k]['fold_mean']['macro_f1']:.3f}"]
        cells += [f"{base[k]['fold_mean']['macro_f1'] + report['variants'][v]['deltas'][k]:.3f}" for v in variants]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(variants) + 2]))


if __name__ == "__main__":
    main()
