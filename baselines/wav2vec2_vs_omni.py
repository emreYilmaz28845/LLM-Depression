#!/usr/bin/env python3
"""Paired comparison: frozen wav2vec2 vs frozen Qwen3-Omni, same subject-mean L2 head.

Both sides are ``baselines/frozen_standalone_heads.py`` outputs on the same
split files, so every fold reports the same subjects (checked, fails closed).
A dataset's score is the fold-mean of subject-level binary-strict metrics
(DAIC: one official-test fold).

Tests: two-sided paired t-tests on macro-F1 differences (wav2vec2 - Omni), no
multiplicity correction. Primary unit = dataset (fold-mean score, so the tested
mean equals the table's mean difference). Secondary unit = dataset x fold cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
METRICS = ("macro_f1", "positive_f1", "uar")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_route(route_dir: Path) -> dict:
    summary = json.loads((route_dir / "summary.json").read_text())
    folds = []
    for fold in summary["folds"]:
        predictions = _read_jsonl(route_dir / f"fold_{fold['fold']}" / "predictions_subject_level.jsonl")
        folds.append({**{m: float(fold[m]) for m in METRICS}, "fold": int(fold["fold"]),
                      "subjects": sorted(str(p["subject_id"]) for p in predictions),
                      "selected_l2": fold["selected_l2_penalty"]})
    return {"folds": folds, "fold_mean": {m: float(np.mean([f[m] for f in folds])) for m in METRICS}}


def paired_t(diffs: list[float]) -> dict:
    d = np.asarray(diffs, dtype=float)
    out = {"n": int(d.size), "mean_diff": float(d.mean()) if d.size else None, "t": None, "p": None}
    if d.size >= 2 and np.ptp(d) > 0:
        result = stats.ttest_1samp(d, 0.0)
        out.update({"t": float(result.statistic), "p": float(result.pvalue)})
    return out


def compare(omni_root: Path, w2v_root: Path, pooling: str) -> dict:
    rows, cells = [], []
    for dataset in DATASETS:
        omni = load_route(omni_root / dataset)
        w2v = load_route(w2v_root / pooling / dataset)
        if [f["fold"] for f in omni["folds"]] != [f["fold"] for f in w2v["folds"]]:
            raise ValueError(f"{dataset}: fold lists differ.")
        for a, b in zip(omni["folds"], w2v["folds"]):
            if a["subjects"] != b["subjects"]:
                raise ValueError(f"{dataset} fold {a['fold']}: reported subjects differ; pairing is invalid.")
            cells.append({"dataset": dataset, "fold": a["fold"], "omni_macro_f1": a["macro_f1"],
                          "w2v_macro_f1": b["macro_f1"], "diff": b["macro_f1"] - a["macro_f1"],
                          "omni_l2": a["selected_l2"], "w2v_l2": b["selected_l2"], "n_subjects": len(a["subjects"])})
        row = {"dataset": dataset, "folds": len(omni["folds"])}
        for m in METRICS:
            row[f"omni_{m}"] = omni["fold_mean"][m]
            row[f"w2v_{m}"] = w2v["fold_mean"][m]
        row["diff_macro_f1"] = row["w2v_macro_f1"] - row["omni_macro_f1"]
        rows.append(row)
    return {
        "pooling": pooling,
        "datasets": rows,
        "cells": cells,
        "mean": {f"{side}_{m}": float(np.mean([r[f"{side}_{m}"] for r in rows]))
                 for side in ("omni", "w2v") for m in METRICS},
        "test_dataset_unit": paired_t([r["diff_macro_f1"] for r in rows]),
        "test_fold_cell_unit": paired_t([c["diff"] for c in cells]),
    }


def _fmt(p):
    return "n/a" if p is None else f"{p:.3f}"


def markdown(results: list[dict]) -> str:
    names = {"daic": "DAIC-WOZ (official test)", "d3tec": "D3TEC", "androids_interview": "Androids interview",
             "cmdc": "CMDC", "turkish": "Turkish pooled"}
    lines = ["| Dataset | Qwen3-Omni frozen Macro-F1 / Pos-F1 / UAR | "
             + " | ".join(f"wav2vec2 {r['pooling']} Macro-F1 / Pos-F1 / UAR" for r in results) + " |",
             "|---|---|" + "---|" * len(results)]
    base = results[0]
    for i, row in enumerate(base["datasets"]):
        cells = [f"{row['omni_macro_f1']:.3f} / {row['omni_positive_f1']:.3f} / {row['omni_uar']:.3f}"]
        for r in results:
            x = r["datasets"][i]
            cells.append(f"{x['w2v_macro_f1']:.3f} / {x['w2v_positive_f1']:.3f} / {x['w2v_uar']:.3f}")
        lines.append(f"| {names[row['dataset']]} | " + " | ".join(cells) + " |")
    mean = [f"{base['mean']['omni_macro_f1']:.3f} / {base['mean']['omni_positive_f1']:.3f} / {base['mean']['omni_uar']:.3f}"]
    mean += [f"{r['mean']['w2v_macro_f1']:.3f} / {r['mean']['w2v_positive_f1']:.3f} / {r['mean']['w2v_uar']:.3f}"
             for r in results]
    lines.append("| Mean of datasets | " + " | ".join(mean) + " |")
    lines.append("")
    for r in results:
        a, b = r["test_dataset_unit"], r["test_fold_cell_unit"]
        lines.append(f"- wav2vec2 {r['pooling']} - Omni, macro-F1: mean diff {a['mean_diff']:+.3f}, "
                     f"paired t-test p {_fmt(a['p'])} (dataset unit, n={a['n']}); "
                     f"fold-cell unit mean {b['mean_diff']:+.3f}, p {_fmt(b['p'])} (n={b['n']}).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omni-root", required=True, type=Path)
    parser.add_argument("--w2v-root", required=True, type=Path, help="Holds <pooling>/<dataset>/ head outputs.")
    parser.add_argument("--poolings", nargs="+", default=["meanvar", "mean"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    results = [compare(args.omni_root, args.w2v_root, p) for p in args.poolings]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    (args.output_dir / "comparison.md").write_text(markdown(results))
    with (args.output_dir / "fold_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pooling", *results[0]["cells"][0]])
        writer.writeheader()
        for r in results:
            writer.writerows({"pooling": r["pooling"], **c} for c in r["cells"])
    print(markdown(results))


if __name__ == "__main__":
    main()
