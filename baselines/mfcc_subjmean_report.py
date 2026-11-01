#!/usr/bin/env python3
"""openSMILE MFCC under the subject-mean L2 head vs Qwen3-Omni, wav2vec2 and the old MFCC XGBoost.

Every head side is a ``frozen_standalone_heads.py`` output on the same split
files, so each fold must report the same subjects (checked, fails closed). The
old fixed XGBoost MFCC head (``outputs/mfcc_xgb/<dataset>/fold_<n>/fixed_heads/xgb_raw``)
is read from its own subject predictions; it is compared only on folds whose
subjects match. Scores are fold means of subject-level binary-strict metrics.
Paired t-tests on Macro-F1 differences (MFCC new head - other), no multiplicity
correction; units = dataset (fold mean) and dataset x fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from baselines.wav2vec2_vs_omni import DATASETS, METRICS, _read_jsonl, load_route, paired_t

OLD_MFCC_DIRS = {"daic": "daic", "d3tec": "d3tec", "androids_interview": "androids_standalone",
                 "cmdc": "cmdc", "turkish": "turkish_pooled"}


def load_old_xgb(root: Path, n_folds: int) -> dict:
    folds = []
    for fold in range(n_folds):
        d = root / f"fold_{fold}" / "fixed_heads" / "xgb_raw"
        m = json.loads((d / "metrics.json").read_text())
        subjects = sorted(str(p["subject_id"]) for p in _read_jsonl(d / "predictions_subject_level.jsonl"))
        folds.append({"fold": fold, "subjects": subjects, "macro_f1": float(m["macro_f1"]),
                      "positive_f1": float(m["positive_f1"]), "uar": float(m["binary_strict_uar"])})
    return {"folds": folds, "fold_mean": {k: float(np.mean([f[k] for f in folds])) for k in METRICS}}


def report(omni_root: Path, w2v_root: Path, mfcc_root: Path, old_root: Path) -> dict:
    sides = ("omni", "w2v_meanvar", "mfcc_xgb_old")
    rows, cells = [], {s: [] for s in sides}
    for d in DATASETS:
        new = load_route(mfcc_root / d)
        ref = {"omni": load_route(omni_root / d), "w2v_meanvar": load_route(w2v_root / "meanvar" / d)}
        ref["mfcc_xgb_old"] = load_old_xgb(old_root / OLD_MFCC_DIRS[d], len(new["folds"]))
        row = {"dataset": d, "mfcc_subjmean": new["fold_mean"],
               "mfcc_selected_l2": [f["selected_l2"] for f in new["folds"]]}
        for side in sides:
            matched = all(a["subjects"] == b["subjects"] for a, b in zip(new["folds"], ref[side]["folds"]))
            if side != "mfcc_xgb_old" and not matched:
                raise ValueError(f"{d}: {side} reports different subjects.")
            row[side] = ref[side]["fold_mean"]
            row[f"{side}_subjects_match"] = matched
            if matched:
                cells[side] += [a["macro_f1"] - b["macro_f1"] for a, b in zip(new["folds"], ref[side]["folds"])]
        rows.append(row)
    tests = {}
    for side in sides:
        usable = [r for r in rows if r[f"{side}_subjects_match"]]
        tests[f"mfcc_subjmean_minus_{side}"] = {
            "datasets": [r["dataset"] for r in usable],
            "dataset_unit": paired_t([r["mfcc_subjmean"]["macro_f1"] - r[side]["macro_f1"] for r in usable]),
            "fold_cell_unit": paired_t(cells[side]),
        }
    mean = {k: float(np.mean([r[k]["macro_f1"] for r in rows])) for k in ("mfcc_subjmean",) + sides}
    return {"datasets": rows, "mean_macro_f1": mean, "tests": tests,
            "aggregation": "unweighted_fold_mean_of_subject_level_metrics"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omni-root", required=True, type=Path, help="<dataset>/ Omni frozen head outputs.")
    parser.add_argument("--w2v-root", required=True, type=Path, help="<pooling>/<dataset>/ wav2vec2 head outputs.")
    parser.add_argument("--mfcc-root", required=True, type=Path, help="<dataset>/ MFCC subject-mean head outputs.")
    parser.add_argument("--old-mfcc-root", required=True, type=Path, help="outputs/mfcc_xgb of the MFCC lane.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    out = report(args.omni_root, args.w2v_root, args.mfcc_root, args.old_mfcc_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for r in out["datasets"]:
        print(r["dataset"], {k: round(r[k]["macro_f1"], 3) for k in ("mfcc_subjmean", "omni", "w2v_meanvar",
                                                                       "mfcc_xgb_old")},
              "old subjects match", r["mfcc_xgb_old_subjects_match"], "L2", r["mfcc_selected_l2"])
    print("mean", {k: round(v, 3) for k, v in out["mean_macro_f1"].items()})
    for k, t in out["tests"].items():
        print(k, t["datasets"], "dataset p", t["dataset_unit"]["p"], "fold p", t["fold_cell_unit"]["p"],
              "n", t["fold_cell_unit"]["n"])


if __name__ == "__main__":
    main()
