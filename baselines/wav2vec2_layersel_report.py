#!/usr/bin/env python3
"""Layer-selected wav2vec2 vs last-layer wav2vec2 vs Qwen3-Omni frozen (same head, folds, subjects).

Reads ``frozen_standalone_heads.py`` outputs (Omni, wav2vec2 last layer) and
``layer_select_heads.py`` outputs (wav2vec2 selected layer). Every fold must
report the same subjects in all three. Paired t-tests on Macro-F1 differences,
no multiplicity correction; units = dataset (fold-mean) and dataset x fold.
The per-layer test profile is copied from ``per_layer_test.json`` as a
diagnostic only (it is never used for selection).
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

from baselines.wav2vec2_vs_omni import DATASETS, load_route, paired_t


def report(omni_root: Path, last_root: Path, layersel_root: Path, pooling: str) -> dict:
    rows, cells_omni, cells_last = [], [], []
    for d in DATASETS:
        omni, last = load_route(omni_root / d), load_route(last_root / pooling / d)
        sel = load_route(layersel_root / pooling / d)
        for a, b, c in zip(omni["folds"], last["folds"], sel["folds"]):
            if not a["subjects"] == b["subjects"] == c["subjects"]:
                raise ValueError(f"{d} fold {a['fold']}: reported subjects differ.")
            cells_omni.append(c["macro_f1"] - a["macro_f1"])
            cells_last.append(c["macro_f1"] - b["macro_f1"])
        summary = json.loads((layersel_root / pooling / d / "summary.json").read_text())
        diag = json.loads((layersel_root / pooling / d / "per_layer_test.json").read_text())["layers"]
        best = max(diag, key=lambda k: diag[k]["mean_macro_f1"])
        n_layers = len(diag)
        rows.append({
            "dataset": d, "omni": omni["fold_mean"], "last": last["fold_mean"], "layersel": sel["fold_mean"],
            "selected_layers": summary["selected_layers"],
            "diag_best_test_layer": int(best), "diag_best_test_macro_f1": diag[best]["mean_macro_f1"],
            "diag_profile_macro_f1": [diag[str(k)]["mean_macro_f1"] for k in range(n_layers)],
            "diag_profile_selection_f1": [float(np.mean(diag[str(k)]["fold_selection_macro_f1"]))
                                          for k in range(n_layers)],
        })
    return {
        "pooling": pooling,
        "datasets": rows,
        "mean": {k: float(np.mean([r[k]["macro_f1"] for r in rows])) for k in ("omni", "last", "layersel")},
        "layersel_vs_omni": {"dataset_unit": paired_t([r["layersel"]["macro_f1"] - r["omni"]["macro_f1"] for r in rows]),
                             "fold_cell_unit": paired_t(cells_omni)},
        "layersel_vs_last": {"dataset_unit": paired_t([r["layersel"]["macro_f1"] - r["last"]["macro_f1"] for r in rows]),
                             "fold_cell_unit": paired_t(cells_last)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omni-root", required=True, type=Path)
    parser.add_argument("--last-root", required=True, type=Path, help="<pooling>/<dataset> last-layer heads.")
    parser.add_argument("--layersel-root", required=True, type=Path, help="<pooling>/<dataset> layer-selected heads.")
    parser.add_argument("--poolings", nargs="+", default=["meanvar", "mean"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = {p: report(args.omni_root, args.last_root, args.layersel_root, p) for p in args.poolings}
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for pool, r in out.items():
        print(f"{pool}: mean Macro-F1 omni {r['mean']['omni']:.3f} last {r['mean']['last']:.3f} "
              f"layersel {r['mean']['layersel']:.3f}; selected vs last p {r['layersel_vs_last']['dataset_unit']['p']}; "
              f"selected vs omni p {r['layersel_vs_omni']['dataset_unit']['p']}")


if __name__ == "__main__":
    main()
