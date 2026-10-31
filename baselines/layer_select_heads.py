#!/usr/bin/env python3
"""Subject-mean L2 head with the hidden layer chosen jointly with L2 on inner validation.

Input: ``<layers-root>/layer_XX`` caches (one per hidden state) from
``src/features/extract_wav2vec2.py --all-layers``; DAIC also takes the matching
official-dev caches. For every layer the fold runs exactly
``baselines/frozen_standalone_heads.run_fold`` (same scaler, head, L2 grid and
inner protocol). The reported (layer, L2) is the candidate with the highest
selection macro-F1, then lower selection BCE, then larger L2, then lower layer:
the same rule as the L2-only head, with the layer added last. Selection never
sees the reported subjects.

``per_layer_test.json`` lists each layer's reported-subject scores. It is a
diagnostic of the layer profile only; it is never used for selection.

Output layout matches ``frozen_standalone_heads.py`` (``summary.json`` and
``fold_<n>/``), so ``baselines/wav2vec2_vs_omni.py`` reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from baselines.frozen_standalone_heads import (
    DEFAULT_L2_GRID,
    INNER_FOLDS,
    INNER_SEED,
    PROTOCOLS,
    _load_caches,
    _write_csv,
    run_fold,
)
from src.features.hidden_classifier_policy import cache_identity, canonical_sha256
from src.utils import read_json, save_json, write_jsonl

SCHEMA_VERSION = "layer_select_heads.v1"
VARIANT = "logreg_subjmean_l2sel_layersel"
LAYER_DIR = re.compile(r"^layer_(\d{2})$")


def layer_dirs(root: Path) -> dict[int, Path]:
    found = {int(m.group(1)): p for p in sorted(root.iterdir()) if (m := LAYER_DIR.match(p.name))}
    if not found or sorted(found) != list(range(len(found))):
        raise ValueError(f"{root}: expected layer_00..layer_NN, found {sorted(found)}")
    return found


def selection_key(best: dict[str, Any], layer: int) -> tuple:
    return (best["macro_f1"], -best["binary_cross_entropy"], best["l2_penalty"], -layer)


def run(*, layers_root: Path, dev_layers_root: Path | None, split_files: list[Path], protocol: str,
        output_dir: Path, route: dict[str, Any], grid=DEFAULT_L2_GRID) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}.")
    if (protocol == "daic_official") != (dev_layers_root is not None):
        raise ValueError("A dev layers root is required for, and only for, daic_official.")
    main_layers = layer_dirs(layers_root)
    dev_layers = layer_dirs(dev_layers_root) if dev_layers_root else None
    if dev_layers is not None and sorted(dev_layers) != sorted(main_layers):
        raise ValueError("Dev and main caches hold different layers.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing output: {output_dir}")
    staged = output_dir.with_name(output_dir.name + ".partial")
    if staged.exists():
        raise ValueError(f"A partial output from an earlier failed run exists: {staged}")

    splits = [read_json(p) for p in split_files]
    per_layer: dict[int, list[dict[str, Any]]] = {}
    sample_ids = None
    manifest_hashes = set()
    for layer, path in main_layers.items():
        cache, meta = _load_caches(path, dev_layers[layer] if dev_layers else None, allow_adapter=False)
        if int(meta.get("hidden_layer_index", -1)) != layer:
            raise ValueError(f"{path}: metadata hidden_layer_index != {layer}.")
        ids = [str(r["sample_id"]) for r in cache.rows]
        if sample_ids is None:
            sample_ids = ids
        elif ids != sample_ids:
            raise ValueError(f"{path}: rows differ from layer 0.")
        manifest_hashes.add(meta.get("manifest_sha256"))
        per_layer[layer] = [run_fold(cache, split, protocol, grid) for split in splits]
    if len(manifest_hashes) != 1:
        raise ValueError("Layer caches come from different window manifests.")

    staged.mkdir(parents=True)
    identity = {
        "schema_version": SCHEMA_VERSION, "variant": VARIANT, "protocol": protocol, "route": route,
        "l2_grid": list(grid), "layers": sorted(main_layers), "inner_folds": INNER_FOLDS,
        "inner_seed": INNER_SEED, "threshold": 0.5,
        "selection_rule": "max selection macro-F1, then min BCE, then max L2, then min layer",
        "cache_identities": {str(k): cache_identity(v) for k, v in main_layers.items()},
        "dev_cache_identities": ({str(k): cache_identity(v) for k, v in dev_layers.items()}
                                 if dev_layers else None),
        "split_files": [{"path": str(p), "sha256": canonical_sha256(read_json(p))} for p in split_files],
        "manifest_sha256": manifest_hashes.pop(),
    }
    identity["config_sha256"] = canonical_sha256(identity)
    save_json(identity, staged / "result_config.json")

    folds = []
    for fold in range(len(splits)):
        choices = []
        for layer, results in per_layer.items():
            sel = results[fold]["selection"]
            best = next(c for c in sel["candidates"] if c["l2_penalty"] == sel["selected_l2_penalty"])
            choices.append((selection_key(best, layer), layer, best))
        _, chosen_layer, chosen = max(choices, key=lambda c: c[0])
        result = per_layer[chosen_layer][fold]
        fold_dir = staged / f"fold_{fold}"
        fold_dir.mkdir()
        save_json({
            "selected_layer": chosen_layer,
            "selected_l2_penalty": result["selection"]["selected_l2_penalty"],
            "selection_source": result["selection"]["selection_source"],
            "per_layer_best": [{"layer": layer, **best} for _, layer, best in sorted(choices, key=lambda c: c[1])],
            "chosen_layer_candidates": result["selection"]["candidates"],
            "inner_folds": result["selection"]["inner_folds"],
        }, fold_dir / "selection.json")
        save_json({**result["metrics"], "fit_info": result["fit_info"], "counts": result["counts"],
                   "selected_layer": chosen_layer}, fold_dir / "metrics.json")
        write_jsonl(result["predictions"], fold_dir / "predictions_subject_level.jsonl")
        _write_csv(result["predictions"], fold_dir / "predictions_subject_level.csv")
        m = result["metrics"]
        folds.append({"fold": fold, "selected_layer": chosen_layer,
                      "selected_l2_penalty": result["selection"]["selected_l2_penalty"],
                      "selection_macro_f1": chosen["macro_f1"],
                      "macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"],
                      "uar": m["binary_strict_uar"], "auroc": m["auroc"],
                      "confusion_matrix": m["confusion_matrix"], **result["counts"]})

    diagnostic = {
        "note": "Reported-subject scores per layer (each with its own inner-selected L2). Diagnostic only; "
                "never used for selection. Not a reportable result.",
        "layers": {
            str(layer): {
                "fold_macro_f1": [r["metrics"]["macro_f1"] for r in results],
                "fold_selection_macro_f1": [
                    next(c["macro_f1"] for c in r["selection"]["candidates"]
                         if c["l2_penalty"] == r["selection"]["selected_l2_penalty"]) for r in results],
                "mean_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in results])),
                "mean_positive_f1": float(np.mean([r["metrics"]["positive_f1"] for r in results])),
                "mean_uar": float(np.mean([r["metrics"]["binary_strict_uar"] for r in results])),
            }
            for layer, results in per_layer.items()
        },
    }
    save_json(diagnostic, staged / "per_layer_test.json")
    summary = {
        "route": route, "protocol": protocol, "variant": VARIANT, "folds": folds,
        "aggregation": "unweighted_fold_mean_of_subject_level_metrics",
        "fold_mean": {k: float(np.mean([f[k] for f in folds])) for k in ("macro_f1", "positive_f1", "uar")},
        "selected_layers": [f["selected_layer"] for f in folds],
        "feature_source": "frozen_base",
    }
    save_json(summary, staged / "summary.json")
    if output_dir.exists():
        output_dir.rmdir()
    staged.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers-root", required=True, type=Path, help="Directory holding layer_XX caches.")
    parser.add_argument("--dev-layers-root", type=Path, help="DAIC only: official-dev layer_XX caches.")
    parser.add_argument("--split-files", required=True, nargs="+", type=Path)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", default="{}")
    args = parser.parse_args()
    summary = run(layers_root=args.layers_root, dev_layers_root=args.dev_layers_root,
                  split_files=args.split_files, protocol=args.protocol, output_dir=args.output_dir,
                  route=json.loads(args.route))
    print(json.dumps({"route": summary["route"], "fold_mean": summary["fold_mean"],
                      "selected_layers": summary["selected_layers"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
