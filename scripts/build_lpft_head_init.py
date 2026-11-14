#!/usr/bin/env python
"""Refit the frozen subject-mean L2 head and export it as an LP-FT warm start.

Uses the same code path as ``baselines/frozen_standalone_heads.py`` (cache
loading, train-only scaler, ``fit_logistic``, L2 chosen on the selection split),
then folds the scaler into the weights (``src/lpft_head.py``) and checks that the
folded head gives the same subject logits on every split. Refuses to write when
the refit does not reproduce the recorded reference metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.frozen_standalone_heads import (  # noqa: E402
    DEFAULT_L2_GRID,
    _features,
    _fit,
    _load_caches,
    _logits,
    _prepare,
    run_fold,
)
from baselines.subject_mean_logreg_head import score_metrics  # noqa: E402
from src.features.hidden_classifier_policy import cache_identity, canonical_sha256  # noqa: E402
from src.lpft_head import HEAD_INIT_SCHEMA_VERSION, file_sha256, fold_standardization  # noqa: E402
from src.utils import read_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="frozen base cache, test view")
    parser.add_argument("--dev-cache", type=Path, required=True, help="frozen base cache, dev view")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="head_init.npz path")
    parser.add_argument("--expect-l2", type=float, required=True)
    parser.add_argument("--expect-macro-f1", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    cache, meta = _load_caches(args.cache_dir, args.dev_cache, allow_adapter=False)
    split = read_json(args.split_file)
    reference = run_fold(cache, split, "daic_official", DEFAULT_L2_GRID)
    selected_l2 = float(reference["selection"]["selected_l2_penalty"])
    if abs(selected_l2 - args.expect_l2) > 0:
        raise SystemExit(f"L2 on official dev is {selected_l2}, expected {args.expect_l2}.")
    if abs(reference["metrics"]["macro_f1"] - args.expect_macro_f1) > args.tolerance:
        raise SystemExit(f"Refit test macro-F1 {reference['metrics']['macro_f1']} != {args.expect_macro_f1}.")

    train = sorted(str(s) for s in split["train_subject_ids"])
    head = _fit(cache, train, selected_l2, None, _prepare(cache, train, None))
    prep, _, weight, bias, info = head
    folded_weight, folded_bias = fold_standardization(prep["mean"], prep["scale"], weight, bias)

    checks = {}
    for name, key in (("train", "train_subject_ids"), ("dev", "selection_subject_ids"),
                      ("test", "final_eval_subject_ids")):
        subjects = sorted(str(s) for s in split[key])
        standard = _logits(cache, head, subjects)
        raw_means = cache.subject_means(cache.x, subjects)
        folded = raw_means @ folded_weight + folded_bias
        gap = float(np.max(np.abs(standard - folded)))
        if gap > 1e-6:
            raise SystemExit(f"Folded head differs from the standardised head on {name}: {gap}.")
        metrics = score_metrics(cache.y(subjects), folded)
        checks[name] = {"subjects": len(subjects), "max_abs_logit_gap": gap,
                        "macro_f1": metrics["macro_f1"], "positive_f1": metrics["positive_f1"],
                        "uar": metrics["binary_strict_uar"], "auroc": metrics["auroc"],
                        "binary_cross_entropy": metrics["binary_cross_entropy"]}
    if abs(checks["test"]["macro_f1"] - reference["metrics"]["macro_f1"]) > args.tolerance:
        raise SystemExit("Folded head test macro-F1 differs from the reference refit.")
    # The standardised design matrix the head was fit on, for the record.
    _ = _features(cache, prep, train, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(HEAD_INIT_SCHEMA_VERSION),
        weight=folded_weight,
        bias=np.asarray(folded_bias),
        mean=prep["mean"],
        scale=prep["scale"],
        standardized_weight=np.asarray(weight, dtype=np.float64),
        standardized_bias=np.asarray(float(bias)),
        l2_penalty=np.asarray(selected_l2),
    )
    identity = {
        "schema_version": HEAD_INIT_SCHEMA_VERSION,
        "head_init_path": str(args.output),
        "head_init_sha256": file_sha256(args.output),
        "source": "baselines/frozen_standalone_heads.py run_fold(protocol=daic_official) refit",
        "selected_l2_penalty": selected_l2,
        "fit_info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))},
        "cache_identity": cache_identity(args.cache_dir),
        "dev_cache_identity": cache_identity(args.dev_cache),
        "split_file": {"path": str(args.split_file), "sha256": canonical_sha256(split)},
        "base_model": meta.get("base_model"),
        "weights_source": meta.get("weights_source"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "pooling": meta.get("pooling"),
        "hidden_layer": meta.get("hidden_layer"),
        "feature_dim": int(folded_weight.shape[0]),
        "checks": checks,
    }
    save_json(identity, args.output.with_suffix(".json"))
    print(json.dumps({"selected_l2": selected_l2, "checks": checks, "sha256": identity["head_init_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
