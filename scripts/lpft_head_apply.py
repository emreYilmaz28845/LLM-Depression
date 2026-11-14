#!/usr/bin/env python
"""Apply a saved LP-FT head (``best_model/lpft_head.pt``) to a hidden-state cache.

The cache is the ``run_qwen_hidden_extract_slurm.sh`` extraction of the same
``best_model`` adapter (final layer, last valid prompt token). The subject logit
is the mean chunk logit, threshold 0, scored with the frozen-head metric code so
the numbers are comparable with ``baselines/frozen_standalone_heads.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.subject_mean_logreg_head import load_partition, score_metrics  # noqa: E402
from src.features.hidden_classifier_policy import cache_identity  # noqa: E402
from src.lpft_head import load_head_state, subject_mean_logits, subject_predictions  # noqa: E402
from src.utils import read_json, save_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--partition", default="final_eval")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-adapter-sha256", default=None,
                        help="refuse a cache extracted from a different adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    meta = read_json(args.cache_dir / "extraction_metadata.json")
    if args.expect_adapter_sha256 and meta.get("adapter_sha256") != args.expect_adapter_sha256:
        raise SystemExit(f"Cache adapter {meta.get('adapter_sha256')} != expected {args.expect_adapter_sha256}.")
    x, rows = load_partition(args.cache_dir, args.partition)
    head = load_head_state(args.head)
    weight = np.asarray(head["weight"], dtype=np.float64)
    if x.shape[1] != weight.shape[0]:
        raise SystemExit(f"Feature dim {x.shape[1]} != head dim {weight.shape[0]}.")
    sample_logits = (x @ weight + float(head["bias"])).tolist()
    subjects, labels, logits = subject_mean_logits(sample_logits, rows)
    metrics = score_metrics(labels, logits)
    metrics["prediction_backend"] = "lpft_head"
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["subject_id"])] = counts.get(str(row["subject_id"]), 0) + 1
    predictions = subject_predictions(subjects, labels, logits, counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "headline_metrics": metrics,
            "head": {"path": str(args.head), "sha256": head["sha256"], "meta": head.get("meta")},
            "cache_identity": cache_identity(args.cache_dir),
            "partition": args.partition,
            "checkpoint_dir": meta.get("checkpoint_dir"),
            "adapter_sha256": meta.get("adapter_sha256"),
            "manifest_sha256": meta.get("manifest_sha256"),
            "aggregation": "subject_level mean chunk logit, threshold 0",
        },
        args.output_dir / "metrics_lpft_head.json",
    )
    write_jsonl(predictions, args.output_dir / "predictions_subject_level.jsonl")
    with open(args.output_dir / "predictions_subject_level.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "positive_f1", "binary_strict_uar", "auroc",
                                              "binary_cross_entropy", "num_subjects")}, indent=1))


if __name__ == "__main__":
    main()
