#!/usr/bin/env python3
"""Run ``merged_subject_mean_heads`` over every merged route, seed and fold.

LoRA: every seed's own features. Frozen: one run per route and fold, using the
seed-1337 row index; the other seeds' rows (minus the checkpoint path) must match it, so the frozen
result does not depend on the seed whose index is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.merged_subject_mean_heads import run  # noqa: E402

ROUTES = (
    ("native", "audio_only"), ("native", "audio_text"), ("native", "text_only"),
    ("english", "audio_text"), ("english", "text_only"),
)
SEEDS = (7, 1337, 2024)
STAGES = tuple(("cv", f) for f in range(5)) + (("final", 0),)
DATASETS = ("androids_interview", "cmdc", "d3tec", "turkish")


def frozen_caches(caches: Path, daic_audio: Path, language: str, modality: str) -> list[Path]:
    paths = [caches / ds / f"{modality}_{language}" / "main" for ds in DATASETS]
    if modality == "audio_only":
        paths += [daic_audio / "hidden_features_test", daic_audio / "hidden_features_dev"]
    else:
        # DAIC is English already; both languages use the native DAIC caches.
        paths += [caches / "daic" / f"{modality}_native" / part for part in ("main", "dev")]
    return paths


def features_dir(root: Path, language: str, modality: str, seed: int, stage: str, fold: int) -> Path:
    return (root / f"qwen3_pooled_{language}" / modality / f"qmsm_{language}_{modality}_s{seed}"
            / stage / f"fold_{fold}" / "features")


def row_index(features: Path) -> list[list[dict]]:
    """Rows of both partitions without the per-seed checkpoint path."""
    return [
        [{k: v for k, v in json.loads(line).items() if k != "checkpoint"}
         for line in (features / f"{part}_rows.jsonl").read_text().splitlines() if line.strip()]
        for part in ("outer_train", "outer_holdout")
    ]


def _job(args):
    features, output, source, caches = args
    summary = run(features, output, source, caches)
    return str(output), {h: {d: m["macro_f1"] for d, m in v["metrics_by_dataset"].items()}
                         for h, v in summary["heads"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged-features-root", required=True, type=Path)
    parser.add_argument("--frozen-caches-root", required=True, type=Path)
    parser.add_argument("--frozen-daic-audio", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for language, modality in ROUTES:
        route = f"{language}_{modality}"
        for stage, fold in STAGES:
            reference = features_dir(args.merged_features_root, language, modality, 1337, stage, fold)
            ref_rows = row_index(reference)
            for seed in SEEDS:
                f = features_dir(args.merged_features_root, language, modality, seed, stage, fold)
                if seed != 1337 and row_index(f) != ref_rows:
                    raise ValueError(f"Row index differs from seed 1337: {f}")
                jobs.append((f, args.output_root / "lora" / route / f"s{seed}" / f"{stage}_fold_{fold}", "lora", None))
            jobs.append((reference, args.output_root / "frozen" / route / f"{stage}_fold_{fold}", "frozen",
                         frozen_caches(args.frozen_caches_root, args.frozen_daic_audio, language, modality)))
    print(f"{len(jobs)} head runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(_job, j) for j in jobs]):
            output, scores = future.result()
            print(output, json.dumps({h: {d: round(v, 3) for d, v in s.items()} for h, s in scores.items()}), flush=True)


if __name__ == "__main__":
    main()
