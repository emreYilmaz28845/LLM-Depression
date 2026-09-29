#!/usr/bin/env python3
"""Expand the Qwen3 DAIC label-vocabulary submission matrix.

The campaign is frozen as fifteen smoke chains (five arms in three
model/modality cells, training seed 1337, one epoch) and forty-five production
chains (the same cells at training seeds 7, 1337 and 2024). This script is the
single source of the run names, group identities and submission parameters that
the managed submit calls consume, so a resumed agent rebuilds the same matrix
instead of inventing names.

Run names carry the UTC campaign id, the model, the modality, the arm tag, the
training seed and the fold. The training seed is always a submission override;
``split.seed`` is never overridden and stays 1337 in every chain, which keeps the
manifest, quarantine list and split identical across seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_qwen3_daic_label_configs import (  # noqa: E402
    ARM_ORDER,
    CAMPAIGN,
    DATASET,
    FOLDS,
    SOURCES,
    generated_name,
)

SCHEMA_VERSION = "audiollm.qwen3_daic_label_vocab.matrix.v1"
SMOKE_CAMPAIGN = "qwen3_daic_label_vocab_smoke_v1"
GROUP_ID = "qwen3-daic-label-vocab-20260928"
# Smoke and production share the lane's linked experiment group: the managed
# submit path refuses any group id other than the linked one, and the smoke
# separation is carried by the campaign and the run-name prefix instead.
SMOKE_GROUP_ID = GROUP_ID
CAMPAIGN_UTC_ID = "20260928T142540Z"
PRODUCTION_SEEDS = (7, 1337, 2024)
SMOKE_SEED = 1337
SMOKE_EPOCHS = 1

QWEN38_ENV = "/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate"
QWEN3OMNI_ENV = "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate"


def env_activate(modality: str) -> str:
    return QWEN38_ENV if SOURCES[modality][2] == "qwen38" else QWEN3OMNI_ENV


def run_name(*, smoke: bool, modality: str, tag: str, seed: int) -> str:
    prefix = "q3dlvsmoke" if smoke else "q3dlv"
    model = SOURCES[modality][1]
    return f"{prefix}_{CAMPAIGN_UTC_ID}_{model}_{modality}_{tag}_s{seed}_f{FOLDS}"


def _entry(*, smoke: bool, modality: str, tag: str, seed: int) -> dict[str, Any]:
    config = f"configs/labels/{generated_name(modality, tag)}"
    overrides = [f"--set=training.num_train_epochs={SMOKE_EPOCHS}"] if smoke else []
    return {
        "stage": "smoke" if smoke else "production",
        "logical_run": f"daic_{modality}_{SOURCES[modality][1]}_{tag}",
        "run_name": run_name(smoke=smoke, modality=modality, tag=tag, seed=seed),
        "config": config,
        "dataset": DATASET,
        "modality": modality,
        "model": SOURCES[modality][1],
        "arm": tag,
        "seed": seed,
        "fold": FOLDS,
        "campaign": SMOKE_CAMPAIGN if smoke else CAMPAIGN,
        "group_id": SMOKE_GROUP_ID if smoke else GROUP_ID,
        "env_activate": env_activate(modality),
        "evaluation_gpus_per_node": 1 if SOURCES[modality][2] == "qwen38" else 4,
        "extra_overrides": overrides,
    }


def build_matrix() -> dict[str, Any]:
    smoke = [
        _entry(smoke=True, modality=modality, tag=tag, seed=SMOKE_SEED)
        for modality in SOURCES
        for tag in ARM_ORDER
    ]
    production = [
        _entry(smoke=False, modality=modality, tag=tag, seed=seed)
        for modality in SOURCES
        for tag in ARM_ORDER
        for seed in PRODUCTION_SEEDS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "smoke_campaign": SMOKE_CAMPAIGN,
        "group_id": GROUP_ID,
        "smoke_group_id": SMOKE_GROUP_ID,
        "campaign_utc_id": CAMPAIGN_UTC_ID,
        "dataset": DATASET,
        "fold": FOLDS,
        "arms": list(ARM_ORDER),
        "production_seeds": list(PRODUCTION_SEEDS),
        "smoke_seed": SMOKE_SEED,
        "smoke_epochs": SMOKE_EPOCHS,
        "smoke": smoke,
        "production": production,
    }


def render(matrix: dict[str, Any]) -> str:
    return json.dumps(matrix, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="write the matrix JSON to this path")
    parser.add_argument("--check", default=None, help="verify this matrix JSON matches the expansion")
    parser.add_argument("--stage", choices=("smoke", "production", "all"), default="all")
    args = parser.parse_args()

    matrix = build_matrix()
    if args.stage != "all":
        payload: dict[str, Any] = {**matrix, args.stage: matrix[args.stage]}
        payload = {key: value for key, value in payload.items() if key not in ("smoke", "production")}
        payload[args.stage] = matrix[args.stage]
        matrix = payload
    if args.check:
        expected = render(matrix)
        current = Path(args.check).read_text(encoding="utf-8") if Path(args.check).is_file() else ""
        if current != expected:
            print(f"matrix drift: {args.check}")
            return 1
        print(f"matrix is current: {args.check}")
        return 0
    if args.out:
        Path(args.out).write_text(render(matrix), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    print(render(matrix), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
