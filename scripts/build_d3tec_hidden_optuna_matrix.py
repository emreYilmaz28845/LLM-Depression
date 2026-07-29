#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


STAGE1_ID = "xgb_optuna_raw_t150_d6_seed1337_inner1337"
SEED_IDS = {
    7: "xgb_optuna_raw_t150_d6_seed1337_inner7",
    2024: "xgb_optuna_raw_t150_d6_seed1337_inner2024",
}
PILOT_CONDITIONS = {"audio_text_normalized", "text_only"}
GATE_THRESHOLD = 0.03


def _base_jobs(path: Path) -> list[dict[str, Any]]:
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = [
        {
            **item,
            "fold": int(fold),
        }
        for item in matrix["experiments"]
        for fold in item["folds"]
    ]
    if len(rows) != int(matrix["expected_jobs"]) or len(rows) != 15:
        raise ValueError(f"D3TEC base matrix must expand to 15 jobs, found {len(rows)}.")
    return rows


def _configured(
    row: dict[str, Any],
    *,
    inner_seed: int,
) -> dict[str, Any]:
    return {
        **{key: value for key, value in row.items() if key != "folds"},
        "experiment_id": (
            STAGE1_ID if inner_seed == 1337 else SEED_IDS[inner_seed]
        ),
        "target_trials": 150,
        "inner_folds": 3,
        "search_profile": "standard_d6",
        "seed": 1337,
        "inner_seed": inner_seed,
        "xgb_threads": 20,
    }


def build(
    stage: str,
    base_matrix: Path,
    stability_summary: Path | None = None,
) -> dict[str, Any]:
    base = _base_jobs(base_matrix)
    if stage == "stage1":
        jobs = [_configured(row, inner_seed=1337) for row in base]
        provenance: dict[str, Any] = {}
    elif stage == "pilot":
        selected = [row for row in base if row["condition"] in PILOT_CONDITIONS]
        if len(selected) != 10:
            raise ValueError(f"Expected 10 pilot folds, found {len(selected)}.")
        jobs = [
            _configured(row, inner_seed=inner_seed)
            for inner_seed in (7, 2024)
            for row in selected
        ]
        provenance = {"reused_inner1337_experiment_id": STAGE1_ID}
    elif stage == "expansion":
        if stability_summary is None:
            raise ValueError("--stability-summary is required for expansion.")
        summary = json.loads(stability_summary.read_text(encoding="utf-8"))
        if float(summary.get("gate_threshold", -1)) != GATE_THRESHOLD:
            raise ValueError("Stability summary uses the wrong gate threshold.")
        if not bool(summary.get("expand_audio_only")):
            raise ValueError("D3TEC stability gate did not trigger.")
        selected = [row for row in base if row["condition"] == "audio_only_normalized"]
        jobs = [
            _configured(row, inner_seed=inner_seed)
            for inner_seed in (7, 2024)
            for row in selected
        ]
        provenance = {
            "stability_summary": str(stability_summary),
            "observed_pilot_max_range": float(summary["observed_pilot_max_range"]),
        }
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    identities = {
        (row["condition"], row["fold"], row["experiment_id"]) for row in jobs
    }
    if len(identities) != len(jobs):
        raise ValueError(f"D3TEC {stage} matrix contains duplicate identities.")
    return {
        "schema_version": "d3tec_hidden_optuna_stage.v1",
        "stage": stage,
        "base_matrix": str(base_matrix),
        "expected_jobs": len(jobs),
        "provenance": provenance,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build staged D3TEC hidden-head Optuna jobs.")
    parser.add_argument("--stage", required=True, choices=("stage1", "pilot", "expansion"))
    parser.add_argument(
        "--base-matrix",
        type=Path,
        default=Path("configs/features/d3tec_hidden_optuna.yaml"),
    )
    parser.add_argument("--stability-summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build(args.stage, args.base_matrix, args.stability_summary)
    text = yaml.safe_dump(payload, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
