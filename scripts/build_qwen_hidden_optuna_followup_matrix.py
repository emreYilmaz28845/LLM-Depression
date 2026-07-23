#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


STAGE1_ID = "xgb_optuna_raw_t150_d6_seed1337_inner1337"
DEPTH8_ID = "xgb_optuna_raw_t150_d8_seed1337_inner1337"
SEED_IDS = {
    7: "xgb_optuna_raw_t150_d6_seed1337_inner7",
    2024: "xgb_optuna_raw_t150_d6_seed1337_inner2024",
}
PILOT_CONDITIONS = {
    ("daic", "text_only"),
    ("cmdc", "audio_text"),
    ("turkish", "text_only"),
}


def _base_jobs(matrix_path: Path) -> list[dict[str, Any]]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for item in matrix["experiments"]:
        condition = str(item.get("condition", item["modality"]))
        for fold in item["folds"]:
            jobs.append(
                {
                    "dataset": str(item["dataset"]),
                    "modality": str(item["modality"]),
                    "condition": condition,
                    "fold": int(fold),
                    "run_dir": str(item["run_dir"]),
                    "objective": str(item["objective"]),
                }
            )
    expected = int(matrix.get("expected_jobs", 33))
    if len(jobs) != expected:
        raise ValueError(f"Base matrix declares {expected} jobs but expands to {len(jobs)}.")
    return jobs


def _configured(
    job: dict[str, Any],
    *,
    experiment_id: str,
    search_profile: str,
    inner_seed: int,
) -> dict[str, Any]:
    return {
        **job,
        "experiment_id": experiment_id,
        "target_trials": 150,
        "inner_folds": 3,
        "search_profile": search_profile,
        "seed": 1337,
        "inner_seed": inner_seed,
        "xgb_threads": 20,
    }


def _stage1(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return (
        [
            _configured(
                job,
                experiment_id=STAGE1_ID,
                search_profile="standard_d6",
                inner_seed=1337,
            )
            for job in jobs
        ],
        {},
    )


def _pilot(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        job for job in jobs if (job["dataset"], job["condition"]) in PILOT_CONDITIONS
    ]
    if len(selected) != 11:
        raise ValueError(f"Expected 11 representative outer evaluations, found {len(selected)}.")
    output = [
        _configured(
            job,
            experiment_id=SEED_IDS[inner_seed],
            search_profile="standard_d6",
            inner_seed=inner_seed,
        )
        for inner_seed in (7, 2024)
        for job in selected
    ]
    return output, {"reused_inner1337_experiment_id": STAGE1_ID}


def _result_dir(results_root: Path, job: dict[str, Any], experiment_id: str) -> Path:
    return (
        results_root
        / job["dataset"]
        / job["condition"]
        / Path(job["run_dir"]).name
        / f"fold_{job['fold']}"
        / experiment_id
    )


def _depth8(
    jobs: list[dict[str, Any]],
    results_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    boundary_folds: list[dict[str, Any]] = []
    for job in jobs:
        result_dir = _result_dir(results_root, job, STAGE1_ID)
        best_path = result_dir / "best_params.json"
        metadata_path = result_dir / "classifier_metadata.json"
        if not best_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete stage-1 result: {result_dir}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(best.get("completed_trial_count", -1)) != 150:
            raise ValueError(f"Stage-1 result does not contain 150 completed trials: {best_path}")
        if metadata.get("experiment_id") != STAGE1_ID:
            raise ValueError(f"Stage-1 experiment identity mismatch: {metadata_path}")
        if int(best["suggested_params"]["max_depth"]) == 6:
            boundary_folds.append(
                {
                    "dataset": job["dataset"],
                    "condition": job["condition"],
                    "fold": job["fold"],
                    "best_trial_number": int(best["best_trial_number"]),
                }
            )
    eligible = {(row["dataset"], row["condition"]) for row in boundary_folds}
    selected = [
        job for job in jobs if (job["dataset"], job["condition"]) in eligible
    ]
    output = [
        _configured(
            job,
            experiment_id=DEPTH8_ID,
            search_profile="depth8",
            inner_seed=1337,
        )
        for job in selected
    ]
    return output, {
        "source_experiment_id": STAGE1_ID,
        "selection_rule": "all folds when any condition fold best max_depth equals 6",
        "boundary_folds": boundary_folds,
        "eligible_conditions": [
            {"dataset": dataset, "condition": condition}
            for dataset, condition in sorted(eligible)
        ],
    }


def _expansion(
    jobs: list[dict[str, Any]],
    stability_summary: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(stability_summary.read_text(encoding="utf-8"))
    if payload.get("source_experiment_ids") != [STAGE1_ID, SEED_IDS[7], SEED_IDS[2024]]:
        raise ValueError("Stability summary was not generated from the required three experiments.")
    if not bool(payload.get("expand_all")):
        raise ValueError("Stability gate did not trigger; refusing to generate full expansion.")
    remaining = [
        job for job in jobs if (job["dataset"], job["condition"]) not in PILOT_CONDITIONS
    ]
    if len(remaining) != 22:
        raise ValueError(f"Expected 22 remaining outer evaluations, found {len(remaining)}.")
    output = [
        _configured(
            job,
            experiment_id=SEED_IDS[inner_seed],
            search_profile="standard_d6",
            inner_seed=inner_seed,
        )
        for inner_seed in (7, 2024)
        for job in remaining
    ]
    return output, {
        "stability_summary": str(stability_summary),
        "gate_threshold": float(payload["gate_threshold"]),
        "observed_max_primary_range": float(payload["observed_max_primary_range"]),
    }


def build_manifest(
    *,
    stage: str,
    base_matrix: Path,
    results_root: Path,
    stability_summary: Path | None,
) -> dict[str, Any]:
    jobs = _base_jobs(base_matrix)
    if stage == "stage1":
        selected, provenance = _stage1(jobs)
    elif stage == "depth8":
        selected, provenance = _depth8(jobs, results_root)
    elif stage == "pilot":
        selected, provenance = _pilot(jobs)
    elif stage == "expansion":
        if stability_summary is None:
            raise ValueError("--stability-summary is required for the expansion stage.")
        selected, provenance = _expansion(jobs, stability_summary)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    identities = {
        (
            row["dataset"],
            row["condition"],
            row["fold"],
            row["experiment_id"],
        )
        for row in selected
    }
    if len(identities) != len(selected):
        raise ValueError(f"Stage {stage} contains duplicate result identities.")
    return {
        "schema_version": "qwen_hidden_optuna_followup_matrix.v1",
        "stage": stage,
        "base_matrix": str(base_matrix),
        "expected_jobs": len(selected),
        "provenance": provenance,
        "jobs": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build staged raw-XGBoost Optuna manifests.")
    parser.add_argument("--stage", required=True, choices=("stage1", "depth8", "pilot", "expansion"))
    parser.add_argument(
        "--base-matrix",
        type=Path,
        default=Path("configs/features/optuna_raw_matrix.yaml"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/hidden_classifiers"),
    )
    parser.add_argument("--stability-summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_manifest(
        stage=args.stage,
        base_matrix=args.base_matrix,
        results_root=args.results_root,
        stability_summary=args.stability_summary,
    )
    text = yaml.safe_dump(payload, sort_keys=False)
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(payload['jobs'])} jobs to {args.output}")


if __name__ == "__main__":
    main()
