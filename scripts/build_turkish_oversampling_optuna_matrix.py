from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import read_json


def build(screen_matrix: Path, screen_summary: Path) -> dict:
    matrix = yaml.safe_load(screen_matrix.read_text(encoding="utf-8"))
    summary = read_json(screen_summary)
    if not summary.get("gate_passed") or summary.get("outer_evaluation_metrics_inspected"):
        raise ValueError("Stage-2 evidence gate did not pass cleanly.")
    ratio = float(summary["selected_ratio"])
    ratio_token = f"ros{int(round(ratio * 100)):03d}"
    base_jobs = [
        {**item, "fold": fold}
        for item in matrix["experiments"]
        for fold in item["folds"]
    ]
    jobs = []
    for item in base_jobs:
        common = {
            **item,
            "objective": "macro_f1",
            "target_trials": 100,
            "inner_folds": 3,
            "seed": 1337,
            "inner_seed": 1337,
            "xgb_threads": 20,
            "search_profile": "standard_d6",
        }
        jobs.append(
            {
                **common,
                "experiment_id": "xgb_optuna_raw_oscontrol_t100_d6_seed1337",
                "sampling_mode": "none",
                "oversampling_ratio": None,
                "oversampling_seed": 1337,
            }
        )
        for oversampling_seed in (7, 1337, 2024):
            jobs.append(
                {
                    **common,
                    "experiment_id": (
                        f"xgb_optuna_raw_{ratio_token}_t100_d6_seed1337_os{oversampling_seed}"
                    ),
                    "sampling_mode": "minority_subject_oversample",
                    "oversampling_ratio": ratio,
                    "oversampling_seed": oversampling_seed,
                }
            )
    return {
        "schema_version": "turkish_oversampling_optuna_matrix.v1",
        "stage2_summary": str(screen_summary),
        "selected_ratio": ratio,
        "expected_jobs": 60,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-matrix", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.screen_matrix, args.screen_summary)
    if len(payload["jobs"]) != payload["expected_jobs"]:
        raise ValueError("Stage-3 matrix job count mismatch.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(payload['jobs'])} jobs to {args.output}")


if __name__ == "__main__":
    main()
