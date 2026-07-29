#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_d3tec_hidden_optuna_matrix import (
    GATE_THRESHOLD,
    PILOT_CONDITIONS,
    SEED_IDS,
    STAGE1_ID,
)
from src.metrics import classification_metrics
from src.utils import read_json, read_jsonl, save_json


EXPERIMENT_SEEDS = {STAGE1_ID: 1337, **{value: key for key, value in SEED_IDS.items()}}


def summarize(
    matrix_path: Path,
    results_root: Path,
    include_audio_only: bool = False,
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    required_conditions = set(PILOT_CONDITIONS)
    if include_audio_only:
        required_conditions.add("audio_only_normalized")
    rows = []
    for item in matrix["experiments"]:
        condition = item["condition"]
        if condition not in required_conditions:
            continue
        run_name = Path(item["run_dir"]).name
        condition_results = []
        for experiment_id, inner_seed in EXPERIMENT_SEEDS.items():
            subject_rows = []
            seen = set()
            for fold in range(5):
                result_dir = (
                    results_root
                    / "d3tec"
                    / condition
                    / run_name
                    / f"fold_{fold}"
                    / experiment_id
                )
                metadata = read_json(result_dir / "classifier_metadata.json")
                if (
                    metadata["experiment_id"] != experiment_id
                    or int(metadata["inner_seed"]) != inner_seed
                    or int(metadata["completed_trials"]) != 150
                ):
                    raise ValueError(f"Incompatible stability result: {result_dir}.")
                fold_rows = read_jsonl(result_dir / "predictions_subject_level.jsonl")
                ids = {str(row["subject_id"]) for row in fold_rows}
                if seen & ids:
                    raise ValueError(f"Repeated held-out subjects: {condition}/{experiment_id}.")
                seen.update(ids)
                subject_rows.extend(fold_rows)
            if len(seen) != 62:
                raise ValueError(f"Expected 62 pooled subjects: {condition}/{experiment_id}.")
            metrics = classification_metrics(
                [int(row["label"]) for row in subject_rows],
                [int(row["prediction"]) for row in subject_rows],
            )
            condition_results.append(
                {
                    "experiment_id": experiment_id,
                    "inner_seed": inner_seed,
                    "pooled_macro_f1": float(metrics["macro_f1"]),
                    "pooled_positive_f1": float(metrics["positive_f1"]),
                }
            )
        values = [row["pooled_macro_f1"] for row in condition_results]
        rows.append(
            {
                "condition": condition,
                "seed_results": sorted(condition_results, key=lambda row: row["inner_seed"]),
                "pooled_macro_f1_min": min(values),
                "pooled_macro_f1_max": max(values),
                "pooled_macro_f1_range": max(values) - min(values),
            }
        )
    if {row["condition"] for row in rows} != required_conditions:
        raise ValueError(
            "Stability results do not contain all required conditions: "
            f"{sorted(required_conditions)}."
        )
    pilot_rows = [row for row in rows if row["condition"] in PILOT_CONDITIONS]
    observed = max(row["pooled_macro_f1_range"] for row in pilot_rows)
    return {
        "schema_version": "d3tec_hidden_stability.v1",
        "source_experiment_ids": list(EXPERIMENT_SEEDS),
        "gate_threshold": GATE_THRESHOLD,
        "observed_pilot_max_range": observed,
        "expand_audio_only": observed >= GATE_THRESHOLD,
        "selection_prohibition": (
            "Inner seeds are a stability analysis only; workbook values use seed 1337."
        ),
        "expansion_audited": include_audio_only,
        "stability_rows": sorted(rows, key=lambda row: row["condition"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the D3TEC inner-seed stability gate.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/features/d3tec_hidden_matrix.yaml"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/hidden_classifiers"),
    )
    parser.add_argument(
        "--include-audio-only",
        action="store_true",
        help="Also require and audit all three inner seeds for audio-only.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.matrix, args.results_root, args.include_audio_only)
    save_json(payload, args.output)
    print(f"expand_audio_only={payload['expand_audio_only']}")


if __name__ == "__main__":
    main()
