from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import read_json, save_json


def summarize(matrix_path: Path, results_root: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    rows = []
    for item in matrix["experiments"]:
        for fold in item["folds"]:
            output = (
                results_root
                / item["dataset"]
                / item["condition"]
                / Path(item["run_dir"]).name
                / f"fold_{fold}"
                / matrix["experiment_id"]
            )
            rows.extend(read_json(output / "screen_summary.json"))
    by_ratio: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["sampling_mode"] == "minority_subject_oversample":
            by_ratio[float(row["oversampling_ratio"])].append(float(row["pooled_macro_f1"]))
    ratio_means = {str(ratio): statistics.mean(values) for ratio, values in sorted(by_ratio.items())}
    if set(by_ratio) != {0.75, 1.0}:
        raise ValueError(f"Expected ratios 0.75 and 1.0, found {sorted(by_ratio)}")
    difference = abs(ratio_means["0.75"] - ratio_means["1.0"])
    selected = 0.75 if difference <= 0.005 else max(by_ratio, key=lambda ratio: ratio_means[str(ratio)])
    return {
        "schema_version": "turkish_oversampling_screen_summary.v1",
        "selection_metric": "mean pooled inner-OOF Macro-F1",
        "selection_scope": "modalities x outer caches x heads x sampling seeds",
        "expected_summary_rows": 210,
        "observed_summary_rows": len(rows),
        "ratio_macro_f1_means": ratio_means,
        "absolute_ratio_difference": difference,
        "tie_threshold": 0.005,
        "selected_ratio": selected,
        "gate_passed": len(rows) == 210,
        "outer_evaluation_metrics_inspected": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.matrix, args.results_root)
    save_json(payload, args.output)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0]))
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if not payload["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
