#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_json, save_json


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "positive_f1",
    "macro_f1",
    "weighted_f1",
    "auroc",
]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)) if values else 0.0,
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def summarize(run_roots: list[Path], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run_root in run_roots:
        summary_path = run_root / "final_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing per-seed summary: {summary_path}")
        summary = read_json(summary_path)
        for fold_row in summary.get("fold_rows", []):
            if not fold_row.get("active_metrics_path"):
                continue
            row = {
                "run_root": str(run_root),
                "run_name": run_root.name,
                "fold": fold_row["fold"],
                "active_backend": fold_row.get("active_backend", ""),
                "active_aggregation_level": fold_row.get("active_aggregation_level", ""),
            }
            for key in METRIC_KEYS:
                value = fold_row.get(f"active_{key}")
                if value is not None:
                    row[key] = float(value)
            rows.append(row)

    metric_summary = {
        key: _summary([float(row[key]) for row in rows if key in row])
        for key in METRIC_KEYS
    }
    payload = {
        "run_roots": [str(path) for path in run_roots],
        "num_fold_seed_runs": len(rows),
        "metrics": metric_summary,
        "rows": rows,
    }
    save_json(payload, output_path)
    csv_path = output_path.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_name", "fold"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} fold×seed results to {output_path} and {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize metrics across CV seed runs.")
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize([Path(value) for value in args.run_roots], Path(args.output))


if __name__ == "__main__":
    main()
