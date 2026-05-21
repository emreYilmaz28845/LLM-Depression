from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_logging, read_json, save_json


METRIC_KEYS = ["accuracy", "precision", "recall", "positive_f1", "macro_f1", "weighted_f1"]


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def summarize_run(run_root: Path) -> None:
    rows = []
    best_metrics_by_key = {key: [] for key in METRIC_KEYS}
    generation_metrics_by_key = {key: [] for key in METRIC_KEYS}
    for fold_dir in sorted(run_root.glob("fold_*")):
        best_likelihood = read_json(fold_dir / "eval" / "best_checkpoint" / "metrics_likelihood.json")
        best_generation = read_json(fold_dir / "eval" / "best_checkpoint" / "metrics_generation.json")
        row = {"fold": fold_dir.name}
        for key in METRIC_KEYS:
            row[f"likelihood_{key}"] = best_likelihood[key]
            row[f"generation_{key}"] = best_generation[key]
            best_metrics_by_key[key].append(float(best_likelihood[key]))
            generation_metrics_by_key[key].append(float(best_generation[key]))
        rows.append(row)

    final_summary = {
        "fold_rows": rows,
        "best_checkpoint_likelihood": {key: _summary_stats(values) for key, values in best_metrics_by_key.items()},
        "best_checkpoint_generation": {key: _summary_stats(values) for key, values in generation_metrics_by_key.items()},
    }
    save_json(final_summary, run_root / "final_summary.json")
    with (run_root / "final_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["fold"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-fold metrics across a run directory.")
    parser.add_argument("--run_root", required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    summarize_run(Path(args.run_root))


if __name__ == "__main__":
    main()
