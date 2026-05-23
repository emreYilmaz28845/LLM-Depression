from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_logging, read_json, save_json


METRIC_KEYS = ["accuracy", "precision", "recall", "positive_f1", "macro_f1", "weighted_f1"]
BACKEND_METRIC_FILES = {
    "likelihood": "metrics_likelihood.json",
    "generation": "metrics_generation.json",
    "original_teacher_forced": "metrics_original_teacher_forced.json",
}


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
    metrics_by_backend = {backend: {key: [] for key in METRIC_KEYS} for backend in BACKEND_METRIC_FILES}
    for fold_dir in sorted(run_root.glob("fold_*")):
        run_config_path = fold_dir / "run_config.yaml"
        active_backend = "likelihood"
        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as handle:
                run_config = yaml.safe_load(handle)
            active_backend = (
                run_config.get("evaluation", {}).get("sample_prediction_mode")
                or run_config.get("config", {}).get("evaluation", {}).get("sample_prediction_mode")
                or active_backend
            )
        row = {"fold": fold_dir.name, "active_backend": active_backend}
        for backend_name, filename in BACKEND_METRIC_FILES.items():
            metrics_path = fold_dir / "eval" / "best_checkpoint" / filename
            if not metrics_path.exists():
                continue
            backend_metrics = read_json(metrics_path)
            row[f"{backend_name}_prediction_backend"] = backend_metrics.get("prediction_backend", backend_name)
            row[f"{backend_name}_evaluation_protocol_name"] = backend_metrics.get("evaluation_protocol_name", "")
            for key in METRIC_KEYS:
                row[f"{backend_name}_{key}"] = backend_metrics[key]
                metrics_by_backend[backend_name][key].append(float(backend_metrics[key]))
        rows.append(row)

    final_summary = {
        "fold_rows": rows,
        "best_checkpoint_metrics_by_backend": {
            backend: {key: _summary_stats(values) for key, values in metric_lists.items() if values}
            for backend, metric_lists in metrics_by_backend.items()
        },
    }
    save_json(final_summary, run_root / "final_summary.json")
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["fold", "active_backend"]
    with (run_root / "final_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
