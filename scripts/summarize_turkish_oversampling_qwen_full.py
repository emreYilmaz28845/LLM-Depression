from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def _run_root(project_root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(
        str(config["output_dirs"]["run_root"]).replace(
            "${PROJECT_ROOT}", str(project_root.resolve())
        )
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = [int(row["label"]) for row in rows]
    p = [int(row["prediction"]) for row in rows]
    matrix = confusion_matrix(y, p, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "negative_f1": float(f1_score(y, p, pos_label=0, zero_division=0)),
        "negative_recall": float(recall_score(y, p, pos_label=0, zero_division=0)),
        "positive_f1": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "auroc": None,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    assert matrix["stage"] == "full"
    grouped = defaultdict(list)
    fold_metrics = defaultdict(list)
    for job in matrix["jobs"]:
        fold_root = (
            _run_root(args.project_root, args.project_root / job["config"])
            / job["run_name"]
            / f"fold_{job['fold']}"
        )
        with (
            fold_root / "eval/best_validation/predictions_subject_level.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        key = (job["modality"], job["profile"])
        grouped[key].extend(rows)
        fold_metrics[key].append(_metrics(rows)["macro_f1"])
    summary_rows = []
    for (modality, profile), rows in sorted(grouped.items()):
        metrics = _metrics(rows)
        values = fold_metrics[(modality, profile)]
        summary_rows.append(
            {
                "dataset": "Turkish (BDI>=17)",
                "modality": modality,
                "profile": profile,
                "sampling_ratio": (
                    matrix["selected_ratio"] if profile == "oversampled" else None
                ),
                **metrics,
                "fold_macro_f1_mean": statistics.mean(values),
                "fold_macro_f1_sd": statistics.stdev(values),
                "evaluation_warning": (
                    "Table-aligned outer validation: the Turkish checkpoints and "
                    "experiment development use these validation folds."
                ),
            }
        )
    controls = {row["modality"]: row for row in summary_rows if row["profile"] == "weighted"}
    for row in summary_rows:
        row["macro_f1_delta_vs_weighted"] = (
            0.0
            if row["profile"] == "weighted"
            else row["macro_f1"] - controls[row["modality"]]["macro_f1"]
        )
    payload = {
        "schema_version": "turkish_oversampling_qwen_full_summary.v1",
        "expected_runs": 30,
        "observed_runs": len(matrix["jobs"]),
        "selected_ratio": matrix["selected_ratio"],
        "auroc_note": "Unavailable for original_teacher_forced hard-label subject outputs.",
        "rows": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
