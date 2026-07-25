from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import yaml


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "subject_id": row["subject_id"],
            "label": int(row["label"]),
            "prediction": int(row["prediction"]),
        }
        for row in rows
    ]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    y = [row["label"] for row in rows]
    p = [row["prediction"] for row in rows]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "negative_recall": tn / (tn + fp) if tn + fp else 0.0,
        "positive_recall": recall_score(y, p, pos_label=1, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    by_profile = {}
    fold_macro = {}
    for job in matrix["jobs"]:
        config = yaml.safe_load(
            (args.project_root / job["config"]).read_text(encoding="utf-8")
        )
        run_root = Path(
            str(config["output_dirs"]["run_root"]).replace(
                "${PROJECT_ROOT}", str(args.project_root.resolve())
            )
        )
        fold_root = run_root / job["run_name"] / f"fold_{job['fold']}"
        rows = _read_predictions(
            fold_root / "eval/best_validation/predictions_subject_level.csv"
        )
        by_profile.setdefault(job["profile"], []).extend(rows)
        metrics = json.loads(
            (
                fold_root
                / "eval/best_validation/metrics_original_teacher_forced.json"
            ).read_text(encoding="utf-8")
        )
        fold_macro[(job["profile"], int(job["fold"]))] = float(metrics["macro_f1"])
    control = _metrics(by_profile["weighted"])
    sampled = _metrics(by_profile["oversampled"])
    fold_gains = [
        fold_macro[("oversampled", fold)] - fold_macro[("weighted", fold)]
        for fold in (0, 1)
    ]
    payload = {
        "schema_version": "turkish_oversampling_qwen_pilot_summary.v1",
        "mean_selected_validation_macro_f1_gain": statistics.mean(fold_gains),
        "minimum_fold_selected_validation_macro_f1_gain": min(fold_gains),
        "pooled_control": control,
        "pooled_oversampled": sampled,
        "pooled_macro_f1_not_below_control": sampled["macro_f1"] >= control["macro_f1"],
        "pooled_negative_recall_not_below_control": (
            sampled["negative_recall"] >= control["negative_recall"]
        ),
    }
    payload["proceed_to_full"] = (
        payload["mean_selected_validation_macro_f1_gain"] >= 0.015
        and payload["minimum_fold_selected_validation_macro_f1_gain"] >= -0.03
        and payload["pooled_macro_f1_not_below_control"]
        and payload["pooled_negative_recall_not_below_control"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
