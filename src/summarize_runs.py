from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_utils import CV_PROTOCOL_TRAIN_VAL, CV_PROTOCOL_TRAIN_VAL_TEST
from src.utils import configure_logging, read_json, save_json


BACKEND_METRIC_FILES = {
    "likelihood": "metrics_likelihood.json",
    "generation": "metrics_generation.json",
    "original_teacher_forced": "metrics_original_teacher_forced.json",
}


def _candidate_metric_paths(fold_dir: Path, filename: str, cv_protocol: str) -> list[Path]:
    if cv_protocol == CV_PROTOCOL_TRAIN_VAL:
        return [fold_dir / "eval" / "best_validation" / filename]
    return [
        fold_dir / "eval" / "best_checkpoint" / filename,
        fold_dir / "best_model" / "standalone_eval" / filename,
    ]


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def _add_confusion_matrices(left: list[list[int]] | None, right: list[list[int]] | None) -> list[list[int]] | None:
    if right is None:
        return left
    if left is None:
        return [[int(value) for value in row] for row in right]
    return [
        [int(left[row_idx][col_idx]) + int(right[row_idx][col_idx]) for col_idx in range(len(right[row_idx]))]
        for row_idx in range(len(right))
    ]


def _metrics_from_confusion_matrix(confusion_matrix: list[list[int]] | None) -> dict[str, float]:
    if not confusion_matrix or len(confusion_matrix) < 2 or any(len(row) < 2 for row in confusion_matrix[:2]):
        return {}
    tn, fp = int(confusion_matrix[0][0]), int(confusion_matrix[0][1])
    fn, tp = int(confusion_matrix[1][0]), int(confusion_matrix[1][1])
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall)
        else 0.0
    )
    macro_f1 = (negative_f1 + positive_f1) / 2
    weighted_f1 = ((tn + fp) * negative_f1 + (fn + tp) * positive_f1) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "positive_f1": positive_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def summarize_run(run_root: Path) -> None:
    rows = []
    cv_protocols: set[str] = set()
    expected_fold_counts: set[int] = set()
    metrics_by_backend = {
        backend: defaultdict(list) for backend in BACKEND_METRIC_FILES
    }
    active_metrics_by_key: dict[str, list[float]] = defaultdict(list)
    active_confusion_matrix: list[list[int]] | None = None
    active_support_negative = 0
    active_support_positive = 0
    for fold_dir in sorted(run_root.glob("fold_*")):
        run_config_path = fold_dir / "run_config.yaml"
        active_backend = "likelihood"
        active_aggregation_level = "subject"
        cv_protocol = CV_PROTOCOL_TRAIN_VAL_TEST
        selection_metadata: dict = {}
        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as handle:
                run_config = yaml.safe_load(handle)
            cv_protocol = (
                run_config.get("cv_protocol")
                or run_config.get("config", {}).get("split", {}).get("cv_protocol")
                or CV_PROTOCOL_TRAIN_VAL_TEST
            )
            configured_fold_count = run_config.get("config", {}).get("split", {}).get("outer_folds")
            if configured_fold_count is not None:
                expected_fold_counts.add(int(configured_fold_count))
            active_backend = (
                run_config.get("evaluation", {}).get("sample_prediction_mode")
                or run_config.get("config", {}).get("evaluation", {}).get("sample_prediction_mode")
                or active_backend
            )
            active_aggregation_level = (
                run_config.get("evaluation", {}).get("aggregation_level")
                or run_config.get("config", {}).get("evaluation", {}).get("aggregation_level")
                or active_aggregation_level
            )
        elif (fold_dir / "eval" / "best_validation").exists():
            cv_protocol = CV_PROTOCOL_TRAIN_VAL
        if cv_protocol not in {CV_PROTOCOL_TRAIN_VAL, CV_PROTOCOL_TRAIN_VAL_TEST}:
            raise ValueError(
                f"Unsupported split.cv_protocol={cv_protocol!r} in {run_config_path}"
            )
        cv_protocols.add(cv_protocol)
        if len(cv_protocols) > 1:
            raise ValueError(
                f"Mixed split.cv_protocol values under {run_root}: {sorted(cv_protocols)}"
            )
        if cv_protocol == CV_PROTOCOL_TRAIN_VAL:
            selection_path = fold_dir / "eval" / "best_validation" / "selection.json"
            if not selection_path.exists():
                raise FileNotFoundError(
                    f"Missing selected validation metadata for {fold_dir.name}: {selection_path}"
                )
            selection_metadata = read_json(selection_path)
            if selection_metadata.get("cv_protocol") not in {None, CV_PROTOCOL_TRAIN_VAL}:
                raise ValueError(
                    f"Invalid selected validation protocol in {selection_path}: "
                    f"{selection_metadata.get('cv_protocol')!r}"
                )
        row = {
            "fold": fold_dir.name,
            "cv_protocol": cv_protocol,
            "score_source": (
                "best_outer_fold_validation"
                if cv_protocol == CV_PROTOCOL_TRAIN_VAL
                else "best_checkpoint_outer_fold_test"
            ),
            "active_backend": active_backend,
            "active_aggregation_level": active_aggregation_level,
        }
        if selection_metadata:
            row.update(
                {
                    "selected_epoch": selection_metadata.get("epoch"),
                    "selection_metric": selection_metadata.get("selection_metric", ""),
                    "selection_metric_mode": selection_metadata.get("selection_metric_mode", ""),
                    "selection_metric_value": selection_metadata.get("selection_metric_value"),
                }
            )
        active_metrics: dict | None = None
        for backend_name, filename in BACKEND_METRIC_FILES.items():
            metrics_path = next(
                (
                    path
                    for path in _candidate_metric_paths(fold_dir, filename, cv_protocol)
                    if path.exists()
                ),
                None,
            )
            if metrics_path is None:
                continue
            backend_metrics = read_json(metrics_path)
            row[f"{backend_name}_prediction_backend"] = backend_metrics.get("prediction_backend", backend_name)
            row[f"{backend_name}_aggregation_level"] = backend_metrics.get("aggregation_level", "")
            row[f"{backend_name}_evaluation_protocol_name"] = backend_metrics.get("evaluation_protocol_name", "")
            row[f"{backend_name}_metrics_path"] = str(metrics_path)
            for key, value in backend_metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                row[f"{backend_name}_{key}"] = value
                metrics_by_backend[backend_name][key].append(float(value))
            if backend_name == active_backend:
                active_metrics = backend_metrics

        if cv_protocol == CV_PROTOCOL_TRAIN_VAL and active_metrics is None:
            expected_path = _candidate_metric_paths(
                fold_dir,
                BACKEND_METRIC_FILES[active_backend],
                cv_protocol,
            )[0]
            raise FileNotFoundError(
                f"Missing active best-validation metrics for {fold_dir.name}: {expected_path}"
            )
        if active_metrics is not None:
            row["active_metrics_path"] = row.get(f"{active_backend}_metrics_path", "")
            for key, value in active_metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                row[f"active_{key}"] = value
                active_metrics_by_key[key].append(float(value))
            confusion_matrix = active_metrics.get("confusion_matrix") or active_metrics.get("binary_strict_confusion_matrix")
            active_confusion_matrix = _add_confusion_matrices(active_confusion_matrix, confusion_matrix)
            active_support_negative += int(active_metrics.get("support_negative", 0) or 0)
            active_support_positive += int(active_metrics.get("support_positive", 0) or 0)
        rows.append(row)

    if len(expected_fold_counts) > 1:
        raise ValueError(
            f"Mixed split.outer_folds values under {run_root}: {sorted(expected_fold_counts)}"
        )
    if cv_protocols == {CV_PROTOCOL_TRAIN_VAL} and expected_fold_counts:
        expected_fold_count = next(iter(expected_fold_counts))
        expected_fold_names = {f"fold_{index}" for index in range(expected_fold_count)}
        actual_fold_names = {row["fold"] for row in rows}
        if actual_fold_names != expected_fold_names:
            raise ValueError(
                "Incomplete train_val CV summary. "
                f"Missing={sorted(expected_fold_names - actual_fold_names)} "
                f"unexpected={sorted(actual_fold_names - expected_fold_names)}"
            )

    active_metric_summary = {
        key: _summary_stats(values) for key, values in active_metrics_by_key.items() if values
    }
    selected_epoch_summary = _summary_stats(
        [float(row["selected_epoch"]) for row in rows if row.get("selected_epoch") is not None]
    )
    active_pooled_metrics = _metrics_from_confusion_matrix(active_confusion_matrix)
    active_summary_row = {
        "folds": len([row for row in rows if row.get("active_metrics_path")]),
        "cv_protocol": next(iter(cv_protocols), ""),
        "score_source": rows[0]["score_source"] if rows else "",
        "selected_epoch_mean": selected_epoch_summary["mean"] if selected_epoch_summary else 0.0,
        "selected_epoch_std": selected_epoch_summary["std"] if selected_epoch_summary else 0.0,
        "active_backend": rows[0]["active_backend"] if rows else "",
        "active_aggregation_level": rows[0]["active_aggregation_level"] if rows else "",
        "accuracy_mean": active_metric_summary.get("accuracy", {}).get("mean", 0.0),
        "accuracy_std": active_metric_summary.get("accuracy", {}).get("std", 0.0),
        "positive_f1_mean": active_metric_summary.get("positive_f1", {}).get("mean", 0.0),
        "positive_f1_std": active_metric_summary.get("positive_f1", {}).get("std", 0.0),
        "precision_mean": active_metric_summary.get("precision", {}).get("mean", 0.0),
        "precision_std": active_metric_summary.get("precision", {}).get("std", 0.0),
        "recall_mean": active_metric_summary.get("recall", {}).get("mean", 0.0),
        "recall_std": active_metric_summary.get("recall", {}).get("std", 0.0),
        "macro_f1_mean": active_metric_summary.get("macro_f1", {}).get("mean", 0.0),
        "macro_f1_std": active_metric_summary.get("macro_f1", {}).get("std", 0.0),
        "weighted_f1_mean": active_metric_summary.get("weighted_f1", {}).get("mean", 0.0),
        "weighted_f1_std": active_metric_summary.get("weighted_f1", {}).get("std", 0.0),
        "auroc_mean": active_metric_summary.get("auroc", {}).get("mean", 0.0),
        "auroc_std": active_metric_summary.get("auroc", {}).get("std", 0.0),
        "pooled_accuracy": active_pooled_metrics.get("accuracy", 0.0),
        "pooled_positive_f1": active_pooled_metrics.get("positive_f1", 0.0),
        "pooled_precision": active_pooled_metrics.get("precision", 0.0),
        "pooled_recall": active_pooled_metrics.get("recall", 0.0),
        "pooled_macro_f1": active_pooled_metrics.get("macro_f1", 0.0),
        "pooled_weighted_f1": active_pooled_metrics.get("weighted_f1", 0.0),
        "pooled_support_negative": active_support_negative,
        "pooled_support_positive": active_support_positive,
        "pooled_confusion_matrix": active_confusion_matrix or [],
    }
    for key, stats in active_metric_summary.items():
        active_summary_row.setdefault(f"{key}_mean", stats["mean"])
        active_summary_row.setdefault(f"{key}_std", stats["std"])
    final_summary = {
        "cv_protocol": next(iter(cv_protocols), ""),
        "score_source": rows[0]["score_source"] if rows else "",
        "fold_rows": rows,
        "active_backend_metric_summary": active_metric_summary,
        "active_backend_pooled_metrics": active_pooled_metrics,
        "active_backend_pooled_confusion_matrix": active_confusion_matrix,
        "active_backend_summary_row": active_summary_row,
        "score_metrics_by_backend": {
            backend: {key: _summary_stats(values) for key, values in metric_lists.items() if values}
            for backend, metric_lists in metrics_by_backend.items()
        },
    }
    final_summary["best_checkpoint_metrics_by_backend"] = final_summary["score_metrics_by_backend"]
    save_json(final_summary, run_root / "final_summary.json")
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["fold", "active_backend"]
    with (run_root / "final_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_root / "final_summary_active.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(active_summary_row.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(active_summary_row)


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
