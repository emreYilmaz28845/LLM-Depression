from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merged.protocol import DATASETS, METHODS
from src.merged.runtime import load_merged_config
from src.metrics import binary_auroc, classification_metrics
from src.utils import ensure_dir, read_json, read_jsonl, save_json


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prediction_value(row: dict[str, Any]) -> int:
    for field in ("prediction", "predicted_class", "teacher_forced_prediction", "parsed_prediction"):
        value = row.get(field)
        if value not in (None, "", "INVALID"):
            try:
                parsed = int(value)
                if parsed in (0, 1):
                    return parsed
            except (TypeError, ValueError):
                pass
    return 1 - int(row["label"])


def _probability_value(row: dict[str, Any], prediction: int) -> float:
    for field in ("probability", "dep_score", "score_margin", "teacher_forced_margin"):
        value = row.get(field)
        if value not in (None, ""):
            try:
                number = float(value)
                if field in {"score_margin", "teacher_forced_margin", "dep_score"}:
                    return 1.0 / (1.0 + math.exp(-number))
                return number
            except (TypeError, ValueError, OverflowError):
                pass
    return float(prediction)


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [_prediction_value(row) for row in rows]
    scores = [_probability_value(row, pred) for row, pred in zip(rows, y_pred)]
    metrics = classification_metrics(y_true, y_pred)
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    metrics.update(
        {
            "negative_f1": 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
            if neg_precision + neg_recall
            else 0.0,
            "auroc": binary_auroc(y_true, scores),
            "invalid_qwen_outputs": sum(
                1 for row in rows if any(row.get(field) == "INVALID" for field in ("prediction_text", "teacher_forced_prediction_text", "generation_prediction_text"))
            ),
            "class_support_negative": int(sum(value == 0 for value in y_true)),
            "class_support_positive": int(sum(value == 1 for value in y_true)),
            "sample_count": len(rows),
        }
    )
    return metrics


def _metric_row(
    *,
    dataset: str,
    modality: str,
    method: str,
    stage: str,
    fold: int | str,
    metrics: dict[str, Any],
    protocol_label: str,
    fold_coverage: int,
) -> dict[str, Any]:
    return {
        "Dataset": dataset,
        "Modality": modality,
        "Method": method,
        "Stage": stage,
        "Fold": fold,
        "Accuracy": metrics.get("accuracy", 0.0),
        "Positive F1": metrics.get("positive_f1", 0.0),
        "Precision": metrics.get("precision", 0.0),
        "Recall": metrics.get("recall", 0.0),
        "Macro-F1": metrics.get("macro_f1", 0.0),
        "Negative F1": metrics.get("negative_f1", 0.0),
        "AUROC": metrics.get("auroc", 0.0),
        "Support Negative": metrics.get("class_support_negative", metrics.get("support_negative", 0)),
        "Support Positive": metrics.get("class_support_positive", metrics.get("support_positive", 0)),
        "Confusion Matrix": json.dumps(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), sort_keys=True),
        "Invalid Qwen Outputs": metrics.get("invalid_qwen_outputs", 0),
        "Fold Coverage": fold_coverage,
        "Protocol": protocol_label,
    }


def collect_stage_rows(config_path: str | Path, *, run_id: str, stage: str) -> list[dict[str, Any]]:
    config = load_merged_config(config_path)
    modality = str(config["modality"])
    root = Path(config["output_dirs"]["merged_root"]) / run_id / stage
    folds = [0] if stage == "final" else list(range(5))
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_root = root / f"fold_{fold}"
        method_paths: dict[str, dict[str, Path]] = {dataset: {} for dataset in DATASETS}
        qwen_root = fold_root / "qwen"
        for dataset in DATASETS:
            path = qwen_root / dataset / "predictions_subject_level.csv"
            if path.is_file():
                method_paths[dataset]["qwen"] = path
        for method in ("logreg", "xgb_fixed", "xgb_optuna"):
            path = fold_root / "heads" / method / "predictions_subject_level.jsonl"
            if path.is_file():
                prediction_rows = _read_prediction_rows(path)
                by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for item in prediction_rows:
                    by_dataset[str(item.get("dataset", "")).lower()].append(item)
                for dataset, values in by_dataset.items():
                    temp = fold_root / "heads" / method / f"{dataset}_predictions.jsonl"
                    if not temp.is_file():
                        temp.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")
                    method_paths.setdefault(dataset, {})[method] = temp
        for dataset in DATASETS:
            if stage == "final" and dataset != "daic":
                continue
            for method in METHODS:
                path = method_paths.get(dataset, {}).get(method)
                if path is None or not path.is_file():
                    continue
                prediction_rows = _read_prediction_rows(path)
                metrics = _metrics_from_rows(prediction_rows)
                rows.append(
                    _metric_row(
                        dataset=dataset,
                        modality=modality,
                        method=method,
                        stage=stage,
                        fold=fold,
                        metrics=metrics,
                        protocol_label="symmetric_merged_cv" if stage == "cv" else "symmetric_merged_daic_official_test",
                        fold_coverage=1,
                    )
                )
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = list(rows[0]) if rows else ["Dataset", "Modality", "Method"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_fold_rows(rows: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Dataset"], row["Modality"], row["Method"])].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, modality, method), values in sorted(grouped.items()):
        out = {
            "Dataset": dataset,
            "Modality": modality,
            "Method": method,
            "Stage": stage,
            "Fold": "mean±std",
            "Fold Coverage": len(values),
            "Protocol": values[0]["Protocol"],
        }
        for metric in ("Accuracy", "Positive F1", "Precision", "Recall", "Macro-F1", "Negative F1", "AUROC"):
            numbers = [float(value[metric]) for value in values]
            out[metric] = f"{statistics.mean(numbers):.6f}±{statistics.pstdev(numbers):.6f}"
        out["Support Negative"] = sum(int(value["Support Negative"]) for value in values)
        out["Support Positive"] = sum(int(value["Support Positive"]) for value in values)
        out["Confusion Matrix"] = "pooled_in_pooled_table"
        out["Invalid Qwen Outputs"] = sum(int(value["Invalid Qwen Outputs"]) for value in values)
        result.append(out)
    return result


def _aggregate_summary(rows: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Modality"], row["Method"])].append(row)
    result: list[dict[str, Any]] = []
    for (modality, method), values in sorted(grouped.items()):
        by_dataset = {row["Dataset"]: float(row["Macro-F1"]) for row in values}
        result.append(
            {
                "Dataset": "five_dataset_mean",
                "Modality": modality,
                "Method": method,
                "Stage": stage,
                "Fold": "aggregate",
                "Accuracy": "",
                "Positive F1": "",
                "Precision": "",
                "Recall": "",
                "Macro-F1": statistics.mean(by_dataset.values()) if by_dataset else 0.0,
                "Negative F1": "",
                "AUROC": "",
                "Support Negative": "",
                "Support Positive": "",
                "Confusion Matrix": "",
                "Invalid Qwen Outputs": sum(int(row["Invalid Qwen Outputs"]) for row in values),
                "Fold Coverage": len(values),
                "Protocol": values[0]["Protocol"],
            }
        )
        result.append(
            {
                "Dataset": "worst_dataset",
                "Modality": modality,
                "Method": method,
                "Stage": stage,
                "Fold": "aggregate",
                "Accuracy": "",
                "Positive F1": "",
                "Precision": "",
                "Recall": "",
                "Macro-F1": min(by_dataset.values()) if by_dataset else 0.0,
                "Negative F1": "",
                "AUROC": "",
                "Support Negative": "",
                "Support Positive": "",
                "Confusion Matrix": "",
                "Invalid Qwen Outputs": sum(int(row["Invalid Qwen Outputs"]) for row in values),
                "Fold Coverage": len(values),
                "Protocol": values[0]["Protocol"],
            }
        )
    return result


def update_workbook(workbook_path: Path, cv_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    from openpyxl import load_workbook
    from openpyxl.worksheet.table import Table, TableStyleInfo

    workbook = load_workbook(workbook_path) if workbook_path.is_file() else None
    if workbook is None:
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.remove(workbook.active)
    for title, rows in (("Merged Symmetric CV", cv_rows), ("Merged DAIC Official", final_rows)):
        if title in workbook.sheetnames:
            del workbook[title]
        sheet = workbook.create_sheet(title)
        if not rows:
            rows = [{"Status": "No completed artifacts"}]
        headers = list(rows[0])
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header, "") for header in headers])
        if sheet.max_row >= 2:
            ref = f"A1:{chr(64 + min(sheet.max_column, 26))}{sheet.max_row}"
            table = Table(displayName="Merged" + ("CV" if "CV" in title else "DAICOfficial"), ref=ref)
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
            sheet.add_table(table)
            sheet.auto_filter.ref = ref
        sheet.freeze_panes = "A2"
    workbook.save(workbook_path)


def generate_reports(
    config_paths: list[str | Path], *, run_id: str, output_dir: str | Path, workbook_path: str | Path | None = None
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    cv_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        cv_rows.extend(collect_stage_rows(config_path, run_id=run_id, stage="cv"))
        final_rows.extend(collect_stage_rows(config_path, run_id=run_id, stage="final"))
    cv_fold_path = output / "symmetric_merged_cv_fold_level.csv"
    cv_summary_path = output / "symmetric_merged_cv_fold_mean_std.csv"
    cv_aggregate_path = output / "symmetric_merged_cv_aggregate.csv"
    final_path = output / "symmetric_merged_daic_official_test.csv"
    _write_rows(cv_rows, cv_fold_path)
    _write_rows(_aggregate_fold_rows(cv_rows, stage="cv"), cv_summary_path)
    _write_rows(_aggregate_summary(cv_rows, stage="cv"), cv_aggregate_path)
    _write_rows(final_rows, final_path)
    workbook = Path(workbook_path) if workbook_path else Path("depression_results_combined_with_posf1_graphs.xlsx")
    if cv_rows or final_rows:
        update_workbook(workbook, cv_rows + _aggregate_fold_rows(cv_rows, stage="cv") + _aggregate_summary(cv_rows, stage="cv"), final_rows)
    report_path = output / "symmetric_merged_execution_results.md"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report_path.write_text(
        "# Symmetric merged execution/results\n\n"
        f"- Run ID: `{run_id}`\n- Git commit: `{commit}`\n"
        f"- CV rows: {len(cv_rows)}\n- DAIC official-test rows: {len(final_rows)}\n"
        "- Protocol: five datasets, three modalities, Qwen + Logistic Regression + fixed XGBoost + 150-trial grouped Optuna XGBoost.\n",
        encoding="utf-8",
    )
    return {
        "cv_rows": len(cv_rows),
        "final_rows": len(final_rows),
        "csv_paths": [str(cv_fold_path), str(cv_summary_path), str(cv_aggregate_path), str(final_path)],
        "workbook": str(workbook),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact symmetric merged CSV/report/workbook artifacts.")
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workbook", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_reports(args.config, run_id=args.run_id, output_dir=args.output_dir, workbook_path=args.workbook)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
