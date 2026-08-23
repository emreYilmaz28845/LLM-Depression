from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.experiment_tracking import registry


def load_selection(selection_path: str | Path) -> list[dict[str, Any]]:
    return load_selection_document(selection_path)["selections"]


def load_selection_document(selection_path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(selection_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("selections"), list):
        raise ValueError("selection file must contain a 'selections' list")
    return data


def resolve_selection(
    connection: Any,
    selection: dict[str, Any],
) -> dict[str, Any]:
    cell = str(selection["cell"])
    dataset = str(selection["dataset"])
    modality = str(selection["modality"])
    metric = str(selection["metric"])
    namespace = str(selection["namespace"])
    backend = str(selection["backend"])
    view = str(selection["view"])
    aggregation = str(selection["aggregation"])
    attempt_id = selection.get("attempt_id")
    rows = registry.best_runs(
        connection,
        dataset=dataset,
        metric=metric,
        namespace=namespace,
        backend=backend,
        view=view,
        aggregation=aggregation,
    )
    if attempt_id is not None:
        rows = [row for row in rows if row["attempt_id"] == attempt_id]
    if not rows:
        return {
            "cell": cell,
            "status": "legacy_unmigrated",
            "reason": "no registry record for the fully qualified query",
            "value": None,
        }
    if len(rows) > 1:
        return {
            "cell": cell,
            "status": "rejected_ambiguous",
            "reason": f"{len(rows)} matching records; selection must be explicit",
            "value": None,
        }
    row = rows[0]
    value = row["metric_value"]
    if value is None:
        return {
            "cell": cell,
            "status": "rejected_missing_value",
            "reason": "metric value is null in the registry record",
            "value": None,
        }
    evidence = registry.provenance_of_metric(connection, int(row["metric_id"]))
    metrics_artifacts = [
        artifact["path"] for artifact in evidence["artifacts"] if artifact["artifact_type"] == "metrics"
    ]
    return {
        "cell": cell,
        "status": "selected",
        "value": value,
        "attempt_id": row["attempt_id"],
        "logical_run_name": row["logical_run_name"],
        "fold": row["fold"],
        "metric": metric,
        "namespace": namespace,
        "backend": backend,
        "evaluation_view": view,
        "aggregation": aggregation,
        "provenance": {
            "metric_id": evidence["metric"]["metric_id"],
            "evaluation_id": evidence["evaluation"]["evaluation_id"],
            "metrics_artifacts": metrics_artifacts,
        },
    }


def resolve_native_en_selection(
    report: dict[str, Any],
    report_path: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a Native vs EN workbook cell from the deterministic report."""
    cell = str(selection["cell"])
    report_key = str(selection["report_provenance_key"])
    metric = str(selection["metric"])
    summary_rows = [
        row for row in report.get("summary", [])
        if str(row.get("provenance_key")) == report_key
    ]
    base = {
        "cell": cell,
        "sheet": "Native vs EN",
        "source_type": "native_en_report",
        "metric": metric,
    }
    if len(summary_rows) != 1:
        return {
            **base,
            "status": "legacy_unmigrated",
            "reason": f"expected one report row for {report_key}, found {len(summary_rows)}",
            "value": None,
        }
    allowed_metrics = {
        "native_macro_mean", "native_macro_sd", "english_macro_mean", "english_macro_sd",
        "delta_macro_mean", "delta_macro_sd", "native_positive_mean", "native_positive_sd",
        "english_positive_mean", "english_positive_sd", "delta_positive_mean", "delta_positive_sd",
    }
    if metric not in allowed_metrics:
        return {
            **base,
            "status": "legacy_unmigrated",
            "reason": f"unsupported Native vs EN metric: {metric}",
            "value": None,
        }
    row = summary_rows[0]
    provenance = {
        "report_path": str(report_path),
        "report_provenance_key": report_key,
        "provenance_status": row.get("provenance_status"),
        "aggregation": row.get("aggregation"),
        "evaluation_view": "harmonized_all_windows_full_coverage",
    }
    value = row.get(metric)
    if value is None:
        return {
            **base,
            "status": "rejected_missing_value",
            "reason": "report metric is null",
            "value": None,
            "provenance": provenance,
        }
    return {
        **base,
        "status": "selected",
        "value": value,
        "provenance": provenance,
    }


def _load_native_en_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "native_en_text_heads_v2_report.v1":
        raise ValueError(f"unsupported Native vs EN report schema: {report_path}")
    if report.get("status") != "passed":
        raise ValueError(f"Native vs EN report is not passed: {report_path}")
    if len(report.get("summary", [])) != 24 or len(report.get("seed_details", [])) != 72:
        raise ValueError(f"Native vs EN report has unexpected row counts: {report_path}")
    return report


def _resolve_report_path(selection_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.is_file():
        return project_relative
    return selection_path.parent / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an explicit workbook-cell selection against the registry. "
        "Never auto-selects the highest-scoring run; missing records are reported, never zeroed."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path",
    )
    parser.add_argument("--selection", required=True, help="Selection YAML with a 'selections' list")
    parser.add_argument("--output", required=True, help="Selected-results JSON output path")
    args = parser.parse_args()

    selection_path = Path(args.selection)
    document = load_selection_document(selection_path)
    selections = document["selections"]
    native_en_report = None
    native_en_report_path = None
    if document.get("source_type") == "native_en_report":
        raw_report_path = document.get("report_path")
        if not isinstance(raw_report_path, str) or not raw_report_path:
            raise ValueError("native_en_report selection requires report_path")
        native_en_report_path = _resolve_report_path(selection_path, raw_report_path)
        if not native_en_report_path.is_file():
            raise FileNotFoundError(f"Native vs EN report does not exist: {native_en_report_path}")
        native_en_report = _load_native_en_report(native_en_report_path)
    connection = registry.connect(args.db)
    try:
        results = [
            resolve_native_en_selection(native_en_report, native_en_report_path, selection)
            if native_en_report is not None
            else resolve_selection(connection, selection)
            for selection in selections
        ]
    finally:
        connection.close()
    statuses: dict[str, int] = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    payload = {
        "schema_version": "audiollm.selected_results.v1",
        "selections": results,
        "status_counts": statuses,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(statuses, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
