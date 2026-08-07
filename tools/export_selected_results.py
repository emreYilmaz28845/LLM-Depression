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
    data = yaml.safe_load(Path(selection_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("selections"), list):
        raise ValueError("selection file must contain a 'selections' list")
    return data["selections"]


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

    selections = load_selection(args.selection)
    connection = registry.connect(args.db)
    try:
        results = [resolve_selection(connection, selection) for selection in selections]
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
