from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry


def _print_rows(rows, columns: tuple[str, ...]) -> None:
    print("\t".join(columns))
    for row in rows:
        print("\t".join(_cell(row[column] if column in row.keys() else None) for column in columns))


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _cmd_list(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.list_runs(connection, dataset=args.dataset, status=args.status)
    finally:
        connection.close()
    _print_rows(
        rows,
        (
            "attempt_id",
            "logical_run_name",
            "dataset",
            "modality",
            "fold",
            "current_state",
            "backend",
            "evaluation_view",
            "aggregation",
            "metric_namespace",
            "headline_positive_f1",
        ),
    )
    return 0


def _cmd_show(args) -> int:
    connection = registry.connect(args.db)
    try:
        payload = registry.show_attempt(connection, args.attempt_id, fold=args.fold)
    finally:
        connection.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_provenance(args) -> int:
    connection = registry.connect(args.db)
    try:
        payload = registry.provenance_of_metric(connection, args.metric_id)
    finally:
        connection.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_jobs(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.list_jobs(connection, failed_only=args.failed)
    finally:
        connection.close()
    _print_rows(
        rows,
        (
            "at_utc",
            "attempt_id",
            "logical_run_name",
            "fold",
            "job_key",
            "job_type",
            "event_type",
            "slurm_job_id",
            "status",
            "resubmission_of_job_id",
        ),
    )
    return 0


def _cmd_best(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.best_runs(
            connection,
            dataset=args.dataset,
            metric=args.metric,
            namespace=args.namespace,
            backend=args.backend,
            view=args.view,
            aggregation=args.aggregation,
            limit=args.limit,
        )
    finally:
        connection.close()
    if not rows:
        print("no matches for the fully qualified query")
        return 0
    _print_rows(
        rows,
        (
            "attempt_id",
            "logical_run_name",
            "fold",
            "metric_value",
            "support",
            "backend",
            "evaluation_view",
            "aggregation",
            "namespace",
            "split_name",
            "checkpoint_role",
            "evidence_artifact_id",
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local experiment registry.")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (default: outputs/experiment_registry/experiments.sqlite)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list runs with headline metrics")
    list_parser.add_argument("--dataset", default=None)
    list_parser.add_argument("--status", default=None)
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="show an attempt with its folds/evaluations/artifacts/jobs")
    show_parser.add_argument("attempt_id")
    show_parser.add_argument("--fold", type=int, default=None)
    show_parser.set_defaults(func=_cmd_show)

    provenance_parser = subparsers.add_parser("provenance", help="show the provenance chain of a metric id")
    provenance_parser.add_argument("metric_id", type=int)
    provenance_parser.set_defaults(func=_cmd_provenance)

    jobs_parser = subparsers.add_parser("jobs", help="list recorded job events")
    jobs_parser.add_argument("--failed", action="store_true", help="only failed/cancelled/timed-out jobs")
    jobs_parser.set_defaults(func=_cmd_jobs)

    best_parser = subparsers.add_parser(
        "best",
        help="fully qualified best-metric query; every qualifier is required to avoid mixing protocols",
    )
    for option in ("--dataset", "--metric", "--namespace", "--backend", "--view", "--aggregation"):
        best_parser.add_argument(option, required=True)
    best_parser.add_argument("--limit", type=int, default=None)
    best_parser.set_defaults(func=_cmd_best)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
