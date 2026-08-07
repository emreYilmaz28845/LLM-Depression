from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry, wandb_export
from src.experiment_tracking.discovery import discover_run_at
from src.experiment_tracking.qualification import qualify_run


def _print_plan(plan: dict) -> None:
    summary = {
        "schema_version": plan["schema_version"],
        "run_id": plan["run_id"],
        "project": plan["project"],
        "job_type": plan["job_type"],
        "attempt_id": plan["identity"]["attempt_id"],
        "fold": plan["identity"]["fold"],
        "logical_run_name": plan["identity"]["logical_run_name"],
        "dataset": plan["identity"]["dataset"],
        "modality": plan["identity"]["modality"],
        "evaluation_id": plan["identity"]["evaluation_id"],
        "backend": _qualifiers(plan),
        "epoch_points": sum(len(points) for points in plan["epoch_curves"].values()),
        "summary_metric_count": len(plan["summary_metrics"]),
        "status": plan["status"],
        "tags": plan["tags"],
        "incomplete_reasons": plan["incomplete_reasons"],
        "exclusion_count": len(plan["exclusions"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _qualifiers(plan: dict) -> dict:
    values = {}
    for name in ("test/", "selection/", "aggregate/"):
        for key, value in plan["summary_metrics"].items():
            if key.startswith(name):
                parts = key.split("/")
                values.setdefault("backend", parts[1])
                values.setdefault("aggregation", parts[2])
                values.setdefault("namespace", parts[3])
                break
        if values:
            break
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evidence-first W&B export plan builder. Dry-run by default; never contacts the network."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Run directory containing fold_<n> subdirectories")
    source.add_argument("--attempt-id", help="Registry attempt id")
    source.add_argument("--group", help="Registry group id (backfill dry-run)")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (required for --attempt-id and --group)",
    )
    parser.add_argument("--project", default="audiollm-depression")
    parser.add_argument("--mode", choices=("dry_run", "offline", "cloud"), default="dry_run")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --mode dry_run")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Backfill report output path (only written when supplied, for --group mode)",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.mode = "dry_run"
    if args.mode != "dry_run":
        print(
            "real W&B export requires an authenticated local W&B account; "
            "exporter dry-run remains available without W&B installed",
            file=sys.stderr,
        )

    plans: list[dict] = []
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            parser.error(f"run directory does not exist: {run_dir}")
        fold_configs = sorted(run_dir.glob("fold_*/run_config.yaml"))
        if not fold_configs:
            parser.error(f"no fold_<n>/run_config.yaml found under {run_dir}")
        for run_config_path in fold_configs:
            discovered = discover_run_at(run_config_path.parent)
            result = qualify_run(discovered)
            plan = wandb_export.build_export_plan_from_result(discovered, result, project=args.project)
            if plan is not None:
                plans.append(plan)
    elif args.group is not None:
        connection = registry.connect(args.db)
        try:
            rows = connection.execute(
                "SELECT a.attempt_id FROM run_attempts a "
                "JOIN logical_runs lr ON lr.logical_run_id = a.logical_run_id "
                "WHERE lr.group_id = ?",
                (args.group,),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            plan = _plan_from_db(args.db, row["attempt_id"], args.project)
            if plan is not None:
                plans.append(plan)
    else:
        plan = _plan_from_db(args.db, args.attempt_id, args.project)
        if plan is None:
            parser.error(f"no qualified export plan for attempt: {args.attempt_id}")
        plans = [plan]

    if args.mode == "dry_run":
        for plan in plans:
            _print_plan(plan)
        if args.output is not None:
            report = {
                "mode": "dry_run",
                "output": args.output,
                "plans": [
                    {
                        "run_id": plan["run_id"],
                        "attempt_id": plan["identity"]["attempt_id"],
                        "fold": plan["identity"]["fold"],
                        "status": plan["status"],
                        "summary_metric_count": len(plan["summary_metrics"]),
                        "exclusion_count": len(plan["exclusions"]),
                    }
                    for plan in plans
                ],
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {output_path}")
        return 0

    adapter = wandb_export.RealWandbAdapter(entity=args.entity)
    results = []
    for plan in plans:
        results.append(wandb_export.execute_export(plan, adapter, mode=args.mode, entity=args.entity))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _plan_from_db(db_path: str, attempt_id: str, project: str) -> dict | None:
    connection = registry.connect(db_path)
    try:
        return wandb_export.build_export_plan(connection, attempt_id, project=project)
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
