from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry, wandb_export
from src.experiment_tracking.canonical import write_json_atomic
from src.experiment_tracking.discovery import discover_run_at
from src.experiment_tracking.qualification import qualify_run
from src.experiment_tracking.workbook_selection import (
    MANIFEST_SCHEMA_VERSION,
    local_evidence_checks,
    payload_hash,
    registry_evidence_hash,
)

DRY_RUN_AUDIT_SCHEMA_VERSION = "audiollm.wandb_dry_run_audit.v1"
EXPORT_AUDIT_SCHEMA_VERSION = "audiollm.wandb_export_audit.v1"


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


def _load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} unreadable: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return payload


def _manifest_units(manifest: dict) -> list[dict]:
    units = manifest.get("export_units")
    if not isinstance(units, list):
        raise ValueError("manifest has no export_units list")
    return units


def _apply_manifest_meta(plan: dict, unit: dict) -> dict:
    plan["tags"] = sorted(set(plan["tags"]) | set(unit.get("tags") or []))
    plan["group"] = unit.get("group")
    plan["selection"] = {
        "selection_ids": unit.get("selection_ids") or [],
        "provenance_keys": unit.get("provenance_keys") or [],
        "source_type": unit.get("source_type"),
    }
    return plan


def _build_manifest_plans(
    db_path: str, manifest: dict, entity: str | None
) -> tuple[list[tuple[dict, dict]], list[str]]:
    connection = registry.connect(db_path)
    try:
        plans: list[tuple[dict, dict]] = []
        problems: list[str] = []
        for unit in _manifest_units(manifest):
            plan = wandb_export.build_export_plan(
                connection, unit["attempt_id"], unit["fold"], project=manifest.get("project") or "audiollm-depression"
            )
            if plan is None:
                problems.append(
                    f"no export plan for {unit['attempt_id']} fold {unit['fold']}"
                )
                continue
            if plan["run_id"] != unit.get("wandb_run_id"):
                problems.append(
                    f"run id mismatch for {unit['attempt_id']} fold {unit['fold']}: "
                    f"plan {plan['run_id']} vs manifest {unit.get('wandb_run_id')}"
                )
            actual_evidence = registry_evidence_hash(connection, unit["attempt_id"], unit["fold"])
            if actual_evidence != unit.get("registry_evidence_sha256"):
                problems.append(
                    f"registry evidence changed since resolution for {unit['attempt_id']} "
                    f"fold {unit['fold']}"
                )
            evidence = local_evidence_checks(
                connection, unit["attempt_id"], unit["fold"], unit.get("evaluation_ids") or [],
                source_type=unit.get("source_type"),
            )
            for key, value in evidence.items():
                if key != "run_dir" and value != "ok":
                    problems.append(
                        f"local evidence changed since resolution for {unit['attempt_id']} "
                        f"fold {unit['fold']}: {key}: {value}"
                    )
            plans.append((plan, unit))
        return plans, problems
    finally:
        connection.close()


def _manifest_dry_run(
    args: argparse.Namespace,
    manifest: dict,
    db_path: str,
) -> int:
    plans, problems = _build_manifest_plans(db_path, manifest, args.entity)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 2
    units_out: list[dict[str, Any]] = []
    for plan, unit in plans:
        plan = _apply_manifest_meta(plan, unit)
        units_out.append(
            {
                "run_id": plan["run_id"],
                "name": plan.get("name"),
                "attempt_id": plan["identity"]["attempt_id"],
                "fold": plan["identity"]["fold"],
                "selection_ids": unit.get("selection_ids") or [],
                "provenance_keys": unit.get("provenance_keys") or [],
                "evaluation_ids": unit.get("evaluation_ids") or [],
                "group": plan.get("group"),
                "tags": plan["tags"],
                "status": plan["status"],
                "incomplete_reasons": plan["incomplete_reasons"],
                "payload_sha256": payload_hash(plan),
                "registry_evidence_sha256": unit.get("registry_evidence_sha256"),
                "exclusion_count": len(plan["exclusions"]),
                "summary_metric_count": len(plan["summary_metrics"]),
            }
        )
        _print_plan(plan)
    audit: dict[str, Any] = {
        "schema_version": DRY_RUN_AUDIT_SCHEMA_VERSION,
        "mode": "dry_run",
        "manifest": {
            "path": str(args.manifest),
            "sha256": _file_sha256(args.manifest),
        },
        "workbook": manifest.get("workbook"),
        "selection": manifest.get("selection"),
        "db_path": db_path,
        "entity": args.entity,
        "project": manifest.get("project"),
        "units": units_out,
        "summary": {
            "unit_count": len(units_out),
            "complete_units": sum(1 for unit in units_out if unit["status"] == "complete"),
            "incomplete_units": sum(1 for unit in units_out if unit["status"] == "incomplete"),
            "total_exclusions": sum(unit["exclusion_count"] for unit in units_out),
            "payload_duplicates": len(units_out)
            - len({unit["payload_sha256"] for unit in units_out}),
        },
    }
    output = Path(args.output or str(PROJECT_ROOT / "outputs/experiment_registry/workbook_wandb_dry_run.json"))
    write_json_atomic(output, audit)
    print(f"wrote {output}")
    return 0


def _file_sha256(path: str | Path) -> str:
    from src.experiment_tracking.canonical import sha256_file

    return sha256_file(path)


def _verify_approved_dry_run(args: argparse.Namespace, manifest: dict, db_path: str) -> list[str]:
    approved_path = Path(args.approved_dry_run)
    approved = _load_json(approved_path, "approved dry-run audit")
    if approved.get("schema_version") != DRY_RUN_AUDIT_SCHEMA_VERSION:
        return [f"approved dry-run audit schema mismatch: {approved.get('schema_version')}"]
    failures: list[str] = []
    manifest_workbook = manifest.get("workbook") or {}
    approved_workbook = approved.get("workbook") or {}
    if manifest_workbook.get("sha256") != approved_workbook.get("sha256"):
        failures.append(
            f"workbook sha256 changed: manifest {manifest_workbook.get('sha256')} "
            f"vs approved audit {approved_workbook.get('sha256')}"
        )
    try:
        actual_workbook = _file_sha256(manifest_workbook.get("path"))
    except OSError as error:
        return [f"workbook unreadable: {error}"]
    if actual_workbook != manifest_workbook.get("sha256"):
        failures.append(
            f"workbook file sha256 changed: {actual_workbook} vs manifest {manifest_workbook.get('sha256')}"
        )
    if manifest.get("selection", {}).get("sha256") != approved.get("selection", {}).get("sha256"):
        failures.append("selection sha256 changed since approved dry run")
    approved_by_run = {unit["run_id"]: unit for unit in approved.get("units", [])}
    for unit in _manifest_units(manifest):
        approved_unit = approved_by_run.get(unit["wandb_run_id"])
        if approved_unit is None:
            failures.append(f"unit {unit['wandb_run_id']} not present in approved dry run")
            continue
        if approved_unit.get("registry_evidence_sha256") != unit.get("registry_evidence_sha256"):
            failures.append(
                f"registry evidence changed since approved dry run for {unit['wandb_run_id']}"
            )
    plans, plan_problems = _build_manifest_plans(db_path, manifest, args.entity)
    failures.extend(plan_problems)
    for plan, unit in plans:
        plan = _apply_manifest_meta(plan, unit)
        approved_unit = approved_by_run.get(plan["run_id"])
        if approved_unit is None:
            continue
        actual = payload_hash(plan)
        if actual != approved_unit.get("payload_sha256"):
            failures.append(
                f"payload changed since approved dry run for {plan['run_id']}: "
                f"{actual} vs approved {approved_unit.get('payload_sha256')}"
            )
    return failures


def _manifest_cloud(args: argparse.Namespace, manifest: dict, db_path: str) -> int:
    failures = _verify_approved_dry_run(args, manifest, db_path)
    if failures:
        print("error: refusing cloud export:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    plans, problems = _build_manifest_plans(db_path, manifest, args.entity)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    adapter = wandb_export.RealWandbAdapter(entity=args.entity)
    results: list[dict[str, Any]] = []
    failed = 0
    for plan, unit in plans:
        plan = _apply_manifest_meta(plan, unit)
        try:
            outcome = wandb_export.execute_export(
                plan, adapter, mode=args.mode, entity=args.entity
            )
            outcome["selection_ids"] = unit.get("selection_ids") or []
            outcome["attempt_id"] = plan["identity"]["attempt_id"]
            outcome["fold"] = plan["identity"]["fold"]
            outcome["group"] = plan.get("group")
        except Exception as error:
            failed += 1
            outcome = {
                "mode": args.mode,
                "run_id": plan["run_id"],
                "attempt_id": plan["identity"]["attempt_id"],
                "fold": plan["identity"]["fold"],
                "error": str(error),
            }
        results.append(outcome)
    audit: dict[str, Any] = {
        "schema_version": EXPORT_AUDIT_SCHEMA_VERSION,
        "mode": args.mode,
        "entity": args.entity,
        "project": manifest.get("project"),
        "manifest": {"path": str(args.manifest), "sha256": _file_sha256(args.manifest)},
        "approved_dry_run": {"path": str(args.approved_dry_run), "sha256": _file_sha256(args.approved_dry_run)},
        "units": results,
        "summary": {
            "unit_count": len(results),
            "failed": failed,
            "succeeded": len(results) - failed,
        },
    }
    output = Path(args.output or str(PROJECT_ROOT / "outputs/experiment_registry/workbook_wandb_export_audit.json"))
    write_json_atomic(output, audit)
    print(f"wrote {output}")
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evidence-first W&B export plan builder. Dry-run by default; never contacts the network."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Run directory containing fold_<n> subdirectories")
    source.add_argument("--attempt-id", help="Registry attempt id")
    source.add_argument("--group", help="Registry group id (backfill dry-run)")
    source.add_argument("--manifest", help="Resolved workbook-selection manifest JSON")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (required for --attempt-id, --group, and --manifest)",
    )
    parser.add_argument(
        "--approved-dry-run",
        default=None,
        help="Approved dry-run audit JSON; required for cloud export with --manifest",
    )
    parser.add_argument("--project", default="audiollm-depression")
    parser.add_argument("--mode", choices=("dry_run", "offline", "cloud"), default="dry_run")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --mode dry_run")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Audit report output path (only written when supplied)",
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

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            parser.error(f"manifest does not exist: {manifest_path}")
        manifest = _load_json(manifest_path, "manifest")
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            parser.error(
                f"manifest schema mismatch: {manifest.get('schema_version')}"
            )
        if args.mode == "cloud" and args.approved_dry_run is None:
            parser.error("cloud export with --manifest requires --approved-dry-run")
        if args.mode in ("cloud", "offline"):
            return _manifest_cloud(args, manifest, args.db)
        return _manifest_dry_run(args, manifest, args.db)

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
