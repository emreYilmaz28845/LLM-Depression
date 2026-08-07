from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry
from src.experiment_tracking.discovery import discover_run_at
from src.experiment_tracking.qualification import qualify_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import one run directory (all its fold_<n> subdirectories) into the registry."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing fold_<n> subdirectories")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (default: outputs/experiment_registry/experiments.sqlite)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover/qualify only; never create or modify the DB")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    fold_dirs = sorted(run_dir.glob("fold_*/run_config.yaml"))
    if not fold_dirs:
        parser.error(f"no fold_<n>/run_config.yaml found under {run_dir}")
    if args.dry_run:
        connection = None
    else:
        connection = registry.ensure_registry(args.db)
    try:
        outcomes = []
        for run_config_path in fold_dirs:
            discovered = discover_run_at(run_config_path.parent)
            result = qualify_run(discovered)
            outcomes.append(registry.import_run(connection, discovered, result, dry_run=args.dry_run))
        print(json.dumps({"run_dir": str(run_dir), "dry_run": args.dry_run, "folds": outcomes}, indent=2, sort_keys=True))
    finally:
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
