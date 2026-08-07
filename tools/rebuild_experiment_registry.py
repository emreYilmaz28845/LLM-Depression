from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the local SQLite experiment registry from run evidence."
    )
    parser.add_argument("--scan-root", required=True, help="Directory containing output_model-style runs")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (default: outputs/experiment_registry/experiments.sqlite)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover/qualify only; never create or modify the DB")
    args = parser.parse_args()
    summary = registry.rebuild_registry(args.scan_root, args.db, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
