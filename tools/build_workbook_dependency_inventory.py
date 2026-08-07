from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import write_json_atomic
from src.experiment_tracking.workbook_selection import (
    DEFAULT_BUILDER_PATH,
    DEFAULT_WORKBOOK_PATH,
    WorkbookSelectionError,
    build_dependency_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read the workbook Provenance sheet into a dependency inventory. "
            "Read-only: never guesses attempt ids, never mutates the workbook or registry."
        )
    )
    parser.add_argument(
        "--workbook",
        default=str(PROJECT_ROOT / DEFAULT_WORKBOOK_PATH),
        help="Workbook path (default: depression_results_clean.xlsx)",
    )
    parser.add_argument(
        "--builder",
        default=str(PROJECT_ROOT / DEFAULT_BUILDER_PATH),
        help="Workbook builder script path (default: scripts/build_clean_workbook.py)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/experiment_registry/workbook_dependency_inventory.json"),
        help="Output path for the dependency inventory JSON",
    )
    args = parser.parse_args()
    try:
        inventory = build_dependency_inventory(args.workbook, builder_path=args.builder)
    except WorkbookSelectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_json_atomic(args.output, inventory)
    summary = inventory["summary"]
    print(json.dumps({"wrote": args.output, **summary}, indent=2, sort_keys=True))
    if summary["malformed_rows"] or summary["duplicate_keys"]:
        print(
            f"warning: {summary['malformed_rows']} malformed row(s), "
            f"{summary['duplicate_keys']} duplicate provenance key(s)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
