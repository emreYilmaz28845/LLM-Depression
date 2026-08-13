from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import write_json_atomic
from src.experiment_tracking.registry import DEFAULT_DB_PATH
from src.experiment_tracking.workbook_selection import (
    DEFAULT_BUILDER_PATH,
    DEFAULT_WORKBOOK_PATH,
    WorkbookSelectionError,
    build_dependency_inventory,
    load_selection,
    resolve_manifest,
    verify_selection_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the reviewed workbook W&B selection against the SQLite registry "
            "into a deduplicated export manifest. No network calls."
        )
    )
    parser.add_argument(
        "--selection",
        required=True,
        help="Reviewed selection YAML (experiments/definitions/workbook_wandb_selection.yaml)",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / DEFAULT_DB_PATH),
        help="SQLite registry path",
    )
    parser.add_argument(
        "--workbook",
        default=str(PROJECT_ROOT / DEFAULT_WORKBOOK_PATH),
        help="Workbook path used to verify the selection workbook hash",
    )
    parser.add_argument(
        "--builder",
        default=str(PROJECT_ROOT / DEFAULT_BUILDER_PATH),
        help="Workbook builder path used to verify the selection builder hash",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/experiment_registry/workbook_wandb_manifest.json"),
        help="Output path for the resolved manifest JSON",
    )
    parser.add_argument(
        "--filter-attempts",
        type=Path,
        default=None,
        help=(
            "Optional task-owned JSON/TSV filter: when given, only export units "
            "whose attempt id is listed are kept in the manifest. Used for "
            "task-filtered dry runs so unrelated approved units cannot be "
            "exported accidentally."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolution summary without writing the manifest",
    )
    args = parser.parse_args()

    try:
        selection = load_selection(args.selection)
        failures = verify_selection_hashes(
            selection, workbook_path=args.workbook, builder_path=args.builder
        )
        if failures:
            print("error: selection hash verification failed:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 2
        inventory = build_dependency_inventory(args.workbook, builder_path=args.builder)
        if inventory["workbook"]["sha256"] != selection["workbook"]["sha256"]:
            print("error: workbook changed between inventory and selection", file=sys.stderr)
            return 2
        manifest = resolve_manifest(
            selection,
            inventory,
            db_path=args.db,
            selection_path=args.selection,
        )
        if args.filter_attempts is not None:
            manifest = _filter_manifest_units(manifest, args.filter_attempts)
    except WorkbookSelectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    for entry in manifest["entries"]:
        if entry["status"] not in ("resolved", "skip_not_run", "skip_derived_only"):
            print(
                f"{entry['status']:>28} {entry['selection_id']}"
                + (f"  -- {'; '.join(entry['reasons'][:2])}" if entry["reasons"] else "")
            )
    if manifest["unresolved_entries"]:
        print("unresolved workbook rows:", file=sys.stderr)
        for row in manifest["unresolved_entries"]:
            print(f"  - {row['provenance_key']} (row {row['row_number']})", file=sys.stderr)
    if manifest["summary"]["unresolved_rows"]:
        return 3
    if args.dry_run:
        print("dry-run: manifest not written")
        return 0
    write_json_atomic(args.output, manifest)
    print(f"wrote {args.output}")
    return 0


def _filter_manifest_units(manifest: dict, filter_path: Path) -> dict:
    """Keep only export units whose attempt id is listed in the filter file.

    The filter file is task-owned JSON (list of attempt ids) or one attempt id
    per line. Unresolved/stale entries are kept untouched so the summary stays
    honest about full-coverage state.
    """
    import json as _json

    raw = filter_path.read_text(encoding="utf-8")
    if filter_path.suffix.lower() == ".json":
        payload = _json.loads(raw)
        if not isinstance(payload, list):
            raise WorkbookSelectionError("filter-attempts JSON must be a list of attempt ids")
        allowed = {str(item) for item in payload}
    else:
        allowed = {line.strip() for line in raw.splitlines() if line.strip()}
    if not allowed:
        raise WorkbookSelectionError("filter-attempts resolved to an empty set")
    units = manifest.get("export_units") or []
    kept = [unit for unit in units if str(unit.get("attempt_id")) in allowed]
    dropped = [unit for unit in units if str(unit.get("attempt_id")) not in allowed]
    if not kept:
        raise WorkbookSelectionError("filter-attempts matched no export units")
    manifest = dict(manifest)
    manifest["export_units"] = kept
    manifest["filtered_out_units"] = [
        {"attempt_id": unit.get("attempt_id"), "group": unit.get("group")} for unit in dropped
    ]
    summary = dict(manifest.get("summary") or {})
    summary["sync_units"] = len(kept)
    manifest["summary"] = summary
    print(f"task filter: kept {len(kept)} unit(s), excluded {len(dropped)} unrelated unit(s)")
    return manifest


if __name__ == "__main__":
    sys.exit(main())
