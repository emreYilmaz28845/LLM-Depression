#!/usr/bin/env python3
"""Create a new translation attempt containing only validation failures as pending."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, save_json, sha256_file, write_jsonl


def _key(row: dict) -> tuple[str, str, int]:
    return str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0))


def prepare(parent_root: Path, retry_root: Path, *, retry_seed: int) -> dict:
    parent_root = parent_root.resolve()
    retry_root = retry_root.resolve()
    if retry_root.exists():
        raise FileExistsError(f"Retry root already exists: {retry_root}")
    if parent_root == retry_root or parent_root in retry_root.parents:
        raise ValueError("Retry root must not equal or be nested inside the parent attempt.")

    required = ("units.jsonl", "candidates.jsonl", "accepted.jsonl", "rejected.jsonl", "audit.json")
    missing = [name for name in required if not (parent_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Parent attempt is missing: {missing}")

    units = read_jsonl(parent_root / "units.jsonl")
    candidates = read_jsonl(parent_root / "candidates.jsonl")
    accepted = read_jsonl(parent_root / "accepted.jsonl")
    rejected = read_jsonl(parent_root / "rejected.jsonl")
    if not rejected:
        raise ValueError("Parent attempt has no rejected translations to retry.")

    unit_keys = {_key(row) for row in units}
    candidate_keys = {_key(row) for row in candidates}
    rejected_keys = {_key(row) for row in rejected}
    if len(unit_keys) != len(units) or len(candidate_keys) != len(candidates):
        raise ValueError("Parent attempt contains duplicate unit or candidate keys.")
    if candidate_keys != unit_keys:
        raise ValueError("Parent candidate coverage does not exactly match its units.")
    if not rejected_keys.issubset(unit_keys):
        raise ValueError("Parent rejected rows contain unknown unit keys.")
    if len(accepted) + len(rejected) != len(units):
        raise ValueError("Parent accepted/rejected coverage does not match its units.")

    retained = [row for row in candidates if _key(row) not in rejected_keys]
    retry_root.mkdir(parents=True)
    write_jsonl(units, retry_root / "units.jsonl")
    write_jsonl(retained, retry_root / "candidates.jsonl")
    if (parent_root / "length_profile.json").is_file():
        shutil.copy2(parent_root / "length_profile.json", retry_root / "length_profile.json")

    provenance = {
        "schema_version": "translation_validation_retry.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_root": str(parent_root),
        "retry_root": str(retry_root),
        "retry_seed": retry_seed,
        "unit_count": len(units),
        "retained_candidate_count": len(retained),
        "pending_rejected_count": len(rejected_keys),
        "action": "regenerate_parent_validation_rejections",
        "parent_hashes": {name: sha256_file(parent_root / name) for name in required},
    }
    save_json(provenance, retry_root / "repair_provenance.json")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--retry-seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(args.parent_root, args.retry_root, retry_seed=args.retry_seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
