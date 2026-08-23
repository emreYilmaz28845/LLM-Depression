#!/usr/bin/env python3
"""Create a new translation cache with explicit native-speaker corrections."""
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

from src.utils import read_jsonl, save_json, sha256_file, sha256_text, write_jsonl


def _key(row: dict) -> tuple[str, str, int]:
    return str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0))


def prepare(parent_root: Path, units_path: Path, corrections_path: Path, output_root: Path) -> dict:
    parent_root = parent_root.resolve()
    units_path = units_path.resolve()
    corrections_path = corrections_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    required = ("units.jsonl", "candidates.jsonl", "accepted.jsonl", "rejected.jsonl", "audit.json")
    missing = [name for name in required if not (parent_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Parent attempt is missing: {missing}")

    old_units = {_key(row): row for row in read_jsonl(parent_root / "units.jsonl")}
    old_candidates = {_key(row): row for row in read_jsonl(parent_root / "candidates.jsonl")}
    new_units = read_jsonl(units_path)
    if len(old_units) != len(old_candidates) or set(old_units) != set(old_candidates):
        raise ValueError("Parent unit and candidate coverage differ.")
    if len(new_units) != len(old_units) or {_key(row) for row in new_units} != set(old_units):
        raise ValueError("Corrected units must preserve the parent unit-key set.")

    corrections: dict[tuple[str, str, int], dict] = {}
    for row in read_jsonl(corrections_path):
        key = _key(row)
        if key in corrections:
            raise ValueError(f"Duplicate translation correction for {key}")
        translation = str(row.get("corrected_translation", "")).strip()
        if not translation or not str(row.get("reviewed_by", "")).strip():
            raise ValueError(f"Translation correction for {key} is incomplete.")
        corrections[key] = row
    if not corrections:
        raise ValueError("At least one reviewed translation correction is required.")

    candidates: list[dict] = []
    reviewed: list[dict] = []
    source_changes = 0
    context_changes = 0
    for unit in new_units:
        key = _key(unit)
        old_unit = old_units[key]
        old_candidate = old_candidates[key]
        source_changed = str(unit["source_sha256"]) != str(old_unit["source_sha256"])
        context_changed = str(unit.get("context_sha256", "")) != str(old_unit.get("context_sha256", ""))
        source_changes += int(source_changed)
        context_changes += int(context_changed)
        correction = corrections.get(key)
        if source_changed != (correction is not None):
            raise ValueError(f"Every and only source-changed unit must have a correction: {key}")
        candidate = dict(old_candidate)
        candidate["source_sha256"] = unit["source_sha256"]
        if correction is not None:
            expected = str(correction.get("expected_source_sha256", ""))
            if expected != str(unit["source_sha256"]):
                raise ValueError(f"Corrected source hash mismatch for {key}")
            translation = str(correction["corrected_translation"]).strip()
            candidate["translation"] = translation
            candidate["translation_sha256"] = sha256_text(translation)
            candidate["status"] = "translated"
            reviewed.append(
                {
                    "unit_id": unit["unit_id"],
                    "status": "human_verified",
                    "reviewed_by": str(correction["reviewed_by"]),
                }
            )
        candidates.append(candidate)

    if set(corrections) != {_key(row) for row in new_units if _key(row) in corrections}:
        raise ValueError("Translation corrections refer to unknown units.")
    output_root.mkdir(parents=True)
    write_jsonl(new_units, output_root / "units.jsonl")
    write_jsonl(candidates, output_root / "candidates.jsonl")
    write_jsonl(reviewed, output_root / "reviewed.jsonl")
    if (parent_root / "length_profile.json").is_file():
        shutil.copy2(parent_root / "length_profile.json", output_root / "length_profile.json")
    provenance = {
        "schema_version": "translation_reviewed_correction.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_root": str(parent_root),
        "output_root": str(output_root),
        "unit_count": len(new_units),
        "retained_candidate_count": len(candidates) - len(corrections),
        "corrected_candidate_count": len(corrections),
        "source_changed_unit_count": source_changes,
        "context_changed_unit_count": context_changes,
        "action": "apply_native_speaker_transcript_and_translation_correction",
        "parent_hashes": {name: sha256_file(parent_root / name) for name in required},
        "units_sha256": sha256_file(output_root / "units.jsonl"),
        "candidates_sha256": sha256_file(output_root / "candidates.jsonl"),
        "reviewed_sha256": sha256_file(output_root / "reviewed.jsonl"),
        "corrections_sha256": sha256_file(corrections_path),
    }
    save_json(provenance, output_root / "repair_provenance.json")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_root, args.units, args.corrections, args.output_root), indent=2))


if __name__ == "__main__":
    main()
