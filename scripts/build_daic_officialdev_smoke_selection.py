#!/usr/bin/env python3
"""Build the deterministic smoke-only subject-selection file for one
officialdev smoke parent.

Reads the smoke training parent's saved split and picks, per class, the two
lowest subject IDs from the saved training partition (fit) and the two lowest
from the saved selection partition (eval). All chosen subjects come from the
official training partition; the file is hashed into the smoke cache identity.
Raw subject IDs stay in ignored evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import read_json, sha256_file


def build_selection(parent_fold_dir: Path, *, per_class: int = 2) -> dict[str, object]:
    import yaml

    split_payload = read_json(parent_fold_dir / "logs" / "split_used.json")
    labels: dict[str, int] = {}
    train_ids = [str(item) for item in split_payload.get("train_subject_ids") or []]
    selection_ids = [str(item) for item in split_payload.get("selection_subject_ids") or []]
    if not train_ids or not selection_ids:
        raise ValueError("smoke parent saved split is empty")
    run_config_path = parent_fold_dir / "run_config.yaml"
    if not run_config_path.is_file():
        raise ValueError("smoke parent has no run_config.yaml")
    run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    partition_path = Path(run_config["split_metadata_path"])
    if not partition_path.is_file():
        raise ValueError("smoke parent run_config has no resolvable partition file")
    for row in read_json(partition_path):
        labels[str(row["subject_id"])] = int(row["label"])

    def pick(ids: list[str]) -> list[str]:
        chosen: list[str] = []
        for label in (0, 1):
            members = sorted(subject_id for subject_id in ids if labels.get(subject_id) == label)
            if len(members) < per_class:
                raise ValueError(
                    f"class {label} has {len(members)} subjects; need at least {per_class} per class"
                )
            chosen.extend(members[:per_class])
        return sorted(chosen)

    outer_train = pick(train_ids)
    final_eval = pick(selection_ids)
    if set(outer_train) & set(final_eval):
        raise ValueError("fit/eval smoke selections overlap")
    return {
        "schema_version": "daic_officialdev_smoke_selection.v1",
        "outer_train": outer_train,
        "final_eval": final_eval,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fold-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args(argv)
    payload = build_selection(args.parent_fold_dir, per_class=args.per_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
