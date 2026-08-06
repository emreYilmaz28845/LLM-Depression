#!/usr/bin/env python
"""Compare two deterministic evaluation passes byte-identically.

Normalizes rows to canonical JSON strings before comparison so formatting and
key order do not matter, while any scientific difference (scores, margins,
predictions, metrics) still fails the gate. Handles JSONL sample rows, CSV
subject rows, and whole-file JSON metrics. Exits nonzero on any difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _normalize_lines(path: Path) -> list[str]:
    rows: list[Any] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [json.loads(path.read_text(encoding="utf-8"))]
    return sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    )


COMPARISON_NAMES = (
    "predictions_sample_level.jsonl",
    "predictions_subject_level.csv",
    "metrics_original_teacher_forced.json",
    "final_and_best_validation_metrics.json",
)


def compare_determinism(pass1_dir: Path, pass2_dir: Path, names=COMPARISON_NAMES) -> list[str]:
    mismatches: list[str] = []
    for name in names:
        first = Path(pass1_dir) / name
        second = Path(pass2_dir) / name
        if not first.is_file() or not second.is_file():
            mismatches.append(f"missing {name} in a pass")
            continue
        a = _normalize_lines(first)
        b = _normalize_lines(second)
        if a != b:
            mismatches.append(f"{name} differs between passes")
            continue
        print(f"pass1 == pass2 ({name}): {len(a)} rows")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass1_dir", type=Path)
    parser.add_argument("pass2_dir", type=Path)
    args = parser.parse_args()
    mismatches = compare_determinism(args.pass1_dir, args.pass2_dir)
    if mismatches:
        print("Determinism gate FAILED: " + "; ".join(mismatches), file=sys.stderr)
        sys.exit(1)
    print("Determinism gate PASSED: both evaluation passes are identical.")


if __name__ == "__main__":
    main()
