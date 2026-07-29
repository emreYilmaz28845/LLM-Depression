#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def allocated_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture byte counts for retained D3TEC artifact trees."
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named path to measure; may be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {}
    for value in args.path:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}.")
        label, raw_path = value.split("=", 1)
        if not label or label in payload:
            raise ValueError(f"Invalid or repeated storage label: {label!r}.")
        payload[label] = allocated_bytes(Path(raw_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
