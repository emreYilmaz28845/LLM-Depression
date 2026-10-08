#!/usr/bin/env python
"""Generate the window15 treatment configs from the frozen control contracts.

Each treatment config is a byte-minimal copy of its canonical control config
with only the declared window/input identity changes:

- recipe marker ``single30`` -> ``single15``;
- DAIC: ``manifest_variant`` -> ``unprocessed_participant_speech_packed30_15s_v1``
  and ``data.participant_chunk_samples`` 480000 -> 240000;
- response datasets: ``data.segment_seconds`` 30.0 -> 15.0.

The generator edits only those exact lines so every other byte (including
prompt templates and their scalar style) stays identical. ``--check`` verifies
the text output and the parsed allowlist (no other key differs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

CONTRACT = LANE / "outputs/qwen3_window15_20261008/contracts/treatment_contract.json"
ALLOWED_KEYS = {"recipe_id", "manifest_variant", "data.segment_seconds", "data.participant_chunk_samples"}


def generate(control_text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    text = control_text

    def sub_once(pattern: str, replacement: str, label: str) -> None:
        nonlocal text
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"expected exactly one match for {label}, found {count}")
        text = new_text
        changes.append(label)

    sub_once(r"^(recipe_id: .*?)single30(.*)$", r"\1single15\2", "recipe_id single30->single15")
    if re.search(r"^  sample_mode: participant_speech_packed30$", control_text, flags=re.MULTILINE):
        sub_once(
            r"^manifest_variant: unprocessed_participant_speech_packed30_v1$",
            "manifest_variant: unprocessed_participant_speech_packed30_15s_v1",
            "manifest_variant -> _15s_v1",
        )
        sub_once(
            r"^  participant_chunk_samples: 480000$",
            "  participant_chunk_samples: 240000",
            "participant_chunk_samples 480000->240000",
        )
    else:
        sub_once(r"^  segment_seconds: 30\.0$", "  segment_seconds: 15.0", "segment_seconds 30->15")
    return text, changes


def parsed_diff(control: dict, treatment: dict) -> set[str]:
    paths: set[str] = set()
    for key in set(control) | set(treatment):
        if control.get(key) != treatment.get(key):
            if key == "data":
                for sub in set(control.get(key) or {}) | set(treatment.get(key) or {}):
                    if (control.get(key) or {}).get(sub) != (treatment.get(key) or {}).get(sub):
                        paths.add(f"data.{sub}")
            else:
                paths.add(key)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    failures: list[str] = []
    written = 0
    for route_id, route in sorted(contract["routes"].items()):
        control_path = LANE / route["control_config"]
        target_path = LANE / route["treatment_config"]
        control_text = control_path.read_text(encoding="utf-8")
        treatment_text, _ = generate(control_text)
        control_parsed = yaml.safe_load(control_text) or {}
        treatment_parsed = yaml.safe_load(treatment_text) or {}
        undeclared = parsed_diff(control_parsed, treatment_parsed) - ALLOWED_KEYS
        if undeclared:
            failures.append(f"{route_id}: undeclared parsed differences: {sorted(undeclared)}")
        if args.check:
            if not target_path.is_file():
                failures.append(f"{route_id}: treatment config missing: {target_path}")
                continue
            if target_path.read_text(encoding="utf-8") != treatment_text:
                failures.append(f"{route_id}: treatment config differs from the generator output")
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(treatment_text, encoding="utf-8")
        written += 1
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=1))
        return 1
    print(json.dumps({"status": "ok", "written": written, "checked": bool(args.check)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
