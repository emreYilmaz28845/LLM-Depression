#!/usr/bin/env python
"""Audit window15 treatment inputs against the control and locked expectations.

DAIC: compares the freshly built 15-second packed manifest against the audited
30-second control and a lock file re-derived from the actual build. The lock
captures the treatment's own observed totals (subjects/labels/splits, retained
participant speech, chunk counts and ranges) so later builds fail on drift.

The lock is written only with ``--lock-write``; the default mode is read-only
verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

CONTROL_RETAINED_FRAMES = 1406614000  # canonical 30s packed participant speech
CONTROL_RETAINED_ROWS = 32373
CONTROL_SUBJECTS = {"train": 107, "val": 35, "test": 47}
CONTROL_PROTOCOL_ID = "daic_participant_speech_packed30_v1"
MAX_CHUNK_SAMPLES = 240000


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def observed_daic(runtime_root: Path) -> dict:
    audit = read_json(runtime_root / "manifests/daic/daic_participant_speech_packed30_corpus_audit.json")
    totals = audit["totals"]
    return {
        "subjects": totals["subjects"],
        "subjects_by_label": totals.get("subjects_by_label"),
        "retained_rows": totals["retained_rows"],
        "retained_frames": totals["retained_frames"],
        "speech_seconds": totals["speech_seconds"],
        "chunks": totals["chunks"],
        "chunks_by_split_label": totals["chunks_by_split_label"],
        "chunks_per_subject_range": totals["chunks_per_subject_range"],
        "final_chunk_samples_range": totals["final_chunk_samples_range"],
        "final_chunk_count": totals["final_chunk_count"],
        "blank_lines": totals["blank_lines"],
        "nonblank_rows": totals["nonblank_rows"],
        "excluded_non_participant_rows": totals["excluded_non_participant_rows"],
        "excluded_empty_participant_rows": totals["excluded_empty_participant_rows"],
        "protocol_id": audit.get("protocol_id"),
        "locked_contract": audit.get("locked_contract"),
    }


def verify(observed: dict, lock: dict | None) -> list[str]:
    issues: list[str] = []
    if observed["subjects"] != CONTROL_SUBJECTS:
        issues.append(f"subjects {observed['subjects']} != control {CONTROL_SUBJECTS}")
    if observed["retained_rows"] != CONTROL_RETAINED_ROWS:
        issues.append(f"retained_rows {observed['retained_rows']} != control {CONTROL_RETAINED_ROWS}")
    if observed["retained_frames"] != CONTROL_RETAINED_FRAMES:
        issues.append(
            f"retained_frames {observed['retained_frames']} != control {CONTROL_RETAINED_FRAMES} "
            "(participant speech coverage must be identical)"
        )
    if observed["protocol_id"] != CONTROL_PROTOCOL_ID:
        issues.append(
            f"protocol_id {observed['protocol_id']!r} != {CONTROL_PROTOCOL_ID!r}; DAIC aggregation/head "
            "policies are gated on this protocol id"
        )
    contract = observed.get("locked_contract") or {}
    if int(contract.get("chunk_samples") or 0) != MAX_CHUNK_SAMPLES:
        issues.append(f"chunk_samples {contract.get('chunk_samples')} != {MAX_CHUNK_SAMPLES}")
    final_range = observed["final_chunk_samples_range"]
    if int(final_range[1]) > MAX_CHUNK_SAMPLES:
        issues.append(f"final chunk max {final_range[1]} exceeds {MAX_CHUNK_SAMPLES}")
    if observed["final_chunk_count"] != len(range(CONTROL_SUBJECTS["train"] + CONTROL_SUBJECTS["val"] + CONTROL_SUBJECTS["test"])):
        issues.append(f"final_chunk_count {observed['final_chunk_count']} != subject count 189")
    if lock is not None:
        for key in (
            "subjects",
            "retained_rows",
            "retained_frames",
            "chunks",
            "chunks_by_split_label",
            "chunks_per_subject_range",
            "final_chunk_samples_range",
            "final_chunk_count",
            "blank_lines",
            "nonblank_rows",
            "excluded_non_participant_rows",
            "excluded_empty_participant_rows",
        ):
            if lock.get(key) != observed.get(key):
                issues.append(f"locked {key} {lock.get(key)!r} != observed {observed.get(key)!r}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--dataset", default="daic", choices=["daic"])
    parser.add_argument(
        "--lock",
        type=Path,
        default=LANE / "outputs/qwen3_window15_20261008/contracts/daic15_expected.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LANE / "outputs/qwen3_window15_20261008/contracts/daic15_audit.json",
    )
    parser.add_argument("--lock-write", action="store_true")
    args = parser.parse_args()

    observed = observed_daic(args.runtime_root)
    lock = None
    if args.lock.is_file():
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
    issues = verify(observed, lock)
    if args.lock_write and not args.lock.is_file():
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(observed, indent=1, sort_keys=True), encoding="utf-8")
        issues = verify(observed, observed)
        print(json.dumps({"lock_written": str(args.lock)}))
    report = {
        "schema_version": "audiollm.qwen3_window15_input_audit.v1",
        "dataset": args.dataset,
        "runtime_root": str(args.runtime_root),
        "observed": observed,
        "issues": issues,
        "status": "passed" if not issues else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": issues, "output": str(args.output)}))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
