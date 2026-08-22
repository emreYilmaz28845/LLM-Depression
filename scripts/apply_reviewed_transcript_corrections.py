#!/usr/bin/env python3
"""Create an audited transcript derivative from native-speaker corrections."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, save_json, sha256_file, sha256_text, write_jsonl


def apply_corrections(source: Path, corrections_path: Path, output: Path, audit_path: Path) -> dict:
    source = source.resolve()
    corrections_path = corrections_path.resolve()
    output = output.resolve()
    audit_path = audit_path.resolve()
    if output.exists() or audit_path.exists():
        raise FileExistsError("Corrected transcript output and audit must be new files.")
    if source == output:
        raise ValueError("The reviewed derivative must not overwrite the source transcript.")

    rows = read_jsonl(source)
    corrections = read_jsonl(corrections_path)
    if not corrections:
        raise ValueError("At least one reviewed correction is required.")
    by_audio: dict[str, dict] = {}
    for correction in corrections:
        basename = str(correction.get("audio_filename", "")).strip()
        if not basename or Path(basename).name != basename:
            raise ValueError("Each correction needs a basename-only audio_filename.")
        if basename in by_audio:
            raise ValueError(f"Duplicate correction for {basename}")
        if not str(correction.get("reviewed_by", "")).strip():
            raise ValueError(f"Correction for {basename} is missing reviewed_by.")
        corrected = str(correction.get("corrected_transcript", "")).strip()
        if not corrected:
            raise ValueError(f"Correction for {basename} has an empty transcript.")
        by_audio[basename] = correction

    applied: list[dict] = []
    seen_audio: set[str] = set()
    output_rows: list[dict] = []
    for row in rows:
        updated = dict(row)
        basename = Path(str(row.get("audio_path", ""))).name
        correction = by_audio.get(basename)
        if correction is not None:
            if basename in seen_audio:
                raise ValueError(f"Source transcript contains duplicate audio basename: {basename}")
            seen_audio.add(basename)
            original = str(row.get("transcript", ""))
            expected_hash = str(correction.get("expected_transcript_sha256", ""))
            if sha256_text(original) != expected_hash:
                raise ValueError(f"Source transcript hash mismatch for {basename}")
            corrected = str(correction["corrected_transcript"]).strip()
            if corrected == original:
                raise ValueError(f"Correction for {basename} does not change the transcript.")
            updated.update(
                {
                    "transcript": corrected,
                    "repair_status": "HUMAN_VERIFIED",
                    "repair_actions": ["native_speaker_asr_correction"],
                    "original_transcript_sha256": sha256_text(original),
                    "manual_review_recommended": False,
                    "manual_review_reason_codes": [],
                }
            )
            applied.append(
                {
                    "audio_filename_sha256": sha256_text(basename),
                    "original_transcript_sha256": sha256_text(original),
                    "corrected_transcript_sha256": sha256_text(corrected),
                    "reviewed_by": str(correction["reviewed_by"]),
                    "reason": str(correction.get("reason", "asr_error")),
                }
            )
        output_rows.append(updated)

    missing = sorted(set(by_audio) - seen_audio)
    if missing:
        raise ValueError(f"Corrections refer to audio missing from the source: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_rows, output)
    audit = {
        "schema_version": "reviewed_transcript_corrections.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "corrections_sha256": sha256_file(corrections_path),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "row_count": len(output_rows),
        "correction_count": len(applied),
        "corrections": applied,
    }
    save_json(audit, audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_corrections(args.source, args.corrections, args.output, args.audit), indent=2))


if __name__ == "__main__":
    main()
