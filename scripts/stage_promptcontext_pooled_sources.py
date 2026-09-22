#!/usr/bin/env python3
"""Stage the audited Turkish pooled source inputs for MN5 path resolution.

The pooled manifest is built on the cluster from four audited source manifest/split
pairs. Their manifests carry the local dataset root, so the MN5 copy must resolve
the same rows against the cluster dataset root. This script performs exactly the
documented translation (``/media/emre/Backup/AudioLLM/Datasets`` ->
``/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets``) used by the pooled
campaign, refuses any path outside those roots, records every input and output
hash, and never modifies the source root.

Output: the eight staged files plus ``source_staging_audit.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_DATASET_ROOT = "/media/emre/Backup/AudioLLM/Datasets"
REMOTE_DATASET_ROOT = "/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets"
DEFAULT_SOURCE_ROOT = Path(
    "/home/emre/worktrees/LLM-Depression-exp-turkish-pooled-qcond-clean-v1/outputs"
    "/turkish_pooled_qcond/exp-turkish-pooled-qcond-clean-v1-20260903/source_inputs"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/prompt_context_pooled_sources/source_inputs_remote"
)

# The four condition/language pairs the pooled builder audits.
SOURCE_FILES = (
    "manifests/pos_native/turkish_manifest.jsonl",
    "splits/pos_native/turkish_folds.json",
    "manifests/neg_native/turkish_manifest.jsonl",
    "splits/neg_native/turkish_folds.json",
    "manifests/pos_english/turkish_manifest.jsonl",
    "splits/pos_english/turkish_folds.json",
    "manifests/neg_english/turkish_manifest.jsonl",
    "splits/neg_english/turkish_folds.json",
)


class StagingError(ValueError):
    """Raised when a pooled source input cannot be staged."""


def translate_audio_path(value: str) -> str:
    if value.startswith(REMOTE_DATASET_ROOT + "/"):
        return value
    if value.startswith(LOCAL_DATASET_ROOT + "/"):
        return REMOTE_DATASET_ROOT + value[len(LOCAL_DATASET_ROOT) :]
    raise StagingError(
        "pooled source audio path is outside the documented dataset roots: " + value
    )


def stage_manifest(source: Path) -> bytes:
    lines: list[str] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise StagingError(f"blank line in {source}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StagingError(f"invalid JSON in {source}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise StagingError(f"row is not an object in {source}:{line_number}")
        if "audio_path" in row:
            if not isinstance(row["audio_path"], str):
                raise StagingError(f"audio_path is not a string in {source}:{line_number}")
            row["audio_path"] = translate_audio_path(row["audio_path"])
        if "audio_paths" in row:
            paths = row["audio_paths"]
            if not isinstance(paths, list) or not all(
                isinstance(path, str) for path in paths
            ):
                raise StagingError(
                    f"audio_paths is not a string list in {source}:{line_number}"
                )
            row["audio_paths"] = [translate_audio_path(path) for path in paths]
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    return "".join(lines).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise StagingError(f"pooled source root is missing: {source_root}")
    output_root = args.output_root.resolve()

    entries: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        source = source_root / relative
        if not source.is_file():
            raise StagingError(f"missing pooled source input: {source}")
        payload = (
            stage_manifest(source)
            if relative.startswith("manifests/")
            else source.read_bytes()
        )
        destination = output_root / relative
        if destination.is_file():
            if destination.read_bytes() != payload:
                raise StagingError(
                    f"refusing to overwrite incompatible staged input: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        entries.append(
            {
                "relative_path": relative,
                "source_path": str(source),
                "source_sha256": _sha256(source.read_bytes()),
                "staged_path": str(destination),
                "staged_sha256": _sha256(payload),
                "bytes": len(payload),
                "audio_paths_translated": relative.startswith("manifests/"),
            }
        )

    audit = {
        "schema_version": "audiollm.promptcontext_pooled_source_staging.v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "translation_rule": {
            "from": LOCAL_DATASET_ROOT,
            "to": REMOTE_DATASET_ROOT,
            "applies_to": "manifest audio_path/audio_paths fields only",
        },
        "files": entries,
    }
    audit_path = output_root / "source_staging_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"staged {len(entries)} files under {output_root}")
    for entry in entries:
        print(f"  {entry['relative_path']:<48} {entry['staged_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
