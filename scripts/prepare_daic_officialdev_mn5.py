#!/usr/bin/env python3
"""MN5-native preflight for the DAIC official-development campaign.

Rebuilds the canonical DAIC packed30 manifest from an officialdev config,
proves the locked 86/21/35 split contract and expected row counts against the
rebuilt manifest, validates that every manifest source path lives under the
MN5 dataset prefix, and writes a task-owned audit JSON declaring the expected
24-job scope. Model-free; no GPU and no internet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)

from scripts.audit_daic_officialdev_split import audit_split, AuditFailure

# One config is enough to rebuild the shared manifest; the audit command
# verifies identity across all six configs.
MANIFEST_BUILD_CONFIG = (
    "configs/main/daic_audio_text_harmonized_selmacrof1_tf_officialdev.yaml"
)
ALL_SIX_CONFIGS = (
    "configs/main/daic_audio_only_harmonized_selmacrof1_tf_officialdev.yaml",
    "configs/main/daic_audio_text_harmonized_selmacrof1_tf_officialdev.yaml",
    "configs/main/daic_text_only_harmonized_selmacrof1_tf_officialdev.yaml",
    "configs/main/daic_audio_only_harmonized_selmacrof1_tf_gemma4_12b_officialdev.yaml",
    "configs/main/daic_audio_text_harmonized_selmacrof1_tf_gemma4_12b_officialdev.yaml",
    "configs/main/daic_text_only_harmonized_selmacrof1_tf_gemma4_12b_officialdev.yaml",
)
JOB_SCOPE = {
    "train_folds": 6,
    "eval_folds": 6,
    "extract_folds": 6,
    "head_folds": 6,
    "principal_jobs": 24,
}


def _path_strings(value: Any, *, key: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            found.extend(_path_strings(child_value, key=str(child_key)))
    elif isinstance(value, list):
        for item in value:
            found.extend(_path_strings(item, key=key))
    elif isinstance(value, str) and (key.endswith("path") or key.endswith("paths")):
        found.append(value)
    return found


def prepare(*, run_id: str, build: bool, required_path_prefix: Path | None) -> dict[str, Any]:
    config_path = resolve_project_path(MANIFEST_BUILD_CONFIG)
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    manifest_dir = resolve_project_path(config["output_dirs"]["manifest_dir"])

    if build:
        print(f"Building MN5 DAIC manifest from {config_path}", flush=True)
        build_for_config(config_path)

    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing manifest metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("build_signature") != manifest_build_signature(config):
        raise ValueError(f"Stale build signature: {metadata_path}")
    manifest_path = resolve_project_path(metadata["manifest_path"])
    rows = read_jsonl(manifest_path)
    if int(metadata.get("manifest_row_count", -1)) != len(rows):
        raise ValueError(f"Manifest row-count mismatch: {manifest_path}")

    prefix = required_path_prefix.resolve() if required_path_prefix else None
    checked_paths: set[str] = set()
    for row in rows:
        for text in _path_strings(row):
            if not text or text.startswith("${PROJECT_ROOT}"):
                continue
            path = Path(text)
            if not path.is_absolute():
                continue
            resolved = path.resolve()
            if prefix is not None and resolved != prefix and prefix not in resolved.parents:
                raise ValueError(f"manifest contains a non-MN5 dataset path: {resolved}")
            if not resolved.exists():
                raise FileNotFoundError(f"manifest path is missing: {resolved}")
            checked_paths.add(str(resolved))
    if not checked_paths:
        raise ValueError(f"No source paths were verified: {manifest_path}")

    # The locked 86/21/35 contract and row counts against the rebuilt manifest.
    split_audit = audit_split(manifest_dir, split_dir, resolve_project_path("configs/main"))

    return {
        "status": "passed",
        "run_id": run_id,
        "source_commit": _provenance_text("git_commit.txt"),
        "source_branch": _provenance_text("git_branch.txt"),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset": dataset,
        "metadata_path": str(metadata_path),
        "split_metadata_sha256": sha256_file(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_hash": metadata.get("manifest_hash"),
        "rows": len(rows),
        "verified_source_paths": len(checked_paths),
        "configs": list(ALL_SIX_CONFIGS),
        "split_audit": split_audit,
        "job_scope": JOB_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _provenance_text(name: str) -> str:
    path = PROJECT_ROOT / ".provenance" / name
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--build", action="store_true", help="Rebuild the manifest before auditing.")
    parser.add_argument("--required-path-prefix", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        record = prepare(
            run_id=args.run_id,
            build=args.build,
            required_path_prefix=args.required_path_prefix,
        )
    except (AuditFailure, ValueError, FileNotFoundError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1

    output = args.output or (
        PROJECT_ROOT / "outputs/daic_officialdev_mn5_preflight" / args.run_id / "audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(record, output)
    print(f"preflight passed: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
