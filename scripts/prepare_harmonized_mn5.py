#!/usr/bin/env python3
"""Build and verify MN5-native harmonized manifests and merged protocols."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.experiment_tracking.manifest_policy import (
    MANIFEST_POLICY_PREBUILT,
    prebuilt_manifest_files,
    validate_manifest_policy,
)
from src.merged.protocol import load_component_records, save_protocol_artifacts
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)


COMPONENT_CONFIGS = (
    "configs/main/d3tec_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
    "configs/main/androids_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "configs/main/daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "configs/main/cmdc_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
)
# The four Qwen3 pooled merged contracts. Their GPU execution is blocked until the
# Qwen3 merged FSDP/postprocess support task lands; CPU preparation (protocol
# artifacts) is allowed and needs the pooled manifests from the task runtime.
MERGED_CONFIGS = (
    "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_audio_text.yaml",
    "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_audio_only.yaml",
    "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_text_only.yaml",
    "configs/experiments/merged/symmetric_merged_qwen3_pooled_english_text_only.yaml",
)
POOLED_COMPONENT_NAME = "turkish"


def _path_strings(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _path_strings(child_value, key=str(child_key))
    elif isinstance(value, list):
        if key.endswith("paths"):
            for item in value:
                if isinstance(item, str):
                    yield item
        else:
            for item in value:
                yield from _path_strings(item, key=key)
    elif isinstance(value, str) and key.endswith("path"):
        yield value


def validate_component(
    config_path: Path, *, required_path_prefix: Path | None
) -> dict[str, Any]:
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing harmonized metadata for {dataset}: {metadata_path}")
    metadata = read_json(metadata_path)
    policy = validate_manifest_policy(config)
    if policy == MANIFEST_POLICY_PREBUILT:
        # The pooled manifest is built outside the worker by
        # scripts/build_turkish_pooled_manifest.py, so its build signature
        # legitimately differs; verify the four prepared files instead.
        files = prebuilt_manifest_files(
            manifest_dir=str(resolve_project_path(config["output_dirs"]["manifest_dir"])),
            split_dir=str(split_dir),
            dataset=dataset,
        )
        missing_files = [path for path in files.values() if not Path(path).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Prebuilt manifest files are missing for {dataset}: {missing_files}"
            )
    elif metadata.get("build_signature") != manifest_build_signature(config):
        raise ValueError(f"Stale build signature for {dataset}: {metadata_path}")
    manifest_path = resolve_project_path(metadata["manifest_path"])
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Empty harmonized manifest for {dataset}: {manifest_path}")
    if int(metadata.get("manifest_row_count", -1)) != len(rows):
        raise ValueError(f"Manifest row-count mismatch for {dataset}: {manifest_path}")

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
                raise ValueError(
                    f"{dataset} manifest contains a non-MN5 dataset path: {resolved}"
                )
            if not resolved.exists():
                raise FileNotFoundError(f"{dataset} manifest path is missing: {resolved}")
            checked_paths.add(str(resolved))
    if not checked_paths:
        raise ValueError(f"No source paths were verified for {dataset}: {manifest_path}")
    return {
        "dataset": dataset,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_policy": policy,
        "metadata_path": str(metadata_path),
        "split_metadata_sha256": sha256_file(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_hash": metadata.get("manifest_hash"),
        "rows": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "verified_source_paths": len(checked_paths),
    }


def apply_pooled_runtime_paths(config: dict[str, Any], pooled_runtime_root: Path) -> dict[str, Any]:
    """Point the pooled Turkish component at the task runtime's built manifests.

    The pooled native/English manifests are built by the pooled preparation job
    into the task runtime, so the merged protocol prep reads them from there
    instead of the checkout's ``outputs/`` tree.
    """
    for component in config.get("components") or []:
        if str(component.get("name")) != POOLED_COMPONENT_NAME:
            continue
        english = "harmonized_en" in str(component.get("manifest_path", ""))
        manifest_root = pooled_runtime_root / ("manifests_en" if english else "manifests")
        split_root = pooled_runtime_root / ("splits_en" if english else "splits")
        component["manifest_path"] = str(manifest_root / "turkish" / "turkish_manifest.jsonl")
        component["metadata_path"] = str(split_root / "turkish" / "turkish_manifest_metadata.json")
    return config


def prepare(
    *, run_id: str, build: bool, required_path_prefix: Path | None,
    build_merged: bool = True, pooled_runtime_root: Path | None = None,
) -> dict[str, Any]:
    component_paths = [resolve_project_path(path) for path in COMPONENT_CONFIGS]
    if build:
        for config_path in component_paths:
            config = load_yaml_with_overrides(config_path, [])
            if validate_manifest_policy(config) == MANIFEST_POLICY_PREBUILT:
                print(f"Skipping build for prebuilt component: {config_path}", flush=True)
                continue
            print(f"Building MN5 harmonized component: {config_path}", flush=True)
            build_for_config(config_path, [])
    components = [
        validate_component(path, required_path_prefix=required_path_prefix)
        for path in component_paths
    ]

    merged: list[dict[str, Any]] = []
    if build_merged:
        for raw_path in MERGED_CONFIGS:
            config_path = resolve_project_path(raw_path)
            config = load_yaml_with_overrides(config_path, [])
            if pooled_runtime_root is not None:
                config = apply_pooled_runtime_paths(config, Path(pooled_runtime_root))
            records = load_component_records(config, require_files=True)
            output_dir = resolve_project_path(config["output_dirs"]["merged_root"])
            payload = save_protocol_artifacts(
                config,
                records,
                output_dir,
                seed=int(config.get("seed", 1337)),
                inner_val_ratio=float(config["protocol_settings"]["inner_val_ratio"]),
            )
            if payload["split_audit"].get("status") != "passed":
                raise ValueError(f"Merged protocol audit failed for {config_path}")
            merged.append(
                {
                    "modality": config["modality"],
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "status": config.get("status"),
                    "blocked_prerequisite": config.get("status") == "blocked_prerequisite",
                    "manifest_path": payload["manifest_path"],
                    "manifest_file_sha256": payload["manifest_file_sha256"],
                    "manifest_hash": payload["manifest"]["manifest_hash"],
                    "split_hash": payload["protocol"]["split_hash"],
                    "artifact_hash": payload["artifact_hash"],
                }
            )

    return {
        "schema_version": "harmonized_mn5_preflight.v1",
        "status": "passed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("HARMONIZED_SOURCE_COMMIT"),
        "source_branch": os.environ.get("HARMONIZED_SOURCE_BRANCH"),
        "required_path_prefix": str(required_path_prefix) if required_path_prefix else None,
        "pooled_runtime_root": str(pooled_runtime_root) if pooled_runtime_root else None,
        "components": components,
        "merged": merged,
        "optuna_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-merged", action="store_true")
    parser.add_argument("--required-path-prefix", type=Path)
    parser.add_argument("--pooled-runtime-root", type=Path,
                        help="task runtime root holding the built pooled manifests/splits")
    parser.add_argument("--audit-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = prepare(
        run_id=args.run_id,
        build=not args.validate_only,
        required_path_prefix=args.required_path_prefix,
        build_merged=not args.skip_merged,
        pooled_runtime_root=args.pooled_runtime_root,
    )
    audit_path = resolve_project_path(
        args.audit_path
        or PROJECT_ROOT / "outputs/harmonized_mn5_preflight" / args.run_id / "audit.json"
    )
    save_json(audit, audit_path)
    print(json.dumps({"status": "passed", "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
