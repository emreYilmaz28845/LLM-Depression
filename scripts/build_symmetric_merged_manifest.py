#!/usr/bin/env python3
"""Build component manifests and the namespaced symmetric merged protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config
from src.merged.protocol import (
    load_component_records,
    resolve_protocol_split_seed,
    save_protocol_artifacts,
)
from src.utils import configure_logging, load_yaml_with_overrides, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--build-components",
        action="store_true",
        help="Run the existing component manifest builders before merging.",
    )
    parser.add_argument(
        "--skip-existing-components",
        action="store_true",
        help="Reuse complete existing component artifacts and fail closed on partial artifacts.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inner-val-ratio", type=float)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Lossless --set=key=value token; repeated for managed runtime paths.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml_with_overrides(config_path, list(args.override))
    if str(config.get("protocol", "")) != "symmetric_merged":
        raise ValueError("The supplied config is not protocol=symmetric_merged.")
    if args.build_components:
        for index, item in enumerate(config["components"]):
            component_path = resolve_project_path(item["config"])
            print(f"Building component manifest: {component_path}", flush=True)
            manifest_path = resolve_project_path(item["manifest_path"])
            metadata_path = resolve_project_path(item["metadata_path"])
            manifest_exists = manifest_path.is_file()
            metadata_exists = metadata_path.is_file()
            if args.skip_existing_components and (manifest_exists or metadata_exists):
                if not (manifest_exists and metadata_exists):
                    raise FileExistsError(
                        f"partial existing component artifacts: {manifest_path} / {metadata_path}"
                    )
                print(f"Reusing existing component artifacts: {manifest_path}", flush=True)
                continue
            component_overrides = [
                f"--set=output_dirs.manifest_dir={manifest_path.parent}",
                f"--set=output_dirs.split_dir={metadata_path.parent}",
            ]
            build_for_config(component_path, component_overrides)
    records = load_component_records(config, require_files=True)
    output_dir = args.output_dir or config["output_dirs"]["merged_root"]
    output_dir = resolve_project_path(output_dir)
    payload = save_protocol_artifacts(
        config,
        records,
        output_dir,
        seed=int(args.seed if args.seed is not None else resolve_protocol_split_seed(config)),
        inner_val_ratio=float(
            args.inner_val_ratio
            if args.inner_val_ratio is not None
            else config.get("protocol_settings", {}).get("inner_val_ratio", 0.2)
        ),
    )
    print(json.dumps(
        {
            "status": "complete",
            "output_dir": str(output_dir),
            "manifest_path": payload["manifest_path"],
            "manifest_hash": payload["manifest"]["manifest_hash"],
            "split_hash": payload["protocol"]["split_hash"],
            "split_audit": payload["split_audit"],
        },
        indent=2,
    ), flush=True)


if __name__ == "__main__":
    main()
