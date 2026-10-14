#!/usr/bin/env python3
"""Build the corrected merged protocol artifacts for the Worker 1 campaign.

Mirrors the submitter's two-step override resolution so each protocol is built
with exactly the resolved component inputs and output roots production will use:
relative component manifest/metadata paths are repointed at ``--input-root`` and
the pooled Turkish component at ``--pooled-runtime-root``.  The corrected
protocol loader then requires the exact official Androids outer folds (fail
closed) and preserves the other components' memberships.

Run this from the immutable deployment code before submitting smoke/CV/final.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.submit_symmetric_merged import (  # noqa: E402
    _runtime_override_tokens,
    load_merged_config,
)
from src.merged.protocol import (  # noqa: E402
    load_component_records,
    resolve_protocol_split_seed,
    save_protocol_artifacts,
)

ROUTES = {
    "native_text_only": ("native", "text_only"),
    "native_audio_only": ("native", "audio_only"),
    "native_audio_text": ("native", "audio_text"),
    "english_text_only": ("english", "text_only"),
    "english_audio_text": ("english", "audio_text"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pooled-runtime-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-base", default="qwen3_androids_official_folds_20261008")
    parser.add_argument("--routes", nargs="*", default=list(ROUTES))
    args = parser.parse_args()

    summaries = []
    failures = []
    for route in args.routes:
        if route not in ROUTES:
            failures.append(f"unknown route {route}")
            continue
        campaign_suffix, modality = ROUTES[route]
        config_path = REPO_ROOT / f"configs/experiments/merged/symmetric_merged_qwen3_pooled_{route}.yaml"
        declared = load_merged_config(config_path, [])
        merged_root = Path(args.output_root) / f"{args.campaign_base}_{campaign_suffix}" / modality
        tokens = [f"--set=output_dirs.merged_root={merged_root}"]
        tokens += _runtime_override_tokens(
            declared,
            input_root=args.input_root,
            pooled_runtime_root=args.pooled_runtime_root,
        )
        config = load_merged_config(config_path, tokens)
        records = load_component_records(config, require_files=True)
        payload = save_protocol_artifacts(
            config,
            records,
            merged_root,
            seed=resolve_protocol_split_seed(config),
            inner_val_ratio=float(
                config.get("protocol_settings", {}).get("inner_val_ratio", 0.2)
            ),
        )
        androids = payload["protocol"]["components"]["androids_interview"]
        summary = {
            "route": route,
            "merged_root": str(merged_root),
            "split_hash": payload["protocol"]["split_hash"],
            "split_audit_status": payload["split_audit"]["status"],
            "split_audit_failures": payload["split_audit"]["failures"],
            "androids_folds_path": androids.get("folds_path"),
            "androids_sources": sorted(
                {payload["protocol"]["components"]["androids_interview"]["folds"][str(f)]["source"] for f in range(5)}
            ),
            "component_folds_paths": {
                dataset: payload["protocol"]["components"][dataset].get("folds_path")
                for dataset in payload["protocol"]["components"]
            },
        }
        summaries.append(summary)
        if payload["split_audit"]["status"] != "passed":
            failures.append(f"{route}: split audit failed: {payload['split_audit']['failures']}")
        if summary["androids_sources"] != ["component_official_folds"]:
            failures.append(f"{route}: androids did not load official folds: {summary['androids_sources']}")
        print(json.dumps(summary, indent=2), flush=True)

    print(json.dumps({"status": "passed" if not failures else "failed", "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
