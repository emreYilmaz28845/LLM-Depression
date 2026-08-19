#!/usr/bin/env python3
"""Qwen3.8 audit CLI: deployment and turkish.

Subcommands:

- ``deployment``: verify the environment record, model/wheelhouse manifests,
  driver probe, per-TP acceptance, and serving selection against the pinned
  contract.
- ``turkish``: run the runbook section 21 audit. Restricted-intermediate
  checks run when the restricted evidence is present (GPFS); the local
  re-run covers the compact outputs, transcript hash, and privacy checks.

The audit writes ``audit.json`` atomically and exits non-zero on any failed
check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen38.audit import (
    audit_deployment,
    audit_turkish,
    audit_wheelhouse,
    write_audit_json,
)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cmd_deployment(args: argparse.Namespace) -> int:
    result = audit_deployment(
        args.deploy_dir,
        args.deployment_id,
        model_dir=args.model_dir,
        wheelhouse_dir=args.wheelhouse_dir,
        environment_dir=args.environment_dir,
        source_commit=args.source_commit,
        selection_file=args.selection_file,
    )
    write_audit_json(result, Path(args.out))
    print(f"deployment audit passed={result['passed']} checks={len(result['checks'])}")
    for check in result["checks"]:
        if not check["passed"]:
            print(f"  failed: {check['check_id']}: {check['description']}", file=sys.stderr)
    return 0 if result["passed"] else 1


def cmd_turkish(args: argparse.Namespace) -> int:
    slurm_metadata = None
    if args.slurm_metadata:
        if Path(args.slurm_metadata).is_file():
            slurm_metadata = _read_json(args.slurm_metadata)
        else:
            print(f"slurm metadata file missing: {args.slurm_metadata}", file=sys.stderr)
            return 1
    result = audit_turkish(
        args.run_dir,
        turkish_run_id=args.turkish_run_id,
        transcript_path=args.transcript,
        deploy_dir=args.deploy_dir,
        deployment_id=args.deployment_id,
        model_dir=args.model_dir,
        wheelhouse_dir=args.wheelhouse_dir,
        source_commit=args.source_commit,
        selection_file=args.selection_file,
        slurm_metadata=slurm_metadata,
        remote_reference=args.remote_reference,
        remote_audit_sha256_path=args.remote_audit_sha256,
    )
    write_audit_json(result, Path(args.out))
    print(f"turkish audit passed={result['passed']} checks={len(result['checks'])}")
    for check in result["checks"]:
        if check["passed"] is False:
            print(f"  failed: {check['check_id']}: {check['description']}", file=sys.stderr)
    return 0 if result["passed"] else 1


def cmd_wheelhouse(args: argparse.Namespace) -> int:
    result = audit_wheelhouse(args.wheelhouse_dir)
    write_audit_json(result, Path(args.out))
    print(f"wheelhouse audit passed={result['passed']} wheels={result['wheel_count']}")
    for error in result["errors"]:
        print(f"  {error}", file=sys.stderr)
    return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.8 audits (deployment / turkish / wheelhouse)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deployment = subparsers.add_parser("deployment", help="audit the deployment records")
    deployment.add_argument("--deploy-dir", required=True, help="REMOTE_DEPLOY root")
    deployment.add_argument("--deployment-id", required=True)
    deployment.add_argument("--model-dir", required=True)
    deployment.add_argument("--wheelhouse-dir", required=True)
    deployment.add_argument("--environment-dir", default=None)
    deployment.add_argument("--source-commit", default=None)
    deployment.add_argument("--selection-file", default=None)
    deployment.add_argument("--out", required=True)
    deployment.set_defaults(func=cmd_deployment)

    turkish = subparsers.add_parser("turkish", help="audit the Turkish compact outputs")
    turkish.add_argument("--run-dir", required=True)
    turkish.add_argument("--turkish-run-id", required=True)
    turkish.add_argument("--transcript", required=True)
    turkish.add_argument("--deploy-dir", required=True)
    turkish.add_argument("--deployment-id", required=True)
    turkish.add_argument("--model-dir", required=True)
    turkish.add_argument("--wheelhouse-dir", required=True)
    turkish.add_argument("--source-commit", required=True)
    turkish.add_argument("--selection-file", required=True)
    turkish.add_argument("--slurm-metadata", default=None, help="JSON with job/state/exit/node/timestamps")
    turkish.add_argument("--remote-reference", default=None, help="remote audit.json to reference")
    turkish.add_argument("--remote-audit-sha256", default=None, help="audit.json SHA-256 sidecar")
    turkish.add_argument("--out", required=True)
    turkish.set_defaults(func=cmd_turkish)

    wheelhouse = subparsers.add_parser("wheelhouse", help="audit wheel tags")
    wheelhouse.add_argument("--wheelhouse-dir", required=True)
    wheelhouse.add_argument("--out", required=True)
    wheelhouse.set_defaults(func=cmd_wheelhouse)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
