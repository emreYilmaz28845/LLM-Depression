#!/usr/bin/env python3
"""Small CLI used by v2 Slurm workers to update head sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.native_en_text_heads_tracking import (
    finish_head_attempt,
    initialize_head_attempt_batch,
    initialize_head_attempt,
    materialize_head_evidence,
    materialize_job_evidence,
    record_head_job,
    transition_head_attempt,
    validate_head_attempt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--attempt-dir", required=True, type=Path)
    init.add_argument("--context", required=True, type=Path)
    init.add_argument("--config", required=True, type=Path)
    init.add_argument("--parent", required=True, type=Path)

    init_batch = sub.add_parser("init-batch")
    init_batch.add_argument("--manifest", required=True, type=Path)

    record = sub.add_parser("record")
    record.add_argument("--attempt-dir", required=True, type=Path)
    record.add_argument("--job-key", required=True)
    record.add_argument("--job-type", required=True)
    record.add_argument("--event-type", required=True)
    record.add_argument("--slurm-job-id")
    record.add_argument("--status")
    record.add_argument("--reason")
    record.add_argument("--dependency-job-id", action="append", default=[])
    record.add_argument("--resubmission-of-job-id")
    record.add_argument("--exit-code")

    transition = sub.add_parser("transition")
    transition.add_argument("--attempt-dir", required=True, type=Path)
    transition.add_argument("--to-state", required=True)
    transition.add_argument("--reason", required=True)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--attempt-dir", required=True, type=Path)
    materialize.add_argument("--predictions", required=True, type=Path)
    materialize.add_argument("--metrics", required=True, type=Path)
    materialize.add_argument("--checkpoint-path", required=True)

    job_materialize = sub.add_parser("job-materialize")
    job_materialize.add_argument("--attempt-dir", required=True, type=Path)
    job_materialize.add_argument("--artifact", action="append", default=[])

    for name, function in (("validate", validate_head_attempt), ("finish", finish_head_attempt)):
        command = sub.add_parser(name)
        command.add_argument("--attempt-dir", required=True, type=Path)
        command.set_defaults(function=function)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = initialize_head_attempt(
            args.attempt_dir,
            context=json.loads(args.context.read_text(encoding="utf-8")),
            config=json.loads(args.config.read_text(encoding="utf-8")),
            parent=json.loads(args.parent.read_text(encoding="utf-8")),
        )
    elif args.command == "init-batch":
        result = initialize_head_attempt_batch(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
    elif args.command == "record":
        result = record_head_job(
            args.attempt_dir,
            job_key=args.job_key,
            job_type=args.job_type,
            event_type=args.event_type,
            slurm_job_id=args.slurm_job_id,
            status=args.status,
            reason=args.reason,
            dependency_job_ids=args.dependency_job_id,
            resubmission_of_job_id=args.resubmission_of_job_id,
            exit_code=args.exit_code,
        )
    elif args.command == "transition":
        result = transition_head_attempt(args.attempt_dir, args.to_state, reason=args.reason)
    elif args.command == "materialize":
        result = materialize_head_evidence(
            args.attempt_dir,
            predictions_path=args.predictions,
            metrics_path=args.metrics,
            checkpoint_path=args.checkpoint_path,
        )
    elif args.command == "job-materialize":
        result = materialize_job_evidence(
            args.attempt_dir,
            artifact_paths=args.artifact,
        )
    else:
        result = args.function(args.attempt_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
