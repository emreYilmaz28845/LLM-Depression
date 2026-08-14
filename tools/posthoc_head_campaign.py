#!/usr/bin/env python3
"""Thin CLI for the general post-hoc head-attempt workflow.

Subcommands: create-attempt, mark-deployed, record-job, transition,
materialize-mn5-evidence, verify-local. All business logic lives in
src.features.posthoc_head_campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import posthoc_head_campaign as campaign  # noqa: E402


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "create-attempt",
            "mark-deployed",
            "record-job",
            "transition",
            "materialize-mn5-evidence",
            "verify-local",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--to-state", choices=("SUBMITTED", "RUNNING", "COMPLETED_ON_MN5", "FAILED", "CANCELLED", "SUPERSEDED"))
    parser.add_argument("--reason", default=None)
    parser.add_argument("--job-key", default="optuna")
    parser.add_argument("--job-type", default="hidden_classifier")
    parser.add_argument("--event-type", choices=("SUBMITTED", "STARTED", "COMPLETED", "FAILED", "CANCELLED"))
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--job-status")
    parser.add_argument("--dependency-job-ids", default=None)
    parser.add_argument("--resubmission-of-job-id", default=None)
    args = parser.parse_args()

    try:
        if args.command == "create-attempt":
            if args.task_spec is None:
                raise SystemExit("create-attempt requires --task-spec")
            payload = campaign.create_attempt(
                repo_root=args.repo_root,
                attempt_dir=args.attempt_dir,
                task_spec=args.task_spec,
            )
        elif args.command == "mark-deployed":
            payload = campaign.mark_deployed(args.attempt_dir, reason=args.reason)
        elif args.command == "record-job":
            dependency_job_ids = (
                args.dependency_job_ids.split(",") if args.dependency_job_ids else None
            )
            payload = campaign.record_job(
                args.attempt_dir,
                job_key=args.job_key,
                job_type=args.job_type,
                event_type=args.event_type,
                slurm_job_id=args.slurm_job_id,
                status=args.job_status,
                reason=args.reason,
                dependency_job_ids=dependency_job_ids,
                resubmission_of_job_id=args.resubmission_of_job_id,
            )
        elif args.command == "transition":
            if args.to_state is None:
                raise SystemExit("transition requires --to-state")
            payload = campaign.transition(args.attempt_dir, args.to_state, reason=args.reason)
        elif args.command == "materialize-mn5-evidence":
            payload = campaign.materialize_mn5_evidence(args.attempt_dir)
        elif args.command == "verify-local":
            payload = campaign.verify_local(args.attempt_dir)
        else:  # pragma: no cover
            raise SystemExit(f"unsupported command {args.command}")
    except (campaign.PosthocError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
