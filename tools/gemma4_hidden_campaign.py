from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import gemma4_hidden_campaign as campaign  # noqa: E402


def _print_result(result: dict) -> None:
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc Gemma 4 DAIC fixed-head campaign CLI (thin wrapper)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-attempt", help="Create a new fixed-head attempt destination.")
    create.add_argument("--repo-root", default=str(PROJECT_ROOT), type=Path)
    create.add_argument("--attempt-dir", required=True, type=Path)
    create.add_argument("--modality", required=True, choices=("audio_text", "audio_only", "text_only"))
    create.add_argument("--run-name", required=True)
    create.add_argument("--group-id", required=True)
    create.add_argument("--parent-fold-dir", required=True, type=Path)
    create.add_argument("--parent-attempt-id", required=True)
    create.add_argument("--merged-sha", required=True, help="Merged implementation SHA on main.")
    create.add_argument("--branch", default="main")
    create.add_argument("--pr-number", type=int)
    create.add_argument("--fold", type=int, default=0)

    deployed = sub.add_parser("mark-deployed", help="Transition PLANNED -> DEPLOYED.")
    deployed.add_argument("--attempt-dir", required=True, type=Path)
    deployed.add_argument("--reason")

    record = sub.add_parser("record-job", help="Append one job event.")
    record.add_argument("--attempt-dir", required=True, type=Path)
    record.add_argument("--job-key", required=True)
    record.add_argument("--job-type", required=True)
    record.add_argument("--event-type", required=True)
    record.add_argument("--slurm-job-id")
    record.add_argument("--dependency-job-ids", nargs="*", default=[])
    record.add_argument("--status")
    record.add_argument("--reason")

    trans = sub.add_parser("transition", help="Enforce a lifecycle transition.")
    trans.add_argument("--attempt-dir", required=True, type=Path)
    trans.add_argument("--to-state", required=True)
    trans.add_argument("--reason")

    materialize = sub.add_parser(
        "materialize-mn5-evidence",
        help="On MN5 after head completion: hash artifacts, create evaluation records, "
        "transition to COMPLETED_ON_MN5.",
    )
    materialize.add_argument("--attempt-dir", required=True, type=Path)
    materialize.add_argument("--parent-fold-dir", required=True, type=Path)

    verify = sub.add_parser(
        "verify-local",
        help="Verify local hashes and recomputed metrics, then transition to REPORTABLE.",
    )
    verify.add_argument("--attempt-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create-attempt":
        result = campaign.create_attempt(
            repo_root=args.repo_root,
            attempt_dir=args.attempt_dir,
            modality=args.modality,
            run_name=args.run_name,
            group_id=args.group_id,
            parent_fold_dir=args.parent_fold_dir,
            parent_attempt_id=args.parent_attempt_id,
            merged_sha=args.merged_sha,
            branch=args.branch,
            pr_number=args.pr_number,
            fold=args.fold,
        )
    elif args.command == "mark-deployed":
        result = campaign.mark_deployed(args.attempt_dir, reason=args.reason)
    elif args.command == "record-job":
        result = campaign.record_job(
            args.attempt_dir,
            job_key=args.job_key,
            job_type=args.job_type,
            event_type=args.event_type,
            slurm_job_id=args.slurm_job_id,
            dependency_job_ids=args.dependency_job_ids,
            status=args.status,
            reason=args.reason,
        )
    elif args.command == "transition":
        result = campaign.transition(args.attempt_dir, args.to_state, reason=args.reason)
    elif args.command == "materialize-mn5-evidence":
        result = campaign.materialize_mn5_evidence(
            args.attempt_dir, args.parent_fold_dir
        )
    elif args.command == "verify-local":
        result = campaign.verify_local(args.attempt_dir)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    _print_result(result)


if __name__ == "__main__":
    main()
