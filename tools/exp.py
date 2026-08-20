from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import registry

PROTECTED_PATHS = [
    Path("/home/emre/Projects/AudioLLM/Teacher-System"),
    Path("/home/emre/Projects/AudioLLM/LLM-Depression-teacher"),
]

def _print_rows(rows, columns: tuple[str, ...]) -> None:
    print("\t".join(columns))
    for row in rows:
        print("\t".join(_cell(row[column] if column in row.keys() else None) for column in columns))


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _cmd_list(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.list_runs(connection, dataset=args.dataset, status=args.status)
    finally:
        connection.close()
    _print_rows(
        rows,
        (
            "attempt_id",
            "logical_run_name",
            "dataset",
            "modality",
            "fold",
            "current_state",
            "backend",
            "evaluation_view",
            "aggregation",
            "metric_namespace",
            "headline_positive_f1",
        ),
    )
    return 0


def _cmd_show(args) -> int:
    connection = registry.connect(args.db)
    try:
        payload = registry.show_attempt(connection, args.attempt_id, fold=args.fold)
    finally:
        connection.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_provenance(args) -> int:
    connection = registry.connect(args.db)
    try:
        payload = registry.provenance_of_metric(connection, args.metric_id)
    finally:
        connection.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_jobs(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.list_jobs(connection, failed_only=args.failed)
    finally:
        connection.close()
    _print_rows(
        rows,
        (
            "at_utc",
            "attempt_id",
            "logical_run_name",
            "fold",
            "job_key",
            "job_type",
            "event_type",
            "slurm_job_id",
            "status",
            "resubmission_of_job_id",
        ),
    )
    return 0


def _cmd_best(args) -> int:
    connection = registry.connect(args.db)
    try:
        rows = registry.best_runs(
            connection,
            dataset=args.dataset,
            metric=args.metric,
            namespace=args.namespace,
            backend=args.backend,
            view=args.view,
            aggregation=args.aggregation,
            limit=args.limit,
        )
    finally:
        connection.close()
    if not rows:
        print("no matches for the fully qualified query")
        return 0
    _print_rows(
        rows,
        (
            "attempt_id",
            "logical_run_name",
            "fold",
            "metric_value",
            "support",
            "backend",
            "evaluation_view",
            "aggregation",
            "namespace",
            "split_name",
            "checkpoint_role",
            "evidence_artifact_id",
        ),
    )
    return 0

# --- New create command for parallel workflow ---

def _run_git(args, cwd=None):
    result = subprocess.run(["git"] + args, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    return result

def _get_git_branch_sha(ref, cwd=None):
    # ref can be branch or sha
    result = _run_git(["rev-parse", "--verify", ref], cwd=cwd)
    if result.returncode == 0:
        return result.stdout.strip()
    # Try rev-parse directly (maybe sha prefix)
    result2 = _run_git(["rev-parse", ref], cwd=cwd)
    if result2.returncode == 0:
        return result2.stdout.strip()
    return None

def _cmd_create(args) -> int:
    slug = args.slug
    tier = args.tier
    from_ref = args.from_ref
    dry_run = args.dry_run
    # Validate tier
    if tier not in (0,1,2):
        print(f"ERROR: tier must be 0,1,2 got {tier}", file=sys.stderr)
        return 1
    # Determine branch name
    # Slug handling: if slug already starts with exp- or feat-, use as is; else add prefix based on tier
    if slug.startswith("exp-") or slug.startswith("feat-"):
        branch_suffix = slug
        # Determine expected prefix based on tier
        expected_prefix = "exp-" if tier == 1 else "feat-" if tier == 2 else None
        # Allow but warn if mismatch? For tier 0, we may allow either
        if tier != 0 and expected_prefix and not slug.startswith(expected_prefix):
            # Allow but note: e.g., feat slug with tier1 should maybe be exp, but we allow
            pass
        branch = f"agent/{slug}"
    else:
        prefix = "exp" if tier == 1 else "feat" if tier == 2 else "exp"
        if tier == 0:
            # For tier 0, we still create branch? Spec says Tier0 stays on main if not editing, but if user requests create with tier0 and slug, we treat as tier1? Let's just allow.
            prefix = "exp"
        branch = f"agent/{prefix}-{slug}"
        branch_suffix = f"{prefix}-{slug}"

    # Worktree path: ~/worktrees/LLM-Depression-<branch_suffix>
    # Use branch_suffix as after agent/
    worktree_name = f"LLM-Depression-{branch_suffix}"
    worktree_path = Path.home() / "worktrees" / worktree_name
    # Resolve canonical
    worktree_path_resolved = worktree_path.resolve()

    # Check protected paths
    for pp in PROTECTED_PATHS:
        pp_res = pp.resolve()
        try:
            if worktree_path_resolved == pp_res or pp_res in worktree_path_resolved.parents:
                print(f"ERROR: worktree path {worktree_path_resolved} is inside protected path {pp_res}", file=sys.stderr)
                return 1
            if pp_res == worktree_path_resolved:
                print(f"ERROR: worktree path equals protected path {pp_res}", file=sys.stderr)
                return 1
        except Exception:
            pass
        # Also check string prefix without resolve (in case path doesn't exist yet)
        if str(worktree_path_resolved).startswith(str(pp_res) + "/"):
            print(f"ERROR: worktree path {worktree_path_resolved} inside protected {pp_res}", file=sys.stderr)
            return 1

    # Determine parent ref
    project_root = PROJECT_ROOT.resolve()
    # Find git top for current worktree
    # Use from_ref if provided
    parent_ref = from_ref if from_ref else "HEAD"
    # If from_ref is branch or sha, resolve
    parent_sha = _get_git_branch_sha(parent_ref, cwd=project_root)
    if parent_sha is None:
        # Try origin/main
        parent_sha = _get_git_branch_sha("origin/main", cwd=project_root)
        if parent_sha is None:
            parent_sha = _get_git_branch_sha("HEAD", cwd=project_root)
    if parent_sha is None:
        print(f"ERROR: could not resolve parent ref {parent_ref}", file=sys.stderr)
        return 1
    # Determine parent branch name for metadata
    parent_branch = from_ref if from_ref else None
    if parent_branch:
        # Check if it's a sha (40 hex) then not branch
        if len(parent_branch) == 40 and all(c in "0123456789abcdef" for c in parent_branch.lower()):
            parent_branch = None  # sha not branch
        else:
            # Verify branch exists, else treat as sha
            check = _run_git(["show-ref", "--verify", f"refs/heads/{parent_branch}"], cwd=project_root)
            if check.returncode != 0:
                # Might be origin/main
                check2 = _run_git(["show-ref", "--verify", f"refs/remotes/{parent_branch}"], cwd=project_root)
                if check2.returncode != 0 and parent_branch != "HEAD":
                    # Could be sha, ignore
                    pass

    # Determine experiment_id
    # Use slug + date, e.g., exp-rotary-20260821
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    # If slug already contains date, use as is
    if date_str in slug:
        experiment_id = slug
    else:
        # Use branch_suffix + date
        experiment_id = f"{branch_suffix}-{date_str}"

    # Definition file path
    definitions_dir = project_root / "experiments" / "definitions"
    definition_path = definitions_dir / f"{experiment_id}.yaml"
    # Alternative: use slug as file name? Spec says tracked definitions under experiments/definitions/
    # Use experiment_id as file name, but ensure unique
    # Check collisions
    errors = []
    # Check branch exists
    branch_exists = _run_git(["show-ref", "--verify", f"refs/heads/{branch}"], cwd=project_root).returncode == 0
    if branch_exists:
        errors.append(f"branch {branch} already exists")
    if worktree_path.exists():
        errors.append(f"worktree path {worktree_path} already exists")
    if definition_path.exists():
        errors.append(f"definition file {definition_path} already exists")

    # Also check for any worktree already registered with that path via git worktree list
    wt_list = _run_git(["worktree", "list", "--porcelain"], cwd=project_root)
    if wt_list.returncode == 0 and str(worktree_path) in wt_list.stdout:
        errors.append(f"worktree {worktree_path} already registered")

    # Print dry-run
    print("=== exp create dry-run ===")
    print(f"slug: {slug}")
    print(f"tier: {tier}")
    print(f"branch: {branch}")
    print(f"worktree: {worktree_path}")
    print(f"experiment_id: {experiment_id}")
    print(f"parent ref: {parent_ref} -> {parent_sha[:8] if parent_sha else 'unknown'}")
    if parent_branch:
        print(f"parent branch: {parent_branch}")
    print(f"definition: {definition_path}")
    print(f"pin: {worktree_path}/.agent-pin.json")
    print(f"allowed_paths: [{worktree_path}]")
    print(f"protected_paths: {PROTECTED_PATHS}")
    if branch_suffix.startswith("exp-"):
        print("merge strategy: squash if selected")
    elif branch_suffix.startswith("feat-"):
        print("merge strategy: merge commit when history helps")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if dry_run:
        print("dry-run: no mutation performed")
        return 0

    # Execute creation
    # Create branch
    print(f"creating branch {branch} from {parent_sha[:8]}")
    res = _run_git(["branch", branch, parent_sha], cwd=project_root)
    if res.returncode != 0:
        print(f"ERROR: failed to create branch {branch}: {res.stderr}", file=sys.stderr)
        return 1

    # Create worktree
    print(f"creating worktree {worktree_path} for branch {branch}")
    # Ensure parent dir exists
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    res = _run_git(["worktree", "add", str(worktree_path), branch], cwd=project_root)
    if res.returncode != 0:
        print(f"ERROR: failed to create worktree: {res.stderr}", file=sys.stderr)
        # Cleanup branch
        _run_git(["branch", "-D", branch], cwd=project_root)
        return 1

    # Create definition file (tracked)
    definitions_dir.mkdir(parents=True, exist_ok=True)
    definition_content = f"""schema_version: audiollm.experiment_group.v1
experiment_id: {experiment_id}
slug: {slug}
tier: {tier}
branch: {branch}
worktree: {worktree_path}
parent_branch: {parent_branch or 'null'}
parent_sha: {parent_sha}
created_at_utc: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
"""
    if tier == 1:
        definition_content += "type: competing\n"
    elif tier == 2:
        definition_content += "type: complementary\n"

    try:
        definition_path.write_text(definition_content, encoding="utf-8")
        print(f"created definition {definition_path}")
    except Exception as e:
        print(f"ERROR: failed to write definition: {e}", file=sys.stderr)
        return 1

    # Create pin file (ignored)
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": experiment_id,
        "worktree": str(worktree_path),
        "branch": branch,
        "allowed_paths": [str(worktree_path)],
        "protected_paths": [str(p) for p in PROTECTED_PATHS],
    }
    if parent_branch:
        pin_data["parent_branch"] = parent_branch
        pin_data["parent_sha"] = parent_sha

    pin_path = worktree_path / ".agent-pin.json"
    try:
        # Ensure worktree exists and write pin
        with pin_path.open("w", encoding="utf-8") as f:
            json.dump(pin_data, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"created pin {pin_path}")
    except Exception as e:
        print(f"ERROR: failed to write pin: {e}", file=sys.stderr)
        return 1

    print(f"created lane {experiment_id} branch {branch} worktree {worktree_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local experiment registry.")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / registry.DEFAULT_DB_PATH),
        help="SQLite registry path (default: outputs/experiment_registry/experiments.sqlite)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list runs with headline metrics")
    list_parser.add_argument("--dataset", default=None)
    list_parser.add_argument("--status", default=None)
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="show an attempt with its folds/evaluations/artifacts/jobs")
    show_parser.add_argument("attempt_id")
    show_parser.add_argument("--fold", type=int, default=None)
    show_parser.set_defaults(func=_cmd_show)

    provenance_parser = subparsers.add_parser("provenance", help="show the provenance chain of a metric id")
    provenance_parser.add_argument("metric_id", type=int)
    provenance_parser.set_defaults(func=_cmd_provenance)

    jobs_parser = subparsers.add_parser("jobs", help="list recorded job events")
    jobs_parser.add_argument("--failed", action="store_true", help="only failed/cancelled/timed-out jobs")
    jobs_parser.set_defaults(func=_cmd_jobs)

    best_parser = subparsers.add_parser(
        "best",
        help="fully qualified best-metric query; every qualifier is required to avoid mixing protocols",
    )
    for option in ("--dataset", "--metric", "--namespace", "--backend", "--view", "--aggregation"):
        best_parser.add_argument(option, required=True)
    best_parser.add_argument("--limit", type=int, default=None)
    best_parser.set_defaults(func=_cmd_best)

    # New create command
    create_parser = subparsers.add_parser("create", help="create new experiment lane (worktree/branch/pin/definition)")
    create_parser.add_argument("slug", help="experiment slug (e.g., exp-rotary or rotary)")
    create_parser.add_argument("--tier", type=int, choices=[0,1,2], required=True, help="Tier 0=CLI-only, 1=competing, 2=complementary")
    create_parser.add_argument("--from", dest="from_ref", default=None, help="parent branch or SHA for stacked lanes")
    create_parser.add_argument("--dry-run", action="store_true", help="show what would be done without mutation")
    create_parser.set_defaults(func=_cmd_create)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
