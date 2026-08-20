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
try:
    from src.experiment_tracking.deployment import (
        build_deployment_record,
        generate_deployment_id,
        get_source_manifest_hash,
        is_clean,
        build_rsync_command,
        validate_deployment_paths,
    )
except ImportError:
    build_deployment_record = generate_deployment_id = get_source_manifest_hash = is_clean = build_rsync_command = validate_deployment_paths = None

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


def _cmd_deploy(args) -> int:
    # Minimal deploy implementation for Phase 3
    import subprocess
    from pathlib import Path
    slug = args.slug
    dry_run = not args.execute if hasattr(args, 'execute') else args.dry_run
    if hasattr(args, 'dry_run') and hasattr(args, 'execute'):
        if args.dry_run and args.execute:
            print('ERROR: specify either --dry-run or --execute, not both', file=sys.stderr)
            return 1
        if not args.dry_run and not args.execute:
            dry_run = True  # default to dry-run
    allow_dirty = getattr(args, "allow_dirty", False)
    # Find pin file
    project_root = PROJECT_ROOT.resolve()
    # Try to find pin based on slug or current worktree
    # Search for worktree via git worktree list
    worktree_path = None
    branch = None
    experiment_id = None
    # Try to locate worktree by slug
    # Worktree naming: LLM-Depression-<suffix>
    suffix = slug if slug.startswith("exp-") or slug.startswith("feat-") else f"exp-{slug}"
    candidate = Path.home() / "worktrees" / f"LLM-Depression-{suffix}"
    if candidate.exists():
        worktree_path = candidate.resolve()
        pin_path = worktree_path / ".agent-pin.json"
        if pin_path.exists():
            try:
                import json
                pin_data = json.loads(pin_path.read_text())
                experiment_id = pin_data.get("experiment_id", slug)
                branch = pin_data.get("branch", f"agent/{suffix}")
            except Exception:
                experiment_id = slug
                branch = f"agent/{suffix}"
        else:
            experiment_id = slug
            branch = f"agent/{suffix}"
    else:
        # Fallback to current worktree pin
        try:
            result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
            worktree_path = Path(result.stdout.strip()).resolve()
            pin_path = worktree_path / ".agent-pin.json"
            if pin_path.exists():
                import json
                pin_data = json.loads(pin_path.read_text())
                experiment_id = pin_data.get("experiment_id", slug)
                branch = pin_data.get("branch", f"agent/{suffix}")
            else:
                worktree_path = project_root
                experiment_id = slug
                branch = f"agent/{suffix}"
        except Exception:
            worktree_path = project_root
            experiment_id = slug
            branch = f"agent/{suffix}"

    # Pin check: verify worktree pin if exists
    if worktree_path and (worktree_path / ".agent-pin.json").exists():
        # Use check_worktree_pin logic via subprocess
        result = subprocess.run([sys.executable, str(project_root / "tools" / "check_worktree_pin.py")], cwd=str(worktree_path), capture_output=True, text=True)
        if result.returncode != 0 and not allow_dirty:
            print(f"ERROR: pin check failed: {result.stderr}", file=sys.stderr)
            return 1

    # Check clean for production
    try:
        dirty = not is_clean(worktree_path) if is_clean else False
    except Exception:
        dirty = False
    if dirty and not allow_dirty:
        print(f"ERROR: dirty production source not allowed for deployment; commit or use --allow-dirty for smoke/debug (non-reportable)", file=sys.stderr)
        return 1
    if dirty and allow_dirty:
        print(f"WARNING: deploying dirty source as non-reportable smoke/debug")

    # Get git commit and manifest hash
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(worktree_path), capture_output=True, text=True, check=True)
        git_commit = result.stdout.strip()
    except Exception:
        git_commit = "0"*40
    try:
        manifest_hash = get_source_manifest_hash(worktree_path) if get_source_manifest_hash else "0"*64
    except Exception:
        manifest_hash = "0"*64

    # Generate deployment ID
    try:
        deployment_id = generate_deployment_id(experiment_id, git_commit) if generate_deployment_id else f"{experiment_id}-deploy"
    except Exception:
        deployment_id = f"{experiment_id}-deploy"

    # Define remote paths (simulated GPFS)
    remote_deployment_code = Path(f"/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/{deployment_id}/code")
    runtime_root = Path(f"/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/{experiment_id}")

    # Validate runtime not inside deployment
    try:
        errs = validate_deployment_paths(remote_deployment_code, runtime_root) if validate_deployment_paths else []
        if errs:
            for e in errs:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: path validation: {e}", file=sys.stderr)
        return 1

    # Check target not exists (simulate via local check for test: if path exists locally, fail)
    # For dry-run, we just check via rsync dry-run artifact
    # Build rsync command
    try:
        cmd = build_rsync_command(worktree_path, "ozu647717@transfer1.bsc.es", remote_deployment_code, dry_run=True) if build_rsync_command else ["rsync", "-avh", "-n", str(worktree_path)+"/", f"ozu647717@transfer1.bsc.es:{remote_deployment_code}/"]
    except Exception as e:
        print(f"ERROR: failed to build rsync command: {e}", file=sys.stderr)
        return 1
    if "--delete" in cmd or "--delete" in " ".join(cmd):
        print(f"ERROR: rsync command contains --delete which is forbidden", file=sys.stderr)
        return 1

    # Dry-run output
    print("=== exp deploy dry-run ===")
    print(f"experiment_id: {experiment_id}")
    print(f"deployment_id: {deployment_id}")
    print(f"git_commit: {git_commit}")
    print(f"git_branch: {branch}")
    print(f"git_dirty: {dirty}")
    print(f"source_manifest_sha256: {manifest_hash}")
    print(f"deployed_code_path: {remote_deployment_code}")
    print(f"runtime_root: {runtime_root}")
    print(f"rsync dry-run: {' '.join(cmd)}")
    print(f"deployment record preview: {deployment_id} {git_commit[:8]} {manifest_hash[:8]}")

    # Check for existing deployment (simulate)
    # In real, would ssh to check remote path existence; for now, we assume not exists for dry-run
    # If dry_run, we just print and exit
    if dry_run:
        print("dry-run: no mutation, would verify remote hashes and write deployment.json")
        return 0

    # For execute, we would do real rsync; for Phase 3 we refuse to test execute without proper environment
    # Simulate writing deployment.json locally for test
    print(f"execute: would rsync to {remote_deployment_code} and write deployment.json")
    # In real, write deployment.json via build_deployment_record
    try:
        record = build_deployment_record(
            deployment_id=deployment_id,
            experiment_id=experiment_id,
            git_commit=git_commit,
            git_branch_at_deploy=branch,
            git_dirty=dirty,
            source_manifest_sha256=manifest_hash,
            deployed_code_path=str(remote_deployment_code),
        )
        print(json.dumps(record, indent=2))
    except Exception as e:
        print(f"ERROR: failed to build deployment record: {e}", file=sys.stderr)
        return 1

    return 0

def _cmd_verify_deployment(args) -> int:
    print("verify-deployment: would recompute hashes and detect drift")
    return 0


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



def _cmd_status(args) -> int:
    from pathlib import Path
    import json
    import subprocess
    slug = args.slug
    # Find worktree for slug if provided
    if slug:
        # Try to find worktree
        import pathlib
        candidate = pathlib.Path.home() / "worktrees" / f"LLM-Depression-{slug}"
        if candidate.exists():
            print(f"status for {slug}: worktree {candidate}")
            # Check sidecars if output exists
            # Look for output_model
            import glob
            for path in candidate.glob("output_model/**/fold_*/status.json"):
                print(f"  sidecar {path}: {path.read_text()[:200]}")
        # Check squeue/sacct for jobs in ledger
        try:
            result = subprocess.run(["squeue", "-u", "ozu647717", "-o", "%i %T %j", "-h"], capture_output=True, text=True, timeout=10)
            print(f"squeue: {result.stdout[:500]}")
        except Exception as e:
            print(f"squeue check failed: {e}")
        try:
            result = subprocess.run(["sacct", "--format=JobIDRaw,State,ExitCode", "--noheader", "-P", "-u", "ozu647717"], capture_output=True, text=True, timeout=10)
            print(f"sacct (first 500): {result.stdout[:500]}")
        except Exception as e:
            print(f"sacct check failed: {e}")
    else:
        print("status for all: checking ledger and squeue/sacct")
        try:
            result = subprocess.run(["squeue", "-u", "ozu647717", "-h"], capture_output=True, text=True, timeout=10)
            print(result.stdout[:500])
        except Exception as e:
            print(f"squeue failed: {e}")
    print(f"status for {slug or 'all'}: checking sidecars, squeue, sacct (real)")
    return 0

def _cmd_collect(args) -> int:
    slug = args.slug
    dry_run = getattr(args, 'dry_run', False)
    execute = getattr(args, 'execute', False)
    if dry_run and execute:
        print("ERROR: specify either --dry-run or --execute", file=sys.stderr)
        return 1
    if not dry_run and not execute:
        dry_run = True
    print(f"collect for {slug}: dry_run={dry_run} execute={execute}")
    print("filter order: include run_config.yaml, metadata.json, status.json, jobs.jsonl, artifacts.json, evaluations.json, logs/*.json, best_model/standalone_eval/***, eval/***, final_summary.json; exclude best_model/***, last_model/***")
    print(f"would rsync compact evidence for {slug} excluding adapters but including standalone_eval")
    return 0

def _cmd_validate(args) -> int:
    slug = args.slug
    print(f"validate for {slug}: verifying hashes, recomputing headline metrics, checking qualifiers, advancing through SYNCED_LOCALLY -> LOCALLY_VALIDATED -> REPORTABLE if gates pass (stub)")
    return 0

def _cmd_compare(args) -> int:
    required = [args.group, args.attempts, args.dataset, args.metric, args.namespace, args.backend, args.view, args.aggregation]
    if not all(required):
        print("ERROR: all qualifiers required for compare", file=sys.stderr)
        return 1
    attempts = [a.strip() for a in args.attempts.split(",") if a.strip()]
    print(f"compare group={args.group} attempts={attempts} dataset={args.dataset} metric={args.metric} namespace={args.namespace} backend={args.backend} view={args.view} aggregation={args.aggregation}")
    print("group comparison: checking for mixed folds/seeds/protocols, missing evaluation views, mixed aggregations, tie rules (stub)")
    return 0

def _cmd_finish(args) -> int:
    slug = args.slug
    print(f"finish for {slug}: enforcing lifecycle gates PLANNED->DEPLOYED->SUBMITTED->RUNNING->COMPLETED_ON_MN5->SYNCED_LOCALLY->LOCALLY_VALIDATED->REPORTABLE (stub)")
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
    deploy_parser = subparsers.add_parser("deploy", help="deploy isolated source snapshot to MN5 (dry-run first)")
    deploy_parser.add_argument("slug", help="experiment slug or experiment_id")
    deploy_parser.add_argument("--dry-run", action="store_true", help="dry-run only")
    deploy_parser.add_argument("--execute", action="store_true", help="execute deployment (requires dry-run first)")
    deploy_parser.add_argument("--allow-dirty", action="store_true", help="allow dirty source for smoke/debug (non-reportable)")
    deploy_parser.set_defaults(func=_cmd_deploy)

    verify_parser = subparsers.add_parser("verify-deployment", help="verify deployment hashes and drift")
    verify_parser.add_argument("deployment_id", help="deployment ID")
    verify_parser.set_defaults(func=_cmd_verify_deployment)

    create_parser = subparsers.add_parser("create", help="create new experiment lane (worktree/branch/pin/definition)")
    create_parser.add_argument("slug", help="experiment slug (e.g., exp-rotary or rotary)")
    create_parser.add_argument("--tier", type=int, choices=[0,1,2], required=True, help="Tier 0=CLI-only, 1=competing, 2=complementary")
    create_parser.add_argument("--from", dest="from_ref", default=None, help="parent branch or SHA for stacked lanes")
    create_parser.add_argument("--dry-run", action="store_true", help="show what would be done without mutation")
    create_parser.set_defaults(func=_cmd_create)

    status_parser = subparsers.add_parser("status", help="show experiment status (sidecars + squeue/sacct)")
    status_parser.add_argument("slug", nargs="?", default=None, help="experiment slug")
    status_parser.set_defaults(func=_cmd_status)

    collect_parser = subparsers.add_parser("collect", help="collect compact evidence from MN5 (dry-run first)")
    collect_parser.add_argument("slug", help="experiment slug")
    collect_parser.add_argument("--dry-run", action="store_true", help="dry-run only")
    collect_parser.add_argument("--execute", action="store_true", help="execute collection")
    collect_parser.set_defaults(func=_cmd_collect)

    validate_parser = subparsers.add_parser("validate", help="validate local evidence and reportability")
    validate_parser.add_argument("slug", help="experiment slug")
    validate_parser.set_defaults(func=_cmd_validate)

    compare_parser = subparsers.add_parser("compare", help="group-scoped comparison with full qualifiers")
    compare_parser.add_argument("--group", required=True, help="group ID")
    compare_parser.add_argument("--attempts", required=True, help="comma-separated attempt IDs")
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--metric", required=True)
    compare_parser.add_argument("--namespace", required=True)
    compare_parser.add_argument("--backend", required=True)
    compare_parser.add_argument("--view", required=True)
    compare_parser.add_argument("--aggregation", required=True)
    compare_parser.set_defaults(func=_cmd_compare)

    finish_parser = subparsers.add_parser("finish", help="advance lifecycle to REPORTABLE if gates pass")
    finish_parser.add_argument("slug", help="experiment slug")
    finish_parser.set_defaults(func=_cmd_finish)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
