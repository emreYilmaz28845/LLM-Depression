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
        DeploymentError,
        RemoteRunner,
        build_deployment_record,
        generate_deployment_id,
        get_source_manifest_hash,
        is_clean,
        build_rsync_command,
        validate_deployment_paths,
        load_source_manifest,
        sha256_of_json,
        plan_deployment,
        execute_deployment,
        verify_deployment,
        DEFAULT_TRANSFER_HOST,
        REMOTE_BASE,
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


def _resolve_lane(slug: str):
    """Resolve a lane slug to (worktree, pin_data) by scanning managed pins."""
    worktrees_root = Path.home() / "worktrees"
    candidates = []
    direct = worktrees_root / f"LLM-Depression-{slug}"
    if direct.exists():
        candidates.append(direct)
    if worktrees_root.exists():
        for entry in sorted(worktrees_root.glob("LLM-Depression-*")):
            pin_path = entry / ".agent-pin.json"
            if not pin_path.exists():
                continue
            try:
                pin = json.loads(pin_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if pin.get("experiment_id") == slug or pin.get("branch") == f"agent/{slug}":
                candidates.append(entry)
    for candidate in candidates:
        pin_path = candidate / ".agent-pin.json"
        if not pin_path.exists():
            continue
        try:
            pin = json.loads(pin_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        return candidate.resolve(), pin
    return None, None


def _check_pin(worktree_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "check_worktree_pin.py"),
         "--pin", str(worktree_path / ".agent-pin.json"),
         "--cwd", str(worktree_path),
         "--target", str(worktree_path)],
        capture_output=True, text=True,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def _capture_provenance(worktree_path: Path) -> tuple[bool, str]:
    script = PROJECT_ROOT / "scripts" / "capture_provenance.sh"
    result = subprocess.run(["bash", str(script)], cwd=str(worktree_path), capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _deploy_evidence_dir(deployment_id: str) -> Path:
    evidence_dir = PROJECT_ROOT / "outputs" / "exp_deploy" / deployment_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def _cmd_deploy(args) -> int:
    slug = args.slug
    allow_dirty = getattr(args, "allow_dirty", False)
    execute = bool(getattr(args, "execute", False))
    dry_run_flag = bool(getattr(args, "dry_run", False))
    if execute and dry_run_flag:
        print("ERROR: specify either --dry-run or --execute, not both", file=sys.stderr)
        return 1

    worktree_path, pin = _resolve_lane(slug)
    if worktree_path is None or pin is None:
        print(f"ERROR: no managed lane with pin found for slug {slug!r} under ~/worktrees/", file=sys.stderr)
        return 1
    experiment_id = pin.get("experiment_id") or slug
    branch = pin.get("branch") or f"agent/{slug}"

    ok, message = _check_pin(worktree_path)
    if not ok:
        print(f"ERROR: pin check failed: {message}", file=sys.stderr)
        return 1

    ok, message = _capture_provenance(worktree_path)
    if not ok:
        print(f"ERROR: provenance capture failed: {message}", file=sys.stderr)
        return 1

    try:
        plan = plan_deployment(
            worktree=worktree_path,
            experiment_id=experiment_id,
            branch=branch,
            allow_dirty=allow_dirty,
            transfer_host=DEFAULT_TRANSFER_HOST,
            remote_base=REMOTE_BASE,
        )
    except DeploymentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    evidence_dir = _deploy_evidence_dir(plan["deployment_id"])
    plan_path = evidence_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner = RemoteRunner(host=plan["transfer_host"])

    print("=== exp deploy plan ===")
    for key in (
        "experiment_id", "deployment_id", "git_commit", "branch", "git_dirty",
        "reportable_allowed", "source_manifest_sha256", "source_manifest_file_count",
        "deployed_code_path", "deployment_record_path", "runtime_root",
        "estimated_transfer_bytes",
    ):
        print(f"{key}: {plan[key]}")
    print(f"rsync dry-run argv: {' '.join(plan['rsync_dry_run_argv'])}")
    print(f"rsync execute argv: {' '.join(plan['rsync_execute_argv'])}")

    from src.experiment_tracking.deployment import (
        require_remote_absent, run_local_rsync,
    )
    from pathlib import Path as _P

    try:
        require_remote_absent(runner, _P(plan["deployment_dir"]), "deployment directory")
        dry_proc = run_local_rsync(list(plan["rsync_dry_run_argv"]))
    except DeploymentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if dry_proc.returncode != 0:
        print(f"ERROR: rsync dry-run failed: {dry_proc.stderr.strip()}", file=sys.stderr)
        return 1
    dry_run_log = evidence_dir / "rsync_dry_run.txt"
    dry_run_log.write_text(
        "$ " + " ".join(plan["rsync_dry_run_argv"]) + "\n"
        + dry_proc.stdout + ("\n[stderr]\n" + dry_proc.stderr if dry_proc.stderr else ""),
        encoding="utf-8",
    )
    transferred = sum(1 for line in dry_proc.stdout.splitlines() if line.startswith(">f"))
    print(f"rsync dry-run through {plan['transfer_host']}: {transferred} files would transfer "
          f"(itemized log: {dry_run_log})")

    if not execute:
        print("dry-run complete; review the itemized log, then run --execute")
        return 0

    try:
        result = execute_deployment(plan, runner, rsync_executor=run_local_rsync)
    except DeploymentError as e:
        print(f"ERROR: deployment aborted, no deployment record was written: {e}", file=sys.stderr)
        return 1

    record_path = evidence_dir / "deployment.json"
    record_path.write_text(json.dumps(result["record"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_log = evidence_dir / "verify_tree.json"
    verify_log.write_text(json.dumps(result["verification"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== exp deploy executed ===")
    print(f"verified_files: {result['verification']['verified_files']}/{result['verification']['expected_files']}")
    print(f"deployment record written once at: {plan['deployment_record_path']}")
    print(f"local evidence: {evidence_dir}")
    print(json.dumps(result["record"], indent=2, sort_keys=True))
    return 0


def _find_deployment_record(experiment_id: str, deployment_id: str | None = None):
    deploy_root = PROJECT_ROOT / "outputs" / "exp_deploy"
    if not deploy_root.exists():
        return None, f"no local deployment evidence under {deploy_root}"
    candidates = []
    for record_path in sorted(deploy_root.glob("*/deployment.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("experiment_id") != experiment_id:
            continue
        if deployment_id and record.get("deployment_id") != deployment_id:
            continue
        candidates.append((record_path, record))
    if not candidates:
        return None, f"no deployment record for experiment {experiment_id}"
    return candidates[-1]


def _cmd_submit(args) -> int:
    from src.experiment_tracking.submit import (
        SubmissionError,
        SshSubmitRunner,
        build_remote_submit_script,
        check_collisions,
        parse_submitted_job_ids,
        resolve_contract,
        DEFAULT_SCHEDULER_HOST,
    )
    from src.experiment_tracking.deployment import (
        DeploymentError,
        RemoteRunner,
        remote_path_exists,
        verify_deployment,
        write_remote_file_once,
        REMOTE_BASE,
    )

    slug = args.slug
    execute = bool(args.execute)
    dry_run_flag = bool(args.dry_run)
    if execute and dry_run_flag:
        print("ERROR: specify either --dry-run or --execute, not both", file=sys.stderr)
        return 1

    worktree_path, pin = _resolve_lane(slug)
    if worktree_path is None or pin is None:
        print(f"ERROR: no managed lane with pin found for slug {slug!r}", file=sys.stderr)
        return 1
    experiment_id = pin.get("experiment_id") or slug
    ok, message = _check_pin(worktree_path)
    if not ok:
        print(f"ERROR: pin check failed: {message}", file=sys.stderr)
        return 1

    found = _find_deployment_record(experiment_id, getattr(args, "deployment_id", None))
    if isinstance(found, tuple) and len(found) == 2 and isinstance(found[0], Path):
        _record_path, deployment = found
    else:
        print(f"ERROR: {found}", file=sys.stderr)
        return 1

    # Resolve config locally with the user's scientific overrides only.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils import load_yaml_with_overrides
    user_overrides = list(args.set or [])
    scientific_tokens = []
    for token in user_overrides:
        scientific_tokens.extend(["--set", token] if "=" in token else [token])
    try:
        config_dict = load_yaml_with_overrides(Path(args.config), scientific_tokens)
    except Exception as e:
        print(f"ERROR: failed to load config with overrides: {e}", file=sys.stderr)
        return 1
    dataset = config_dict.get("dataset")
    if args.dataset and dataset != args.dataset:
        print(f"ERROR: --dataset {args.dataset} does not match config dataset {dataset}", file=sys.stderr)
        return 1

    # The remote config path lives inside the deployment code snapshot.
    rel_config = Path(args.config)
    marker = "configs" + os.sep
    if marker in str(rel_config):
        idx = str(rel_config).index(marker)
        remote_rel = str(rel_config)[idx:].replace(os.sep, "/")
    elif "configs/" in str(rel_config):
        idx = str(rel_config).index("configs/")
        remote_rel = str(rel_config)[idx:]
    else:
        remote_rel = f"configs/main/{rel_config.name}"
    config_path_remote = f"{deployment['deployed_code_path']}/{remote_rel}"

    try:
        contract = resolve_contract(
            experiment_id=experiment_id,
            deployment=deployment,
            config_path_remote=config_path_remote,
            config_dict=config_dict,
            fold=args.fold,
            seed=args.seed,
            run_name=args.run_name,
            campaign=args.campaign,
            modality=args.modality,
            dataset=dataset,
            extra_overrides=[f"--set={t}" if "=" in t and not t.startswith("--set") else t for t in user_overrides],
            scheduler_host=getattr(args, "scheduler_host", None) or DEFAULT_SCHEDULER_HOST,
            supersedes_attempt_id=getattr(args, "supersedes_attempt_id", None),
            group_id=getattr(args, "group_id", None),
            github_issue=os.environ.get("GITHUB_ISSUE"),
            github_pr=os.environ.get("GITHUB_PR"),
        )
    except SubmissionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    evidence_dir = PROJECT_ROOT / "outputs" / "exp_submit" / contract["attempt_id"]
    evidence_dir.mkdir(parents=True, exist_ok=True)

    (evidence_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== exp submit contract ===")
    printable = {k: v for k, v in contract.items() if k not in ("context", "overrides_b64")}
    print(json.dumps(printable, indent=2, sort_keys=True))
    script = build_remote_submit_script(contract)
    (evidence_dir / "submit_script.sh").write_text(script, encoding="utf-8")
    (evidence_dir / "context.json").write_text(json.dumps(contract["context"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not execute:
        print("--- remote submit script (would run on scheduler login) ---")
        print(script)
        print("dry-run complete; no mutation performed")
        return 0

    runner = RemoteRunner(host="ozu647717@transfer1.bsc.es")
    try:
        result = verify_deployment(
            runner,
            deployment["deployment_id"],
            remote_base=REMOTE_BASE,
            expected_git_commit=deployment.get("git_commit"),
            expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
        )
        print(f"deployment verified: {result['deployment_id']} "
              f"({result['tree_verification']['verified_files']}/{result['tree_verification']['expected_files']} files)")

        def exists(path: str) -> bool:
            return remote_path_exists(runner, path)

        check_collisions(contract, exists)

        context_payload = json.dumps(contract["context"], indent=2, sort_keys=True) + "\n"
        ctx_parent = str(Path(contract["context_path"]).parent)
        proc = runner.run("mkdir -p " + shlex.quote(ctx_parent))
        if proc.returncode != 0:
            raise DeploymentError(f"failed to create context dir: {proc.stderr.strip()}")
        write_remote_file_once(runner, contract["context_path"], context_payload, "experiment context")
        print(f"context written: {contract['context_path']}")
    except (DeploymentError, Exception) as e:
        if isinstance(e, (DeploymentError, SubmissionError)):
            print(f"ERROR: submission aborted before sbatch: {e}", file=sys.stderr)
            return 1
        raise

    submit_runner = SshSubmitRunner(host=contract["scheduler_host"])
    proc = submit_runner.run_script(script)
    (evidence_dir / "submit_output.log").write_text(
        "$ ssh " + contract["scheduler_host"] + " bash -s < submit_script.sh\n"
        + proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"ERROR: remote submission failed rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
        return 1
    job_ids = parse_submitted_job_ids(proc.stdout)
    if not job_ids:
        print("ERROR: no job IDs parsed from wrapper output; refusing to record events", file=sys.stderr)
        return 1
    print(f"submitted jobs: {job_ids}")

    STATE_PATH = PROJECT_ROOT / "outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json"
    q = shlex.quote
    qualifiers = contract["qualifiers"]
    dep_ids = []
    for graph_job in contract["job_graph"]:
        jid = job_ids.get(graph_job["job_key"])
        if not jid:
            continue
        cmd = [
            sys.executable, str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
            "record-job", "--state", str(STATE_PATH),
            "--attempt-id", contract["attempt_id"],
            "--job-key", graph_job["job_key"],
            "--job-type", graph_job["job_type"],
            "--event-type", "SUBMITTED",
            "--slurm-job-id", jid,
            "--status", "PENDING",
            "--fold", str(contract["fold"]),
            "--deployment-id", contract["deployment_id"],
            "--evaluation-view", qualifiers["evaluation_view"],
            "--backend", qualifiers["backend"],
            "--aggregation", qualifiers["aggregation"],
        ]
        if graph_job.get("depends_on"):
            resolved_dep = [job_ids[d] for d in graph_job["depends_on"] if d in job_ids]
            if resolved_dep:
                cmd += ["--dependency-job-ids", ",".join(resolved_dep)]
        rec = subprocess.run(cmd, capture_output=True, text=True)
        if rec.returncode != 0:
            print(f"ERROR: failed to record job event: {rec.stderr.strip()}", file=sys.stderr)
            return 1
        dep_ids.append(jid)
    print(f"ledger updated with {len(dep_ids)} SUBMITTED job events")
    return 0


def _cmd_verify_deployment(args) -> int:
    deployment_id = args.deployment_id
    expected_commit = getattr(args, "expected_git_commit", None)
    expected_manifest = getattr(args, "expected_source_manifest_sha256", None)
    runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    try:
        result = verify_deployment(
            runner,
            deployment_id,
            remote_base=REMOTE_BASE,
            expected_git_commit=expected_commit,
            expected_source_manifest_sha256=expected_manifest,
        )
    except DeploymentError as e:
        print(f"VERIFY-DEPLOYMENT FAILED: {e}", file=sys.stderr)
        return 1
    tree = result["tree_verification"]
    print("=== verify-deployment ===")
    print(f"deployment_id: {deployment_id}")
    print(f"git_commit: {result['record'].get('git_commit')}")
    print(f"source_manifest_sha256: {result['source_manifest_sha256']}")
    print(f"verified_files: {tree['verified_files']}/{tree['expected_files']}")
    print(f"unexpected_files: {len(tree['unexpected'])}")
    print("VERIFIED: deployment matches its record and manifest")
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
    from src.experiment_tracking.monitor import (
        MonitorError,
        SchedulerClient,
        reconcile_job,
    )

    slug = args.slug
    worktree_path, pin = (None, None)
    experiment_id = None
    if slug:
        worktree_path, pin = _resolve_lane(slug)
        if pin is None:
            print(f"ERROR: no managed lane with pin found for slug {slug!r}", file=sys.stderr)
            return 1
        experiment_id = pin.get("experiment_id") or slug

    state_path = PROJECT_ROOT / "outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json"
    ledger = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    lane_deployments = {
        d.get("deployment_id")
        for d in ledger.get("deployments", [])
        if not experiment_id or d.get("experiment_id") == experiment_id
    }
    if experiment_id:
        deploy_root = PROJECT_ROOT / "outputs" / "exp_deploy"
        if deploy_root.exists():
            for record_path in deploy_root.glob("*/deployment.json"):
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if record.get("experiment_id") == experiment_id:
                    lane_deployments.add(record.get("deployment_id"))
    lane_jobs = [
        j for j in ledger.get("jobs", [])
        if j.get("deployment_id") in lane_deployments or (not experiment_id and True)
    ]
    if experiment_id:
        lane_jobs = [j for j in lane_jobs if j.get("deployment_id") in lane_deployments]

    job_ids = sorted({str(j["slurm_job_id"]) for j in lane_jobs if j.get("slurm_job_id")})
    scheduler = SchedulerClient()
    had_error = False
    try:
        queue = scheduler.squeue(job_ids)
        accounting = scheduler.sacct(job_ids)
    except MonitorError as e:
        print(f"ERROR: remote scheduler query failed: {e}", file=sys.stderr)
        return 1

    unknown_ids = set(job_ids) - set(queue) - set(accounting)
    if unknown_ids:
        print(f"ERROR: jobs missing from both queue and accounting: {sorted(unknown_ids)}", file=sys.stderr)
        had_error = True

    print(f"=== exp status {slug or '(all lanes)'} ===")
    print(f"lane deployments: {sorted(d for d in lane_deployments if d)}")
    print(f"recorded jobs: {len(lane_jobs)}; scheduler-visible: {len(job_ids)}")
    header = f"{'attempt':<64} {'job_key':<10} {'slurm':>9} {'queue':<12} {'accounting':<12} {'exit':>5} {'artifacts':<9} classification"
    print(header)
    for record in lane_jobs:
        jid = str(record.get("slurm_job_id") or "")
        try:
            rec = reconcile_job(record, queue, accounting, artifacts_ok=None)
        except MonitorError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            had_error = True
            continue
        status_cell = rec.account_state or rec.queue_state or "UNKNOWN"
        cls = rec.classification or ("running" if rec.queue_state else ("-"))
        print(
            f"{str(record.get('attempt_id', '-')):<64} {rec.job_key:<10} {jid:>9} "
            f"{rec.queue_state or '-':<12} {status_cell:<12} {rec.exit_code or '-':>5} "
            f"{'n/a':<9} {cls}"
        )
        # Append terminal evidence through official APIs when newly terminal.
        if rec.terminal_failure or (rec.account_state == "COMPLETED" and rec.exit_code == "0:0"):
            already_terminal = any(
                str(j.get("slurm_job_id")) == jid and j.get("event_type") == "TERMINAL"
                for j in ledger.get("jobs", [])
            )
            if not already_terminal:
                terminal_status = rec.account_state
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
                    "record-job", "--state", str(state_path),
                    "--attempt-id", str(record.get("attempt_id")),
                    "--job-key", rec.job_key,
                    "--job-type", str(record.get("job_type", "train")),
                    "--event-type", "TERMINAL",
                    "--slurm-job-id", jid,
                    "--status", terminal_status,
                    "--fold", str(record.get("fold", 0)),
                    "--exit-code", rec.exit_code or "-",
                ]
                if rec.classification:
                    cmd += ["--reason", f"classification={rec.classification}"]
                term = subprocess.run(cmd, capture_output=True, text=True)
                if term.returncode != 0:
                    print(f"ERROR: failed to append terminal event: {term.stderr.strip()}", file=sys.stderr)
                    had_error = True
                else:
                    print(f"  appended TERMINAL event for {jid} ({terminal_status})")

    if had_error:
        return 1
    return 0

def _cmd_collect(args) -> int:
    from src.experiment_tracking.collect import (
        CollectionError,
        RemoteRunner,
        execute_collection,
        plan_collection,
        validate_fold_path,
    )

    slug = args.slug
    dry_run_flag = bool(getattr(args, "dry_run", False))
    execute = bool(getattr(args, "execute", False))
    if dry_run_flag and execute:
        print("ERROR: specify either --dry-run or --execute, not both", file=sys.stderr)
        return 1

    fold_dir = getattr(args, "fold_dir", None)
    attempt_id = getattr(args, "attempt_id", None)
    if not fold_dir:
        # Resolve the exact remote fold path from recorded submission evidence.
        submit_root = PROJECT_ROOT / "outputs" / "exp_submit"
        candidates = []
        if attempt_id:
            contract_path = submit_root / attempt_id / "contract.json"
            if contract_path.is_file():
                candidates.append(contract_path)
        elif submit_root.exists():
            for contract_path in sorted(submit_root.glob("*/contract.json")):
                candidates.append(contract_path)
        lane_pin_experiment = None
        if slug:
            _, pin = _resolve_lane(slug)
            lane_pin_experiment = pin.get("experiment_id") if pin else None
        chosen = None
        for contract_path in reversed(candidates):
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if attempt_id and contract.get("attempt_id") != attempt_id:
                continue
            if lane_pin_experiment and contract.get("experiment_id") != lane_pin_experiment:
                continue
            chosen = contract
            break
        if chosen is None:
            print("ERROR: no recorded submission contract found; pass --fold-dir or run exp submit first", file=sys.stderr)
            return 1
        fold_dir = chosen["fold_dir"]
        if attempt_id is None:
            attempt_id = chosen.get("attempt_id")

    local_fold = getattr(args, "output", None)
    if not local_fold:
        if not attempt_id:
            print("ERROR: --output required when the attempt id cannot be resolved", file=sys.stderr)
            return 1
        local_fold = str(PROJECT_ROOT / "output_model" / "collected" / attempt_id / Path(fold_dir).name)

    try:
        validate_fold_path(fold_dir)
    except CollectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    plan = plan_collection(fold_dir, local_fold)
    evidence_dir = PROJECT_ROOT / "outputs" / "exp_collect" / (attempt_id or "adhoc")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== exp collect plan ===")
    print(f"remote_fold: {plan['remote_fold']}")
    print(f"local_fold:  {plan['local_fold']}")
    print(f"rsync dry-run argv: {' '.join(plan['rsync_dry_run_argv'])}")
    print(f"rsync execute argv: {' '.join(plan['rsync_execute_argv'])}")

    runner = RemoteRunner()
    from src.experiment_tracking.collect import remote_inventory
    try:
        inv = remote_inventory(runner, plan["remote_fold"])
    except CollectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    (evidence_dir / "remote_inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"remote compact inventory: {len(inv)} files")

    if not execute:
        print("dry-run complete; review the plan, then run --execute")
        return 0

    try:
        result = execute_collection(plan, runner)
    except CollectionError as e:
        print(f"ERROR: collection failed: {e}", file=sys.stderr)
        return 1
    (evidence_dir / "collection_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected and hash-verified: {result['inventory']['matched']} files -> {result['local_fold']}")
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

    verify_parser = subparsers.add_parser("verify-deployment", help="verify deployment identity, manifest hash, and tree drift (read-only)")
    verify_parser.add_argument("deployment_id", help="deployment ID")
    verify_parser.add_argument("--expected-git-commit", default=None, help="fail if record commit differs")
    verify_parser.add_argument("--expected-source-manifest-sha256", default=None, help="fail if manifest hash differs")
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

    submit_parser = subparsers.add_parser("submit", help="submit train+standalone-eval job graph for a lane deployment (dry-run first)")
    submit_parser.add_argument("slug", help="experiment slug or experiment_id")
    submit_parser.add_argument("--config", required=True, help="config YAML path (local path; resolved inside the deployment)")
    submit_parser.add_argument("--fold", type=int, default=0)
    submit_parser.add_argument("--seed", type=int, default=None)
    submit_parser.add_argument("--run-name", required=True)
    submit_parser.add_argument("--campaign", required=True)
    submit_parser.add_argument("--modality", required=True)
    submit_parser.add_argument("--dataset", default=None, help="must match config dataset when given")
    submit_parser.add_argument("--deployment-id", default=None, help="defaults to latest local record for the lane")
    submit_parser.add_argument("--set", action="append", dest="set", default=[], help="scientific override key=value (repeatable; applied identically to train and eval)")
    submit_parser.add_argument("--scheduler-host", default=None, help="override scheduler login host")
    submit_parser.add_argument("--group-id", default=None)
    submit_parser.add_argument("--supersedes-attempt-id", default=None)
    submit_parser.add_argument("--dry-run", action="store_true", help="print the full resolved contract and exact commands without mutation")
    submit_parser.add_argument("--execute", action="store_true", help="verify deployment, transfer context, and submit through Slurm")
    submit_parser.set_defaults(func=_cmd_submit)

    collect_parser = subparsers.add_parser("collect", help="collect compact evidence from MN5 (dry-run first)")
    collect_parser.add_argument("slug", nargs="?", default=None, help="experiment slug")
    collect_parser.add_argument("--attempt-id", default=None, help="resolve fold path from this attempt's recorded contract")
    collect_parser.add_argument("--fold-dir", default=None, help="explicit remote fold dir (must end in fold_<n>)")
    collect_parser.add_argument("--output", default=None, help="local destination dir")
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
