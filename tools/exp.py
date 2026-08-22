from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import os
import hashlib

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

# The execution ledger lives in the canonical main checkout, not in whichever
# worktree invokes this tool. Override with PARALLEL_WORKFLOW_STATE if needed.
EXECUTION_LEDGER_PATH = Path(
    os.environ.get(
        "PARALLEL_WORKFLOW_STATE",
        "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/"
        "parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json",
    )
)

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


def _get_git_branch_name(cwd=None):
    result = _run_git(["branch", "--show-current"], cwd=cwd)
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _rollback_created_lane(project_root: Path, worktree_path: Path, branch: str) -> None:
    """Remove only resources created by a failed lane-creation attempt."""
    _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=project_root)
    _run_git(["branch", "-D", branch], cwd=project_root)


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
    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(worktree_path)
    result = subprocess.run(["bash", str(script)], cwd=str(worktree_path),
                            capture_output=True, text=True, env=env)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _deploy_evidence_dir(deployment_id: str) -> Path:
    evidence_dir = PROJECT_ROOT / "outputs" / "exp_deploy" / deployment_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def _load_linked_experiment_group(worktree_path: Path, pin: dict) -> dict:
    """Load and validate the scientific group linked by an operational lane."""
    import yaml
    from src.experiment_tracking.constants import SCHEMA_VERSION_EXPERIMENT_LANE
    from src.experiment_tracking.schemas import validate_experiment_group, validate_experiment_lane

    definition_rel = pin.get("definition_path")
    if not isinstance(definition_rel, str) or not definition_rel:
        raise ValueError("lane pin has no definition_path; recreate or migrate this legacy lane")
    lane_path = worktree_path / definition_rel
    try:
        lane = yaml.safe_load(lane_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not load lane definition {definition_rel}: {exc}") from exc
    valid, errors = validate_experiment_lane(lane)
    if not valid or lane.get("schema_version") != SCHEMA_VERSION_EXPERIMENT_LANE:
        raise ValueError("invalid experiment lane: " + "; ".join(errors))
    group_rel_value = lane.get("experiment_group_path")
    if not group_rel_value:
        raise ValueError(
            f"lane {lane['experiment_id']} has no experiment_group_path; link a complete "
            "audiollm.experiment_group.v1 definition before deploy or submit"
        )
    group_rel = Path(group_rel_value)
    definitions_root = (worktree_path / "experiments" / "definitions").resolve()
    group_path = (worktree_path / group_rel).resolve()
    if group_rel.is_absolute() or ".." in group_rel.parts or definitions_root not in group_path.parents:
        raise ValueError("experiment_group_path must be a relative file under experiments/definitions/")
    if "lanes" in group_path.relative_to(definitions_root).parts:
        raise ValueError("experiment_group_path cannot point into the operational lanes directory")
    try:
        raw = group_path.read_bytes()
        group = yaml.safe_load(raw)
    except Exception as exc:
        raise ValueError(f"could not load experiment group {group_rel_value}: {exc}") from exc
    valid, errors = validate_experiment_group(group)
    if not valid:
        raise ValueError("invalid linked experiment group: " + "; ".join(errors))
    return {
        "experiment_group_id": group["group_id"],
        "experiment_group_path": group_path.relative_to(worktree_path.resolve()).as_posix(),
        "experiment_group_sha256": hashlib.sha256(raw).hexdigest(),
    }


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

    try:
        group_identity = _load_linked_experiment_group(worktree_path, pin)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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

    plan.update(group_identity)
    plan["record"].update(group_identity)

    evidence_dir = _deploy_evidence_dir(plan["deployment_id"])
    plan_path = evidence_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner = RemoteRunner(host=plan["transfer_host"])

    print("=== exp deploy plan ===")
    for key in (
        "experiment_id", "deployment_id", "git_commit", "branch", "git_dirty",
        "reportable_allowed", "source_manifest_sha256", "source_manifest_file_count",
        "experiment_group_id", "experiment_group_path", "experiment_group_sha256",
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
    transferred = sum(
        1 for line in dry_proc.stdout.splitlines()
        if len(line) > 9 and line[1] == "f" and "+" in line
    )
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


def _find_deployment_record(experiment_id: str, deployment_id: str | None = None, allow_plan: bool = False):
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
    if not candidates and allow_plan:
        # Dry-run mode may plan against a reviewed deploy plan (no execution yet).
        for plan_path in sorted(deploy_root.glob("*/plan.json")):
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            record = plan.get("record")
            if not isinstance(record, dict):
                continue
            if record.get("experiment_id") != experiment_id:
                continue
            if deployment_id and record.get("deployment_id") != deployment_id:
                continue
            candidates.append((plan_path, record))
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
        require_complete_job_ids,
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

    try:
        group_identity = _load_linked_experiment_group(worktree_path, pin)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    found = _find_deployment_record(experiment_id, getattr(args, "deployment_id", None), allow_plan=not execute)
    if isinstance(found, tuple) and len(found) == 2 and isinstance(found[0], Path):
        _record_path, deployment = found
    else:
        print(f"ERROR: {found}", file=sys.stderr)
        return 1

    for key in ("experiment_group_id", "experiment_group_path", "experiment_group_sha256"):
        if deployment.get(key) != group_identity[key]:
            print(
                f"ERROR: deployment {key}={deployment.get(key)!r} does not match current lane "
                f"{group_identity[key]!r}; deploy the reviewed scientific definition",
                file=sys.stderr,
            )
            return 1
    requested_group_id = getattr(args, "group_id", None)
    if requested_group_id and requested_group_id != group_identity["experiment_group_id"]:
        print(
            f"ERROR: --group-id {requested_group_id!r} does not match linked experiment group "
            f"{group_identity['experiment_group_id']!r}", file=sys.stderr,
        )
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

    env_exports: dict[str, str] = {}
    if str(config_dict.get("model_backend", "")).strip().lower() == "gemma4":
        gemma_env = os.environ.get("GEMMA_ENV", "/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1")
        env_exports["ENV_ACTIVATE"] = (
            gemma_env if gemma_env.endswith("/bin/activate") else f"{gemma_env}/bin/activate"
        )
        model_path = os.environ.get("GEMMA4_MODEL_PATH") or str(
            config_dict.get("model_name_or_path") or ""
        )
        if model_path:
            env_exports["MODEL_PATH"] = model_path

    try:
        contract = resolve_contract(
            experiment_id=experiment_id,
            extra_env=env_exports,
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
            group_id=group_identity["experiment_group_id"],
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
    try:
        job_ids = require_complete_job_ids(
            parse_submitted_job_ids(proc.stdout), contract["job_graph"]
        )
    except SubmissionError as e:
        print(f"ERROR: {e}; refusing to record a partial job graph", file=sys.stderr)
        return 1
    print(f"submitted jobs: {job_ids}")

    STATE_PATH = EXECUTION_LEDGER_PATH
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
    if tier not in (1, 2):
        print(f"ERROR: managed worktrees support only tier 1 or 2, got {tier}", file=sys.stderr)
        return 1
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        print(
            "ERROR: slug must use lowercase letters, digits, and single hyphen-separated words",
            file=sys.stderr,
        )
        return 1
    # Determine branch name
    # Slug handling: if slug already starts with exp- or feat-, use as is; else add prefix based on tier
    if slug.startswith("exp-") or slug.startswith("feat-"):
        branch_suffix = slug
        expected_prefix = "exp-" if tier == 1 else "feat-" if tier == 2 else None
        if tier != 0 and expected_prefix and not slug.startswith(expected_prefix):
            print(
                f"ERROR: tier {tier} slug must start with {expected_prefix!r}, got {slug!r}",
                file=sys.stderr,
            )
            return 1
        branch = f"agent/{slug}"
    else:
        prefix = "exp" if tier == 1 else "feat"
        branch = f"agent/{prefix}-{slug}"
        branch_suffix = f"{prefix}-{slug}"

    # Worktree path: ~/worktrees/LLM-Depression-<branch_suffix>
    # Use branch_suffix as after agent/
    worktree_name = f"LLM-Depression-{branch_suffix}"
    worktree_path = Path.home() / "worktrees" / worktree_name
    # Resolve canonical
    worktree_path_resolved = worktree_path.resolve()
    worktrees_root = (Path.home() / "worktrees").resolve()
    if worktrees_root not in worktree_path_resolved.parents:
        print(f"ERROR: worktree path escapes managed root {worktrees_root}", file=sys.stderr)
        return 1

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

    # Determine parent ref. An explicit --from is provenance-critical and must
    # never silently fall back to another branch or commit.
    project_root = PROJECT_ROOT.resolve()
    parent_ref = from_ref if from_ref else "HEAD"
    parent_sha = _get_git_branch_sha(parent_ref, cwd=project_root)
    if parent_sha is None:
        print(f"ERROR: could not resolve parent ref {parent_ref}", file=sys.stderr)
        return 1
    parent_branch = from_ref if from_ref else _get_git_branch_name(cwd=project_root)
    if parent_branch and len(parent_branch) == 40 and all(c in "0123456789abcdef" for c in parent_branch.lower()):
        parent_branch = None

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
    definitions_dir = worktree_path / "experiments" / "definitions" / "lanes"
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
    definition_relpath = f"experiments/definitions/lanes/{experiment_id}.yaml"
    parent_definition = _run_git(["cat-file", "-e", f"{parent_sha}:{definition_relpath}"], cwd=project_root)
    if parent_definition.returncode == 0:
        errors.append(f"definition file {definition_relpath} already exists in parent {parent_sha[:8]}")

    branch_check = _run_git(["check-ref-format", "--branch", branch], cwd=project_root)
    if branch_check.returncode != 0:
        errors.append(f"invalid branch derived from slug: {branch}")

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
    try:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: failed to create worktree parent: {e}", file=sys.stderr)
        _rollback_created_lane(project_root, worktree_path, branch)
        return 1
    res = _run_git(["worktree", "add", str(worktree_path), branch], cwd=project_root)
    if res.returncode != 0:
        print(f"ERROR: failed to create worktree: {res.stderr}", file=sys.stderr)
        _rollback_created_lane(project_root, worktree_path, branch)
        return 1

    # Create definition file in the new worktree so the pin checker and the
    # eventual commit see the same tracked experiment identity.
    definition_content = f"""schema_version: audiollm.experiment_lane.v1
experiment_id: {experiment_id}
slug: {slug}
tier: {tier}
branch: {branch}
worktree: {worktree_path}
parent_branch: {parent_branch or 'null'}
parent_sha: {parent_sha}
created_at_utc: "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"
experiment_group_path: null
"""
    if tier == 1:
        definition_content += "type: competing\n"
    elif tier == 2:
        definition_content += "type: complementary\n"

    try:
        definitions_dir.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(definition_content, encoding="utf-8")
        print(f"created definition {definition_path}")
    except Exception as e:
        print(f"ERROR: failed to write definition: {e}", file=sys.stderr)
        _rollback_created_lane(project_root, worktree_path, branch)
        return 1

    # Create pin file (ignored)
    pin_data = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": experiment_id,
        "tier": tier,
        "worktree": str(worktree_path),
        "branch": branch,
        "parent_branch": parent_branch,
        "parent_sha": parent_sha,
        "definition_path": definition_relpath,
        "allowed_paths": [str(worktree_path)],
        "protected_paths": [str(p) for p in PROTECTED_PATHS],
    }

    pin_path = worktree_path / ".agent-pin.json"
    try:
        # Ensure worktree exists and write pin
        with pin_path.open("w", encoding="utf-8") as f:
            json.dump(pin_data, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"created pin {pin_path}")
    except Exception as e:
        print(f"ERROR: failed to write pin: {e}", file=sys.stderr)
        _rollback_created_lane(project_root, worktree_path, branch)
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

    state_path = EXECUTION_LEDGER_PATH
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
            # Mirror the terminal evidence into the attempt's fold jobs.jsonl
            # through the official append-only API, then advance the fold
            # lifecycle RUNNING -> COMPLETED_ON_MN5 when every required job
            # has COMPLETED 0:0.
            _mirror_terminal_to_fold(record, rec)

    if had_error:
        return 1
    return 0


def _mirror_terminal_to_fold(record: dict, rec) -> None:
    """Append the terminal job event to the local fold sidecar (if collected)
    and advance RUNNING -> COMPLETED_ON_MN5 once train+eval are COMPLETED 0:0."""
    from src.experiment_tracking import lifecycle
    from src.experiment_tracking.sidecars import is_modern_tracked

    attempt_id = record.get("attempt_id")
    submit_contract = PROJECT_ROOT / "outputs" / "exp_submit" / str(attempt_id) / "contract.json"
    if not submit_contract.is_file():
        return
    try:
        contract = json.loads(submit_contract.read_text(encoding="utf-8"))
    except Exception:
        return
    fold_dir = PROJECT_ROOT / contract.get("local_fold_rel", "")
    jobs_path = fold_dir / "jobs.jsonl"
    if not is_modern_tracked(fold_dir):
        return
    events = lifecycle.read_job_events(jobs_path)
    jid = str(rec.slurm_job_id)
    from src.experiment_tracking.monitor import terminal_event_type

    event_type = terminal_event_type(rec.account_state or "", rec.exit_code or "")
    already = any(
        str(e.get("slurm_job_id")) == jid
        and e.get("event_type") in {"COMPLETED", "FAILED", "CANCELLED"}
        for e in events
    )
    if not already:
        event = lifecycle.new_job_event(
            job_key=rec.job_key,
            job_type=str(record.get("job_type", "train")),
            event_type=event_type,
            attempt_id=str(attempt_id),
            fold=int(record.get("fold", 0)),
            slurm_job_id=jid,
            status=rec.account_state,
        )
        event["exit_code"] = rec.exit_code
        try:
            lifecycle.append_job_event(jobs_path, event)
        except Exception as e:
            print(f"WARNING: could not append to {jobs_path}: {e}", file=sys.stderr)
            return
    # Advance the fold lifecycle when both required jobs are COMPLETED 0:0.
    status_path = fold_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if status.get("state") not in {"SUBMITTED", "RUNNING"}:
        return
    if event_type in {"FAILED", "CANCELLED"}:
        target_state = event_type
        record_obj = lifecycle.StatusRecord.from_dict(status)
        try:
            record_obj.transition(
                target_state,
                reason=f"{rec.job_key} ended with {rec.account_state} {rec.exit_code}",
            )
        except Exception as e:
            print(f"WARNING: lifecycle transition refused: {e}", file=sys.stderr)
            return
        lifecycle.write_status(status_path, record_obj)
        print(f"  fold lifecycle advanced to {target_state} ({fold_dir.name})")
        return
    if status.get("state") != "RUNNING":
        return
    refreshed = lifecycle.read_job_events(jobs_path)
    done_keys = {
        str(e.get("job_key"))
        for e in refreshed
        if e.get("event_type") == "COMPLETED" and e.get("status") == "COMPLETED"
        and str(e.get("exit_code", "")).startswith("0:0")
    }
    if {"train", "best_eval"} <= done_keys:
        record_obj = lifecycle.StatusRecord.from_dict(status)
        try:
            record_obj.transition("COMPLETED_ON_MN5",
                                  reason="train and standalone eval COMPLETED 0:0 per sacct reconciliation")
        except Exception as e:
            print(f"WARNING: lifecycle transition refused: {e}", file=sys.stderr)
            return
        lifecycle.write_status(status_path, record_obj)
        print(f"  fold lifecycle advanced to COMPLETED_ON_MN5 ({fold_dir.name})")

def _study_submission_context(args, *, execute: bool):
    """Shared lane/pin/group/deployment resolution for study subcommands."""

    worktree_path, pin = _resolve_lane(args.slug)
    if worktree_path is None or pin is None:
        print("ERROR: no managed lane with pin found for slug %r" % args.slug, file=sys.stderr)
        return None
    experiment_id = pin.get("experiment_id") or args.slug
    ok, message = _check_pin(worktree_path)
    if not ok:
        print(f"ERROR: pin check failed: {message}", file=sys.stderr)
        return None
    try:
        group_identity = _load_linked_experiment_group(worktree_path, pin)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    found = _find_deployment_record(
        experiment_id, getattr(args, "deployment_id", None), allow_plan=not execute
    )
    if isinstance(found, tuple) and len(found) == 2 and isinstance(found[0], Path):
        _record_path, deployment = found
    else:
        print(f"ERROR: {found}", file=sys.stderr)
        return None
    for key in ("experiment_group_id", "experiment_group_path", "experiment_group_sha256"):
        if deployment.get(key) != group_identity[key]:
            print(
                f"ERROR: deployment {key}={deployment.get(key)!r} does not match current lane "
                f"{group_identity[key]!r}; deploy the reviewed scientific definition",
                file=sys.stderr,
            )
            return None
    requested_group_id = getattr(args, "group_id", None)
    if requested_group_id and requested_group_id != group_identity["experiment_group_id"]:
        print(
            f"ERROR: --group-id {requested_group_id!r} does not match linked experiment group "
            f"{group_identity['experiment_group_id']!r}",
            file=sys.stderr,
        )
        return None
    return {
        "worktree": worktree_path,
        "pin": pin,
        "experiment_id": experiment_id,
        "group_identity": group_identity,
        "deployment": deployment,
    }


def _study_remote_config_rel(config_str: str) -> str:
    marker = "configs/"
    idx = str(config_str).find(marker)
    if idx < 0:
        return f"configs/main/{Path(config_str).name}"
    return str(config_str)[idx:].replace(os.sep, "/")


def _require_remote_absent(runner, paths: list[str]) -> str | None:
    from src.experiment_tracking.deployment import remote_path_exists

    existing = [p for p in paths if remote_path_exists(runner, p)]
    if existing:
        return "; ".join(existing)
    return None


def _new_attempt_id(run_name: str, commit: str) -> str:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.experiment_tracking.identity import new_attempt_id

    return new_attempt_id(run_name, commit)


def _cmd_submit_merged(args) -> int:
    """Managed merged CV/final/smoke submission for a lane deployment."""

    execute = bool(args.execute)
    dry_run_flag = bool(args.dry_run)
    if execute == dry_run_flag:
        print("ERROR: specify exactly one of --dry-run or --execute", file=sys.stderr)
        return 1
    ctx = _study_submission_context(args, execute=execute)
    if ctx is None:
        return 1
    group_identity = ctx["group_identity"]
    deployment = ctx["deployment"]

    import tools.native_en_submit as ns

    import yaml as _yaml

    config_path = Path(args.config)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
        config_dict = _yaml.safe_load(raw_text)
    except Exception as exc:
        print(f"ERROR: cannot read merged config: {exc}", file=sys.stderr)
        return 1
    if str(config_dict.get("protocol")) != "symmetric_merged":
        print("ERROR: config protocol must be symmetric_merged", file=sys.stderr)
        return 1
    component_names = sorted(str(item["name"]) for item in config_dict.get("components") or [])
    if component_names != sorted(ns.MERGED_DATASETS):
        print(
            f"ERROR: merged components must be exactly {sorted(ns.MERGED_DATASETS)}, got {component_names}",
            file=sys.stderr,
        )
        return 1
    condition = str(getattr(args, "condition"))
    backbone = str(getattr(args, "backbone"))
    try:
        derived_text = ns.materialize_merged_config(config_dict, seed=int(args.seed))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    import hashlib

    derived_sha = hashlib.sha256(derived_text.encode("utf-8")).hexdigest()

    stage = str(args.stage)
    folds = (
        [int(v) for v in str(args.folds).split(",")]
        if getattr(args, "folds", None)
        else ns.merged_stage_folds(stage)
    )
    run_id = str(args.run_id)

    # Final-stage epoch derivation comes only from completed CV evidence.
    epochs = getattr(args, "epochs", None)
    if stage == "final" and epochs is None and execute:
        runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
        selections = []
        cv_condition = getattr(args, "cv_condition_for_epochs", None) or condition
        for fold in range(5):
            sel_path = (
                f"{ns.merged_run_root(cv_condition, backbone)}/"
                f"{getattr(args, 'cv_run_id_for_epochs', None) or run_id}/cv/fold_{fold}/logs/selected_checkpoint.json"
            )
            proc = runner.run("cat " + shlex.quote(sel_path))
            if proc.returncode != 0:
                print(f"ERROR: cannot read {sel_path}: {proc.stderr.strip()}", file=sys.stderr)
                return 1
            payload = json.loads(proc.stdout)
            value = payload.get("selected_epoch")
            if value is None:
                print(f"ERROR: {sel_path} has no selected_epoch", file=sys.stderr)
                return 1
            selections.append(int(value))
        epochs = ns.rounded_median_epoch(selections)
        print(f"derived final epochs from completed CV evidence: {epochs} (selections={selections})")

    code_path = str(deployment["deployed_code_path"])
    source_commit = str(deployment.get("git_commit"))
    group_id = group_identity["experiment_group_id"]
    from src.experiment_tracking.submit import REMOTE_RUNTIME_BASE

    runtime_root = str(REMOTE_RUNTIME_BASE / ctx["experiment_id"])

    plans = []
    for fold in folds:
        attempt_id = _new_attempt_id(f"{run_id}-{stage}-s{args.seed}-f{fold}", source_commit)
        paths = ns.merged_fold_paths(
            condition=condition, backbone=backbone, run_id=run_id, stage=stage, fold=fold
        )
        derived_config_remote = (
            f"{runtime_root}/configs/{run_id}/seed_{int(args.seed)}/{config_path.name}"
        )
        context_path = f"{runtime_root}/contexts/{attempt_id}/fold_{fold}/context.json"
        context_payload = {
            "schema_version": "audiollm.tracking_context.v1",
            "group_id": group_id,
            "logical_run_name": run_id,
            "attempt_id": attempt_id,
            "fold": int(fold),
            "seed": int(args.seed),
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "source": {
                "git_commit": source_commit,
                "git_branch_at_deploy": deployment.get("git_branch_at_deploy"),
                "git_dirty": False,
                "deployed_source_sha256": deployment.get("source_manifest_sha256"),
                "deployment_id": deployment.get("deployment_id"),
            },
            "research": {
                "github_issue": os.environ.get("GITHUB_ISSUE"),
                "github_pr": os.environ.get("GITHUB_PR"),
            },
            "condition": condition,
            "backbone": backbone,
            "stage": stage,
        }
        supersedes = getattr(args, "supersedes_attempt_id", None)
        if supersedes:
            context_payload["supersedes_attempt_id"] = supersedes
        script = ns.render_merged_chain_script(
            code_path=code_path,
            derived_config_path=derived_config_remote,
            derived_config_sha256=derived_sha,
            condition=condition,
            backbone=backbone,
            run_id=run_id,
            stage=stage,
            fold=fold,
            fold_dir=paths["fold_dir"],
            checkpoint_dir=paths["checkpoint_dir"],
            features_dir=paths["features_dir"],
            source_commit=source_commit,
            context_path=context_path,
            log_root_train=f"{runtime_root}/logs/merged_train/{run_id}",
            log_root_post=f"{runtime_root}/logs/merged_postprocess/{run_id}",
            log_root_head=f"{runtime_root}/logs/merged_head/{run_id}",
            epochs=int(epochs) if epochs is not None else None,
            subjects_per_class=getattr(args, "subjects_per_class", None),
            head_trials=0,
        )
        plans.append(
            {
                "fold": int(fold),
                "attempt_id": attempt_id,
                "paths": paths,
                "derived_config_remote": derived_config_remote,
                "context_path": context_path,
                "context_payload": context_payload,
                "script": script,
            }
        )

    evidence_dir = PROJECT_ROOT / "outputs" / "exp_submit_merged" / f"{run_id}-{stage}-s{args.seed}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "plan.json").write_text(
        json.dumps(
            [
                {k: v for k, v in plan.items() if k != "script"}
                | {"derived_config_sha256": derived_sha}
                for plan in plans
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for plan in plans:
        (evidence_dir / f"submit_fold_{plan['fold']}.sh").write_text(plan["script"], encoding="utf-8")

    print("=== exp submit-merged contract ===")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "stage": stage,
                "condition": condition,
                "backbone": backbone,
                "seed": int(args.seed),
                "folds": folds,
                "epochs": epochs,
                "deployment_id": deployment.get("deployment_id"),
                "git_commit": source_commit,
                "group_id": group_id,
                "derived_config_sha256": derived_sha,
                "jobs_per_fold": ["train", "postprocess", "head"],
                "optuna_separate_submission": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not execute:
        print("--- submit script (fold 0; identical shape per fold) ---")
        print(plans[0]["script"])
        print("dry-run complete; no mutation performed")
        return 0

    from src.experiment_tracking.deployment import (
        DeploymentError,
        RemoteRunner,
        verify_deployment,
        write_remote_file_once,
    )
    from src.experiment_tracking.submit import SshSubmitRunner

    transfer_runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    try:
        result = verify_deployment(
            transfer_runner,
            deployment["deployment_id"],
            remote_base=REMOTE_BASE,
            expected_git_commit=deployment.get("git_commit"),
            expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
        )
        collisions = _require_remote_absent(
            transfer_runner,
            [plan["paths"]["fold_dir"] for plan in plans]
            + [plan["context_path"] for plan in plans]
            + [plan["derived_config_remote"] for plan in plans],
        )
        if collisions:
            print(f"ERROR: collision(s): {collisions}", file=sys.stderr)
            return 1
        derived_parent = str(Path(plans[0]["derived_config_remote"]).parent)
        proc = transfer_runner.run("mkdir -p " + shlex.quote(derived_parent))
        if proc.returncode != 0:
            raise DeploymentError(f"config dir mkdir failed: {proc.stderr.strip()}")
        write_remote_file_once(
            transfer_runner,
            plans[0]["derived_config_remote"],
            derived_text,
            "derived merged config",
        )
        for plan in plans:
            parent = str(Path(plan["context_path"]).parent)
            proc = transfer_runner.run("mkdir -p " + shlex.quote(parent))
            if proc.returncode != 0:
                raise DeploymentError(f"context mkdir failed: {proc.stderr.strip()}")
            write_remote_file_once(
                transfer_runner,
                plan["context_path"],
                json.dumps(plan["context_payload"], indent=2, sort_keys=True) + "\n",
                "experiment context",
            )
    except Exception as exc:
        print(f"ERROR: submission aborted before sbatch: {exc}", file=sys.stderr)
        return 1

    scheduler_host = getattr(args, "scheduler_host", None) or DEFAULT_SCHEDULER_HOST
    submit_runner = SshSubmitRunner(host=scheduler_host)
    recorded = []
    for plan in plans:
        proc = submit_runner.run_script(plan["script"])
        log_path = evidence_dir / f"submit_fold_{plan['fold']}.log"
        log_path.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
        if proc.returncode != 0:
            print(
                f"ERROR: remote submission failed for fold {plan['fold']}: "
                f"{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        ids = {}
        for line in proc.stdout.splitlines():
            for key, marker in (
                ("train", "Submitted training job:"),
                ("postprocess", "Submitted postprocess job:"),
                ("head", "Submitted head job:"),
            ):
                if line.startswith(marker):
                    ids[key] = line.split(marker, 1)[1].strip()
        missing = {"train", "postprocess", "head"} - set(ids)
        if missing:
            print(
                f"ERROR: incomplete job graph for fold {plan['fold']}; missing {sorted(missing)}; "
                "refusing to record a partial chain",
                file=sys.stderr,
            )
            return 1
        recorded.append((plan, ids))
        print(f"fold {plan['fold']} jobs: {ids}")

    q = shlex.quote
    for plan, ids in recorded:
        for job_key in ("train", "postprocess", "head"):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
                "record-job",
                "--state", str(EXECUTION_LEDGER_PATH),
                "--attempt-id", plan["attempt_id"],
                "--job-key", job_key,
                "--job-type", f"merged_{job_key}",
                "--event-type", "SUBMITTED",
                "--slurm-job-id", ids[job_key],
                "--status", "PENDING",
                "--fold", str(plan["fold"]),
                "--deployment-id", str(deployment.get("deployment_id")),
                "--evaluation-view", ns.EVALUATION_VIEW,
                "--backend", "original_teacher_forced",
                "--aggregation", ns.AGGREGATION,
            ]
            subprocess.run(cmd, capture_output=True, text=True)
    print(f"recorded {len(recorded)} fold chains in the execution ledger")
    return 0

def _cmd_submit_hidden(args) -> int:
    """Managed standalone hidden-state extraction + raw LogReg submission."""

    execute = bool(args.execute)
    dry_run_flag = bool(args.dry_run)
    if execute == dry_run_flag:
        print("ERROR: specify exactly one of --dry-run or --execute", file=sys.stderr)
        return 1
    ctx = _study_submission_context(args, execute=execute)
    if ctx is None:
        return 1
    group_identity = ctx["group_identity"]
    deployment = ctx["deployment"]

    import tools.native_en_submit as ns

    condition = str(args.condition)
    backbone = str(args.backbone)
    if backbone not in ns.BACKBONES or condition not in ns.CONDITIONS:
        print("ERROR: --backbone must be qwen|gemma4 and --condition native|english", file=sys.stderr)
        return 1
    dataset = str(args.dataset)
    run_name = str(args.run_name)
    fold = int(args.fold)
    campaign = ns.campaign_for(condition, backbone)

    cache_paths = ns.standalone_cache_paths(
        dataset=dataset, condition=condition, run_name=run_name, fold=fold
    )
    attempt_dir = ns.standalone_attempt_path(
        campaign=campaign,
        dataset=dataset,
        run_name=run_name,
        fold=fold,
        experiment_id=ns.LOGREG_EXPERIMENT_ID,
    )
    parent_fold_dir = str(args.parent_fold_dir).rstrip("/")
    spec = ns.build_logreg_task_spec(
        family="standalone",
        backend=backbone,
        dataset=dataset,
        modality="text_only",
        condition=f"{condition}_{backbone}",
        fold=fold,
        seed=int(args.seed),
        stage=None,
        cache_dir=cache_paths["cache_dir"],
        group_id=group_identity["experiment_group_id"],
        run_name=run_name,
        branch=str(deployment.get("git_branch_at_deploy") or ""),
        merged_sha=str(deployment.get("git_commit")),
        parent_checkpoint_path=f"{parent_fold_dir}/best_model",
        github_issue=int(os.environ["GITHUB_ISSUE"]) if os.environ.get("GITHUB_ISSUE") else None,
        github_pr=int(os.environ["GITHUB_PR"]) if os.environ.get("GITHUB_PR") else None,
    )
    from src.experiment_tracking.submit import REMOTE_RUNTIME_BASE

    runtime_root = REMOTE_RUNTIME_BASE / ctx["experiment_id"]
    spec_remote = (
        f"{runtime_root}/specs/logreg/{condition}_{backbone}/{dataset}/{run_name}/fold_{fold}/task_spec.json"
    )
    code_path = str(deployment["deployed_code_path"])
    exports = [
        ("PROJECT_ROOT", code_path),
        ("BACKBONE", backbone),
        ("PARENT_FOLD_DIR", parent_fold_dir),
        ("CACHE_DIR", cache_paths["cache_dir"]),
        ("ATTEMPT_DIR", attempt_dir),
        ("TASK_SPEC_PATH", spec_remote),
        ("ENV_ACTIVATE", ns.QWEN_ENV_DEFAULT),
    ]
    if backbone == "gemma4":
        exports.append(("ENV_ACTIVATE", ns.GEMMA_ENV_DEFAULT))
        model_path = getattr(args, "model_path", None) or ns.GEMMA4_MODEL_DEFAULT
        exports.append(("MODEL_PATH", model_path))
    exports.append(
        ("LOG_ROOT", f"{runtime_root}/logs/logreg/{condition}_{backbone}/{dataset}/{run_name}")
    )
    script = ns.render_study_job_script(
        code_path=code_path,
        worker_relpath="scripts/run_native_en_logreg_attempt_slurm.sh",
        job_name=f"nmq-logreg-{dataset}-{fold}",
        exports=exports,
        after_job_ids=[str(args.after_job_id)] if getattr(args, "after_job_id", None) else [],
        echo_label="logreg attempt",
    )
    contract = {
        "attempt_dir": attempt_dir,
        "cache_dir": cache_paths["cache_dir"],
        "parent_fold_dir": parent_fold_dir,
        "backbone": backbone,
        "spec": spec,
        "deployment_id": deployment.get("deployment_id"),
    }
    evidence_dir = PROJECT_ROOT / "outputs" / "exp_submit_hidden" / f"{condition}-{backbone}-{dataset}-{run_name}-f{fold}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (evidence_dir / "submit_script.sh").write_text(script, encoding="utf-8")
    print("=== exp submit-hidden contract ===")
    printable = {k: v for k, v in contract.items() if k != "spec"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(json.dumps({"backend_policy": "harmonized_hidden_logreg_raw_v1", "head_seed": 1337}, indent=2))
    if not execute:
        print("--- remote submit script ---")
        print(script)
        print("dry-run complete; no mutation performed")
        return 0

    from src.experiment_tracking.deployment import verify_deployment, write_remote_file_once
    from src.experiment_tracking.submit import SshSubmitRunner

    transfer_runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    try:
        result = verify_deployment(
            transfer_runner,
            deployment["deployment_id"],
            remote_base=REMOTE_BASE,
            expected_git_commit=deployment.get("git_commit"),
            expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
        )
        collisions = _require_remote_absent(
            transfer_runner, [attempt_dir, cache_paths["cache_dir"], spec_remote]
        )
        if collisions:
            print(f"ERROR: collision(s): {collisions}", file=sys.stderr)
            return 1
        proc = transfer_runner.run("mkdir -p " + shlex.quote(str(Path(spec_remote).parent)))
        if proc.returncode != 0:
            raise Exception(f"spec mkdir failed: {proc.stderr.strip()}")
        write_remote_file_once(
            transfer_runner,
            spec_remote,
            json.dumps(spec, indent=2, sort_keys=True) + "\n",
            "logreg task spec",
        )
    except Exception as exc:
        print(f"ERROR: submission aborted before sbatch: {exc}", file=sys.stderr)
        return 1

    scheduler_host = getattr(args, "scheduler_host", None) or DEFAULT_SCHEDULER_HOST
    submit_runner = SshSubmitRunner(host=scheduler_host)
    proc = submit_runner.run_script(script)
    (evidence_dir / "submit_output.log").write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        print(f"ERROR: remote submission failed: {proc.stderr.strip()}", file=sys.stderr)
        return 1
    marker = "Submitted logreg attempt job:"
    job_id = next((ln.split(marker, 1)[1].strip() for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
    if not job_id:
        print("ERROR: scheduler did not report a job id; refusing to continue", file=sys.stderr)
        return 1
    print(f"submitted logreg attempt job {job_id} (attempt id minted worker-side)")
    return 0


def _cmd_submit_optuna100(args) -> int:
    from src.experiment_tracking.submit import DEFAULT_SCHEDULER_HOST

    """Managed Optuna-100 XGBoost study submission (standalone or merged)."""

    execute = bool(args.execute)
    dry_run_flag = bool(args.dry_run)
    if execute == dry_run_flag:
        print("ERROR: specify exactly one of --dry-run or --execute", file=sys.stderr)
        return 1
    ctx = _study_submission_context(args, execute=execute)
    if ctx is None:
        return 1
    group_identity = ctx["group_identity"]
    deployment = ctx["deployment"]

    import tools.native_en_submit as ns

    family = str(args.family)
    if family not in {"standalone", "merged"}:
        print("ERROR: --family must be standalone|merged", file=sys.stderr)
        return 1
    target_trials = int(getattr(args, "target_trials", 100))
    stage = str(getattr(args, "stage") or "")
    if family == "merged":
        if target_trials == 100 and stage not in {"cv", "final"}:
            print("ERROR: merged production requires --stage cv|final", file=sys.stderr)
            return 1
        if target_trials not in {100, ns.STUDY_OPTUNA_SMOKE_TRIALS}:
            print("ERROR: --target-trials must be 100 (production) or 2 (stage smoke)", file=sys.stderr)
            return 1
        if target_trials == 2 and stage != "smoke":
            print("ERROR: the two-trial target is only valid with --stage smoke", file=sys.stderr)
            return 1
    else:
        # The locked smoke gate exercises Optuna resumability with exactly
        # two completed trials; production requires the full 100.
        if target_trials == 2:
            if stage != "smoke":
                print("ERROR: standalone two-trial studies require --stage smoke", file=sys.stderr)
                return 1
        elif target_trials == 100:
            if stage not in {"", "cv"}:
                print("ERROR: standalone production takes no --stage", file=sys.stderr)
                return 1
            stage = ""
        else:
            print(
                "ERROR: --target-trials must be 100 (production) or 2 with --stage smoke",
                file=sys.stderr,
            )
            return 1

    condition = str(args.condition)
    backbone = str(args.backbone)
    campaign = ns.campaign_for(condition, backbone)
    fold = int(args.fold)
    run_name = str(args.run_name)
    if family == "merged":
        features_dir = str(args.features_dir).rstrip("/")
        dataset = "merged"
        merged_config_remote = str(args.merged_config)
        attempt_dir = ns.merged_attempt_path(
            campaign=campaign,
            run_id=run_name,
            stage=stage,
            fold=fold,
            experiment_id=ns.OPTUNA_EXPERIMENT_ID,
        )
        parent_ckpt = getattr(args, "parent_checkpoint_path", None)
    else:
        dataset = str(args.dataset)
        if getattr(args, "cache_dir", None):
            features_dir = str(args.cache_dir).rstrip("/")
        else:
            features_dir = ns.standalone_cache_paths(
                dataset=dataset, condition=condition, run_name=run_name, fold=fold
            )["cache_dir"]
        merged_config_remote = ""
        attempt_dir = ns.standalone_attempt_path(
            campaign=campaign,
            dataset=dataset,
            run_name=run_name,
            fold=fold,
            experiment_id=ns.OPTUNA_EXPERIMENT_ID,
        )
        parent_ckpt = getattr(args, "parent_checkpoint_path", None)

    # The cache identity check compares spec.condition against the
    # extraction metadata's own record; always take the authoritative value.
    try:
        meta_proc = transfer_runner_probe = None
        from src.experiment_tracking.deployment import RemoteRunner as _RR
        _probe = _RR(host=DEFAULT_TRANSFER_HOST)
        meta_path = f"{features_dir}/extraction_metadata.json"
        mp = _probe.run("cat " + shlex.quote(meta_path))
        if mp.returncode == 0:
            cache_condition = str(json.loads(mp.stdout).get("condition") or "")
            if cache_condition:
                condition = cache_condition
    except Exception:
        pass
    spec = ns.build_optuna_task_spec(
        family=family,
        backend=backbone,
        dataset=dataset,
        modality="text_only",
        condition=condition,
        fold=fold,
        seed=int(args.seed),
        stage=stage or None,
        cache_dir=features_dir,
        group_id=group_identity["experiment_group_id"],
        run_name=run_name,
        branch=str(deployment.get("git_branch_at_deploy") or ""),
        merged_sha=str(deployment.get("git_commit")),
        parent_checkpoint_path=parent_ckpt,
        merged_config=merged_config_remote or None,
        target_trials=target_trials,
        github_issue=int(os.environ["GITHUB_ISSUE"]) if os.environ.get("GITHUB_ISSUE") else None,
        github_pr=int(os.environ["GITHUB_PR"]) if os.environ.get("GITHUB_PR") else None,
    )
    from src.experiment_tracking.submit import REMOTE_RUNTIME_BASE

    runtime_root = REMOTE_RUNTIME_BASE / ctx["experiment_id"]
    spec_tag = getattr(args, "spec_tag", None)
    spec_remote = (
        f"{runtime_root}/specs/optuna100/{family}/{campaign}/{dataset}/{run_name}/"
        f"f{fold}{('_' + stage) if stage else ''}{('/' + spec_tag) if spec_tag else ''}/task_spec.json"
    )
    code_path = str(deployment["deployed_code_path"])
    exports = [
        ("PROJECT_ROOT", code_path),
        ("MODE", family),
        ("ATTEMPT_DIR", attempt_dir),
        ("CACHE_DIR", features_dir),
        ("TASK_SPEC_PATH", spec_remote),
        ("TARGET_TRIALS", str(target_trials)),
        ("LOG_ROOT", f"{runtime_root}/logs/optuna100/{family}/{campaign}/{run_name}-f{fold}{('-' + stage) if stage else ''}"),
    ]
    after_ids = [str(args.after_job_id)] if getattr(args, "after_job_id", None) else []
    if family == "merged":
        exports += [
            ("MERGED_CONFIG", merged_config_remote),
            ("STAGE", stage),
            ("FOLD", str(fold)),
            ("RUN_ID", run_name),
        ]
    script = ns.render_study_job_script(
        code_path=code_path,
        worker_relpath="scripts/run_native_en_optuna100_attempt_slurm.sh",
        job_name=f"nmq-optuna-{dataset}-f{fold}",
        exports=exports,
        after_job_ids=after_ids,
        echo_label="optuna attempt",
    )
    contract = {
        "attempt_dir": attempt_dir,
        "features_dir": features_dir,
        "family": family,
        "target_trials": target_trials,
        "spec": spec,
        "deployment_id": deployment.get("deployment_id"),
    }
    tag = f"{condition}-{backbone}-{dataset}-{run_name}-f{fold}" + (f"-{stage}" if stage else "")
    evidence_dir = PROJECT_ROOT / "outputs" / "exp_submit_optuna100" / tag
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (evidence_dir / "submit_script.sh").write_text(script, encoding="utf-8")
    print("=== exp submit-optuna100 contract ===")
    printable = {k: v for k, v in contract.items() if k != "spec"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    if not execute:
        print("--- remote submit script ---")
        print(script)
        print("dry-run complete; no mutation performed")
        return 0

    from src.experiment_tracking.deployment import verify_deployment, write_remote_file_once
    from src.experiment_tracking.submit import SshSubmitRunner

    transfer_runner = RemoteRunner(host=DEFAULT_TRANSFER_HOST)
    try:
        result = verify_deployment(
            transfer_runner,
            deployment["deployment_id"],
            remote_base=REMOTE_BASE,
            expected_git_commit=deployment.get("git_commit"),
            expected_source_manifest_sha256=deployment.get("source_manifest_sha256"),
        )
        collisions = _require_remote_absent(transfer_runner, [attempt_dir, spec_remote])
        if collisions:
            print(f"ERROR: collision(s): {collisions}", file=sys.stderr)
            return 1
        proc = transfer_runner.run("mkdir -p " + shlex.quote(str(Path(spec_remote).parent)))
        if proc.returncode != 0:
            raise Exception(f"spec mkdir failed: {proc.stderr.strip()}")
        write_remote_file_once(
            transfer_runner,
            spec_remote,
            json.dumps(spec, indent=2, sort_keys=True) + "\n",
            "optuna task spec",
        )
    except Exception as exc:
        print(f"ERROR: submission aborted before sbatch: {exc}", file=sys.stderr)
        return 1

    scheduler_host = getattr(args, "scheduler_host", None) or DEFAULT_SCHEDULER_HOST
    submit_runner = SshSubmitRunner(host=scheduler_host)
    proc = submit_runner.run_script(script)
    (evidence_dir / "submit_output.log").write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        print(f"ERROR: remote submission failed: {proc.stderr.strip()}", file=sys.stderr)
        return 1
    marker = "Submitted optuna attempt job:"
    job_id = next((ln.split(marker, 1)[1].strip() for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
    if not job_id:
        print("ERROR: scheduler did not report a job id; refusing to continue", file=sys.stderr)
        return 1
    print(f"submitted optuna attempt job {job_id} (attempt id minted worker-side)")
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
    contract_holder: dict = {}
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
        contract_holder = chosen
        if attempt_id is None:
            attempt_id = chosen.get("attempt_id")

    local_fold = getattr(args, "output", None)
    if not local_fold:
        if contract_holder.get("local_fold_rel"):
            local_fold = str(PROJECT_ROOT / contract_holder["local_fold_rel"])
        elif attempt_id:
            local_fold = str(PROJECT_ROOT / "output_model" / "collected" / attempt_id / Path(fold_dir).name)
        else:
            print("ERROR: --output required when the local destination cannot be resolved", file=sys.stderr)
            return 1

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

def _resolve_attempt_fold_dir(args):
    """Resolve (fold_dir, contract) from --attempt-id/--fold-dir or latest contract."""
    from src.experiment_tracking.collect import CollectionError, validate_fold_path

    fold_dir = getattr(args, "fold_dir", None)
    attempt_id = getattr(args, "attempt_id", None)
    contract = None
    if not fold_dir:
        submit_root = PROJECT_ROOT / "outputs" / "exp_submit"
        candidates = []
        if attempt_id:
            p = submit_root / attempt_id / "contract.json"
            if p.is_file():
                candidates.append(p)
        elif submit_root.exists():
            candidates = sorted(submit_root.glob("*/contract.json"))
        for contract_path in reversed(candidates):
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if attempt_id and contract.get("attempt_id") != attempt_id:
                continue
            break
        if contract is None:
            return None, None, "no recorded submission contract found; pass --attempt-id or --fold-dir"
        fold_dir = str(PROJECT_ROOT / contract["local_fold_rel"])
        attempt_id = contract.get("attempt_id")
    try:
        validate_fold_path(str(fold_dir))
    except CollectionError:
        pass  # local dirs may legitimately differ; remote check happens in collect
    return fold_dir, contract, None


def _cmd_validate(args) -> int:
    from src.experiment_tracking.validate import (
        ValidationError,
        advance_lifecycle,
        read_state,
        validate_attempt,
    )

    fold_dir, contract, err = _resolve_attempt_fold_dir(args)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    qualifiers = (contract or {}).get("qualifiers", {})
    try:
        result = validate_attempt(
            fold_dir,
            expected_attempt_id=(contract or {}).get("attempt_id"),
            expected_dataset=(contract or {}).get("dataset"),
            expected_evaluation_view=qualifiers.get("evaluation_view"),
            expected_backend=qualifiers.get("backend"),
            expected_aggregation=qualifiers.get("aggregation"),
            require_standalone_eval=True,
        )
    except ValidationError as e:
        print(f"VALIDATE FAILED: {e}", file=sys.stderr)
        return 1
    state, _ = read_state(fold_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        print("VALIDATE FAILED", file=sys.stderr)
        return 1
    # Official local verification of artifacts and evaluations (sets the
    # locally_verified flags the registry importer requires).
    from src.experiment_tracking.evidence import (
        verify_artifacts_locally,
        verify_evaluations_locally,
    )
    art = verify_artifacts_locally(fold_dir)
    evs = verify_evaluations_locally(fold_dir)
    print(f"verified artifacts: {art.get('verified_artifacts', '?')}/{art.get('total_artifacts', '?')}; "
          f"evaluations: {evs.get('verified_evaluations', '?')}/{evs.get('total_evaluations', '?')}")
    # Stepwise official advancement: COMPLETED_ON_MN5 -> SYNCED_LOCALLY -> LOCALLY_VALIDATED
    advanced = []
    try:
        if state == "COMPLETED_ON_MN5":
            advanced.append(advance_lifecycle(fold_dir, "SYNCED_LOCALLY"))
            state = "SYNCED_LOCALLY"
        if state == "SYNCED_LOCALLY":
            advanced.append(advance_lifecycle(fold_dir, "LOCALLY_VALIDATED"))
            state = "LOCALLY_VALIDATED"
    except ValidationError as e:
        print(f"VALIDATE FAILED: {e}", file=sys.stderr)
        return 1
    if advanced:
        print(f"lifecycle advanced: {' -> '.join([state.split('->')[0].strip()] + advanced)}")
    print(f"VALIDATE OK (state: {state})")
    return 0


def _cmd_finish(args) -> int:
    from src.experiment_tracking.validate import finish_gates

    fold_dir, contract, err = _resolve_attempt_fold_dir(args)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    qualifiers = (contract or {}).get("qualifiers", {})
    result = finish_gates(
        fold_dir,
        expected_attempt_id=(contract or {}).get("attempt_id"),
        expected_dataset=(contract or {}).get("dataset"),
        expected_evaluation_view=qualifiers.get("evaluation_view"),
        expected_backend=qualifiers.get("backend"),
        expected_aggregation=qualifiers.get("aggregation"),
        required_jobs_terminal_success=not getattr(args, "skip_job_gate", False),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        print(f"FINISH INCOMPLETE — next action: {result['next_action']}", file=sys.stderr)
        return 1
    print(f"FINISH OK: {result['state']}")
    return 0

def _cmd_compare(args) -> int:
    from src.experiment_tracking.compare import (
        ComparisonError,
        compare_group,
        write_comparison_audit,
    )

    attempts = [a.strip() for a in args.attempts.split(",") if a.strip()]
    try:
        audit = compare_group(
            PROJECT_ROOT,
            group_id=args.group,
            attempt_ids=attempts,
            dataset=args.dataset,
            metric=args.metric,
            namespace=args.namespace,
            backend=args.backend,
            view=args.view,
            aggregation=args.aggregation,
            tie_rule=args.tie_rule,
            tie_tolerance=args.tie_tolerance,
        )
    except ComparisonError as e:
        print(f"COMPARE REFUSED: {e}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "outputs" / "comparisons" / f"{args.group}_{args.dataset}_{args.metric}.json"
    )
    write_comparison_audit(audit, output)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"audit written: {output}")
    return 0


def _cmd_plan_integration(args) -> int:
    from src.experiment_tracking.compare import ComparisonError, plan_integration

    repo = Path(args.repo) if getattr(args, "repo", None) else PROJECT_ROOT
    try:
        plan = plan_integration(
            repo,
            branch_a=args.branch_a,
            branch_b=args.branch_b,
            base=args.base,
        )
    except ComparisonError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "outputs" / "integration_plans" / f"{Path(args.branch_a).name}__{Path(args.branch_b).name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(f"plan written: {output}")
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
    create_parser.add_argument("--tier", type=int, choices=[1,2], required=True, help="Tier 1=competing, 2=complementary")
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

    validate_parser = subparsers.add_parser("validate", help="verify local evidence hashes, recompute headline metrics, check qualifiers, advance lifecycle stepwise")
    validate_parser.add_argument("slug", nargs="?", default=None)
    validate_parser.add_argument("--attempt-id", default=None)
    validate_parser.add_argument("--fold-dir", default=None, help="local fold dir with collected evidence")
    validate_parser.set_defaults(func=_cmd_validate)

    compare_parser = subparsers.add_parser("compare", help="group-scoped comparison with full qualifiers and deterministic audit")
    compare_parser.add_argument("--group", required=True, help="group ID")
    compare_parser.add_argument("--attempts", required=True, help="comma-separated attempt IDs (all must be REPORTABLE)")
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--metric", required=True)
    compare_parser.add_argument("--namespace", required=True)
    compare_parser.add_argument("--backend", required=True)
    compare_parser.add_argument("--view", required=True)
    compare_parser.add_argument("--aggregation", required=True)
    compare_parser.add_argument("--tie-rule", choices=["max", "min"], default="max")
    compare_parser.add_argument("--tie-tolerance", type=float, default=0.0)
    compare_parser.add_argument("--output", default=None, help="audit output path")
    compare_parser.set_defaults(func=_cmd_compare)

    planint_parser = subparsers.add_parser("plan-integration", help="read-only integration plan: ancestry, Git conflicts, semantic-conflict candidates")
    planint_parser.add_argument("--branch-a", required=True)
    planint_parser.add_argument("--branch-b", required=True)
    planint_parser.add_argument("--base", default="origin/main")
    planint_parser.add_argument("--repo", default=None, help="repository path (default: this repo)")
    planint_parser.add_argument("--output", default=None)
    planint_parser.set_defaults(func=_cmd_plan_integration)

    finish_parser = subparsers.add_parser("finish", help="gate orchestrator: advance to REPORTABLE only when every gate passes")
    finish_parser.add_argument("slug", nargs="?", default=None)
    finish_parser.add_argument("--attempt-id", default=None)
    finish_parser.add_argument("--fold-dir", default=None, help="local fold dir with collected evidence")
    finish_parser.add_argument("--skip-job-gate", action="store_true", help="test-only: bypass TERMINAL job-event gate")
    finish_parser.set_defaults(func=_cmd_finish)


    merged_parser = subparsers.add_parser(
        "submit-merged",
        help="submit a managed merged train->postprocess->head chain (dry-run first)",
    )
    merged_parser.add_argument("slug")
    merged_parser.add_argument("--config", required=True)
    merged_parser.add_argument("--stage", required=True, choices=("smoke", "cv", "final"))
    merged_parser.add_argument("--run-id", required=True)
    merged_parser.add_argument("--seed", type=int, required=True)
    merged_parser.add_argument("--condition", required=True, choices=("native", "english"))
    merged_parser.add_argument("--backbone", required=True, choices=("qwen", "gemma4"))
    merged_parser.add_argument("--folds", default=None, help="comma list; defaults per stage")
    merged_parser.add_argument("--epochs", type=int, default=None, help="final stage only; derived from CV evidence when omitted on execute")
    merged_parser.add_argument("--subjects-per-class", type=int, default=None)
    merged_parser.add_argument("--cv-run-id-for-epochs", default=None)
    merged_parser.add_argument("--cv-condition-for-epochs", default=None)
    merged_parser.add_argument("--deployment-id", default=None)
    merged_parser.add_argument("--group-id", default=None)
    merged_parser.add_argument("--scheduler-host", default=None)
    merged_parser.add_argument("--supersedes-attempt-id", default=None)
    merged_group = merged_parser.add_mutually_exclusive_group(required=True)
    merged_group.add_argument("--dry-run", action="store_true")
    merged_group.add_argument("--execute", action="store_true")
    merged_parser.set_defaults(func=_cmd_submit_merged)

    hidden_parser = subparsers.add_parser(
        "submit-hidden",
        help="submit one standalone extract+LogReg attempt job (dry-run first)",
    )
    hidden_parser.add_argument("slug")
    hidden_parser.add_argument("--parent-fold-dir", required=True)
    hidden_parser.add_argument("--dataset", required=True)
    hidden_parser.add_argument("--condition", required=True, choices=("native", "english"))
    hidden_parser.add_argument("--backbone", required=True, choices=("qwen", "gemma4"))
    hidden_parser.add_argument("--run-name", required=True)
    hidden_parser.add_argument("--fold", type=int, required=True)
    hidden_parser.add_argument("--seed", type=int, required=True)
    hidden_parser.add_argument("--model-path", default=None)
    hidden_parser.add_argument("--after-job-id", default=None)
    hidden_parser.add_argument("--deployment-id", default=None)
    hidden_parser.add_argument("--group-id", default=None)
    hidden_parser.add_argument("--scheduler-host", default=None)
    hidden_group = hidden_parser.add_mutually_exclusive_group(required=True)
    hidden_group.add_argument("--dry-run", action="store_true")
    hidden_group.add_argument("--execute", action="store_true")
    hidden_parser.set_defaults(func=_cmd_submit_hidden)

    optuna_parser = subparsers.add_parser(
        "submit-optuna100",
        help="submit one Optuna-100 XGBoost study attempt job (dry-run first)",
    )
    optuna_parser.add_argument("slug")
    optuna_parser.add_argument("--family", required=True, choices=("standalone", "merged"))
    optuna_parser.add_argument("--condition", required=True, choices=("native", "english"))
    optuna_parser.add_argument("--backbone", required=True, choices=("qwen", "gemma4"))
    optuna_parser.add_argument("--dataset", default=None)
    optuna_parser.add_argument("--run-name", required=True)
    optuna_parser.add_argument("--fold", type=int, required=True)
    optuna_parser.add_argument("--seed", type=int, required=True)
    optuna_parser.add_argument("--features-dir", default=None, help="merged family: remote features dir")
    optuna_parser.add_argument("--merged-config", default=None, help="merged family: remote derived config path")
    optuna_parser.add_argument("--stage", default="cv", choices=("", "smoke", "cv", "final"))
    optuna_parser.add_argument("--target-trials", type=int, default=100)
    optuna_parser.add_argument("--parent-checkpoint-path", default=None)
    optuna_parser.add_argument("--after-job-id", default=None)
    optuna_parser.add_argument("--deployment-id", default=None)
    optuna_parser.add_argument("--group-id", default=None)
    optuna_parser.add_argument("--scheduler-host", default=None)
    optuna_parser.add_argument("--cache-dir", default=None,
                               help="standalone family: explicit remote hidden-features dir")
    optuna_parser.add_argument("--spec-tag", default=None, help="disambiguates re-submissions whose prior spec file exists")
    optuna_group = optuna_parser.add_mutually_exclusive_group(required=True)
    optuna_group.add_argument("--dry-run", action="store_true")
    optuna_group.add_argument("--execute", action="store_true")
    optuna_parser.set_defaults(func=_cmd_submit_optuna100)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
