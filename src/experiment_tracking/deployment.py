"""Immutable MN5 source deployments.

Real execution path: plan -> dry-run rsync -> remote absence checks -> rsync
transfer (never --delete) -> remote manifest/hash verification -> one-time
deployment.json creation -> writable runtime directories.

All remote operations go through :class:`RemoteRunner` so tests can inject a
fake runner that records the exact SSH argv and returns canned results.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = "audiollm.deployment.v1"
SOURCE_MANIFEST_SCHEMA = "audiollm.source_manifest.v1"
DEFAULT_TRANSFER_HOST = "ozu647717@transfer1.bsc.es"
REMOTE_BASE = Path("/gpfs/projects/etur92/ozu647717/AudioLLM")
ALLOWED_REMOTE_EXTRAS_PREFIXES = (".provenance/",)


class DeploymentError(RuntimeError):
    """Raised when a deployment or verification step must fail closed."""


class RemoteCommandError(DeploymentError):
    """Raised when an SSH command itself fails (not an expected nonzero)."""


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_commit(cwd: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_git_branch(cwd: Path) -> str:
    result = subprocess.run(["git", "branch", "--show-current"], cwd=str(cwd), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def is_clean(cwd: Path) -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=str(cwd), capture_output=True, text=True, check=True)
    return result.stdout.strip() == ""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_of_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_source_manifest(cwd: Path) -> dict[str, Any]:
    """Build the deterministic tracked-file manifest from a worktree."""
    result = subprocess.run(["git", "-C", str(cwd), "ls-files"], capture_output=True, text=True, check=True)
    records = []
    for relative in sorted(line for line in result.stdout.splitlines() if line.strip()):
        path = cwd / relative
        if not path.is_file():
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": relative, "sha256": sha, "size_bytes": path.stat().st_size})
    return {"schema_version": SOURCE_MANIFEST_SCHEMA, "file_count": len(records), "files": records}


def load_source_manifest(cwd: Path) -> dict[str, Any]:
    prov = cwd / ".provenance" / "source_manifest.json"
    if not prov.exists():
        raise DeploymentError(f"missing {prov}; run scripts/capture_provenance.sh first")
    data = json.loads(prov.read_text(encoding="utf-8"))
    if data.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise DeploymentError(f"unexpected source manifest schema in {prov}")
    return data


def get_source_manifest_hash(cwd: Path) -> str:
    try:
        manifest = load_source_manifest(cwd)
        return sha256_of_json(manifest)
    except DeploymentError:
        pass
    except Exception:
        pass
    return sha256_of_json(build_source_manifest(cwd))


def generate_deployment_id(experiment_id: str, git_commit: str, timestamp: str | None = None) -> str:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    short = git_commit[:8] if len(git_commit) >= 8 else git_commit
    safe_exp = "".join(c if c.isalnum() or c in "-_" else "-" for c in experiment_id)
    return f"{safe_exp}-{timestamp}-{short}-{suffix}"


def build_deployment_record(
    deployment_id: str,
    experiment_id: str,
    git_commit: str,
    git_branch_at_deploy: str,
    git_dirty: bool,
    source_manifest_sha256: str,
    deployed_code_path: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if created_at_utc is None:
        created_at_utc = utc_now_str()
    uncommitted_patch_sha256 = (
        hashlib.sha256(b"dirty").hexdigest() if git_dirty else hashlib.sha256(b"").hexdigest()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "deployment_id": deployment_id,
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "git_branch_at_deploy": git_branch_at_deploy,
        "git_dirty": git_dirty,
        "source_manifest_sha256": source_manifest_sha256,
        "uncommitted_patch_sha256": uncommitted_patch_sha256,
        "deployed_code_path": deployed_code_path,
        "created_at_utc": created_at_utc,
    }


def validate_deployment_paths(deployment_code_path: Path, runtime_root: Path) -> list[str]:
    errors = []
    try:
        dep_res = deployment_code_path.resolve()
        run_res = runtime_root.resolve()
        if dep_res == run_res:
            errors.append("deployment code path and runtime root are same")
        elif is_inside(dep_res, run_res):
            errors.append(f"runtime root {run_res} is inside deployment code {dep_res}")
    except Exception as e:
        errors.append(str(e))
    return errors


def is_inside(parent: Path, child: Path) -> bool:
    try:
        return parent == child or parent in child.resolve().parents
    except Exception:
        return False


def assert_no_delete(argv: Iterable[str]) -> None:
    for token in argv:
        if token == "--delete" or token.startswith("--delete="):
            raise DeploymentError("rsync --delete is forbidden")


def build_rsync_command(
    local_project: Path,
    remote_host: str,
    remote_deployment_path: Path,
    dry_run: bool = True,
    files_from: Path | None = None,
) -> list[str]:
    """Manifest-driven transfer command.

    When ``files_from`` is given (a newline-separated list of relative paths),
    rsync copies exactly those files so the deployed tree matches the source
    manifest by construction. Otherwise it falls back to gitignore filtering.
    """
    cmd = ["rsync", "-avh", "--itemize-changes"]
    if dry_run:
        cmd.append("-n")
    if files_from is not None:
        cmd.extend(["--files-from=" + str(files_from), str(local_project) + "/"])
    else:
        cmd.extend(
            [
                "--include=.provenance/***",
                "--filter=:- .gitignore",
                "--exclude=.git/",
                "--exclude=.git",
                str(local_project) + "/",
            ]
        )
    cmd.append(f"{remote_host}:{remote_deployment_path}/")
    assert_no_delete(cmd)
    return cmd


def estimate_transfer_bytes(manifest: dict[str, Any]) -> int:
    return int(sum(int(f.get("size_bytes", 0)) for f in manifest.get("files", [])))


class RemoteRunner:
    """Executes shell snippets on a remote host over non-interactive SSH."""

    def __init__(
        self,
        host: str = DEFAULT_TRANSFER_HOST,
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
        ssh_opts: list[str] | None = None,
    ) -> None:
        self.host = host
        self._runner = runner
        self.ssh_opts = ssh_opts or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

    def run(self, script: str, input: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
        argv = ["ssh", *self.ssh_opts, self.host, script]
        if self._runner is not None:
            return self._runner(argv)
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=input)


def remote_path_exists(runner: RemoteRunner, path: Path | str) -> bool:
    proc = runner.run(f"test -e {shlex.quote(str(path))}")
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise RemoteCommandError(f"ssh existence check failed rc={proc.returncode}: {proc.stderr.strip()}")


def require_remote_absent(runner: RemoteRunner, path: Path | str, label: str) -> None:
    if remote_path_exists(runner, path):
        raise DeploymentError(f"{label} already exists on remote: {path}; new-target-only deployment refused")


def run_local_rsync(argv: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    assert_no_delete(argv)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def verify_remote_tree(
    runner: RemoteRunner,
    code_path: Path,
    manifest: dict[str, Any],
    strict_unexpected: bool = True,
) -> dict[str, Any]:
    """Verify the deployed tree against the source manifest.

    Checks every manifest file exists remotely with matching SHA-256 and, when
    ``strict_unexpected`` is true, fails on files outside the manifest (except
    ``.provenance/**``).
    """
    expected = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
    if not expected:
        raise DeploymentError("source manifest contains no files")

    quoted_code = shlex.quote(str(code_path))
    inv_proc = runner.run(
        f"cd {quoted_code} && find . -type f -printf '%P\\n' | LC_ALL=C sort"
    )
    if inv_proc.returncode != 0:
        raise RemoteCommandError(f"remote inventory failed: {inv_proc.stderr.strip()}")
    actual_paths = [line for line in inv_proc.stdout.splitlines() if line.strip()]
    actual_set = set(actual_paths)
    missing = sorted(set(expected) - actual_set)
    allowed_extras = {
        p for p in actual_set - set(expected)
        if any(p.startswith(prefix) for prefix in ALLOWED_REMOTE_EXTRAS_PREFIXES)
    }
    unexpected = sorted((actual_set - set(expected)) - allowed_extras)

    mismatched: list[dict[str, str]] = []
    paths_sorted = sorted(expected)
    chunk_size = 200
    for start in range(0, len(paths_sorted), chunk_size):
        chunk = paths_sorted[start:start + chunk_size]
        script = f"cd {quoted_code} && sha256sum -- " + " ".join(shlex.quote(p) for p in chunk)
        hash_proc = runner.run(script)
        remote_hashes: dict[str, str] = {}
        for line in hash_proc.stdout.splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            if not rel:
                # A filename containing two spaces; re-split conservatively.
                parts = line.split("  ", 1)
                if len(parts) != 2:
                    continue
                digest, rel = parts[0], parts[1]
            remote_hashes[rel] = digest.strip()
        # sha256sum exits 1 when any listed file is missing; that is handled
        # below as a missing-file finding, not a command failure.
        if hash_proc.returncode not in (0, 1):
            raise RemoteCommandError(f"remote sha256sum failed: {hash_proc.stderr.strip()}")
        for rel in chunk:
            got = remote_hashes.get(rel)
            if got is None:
                mismatched.append({"path": rel, "reason": "missing_from_hash_output"})
            elif got != expected[rel]:
                mismatched.append({"path": rel, "reason": "hash_mismatch"})

    result = {
        "expected_files": len(expected),
        "verified_files": len(expected) - len(mismatched),
        "missing": missing,
        "unexpected": unexpected,
        "allowed_extras": sorted(allowed_extras),
        "mismatched": mismatched,
        "ok": not missing and not mismatched and (not strict_unexpected or not unexpected),
    }
    if missing or mismatched:
        raise DeploymentError(
            f"deployed tree does not match source manifest: "
            f"{len(missing)} missing, {len(mismatched)} mismatched; "
            f"first problems: {(missing + [m['path'] for m in mismatched])[:5]}"
        )
    if strict_unexpected and unexpected:
        raise DeploymentError(f"unexpected files in clean-source deployment: {unexpected[:5]}")
    return result


def write_remote_file_once(runner: RemoteRunner, path: Path | str, content: str, label: str) -> None:
    """Create a remote file only if it does not exist (new-target-only)."""
    require_remote_absent(runner, path, label)
    target = shlex.quote(str(path))
    tmp = shlex.quote(str(path) + ".tmp")
    proc = runner.run(f"cat > {tmp} && mv {tmp} {target}", input=content)
    if proc.returncode != 0:
        raise RemoteCommandError(f"failed to create {label} at {path}: {proc.stderr.strip()}")


def read_remote_file(runner: RemoteRunner, path: Path | str) -> str:
    proc = runner.run(f"cat {shlex.quote(str(path))}")
    if proc.returncode != 0:
        raise RemoteCommandError(f"failed to read {path}: {proc.stderr.strip()}")
    return proc.stdout


def create_runtime_dirs(runner: RemoteRunner, runtime_root: Path | str) -> list[str]:
    subdirs = [
        str(runtime_root),
        f"{runtime_root}/contexts",
        f"{runtime_root}/manifests",
        f"{runtime_root}/splits",
        f"{runtime_root}/logs/slurm_train",
        f"{runtime_root}/logs/slurm_eval",
    ]
    script = "mkdir -p " + " ".join(shlex.quote(d) for d in subdirs)
    proc = runner.run(script)
    if proc.returncode != 0:
        raise RemoteCommandError(f"failed to create runtime dirs: {proc.stderr.strip()}")
    return subdirs


def plan_deployment(
    worktree: Path,
    experiment_id: str,
    branch: str,
    allow_dirty: bool,
    transfer_host: str = DEFAULT_TRANSFER_HOST,
    remote_base: Path = REMOTE_BASE,
    deployment_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the complete deployment plan without touching the network."""
    dirty = not is_clean(worktree)
    if dirty and not allow_dirty:
        raise DeploymentError(
            "dirty production source not allowed; commit first or use --allow-dirty for non-reportable smoke/debug"
        )
    commit = get_git_commit(worktree)
    manifest = load_source_manifest(worktree)
    manifest_hash = sha256_of_json(manifest)
    if deployment_id is None:
        deployment_id = generate_deployment_id(experiment_id, commit, timestamp=None)
    deployment_dir = remote_base / "deployments" / deployment_id
    code_path = deployment_dir / "code"
    record_path = deployment_dir / "deployment.json"
    runtime_root = remote_base / "experiment_runtime" / experiment_id
    path_errors = validate_deployment_paths(code_path, runtime_root)
    if path_errors:
        raise DeploymentError("; ".join(path_errors))
    # Manifest-driven file list: every tracked file plus the .provenance
    # snapshot, so the deployed tree matches the manifest exactly.
    provenance_dir = worktree / ".provenance"
    if not provenance_dir.is_dir():
        raise DeploymentError(f"missing {provenance_dir}; run scripts/capture_provenance.sh first")
    transfer_list = [f["path"] for f in manifest.get("files", [])]
    transfer_list += sorted(
        str(p.relative_to(worktree)) for p in provenance_dir.rglob("*") if p.is_file()
    )
    if not transfer_list:
        raise DeploymentError("empty transfer list")
    files_from = worktree / ".provenance" / "deployment_files_from.txt"
    files_from.write_text("\n".join(transfer_list) + "\n", encoding="utf-8")
    record = build_deployment_record(
        deployment_id=deployment_id,
        experiment_id=experiment_id,
        git_commit=commit,
        git_branch_at_deploy=branch,
        git_dirty=dirty,
        source_manifest_sha256=manifest_hash,
        deployed_code_path=str(code_path),
        created_at_utc=created_at_utc,
    )
    return {
        "experiment_id": experiment_id,
        "branch": branch,
        "worktree": str(worktree),
        "allow_dirty": allow_dirty,
        "reportable_allowed": not dirty,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_manifest_sha256": manifest_hash,
        "source_manifest_file_count": manifest.get("file_count", len(manifest.get("files", []))),
        "deployment_id": deployment_id,
        "remote_base": str(remote_base),
        "deployment_dir": str(deployment_dir),
        "deployed_code_path": str(code_path),
        "deployment_record_path": str(record_path),
        "runtime_root": str(runtime_root),
        "transfer_host": transfer_host,
        "rsync_dry_run_argv": build_rsync_command(worktree, transfer_host, code_path, dry_run=True, files_from=files_from),
        "rsync_execute_argv": build_rsync_command(worktree, transfer_host, code_path, dry_run=False, files_from=files_from),
        "estimated_transfer_bytes": estimate_transfer_bytes(manifest),
        "record": record,
    }


def execute_deployment(
    plan: dict[str, Any],
    runner: RemoteRunner,
    rsync_executor: Callable[[list[str]], subprocess.CompletedProcess] = run_local_rsync,
    strict_unexpected: bool | None = None,
) -> dict[str, Any]:
    """Execute a reviewed plan: absence checks, rsync, verification, record.

    The deployment record is written only after the transferred tree verifies.
    A failed transfer therefore cannot leave a valid deployment record.
    """
    code_path = Path(plan["deployed_code_path"])
    record_path = Path(plan["deployment_record_path"])
    deployment_dir = Path(plan["deployment_dir"])
    if strict_unexpected is None:
        strict_unexpected = not plan["git_dirty"]

    require_remote_absent(runner, deployment_dir, "deployment directory")
    worktree = Path(plan["worktree"])
    manifest = load_source_manifest(worktree)

    # rsync cannot create multiple missing path components; create the
    # deployment directory itself, then let rsync build code/ inside it.
    mk = runner.run(f"mkdir -p {shlex.quote(str(deployment_dir))}")
    if mk.returncode != 0:
        raise DeploymentError(f"failed to create deployment directory: {mk.stderr.strip()}")

    dry_argv = list(plan["rsync_dry_run_argv"])
    assert_no_delete(dry_argv)
    dry_proc = rsync_executor(dry_argv)
    if dry_proc.returncode != 0:
        raise DeploymentError(f"rsync dry-run failed: {dry_proc.stderr.strip()}")

    exec_argv = list(plan["rsync_execute_argv"])
    assert_no_delete(exec_argv)
    exec_proc = rsync_executor(exec_argv)
    if exec_proc.returncode != 0:
        raise DeploymentError(f"rsync transfer failed: {exec_proc.stderr.strip()}")

    verification = verify_remote_tree(runner, code_path, manifest, strict_unexpected=strict_unexpected)

    record = plan["record"]
    write_remote_file_once(
        runner,
        record_path,
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        "deployment record",
    )
    runtime_subdirs = create_runtime_dirs(runner, plan["runtime_root"])

    return {
        "deployment_id": plan["deployment_id"],
        "record": record,
        "verification": verification,
        "runtime_subdirs": runtime_subdirs,
        "rsync_dry_run_stdout_lines": len(dry_proc.stdout.splitlines()),
        "rsync_execute_stdout_lines": len(exec_proc.stdout.splitlines()),
    }


def verify_deployment(
    runner: RemoteRunner,
    deployment_id: str,
    remote_base: Path = REMOTE_BASE,
    expected_git_commit: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Read-only post-deploy verification: identity, drift, hashes."""
    deployment_dir = remote_base / "deployments" / deployment_id
    record_path = deployment_dir / "deployment.json"
    code_path = deployment_dir / "code"
    if not remote_path_exists(runner, record_path):
        raise DeploymentError(f"deployment record missing: {record_path}")
    raw = read_remote_file(runner, record_path)
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DeploymentError(f"deployment record is not valid JSON: {e}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise DeploymentError(f"unexpected deployment record schema: {record.get('schema_version')}")
    if record.get("deployment_id") != deployment_id:
        raise DeploymentError(
            f"identity mismatch: record deployment_id {record.get('deployment_id')!r} != requested {deployment_id!r}"
        )
    if record.get("deployed_code_path") != str(code_path):
        raise DeploymentError(
            f"identity mismatch: record deployed_code_path {record.get('deployed_code_path')!r} != {str(code_path)!r}"
        )
    if expected_git_commit is not None and record.get("git_commit") != expected_git_commit:
        raise DeploymentError(
            f"identity mismatch: record git_commit {record.get('git_commit')!r} != expected {expected_git_commit!r}"
        )

    manifest_raw = read_remote_file(runner, code_path / ".provenance" / "source_manifest.json")
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as e:
        raise DeploymentError(f"remote source manifest is not valid JSON: {e}")
    manifest_hash = sha256_of_json(manifest)
    if manifest_hash != record.get("source_manifest_sha256"):
        raise DeploymentError(
            f"manifest hash drift: remote manifest {manifest_hash} != record {record.get('source_manifest_sha256')}"
        )
    if expected_source_manifest_sha256 is not None and manifest_hash != expected_source_manifest_sha256:
        raise DeploymentError(
            f"manifest hash drift: remote manifest {manifest_hash} != expected {expected_source_manifest_sha256}"
        )

    tree = verify_remote_tree(runner, code_path, manifest, strict_unexpected=True)
    return {
        "deployment_id": deployment_id,
        "record": record,
        "source_manifest_sha256": manifest_hash,
        "tree_verification": tree,
        "ok": True,
    }
