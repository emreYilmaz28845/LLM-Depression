from __future__ import annotations
import hashlib
import json
import subprocess
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "audiollm.deployment.v1"

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

def get_source_manifest_hash(cwd: Path) -> str:
    prov = cwd / ".provenance" / "source_manifest.json"
    if prov.exists():
        try:
            data = json.loads(prov.read_text(encoding="utf-8"))
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except Exception:
            pass
    result = subprocess.run(["git", "-C", str(cwd), "ls-files"], capture_output=True, text=True, check=True)
    records = []
    for relative in sorted(line for line in result.stdout.splitlines() if line.strip()):
        path = cwd / relative
        if not path.is_file():
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": relative, "sha256": sha, "size_bytes": path.stat().st_size})
    payload = {"schema_version": "audiollm.source_manifest.v1", "file_count": len(records), "files": records}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
    if git_dirty:
        uncommitted_patch_sha256 = hashlib.sha256(b"dirty").hexdigest()
    else:
        uncommitted_patch_sha256 = hashlib.sha256(b"").hexdigest()
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

def build_rsync_command(
    local_project: Path,
    remote_host: str,
    remote_deployment_path: Path,
    dry_run: bool = True,
) -> list[str]:
    cmd = ["rsync", "-avh", "--itemize-changes", "--relative"]
    if dry_run:
        cmd.append("-n")
    cmd.extend([str(local_project) + "/", f"{remote_host}:{remote_deployment_path}/"])
    return cmd

def check_deployment_target_exists(remote_path: Path) -> bool:
    return remote_path.exists()
