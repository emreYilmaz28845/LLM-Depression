"""Real compact-evidence collection from MN5 to local.

Dry-run emits the exact rsync plan and inventory. Execute transfers through
transfer1 without --delete, refuses incompatible local overwrites, preserves
best_model/standalone_eval evidence while excluding adapter weights, and
verifies remote/local hash agreement for the compact set.
"""
from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

DEFAULT_TRANSFER_HOST = "ozu647717@transfer1.bsc.es"

PLACEHOLDER_PATTERN = re.compile(r"<[a-z_][a-z0-9_]*>")

# Ordered rsync filters: includes before excludes; standalone_eval survives
# the best_model exclusion. No --delete is ever generated.
FILTER_RULES: list[str] = [
    "--include=*/",
    "--include=run_config.yaml",
    "--include=metadata.json",
    "--include=status.json",
    "--include=jobs.jsonl",
    "--include=artifacts.json",
    "--include=evaluations.json",
    "--include=final_summary.json",
    "--include=predictions_subject_level.csv",
    "--include=predictions_subject_level.json",
    "--include=*.audit.json",
    "--include=eval/**",
    "--include=best_model/standalone_eval/**",
    "--include=logs/*.json",
    "--include=logs/*.jsonl",
    "--exclude=best_model/**",
    "--exclude=last_model/**",
    "--exclude=*",
]

REQUIRED_EVIDENCE_FILES = [
    "run_config.yaml",
    "metadata.json",
    "status.json",
    "jobs.jsonl",
    "artifacts.json",
    "evaluations.json",
]


class CollectionError(RuntimeError):
    """Raised when collection must fail closed."""


def validate_fold_path(remote_fold: str) -> None:
    if PLACEHOLDER_PATTERN.search(remote_fold):
        raise CollectionError(
            f"remote fold path contains an unresolved placeholder: {remote_fold}; "
            "execute mode requires exact paths from recorded deployment/attempt evidence"
        )
    if not re.search(r"fold_\d+$", remote_fold.rstrip("/")):
        raise CollectionError(f"remote fold path must end in fold_<n>: {remote_fold}")


def build_collect_argv(
    source: str,
    local_fold: str,
    dry_run: bool,
    remote_host: str | None = DEFAULT_TRANSFER_HOST,
) -> list[str]:
    argv = ["rsync", "-avh", "--itemize-changes", "--prune-empty-dirs"]
    if dry_run:
        argv.append("-n")
    argv.extend(FILTER_RULES)
    if remote_host:
        argv.append(f"{remote_host}:{source.rstrip('/')}/")
    else:
        argv.append(source.rstrip("/") + "/")
    argv.append(local_fold.rstrip("/") + "/")
    for token in argv:
        if token == "--delete" or token.startswith("--delete="):
            raise CollectionError("rsync --delete is forbidden")
    return argv


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def local_inventory(root: Path) -> dict[str, str]:
    inv: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            inv[str(p.relative_to(root))] = sha256_file(p)
    return inv


class RemoteRunner:
    def __init__(
        self,
        host: str = DEFAULT_TRANSFER_HOST,
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.host = host
        self._runner = runner

    def run(self, script: str, timeout: int = 300) -> subprocess.CompletedProcess:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.host, script]
        if self._runner is not None:
            return self._runner(argv)
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def remote_inventory(runner: RemoteRunner, root: str) -> dict[str, str]:
    """Hash every file under the remote fold (compact tree is small)."""
    q = shlex.quote(root)
    proc = runner.run(f"cd {q} && find . -type f -printf '%P\\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum --")
    if proc.returncode != 0:
        raise CollectionError(f"remote inventory failed: {proc.stderr.strip()}")
    inv: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        digest, _, rel = line.partition("  ")
        if rel:
            inv[rel] = digest.strip()
    return inv


def check_local_overwrite_safety(remote_inv: dict[str, str], local_fold: Path) -> None:
    """Refuse incompatible overwrites: same relative path with different content."""
    if not local_fold.exists():
        return
    local_inv = local_inventory(local_fold)
    conflicts = [
        rel for rel, digest in remote_inv.items()
        if rel in local_inv and local_inv[rel] != digest
    ]
    if conflicts:
        raise CollectionError(
            "incompatible local overwrite refused; differing files: "
            + ", ".join(sorted(conflicts)[:5])
        )


def verify_required_evidence(local_fold: Path) -> None:
    missing = [f for f in REQUIRED_EVIDENCE_FILES if not (local_fold / f).is_file()]
    if missing:
        raise CollectionError(f"required compact evidence missing after sync: {missing}")


def _rule_matches(rule: str, rel: str) -> bool:
    """Approximate rsync pattern matching for our fixed rule set."""
    pattern = rule.split("=", 1)[1]
    if pattern == "*":
        return True
    if pattern.endswith("/**"):
        base = pattern[:-3]
        return rel == base or rel.startswith(base + "/")
    if "/" in pattern:
        import fnmatch
        return fnmatch.fnmatch(rel, pattern)
    import fnmatch
    return fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)


def compact_expected(rel: str) -> bool:
    """First-match-wins evaluation of FILTER_RULES for a file path."""
    for rule in FILTER_RULES:
        if rule.startswith("--include="):
            pattern = rule.split("=", 1)[1]
            if pattern == "*/":
                continue  # directory traversal rule, not a file include
            if _rule_matches(rule, rel):
                return True
        elif rule.startswith("--exclude="):
            if _rule_matches(rule, rel):
                return False
    return False


def verify_compact_hash_agreement(
    remote_inv: dict[str, str], local_fold: Path
) -> dict[str, Any]:
    local_inv = local_inventory(local_fold)
    expected_remote = {rel: d for rel, d in remote_inv.items() if compact_expected(rel)}
    mismatched = [rel for rel, d in expected_remote.items() if local_inv.get(rel) != d]
    unexpected_local = sorted(
        rel for rel in local_inv
        if rel not in remote_inv or not compact_expected(rel)
    )
    if mismatched:
        raise CollectionError(
            f"remote/local hash mismatch for {len(mismatched)} files; first: {sorted(mismatched)[:5]}"
        )
    if unexpected_local:
        raise CollectionError(
            f"local tree contains files outside the compact evidence set: {sorted(unexpected_local)[:5]}"
        )
    return {
        "remote_files": len(remote_inv),
        "expected_compact_files": len(expected_remote),
        "local_files": len(local_inv),
        "matched": len(expected_remote),
        "extra_local": [],
    }


def plan_collection(remote_fold: str, local_fold: str) -> dict[str, Any]:
    validate_fold_path(remote_fold)
    return {
        "remote_fold": remote_fold,
        "local_fold": str(Path(local_fold).resolve()),
        "rsync_dry_run_argv": build_collect_argv(remote_fold, local_fold, dry_run=True),
        "rsync_execute_argv": build_collect_argv(remote_fold, local_fold, dry_run=False),
    }


def execute_collection(
    plan: dict[str, Any],
    runner: RemoteRunner,
    rsync_executor: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> dict[str, Any]:
    executor = rsync_executor or (
        lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    )
    remote_fold = plan["remote_fold"]
    local_fold = Path(plan["local_fold"])

    remote_inv = remote_inventory(runner, remote_fold)
    if not remote_inv:
        raise CollectionError(f"remote fold has no files: {remote_fold}")

    check_local_overwrite_safety(remote_inv, local_fold)
    local_fold.parent.mkdir(parents=True, exist_ok=True)

    dry_proc = executor(list(plan["rsync_dry_run_argv"]))
    if dry_proc.returncode != 0:
        raise CollectionError(f"rsync dry-run failed: {dry_proc.stderr.strip()}")

    exec_proc = executor(list(plan["rsync_execute_argv"]))
    if exec_proc.returncode != 0:
        raise CollectionError(f"rsync transfer failed: {exec_proc.stderr.strip()}")

    verify_required_evidence(local_fold)
    agreement = verify_compact_hash_agreement(remote_inv, local_fold)
    return {
        "remote_fold": remote_fold,
        "local_fold": str(local_fold),
        "inventory": agreement,
        "dry_run_stdout_lines": len(dry_proc.stdout.splitlines()),
        "execute_stdout_lines": len(exec_proc.stdout.splitlines()),
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Collect compact experiment evidence from MN5")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show the exact rsync plan and inventory (default)")
    mode.add_argument("--execute", action="store_true", help="execute the transfer after safety checks")
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--fold-dir", default=None, help="remote fold dir; must end in fold_<n>")
    parser.add_argument("--output", default=None, help="local destination dir")
    args = parser.parse_args(argv)

    from pathlib import Path as _P

    project_root = _P(__file__).resolve().parents[2]
    if not args.fold_dir:
        submit_root = project_root / "outputs" / "exp_submit"
        candidates = []
        if args.attempt_id:
            p = submit_root / args.attempt_id / "contract.json"
            if p.is_file():
                candidates.append(p)
        elif submit_root.exists():
            candidates = sorted(submit_root.glob("*/contract.json"))
        chosen = None
        for contract_path in reversed(candidates):
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if args.attempt_id and contract.get("attempt_id") != args.attempt_id:
                continue
            chosen = contract
            break
        if chosen is None:
            print("ERROR: --fold-dir or a recorded attempt contract is required", file=sys.stderr)
            return 1
        args.fold_dir = chosen["fold_dir"]
        if not args.output:
            args.output = str(project_root / "output_model" / "collected" / chosen["attempt_id"] / _P(args.fold_dir).name)

    try:
        plan = plan_collection(args.fold_dir, args.output)
        runner = RemoteRunner()
        if args.execute:
            result = execute_collection(plan, runner)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except CollectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
