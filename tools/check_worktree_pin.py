from __future__ import annotations
import argparse
import json
import re
import sys
import subprocess
from pathlib import Path

PROTECTED_PATHS = [
    Path("/home/emre/Projects/AudioLLM/Teacher-System").resolve(),
    Path("/home/emre/Projects/AudioLLM/LLM-Depression-teacher").resolve(),
]

SCHEMA_VERSION = "audiollm.agent_pin.v1"

def get_git_toplevel(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except subprocess.CalledProcessError:
        return None

def get_git_branch(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def is_inside(parent: Path, child: Path) -> bool:
    try:
        # Use resolved paths and check relative
        child_resolved = child.resolve()
        parent_resolved = parent.resolve()
        # Handle symlink escape by resolving
        return parent_resolved == child_resolved or parent_resolved in child_resolved.parents
    except Exception:
        return False

def load_pin(pin_path: Path) -> dict:
    with pin_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_simple_definition(definition_path: Path) -> dict:
    """Load the scalar YAML emitted by tools/exp.py without adding PyYAML."""
    data = {}
    for line_number, raw_line in enumerate(
        definition_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"line {line_number} is not a key/value pair")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError(f"line {line_number} has an empty key")
        if value in {"null", "~"}:
            parsed = None
        elif value.isdigit():
            parsed = int(value)
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            parsed = value[1:-1]
        else:
            parsed = value
        data[key] = parsed
    return data

def validate_pin(pin_data: dict, pin_path: Path, cwd: Path, target_path: Path | None) -> list[str]:
    errors = []
    # Schema
    if pin_data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"invalid schema_version {pin_data.get('schema_version')} expected {SCHEMA_VERSION}")
    worktree = Path(pin_data.get("worktree", "")).resolve() if pin_data.get("worktree") else None
    branch = pin_data.get("branch")
    allowed_paths = [Path(p).resolve() for p in pin_data.get("allowed_paths", [])]
    protected_paths = [Path(p).resolve() for p in pin_data.get("protected_paths", [])]
    experiment_id = pin_data.get("experiment_id")

    required = {
        "experiment_id": experiment_id,
        "worktree": pin_data.get("worktree"),
        "branch": branch,
        "allowed_paths": pin_data.get("allowed_paths"),
    }
    for key, value in required.items():
        if value is None or value == "" or value == []:
            errors.append(f"pin is missing required field {key!r}")
    if experiment_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(experiment_id)):
        errors.append(f"invalid experiment_id {experiment_id!r}")

    # Add default protected paths if not present? Ensure they are checked
    for pp in PROTECTED_PATHS:
        if pp not in protected_paths:
            protected_paths.append(pp)

    # Check real CWD inside pinned worktree
    cwd_resolved = cwd.resolve()
    if worktree and not is_inside(worktree, cwd_resolved):
        errors.append(f"CWD {cwd_resolved} is not inside pinned worktree {worktree}")

    # Check git top-level equals pinned worktree
    git_top = get_git_toplevel(cwd)
    if git_top is None:
        errors.append(f"not a git repository at {cwd}")
    elif worktree and git_top != worktree:
        errors.append(f"git top-level {git_top} does not equal pinned worktree {worktree}")

    # Check branch
    git_branch = get_git_branch(cwd)
    if git_branch is None:
        errors.append("could not determine git branch")
    elif branch and git_branch != branch:
        errors.append(f"checked-out branch {git_branch!r} does not equal pinned branch {branch!r}")

    # Check the exact operational lane definition emitted for this experiment.
    # Older pins use the original definition location and mislabeled schema;
    # keep them usable, but new pins must point at the dedicated lane schema.
    if experiment_id:
        project_root = git_top if git_top else cwd
        definition_rel = pin_data.get("definition_path")
        if definition_rel:
            rel_path = Path(definition_rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                errors.append(f"invalid definition_path {definition_rel!r}")
                definition_path = project_root / "__invalid_definition_path__"
            else:
                definition_path = project_root / rel_path
        else:
            definition_path = project_root / "experiments" / "definitions" / f"{experiment_id}.yaml"
        if not definition_path.is_file():
            errors.append(f"experiment definition for {experiment_id!r} not found at {definition_path}")
        else:
            try:
                definition = load_simple_definition(definition_path)
            except Exception as exc:
                errors.append(f"could not parse experiment definition {definition_path}: {exc}")
            else:
                expected = {
                    "schema_version": (
                        "audiollm.experiment_lane.v1" if definition_rel
                        else "audiollm.experiment_group.v1"
                    ),
                    "experiment_id": experiment_id,
                    "branch": branch,
                    "worktree": str(worktree) if worktree else None,
                }
                for optional_key in ("tier", "parent_branch", "parent_sha"):
                    if optional_key in pin_data:
                        expected[optional_key] = pin_data.get(optional_key)
                for key, expected_value in expected.items():
                    actual_value = definition.get(key)
                    if actual_value != expected_value:
                        errors.append(
                            f"experiment definition {key}={actual_value!r} does not match pin {expected_value!r}"
                        )

    # Check target path
    check_target = target_path.resolve() if target_path else cwd_resolved
    # Check inside allowed_paths
    if allowed_paths:
        inside_allowed = any(is_inside(ap, check_target) for ap in allowed_paths)
        if not inside_allowed:
            errors.append(f"target path {check_target} is not inside allowed_paths {allowed_paths}")
    # Check outside protected paths
    for pp in protected_paths:
        if is_inside(pp, check_target) or check_target == pp:
            errors.append(f"target path {check_target} is inside protected path {pp}")

    # Also check for symlink escape and .. escape via resolve already handled
    # Check that worktree itself is not inside protected path
    if worktree:
        for pp in protected_paths:
            if is_inside(pp, worktree) or worktree == pp:
                errors.append(f"pinned worktree {worktree} is inside protected path {pp}")

    # Check for .. escape in original target before resolve - if target contains .. that would escape allowed
    # Already handled via resolve, but we can explicitly check if original target string contains .. and resolved is outside allowed
    return errors

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify worktree pin before mutation")
    parser.add_argument("--pin", default=None, help="path to .agent-pin.json (default: <git top>/.agent-pin.json)")
    parser.add_argument("--target", default=None, help="target path to check (default: current directory)")
    parser.add_argument("--cwd", default=None, help="working directory to check (default: current directory)")
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve()
    pin_path = Path(args.pin).resolve() if args.pin else None
    if pin_path is None:
        # Find pin file from git top or cwd
        git_top = get_git_toplevel(cwd)
        search_root = git_top if git_top else cwd
        pin_path = search_root / ".agent-pin.json"
    if not pin_path.exists():
        print(f"ERROR: pin file not found at {pin_path}", file=sys.stderr)
        return 1

    try:
        pin_data = load_pin(pin_path)
    except Exception as e:
        print(f"ERROR: failed to load pin {pin_path}: {e}", file=sys.stderr)
        return 1

    target_path = Path(args.target).resolve() if args.target else cwd
    # Use original target for .. check - resolve handles it
    errors = validate_pin(pin_data, pin_path, cwd, target_path)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"pin verified: worktree={pin_data.get('worktree')} branch={pin_data.get('branch')} experiment={pin_data.get('experiment_id')} CWD={cwd} target={target_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
