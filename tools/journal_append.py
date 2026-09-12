"""Append one entry to the agent journal.

The journal is the append-only narrative index of agent work for this
repository. It lives outside the repository so that every checkout, lane
worktree, and agent sees the same file. Several agents can append at the same
time, so this tool is the only supported writer: it takes an exclusive lock,
writes the whole entry with a single write call, and then commits the result.

Never edit the journal file directly. A direct edit bypasses the lock and can
silently drop another agent's entry.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEFAULT_JOURNAL_ROOT = Path("/home/emre/Projects/AudioLLM/Agent-Journal")
PROJECT_DIR_NAME = "LLM-Depression"
PIN_FILE_NAME = ".agent-pin.json"
LOCK_POLL_SECONDS = 0.05

try:
    from zoneinfo import ZoneInfo

    ISTANBUL = ZoneInfo("Europe/Istanbul")
except Exception:  # pragma: no cover - only when tzdata is unavailable
    ISTANBUL = timezone(timedelta(hours=3), "Europe/Istanbul")


class JournalError(Exception):
    """Raised when the journal entry cannot be written safely."""


@dataclass(frozen=True)
class Identity:
    label: str
    detail: str


def journal_root() -> Path:
    override = os.environ.get("AGENT_JOURNAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_JOURNAL_ROOT


def journal_dir(root: Path | None = None) -> Path:
    return (root or journal_root()) / PROJECT_DIR_NAME


def journal_file(when: date, root: Path | None = None) -> Path:
    return journal_dir(root) / f"agent-journal-{when.year}.md"


def istanbul_now() -> datetime:
    return datetime.now(ISTANBUL)


def _git(cwd: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _load_pin(pin_path: Path) -> dict | None:
    if not pin_path.is_file():
        return None
    try:
        return json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JournalError(f"could not read pin {pin_path}: {exc}") from exc


def derive_identity(cwd: Path) -> Identity:
    """Derive the writer identity from the lane pin or from Git.

    The identity is never typed by the agent: it comes from the pin that the
    lane already recorded, or from the current branch and commit.
    """
    git_top = _git(cwd, "rev-parse", "--show-toplevel")
    search_dirs = [Path(git_top)] if git_top else []
    if cwd.resolve() not in [d.resolve() for d in search_dirs]:
        search_dirs.append(cwd)

    for directory in search_dirs:
        pin = _load_pin(directory / PIN_FILE_NAME)
        if pin is None:
            continue
        branch = pin.get("branch")
        experiment_id = pin.get("experiment_id")
        if not branch or not experiment_id:
            raise JournalError(
                f"pin {directory / PIN_FILE_NAME} is missing branch or experiment_id"
            )
        worktree = pin.get("worktree") or str(directory)
        detail = (
            f"experiment_id={experiment_id} branch={branch} worktree={worktree}"
        )
        return Identity(label=str(branch), detail=detail)

    if not git_top:
        raise JournalError(
            f"{cwd} is neither a Git work tree nor a pinned lane; refusing to "
            "guess an identity"
        )
    branch = _git(cwd, "branch", "--show-current") or "detached"
    commit = _git(cwd, "rev-parse", "--short", "HEAD")
    if not commit:
        raise JournalError(f"could not resolve HEAD in {git_top}")
    return Identity(
        label=f"{branch}@{commit}",
        detail=f"repo={git_top} branch={branch} commit={commit}",
    )


def build_entry(
    title: str, body: str, identity: Identity, when_utc: datetime
) -> str:
    title = title.strip()
    body = body.strip("\n")
    if not title:
        raise JournalError("entry title is empty")
    if not body.strip():
        raise JournalError("entry body is empty")
    stamp = when_utc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    return (
        f"## {stamp} — {identity.label} — {title}\n\n"
        f"Agent: {identity.detail}\n\n"
        f"{body}\n\n"
    )


def _lock_with_timeout(fd: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise JournalError(
                    f"could not lock the journal within {timeout_seconds:g}s; "
                    "another writer is holding it"
                ) from None
            time.sleep(LOCK_POLL_SECONDS)


def _git_commit(root: Path, path: Path, message: str) -> None:
    if _git(root, "rev-parse", "--git-dir") is None:
        raise JournalError(f"{root} is not a Git repository; cannot commit")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise JournalError(f"{path} is outside the journal repository {root}") from exc
    add = subprocess.run(
        ["git", "-C", str(root), "add", "--", relative],
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise JournalError(f"git add failed: {add.stderr.strip()}")
    commit = subprocess.run(
        ["git", "-C", str(root), "commit", "-m", message, "--", relative],
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise JournalError(f"git commit failed: {commit.stderr.strip()}")


def git_push(root: Path) -> tuple[bool, str]:
    """Best-effort push of the journal repository. Returns (ok, message)."""
    if _git(root, "rev-parse", "--git-dir") is None:
        return False, f"{root} is not a Git repository"
    remotes = subprocess.run(
        ["git", "-C", str(root), "remote"], capture_output=True, text=True
    ).stdout.split()
    if not remotes:
        return False, "no remote configured"
    remote = "origin" if "origin" in remotes else remotes[0]
    push = subprocess.run(
        ["git", "-C", str(root), "push", "--set-upstream", remote, "HEAD"],
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        return False, push.stderr.strip() or "git push failed"
    return True, "pushed"


def append_entry(
    path: Path,
    entry: str,
    lock_timeout: float,
    commit_message: str | None = None,
    root: Path | None = None,
    header: str = "",
) -> None:
    """Append one entry under an exclusive lock and commit it."""
    if not path.parent.is_dir():
        raise JournalError(
            f"journal directory {path.parent} does not exist; create the journal "
            "repository first"
        )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        _lock_with_timeout(fd, lock_timeout)
        try:
            prefix = ""
            if os.fstat(fd).st_size == 0 and header:
                prefix = header
            payload = (prefix + entry).encode("utf-8")
            written = os.write(fd, payload)
            if written != len(payload):
                raise JournalError(
                    f"short write: wrote {written} of {len(payload)} bytes"
                )
            os.fsync(fd)
            if commit_message is not None and root is not None:
                _git_commit(root, path, commit_message)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_body(body_file: str | None) -> str:
    if body_file is None:
        if sys.stdin.isatty():
            raise JournalError(
                "no entry body: pipe it on stdin or pass --body-file"
            )
        return sys.stdin.read()
    if body_file == "-":
        return sys.stdin.read()
    return Path(body_file).read_text(encoding="utf-8")


def _parse_now(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _command_sync(root: Path) -> int:
    ok, message = git_push(root)
    if ok:
        print(f"journal push: {message}")
        return 0
    print(f"ERROR: journal push failed: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one entry to the agent journal (single supported writer)"
    )
    parser.add_argument("--title", help="short entry title")
    parser.add_argument(
        "--body-file",
        default=None,
        help="read the body from this file, or '-' for stdin (default: stdin)",
    )
    parser.add_argument(
        "--now",
        default=None,
        help="UTC instant for the entry and the rotation year (ISO 8601)",
    )
    parser.add_argument("--cwd", default=None, help="directory used to derive the identity")
    parser.add_argument(
        "--lock-timeout", type=float, default=30.0, help="seconds to wait for the lock"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the entry only")
    parser.add_argument(
        "--no-git", action="store_true", help="write without committing to the journal repo"
    )
    parser.add_argument(
        "--sync", action="store_true", help="push pending journal commits and exit"
    )
    parser.add_argument(
        "--print-identity", action="store_true", help="print the derived identity and exit"
    )
    parser.add_argument(
        "--print-path", action="store_true", help="print the target journal file and exit"
    )
    args = parser.parse_args(argv)

    root = journal_root()
    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()

    if args.sync:
        if args.title:
            print("ERROR: --sync does not take --title", file=sys.stderr)
            return 1
        return _command_sync(root)

    try:
        identity = derive_identity(cwd)
        when_utc = _parse_now(args.now)
        local_date = when_utc.astimezone(ISTANBUL).date()
        target = journal_file(local_date, root)

        if args.print_identity:
            print(f"{identity.label}\nAgent: {identity.detail}")
            return 0
        if args.print_path:
            print(str(target))
            return 0
        if not args.title:
            print("ERROR: --title is required to append an entry", file=sys.stderr)
            return 1

        entry = build_entry(args.title, _read_body(args.body_file), identity, when_utc)
        if args.dry_run:
            print(f"journal file: {target}")
            print("--- entry ---")
            print(entry, end="")
            return 0

        commit_message = None
        if not args.no_git:
            commit_message = f"journal: [{identity.label}] {args.title.strip()}"[:120]
        append_entry(
            target,
            entry,
            lock_timeout=args.lock_timeout,
            commit_message=commit_message,
            root=root,
            header=f"# Agent journal — {local_date.year}\n\n",
        )
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"appended to {target} as {identity.label}")
    if not args.no_git:
        ok, message = git_push(root)
        if not ok:
            print(
                f"WARNING: local commit kept, push skipped ({message}); run "
                "--sync later",
                file=sys.stderr,
            )
        else:
            print(f"journal push: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
