"""Append one entry to the agent journal, which now lives in SecondBrain.

The journal is the append-only narrative index of agent work for this
repository. Since 2026-10-12 it lives in the shared SecondBrain vault
(``~/SecondBrain/20-Projects/LLM-Depression/journal/``, one file per UTC month)
and the only writer is ``sb log``. This tool keeps its old command line and
forwards each entry to ``sb log --project LLM-Depression``, which takes the
vault lock, writes the entry in one call, commits, and pushes to the backup
remote.

The writer must be declared with ``SECONDBRAIN_AGENT`` (for example ``codex``
or ``claude-code``); the tool refuses to guess. The lane/branch context is still
derived from the pin or Git and recorded by ``sb`` in the entry.

The yearly-file helpers below (``legacy_*``, ``build_entry``, ``append_entry``)
describe the pre-SecondBrain journal and are kept for
``tools/migrate_agent_journal.py`` and its tests. The 2026 legacy file was copied
byte-for-byte into the vault as ``LLM-Depression-legacy-2026.md``.
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

DEFAULT_JOURNAL_ROOT = Path.home() / "SecondBrain"
LEGACY_JOURNAL_ROOT = Path("/home/emre/Projects/AudioLLM/Agent-Journal")
PROJECT_DIR_NAME = "LLM-Depression"
SB_PROJECT = "LLM-Depression"
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
    """Root of the SecondBrain vault that holds the journal.

    ``AGENT_JOURNAL_ROOT`` (tests, backfills) wins, then ``SECONDBRAIN_ROOT``.
    """
    for name in ("AGENT_JOURNAL_ROOT", "SECONDBRAIN_ROOT"):
        override = os.environ.get(name)
        if override:
            return Path(override).expanduser().resolve()
    return DEFAULT_JOURNAL_ROOT


def journal_dir(root: Path | None = None) -> Path:
    return (root or journal_root()) / "20-Projects" / SB_PROJECT / "journal"


def journal_file(when: date, root: Path | None = None) -> Path:
    """Monthly SecondBrain journal file. ``sb`` picks the month from the UTC date."""
    return journal_dir(root) / f"{SB_PROJECT}-{when:%Y-%m}.md"


def legacy_journal_dir(root: Path | None = None) -> Path:
    return (root or LEGACY_JOURNAL_ROOT) / PROJECT_DIR_NAME


def legacy_journal_file(when: date, root: Path | None = None) -> Path:
    return legacy_journal_dir(root) / f"agent-journal-{when.year}.md"


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


def sb_command() -> list[str]:
    raw = os.environ.get("SECONDBRAIN_SB", "sb")
    return raw.split()


def _run_sb(args: list[str], stdin: str | None = None) -> int:
    try:
        result = subprocess.run(
            [*sb_command(), *args], input=stdin, capture_output=True, text=True
        )
    except FileNotFoundError:
        print(
            "ERROR: the `sb` command (SecondBrain) was not found. Install it from "
            "~/Projects/SecondBrain-tool (./install.sh) or set SECONDBRAIN_SB.",
            file=sys.stderr,
        )
        return 1
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return 0 if result.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one entry to the agent journal (forwards to `sb log` in SecondBrain)"
    )
    parser.add_argument("--title", help="short entry title")
    parser.add_argument(
        "--body-file",
        default=None,
        help="read the body from this file, or '-' for stdin (default: stdin)",
    )
    parser.add_argument("--cwd", default=None, help="directory used to derive the identity")
    parser.add_argument("--dry-run", action="store_true", help="print the entry only")
    parser.add_argument(
        "--sync", action="store_true", help="push pending SecondBrain commits (`sb sync`) and exit"
    )
    parser.add_argument(
        "--print-identity", action="store_true", help="print the derived lane identity and exit"
    )
    parser.add_argument(
        "--print-path", action="store_true", help="print the current journal file and exit"
    )
    args = parser.parse_args(argv)

    root = journal_root()
    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    env_root = {"SECONDBRAIN_ROOT": str(root)}

    if args.sync:
        if args.title:
            print("ERROR: --sync does not take --title", file=sys.stderr)
            return 1
        os.environ.update(env_root)
        return _run_sb(["sync"])

    try:
        identity = derive_identity(cwd)
        if args.print_identity:
            print(f"{identity.label}\nAgent: {identity.detail}")
            return 0
        if args.print_path:
            print(str(journal_file(datetime.now(timezone.utc).date(), root)))
            return 0
        if not args.title or not args.title.strip():
            raise JournalError("--title is required and must not be empty")
        body = _read_body(args.body_file)
        if not body.strip():
            raise JournalError("entry body is empty")
        if args.dry_run:
            print(f"journal file: {journal_file(datetime.now(timezone.utc).date(), root)}")
            print(f"--- entry (via sb log --project {SB_PROJECT}) ---")
            print(f"{args.title.strip()}\n\n{body.strip()}")
            return 0
        if not os.environ.get("SECONDBRAIN_AGENT", "").strip():
            raise JournalError(
                "SECONDBRAIN_AGENT is not set; declare who is writing, e.g. "
                "`export SECONDBRAIN_AGENT=codex` (or claude-code, opencode, pi)"
            )
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    os.environ.update(env_root)
    previous = os.getcwd()
    try:
        os.chdir(cwd)  # sb records the lane pin / branch of the working directory
        return _run_sb(
            ["log", "--project", SB_PROJECT, "--title", args.title.strip(), "--body-file", "-"],
            stdin=body,
        )
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    sys.exit(main())
