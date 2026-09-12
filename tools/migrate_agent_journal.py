"""One-time migration of the historical journal into the dedicated journal repo.

The historical journal lived in `docs/agent-journal/YYYY-MM-DD.md` (one file per
day). It now lives in a single yearly file outside the repository. This tool
merges the historical files, demoting every heading one level so that a day
becomes a section and its entries become subsections.

The tool never deletes the sources. Removing `docs/agent-journal/` is a separate,
explicit step that happens only after `--verify` passes.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_append import journal_dir, journal_file  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / "docs" / "agent-journal"
SOURCE_NAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.md$")
HEADING_RE = re.compile(r"^(#{1,5})(\s)")
SEPARATOR = "\n"


class MigrationError(Exception):
    """Raised when the migration cannot run or verify successfully."""


def demote_headings(text: str) -> str:
    """Demote every ATX heading one level, skipping fenced code blocks."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and HEADING_RE.match(line):
            out.append("#" + line)
        else:
            out.append(line)
    return "".join(out)


def source_date(path: Path) -> date:
    match = SOURCE_NAME_RE.match(path.name)
    if not match:
        raise MigrationError(
            f"unexpected source file name {path.name!r}; expected YYYY-MM-DD.md"
        )
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def load_sources(source_dir: Path) -> dict[int, list[tuple[Path, str]]]:
    if not source_dir.is_dir():
        raise MigrationError(
            f"source directory {source_dir} does not exist. The migration removed "
            "the historical sources after it verified them; point --source at a "
            "copy of them if you need to re-verify."
        )
    grouped: dict[int, list[tuple[Path, str]]] = {}
    for path in sorted(source_dir.glob("*.md")):
        when = source_date(path)
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise MigrationError(f"source file {path} is empty")
        grouped.setdefault(when.year, []).append((path, demote_headings(text)))
    if not grouped:
        raise MigrationError(f"no journal source files found in {source_dir}")
    for year in grouped:
        grouped[year].sort(key=lambda item: item[0].name)
    return grouped


def build_year_content(year: int, parts: list[tuple[Path, str]]) -> str:
    header = f"# Agent journal — {year}\n\n"
    bodies = []
    for _, text in parts:
        bodies.append(text if text.endswith("\n") else text + "\n")
    return header + SEPARATOR.join(bodies)


def plan(source_dir: Path, root: Path | None = None) -> list[tuple[Path, str]]:
    grouped = load_sources(source_dir)
    planned = []
    for year in sorted(grouped):
        target = journal_file(date(year, 1, 1), root)
        planned.append((target, build_year_content(year, grouped[year])))
    return planned


def execute(source_dir: Path, root: Path | None = None, force: bool = False) -> None:
    for target, content in plan(source_dir, root):
        if not target.parent.is_dir():
            raise MigrationError(
                f"journal directory {target.parent} does not exist; create the "
                "journal repository first"
            )
        if target.exists() and target.read_text(encoding="utf-8").strip() and not force:
            raise MigrationError(
                f"target {target} already has content; refusing to overwrite "
                "(use --force to replace it)"
            )
        target.write_text(content, encoding="utf-8")


def verify(source_dir: Path, root: Path | None = None) -> list[str]:
    """Check that every source survived the transform byte for byte.

    Later entries appended by the journal tool are allowed: the migrated
    content must be a byte-exact prefix of the current file, so verification
    still works after the journal has grown.
    """
    grouped = load_sources(source_dir)
    messages = []
    for year in sorted(grouped):
        target = journal_file(date(year, 1, 1), root)
        if not target.is_file():
            raise MigrationError(f"target {target} is missing")
        text = target.read_text(encoding="utf-8")
        expected = build_year_content(year, grouped[year])
        if not text.startswith(expected):
            raise MigrationError(
                f"target {target} does not start with the migration of {source_dir}"
            )
        cursor = len(f"# Agent journal — {year}\n\n")
        for path, demoted in grouped[year]:
            body = demoted if demoted.endswith("\n") else demoted + "\n"
            chunk = text[cursor : cursor + len(body)]
            if chunk != body:
                raise MigrationError(
                    f"source {path} is not preserved in {target} at offset {cursor}"
                )
            cursor += len(body)
            if text[cursor : cursor + len(SEPARATOR)] == SEPARATOR:
                cursor += len(SEPARATOR)
        digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        trailing = len(text) - len(expected)
        messages.append(
            f"{target}: {len(grouped[year])}/{len(grouped[year])} sources preserved, "
            f"migrated_sha256={digest}, {trailing} bytes appended afterwards"
        )
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge the historical agent journal")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="source directory")
    parser.add_argument("--execute", action="store_true", help="write the merged files")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only (default)")
    parser.add_argument("--verify", action="store_true", help="verify the merged files")
    parser.add_argument("--force", action="store_true", help="replace non-empty targets")
    args = parser.parse_args(argv)

    source_dir = Path(args.source).expanduser().resolve()
    if args.verify:
        try:
            for message in verify(source_dir):
                print(message)
        except MigrationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("VERIFIED: every source is preserved")
        return 0

    try:
        planned = plan(source_dir)
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.execute:
        try:
            execute(source_dir, force=args.force)
        except MigrationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        for target, content in planned:
            print(f"wrote {target} ({len(content)} bytes)")
        print("run --verify next, then remove the source directory explicitly")
        return 0

    for target, content in planned:
        print(f"would write {target} ({len(content)} bytes)")
    print("dry-run: nothing written; pass --execute to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
