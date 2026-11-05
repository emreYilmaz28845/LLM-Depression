"""Shared pytest configuration.

``real_data`` marks tests that read local-only evidence: the private ``docs/``
overlay or untracked ``output_model/``, ``outputs/``, ``logs/`` or ``reports/``
results. These paths exist on the main checkout but not in managed lane
worktrees (``tools/exp.py create`` copies neither) or in a fresh clone.

Usage::

    pytestmark = pytest.mark.real_data("output_model/harmonized_v1", "docs/archive")

or on a single test. Each argument is a path relative to the repository root
(or to ``LLMDEP_EVIDENCE_ROOT`` when that is set, which is what the workbook
tests already read), or an absolute path such as a local dataset root. When
any listed path is missing, the test is skipped with
the missing path in the reason. When every path exists, the test runs and a
failure is a real failure. Run only the fast unit tests with::

    python -m pytest tests/ -m "not real_data"

``slow`` marks tests that take about 15 s or more on their own (full 100-trial
Optuna studies, manifest builders, generators). They run in the default full
suite; skip them for a quick check with ``-m "not slow and not real_data"``.

Checkout write guard: tests must write only under ``tmp_path``. At session
start the guard records the size and mtime of every file in the checkout
(except ``.git``, ``__pycache__`` and ``.pytest_cache``). At session end any
file that was added, changed or removed fails the run and is listed. This
protects canonical manifests, splits and configs from being rewritten by a
test. A concurrent process writing into the same checkout also trips it; set
``LLMDEP_SKIP_WRITE_GUARD=1`` only when you know that is the cause.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
_GUARD_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
_GUARD_REPORT_LIMIT = 40
_guard_snapshot: dict[str, tuple[int, int]] | None = None
_guard_changes: list[str] = []


def _checkout_snapshot() -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(CODE_ROOT):
        dirnames[:] = [name for name in dirnames if name not in _GUARD_SKIP_DIRS]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                stat = os.lstat(path)
            except FileNotFoundError:
                continue
            snapshot[os.path.relpath(path, CODE_ROOT)] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _guard_enabled() -> bool:
    return os.environ.get("LLMDEP_SKIP_WRITE_GUARD") != "1" and "PYTEST_XDIST_WORKER" not in os.environ


def pytest_sessionstart(session: pytest.Session) -> None:
    global _guard_snapshot
    _guard_snapshot = _checkout_snapshot() if _guard_enabled() else None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _guard_snapshot is None:
        return
    after = _checkout_snapshot()
    changes = [f"changed {path}" for path in sorted(after) if path in _guard_snapshot and after[path] != _guard_snapshot[path]]
    changes += [f"added {path}" for path in sorted(after) if path not in _guard_snapshot]
    changes += [f"removed {path}" for path in sorted(_guard_snapshot) if path not in after]
    _guard_changes[:] = changes
    if changes:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if not _guard_changes:
        return
    terminalreporter.section("checkout write guard", red=True)
    terminalreporter.write_line(
        f"FAILED: the test session wrote {len(_guard_changes)} file(s) in the checkout "
        f"{CODE_ROOT}; tests must write under tmp_path."
    )
    for line in _guard_changes[:_GUARD_REPORT_LIMIT]:
        terminalreporter.write_line(f"  {line}")
    if len(_guard_changes) > _GUARD_REPORT_LIMIT:
        terminalreporter.write_line(f"  ... and {len(_guard_changes) - _GUARD_REPORT_LIMIT} more")


def _evidence_root() -> Path:
    return Path(os.environ.get("LLMDEP_EVIDENCE_ROOT", CODE_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_data(*paths): needs local-only evidence (private docs/ or untracked "
        "output_model/, outputs/, logs/, reports/); skipped when a listed path is missing",
    )
    config.addinivalue_line(
        "markers",
        "slow: takes about 15 s or more alone; still part of the default full run",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    root = _evidence_root()
    for marker in item.iter_markers(name="real_data"):
        if not marker.args:
            raise pytest.UsageError(f"{item.nodeid}: real_data marker needs at least one path")
        missing = [path for path in marker.args if not (root / path).exists()]
        if missing:
            pytest.skip(f"real_data: missing local-only evidence {', '.join(missing)} under {root}")
