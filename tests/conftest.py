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
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


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
