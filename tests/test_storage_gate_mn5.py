"""Parser and gate validation for the MN5 storage admission gate.

The gate must read the real ``bsc_quota`` project row (Usage, soft Quota,
In doubt), never shared-filesystem ``df`` occupancy, and must fail closed on
SSH, group, or parser failures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import storage_gate_mn5 as gate

REAL_OUTPUT = """
\u001b[1m\u001b[38;5;117m Printing quota for group etur92:\u001b[22m\u001b[0m

\u001b[1;30m\u001b[47m    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace  |       Files  In doubt  \u001b[0m
\u001b[38;5;252m\u001b[48;5;235m     gpfs_home    USR       31.88 GB       80.00 GB       84.00 GB\u001b[38;5;252m      0.00 KB      None  |      115940         0  \u001b[0m
\u001b[38;5;252m\u001b[48;5;235m gpfs_projects    GRP        1.79 TB        3.91 TB        4.10 TB\u001b[38;5;252m      4.69 GB      None  |     1723914       469  \u001b[0m
\u001b[38;5;252m\u001b[48;5;235m  gpfs_scratch    GRP        1.75 TB        1.95 TB        2.05 TB\u001b[38;5;252m     89.94 MB      None  |     1665976        39  \u001b[0m
"""

GB_OUTPUT = """
 Printing quota for group etur92:

    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace
 gpfs_projects    GRP     1758.18 GB     4000.00 GB     4200.00 GB      12.86 GB      None
"""


def test_parse_real_grouped_output_with_ansi() -> None:
    quota = gate.parse_bsc_quota_projects(REAL_OUTPUT)
    assert quota["usage_gb"] == pytest.approx(1.79 * 1024)
    assert quota["soft_quota_gb"] == pytest.approx(3.91 * 1024)
    assert quota["hard_limit_gb"] == pytest.approx(4.10 * 1024)
    assert quota["in_doubt_gb"] == pytest.approx(4.69)
    assert quota["remaining_gb"] == pytest.approx(3.91 * 1024 - 1.79 * 1024 - 4.69)


def test_parse_gb_output() -> None:
    quota = gate.parse_bsc_quota_projects(GB_OUTPUT)
    assert quota["usage_gb"] == pytest.approx(1758.18)
    assert quota["soft_quota_gb"] == pytest.approx(4000.0)
    assert quota["in_doubt_gb"] == pytest.approx(12.86)
    assert quota["remaining_gb"] == pytest.approx(4000.0 - 1758.18 - 12.86)


def test_parse_missing_project_row_fails_closed() -> None:
    with pytest.raises(gate.StorageGateError):
        gate.parse_bsc_quota_projects(" Printing quota for group etur92:\n nothing here\n")


def test_parse_wrong_group_fails_closed() -> None:
    with pytest.raises(gate.StorageGateError):
        gate.parse_bsc_quota_projects(GB_OUTPUT.replace("etur92", "othergroup"))


def test_evaluate_gate_boundaries() -> None:
    quota = {"remaining_gb": 500.0}
    ok, reason = gate.evaluate_gate(quota, 50 * 1024**3)
    assert ok and reason == "ok"
    ok, reason = gate.evaluate_gate({"remaining_gb": 499.0}, 50 * 1024**3)
    assert not ok and "project reserve" in reason
    ok, reason = gate.evaluate_gate(quota, 49 * 1024**3)
    assert not ok and "local free" in reason


def test_read_bsc_quota_rc_failure_fails_closed() -> None:
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="Permission denied")

    with pytest.raises(gate.StorageGateError):
        gate.read_bsc_quota(runner=runner)


def test_check_fails_closed_on_ssh_exception() -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 120)

    with pytest.raises(gate.StorageGateError):
        gate.check(runner=runner)


def test_check_passes_and_reports_evidence(tmp_path: Path, monkeypatch) -> None:
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=GB_OUTPUT, stderr="")

    monkeypatch.setattr(
        gate.shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=60 * 1024**3)
    )
    evidence = gate.check(runner=runner, local_path=tmp_path)
    assert evidence["ok"] is True
    assert evidence["remaining_gb"] == pytest.approx(2228.96, abs=0.01)
    assert evidence["local_free_gb"] == pytest.approx(60.0)


def test_check_refuses_on_low_project_reserve(monkeypatch) -> None:
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=REAL_OUTPUT, stderr="")

    monkeypatch.setattr(
        gate.shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=60 * 1024**3)
    )
    monkeypatch.setattr(gate, "evaluate_gate", lambda *a, **k: (False, "project reserve low"))
    with pytest.raises(gate.StorageGateError) as info:
        gate.check(runner=runner)
    assert info.value.kind == "threshold"
