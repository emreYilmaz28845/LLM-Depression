"""Tests for the fail-closed project storage gate used by the refill watcher."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qwen3_window_cap_storage_gate as gate  # noqa: E402

ANSI_SAMPLE = (
    "\x1b[1m\x1b[38;5;117m Printing quota for group etur92:\x1b[22m\x1b[0m\n"
    "\x1b[1;30m\x1b[47m    Filesystem   Type          Usage          Quota"
    "          Limit     In doubt     Grace  |       Files  In doubt  \x1b[0m\n"
    "\x1b[38;5;252m\x1b[48;5;235m     gpfs_home    USR       31.88 GB       80.00 GB"
    "       84.00 GB\x1b[38;5;252m      0.00 KB      None  |      115940         0  \x1b[0m\n"
    "\x1b[38;5;252m\x1b[48;5;235m gpfs_projects    GRP        1.79 TB        3.91 TB"
    "        4.10 TB\x1b[38;5;252m      4.69 GB      None  |     1723914       469  \x1b[0m\n"
    "\x1b[38;5;252m\x1b[48;5;235m  gpfs_scratch    GRP        1.75 TB        1.95 TB"
    "        2.05 TB\x1b[38;5;252m     89.94 MB      None  |     1665976        39  \x1b[0m\n"
)

PLAIN_SAMPLE = gate.strip_ansi(ANSI_SAMPLE)


def test_parse_real_sample_with_ansi() -> None:
    parsed = gate.parse_bsc_quota(ANSI_SAMPLE)
    assert parsed["filesystem"] == "gpfs_projects"
    assert parsed["usage_gb"] == pytest.approx(1.79 * 1024)
    assert parsed["quota_gb"] == pytest.approx(3.91 * 1024)
    assert parsed["limit_gb"] == pytest.approx(4.10 * 1024)
    assert parsed["in_doubt_gb"] == pytest.approx(4.69)


def test_evaluate_allows_the_real_snapshot() -> None:
    parsed = gate.parse_bsc_quota(PLAIN_SAMPLE)
    ok, reason, remaining = gate.evaluate(
        parsed, min_remaining_gb=500.0, local_free_gb=232.0, min_local_gb=50.0
    )
    assert ok is True
    assert remaining == pytest.approx(3.91 * 1024 - 1.79 * 1024 - 4.69)
    assert "remaining" in reason


def test_evaluate_refuses_below_project_reserve() -> None:
    parsed = gate.parse_bsc_quota(PLAIN_SAMPLE)
    parsed["usage_gb"] = 3.70 * 1024
    ok, reason, remaining = gate.evaluate(parsed, min_remaining_gb=500.0)
    assert ok is False
    assert "remaining" in reason
    assert remaining < 500.0


def test_evaluate_refuses_low_local_space() -> None:
    parsed = gate.parse_bsc_quota(PLAIN_SAMPLE)
    ok, reason, _ = gate.evaluate(
        parsed, min_remaining_gb=500.0, local_free_gb=40.0, min_local_gb=50.0
    )
    assert ok is False
    assert "local free" in reason


def test_parse_refuses_missing_row() -> None:
    text = "Filesystem Type Usage Quota Limit In doubt Grace\n gpfs_home USR 1 GB 2 GB 3 GB 0 GB None\n"
    with pytest.raises(gate.StorageGateError, match="no gpfs_projects row"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_empty_output() -> None:
    with pytest.raises(gate.StorageGateError, match="empty"):
        gate.parse_bsc_quota("   \n")


def test_parse_refuses_unknown_unit() -> None:
    text = "gpfs_projects GRP 1.79 XB 3.91 TB 4.10 TB 4.69 GB None\n"
    with pytest.raises(gate.StorageGateError, match="unknown size unit"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_unparseable_number() -> None:
    text = "gpfs_projects GRP many TB 3.91 TB 4.10 TB 4.69 GB None\n"
    with pytest.raises(gate.StorageGateError, match="unparseable size"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_non_group_row() -> None:
    text = "gpfs_projects USR 1.79 TB 3.91 TB 4.10 TB 4.69 GB None\n"
    with pytest.raises(gate.StorageGateError, match="not a group row"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_short_row() -> None:
    text = "gpfs_projects GRP 1.79 TB 3.91 TB\n"
    with pytest.raises(gate.StorageGateError, match="too short"):
        gate.parse_bsc_quota(text)


def test_query_failure_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        returncode = 255
        stdout = ""
        stderr = "ssh: connect to host transfer1 port 22: Connection timed out"

    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: Fake())
    with pytest.raises(gate.StorageGateError, match="bsc_quota query failed"):
        gate.query_quota("user@host")


def test_size_units() -> None:
    assert gate._to_gb("1", "TB") == pytest.approx(1024.0)
    assert gate._to_gb("512", "GB") == pytest.approx(512.0)
    assert gate._to_gb("1", "MB") == pytest.approx(1.0 / 1024.0)
    assert gate._to_gb("1", "B") == pytest.approx(1.0 / (1024.0 ** 3))


def test_cli_parse_file_ok_and_refused(tmp_path: Path) -> None:
    sample = tmp_path / "quota.txt"
    sample.write_text(PLAIN_SAMPLE, encoding="utf-8")
    assert gate.main(["--parse-file", str(sample), "--json"]) == 0
    heavy = tmp_path / "heavy.txt"
    heavy.write_text(
        PLAIN_SAMPLE.replace("1.79 TB", "3.90 TB"), encoding="utf-8"
    )
    assert gate.main(["--parse-file", str(heavy), "--json"]) == 1


def test_cli_local_free_uses_real_disk() -> None:
    sample_free = gate.shutil.disk_usage("/home/emre/Projects").free / gate.BYTES_PER_GB
    assert sample_free > 0
