"""Tests for the fail-closed project storage gate used by the refill watcher.

The live admission path queries `bsc_quota projects --unit GB --no-color` so
the soft quota is exact (4000.00 GB, not the rounded 3.91 TB that reconstructs
4003.84 GB and can wrongly admit at the 500 GB reserve boundary). The parser
requires the expected group header, GB units in live mode, and finite
nonnegative sizes; every ambiguity refuses.
"""

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

GB_SAMPLE = (
    " Printing quota for group etur92:\n"
    "    Filesystem   Type          Usage          Quota          Limit"
    "     In doubt     Grace  |       Files  In doubt\n"
    "     gpfs_home    USR       31.88 GB       80.00 GB       84.00 GB"
    "      0.00 GB      None  |      115940         0\n"
    " gpfs_projects    GRP     1839.71 GB     4000.00 GB     4200.00 GB"
    "     12.13 GB      None  |     1726126      1831\n"
    "  gpfs_scratch    GRP     1790.89 GB     2000.00 GB     2100.00 GB"
    "      0.00 GB      None  |     1665977         0\n"
)

ANSI_TB_SAMPLE = (
    "\x1b[1m\x1b[38;5;117m Printing quota for group etur92:\x1b[22m\x1b[0m\n"
    "\x1b[1;30m\x1b[47m    Filesystem   Type          Usage          Quota"
    "          Limit     In doubt     Grace  |       Files  In doubt  \x1b[0m\n"
    "\x1b[38;5;252m\x1b[48;5;235m gpfs_projects    GRP        1.79 TB        3.91 TB"
    "        4.10 TB\x1b[38;5;252m      4.69 GB      None  |     1723914       469  \x1b[0m\n"
)


def _with_header(row: str) -> str:
    return " Printing quota for group etur92:\n" + row + "\n"


def test_parse_live_gb_sample() -> None:
    parsed = gate.parse_bsc_quota(GB_SAMPLE)
    assert parsed["filesystem"] == "gpfs_projects"
    assert parsed["usage_gb"] == pytest.approx(1839.71)
    assert parsed["quota_gb"] == pytest.approx(4000.00)
    assert parsed["limit_gb"] == pytest.approx(4200.00)
    assert parsed["in_doubt_gb"] == pytest.approx(12.13)


def test_evaluate_allows_the_live_snapshot() -> None:
    parsed = gate.parse_bsc_quota(GB_SAMPLE)
    ok, reason, remaining = gate.evaluate(
        parsed, min_remaining_gb=500.0, local_free_gb=231.0, min_local_gb=50.0
    )
    assert ok is True
    assert remaining == pytest.approx(4000.00 - 1839.71 - 12.13)
    assert "remaining" in reason


def test_reserve_boundary_is_exact_in_gb() -> None:
    parsed = gate.parse_bsc_quota(GB_SAMPLE)
    parsed["usage_gb"] = 4000.00 - 12.13 - 500.0
    ok, _, remaining = gate.evaluate(parsed, min_remaining_gb=500.0)
    assert ok is True and remaining == pytest.approx(500.0)
    parsed["usage_gb"] = 4000.00 - 12.13 - 499.99
    ok, _, remaining = gate.evaluate(parsed, min_remaining_gb=500.0)
    assert ok is False and remaining < 500.0


def test_evaluate_refuses_below_project_reserve() -> None:
    parsed = gate.parse_bsc_quota(GB_SAMPLE)
    parsed["usage_gb"] = 3600.0
    ok, reason, remaining = gate.evaluate(parsed, min_remaining_gb=500.0)
    assert ok is False
    assert "remaining" in reason
    assert remaining < 500.0


def test_evaluate_refuses_low_local_space() -> None:
    parsed = gate.parse_bsc_quota(GB_SAMPLE)
    ok, reason, _ = gate.evaluate(
        parsed, min_remaining_gb=500.0, local_free_gb=40.0, min_local_gb=50.0
    )
    assert ok is False
    assert "local free" in reason


def test_parse_tb_sample_with_conversion_when_gb_not_required() -> None:
    parsed = gate.parse_bsc_quota(ANSI_TB_SAMPLE, require_gb=False)
    assert parsed["usage_gb"] == pytest.approx(1.79 * 1024)
    assert parsed["quota_gb"] == pytest.approx(3.91 * 1024)
    assert parsed["in_doubt_gb"] == pytest.approx(4.69)


def test_parse_refuses_missing_group_header() -> None:
    text = "    Filesystem Type Usage Quota Limit In doubt Grace\n gpfs_projects GRP 1 GB 2 GB 3 GB 0 GB None\n"
    with pytest.raises(gate.StorageGateError, match="no 'group <name>' header"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_wrong_group_header() -> None:
    text = " Printing quota for group other42:\n gpfs_projects GRP 1 GB 4000 GB 4200 GB 0 GB None\n"
    with pytest.raises(gate.StorageGateError, match="expected 'etur92'"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_non_gb_units_in_live_mode() -> None:
    text = _with_header(" gpfs_projects GRP 1.79 TB 3.91 TB 4.10 TB 4.69 GB None")
    with pytest.raises(gate.StorageGateError, match="not in GB units"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_missing_row() -> None:
    text = _with_header(" gpfs_home USR 31.88 GB 80.00 GB 84.00 GB 0.00 GB None")
    with pytest.raises(gate.StorageGateError, match="no gpfs_projects row"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_empty_output() -> None:
    with pytest.raises(gate.StorageGateError, match="empty"):
        gate.parse_bsc_quota("   \n")


def test_parse_refuses_unknown_unit() -> None:
    text = _with_header(" gpfs_projects GRP 1.79 XB 4000.00 GB 4200.00 GB 12.13 GB None")
    with pytest.raises(gate.StorageGateError, match="not in GB units"):
        gate.parse_bsc_quota(text)
    with pytest.raises(gate.StorageGateError, match="unknown size unit"):
        gate.parse_bsc_quota(text, require_gb=False)


def test_parse_refuses_unparseable_number() -> None:
    text = _with_header(" gpfs_projects GRP many GB 4000.00 GB 4200.00 GB 12.13 GB None")
    with pytest.raises(gate.StorageGateError, match="unparseable size"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_non_finite_and_negative() -> None:
    with pytest.raises(gate.StorageGateError, match="non-finite"):
        gate.parse_bsc_quota(
            _with_header(" gpfs_projects GRP nan GB 4000.00 GB 4200.00 GB 12.13 GB None")
        )
    with pytest.raises(gate.StorageGateError, match="negative"):
        gate.parse_bsc_quota(
            _with_header(" gpfs_projects GRP -1.00 GB 4000.00 GB 4200.00 GB 12.13 GB None")
        )


def test_parse_refuses_non_group_row() -> None:
    text = _with_header(" gpfs_projects USR 1.00 GB 4000.00 GB 4200.00 GB 0.00 GB None")
    with pytest.raises(gate.StorageGateError, match="not a group row"):
        gate.parse_bsc_quota(text)


def test_parse_refuses_short_row() -> None:
    text = _with_header(" gpfs_projects GRP 1.00 GB 4000.00 GB")
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


def test_query_uses_exact_gb_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Fake:
        returncode = 0
        stdout = GB_SAMPLE
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return Fake()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    output = gate.query_quota("user@host")
    assert "bsc_quota projects --unit GB --no-color" in captured["argv"]
    assert "etur92" in output


def test_size_units() -> None:
    assert gate._to_gb("1", "TB") == pytest.approx(1024.0)
    assert gate._to_gb("512", "GB") == pytest.approx(512.0)
    assert gate._to_gb("1", "MB") == pytest.approx(1.0 / 1024.0)
    assert gate._to_gb("1", "B") == pytest.approx(1.0 / (1024.0 ** 3))


def test_cli_parse_file_ok_and_refused(tmp_path: Path) -> None:
    sample = tmp_path / "quota.txt"
    sample.write_text(GB_SAMPLE, encoding="utf-8")
    assert gate.main(["--parse-file", str(sample), "--json"]) == 0
    heavy = tmp_path / "heavy.txt"
    heavy.write_text(GB_SAMPLE.replace("1839.71 GB", "3900.00 GB"), encoding="utf-8")
    assert gate.main(["--parse-file", str(heavy), "--json"]) == 1
