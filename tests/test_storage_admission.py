from __future__ import annotations

import pytest

from scripts.dispatch_androids_fixed_merged import (
    AdmissionError,
    parse_bsc_quota_projects,
    storage_admission,
)

REAL_SAMPLE = """
 Printing quota for group etur92:

    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace  |       Files  In doubt
     gpfs_home    USR       31.88 GB       80.00 GB       84.00 GB      0.00 KB      None  |      115940         0
 gpfs_projects    GRP        1.79 TB        3.91 TB        4.10 TB      4.69 GB      None  |     1723914       469
  gpfs_scratch    GRP        1.75 TB        1.95 TB        2.05 TB     89.94 MB      None  |     1665976        39
"""

ANSI_SAMPLE = (
    "\x1b[1m\x1b[38;5;117m Printing quota for group etur92:\x1b[22m\x1b[0m\n"
    "\x1b[1;30m\x1b[47m    Filesystem   Type          Usage          Quota"
    "          Limit     In doubt     Grace\x1b[0m\n"
    "\x1b[38;5;252m\x1b[48;5;235m gpfs_projects    GRP        1.79 TB        3.91 TB"
    "        4.10 TB\x1b[38;5;252m      4.69 GB      None  |     1723914       469  \x1b[0m\n"
)


def test_parse_real_sample() -> None:
    parsed = parse_bsc_quota_projects(REAL_SAMPLE)
    assert parsed["usage_gib"] == pytest.approx(1.79 * 1024)
    assert parsed["soft_quota_gib"] == pytest.approx(3.91 * 1024)
    assert parsed["hard_limit_gib"] == pytest.approx(4.10 * 1024)
    assert parsed["in_doubt_gib"] == pytest.approx(4.69)


def test_parse_strips_ansi_codes() -> None:
    parsed = parse_bsc_quota_projects(ANSI_SAMPLE)
    assert parsed["soft_quota_gib"] == pytest.approx(3.91 * 1024)
    assert parsed["in_doubt_gib"] == pytest.approx(4.69)


def test_parse_missing_group_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("group etur92", "group other"))


def test_parse_missing_row_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("gpfs_projects", "gpfs_other"))


def test_parse_non_numeric_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("1.79 TB", "abc TB"))


def test_parse_unknown_unit_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("3.91 TB", "3.91 XB"))


def test_parse_negative_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("4.69 GB", "-4.69 GB"))


def test_storage_gate_passes_real_values() -> None:
    result = storage_admission(REAL_SAMPLE, local_free_gib=100.0)
    expected = 3.91 * 1024 - 1.79 * 1024 - 4.69
    assert result["project_remaining_gib"] == pytest.approx(expected)
    assert result["local_free_gib"] == 100.0


def test_storage_gate_project_reserve_refuses() -> None:
    tight = REAL_SAMPLE.replace("1.79 TB", "3.60 TB")
    with pytest.raises(AdmissionError):
        storage_admission(tight, local_free_gib=100.0)


def test_storage_gate_local_reserve_refuses() -> None:
    with pytest.raises(AdmissionError):
        storage_admission(REAL_SAMPLE, local_free_gib=49.9)


def test_storage_gate_boundary() -> None:
    just_enough = REAL_SAMPLE.replace("1.79 TB", "3.42 TB").replace("4.69 GB", "0.00 KB")
    result = storage_admission(just_enough, local_free_gib=50.0)
    assert result["project_remaining_gib"] == pytest.approx(501.76, abs=0.01)
    too_tight = REAL_SAMPLE.replace("1.79 TB", "3.43 TB").replace("4.69 GB", "0.00 KB")
    with pytest.raises(AdmissionError):
        storage_admission(too_tight, local_free_gib=50.0)
