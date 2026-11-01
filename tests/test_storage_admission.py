from __future__ import annotations

import pytest

from scripts.dispatch_androids_fixed_merged import (
    AdmissionError,
    BSC_QUOTA_COMMAND,
    parse_bsc_quota_projects,
    storage_admission,
)

REAL_SAMPLE = """
 Printing quota for group etur92:

    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace  |       Files  In doubt
     gpfs_home    USR       31.88 GB       80.00 GB       84.00 GB      0.00 GB      None  |      115940         0
 gpfs_projects    GRP     1875.47 GB     4000.00 GB     4200.00 GB      3.84 GB      None  |     1727031       414
  gpfs_scratch    GRP     1790.89 GB     2000.00 GB     2100.00 GB      0.00 GB      None  |     1665977         0
"""

ANSI_SAMPLE = (
    "\x1b[0m\x1b[0m Printing quota for group etur92:\x1b[22m\x1b[0m\n"
    "\x1b[1;30m\x1b[0m    Filesystem   Type          Usage          Quota"
    "          Limit     In doubt     Grace\x1b[0m\n"
    "\x1b[0m gpfs_projects    GRP     1875.47 GB     4000.00 GB     4200.00 GB"
    "\x1b[38;5;252m      3.84 GB      None  |     1727031       414  \x1b[0m\n"
)


def test_command_is_exact_gb_no_color() -> None:
    assert BSC_QUOTA_COMMAND == "bsc_quota projects --unit GB --no-color"


def test_parse_real_gb_sample() -> None:
    parsed = parse_bsc_quota_projects(REAL_SAMPLE)
    assert parsed["usage_gib"] == pytest.approx(1875.47)
    assert parsed["soft_quota_gib"] == pytest.approx(4000.00)
    assert parsed["hard_limit_gib"] == pytest.approx(4200.00)
    assert parsed["in_doubt_gib"] == pytest.approx(3.84)


def test_parse_strips_residual_ansi_codes() -> None:
    parsed = parse_bsc_quota_projects(ANSI_SAMPLE)
    assert parsed["soft_quota_gib"] == pytest.approx(4000.00)
    assert parsed["in_doubt_gib"] == pytest.approx(3.84)


def test_parse_missing_group_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("group etur92", "group other"))


def test_parse_missing_row_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("gpfs_projects", "gpfs_other"))


def test_parse_duplicate_row_refuses() -> None:
    duplicated = REAL_SAMPLE + " gpfs_projects    GRP     1875.47 GB     4000.00 GB     4200.00 GB      3.84 GB      None\n"
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(duplicated)


def test_parse_wrong_row_type_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("GRP     1875.47", "USR     1875.47"))


def test_parse_non_finite_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("1875.47 GB", "nan GB"))
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("4000.00 GB", "inf GB"))
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("3.84 GB", "-3.84 GB"))


def test_parse_unknown_unit_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("4000.00 GB", "4000.00 XB"))


def test_parse_hard_limit_below_soft_refuses() -> None:
    with pytest.raises(AdmissionError):
        parse_bsc_quota_projects(REAL_SAMPLE.replace("4200.00 GB", "3900.00 GB"))


def test_storage_gate_passes_real_values() -> None:
    result = storage_admission(REAL_SAMPLE, local_free_gib=231.03)
    assert result["project_remaining_gib"] == pytest.approx(4000.00 - 1875.47 - 3.84)
    assert result["local_free_gib"] == pytest.approx(231.03)


def test_storage_gate_project_reserve_refuses() -> None:
    tight = REAL_SAMPLE.replace("1875.47 GB", "3600.00 GB")
    with pytest.raises(AdmissionError):
        storage_admission(tight, local_free_gib=100.0)


def test_storage_gate_local_reserve_refuses() -> None:
    with pytest.raises(AdmissionError):
        storage_admission(REAL_SAMPLE, local_free_gib=49.9)


def test_storage_gate_non_finite_local_refuses() -> None:
    import math

    with pytest.raises(AdmissionError):
        storage_admission(REAL_SAMPLE, local_free_gib=float("nan"))
    with pytest.raises(AdmissionError):
        storage_admission(REAL_SAMPLE, local_free_gib=math.inf)
    with pytest.raises(AdmissionError):
        storage_admission(REAL_SAMPLE, local_free_gib=-1.0)


def test_storage_gate_boundary() -> None:
    just_enough = REAL_SAMPLE.replace("1875.47 GB", "3496.16 GB").replace("3.84 GB", "0.00 GB")
    result = storage_admission(just_enough, local_free_gib=50.0)
    assert result["project_remaining_gib"] == pytest.approx(503.84, abs=0.01)
    too_tight = REAL_SAMPLE.replace("1875.47 GB", "3501.00 GB").replace("3.84 GB", "0.00 GB")
    with pytest.raises(AdmissionError):
        storage_admission(too_tight, local_free_gib=50.0)
