from __future__ import annotations

from src.experiment_tracking.validate import _canonical_aggregation


def test_subject_aggregation_config_and_evidence_spellings_match() -> None:
    assert _canonical_aggregation("subject") == _canonical_aggregation("subject_level")


def test_other_aggregation_values_remain_distinct() -> None:
    assert _canonical_aggregation("response_subject") != _canonical_aggregation("subject_level")
