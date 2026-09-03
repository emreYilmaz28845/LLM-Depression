from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.turkish_question_condition_preflight import (
    EXPECTED_LABELS,
    PreflightError,
    _subject_contract,
    _translation_audit,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"sample_id": "a", "subject_id": "s1", "label": 1},
        {"sample_id": "b", "subject_id": "s1", "label": 1},
        {"sample_id": "c", "subject_id": "s2", "label": 0},
    ]


def test_subject_contract_rejects_duplicate_or_mixed_labels() -> None:
    with pytest.raises(PreflightError, match="duplicate sample IDs"):
        _subject_contract(_rows() + [_rows()[0]], condition="pos_only")
    with pytest.raises(PreflightError, match="inconsistent labels"):
        _subject_contract(_rows() + [{"sample_id": "d", "subject_id": "s1", "label": 0}], condition="pos_only")


def test_translation_audit_is_fail_closed(tmp_path: Path) -> None:
    cache = tmp_path / "accepted.jsonl"
    cache.write_text(
        json.dumps(
            {
                "unit_id": "u1",
                "field": "transcript",
                "part_index": 0,
                "status": "automatic_high",
                "translation_sha256": "abc",
                "fallback": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = _translation_audit(
        {
            "transcripts": {"cache_path": str(cache)},
        },
        "pos_only",
        1,
    )
    assert payload["accepted_rows"] == 1
    assert payload["failures"] == []

    cache.write_text(
        json.dumps(
            {
                "unit_id": "u1",
                "field": "transcript",
                "part_index": 0,
                "status": "rejected",
                "translation_sha256": "",
                "fallback": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rejected = _translation_audit({"transcripts": {"cache_path": str(cache)}}, "pos_only", 1)
    assert rejected["failures"]


def test_locked_label_contract_is_explicit() -> None:
    assert EXPECTED_LABELS == {0: 37, 1: 83}
