"""Native/English manifest equivalence audit.

The audit is the Phase B gate that proves an English cell differs from its
native counterpart by the transcript text alone: same rows, subjects, labels and
audio files, with translation provenance on every row and no native fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_en_manifest_equivalence as audit


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _rows(*, text_prefix: str = "text", language: str = "en", variant: str = "english",
          sha: str = "a" * 64, labels: tuple[int, ...] = (1, 0)) -> list[dict]:
    return [
        {
            "dataset": "cmdc",
            "subject_id": f"s{index}",
            "sample_id": f"s{index}_Q1",
            "label": label,
            "audio_path": f"/data/s{index}/Q1.wav",
            "transcript": f"{text_prefix}-{index}",
            "language": language,
            "transcript_variant": variant,
            "translation_sha256": sha,
        }
        for index, label in enumerate(labels)
    ]


def test_matching_pair_passes(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="native"))
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="english"))
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "passed"
    assert record["failures"] == []
    assert record["rows"] == {"native": 2, "english": 2}


def test_label_drift_fails(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="native"))
    english_rows = _rows(text_prefix="english")
    english_rows[0]["label"] = 0
    english = _write(tmp_path / "english.jsonl", english_rows)
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "failed"
    assert any("label differs" in failure for failure in record["failures"])


def test_native_fallback_fails(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="same"))
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="same"))
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "failed"
    assert any("native fallback" in failure for failure in record["failures"])


def test_missing_translation_markers_fail(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="native"))
    english = _write(
        tmp_path / "english.jsonl",
        _rows(text_prefix="english", language="", variant="", sha=""),
    )
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "failed"
    assert sum("lack" in failure for failure in record["failures"]) == 3


def test_geriatri_rows_fail(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="native"))
    english_rows = _rows(text_prefix="english")
    english_rows[0]["subject_id"] = "geriatri:s1"
    english = _write(tmp_path / "english.jsonl", english_rows)
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "failed"
    assert any("geriatri" in failure for failure in record["failures"])


def test_row_set_drift_fails(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="native"))
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="english")[:1])
    record = audit.audit_pair("cmdc", native, english)
    assert record["status"] == "failed"
    assert any("only in the native manifest" in failure for failure in record["failures"])


def test_duplicate_sample_ids_fail(tmp_path: Path) -> None:
    rows = _rows(text_prefix="native")
    native = _write(tmp_path / "native.jsonl", rows + [rows[0]])
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="english"))
    with pytest.raises(audit.AuditError, match="duplicate sample_ids"):
        audit.audit_pair("cmdc", native, english)


def test_missing_manifest_fails(tmp_path: Path) -> None:
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="english"))
    with pytest.raises(audit.AuditError, match="missing native manifest"):
        audit.audit_pair("cmdc", tmp_path / "absent.jsonl", english)


def test_cli_writes_the_audit_and_returns_nonzero_on_failure(tmp_path: Path) -> None:
    native = _write(tmp_path / "native.jsonl", _rows(text_prefix="same"))
    english = _write(tmp_path / "english.jsonl", _rows(text_prefix="same"))
    output = tmp_path / "audit.json"
    code = audit.main(
        ["--pair", "cmdc", str(native), str(english), "--output", str(output)]
    )
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["pairs"][0]["dataset"] == "cmdc"
