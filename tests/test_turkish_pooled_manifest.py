"""Pooled manifest concat + split verification (plan Step 2)."""

from __future__ import annotations

import json

import pytest

from scripts.build_turkish_pooled_manifest import (
    concat_and_audit,
    verify_splits_identical,
)


def _row(sample_id: str, subject_id: str, variant: str, label: int = 1) -> dict:
    return {
        "dataset": "turkish",
        "subject_id": subject_id,
        "sample_id": sample_id,
        "audio_path": f"/tmp/{sample_id}.wav",
        "transcript": "merhaba",
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "dataset_variant": variant,
        "translation_sha256": "abc123",
    }


def test_concat_stacks_and_preserves_rows_byte_identically(tmp_path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    pos = [_row("p1", "s1", "pos_only_t17"), _row("p2", "s2", "pos_only_t17", label=0)]
    neg = [_row("n1", "s1", "negative_only_t17"), _row("n2", "s2", "negative_only_t17", label=0)]
    for row in pos + neg:
        row["audio_path"] = str(audio)
    pooled, audit = concat_and_audit(
        pos, neg, expected_pos=2, expected_neg=2, require_translation_sha=True
    )
    assert len(pooled) == 4
    assert audit["pooled_subject_count"] == 2
    # rows byte-identical to sources (same dict content)
    by_id = {row["sample_id"]: row for row in pooled}
    assert by_id["p1"]["transcript"] == "merhaba"
    assert by_id["p1"]["dataset_variant"] == "pos_only_t17"
    assert by_id["n1"]["dataset_variant"] == "negative_only_t17"
    assert "transcript_asymmetry" in audit
    assert audit["manifest_hash"]


def test_concat_fail_closed(tmp_path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    def rows_with_audio(rows: list[dict]) -> list[dict]:
        for row in rows:
            row["audio_path"] = str(audio)
        return rows

    pos = rows_with_audio([_row("p1", "s1", "pos_only_t17")])
    # wrong count
    with pytest.raises(ValueError):
        concat_and_audit(pos, [], expected_pos=2, expected_neg=0, check_audio_exists=False)
    # wrong variant in pos pile
    bad = rows_with_audio([_row("p1", "s1", "negative_only_t17")])
    with pytest.raises(ValueError):
        concat_and_audit(bad, [], expected_pos=1, expected_neg=0, check_audio_exists=False)
    # duplicate sample id across piles
    dup_neg = rows_with_audio([_row("p1", "s9", "negative_only_t17")])
    with pytest.raises(ValueError):
        concat_and_audit(pos, dup_neg, expected_pos=1, expected_neg=1, check_audio_exists=False)
    # conflicting labels for one subject
    conflict_neg = rows_with_audio([_row("n1", "s1", "negative_only_t17", label=0)])
    with pytest.raises(ValueError):
        concat_and_audit(pos, conflict_neg, expected_pos=1, expected_neg=1, check_audio_exists=False)
    # missing audio
    missing = [_row("n1", "s1", "negative_only_t17")]
    missing[0]["audio_path"] = str(tmp_path / "nope.wav")
    with pytest.raises(ValueError):
        concat_and_audit(pos, missing, expected_pos=1, expected_neg=1)
    # missing translation sha in EN mode
    no_sha = rows_with_audio([_row("n1", "s1", "negative_only_t17")])
    del no_sha[0]["translation_sha256"]
    with pytest.raises(ValueError):
        concat_and_audit(
            pos, no_sha, expected_pos=1, expected_neg=1, require_translation_sha=True
        )


def test_verify_splits_identical_pass_and_abort(tmp_path) -> None:
    folds = {"0": {"outer_train_subject_ids": ["s1"], "final_eval_subject_ids": ["s2"]}}
    pos = tmp_path / "pos_folds.json"
    neg = tmp_path / "neg_folds.json"
    pos.write_text(json.dumps(folds), encoding="utf-8")
    neg.write_text(json.dumps(folds), encoding="utf-8")
    audit = verify_splits_identical(pos, neg)
    assert audit["folds_match"] is True
    assert audit["pos_folds_sha256"] == audit["neg_folds_sha256"]

    neg.write_text(json.dumps({"0": {"outer_train_subject_ids": ["s2"], "final_eval_subject_ids": ["s1"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="ABORT"):
        verify_splits_identical(pos, neg)
