from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.build_manifest import manifest_build_signature
from src.translation.overlay import apply_overlay
from src.translation.translate import _load_candidates, _load_units, run_translation
from src.translation.units import (
    DATASET_UNIT_FIELDS,
    SOURCE_LANGUAGE_BY_DATASET,
    export_units,
    split_source_for_budget,
    unit_rows_for_dataset,
)
from src.translation.validate import (
    english_only,
    leakage_checks,
    number_preservation_ok,
    run_validation,
    sensitive_term_violations,
)
from src.utils import sha256_text, write_jsonl


def _cmdc_rows() -> list[dict]:
    return [
        {
            "dataset": "cmdc",
            "subject_id": "MDD001",
            "sample_id": "MDD001_Q1",
            "transcript": "我最近睡得很不好。",
            "label": 1,
            "score": 12,
        },
        {
            "dataset": "cmdc",
            "subject_id": "MDD001",
            "sample_id": "MDD001_Q2",
            "transcript": "医生给我开了药。",
            "label": 1,
            "score": 12,
        },
        {
            "dataset": "cmdc",
            "subject_id": "HC001",
            "sample_id": "HC001_Q1",
            "transcript": "我很好，谢谢。",
            "label": 0,
            "score": "",
        },
    ]


def _turkish_rows() -> list[dict]:
    return [
        {
            "dataset": "turkish",
            "subject_id": "P01",
            "sample_id": "P01-a-001-b.wav",
            "chunk_id": "001",
            "transcript": "Bugün kendimi iyi hissetmiyorum.",
            "label": 1,
            "score": 18.0,
        },
        {
            "dataset": "turkish",
            "subject_id": "P01",
            "sample_id": "P01-a-002-b.wav",
            "chunk_id": "002",
            "transcript": "Üç gündür uyuyamıyorum.",
            "label": 1,
            "score": 18.0,
        },
    ]


def _d3tec_rows() -> list[dict]:
    return [
        {
            "dataset": "d3tec",
            "subject_id": "001",
            "response_id": "001_p0",
            "sample_id": "001_p0_s0",
            "segment_index": 0,
            "prompt_id": 0,
            "transcript": "No he dormido bien esta semana.",
            "segment_transcript": "No he dormido bien esta semana.",
            "full_response_transcript": "No he dormido bien esta semana. Me siento muy cansado.",
            "label": 1,
            "score": 12,
        },
        {
            "dataset": "d3tec",
            "subject_id": "001",
            "response_id": "001_p0",
            "sample_id": "001_p0_s1",
            "segment_index": 1,
            "prompt_id": 0,
            "transcript": "Me siento muy cansado.",
            "segment_transcript": "Me siento muy cansado.",
            "full_response_transcript": "No he dormido bien esta semana. Me siento muy cansado.",
            "label": 1,
            "score": 12,
        },
    ]


def _androids_rows() -> list[dict]:
    return [
        {
            "dataset": "androids_interview",
            "subject_id": "01_P",
            "response_id": "01_PF35_14_1",
            "turn_id": 1,
            "window_index": 0,
            "sample_id": "01_PF35_14_1_w00",
            "transcript": "Non riesco a dormire da due settimane.",
            "segment_transcript": "Non riesco a dormire da due settimane.",
            "full_turn_transcript": "Non riesco a dormire da due settimane. Prendo farmaci.",
            "label": 1,
        },
    ]


def _write_rows(tmp_path: Path, rows: list[dict], name: str = "manifest.jsonl") -> Path:
    path = tmp_path / name
    write_jsonl(rows, path)
    return path


def test_export_cmdc_is_label_free_and_deterministic() -> None:
    rows = _cmdc_rows()
    units = unit_rows_for_dataset(rows, "cmdc")
    assert [unit["unit_id"] for unit in units] == ["HC001_Q1", "MDD001_Q1", "MDD001_Q2"]
    assert units[1]["source_language"] == "zh"
    assert units[1]["target_language"] == "en"
    assert units[1]["source_sha256"] == sha256_text(units[1]["source_text"])
    assert units[1]["context_id"] == ""
    assert units[2]["context_id"] == "MDD001_context"
    assert "最近睡得很不好" in units[2]["context_text"]
    for unit in units:
        serialized = json.dumps(unit, sort_keys=True)
        assert "label" not in serialized
        assert "score" not in serialized
        assert "fold" not in serialized


def test_export_turkish_context_uses_chunk_order() -> None:
    units = unit_rows_for_dataset(_turkish_rows(), "turkish")
    second = [unit for unit in units if unit["unit_id"] == "P01-a-002-b.wav"][0]
    assert second["context_id"] == "P01_context"
    assert "iyi hissetmiyorum" in second["context_text"]
    first = [unit for unit in units if unit["unit_id"] == "P01-a-001-b.wav"][0]
    assert first["context_text"] == ""


def test_export_d3tec_emits_full_and_segment_units() -> None:
    units = unit_rows_for_dataset(_d3tec_rows(), "d3tec")
    fields = {unit["field"] for unit in units}
    assert fields == {"full_response_transcript", "segment_transcript"}
    full = [unit for unit in units if unit["field"] == "full_response_transcript"]
    segments = [unit for unit in units if unit["field"] == "segment_transcript"]
    assert len(full) == 1
    assert len(segments) == 2
    assert full[0]["unit_id"] == "001_p0"
    assert segments[0]["unit_id"] == "001_p0_s0"
    assert segments[0]["context_id"] == "001_p0"
    assert "cansado" in segments[0]["context_text"]


def test_export_androids_emits_full_turn_and_window_units() -> None:
    units = unit_rows_for_dataset(_androids_rows(), "androids_interview")
    fields = {unit["field"] for unit in units}
    assert fields == {"full_turn_transcript", "segment_transcript"}
    full = [unit for unit in units if unit["field"] == "full_turn_transcript"]
    assert full[0]["unit_id"] == "01_PF35_14_1"


def test_export_rejects_unsupported_dataset() -> None:
    with pytest.raises(ValueError):
        unit_rows_for_dataset(_cmdc_rows(), "daic")


def test_export_end_to_end(tmp_path) -> None:
    manifest = _write_rows(tmp_path, _cmdc_rows())
    audit = export_units(manifest, "cmdc", tmp_path / "units.jsonl")
    assert audit["unit_count"] == 3
    written = tmp_path / "units.jsonl"
    assert len(written.read_text().splitlines()) == 3


def test_split_source_for_budget_splits_at_sentence_boundaries() -> None:
    text = "第一句。第二句！第三句？第四句。"
    parts = split_source_for_budget(text, max_chars=8)
    assert len(parts) > 1
    assert "第一句。" in parts[0]
    assert "".join(parts) == text
    with pytest.raises(ValueError):
        split_source_for_budget("很长的单句没有分号。", max_chars=3)


def _candidate(unit: dict, translation: str) -> dict:
    return {
        "dataset": unit["dataset"],
        "unit_id": unit["unit_id"],
        "field": unit["field"],
        "part_index": unit.get("part_index", 0),
        "part_count": unit.get("part_count", 1),
        "translation": translation,
        "translation_sha256": sha256_text(translation),
        "model": "Qwen/Qwen3.6-27B",
        "model_revision": "rev1",
        "precision": "bf16",
        "prompt_version": "clinical_faithful_v1",
        "source_sha256": unit["source_sha256"],
        "status": "translated",
    }


def _accepted(unit: dict, translation: str, status: str = "automatic_high") -> dict:
    return {
        "dataset": unit["dataset"],
        "unit_id": unit["unit_id"],
        "field": unit["field"],
        "part_index": unit.get("part_index", 0),
        "part_count": unit.get("part_count", 1),
        "translation": translation,
        "translation_sha256": sha256_text(translation),
        "model": "Qwen/Qwen3.6-27B",
        "model_revision": "rev1",
        "precision": "bf16",
        "prompt_version": "clinical_faithful_v1",
        "source_sha256": unit["source_sha256"],
        "status": status,
    }


def test_overlay_replaces_fields_and_adds_provenance() -> None:
    rows = _cmdc_rows()
    units = unit_rows_for_dataset(rows, "cmdc")
    accepted = [
        _accepted(units[0], "I am fine, thank you."),
        _accepted(units[1], "I have been sleeping very poorly lately."),
        _accepted(units[2], "The doctor prescribed me some medicine."),
    ]
    overlaid, audit = apply_overlay(
        rows, "cmdc", accepted, minimum_status="automatic_high", require_complete=True
    )
    assert audit["replaced_rows"] == 3
    row = next(row for row in overlaid if row["sample_id"] == "HC001_Q1")
    assert row["transcript"] == "I am fine, thank you."
    assert row["transcript_original"] == "我很好，谢谢。"
    assert row["language"] == "en"
    assert row["source_language"] == "zh"
    assert row["transcript_variant"] == "english"
    assert row["translation_status"] == "automatic_high"
    assert row["translation_sha256"] == sha256_text(row["transcript"])


def test_overlay_joins_parts_in_order() -> None:
    source = "第一句。第二句。"
    parts = split_source_for_budget(source, max_chars=5)
    assert len(parts) == 2
    row = {
        "dataset": "cmdc",
        "subject_id": "MDD001",
        "sample_id": "MDD001_Q1",
        "transcript": source,
    }
    units = unit_rows_for_dataset([row], "cmdc", max_source_chars=5)
    assert len(units) == 2
    accepted = [
        _accepted(units[0], "First sentence."),
        _accepted(units[1], "Second sentence."),
    ]
    overlaid, _ = apply_overlay(
        [row], "cmdc", accepted, minimum_status="automatic_high", require_complete=True
    )
    assert overlaid[0]["transcript"] == "First sentence. Second sentence."


def test_overlay_require_complete_fails_on_missing_unit() -> None:
    rows = _cmdc_rows()
    units = unit_rows_for_dataset(rows, "cmdc")
    accepted = [_accepted(units[0], "Partial coverage.")]
    with pytest.raises(ValueError):
        apply_overlay(
            rows, "cmdc", accepted, minimum_status="automatic_high", require_complete=True
        )


def test_overlay_minimum_status_filters_low_status() -> None:
    rows = _cmdc_rows()
    units = unit_rows_for_dataset(rows, "cmdc")
    accepted = [
        _accepted(units[0], "Low confidence.", status="automatic_low"),
        _accepted(units[1], "The doctor prescribed me some medicine."),
        _accepted(units[2], "I am fine, thank you."),
    ]
    with pytest.raises(ValueError):
        apply_overlay(
            rows, "cmdc", accepted, minimum_status="automatic_high", require_complete=True
        )
    overlaid, audit = apply_overlay(
        rows, "cmdc", accepted, minimum_status="automatic_low", require_complete=True
    )
    assert audit["replaced_rows"] == 3


def test_overlay_d3tec_multi_field() -> None:
    rows = _d3tec_rows()
    units = unit_rows_for_dataset(rows, "d3tec")
    accepted = [
        _accepted(units[0], "I have not slept well this week. I feel very tired."),
        _accepted(units[1], "I have not slept well this week."),
        _accepted(units[2], "I feel very tired."),
    ]
    overlaid, audit = apply_overlay(
        rows, "d3tec", accepted, minimum_status="automatic_high", require_complete=True
    )
    assert audit["replaced_rows"] == 2
    row = overlaid[0]
    assert row["segment_transcript"] == "I have not slept well this week."
    assert row["full_response_transcript"] == "I have not slept well this week. I feel very tired."
    assert row["transcript"] == row["segment_transcript"]
    assert row["language"] == "en"
    assert row["source_language"] == "es"
    assert row["transcript_original"]["segment_transcript"] == "No he dormido bien esta semana."


def test_manifest_signature_ignores_original_variant_and_includes_english() -> None:
    base = {
        "dataset": "cmdc",
        "seed": 1337,
        "output_dirs": {},
        "data": {"segment_seconds": 30.0},
        "split": {"seed": 1337, "cv_protocol": "train_val"},
    }
    without_block = manifest_build_signature(base)
    with_original = manifest_build_signature({**base, "transcripts": {"variant": "original"}})
    assert without_block == with_original
    with_english = manifest_build_signature(
        {
            **base,
            "transcripts": {
                "variant": "english",
                "cache_path": "/tmp/accepted.jsonl",
                "minimum_status": "automatic_high",
            },
        }
    )
    assert with_english != without_block
    assert "transcripts" in with_english["builder_options"]


def test_translate_resume_skips_matching_and_rejects_conflicts(tmp_path) -> None:
    units = unit_rows_for_dataset(_cmdc_rows(), "cmdc")
    units_path = tmp_path / "units.jsonl"
    write_jsonl(units, units_path)
    candidates_path = tmp_path / "candidates.jsonl"
    done = _candidate(units[0], "I have been sleeping very poorly lately.")
    done["model_revision"] = "rev1"
    write_jsonl([done], candidates_path)
    failed_path = tmp_path / "failed.jsonl"

    translated: list[dict] = []

    async def stub_translate_pending(base_url, model, model_revision, pending, *, batch_size, seed, max_retries):
        candidates = [_candidate(unit, f"EN-{unit['unit_id']}") for unit in pending]
        translated.extend(candidates)
        return [(unit, candidate, None) for unit, candidate in zip(pending, candidates)]

    import src.translation.translate as translate_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translate_module, "_translate_pending", stub_translate_pending)
    try:
        summary = run_translation(
            units_path,
            candidates_path,
            failed_path,
            base_url="http://x/v1",
            model="qwen3.6-27b",
            model_revision="rev1",
            batch_size=8,
            seed=42,
            max_retries=1,
            force_resync=False,
        )
        assert summary["skipped_on_resume"] == 1
        assert summary["completed"] == 3

        changed = dict(units[0])
        changed["source_text"] = "不同文本。"
        changed["source_sha256"] = sha256_text(changed["source_text"])
        write_jsonl([changed] + units[1:], units_path)
        with pytest.raises(ValueError, match="Resume conflict"):
            run_translation(
                units_path,
                candidates_path,
                failed_path,
                base_url="http://x/v1",
                model="qwen3.6-27b",
                model_revision="rev1",
                batch_size=8,
                seed=42,
                max_retries=1,
                force_resync=False,
            )
        summary = run_translation(
            units_path,
            candidates_path,
            failed_path,
            base_url="http://x/v1",
            model="qwen3.6-27b",
            model_revision="rev1",
            batch_size=8,
            seed=42,
            max_retries=1,
            force_resync=True,
        )
        assert summary["completed"] == 3
    finally:
        monkeypatch.undo()


def test_validate_statuses_and_checks(tmp_path) -> None:
    units = unit_rows_for_dataset(_cmdc_rows(), "cmdc")
    units_path = tmp_path / "units.jsonl"
    write_jsonl(units, units_path)
    candidates = [
        _candidate(units[0], "I have been sleeping very poorly lately."),
        _candidate(units[1], "The doctor prescribed me some medicine."),
        _candidate(units[2], "我很好，谢谢。"),
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl(candidates, candidates_path)
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    audit_path = tmp_path / "audit.json"
    audit = run_validation(
        units_path,
        candidates_path,
        accepted_path,
        rejected_path,
        audit_path,
        nllb_model=None,
        verifier_base_url=None,
        verifier_model=None,
        reviewed_path=None,
        seed=42,
    )
    assert audit["status_counts"]["automatic_high"] == 2
    assert audit["status_counts"]["failed"] == 1
    accepted = [json.loads(line) for line in accepted_path.read_text().splitlines()]
    rejected = [json.loads(line) for line in rejected_path.read_text().splitlines()]
    assert len(accepted) == 2
    assert len(rejected) == 1
    assert "CJK" in rejected[0]["reasons"][0]
    assert audit["accepted_cache_record_count"] == 2


def test_validate_coverage_failure(tmp_path) -> None:
    units = unit_rows_for_dataset(_cmdc_rows(), "cmdc")
    units_path = tmp_path / "units.jsonl"
    write_jsonl(units, units_path)
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl([_candidate(units[0], "Only one.")], candidates_path)
    with pytest.raises(ValueError, match="Coverage failure"):
        run_validation(
            units_path,
            candidates_path,
            tmp_path / "accepted.jsonl",
            tmp_path / "rejected.jsonl",
            tmp_path / "audit.json",
            nllb_model=None,
            verifier_base_url=None,
            verifier_model=None,
            reviewed_path=None,
            seed=42,
        )


def test_translate_failure_path_records_failed_unit(tmp_path) -> None:
    units = unit_rows_for_dataset(_cmdc_rows(), "cmdc")
    units_path = tmp_path / "units.jsonl"
    write_jsonl(units, units_path)
    candidates_path = tmp_path / "candidates.jsonl"
    failed_path = tmp_path / "failed.jsonl"

    async def stub_with_failure(base_url, model, model_revision, pending, *, batch_size, seed, max_retries):
        results = []
        for index, unit in enumerate(pending):
            if index == 0:
                results.append((unit, None, "translation_failed"))
            else:
                results.append((unit, _candidate(unit, f"EN-{unit['unit_id']}"), None))
        return results

    import src.translation.translate as translate_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translate_module, "_translate_pending", stub_with_failure)
    try:
        summary = run_translation(
            units_path,
            candidates_path,
            failed_path,
            base_url="http://x/v1",
            model="qwen3.6-27b",
            model_revision="rev1",
            batch_size=8,
            seed=42,
            max_retries=1,
            force_resync=False,
        )
    finally:
        monkeypatch.undo()
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    failed = [json.loads(line) for line in failed_path.read_text().splitlines()]
    assert failed[0]["unit_id"] == units[0]["unit_id"]
    assert failed[0]["status"] == "failed"
    assert failed[0]["reason"] == "translation_failed"


def test_validate_low_status_for_number_loss(tmp_path) -> None:
    unit = {
        "dataset": "cmdc",
        "unit_id": "u1",
        "field": "transcript",
        "scope": "response",
        "source_language": "zh",
        "target_language": "en",
        "source_text": "我吃了35片药，持续了2周。",
        "source_sha256": sha256_text("我吃了35片药，持续了2周。"),
        "context_id": "",
        "context_text": "",
        "context_sha256": "",
        "part_index": 0,
        "part_count": 1,
    }
    units_path = tmp_path / "units.jsonl"
    write_jsonl([unit], units_path)
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl([_candidate(unit, "I took some pills for a while.")], candidates_path)
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    audit_path = tmp_path / "audit.json"
    audit = run_validation(
        units_path,
        candidates_path,
        accepted_path,
        rejected_path,
        audit_path,
        nllb_model=None,
        verifier_base_url=None,
        verifier_model=None,
        reviewed_path=None,
        seed=42,
    )
    assert audit["status_counts"]["automatic_low"] == 1
    accepted = [json.loads(line) for line in accepted_path.read_text().splitlines()]
    assert "numbers not preserved" in accepted[0]["reasons"]


def test_validate_verifier_pass_runs_and_flags(tmp_path) -> None:
    unit = {
        "dataset": "cmdc",
        "unit_id": "long1",
        "field": "transcript",
        "scope": "response",
        "source_language": "zh",
        "target_language": "en",
        "source_text": "我最近睡得很不好。" * 400,
        "source_sha256": sha256_text("我最近睡得很不好。" * 400),
        "context_id": "",
        "context_text": "",
        "context_sha256": "",
        "part_index": 0,
        "part_count": 1,
    }
    units_path = tmp_path / "units.jsonl"
    write_jsonl([unit], units_path)
    candidate = _candidate(unit, "I have been sleeping very poorly lately.")
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl([candidate], candidates_path)

    async def fake_create(*, model, messages, temperature, top_p, max_tokens, seed, extra_body):
        class FakeMessage:
            content = '{"missing": ["the number 35"], "added": []}'

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        return FakeResponse()

    import src.translation.validate as validate_module

    class FakeCompletions:
        async def create(self, **kwargs):
            return await fake_create(**kwargs)

    class FakeChat:
        completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = FakeChat()

        async def close(self):
            pass

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(validate_module, "AsyncOpenAI", FakeAsyncOpenAI)
    try:
        audit = run_validation(
            units_path,
            candidates_path,
            tmp_path / "accepted.jsonl",
            tmp_path / "rejected.jsonl",
            tmp_path / "audit.json",
            nllb_model=None,
            verifier_base_url="http://x/v1",
            verifier_model="qwen3.6-27b",
            reviewed_path=None,
            seed=42,
        )
    finally:
        monkeypatch.undo()
    assert audit["verifier_pass"] is True
    assert audit["status_counts"]["automatic_low"] == 1
    accepted = [json.loads(line) for line in (tmp_path / "accepted.jsonl").read_text().splitlines()]
    assert any("missing invariants" in reason for reason in accepted[0]["reasons"])


def test_validate_reviewed_override(tmp_path) -> None:
    units = unit_rows_for_dataset(_cmdc_rows(), "cmdc")
    units_path = tmp_path / "units.jsonl"
    write_jsonl(units, units_path)
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl(
        [
            _candidate(units[0], "I have been sleeping very poorly lately."),
            _candidate(units[1], "The doctor prescribed me some medicine."),
            _candidate(units[2], "我很好，谢谢。"),
        ],
        candidates_path,
    )
    reviewed_path = tmp_path / "reviewed.jsonl"
    write_jsonl(
        [{"unit_id": "MDD001_Q2", "status": "human_verified", "reviewed_by": "user"}],
        reviewed_path,
    )
    audit = run_validation(
        units_path,
        candidates_path,
        tmp_path / "accepted.jsonl",
        tmp_path / "rejected.jsonl",
        tmp_path / "audit.json",
        nllb_model=None,
        verifier_base_url=None,
        verifier_model=None,
        reviewed_path=reviewed_path,
        seed=42,
    )
    assert audit["status_counts"]["human_verified"] == 1
    assert audit["reviewed_units"] == 1


def test_english_only_and_leakage_helpers() -> None:
    assert english_only("I am fine.", "zh") == []
    assert english_only("我很好。", "zh")
    assert english_only("Bugün şeyler iyiydi.", "tr")
    assert english_only("¿Cómo estás?", "es")
    unit = {
        "source_text": "这是一个测试。",
        "unit_id": "u1",
    }
    assert leakage_checks(unit, "Translation: 这是一个测试。")
    assert leakage_checks(unit, "<target> 这是一个测试。 </target>")


def test_sensitive_term_violations() -> None:
    unit = {
        "source_language": "tr",
        "source_text": "İntihar etmek istiyorum.",
    }
    assert sensitive_term_violations(unit, "I want to do something nice.")


def test_translate_loaders_reject_duplicates(tmp_path) -> None:
    rows = [{"unit_id": "a", "field": "transcript", "part_index": 0}, {"unit_id": "a", "field": "transcript", "part_index": 0}]
    path = tmp_path / "units.jsonl"
    write_jsonl(rows, path)
    with pytest.raises(ValueError, match="Duplicate unit key"):
        _load_units(path)
    candidates = [
        {"unit_id": "b", "field": "transcript", "part_index": 0},
        {"unit_id": "b", "field": "transcript", "part_index": 0},
    ]
    path2 = tmp_path / "candidates.jsonl"
    write_jsonl(candidates, path2)
    with pytest.raises(ValueError, match="Duplicate candidate key"):
        _load_candidates(path2)
