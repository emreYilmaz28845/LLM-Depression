from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.classify_androids_interviewer_context import (
    _extract_json_object,
    finalize,
    grounding_ratio,
    normalize_text,
    stable_id,
    validate_prediction,
    write_jsonl_atomic,
)


def test_model_json_and_prediction_validation() -> None:
    parsed = _extract_json_object(
        '```json\n{"context_type":"question_or_prompt",'
        '"cleaned_questions_it":["Come stai?"],"topic_codes":["health"],'
        '"confidence":"high","notes":"direct question"}\n```'
    )
    validated = validate_prediction(parsed)
    assert validated["cleaned_questions_it"] == ["Come stai?"]
    assert validated["topic_codes"] == ["health"]


def test_normalization_and_grounding_are_conservative() -> None:
    assert normalize_text("  Com'è, andata? ") == "com è andata"
    assert stable_id("aq", "same") == stable_id("aq", "same")
    assert grounding_ratio("Come è andata la settimana?", "Come è andata questa settimana") >= 0.6
    assert grounding_ratio("Dove vuoi vivere?", "Parlami della famiglia") < 0.6


def test_finalize_maps_every_turn_and_separates_verification_scopes(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    turn_map = tmp_path / "turn_map.jsonl"
    inventory = tmp_path / "inventory.jsonl"
    report = tmp_path / "report.json"
    rows = [
        {
            "context_id": "c1",
            "recording_id": "recording-a",
            "turn_id": 1,
            "context_start": 0.0,
            "context_end": 2.0,
            "participant_start": 2.0,
            "participant_end": 4.0,
            "interviewer_context_transcript": "Come è andata questa settimana",
        },
        {
            "context_id": "c2",
            "recording_id": "recording-a",
            "turn_id": 2,
            "context_start": 4.0,
            "context_end": 4.5,
            "participant_start": 4.5,
            "participant_end": 8.0,
            "interviewer_context_transcript": "sì",
        },
    ]
    write_jsonl_atomic(source, rows)
    pred_rows = [
        {
            "context_id": "c1",
            "source_asr_sha256": hashlib.sha256(rows[0]["interviewer_context_transcript"].encode()).hexdigest(),
            "cleaned_questions_it": ["Come è andata questa settimana?"],
            "topic_codes": ["recent_activities"],
            "context_type": "question_or_prompt",
            "confidence": "high",
            "classifier_model": "fake",
        },
        {
            "context_id": "c2",
            "source_asr_sha256": hashlib.sha256(rows[1]["interviewer_context_transcript"].encode()).hexdigest(),
            "cleaned_questions_it": [],
            "topic_codes": ["none"],
            "context_type": "nonquestion",
            "confidence": "high",
            "classifier_model": "fake",
        },
    ]
    write_jsonl_atomic(predictions, pred_rows)
    finalize(
        argparse.Namespace(
            input=source,
            predictions=predictions,
            turn_map=turn_map,
            question_inventory=inventory,
            report=report,
            grounding_threshold=0.6,
        )
    )
    mapped = [json.loads(line) for line in turn_map.read_text().splitlines()]
    summary = json.loads(report.read_text())
    assert len(mapped) == 2
    assert mapped[0]["interval_mapping_verified"] is True
    assert mapped[0]["question_text_verification"] == "asr_grounded_auto_high"
    assert mapped[1]["question_text_verification"] == "high_confidence_no_question"
    assert mapped[0]["turn_question_relation"] == "direct_preceding_context"
    assert mapped[1]["turn_question_relation"] == "carried_forward_prior_question"
    assert mapped[1]["governing_question_ids"] == mapped[0]["question_ids"]
    assert mapped[1]["turn_question_relation_review_required"] is True
    assert summary["turns_with_governing_question"] == 2
    assert summary["num_unique_exact_normalized_questions"] == 1
