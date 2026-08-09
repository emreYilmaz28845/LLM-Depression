#!/usr/bin/env python3
"""Extract and conservatively deduplicate Androids interviewer questions.

The input is the JSONL produced by ``recover_androids_interviewer_context.py``.
Classification uses an OpenAI-compatible chat endpoint (normally a local vLLM
server inside an MN5 Slurm job). Finalization is deterministic and separates:

* the verified timestamp mapping from interviewer context to participant turn;
* question text grounded in the ASR transcript; and
* a derived broad topic taxonomy, which is not claimed to be the corpus's
  unavailable original question sheet.

No canonical dataset file is modified. Outputs are resumable JSONL/JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


TOPICS: dict[str, str] = {
    "recent_activities": "Recent days, week, or weekend",
    "family": "Family members and family life",
    "work_study": "Work, school, study, or education",
    "daily_routine": "Daily routine and ordinary activities",
    "sleep": "Sleep and sleeping habits",
    "hobbies": "Hobbies and free-time interests",
    "sport": "Sport, teams, or exercise",
    "travel_places": "Travel, places visited, or preferred places",
    "food": "Food, meals, or cooking",
    "friends_relationships": "Friends and other relationships",
    "past_childhood": "Childhood, personal history, or past experiences",
    "future_plans": "Plans, wishes, or the future",
    "health": "Physical or mental health",
    "mood_feelings": "Mood, emotions, or feelings",
    "other": "A question or prompt outside the listed topics",
    "none": "No recoverable question or prompt",
}

CONTEXT_TYPES = {
    "question_or_prompt",
    "mixed_question_and_nonquestion",
    "nonquestion",
    "unclear",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    text = re.sub(r"[^\w\sàèéìòóù]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _topic_lines() -> str:
    return "\n".join(f"- {key}: {description}" for key, description in TOPICS.items())


def build_prompt(transcript: str) -> str:
    return f"""You are cleaning an Italian interviewer's ASR transcript.

The interval occurs immediately before one participant answer. It can contain a direct
question, an imperative prompt such as 'raccontami' or 'parlami', a follow-up,
acknowledgement, hesitation, silence, or several of these. Extract only questions and
prompts addressed to the participant. Keep their meaning and wording close to the ASR;
fix punctuation and only obvious ASR grammar. Do not invent an absent question.

Choose zero or more topic codes from this closed list:
{_topic_lines()}

Return one JSON object and no surrounding prose:
{{
  "context_type": "question_or_prompt|mixed_question_and_nonquestion|nonquestion|unclear",
  "cleaned_questions_it": ["one complete Italian question or prompt", "optional second"],
  "topic_codes": ["one_or_more_codes"],
  "confidence": "high|medium|low",
  "notes": "short reason without personal names or participant details"
}}

Rules:
- A request to speak or describe something counts as a prompt even without a question mark.
- Remove greetings, names, acknowledgements, and interviewer filler from extracted questions.
- If there is no recoverable question or prompt, return an empty list and topic_codes ["none"].
- Do not copy personal names into notes.

ASR transcript:
{transcript}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model output contains no JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")
    return value


def validate_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    context_type = str(payload.get("context_type", "")).strip()
    confidence = str(payload.get("confidence", "")).strip()
    questions = payload.get("cleaned_questions_it", [])
    topics = payload.get("topic_codes", [])
    if context_type not in CONTEXT_TYPES:
        raise ValueError(f"Invalid context_type: {context_type!r}")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    if not isinstance(questions, list) or not all(isinstance(x, str) for x in questions):
        raise ValueError("cleaned_questions_it must be a list of strings")
    if not isinstance(topics, list) or not all(str(x) in TOPICS for x in topics):
        raise ValueError("topic_codes contains an unknown topic")
    questions = [re.sub(r"\s+", " ", x).strip() for x in questions if x.strip()]
    topics = list(dict.fromkeys(str(x) for x in topics))
    if questions and (not topics or topics == ["none"]):
        raise ValueError("Questions require at least one non-none topic")
    if not questions:
        topics = ["none"]
    return {
        "context_type": context_type,
        "cleaned_questions_it": questions,
        "topic_codes": topics,
        "confidence": confidence,
        "notes": re.sub(r"\s+", " ", str(payload.get("notes", ""))).strip(),
    }


def _post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def classify_one(
    row: dict[str, Any], *, base_url: str, model: str, seed: int, timeout_sec: float, retries: int
) -> dict[str, Any]:
    transcript = str(row.get("interviewer_context_transcript", "")).strip()
    if not transcript:
        prediction = validate_prediction(
            {
                "context_type": "nonquestion",
                "cleaned_questions_it": [],
                "topic_codes": ["none"],
                "confidence": "high",
                "notes": "Empty ASR context.",
            }
        )
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(transcript)}],
            "temperature": 0.0,
            "max_tokens": 384,
            "seed": int(seed),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = _post_json(base_url, body, timeout_sec)
                content = response["choices"][0]["message"]["content"]
                prediction = validate_prediction(_extract_json_object(content))
                break
            except (KeyError, ValueError, json.JSONDecodeError, TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(f"Classification failed for {row.get('context_id')}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        else:  # pragma: no cover
            raise RuntimeError(str(last_error))
    return {
        "context_id": row["context_id"],
        "recording_id": row["recording_id"],
        "turn_id": row["turn_id"],
        "context_start": row["context_start"],
        "context_end": row["context_end"],
        "participant_start": row["participant_start"],
        "participant_end": row["participant_end"],
        "source_asr_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "classifier_model": model,
        "classifier_seed": int(seed),
        **prediction,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    tmp_path.replace(path)


def classify_file(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.input)
    existing = read_jsonl(args.output) if args.resume else []
    by_id = {str(row["context_id"]): row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError("Existing output has duplicate context IDs")
    pending = [row for row in source_rows if str(row["context_id"]) not in by_id]
    if args.limit is not None and len(pending) > args.limit:
        # A smoke should cover the corpus, not only the first interview. Choose
        # deterministic evenly spaced rows while retaining their source order.
        if args.limit <= 0:
            pending = []
        elif args.limit == 1:
            pending = [pending[0]]
        else:
            indices = [round(i * (len(pending) - 1) / (args.limit - 1)) for i in range(args.limit)]
            pending = [pending[index] for index in indices]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                classify_one,
                row,
                base_url=args.base_url,
                model=args.model,
                seed=args.seed,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
            ): row["context_id"]
            for row in pending
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            by_id[str(result["context_id"])] = result
            completed += 1
            if completed % 16 == 0 or completed == len(pending):
                ordered = [by_id[str(row["context_id"])] for row in source_rows if str(row["context_id"]) in by_id]
                write_jsonl_atomic(args.output, ordered)
                print(f"Classified {len(by_id)}/{len(source_rows)} contexts", flush=True)
    if not pending:
        print(f"No pending contexts; output already has {len(by_id)} rows", flush=True)


def grounding_ratio(question: str, source: str) -> float:
    q_tokens = set(normalize_text(question).split())
    source_tokens = set(normalize_text(source).split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & source_tokens) / len(q_tokens)


def finalize(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.input)
    predictions = read_jsonl(args.predictions)
    source_by_id = {str(row["context_id"]): row for row in source_rows}
    pred_by_id = {str(row["context_id"]): row for row in predictions}
    if len(source_by_id) != len(source_rows) or len(pred_by_id) != len(predictions):
        raise ValueError("Duplicate context IDs in input or predictions")
    if set(source_by_id) != set(pred_by_id):
        missing = sorted(set(source_by_id) - set(pred_by_id))
        extra = sorted(set(pred_by_id) - set(source_by_id))
        raise ValueError(f"Coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    inventory: dict[str, dict[str, Any]] = {}
    turn_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    for source in source_rows:
        context_id = str(source["context_id"])
        pred = pred_by_id[context_id]
        transcript = str(source.get("interviewer_context_transcript", "")).strip()
        if pred.get("source_asr_sha256") != hashlib.sha256(transcript.encode("utf-8")).hexdigest():
            raise ValueError(f"Source ASR changed for context {context_id}")
        question_ids: list[str] = []
        ratios: list[float] = []
        for question in pred["cleaned_questions_it"]:
            normalized = normalize_text(question)
            if not normalized:
                continue
            qid = stable_id("aq", normalized)
            question_ids.append(qid)
            ratios.append(grounding_ratio(question, transcript))
            item = inventory.setdefault(
                qid,
                {
                    "question_id": qid,
                    "normalized_question_it": normalized,
                    "representative_question_it": question,
                    "occurrence_count": 0,
                    "topic_codes": [],
                    "context_ids": [],
                },
            )
            item["occurrence_count"] += 1
            item["context_ids"].append(context_id)
            item["topic_codes"] = sorted(set(item["topic_codes"]) | set(pred["topic_codes"]))

        high_grounded = bool(question_ids) and pred["confidence"] == "high" and min(ratios) >= args.grounding_threshold
        if high_grounded:
            verification = "asr_grounded_auto_high"
            review_required = False
        elif not question_ids and pred["confidence"] == "high" and pred["context_type"] == "nonquestion":
            verification = "high_confidence_no_question"
            review_required = False
        else:
            verification = "review_required"
            review_required = True
        status_counts[verification] += 1
        topic_counts.update(pred["topic_codes"])
        turn_rows.append(
            {
                "context_id": context_id,
                "recording_id": source["recording_id"],
                "turn_id": source["turn_id"],
                "context_start": source["context_start"],
                "context_end": source["context_end"],
                "participant_start": source["participant_start"],
                "participant_end": source["participant_end"],
                "interviewer_context_transcript": transcript,
                "cleaned_questions_it": pred["cleaned_questions_it"],
                "question_ids": question_ids,
                "topic_codes": pred["topic_codes"],
                "context_type": pred["context_type"],
                "classifier_confidence": pred["confidence"],
                "grounding_ratios": ratios,
                "interval_mapping_verified": True,
                "question_text_verification": verification,
                "manual_review_required": review_required,
                "classifier_model": pred["classifier_model"],
            }
        )

    inventory_rows = sorted(inventory.values(), key=lambda row: (-row["occurrence_count"], row["question_id"]))
    write_jsonl_atomic(args.turn_map, turn_rows)
    write_jsonl_atomic(args.question_inventory, inventory_rows)
    report = {
        "dataset": "androids",
        "task": "interviewer_question_classification_and_conservative_deduplication",
        "num_turns": len(turn_rows),
        "num_unique_exact_normalized_questions": len(inventory_rows),
        "question_occurrences": sum(len(row["question_ids"]) for row in turn_rows),
        "verification_status_counts": dict(sorted(status_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "grounding_threshold": args.grounding_threshold,
        "deduplication": "Exact match after Unicode, case, whitespace, and punctuation normalization. Broad topic codes are derived semantic groups, not original corpus prompt IDs.",
        "verification_scope": "interval_mapping_verified is timestamp-derived. asr_grounded_auto_high verifies extraction against ASR text, not against a human transcript or original question sheet.",
        "source_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "turn_map_sha256": hashlib.sha256(args.turn_map.read_bytes()).hexdigest(),
        "question_inventory_sha256": hashlib.sha256(args.question_inventory.read_bytes()).hexdigest(),
        "topics": TOPICS,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    classify = sub.add_parser("classify")
    classify.add_argument("--input", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    classify.add_argument("--model", default="qwen3.6-27b")
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument("--workers", type=int, default=16)
    classify.add_argument("--timeout-sec", type=float, default=180.0)
    classify.add_argument("--retries", type=int, default=2)
    classify.add_argument("--limit", type=int)
    classify.add_argument("--resume", action="store_true")

    finish = sub.add_parser("finalize")
    finish.add_argument("--input", type=Path, required=True)
    finish.add_argument("--predictions", type=Path, required=True)
    finish.add_argument("--turn-map", type=Path, required=True)
    finish.add_argument("--question-inventory", type=Path, required=True)
    finish.add_argument("--report", type=Path, required=True)
    finish.add_argument("--grounding-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "classify":
        classify_file(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
