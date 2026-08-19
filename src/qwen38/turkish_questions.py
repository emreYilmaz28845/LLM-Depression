"""Private Turkish question-recovery pipeline (runbook sections 18-20).

Stages:

- ``prepare``: parse the ASR transcript JSONL, group the 1,186 windows into
  135 subject sequences, assign opaque sequence IDs S0001-S0135 ordered by
  SHA-256 of the source subject string, and write owner-only prepared
  packets. Never reads the metadata CSV, never sends the condition tag, and
  marks windows as ``[WINDOW n]`` inside the prompt.
- ``infer-subjects``: one deterministic request per complete sequence,
  resumable by sequence ID, refusing resume when any provenance hash differs.
- ``consolidate``: five fixed sequence batches (32, 32, 32, 32, 7), each
  merged by the model into clusters, then one final merge request; Python
  enforces that every candidate lands in exactly one cluster and every
  cluster in exactly one family.
- ``render``: deterministic CSV/JSON/Markdown tables from the final families
  with the exact runbook column order.

All intermediates are written atomically and stay owner-only; raw transcripts
and subject identifiers never appear in the compact outputs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.qwen38.contracts import (
    CONFIDENCE_WEIGHTS,
    FINAL_TABLE_COLUMNS,
    MODEL_ID,
    MODEL_REVISION,
    SERVED_MODEL,
    TURKISH_CONSOLIDATION_BATCHES,
    TURKISH_EXPECTED_SEQUENCES,
    TURKISH_EXPECTED_WINDOWS,
    TURKISH_INFERENCE_CONCURRENCY,
    TURKISH_MAX_TOKENS,
    TURKISH_SOURCE_HASH,
    Confidence,
    Label,
    WordingStatus,
    generation_settings_hash,
    parse_filename_stem,
    request_settings,
    structured_output_schema,
    validate_consolidation_batch,
    validate_consolidation_final,
    validate_subject_inference,
)

SUBJECT_SYSTEM_PROMPT = (
    "You infer recurring interviewer-question families from answer-only interview "
    "transcripts. You see only the participant's answers, each marked [WINDOW n] in "
    "chronological order; you never see the interviewer's questions and the window "
    "numbers are sequential 20-second window indices, not question IDs.\n"
    "For every distinct interviewer prompt family you can support from these answers, "
    "return one episode in chronological order. Classify the inferred interviewer "
    "prompt, not the emotional tone of the answer:\n"
    "- POSITIVE asks about happiness, enjoyment, strengths, hope, support, positive "
    "memories, or pleasant events.\n"
    "- NEGATIVE asks about sadness, distress, problems, symptoms, loss, fear, conflict, "
    "or unpleasant events.\n"
    "- NEUTRAL is factual, descriptive, demographic, procedural, or open framing that "
    "does not ask for positive or negative valence.\n"
    "- MIXED asks for both positive and negative material, or combines opposing "
    "valences in one question.\n"
    "Set wording_status to EXPLICIT_ECHO only when the answer clearly repeats enough "
    "of the missing question to support close wording; otherwise use "
    "INFERRED_PARAPHRASE and never present your paraphrase as the verbatim question.\n"
    "Set confidence to HIGH, MEDIUM, or LOW. List the exact evidence window indices. "
    "evidence_basis must be a short, non-quoted description and must not copy any "
    "transcript text. When an answer does not support any question family, return the "
    "episode with empty question_tr and question_en and a short abstain_reason.\n"
    "Return exactly one JSON object, no prose, no markdown fences, with this exact "
    "shape:\n"
    '{\n'
    '  "episodes": [\n'
    '    {\n'
    '      "sequence_id": "S0001 (the exact value shown at the top of the input as \'Sequence id: ...\')",\n'
    '      "episode_order": 1,\n'
    '      "question_tr": "concise Turkish question",\n'
    '      "question_en": "concise English question",\n'
    '      "label": "POSITIVE | NEGATIVE | NEUTRAL | MIXED",\n'
    '      "wording_status": "EXPLICIT_ECHO | INFERRED_PARAPHRASE",\n'
    '      "confidence": "HIGH | MEDIUM | LOW",\n'
    '      "evidence_window_indices": [1, 2],\n'
    '      "evidence_basis": "short non-quoted description",\n'
    '      "abstain_reason": ""\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "Use exactly the field names shown. The sequence_id field must equal the "
    "sequence id provided to you; never invent a different id."
)

CONSOLIDATION_BATCH_SYSTEM_PROMPT = (
    "You merge inferred interviewer-question episodes into clusters of paraphrases. "
    "Each candidate carries an opaque sequence ID, its episode order, a Turkish and an "
    "English inferred question, a label (POSITIVE, NEGATIVE, NEUTRAL, MIXED), a wording "
    "status (EXPLICIT_ECHO, INFERRED_PARAPHRASE), and a confidence (HIGH, MEDIUM, LOW).\n"
    "Merge paraphrases that plausibly represent the same interviewer prompt while "
    "keeping distinct topics separate. Every candidate ID must appear in exactly one "
    "cluster. Propose concise canonical Turkish and English wording for each cluster.\n"
    "Return exactly one JSON object, no prose, no markdown fences, with this exact "
    "shape:\n"
    '{\n'
    '  "clusters": [\n'
    '    {\n'
    '      "cluster_id": "c1",\n'
    '      "canonical_question_tr": "concise Turkish wording",\n'
    '      "canonical_question_en": "concise English wording",\n'
    '      "member_candidate_ids": ["S0001-1", "S0002-3"]\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "Use exactly the field names shown. Every candidate id must appear in exactly "
    "one member_candidate_ids list."
)

CONSOLIDATION_FINAL_SYSTEM_PROMPT = (
    "You merge batch cluster summaries into final recurring-question families. Each "
    "cluster summary carries a cluster ID, canonical Turkish and English wording, the "
    "number of supporting candidate episodes, and the label mix across its members.\n"
    "Merge clusters that represent the same interviewer question family across "
    "batches; keep distinct topics separate. Every cluster ID must appear in exactly "
    "one family. For each final family, propose concise, semantically aligned "
    "canonical wording in Turkish and English. Do not include any subject ID, "
    "filename, window marker, quotation, diagnostic label, or outcome in the wording.\n"
    "Return exactly one JSON object, no prose, no markdown fences, with this exact "
    "shape:\n"
    '{\n'
    '  "families": [\n'
    '    {\n'
    '      "family_id": "f1",\n'
    '      "question_tr": "concise Turkish wording",\n'
    '      "question_en": "concise English wording",\n'
    '      "member_cluster_ids": ["c1", "c2"]\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "Use exactly the field names shown. Every cluster id must appear in exactly "
    "one member_cluster_ids list."
)

SCHEMA_CORRECTION_MESSAGE = (
    "Your previous answer was not valid JSON matching the required schema. Return only "
    "the single JSON object, no prose."
)

PROMPT_VERSION = "qwen38_turkish_v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _write_jsonl_atomic(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _restrict(path: str | Path, mode: int = 0o600) -> None:
    path = Path(path)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def load_transcript_rows(transcript_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(transcript_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{transcript_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{transcript_path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def prepare_sequences(
    transcript_path: str | Path,
    *,
    run_dir: str | Path,
    deployment_id: str,
    model_revision: str,
    source_commit: str,
    source_sha256: str = TURKISH_SOURCE_HASH,
    expected_sequences: int = TURKISH_EXPECTED_SEQUENCES,
    expected_windows: int = TURKISH_EXPECTED_WINDOWS,
) -> dict[str, Any]:
    """Parse, group, and verify the Turkish windows; write restricted packets."""
    transcript_path = Path(transcript_path)
    if not transcript_path.is_file():
        raise ValueError(f"transcript file not found: {transcript_path}")
    actual_hash = _sha256_file(transcript_path)
    if actual_hash != source_sha256:
        raise ValueError(
            f"transcript hash mismatch: expected {source_sha256}, found {actual_hash}"
        )

    windows_by_subject: dict[str, list[tuple[int, str]]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    total_rows = 0
    for row in load_transcript_rows(transcript_path):
        total_rows += 1
        audio_path = row.get("audio_path")
        transcript = row.get("transcript")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("transcript row without audio_path")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"transcript row {audio_path} has an empty transcript")
        stem = Path(audio_path).stem
        parts = parse_filename_stem(stem)
        if parts is None:
            raise ValueError(f"unparseable filename stem: {stem}")
        pair = (parts.subject, parts.window)
        if pair in seen_pairs:
            raise ValueError(f"duplicate (subject, window) pair: {pair}")
        seen_pairs.add(pair)
        windows_by_subject.setdefault(parts.subject, []).append((parts.window, transcript.strip()))

    subjects = sorted(windows_by_subject)
    if len(subjects) != expected_sequences:
        raise ValueError(
            f"expected {expected_sequences} subjects, found {len(subjects)}"
        )
    total_windows = sum(len(windows) for windows in windows_by_subject.values())
    if total_windows != expected_windows:
        raise ValueError(
            f"expected {expected_windows} windows, found {total_windows}"
        )

    subject_hash = {subject: hashlib.sha256(subject.encode("utf-8")).hexdigest() for subject in subjects}
    ordered_subjects = sorted(subjects, key=lambda subject: subject_hash[subject])
    sequence_ids = {subject: f"S{index:04d}" for index, subject in enumerate(ordered_subjects, start=1)}

    subject_map = {sequence_ids[subject]: subject for subject in subjects}
    generation_hash = generation_settings_hash(TURKISH_MAX_TOKENS)

    sequences: list[dict[str, Any]] = []
    for subject in ordered_subjects:
        sequence_id = sequence_ids[subject]
        windows = sorted(windows_by_subject[subject], key=lambda item: item[0])
        blocks = [f"[WINDOW {window}]\n{text}" for window, text in windows]
        user_prompt = f"Sequence id: {sequence_id}\n\n" + "\n\n".join(blocks)
        sequences.append(
            {
                "sequence_id": sequence_id,
                "subject_sha256": subject_hash[subject],
                "window_count": len(windows),
                "windows": [{"window": window, "text": text} for window, text in windows],
                "user_prompt": user_prompt,
                "prompt_hash": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                "source_sha256": actual_hash,
                "source_commit": source_commit,
                "model_id": MODEL_ID,
                "model_revision": model_revision,
                "generation_settings_hash": generation_hash,
                "prompt_version": PROMPT_VERSION,
                "deployment_id": deployment_id,
            }
        )

    run_dir = Path(run_dir)
    restricted = run_dir / "restricted"
    restricted.mkdir(parents=True, exist_ok=True)
    map_path = restricted / "subject_map.json"
    packets_path = restricted / "prepared_sequences.jsonl"
    _atomic_write_json(subject_map, map_path)
    _write_jsonl_atomic(sequences, packets_path)
    _restrict(map_path)
    _restrict(packets_path)
    for subject_dir in (restricted / "subject_inferences", restricted / "consolidation_batches"):
        subject_dir.mkdir(parents=True, exist_ok=True)

    return {
        "deployment_id": deployment_id,
        "source_sha256": actual_hash,
        "windows": total_windows,
        "sequences": len(sequences),
        "subject_map_path": str(map_path),
        "prepared_sequences_path": str(packets_path),
        "prompt_version": PROMPT_VERSION,
        "source_commit": source_commit,
    }


def load_prepared_sequences(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    ids = [row.get("sequence_id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate sequence IDs")
    return rows


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```"):
        cleaned = stripped.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _validate_episodes(payload: dict[str, Any], sequence_id: str) -> list[dict[str, Any]]:
    errors = validate_subject_inference(payload)
    if errors:
        raise ValueError(f"{sequence_id}: schema errors: {'; '.join(errors[:5])}")
    episodes = payload["episodes"]
    orders: set[int] = set()
    for index, episode in enumerate(episodes):
        if episode["sequence_id"] != sequence_id:
            raise ValueError(
                f"{sequence_id}: episode {index} carries sequence {episode['sequence_id']!r}"
            )
        order = episode["episode_order"]
        if order in orders:
            raise ValueError(f"{sequence_id}: duplicate episode_order {order}")
        orders.add(order)
        if episode["label"] not in (label.value for label in Label):
            raise ValueError(f"{sequence_id}: invalid label {episode['label']!r}")
        if episode["wording_status"] not in (status.value for status in WordingStatus):
            raise ValueError(f"{sequence_id}: invalid wording_status {episode['wording_status']!r}")
        if episode["confidence"] not in (conf.value for conf in Confidence):
            raise ValueError(f"{sequence_id}: invalid confidence {episode['confidence']!r}")
        if episode["evidence_basis"] and ('"' in episode["evidence_basis"] or "'" in episode["evidence_basis"]):
            raise ValueError(f"{sequence_id}: evidence_basis must be non-quoted")
        if len(episode["evidence_basis"]) > 200:
            raise ValueError(f"{sequence_id}: evidence_basis too long")
        if not episode["abstain_reason"] and not episode["question_tr"].strip():
            raise ValueError(f"{sequence_id}: candidate episode without question_tr")
    return episodes


def _episode_provenance(sequence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha256": sequence["source_sha256"],
        "source_commit": sequence["source_commit"],
        "model_revision": sequence["model_revision"],
        "generation_settings_hash": sequence["generation_settings_hash"],
        "prompt_hash": sequence["prompt_hash"],
    }


def _completed_inference(inference_path: Path) -> dict[str, Any] | None:
    if not inference_path.is_file():
        return None
    with inference_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) and data.get("status") == "completed" else None


def infer_subjects(
    prepared_path: str | Path,
    inferences_dir: str | Path,
    *,
    base_url: str,
    model: str = SERVED_MODEL,
    concurrency: int = TURKISH_INFERENCE_CONCURRENCY,
    seed: int = 42,
    max_tokens: int = TURKISH_MAX_TOKENS,
) -> dict[str, Any]:
    """Run one deterministic request per sequence; resumable by sequence ID."""
    sequences = load_prepared_sequences(prepared_path)
    inferences_dir = Path(inferences_dir)
    inferences_dir.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    completed = 0
    for sequence in sequences:
        sequence_id = sequence["sequence_id"]
        inference_path = inferences_dir / f"{sequence_id}.json"
        existing = _completed_inference(inference_path)
        if existing is not None:
            expected = _episode_provenance(sequence)
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ValueError(
                        f"resume refused for {sequence_id}: {key} changed "
                        f"(stored {existing.get(key)!r} != expected {value!r})"
                    )
            completed += 1
            continue
        if inference_path.exists():
            raise ValueError(f"resume refused for {sequence_id}: incomplete or invalid record")
        pending.append(sequence)

    settings = request_settings(max_tokens)

    async def infer_one(sequence: dict[str, Any]) -> tuple[str, bool, str | None]:
        sequence_id = sequence["sequence_id"]
        messages = [
            {"role": "system", "content": SUBJECT_SYSTEM_PROMPT},
            {"role": "user", "content": sequence["user_prompt"]},
        ]
        for attempt in (1, 2):
            try:
                if attempt > 1:
                    messages = messages + [
                        {"role": "user", "content": SCHEMA_CORRECTION_MESSAGE}
                    ]
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    max_tokens=settings["max_tokens"],
                    seed=seed,
                    extra_body={"chat_template_kwargs": settings["chat_template_kwargs"]},
                    stream=True,
                )
                chunks: list[str] = []
                async for chunk in stream:
                    if getattr(chunk, "choices", None):
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            chunks.append(content)
                content = "".join(chunks)
                if not content.strip():
                    if attempt == 1:
                        continue
                    return sequence_id, False, "empty response"
                parsed = _parse_json_object(content)
                if parsed is None:
                    if attempt == 1:
                        continue
                    return sequence_id, False, "invalid JSON twice"
                episodes = _validate_episodes(parsed, sequence_id)
                record = dict(_episode_provenance(sequence))
                record.update(
                    {
                        "sequence_id": sequence_id,
                        "status": "completed",
                        "episodes": episodes,
                        "episode_count": len(episodes),
                        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
                inference_path = inferences_dir / f"{sequence_id}.json"
                _atomic_write_json(record, inference_path)
                _restrict(inference_path)
                return sequence_id, True, None
            except Exception as exc:
                if attempt == 1:
                    continue
                return sequence_id, False, f"{type(exc).__name__}: {exc}"
        return sequence_id, False, "unreachable"

    client: Any = None
    import asyncio

    async def orchestrate() -> list[tuple[str, bool, str | None]]:
        nonlocal client
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)
        semaphore = asyncio.Semaphore(concurrency)

        async def guarded(sequence: dict[str, Any]) -> tuple[str, bool, str | None]:
            async with semaphore:
                return await infer_one(sequence)

        results = await asyncio.gather(*(guarded(sequence) for sequence in pending))
        await client.close()
        return list(results)

    failures: list[str] = []
    if pending:
        results = asyncio.run(orchestrate())
        for sequence_id, ok, error in results:
            if not ok:
                failures.append(f"{sequence_id}: {error}")

    total_completed = completed + (len(pending) - len(failures))
    return {
        "sequences_total": len(sequences),
        "completed_before_resume": completed,
        "completed_now": len(pending) - len(failures),
        "completed_total": total_completed,
        "failed": failures,
        "complete": total_completed == len(sequences) and not failures,
    }


# --------------------------------------------------------------------------
# consolidation
# --------------------------------------------------------------------------


def _load_inferences(inferences_dir: str | Path) -> list[dict[str, Any]]:
    inferences_dir = Path(inferences_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(inferences_dir.glob("S*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "completed":
            raise ValueError(f"{path.name}: not completed")
        records.append(record)
    return records


def _candidate_id(sequence_id: str, episode_order: int) -> str:
    return f"{sequence_id}-e{episode_order}"


def collect_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        sequence_id = record["sequence_id"]
        for episode in record["episodes"]:
            if episode["abstain_reason"]:
                continue
            candidates.append(
                {
                    "candidate_id": _candidate_id(sequence_id, episode["episode_order"]),
                    "sequence_id": sequence_id,
                    "episode_order": episode["episode_order"],
                    "question_tr": episode["question_tr"],
                    "question_en": episode["question_en"],
                    "label": episode["label"],
                    "wording_status": episode["wording_status"],
                    "confidence": episode["confidence"],
                    "evidence_window_indices": episode["evidence_window_indices"],
                    "evidence_basis": episode["evidence_basis"],
                }
            )
    return candidates


def _check_cluster_assignment(
    clusters: list[dict[str, Any]],
    candidate_ids: list[str],
    batch_label: str,
) -> dict[str, Any]:
    seen: set[str] = set()
    assignment: dict[str, str] = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        if not cluster_id or not re.fullmatch(r"[A-Za-z0-9_-]+", cluster_id):
            raise ValueError(f"{batch_label}: invalid cluster_id {cluster_id!r}")
        for member in cluster["member_candidate_ids"]:
            if member not in candidate_ids:
                raise ValueError(f"{batch_label}: unknown candidate {member!r} in {cluster_id}")
            if member in seen:
                raise ValueError(f"{batch_label}: candidate {member!r} assigned twice")
            seen.add(member)
            assignment[member] = cluster_id
        if not cluster["canonical_question_tr"].strip() or not cluster["canonical_question_en"].strip():
            raise ValueError(f"{batch_label}: cluster {cluster_id} has empty canonical wording")
    missing = [candidate for candidate in candidate_ids if candidate not in seen]
    if missing:
        raise ValueError(f"{batch_label}: {len(missing)} candidates unassigned: {missing[:5]} ...")
    if not clusters:
        raise ValueError(f"{batch_label}: empty cluster list")
    return assignment


def _check_family_assignment(
    families: list[dict[str, Any]],
    cluster_ids: list[str],
) -> dict[str, Any]:
    seen: set[str] = set()
    assignment: dict[str, str] = {}
    for family in families:
        family_id = family["family_id"]
        if not family_id or not re.fullmatch(r"[A-Za-z0-9_-]+", family_id):
            raise ValueError(f"invalid family_id {family_id!r}")
        for member in family["member_cluster_ids"]:
            if member not in cluster_ids:
                raise ValueError(f"unknown cluster {member!r} in family {family_id}")
            if member in seen:
                raise ValueError(f"cluster {member!r} assigned twice")
            seen.add(member)
            assignment[member] = family_id
        if not family["question_tr"].strip() or not family["question_en"].strip():
            raise ValueError(f"family {family_id} has empty wording")
    missing = [cluster for cluster in cluster_ids if cluster not in seen]
    if missing:
        raise ValueError(f"{len(missing)} clusters unassigned: {missing[:5]} ...")
    if not families:
        raise ValueError("empty family list")
    return assignment


async def _consolidation_request(
    client: Any,
    model: str,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    schema: dict[str, Any],
    max_tokens: int,
    seed: int,
    settings: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    for attempt in (1, 2):
        try:
            if attempt > 1:
                messages = messages + [{"role": "user", "content": SCHEMA_CORRECTION_MESSAGE}]
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                max_tokens=settings["max_tokens"],
                seed=seed,
                extra_body={"chat_template_kwargs": settings["chat_template_kwargs"]},
                stream=True,
            )
            chunks: list[str] = []
            async for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        chunks.append(content)
            content = "".join(chunks)
            parsed = _parse_json_object(content)
            if parsed is None:
                if attempt == 1:
                    continue
                raise ValueError(f"{label}: invalid JSON twice")
            return parsed
        except Exception as exc:
            if attempt == 1 and not isinstance(exc, ValueError):
                continue
            raise ValueError(f"{label}: request failed: {exc}") from exc
    raise ValueError(f"{label}: unreachable")


def consolidate(
    inferences_dir: str | Path,
    consolidation_dir: str | Path,
    *,
    base_url: str,
    model: str = SERVED_MODEL,
    seed: int = 42,
    max_tokens: int = TURKISH_MAX_TOKENS,
) -> dict[str, Any]:
    """Two-level consolidation: five sequence batches, then one final merge."""
    import asyncio

    records = _load_inferences(inferences_dir)
    sequence_ids = sorted(record["sequence_id"] for record in records)
    if len(sequence_ids) != TURKISH_EXPECTED_SEQUENCES:
        raise ValueError(
            f"expected {TURKISH_EXPECTED_SEQUENCES} completed sequences, found {len(sequence_ids)}"
        )
    candidates = collect_candidates(records)
    candidates.sort(key=lambda c: (c["sequence_id"], c["episode_order"]))
    candidate_by_id = {c["candidate_id"]: c for c in candidates}
    candidate_ids_all = [c["candidate_id"] for c in candidates]

    consolidation_dir = Path(consolidation_dir)
    consolidation_dir.mkdir(parents=True, exist_ok=True)

    settings = request_settings(max_tokens)
    batch_boundaries: list[tuple[int, int]] = []
    cursor = 0
    for size in TURKISH_CONSOLIDATION_BATCHES:
        batch_boundaries.append((cursor, cursor + size))
        cursor += size
    if cursor != len(sequence_ids):
        raise ValueError(f"consolidation batches do not cover {len(sequence_ids)} sequences")

    BATCH_SCHEMA = {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "canonical_question_tr": {"type": "string"},
                        "canonical_question_en": {"type": "string"},
                        "member_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "cluster_id",
                        "canonical_question_tr",
                        "canonical_question_en",
                        "member_candidate_ids",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clusters"],
        "additionalProperties": False,
    }
    FINAL_SCHEMA = {
        "type": "object",
        "properties": {
            "families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "family_id": {"type": "string"},
                        "question_tr": {"type": "string"},
                        "question_en": {"type": "string"},
                        "member_cluster_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "family_id",
                        "question_tr",
                        "question_en",
                        "member_cluster_ids",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["families"],
        "additionalProperties": False,
    }

    async def orchestrate() -> dict[str, Any]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)
        try:
            batch_results: list[dict[str, Any]] = []
            all_cluster_ids: list[str] = []
            for batch_index, (start, end) in enumerate(batch_boundaries, start=1):
                batch_sequences = sequence_ids[start:end]
                batch_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["sequence_id"] in batch_sequences
                ]
                user_payload = {
                    "batch_index": batch_index,
                    "candidates": [
                        {
                            "candidate_id": c["candidate_id"],
                            "sequence_id": c["sequence_id"],
                            "episode_order": c["episode_order"],
                            "question_tr": c["question_tr"],
                            "question_en": c["question_en"],
                            "label": c["label"],
                            "wording_status": c["wording_status"],
                            "confidence": c["confidence"],
                        }
                        for c in batch_candidates
                    ],
                }
                parsed = await _consolidation_request(
                    client,
                    model,
                    system_prompt=CONSOLIDATION_BATCH_SYSTEM_PROMPT,
                    user_payload=user_payload,
                    schema=BATCH_SCHEMA,
                    max_tokens=max_tokens,
                    seed=seed,
                    settings=settings,
                    label=f"batch {batch_index}",
                )
                errors = validate_consolidation_batch(parsed)
                if errors:
                    raise ValueError(f"batch {batch_index}: {'; '.join(errors[:5])}")
                batch_candidate_ids = [c["candidate_id"] for c in batch_candidates]
                assignment = _check_cluster_assignment(
                    parsed["clusters"], batch_candidate_ids, f"batch {batch_index}"
                )
                record = {
                    "batch_index": batch_index,
                    "sequence_ids": batch_sequences,
                    "candidate_count": len(batch_candidate_ids),
                    "clusters": parsed["clusters"],
                    "assignment": assignment,
                }
                batch_results.append(record)
                all_cluster_ids.extend(cluster["cluster_id"] for cluster in parsed["clusters"])
                _atomic_write_json(record, consolidation_dir / f"batch_{batch_index:02d}.json")
                _restrict(consolidation_dir / f"batch_{batch_index:02d}.json")

            cluster_summaries: list[dict[str, Any]] = []
            for record in batch_results:
                for cluster in record["clusters"]:
                    label_counts: Counter[str] = Counter()
                    for member in cluster["member_candidate_ids"]:
                        label_counts[candidate_by_id[member]["label"]] += 1
                    cluster_summaries.append(
                        {
                            "cluster_id": cluster["cluster_id"],
                            "batch_index": record["batch_index"],
                            "canonical_question_tr": cluster["canonical_question_tr"],
                            "canonical_question_en": cluster["canonical_question_en"],
                            "member_count": len(cluster["member_candidate_ids"]),
                            "label_counts": dict(label_counts),
                        }
                    )

            final_parsed = await _consolidation_request(
                client,
                model,
                system_prompt=CONSOLIDATION_FINAL_SYSTEM_PROMPT,
                user_payload={"cluster_summaries": cluster_summaries},
                schema=FINAL_SCHEMA,
                max_tokens=max_tokens,
                seed=seed,
                settings=settings,
                label="final merge",
            )
            errors = validate_consolidation_final(final_parsed)
            if errors:
                raise ValueError(f"final merge: {'; '.join(errors[:5])}")
            family_assignment = _check_family_assignment(
                final_parsed["families"], all_cluster_ids
            )
            final_record = {
                "families": final_parsed["families"],
                "cluster_assignment": family_assignment,
                "cluster_count": len(all_cluster_ids),
                "candidate_count": len(candidate_ids_all),
            }
            _atomic_write_json(final_record, consolidation_dir / "final_merge.json")
            _restrict(consolidation_dir / "final_merge.json")
        finally:
            await client.close()

        cluster_to_candidate: dict[str, list[str]] = {}
        for record in batch_results:
            for cluster in record["clusters"]:
                cluster_to_candidate[cluster["cluster_id"]] = cluster["member_candidate_ids"]
        return {
            "sequences": len(sequence_ids),
            "candidates": len(candidate_ids_all),
            "batches": len(batch_results),
            "clusters": len(all_cluster_ids),
            "families": len(final_parsed["families"]),
            "cluster_assignment": family_assignment,
            "cluster_to_candidate": cluster_to_candidate,
            "final_record_path": str(consolidation_dir / "final_merge.json"),
        }

    return asyncio.run(orchestrate())


# --------------------------------------------------------------------------
# final aggregation and rendering
# --------------------------------------------------------------------------


def aggregate_families(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    families: list[dict[str, Any]],
    cluster_to_candidate: dict[str, list[str]],
    cluster_assignment: dict[str, str],
) -> list[dict[str, Any]]:
    """Deterministic per-family aggregation (runbook section 19.4)."""
    candidate_by_id = {c["candidate_id"]: c for c in candidates}
    episode_count_by_sequence = Counter()
    for record in records:
        episode_count_by_sequence[record["sequence_id"]] = record.get("episode_count", 0)

    def episode_count(sequence_id: str) -> int:
        return max(episode_count_by_sequence.get(sequence_id, 0), 1)

    rows: list[dict[str, Any]] = []
    for family in families:
        member_candidates: list[dict[str, Any]] = []
        for cluster_id in family["member_cluster_ids"]:
            for candidate_id in cluster_to_candidate.get(cluster_id, []):
                candidate = candidate_by_id.get(candidate_id)
                if candidate is None:
                    raise ValueError(f"family {family['family_id']}: unknown candidate {candidate_id}")
                member_candidates.append(candidate)
        if not member_candidates:
            raise ValueError(f"family {family['family_id']} has no supporting candidates")

        total_weight = sum(CONFIDENCE_WEIGHTS[Confidence(c["confidence"])] for c in member_candidates)
        shares: dict[str, float] = {}
        for label in Label:
            shares[label.value] = (
                sum(
                    CONFIDENCE_WEIGHTS[Confidence(c["confidence"])]
                    for c in member_candidates
                    if c["label"] == label.value
                )
                / total_weight
            )
        agreement = max(shares.values())

        non_mixed = {label.value: shares[label.value] for label in (Label.NEUTRAL, Label.NEGATIVE, Label.POSITIVE)}
        winner = max(non_mixed, key=lambda label: (non_mixed[label], -["NEUTRAL", "NEGATIVE", "POSITIVE"].index(label)))
        positive = shares[Label.POSITIVE.value]
        negative = shares[Label.NEGATIVE.value]
        neutral = shares[Label.NEUTRAL.value]
        mixed = shares[Label.MIXED.value]
        if winner == Label.POSITIVE.value and positive >= 0.75 and negative + mixed < 0.20:
            final_label = Label.POSITIVE.value
        elif winner == Label.NEGATIVE.value and negative >= 0.75 and positive + mixed < 0.20:
            final_label = Label.NEGATIVE.value
        elif winner == Label.NEUTRAL.value and neutral >= 0.75 and positive + negative + mixed < 0.20:
            final_label = Label.NEUTRAL.value
        else:
            final_label = Label.MIXED.value

        supporting_sequences = {c["sequence_id"] for c in member_candidates}
        support = len(supporting_sequences)
        explicit_sequences = {
            c["sequence_id"] for c in member_candidates if c["wording_status"] == WordingStatus.EXPLICIT_ECHO.value
        }
        if len(explicit_sequences) >= 3 and len(explicit_sequences) > support / 2:
            final_status = WordingStatus.EXPLICIT_ECHO.value
        else:
            final_status = WordingStatus.INFERRED_PARAPHRASE.value

        if support >= 10 and agreement >= 0.80 and sum(
            1 for c in member_candidates if c["confidence"] in (Confidence.HIGH.value, Confidence.MEDIUM.value)
        ) >= 0.70 * len(member_candidates):
            final_confidence = Confidence.HIGH.value
        elif support >= 3 and agreement >= 0.60:
            final_confidence = Confidence.MEDIUM.value
        else:
            final_confidence = Confidence.LOW.value

        normalized_positions = [
            c["episode_order"] / episode_count(c["sequence_id"]) for c in member_candidates
        ]
        median_position = statistics.median(normalized_positions)

        basis_counter: Counter[str] = Counter()
        for c in member_candidates:
            if c["evidence_basis"]:
                basis_counter[c["evidence_basis"]] += 1
        top_basis = sorted(basis_counter, key=lambda text: (-basis_counter[text], text))[:3]
        evidence_basis = "; ".join(top_basis) if top_basis else "no explicit question wording available"

        rows.append(
            {
                "family_id": family["family_id"],
                "question_tr": family["question_tr"].strip(),
                "question_en": family["question_en"].strip(),
                "label": final_label,
                "wording_status": final_status,
                "confidence": final_confidence,
                "supporting_subjects": support,
                "evidence_basis": evidence_basis,
                "weighted_label_agreement": agreement,
                "median_normalized_position": median_position,
                "shares": {label: round(value, 6) for label, value in shares.items()},
            }
        )

    rows.sort(key=lambda row: (row["median_normalized_position"], row["question_tr"]))
    for order, row in enumerate(rows, start=1):
        row["order"] = order
    return rows


def render_tables(
    rows: list[dict[str, Any]],
    *,
    run_dir: str | Path,
    deployment_id: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    source_commit: str = "",
) -> dict[str, Any]:
    """Write the compact CSV, JSON, and Markdown tables atomically."""
    table_rows = []
    for row in rows:
        table_row = {column: row[column] for column in FINAL_TABLE_COLUMNS}
        table_rows.append(table_row)
        for column in ("question_tr", "question_en", "evidence_basis"):
            text = table_row[column]
            for marker in ("[WINDOW", "[", "]", '"', "'", "subject_id", "filename"):
                if marker in text:
                    raise ValueError(f"render refused: {column} contains forbidden marker {marker!r}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "turkish_inferred_questions.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FINAL_TABLE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in table_rows:
        writer.writerow(row)
    _atomic_write_text(buffer.getvalue(), csv_path)

    payload = {
        "deployment_id": deployment_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "source_commit": source_commit,
        "rows": table_rows,
    }
    json_path = run_dir / "turkish_inferred_questions.json"
    _atomic_write_json(payload, json_path)

    lines = [
        "# Turkish inferred recurring question families",
        "",
        f"Deployment: `{deployment_id}` — model `{model_id}` revision `{model_revision}`.",
        "",
        "Model-inferred recurring interviewer-question families from answer-only ASR",
        "transcripts. Exact original wording is unavailable unless an answer explicitly",
        "echoes it. Sequential 20-second window numbers are not question IDs.",
        "",
        "| " + " | ".join(FINAL_TABLE_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(FINAL_TABLE_COLUMNS)) + "|",
    ]
    for row in table_rows:
        cells = []
        for column in FINAL_TABLE_COLUMNS:
            value = str(row[column])
            value = value.replace("|", "\\|").replace("\n", " ")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    md_path = run_dir / "turkish_inferred_questions.md"
    _atomic_write_text("\n".join(lines), md_path)

    serialized = json.dumps(table_rows, ensure_ascii=False, sort_keys=True)
    rows_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        "rows": len(table_rows),
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
        "rows_sha256": rows_hash,
    }


def _atomic_write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
    tmp.replace(path)


def load_table_rows(path: str | Path, fmt: str = "json") -> list[dict[str, Any]]:
    path = Path(path)
    if fmt == "json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        return [dict(row) for row in rows]
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [_coerce_row(row) for row in reader]
    if fmt == "md":
        rows: list[dict[str, Any]] = []
        for line in path.open("r", encoding="utf-8"):
            if not line.startswith("| "):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                continue
            if cells == list(FINAL_TABLE_COLUMNS):
                continue
            if len(cells) == len(FINAL_TABLE_COLUMNS):
                rows.append(
                    _coerce_row({column: cells[index] for index, column in enumerate(FINAL_TABLE_COLUMNS)})
                )
        return rows
    raise ValueError(f"unknown format {fmt!r}")


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize CSV/Markdown rows to the same types as the JSON rows."""
    coerced = dict(row)
    for column in ("order", "supporting_subjects"):
        if column in coerced and coerced[column] not in (None, ""):
            coerced[column] = int(coerced[column])
    return coerced
