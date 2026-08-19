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
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.qwen38.contracts import (
    CONFIDENCE_WEIGHTS,
    CONSOLIDATION_FINAL_SIMPLIFIED_SCHEMA,
    FALLBACK_INFERENCE_SCHEMA,
    FINAL_TABLE_COLUMNS,
    INFERENCE_POLICY,
    INFERENCE_POLICY_VERSION,
    FALLBACK_PROMPT_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    STRICT_PROMPT_VERSION,
    SUBJECT_INFERENCE_SCHEMA,
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
    ngram_overlap_at_least,
    parse_filename_stem,
    request_settings,
    structured_output_schema,
    validate_consolidation_batch,
    validate_consolidation_final,
    validate_consolidation_final_simplified,
    validate_fallback_inference,
    validate_subject_inference,
)

PROMPT_VERSION = "qwen38_turkish_v3"
STRICT_PROMPT_VERSION = STRICT_PROMPT_VERSION
FALLBACK_PROMPT_VERSION = FALLBACK_PROMPT_VERSION
INFERENCE_POLICY_VERSION = INFERENCE_POLICY_VERSION
EPISODE_SAFETY_POLICY_VERSION = "qwen38_episode_safety_v1"
EPISODE_TEXT_FIELDS = (
    "question_tr",
    "question_en",
    "evidence_basis",
    "abstain_reason",
)
EPISODE_SAFETY_REASON_CODES = (
    "forbidden_marker_or_identifier",
    "privacy_overlap_12_tokens",
)
CORRECTION_REASON_CODES = (
    "invalid_json",
    "invalid_schema",
    *EPISODE_SAFETY_REASON_CODES,
)
EPISODE_SAFETY_FIELD_NAMES = EPISODE_TEXT_FIELDS
_CANONICAL_SEQUENCE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])S[0-9]{4}(?![A-Za-z0-9])")

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
    "Do not use any quote, apostrophe, or backtick character in evidence_basis.\n"
    "Do not include window markers, sequence identifiers, or copied answer text in any free-text field.\n"
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
    "Correct the previous response. The permitted validation categories are "
    "invalid_json, invalid_schema, forbidden_marker_or_identifier, and "
    "privacy_overlap_12_tokens. Return exactly one JSON object matching the "
    "subject schema from the original request, with no validation explanation "
    "and no private content."
)

FALLBACK_SYSTEM_PROMPT = (
    "You infer recurring interviewer questions from answer-only Turkish transcripts. "
    "You see only participant answers marked [WINDOW n] in chronological order; "
    "questions are inferred, not recovered verbatim. The window numbers are sequential "
    "20-second indices, not question IDs. For each distinct inferred question family, "
    "return one concise Turkish and English question and its framing label. Classify "
    "the inferred question framing, not answer tone:\n"
    "- POSITIVE asks about happiness, enjoyment, strengths, hope, support, positive "
    "memories, or pleasant events.\n"
    "- NEGATIVE asks about sadness, distress, problems, symptoms, loss, fear, conflict, "
    "or unpleasant events.\n"
    "- NEUTRAL is factual, descriptive, demographic, procedural, or open framing.\n"
    "- MIXED asks for both positive and negative material in one question.\n"
    "Do not include window markers, sequence identifiers, quotes, apostrophes, "
    "backticks, copied answer text, or reasoning in any field. Return exactly one JSON object, "
    "no prose, no markdown fences, with this exact shape:\n"
    '{\n'
    '  "questions": [\n'
    '    {\n'
    '      "question_tr": "concise inferred Turkish question",\n'
    '      "question_en": "concise inferred English question",\n'
    '      "label": "POSITIVE | NEGATIVE | NEUTRAL | MIXED",\n'
    '      "confidence": "HIGH | MEDIUM | LOW"\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "Use exactly the field names shown."
)

FALLBACK_CORRECTION_MESSAGE = (
    "Correct the previous simplified response. The permitted categories are "
    "invalid_json, invalid_schema, forbidden_marker_or_identifier, and "
    "privacy_overlap_12_tokens. Return exactly one JSON object matching the "
    "simplified schema with question_tr, question_en, label, and confidence only, "
    "with no extra fields, no window markers, no identifiers, no quotes, no copied "
    "answer text, and no private content."
)

CONSOLIDATION_CORRECTION_MESSAGE = (
    "Correct the previous consolidation response. The permitted categories are "
    "invalid_json, invalid_schema, and assignment coverage. Return exactly one JSON "
    "object matching the required schema with exact candidate or cluster assignment, "
    "with no extra fields and no private content."
)

EVIDENCE_BASIS_FALLBACK = "Inferred from response topic and framing"
_EVIDENCE_QUOTES = '"\'`“”‘’«»‹›„‟‚‛'
_EVIDENCE_QUOTE_TRANSLATION = str.maketrans({character: None for character in _EVIDENCE_QUOTES})
_PROMPT_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


EPISODE_SAFETY_POLICY = {
    "version": EPISODE_SAFETY_POLICY_VERSION,
    "text_fields": list(EPISODE_TEXT_FIELDS),
    "normalization": {
        "unicode": "NFKC",
        "removed_quote_characters": list(_EVIDENCE_QUOTES),
        "replace_with_space": ["newline", "carriage_return", "tab"],
        "collapse_whitespace": "unicode",
        "strip": True,
        "evidence_basis_max_characters": 200,
        "evidence_basis_fallback": EVIDENCE_BASIS_FALLBACK,
    },
    "forbidden_marker_or_identifier": {
        "markers": ["[WINDOW", "]"],
        "current_sequence_id": True,
        "canonical_sequence_token_regex": r"S[0-9]{4}",
    },
    "privacy_overlap_tokens": 12,
    "reason_codes": list(EPISODE_SAFETY_REASON_CODES),
}
EPISODE_SAFETY_POLICY_SHA256 = _sha256_text(_canonical_json(EPISODE_SAFETY_POLICY))

FALLBACK_PROMPT_SHA256 = _sha256_text(FALLBACK_SYSTEM_PROMPT)
FALLBACK_CORRECTION_SHA256 = _sha256_text(FALLBACK_CORRECTION_MESSAGE)
CONSOLIDATION_CORRECTION_SHA256 = _sha256_text(CONSOLIDATION_CORRECTION_MESSAGE)
STRICT_SCHEMA_SHA256 = _sha256_text(_canonical_json(SUBJECT_INFERENCE_SCHEMA))
FALLBACK_SCHEMA_SHA256 = _sha256_text(_canonical_json(FALLBACK_INFERENCE_SCHEMA))
INFERENCE_POLICY_SHA256 = _sha256_text(_canonical_json(INFERENCE_POLICY))


def normalize_episode_text(value: str) -> str:
    """Apply only the harmless formatting normalization allowed by the plan."""
    if not isinstance(value, str):
        raise TypeError("episode text fields must be strings")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(_EVIDENCE_QUOTE_TRANSLATION)
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _PROMPT_WHITESPACE_RE.sub(" ", normalized).strip()


def sanitize_evidence_basis(value: str) -> str:
    """Normalize and bound only the evidence description."""
    normalized = normalize_episode_text(value)
    if len(normalized) > 200:
        truncated = normalized[:200]
        boundary = truncated.rfind(" ")
        normalized = truncated[:boundary].rstrip() if boundary > 0 else truncated.rstrip()
    return normalized or EVIDENCE_BASIS_FALLBACK


def prompt_contract_payload(
    *,
    model_revision: str = MODEL_REVISION,
    max_tokens: int = TURKISH_MAX_TOKENS,
    seed: int = 42,
) -> dict[str, Any]:
    """Return the stable prompt contract, excluding a subject's user prompt."""
    settings = request_settings(max_tokens)
    return {
        "prompt_version": PROMPT_VERSION,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "inference_policy_sha256": INFERENCE_POLICY_SHA256,
        "inference_policy": INFERENCE_POLICY,
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "episode_safety_policy": EPISODE_SAFETY_POLICY,
        "SUBJECT_SYSTEM_PROMPT": SUBJECT_SYSTEM_PROMPT,
        "FALLBACK_SYSTEM_PROMPT": FALLBACK_SYSTEM_PROMPT,
        "SCHEMA_CORRECTION_MESSAGE": SCHEMA_CORRECTION_MESSAGE,
        "FALLBACK_CORRECTION_MESSAGE": FALLBACK_CORRECTION_MESSAGE,
        "CONSOLIDATION_CORRECTION_MESSAGE": CONSOLIDATION_CORRECTION_MESSAGE,
        "subject_output_schema": SUBJECT_INFERENCE_SCHEMA,
        "fallback_output_schema": FALLBACK_INFERENCE_SCHEMA,
        "strict_schema_sha256": STRICT_SCHEMA_SHA256,
        "fallback_schema_sha256": FALLBACK_SCHEMA_SHA256,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "seed": seed,
        "max_tokens": settings["max_tokens"],
        "enable_thinking": settings["chat_template_kwargs"]["enable_thinking"],
        "preserve_thinking": settings["chat_template_kwargs"]["preserve_thinking"],
    }


def prompt_contract_sha256(
    *,
    model_revision: str = MODEL_REVISION,
    max_tokens: int = TURKISH_MAX_TOKENS,
    seed: int = 42,
) -> str:
    return _sha256_text(_canonical_json(prompt_contract_payload(
        model_revision=model_revision, max_tokens=max_tokens, seed=seed
    )))


def prompt_bundle_sha256(
    user_prompt: str,
    *,
    model_revision: str = MODEL_REVISION,
    max_tokens: int = TURKISH_MAX_TOKENS,
    seed: int = 42,
) -> str:
    payload = prompt_contract_payload(
        model_revision=model_revision, max_tokens=max_tokens, seed=seed
    )
    payload["user_prompt"] = user_prompt
    return _sha256_text(_canonical_json(payload))


def prompt_component_hashes(
    user_prompt: str,
    *,
    model_revision: str = MODEL_REVISION,
    max_tokens: int = TURKISH_MAX_TOKENS,
    seed: int = 42,
) -> dict[str, str]:
    return {
        "user_prompt_sha256": _sha256_text(user_prompt),
        "system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
        "fallback_prompt_sha256": FALLBACK_PROMPT_SHA256,
        "correction_message_sha256": _sha256_text(SCHEMA_CORRECTION_MESSAGE),
        "fallback_correction_sha256": FALLBACK_CORRECTION_SHA256,
        "subject_schema_sha256": STRICT_SCHEMA_SHA256,
        "fallback_schema_sha256": FALLBACK_SCHEMA_SHA256,
        "generation_settings_hash": generation_settings_hash(max_tokens),
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "inference_policy_sha256": INFERENCE_POLICY_SHA256,
        "prompt_contract_sha256": prompt_contract_sha256(
            model_revision=model_revision, max_tokens=max_tokens, seed=seed
        ),
        "prompt_bundle_sha256": prompt_bundle_sha256(
            user_prompt, model_revision=model_revision, max_tokens=max_tokens, seed=seed
        ),
    }


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


TURKISH_RUN_ID_RE = re.compile(r"^q38tr_[0-9a-f]{12}_attempt[0-9]+$")


def validate_turkish_run_id(turkish_run_id: str) -> None:
    if not isinstance(turkish_run_id, str) or TURKISH_RUN_ID_RE.fullmatch(turkish_run_id) is None:
        raise ValueError(
            "turkish_run_id must match q38tr_<12 lowercase source-SHA hex>_attempt<N>"
        )


def _selection_metadata(
    selection_file: str | Path | None,
    selected_tp: int | None,
) -> dict[str, Any]:
    if selection_file is None:
        return {
            "selection_file": None,
            "selection_file_sha256": None,
            "selected_tp": selected_tp,
            "selection_version": None,
        }
    path = Path(selection_file)
    if not path.is_file():
        raise ValueError(f"selection file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("selection_version") != 2:
        raise ValueError("Turkish analysis requires serving_selection_v2.json")
    file_tp = selection.get("selected_tp")
    if file_tp not in (1, 2, 4):
        raise ValueError(f"selection v2 has invalid selected_tp: {file_tp!r}")
    if selected_tp is not None and int(selected_tp) != int(file_tp):
        raise ValueError(f"selected TP {selected_tp} does not match selection v2 {file_tp}")
    return {
        "selection_file": str(path),
        "selection_file_sha256": _sha256_file(path),
        "selected_tp": int(file_tp),
        "selection_version": 2,
    }


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
    turkish_run_id: str | None = None,
    analysis_attempt: int = 1,
    selected_tp: int | None = None,
    selection_file: str | Path | None = None,
    supersedes_job_ids: Sequence[str] | None = None,
    seed: int = 42,
    source_sha256: str = TURKISH_SOURCE_HASH,
    expected_sequences: int = TURKISH_EXPECTED_SEQUENCES,
    expected_windows: int = TURKISH_EXPECTED_WINDOWS,
) -> dict[str, Any]:
    """Parse, group, and verify the Turkish windows; write restricted packets."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError(f"refusing to reuse existing Turkish run root: {run_dir}")
    if turkish_run_id is not None:
        validate_turkish_run_id(turkish_run_id)
    if analysis_attempt < 1:
        raise ValueError("analysis_attempt must be positive")
    selection = _selection_metadata(selection_file, selected_tp)
    if selection["selected_tp"] is not None:
        selected_tp = selection["selected_tp"]
    turkish_run_id = turkish_run_id or deployment_id
    supersedes_job_ids = [str(job_id) for job_id in (supersedes_job_ids or [])]
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
    contract_hash = prompt_contract_sha256(
        model_revision=model_revision, max_tokens=TURKISH_MAX_TOKENS, seed=seed
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    restricted = run_dir / "restricted"
    restricted.mkdir(parents=True, exist_ok=False)
    manifest_payload = {
        "turkish_run_id": turkish_run_id,
        "analysis_attempt": int(analysis_attempt),
        "deployment_id": deployment_id,
        "source_sha256": actual_hash,
        "source_commit": source_commit,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "prompt_version": PROMPT_VERSION,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "inference_policy_sha256": INFERENCE_POLICY_SHA256,
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "prompt_contract_sha256": contract_hash,
        "system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
        "fallback_prompt_sha256": FALLBACK_PROMPT_SHA256,
        "correction_message_sha256": _sha256_text(SCHEMA_CORRECTION_MESSAGE),
        "fallback_correction_sha256": FALLBACK_CORRECTION_SHA256,
        "subject_schema_sha256": STRICT_SCHEMA_SHA256,
        "fallback_schema_sha256": FALLBACK_SCHEMA_SHA256,
        "generation_settings_hash": generation_hash,
        "request_settings": request_settings(TURKISH_MAX_TOKENS),
        "selected_tp": selected_tp,
        "selection_version": selection["selection_version"],
        "selection_file": selection["selection_file"],
        "selection_file_sha256": selection["selection_file_sha256"],
        "expected_windows": expected_windows,
        "expected_sequences": expected_sequences,
        "consolidation_batches": list(TURKISH_CONSOLIDATION_BATCHES),
        "supersedes_job_ids": supersedes_job_ids,
    }
    manifest_path = run_dir / "run_manifest.json"
    _atomic_write_json(manifest_payload, manifest_path)
    _restrict(manifest_path)
    run_manifest_sha256 = _sha256_file(manifest_path)

    sequences: list[dict[str, Any]] = []
    for subject in ordered_subjects:
        sequence_id = sequence_ids[subject]
        windows = sorted(windows_by_subject[subject], key=lambda item: item[0])
        blocks = [f"[WINDOW {window}]\n{text}" for window, text in windows]
        user_prompt = f"Sequence id: {sequence_id}\n\n" + "\n\n".join(blocks)
        components = prompt_component_hashes(
            user_prompt,
            model_revision=model_revision,
            max_tokens=TURKISH_MAX_TOKENS,
            seed=seed,
        )
        sequences.append(
            {
                "turkish_run_id": turkish_run_id,
                "analysis_attempt": int(analysis_attempt),
                "sequence_id": sequence_id,
                "subject_sha256": subject_hash[subject],
                "window_count": len(windows),
                "windows": [{"window": window, "text": text} for window, text in windows],
                "user_prompt": user_prompt,
                "prompt_hash": components["user_prompt_sha256"],
                **components,
                "run_manifest_sha256": run_manifest_sha256,
                "source_sha256": actual_hash,
                "source_commit": source_commit,
                "model_id": MODEL_ID,
                "model_revision": model_revision,
                "generation_settings_hash": generation_hash,
                "prompt_version": PROMPT_VERSION,
                "strict_prompt_version": STRICT_PROMPT_VERSION,
                "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
                "inference_policy_version": INFERENCE_POLICY_VERSION,
                "inference_policy_sha256": INFERENCE_POLICY_SHA256,
                "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
                "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
                "deployment_id": deployment_id,
            }
        )

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
        "turkish_run_id": turkish_run_id,
        "analysis_attempt": int(analysis_attempt),
        "source_sha256": actual_hash,
        "windows": total_windows,
        "sequences": len(sequences),
        "subject_map_path": str(map_path),
        "prepared_sequences_path": str(packets_path),
        "prompt_version": PROMPT_VERSION,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "inference_policy_sha256": INFERENCE_POLICY_SHA256,
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "prompt_contract_sha256": contract_hash,
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": run_manifest_sha256,
        "selected_tp": selected_tp,
        "selection_file": selection["selection_file"],
        "selection_file_sha256": selection["selection_file_sha256"],
        "supersedes_job_ids": supersedes_job_ids,
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


def _normalize_episode_fields(episode: dict[str, Any]) -> dict[str, Any]:
    """Normalize all four free-text fields before episode safety checks."""
    for field_name in EPISODE_TEXT_FIELDS:
        value = episode[field_name]
        if field_name == "evidence_basis":
            episode[field_name] = sanitize_evidence_basis(value)
        else:
            episode[field_name] = normalize_episode_text(value)
    return episode


def _validate_subject_schema(payload: Any, sequence_id: str) -> list[dict[str, Any]]:
    """Parse-independent schema and semantic validation, without safety checks."""
    errors = validate_subject_inference(payload)
    if errors:
        raise ValueError("invalid subject schema")
    episodes = payload["episodes"]
    orders: set[int] = set()
    for episode in episodes:
        _normalize_episode_fields(episode)
        if episode["sequence_id"] != sequence_id:
            raise ValueError("invalid subject schema")
        order = episode["episode_order"]
        if order in orders:
            raise ValueError("invalid subject schema")
        orders.add(order)
        if not episode["abstain_reason"] and not episode["question_tr"].strip():
            raise ValueError("invalid subject schema")
    return episodes


def classify_episode_safety(
    episode: dict[str, Any],
    sequence_id: str,
    transcript_windows: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Return safe reason codes and field names without retaining trigger text."""
    marker_fields: list[str] = []
    overlap_fields: list[str] = []
    for field_name in EPISODE_SAFETY_FIELD_NAMES:
        text = episode[field_name]
        has_marker_or_identifier = (
            "[window" in text.casefold()
            or "]" in text
            or sequence_id in text
            or _CANONICAL_SEQUENCE_TOKEN_RE.search(text) is not None
        )
        if has_marker_or_identifier:
            marker_fields.append(field_name)
        if transcript_windows and ngram_overlap_at_least(text, transcript_windows, 12):
            overlap_fields.append(field_name)

    reason_codes: list[str] = []
    field_names: list[str] = []
    if marker_fields:
        reason_codes.append("forbidden_marker_or_identifier")
        field_names.extend(marker_fields)
    if overlap_fields:
        reason_codes.append("privacy_overlap_12_tokens")
        field_names.extend(overlap_fields)
    return {
        "reason_codes": reason_codes,
        "field_names": list(dict.fromkeys(field_names)),
    }


def filter_safe_episodes(
    episodes: Sequence[dict[str, Any]],
    sequence_id: str,
    transcript_windows: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain safe episodes and record only bounded exclusion metadata."""
    retained: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for episode in episodes:
        safety = classify_episode_safety(episode, sequence_id, transcript_windows)
        if safety["reason_codes"]:
            exclusions.append(
                {
                    "episode_order": episode["episode_order"],
                    "reason_codes": safety["reason_codes"],
                    "field_names": safety["field_names"],
                }
            )
        else:
            retained.append(episode)
    return retained, exclusions

def _validate_fallback_payload(payload: Any) -> list[dict[str, Any]]:
    """Validate simplified question schema; returns normalized questions."""
    errors = validate_fallback_inference(payload)
    if errors:
        raise ValueError("invalid fallback schema")
    questions = payload["questions"]
    for idx, q in enumerate(questions):
        if not isinstance(q["question_tr"], str) or not q["question_tr"].strip():
            raise ValueError("invalid fallback schema")
        if not isinstance(q["question_en"], str) or not q["question_en"].strip():
            raise ValueError("invalid fallback schema")
    return questions


def _fallback_to_episodes(
    questions: list[dict[str, Any]],
    sequence_id: str,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        episodes.append(
            {
                "sequence_id": sequence_id,
                "episode_order": idx,
                "question_tr": q["question_tr"],
                "question_en": q["question_en"],
                "label": q["label"],
                "wording_status": WordingStatus.INFERRED_PARAPHRASE.value,
                "confidence": q["confidence"],
                "evidence_window_indices": [],
                "evidence_basis": EVIDENCE_BASIS_FALLBACK,
                "abstain_reason": "",
            }
        )
    return episodes


def _fallback_response_validation(
    content: str,
    sequence_id: str,
    transcript_windows: Sequence[str],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, list[str]]:
    if not content.strip():
        return None, None, ["invalid_json"]
    parsed = _parse_json_object(content)
    if parsed is None:
        return None, None, ["invalid_json"]
    try:
        questions = _validate_fallback_payload(parsed)
    except Exception:
        return None, None, ["invalid_schema"]
    episodes = _fallback_to_episodes(questions, sequence_id)
    # Normalize (including question fields) before safety checks
    for ep in episodes:
        _normalize_episode_fields(ep)
    retained, exclusions = filter_safe_episodes(episodes, sequence_id, transcript_windows)
    reasons = _ordered_reason_codes(
        reason for exclusion in exclusions for reason in exclusion["reason_codes"]
    )
    return retained, exclusions, reasons



def _validate_episodes(
    payload: dict[str, Any],
    sequence_id: str,
    transcript_windows: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Strictly validate a response, including episode safety."""
    episodes = _validate_subject_schema(payload, sequence_id)
    _, exclusions = filter_safe_episodes(episodes, sequence_id, transcript_windows)
    if exclusions:
        reasons = sorted(
            {reason for exclusion in exclusions for reason in exclusion["reason_codes"]},
            key=EPISODE_SAFETY_REASON_CODES.index,
        )
        raise ValueError(f"{sequence_id}: {'; '.join(reasons)}")
    return episodes


def _episode_provenance(sequence: dict[str, Any]) -> dict[str, Any]:
    components = prompt_component_hashes(
        sequence["user_prompt"],
        model_revision=sequence["model_revision"],
        max_tokens=TURKISH_MAX_TOKENS,
        seed=42,
    )
    return {
        "turkish_run_id": sequence.get("turkish_run_id"),
        "analysis_attempt": sequence.get("analysis_attempt"),
        "deployment_id": sequence.get("deployment_id"),
        "model_id": sequence.get("model_id", MODEL_ID),
        "prompt_version": sequence.get("prompt_version"),
        "episode_safety_policy_version": sequence.get("episode_safety_policy_version"),
        "episode_safety_policy_sha256": sequence.get("episode_safety_policy_sha256"),
        "run_manifest_sha256": sequence.get("run_manifest_sha256"),
        "source_sha256": sequence["source_sha256"],
        "source_commit": sequence["source_commit"],
        "model_revision": sequence["model_revision"],
        **components,
        "prompt_hash": components["user_prompt_sha256"],
    }


def _verify_current_prompt_contract(sequence: dict[str, Any], *, max_tokens: int, seed: int) -> None:
    expected = prompt_component_hashes(
        sequence["user_prompt"],
        model_revision=sequence["model_revision"],
        max_tokens=max_tokens,
        seed=seed,
    )
    if sequence.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(
            f"resume refused for {sequence.get('sequence_id')}: prompt version changed"
        )
    if sequence.get("episode_safety_policy_version") != EPISODE_SAFETY_POLICY_VERSION:
        raise ValueError(
            f"resume refused for {sequence.get('sequence_id')}: episode safety policy version changed"
        )
    if sequence.get("episode_safety_policy_sha256") != EPISODE_SAFETY_POLICY_SHA256:
        raise ValueError(
            f"resume refused for {sequence.get('sequence_id')}: episode safety policy changed"
        )
    for key, value in expected.items():
        if sequence.get(key) != value:
            raise ValueError(
                f"resume refused for {sequence.get('sequence_id')}: {key} changed "
                f"(stored {sequence.get(key)!r} != current {value!r})"
            )


def _completed_inference(inference_path: Path) -> dict[str, Any] | None:
    if not inference_path.is_file():
        return None
    with inference_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) and data.get("status") == "completed" else None


def _ordered_reason_codes(reason_codes: Iterable[str]) -> list[str]:
    present = set(reason_codes)
    return [code for code in CORRECTION_REASON_CODES if code in present]


def _response_validation(
    content: str,
    sequence_id: str,
    transcript_windows: Sequence[str],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, list[str]]:
    """Parse, schema-check, normalize, and classify one model response."""
    if not content.strip():
        return None, None, ["invalid_json"]
    parsed = _parse_json_object(content)
    if parsed is None:
        return None, None, ["invalid_json"]
    try:
        episodes = _validate_subject_schema(parsed, sequence_id)
    except Exception:
        return None, None, ["invalid_schema"]
    retained, exclusions = filter_safe_episodes(episodes, sequence_id, transcript_windows)
    reasons = _ordered_reason_codes(
        reason
        for exclusion in exclusions
        for reason in exclusion["reason_codes"]
    )
    return retained, exclusions, reasons


def correction_message(reason_codes: Sequence[str]) -> str:
    """Build a category-only correction message without rejected content."""
    categories = _ordered_reason_codes(reason_codes)
    if not categories:
        categories = list(CORRECTION_REASON_CODES)
    return f"{SCHEMA_CORRECTION_MESSAGE} Categories observed: {', '.join(categories)}."



def _strict_response_validation(
    content: str,
    sequence_id: str,
    transcript_windows: Sequence[str],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, list[str]]:
    """Strict route validation: JSON/schema + safety filtering."""
    if not content.strip():
        return None, None, ["invalid_json"]
    parsed = _parse_json_object(content)
    if parsed is None:
        return None, None, ["invalid_json"]
    try:
        episodes = _validate_subject_schema(parsed, sequence_id)
    except Exception:
        return None, None, ["invalid_schema"]
    retained, exclusions = filter_safe_episodes(episodes, sequence_id, transcript_windows)
    reasons = _ordered_reason_codes(
        reason for exclusion in exclusions for reason in exclusion["reason_codes"]
    )
    return retained, exclusions, reasons


def infer_subjects(
    prepared_path: str | Path,
    inferences_dir: str | Path,
    *,
    base_url: str,
    model: str = SERVED_MODEL,
    concurrency: int = TURKISH_INFERENCE_CONCURRENCY,
    seed: int = 42,
    max_tokens: int = TURKISH_MAX_TOKENS,
    source_commit: str | None = None,
    turkish_run_id: str | None = None,
) -> dict[str, Any]:
    """Run deterministic ladder per subject: strict -> fallback -> exclusion."""
    sequences = load_prepared_sequences(prepared_path)
    inferences_dir = Path(inferences_dir)
    inferences_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(prepared_path).parent.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest missing: {manifest_path}")
    manifest_hash = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if turkish_run_id is not None and manifest.get("turkish_run_id") != turkish_run_id:
        raise ValueError("resume refused: Turkish run identity changed")
    if source_commit is not None and manifest.get("source_commit") != source_commit:
        raise ValueError("resume refused: source commit changed")
    if manifest.get("run_manifest_sha256") not in (None, manifest_hash):
        raise ValueError("run manifest self-reference is invalid")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("resume refused: prompt version changed")
    if manifest.get("strict_prompt_version") != STRICT_PROMPT_VERSION:
        raise ValueError("resume refused: strict prompt version changed")
    if manifest.get("fallback_prompt_version") != FALLBACK_PROMPT_VERSION:
        raise ValueError("resume refused: fallback prompt version changed")
    if manifest.get("inference_policy_version") != INFERENCE_POLICY_VERSION:
        raise ValueError("resume refused: inference policy version changed")
    if manifest.get("episode_safety_policy_version") != EPISODE_SAFETY_POLICY_VERSION:
        raise ValueError("resume refused: episode safety policy version changed")
    if manifest.get("episode_safety_policy_sha256") != EPISODE_SAFETY_POLICY_SHA256:
        raise ValueError("resume refused: episode safety policy changed")
    expected_contract = prompt_contract_sha256(
        model_revision=manifest.get("model_revision", MODEL_REVISION),
        max_tokens=max_tokens,
        seed=seed,
    )
    if manifest.get("prompt_contract_sha256") != expected_contract:
        raise ValueError("resume refused: prompt contract changed")
    for sequence in sequences:
        if sequence.get("run_manifest_sha256") != manifest_hash:
            raise ValueError(f"{sequence.get('sequence_id')}: run manifest hash mismatch")
        if sequence.get("source_commit") != manifest.get("source_commit"):
            raise ValueError(f"{sequence.get('sequence_id')}: source commit mismatch")
        _verify_current_prompt_contract(sequence, max_tokens=max_tokens, seed=seed)
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
            # also verify new ladder fields if present in existing record
            # require subject_status etc to be present
            if "subject_status" not in existing or "route_used" not in existing:
                raise ValueError(f"resume refused for {sequence_id}: ladder provenance missing")
            completed += 1
            continue
        if inference_path.exists():
            raise ValueError(f"resume refused for {sequence_id}: incomplete or invalid record")
        pending.append(sequence)

    settings = request_settings(max_tokens)

    async def infer_one(sequence: dict[str, Any]) -> tuple[str, bool, str | None]:
        sequence_id = sequence["sequence_id"]
        transcript_windows = [window["text"] for window in sequence.get("windows", [])]
        strict_gen_count = 0
        fallback_gen_count = 0
        strict_failure_cats: list[str] = []
        fallback_failure_cats: list[str] = []
        # ---------- Route A: strict ----------
        strict_retained: list[dict[str, Any]] | None = None
        strict_exclusions: list[dict[str, Any]] | None = None
        strict_episodes_returned = 0
        first_strict_failure: list[str] = []
        strict_success = False
        for attempt in (1, 2):
            strict_gen_count += 1
            messages = [
                {"role": "system", "content": SUBJECT_SYSTEM_PROMPT},
                {"role": "user", "content": sequence["user_prompt"]},
            ]
            if attempt == 2:
                messages.append({"role": "user", "content": correction_message(first_strict_failure)})
            try:
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
            except Exception:
                return sequence_id, False, "request_failed"
            retained, exclusions, reasons = _strict_response_validation(content, sequence_id, transcript_windows)
            if retained is None:
                # invalid json/schema
                cats = _ordered_reason_codes(reasons)
                if attempt == 1:
                    first_strict_failure = cats
                    strict_failure_cats = cats
                    continue
                else:
                    strict_failure_cats = _ordered_reason_codes(strict_failure_cats + cats)
                    break
            else:
                # valid schema, possibly with exclusions
                if reasons:
                    # some episodes unsafe
                    if attempt == 1:
                        first_strict_failure = _ordered_reason_codes(reasons)
                        strict_failure_cats = first_strict_failure
                        continue
                    else:
                        # second attempt: have retained/exclusions
                        # Record failure cats from first attempt
                        # If retained non-empty, success via strict
                        if retained:
                            strict_retained = retained
                            strict_exclusions = exclusions
                            strict_episodes_returned = len(retained) + len(exclusions)
                            strict_failure_cats = first_strict_failure
                            strict_success = True
                            break
                        else:
                            # valid but all filtered -> treat as strict failure needing fallback
                            strict_failure_cats = _ordered_reason_codes(first_strict_failure + reasons)
                            # keep retained/exclusions for fallback decision but not success
                            strict_retained = retained
                            strict_exclusions = exclusions
                            break
                else:
                    # no safety reasons, fully valid
                    strict_retained = retained
                    strict_exclusions = exclusions
                    strict_episodes_returned = len(retained) + len(exclusions)
                    strict_failure_cats = first_strict_failure if attempt == 2 else []
                    strict_success = True
                    break
        if strict_success and strict_retained is not None:
            # Route A succeeded: keep safe episodes, ignore fallback
            record = dict(_episode_provenance(sequence))
            record.update(
                {
                    "sequence_id": sequence_id,
                    "status": "completed",
                    "subject_status": "INCLUDED",
                    "route_used": "STRICT",
                    "strict_generation_count": strict_gen_count,
                    "fallback_generation_count": 0,
                    "strict_failure_categories": strict_failure_cats,
                    "fallback_failure_categories": [],
                    "generation_attempts": strict_gen_count,
                    "correction_reason_codes": strict_failure_cats,
                    "episodes_returned": len(strict_retained) + len(strict_exclusions or []),
                    "episodes_retained": len(strict_retained),
                    "episode_exclusions": strict_exclusions or [],
                    "subject_exclusion_reason": "",
                    "episodes": strict_retained,
                    "episode_count": len(strict_retained),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "strict_prompt_version": STRICT_PROMPT_VERSION,
                    "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
                    "inference_policy_version": INFERENCE_POLICY_VERSION,
                    "strict_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
                    "fallback_prompt_sha256": FALLBACK_PROMPT_SHA256,
                    "inference_policy_sha256": INFERENCE_POLICY_SHA256,
                }
            )
            inference_path = inferences_dir / f"{sequence_id}.json"
            _atomic_write_json(record, inference_path)
            _restrict(inference_path)
            return sequence_id, True, None
        # If strict had valid schema but all episodes filtered (retained empty with exclusions),
        # we do NOT return success; fall through to fallback.
        # Similarly if strict both attempts invalid, fall through.

        # ---------- Route B: fallback simplified ----------
        fallback_retained: list[dict[str, Any]] | None = None
        fallback_exclusions: list[dict[str, Any]] | None = None
        first_fallback_failure: list[str] = []
        fallback_success = False
        for fb_attempt in (1, 2):
            fallback_gen_count += 1
            messages = [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": sequence["user_prompt"]},
            ]
            if fb_attempt == 2:
                # category-only correction for fallback
                cats = first_fallback_failure if first_fallback_failure else strict_failure_cats
                correction = f"{FALLBACK_CORRECTION_MESSAGE} Categories observed: {', '.join(_ordered_reason_codes(cats) or list(CORRECTION_REASON_CODES))}."
                messages.append({"role": "user", "content": correction})
            try:
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
            except Exception:
                return sequence_id, False, "request_failed"
            retained, exclusions, reasons = _fallback_response_validation(content, sequence_id, transcript_windows)
            if retained is None:
                cats = _ordered_reason_codes(reasons)
                if fb_attempt == 1:
                    first_fallback_failure = cats
                    fallback_failure_cats = cats
                    continue
                else:
                    fallback_failure_cats = _ordered_reason_codes(fallback_failure_cats + cats)
                    break
            else:
                if reasons:
                    # some unsafe in fallback
                    if fb_attempt == 1:
                        first_fallback_failure = _ordered_reason_codes(reasons)
                        fallback_failure_cats = first_fallback_failure
                        continue
                    else:
                        if retained:
                            fallback_retained = retained
                            fallback_exclusions = exclusions
                            fallback_failure_cats = first_fallback_failure
                            fallback_success = True
                            break
                        else:
                            # all filtered
                            fallback_failure_cats = _ordered_reason_codes(first_fallback_failure + reasons)
                            break
                else:
                    # no safety reasons
                    if retained:
                        fallback_retained = retained
                        fallback_exclusions = exclusions
                        fallback_failure_cats = first_fallback_failure if fb_attempt == 2 else []
                        fallback_success = True
                        break
                    else:
                        # empty questions array -> treat as zero retained
                        # Consider this as success with zero episodes? But spec says exclusion if all valid items unsafe.
                        # Empty is not unsafe, but yields zero. We'll treat as success with zero.
                        fallback_retained = retained
                        fallback_exclusions = exclusions
                        fallback_success = True
                        break
        if fallback_success and fallback_retained is not None:
            # Route B succeeded with at least one retained (or zero but still valid)
            # For spec, if fallback returned at least one safe, it's INCLUDED
            # If fallback returned zero but valid (empty), also INCLUDED zero? We'll treat as INCLUDED.
            record = dict(_episode_provenance(sequence))
            record.update(
                {
                    "sequence_id": sequence_id,
                    "status": "completed",
                    "subject_status": "INCLUDED",
                    "route_used": "SIMPLIFIED",
                    "strict_generation_count": strict_gen_count,
                    "fallback_generation_count": fallback_gen_count,
                    "strict_failure_categories": strict_failure_cats,
                    "fallback_failure_categories": fallback_failure_cats,
                    "generation_attempts": strict_gen_count + fallback_gen_count,
                    "correction_reason_codes": strict_failure_cats,
                    "episodes_returned": len(fallback_retained) + len(fallback_exclusions or []),
                    "episodes_retained": len(fallback_retained),
                    "episode_exclusions": fallback_exclusions or [],
                    "subject_exclusion_reason": "",
                    "episodes": fallback_retained,
                    "episode_count": len(fallback_retained),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "strict_prompt_version": STRICT_PROMPT_VERSION,
                    "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
                    "inference_policy_version": INFERENCE_POLICY_VERSION,
                    "strict_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
                    "fallback_prompt_sha256": FALLBACK_PROMPT_SHA256,
                    "inference_policy_sha256": INFERENCE_POLICY_SHA256,
                }
            )
            inference_path = inferences_dir / f"{sequence_id}.json"
            _atomic_write_json(record, inference_path)
            _restrict(inference_path)
            return sequence_id, True, None

        # ---------- Route C: safe exclusion ----------
        # Build excluded record with zero episodes
        # Need to collect failure cats: if strict both invalid, those cats; fallback cats similarly.
        # If fallback had valid but all filtered, cats include safety reasons.
        # For excluded, strict_generation_count up to 2, fallback up to 2, total max 4.
        # Use collected cats.
        record = dict(_episode_provenance(sequence))
        # Ensure we have at least 2 counts per route if not already
        # If strict_gen_count <2, set to 2? But we already incremented per attempt.
        # For excluded, we need to ensure we attempted max: if strict had 1 attempt and fell back, strict_gen_count may be 1? But spec says max across both routes 4, and exclusion after bounded repair means we attempted max.
        # To reflect bounded repair, if strict succeeded partially? No, exclusion means strict failed both attempts or all filtered, so strict_gen_count should be 2.
        # We'll set counts as actually attempted; audit can check totals.
        record.update(
            {
                "sequence_id": sequence_id,
                "status": "completed",
                "subject_status": "EXCLUDED",
                "route_used": "NONE",
                "strict_generation_count": strict_gen_count,
                "fallback_generation_count": fallback_gen_count,
                "strict_failure_categories": strict_failure_cats,
                "fallback_failure_categories": fallback_failure_cats,
                "generation_attempts": strict_gen_count + fallback_gen_count,
                "correction_reason_codes": strict_failure_cats,
                "episodes_returned": 0,
                "episodes_retained": 0,
                "episode_exclusions": [],
                "subject_exclusion_reason": "MODEL_OUTPUT_INVALID_AFTER_BOUNDED_REPAIR",
                "episodes": [],
                "episode_count": 0,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "strict_prompt_version": STRICT_PROMPT_VERSION,
                "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
                "inference_policy_version": INFERENCE_POLICY_VERSION,
                "strict_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
                "fallback_prompt_sha256": FALLBACK_PROMPT_SHA256,
                "inference_policy_sha256": INFERENCE_POLICY_SHA256,
            }
        )
        inference_path = inferences_dir / f"{sequence_id}.json"
        _atomic_write_json(record, inference_path)
        _restrict(inference_path)
        return sequence_id, True, None

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
    # Compute provenance aggregates for audit: subjects_by_route etc
    # Load all records to count
    all_records = []
    for seq in sequences:
        p = inferences_dir / f"{seq['sequence_id']}.json"
        if p.is_file():
            try:
                all_records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    subjects_included = sum(1 for r in all_records if r.get("subject_status") == "INCLUDED")
    subjects_excluded = sum(1 for r in all_records if r.get("subject_status") == "EXCLUDED")
    subjects_by_route = {
        "STRICT": sum(1 for r in all_records if r.get("route_used") == "STRICT"),
        "SIMPLIFIED": sum(1 for r in all_records if r.get("route_used") == "SIMPLIFIED"),
        "NONE": sum(1 for r in all_records if r.get("route_used") == "NONE"),
    }
    strict_total = sum(int(r.get("strict_generation_count", 0)) for r in all_records)
    fallback_total = sum(int(r.get("fallback_generation_count", 0)) for r in all_records)
    episodes_returned = sum(int(r.get("episodes_returned", 0)) for r in all_records)
    episodes_retained = sum(int(r.get("episodes_retained", 0)) for r in all_records)
    episodes_excluded = sum(len(r.get("episode_exclusions", [])) for r in all_records)
    # exclusions by safe reason
    from collections import Counter
    reason_counter = Counter()
    for r in all_records:
        for excl in r.get("episode_exclusions", []):
            for rc in excl.get("reason_codes", []):
                reason_counter[rc] += 1
    return {
        "sequences_total": len(sequences),
        "completed_before_resume": completed,
        "completed_now": len(pending) - len(failures),
        "completed_total": total_completed,
        "failed": failures,
        "complete": total_completed == len(sequences) and not failures,
        "prompt_version": PROMPT_VERSION,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "subjects_processed": len(sequences),
        "subjects_included": subjects_included,
        "subjects_excluded": subjects_excluded,
        "subjects_by_route": subjects_by_route,
        "strict_generation_total": strict_total,
        "fallback_generation_total": fallback_total,
        "episodes_returned": episodes_returned,
        "episodes_retained": episodes_retained,
        "episodes_excluded": episodes_excluded,
        "exclusions_by_reason": dict(reason_counter),
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
    if not clusters and not candidate_ids:
        return {}
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
    if not families and not cluster_ids:
        return {}
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
                # Use consolidation-specific correction for batches/final, else generic
                correction = CONSOLIDATION_CORRECTION_MESSAGE if "batch" in label or "final" in label else SCHEMA_CORRECTION_MESSAGE
                messages = messages + [{"role": "user", "content": correction}]
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
                # Try to repair truncated JSON by closing brackets
                repaired = content.strip()
                # Count open braces/brackets
                open_braces = repaired.count("{") - repaired.count("}")
                open_brackets = repaired.count("[") - repaired.count("]")
                repaired += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
                # Also try to extract from first { to last }
                start, end = repaired.find("{"), repaired.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(repaired[start:end+1])
                    except Exception:
                        parsed = None
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



async def _batch_consolidate_with_split(
    client,
    model,
    *,
    batch_candidates,
    batch_sequences,
    batch_index,
    depth,
    max_tokens,
    seed,
    settings,
    consolidation_dir,
):
    """Try batch consolidation; on failure split deterministically up to depth 2."""
    if depth > 3:
        raise ValueError(f"batch {batch_index}: split depth exceeded")
    batch_candidate_ids = [c["candidate_id"] for c in batch_candidates]
    # Empty batch: no candidates -> produce empty clusters without model call?
    # But spec says every candidate must be assigned exactly once; empty batch should yield empty clusters.
    if not batch_candidates:
        record = {
            "batch_index": batch_index,
            "sequence_ids": batch_sequences,
            "candidate_count": 0,
            "clusters": [],
            "assignment": {},
            "split_depth": depth,
            "split": False,
        }
        # Only write if top-level batch (depth 0) - split halves will be merged into parent.
        return [], {}, record
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
    # Use smaller output budget for batches to stay within 8192 context (input often >6000 tokens)
    batch_max_tokens = 1024
    batch_settings = dict(settings)
    batch_settings["max_tokens"] = batch_max_tokens
    try:
        parsed = await _consolidation_request(
            client,
            model,
            system_prompt=CONSOLIDATION_BATCH_SYSTEM_PROMPT,
            user_payload=user_payload,
            schema=BATCH_SCHEMA,
            max_tokens=batch_max_tokens,
            seed=seed,
            settings=batch_settings,
            label=f"batch {batch_index} depth {depth}",
        )
        errors = validate_consolidation_batch(parsed)
        if errors:
            raise ValueError(f"batch {batch_index}: {'; '.join(errors[:5])}")
        assignment = _check_cluster_assignment(parsed["clusters"], batch_candidate_ids, f"batch {batch_index}")
        record = {
            "batch_index": batch_index,
            "sequence_ids": batch_sequences,
            "candidate_count": len(batch_candidate_ids),
            "clusters": parsed["clusters"],
            "assignment": assignment,
            "split_depth": depth,
            "split": False,
        }
        return parsed["clusters"], assignment, record
    except Exception as exc:
        if depth >= 3:
            # Ultimate fallback: each candidate becomes its own cluster (deterministic, no semantic merging)
            # This preserves assignment coverage without requiring model
            clusters = []
            assignment = {}
            for cand in batch_candidates:
                cid = f"b{batch_index}-c{cand['candidate_id']}"
                # Use candidate's own wording as canonical (already normalized, safe)
                clusters.append({
                    "cluster_id": cid,
                    "canonical_question_tr": cand["question_tr"],
                    "canonical_question_en": cand["question_en"],
                    "member_candidate_ids": [cand["candidate_id"]],
                })
                assignment[cand["candidate_id"]] = cid
            record = {
                "batch_index": batch_index,
                "sequence_ids": batch_sequences,
                "candidate_count": len(batch_candidate_ids),
                "clusters": clusters,
                "assignment": assignment,
                "split_depth": depth,
                "split": depth > 0,
                "fallback_trivial": True,
                "original_error": str(exc),
            }
            return clusters, assignment, record
        # Split deterministically in half
        # Split deterministically in half
        sorted_candidates = sorted(batch_candidates, key=lambda c: c["candidate_id"])
        mid = len(sorted_candidates) // 2
        if mid == 0 or mid == len(sorted_candidates):
            raise
        left_candidates = sorted_candidates[:mid]
        right_candidates = sorted_candidates[mid:]
        # For split, we keep same batch_index but process halves sequentially with depth+1
        # Need to generate distinct sub-records but final combined will have same batch_index.
        left_clusters, left_assign, left_record = await _batch_consolidate_with_split(
            client, model, batch_candidates=left_candidates, batch_sequences=batch_sequences,
            batch_index=batch_index, depth=depth+1, max_tokens=max_tokens, seed=seed,
            settings=settings, consolidation_dir=consolidation_dir
        )
        right_clusters, right_assign, right_record = await _batch_consolidate_with_split(
            client, model, batch_candidates=right_candidates, batch_sequences=batch_sequences,
            batch_index=batch_index, depth=depth+1, max_tokens=max_tokens, seed=seed,
            settings=settings, consolidation_dir=consolidation_dir
        )
        # Merge clusters and assignments, ensure no duplicate cluster_id
        combined_clusters = left_clusters + right_clusters
        combined_assignment = {**left_assign, **right_assign}
        # Validate combined assignment covers all
        _check_cluster_assignment(combined_clusters, batch_candidate_ids, f"batch {batch_index} merged")
        record = {
            "batch_index": batch_index,
            "sequence_ids": batch_sequences,
            "candidate_count": len(batch_candidate_ids),
            "clusters": combined_clusters,
            "assignment": combined_assignment,
            "split_depth": depth,
            "split": True,
            "left": left_record,
            "right": right_record,
            "original_error": str(exc),
        }
        return combined_clusters, combined_assignment, record


async def _final_consolidate_with_fallback(
    client,
    model,
    cluster_summaries,
    all_cluster_ids,
    max_tokens,
    seed,
    settings,
):
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
    SIMPLIFIED_SCHEMA = {
        "type": "object",
        "properties": {
            "families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_tr": {"type": "string"},
                        "question_en": {"type": "string"},
                        "member_cluster_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question_tr", "question_en", "member_cluster_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["families"],
        "additionalProperties": False,
    }
    # First try normal schema with correction (via _consolidation_request which already has one correction)
    try:
        parsed = await _consolidation_request(
            client, model,
            system_prompt=CONSOLIDATION_FINAL_SYSTEM_PROMPT,
            user_payload={"cluster_summaries": cluster_summaries},
            schema=FINAL_SCHEMA,
            max_tokens=max_tokens,
            seed=seed,
            settings=settings,
            label="final merge",
        )
        errors = validate_consolidation_final(parsed)
        if errors:
            raise ValueError(f"final merge: {'; '.join(errors[:5])}")
        family_assignment = _check_family_assignment(parsed["families"], all_cluster_ids)
        return parsed, family_assignment, False
    except Exception as exc:
        # Retry once using simplified schema containing only question_tr, question_en, member_cluster_ids
        # Python attaches deterministic family IDs in list order
        try:
            parsed_simple = await _consolidation_request(
                client, model,
                system_prompt=CONSOLIDATION_FINAL_SYSTEM_PROMPT + " Use the simplified schema with question_tr, question_en, member_cluster_ids only; do not add family_id.",
                user_payload={"cluster_summaries": cluster_summaries},
                schema=SIMPLIFIED_SCHEMA,
                max_tokens=max_tokens,
                seed=seed,
                settings=settings,
                label="final merge simplified",
            )
            errors = validate_consolidation_final_simplified(parsed_simple)
            if errors:
                raise ValueError(f"final merge simplified: {'; '.join(errors[:5])}")
            # Attach deterministic family IDs in list order
            families = []
            for idx, fam in enumerate(parsed_simple["families"], start=1):
                families.append({
                    "family_id": f"f{idx}",
                    "question_tr": fam["question_tr"],
                    "question_en": fam["question_en"],
                    "member_cluster_ids": fam["member_cluster_ids"],
                })
            family_assignment = _check_family_assignment(families, all_cluster_ids)
            return {"families": families, "cluster_assignment": family_assignment}, family_assignment, True
        except Exception as exc2:
            raise ValueError(f"final merge failed after fallback: {exc} | simplified: {exc2}") from exc2


def consolidate(
    inferences_dir: str | Path,
    consolidation_dir: str | Path,
    *,
    base_url: str,
    model: str = SERVED_MODEL,
    seed: int = 42,
    max_tokens: int = TURKISH_MAX_TOKENS,
) -> dict[str, Any]:
    """Two-level consolidation with bounded recovery ladder."""
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
                clusters, assignment, record = await _batch_consolidate_with_split(
                    client, model,
                    batch_candidates=batch_candidates,
                    batch_sequences=batch_sequences,
                    batch_index=batch_index,
                    depth=0,
                    max_tokens=max_tokens,
                    seed=seed,
                    settings=settings,
                    consolidation_dir=consolidation_dir,
                )
                batch_results.append(record)
                all_cluster_ids.extend(cluster["cluster_id"] for cluster in clusters)
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

            final_parsed, family_assignment, used_simplified = await _final_consolidate_with_fallback(
                client, model, cluster_summaries, all_cluster_ids, max_tokens, seed, settings
            )
            # final_parsed may already have families
            families = final_parsed["families"] if "families" in final_parsed else final_parsed.get("families")
            # Ensure we have families list
            if isinstance(final_parsed, dict) and "families" not in final_parsed:
                families = final_parsed.get("families")
            else:
                families = final_parsed["families"]
            final_record = {
                "families": families,
                "cluster_assignment": family_assignment,
                "cluster_count": len(all_cluster_ids),
                "candidate_count": len(candidate_ids_all),
                "used_simplified_fallback": used_simplified,
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

def _consolidation_request_with_correction(
    client,
    model,
    system_prompt,
    user_payload,
    schema,
    max_tokens,
    seed,
    settings,
    label,
):
    """Wrapper that maps to existing _consolidation_request with one correction."""
    return  # placeholder, will be replaced via direct call
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


def aggregate_episode_exclusion_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Return only safe aggregate exclusion counts for compact provenance."""
    counts = Counter()
    for record in records:
        for exclusion in record.get("episode_exclusions", []):
            for reason_code in exclusion.get("reason_codes", []):
                if reason_code in EPISODE_SAFETY_REASON_CODES:
                    counts[reason_code] += 1
    return {reason_code: int(counts.get(reason_code, 0)) for reason_code in EPISODE_SAFETY_REASON_CODES}


def render_tables(
    rows: list[dict[str, Any]],
    *,
    run_dir: str | Path,
    deployment_id: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    source_commit: str = "",
    turkish_run_id: str | None = None,
    analysis_attempt: int | None = None,
    prompt_contract_hash: str | None = None,
    run_manifest_sha256: str | None = None,
    selection_file_sha256: str | None = None,
    episode_exclusion_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write the compact CSV, JSON, and Markdown tables atomically."""
    table_rows = []
    for row in rows:
        table_row = {column: row[column] for column in FINAL_TABLE_COLUMNS}
        table_rows.append(table_row)
        for column in ("question_tr", "question_en", "evidence_basis"):
            text = table_row[column]
            for marker in ("[WINDOW", "[", "]", "subject_id", "filename") + tuple(_EVIDENCE_QUOTES):
                if marker in text:
                    raise ValueError(f"render refused: {column} contains forbidden marker {marker!r}")
            if _CANONICAL_SEQUENCE_TOKEN_RE.search(text):
                raise ValueError(
                    f"render refused: {column} contains a sequence identifier marker"
                )

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
        "turkish_run_id": turkish_run_id,
        "analysis_attempt": analysis_attempt,
        "model_id": model_id,
        "model_revision": model_revision,
        "source_commit": source_commit,
        "prompt_version": PROMPT_VERSION,
        "strict_prompt_version": STRICT_PROMPT_VERSION,
        "fallback_prompt_version": FALLBACK_PROMPT_VERSION,
        "inference_policy_version": INFERENCE_POLICY_VERSION,
        "inference_policy_sha256": INFERENCE_POLICY_SHA256,
        "episode_safety_policy_version": EPISODE_SAFETY_POLICY_VERSION,
        "episode_safety_policy_sha256": EPISODE_SAFETY_POLICY_SHA256,
        "prompt_contract_sha256": prompt_contract_hash,
        "generation_settings_hash": generation_settings_hash(TURKISH_MAX_TOKENS),
        "run_manifest_sha256": run_manifest_sha256,
        "selection_file_sha256": selection_file_sha256,
        "episode_exclusion_counts": episode_exclusion_counts
        or {reason_code: 0 for reason_code in EPISODE_SAFETY_REASON_CODES},
        "rows": table_rows,
    }
    json_path = run_dir / "turkish_inferred_questions.json"
    _atomic_write_json(payload, json_path)

    lines = [
        "# Turkish inferred recurring question families",
        "",
        f"Deployment: `{deployment_id}` — model `{model_id}` revision `{model_revision}`.",
        f"Prompt policy: `{PROMPT_VERSION}` / `{EPISODE_SAFETY_POLICY_VERSION}`.",
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
