"""Prompt definitions for the clinical translation pipeline (clinical_faithful_v1).

The instruction requires translating only the marked target passage, using the
context solely to resolve ambiguity, preserving exact meaning without
summarizing/diagnosing/censoring/interpreting, avoiding ASR repair, and
returning only schema-valid JSON.
"""
from __future__ import annotations

from typing import Any

PROMPT_VERSION = "clinical_faithful_v1"

SYSTEM_PROMPT = (
    "You are a professional clinical transcript translator. Translate the marked "
    "target passage below into English.\n"
    "Requirements:\n"
    "- Translate only the passage marked <target>...</target>. Use the "
    "<context>...</context> block solely to resolve ambiguity; never translate "
    "the context.\n"
    "- Preserve exact meaning. Do not summarize, diagnose, censor, or interpret.\n"
    "- Do not repair or fill suspected speech-recognition errors; keep disfluencies, "
    "repetitions, and unfinished thoughts.\n"
    "- Preserve negation, uncertainty, speaker attribution, repetition, unfinished "
    "thoughts, symptom severity, frequency, duration, dates, numbers, names, "
    "medications, and self-harm language.\n"
    "- Do not add explanations or commentary. Respond with exactly one JSON object: "
    '{"translation": "the English translation only"}.'
)

_CORRECTIVE_SUFFIX = (
    "\nYour previous response was not valid JSON. Respond again with exactly one "
    'JSON object: {"translation": "the English translation only"}.'
)


def user_prompt(unit: dict[str, Any], corrective: bool = False) -> str:
    parts: list[str] = []
    context_text = str(unit.get("context_text", "")).strip()
    if context_text:
        parts.append(f"<context>\n{context_text}\n</context>")
    parts.append(f"<target>\n{unit['source_text']}\n</target>")
    parts.append('Translate the target passage into English and return JSON: {"translation": "..."}.')
    if corrective:
        parts.append(_CORRECTIVE_SUFFIX)
    return "\n".join(parts)


def verifier_prompt(unit: dict[str, Any], translation: str) -> str:
    return (
        "You are a clinical translation verifier. Compare the source passage with "
        "its English translation.\n"
        f"Source:\n{unit['source_text']}\n\n"
        f"English translation:\n{translation}\n\n"
        "List every semantic invariant that is missing from the translation or "
        "added to it, restricted to: negation, uncertainty, numbers, dates, "
        "durations, names, medications, symptom severity/frequency, and "
        "self-harm language.\n"
        'Respond with exactly one JSON object: {"missing": [...], "added": [...]}. '
        "Use empty lists when the translation is faithful."
    )
