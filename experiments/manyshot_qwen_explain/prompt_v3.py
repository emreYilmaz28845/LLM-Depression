"""Advisor many-shot prompt (v3_advisor): numbered labelled examples, explanation before the label.

Cenk Hoca, 2026-11-23 (WhatsApp): the prompt says, as is, "Base your answer on the examples in the
prompt and explain your answer". The model is NOT required to name the 2-3 most similar examples
(he: "Benzeri bulmasi zor"), but examples stay numbered so it can cite them. Only DAIC (English) and
Turkish. The dataset context sentences follow ``src/data/prompt_context.py`` (promptcontext_v1).

Shared by the Qwen3.8 runner and, if wanted, the Claude runner, so both see the same text.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "v3_advisor"
ADVISOR_SENTENCE = "Base your answer on the examples in the prompt and explain your answer."
LABEL_TEXT = {1: "Depressed", 0: "Non-depressed"}
TEXT_LABEL = {v.lower(): k for k, v in LABEL_TEXT.items()}

DATASET_CONTEXT = {
    "daic": (
        "Each transcript contains only the participant's side of an English semi-structured clinical "
        "interview with the virtual interviewer Ellie (DAIC-WOZ). It mixes everyday conversation with "
        "questions about mood, sleep, diagnosis, and treatment. Labels come from the PHQ-8 "
        "questionnaire: Depressed means PHQ-8 >= 10."
    ),
    "turkish": (
        "Each transcript contains a participant's spoken answers, in Turkish, to two question sets: a "
        "positive set and a negative set. The sets concern positive or negative material; the framing "
        "of a set is not the participant's label. The exact question text is not available. Each "
        "participant has one label across both sets. Labels come from the Beck Depression Inventory: "
        "Depressed means BDI >= 17."
    ),
}


def system_prompt(dataset: str) -> str:
    """Instructions only (the shot block is appended after them)."""
    return (
        "You are assisting a research study on depression screening from interview transcripts. "
        + DATASET_CONTEXT[dataset] + "\n\n"
        "You will first see numbered labelled examples: transcripts of other participants of this "
        "study, each followed by that participant's true label. Then classify the final participant.\n\n"
        + ADVISOR_SENTENCE + "\n\n"
        "Answer with exactly one JSON object and nothing else, with the explanation first:\n"
        '{"explanation": "<3-6 sentences>", "label": "Depressed" or "Non-depressed"}\n'
        "You may refer to the labelled examples by their number."
    )


def shot_block(shots: list[dict]) -> str:
    parts = ["LABELLED EXAMPLES"]
    for i, s in enumerate(shots, 1):
        parts.append(f"### Example {i}\nTranscript:\n{s['text']}\nLabel: {LABEL_TEXT[s['label']]}")
    return "\n\n".join(parts)


def system_with_shots(dataset: str, shots: list[dict]) -> str:
    return system_prompt(dataset) + "\n\n" + shot_block(shots)


def target_block(target: dict) -> str:
    return ("### Final participant (classify this one)\nTranscript:\n" + target["text"]
            + "\n\nReturn the JSON object now.")


def parse_answer(text: str) -> tuple[int | None, str]:
    """Label from the last JSON object with a valid label; None = INVALID (scored as wrong)."""
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        label = TEXT_LABEL.get(str(obj.get("label", "")).strip().lower())
        if label is not None:
            return label, str(obj.get("explanation", "")).strip()
    return None, ""


def cited_examples(explanation: str) -> list[int]:
    """Example numbers named in an explanation ('Example 12', 'Examples 3 and 7', 'Example #5')."""
    found: list[int] = []
    for m in re.finditer(r"Examples?\s*#?\s*((?:\d+)(?:\s*(?:,|and|&|or)\s*#?\s*\d+)*)", explanation,
                         flags=re.I):
        for n in re.findall(r"\d+", m.group(1)):
            if int(n) not in found:
                found.append(int(n))
    return found


LABEL_VALUE_START = re.compile(r'"label"\s*:\s*"')
