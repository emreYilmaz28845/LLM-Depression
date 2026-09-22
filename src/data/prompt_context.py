"""Central, versioned prompt context for the depression-classification prompts.

One prompt version owns three things: the shared system instruction, one short
source-grounded recording-context block per dataset, and the Turkish pooled
question-context sentences. Dataset configs select the version and their context
key instead of carrying independent copies of the text.

A config without ``prompt.version`` keeps its own inline ``prompt.system`` and
renders byte-identically to before, and a question-conditioned config without
``prompt.question_context_version`` keeps the legacy sentence wording.

Prompt text is taken from ``docs/PROMPT_PROPOSAL_DEPRESSION_20260920.md``. The
audio-bearing modalities keep the approved wording verbatim. Text-only inputs
carry no audio, so the text-only rendering drops the audio-availability claims
instead of describing an input the model does not receive.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.utils import (
    INPUT_MODALITY_TEXT_ONLY,
    resolve_input_modality,
)

PROMPT_CONTEXT_VERSION = "promptcontext_v1"
DEFAULT_QUESTION_CONTEXT_SENTENCE_SET = "legacy_v1"

# The approved shared instruction, used for every modality that is not listed
# with its own wording below (audio-only and audio+text keep it verbatim).
SHARED_INSTRUCTION = (
    "You are classifying a participant's depression study label from the provided "
    "speech audio and/or transcript. Use the recording context below to interpret "
    "the input. Possible relevant signs include persistent low mood, loss of "
    "interest, hopelessness, and changes in sleep or energy. A question's topic, a "
    "single phrase, or one vocal feature is not conclusive. The label belongs to "
    "the participant, even when the input is only one recording or audio window. "
    "Make a research label prediction, not a clinical diagnosis."
)

# Text-only variant: same instruction and the same caution about a single topic or
# phrase, with no claim that the example contains audio and no audio-only cue.
SHARED_INSTRUCTION_TEXT_ONLY = (
    "You are classifying a participant's depression study label from the provided "
    "transcript. Use the recording context below to interpret the input. Possible "
    "relevant signs include persistent low mood, loss of interest, hopelessness, and "
    "changes in sleep or energy. A question's topic or a single phrase is not "
    "conclusive. The label belongs to the participant, even when the input is only "
    "one session. Make a research label prediction, not a clinical diagnosis."
)

ANDROIDS_CONTEXT = (
    "Recording context: This is an Italian interview with a human interviewer. "
    "Questions mainly concern everyday life, including family, work, recent "
    "activities, and hobbies. It is not a systematic symptom questionnaire. A "
    "neutral or cheerful topic does not determine the participant's study label. "
    "The label comes from a psychiatrist's DSM-5 diagnosis."
)

D3TEC_CONTEXT = (
    "Recording context: This is Spanish speech from a non-interactive slideshow of "
    "27 tasks. Tasks include open answers, reading words or passages, and describing "
    "images with different emotional content. Some emotion in a response may come "
    "from the assigned task. The study label uses PHQ-9 ≥ 10."
)

DAIC_CONTEXT = (
    "Recording context: This is an English semi-structured interview with the "
    "virtual interviewer Ellie. It mixes everyday conversation with questions about "
    "mood, sleep, diagnosis, and treatment. The audio is a short window of "
    "participant speech; if a transcript is provided, it can cover more of the "
    "participant's session than the audio. The study uses a PHQ-8 binary label "
    "associated with a threshold of 10."
)

# Text-only DAIC variant: the harmonized text-only input is the participant's
# transcript, so the sentence about the audio window is dropped.
DAIC_CONTEXT_TEXT_ONLY = (
    "Recording context: This is an English semi-structured interview with the "
    "virtual interviewer Ellie. It mixes everyday conversation with questions about "
    "mood, sleep, diagnosis, and treatment. The study uses a PHQ-8 binary label "
    "associated with a threshold of 10."
)

CMDC_CONTEXT = (
    "Recording context: This is Mandarin speech from a face-to-face, "
    "symptom-focused interview. Its topics include mood, sleep, appetite, energy, "
    "concentration, worries, self-harm, and changes in movement or speech. The topic "
    "of a question is not itself evidence that the participant has depression. The "
    "study label is a clinical MDD group under DSM-IV, confirmed with MINI."
)

TURKISH_POOLED_CONTEXT = (
    "Recording context: This is Turkish speech from one of two question sets "
    "completed by the same participants. The sets concern positive or negative "
    "material. The exact question text is not available in this input. The "
    "participant has one study label across both sets. The study label is BDI ≥ 17."
)

# ``default`` is the wording used unless the resolved input modality overrides it.
DATASET_CONTEXT_BLOCKS: dict[str, dict[str, dict[str, str]]] = {
    PROMPT_CONTEXT_VERSION: {
        "androids": {"default": ANDROIDS_CONTEXT},
        "d3tec": {"default": D3TEC_CONTEXT},
        "daic": {
            "default": DAIC_CONTEXT,
            INPUT_MODALITY_TEXT_ONLY: DAIC_CONTEXT_TEXT_ONLY,
        },
        "cmdc": {"default": CMDC_CONTEXT},
        "turkish_pooled": {"default": TURKISH_POOLED_CONTEXT},
    }
}

# The shared instruction is version-wide, so the modality override lives beside it
# rather than inside a dataset block.
SHARED_INSTRUCTION_OVERRIDES: dict[str, dict[str, str]] = {
    PROMPT_CONTEXT_VERSION: {INPUT_MODALITY_TEXT_ONLY: SHARED_INSTRUCTION_TEXT_ONLY}
}

LEGACY_QUESTION_CONTEXT_SENTENCES: dict[str, str] = {
    "pos_only_t17": "The following speech is the subject's response to positive interview questions.",
    "negative_only_t17": "The following speech is the subject's response to negative interview questions.",
}

PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES: dict[str, str] = {
    "pos_only_t17": (
        "Positive set: This recording answers a question from the positive set. "
        "The question's positive framing is not the participant's depression label."
    ),
    "negative_only_t17": (
        "Negative set: This recording answers a question from the negative set. "
        "The question's negative framing is not the participant's depression label."
    ),
}

QUESTION_CONTEXT_SENTENCE_SETS: dict[str, dict[str, str]] = {
    DEFAULT_QUESTION_CONTEXT_SENTENCE_SET: LEGACY_QUESTION_CONTEXT_SENTENCES,
    PROMPT_CONTEXT_VERSION: PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES,
}

DATASET_CONTEXT_KEYS = tuple(sorted(DATASET_CONTEXT_BLOCKS[PROMPT_CONTEXT_VERSION]))


def _prompt_cfg(config: dict[str, Any]) -> dict[str, Any]:
    prompt_cfg = config.get("prompt", {}) or {}
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt must be a mapping.")
    return prompt_cfg


def resolve_prompt_context_version(config: dict[str, Any]) -> str | None:
    """Return the selected prompt version, or ``None`` for an inline config."""
    raw_value = _prompt_cfg(config).get("version")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    version = str(raw_value).strip()
    if version not in DATASET_CONTEXT_BLOCKS:
        raise ValueError(
            f"Unsupported prompt.version={raw_value!r}. "
            f"Expected one of {sorted(DATASET_CONTEXT_BLOCKS)}."
        )
    return version


def resolve_dataset_context_key(config: dict[str, Any]) -> str:
    raw_value = _prompt_cfg(config).get("dataset_context")
    key = str(raw_value).strip() if raw_value is not None else ""
    if not key:
        raise ValueError(
            "prompt.dataset_context is required when prompt.version is set. "
            f"Expected one of {list(DATASET_CONTEXT_KEYS)}."
        )
    version = resolve_prompt_context_version(config)
    blocks = DATASET_CONTEXT_BLOCKS[version]
    if key not in blocks:
        raise ValueError(
            f"Unsupported prompt.dataset_context={raw_value!r} for prompt.version="
            f"{version!r}. Expected one of {sorted(blocks)}."
        )
    return key


def _pick(mapping: dict[str, str], modality: str) -> str:
    if modality in mapping:
        return mapping[modality]
    if "default" in mapping:
        return mapping["default"]
    raise ValueError(
        f"No prompt wording for input modality {modality!r}; "
        f"available: {sorted(mapping)}."
    )


def resolve_system_prompt(config: dict[str, Any]) -> str:
    """Resolve the system prompt for a config, fail-closed on unknown selection."""
    version = resolve_prompt_context_version(config)
    if version is None:
        inline = str(_prompt_cfg(config).get("system", "")).strip()
        if not inline:
            raise ValueError(
                "prompt.system is required when prompt.version is absent."
            )
        return inline
    modality = resolve_input_modality(config)
    key = resolve_dataset_context_key(config)
    instruction = _pick(SHARED_INSTRUCTION_OVERRIDES.get(version, {}), modality)
    context = _pick(DATASET_CONTEXT_BLOCKS[version][key], modality)
    return f"{instruction}\n\n{context}"


def resolve_question_context_sentences(config: dict[str, Any]) -> dict[str, str]:
    """Resolve the pooled question-context sentences for a config."""
    raw_version = _prompt_cfg(config).get("question_context_version")
    if raw_version is None or str(raw_version).strip() == "":
        version = DEFAULT_QUESTION_CONTEXT_SENTENCE_SET
    else:
        version = str(raw_version).strip()
    if version not in QUESTION_CONTEXT_SENTENCE_SETS:
        raise ValueError(
            f"Unsupported prompt.question_context_version={raw_version!r}. "
            f"Expected one of {sorted(QUESTION_CONTEXT_SENTENCE_SETS)}."
        )
    return QUESTION_CONTEXT_SENTENCE_SETS[version]


def prompt_context_record(config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic provenance record of the prompt a run renders.

    Written into ``run_config.yaml`` and ``eval_config.yaml`` so the evidence
    carries the exact system prompt next to its hash.
    """
    version = resolve_prompt_context_version(config)
    system_prompt = resolve_system_prompt(config)
    question_context_version = str(
        _prompt_cfg(config).get("question_context_version")
        or DEFAULT_QUESTION_CONTEXT_SENTENCE_SET
    ).strip()
    return {
        "version": version,
        "dataset_context": (
            resolve_dataset_context_key(config) if version is not None else None
        ),
        "question_context_version": question_context_version,
        "input_modality": resolve_input_modality(config),
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
    }
