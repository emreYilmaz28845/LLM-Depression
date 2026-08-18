"""Fixed Qwen3.8 deployment contracts: pins, enums, schemas, parsing, rules.

Pure stdlib module. Everything here is deterministic and unit-testable:
pinned versions, request settings, the strict response schemas, filename-stem
parsing for the Turkish source windows, text normalization used for
determinism and privacy checks, and the serving-configuration selection rules
from runbook section 17.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# Fixed pins (runbook section 2)
# --------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL = "qwen3.8-27b"
PORT = 8000

PYTHON_MAJOR = 3
PYTHON_MINOR = 10
VLLM_VERSION = "0.25.1"
TRANSFORMERS_VERSION = "5.8.0"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
OPENAI_VERSION = "3.2.0"
HUGGINGFACE_HUB_VERSION = "1.28.0"

MIN_DRIVER_VERSION = 580.00
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
SERVER_STARTUP_TIMEOUT_SECONDS = 600
SERVER_READY_POLL_SECONDS = 2

VALIDATION_MAX_TOKENS = 1024
TURKISH_MAX_TOKENS = 2048
TEMPERATURE = 0
TOP_P = 1
SEED = 42
ENABLE_THINKING = False
PRESERVE_THINKING = False

CONCURRENCY_LEVELS = (1, 8, 16, 32)
SYNTHETIC_CASE_COUNT = 64
SYNTHETIC_LANGUAGES = ("tr", "en")
SYNTHETIC_LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED")
RESTART_SUBSET_SIZE = 8
TURKISH_INFERENCE_CONCURRENCY = 8

# Section 17 projection: 135 subject sequences + 5 batch consolidations +
# 1 final merge + 1 render/audit allowance.
TURKISH_PROJECTED_REQUESTS = 142
TURKISH_WALL_LIMIT_SECONDS = 2 * 3600
TURKISH_TP2_UPPER_LIMIT_SECONDS = 4 * 3600
SAFETY_FACTOR = 1.25

# Turkish source facts (runbook sections 8 and 18)
TURKISH_SOURCE_HASH = "b7e3a64af3df2d9aa490fb4f321d0a76892c9875b0ba7250437aca3d151649eb"
TURKISH_EXPECTED_WINDOWS = 1186
TURKISH_EXPECTED_SEQUENCES = 135
TURKISH_CONSOLIDATION_BATCHES = (32, 32, 32, 32, 7)

CONDITION_TAGS = ("ank", "depr", "depr+ank", "ank+depr", "dep+ank")

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Label(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class WordingStatus(str, Enum):
    EXPLICIT_ECHO = "EXPLICIT_ECHO"
    INFERRED_PARAPHRASE = "INFERRED_PARAPHRASE"


CONFIDENCE_WEIGHTS = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
FINAL_TABLE_COLUMNS = (
    "order",
    "question_tr",
    "question_en",
    "label",
    "wording_status",
    "confidence",
    "supporting_subjects",
    "evidence_basis",
)

# --------------------------------------------------------------------------
# Filename-stem parsing (runbook section 18)
# --------------------------------------------------------------------------

STEM_PATTERN = r"^(?P<subject>.+)-1-(?P<window>[0-9]+)-(?P<condition>ank|depr|depr\+ank|ank\+depr|dep\+ank)$"
STEM_RE = re.compile(STEM_PATTERN)


@dataclass(frozen=True)
class FilenameParts:
    subject: str
    window: int
    condition: str


def parse_filename_stem(stem: str) -> FilenameParts | None:
    """Parse ``<subject>-1-<window>-<condition>``.

    Returns None for any stem that does not match exactly, including stems
    with a non-canonical condition tag or a non-numeric window field.
    """
    match = STEM_RE.fullmatch(stem.strip())
    if match is None:
        return None
    parts = match.groupdict()
    if parts["condition"] not in CONDITION_TAGS:
        return None
    return FilenameParts(
        subject=parts["subject"],
        window=int(parts["window"]),
        condition=parts["condition"],
    )


# --------------------------------------------------------------------------
# Text normalization and privacy helpers
# --------------------------------------------------------------------------

_PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_for_determinism(text: str) -> str:
    """Casefold, collapse whitespace, and strip punctuation."""
    folded = _PUNCTUATION_RE.sub(" ", text.casefold())
    return _WS_RE.sub(" ", folded).strip()


def tokenize_for_privacy(text: str) -> list[str]:
    """Tokenize after casefolding, whitespace collapse, punctuation removal."""
    folded = _PUNCTUATION_RE.sub(" ", text.casefold())
    return [tok for tok in _WS_RE.sub(" ", folded).strip().split(" ") if tok]


def ngram_overlap_at_least(text: str, transcript_windows: Sequence[str], n: int = 12) -> bool:
    """True if any contiguous n-token sequence of ``text`` appears in a window.

    The comparison uses the same normalization the audit applies: Unicode
    case folding, whitespace collapse, and punctuation removal. Shorter
    overlap is allowed by design.
    """
    tokens = tokenize_for_privacy(text)
    if len(tokens) < n:
        return False
    text_ngrams: set[tuple[str, ...]] = set()
    for start in range(0, len(tokens) - n + 1):
        text_ngrams.add(tuple(tokens[start : start + n]))
    for window in transcript_windows:
        window_tokens = tokenize_for_privacy(window)
        if len(window_tokens) < n:
            continue
        window_ngrams: set[tuple[str, ...]] = set()
        for start in range(0, len(window_tokens) - n + 1):
            window_ngrams.add(tuple(window_tokens[start : start + n]))
        if text_ngrams & window_ngrams:
            return True
    return False


# --------------------------------------------------------------------------
# Request settings (runbook section 11.2)
# --------------------------------------------------------------------------


def request_settings(max_tokens: int) -> dict[str, Any]:
    """Fixed request parameters every validation/Turkish request must use."""
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": ENABLE_THINKING,
            "preserve_thinking": PRESERVE_THINKING,
        },
    }


def generation_settings_hash(max_tokens: int) -> str:
    from hashlib import sha256

    return sha256(
        json.dumps(request_settings(max_tokens), sort_keys=True).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------
# Strict response schemas
# --------------------------------------------------------------------------

VALIDATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "inferred_question": {"type": "string"},
        "label": {"type": "string", "enum": list(SYNTHETIC_LABELS)},
        "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
    },
    "required": ["case_id", "inferred_question", "label", "confidence"],
    "additionalProperties": False,
}

EPISODE_SCHEMA = {
    "type": "object",
    "properties": {
        "sequence_id": {"type": "string"},
        "episode_order": {"type": "integer", "minimum": 1},
        "question_tr": {"type": "string"},
        "question_en": {"type": "string"},
        "label": {"type": "string", "enum": [l.value for l in Label]},
        "wording_status": {"type": "string", "enum": [w.value for w in WordingStatus]},
        "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
        "evidence_window_indices": {"type": "array", "items": {"type": "integer"}},
        "evidence_basis": {"type": "string"},
        "abstain_reason": {"type": "string"},
    },
    "required": [
        "sequence_id",
        "episode_order",
        "question_tr",
        "question_en",
        "label",
        "wording_status",
        "confidence",
        "evidence_window_indices",
        "evidence_basis",
        "abstain_reason",
    ],
    "additionalProperties": False,
}

SUBJECT_INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "episodes": {"type": "array", "items": EPISODE_SCHEMA},
    },
    "required": ["episodes"],
    "additionalProperties": False,
}

CONSOLIDATION_CLUSTER_SCHEMA = {
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
}

CONSOLIDATION_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {"type": "array", "items": CONSOLIDATION_CLUSTER_SCHEMA},
    },
    "required": ["clusters"],
    "additionalProperties": False,
}

FAMILY_SCHEMA = {
    "type": "object",
    "properties": {
        "family_id": {"type": "string"},
        "question_tr": {"type": "string"},
        "question_en": {"type": "string"},
        "member_cluster_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["family_id", "question_tr", "question_en", "member_cluster_ids"],
    "additionalProperties": False,
}

CONSOLIDATION_FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "families": {"type": "array", "items": FAMILY_SCHEMA},
    },
    "required": ["families"],
    "additionalProperties": False,
}


def structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI JSON-schema response format payload for vLLM 0.25.1."""
    return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}


def _type_errors(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{path}: expected object, got {type(value).__name__}"]
    for prop in schema.get("required", []):
        if prop not in value:
            errors.append(f"{path}: missing required field {prop!r}")
            continue
        field_schema = schema["properties"].get(prop, {})
        errors.extend(_check_field(value[prop], field_schema, f"{path}.{prop}"))
    for key in value:
        if key not in schema.get("properties", {}):
            errors.append(f"{path}: unknown field {key!r}")
    return errors


def _check_field(value: Any, field_schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = field_schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
        elif "enum" in field_schema and value not in field_schema["enum"]:
            errors.append(f"{path}: invalid enum value {value!r}")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
        else:
            minimum = field_schema.get("minimum")
            if minimum is not None and value < minimum:
                errors.append(f"{path}: value {value} below minimum {minimum}")
    elif expected == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
        else:
            item_schema = field_schema.get("items", {})
            for index, item in enumerate(value):
                errors.extend(_check_field(item, item_schema, f"{path}[{index}]"))
    elif expected == "object":
        errors.extend(_type_errors(value, field_schema, path))
    return errors


def validate_validation_response(value: Any) -> list[str]:
    """Strictly validate one validation-case response object."""
    return _type_errors(value, VALIDATION_RESPONSE_SCHEMA, "response")


def validate_subject_inference(value: Any) -> list[str]:
    """Strictly validate a subject-inference response object."""
    return _type_errors(value, SUBJECT_INFERENCE_SCHEMA, "response")


def validate_consolidation_batch(value: Any) -> list[str]:
    return _type_errors(value, CONSOLIDATION_BATCH_SCHEMA, "response")


def validate_consolidation_final(value: Any) -> list[str]:
    return _type_errors(value, CONSOLIDATION_FINAL_SCHEMA, "response")


# --------------------------------------------------------------------------
# Serving-configuration selection (runbook section 17)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    tp: int
    passed: bool
    request_rate_c1: float | None = None
    request_rate_c8: float | None = None
    metrics_path: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    selected_tp: int | None
    decision_rule: str
    candidate_results: dict[int, CandidateResult] = field(default_factory=dict)
    projected_requests: int = TURKISH_PROJECTED_REQUESTS
    projected_wall_seconds: dict[int, float] = field(default_factory=dict)


def project_turkish_wall_seconds(rate_c1: float, rate_c8: float) -> float:
    """Projected Turkish wall time: slower rate, safety factor 1.25."""
    if rate_c1 is None or rate_c8 is None or rate_c1 <= 0 or rate_c8 <= 0:
        raise ValueError("request rates must be positive")
    slower = min(rate_c1, rate_c8)
    return TURKISH_PROJECTED_REQUESTS / slower * SAFETY_FACTOR


def _projected(candidate: CandidateResult) -> float | None:
    if not candidate.passed or candidate.request_rate_c1 is None or candidate.request_rate_c8 is None:
        return None
    return project_turkish_wall_seconds(candidate.request_rate_c1, candidate.request_rate_c8)


def select_serving_configuration(candidates: Mapping[int, CandidateResult]) -> SelectionResult:
    """Apply runbook section 17 rules in their exact order.

    Rules:
    1. TP=2 must have passed; otherwise no configuration is selected.
    2. TP=1 is eligible only when it passed and its projection is <= 2 hours.
    3. An eligible TP=1 is selected (minimizes GPU-hours).
    4. TP=4 may replace TP=2 only when TP=4 passed, TP=2 projects > 4 hours,
       and TP=4 is at least 30% faster than TP=2.
    5. Otherwise select TP=2.
    6. If eligible configurations differ by <= 10% in projected wall time,
       select the one with fewer GPUs.
    """
    candidates = dict(candidates)
    tp2 = candidates.get(2)
    if tp2 is None or not tp2.passed:
        return SelectionResult(
            selected_tp=None,
            decision_rule="rule1_tp2_not_passed_no_selection",
            candidate_results=candidates,
        )

    projections = {tp: _projected(cand) for tp, cand in candidates.items()}
    tp1 = candidates.get(1)
    tp1_eligible = bool(
        tp1 is not None
        and tp1.passed
        and projections.get(1) is not None
        and projections[1] <= TURKISH_WALL_LIMIT_SECONDS
    )
    if tp1_eligible:
        selected, rule = 1, "rule3_select_tp1_eligible_within_2h"
    else:
        tp4 = candidates.get(4)
        tp2_proj = projections.get(2)
        tp4_proj = projections.get(4)
        tp4_replaces = bool(
            tp4 is not None
            and tp4.passed
            and tp2_proj is not None
            and tp4_proj is not None
            and tp2_proj > TURKISH_TP2_UPPER_LIMIT_SECONDS
            and tp4_proj <= 0.70 * tp2_proj
        )
        if tp4_replaces:
            selected, rule = 4, "rule4_tp4_replaces_tp2_over4h_and_30pct_faster"
        else:
            selected, rule = 2, "rule5_select_tp2"

    # Rule 6 tie-break: among configurations eligible under rules 2-4
    # (TP=1 eligible per rule 2, TP=2/TP=4 passed), prefer fewer GPUs when
    # the projected wall times differ by 10% or less. TP=1 is never made
    # eligible by this rule when its projection exceeds the two-hour limit.
    comparable: list[tuple[int, float]] = []
    if tp1_eligible and projections.get(1) is not None:
        comparable.append((1, projections[1]))
    for tp in (2, 4):
        cand = candidates.get(tp)
        proj = projections.get(tp)
        if cand is not None and cand.passed and proj is not None:
            comparable.append((tp, proj))
    selected_proj = projections.get(selected)
    for tp, proj in comparable:
        if selected_proj is not None and abs(proj - selected_proj) <= 0.10 * selected_proj:
            if tp < selected:
                selected, rule = tp, f"{rule}_then_rule6_fewer_gpus"

    return SelectionResult(
        selected_tp=selected,
        decision_rule=rule,
        candidate_results=candidates,
        projected_requests=TURKISH_PROJECTED_REQUESTS,
        projected_wall_seconds={tp: proj for tp, proj in projections.items() if proj is not None},
    )
