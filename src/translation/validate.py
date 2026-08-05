"""Structural and semantic validation of translation candidates.

Assigns translation statuses (human_verified / automatic_high /
automatic_medium / automatic_low / failed), verifies coverage and hashes,
checks English-only output, source-language leakage, plausible length ratios,
preserved numbers/dates/named terms, and full-vs-segment consistency for D3TEC
and ANDROIDS. Optional independent NLLB comparison for short units and an
optional Qwen verification pass for long units. A verifier may flag or reject
output but never silently rewrites an accepted translation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, verifier_prompt
from src.utils import (
    configure_logging,
    get_logger,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_jsonl_rows,
    sha256_text,
    write_jsonl,
)


LOGGER = get_logger(__name__)

STATUS_ORDER = ("human_verified", "automatic_high", "automatic_medium", "automatic_low", "failed")
STATUS_RANK = {status: index for index, status in enumerate(STATUS_ORDER)}

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
TURKISH_ONLY_RE = re.compile(r"[\u011f\u015f\u0131\u0130]")
SPANISH_ONLY_RE = re.compile(r"[\u00d1\u00f1\u00bf\u00a1]")
ITALIAN_ONLY_RE = re.compile(r"[\u00f9]")
EXPLANATION_RE = re.compile(
    r"\b(?:translation|explanation|reason|analysis|interpretation|summary)\s*:\s*|"
    r"\b(?:here is|here's|sure|of course)\b|```|"
    r"\b(?:missing|added)\s*(?:invariant|fact|number|name)s?\b",
    re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")
CAPITALIZED_RE = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z]+")

SENSITIVE_TERMS: dict[str, tuple[str, ...]] = {
    "en": ("suicide", "suicidal", "self-harm", "kill myself", "end my life", "die", "death"),
    "zh": ("自杀", "自残", "想死", "死"),
    "tr": ("intihar", "kendimi öldür", "ölmek", "ölüm"),
    "es": ("suicidio", "suicida", "autolesión", "matarme", "morir"),
    "it": ("suicidio", "suicida", "autolesion", "uccidermi", "morire"),
}

MIN_LENGTH_RATIO = 0.25
MAX_LENGTH_RATIO = 3.5
MAX_NUMBER_LOSS_FRACTION = 0.2
MIN_ENTITY_OVERLAP = 0.3
NLLB_SHORT_UNIT_MAX_CHARS = 800
VERIFIER_LONG_UNIT_MIN_CHARS = 1500

NLLB_LANGUAGE_CODES = {
    "cmdc": ("zho_Hans", "eng_Latn"),
    "turkish": ("tur_Latn", "eng_Latn"),
    "d3tec": ("spa_Latn", "eng_Latn"),
    "androids_interview": ("ita_Latn", "eng_Latn"),
}


def _strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _digit_sequences(text: str) -> list[str]:
    return [match.group(0) for match in DIGIT_RE.finditer(text)]


def _normalized_numbers(text: str) -> set[str]:
    normalized: set[str] = set()
    for match in DIGIT_RE.finditer(text):
        value = match.group(0).replace(",", "").replace(".", "")
        if value and len(value) <= 6:
            normalized.add(value.lstrip("0") or "0")
    return normalized


def _capitalized_tokens(text: str) -> set[str]:
    tokens = set()
    for match in CAPITALIZED_RE.finditer(text):
        token = _strip_accents(match.group(0)).lower()
        if len(token) >= 2:
            tokens.add(token)
    return tokens


def english_only(translation: str, source_language: str) -> list[str]:
    failures: list[str] = []
    if CJK_RE.search(translation):
        failures.append("CJK characters present in translation")
    source_checks = {
        "tr": (TURKISH_ONLY_RE, "Turkish-only characters present in translation"),
        "es": (SPANISH_ONLY_RE, "Spanish-only characters present in translation"),
        "it": (ITALIAN_ONLY_RE, "Italian-only characters present in translation"),
    }
    if source_language in source_checks:
        pattern, message = source_checks[source_language]
        if pattern.search(translation):
            failures.append(message)
    return failures


def leakage_checks(unit: dict[str, Any], translation: str) -> list[str]:
    failures: list[str] = []
    if EXPLANATION_RE.search(translation):
        failures.append("explanation or formatting text present")
    source_text = str(unit.get("source_text", "")).strip()
    if source_text and len(source_text) <= 500 and source_text in translation:
        failures.append("source text copied verbatim into translation")
    if any(marker in translation for marker in ("<target>", "<context>", "clinical transcript translator")):
        failures.append("prompt leakage into translation")
    return failures


def length_ratio_ok(unit: dict[str, Any], translation: str) -> bool:
    source_language = str(unit.get("source_language", "")).strip().lower()
    source_text = str(unit.get("source_text", "")).strip()
    if not source_text:
        return True
    source_units = len(source_text) if source_language == "zh" else len(source_text.split())
    target_units = len(translation.split())
    if source_units <= 0:
        return True
    ratio = target_units / source_units
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def _normalize_text(text: str, *, source_language: str) -> str:
    if source_language == "zh":
        return unicodedata.normalize("NFC", text).casefold()
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )


def sensitive_term_violations(unit: dict[str, Any], translation: str) -> list[str]:
    source_language = str(unit.get("source_language", "")).strip().lower()
    source_text = _normalize_text(str(unit.get("source_text", "")), source_language=source_language)
    translation_lower = _normalize_text(translation, source_language=source_language)
    english_terms = tuple(
        _normalize_text(term, source_language=source_language)
        for term in SENSITIVE_TERMS["en"]
        if _normalize_text(term, source_language=source_language)
    )
    violations: list[str] = []
    for raw_term in SENSITIVE_TERMS.get(source_language, ()):
        term = _normalize_text(raw_term, source_language=source_language)
        if not term:
            continue
        if term in source_text and not any(
            english_term in translation_lower for english_term in english_terms
        ):
            violations.append(f"sensitive term {raw_term!r} not preserved in translation")
            break
    return violations


def number_preservation_ok(unit: dict[str, Any], translation: str) -> bool:
    source_numbers = _normalized_numbers(str(unit.get("source_text", "")))
    translation_numbers = _normalized_numbers(translation)
    if not source_numbers:
        return True
    lost = len(source_numbers - translation_numbers)
    return lost / len(source_numbers) <= MAX_NUMBER_LOSS_FRACTION


def entity_overlap_ok(unit: dict[str, Any], translation: str) -> bool:
    source_language = str(unit.get("source_language", "")).strip().lower()
    if source_language == "zh":
        return True
    source_entities = _capitalized_tokens(str(unit.get("source_text", "")))
    if not source_entities:
        return True
    translation_entities = _capitalized_tokens(translation)
    overlap = len(source_entities & translation_entities) / len(source_entities)
    return overlap >= MIN_ENTITY_OVERLAP


def _chrf_precision(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    counter = Counter(ref_tokens)
    matched = 0
    for token in pred_tokens:
        if counter.get(token, 0) > 0:
            counter[token] -= 1
            matched += 1
    return matched / len(pred_tokens)


def nllb_translate(
    nllb_model: Any,
    nllb_tokenizer: Any,
    text: str,
    source_language: str,
    target_language: str,
    device: str,
) -> str | None:
    try:
        import torch

        inputs = nllb_tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(target_language),
            max_new_tokens=512,
        )
        output = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return str(output).strip()
    except Exception as exc:
        LOGGER.warning("NLLB translation failed for a unit: %s", exc)
        return None


def _load_nllb(model_path: str | None) -> tuple[Any, Any, str] | None:
    if not model_path:
        return None
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("NLLB comparison unavailable (imports): %s", exc)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as exc:
        LOGGER.warning("NLLB comparison unavailable (load): %s", exc)
        return None


def _load_verifier_client(base_url: str | None, model: str | None) -> Any | None:
    if not base_url or not model:
        return None
    try:
        from openai import OpenAI

        return OpenAI(base_url=base_url, api_key="EMPTY")
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Verifier pass unavailable: %s", exc)
        return None


def verify_with_qwen(client: Any, model: str, unit: dict[str, Any], translation: str, seed: int) -> list[str]:
    if client is None:
        return []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": verifier_prompt(unit, translation)},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = (response.choices[0].message.content or "").strip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return ["verifier returned no structured result"]
        payload = json.loads(content[start : end + 1])
        missing = [str(item) for item in payload.get("missing", []) if str(item).strip()]
        added = [str(item) for item in payload.get("added", []) if str(item).strip()]
        flags: list[str] = []
        if missing:
            flags.append("verifier: missing invariants: " + "; ".join(missing[:5]))
        if added:
            flags.append("verifier: added invariants: " + "; ".join(added[:5]))
        return flags
    except Exception as exc:
        LOGGER.warning("Verifier request failed: %s", exc)
        return ["verifier request failed"]


def validate_candidate(
    unit: dict[str, Any],
    candidate: dict[str, Any],
    *,
    nllb: tuple[Any, Any, str] | None,
    verifier_client: Any,
    verifier_model: str | None,
    seed: int,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    source_language = str(unit.get("source_language", "")).strip().lower()
    translation = str(candidate.get("translation", "")).strip()

    if candidate.get("status") != "translated":
        return "failed", ["candidate not in translated state"]
    if candidate.get("source_sha256") != unit.get("source_sha256"):
        return "failed", ["source hash mismatch"]
    if candidate.get("translation_sha256") != sha256_text(translation):
        return "failed", ["translation hash mismatch"]
    if not translation:
        return "failed", ["empty translation"]
    failures.extend(english_only(translation, source_language))
    failures.extend(leakage_checks(unit, translation))
    if failures:
        return "failed", failures

    if not length_ratio_ok(unit, translation):
        warnings.append("implausible source-to-target length ratio")
    violations = sensitive_term_violations(unit, translation)
    warnings.extend(violations)
    if not number_preservation_ok(unit, translation):
        warnings.append("numbers not preserved")
    if not entity_overlap_ok(unit, translation):
        warnings.append("named entities not preserved")

    verifier_flags: list[str] = []
    if (
        verifier_client is not None
        and verifier_model
        and len(str(unit.get("source_text", ""))) >= VERIFIER_LONG_UNIT_MIN_CHARS
    ):
        verifier_flags = verify_with_qwen(verifier_client, verifier_model, unit, translation, seed)
    warnings.extend(verifier_flags)

    if warnings:
        return "automatic_low", warnings

    disagreement = False
    if nllb is not None and len(str(unit.get("source_text", ""))) <= NLLB_SHORT_UNIT_MAX_CHARS:
        model, tokenizer, device = nllb
        source_code, target_code = NLLB_LANGUAGE_CODES.get(
            str(unit.get("dataset", "")), ("eng_Latn", "eng_Latn")
        )
        reference = nllb_translate(model, tokenizer, unit["source_text"], source_code, target_code, device)
        if reference:
            f1 = _chrf_precision(translation, reference)
            if f1 < 0.5:
                disagreement = True
    if disagreement:
        return "automatic_medium", ["large Qwen-versus-NLLB disagreement"]
    return "automatic_high", []


def consistency_checks(
    units: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    full_units = [unit for unit in units if unit["scope"].startswith("full_")]
    segment_units = [unit for unit in units if unit["scope"] == "segment"]
    full_by_id = {unit["unit_id"]: unit for unit in full_units}
    candidate_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(candidate["unit_id"]), str(candidate["field"])): candidate for candidate in candidates
    }
    for segment in segment_units:
        parent_id = str(segment.get("context_id", ""))
        if not parent_id or parent_id not in full_by_id:
            failures.append(f"segment {segment['unit_id']} has no full unit {parent_id!r}")
    segment_join_by_parent: dict[str, list[str]] = defaultdict(list)
    for segment in segment_units:
        parent_id = str(segment.get("context_id", ""))
        key = (str(segment["unit_id"]), str(segment["field"]))
        if key in candidate_by_key:
            segment_join_by_parent[parent_id].append(candidate_by_key[key]["translation"])
    for parent_id, full_unit in full_by_id.items():
        key = (str(full_unit["unit_id"]), str(full_unit["field"]))
        full_candidate = candidate_by_key.get(key)
        if full_candidate is None:
            continue
        joined = " ".join(segment_join_by_parent.get(parent_id, []))
        full_numbers = _normalized_numbers(full_candidate["translation"])
        joined_numbers = _normalized_numbers(joined)
        missing_from_segments = full_numbers - joined_numbers
        if missing_from_segments:
            failures.append(
                f"full unit {parent_id} numbers missing from its segments: {sorted(missing_from_segments)[:5]}"
            )
    return failures, {}


def load_reviewed(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    rows = read_jsonl(path)
    reviewed: dict[str, str] = {}
    for row in rows:
        unit_id = str(row.get("unit_id", "")).strip()
        status = str(row.get("status", "")).strip()
        if not unit_id:
            raise ValueError(f"Reviewed row missing unit_id: {row}")
        if status != "human_verified":
            raise ValueError(f"Reviewed row for {unit_id} must declare status=human_verified")
        reviewed[unit_id] = status
    return reviewed


def run_validation(
    units_path: str | Path,
    candidates_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    audit_path: str | Path,
    *,
    nllb_model: str | None,
    verifier_base_url: str | None,
    verifier_model: str | None,
    reviewed_path: str | Path | None,
    seed: int,
    run_consistency: bool = True,
) -> dict[str, Any]:
    units = read_jsonl(units_path)
    candidates = read_jsonl(candidates_path)
    reviewed = load_reviewed(resolve_project_path(reviewed_path) if reviewed_path else None)

    unit_keys = {
        (str(unit["unit_id"]), str(unit["field"]), int(unit.get("part_index", 0))) for unit in units
    }
    candidate_keys: set[tuple[str, str, int]] = set()
    for candidate in candidates:
        key = (
            str(candidate["unit_id"]),
            str(candidate["field"]),
            int(candidate.get("part_index", 0)),
        )
        if key in candidate_keys:
            raise ValueError(f"Duplicate candidate for {key}")
        candidate_keys.add(key)
    missing = sorted(unit_keys - candidate_keys)
    extra = sorted(candidate_keys - unit_keys)
    if missing:
        raise ValueError(f"Coverage failure: {len(missing)} units without candidates: {missing[:10]}")

    units_by_key = {(u["unit_id"], u["field"], u.get("part_index", 0)): u for u in units}
    nllb = _load_nllb(nllb_model)
    verifier_client = _load_verifier_client(verifier_base_url, verifier_model)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    dataset = str(units[0]["dataset"]) if units else ""
    for candidate in candidates:
        key = (
            str(candidate["unit_id"]),
            str(candidate["field"]),
            int(candidate.get("part_index", 0)),
        )
        unit = units_by_key[key]
        status, reasons = validate_candidate(
            unit,
            candidate,
            nllb=nllb,
            verifier_client=verifier_client,
            verifier_model=verifier_model,
            seed=seed,
        )
        if str(unit["unit_id"]) in reviewed:
            status = "human_verified"
            reasons = []
        for reason in reasons:
            failure_reasons[reason] += 1
        status_counts[status] += 1
        record = {
            "dataset": unit["dataset"],
            "unit_id": unit["unit_id"],
            "field": unit["field"],
            "part_index": unit.get("part_index", 0),
            "part_count": unit.get("part_count", 1),
            "translation": candidate["translation"],
            "translation_sha256": candidate["translation_sha256"],
            "model": candidate["model"],
            "model_revision": candidate.get("model_revision", ""),
            "precision": candidate.get("precision", "bf16"),
            "prompt_version": candidate.get("prompt_version", PROMPT_VERSION),
            "source_sha256": candidate["source_sha256"],
            "context_sha256": unit.get("context_sha256", ""),
            "status": status,
            "reasons": reasons,
        }
        if status == "failed":
            rejected.append(record)
        else:
            accepted.append(record)

    consistency_failures: list[str] = []
    if run_consistency and dataset in {"d3tec", "androids_interview"}:
        consistency_failures, _ = consistency_checks(units, candidates)
        if consistency_failures:
            for failure in consistency_failures:
                failure_reasons[f"consistency: {failure}"] += 1

    accepted_sorted = sorted(
        accepted, key=lambda row: (row["unit_id"], row["field"], row["part_index"])
    )
    rejected_sorted = sorted(
        rejected, key=lambda row: (row["unit_id"], row["field"], row["part_index"])
    )
    write_jsonl(accepted_sorted, accepted_path)
    write_jsonl(rejected_sorted, rejected_path)

    accepted_cache_hash = sha256_jsonl_rows(accepted_sorted)
    audit = {
        "dataset": dataset,
        "model": candidates[0]["model"] if candidates else "",
        "model_revision": candidates[0].get("model_revision", "") if candidates else "",
        "precision": candidates[0].get("precision", "bf16") if candidates else "",
        "prompt_version": PROMPT_VERSION,
        "prompt_system_sha256": sha256_text(SYSTEM_PROMPT),
        "seed": seed,
        "units_file_sha256": sha256_text("\n".join(line for line in Path(units_path).read_text("utf-8").splitlines() if line.strip())),
        "candidates_file_sha256": sha256_text("\n".join(line for line in Path(candidates_path).read_text("utf-8").splitlines() if line.strip())),
        "unit_count": len(unit_keys),
        "candidate_count": len(candidates),
        "extra_candidates": len(extra),
        "status_counts": dict(status_counts),
        "failure_reasons": dict(failure_reasons),
        "consistency_failures": consistency_failures,
        "nllb_comparison": bool(nllb),
        "verifier_pass": verifier_client is not None,
        "reviewed_units": len(reviewed),
        "accepted_cache_sha256": accepted_cache_hash,
        "accepted_cache_record_count": len(accepted_sorted),
    }
    save_json(audit, audit_path)
    LOGGER.info(
        "Validation summary: statuses=%s accepted=%s rejected=%s cache_hash=%s",
        dict(status_counts),
        len(accepted_sorted),
        len(rejected_sorted),
        accepted_cache_hash,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate translation candidates and write accepted/rejected caches.")
    parser.add_argument("--units", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--nllb-model", help="Optional NLLB-200 checkpoint for independent short-unit comparison.")
    parser.add_argument("--verifier-base-url", help="Optional vLLM endpoint for the long-unit Qwen verification pass.")
    parser.add_argument("--verifier-model", default="qwen3.6-27b")
    parser.add_argument("--reviewed", help="Optional JSONL of user-reviewed units (status=human_verified).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-consistency", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_validation(
        args.units,
        args.candidates,
        args.accepted,
        args.rejected,
        args.audit,
        nllb_model=args.nllb_model,
        verifier_base_url=args.verifier_base_url,
        verifier_model=args.verifier_model,
        reviewed_path=args.reviewed,
        seed=args.seed,
        run_consistency=not args.no_consistency,
    )


if __name__ == "__main__":
    main()
