"""Qwen3.8 validation harness (runbook sections 11.3 and 11.4).

The harness runs the 64 synthetic cases at concurrency levels 1, 8, 16, and
32 against a localhost vLLM server, repeats concurrency 1 for determinism,
and reports JSON/schema validity, label accuracy, required-concept coverage,
request errors, TTFT p50/p95, end-to-end latency p50/p95, output tokens per
second, aggregate requests per second, startup time, and peak allocated GPU
memory sampled from ``nvidia-smi``.

Nothing here is a scientific result: the synthetic fixture is a serving
health check only.
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

VALIDATION_SYSTEM_PROMPT = (
    "You recover interviewer questions from answer-only transcripts. "
    "For this synthetic validation case, infer the missing interviewer "
    "question from the answer and return exactly one JSON object with "
    "case_id, inferred_question, label (POSITIVE, NEGATIVE, NEUTRAL, or "
    "MIXED), and confidence (HIGH, MEDIUM, or LOW). Classify the inferred "
    "question's framing, not the emotional tone of the answer:\n"
    "- POSITIVE asks about happiness, enjoyment, strengths, hope, support, "
    "positive memories, or pleasant events.\n"
    "- NEGATIVE asks about sadness, distress, problems, symptoms, loss, fear, "
    "conflict, or unpleasant events.\n"
    "- NEUTRAL is factual, descriptive, demographic, procedural, or open "
    "framing that does not ask for positive or negative valence.\n"
    "- MIXED asks for both positive and negative material, or combines "
    "opposing valences in one question."
)
assert isinstance(VALIDATION_SYSTEM_PROMPT, str)

from src.qwen38.contracts import (
    CONFIDENCE_WEIGHTS,
    CONCURRENCY_LEVELS,
    HUGGINGFACE_HUB_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    OPENAI_VERSION,
    RESTART_SUBSET_SIZE,
    SERVED_MODEL,
    SYNTHETIC_CASE_COUNT,
    SYNTHETIC_LABELS,
    SYNTHETIC_LANGUAGES,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    VLLM_VERSION,
    Confidence,
    Label,
    WordingStatus,
    normalize_for_determinism,
    request_settings,
    structured_output_schema,
    validate_validation_response,
)

SERVER_STARTUP_TIMEOUT_SECONDS = 600
SERVER_READY_POLL_SECONDS = 2
MIN_LABEL_ACCURACY_C1 = 0.95
MIN_CONCEPT_COVERAGE = 0.90


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    language: str
    answer_text: str
    expected_label: str
    required_question_concepts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "answer_text": self.answer_text,
            "expected_label": self.expected_label,
            "required_question_concepts": list(self.required_question_concepts),
        }


def load_synthetic_cases(path: str | Path) -> list[SyntheticCase]:
    """Load and strictly validate the synthetic fixture distribution.

    Required: 64 cases, 32 Turkish + 32 English, 16 of each label, and
    exactly eight cases per language/label pair.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if len(rows) != SYNTHETIC_CASE_COUNT:
        raise ValueError(f"{path}: expected {SYNTHETIC_CASE_COUNT} cases, found {len(rows)}")

    cases: list[SyntheticCase] = []
    seen_ids: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        errors: list[str] = []
        for required in ("case_id", "language", "answer_text", "expected_label", "required_question_concepts"):
            if required not in row:
                errors.append(f"missing {required!r}")
        if errors:
            raise ValueError(f"{path} row {index + 1}: {', '.join(errors)}")
        case_id = str(row["case_id"])
        language = str(row["language"])
        answer_text = str(row["answer_text"])
        expected_label = str(row["expected_label"])
        concepts = tuple(str(c) for c in row["required_question_concepts"])
        if case_id in seen_ids:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        seen_ids.add(case_id)
        if language not in SYNTHETIC_LANGUAGES:
            raise ValueError(f"{path}: case {case_id} has unsupported language {language!r}")
        if expected_label not in SYNTHETIC_LABELS:
            raise ValueError(f"{path}: case {case_id} has unsupported label {expected_label!r}")
        if not answer_text.strip():
            raise ValueError(f"{path}: case {case_id} has an empty answer_text")
        if not concepts or any(not concept.strip() for concept in concepts):
            raise ValueError(f"{path}: case {case_id} needs non-empty required_question_concepts")
        counts[(language, expected_label)] = counts.get((language, expected_label), 0) + 1
        cases.append(
            SyntheticCase(
                case_id=case_id,
                language=language,
                answer_text=answer_text,
                expected_label=expected_label,
                required_question_concepts=concepts,
            )
        )

    expected_per_pair = SYNTHETIC_CASE_COUNT // (len(SYNTHETIC_LANGUAGES) * len(SYNTHETIC_LABELS))
    for language in SYNTHETIC_LANGUAGES:
        language_total = sum(counts.get((language, label), 0) for label in SYNTHETIC_LABELS)
        if language_total != SYNTHETIC_CASE_COUNT // 2:
            raise ValueError(
                f"{path}: expected {SYNTHETIC_CASE_COUNT // 2} {language} cases, found {language_total}"
            )
        for label in SYNTHETIC_LABELS:
            pair = counts.get((language, label), 0)
            if pair != expected_per_pair:
                raise ValueError(
                    f"{path}: expected {expected_per_pair} {language}/{label} cases, found {pair}"
                )
    return cases


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


def _concept_coverage(parsed: dict[str, Any], case: SyntheticCase) -> bool:
    question = str(parsed.get("inferred_question", ""))
    normalized = normalize_for_determinism(question)
    return all(normalize_for_determinism(concept) in normalized for concept in case.required_question_concepts)


def _label_correct(parsed: dict[str, Any], case: SyntheticCase) -> bool:
    return str(parsed.get("label", "")) == case.expected_label


async def _server_ready(base_url: str, timeout: int) -> tuple[bool, float]:
    import urllib.request

    started = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return True, time.monotonic() - started
        except Exception:
            pass
        if time.monotonic() - started >= timeout:
            return False, time.monotonic() - started
        await asyncio.sleep(SERVER_READY_POLL_SECONDS)


async def _nvidia_memory_sampler(interval: float, stop: asyncio.Event) -> dict[str, Any]:
    import subprocess

    samples: list[list[int]] = []
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                values = [int(v.strip()) for v in result.stdout.splitlines() if v.strip()]
                if values:
                    samples.append(values)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
    if not samples:
        return {"sampled": False, "gpu_count": 0, "peak_mib": None, "peak_per_gpu_mib": []}
    per_gpu = list(range(len(samples[0])))
    peak_per_gpu = [max(s[gpu] for s in samples) for gpu in per_gpu]
    return {
        "sampled": True,
        "gpu_count": len(samples[0]),
        "peak_mib": max(peak_per_gpu),
        "peak_per_gpu_mib": peak_per_gpu,
        "samples_taken": len(samples),
    }


async def _run_case(
    client: Any,
    model: str,
    case: SyntheticCase,
    *,
    max_tokens: int,
    seed: int,
    repair: bool,
) -> dict[str, Any]:
    settings = request_settings(max_tokens)
    schema_payload = structured_output_schema(
        {
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
    )

    async def one_attempt(corrective: bool) -> tuple[dict[str, Any], float, float, int, str | None]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": VALIDATION_SYSTEM_PROMPT,
            },
            {"role": "user", "content": case.answer_text},
        ]
        if corrective:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was not valid JSON matching the required "
                        "schema. Return only the single JSON object, no prose."
                    ),
                }
            )
        started = time.monotonic()
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_tokens=settings["max_tokens"],
            seed=seed,
            response_format=schema_payload,
            extra_body={"chat_template_kwargs": settings["chat_template_kwargs"]},
            stream=True,
        )
        chunks: list[str] = []
        first_token_at: float | None = None
        async for chunk in stream:
            if first_token_at is None and chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    first_token_at = time.monotonic()
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    chunks.append(content)
        finished_at = time.monotonic()
        content = "".join(chunks)
        ttft = (first_token_at - started) if first_token_at is not None else None
        return (
            {
                "content": content,
                "ttft_seconds": ttft,
                "e2e_seconds": finished_at - started,
                "completion_tokens": len(content.split()) if content.strip() else 0,
            },
            ttft,
            finished_at - started,
            len(content.split()) if content.strip() else 0,
            None,
        )

    last_error: str | None = None
    attempts = 0
    while attempts <= (1 if repair else 0):
        attempts += 1
        try:
            attempt_result, _ttft, _e2e, _tokens, _error = await one_attempt(corrective=attempts > 1)
            parsed = _parse_json_object(attempt_result["content"])
            schema_errors = [] if parsed is not None else ["not a JSON object"]
            if parsed is not None:
                schema_errors = validate_validation_response(parsed)
            if not schema_errors and parsed is not None:
                return {
                    "case_id": case.case_id,
                    "json_valid": True,
                    "schema_valid": True,
                    "parsed": parsed,
                    "label_correct": _label_correct(parsed, case),
                    "concepts_covered": _concept_coverage(parsed, case),
                    "empty_output": not attempt_result["content"].strip(),
                    "request_error": None,
                    "repair_used": attempts > 1,
                    "ttft_seconds": _ttft,
                    "e2e_seconds": _e2e,
                    "completion_tokens": _tokens,
                }
            last_error = "; ".join(schema_errors or ["invalid JSON"])
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "case_id": case.case_id,
        "json_valid": False,
        "schema_valid": False,
        "parsed": None,
        "label_correct": False,
        "concepts_covered": False,
        "empty_output": False,
        "request_error": last_error,
        "repair_used": False,
        "ttft_seconds": None,
        "e2e_seconds": None,
        "completion_tokens": 0,
    }


async def _run_level(
    client: Any,
    model: str,
    cases: Sequence[SyntheticCase],
    concurrency: int,
    *,
    max_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(case: SyntheticCase) -> dict[str, Any]:
        async with semaphore:
            return await _run_case(
                client, model, case, max_tokens=max_tokens, seed=seed, repair=True
            )

    results = await asyncio.gather(*(guarded(case) for case in cases))
    return list(results)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _level_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in results if not r["request_error"]]
    json_valid = sum(1 for r in results if r["json_valid"])
    schema_valid = sum(1 for r in results if r["schema_valid"])
    label_correct = sum(1 for r in results if r["label_correct"])
    concepts = sum(1 for r in results if r["concepts_covered"])
    empty = sum(1 for r in results if r["empty_output"])
    ttfts = [r["ttft_seconds"] for r in complete if r["ttft_seconds"] is not None]
    e2es = [r["e2e_seconds"] for r in complete if r["e2e_seconds"] is not None]
    tokens = [r["completion_tokens"] for r in complete]
    total_tokens = sum(tokens)
    total_seconds = sum(r["e2e_seconds"] for r in complete if r["e2e_seconds"] is not None)
    return {
        "cases_total": len(results),
        "cases_complete": len(complete),
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "label_correct": label_correct,
        "concepts_covered": concepts,
        "empty_outputs": empty,
        "request_errors": len(results) - len(complete),
        "ttft_p50_seconds": _percentile(ttfts, 0.50),
        "ttft_p95_seconds": _percentile(ttfts, 0.95),
        "e2e_p50_seconds": _percentile(e2es, 0.50),
        "e2e_p95_seconds": _percentile(e2es, 0.95),
        "total_completion_tokens": total_tokens,
        "requests_per_second": total_tokens / total_seconds if total_seconds > 0 else math.nan,
        "output_tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else math.nan,
    }


def _aggregate_rate(results: list[dict[str, Any]]) -> float:
    complete = [r for r in results if r["e2e_seconds"] is not None]
    if not complete:
        return math.nan
    return len(complete) / sum(r["e2e_seconds"] for r in complete)


def summarize_level(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _level_summary(results)
    summary["aggregate_requests_per_second"] = _aggregate_rate(results)
    return summary


def compare_determinism(
    pass_a: list[dict[str, Any]], pass_b: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """Two concurrency-1 passes must match labels and normalized questions."""
    by_id_a = {r["case_id"]: r for r in pass_a}
    by_id_b = {r["case_id"]: r for r in pass_b}
    mismatches: list[str] = []
    if set(by_id_a) != set(by_id_b):
        return False, ["concurrency-1 passes cover different case sets"]
    for case_id in sorted(by_id_a):
        a = by_id_a[case_id]
        b = by_id_b[case_id]
        label_a = str(a.get("parsed", {}).get("label", "")) if a.get("parsed") else ""
        label_b = str(b.get("parsed", {}).get("label", "")) if b.get("parsed") else ""
        question_a = (
            normalize_for_determinism(str(a["parsed"].get("inferred_question", "")))
            if a.get("parsed")
            else ""
        )
        question_b = (
            normalize_for_determinism(str(b["parsed"].get("inferred_question", "")))
            if b.get("parsed")
            else ""
        )
        if label_a != label_b:
            mismatches.append(f"{case_id}: label {label_a!r} != {label_b!r}")
        if question_a != question_b:
            mismatches.append(f"{case_id}: normalized question differs")
    return not mismatches, mismatches


def run_validation(
    *,
    base_url: str,
    model: str,
    cases: Sequence[SyntheticCase],
    concurrency_levels: Sequence[int] = CONCURRENCY_LEVELS,
    max_tokens: int = 1024,
    seed: int = 42,
    restart_subset: bool = False,
    server_startup_timeout: int = SERVER_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the full validation workload and return the results payload.

    The OpenAI asynchronous client and the GPU-memory sampler run inside one
    event loop. The caller decides whether this is the main pass or the
    post-restart subset pass.
    """
    from openai import AsyncOpenAI

    async def orchestrate() -> dict[str, Any]:
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)
        stop = asyncio.Event()
        sampler_task = asyncio.create_task(_nvidia_memory_sampler(2.0, stop))
        try:
            ready, startup_seconds = await _server_ready(base_url, server_startup_timeout)
            if not ready:
                return {
                    "ready": False,
                    "startup_seconds": startup_seconds,
                    "error": "server did not become ready",
                    "levels": {},
                    "gpu_memory": {},
                }
            levels: dict[str, list[dict[str, Any]]] = {}
            if restart_subset:
                subset = list(cases[:RESTART_SUBSET_SIZE])
                levels["restart_subset"] = await _run_level(
                    client, model, subset, 1, max_tokens=max_tokens, seed=seed
                )
            else:
                levels["c1_pass_a"] = await _run_level(
                    client, model, cases, 1, max_tokens=max_tokens, seed=seed
                )
                for level in concurrency_levels:
                    if level == 1:
                        continue
                    levels[f"c{level}"] = await _run_level(
                        client, model, cases, level, max_tokens=max_tokens, seed=seed
                    )
                levels["c1_pass_b"] = await _run_level(
                    client, model, cases, 1, max_tokens=max_tokens, seed=seed
                )
            await client.close()
            stop.set()
            await sampler_task
            gpu_memory = sampler_task.result()
            return {
                "ready": True,
                "startup_seconds": startup_seconds,
                "levels": levels,
                "gpu_memory": gpu_memory,
            }
        finally:
            stop.set()
            await client.close()

    return asyncio.run(orchestrate())


def summarize_acceptance(
    results: dict[str, Any],
    restart_results: dict[str, Any] | None,
    *,
    model_revision: str,
    deployment_model_revision: str,
    environment_versions: dict[str, Any],
    deployment_environment_versions: dict[str, Any],
    model_manifest_sha256: str | None,
    deployment_model_manifest_sha256: str | None,
    wheelhouse_manifest_sha256: str | None,
    deployment_wheelhouse_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Evaluate the runbook section 11.4 acceptance gates and report every check."""
    checks: list[dict[str, Any]] = []
    passed = True

    def record(check_id: str, description: str, ok: bool, details: Any = None) -> None:
        nonlocal passed
        if not ok:
            passed = False
        checks.append(
            {"check_id": check_id, "description": description, "passed": ok, "details": details}
        )

    if not results.get("ready"):
        record("server_ready", "server became ready within ten minutes", False, results.get("error"))
        return {"passed": False, "checks": checks, "metrics": {}, "summary": {}}

    startup_seconds = float(results["startup_seconds"])
    record(
        "server_ready",
        "server became ready within ten minutes",
        startup_seconds <= SERVER_STARTUP_TIMEOUT_SECONDS,
        {"startup_seconds": startup_seconds},
    )

    levels = results["levels"]
    expected_levels = [f"c{level}" for level in CONCURRENCY_LEVELS if level != 1]
    expected_levels += ["c1_pass_a", "c1_pass_b"]

    level_summaries: dict[str, dict[str, Any]] = {}
    for name in expected_levels:
        level_results = levels.get(name)
        if level_results is None:
            record(
                f"complete_{name}",
                f"all 64 cases complete at {name}",
                False,
                "level missing",
            )
            passed = False
            continue
        summary = summarize_level(level_results)
        level_summaries[name] = summary
        ok = summary["cases_complete"] == SYNTHETIC_CASE_COUNT
        record(
            f"complete_{name}",
            f"all 64 cases complete at {name}",
            ok,
            {"cases_complete": summary["cases_complete"]},
        )
        record(
            f"json_validity_{name}",
            f"JSON validity 100% at {name}",
            summary["json_valid"] == SYNTHETIC_CASE_COUNT,
            {"json_valid": summary["json_valid"]},
        )
        record(
            f"schema_validity_{name}",
            f"schema validity 100% at {name}",
            summary["schema_valid"] == SYNTHETIC_CASE_COUNT,
            {"schema_valid": summary["schema_valid"]},
        )
        record(
            f"zero_errors_{name}",
            f"zero HTTP errors at {name}",
            summary["request_errors"] == 0,
            {"request_errors": summary["request_errors"]},
        )
        record(
            f"zero_empty_{name}",
            f"zero empty outputs at {name}",
            summary["empty_outputs"] == 0,
            {"empty_outputs": summary["empty_outputs"]},
        )

    c1 = level_summaries.get("c1_pass_a")
    if c1 is None:
        record("label_accuracy_c1", "label accuracy >= 95% at concurrency 1", False, "level missing")
    else:
        record(
            "label_accuracy_c1",
            "label accuracy >= 95% at concurrency 1",
            c1["label_correct"] >= MIN_LABEL_ACCURACY_C1 * SYNTHETIC_CASE_COUNT,
            {"label_correct": c1["label_correct"], "total": SYNTHETIC_CASE_COUNT},
        )
        record(
            "concept_coverage_c1",
            "required-concept coverage >= 90%",
            c1["concepts_covered"] >= MIN_CONCEPT_COVERAGE * SYNTHETIC_CASE_COUNT,
            {"concepts_covered": c1["concepts_covered"], "total": SYNTHETIC_CASE_COUNT},
        )
        for name in [f"c{level}" for level in CONCURRENCY_LEVELS if level != 1]:
            other = level_summaries.get(name)
            if other is None:
                record(
                    f"label_accuracy_{name}",
                    f"label accuracy within one case of concurrency 1 at {name}",
                    False,
                    "level missing",
                )
                continue
            ok = other["label_correct"] >= c1["label_correct"] - 1
            record(
                f"label_accuracy_{name}",
                f"label accuracy within one case of concurrency 1 at {name}",
                ok,
                {"label_correct": other["label_correct"], "c1_label_correct": c1["label_correct"]},
            )

    determinism_ok, determinism_details = compare_determinism(
        levels.get("c1_pass_a", []), levels.get("c1_pass_b", [])
    )
    record(
        "determinism_c1",
        "two concurrency-1 passes identical labels and normalized questions",
        determinism_ok,
        {"mismatches": determinism_details},
    )

    gpu = results.get("gpu_memory", {})
    nan_metrics = False
    for level_name in expected_levels:
        level_records = levels.get(level_name) or []
        for level_record in level_records:
            for key in ("ttft_seconds", "e2e_seconds", "completion_tokens"):
                value = level_record.get(key)
                if value is not None and not math.isfinite(float(value)):
                    nan_metrics = True
    for name, summary in level_summaries.items():
        for key in ("ttft_p50_seconds", "ttft_p95_seconds", "e2e_p50_seconds", "e2e_p95_seconds"):
            value = summary.get(key)
            if value is not None and (math.isnan(value) or math.isinf(value)):
                nan_metrics = True
    record(
        "no_nan_metrics",
        "no NaN or infinite latency/token metrics",
        not nan_metrics,
        {"gpu_peak_mib": gpu.get("peak_mib"), "gpu_sampled": gpu.get("sampled")},
    )

    if restart_results is not None:
        restart_ok = bool(restart_results.get("ready"))
        restart_levels = (restart_results.get("levels") or {}).get("restart_subset", [])
        restart_summary = summarize_level(restart_levels)
        restart_complete = restart_summary["cases_complete"] == RESTART_SUBSET_SIZE
        restart_valid = restart_summary["json_valid"] == RESTART_SUBSET_SIZE and (
            restart_summary["schema_valid"] == RESTART_SUBSET_SIZE
        )
        restart_errors = restart_summary["request_errors"] == 0
        restart_labels = restart_summary["label_correct"] >= math.ceil(
            MIN_LABEL_ACCURACY_C1 * RESTART_SUBSET_SIZE
        )
        record(
            "restart_subset",
            "server restart with offline variables and 8-case subset passes",
            restart_ok and restart_complete and restart_valid and restart_errors and restart_labels,
            {
                "ready": restart_ok,
                "cases_complete": restart_summary["cases_complete"],
                "json_valid": restart_summary["json_valid"],
                "schema_valid": restart_summary["schema_valid"],
                "request_errors": restart_summary["request_errors"],
                "label_correct": restart_summary["label_correct"],
            },
        )
    else:
        record("restart_subset", "server restart and 8-case subset", False, "no restart results")

    record(
        "model_revision_match",
        "model revision matches deployment record",
        model_revision == deployment_model_revision,
        {"actual": model_revision, "expected": deployment_model_revision},
    )
    record(
        "environment_match",
        "environment versions match deployment record",
        environment_versions == deployment_environment_versions,
        {"actual": environment_versions, "expected": deployment_environment_versions},
    )
    record(
        "model_manifest_match",
        "model manifest hash matches deployment record",
        model_manifest_sha256 is not None
        and model_manifest_sha256 == deployment_model_manifest_sha256,
        {"actual": model_manifest_sha256, "expected": deployment_model_manifest_sha256},
    )
    record(
        "wheelhouse_manifest_match",
        "wheelhouse manifest hash matches deployment record",
        wheelhouse_manifest_sha256 is not None
        and wheelhouse_manifest_sha256 == deployment_wheelhouse_manifest_sha256,
        {"actual": wheelhouse_manifest_sha256, "expected": deployment_wheelhouse_manifest_sha256},
    )

    metrics = {
        "startup_seconds": startup_seconds,
        "gpu_peak_mib": gpu.get("peak_mib"),
        "gpu_sampled": gpu.get("sampled"),
        "levels": {name: summarize_level(levels[name]) for name in expected_levels if name in levels},
    }
    c1_summary = metrics["levels"].get("c1_pass_a", {})
    c8_summary = metrics["levels"].get("c8", {})
    metrics["requests_per_second_c1"] = c1_summary.get("aggregate_requests_per_second")
    metrics["requests_per_second_c8"] = c8_summary.get("aggregate_requests_per_second")
    return {"passed": passed, "checks": checks, "metrics": metrics, "summary": level_summaries}
