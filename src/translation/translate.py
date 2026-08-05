"""Deterministic, resumable Qwen3.6-27B translation worker.

Connects to a vLLM OpenAI-compatible endpoint (BF16, tensor-parallel=2,
text-only, deterministic non-thinking generation) and translates the exported
units. Candidate records are flushed incrementally; resume is allowed only when
unit IDs and source hashes match, changed sources are regenerated, duplicate
keys are rejected, and an incompatible completed run is never overwritten
without ``--force-resync``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt
from src.utils import (
    configure_logging,
    get_logger,
    read_jsonl,
    resolve_project_path,
    sha256_text,
    write_jsonl,
)


LOGGER = get_logger(__name__)

PRECISION = "bf16"

DEFAULT_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Response is not valid JSON")


def _request_messages(unit: dict[str, Any], corrective: bool = False) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(unit, corrective=corrective)},
    ]


def _openai_client(base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key="EMPTY")


def _estimate_max_tokens(unit: dict[str, Any]) -> int:
    return max(512, min(4096, int(len(unit["source_text"]) * 1.6)))


def translate_batch(
    client: OpenAI,
    model: str,
    units: list[dict[str, Any]],
    *,
    seed: int,
    max_retries: int,
    model_identity: str = "Qwen/Qwen3.6-27B",
) -> list[tuple[dict[str, Any], str | None]]:
    results: list[tuple[dict[str, Any], str | None]] = []
    for unit in units:
        translation: str | None = None
        reason: str | None = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=_request_messages(unit, corrective=attempt > 0),
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=_estimate_max_tokens(unit),
                    seed=seed,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    reason = "empty_response"
                    continue
                parsed = _parse_json_object(content)
                translation = str(parsed.get("translation", "")).strip()
                if not translation:
                    reason = "missing_translation_key"
                    continue
                break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                continue
        if translation is None:
            results.append((unit, reason))
            continue
        results.append(
            (
                {
                    "dataset": unit["dataset"],
                    "unit_id": unit["unit_id"],
                    "field": unit["field"],
                    "part_index": unit["part_index"],
                    "part_count": unit["part_count"],
                    "translation": translation,
                    "translation_sha256": sha256_text(translation),
                    "model": model_identity,
                    "model_revision": unit.get("_model_revision", ""),
                    "precision": PRECISION,
                    "prompt_version": PROMPT_VERSION,
                    "source_sha256": unit["source_sha256"],
                    "status": "translated",
                },
                None,
            )
        )
    return results


def _load_units(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0)))
        if key in seen:
            raise ValueError(f"Duplicate unit key in {path}: {key}")
        seen.add(key)
    return rows


def _load_candidates(path: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row["unit_id"]), str(row["field"]), int(row.get("part_index", 0)))
        if key in index:
            raise ValueError(f"Duplicate candidate key in {path}: {key}")
        index[key] = row
    return index


def run_translation(
    units_path: str | Path,
    candidates_path: str | Path,
    failed_path: str | Path,
    *,
    base_url: str,
    model: str,
    model_revision: str,
    batch_size: int,
    seed: int,
    max_retries: int,
    force_resync: bool,
) -> dict[str, Any]:
    units = _load_units(resolve_project_path(units_path))
    for unit in units:
        unit["_model_revision"] = model_revision
    candidates = _load_candidates(resolve_project_path(candidates_path))

    pending: list[dict[str, Any]] = []
    skipped = 0
    for unit in units:
        key = (unit["unit_id"], unit["field"], unit["part_index"])
        existing = candidates.get(key)
        if existing is None:
            pending.append(unit)
            continue
        if existing.get("source_sha256") != unit["source_sha256"]:
            if not force_resync:
                raise ValueError(
                    f"Resume conflict for {key}: candidate source hash "
                    f"{existing.get('source_sha256')} does not match unit hash "
                    f"{unit['source_sha256']}. Refusing to overwrite an "
                    "incompatible completed run; pass --force-resync to "
                    "regenerate changed sources."
                )
            pending.append(unit)
            continue
        if existing.get("status") != "translated":
            pending.append(unit)
            continue
        skipped += 1

    LOGGER.info(
        "Resume state: units=%s done=%s pending=%s",
        len(units),
        skipped,
        len(pending),
    )

    if pending:
        client = _openai_client(base_url)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = translate_batch(
                client,
                model,
                batch,
                seed=seed,
                max_retries=max_retries,
            )
            for unit, reason in results:
                key = (unit["unit_id"], unit["field"], unit["part_index"])
                if reason is None:
                    candidate = dict(unit)
                    candidate.pop("_model_revision", None)
                    candidate["model_revision"] = model_revision
                    candidates[key] = candidate
                else:
                    failed = {
                        "dataset": unit["dataset"],
                        "unit_id": unit["unit_id"],
                        "field": unit["field"],
                        "part_index": unit["part_index"],
                        "part_count": unit["part_count"],
                        "source_sha256": unit["source_sha256"],
                        "status": "failed",
                        "reason": reason,
                    }
                    LOGGER.warning("Failed unit %s: %s", key, reason)
                    _append_failed(failed_path, failed)
            write_jsonl(
                sorted(candidates.values(), key=lambda row: (row["unit_id"], row["part_index"])),
                candidates_path,
            )

    candidates = _load_candidates(resolve_project_path(candidates_path))
    return {
        "units": len(units),
        "completed": sum(1 for row in candidates.values() if row.get("status") == "translated"),
        "failed": _count_failed(failed_path),
        "skipped_on_resume": skipped,
    }


def _append_failed(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_failed(path: str | Path) -> int:
    path = Path(path)
    if not path.is_file():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate exported units with Qwen3.6-27B via vLLM.")
    parser.add_argument("--units", required=True, help="Units JSONL produced by src.translation.units.")
    parser.add_argument("--out", required=True, help="Candidates JSONL (incremental flush, resumable).")
    parser.add_argument("--failed", required=True, help="Failed records JSONL.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--force-resync", action="store_true", help="Regenerate candidates whose source changed.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    summary = run_translation(
        args.units,
        args.out,
        args.failed,
        base_url=args.base_url,
        model=args.model,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        seed=args.seed,
        max_retries=args.max_retries,
        force_resync=args.force_resync,
    )
    LOGGER.info(
        "Translation summary: completed=%s failed=%s skipped=%s",
        summary["completed"],
        summary["failed"],
        summary["skipped_on_resume"],
    )
    if summary["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
