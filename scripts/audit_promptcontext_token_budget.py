#!/usr/bin/env python3
"""Token-budget audit for the prompt-context family.

Measures the longest available rendered prompt per dataset against the model's
context limit, using the real Qwen3.8 tokenizer when it is locally available and
the locally built manifests as the audit input. When the tokenizer or a dataset's
manifest is missing, the row is recorded as not measured with the reason instead
of being guessed, and the summary says the measurement is a proxy.

The audit never writes transcript text or subject identifiers: only counts,
lengths and hashes. Output: ``outputs/prompt_context_token_budget/token_budget_audit.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prompt_context import resolve_system_prompt  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/prompt_context_token_budget/token_budget_audit.json"
DEFAULT_MODEL_DIR = Path("/media/emre/Backup/AudioLLM/models/Qwen3.8-27B")
MATRIX = PROJECT_ROOT / "configs/experiments/promptcontext_qwen38/matrix.yaml"
QWEN38_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SAFETY_MARGIN_TOKENS = 1024

# cell_id -> manifest directory name under the harmonized manifest root. The
# pooled cell is measured from both source conditions.
MANIFEST_INPUTS = {
    "daic": "daic",
    "d3tec": "d3tec",
    "androids": "androids",
    "cmdc": "cmdc",
    "turkish_pooled": "turkish_pos_only_t17_qwen3asr",
    "turkish_pooled_negative": "turkish_t17_qwen3asr",
}


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return tokenizer


def count_tokens(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids)


def _longest_subject_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the longest subjects, or every subject when ``limit`` is not positive."""
    if limit <= 0:
        return list(rows)
    lengths: dict[str, int] = {}
    for row in rows:
        subject = str(row["subject_id"])
        text = str(
            row.get("full_participant_transcript")
            or row.get("full_subject_transcript")
            or row.get("full_response_transcript")
            or row.get("transcript")
            or ""
        )
        lengths[subject] = lengths.get(subject, 0) + len(text)
    keep = {
        subject
        for subject, _ in sorted(lengths.items(), key=lambda item: (-item[1], item[0]))[:limit]
    }
    return [row for row in rows if str(row["subject_id"]) in keep]


def _pooled_condition_prompts(
    config: dict[str, Any], rows: list[dict[str, Any]], condition: str
) -> list[tuple[str, str]]:
    """Render one pooled text-only example per subject for a single condition.

    The pooled runner builds one example per participant per condition, and its
    pair guard refuses a manifest that carries only one condition. A source
    manifest is exactly that, so the audit renders the same prompt the pair
    builder renders for this condition: the subject's concatenated transcript
    through the pooled user template.
    """
    from src.data.runtime import (
        _harmonized_subject_transcripts,
        _ordered_subject_rows,
        build_prompt_text,
        render_user_prompt_text,
    )
    from src.utils import internal_label_text_from_int

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)
    rendered: list[tuple[str, str]] = []
    for subject_id in sorted(grouped):
        subject_rows = _ordered_subject_rows(grouped[subject_id])
        transcript = _harmonized_subject_transcripts(subject_rows, "turkish").get(subject_id, "")
        if not transcript:
            continue
        user_text = render_user_prompt_text(
            config, transcript, question_condition=condition
        )
        prompt_text = build_prompt_text(
            system_prompt=resolve_system_prompt(config),
            user_text=user_text,
            num_audios=0,
            use_audio=False,
        )
        label_text = internal_label_text_from_int(config, int(subject_rows[0]["label"]))
        rendered.append((prompt_text + label_text, subject_id))
    return rendered


def measure_dataset(
    *,
    cell_id: str,
    config: dict[str, Any],
    manifest_dir: Path,
    tokenizer: Any,
    context_limit: int,
    pooled_condition: str | None = None,
    subject_limit: int = 0,
) -> dict[str, Any]:
    from src.data.runtime import build_examples
    from src.utils import read_jsonl

    manifest_path = manifest_dir / f"{config['dataset']}_manifest.jsonl"
    if not manifest_path.is_file():
        return {
            "cell_id": cell_id,
            "dataset": config["dataset"],
            "measured": False,
            "reason": f"local manifest not available: {manifest_path}",
        }
    rows = read_jsonl(manifest_path)
    selected = _longest_subject_rows(rows, subject_limit)
    if pooled_condition is not None:
        rendered = _pooled_condition_prompts(config, selected, pooled_condition)
        if not rendered:
            return {
                "cell_id": cell_id,
                "dataset": config["dataset"],
                "measured": False,
                "reason": "no rendered pooled condition examples for the selected subjects",
            }
        counts = [count_tokens(tokenizer, prompt) for prompt, _subject in rendered]
        subject_count = len({subject for _prompt, subject in rendered})
        example_count = len(rendered)
    else:
        examples = build_examples(selected, config, partition_name="token_budget_audit")
        if not examples:
            return {
                "cell_id": cell_id,
                "dataset": config["dataset"],
                "measured": False,
                "reason": "no rendered examples for the selected subjects",
            }
        counts = [
            count_tokens(tokenizer, example["prompt_text"] + example["internal_label_text"])
            for example in examples
        ]
        subject_count = len({str(example["subject_id"]) for example in examples})
        example_count = len(examples)
    maximum = max(counts)
    return {
        "cell_id": cell_id,
        "dataset": config["dataset"],
        "dataset_variant": str(config.get("dataset_variant", "") or ""),
        "input_modality": "text_only",
        "measured": True,
        "measured_subjects": subject_count,
        "measured_examples": example_count,
        "max_prompt_tokens": maximum,
        "min_prompt_tokens": min(counts),
        "context_limit_tokens": context_limit,
        "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
        "exceeds_context_limit": maximum + SAFETY_MARGIN_TOKENS > context_limit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/manifests_harmonized",
        help="root holding the locally built per-dataset manifest directories",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="explicit tokenizer directory (default: --model-dir)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="measure only the N longest subjects per dataset (0 = every subject)",
    )
    args = parser.parse_args(argv)

    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else args.model_dir
    audit: dict[str, Any] = {
        "schema_version": "audiollm.promptcontext_token_budget.v1",
        "prompt_context_version": matrix["prompt_context_version"],
        "tokenizer_identity": {},
        "model_context_limit_tokens": None,
        "measurement_kind": None,
        "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
        "results": [],
    }
    try:
        tokenizer = load_tokenizer(tokenizer_dir)
        audit["tokenizer_identity"] = {
            "source": "local_huggingface_tokenizer",
            "path": str(tokenizer_dir),
            "class": type(tokenizer).__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
            "model_revision": QWEN38_MODEL_REVISION,
        }
        audit["model_context_limit_tokens"] = int(
            getattr(tokenizer, "model_max_length", 0) or 0
        )
        audit["measurement_kind"] = "real_qwen38_tokenizer"
    except Exception as exc:  # noqa: BLE001 - a missing tokenizer is a recorded reason
        audit["measurement_kind"] = "not_measured"
        audit["tokenizer_error"] = f"{type(exc).__name__}: {exc}"

    for cell in matrix["experiments"]:
        config = yaml.safe_load((PROJECT_ROOT / cell["config"]).read_text(encoding="utf-8"))
        cell_id = cell["cell_id"]
        inputs = MANIFEST_INPUTS.get(cell_id)
        if audit["measurement_kind"] != "real_qwen38_tokenizer":
            audit["results"].append(
                {
                    "cell_id": cell_id,
                    "dataset": config["dataset"],
                    "measured": False,
                    "reason": "the Qwen3.8 tokenizer is not locally available",
                }
            )
            continue
        if inputs is None:
            audit["results"].append(
                {
                    "cell_id": cell_id,
                    "dataset": config["dataset"],
                    "measured": False,
                    "reason": "no local audit input configured for this cell",
                }
            )
            continue
        pooled_condition = (
            "pos_only_t17"
            if cell_id == "turkish_pooled"
            else ("negative_only_t17" if cell_id == "turkish_pooled_negative" else None)
        )
        audit["results"].append(
            measure_dataset(
                cell_id=cell_id,
                config=config,
                manifest_dir=Path(args.manifest_root) / inputs,
                tokenizer=tokenizer,
                context_limit=int(audit["model_context_limit_tokens"] or 0),
                pooled_condition=pooled_condition,
                subject_limit=args.max_subjects,
            )
        )

    # The pooled cell's second source condition is measured explicitly: the
    # matrix carries one pooled cell, whose manifest comes from both sources.
    pooled_cells = [
        cell for cell in matrix["experiments"] if cell["cell_id"] == "turkish_pooled"
    ]
    if pooled_cells and audit["measurement_kind"] == "real_qwen38_tokenizer":
        pooled_config = yaml.safe_load(
            (PROJECT_ROOT / pooled_cells[0]["config"]).read_text(encoding="utf-8")
        )
        audit["results"].append(
            measure_dataset(
                cell_id="turkish_pooled_negative",
                config=pooled_config,
                manifest_dir=Path(args.manifest_root)
                / MANIFEST_INPUTS["turkish_pooled_negative"],
                tokenizer=tokenizer,
                context_limit=int(audit["model_context_limit_tokens"] or 0),
                pooled_condition="negative_only_t17",
                subject_limit=args.max_subjects,
            )
        )

    # The pooled cell carries both source conditions; its row reports the worst
    # condition, and the helper row for the negative sources is folded into it.
    pooled = [row for row in audit["results"] if row["cell_id"] == "turkish_pooled"]
    negative = [row for row in audit["results"] if row["cell_id"] == "turkish_pooled_negative"]
    audit["results"] = [
        row for row in audit["results"] if row["cell_id"] != "turkish_pooled_negative"
    ]
    if pooled and negative:
        condition_rows = {
            "positive_sources": dict(pooled[0]),
            "negative_sources": dict(negative[0]),
        }
        pooled[0]["condition_rows"] = condition_rows
        measured_conditions = [
            row for row in condition_rows.values() if row.get("measured")
        ]
        if measured_conditions:
            worst = max(measured_conditions, key=lambda row: row["max_prompt_tokens"])
            pooled[0].update(
                {
                    "measured": True,
                    "max_prompt_tokens": worst["max_prompt_tokens"],
                    "min_prompt_tokens": min(
                        row["min_prompt_tokens"] for row in measured_conditions
                    ),
                    "measured_subjects": sum(
                        row["measured_subjects"] for row in measured_conditions
                    ),
                    "measured_examples": sum(
                        row["measured_examples"] for row in measured_conditions
                    ),
                    "longest_condition": worst["cell_id"],
                    "exceeds_context_limit": any(
                        row["exceeds_context_limit"] for row in measured_conditions
                    ),
                }
            )
            pooled[0].pop("reason", None)
        else:
            pooled[0]["measured"] = False
            pooled[0]["reason"] = negative[0].get("reason", "not measured")
    measured = [row for row in audit["results"] if row.get("measured")]
    audit["measured_cells"] = len(measured)
    audit["unmeasured_cells"] = [
        {"cell_id": row["cell_id"], "reason": row.get("reason", "")}
        for row in audit["results"]
        if not row.get("measured")
    ]
    audit["any_cell_exceeds_context_limit"] = any(
        row.get("exceeds_context_limit") for row in measured
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        f"measurement_kind={audit['measurement_kind']} "
        f"measured_cells={audit['measured_cells']} "
        f"exceeds_limit={audit['any_cell_exceeds_context_limit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
