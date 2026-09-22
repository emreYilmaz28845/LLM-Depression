#!/usr/bin/env python3
"""Render every prompt-context cell so the exact prompt text is reviewable.

Each cell renders the system prompt and the user prompt through the same code
path a run uses, with a fixed placeholder transcript instead of participant
data: the audit shows the wording, not the data. Checks per cell: the selected
dataset context appears exactly once, no other dataset's context appears, and a
text-only prompt carries no audio claim and no audio placeholder.

Output: ``outputs/prompt_context_rendering/prompt_rendering_audit.json``.
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

from src.data.prompt_context import (  # noqa: E402
    DATASET_CONTEXT_BLOCKS,
    PROMPT_CONTEXT_VERSION,
    prompt_context_record,
    resolve_question_context_sentences,
)
from src.data.runtime import build_prompt_text, render_user_prompt_text  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/prompt_context_rendering/prompt_rendering_audit.json"
MATRIX = PROJECT_ROOT / "configs/experiments/promptcontext_qwen38/matrix.yaml"
PLACEHOLDER_TRANSCRIPT = "<participant transcript>"
PLACEHOLDER_LABEL = "<label>"
AUDIO_CLAIMS = (
    "speech audio",
    "audio window",
    "vocal feature",
    "The audio is",
    "audio is provided",
    "<|AUDIO|>",
    "<|audio_bos|>",
    "<|audio_start|>",
)
CONTEXT_MARKERS = {
    "androids": "This is an Italian interview with a human interviewer.",
    "d3tec": "This is Spanish speech from a non-interactive slideshow of 27 tasks.",
    "daic": "This is an English semi-structured interview with the virtual interviewer Ellie.",
    "cmdc": "This is Mandarin speech from a face-to-face, symptom-focused interview.",
    "turkish_pooled": "This is Turkish speech from one of two question sets completed by the same participants.",
}


def _audit_cell(cell: dict[str, Any], condition: str | None) -> dict[str, Any]:
    config = yaml.safe_load((PROJECT_ROOT / cell["config"]).read_text(encoding="utf-8"))
    record = prompt_context_record(config)
    system_prompt = record["system_prompt"]
    user_text = render_user_prompt_text(
        config, PLACEHOLDER_TRANSCRIPT, question_condition=condition
    )
    prompt_text = build_prompt_text(
        system_prompt=system_prompt, user_text=user_text, num_audios=0, use_audio=False
    )
    own_marker = CONTEXT_MARKERS[cell["prompt_context_dataset"]]
    checks = {
        "own_context_present_once": system_prompt.count(own_marker) == 1,
        "other_context_absent": all(
            marker not in system_prompt
            for key, marker in CONTEXT_MARKERS.items()
            if key != cell["prompt_context_dataset"]
        ),
        "no_audio_claim": all(
            claim not in f"{system_prompt}\n{user_text}" for claim in AUDIO_CLAIMS
        ),
        "no_audio_placeholder": "<|AUDIO|>" not in prompt_text
        and "<|audio_bos|>" not in prompt_text,
        "transcript_in_user_prompt": PLACEHOLDER_TRANSCRIPT in user_text,
        "label_contract_in_user_prompt": "Depressed" in user_text
        and "Non-depressed" in user_text,
    }
    entry: dict[str, Any] = {
        "cell_id": cell["cell_id"],
        "condition": condition,
        "config": cell["config"],
        "prompt_context": record,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "system_prompt": system_prompt,
        "user_prompt": user_text,
    }
    if condition is not None:
        sentences = resolve_question_context_sentences(config)
        entry["question_context_sentence"] = sentences[condition]
        entry["checks"]["condition_sentence_present"] = sentences[condition] in user_text
        other = "negative_only_t17" if condition == "pos_only_t17" else "pos_only_t17"
        entry["checks"]["other_condition_sentence_absent"] = sentences[other] not in user_text
        entry["all_checks_pass"] = all(entry["checks"].values())
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for cell in matrix["experiments"]:
        if cell["cell_id"] == "turkish_pooled":
            entries.append(_audit_cell(cell, "pos_only_t17"))
            entries.append(_audit_cell(cell, "negative_only_t17"))
        else:
            entries.append(_audit_cell(cell, None))

    audit = {
        "schema_version": "audiollm.promptcontext_rendering.v1",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "dataset_context_blocks": sorted(DATASET_CONTEXT_BLOCKS[PROMPT_CONTEXT_VERSION]),
        "transcript_redacted": True,
        "entries": entries,
        "all_checks_pass": all(entry["all_checks_pass"] for entry in entries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for entry in entries:
        condition = entry["condition"] or "-"
        print(
            f"{entry['cell_id']:<16} condition={condition:<16} "
            f"system_sha256={entry['system_prompt_sha256'][:12]} "
            f"checks={'ok' if entry['all_checks_pass'] else 'FAILED'}"
        )
    return 0 if audit["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
