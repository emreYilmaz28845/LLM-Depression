#!/usr/bin/env python3
"""Render the Worker 4 legacy/current prompt contract for all 15 Native routes.

This audit is public-safe by construction: it renders only a fixed synthetic
transcript and reads only tracked configs and archived/marked prompt sources.
No study data, subject identifiers, manifests or model files are read.

For every dataset/modality route it records the rendered system and user
prompts and their sha256 hashes for:

- the legacy-prompt treatment config (``configs/main/*_legacyprompt_v1*``);
- its canonical promptcontext_v1 control config.

Fail-closed checks:

- the treatment config resolves no ``prompt.version`` and renders the exact
  inline ``prompt.system`` of its source-grounded legacy reference;
- the treatment user prompt equals the legacy reference rendering on the same
  synthetic transcript (both Turkish question conditions when tagged);
- the control config still resolves ``promptcontext_v1`` and its rendered
  prompt differs from the treatment (the prompts are genuinely different arms).

The output is deterministic and contains no timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prompt_context import (
    resolve_prompt_context_version,
    resolve_question_context_sentences,
    resolve_system_prompt,
)
from src.data.runtime import render_user_prompt_text

from scripts.build_qwen3_legacy_prompt_configs import (
    control_name,
    legacy_name,
    treatment_name,
)

MAIN = ROOT / "configs/main"
LEGACY_ARCHIVE = ROOT / "configs/archive/pre_default_backbone_20260923"

SCHEMA_VERSION = "audiollm.worker4_prompt_render_audit.v1"
CAMPAIGN = "qwen3_legacy_prompt_20261008"
SYNTHETIC_TRANSCRIPT = "Synthetic render-audit transcript only; no study data."
SYNTHETIC_AUDIO_CONTEXT = "SYNTHETIC_AUDIO_CONTEXT_BLOCK"

# ``render_user_prompt_text`` question keys use the dataset_variant vocabulary.
QUESTION_CONDITION_KEYS = {
    "positive_only_t17": "pos_only_t17",
    "negative_only_t17": "negative_only_t17",
}

LEGACY_SYSTEM = {
    "text_only": (
        "You are a psychologist analyzing transcript information for depression screening."
    ),
    "audio_only": "You are a psychologist analyzing speech audio for depression screening.",
    "audio_text": (
        "You are a psychologist analyzing speech and transcript information "
        "for depression screening."
    ),
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def legacy_reference_path(dataset: str, modality: str) -> Path:
    if dataset == "turkish":
        return MAIN / legacy_name(dataset, modality)
    return LEGACY_ARCHIVE / legacy_name(dataset, modality)


def route_entry(dataset: str, modality: str) -> dict[str, Any]:
    route = f"{dataset}_{modality}"
    treatment_path = MAIN / treatment_name(dataset, modality)
    control_path = MAIN / control_name(dataset, modality)
    legacy_path = legacy_reference_path(dataset, modality)

    failures: list[str] = []
    if not treatment_path.is_file():
        raise SystemExit(f"missing treatment config: {treatment_path}")
    treatment = load_yaml(treatment_path)
    control = load_yaml(control_path)
    legacy = load_yaml(legacy_path)

    if resolve_prompt_context_version(treatment) is not None:
        failures.append("treatment resolves a prompt version")
    if str(treatment["prompt"].get("system", "")).strip() != LEGACY_SYSTEM[modality]:
        failures.append("treatment system differs from the recovered modality wording")

    treatment_user: dict[str, str] = {}
    control_user: dict[str, str] = {}
    legacy_user: dict[str, str] = {}
    if "{question_context}" in str(treatment["prompt"]["user_template"]):
        for label, key in QUESTION_CONDITION_KEYS.items():
            treatment_user[label] = render_user_prompt_text(
                treatment, SYNTHETIC_TRANSCRIPT, question_condition=key
            )
            control_user[label] = render_user_prompt_text(
                control, SYNTHETIC_TRANSCRIPT, question_condition=key
            )
            legacy_user[label] = render_user_prompt_text(
                legacy, SYNTHETIC_TRANSCRIPT, question_condition=key
            )
        expected_sentences = resolve_question_context_sentences(treatment)
        expected_legacy = resolve_question_context_sentences(legacy)
        if expected_sentences != expected_legacy:
            failures.append("treatment question sentences differ from the legacy reference")
    else:
        treatment_user["untagged"] = render_user_prompt_text(treatment, SYNTHETIC_TRANSCRIPT)
        control_user["untagged"] = render_user_prompt_text(control, SYNTHETIC_TRANSCRIPT)
        legacy_user["untagged"] = render_user_prompt_text(legacy, SYNTHETIC_TRANSCRIPT)

    question_tagged = "{question_context}" in str(treatment["prompt"]["user_template"])
    for label in treatment_user:
        if treatment_user[label] != legacy_user[label]:
            failures.append(f"treatment user prompt differs from legacy reference ({label})")
    if question_tagged:
        # The Turkish pooled arm also changes the question-context sentence set,
        # so the user prompt must differ from the control's.
        for label in treatment_user:
            if treatment_user[label] == control_user[label]:
                failures.append(f"treatment and control user prompts are identical ({label})")

    control_system = resolve_system_prompt(control)
    if resolve_prompt_context_version(control) != "promptcontext_v1":
        failures.append("control does not resolve promptcontext_v1")
    if control_system == LEGACY_SYSTEM[modality]:
        failures.append("control system equals the legacy wording")

    return {
        "route": route,
        "dataset": dataset,
        "modality": modality,
        "treatment_config": str(treatment_path.relative_to(ROOT)),
        "control_config": str(control_path.relative_to(ROOT)),
        "legacy_reference": str(legacy_path.relative_to(ROOT)),
        "treatment": {
            "system_prompt": resolve_system_prompt(treatment),
            "system_prompt_sha256": sha256(resolve_system_prompt(treatment)),
            "user_prompts": {
                label: {"text": text, "sha256": sha256(text)}
                for label, text in treatment_user.items()
            },
        },
        "control": {
            "system_prompt": control_system,
            "system_prompt_sha256": sha256(control_system),
            "user_prompts": {
                label: {"text": text, "sha256": sha256(text)}
                for label, text in control_user.items()
            },
        },
        "failures": failures,
    }


def build_audit() -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in ("daic", "d3tec", "androids", "cmdc", "turkish"):
        for modality in ("text_only", "audio_only", "audio_text"):
            entry = route_entry(dataset, modality)
            routes.append(entry)
            failures.extend(f"{entry['route']}: {item}" for item in entry["failures"])
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "synthetic_inputs": {
            "transcript": SYNTHETIC_TRANSCRIPT,
            "audio_context_block": SYNTHETIC_AUDIO_CONTEXT,
        },
        "expected_legacy_system": LEGACY_SYSTEM,
        "routes": routes,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / CAMPAIGN / "prompt_render_audit.json"),
    )
    args = parser.parse_args()
    audit = build_audit()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if audit["failures"]:
        print("Prompt render audit FAILED:")
        for failure in audit["failures"]:
            print(f"- {failure}")
        return 1
    print(f"Prompt render audit passed for {len(audit['routes'])} routes -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
