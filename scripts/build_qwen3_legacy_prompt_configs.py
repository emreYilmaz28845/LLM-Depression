#!/usr/bin/env python3
"""Build the Worker 4 legacy-prompt treatment configs.

The legacy-prompt versus promptcontext_v1 comparison needs one treatment arm:
the historical harmonized inline prompt on the current Qwen3 contracts. The
treatment configs are derived deterministically from the current canonical
promptcontext_v1 control configs; only the prompt block, ``recipe_id`` and
``output_dirs.run_root`` change.

Legacy prompt text is source-grounded, never hardcoded here:

- D3TEC, Androids, DAIC and CMDC: the archived harmonized inline prompt blocks in
  ``configs/archive/pre_default_backbone_20260923`` (read-only references; the
  archived configs are never executed).
- Turkish pooled: the pre-promptcontext pooled inline prompt blocks in
  ``configs/main/turkish_pooled_t17_*_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml``.

The treatment keeps the exact historical inline ``prompt.system`` with no
``prompt.version``, the control's ``user_template`` and ``prompt_language``;
for Turkish that means the legacy question-context sentences render because no
``prompt.question_context_version`` is declared. Everything else (backends,
model revisions, data, splits, LoRA, training, evaluation, resources) is copied
from the control byte-for-byte in value, so the only scientific change is the
rendered prompt.

Use ``--check`` to fail when the generated files are missing or stale.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prompt_context import (
    LEGACY_QUESTION_CONTEXT_SENTENCES,
    resolve_prompt_context_version,
    resolve_question_context_sentences,
    resolve_system_prompt,
)

MAIN = ROOT / "configs/main"
LEGACY_ARCHIVE = ROOT / "configs/archive/pre_default_backbone_20260923"

CAMPAIGN = "qwen3_legacy_prompt_20261008"
MARKER = "legacyprompt_v1"

MODALITIES = ("text_only", "audio_only", "audio_text")

# dataset key -> output dataset directory used by the managed run layout
DATASET_DIRS = {
    "daic": "daic",
    "d3tec": "d3tec",
    "androids": "androids_interview",
    "cmdc": "cmdc",
    "turkish": "turkish",
}

NON_TURKISH = ("daic", "d3tec", "androids", "cmdc")


def control_name(dataset: str, modality: str) -> str:
    if dataset in NON_TURKISH:
        return f"{dataset}_{modality}_harmonized_selmacrof1_likelihood_v1.yaml"
    if modality == "text_only":
        return (
            "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
            "_promptcontext_v1_qwen38_27b.yaml"
        )
    return (
        f"turkish_pooled_t17_{modality}_harmonized_selmacrof1_likelihood_v1"
        "_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    )


def legacy_name(dataset: str, modality: str) -> str:
    """The read-only historical reference that carries the legacy prompt."""
    if dataset in NON_TURKISH:
        return f"{dataset}_{modality}_harmonized_selmacrof1_likelihood_v1.yaml"
    return f"turkish_pooled_t17_{modality}_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml"


def treatment_name(dataset: str, modality: str) -> str:
    if dataset in NON_TURKISH:
        return f"{dataset}_{modality}_harmonized_selmacrof1_likelihood_v1_{MARKER}.yaml"
    if modality == "text_only":
        return (
            "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
            f"_{MARKER}_qwen38_27b.yaml"
        )
    return (
        f"turkish_pooled_t17_{modality}_harmonized_selmacrof1_likelihood_v1"
        f"_qwen3asr_{MARKER}_qwen3omni_30b_a3b.yaml"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def with_marker(recipe_id: str) -> str:
    if not recipe_id.endswith("_promptcontext_v1"):
        raise ValueError(
            f"control recipe_id {recipe_id!r} does not end with _promptcontext_v1; "
            "the legacy-prompt derivation expects a promptcontext control."
        )
    return recipe_id[: -len("_promptcontext_v1")] + f"_{MARKER}"


def derive(dataset: str, modality: str) -> dict[str, Any]:
    control_path = MAIN / control_name(dataset, modality)
    legacy_path = (
        LEGACY_ARCHIVE / legacy_name(dataset, modality)
        if dataset in NON_TURKISH
        else MAIN / legacy_name(dataset, modality)
    )
    control = load_yaml(control_path)
    legacy = load_yaml(legacy_path)

    control_prompt = control["prompt"]
    legacy_prompt = legacy["prompt"]

    if str(legacy_prompt.get("version", "")).strip():
        raise ValueError(f"{legacy_path.name}: legacy reference must not declare prompt.version")
    legacy_system = str(legacy_prompt.get("system", "")).strip()
    if not legacy_system:
        raise ValueError(f"{legacy_path.name}: legacy reference has no prompt.system")
    if legacy_prompt.get("user_template") != control_prompt.get("user_template"):
        raise ValueError(
            f"{legacy_path.name}: legacy user_template differs from the control; "
            "the whole-prompt comparison would not be source-grounded."
        )

    config = copy.deepcopy(control)
    config["recipe_id"] = with_marker(str(control["recipe_id"]))
    config["prompt"] = {
        "system": legacy_system,
        "user_template": control_prompt["user_template"],
        "prompt_language": control_prompt.get("prompt_language", "english"),
    }
    config["output_dirs"]["run_root"] = (
        "${PROJECT_ROOT}/output_model/"
        f"{CAMPAIGN}/{modality}/{DATASET_DIRS[dataset]}"
    )

    if resolve_prompt_context_version(config) is not None:
        raise ValueError("treatment config must not resolve a prompt context version")
    if resolve_system_prompt(config) != legacy_system:
        raise ValueError("treatment config does not render the legacy system prompt")
    if "{question_context}" in str(config["prompt"]["user_template"]):
        if resolve_question_context_sentences(config) != LEGACY_QUESTION_CONTEXT_SENTENCES:
            raise ValueError("treatment Turkish config must render the legacy question sentences")
    return config


def render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000)


def build(*, check: bool) -> int:
    mismatches: list[str] = []
    for dataset in (*NON_TURKISH, "turkish"):
        for modality in MODALITIES:
            expected = render(derive(dataset, modality))
            target = MAIN / treatment_name(dataset, modality)
            if check:
                if not target.is_file() or target.read_text(encoding="utf-8") != expected:
                    mismatches.append(str(target.relative_to(ROOT)))
            else:
                target.write_text(expected, encoding="utf-8")
    if mismatches:
        print("Legacy-prompt treatment config drift:")
        for path in mismatches:
            print(f"- {path}")
        return 1
    if check:
        print("Legacy-prompt treatment configs are current (15 files).")
    else:
        print("Wrote 15 legacy-prompt treatment configs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
