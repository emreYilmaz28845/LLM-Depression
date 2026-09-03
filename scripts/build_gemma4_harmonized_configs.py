#!/usr/bin/env python3
"""Generate the 20 Gemma 4 harmonized standalone configs from their Qwen bases.

Native family (12): four non-DAIC datasets x three modalities.
English family (8): four datasets x (audio+text, text-only).

The Gemma config differs from its Qwen counterpart only in the backend
allowlist: ``model_backend``, ``model_name_or_path`` (pinned revision path),
``model_revision``, ``output_dirs.run_root``, ``lora.target_modules`` (exact
288-module decoder regex), and ``evaluation.evaluation_view``.

Manifest/split dirs and every scientific field are inherited byte-for-byte so
Qwen and Gemma consume identical manifests, splits, prompts, labels, weights,
and preprocessing for each paired cell.

The script is deterministic and idempotent: it refuses to overwrite an
existing file whose content differs from the derived content.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "configs/main"

GEMMA4_MODEL_PATH = (
    "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/"
    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
)
GEMMA4_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA4_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
GEMMA4_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"

NATIVE_BASES = {
    "d3tec": ("audio_text", "audio_only", "text_only"),
    "androids": ("audio_text", "audio_only", "text_only"),
    "cmdc": ("audio_text", "audio_only", "text_only"),
    "turkish_pos_only_t17": ("audio_text", "audio_only", "text_only"),
}
ENGLISH_BASES = {
    "d3tec": ("audio_text", "text_only"),
    "androids": ("audio_text", "text_only"),
    "cmdc": ("audio_text", "text_only"),
    "turkish_pos_only_t17": ("audio_text", "text_only"),
}


def qwen_config_name(dataset: str, modality: str, english: bool) -> str:
    suffix = "_en" if english else ""
    tail = "qwen3asr" if dataset == "turkish_pos_only_t17" else None
    name = f"{dataset}_{modality}_harmonized_selmacrof1_tf"
    if tail:
        name += f"_{tail}"
    return f"{name}{suffix}.yaml"


def gemma_config_name(dataset: str, modality: str, english: bool) -> str:
    suffix = "_en" if english else ""
    tail = "qwen3asr" if dataset == "turkish_pos_only_t17" else None
    name = f"{dataset}_{modality}_harmonized_selmacrof1_tf"
    if tail:
        name += f"_{tail}"
    return f"{name}{suffix}_gemma4_12b.yaml"


def derive_gemma_config(source: dict[str, object], english: bool) -> dict[str, object]:
    config = copy.deepcopy(source)
    config["model_backend"] = "gemma4"
    config["model_name_or_path"] = "${GEMMA4_MODEL_PATH:-" + GEMMA4_MODEL_PATH + "}"
    config["model_revision"] = GEMMA4_REVISION
    run_root = str(config["output_dirs"]["run_root"])
    old_root = "output_model/harmonized_v1_en" if english else "output_model/harmonized_v1"
    new_root = "output_model/harmonized_v1_en_gemma4" if english else "output_model/harmonized_v1_gemma4"
    if old_root not in run_root:
        raise ValueError(f"Cannot derive Gemma run root from {run_root!r}.")
    config["output_dirs"]["run_root"] = run_root.replace(old_root, new_root, 1)
    config["lora"]["target_modules"] = GEMMA4_LORA_TARGET_REGEX
    config["evaluation"]["evaluation_view"] = GEMMA4_EVALUATION_VIEW
    return config


def main() -> int:
    check_only = "--check" in sys.argv
    failures: list[str] = []
    written: list[Path] = []
    for english, bases in ((False, NATIVE_BASES), (True, ENGLISH_BASES)):
        for dataset, modalities in bases.items():
            for modality in modalities:
                source_name = qwen_config_name(dataset, modality, english)
                target_name = gemma_config_name(dataset, modality, english)
                source_path = MAIN / source_name
                target_path = MAIN / target_name
                if not source_path.is_file():
                    failures.append(f"missing Qwen base: {source_path}")
                    continue
                source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
                derived = derive_gemma_config(source, english)
                rendered = yaml.safe_dump(
                    derived, sort_keys=False, default_flow_style=False, allow_unicode=True
                )
                if target_path.is_file():
                    existing = target_path.read_text(encoding="utf-8")
                    if existing != rendered:
                        failures.append(
                            f"existing Gemma config differs from derived content: {target_path}"
                        )
                    continue
                if not check_only:
                    target_path.write_text(rendered, encoding="utf-8")
                    written.append(target_path)
                    print(f"wrote {target_path.name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"ok: {len(written)} written, all derived configs consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
