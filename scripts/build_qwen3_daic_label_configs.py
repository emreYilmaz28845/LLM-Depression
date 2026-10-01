#!/usr/bin/env python3
"""Build the Qwen3 DAIC label-vocabulary configs from the canonical DAIC configs.

The campaign keeps the canonical harmonized DAIC recipe fixed and changes only the
answer-label vocabulary, so each generated config must equal its canonical source
except for exactly three things:

* ``labels`` - the vocabulary and its explicit positive/negative internal labels,
* ``recipe_id`` - the canonical recipe id plus the arm tag,
* ``output_dirs.run_root`` - the isolated campaign directory.

The training seed is a submission override (``--seed``), never a config
difference, and ``split.seed`` stays 1337 in every file. The script is
deterministic and supports ``--check`` so drift is caught before a submission.
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

from src.model.qwen38_lora import validate_qwen38_config
from src.model.qwen3omni_lora import validate_qwen3omni_config

MAIN = ROOT / "configs/main"
LABELS = ROOT / "configs/labels"

DATASET = "daic"
CAMPAIGN = "qwen3_daic_label_vocab_v1"
FOLDS = 0

# modality -> (canonical source file name, model token used in generated names, expected backend)
SOURCES: dict[str, tuple[str, str, str]] = {
    "text_only": (
        "daic_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "qwen38_27b",
        "qwen38",
    ),
    "audio_only": (
        "daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
        "qwen3omni_30b_a3b",
        "qwen3omni",
    ),
    "audio_text": (
        "daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
        "qwen3omni_30b_a3b",
        "qwen3omni",
    ),
}

# arm tag -> (label_vocab_version, internal positive label, internal negative label)
ARMS: dict[str, tuple[str, str, str]] = {
    "ab": ("short_internal_ab_labels", "A", "B"),
    "01": ("binary_01_labels", "1", "0"),
    "truefalse": ("truefalse_labels", "True", "False"),
    "yesno": ("yesno_labels", "Yes", "No"),
    "en": ("legacy_english_labels", "Depressed", "Non-depressed"),
}

ARM_ORDER = ("ab", "01", "truefalse", "yesno", "en")
# `training.gradient_accumulation_steps` and `resources.train_nodes` are allowed
# because this family pins its own four-rank shape (see `derive`), while the
# canonical DAIC audio source now declares the two-node audio-lane default.
ALLOWED_DIFF_KEYS = (
    "recipe_id",
    "labels",
    "output_dirs.run_root",
    "training.gradient_accumulation_steps",
    "resources.train_nodes",
)
# Sections whose content may change as a whole; the tests check their exact content.
OPAQUE_DIFF_KEYS = ("labels",)


class ConfigBuildError(RuntimeError):
    """Raised when a source config or a derived config breaks the contract."""


def generated_name(modality: str, tag: str) -> str:
    model = SOURCES[modality][1]
    return f"{DATASET}_{modality}_harmonized_selmacrof1_likelihood_{model}_{tag}_v1.yaml"


def labels_block(tag: str) -> dict[str, str]:
    vocab, positive, negative = ARMS[tag]
    return {
        "label_vocab_version": vocab,
        "internal_positive_label": positive,
        "internal_negative_label": negative,
        "external_positive_label": "Depressed",
        "external_negative_label": "Non-depressed",
    }


def derive(source: dict[str, Any], *, modality: str, tag: str) -> dict[str, Any]:
    config = copy.deepcopy(source)
    expected_backend = SOURCES[modality][2]
    if config.get("model_backend") != expected_backend:
        raise ConfigBuildError(
            f"{modality}: canonical source backend is {config.get('model_backend')!r}, expected {expected_backend!r}"
        )
    if config.get("dataset") != DATASET:
        raise ConfigBuildError(f"{modality}: canonical source dataset is {config.get('dataset')!r}")
    if config.get("prompt", {}).get("version") != "promptcontext_v1":
        raise ConfigBuildError(f"{modality}: canonical source is not promptcontext_v1")
    evaluation = config.get("evaluation", {})
    if evaluation.get("sample_prediction_mode") != "likelihood" or evaluation.get("headline_mode") != "likelihood":
        raise ConfigBuildError(f"{modality}: canonical source is not the likelihood recipe")
    if evaluation.get("evaluation_view") != "harmonized_all_windows_full_coverage":
        raise ConfigBuildError(f"{modality}: canonical source view is not harmonized_all_windows_full_coverage")
    if int(config.get("split", {}).get("seed", -1)) != 1337:
        raise ConfigBuildError(f"{modality}: canonical source split.seed is not 1337")

    config["recipe_id"] = f"{source['recipe_id']}_{tag}"
    config["labels"] = labels_block(tag)
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{CAMPAIGN}/{modality}/{DATASET}"
    )
    if modality != "text_only":
        # This family is a separate, completed DAIC campaign: it keeps the
        # four-rank shape its runs were submitted with, so the two-node audio-lane
        # default of the three-seed matrix does not change these configs.
        config["training"]["gradient_accumulation_steps"] = 32
        config["resources"] = {"eval_nodes": 1, "eval_gpus_per_node": 4}

    if modality == "text_only":
        validate_qwen38_config(config)
    else:
        validate_qwen3omni_config(config)
    if config.get("training", {}).get("selection_metric") != "inner_val_macro_f1":
        raise ConfigBuildError(f"{modality}: checkpoint selection is not inner_val_macro_f1")
    return config


def diff_keys(source: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Return the dotted paths that differ between two configs."""
    differences: list[str] = []

    def walk(left: Any, right: Any, prefix: str) -> None:
        if prefix and prefix in OPAQUE_DIFF_KEYS:
            if left != right:
                differences.append(prefix)
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in left or key not in right:
                    differences.append(path)
                else:
                    walk(left[key], right[key], path)
            return
        if left != right:
            differences.append(prefix)

    walk(source, candidate, "")
    return differences


def render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000)


def build(*, check: bool) -> int:
    mismatches: list[str] = []
    for modality in SOURCES:
        source_path = MAIN / SOURCES[modality][0]
        if not source_path.is_file():
            raise ConfigBuildError(f"missing canonical source: {source_path}")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        for tag in ARM_ORDER:
            expected = derive(source, modality=modality, tag=tag)
            drift = set(diff_keys(source, expected))
            unexpected = drift - set(ALLOWED_DIFF_KEYS)
            if unexpected:
                raise ConfigBuildError(
                    f"{modality}/{tag}: derived config differs outside the allowlist: {sorted(unexpected)}"
                )
            target_path = LABELS / generated_name(modality, tag)
            rendered = render(expected)
            if check:
                if not target_path.is_file() or target_path.read_text(encoding="utf-8") != rendered:
                    mismatches.append(str(target_path.relative_to(ROOT)))
            else:
                target_path.write_text(rendered, encoding="utf-8")

    if mismatches:
        print("Qwen3 DAIC label-vocabulary config drift:")
        for path in mismatches:
            print(f"- {path}")
        return 1
    print(
        "Qwen3 DAIC label-vocabulary configs are current."
        if check
        else f"Wrote {len(SOURCES) * len(ARM_ORDER)} Qwen3 DAIC label-vocabulary configs."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
