#!/usr/bin/env python3
"""Generate the prompt-context Qwen/Gemma config family.

Thirty standalone cells (five datasets x three modalities x two backbones) and
six merged cells (two backbones x three modalities) are derived from their
current canonical sources. Only the prompt recipe changes:

* the shared system instruction is replaced by the common instruction plus the
  current dataset's recording-context block (``docs/PROMPT_PROPOSAL_DEPRESSION_20260920.md``);
* the Turkish pooled cells select the versioned question-context sentence set
  through ``prompt.question_context_version``;
* the recipe id, the output root and the explicit evaluation view are new.

Every other field, including the split protocol, seed, LoRA settings,
checkpoint selection, weights, sampling and evaluation backend, is inherited
byte-for-byte from the canonical source so the family represents a prompt
change only. The merged cells additionally repoint the Turkish component from
the positive-only config to the pooled config and manifest.

The Gemma configs are derived from the generated Qwen configs and differ only
where the backend requires it: ``model_backend``, ``model_name_or_path``,
``model_revision``, ``lora.target_modules`` and ``output_dirs.run_root``.

The script is deterministic and idempotent: it refuses to overwrite an existing
file whose content differs from the derived content, and ``--check`` reports
inconsistencies without writing.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "configs/main"
MERGED = PROJECT_ROOT / "configs/experiments/merged"

RECIPE_ID = (
    "harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_v2_promptcontext_v1"
)
EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
QUESTION_CONTEXT_VERSION = "promptcontext_v1"
QWEN_RUN_ROOT_CAMPAIGN = "promptcontext_v1_likelihood"
GEMMA_RUN_ROOT_CAMPAIGN = "promptcontext_v1_gemma4_likelihood"
QWEN_MERGED_OUTPUT_CAMPAIGN = "promptcontext_v1"
GEMMA_MERGED_OUTPUT_CAMPAIGN = "promptcontext_v1_gemma4"
QWEN_MERGED_RUN_CAMPAIGN = "symmetric_merged/promptcontext_v1_likelihood"
GEMMA_MERGED_RUN_CAMPAIGN = "symmetric_merged/promptcontext_v1_gemma4_likelihood"

GEMMA4_MODEL_PATH = (
    "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/"
    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
)
GEMMA4_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA4_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)

COMMON_INSTRUCTION = (
    "You are classifying a participant's depression study label from the provided "
    "speech audio and/or transcript. Use the recording context below to interpret "
    "the input. Possible relevant signs include persistent low mood, loss of "
    "interest, hopelessness, and changes in sleep or energy. A question's topic, a "
    "single phrase, or one vocal feature is not conclusive. The label belongs to "
    "the participant, even when the input is only one recording or audio window. "
    "Make a research label prediction, not a clinical diagnosis."
)

DATASET_CONTEXT = {
    "androids_interview": (
        "Recording context: This is an Italian interview with a human interviewer. "
        "Questions mainly concern everyday life, including family, work, recent "
        "activities, and hobbies. It is not a systematic symptom questionnaire. A "
        "neutral or cheerful topic does not determine the participant's study label. "
        "The label comes from a psychiatrist's DSM-5 diagnosis."
    ),
    "d3tec": (
        "Recording context: This is Spanish speech from a non-interactive slideshow "
        "of 27 tasks. Tasks include open answers, reading words or passages, and "
        "describing images with different emotional content. Some emotion in a "
        "response may come from the assigned task. The study label uses PHQ-9 \u2265 10."
    ),
    "daic": (
        "Recording context: This is an English semi-structured interview with the "
        "virtual interviewer Ellie. It mixes everyday conversation with questions "
        "about mood, sleep, diagnosis, and treatment. The audio is a short window of "
        "participant speech; if a transcript is provided, it can cover more of the "
        "participant's session than the audio. The study uses a PHQ-8 binary label "
        "associated with a threshold of 10."
    ),
    "cmdc": (
        "Recording context: This is Mandarin speech from a face-to-face, "
        "symptom-focused interview. Its topics include mood, sleep, appetite, energy, "
        "concentration, worries, self-harm, and changes in movement or speech. The "
        "topic of a question is not itself evidence that the participant has "
        "depression. The study label is a clinical MDD group under DSM-IV, confirmed "
        "with MINI."
    ),
    "turkish": (
        "Recording context: This is Turkish speech from one of two question sets "
        "completed by the same participants. The sets concern positive or negative "
        "material. The exact question text is not available in this input. The "
        "participant has one study label across both sets. The study label is "
        "BDI \u2265 17."
    ),
}

# dataset key -> (config prefix, suffix carrying the transcript provenance)
STANDALONE_DATASETS = {
    "androids_interview": ("androids", ""),
    "d3tec": ("d3tec", ""),
    "cmdc": ("cmdc", ""),
    "daic": ("daic", ""),
    "turkish": ("turkish_pooled_t17", "_qwen3asr"),
}
STANDALONE_DATASET_ORDER = ("androids_interview", "d3tec", "daic", "cmdc", "turkish")
MODALITIES = ("audio_only", "text_only", "audio_text")

MERGED_COMPONENTS = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")


class GenerationError(RuntimeError):
    """Raised when a config cannot be derived safely."""


def system_prompt(dataset: str) -> str:
    try:
        context = DATASET_CONTEXT[dataset]
    except KeyError as exc:
        raise GenerationError(f"No recording-context block for dataset {dataset!r}.") from exc
    return f"{COMMON_INSTRUCTION}\n\n{context}"


def source_name(dataset: str, modality: str) -> str:
    prefix, tail = STANDALONE_DATASETS[dataset]
    return f"{prefix}_{modality}_harmonized_selmacrof1_likelihood_v1{tail}.yaml"


def target_name(dataset: str, modality: str, *, gemma: bool) -> str:
    prefix, tail = STANDALONE_DATASETS[dataset]
    backend = "_gemma4" if gemma else ""
    return (
        f"{prefix}_{modality}_harmonized_selmacrof1_likelihood_v2_promptcontext"
        f"{tail}{backend}.yaml"
    )


def merged_source_name(modality: str, *, gemma: bool) -> str:
    backend = "gemma4_" if gemma else ""
    return f"symmetric_merged_harmonized_{backend}{modality}_likelihood_v1.yaml"


def merged_target_name(modality: str, *, gemma: bool) -> str:
    backend = "gemma4_" if gemma else ""
    return f"symmetric_merged_harmonized_{backend}promptcontext_{modality}.yaml"


def _run_root(dataset: str, modality: str) -> str:
    return (
        f"${{PROJECT_ROOT}}/output_model/{QWEN_RUN_ROOT_CAMPAIGN}/{modality}/{dataset}"
    )


def derive_standalone(source: dict, dataset: str, modality: str, *, gemma: bool) -> dict:
    config = copy.deepcopy(source)
    if config.get("dataset") != dataset:
        raise GenerationError(
            f"source config dataset {config.get('dataset')!r} does not match {dataset!r}"
        )
    config["recipe_id"] = RECIPE_ID
    prompt = config.setdefault("prompt", {})
    prompt["system"] = system_prompt(dataset)
    if dataset == "turkish":
        prompt["question_context_version"] = QUESTION_CONTEXT_VERSION
    config["output_dirs"]["run_root"] = _run_root(dataset, modality)
    config["evaluation"]["evaluation_view"] = EVALUATION_VIEW
    if gemma:
        config = derive_gemma(config)
    return config


def derive_gemma(source: dict) -> dict:
    config = copy.deepcopy(source)
    config["model_backend"] = "gemma4"
    config["model_name_or_path"] = "${GEMMA4_MODEL_PATH:-" + GEMMA4_MODEL_PATH + "}"
    config["model_revision"] = GEMMA4_REVISION
    config["lora"]["target_modules"] = GEMMA4_LORA_TARGET_REGEX
    qwen_root = str(config["output_dirs"]["run_root"])
    if QWEN_RUN_ROOT_CAMPAIGN not in qwen_root:
        raise GenerationError(f"Cannot derive the Gemma run root from {qwen_root!r}.")
    config["output_dirs"]["run_root"] = qwen_root.replace(
        QWEN_RUN_ROOT_CAMPAIGN, GEMMA_RUN_ROOT_CAMPAIGN, 1
    )
    return config


def component_config_name(name: str, modality: str, *, gemma: bool) -> str:
    dataset = "turkish" if name == "turkish" else name
    return target_name(dataset, modality, gemma=gemma)


def derive_merged(source: dict, modality: str, *, gemma: bool) -> dict:
    config = copy.deepcopy(source)
    if str(config.get("modality")) != modality:
        raise GenerationError(
            f"merged source modality {config.get('modality')!r} does not match {modality!r}"
        )
    config["name"] = (
        f"symmetric_merged_harmonized_{'gemma4_' if gemma else ''}promptcontext_{modality}"
    )
    config["recipe_id"] = RECIPE_ID
    seen: set[str] = set()
    for component in config["components"]:
        name = str(component["name"])
        seen.add(name)
        if name not in MERGED_COMPONENTS:
            raise GenerationError(f"unexpected merged component {name!r}")
        component["config"] = f"configs/main/{component_config_name(name, modality, gemma=gemma)}"
        if name == "turkish":
            component["manifest_path"] = (
                "outputs/manifests_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest.jsonl"
            )
            component["metadata_path"] = (
                "outputs/splits_harmonized/turkish_pooled_t17_qwen3asr/"
                "turkish_manifest_metadata.json"
            )
    if seen != set(MERGED_COMPONENTS):
        raise GenerationError(f"merged components {sorted(seen)} do not match the locked set")
    output_campaign = (
        GEMMA_MERGED_OUTPUT_CAMPAIGN if gemma else QWEN_MERGED_OUTPUT_CAMPAIGN
    )
    run_campaign = GEMMA_MERGED_RUN_CAMPAIGN if gemma else QWEN_MERGED_RUN_CAMPAIGN
    config["output_dirs"]["merged_root"] = (
        f"${{PROJECT_ROOT}}/outputs/symmetric_merged/{output_campaign}/{modality}"
    )
    config["output_dirs"]["run_root"] = (
        f"${{PROJECT_ROOT}}/output_model/{run_campaign}/{modality}"
    )
    return config


def _str_representer(dumper: yaml.Dumper, value: str):
    """Render multi-line strings as literal blocks so prompts stay readable."""
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


class _ConfigDumper(yaml.SafeDumper):
    """Local dumper: the block-scalar style must not leak into other callers."""


_ConfigDumper.add_representer(str, _str_representer)


def _render(config: dict) -> str:
    return yaml.dump(
        config,
        Dumper=_ConfigDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def _emit(target: Path, config: dict, *, check_only: bool, failures: list[str], written: list[Path]) -> None:
    rendered = _render(config)
    if yaml.safe_load(rendered) != config:
        failures.append(f"rendered config does not round-trip: {target}")
        return
    if target.is_file():
        if target.read_text(encoding="utf-8") != rendered:
            failures.append(f"existing config differs from derived content: {target}")
        return
    if check_only:
        failures.append(f"missing derived config: {target}")
        return
    target.write_text(rendered, encoding="utf-8")
    written.append(target)
    print(f"wrote {target.relative_to(PROJECT_ROOT)}")


def build_standalone() -> list[tuple[dict, Path]]:
    planned: list[tuple[dict, Path]] = []
    for dataset in STANDALONE_DATASET_ORDER:
        for modality in MODALITIES:
            source_path = MAIN / source_name(dataset, modality)
            if not source_path.is_file():
                raise GenerationError(f"missing canonical source config: {source_path}")
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            for gemma in (False, True):
                config = derive_standalone(source, dataset, modality, gemma=gemma)
                planned.append((config, MAIN / target_name(dataset, modality, gemma=gemma)))
    return planned


def build_merged() -> list[tuple[dict, Path]]:
    planned: list[tuple[dict, Path]] = []
    for modality in MODALITIES:
        for gemma in (False, True):
            source_path = MERGED / merged_source_name(modality, gemma=gemma)
            if not source_path.is_file():
                raise GenerationError(f"missing canonical merged config: {source_path}")
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            config = derive_merged(source, modality, gemma=gemma)
            planned.append((config, MERGED / merged_target_name(modality, gemma=gemma)))
    return planned


def main() -> int:
    check_only = "--check" in sys.argv
    failures: list[str] = []
    written: list[Path] = []
    try:
        planned = build_standalone() + build_merged()
    except GenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for config, target in planned:
        _emit(target, config, check_only=check_only, failures=failures, written=written)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    standalone = sum(1 for _config, path in planned if path.parent == MAIN)
    print(
        f"ok: {standalone} standalone + {len(planned) - standalone} merged configs consistent "
        f"({len(written)} written)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
