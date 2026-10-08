"""Worker 4 legacy-prompt treatment contract tests.

These tests freeze the source-grounded legacy/current prompt map for the
legacy-prompt versus promptcontext_v1 comparison and prove that the treatment
family changes only the rendered prompt, ``recipe_id`` and output run root
relative to the canonical promptcontext_v1 controls. They also pin that the
default canonical generators and default selections do not change.

The tests are hermetic: no dataset, model, manifest or cache is required.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import yaml

from scripts import build_qwen3_legacy_prompt_configs as legacy_configs
from tools import audit_qwen3_legacy_prompt_controls as controls_audit
from tools import qwen3_legacy_prompt_plan as planner
from src.data.prompt_context import (
    LEGACY_QUESTION_CONTEXT_SENTENCES,
    PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES,
    resolve_prompt_context_version,
    resolve_question_context_sentences,
    resolve_system_prompt,
)
from src.data.runtime import render_user_prompt_text

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
ARCHIVE = ROOT / "configs/archive/pre_default_backbone_20260923"
CAMPAIGN = "qwen3_legacy_prompt_20261008"

DATASETS = ("daic", "d3tec", "androids", "cmdc", "turkish")
MODALITIES = ("text_only", "audio_only", "audio_text")

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

SYNTHETIC_TRANSCRIPT = "Synthetic unit-test transcript only; no study data."

QUESTION_CONDITION_KEYS = {
    "positive_only_t17": "pos_only_t17",
    "negative_only_t17": "negative_only_t17",
}


def cells() -> list[tuple[str, str]]:
    return [(dataset, modality) for dataset in DATASETS for modality in MODALITIES]


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def treatment_path(dataset: str, modality: str) -> Path:
    return MAIN / legacy_configs.treatment_name(dataset, modality)


def control_path(dataset: str, modality: str) -> Path:
    return MAIN / legacy_configs.control_name(dataset, modality)


def legacy_reference_path(dataset: str, modality: str) -> Path:
    name = legacy_configs.legacy_name(dataset, modality)
    return (MAIN if dataset == "turkish" else ARCHIVE) / name


def test_generator_check_is_current_and_emits_fifteen_files() -> None:
    assert legacy_configs.build(check=True) == 0
    generated = sorted(MAIN.glob(f"*_{legacy_configs.MARKER}*.yaml"))
    assert len(generated) == 15
    assert all(path.name.endswith(".yaml") for path in generated)


def test_treatment_differs_from_control_only_in_allowlisted_keys() -> None:
    allowed_top_level = {"recipe_id", "prompt", "output_dirs"}
    for dataset, modality in cells():
        control = load(control_path(dataset, modality))
        treatment = load(treatment_path(dataset, modality))
        assert set(control) == set(treatment)
        for key in sorted(control):
            if key in allowed_top_level:
                continue
            assert treatment[key] == control[key], f"unexpected {key} change for {dataset}/{modality}"
        control_dirs = copy.deepcopy(control["output_dirs"])
        treatment_dirs = copy.deepcopy(treatment["output_dirs"])
        control_run_root = control_dirs.pop("run_root")
        treatment_run_root = treatment_dirs.pop("run_root")
        assert treatment_dirs == control_dirs
        assert control_run_root != treatment_run_root
        assert treatment_run_root == (
            "${PROJECT_ROOT}/output_model/"
            f"{CAMPAIGN}/{modality}/{legacy_configs.DATASET_DIRS[dataset]}"
        )


def test_treatment_prompt_is_the_source_grounded_inline_prompt() -> None:
    for dataset, modality in cells():
        treatment = load(treatment_path(dataset, modality))
        legacy = load(legacy_reference_path(dataset, modality))
        prompt = treatment["prompt"]
        assert set(prompt) == {"system", "user_template", "prompt_language"}
        assert prompt["system"] == LEGACY_SYSTEM[modality]
        assert prompt["system"] == lstr(legacy["prompt"]["system"])
        assert prompt["user_template"] == lstr(legacy["prompt"]["user_template"])
        assert resolve_prompt_context_version(treatment) is None
        assert resolve_system_prompt(treatment) == LEGACY_SYSTEM[modality]
        if "{question_context}" in prompt["user_template"]:
            assert resolve_question_context_sentences(treatment) == LEGACY_QUESTION_CONTEXT_SENTENCES


def lstr(value: object) -> str:
    return str(value).strip()


def test_rendered_prompts_match_the_legacy_reference() -> None:
    for dataset, modality in cells():
        treatment = load(treatment_path(dataset, modality))
        legacy = load(legacy_reference_path(dataset, modality))
        if "{question_context}" in str(treatment["prompt"]["user_template"]):
            for label, key in QUESTION_CONDITION_KEYS.items():
                assert render_user_prompt_text(
                    treatment, SYNTHETIC_TRANSCRIPT, question_condition=key
                ) == render_user_prompt_text(legacy, SYNTHETIC_TRANSCRIPT, question_condition=key)
        else:
            assert render_user_prompt_text(
                treatment, SYNTHETIC_TRANSCRIPT
            ) == render_user_prompt_text(legacy, SYNTHETIC_TRANSCRIPT)


def test_controls_still_render_promptcontext_v1_and_turkish_differs() -> None:
    for dataset, modality in cells():
        control = load(control_path(dataset, modality))
        treatment = load(treatment_path(dataset, modality))
        assert resolve_prompt_context_version(control) == "promptcontext_v1"
        assert resolve_system_prompt(control) != LEGACY_SYSTEM[modality]
        assert "Recording context:" in resolve_system_prompt(control)
        if "{question_context}" in str(control["prompt"]["user_template"]):
            assert resolve_question_context_sentences(control) == PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES
            for key in QUESTION_CONDITION_KEYS.values():
                assert render_user_prompt_text(
                    control, SYNTHETIC_TRANSCRIPT, question_condition=key
                ) != render_user_prompt_text(
                    treatment, SYNTHETIC_TRANSCRIPT, question_condition=key
                )


def test_default_selections_do_not_include_the_treatment_family() -> None:
    matrix = (ROOT / "configs/experiments/harmonized/standalone_matrix.yaml").read_text(
        encoding="utf-8"
    )
    assert legacy_configs.MARKER not in matrix
    merged = (ROOT / "configs/experiments/merged").glob("*.yaml")
    for path in merged:
        assert legacy_configs.MARKER not in path.read_text(encoding="utf-8"), path.name


def test_canonical_generators_and_default_selection_remain_current() -> None:
    commands = (
        [sys.executable, "scripts/build_canonical_backend_configs.py", "--check"],
        [sys.executable, "scripts/build_qwen3_english_configs.py", "--check"],
        [sys.executable, "scripts/build_qwen3_pooled_merged_configs.py", "--check"],
        [sys.executable, "tools/qwen3_pooled_defaults.py", "--check"],
    )
    for command in commands:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=300
        )
        assert completed.returncode == 0, (command, completed.stdout, completed.stderr)


def test_planner_freezes_the_189_fit_matrix() -> None:
    matrix = planner.build_matrix()
    assert planner.check_matrix(matrix) == []
    assert matrix["summary"] == {
        "routes": 15,
        "fits": 189,
        "text_fits": 63,
        "audio_fits": 126,
    }
    assert matrix["training_seeds"] == [7, 1337, 2024]
    assert matrix["split_seed"] == 1337 and matrix["head_seed"] == 1337
    run_names = [fit["run_name"] for fit in matrix["fits"]]
    assert len(set(run_names)) == 189
    assert all(name.startswith("q3lp_") for name in run_names)
    assert all(legacy_configs.MARKER in fit["config"] for fit in matrix["fits"])
    for route in matrix["routes"]:
        expected = (0,) if route["dataset"] == "daic" else (0, 1, 2, 3, 4)
        assert tuple(route["folds"]) == expected
        assert route["run_root"].startswith(
            "${PROJECT_ROOT}/output_model/qwen3_legacy_prompt_20261008/"
        )
        assert route["control_config"] != route["config"]


def test_planner_production_graph_counts() -> None:
    matrix = planner.build_matrix()
    graph = planner.build_production_graph(matrix, "0" * 64)
    assert graph["core_training"]["total_scheduler_jobs"] == 378
    assert graph["downstream_heads"]["total_scheduler_jobs"] == 378
    assert graph["total_with_downstream_scheduler_jobs"] == 756
    assert sum(wave["fits"] for wave in graph["planned_waves"]) == 189
    assert all(wave["scheduler_jobs"] <= 80 for wave in graph["planned_waves"])
    assert graph["slot_policy"]["lane_cap_nonterminal"] == 80
    envs = graph["environments"]["per_backend"]
    assert set(envs) == {"qwen38", "qwen3omni"}
    assert all("qwen_mn5_rebuilt" not in path for path in envs.values())
    assert all(path.endswith("/bin/activate") for path in envs.values())


def test_control_audit_normalizes_both_key_formats() -> None:
    assert controls_audit.normalize_key("daic_text_only_native|1337|0") == (
        "daic_text_only_native",
        1337,
        0,
    )
    assert controls_audit.normalize_key("daic_text_only_native|s7|f0") == (
        "daic_text_only_native",
        7,
        0,
    )
    assert controls_audit.normalize_key("not-a-key") is None
