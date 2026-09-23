from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_qwen3omni_daic_configs as generator
from scripts import qwen3omni_backend_probe as probe
from scripts import qwen3omni_risk_inventory as inventory
from src.data.prompt_context import (
    PROMPT_CONTEXT_VERSION,
    PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES,
    resolve_question_context_sentences,
    resolve_system_prompt,
)
from src.experiment_tracking.manifest_policy import (
    MANIFEST_POLICY_PREBUILT,
    ManifestPolicyError,
    prebuilt_manifest_files,
    validate_manifest_policy,
)
from src.model.qwen3omni_lora import (
    QWEN3OMNI_EVALUATION_VIEW,
    QWEN3OMNI_LORA_TARGET_REGEX,
    validate_qwen3omni_config,
)
from src.utils import (
    MODEL_BACKEND_QWEN3OMNI,
    load_yaml,
    resolve_evaluation_resource_shape,
    resolve_model_backend,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
GROUP_PATH = ROOT / "experiments/definitions/qwen3omni-standalone-promptcontext-20260923.yaml"

POOLED_SOURCES = {
    "audio_only": MAIN / "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml",
    "audio_text": MAIN / "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml",
}
POOLED_TARGETS = {
    "audio_only": MAIN
    / "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
    "_promptcontext_v1_qwen3omni_30b_a3b.yaml",
    "audio_text": MAIN
    / "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr"
    "_promptcontext_v1_qwen3omni_30b_a3b.yaml",
}

# The eight production cells the campaign plan fixes.
EXPECTED_CELLS = {
    ("d3tec", "audio_only"),
    ("d3tec", "audio_text"),
    ("androids_interview", "audio_only"),
    ("androids_interview", "audio_text"),
    ("cmdc", "audio_only"),
    ("cmdc", "audio_text"),
    ("turkish", "audio_only"),
    ("turkish", "audio_text"),
}
EXPECTED_DATASET_CONTEXT = {
    "d3tec": "d3tec",
    "androids_interview": "androids",
    "cmdc": "cmdc",
    "turkish": "turkish_pooled",
}

# Leaf paths that must never move between a pooled source and its Omni target:
# the scientific recipe is inherited, only the documented contract changes.
FROZEN_POOLED_PREFIXES = (
    "split.",
    "data.",
    "labels.",
    "audio_adapter.",
    "dataset_root",
    "metadata_csv",
    "transcript_file",
    "threshold",
    "metadata_schema",
    "quarantine_path",
    "seed",
)
FROZEN_POOLED_LEAVES = {
    "evaluation.aggregation_level",
    "evaluation.hierarchical_score_aggregation",
    "evaluation.headline_mode",
    "evaluation.sample_prediction_mode",
    "lora.rank",
    "lora.alpha",
    "lora.dropout",
    "lora.bias",
    "output_dirs.manifest_dir",
    "output_dirs.split_dir",
    "training.num_train_epochs",
    "training.learning_rate",
    "training.gradient_accumulation_steps",
    "training.selection_metric",
    "training.selection_metric_mode",
    "training.class_balance",
    "training.bf16",
    "training.gradient_checkpointing",
    "training.early_stopping",
}


def _pooled_pair(modality: str) -> tuple[dict, dict]:
    return load_yaml(POOLED_SOURCES[modality]), load_yaml(POOLED_TARGETS[modality])


@pytest.mark.parametrize("modality", sorted(POOLED_TARGETS))
def test_pooled_targets_exist_and_honour_the_omni_contract(modality: str) -> None:
    assert POOLED_TARGETS[modality].is_file(), POOLED_TARGETS[modality]
    config = load_yaml(POOLED_TARGETS[modality])
    assert config["dataset"] == "turkish"
    assert config["dataset_variant"] == "pooled_t17"
    assert config["model_backend"] == MODEL_BACKEND_QWEN3OMNI
    assert resolve_model_backend(config) == MODEL_BACKEND_QWEN3OMNI
    assert config["model_attn_implementation"] == "sdpa"
    # The file declares the offline snapshot through the documented env default.
    raw = POOLED_TARGETS[modality].read_text(encoding="utf-8")
    assert "model_name_or_path: ${QWEN3_OMNI_MODEL_PATH:-/gpfs/" in raw
    assert config["model_name_or_path"].endswith("Qwen3-Omni-30B-A3B-Instruct")
    assert config["lora"]["target_modules"] == QWEN3OMNI_LORA_TARGET_REGEX
    assert config["training"]["strategy"] == "fsdp"
    assert config["training"]["activation_offload"] == "cpu"
    assert config["training"]["run_final_eval_in_train"] is False
    assert config["evaluation"]["evaluation_view"] == QWEN3OMNI_EVALUATION_VIEW
    assert config["evaluation"]["inference_dtype"] == "bf16"
    assert config["manifest_policy"] == MANIFEST_POLICY_PREBUILT
    assert config["output_dirs"]["run_root"].endswith(
        f"output_model/promptcontext_v1_qwen3omni_likelihood/{modality}/turkish"
    )
    prompt = config["prompt"]
    assert prompt["version"] == PROMPT_CONTEXT_VERSION
    assert prompt["dataset_context"] == "turkish_pooled"
    assert prompt["question_context_version"] == PROMPT_CONTEXT_VERSION
    assert "{question_context}" in prompt["user_template"]
    assert "system" not in prompt
    validate_qwen3omni_config(config)
    # The prompt resolves through the central resolver, not through local prose.
    system_prompt = resolve_system_prompt(config)
    assert "Turkish speech from one of two question sets" in system_prompt
    assert resolve_evaluation_resource_shape(config)["gpus_per_node"] == 4


@pytest.mark.parametrize("modality", sorted(POOLED_TARGETS))
def test_pooled_diff_is_limited_to_the_pooled_allowlist(modality: str) -> None:
    source, target = _pooled_pair(modality)
    changed = set(generator.diff_paths(source, target))
    assert changed
    assert changed <= generator.POOLED_ALLOWED_DIFF_PATHS
    # The recipe leaves above are inherited unchanged.
    for prefix in FROZEN_POOLED_PREFIXES:
        assert not [path for path in changed if path.startswith(prefix)], prefix
    for leaf in FROZEN_POOLED_LEAVES:
        assert leaf not in changed, leaf
    # The documented contract differences are present.
    assert {
        "model_backend",
        "model_name_or_path",
        "model_attn_implementation",
        "recipe_id",
        "manifest_policy",
        "prompt.version",
        "prompt.dataset_context",
        "prompt.question_context_version",
        "prompt.system",
        "lora.target_modules",
        "training.strategy",
        "training.activation_offload",
        "training.run_final_eval_in_train",
        "evaluation.inference_dtype",
        "output_dirs.run_root",
    } <= changed


@pytest.mark.parametrize("modality", sorted(POOLED_TARGETS))
def test_pooled_prompt_uses_the_versioned_question_context_sentences(modality: str) -> None:
    config = load_yaml(POOLED_TARGETS[modality])
    sentences = resolve_question_context_sentences(config)
    assert sentences == PROMPTCONTEXT_QUESTION_CONTEXT_SENTENCES
    assert sentences["pos_only_t17"].startswith("Positive set: This recording answers a question")
    assert sentences["negative_only_t17"].startswith("Negative set: This recording answers a question")


def test_pooled_prebuilt_manifest_policy_fails_closed() -> None:
    target = load_yaml(POOLED_TARGETS["audio_only"])
    assert validate_manifest_policy(target) == MANIFEST_POLICY_PREBUILT
    files = prebuilt_manifest_files(
        manifest_dir="/runtime/manifests/turkish", split_dir="/runtime/splits/turkish", dataset="turkish"
    )
    assert set(files) == {"manifest", "manifest_csv", "folds", "split_metadata"}
    assert all(path.startswith("/runtime/") for path in files.values())
    # A pooled config without the explicit policy still refuses the build route.
    source = load_yaml(POOLED_SOURCES["audio_only"])
    assert "manifest_policy" not in source
    with pytest.raises(ManifestPolicyError):
        validate_manifest_policy(source)


def test_generator_check_mode_reports_both_families_without_drift(tmp_path: Path) -> None:
    audit_path = tmp_path / "config_diff_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_qwen3omni_daic_configs.py",
            "--check",
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["prompt_context_version"] == PROMPT_CONTEXT_VERSION
    entries = {entry["cell_id"]: entry for entry in audit["configs"]}
    assert len(entries) == 4
    assert {entry["family"] for entry in audit["configs"]} == {"daic", "turkish_pooled"}
    for entry in audit["configs"]:
        assert entry["allowed"] is True, entry
        assert set(entry["changed_paths"]) <= set(entry["allowed_paths"])
    assert entries["daic_audio_only"]["family"] == "daic"
    assert entries["turkish_pooled_audio_only"]["family"] == "turkish_pooled"
    assert set(audit["family_allowed_paths"]["turkish_pooled"]) > set(
        audit["family_allowed_paths"]["daic"]
    )


def test_group_definition_expands_to_eight_cells_and_forty_fits() -> None:
    group = load_yaml(GROUP_PATH)
    assert group["schema_version"] == "audiollm.experiment_group.v1"
    assert group["group_id"] == "qwen3omni-standalone-promptcontext-20260923"
    assert group["dataset"] == "multi_dataset"
    assert group["expected_seeds"] == [1337]
    assert group["expected_folds"] == [0, 1, 2, 3, 4]
    runs = group["runs"]
    assert len(runs) == 8
    cells = set()
    fits = 0
    for run in runs:
        assert run["folds"] == [0, 1, 2, 3, 4], run["logical_run"]
        config_path = ROOT / run["config"]
        assert config_path.is_file(), config_path
        config = load_yaml(config_path)
        assert config["dataset"] == run["dataset"]
        cells.add((run["dataset"], run["modality"]))
        fits += len(run["folds"])
    assert cells == EXPECTED_CELLS
    assert fits == 40
    assert group["scope"]["campaign"] == "promptcontext_v1_qwen3omni_likelihood"
    assert group["scope"]["checkpoint_role"] == "best_model"
    assert group["scope"]["evaluation_jobs_per_fit"] == 1
    assert group["primary_metric"] == {
        "namespace": "headline/binary_strict",
        "name": "macro_f1",
        "backend": "likelihood",
        "aggregation": "subject_level",
        "evaluation_view": QWEN3OMNI_EVALUATION_VIEW,
    }


def test_group_definition_excludes_the_unplanned_scope() -> None:
    group = load_yaml(GROUP_PATH)
    assert {
        "omni_text_only_training",
        "daic_retraining",
        "merged_training",
        "native_versus_english_comparison",
        "translation",
        "logreg_xgboost_optuna",
        "teacher_forced_or_generation_evaluation",
        "talker_loading_or_training",
        "quantization",
        "wandb_export",
        "lora_target_comparison",
        "unplanned_dataset_or_seed_expansion",
        "PR_merge",
    } <= set(group["exclusions"])
    contract = group["model_contract"]
    assert contract["backend"] == MODEL_BACKEND_QWEN3OMNI
    assert contract["trainable_class"] == "Qwen3OmniMoeThinkerForConditionalGeneration"
    assert contract["training_strategy"] == "fsdp"
    assert "world size 8" in contract["training_shape"]
    assert "accumulation 16" in contract["training_shape"]
    assert "4 H100" in contract["evaluation_shape"]


def test_group_configs_resolve_to_the_omni_prompt_context_contract() -> None:
    group = load_yaml(GROUP_PATH)
    for run in group["runs"]:
        config = load_yaml(ROOT / run["config"])
        validate_qwen3omni_config(config)
        assert resolve_model_backend(config) == MODEL_BACKEND_QWEN3OMNI
        assert config["prompt"]["version"] == PROMPT_CONTEXT_VERSION
        assert config["prompt"]["dataset_context"] == EXPECTED_DATASET_CONTEXT[run["dataset"]]
        assert config["evaluation"]["evaluation_view"] == QWEN3OMNI_EVALUATION_VIEW
        assert config["evaluation"]["inference_dtype"] == "bf16"
        assert resolve_evaluation_resource_shape(config)["gpus_per_node"] == 4
        assert "promptcontext_v1_qwen3omni_likelihood" in config["output_dirs"]["run_root"]
        # The old prompt either disappeared (promptcontext configs) or is unused.
        if config["prompt"].get("version") == PROMPT_CONTEXT_VERSION:
            assert "system" not in config["prompt"]


def test_risk_inventory_reads_spans_window_bounds_and_segment_duration() -> None:
    spans = {"audio_spans": [{"start_frame": 0, "end_frame": 16000}]}
    assert inventory._audio_seconds(spans) == 1.0
    assert inventory._audio_seconds({"start_time": 2.0, "end_time": 32.5}) == 30.5
    assert inventory._audio_seconds({"segment_duration": 12.25}) == 12.25
    assert inventory._audio_seconds({}) == 0.0


def test_probe_keeps_each_datasets_declared_transcript_scope() -> None:
    packed30 = {"data": {"use_audio": True, "use_text": True, "audio_text_transcript_scope": "full_participant"}}
    windows = {"data": {"use_audio": True, "use_text": True, "audio_text_transcript_scope": "full_subject"}}
    undeclared = {"data": {"use_audio": True, "use_text": True}}
    assert probe._modality_config(packed30, "audio_text")["data"]["audio_text_transcript_scope"] == "full_participant"
    assert probe._modality_config(windows, "audio_text")["data"]["audio_text_transcript_scope"] == "full_subject"
    assert probe._modality_config(undeclared, "audio_text")["data"]["audio_text_transcript_scope"] == "full_participant"
    audio_only = probe._modality_config(windows, "audio_only")["data"]
    assert audio_only["use_audio"] is True and audio_only["use_text"] is False
    assert "audio_text_transcript_scope" not in audio_only
    # The source configs are not mutated.
    assert windows["data"]["use_text"] is True


def test_no_script_still_references_the_renamed_inventory() -> None:
    stale = [
        path
        for path in sorted(ROOT.glob("scripts/*.sh")) + sorted(ROOT.glob("scripts/*.py"))
        if "qwen3omni_daic_risk_inventory" in path.read_text(encoding="utf-8")
    ]
    assert not stale, stale
