"""Locked matrix and qualifier contract for the v2 native/English study.

This module is deliberately model-free.  It is used by dry-runs, preflight,
submission bookkeeping, and the deterministic report builder so the planned
matrix is not re-created independently by shell wrappers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.features import optuna100_policy
from src.merged.heads import merged_method_prediction_backend
from src.utils import load_yaml_with_overrides, resolve_project_path


GROUP_ID = "native-en-text-heads-v2-20260822"
TRAINING_SEEDS = (7, 1337, 2024)
SPLIT_SEED = 1337
HEAD_SEED = 1337
OUTER_FOLDS = (0, 1, 2, 3, 4)
CONDITIONS = ("native", "english")
BACKBONES = ("qwen", "gemma4")
STANDALONE_DATASETS = ("d3tec", "androids_interview", "cmdc", "turkish")
MERGED_DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
HEADS = ("logreg", "xgb_optuna100")

STANDALONE_CONFIGS = {
    ("native", "qwen", "d3tec"): "configs/main/d3tec_text_only_harmonized_selmacrof1_tf.yaml",
    ("native", "qwen", "androids_interview"): "configs/main/androids_text_only_harmonized_selmacrof1_tf.yaml",
    ("native", "qwen", "cmdc"): "configs/main/cmdc_text_only_harmonized_selmacrof1_tf.yaml",
    ("native", "qwen", "turkish"): "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml",
    ("english", "qwen", "d3tec"): "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en.yaml",
    ("english", "qwen", "androids_interview"): "configs/main/androids_text_only_harmonized_selmacrof1_tf_en.yaml",
    ("english", "qwen", "cmdc"): "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en.yaml",
    ("english", "qwen", "turkish"): "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
    ("native", "gemma4", "d3tec"): "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
    ("native", "gemma4", "androids_interview"): "configs/main/androids_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
    ("native", "gemma4", "cmdc"): "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
    ("native", "gemma4", "turkish"): "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml",
    ("english", "gemma4", "d3tec"): "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
    ("english", "gemma4", "androids_interview"): "configs/main/androids_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
    ("english", "gemma4", "cmdc"): "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
    ("english", "gemma4", "turkish"): "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml",
}

MERGED_CONFIGS = {
    ("native", "qwen"): "configs/experiments/merged/native_en_text_heads_v2_qwen_native.yaml",
    ("english", "qwen"): "configs/experiments/merged/native_en_text_heads_v2_qwen_english.yaml",
    ("native", "gemma4"): "configs/experiments/merged/native_en_text_heads_v2_gemma4_native.yaml",
    ("english", "gemma4"): "configs/experiments/merged/native_en_text_heads_v2_gemma4_english.yaml",
}

# Smoke deliberately rotates representative adapters so the four standalone
# panels cover all four dataset builders while keeping the contracted 16 jobs.
SMOKE_STANDALONE_DATASET = {
    ("native", "qwen"): "d3tec",
    ("english", "qwen"): "androids_interview",
    ("native", "gemma4"): "cmdc",
    ("english", "gemma4"): "turkish",
}


@dataclass(frozen=True)
class MatrixJob:
    logical_run_name: str
    condition: str
    backbone: str
    endpoint: str
    job_type: str
    method: str | None
    dataset: str
    seed: int
    fold: int
    config: str
    dependency_keys: tuple[str, ...]
    shape: str
    evaluation_view: str
    aggregation: str
    backend: str
    optuna_trials: int | None

    @property
    def key(self) -> str:
        return ":".join(
            (
                self.logical_run_name,
                self.endpoint,
                str(self.fold),
                self.job_type,
                self.method or "none",
            )
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependency_keys"] = list(self.dependency_keys)
        value["key"] = self.key
        return value


def standalone_logical_run(condition: str, backbone: str, dataset: str, seed: int) -> str:
    return f"native_en_text_heads_v2_{condition}_{backbone}_{dataset}_s{int(seed)}"


def merged_logical_run(condition: str, backbone: str, seed: int, stage: str) -> str:
    return f"native_en_text_heads_v2_{condition}_{backbone}_merged_{stage}_s{int(seed)}"


def standalone_campaign(condition: str, backbone: str) -> str:
    return f"native_en_text_heads_v2_{condition}_{backbone}"


def _standalone_backend(backbone: str, head: str) -> str:
    prefix = "gemma4" if backbone == "gemma4" else "qwen"
    return f"{prefix}_hidden_logreg_raw" if head == "logreg" else f"{prefix}_hidden_xgb_optuna100"


def _standalone_aggregation(dataset: str) -> str:
    if dataset in {"d3tec", "androids_interview"}:
        return "pooled_subject_level_across_five_outer_folds"
    return "unweighted_fold_mean_across_five_outer_folds"


def _merged_backend(backbone: str, head: str) -> str:
    return merged_method_prediction_backend(
        "logreg" if head == "logreg" else "xgb_optuna",
        backbone,
    ) or ""


def _validate_keys() -> None:
    expected = {(condition, backbone, dataset) for condition in CONDITIONS for backbone in BACKBONES for dataset in STANDALONE_DATASETS}
    if set(STANDALONE_CONFIGS) != expected:
        raise ValueError("standalone config matrix does not cover exactly the locked study")
    if set(MERGED_CONFIGS) != {(condition, backbone) for condition in CONDITIONS for backbone in BACKBONES}:
        raise ValueError("merged config matrix does not cover exactly the locked study")


def validate_configs(project_root: str | Path) -> list[dict[str, Any]]:
    """Validate all study configs without requiring manifests or model imports."""

    root = Path(project_root).resolve()
    _validate_keys()
    identities: list[dict[str, Any]] = []
    for (condition, backbone, dataset), relative in sorted(STANDALONE_CONFIGS.items()):
        path = root / relative
        config = load_yaml_with_overrides(path, [])
        if str(config.get("dataset", "")).lower() != dataset:
            raise ValueError(f"{path} dataset does not match {dataset}")
        if str(config.get("data", {}).get("use_audio")) != "False" and config.get("data", {}).get("use_audio") is not False:
            raise ValueError(f"{path} is not text-only")
        configured_view = config.get("evaluation", {}).get("evaluation_view")
        if configured_view not in (None, "harmonized_all_windows_full_coverage"):
            raise ValueError(f"{path} has an incompatible evaluation view: {configured_view!r}")
        if condition == "english" and dataset != "daic":
            if config.get("transcripts", {}).get("variant") != "english":
                raise ValueError(f"English study config is not an English transcript config: {path}")
        identities.append({"kind": "standalone", "condition": condition, "backbone": backbone, "dataset": dataset, "path": relative})
    for (condition, backbone), relative in sorted(MERGED_CONFIGS.items()):
        path = root / relative
        config = load_yaml_with_overrides(path, [])
        components = config.get("components") or []
        if len(components) != 5 or {str(item.get("name")) for item in components} != set(MERGED_DATASETS):
            raise ValueError(f"{path} does not contain exactly the five locked merged components")
        if config.get("protocol_settings", {}).get("split_seed") != SPLIT_SEED:
            raise ValueError(f"{path} does not fix protocol_settings.split_seed={SPLIT_SEED}")
        if config.get("heads", {}).get("fixed_seed") != HEAD_SEED:
            raise ValueError(f"{path} does not fix heads.fixed_seed={HEAD_SEED}")
        optuna = config.get("heads", {}).get("optuna", {})
        if optuna.get("protocol_profile") != optuna100_policy.PROTOCOL_PROFILE or optuna.get("target_trials") != 100:
            raise ValueError(f"{path} does not declare the locked Optuna-100 profile")
        identities.append({"kind": "merged", "condition": condition, "backbone": backbone, "path": relative})
    return identities


def build_matrix(stage: str = "production") -> list[MatrixJob]:
    """Expand one locked execution stage into deterministic job records."""

    _validate_keys()
    stage = str(stage).lower()
    if stage not in {"smoke", "production"}:
        raise ValueError(f"unsupported matrix stage: {stage}")
    jobs: list[MatrixJob] = []
    seeds = (1337,) if stage == "smoke" else TRAINING_SEEDS
    panels = [(condition, backbone) for condition in CONDITIONS for backbone in BACKBONES]
    for condition, backbone in panels:
        datasets = (SMOKE_STANDALONE_DATASET[(condition, backbone)],) if stage == "smoke" else STANDALONE_DATASETS
        for seed in seeds:
            for dataset in datasets:
                config = STANDALONE_CONFIGS[(condition, backbone, dataset)]
                logical = standalone_logical_run(condition, backbone, dataset, seed)
                for fold in ((0,) if stage == "smoke" else OUTER_FOLDS):
                    train_key = f"{logical}:standalone:{fold}:train:none"
                    eval_key = f"{logical}:standalone:{fold}:best_eval:none"
                    head_base = f"{logical}:standalone:{fold}:head"
                    jobs.extend(
                        [
                            MatrixJob(logical, condition, backbone, "standalone", "train", None, dataset, seed, fold, config, (), "1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4", "original_teacher_forced", "subject", "original_teacher_forced", None),
                            MatrixJob(logical, condition, backbone, "standalone", "best_eval", None, dataset, seed, fold, config, (train_key,), "1 node, 1 task, 1 H100", "harmonized_all_windows_full_coverage", "subject_level", "original_teacher_forced", None),
                            MatrixJob(logical, condition, backbone, "standalone", "head", "logreg", dataset, seed, fold, config, (eval_key,), "1 node, 1 task, 1 H100", "harmonized_all_windows_full_coverage", "subject_level", _standalone_backend(backbone, "logreg"), None),
                            MatrixJob(logical, condition, backbone, "standalone", "head", "xgb_optuna100", dataset, seed, fold, config, (f"{head_base}:logreg",), "1 node, 1 task, 1 CPU lane", "harmonized_all_windows_full_coverage", "subject_level", _standalone_backend(backbone, "xgb_optuna100"), 2 if stage == "smoke" else 100),
                        ]
                    )
    for condition, backbone in panels:
        for seed in seeds:
            logical = merged_logical_run(condition, backbone, seed, "cv")
            config = MERGED_CONFIGS[(condition, backbone)]
            for fold in ((0,) if stage == "smoke" else OUTER_FOLDS):
                train_key = f"{logical}:merged:{fold}:train:none"
                post_key = f"{logical}:merged:{fold}:postprocess:none"
                logreg_key = f"{logical}:merged:{fold}:head:logreg"
                jobs.extend(
                    [
                        MatrixJob(logical, condition, backbone, "merged_cv", "train", None, "merged", seed, fold, config, (), "1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4", "original_teacher_forced", "merged_cv_fold", "original_teacher_forced", None),
                        MatrixJob(logical, condition, backbone, "merged_cv", "postprocess", None, "merged", seed, fold, config, (train_key,), "1 node, 1 task, 1 H100", "harmonized_all_windows_full_coverage", "subject_level", "original_teacher_forced", None),
                        MatrixJob(logical, condition, backbone, "merged_cv", "head", "logreg", "merged", seed, fold, config, (post_key,), "1 node, 1 task, 1 CPU lane", "harmonized_all_windows_full_coverage", "merged_cv_fold", _merged_backend(backbone, "logreg"), None),
                        MatrixJob(logical, condition, backbone, "merged_cv", "head", "xgb_optuna100", "merged", seed, fold, config, (logreg_key,), "1 node, 1 task, 1 CPU lane", "harmonized_all_windows_full_coverage", "merged_cv_fold", _merged_backend(backbone, "xgb_optuna100"), 2 if stage == "smoke" else 100),
                    ]
                )
            if stage == "production":
                final_logical = merged_logical_run(condition, backbone, seed, "final")
                final_train_key = f"{final_logical}:merged_final:0:train:none"
                final_post_key = f"{final_logical}:merged_final:0:postprocess:none"
                final_logreg_key = f"{final_logical}:merged_final:0:head:logreg"
                jobs.extend(
                    [
                        MatrixJob(final_logical, condition, backbone, "merged_final", "train", None, "merged", seed, 0, config, (), "1 node, 4 tasks, 4 H100, NPROC_PER_NODE=4", "original_teacher_forced", "final_daic_subject_level", "original_teacher_forced", None),
                        MatrixJob(final_logical, condition, backbone, "merged_final", "postprocess", None, "merged", seed, 0, config, (final_train_key,), "1 node, 1 task, 1 H100", "harmonized_all_windows_full_coverage", "final_daic_subject_level", "original_teacher_forced", None),
                        MatrixJob(final_logical, condition, backbone, "merged_final", "head", "logreg", "merged", seed, 0, config, (final_post_key,), "1 node, 1 task, 1 CPU lane", "harmonized_all_windows_full_coverage", "final_daic_subject_level", _merged_backend(backbone, "logreg"), None),
                        MatrixJob(final_logical, condition, backbone, "merged_final", "head", "xgb_optuna100", "merged", seed, 0, config, (final_logreg_key,), "1 node, 1 task, 1 CPU lane", "harmonized_all_windows_full_coverage", "final_daic_subject_level", _merged_backend(backbone, "xgb_optuna100"), 100),
                    ]
                )
    return jobs


def matrix_counts(stage: str = "production") -> dict[str, int]:
    jobs = build_matrix(stage)
    return {
        "total": len(jobs),
        "train": sum(job.job_type == "train" for job in jobs),
        "best_eval": sum(job.job_type == "best_eval" for job in jobs),
        "postprocess": sum(job.job_type == "postprocess" for job in jobs),
        "logreg": sum(job.method == "logreg" for job in jobs),
        "xgb_optuna100": sum(job.method == "xgb_optuna100" for job in jobs),
    }


def matrix_payload(stage: str = "production") -> dict[str, Any]:
    jobs = build_matrix(stage)
    return {
        "schema_version": "native_en_text_heads_v2_matrix.v1",
        "group_id": GROUP_ID,
        "stage": stage,
        "training_seeds": list((1337,) if stage == "smoke" else TRAINING_SEEDS),
        "split_seed": SPLIT_SEED,
        "head_seed": HEAD_SEED,
        "counts": matrix_counts(stage),
        "jobs": [job.as_dict() for job in jobs],
    }
