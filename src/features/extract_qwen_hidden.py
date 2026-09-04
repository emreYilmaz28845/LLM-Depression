from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.emotion import load_emotion_cache, report_cache_coverage, resolve_missing_policy, use_emotion
from src.data.runtime import (
    JOINT_PACKED30_MODE,
    QUESTION_CONTEXT_SENTENCES,
    build_examples,
    filter_rows_by_subjects,
    load_manifest_rows,
    render_joint_packed30_bundle,
)
from src.daic_chunking import build_joint_epoch_schedule
from src.features.gemma4_hidden_collator import (
    GEMMA4_MODEL_INPUT_KEYS,
    Gemma4PromptOnlyExtractionCollator,
)
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator, load_prompt_audio
from src.model.runtime import (
    load_model_for_inference,
    load_processor,
    prepare_backend_examples,
    resolve_processor_sampling_rate,
)
from src.utils import (
    MODEL_BACKEND_GEMMA4,
    MODEL_BACKEND_QWEN2AUDIO,
    MODEL_BACKEND_TEXT,
    read_json,
    resolve_model_backend,
    save_json,
    sha256_file,
    sha256_jsonl_rows,
    sha256_text,
    write_jsonl,
)


BACKEND_HIDDEN_SIZES: dict[str, set[int]] = {
    MODEL_BACKEND_GEMMA4: {3840},
    MODEL_BACKEND_QWEN2AUDIO: {4096},
    MODEL_BACKEND_TEXT: {3584},
}
QWEN_HIDDEN_SIZES = {3584, 4096}
POOLING_NAME = "last_valid_prompt_token"
CONDITION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
CACHE_SCHEMA_VERSION_QWEN = "qwen_hidden_cache.v2"
CACHE_SCHEMA_VERSION_GEMMA4 = "gemma4_hidden_cache.v1"
CACHE_ARTIFACT_NAMES = (
    "outer_train.npz",
    "outer_train_rows.jsonl",
    "final_eval.npz",
    "final_eval_rows.jsonl",
    "extraction_metadata.json",
)


def resolve_condition(value: str | None, input_modality: str | None, emotion_enabled: bool) -> str:
    condition = str(value or input_modality or "").strip()
    if not CONDITION_PATTERN.fullmatch(condition):
        raise ValueError(f"Invalid experiment condition: {condition!r}.")
    if emotion_enabled and condition == str(input_modality):
        raise ValueError("Emotion runs require an explicit condition distinct from input_modality.")
    return condition


def _emotion_provenance(
    config: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    partition_subject_ids: dict[str, list[str]],
    *,
    source: str | None,
    language: str | None,
) -> dict[str, Any] | None:
    if not use_emotion(config):
        if source or language:
            raise ValueError("Emotion source/language metadata was supplied for a non-emotion run.")
        return None
    if not source or not language:
        raise ValueError("Emotion runs require explicit source and language provenance.")
    data = config["data"]
    cache_path = _saved_path(data["emotion_cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(f"Saved emotion cache is unavailable: {cache_path}")
    caption_field = str(data.get("emotion_caption_field", "emotion_en"))
    cache = load_emotion_cache(cache_path, caption_field=caption_field)
    train_ids = set(partition_subject_ids["outer_train"])
    heldout_ids = set(partition_subject_ids["final_eval"])

    def coverage(subject_ids: set[str]) -> dict[str, int]:
        sample_ids = [
            str(row["sample_id"])
            for row in manifest_rows
            if str(row["subject_id"]) in subject_ids
        ]
        return report_cache_coverage(cache, sample_ids)

    policy = resolve_missing_policy(config)
    partition_coverage = {
        "outer_train": coverage(train_ids),
        "final_eval": coverage(heldout_ids),
    }
    missing_total = sum(
        stats["missing_row"] + stats["null_caption"] for stats in partition_coverage.values()
    )
    return {
        "enabled": True,
        "source": source,
        "language": language,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "caption_field": caption_field,
        "missing_policy": policy,
        "partition_coverage": partition_coverage,
        "fallback_caption_count": missing_total if policy == "neutral_fallback" else 0,
        "dropped_caption_count": missing_total if policy == "drop_emotion_line" else 0,
    }


def _saved_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    marker = "LLM-Depression/"
    text = str(path)
    if marker in text:
        candidate = PROJECT_ROOT / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return path


def _load_saved_run(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    fold_dir = checkpoint_dir.parent
    run_config_path = fold_dir / "run_config.yaml"
    split_path = fold_dir / "logs" / "split_used.json"
    if not run_config_path.exists() or not split_path.exists():
        raise FileNotFoundError("Checkpoint requires sibling run_config.yaml and logs/split_used.json.")
    import yaml

    saved = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise ValueError(f"Invalid saved run configuration: {run_config_path}")
    return saved, saved["config"], run_config_path, split_path


def _git_commit() -> str:
    # Cluster copies are synchronized without .git; an old checkout may still
    # leave a stale .git directory behind, so the captured sync provenance wins.
    provenance = PROJECT_ROOT / ".provenance" / "git_commit.txt"
    if provenance.exists():
        return provenance.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _decoder_hidden_size(model, config: dict[str, Any] | None = None) -> int:
    backend = resolve_model_backend(config or {})
    supported = BACKEND_HIDDEN_SIZES.get(backend, QWEN_HIDDEN_SIZES)
    configs = [getattr(model, "config", None)]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        configs.append(getattr(base_model, "config", None))
        configs.append(getattr(getattr(base_model, "model", None), "config", None))
    for config in configs:
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if hidden_size is not None:
            hidden_size = int(hidden_size)
            if hidden_size not in supported:
                raise ValueError(
                    f"Unexpected decoder hidden size {hidden_size} for backend "
                    f"{backend or 'default'}; expected one of {sorted(supported)}."
                )
            return hidden_size
    raise ValueError("Could not resolve the decoder hidden size from the loaded model configuration.")


def _backend_cache_schema(config: dict[str, Any]) -> str:
    if resolve_model_backend(config) == MODEL_BACKEND_GEMMA4:
        return CACHE_SCHEMA_VERSION_GEMMA4
    return CACHE_SCHEMA_VERSION_QWEN


def _parent_attempt_id(checkpoint_dir: Path) -> str | None:
    metadata_path = checkpoint_dir.parent / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
    except (ValueError, OSError):
        return None
    attempt_id = metadata.get("attempt_id")
    return str(attempt_id) if isinstance(attempt_id, str) and attempt_id else None


def _is_joint_packed30_recipe(config: dict[str, Any]) -> bool:
    return (
        str(config.get("dataset", "")).lower() == "daic"
        and str(config.get("data", {}).get("sample_mode", "")).strip().lower()
        == JOINT_PACKED30_MODE
        and str(config.get("data", {}).get("train_chunk_policy", "")).strip().lower()
        == "joint_random_k"
    )


def _is_turkish_pooled(config: dict[str, Any]) -> bool:
    return (
        str(config.get("dataset", "")).lower() == "turkish"
        and str(config.get("dataset_variant", "")).strip() == "pooled_t17"
    )


def _is_turkish_pooled_text(config: dict[str, Any]) -> bool:
    return (
        _is_turkish_pooled(config)
        and not bool(config.get("data", {}).get("use_audio", False))
        and bool(config.get("data", {}).get("use_text", False))
    )


def _selected_epoch_fit_examples(
    checkpoint_dir: Path,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    fold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the exact random joint schedule of the epoch that produced
    ``best_model`` and return one prompt-only example per training subject.

    Locked head-fit view: read ``selected_epoch`` (one-based) from
    ``logs/selected_checkpoint_selection_metrics.json``, rebuild all random
    joint schedules from the saved config and seed, select zero-based schedule
    index ``selected_epoch - 1``, and prove that its canonical hash and every
    subject's ``bundle_chunk_ids`` match ``logs/daic_chunk_schedule_audit.json``.
    Any missing or inconsistent provenance is a failure.
    """
    fold_dir = checkpoint_dir.parent
    selection_metrics_path = fold_dir / "logs" / "selected_checkpoint_selection_metrics.json"
    saved_audit_path = fold_dir / "logs" / "daic_chunk_schedule_audit.json"
    if not selection_metrics_path.is_file():
        raise FileNotFoundError(
            f"Selected-epoch head fit requires {selection_metrics_path}."
        )
    if not saved_audit_path.is_file():
        raise FileNotFoundError(
            f"Selected-epoch head fit requires {saved_audit_path}."
        )
    selection_metrics = read_json(selection_metrics_path)
    selected_epoch = int(selection_metrics.get("selected_epoch"))
    if selected_epoch < 1:
        raise ValueError(
            f"Invalid selected_epoch={selected_epoch}; expected a one-based epoch."
        )
    schedule_index = selected_epoch - 1
    saved_audit = read_json(saved_audit_path)
    train_examples = build_examples(
        train_rows, config, partition_name="outer_train", truncation_log_path=None
    )
    schedules, rebuilt_audit = build_joint_epoch_schedule(
        train_examples,
        policy=str(config["data"]["train_chunk_policy"]),
        k=int(config["data"]["train_chunks_per_subject"]),
        seed=int(config["seed"]),
        epochs=int(config["training"]["num_train_epochs"]),
        loss_weight_rescale=str(config["data"].get("loss_weight_rescale", "mean_one")),
        class_balance=str(config.get("training", {}).get("class_balance", "none"))
        == "subject_inverse_frequency",
    )
    if schedule_index >= len(schedules):
        raise ValueError(
            f"selected_epoch={selected_epoch} exceeds the rebuilt schedule "
            f"({len(schedules)} epochs)."
        )
    if saved_audit.get("schedule_sha256") != rebuilt_audit["schedule_sha256"]:
        raise ValueError(
            "Rebuilt joint schedule hash does not match the saved "
            "daic_chunk_schedule_audit.json."
        )
    if saved_audit.get("bundle_membership_sha256") != rebuilt_audit["bundle_membership_sha256"]:
        raise ValueError(
            "Rebuilt bundle memberships do not match the saved "
            "daic_chunk_schedule_audit.json."
        )
    saved_memberships: dict[str, list[str]] = {}
    for row in saved_audit.get("rows", []):
        if int(row["epoch"]) != schedule_index:
            continue
        saved_memberships[str(row["subject_id"])] = list(row["bundle_chunk_ids"])
    rebuilt_rows = schedules[schedule_index]
    for row in rebuilt_rows:
        subject_id = str(row["subject_id"])
        saved_ids = saved_memberships.get(subject_id)
        if saved_ids is None:
            raise ValueError(
                f"Saved schedule epoch {schedule_index} has no membership for "
                f"subject {subject_id}."
            )
        if saved_ids != list(row["bundle_chunk_ids"]):
            raise ValueError(
                f"Rebuilt bundle membership for subject {subject_id} in epoch "
                f"{schedule_index} does not match the saved schedule."
            )
    fit_examples: list[dict[str, Any]] = []
    for row in rebuilt_rows:
        prompt_text, training_text = render_joint_packed30_bundle(
            row, len(row["audio_span_groups"])
        )
        fit_example = dict(row)
        fit_example["prompt_text"] = prompt_text
        fit_example["training_text"] = training_text
        fit_example["partition"] = "outer_train"
        fit_example["fold"] = fold
        fit_examples.append(fit_example)
    fit_subjects = {str(example["subject_id"]) for example in fit_examples}
    if len(fit_examples) != len(fit_subjects):
        raise ValueError(
            "Selected-epoch head fit must emit exactly one bundle per training "
            "subject."
        )
    expected_subjects = {str(row["subject_id"]) for row in train_rows}
    if fit_subjects != expected_subjects:
        raise ValueError(
            "Selected-epoch head fit subjects differ from the training partition."
        )
    provenance = {
        "head_fit_view": "selected_checkpoint_training_epoch",
        "selected_epoch": selected_epoch,
        "schedule_epoch_index": schedule_index,
        "schedule_sha256": rebuilt_audit["schedule_sha256"],
        "bundle_membership_sha256": rebuilt_audit["bundle_membership_sha256"],
        "saved_schedule_audit": str(saved_audit_path),
        "selection_metrics": str(selection_metrics_path),
    }
    return fit_examples, provenance


def _partition_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_subject_ids: dict[str, list[str]],
    fold: int,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    result: dict[str, list[dict[str, Any]]] = {}
    joint_provenance: dict[str, Any] | None = None
    for partition in ("outer_train", "final_eval"):
        ids = partition_subject_ids[partition]
        rows = filter_rows_by_subjects(manifest_rows, ids)
        if partition == "outer_train" and _is_joint_packed30_recipe(config):
            if checkpoint_dir is None:
                raise ValueError(
                    "Joint packed30 head-fit extraction requires a checkpoint dir."
                )
            examples, joint_provenance = _selected_epoch_fit_examples(
                checkpoint_dir, config, rows, fold
            )
        else:
            examples = build_examples(rows, config, partition_name=partition, truncation_log_path=None)
        for example in examples:
            example["partition"] = partition
            example["fold"] = fold
        example_subjects = {str(item["subject_id"]) for item in examples}
        if example_subjects != set(ids):
            raise ValueError(
                f"{partition} example subjects differ from saved split: "
                f"missing={sorted(set(ids)-example_subjects)[:10]} extra={sorted(example_subjects-set(ids))[:10]}"
            )
        result[partition] = examples
    return result, joint_provenance or {}


def _is_daic_chunking(config: dict[str, Any]) -> bool:
    return (
        str(config.get("dataset", "")).lower() == "daic"
        and (
            str(config.get("protocol_id", "")).strip().lower()
            == "daic_participant_speech_packed30_v1"
            or str(
                config.get("evaluation", {}).get("subject_score_aggregation", "")
            ).lower()
            == "mean_score"
        )
    )


def _resolve_subject_partitions(
    saved: dict[str, Any],
    config: dict[str, Any],
    split_payload: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    cv_protocol = str(
        saved.get("cv_protocol")
        or config.get("split", {}).get("cv_protocol")
        or ""
    )
    is_daic_chunking = _is_daic_chunking(config)
    if is_daic_chunking:
        train_sources = ("train_subject_ids",)
        heldout_source = "final_eval_subject_ids"
        final_partition = str(config.get("split", {}).get("final_eval_partition", "test"))
        has_selection_partition = bool(
            str(config.get("split", {}).get("selection_partition", "")).strip()
        )
        if final_partition == "val" and not has_selection_partition:
            # Official-development protocol: the saved train_subject_ids are the
            # 86 inner-training subjects and the saved final_eval_subject_ids
            # are the 35 official development subjects. Derived from the saved
            # config, never from hard-coded test strings.
            evaluation_protocol = "daic_official_train_inner_split_dev_evaluation"
        else:
            evaluation_protocol = "daic_official_train_fit_locked_test_evaluation"
    elif cv_protocol == "train_val":
        train_sources = ("train_subject_ids",)
        heldout_source = "selection_subject_ids"
        evaluation_protocol = "table_aligned_outer_validation"
    else:
        train_sources = ("train_subject_ids", "selection_subject_ids")
        heldout_source = "final_eval_subject_ids"
        evaluation_protocol = "saved_final_evaluation"

    train_ids = sorted({
        str(subject_id)
        for source in train_sources
        for subject_id in split_payload.get(source, [])
    })
    heldout_ids = sorted({str(subject_id) for subject_id in split_payload.get(heldout_source, [])})
    if not train_ids:
        raise ValueError("Resolved outer_train subject set is empty.")
    if not heldout_ids:
        raise ValueError(
            f"Resolved final_eval subject set is empty (source={heldout_source}, cv_protocol={cv_protocol!r})."
        )
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise ValueError(f"Training/held-out subject overlap: {overlap[:10]}")
    partitions = {"outer_train": train_ids, "final_eval": heldout_ids}
    provenance = {
        "evaluation_protocol": evaluation_protocol,
        "split_name": str(config.get("split", {}).get("final_eval_partition", "test")),
        "saved_cv_protocol": cv_protocol or None,
        "partition_sources": {
            "outer_train": list(train_sources),
            "final_eval": [heldout_source],
        },
        "compatibility_note": (
            "The final_eval cache contains the saved outer-fold validation subjects."
            if cv_protocol == "train_val"
            else None
        ),
    }
    return partitions, provenance


def _validate_saved_split(
    saved: dict[str, Any],
    config: dict[str, Any],
    partition_subject_ids: dict[str, list[str]],
    fold: int,
    train_source_count: int,
    split_payload: dict[str, Any] | None = None,
) -> Path:
    split_metadata_path = _saved_path(saved["split_metadata_path"])
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Saved split metadata is unavailable: {split_metadata_path}")
    if saved.get("split_metadata_hash") and sha256_file(split_metadata_path) != saved["split_metadata_hash"]:
        raise ValueError("Current split metadata hash does not match the checkpoint's saved split hash.")
    train_ids = set(partition_subject_ids["outer_train"])
    heldout_ids = set(partition_subject_ids["final_eval"])
    split_metadata = read_json(split_metadata_path)
    if str(saved.get("split_mode", "")) == "cv":
        fold_payload = split_metadata[str(fold)] if str(fold) in split_metadata else split_metadata[fold]
        expected_train = set(fold_payload["outer_train_subject_ids"])
        expected_heldout = set(fold_payload["final_eval_subject_ids"])
    else:
        is_daic_chunking = _is_daic_chunking(config)
        split_payload = split_payload or {}
        split_names = split_payload.get("split_names") or {}
        inner_split_mode = str(split_names.get("train", "")) == "train_inner"
        official_train_set: set[str] | None = None
        official_heldout_set: set[str] | None = None
        if is_daic_chunking and inner_split_mode:
            # Official-development inner-split protocol: the saved
            # train_subject_ids are the inner-training subjects and the saved
            # final_eval_subject_ids are the official development partition.
            # The saved split must prove its official origins: train_inner and
            # val_inner partition the official train partition exactly, the
            # val_inner set equals the saved selection set, and the official
            # development partition stays disjoint.
            train_partition = str(config["split"]["train_partition"])
            final_partition = str(config["split"]["final_eval_partition"])
            official_train_set = {
                str(row["subject_id"])
                for row in split_metadata
                if str(row["partition"]) == train_partition
            }
            official_heldout_set = {
                str(row["subject_id"])
                for row in split_metadata
                if str(row["partition"]) == final_partition
            }
            saved_train_inner = {
                str(subject_id)
                for subject_id in split_payload.get("train_inner_subject_ids", [])
            }
            saved_val_inner = {
                str(subject_id)
                for subject_id in split_payload.get("val_inner_subject_ids", [])
            }
            saved_selection = {
                str(subject_id)
                for subject_id in split_payload.get("selection_subject_ids", [])
            }
            expected_train = saved_train_inner
            expected_heldout = official_heldout_set
            if not saved_train_inner:
                raise ValueError(
                    "Inner-split checkpoint has no saved train_inner_subject_ids."
                )
            smoke_limit = int(config.get("split", {}).get("smoke_subject_limit", 0) or 0)
            if smoke_limit <= 0:
                # The exact official-origins proof applies to production
                # checkpoints only. Smoke checkpoints are limited subsets and
                # are validated as subsets below.
                if saved_train_inner | saved_val_inner != official_train_set:
                    raise ValueError(
                        "Inner-split checkpoint does not partition the official "
                        "train partition exactly."
                    )
                if saved_val_inner != saved_selection:
                    raise ValueError(
                        "Inner-split checkpoint val_inner set differs from its "
                        "saved selection set."
                    )
                if not saved_train_inner.isdisjoint(official_heldout_set):
                    raise ValueError(
                        "Inner-split checkpoint training subjects overlap the "
                        "official development partition."
                    )
        else:
            dev_partitions = (
                {str(config["split"]["train_partition"])}
                if is_daic_chunking
                else set(config["split"].get("dev_pool_partitions") or [
                    config["split"]["train_partition"],
                    config["split"]["selection_partition"],
                ])
            )
            final_partition = str(config["split"]["final_eval_partition"])
            expected_train = {
                str(row["subject_id"]) for row in split_metadata if str(row["partition"]) in dev_partitions
            }
            expected_heldout = {
                str(row["subject_id"]) for row in split_metadata if str(row["partition"]) == final_partition
            }
    smoke_limit = int(config.get("split", {}).get("smoke_subject_limit", 0) or 0)
    if smoke_limit > 0:
        if not train_ids.issubset(expected_train) or not heldout_ids.issubset(
            expected_heldout
        ):
            raise ValueError(
                "Smoke checkpoint split_used.json is not a subset of its hashed "
                "official split metadata."
            )
        # Fixed train/val/test extraction deliberately fits classifiers on the
        # union of the saved train and selection partitions. A smoke limit is
        # applied independently to every source partition by split creation
        # (see train._apply_smoke_subject_limit), so the outer-train union may
        # contain up to train_source_count * smoke_limit subjects. The count is
        # resolved by the caller from the protocol, mirroring
        # _resolve_subject_partitions (train_val/daic: 1, train_val_test: 2).
        if len(train_ids) > smoke_limit * train_source_count or len(heldout_ids) > smoke_limit:
            raise ValueError(
                "Smoke checkpoint split exceeds split.smoke_subject_limit."
            )
    elif train_ids != expected_train or heldout_ids != expected_heldout:
        raise ValueError(
            "Checkpoint split_used.json does not match its hashed split metadata."
        )
    return split_metadata_path


def _extract_partition(
    *,
    model,
    processor,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    output_dir: Path,
    partition: str,
    max_examples: int | None,
    expected_hidden_size: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    sampling_rate = resolve_processor_sampling_rate(processor)
    gemma_backend = resolve_model_backend(config) == MODEL_BACKEND_GEMMA4
    collator = (
        Gemma4PromptOnlyExtractionCollator(
            processor,
            require_unit_range=str(config.get("dataset", "")).lower() == "daic",
        )
        if gemma_backend
        else PromptOnlyExtractionCollator(processor)
    )
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    mask_sources: dict[str, int] = {}
    determinism_max_abs_diff: float | None = None
    selected = examples[:max_examples] if max_examples else examples
    seen_samples: set[str] = set()
    subject_labels: dict[str, int] = {}
    pooled_turkish = _is_turkish_pooled(config)
    pooled_text = _is_turkish_pooled_text(config)
    condition_counts: Counter[str] = Counter()
    subject_conditions: dict[str, set[str]] = defaultdict(set)
    subject_condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_context_hashes: set[str] = set()
    transcript_hashes: set[str] = set()
    transcript_char_counts: list[int] = []
    for index, raw_example in enumerate(selected, start=1):
        example = load_prompt_audio(raw_example, sampling_rate, bool(config["data"].get("silence_audio", False)))
        model_inputs, metadata_rows = collator([example])
        metadata = metadata_rows[0]
        prompt_text = metadata.pop("prompt_text")
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        if "labels" in model_inputs:
            raise AssertionError("Gold labels must never be passed to the model during extraction.")
        if gemma_backend:
            base_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
            missing_base = base_keys - set(model_inputs)
            if missing_base:
                raise AssertionError(
                    f"Gemma extraction inputs missing required keys: {sorted(missing_base)}"
                )
            if any(
                example.get(key)
                for key in ("audio_paths", "audio_spans", "audio_span_groups", "audio_path")
            ):
                missing_audio = {"input_features", "input_features_mask"} - set(model_inputs)
                if missing_audio:
                    raise AssertionError(
                        f"Gemma audio extraction inputs missing feature keys: {sorted(missing_audio)}"
                    )
            else:
                extra_audio = {"input_features", "input_features_mask"}.intersection(model_inputs)
                if extra_audio:
                    raise AssertionError(
                        f"Gemma text-only extraction must omit audio feature tensors: "
                        f"{sorted(extra_audio)}"
                    )
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                labels=None,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        output_mask = getattr(outputs, "attention_mask", None)
        if gemma_backend:
            if tuple(hidden.shape[:2]) != tuple(model_inputs["attention_mask"].shape):
                raise ValueError(
                    "Gemma hidden states must align with the processor input "
                    f"attention mask; hidden {tuple(hidden.shape[:2])} versus "
                    f"mask {tuple(model_inputs['attention_mask'].shape)}."
                )
            mask, mask_source = model_inputs["attention_mask"].to(hidden.device), "processor_input"
        else:
            mask, mask_source = aligned_attention_mask(hidden, model_inputs["attention_mask"], output_mask)
        vector = last_valid_token(hidden, mask).cpu().numpy()[0].astype(np.float32, copy=False)
        if vector.shape != (expected_hidden_size,):
            raise ValueError(f"Expected {expected_hidden_size} features, got {vector.shape}.")
        if not bool(np.isfinite(vector).all()):
            raise ValueError(f"Non-finite vector for sample {metadata['sample_id']}.")
        if index == 1:
            with torch.inference_mode():
                repeated_outputs = model(
                    **model_inputs,
                    labels=None,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            repeated_hidden = repeated_outputs.hidden_states[-1]
            if gemma_backend:
                repeated_mask = model_inputs["attention_mask"].to(repeated_hidden.device)
            else:
                repeated_mask, _ = aligned_attention_mask(
                    repeated_hidden,
                    model_inputs["attention_mask"],
                    getattr(repeated_outputs, "attention_mask", None),
                )
            repeated_vector = (
                last_valid_token(repeated_hidden, repeated_mask).cpu().numpy()[0].astype(np.float32, copy=False)
            )
            determinism_max_abs_diff = float(np.max(np.abs(vector - repeated_vector)))
            if not np.allclose(vector, repeated_vector, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"Determinism check failed for {metadata['sample_id']}: "
                    f"max_abs_diff={determinism_max_abs_diff}"
                )
        sample_id = metadata["sample_id"]
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample ID within {partition}: {sample_id}")
        seen_samples.add(sample_id)
        subject_id = metadata["subject_id"]
        label = int(metadata["label"])
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        subject_labels[subject_id] = label
        if pooled_turkish:
            condition = str(metadata.get("question_condition", "")).strip()
            if condition not in {"pos_only_t17", "negative_only_t17"}:
                raise ValueError(
                    "Pooled Turkish hidden extraction requires exactly the two "
                    f"question conditions, got {condition!r}."
                )
            condition_counts[condition] += 1
            subject_conditions[subject_id].add(condition)
            subject_condition_counts[subject_id][condition] += 1
            transcript = str(example.get("transcript", ""))
            transcript_block = (
                f"The transcript of the subject's speech is:\n{transcript}\n\n"
                if bool(config.get("data", {}).get("use_text", False))
                else ""
            )
            if transcript_block:
                if transcript_block not in prompt_text:
                    raise ValueError(
                        "Pooled Turkish hidden extraction could not locate the "
                        "transcript block for redaction."
                    )
                prompt_context = prompt_text.replace(
                    transcript_block,
                    "The transcript of the subject's speech is:\n<TRANSCRIPT>\n\n",
                    1,
                )
            else:
                prompt_context = prompt_text
            prompt_context_hash = sha256_text(prompt_context)
            transcript_hash = sha256_text(transcript)
            prompt_context_hashes.add(prompt_context_hash)
            transcript_hashes.add(transcript_hash)
            transcript_char_counts.append(len(transcript))
        else:
            prompt_context_hash = ""
            transcript_hash = ""
        metadata.update({
            "prompt_sha256": sha256_text(prompt_text),
            "hidden_layer": "final",
            "pooling": POOLING_NAME,
            "vector_dimension": expected_hidden_size,
            "vector_dtype": "float32",
            "mask_source": mask_source,
            "checkpoint": str(checkpoint_dir),
        })
        if pooled_turkish:
            metadata.update({
                "prompt_context_sha256": prompt_context_hash,
                "transcript_sha256": transcript_hash,
                "transcript_chars": len(transcript),
            })
        rows.append(metadata)
        vectors.append(vector)
        mask_sources[mask_source] = mask_sources.get(mask_source, 0) + 1
        if index % 25 == 0 or index == len(selected):
            print(f"{partition}: extracted {index}/{len(selected)}", flush=True)
    matrix = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, expected_hidden_size), np.float32)
    )
    np.savez_compressed(output_dir / f"{partition}.npz", vectors=matrix)
    write_jsonl(rows, output_dir / f"{partition}_rows.jsonl")
    summary = {
        "rows": len(rows),
        "subjects": len(subject_labels),
        "mask_sources": mask_sources,
        "determinism_rtol": 1e-5,
        "determinism_atol": 1e-5,
        "determinism_max_abs_diff": determinism_max_abs_diff,
    }
    if pooled_turkish:
        expected = {"pos_only_t17", "negative_only_t17"}
        if set(condition_counts) != expected:
            raise ValueError(
                "Pooled Turkish hidden extraction conditions differ from the "
                f"locked pair: {sorted(condition_counts)}"
            )
        summary.update({
            "condition_counts": dict(sorted(condition_counts.items())),
            "prompt_context_sha256": sorted(prompt_context_hashes),
            "transcript_sha256": sorted(transcript_hashes),
            "transcript_chars": {
                "min": min(transcript_char_counts) if transcript_char_counts else 0,
                "max": max(transcript_char_counts) if transcript_char_counts else 0,
                "total": sum(transcript_char_counts),
            },
        })
        if pooled_text:
            invalid_subjects = sorted(
                subject_id for subject_id, conditions in subject_conditions.items()
                if conditions != expected
            )
            if invalid_subjects:
                raise ValueError(
                    "Pooled Turkish text extraction must contain both conditions for every subject: "
                    f"{invalid_subjects[:10]}"
                )
            duplicate_subjects = sorted(
                subject_id
                for subject_id, counts in subject_condition_counts.items()
                if counts != Counter({condition: 1 for condition in expected})
            )
            if duplicate_subjects:
                raise ValueError(
                    "Pooled Turkish text extraction must contain exactly one example "
                    "per condition for every subject: "
                    f"{duplicate_subjects[:10]}"
                )
            summary["paired_text_examples_per_subject"] = 2
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract prompt-only final hidden vectors.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--manifest-path", type=Path, help="Relocated copy of the saved manifest; hash must match.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--condition", help="Unique experiment condition used in metadata and output grouping.")
    parser.add_argument("--emotion-source", help="Predeclared source label for an emotion cache.")
    parser.add_argument("--emotion-language", help="Predeclared language label for emotion captions.")
    parser.add_argument(
        "--subject-selection",
        type=Path,
        help="Isolated-smoke-only JSON listing explicit train and test subject IDs; "
        "hashed into the cache identity. Production attempts must omit it.",
    )
    parser.add_argument(
        "--eval-chunk-policy",
        choices=("fixed_k", "balanced_joint_cover", "fixed_count_balanced_joint_cover", "all", "matched_k"),
        help="DAIC evaluation-view override; recorded in cache identity.",
    )
    parser.add_argument("--eval-chunks-per-subject")
    parser.add_argument("--eval-bundles-per-subject", type=int)
    return parser.parse_args()


def _require_best_model(checkpoint_dir: Path) -> None:
    if Path(checkpoint_dir).name != "best_model":
        raise ValueError(
            "Primary experiment requires the fold-specific best_model checkpoint; "
            f"got {Path(checkpoint_dir).name!r}. last_model must never be substituted."
        )


def load_subject_selection(path: Path | None) -> dict[str, Any] | None:
    """Load and validate an isolated-smoke subject-selection file.

    The file must list explicit ``outer_train`` and ``final_eval`` subject IDs.
    Production attempts must omit the argument entirely; the campaign wrapper
    refuses to pass it. Its canonical hash becomes part of the cache identity,
    so a smoke cache can never collide with a production cache.
    """
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Subject-selection file is unavailable: {target}")
    try:
        payload = read_json(target)
    except (ValueError, OSError) as error:
        raise ValueError(f"Unreadable subject-selection file {target}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Subject-selection file must be a JSON object.")
    missing = [name for name in ("outer_train", "final_eval") if name not in payload]
    if missing:
        raise ValueError(
            f"Subject-selection file {target} is missing keys: {missing}."
        )
    resolved: dict[str, list[str]] = {}
    for name in ("outer_train", "final_eval"):
        values = payload[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Subject-selection {name} must be a non-empty list.")
        ids = sorted({str(value) for value in values})
        if not ids or len(ids) != len(values):
            raise ValueError(f"Subject-selection {name} contains duplicates or empties.")
        resolved[name] = ids
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "outer_train": resolved["outer_train"],
        "final_eval": resolved["final_eval"],
    }


def _apply_subject_selection(
    partition_subject_ids: dict[str, list[str]],
    selection: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
    split_payload: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Restrict saved partition subjects to the explicit smoke selection.

    Production smokes for the official-test protocol restrict both partitions
    to subsets of the saved split. Official-development smokes draw their
    fit subjects from the saved training partition and their eval subjects
    from the official training pool (train plus selection of the saved split)
    so that no official-development or official-test subject can enter a
    smoke; the fit and eval smoke sets must stay disjoint.
    """
    if selection is None:
        return partition_subject_ids
    officialdev = (
        config is not None
        and str(config.get("split", {}).get("final_eval_partition", "")) == "val"
        and not str(config.get("split", {}).get("selection_partition", "")).strip()
    )
    restricted: dict[str, list[str]] = {}
    for partition in ("outer_train", "final_eval"):
        saved = set(partition_subject_ids[partition])
        chosen = set(selection[partition])
        if partition == "final_eval" and officialdev:
            # Official-development smokes must draw their eval subjects from
            # the official training pool, never from the saved official
            # development partition or the official test partition.
            pool = set(partition_subject_ids["outer_train"]) | set(
                str(item) for item in (split_payload or {}).get("selection_subject_ids", [])
            )
            if not chosen.issubset(pool):
                raise ValueError(
                    f"Subject-selection final_eval subjects not in the official "
                    f"training pool: {sorted(chosen - pool)[:10]}"
                )
            if chosen & set(selection.get("outer_train", [])):
                raise ValueError(
                    "Smoke fit and eval subject sets must stay disjoint."
                )
        elif not chosen.issubset(saved):
            raise ValueError(
                f"Subject-selection {partition} subjects not in the saved split: "
                f"{sorted(chosen - saved)[:10]}"
            )
        if not chosen:
            raise ValueError(f"Subject-selection {partition} resolved empty.")
        restricted[partition] = sorted(chosen)
    return restricted


def _existing_cache_decision(
    output_dir: Path,
    cache_config: dict[str, Any],
    cache_config_sha256: str,
) -> str:
    """Decide whether an existing cache output may be reused.

    Returns ``"write"`` for an absent or empty output dir,
    ``"skipped_compatible_complete_cache"`` for a complete cache whose identity
    matches exactly, and raises on a partial or identity-mismatched cache.
    Never overwrites evidence.
    """
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return "write"
    metadata_path = output_dir / "extraction_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"Non-empty hidden cache has no extraction_metadata.json: {output_dir}. "
            "Refusing to overwrite a partial cache."
        )
    existing = read_json(metadata_path)
    if (
        existing.get("cache_config") != cache_config
        or existing.get("cache_config_sha256") != cache_config_sha256
    ):
        raise ValueError(f"Existing hidden cache is incompatible: {output_dir}.")
    if not all((output_dir / name).is_file() for name in CACHE_ARTIFACT_NAMES):
        raise ValueError(f"Existing hidden cache is partial: {output_dir}.")
    return "skipped_compatible_complete_cache"


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    saved, config, run_config_path, split_path = _load_saved_run(checkpoint_dir)
    config = json.loads(json.dumps(config))
    if args.eval_chunk_policy:
        config.setdefault("data", {})["eval_chunk_policy"] = args.eval_chunk_policy
    if args.eval_chunks_per_subject:
        value = args.eval_chunks_per_subject
        config.setdefault("data", {})["eval_chunks_per_subject"] = (
            value if value == "all" else int(value)
        )
    if args.eval_bundles_per_subject is not None:
        config.setdefault("data", {})["eval_bundles_per_subject"] = args.eval_bundles_per_subject
    extraction_dtype = os.environ.get("EXTRACTION_INFERENCE_DTYPE", "").strip().lower()
    if extraction_dtype:
        config.setdefault("evaluation", {})["inference_dtype"] = extraction_dtype
    fold = int(saved["fold"])
    _require_best_model(checkpoint_dir)
    split_payload = read_json(split_path)
    partition_subject_ids, evaluation_provenance = _resolve_subject_partitions(
        saved, config, split_payload
    )
    cv_protocol = str(
        saved.get("cv_protocol") or config.get("split", {}).get("cv_protocol") or ""
    )
    train_source_count = 1 if (cv_protocol == "train_val" or _is_daic_chunking(config)) else 2
    split_metadata_path = _validate_saved_split(
        saved, config, partition_subject_ids, fold, train_source_count, split_payload=split_payload
    )
    manifest_path = args.manifest_path.resolve() if args.manifest_path else _saved_path(saved["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest is unavailable: {manifest_path}")
    manifest_rows = load_manifest_rows(manifest_path)
    canonical_manifest_hash = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and canonical_manifest_hash != saved["manifest_hash"]:
        raise ValueError("Current manifest hash does not match the checkpoint's saved manifest hash.")
    condition = resolve_condition(args.condition, saved.get("input_modality"), use_emotion(config))
    gemma_backend = resolve_model_backend(config) == MODEL_BACKEND_GEMMA4
    subject_selection = load_subject_selection(args.subject_selection)
    cache_config = {
        "schema_version": _backend_cache_schema(config),
        "dataset": config["dataset"],
        "condition": condition,
        "input_modality": saved.get("input_modality"),
        "protocol_id": config.get("protocol_id", ""),
        "fold": fold,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_sha256": sha256_file(checkpoint_dir / "adapter_config.json"),
        "adapter_sha256": sha256_file(checkpoint_dir / "adapter_model.safetensors"),
        "saved_run_config_sha256": sha256_file(run_config_path),
        "saved_split_sha256": sha256_file(split_path),
        "split_metadata_sha256": sha256_file(split_metadata_path),
        "manifest_sha256": canonical_manifest_hash,
        "max_examples": args.max_examples,
        "evaluation_view": {
            "sample_mode": config.get("data", {}).get("sample_mode"),
            "eval_chunk_policy": config.get("data", {}).get("eval_chunk_policy"),
            "eval_chunks_per_subject": config.get("data", {}).get(
                "eval_chunks_per_subject",
                config.get("data", {}).get("chunks_per_subject"),
            ),
            "subject_score_aggregation": config.get("evaluation", {}).get(
                "subject_score_aggregation"
            ),
        },
        "subject_selection_sha256": (
            subject_selection["sha256"] if subject_selection is not None else None
        ),
    }
    if _is_turkish_pooled(config):
        cache_config["dataset_variant"] = config.get("dataset_variant", "")
        cache_config["prompt_context_contract"] = {
            "user_template_redacted": str(config.get("prompt", {}).get("user_template", "")),
            "transcript_block_redacted": True,
            "question_context_sentences": dict(QUESTION_CONTEXT_SENTENCES),
        }
        if _is_turkish_pooled_text(config):
            cache_config.update({
                "aggregation_policy": config.get("evaluation", {}).get("subject_score_aggregation"),
                "paired_text_examples_per_subject": 2,
            })
    if gemma_backend:
        cache_config["model_backend"] = MODEL_BACKEND_GEMMA4
        cache_config["base_model_revision"] = str(config.get("model_revision", ""))
        cache_config["parent_attempt_id"] = _parent_attempt_id(checkpoint_dir)
        cache_config["parent_checkpoint_role"] = "best_model"
        cache_config["parent_checkpoint_path"] = str(checkpoint_dir)
        cache_config["source_git_sha256"] = _git_commit()
    cache_config_sha256 = sha256_text(
        json.dumps(cache_config, sort_keys=True, separators=(",", ":"))
    )
    cache_decision = _existing_cache_decision(output_dir, cache_config, cache_config_sha256)
    if cache_decision == "skipped_compatible_complete_cache":
        print(
            json.dumps(
                {
                    "status": cache_decision,
                    "output_dir": str(output_dir),
                    "cache_config_sha256": cache_config_sha256,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    emotion_provenance = _emotion_provenance(
        config,
        manifest_rows,
        partition_subject_ids,
        source=args.emotion_source,
        language=args.emotion_language,
    )
    if subject_selection is not None:
        partition_subject_ids = _apply_subject_selection(
            partition_subject_ids, subject_selection, config=config, split_payload=split_payload
        )
    examples, joint_head_fit_provenance = _partition_examples(
        manifest_rows, config, partition_subject_ids, fold, checkpoint_dir=checkpoint_dir
    )
    model_name = args.model_name_or_path or saved.get("resolved_model_name_or_path") or config["model_name_or_path"]
    processor = load_processor(checkpoint_dir, config)
    for partition in examples:
        examples[partition] = prepare_backend_examples(
            examples[partition], config, processor
        )
    model = load_model_for_inference(str(model_name), checkpoint_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and bool(config.get("training", {}).get("bf16", False)) else None
    model.to(device=device, dtype=dtype)
    model.eval()
    expected_hidden_size = _decoder_hidden_size(model, config)
    partition_summaries = {}
    for partition in ("outer_train", "final_eval"):
        partition_summaries[partition] = _extract_partition(
            model=model,
            processor=processor,
            examples=examples[partition],
            config=config,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            partition=partition,
            max_examples=args.max_examples,
            expected_hidden_size=expected_hidden_size,
        )
    metadata = {
        "dataset": config["dataset"],
        "condition": condition,
        "input_modality": saved.get("input_modality"),
        "protocol_id": config.get("protocol_id", ""),
        "fold": fold,
        "model_backend": resolve_model_backend(config),
        "checkpoint_type": "best_model",
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_sha256": sha256_file(checkpoint_dir / "adapter_config.json"),
        "adapter_sha256": sha256_file(checkpoint_dir / "adapter_model.safetensors"),
        "base_model": str(model_name),
        "saved_run_config": str(run_config_path),
        "saved_run_config_sha256": sha256_file(run_config_path),
        "saved_split": str(split_path),
        "saved_split_sha256": sha256_file(split_path),
        "evaluation_provenance": evaluation_provenance,
        "head_fit_provenance": joint_head_fit_provenance or None,
        "evaluation_view": cache_config["evaluation_view"],
        "split_metadata": str(split_metadata_path),
        "split_metadata_sha256": sha256_file(split_metadata_path),
        "manifest": str(manifest_path),
        "manifest_sha256": canonical_manifest_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "emotion_provenance": emotion_provenance,
        "hidden_layer": "final",
        "pooling": POOLING_NAME,
        "vector_dimension": expected_hidden_size,
        "vector_dtype": "float32",
        "gold_label_protection": {
            "input_field": "prompt_text",
            "generation_used": False,
            "labels_passed_to_model": False,
        },
        "cache_config": cache_config,
        "cache_config_sha256": cache_config_sha256,
        "cache_collision_policy": (
            "skip_compatible_complete_refuse_partial_or_incompatible"
        ),
        "partitions": partition_summaries,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "project_git": _git_commit(),
        },
    }
    if _is_turkish_pooled(config):
        metadata["dataset_variant"] = config.get("dataset_variant", "")
        if _is_turkish_pooled_text(config):
            metadata.update({
                "aggregation_policy": cache_config.get("aggregation_policy"),
                "paired_text_examples_per_subject": 2,
            })
    save_json(metadata, output_dir / "extraction_metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
