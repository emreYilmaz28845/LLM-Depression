from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.data.d3tec import build_d3tec_training_schedule
from src.data.androids import apply_androids_training_weights
from src.daic_chunking import (
    JOINT_PACKED30_MODE,
    build_independent_epoch_schedule,
    build_joint_epoch_schedule,
    gradient_accumulation_for_reference_updates,
    resolve_chunking_controls,
)
from src.daic_mil import candidate_mean_token_logprob, streaming_subject_mil_backward
from src.data.runtime import (
    AudioTextDataset,
    build_examples,
    build_subject_label_map,
    filter_rows_by_subjects,
    load_manifest_rows,
    qwen2audio_audio_token_length,
    render_joint_packed30_bundle,
    save_partition_subjects,
)
from src.data.split_utils import (
    CV_PROTOCOL_TRAIN_VAL,
    FIXED_PROTOCOL_TRAIN_VAL,
    SPLIT_MODE_CV,
    SPLIT_MODE_FIXED,
    SPLIT_MODE_FULL_TRAIN,
    deterministic_inner_split,
    read_fold_payload,
    resolve_cv_protocol,
    resolve_dev_pool_partitions,
    resolve_fixed_protocol,
    resolve_requested_split_mode,
    resolve_split_mode,
    subject_ids_for_partitions,
)
from src.evaluate import _processor_inputs, evaluate_examples
from src.model.collator import Qwen2AudioSFTCollator
from src.model.runtime import (
    load_model_for_inference,
    load_model_for_training,
    load_processor,
    resolve_processor_sampling_rate,
    prepare_model_for_evaluation,
    restore_model_for_training,
    resolve_audio_adapter_config,
    save_adapter_and_processor,
)
from src.model.lora_common import resolved_lora_layer_selection
from src.sampling import (
    SAMPLING_MODE_SUBJECT_OVERSAMPLE,
    build_subject_oversampling,
)
from src.utils import (
    configure_logging,
    ensure_dir,
    evaluation_protocol_name,
    get_logger,
    internal_label_text_from_int,
    load_yaml_with_overrides,
    log_resolved_config,
    read_json,
    resolve_input_modality,
    resolve_metadata_paths,
    resolve_aggregation_level,
    resolve_model_name_or_path,
    resolve_prediction_mode,
    resolve_project_path,
    save_json,
    save_json_atomic,
    save_yaml,
    set_seed,
    sha256_file,
)

from src.experiment_tracking.canonical import (
    append_jsonl_atomic,
    canonical_sha256,
    format_utc_timestamp,
    sha256_file as tracking_sha256_file,
    utc_now,
    write_json_atomic,
)
from src.experiment_tracking.constants import (
    SCHEMA_VERSION_ARTIFACTS,
    SCHEMA_VERSION_METADATA,
    WANDB_PROJECT,
)
from src.experiment_tracking.identity import artifact_id, validate_attempt_id
from src.experiment_tracking.lifecycle import (
    StatusRecord,
    append_job_event,
    new_job_event,
    read_job_events,
    write_status,
)


LOGGER = get_logger(__name__)

ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY = "all_chunks_subject_normalized"
PACKED30_LOSS_WEIGHT_SCHEMA_VERSION = "packed30_subject_normalized_loss_weight.v1"


def apply_subject_normalized_chunk_weights(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Static subject-normalized chunk weighting behind the locked policy name.

    Every subject contributes equal total loss weight: each chunk receives raw
    weight ``1 / number_of_training_chunks_for_subject``, then all weights are
    rescaled so the arithmetic mean example weight is exactly one. The audit
    proves raw totals sum to one per subject and emitted weights average one.
    """
    if not examples:
        raise ValueError("Cannot weight an empty training partition.")
    chunk_counts: dict[str, int] = defaultdict(int)
    labels_by_subject: dict[str, set[int]] = defaultdict(set)
    sample_ids_by_subject: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        subject_id = str(example["subject_id"])
        chunk_counts[subject_id] += 1
        labels_by_subject[subject_id].add(int(example["label"]))
        sample_ids_by_subject[subject_id].add(str(example.get("sample_id", "")))
    for subject_id, labels in sorted(labels_by_subject.items()):
        if len(labels) != 1:
            raise ValueError(
                f"{ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} requires one label per subject; "
                f"subject_id={subject_id} has {sorted(labels)}."
            )
        sample_ids = sample_ids_by_subject[subject_id]
        if not all(sample_ids) or len(sample_ids) != chunk_counts[subject_id]:
            raise ValueError(
                f"{ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} requires unique sample IDs per "
                f"subject; subject_id={subject_id} has {chunk_counts[subject_id]} chunks "
                f"but {len(sample_ids)} distinct sample IDs."
            )
    raw_weights = [1.0 / chunk_counts[str(example["subject_id"])] for example in examples]
    scale = len(raw_weights) / sum(raw_weights)
    weighted: list[dict[str, Any]] = []
    for example, raw_weight in zip(examples, raw_weights):
        weighted.append(
            {
                **example,
                "raw_loss_weight": float(raw_weight),
                "loss_weight": float(raw_weight) * scale,
            }
        )
    raw_subject_totals: dict[str, float] = defaultdict(float)
    for example in weighted:
        raw_subject_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    for subject_id, total in raw_subject_totals.items():
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} raw weight total is not one "
                f"for subject_id={subject_id}: {total}"
            )
    mean_weight = sum(float(row["loss_weight"]) for row in weighted) / len(weighted)
    if not math.isclose(mean_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} emitted weights do not average one: {mean_weight}"
        )
    audit = {
        "schema_version": PACKED30_LOSS_WEIGHT_SCHEMA_VERSION,
        "policy": ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY,
        "formula": "raw = 1 / training_chunks_for_subject; rescaled so mean example weight is 1",
        "sample_count": len(weighted),
        "subject_count": len(chunk_counts),
        "chunks_per_subject": dict(sorted(chunk_counts.items())),
        "raw_subject_weight_totals": dict(sorted(raw_subject_totals.items())),
        "mean_loss_weight": float(mean_weight),
        "rescale_factor": float(scale),
        "equal_total_subject_weight": True,
    }
    return weighted, audit


def _metadata_artifacts_are_usable(metadata: dict[str, Any]) -> tuple[bool, str]:
    manifest_path = Path(metadata["manifest_path"])
    if not manifest_path.exists():
        return False, f"manifest_missing:{manifest_path}"
    for key, value in metadata.items():
        if key.endswith("_path") and key != "manifest_path" and isinstance(value, str) and value:
            if not Path(value).exists():
                return False, f"artifact_missing:{key}:{value}"
    preview_rows = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                preview_rows.append(json.loads(line))
                if len(preview_rows) >= 3:
                    break
    except Exception as exc:
        return False, f"manifest_read_error:{exc}"
    if not preview_rows:
        return False, "manifest_empty"
    for row in preview_rows:
        audio_paths = row.get("audio_paths") or ([row["audio_path"]] if row.get("audio_path") else [])
        for audio_path in audio_paths:
            if audio_path and not Path(audio_path).exists():
                return False, f"stale_audio_path:{audio_path}"
    return True, "ok"


def _wait_for_usable_metadata(metadata_path: Path, timeout_seconds: int = 600) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_reason = "metadata_not_ready"
    while time.time() < deadline:
        if metadata_path.exists():
            metadata = resolve_metadata_paths(read_json(metadata_path))
            usable, reason = _metadata_artifacts_are_usable(metadata)
            if usable:
                return metadata
            last_reason = reason
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for usable metadata at {metadata_path}. Last reason: {last_reason}")


def _required_split_metadata_keys(config: dict[str, Any]) -> list[str]:
    requested_mode = resolve_requested_split_mode(config)
    if requested_mode == SPLIT_MODE_CV:
        return ["folds_path"]
    if requested_mode in {SPLIT_MODE_FIXED, SPLIT_MODE_FULL_TRAIN}:
        return ["subject_partition_path"]
    split_cfg = config.get("split", {})
    if any(key in split_cfg for key in ("train_partition", "train_partitions", "selection_partition", "dev_pool_partitions")):
        return ["subject_partition_path"]
    return ["folds_path"]


def _load_metadata_or_build(config_path: str | Path, config: dict[str, Any], config_overrides: list[str] | None = None) -> dict[str, Any]:
    metadata_path = resolve_project_path(Path(config["output_dirs"]["split_dir"]) / f"{config['dataset']}_manifest_metadata.json")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if metadata_path.exists():
        metadata = resolve_metadata_paths(read_json(metadata_path))
        usable, reason = _metadata_artifacts_are_usable(metadata)
        missing_split_keys = [key for key in _required_split_metadata_keys(config) if not metadata.get(key)]
        if usable and missing_split_keys:
            usable = False
            reason = f"split_metadata_missing:{','.join(missing_split_keys)}"
        if usable and metadata.get("build_signature") != manifest_build_signature(config):
            usable = False
            reason = "build_signature_mismatch"
        if usable:
            return metadata
        LOGGER.warning("Refreshing stale metadata for %s: %s", config["dataset"], reason)
    if local_rank == 0:
        build_for_config(config_path, config_overrides)
    return _wait_for_usable_metadata(metadata_path)


def _load_subject_partition_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if not metadata.get("subject_partition_path"):
        raise ValueError("Split metadata does not include subject_partition_path.")
    return read_json(metadata["subject_partition_path"])


def _resolve_fixed_outer_partitions(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, list[str]]:
    partition_rows = _load_subject_partition_rows(metadata)
    train_partitions_cfg = config["split"].get("train_partitions")
    if train_partitions_cfg:
        train_partitions = [str(item) for item in train_partitions_cfg]
    else:
        train_partitions = [str(config["split"]["train_partition"])]
    selection_partition = str(config["split"].get("selection_partition", "")).strip()
    final_eval_partition = str(config["split"]["final_eval_partition"])
    payload = {
        "outer_train_subject_ids": subject_ids_for_partitions(partition_rows, train_partitions),
        "final_eval_subject_ids": subject_ids_for_partitions(partition_rows, [final_eval_partition]),
    }
    if selection_partition:
        payload["selection_subject_ids"] = subject_ids_for_partitions(partition_rows, [selection_partition])
    return payload


def _resolve_cv_outer_partitions(metadata: dict[str, Any], fold: int) -> dict[str, list[str]]:
    fold_payload = read_fold_payload(metadata, fold)
    return {
        "outer_train_subject_ids": sorted(fold_payload["outer_train_subject_ids"]),
        "final_eval_subject_ids": sorted(fold_payload["final_eval_subject_ids"]),
    }


def _resolve_full_train_outer_partitions(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, list[str]]:
    partition_rows = _load_subject_partition_rows(metadata)
    final_eval_partition = str(config["split"]["final_eval_partition"])
    train_subject_ids = subject_ids_for_partitions(partition_rows, resolve_dev_pool_partitions(config))
    final_eval_subject_ids = subject_ids_for_partitions(partition_rows, [final_eval_partition])
    overlap = sorted(set(train_subject_ids).intersection(final_eval_subject_ids))
    if overlap:
        raise ValueError(
            "split.mode=full_train requires dev_pool_partitions to be disjoint from final_eval_partition. "
            f"Overlap detected with final_eval_partition={final_eval_partition!r}: {overlap[:10]}"
        )
    return {
        "outer_train_subject_ids": train_subject_ids,
        "final_eval_subject_ids": final_eval_subject_ids,
    }


def _resolve_outer_partitions(config: dict[str, Any], metadata: dict[str, Any], fold: int) -> dict[str, list[str]]:
    split_mode = resolve_split_mode(config, metadata)
    if split_mode == SPLIT_MODE_FIXED:
        return _resolve_fixed_outer_partitions(config, metadata)
    if split_mode == SPLIT_MODE_CV:
        return _resolve_cv_outer_partitions(metadata, fold)
    if split_mode == SPLIT_MODE_FULL_TRAIN:
        return _resolve_full_train_outer_partitions(config, metadata)
    raise ValueError(f"Unsupported split mode: {split_mode}")


def _resolve_training_subject_splits(
    config: dict[str, Any],
    metadata: dict[str, Any],
    subject_labels: dict[str, int],
    fold: int,
) -> dict[str, Any]:
    split_mode = resolve_split_mode(config, metadata)
    outer_partitions = _resolve_outer_partitions(config, metadata, fold)
    cv_protocol = resolve_cv_protocol(config) if split_mode == SPLIT_MODE_CV else None
    if split_mode == SPLIT_MODE_FULL_TRAIN:
        return {
            "split_mode": split_mode,
            "cv_protocol": None,
            "selection_mode": "none",
            "selection_enabled": False,
            "uses_inner_split": False,
            "outer_partitions": outer_partitions,
            "train_subject_ids": outer_partitions["outer_train_subject_ids"],
            "selection_subject_ids": [],
            "final_eval_subject_ids": outer_partitions["final_eval_subject_ids"],
            "train_split_name": "train_full",
            "selection_split_name": "none",
            "final_eval_split_name": str(config["split"]["final_eval_partition"]),
            "selection_log_dir_name": "selection_disabled",
        }

    if split_mode == SPLIT_MODE_CV and cv_protocol == CV_PROTOCOL_TRAIN_VAL:
        return {
            "split_mode": split_mode,
            "cv_protocol": cv_protocol,
            "selection_mode": "outer_fold_validation",
            "selection_enabled": True,
            "uses_inner_split": False,
            "outer_partitions": outer_partitions,
            "train_subject_ids": outer_partitions["outer_train_subject_ids"],
            "selection_subject_ids": outer_partitions["final_eval_subject_ids"],
            "final_eval_subject_ids": [],
            "train_split_name": "train_outer",
            "selection_split_name": "fold_validation",
            "final_eval_split_name": "none",
            "selection_log_dir_name": "fold_validation",
        }

    selection_partition = str(config["split"].get("selection_partition", "")).strip()
    if split_mode == SPLIT_MODE_FIXED and resolve_fixed_protocol(config) == FIXED_PROTOCOL_TRAIN_VAL:
        selection_subject_ids = outer_partitions.get("selection_subject_ids") or []
        if not selection_subject_ids:
            raise ValueError(
                "split.fixed_protocol=train_val requires split.selection_partition with resolvable subject ids."
            )
        # Fixed-partition analogue of the CV train_val protocol: train on the train
        # partition, select on the selection partition, and report the best
        # selection-partition results. Setting cv_protocol=train_val on the plan
        # reuses that protocol's reporting plumbing (best_validation copy, forced
        # run_final_eval_in_train=False, summarize_runs score source) unchanged.
        # split.final_eval_partition stays manifest bookkeeping and is never
        # evaluated here.
        return {
            "split_mode": split_mode,
            "cv_protocol": CV_PROTOCOL_TRAIN_VAL,
            "selection_mode": "fixed_partition_validation",
            "selection_enabled": True,
            "uses_inner_split": False,
            "outer_partitions": outer_partitions,
            "train_subject_ids": outer_partitions["outer_train_subject_ids"],
            "selection_subject_ids": selection_subject_ids,
            "final_eval_subject_ids": [],
            "train_split_name": str(config["split"].get("train_partition", "train")),
            "selection_split_name": selection_partition,
            "final_eval_split_name": "none",
            "selection_log_dir_name": selection_partition,
        }

    if split_mode == SPLIT_MODE_FIXED and selection_partition:
        selection_subject_ids = outer_partitions.get("selection_subject_ids") or []
        if not selection_subject_ids:
            raise ValueError(
                "split.selection_partition was configured, but no subject ids were resolved for that partition."
            )
        return {
            "split_mode": split_mode,
            "cv_protocol": None,
            "selection_mode": "fixed_partition",
            "selection_enabled": True,
            "uses_inner_split": False,
            "outer_partitions": outer_partitions,
            "train_subject_ids": outer_partitions["outer_train_subject_ids"],
            "selection_subject_ids": selection_subject_ids,
            "final_eval_subject_ids": outer_partitions["final_eval_subject_ids"],
            "train_split_name": str(config["split"].get("train_partition", "train")),
            "selection_split_name": selection_partition,
            "final_eval_split_name": str(config["split"]["final_eval_partition"]),
            "selection_log_dir_name": selection_partition,
        }

    inner_split = deterministic_inner_split(
        subject_labels,
        outer_partitions["outer_train_subject_ids"],
        seed=int(config["split"]["seed"]) + int(fold),
        val_ratio=float(config["split"]["inner_val_ratio"]),
    )
    return {
        "split_mode": split_mode,
        "cv_protocol": cv_protocol if split_mode == SPLIT_MODE_CV else None,
        "selection_mode": "inner_split",
        "selection_enabled": True,
        "uses_inner_split": True,
        "outer_partitions": outer_partitions,
        "train_subject_ids": inner_split["train_inner_subject_ids"],
        "selection_subject_ids": inner_split["val_inner_subject_ids"],
        "final_eval_subject_ids": outer_partitions["final_eval_subject_ids"],
        "train_split_name": "train_inner",
        "selection_split_name": "val_inner",
        "final_eval_split_name": "fold_holdout" if split_mode == SPLIT_MODE_CV else "final_eval",
        "selection_log_dir_name": "inner_val",
    }


JOINT_SELECTION_PRIMARY_ONLY = "primary_only"
JOINT_SELECTION_MEAN_POSITIVE_F1 = "mean_positive_f1"
_SUPPORTED_JOINT_SELECTION_MODES = (JOINT_SELECTION_PRIMARY_ONLY, JOINT_SELECTION_MEAN_POSITIVE_F1)


def _resolve_joint_selection_mode(config: dict[str, Any]) -> str:
    raw_mode = str((config.get("joint_train") or {}).get("selection_mode", "")).strip().lower()
    if not raw_mode:
        return JOINT_SELECTION_PRIMARY_ONLY
    if raw_mode not in _SUPPORTED_JOINT_SELECTION_MODES:
        raise ValueError(
            f"Unsupported joint_train.selection_mode={raw_mode!r}. "
            f"Expected one of {_SUPPORTED_JOINT_SELECTION_MODES}."
        )
    return raw_mode


def _subject_overlap_proof(
    *,
    train_subject_ids: list[str],
    selection_subject_ids: list[str],
    final_eval_subject_ids: list[str],
) -> dict[str, Any]:
    overlaps = {
        "train_selection": sorted(set(train_subject_ids) & set(selection_subject_ids)),
        "train_final_eval": sorted(set(train_subject_ids) & set(final_eval_subject_ids)),
        "selection_final_eval": sorted(set(selection_subject_ids) & set(final_eval_subject_ids)),
    }
    return {
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "overlaps": overlaps,
        "passed": all(not values for values in overlaps.values()),
    }


def _assert_no_subject_overlap(dataset_name: str, proof: dict[str, Any]) -> None:
    if proof["passed"]:
        return
    non_empty = {name: values for name, values in proof["overlaps"].items() if values}
    raise ValueError(f"{dataset_name} train/selection/final-eval subject overlap: {non_empty}")


def _example_class_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    labels = [int(example["label"]) for example in examples]
    return {
        "depressed_samples": int(sum(labels)),
        "non_depressed_samples": int(len(labels) - sum(labels)),
        "total_samples": len(labels),
    }


def _build_joint_train_examples(
    config: dict[str, Any],
    logs_dir: Path,
    primary_model_name_or_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Auxiliary train (and optional selection) examples for opt-in joint training.

    ``config.joint_train.extra_datasets`` entries each name a standalone dataset
    config; that config's fold-``fold`` outer-train subjects are built into
    examples WITH THAT CONFIG (its prompts, sample_mode, transcript budget) and
    appended to the primary train set. The entry's fold holdout is never trained
    on, so it remains clean for a standalone eval of the joint checkpoint
    (src.evaluate --config <entry config> --checkpoint_dir <best_model> --fold k).

    ``joint_train.selection_mode`` controls checkpoint selection:
      - ``primary_only`` (default): all outer-train subjects go to training and
        selection stays primary-only — the original joint behavior.
      - ``mean_positive_f1``: a deterministic inner-val is carved from each
        entry's outer-train subjects (never from the holdout); those subjects are
        excluded from training and evaluated per epoch, and the checkpoint is
        selected on ``joint_val_positive_f1`` = unweighted mean of the primary
        and per-entry selection positive-F1 (see ``_joint_selection_values``).

    Returns ``(extra_train_examples, selection_groups, final_eval_groups, composition)``.
    Selection groups carry inner-val examples for per-epoch selection eval; final
    eval groups carry the untouched fold holdout for optional best-checkpoint
    evaluation.
    """
    joint_cfg = config.get("joint_train") or {}
    entries = joint_cfg.get("extra_datasets") or []
    selection_mode = _resolve_joint_selection_mode(config)
    extra_examples: list[dict[str, Any]] = []
    selection_groups: list[dict[str, Any]] = []
    final_eval_groups: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for entry in entries:
        extra_config_path = str(entry["config"])
        extra_config = load_yaml_with_overrides(extra_config_path, [])
        extra_model = str(resolve_model_name_or_path(None, extra_config))
        if extra_model != str(primary_model_name_or_path):
            raise ValueError(
                "joint_train.extra_datasets entries must resolve to the primary model. "
                f"Primary={primary_model_name_or_path!r} extra={extra_model!r} in {extra_config_path}"
            )
        extra_fold = int(entry.get("fold", 0))
        extra_metadata = _load_metadata_or_build(extra_config_path, extra_config)
        extra_rows = load_manifest_rows(extra_metadata["manifest_path"])
        outer_partitions = _resolve_outer_partitions(extra_config, extra_metadata, extra_fold)
        train_ids = list(outer_partitions["outer_train_subject_ids"])
        heldout_ids = list(outer_partitions["final_eval_subject_ids"])
        selection_ids: list[str] = []
        if selection_mode == JOINT_SELECTION_MEAN_POSITIVE_F1:
            extra_subject_labels = build_subject_label_map(extra_rows)
            inner_split = deterministic_inner_split(
                extra_subject_labels,
                train_ids,
                seed=int(extra_config["split"]["seed"]) + extra_fold,
                val_ratio=float(entry.get("inner_val_ratio", extra_config["split"].get("inner_val_ratio", 0.2))),
            )
            train_ids = inner_split["train_inner_subject_ids"]
            selection_ids = inner_split["val_inner_subject_ids"]
        dataset_name = str(extra_config["dataset"])
        overlap_proof = _subject_overlap_proof(
            train_subject_ids=train_ids,
            selection_subject_ids=selection_ids,
            final_eval_subject_ids=heldout_ids,
        )
        _assert_no_subject_overlap(dataset_name, overlap_proof)
        examples = build_examples(
            filter_rows_by_subjects(extra_rows, train_ids),
            extra_config,
            partition_name=f"joint_{dataset_name}_train",
            truncation_log_path=logs_dir / f"joint_{dataset_name}_train_truncation.jsonl",
        )
        selection_examples: list[dict[str, Any]] = []
        if selection_ids:
            selection_examples = build_examples(
                filter_rows_by_subjects(extra_rows, selection_ids),
                extra_config,
                partition_name=f"joint_{dataset_name}_inner_val",
                truncation_log_path=logs_dir / f"joint_{dataset_name}_inner_val_truncation.jsonl",
            )
        heldout_examples = build_examples(
            filter_rows_by_subjects(extra_rows, heldout_ids),
            extra_config,
            partition_name=f"joint_{dataset_name}_fold_holdout",
            truncation_log_path=logs_dir / f"joint_{dataset_name}_fold_holdout_truncation.jsonl",
        )
        extra_examples.extend(examples)
        composition.append(
            {
                "dataset": dataset_name,
                "config": extra_config_path,
                "fold": extra_fold,
                "selection_mode": selection_mode,
                "train_subject_count": len(train_ids),
                "train_sample_count": len(examples),
                "train_class_counts": _example_class_counts(examples),
                "selection_subject_count": len(selection_ids),
                "selection_sample_count": len(selection_examples),
                "selection_class_counts": _example_class_counts(selection_examples),
                "train_subject_ids": sorted(train_ids),
                "selection_subject_ids": sorted(selection_ids),
                "heldout_subject_count": len(heldout_ids),
                "heldout_sample_count": len(heldout_examples),
                "heldout_class_counts": _example_class_counts(heldout_examples),
                "heldout_subject_ids": sorted(heldout_ids),
                "subject_overlap_proof": overlap_proof,
            }
        )
        if selection_examples:
            selection_groups.append(
                {
                    "dataset": dataset_name,
                    "config": extra_config,
                    "config_path": extra_config_path,
                    "fold": extra_fold,
                    "split_name": f"{dataset_name}_inner_val",
                    "subject_ids": sorted(selection_ids),
                    "examples": selection_examples,
                }
            )
        final_eval_groups.append(
            {
                "dataset": dataset_name,
                "config": extra_config,
                "config_path": extra_config_path,
                "fold": extra_fold,
                "split_name": f"{dataset_name}_fold_holdout",
                "subject_ids": sorted(heldout_ids),
                "examples": heldout_examples,
            }
        )
    return extra_examples, selection_groups, final_eval_groups, composition


def _joint_selection_values(component_headlines: list[tuple[str, dict[str, Any]]]) -> dict[str, float]:
    """Per-dataset selection diagnostics plus the combined joint metric.

    ``joint_val_positive_f1`` is the UNWEIGHTED MEAN of the components'
    positive-F1 — never a pooled F1 over concatenated predictions, because the
    component datasets differ in sample count and class balance, and pooling
    would let the larger one dominate checkpoint selection.
    """
    values: dict[str, float] = {}
    positive_f1s: list[float] = []
    for name, headline in component_headlines:
        for key in ("positive_f1", "macro_f1", "accuracy", "precision", "recall"):
            values[f"{name}_{key}"] = float(headline[key])
        positive_f1s.append(float(headline["positive_f1"]))
    values["joint_val_positive_f1"] = sum(positive_f1s) / max(1, len(positive_f1s))
    return values


def _print_partition_counts(payload: dict[str, Any]) -> None:
    for name, counts in payload["class_counts"].items():
        LOGGER.info(
            "%s | depressed_subjects=%s non_depressed_subjects=%s total_subjects=%s",
            name,
            counts["depressed"],
            counts["non_depressed"],
            counts["depressed"] + counts["non_depressed"],
        )


def _emit_label_mask_debug(dataset: AudioTextDataset, collator: Qwen2AudioSFTCollator, processor, logs_dir: Path) -> None:
    preview_examples = [dataset[index] for index in range(min(2, len(dataset)))]
    if not preview_examples:
        return
    collator(preview_examples[:1])
    if not collator.last_debug_example:
        return
    debug_payload = collator.last_debug_example.copy()
    debug_payload["unmasked_tokens"] = processor.tokenizer.convert_ids_to_tokens(debug_payload["unmasked_token_ids"])
    save_json(debug_payload, logs_dir / "label_mask_debug.json")
    LOGGER.info("Label-mask debug saved to %s", logs_dir / "label_mask_debug.json")


def _sample_partition_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(int(example["label"]) for example in examples)
    return {
        "depressed_samples": int(counter[1]),
        "non_depressed_samples": int(counter[0]),
        "total_samples": len(examples),
    }


def _limit_subject_ids_for_smoke(
    subject_ids: list[str],
    subject_labels: dict[str, int],
    *,
    limit: int,
    seed: int,
    preserve_class_ratio: bool = False,
) -> list[str]:
    if limit <= 0 or len(subject_ids) <= limit:
        return sorted(subject_ids)
    rng = random.Random(seed)
    by_label = {
        label: [subject_id for subject_id in sorted(subject_ids) if int(subject_labels[subject_id]) == label]
        for label in (0, 1)
    }
    for values in by_label.values():
        rng.shuffle(values)
    if preserve_class_ratio:
        negative_target = max(
            1,
            min(
                limit - 1,
                round(limit * len(by_label[0]) / (len(by_label[0]) + len(by_label[1]))),
            ),
        )
        positive_target = limit - negative_target
        selected = by_label[0][:negative_target] + by_label[1][:positive_target]
        if len(selected) != limit:
            raise ValueError("Smoke subject cap could not retain both classes proportionally.")
        return sorted(selected)
    selected: list[str] = []
    while len(selected) < limit and any(by_label.values()):
        for label in (1, 0):
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop())
    return sorted(selected)


def _apply_smoke_subject_limit(
    partition_plan: dict[str, Any],
    subject_labels: dict[str, int],
    config: dict[str, Any],
    fold: int,
) -> None:
    limit = int(config.get("split", {}).get("smoke_subject_limit", 0) or 0)
    if limit <= 0:
        return
    preserve_class_ratio = (
        str(config.get("training", {}).get("class_balance", "")).strip().lower()
        == SAMPLING_MODE_SUBJECT_OVERSAMPLE
    )
    for offset, key in enumerate(
        ("train_subject_ids", "selection_subject_ids", "final_eval_subject_ids")
    ):
        partition_plan[key] = _limit_subject_ids_for_smoke(
            partition_plan[key],
            subject_labels,
            limit=limit,
            seed=int(config["seed"]) + int(fold) * 10 + offset,
            preserve_class_ratio=preserve_class_ratio,
        )
    LOGGER.warning(
        "Smoke subject cap enabled: at most %s subjects per train/selection/final-eval partition.",
        limit,
    )


def _build_weighted_train_sampler(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> WeightedRandomSampler | None:
    mode = str(config.get("training", {}).get("class_balance", "none")).strip().lower()
    if mode in {"", "none", "off", "false", "subject_inverse_frequency"}:
        return None
    if mode == SAMPLING_MODE_SUBJECT_OVERSAMPLE:
        return None
    if mode != "weighted_sampler":
        raise ValueError(
            "Unsupported training.class_balance="
            f"{mode!r}. Expected 'none', 'weighted_sampler', or "
            f"'{SAMPLING_MODE_SUBJECT_OVERSAMPLE}'."
        )
    class_counts = Counter(int(example["label"]) for example in examples)
    if set(class_counts) != {0, 1}:
        raise ValueError("Weighted training sampler requires both classes.")
    weights = [1.0 / class_counts[int(example["label"])] for example in examples]
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    LOGGER.info(
        "Train-only weighted sampler enabled | depressed_samples=%s non_depressed_samples=%s",
        class_counts[1],
        class_counts[0],
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _apply_subject_oversampling(
    train_examples: list[dict[str, Any]],
    selection_examples: list[dict[str, Any]],
    final_eval_examples: list[dict[str, Any]],
    config: dict[str, Any],
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, torch.Generator | None]:
    training_cfg = config.get("training", {})
    mode = str(training_cfg.get("class_balance", "none")).strip().lower()
    if mode != SAMPLING_MODE_SUBJECT_OVERSAMPLE:
        return train_examples, None, None
    if config.get("joint_train", {}).get("extra_datasets"):
        raise ValueError("Subject oversampling is not supported for joint_train experiments.")

    ratio = training_cfg.get("oversampling_ratio")
    sampling_seed = int(training_cfg.get("oversampling_seed", config["seed"]))
    result = build_subject_oversampling(
        train_examples,
        ratio=ratio,
        seed=sampling_seed,
        expected_minority_label=0,
        validation_rows=selection_examples,
        evaluation_rows=final_eval_examples,
    )
    identity = {
        "schema_version": "qwen_sampling_identity.v1",
        "strategy": result.audit["strategy"],
        "requested_ratio": result.audit["requested_ratio"],
        "sampling_seed": result.audit["sampling_seed"],
        "source_row_assignments_sha256": result.audit[
            "source_row_assignments_sha256"
        ],
        "source_subject_assignments_sha256": result.audit[
            "source_subject_assignments_sha256"
        ],
    }
    identity_path = run_root / "sampling_identity.json"
    if int(os.environ.get("RANK", "0")) == 0:
        if identity_path.exists():
            existing = read_json(identity_path)
            if existing != identity:
                raise ValueError(
                    f"Incompatible sampling configuration already exists at {identity_path}."
                )
        else:
            save_json_atomic(identity, identity_path)
        save_json_atomic(result.audit, run_root / "sampling_audit.json")

    expanded_examples = [train_examples[index] for index in result.indices]
    generator = torch.Generator()
    generator.manual_seed(sampling_seed)
    LOGGER.info(
        "Train-only subject oversampling enabled | ratio=%s seed=%s "
        "original_samples=%s expanded_samples=%s subject_occurrences=%s",
        result.audit["requested_ratio"],
        sampling_seed,
        len(train_examples),
        len(expanded_examples),
        result.audit["final_subject_occurrence_counts_by_class"],
    )
    return expanded_examples, result.audit, generator


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


AUDIO_TOKEN = "<|AUDIO|>"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[int(pos)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def _log_peak_gpu_memory(logger, tag: str) -> dict[str, float] | None:
    """Log the running peak CUDA memory (allocated + reserved) since the last reset."""
    if not torch.cuda.is_available():
        return None
    gib = 1024 ** 3
    allocated = float(torch.cuda.max_memory_allocated()) / gib
    reserved = float(torch.cuda.max_memory_reserved()) / gib
    logger.info(
        "Peak GPU memory [%s] | max_allocated=%.2f GiB | max_reserved=%.2f GiB",
        tag,
        allocated,
        reserved,
    )
    return {"max_allocated_gib": allocated, "max_reserved_gib": reserved}


def _audit_audio_budget(
    dataset: AudioTextDataset,
    collator: Qwen2AudioSFTCollator,
    processor,
    sampling_rate: int | None,
    partition_name: str,
    logs_dir: Path,
    max_examples: int | None = 256,
) -> dict[str, Any] | None:
    """One-time per-example audio-budget trace for VRAM planning.

    Reports mean / p95 / max of audio seconds, audio tokens, and total input
    sequence tokens per example. Audio tokens are read from the expanded
    ``<|AUDIO|>`` ids in ``input_ids`` (version-matched) when available, falling
    back to the Qwen2-Audio conv-length formula on ``feature_attention_mask``.

    Runs on a dedicated dataset instance so it never perturbs the training RNG.
    """
    if sampling_rate is None:
        return None
    try:
        audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(AUDIO_TOKEN))
    except Exception:  # pragma: no cover - tokenizer without the audio token
        audio_token_id = None
    total = len(dataset)
    audit_count = total if max_examples is None else min(total, max_examples)
    seconds: list[float] = []
    audio_tokens: list[float] = []
    seq_lengths: list[float] = []
    for index in range(audit_count):
        try:
            item = dataset[index]
        except Exception as exc:
            LOGGER.warning("Audio budget audit: failed to load example %s: %s", index, exc)
            continue
        audio_arrays = item.get("audio_arrays") or []
        if not audio_arrays:
            continue
        try:
            batch = collator([item])
        except Exception as exc:
            LOGGER.warning("Audio budget audit: collator failed on example %s: %s", index, exc)
            continue
        input_ids = batch["input_ids"][0]
        example_tokens = 0
        if audio_token_id is not None:
            audio_id_count = int((input_ids == audio_token_id).sum().item())
            if audio_id_count > len(audio_arrays):
                example_tokens = audio_id_count
        if example_tokens == 0 and "feature_attention_mask" in batch:
            mel_lengths = batch["feature_attention_mask"].sum(dim=-1).tolist()
            example_tokens = sum(qwen2audio_audio_token_length(int(length)) for length in mel_lengths)
        seconds.append(sum(len(arr) / float(sampling_rate) for arr in audio_arrays))
        audio_tokens.append(float(example_tokens))
        seq_lengths.append(float(int(input_ids.shape[0])))

    if not seconds:
        return None

    def _stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": sum(values) / len(values),
            "p95": _percentile(values, 0.95),
            "max": max(values),
        }

    payload = {
        "partition": partition_name,
        "examples_audited": len(seconds),
        "examples_total": total,
        "audited_all": len(seconds) >= total,
        "sampling_rate": int(sampling_rate),
        "audio_seconds_per_example": _stats(seconds),
        "audio_tokens_per_example": _stats(audio_tokens),
        "total_sequence_tokens_per_example": _stats(seq_lengths),
    }
    save_json(payload, logs_dir / f"audio_budget_audit_{partition_name}.json")
    sec_stats = payload["audio_seconds_per_example"]
    tok_stats = payload["audio_tokens_per_example"]
    seq_stats = payload["total_sequence_tokens_per_example"]
    LOGGER.info(
        "Audio budget audit [%s] examples=%s/%s | audio_sec/ex mean=%.1f p95=%.1f max=%.1f | "
        "audio_tok/ex mean=%.0f p95=%.0f max=%.0f | total_seq_tok/ex mean=%.0f p95=%.0f max=%.0f",
        partition_name,
        payload["examples_audited"],
        total,
        sec_stats["mean"],
        sec_stats["p95"],
        sec_stats["max"],
        tok_stats["mean"],
        tok_stats["p95"],
        tok_stats["max"],
        seq_stats["mean"],
        seq_stats["p95"],
        seq_stats["max"],
    )
    return payload


def _compute_dataset_loss(
    model,
    data_loader: DataLoader,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def _save_best_checkpoint(save_strategy: str) -> bool:
    return save_strategy in {"full", "best_only"}


def _save_last_checkpoint(save_strategy: str) -> bool:
    return save_strategy == "full"


def _resolve_early_stopping(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config["training"]
    early_cfg = training_cfg.get("early_stopping") or {}
    enabled = bool(early_cfg.get("enabled", False))
    metric_name = str(early_cfg.get("metric", "inner_val_positive_f1"))
    mode = str(early_cfg.get("mode", "auto")).lower()
    if mode == "auto":
        mode = "min" if metric_name.endswith("_loss") else "max"
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported early_stopping.mode={mode!r}. Expected 'min', 'max', or 'auto'.")
    return {
        "enabled": enabled,
        "metric": metric_name,
        "mode": mode,
        "patience": int(early_cfg.get("patience", 0)),
        "min_delta": float(early_cfg.get("min_delta", 0.0)),
    }


_SELECTION_METRIC_DEFAULT = "inner_val_positive_f1"


def _resolve_selection_metric(config: dict[str, Any]) -> dict[str, Any]:
    """Which inner-val metric chooses the saved ``best_model`` checkpoint.

    Independent of ``early_stopping`` (that only decides *when* to stop). Defaults to
    ``inner_val_positive_f1`` (max) so existing configs behave exactly as before; a
    config may set ``training.selection_metric`` (e.g. ``inner_val_loss``) to pick the
    checkpoint on a smoother, lower-variance signal. ``mode: auto`` resolves to ``min``
    for ``*_loss`` metrics, else ``max``.
    """
    training_cfg = config["training"]
    metric_name = str(training_cfg.get("selection_metric", _SELECTION_METRIC_DEFAULT))
    mode = str(training_cfg.get("selection_metric_mode", "auto")).lower()
    if mode == "auto":
        mode = "min" if metric_name.endswith("_loss") else "max"
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported training.selection_metric_mode={mode!r}. Expected 'min', 'max', or 'auto'.")
    return {"metric": metric_name, "mode": mode}


def _selection_metric_values(metric_value: float, headline_metrics: dict[str, Any], selection_loss: float) -> dict[str, float]:
    values = {
        "selection_positive_f1": metric_value,
        "selection_macro_f1": float(headline_metrics["macro_f1"]),
        "selection_accuracy": float(headline_metrics["accuracy"]),
        "selection_precision": float(headline_metrics["precision"]),
        "selection_recall": float(headline_metrics["recall"]),
        "selection_loss": selection_loss,
        "selection_auroc": float(headline_metrics.get("auroc", 0.0)),
    }
    values.update(
        {
            "inner_val_positive_f1": values["selection_positive_f1"],
            "inner_val_macro_f1": values["selection_macro_f1"],
            "inner_val_accuracy": values["selection_accuracy"],
            "inner_val_precision": values["selection_precision"],
            "inner_val_recall": values["selection_recall"],
            "inner_val_loss": values["selection_loss"],
            "inner_val_auroc": values["selection_auroc"],
        }
    )
    return values


def _metric_improved(metric_value: float, best_value: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return metric_value < (best_value - min_delta)
    return metric_value > (best_value + min_delta)


def _write_trial_progress(
    progress_path: str | None,
    *,
    epoch: int,
    metric_name: str,
    metric_value: float,
    best_metric: float,
    best_epoch: int,
    run_root: Path,
    config_overrides: list[str],
) -> None:
    if not progress_path:
        return
    save_json_atomic(
        {
            "epoch": int(epoch),
            "step": int(epoch),
            "metric_name": metric_name,
            "metric": float(metric_value),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "run_root": str(run_root),
            "config_overrides": list(config_overrides),
        },
        progress_path,
    )


def _write_trial_result(result_path: str | None, payload: dict[str, Any]) -> None:
    if not result_path:
        return
    save_json_atomic(payload, result_path)


def _load_experiment_context(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "experiment_context", None):
        return None
    context_path = Path(args.experiment_context)
    if not context_path.is_file():
        raise ValueError(f"experiment context file not found: {context_path}")
    context = read_json(context_path)
    if not isinstance(context, dict):
        raise ValueError(f"experiment context must be an object: {context_path}")
    attempt_id = context.get("attempt_id")
    if not isinstance(attempt_id, str) or not validate_attempt_id(attempt_id):
        raise ValueError(f"experiment context has invalid attempt_id: {attempt_id!r}")
    if context.get("fold") != args.fold:
        raise ValueError(
            f"experiment context fold {context.get('fold')!r} does not match --fold {args.fold}"
        )
    return context


def _attach_tracking_block(
    run_config: dict[str, Any], context: dict[str, Any], fold: int
) -> None:
    run_config["tracking"] = {
        "schema_version": "audiollm.tracking.v1",
        "group_id": context.get("group_id"),
        "logical_run_name": context.get("logical_run_name"),
        "attempt_id": context["attempt_id"],
        "fold": fold,
    }


def _initialize_tracking_sidecars(
    args: argparse.Namespace,
    context: dict[str, Any],
    run_root: Path,
    run_config: dict[str, Any],
) -> None:
    fold = int(args.fold)
    attempt_id = str(context["attempt_id"])
    now = format_utc_timestamp(utc_now())
    metadata = {
        "schema_version": SCHEMA_VERSION_METADATA,
        "group_id": context.get("group_id"),
        "logical_run_name": context.get("logical_run_name") or args.run_name,
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": context.get("seed"),
        "created_at_utc": now,
        "source": {
            "git_commit": context.get("source", {}).get("git_commit"),
            "git_branch": context.get("source", {}).get("git_branch"),
            "git_dirty": context.get("source", {}).get("git_dirty"),
            "deployed_source_sha256": context.get("source", {}).get("deployed_source_sha256"),
        },
        "research": {
            "github_issue": context.get("research", {}).get("github_issue"),
            "github_pr": context.get("research", {}).get("github_pr"),
        },
        "hashes": {
            "resolved_config_sha256": canonical_sha256(run_config.get("config") or {}),
            "manifest_sha256": context.get("hashes", {}).get("manifest_sha256")
            or run_config.get("manifest_hash"),
            "split_sha256": context.get("hashes", {}).get("split_sha256")
            or run_config.get("split_metadata_hash"),
        },
        "paths": {
            "run_config": "run_config.yaml",
            "best_model": "best_model",
            "local_evidence_root": None,
        },
        "wandb": {
            "project": WANDB_PROJECT,
            "entity": None,
            "run_id": f"{attempt_id}-fold{fold}",
            "url": None,
            "sync_status": "NOT_EXPORTED",
        },
    }
    write_json_atomic(run_root / "metadata.json", metadata)
    status = StatusRecord(attempt_id, fold, state="SUBMITTED")
    status.transition("RUNNING", reason="training job started")
    write_status(run_root / "status.json", status)
    slurm = context.get("slurm") if isinstance(context.get("slurm"), dict) else {}
    append_job_event(
        run_root / "jobs.jsonl",
        new_job_event(
            job_key="train",
            job_type="train",
            event_type="STARTED",
            attempt_id=attempt_id,
            fold=fold,
            slurm_job_id=slurm.get("train_job_id"),
            status="RUNNING",
        ),
    )
    _update_artifacts_json(
        run_root,
        attempt_id,
        fold,
        [
            ("run_config.yaml", "run_config", "run_config"),
        ],
    )


def _finalize_tracking_artifacts(
    args: argparse.Namespace,
    context: dict[str, Any],
    run_root: Path,
    run_config: dict[str, Any],
) -> None:
    fold = int(args.fold)
    attempt_id = str(context["attempt_id"])
    records: list[tuple[str, str, str]] = [
        ("logs/training_history.json", "training_history", "training_history"),
        ("logs/split_used.json", "split", "split_used"),
        ("logs/selected_checkpoint_selection_metrics.json", "metrics", "selection_metrics"),
        ("logs/sample_partition_counts.json", "audit", "partition_counts"),
        ("logs/peak_gpu_memory.json", "audit", "peak_gpu_memory"),
        ("logs/audio_budget_audit_train.json", "audit", "audio_budget_audit"),
    ]
    for checkpoint in ("best_model", "last_model"):
        if (run_root / checkpoint).is_dir():
            records.append((checkpoint, "checkpoint", "checkpoint_dir"))
    _update_artifacts_json(run_root, attempt_id, fold, records)
    slurm = context.get("slurm") if isinstance(context.get("slurm"), dict) else {}
    append_job_event(
        run_root / "jobs.jsonl",
        new_job_event(
            job_key="train",
            job_type="train",
            event_type="COMPLETED",
            attempt_id=attempt_id,
            fold=fold,
            slurm_job_id=slurm.get("train_job_id"),
            status="COMPLETED",
        ),
    )


def _update_artifacts_json(
    run_root: Path,
    attempt_id: str,
    fold: int,
    records: list[tuple[str, str, str]],
) -> None:
    path = run_root / "artifacts.json"
    existing: dict[str, Any] = {"artifacts": []}
    if path.is_file():
        try:
            existing = read_json(path)
        except (ValueError, OSError):
            existing = {"artifacts": []}
    known_paths = {artifact.get("path") for artifact in existing.get("artifacts", [])}
    for relative_path, artifact_type, role in records:
        if relative_path in known_paths:
            continue
        full_path = run_root / relative_path
        if not full_path.is_file() and artifact_type != "checkpoint":
            continue
        if artifact_type == "checkpoint" and not full_path.is_dir():
            continue
        try:
            sha256 = tracking_sha256_file(full_path) if full_path.is_file() else None
            size_bytes = full_path.stat().st_size if full_path.is_file() else None
        except OSError:
            continue
        existing.setdefault("artifacts", []).append(
            {
                "artifact_id": artifact_id(
                    attempt_id=attempt_id,
                    fold=fold,
                    role=role,
                    relative_path=relative_path,
                    artifact_sha256=sha256,
                ),
                "artifact_type": artifact_type,
                "role": role,
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "exists_on_mn5": True,
                "exists_locally": False,
                "locally_verified": False,
            }
        )
    existing.setdefault("schema_version", SCHEMA_VERSION_ARTIFACTS)
    existing.setdefault("attempt_id", attempt_id)
    existing.setdefault("fold", fold)
    write_json_atomic(path, existing)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe Qwen2-Audio depression detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--run_name", default="reproduction")
    parser.add_argument("--experiment-context", default=None, help="Path to experiment context JSON (attempt identity sidecars)")
    parser.add_argument("--label_mask_debug", action="store_true")
    parser.add_argument("--trial-progress-file", default=None)
    parser.add_argument("--trial-result-file", default=None)
    parser.add_argument(
        "--save_strategy",
        choices=("full", "best_only", "hpo_minimal"),
        default="full",
        help="Artifact retention strategy. Use hpo_minimal for Optuna trials.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE, using dot paths for nested keys.",
    )
    return parser.parse_args(argv)


def main() -> None:
    configure_logging()
    args = parse_args()
    tracking_context = _load_experiment_context(args)
    config = load_yaml_with_overrides(args.config, args.config_overrides)
    log_resolved_config(
        LOGGER,
        base_config_path=args.config,
        config_overrides=args.config_overrides,
        resolved_config=config,
    )
    set_seed(
        int(config["seed"]),
        deterministic=bool(config.get("training", {}).get("deterministic", True)),
    )
    input_modality = resolve_input_modality(config)
    LOGGER.info(
        "Resolved input modality=%s | use_audio=%s | use_text=%s",
        input_modality,
        bool(config["data"].get("use_audio", False)),
        bool(config["data"].get("use_text", False)),
    )
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    sample_prediction_mode = resolve_prediction_mode(config)
    aggregation_level = resolve_aggregation_level(config)
    metadata = _load_metadata_or_build(args.config, config, args.config_overrides)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    subject_labels = build_subject_label_map(manifest_rows)
    partition_plan = _resolve_training_subject_splits(config, metadata, subject_labels, args.fold)
    _apply_smoke_subject_limit(partition_plan, subject_labels, config, args.fold)
    split_mode = str(partition_plan["split_mode"])
    selection_enabled = bool(partition_plan["selection_enabled"])
    active_split_metadata_path = (
        metadata.get("folds_path")
        if split_mode == SPLIT_MODE_CV
        else metadata.get("subject_partition_path") or metadata.get("folds_path")
    )

    run_root = Path(config["output_dirs"]["run_root"]) / args.run_name / f"fold_{args.fold}"
    logs_dir = ensure_dir(run_root / "logs")
    eval_dir = run_root / "eval"
    best_dir = run_root / "best_model"
    last_dir = run_root / "last_model"

    split_payload = save_partition_subjects(
        logs_dir / "split_used.json",
        train_subject_ids=partition_plan["train_subject_ids"],
        selection_subject_ids=partition_plan["selection_subject_ids"],
        final_eval_subject_ids=partition_plan["final_eval_subject_ids"],
        subject_labels=subject_labels,
        train_split_name=partition_plan["train_split_name"],
        selection_split_name=partition_plan["selection_split_name"],
        final_eval_split_name=partition_plan["final_eval_split_name"],
    )
    _print_partition_counts(split_payload)

    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, config)
    processor = load_processor(model_name_or_path, config)
    processor_sampling_rate = resolve_processor_sampling_rate(processor)
    train_examples = build_examples(
        filter_rows_by_subjects(manifest_rows, partition_plan["train_subject_ids"]),
        config,
        partition_name=partition_plan["train_split_name"],
        truncation_log_path=logs_dir / "train_truncation.jsonl",
    )
    selection_examples: list[dict[str, Any]] = []
    if selection_enabled:
        selection_examples = build_examples(
            filter_rows_by_subjects(manifest_rows, partition_plan["selection_subject_ids"]),
            config,
            partition_name=partition_plan["selection_split_name"],
            truncation_log_path=logs_dir / "val_truncation.jsonl",
        )
    final_eval_examples: list[dict[str, Any]] = []
    if partition_plan["final_eval_subject_ids"]:
        final_eval_examples = build_examples(
            filter_rows_by_subjects(manifest_rows, partition_plan["final_eval_subject_ids"]),
            config,
            partition_name=partition_plan["final_eval_split_name"],
            truncation_log_path=logs_dir / "final_eval_truncation.jsonl",
        )
    sample_count_payload = {
        partition_plan["train_split_name"]: _sample_partition_counts(train_examples),
        partition_plan["final_eval_split_name"]: _sample_partition_counts(final_eval_examples),
    }
    sample_count_payload[partition_plan["selection_split_name"]] = _sample_partition_counts(selection_examples)
    save_json(sample_count_payload, logs_dir / "sample_partition_counts.json")
    for partition_name, counts in sample_count_payload.items():
        LOGGER.info(
            "%s | depressed_samples=%s non_depressed_samples=%s total_samples=%s",
            partition_name,
            counts["depressed_samples"],
            counts["non_depressed_samples"],
            counts["total_samples"],
        )

    joint_extra_examples, joint_selection_groups, joint_final_eval_groups, joint_composition = _build_joint_train_examples(config, logs_dir, model_name_or_path)
    primary_overlap_proof = _subject_overlap_proof(
        train_subject_ids=partition_plan["train_subject_ids"],
        selection_subject_ids=partition_plan["selection_subject_ids"],
        final_eval_subject_ids=partition_plan["final_eval_subject_ids"],
    )
    _assert_no_subject_overlap(str(config["dataset"]), primary_overlap_proof)
    if joint_extra_examples:
        primary_train_sample_count = len(train_examples)
        primary_composition = {
            "dataset": str(config["dataset"]),
            "config": str(Path(args.config)),
            "fold": int(args.fold),
            "train_split_name": partition_plan["train_split_name"],
            "selection_split_name": partition_plan["selection_split_name"],
            "final_eval_split_name": partition_plan["final_eval_split_name"],
            "train_subject_count": len(partition_plan["train_subject_ids"]),
            "selection_subject_count": len(partition_plan["selection_subject_ids"]),
            "final_eval_subject_count": len(partition_plan["final_eval_subject_ids"]),
            "train_sample_count": len(train_examples),
            "selection_sample_count": len(selection_examples),
            "final_eval_sample_count": len(final_eval_examples),
            "train_class_counts": _example_class_counts(train_examples),
            "selection_class_counts": _example_class_counts(selection_examples),
            "final_eval_class_counts": _example_class_counts(final_eval_examples),
            "train_subject_ids": sorted(partition_plan["train_subject_ids"]),
            "selection_subject_ids": sorted(partition_plan["selection_subject_ids"]),
            "final_eval_subject_ids": sorted(partition_plan["final_eval_subject_ids"]),
            "subject_overlap_proof": primary_overlap_proof,
        }
        train_examples = train_examples + joint_extra_examples
        save_json(
            {
                "primary_dataset": str(config["dataset"]),
                "joint_selection_mode": _resolve_joint_selection_mode(config),
                "primary_train_sample_count": primary_train_sample_count,
                "combined_train_sample_count": len(train_examples),
                "primary": primary_composition,
                "extra_datasets": joint_composition,
            },
            logs_dir / "joint_train_composition.json",
        )
        for item in joint_composition:
            train_counts = item["train_class_counts"]
            LOGGER.info(
                "Joint train | dataset=%s fold=%s train_subjects=%s train_samples=%s (dep=%s non=%s) heldout_subjects=%s",
                item["dataset"],
                item["fold"],
                item["train_subject_count"],
                item["train_sample_count"],
                train_counts["depressed_samples"],
                train_counts["non_depressed_samples"],
                item["heldout_subject_count"],
            )
        LOGGER.info(
            "Joint train combined | primary=%s primary_samples=%s combined_samples=%s",
            config["dataset"],
            primary_train_sample_count,
            len(train_examples),
        )

    original_train_sample_count = len(train_examples)
    train_examples, sampling_audit, train_shuffle_generator = _apply_subject_oversampling(
        train_examples,
        selection_examples,
        final_eval_examples,
        config,
        run_root,
    )
    androids_weight_audit: dict[str, Any] | None = None
    if (
        str(config["dataset"]).lower() == "androids_interview"
        and input_modality != "text_only"
    ):
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError(
                "ANDROIDS Interview hierarchical weighting requires "
                "per_device_train_batch_size=1."
            )
        train_examples, androids_weight_audit = apply_androids_training_weights(
            train_examples
        )
        save_json(
            androids_weight_audit,
            logs_dir / "androids_training_weight_audit.json",
        )
    packed30_weight_audit: dict[str, Any] | None = None
    if (
        str(config["data"].get("train_chunk_policy", "")).strip().lower()
        == ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY
        and input_modality != "text_only"
    ):
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError(
                f"train_chunk_policy={ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} requires "
                "per_device_train_batch_size=1."
            )
        if str(config.get("training", {}).get("class_balance", "none")).strip().lower() != "none":
            raise ValueError(
                f"train_chunk_policy={ALL_CHUNKS_SUBJECT_NORMALIZED_POLICY} requires "
                "training.class_balance=none (subject-normalized loss weights are the "
                "only v1 weighting)."
            )
        train_examples, packed30_weight_audit = apply_subject_normalized_chunk_weights(
            train_examples
        )
        save_json(
            packed30_weight_audit,
            logs_dir / "packed30_training_weight_audit.json",
        )
    d3tec_epoch_schedule: list[list[dict[str, Any]]] | None = None
    d3tec_schedule_audit: dict[str, Any] | None = None
    daic_epoch_schedule: list[list[dict[str, Any]]] | None = None
    daic_schedule_audit: dict[str, Any] | None = None
    if str(config["dataset"]).lower() == "d3tec" and input_modality != "text_only":
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError("D3TEC audio policies require per_device_train_batch_size=1.")
        if int(config["training"]["dataloader_num_workers"]) != 0:
            raise ValueError("D3TEC epoch schedules require training.dataloader_num_workers=0.")
        virtual_epochs = int(
            config["training"].get(
                "reference_virtual_epochs",
                config["training"]["num_train_epochs"],
            )
        )
        if int(config["training"]["num_train_epochs"]) != virtual_epochs:
            raise ValueError(
                "D3TEC num_train_epochs must equal training.reference_virtual_epochs."
            )
        reference_examples = int(
            config["training"].get("reference_examples_per_response", 1)
        )
        if reference_examples != 1:
            raise ValueError(
                "D3TEC requires training.reference_examples_per_response=1."
            )
        d3tec_epoch_schedule, d3tec_schedule_audit = build_d3tec_training_schedule(
            train_examples,
            policy=str(config["data"]["train_chunk_policy"]),
            seed=int(config["seed"]),
            virtual_epochs=virtual_epochs,
            responses_per_subject=27,
        )
        train_examples = d3tec_epoch_schedule[0]
        save_json(d3tec_schedule_audit, logs_dir / "d3tec_training_schedule_audit.json")
    if (
        str(config["dataset"]).lower() == "daic"
        and str(config["data"].get("sample_mode", "")).lower() in {"subject_chunks", "subject_mil"}
        and str(config.get("training", {}).get("objective", "token_ce")) == "token_ce"
    ):
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError("DAIC independent chunk weighting requires per_device_train_batch_size=1.")
        if int(config["training"]["dataloader_num_workers"]) != 0:
            raise ValueError("DAIC rotary schedules require training.dataloader_num_workers=0.")
        controls = resolve_chunking_controls(config)
        daic_epoch_schedule, daic_schedule_audit = build_independent_epoch_schedule(
            train_examples,
            policy=controls["train_chunk_policy"],
            chunks_per_subject=controls["train_chunks_per_subject"],
            seed=int(config["seed"]),
            epochs=int(config["training"]["num_train_epochs"]),
            loss_weight_rescale=controls["loss_weight_rescale"],
            equal_row_weight=bool(config["data"].get("equal_row_weight", False)),
            class_balance=str(config["training"].get("class_balance", "none")) == "subject_inverse_frequency",
        )
        train_examples = daic_epoch_schedule[0]
        if bool(config["training"].get("match_joint_optimizer_updates", True)):
            reference_accumulation = int(
                config["training"].get("joint_reference_gradient_accumulation_steps", 8)
            )
            resolved_accumulation = gradient_accumulation_for_reference_updates(
                independent_examples_per_epoch=len(train_examples),
                reference_subjects=len(partition_plan["train_subject_ids"]),
                reference_gradient_accumulation=reference_accumulation,
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                per_device_batch_size=int(config["training"]["per_device_train_batch_size"]),
            )
            config["training"]["gradient_accumulation_steps"] = resolved_accumulation
            daic_schedule_audit["gradient_accumulation"] = {
                "policy": "match_joint_optimizer_updates_per_epoch",
                "joint_reference_gradient_accumulation_steps": reference_accumulation,
                "resolved_gradient_accumulation_steps": resolved_accumulation,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            }
        save_json(daic_schedule_audit, logs_dir / "daic_chunk_schedule_audit.json")
    if (
        str(config["dataset"]).lower() == "daic"
        and str(config["data"].get("sample_mode", "")).lower()
        in {"subject_audio", JOINT_PACKED30_MODE}
        and str(config["data"].get("train_chunk_policy", "random_k")) in {"joint_random_k", "joint_rotary_k", "joint_balanced_cover"}
    ):
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError("DAIC joint weighted schedules require per_device_train_batch_size=1.")
        if int(config["training"]["dataloader_num_workers"]) != 0:
            raise ValueError("DAIC joint schedules require training.dataloader_num_workers=0.")
        controls = resolve_chunking_controls(config)
        daic_epoch_schedule, daic_schedule_audit = build_joint_epoch_schedule(
            train_examples,
            policy=controls["train_chunk_policy"],
            k=int(controls["train_chunks_per_subject"]),
            seed=int(config["seed"]),
            epochs=int(config["training"]["num_train_epochs"]),
            loss_weight_rescale=controls["loss_weight_rescale"],
            class_balance=str(config["training"].get("class_balance", "none")) == "subject_inverse_frequency",
        )
        if (
            str(config["data"].get("sample_mode", "")).lower() == JOINT_PACKED30_MODE
            and str(config["data"].get("train_chunk_policy", "")) in {"joint_random_k", "joint_rotary_k"}
        ):
            if str(config["data"].get("loss_weight_rescale", "none")) != "mean_one":
                raise ValueError(
                    f"sample_mode={JOINT_PACKED30_MODE} requires "
                    "data.loss_weight_rescale=mean_one."
                )
            # Span-group bundles re-render the prompt per epoch because the
            # placeholder count follows the current bundle size (K, or 3 for
            # subject 385). Training, evaluation, and hidden extraction share
            # the same renderer. joint_random_k and joint_rotary_k both draw
            # K=min(requested_k, N) chunks per subject per epoch, so both need
            # the re-render.
            for epoch_rows in daic_epoch_schedule:
                for row in epoch_rows:
                    row["prompt_text"], row["training_text"] = render_joint_packed30_bundle(
                        row, len(row["audio_span_groups"])
                    )
        train_examples = daic_epoch_schedule[0]
        if bool(config["training"].get("match_joint_optimizer_updates", True)):
            reference_accumulation = int(config["training"].get("joint_reference_gradient_accumulation_steps", 8))
            resolved_accumulation = gradient_accumulation_for_reference_updates(
                independent_examples_per_epoch=len(train_examples),
                reference_subjects=len(partition_plan["train_subject_ids"]),
                reference_gradient_accumulation=reference_accumulation,
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                per_device_batch_size=int(config["training"]["per_device_train_batch_size"]),
            )
            config["training"]["gradient_accumulation_steps"] = resolved_accumulation
            daic_schedule_audit["gradient_accumulation"] = {
                "policy": "match_jr4_optimizer_updates_per_epoch",
                "joint_reference_gradient_accumulation_steps": reference_accumulation,
                "resolved_gradient_accumulation_steps": resolved_accumulation,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            }
        save_json(daic_schedule_audit, logs_dir / "daic_chunk_schedule_audit.json")

    # Training is the only place stochastic per-epoch chunk sampling is allowed.
    # Selection/eval datasets stay deterministic (baked audio_paths) so reported
    # validation/test metrics never depend on random sampling.
    train_chunk_sampling = (
        "random" if (
            str(config["data"].get("sample_mode", "")).lower() == "subject_audio"
            and str(config["data"].get("train_chunk_policy", "random_k")) == "random_k"
        ) else None
    )
    # Train-only waveform acoustic augmentation. Passed ONLY to the train dataset;
    # selection/audit stay clean and eval never touches AudioTextDataset, so the
    # eval-determinism rule (handoff §3) holds.
    audio_augment_cfg = config["data"].get("audio_augment")
    if audio_augment_cfg and audio_augment_cfg.get("enabled"):
        LOGGER.info("Train audio augmentation enabled | cfg=%s", audio_augment_cfg)
    train_dataset = AudioTextDataset(
        train_examples,
        processor_sampling_rate=processor_sampling_rate,
        silence_audio=bool(config["data"].get("silence_audio", False)),
        chunk_sampling=train_chunk_sampling,
        chunk_sampling_seed=int(config["seed"]),
        audio_augment=audio_augment_cfg,
    )
    selection_dataset = None
    if selection_enabled:
        selection_dataset = AudioTextDataset(
            selection_examples,
            processor_sampling_rate=processor_sampling_rate,
            silence_audio=bool(config["data"].get("silence_audio", False)),
            chunk_sampling="deterministic",
        )
    collator = Qwen2AudioSFTCollator(processor=processor, debug=args.label_mask_debug)
    if args.label_mask_debug:
        _emit_label_mask_debug(train_dataset, collator, processor, logs_dir)

    train_sampler = _build_weighted_train_sampler(train_examples, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["per_device_train_batch_size"]),
        shuffle=False if (d3tec_epoch_schedule is not None or daic_epoch_schedule is not None) else train_sampler is None,
        sampler=train_sampler,
        generator=train_shuffle_generator,
        num_workers=int(config["training"]["dataloader_num_workers"]),
        collate_fn=collator,
    )
    selection_loss_loader = None
    if selection_enabled and selection_dataset is not None:
        selection_loss_loader = DataLoader(
            selection_dataset,
            batch_size=int(config["training"]["per_device_eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["dataloader_num_workers"]),
            collate_fn=Qwen2AudioSFTCollator(processor=processor, debug=False),
        )
    selection_components: list[dict[str, Any]] = []
    if selection_enabled:
        selection_components.append(
            {
                "name": f"{str(config['dataset']).lower()}_val",
                "dataset": str(config["dataset"]).lower(),
                "split_name": partition_plan["selection_split_name"],
                "examples": selection_examples,
                "config": config,
                "loss_loader": selection_loss_loader,
                "log_dir_prefix": partition_plan["selection_log_dir_name"],
                "subject_count": len(partition_plan["selection_subject_ids"]),
                "sample_count": len(selection_examples),
            }
        )
        for group in joint_selection_groups:
            group_dataset = AudioTextDataset(
                group["examples"],
                processor_sampling_rate=processor_sampling_rate,
                silence_audio=bool(group["config"]["data"].get("silence_audio", False)),
                chunk_sampling="deterministic",
            )
            group_loader = DataLoader(
                group_dataset,
                batch_size=int(config["training"]["per_device_eval_batch_size"]),
                shuffle=False,
                num_workers=int(config["training"]["dataloader_num_workers"]),
                collate_fn=Qwen2AudioSFTCollator(processor=processor, debug=False),
            )
            selection_components.append(
                {
                    "name": f"{str(group['dataset']).lower()}_inner_val",
                    "dataset": str(group["dataset"]).lower(),
                    "split_name": group["split_name"],
                    "examples": group["examples"],
                    "config": group["config"],
                    "loss_loader": group_loader,
                    "log_dir_prefix": group["split_name"],
                    "subject_count": len(group["subject_ids"]),
                    "sample_count": len(group["examples"]),
                }
            )

    model = load_model_for_training(model_name_or_path, config)
    lora_layer_selection = resolved_lora_layer_selection(model)
    optimizer = AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    objective = str(config.get("training", {}).get("objective", "token_ce"))
    if objective not in {"token_ce", "subject_mean_margin_mil"}:
        raise ValueError(f"Unsupported training.objective={objective!r}.")
    if objective == "subject_mean_margin_mil":
        if str(config["dataset"]).lower() != "daic" or str(config["data"].get("sample_mode", "")).lower() != "subject_mil":
            raise ValueError("subject_mean_margin_mil requires dataset=daic and data.sample_mode=subject_mil.")
        if int(config["training"]["per_device_train_batch_size"]) != 1:
            raise ValueError("subject_mean_margin_mil requires per_device_train_batch_size=1.")
        if int(config["training"]["dataloader_num_workers"]) != 0:
            raise ValueError("subject_mean_margin_mil requires dataloader_num_workers=0.")
        if int(config["training"].get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError("subject_mean_margin_mil requires gradient_accumulation_steps=1 so a subject cannot be split across updates.")
    micro_batches_per_epoch = len(train_loader)
    if objective == "subject_mean_margin_mil":
        micro_batches_per_epoch = len({str(row["subject_id"]) for row in train_examples})
    total_steps = max(
        1,
        math.ceil(micro_batches_per_epoch / int(config["training"]["gradient_accumulation_steps"])) * int(config["training"]["num_train_epochs"]),
    )
    warmup_steps = int(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dist_timeout_minutes = int(config.get("training", {}).get("dist_timeout_minutes", 0) or 0)
    if dist_timeout_minutes > 0 and not torch.distributed.is_initialized():
        # Rank-0-only per-epoch selection evaluation (e.g. 445 balanced-cover
        # bundles) can hold the other ranks at the next collective for longer
        # than torch's default 600s NCCL watchdog timeout. Pre-initialize the
        # process group with a longer timeout; Accelerate reuses an already
        # initialized group. Opt-in via training.dist_timeout_minutes so other
        # recipes keep the default behavior.
        import datetime

        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(minutes=dist_timeout_minutes)
        )
        LOGGER.info(
            "Pre-initialized NCCL process group with timeout=%s minutes "
            "(training.dist_timeout_minutes).",
            dist_timeout_minutes,
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(config["training"].get("bf16", False)) else "no",
        kwargs_handlers=[ddp_kwargs],
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    if daic_schedule_audit is not None:
        gradient_accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
        schedule_micro_batches = len(train_loader)
        schedule_epochs = int(daic_schedule_audit.get("epochs", len(daic_epoch_schedule or [])))
        schedule_updates = math.ceil(schedule_micro_batches / max(1, gradient_accumulation))
        daic_schedule_audit.update(
            {
                "micro_batches_per_epoch": [schedule_micro_batches] * schedule_epochs,
                "optimizer_updates_per_epoch": [schedule_updates] * schedule_epochs,
                "gradient_accumulation_steps": gradient_accumulation,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "optimizer_step_unit": "weighted_chunk_or_bundle_example",
            }
        )
        save_json(daic_schedule_audit, logs_dir / "daic_chunk_schedule_audit.json")

    run_config = {
        "config": config,
        "base_config_path": str(Path(args.config)),
        "config_overrides": list(args.config_overrides),
        "evaluation": {
            "sample_prediction_mode": sample_prediction_mode,
            "aggregation_level": aggregation_level,
            "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
        },
        "input_modality": input_modality,
        "audio_adapter": audio_adapter_cfg,
        "lora_resolution": lora_layer_selection,
        "resolved_model_name_or_path": model_name_or_path,
        "manifest_path": metadata["manifest_path"],
        "manifest_hash": metadata["manifest_hash"],
        "split_metadata_path": active_split_metadata_path,
        "split_metadata_hash": sha256_file(active_split_metadata_path),
        "split_mode": split_mode,
        "cv_protocol": partition_plan["cv_protocol"],
        "fold": int(args.fold),
        "save_strategy": args.save_strategy,
        "sampling": {
            "mode": str(config.get("training", {}).get("class_balance", "none")).strip().lower(),
            "oversampling_ratio": config.get("training", {}).get("oversampling_ratio"),
            "oversampling_seed": config.get("training", {}).get("oversampling_seed"),
            "original_train_sample_count": original_train_sample_count,
            "effective_train_sample_count": len(train_examples),
            "audit_path": str(run_root / "sampling_audit.json") if sampling_audit else None,
            "audit": sampling_audit,
            "d3tec_schedule_audit_path": (
                str(logs_dir / "d3tec_training_schedule_audit.json")
                if d3tec_schedule_audit is not None
                else None
            ),
            "daic_schedule_audit_path": (
                str(logs_dir / "daic_chunk_schedule_audit.json")
                if daic_schedule_audit is not None
                else None
            ),
            "androids_weight_audit_path": (
                str(logs_dir / "androids_training_weight_audit.json")
                if androids_weight_audit is not None
                else None
            ),
            "packed30_weight_audit_path": (
                str(logs_dir / "packed30_training_weight_audit.json")
                if packed30_weight_audit is not None
                else None
            ),
        },
        "subject_overlap_proof": primary_overlap_proof,
        "selection_protocol": {
            "mode": partition_plan["selection_mode"],
            "joint_selection_mode": _resolve_joint_selection_mode(config),
            "metric_name": None,
            "selection_split_name": partition_plan["selection_split_name"],
            "selection_subject_count": len(partition_plan["selection_subject_ids"]),
            "selection_sample_count": len(selection_examples),
            "selection_components": [
                {
                    "name": component["name"],
                    "dataset": component["dataset"],
                    "split_name": component["split_name"],
                    "subject_count": component["subject_count"],
                    "sample_count": component["sample_count"],
                }
                for component in selection_components
            ],
            "selection_prediction_backend": sample_prediction_mode if selection_enabled else None,
            "selection_aggregation_level": aggregation_level if selection_enabled else None,
            "selection_evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode) if selection_enabled else None,
        },
        "final_eval_protocol": {
            "final_eval_split_name": partition_plan["final_eval_split_name"],
            "final_eval_partition": None if split_mode == SPLIT_MODE_CV else str(config["split"]["final_eval_partition"]),
            "final_eval_subject_count": len(partition_plan["final_eval_subject_ids"]),
            "final_eval_sample_count": len(final_eval_examples),
            "final_eval_aggregation_level": aggregation_level,
            "final_eval_components": [
                {
                    "name": f"{str(config['dataset']).lower()}_{partition_plan['final_eval_split_name']}",
                    "dataset": str(config["dataset"]).lower(),
                    "split_name": partition_plan["final_eval_split_name"],
                    "subject_count": len(partition_plan["final_eval_subject_ids"]),
                    "sample_count": len(final_eval_examples),
                    "aggregation_level": aggregation_level,
                }
            ]
            + [
                {
                    "name": f"{str(group['dataset']).lower()}_fold_holdout",
                    "dataset": str(group["dataset"]).lower(),
                    "split_name": group["split_name"],
                    "subject_count": len(group["subject_ids"]),
                    "sample_count": len(group["examples"]),
                    "aggregation_level": resolve_aggregation_level(group["config"]),
                }
                for group in joint_final_eval_groups
            ],
            "run_final_eval_in_train": (
                bool(config["training"].get("run_final_eval_in_train", False))
                and partition_plan["cv_protocol"] != CV_PROTOCOL_TRAIN_VAL
            ),
        },
    }

    selection_metric_cfg = _resolve_selection_metric(config)
    run_config["selection_protocol"]["metric_name"] = selection_metric_cfg["metric"] if selection_enabled else None
    run_config["selection_protocol"]["metric_mode"] = selection_metric_cfg["mode"] if selection_enabled else None
    if tracking_context is not None and accelerator.is_main_process:
        _attach_tracking_block(run_config, tracking_context, int(args.fold))
    if accelerator.is_main_process:
        save_yaml(run_config, run_root / "run_config.yaml")
        if tracking_context is not None:
            _initialize_tracking_sidecars(args, tracking_context, run_root, run_config)
    best_metric: float | None = (
        None
        if not selection_enabled
        else (float("inf") if selection_metric_cfg["mode"] == "min" else float("-inf"))
    )
    best_epoch = -1 if selection_enabled else 0
    best_selection_auroc = float("-inf")
    best_selection_loss = float("inf")
    if selection_enabled:
        LOGGER.info(
            "Checkpoint selection metric=%s mode=%s (early_stopping metric handled separately).",
            selection_metric_cfg["metric"],
            selection_metric_cfg["mode"],
        )
    early_stop_cfg = _resolve_early_stopping(config)
    if not selection_enabled and early_stop_cfg["enabled"]:
        LOGGER.info("Disabling early stopping because split.mode=%s does not create a selection split.", split_mode)
        early_stop_cfg["enabled"] = False
    early_stop_best = float("inf") if early_stop_cfg["mode"] == "min" else float("-inf")
    early_stop_best_epoch = -1
    early_stop_bad_epochs = 0
    stopped_early = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    history: list[dict[str, Any]] = []
    mil_training_audit: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if (
        accelerator.is_main_process
        and bool(config["data"].get("use_audio", False))
        and bool(config["training"].get("audio_budget_audit", True))
    ):
        # Dedicated instance with the same seed/sampler as training so the audit
        # reflects the real (possibly stochastic) training audio path without
        # advancing the training dataset's RNG.
        audit_dataset = AudioTextDataset(
            train_examples,
            processor_sampling_rate=processor_sampling_rate,
            silence_audio=bool(config["data"].get("silence_audio", False)),
            chunk_sampling=train_chunk_sampling,
            chunk_sampling_seed=int(config["seed"]),
        )
        _audit_audio_budget(
            audit_dataset,
            Qwen2AudioSFTCollator(processor=processor, debug=False),
            processor,
            processor_sampling_rate,
            partition_plan["train_split_name"],
            logs_dir,
        )

    for epoch in range(1, int(config["training"]["num_train_epochs"]) + 1):
        if d3tec_epoch_schedule is not None:
            train_dataset.examples = d3tec_epoch_schedule[epoch - 1]
        if daic_epoch_schedule is not None:
            train_dataset.examples = daic_epoch_schedule[epoch - 1]
        model.train()
        # The previous epoch's selection eval disables gradient checkpointing and
        # enables use_cache; restore the training-time memory config before this
        # epoch's forward, or activations balloon and OOM the heaviest configs.
        restore_model_for_training(accelerator.unwrap_model(model), config)
        epoch_losses: list[float] = []
        objective = str(config.get("training", {}).get("objective", "token_ce"))
        if objective == "subject_mean_margin_mil":
            if accelerator.num_processes != 1:
                raise ValueError("subject_mean_margin_mil requires exactly one process/GPU.")
            grouped_mil: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for example in train_examples:
                grouped_mil[str(example["subject_id"])].append(example)
            from src.daic_chunking import deterministic_subject_order
            mil_subjects = deterministic_subject_order(grouped_mil, seed=int(config["seed"]), epoch=epoch - 1)
            mil_batches = []
            for subject_id in mil_subjects:
                subject_examples = sorted(grouped_mil[subject_id], key=lambda row: str(row["sample_id"]))
                label = int(subject_examples[0]["label"])

                def margin_fn(example):
                    prompt_inputs = _processor_inputs(
                        processor, example, example["prompt_text"], accelerator.device,
                        bool(config["data"].get("silence_audio", False)),
                    )
                    prompt_len = int(prompt_inputs["input_ids"].shape[1])
                    scores = []
                    for candidate_label in (
                        internal_label_text_from_int(config, 1),
                        internal_label_text_from_int(config, 0),
                    ):
                        full_inputs = _processor_inputs(
                            processor, example, example["prompt_text"] + candidate_label,
                            accelerator.device, bool(config["data"].get("silence_audio", False)),
                        )
                        scores.append(candidate_mean_token_logprob(model, full_inputs, prompt_len))
                    return scores[0] - scores[1]

                with accelerator.accumulate(model):
                    result = streaming_subject_mil_backward(
                        subject_examples, label=label, margin_fn=margin_fn,
                        backward_fn=accelerator.backward,
                    )
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                mil_batches.append(result)
            training_batches = []
        elif objective == "token_ce":
            mil_batches = []
            training_batches = train_loader
        else:
            raise ValueError(f"Unsupported training.objective={objective!r}.")
        for step, batch in enumerate(training_batches, start=1):
            with accelerator.accumulate(model):
                loss_weights = batch.pop("loss_weight", None)
                outputs = model(**batch)
                loss = outputs.loss
                if loss_weights is not None:
                    if int(loss_weights.numel()) != 1:
                        raise ValueError(
                            "Per-example hierarchical loss weighting currently "
                            "requires batch size 1."
                        )
                    loss = loss * loss_weights.reshape(-1)[0].to(loss.device)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_losses.append(float(loss.detach().item()))
            if accelerator.is_main_process and step % int(config["training"]["logging_steps"]) == 0:
                LOGGER.info("epoch=%s step=%s loss=%.6f", epoch, step, sum(epoch_losses) / len(epoch_losses))
        if mil_batches:
            epoch_losses.extend(float(row["loss"]) for row in mil_batches)
            mil_training_audit.append(
                {
                    "epoch": epoch,
                    "subject_updates": len(mil_batches),
                    "class_updates": {
                        "non_depressed": sum(
                            int(int(grouped_mil[str(subject_id)][0]["label"]) == 0)
                            for subject_id in mil_subjects
                        ),
                        "depressed": sum(
                            int(int(grouped_mil[str(subject_id)][0]["label"]) == 1)
                            for subject_id in mil_subjects
                        ),
                    },
                    "complete_subject_updates": len(mil_batches),
                    "mean_subject_loss": sum(float(row["loss"]) for row in mil_batches) / len(mil_batches),
                    "subject_chunk_counts": {
                        str(subject_id): len(grouped_mil[str(subject_id)]) for subject_id in mil_subjects
                    },
                }
            )
            LOGGER.info(
                "epoch=%s MIL subjects=%s mean_loss=%.6f", epoch, len(mil_batches),
                sum(epoch_losses) / len(epoch_losses),
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if selection_enabled:
                component_headlines: list[tuple[str, dict[str, Any]]] = []
                component_losses: dict[str, float] = {}
                component_eval_dirs: dict[str, str] = {}
                primary_headline_metrics: dict[str, Any] | None = None
                primary_selection_loss: float | None = None
                primary_eval_dir: Path | None = None
                for component in selection_components:
                    component_eval_dir = ensure_dir(logs_dir / f"{component['log_dir_prefix']}_epoch_{epoch}")
                    component_loss = _compute_dataset_loss(unwrapped, component["loss_loader"])
                    LOGGER.info(
                        "Selection evaluation dataset=%s split=%s | backend=%s | aggregation_level=%s | protocol=%s",
                        component["dataset"],
                        component["split_name"],
                        sample_prediction_mode,
                        resolve_aggregation_level(component["config"]),
                        evaluation_protocol_name(sample_prediction_mode),
                    )
                    metrics = evaluate_examples(
                        unwrapped,
                        processor,
                        component["examples"],
                        component["config"],
                        component_eval_dir,
                        checkpoint_name=f"epoch_{epoch}",
                        sample_prediction_mode=sample_prediction_mode,
                    )
                    headline_metrics = metrics["backend_results"][sample_prediction_mode]["headline_metrics"]
                    component_headlines.append((component["name"], headline_metrics))
                    component_losses[f"{component['name']}_loss"] = float(component_loss)
                    component_eval_dirs[component["name"]] = str(component_eval_dir)
                    if component is selection_components[0]:
                        primary_headline_metrics = headline_metrics
                        primary_selection_loss = float(component_loss)
                        primary_eval_dir = component_eval_dir
                    LOGGER.info(
                        "Selection component epoch=%s | dataset=%s split=%s aggregation_level=%s | loss=%.6f "
                        "positive_f1=%.6f macro_f1=%.6f precision=%.6f recall=%.6f",
                        epoch,
                        component["dataset"],
                        component["split_name"],
                        resolve_aggregation_level(component["config"]),
                        component_loss,
                        float(headline_metrics["positive_f1"]),
                        float(headline_metrics["macro_f1"]),
                        float(headline_metrics["precision"]),
                        float(headline_metrics["recall"]),
                    )
                if primary_headline_metrics is None or primary_selection_loss is None or primary_eval_dir is None:
                    raise RuntimeError("Selection was enabled but no selection components were evaluated.")
                metric_value = float(primary_headline_metrics["positive_f1"])
                metric_values = _selection_metric_values(metric_value, primary_headline_metrics, primary_selection_loss)
                if len(component_headlines) > 1:
                    metric_values.update(_joint_selection_values(component_headlines))
                    metric_values.update(component_losses)
                selection_loss = primary_selection_loss
                selection_eval_dir = primary_eval_dir
                headline_metrics = primary_headline_metrics
                selection_metric_name = selection_metric_cfg["metric"]
                if selection_metric_name not in metric_values:
                    raise ValueError(
                        f"Unknown training.selection_metric={selection_metric_name!r}. "
                        f"Available: {sorted(metric_values)}"
                    )
                selection_value = float(metric_values[selection_metric_name])
                history_row = {
                    "epoch": epoch,
                    "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                    "selection_split_name": partition_plan["selection_split_name"],
                    "selection_loss": selection_loss,
                    "selection_prediction_backend": sample_prediction_mode,
                    "selection_aggregation_level": aggregation_level,
                    "selection_evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
                    "inner_val_loss": selection_loss,
                    "inner_val_prediction_backend": sample_prediction_mode,
                    "inner_val_aggregation_level": aggregation_level,
                    "inner_val_evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
                    "selection_component_eval_dirs": component_eval_dirs,
                    **metric_values,
                }
                history.append(history_row)
                LOGGER.info(
                    "Selection epoch=%s | selected_metric=%s value=%.6f | primary_split=%s | aggregation_level=%s | "
                    "loss=%.6f ACC=%.6f positive_f1=%.6f macro_f1=%.6f Precision=%.6f Recall=%.6f",
                    epoch,
                    selection_metric_name,
                    selection_value,
                    partition_plan["selection_split_name"],
                    aggregation_level,
                    selection_loss,
                    float(headline_metrics["accuracy"]),
                    float(headline_metrics["positive_f1"]),
                    float(headline_metrics["macro_f1"]),
                    float(headline_metrics["precision"]),
                    float(headline_metrics["recall"]),
                )
                selection_improved = _metric_improved(
                    selection_value,
                    best_metric,
                    selection_metric_cfg["mode"],
                    0.0,
                )
                if (
                    str(config["dataset"]).lower() == "d3tec"
                    and not selection_improved
                    and float(selection_value) == float(best_metric)
                ):
                    current_auroc = float(metric_values.get("inner_val_auroc", 0.0))
                    selection_improved = (
                        current_auroc > best_selection_auroc
                        or (
                            current_auroc == best_selection_auroc
                            and float(selection_loss) < best_selection_loss
                        )
                    )
                if selection_improved:
                    best_metric = selection_value
                    best_epoch = epoch
                    best_selection_auroc = float(metric_values.get("inner_val_auroc", 0.0))
                    best_selection_loss = float(selection_loss)
                    if _save_best_checkpoint(args.save_strategy):
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        save_adapter_and_processor(unwrapped, processor, best_dir, config=config)
                    if partition_plan["cv_protocol"] == CV_PROTOCOL_TRAIN_VAL:
                        best_validation_dir = eval_dir / "best_validation"
                        if best_validation_dir.exists():
                            shutil.rmtree(best_validation_dir)
                        ensure_dir(eval_dir)
                        shutil.copytree(selection_eval_dir, best_validation_dir)
                        save_json(
                            {
                                "cv_protocol": CV_PROTOCOL_TRAIN_VAL,
                                "score_source": "best_outer_fold_validation",
                                "fold": int(args.fold),
                                "epoch": int(epoch),
                                "selection_metric": selection_metric_name,
                                "selection_metric_mode": selection_metric_cfg["mode"],
                                "selection_metric_value": selection_value,
                                "component_selection_metrics": metric_values,
                                "component_eval_dirs": component_eval_dirs,
                                "prediction_backend": sample_prediction_mode,
                                "aggregation_level": aggregation_level,
                                "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
                                "metrics_path": str(best_validation_dir / {
                                    "likelihood": "metrics_likelihood.json",
                                    "generation": "metrics_generation.json",
                                    "original_teacher_forced": "metrics_original_teacher_forced.json",
                                }[sample_prediction_mode]),
                                "active_backend_metrics": headline_metrics,
                            },
                            best_validation_dir / "selection.json",
                        )
                _write_trial_progress(
                    args.trial_progress_file,
                    epoch=epoch,
                    metric_name=selection_metric_name,
                    metric_value=selection_value,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    run_root=run_root,
                    config_overrides=args.config_overrides,
                )
                if early_stop_cfg["enabled"]:
                    monitor_value = float(metric_values[early_stop_cfg["metric"]])
                    if _metric_improved(
                        monitor_value,
                        early_stop_best,
                        early_stop_cfg["mode"],
                        early_stop_cfg["min_delta"],
                    ):
                        early_stop_best = monitor_value
                        early_stop_best_epoch = epoch
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                    LOGGER.info(
                        "Early stopping monitor=%s mode=%s epoch=%s value=%.6f best=%.6f best_epoch=%s bad_epochs=%s patience=%s",
                        early_stop_cfg["metric"],
                        early_stop_cfg["mode"],
                        epoch,
                        monitor_value,
                        early_stop_best,
                        early_stop_best_epoch,
                        early_stop_bad_epochs,
                        early_stop_cfg["patience"],
                    )
                LOGGER.info("Finished epoch=%s | best_epoch=%s best_metric=%.6f", epoch, best_epoch, best_metric)
            else:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                        "selection_protocol_mode": "none",
                    }
                )
                LOGGER.info("Finished epoch=%s | selection disabled | train_loss=%.6f", epoch, sum(epoch_losses) / max(1, len(epoch_losses)))
            _log_peak_gpu_memory(LOGGER, f"epoch_{epoch}")
        stop_training_tensor = torch.tensor(0, device=accelerator.device, dtype=torch.int32)
        if accelerator.is_main_process and early_stop_cfg["enabled"] and early_stop_bad_epochs >= early_stop_cfg["patience"]:
            stopped_early = True
            stop_epoch = epoch
            stop_reason = (
                f"early_stopping:{early_stop_cfg['metric']}:{early_stop_cfg['mode']}:"
                f"patience={early_stop_cfg['patience']}:best_epoch={early_stop_best_epoch}"
            )
            LOGGER.info("Stopping early at epoch=%s | %s", epoch, stop_reason)
            stop_training_tensor.fill_(1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_training_tensor, src=0)
        accelerator.wait_for_everyone()
        if int(stop_training_tensor.item()) == 1:
            break
        # Release the per-epoch selection-eval allocations (run on rank 0 only)
        # before the next epoch's training step. Without this, the fragmented
        # cached blocks left by evaluation can prevent the next training forward
        # from finding a contiguous block for the fp32 logits, OOMing on the
        # heaviest LoRA configs even though epoch 1 fit.
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if objective == "subject_mean_margin_mil":
            save_json(
                {
                    "schema_version": "daic_subject_mean_margin_mil_audit.v1",
                    "objective": objective,
                    "epochs": mil_training_audit,
                    "complete_subject_updates": sum(
                        int(row["complete_subject_updates"]) for row in mil_training_audit
                    ),
                    "subjects_per_epoch": [int(row["subject_updates"]) for row in mil_training_audit],
                    "gradient_passes_per_subject": 2,
                    "optimizer_step_unit": "complete_subject",
                },
                logs_dir / "mil_training_audit.json",
            )
        completed_epochs = len(history)
        save_json(history, logs_dir / "training_history.json")
        selected_history_row = next(
            (row for row in history if int(row.get("epoch", -1)) == int(best_epoch)),
            None,
        )
        if selection_enabled:
            save_json(
                {
                    "selected_epoch": int(best_epoch),
                    "selection_metric": selection_metric_cfg["metric"],
                    "selection_metric_mode": selection_metric_cfg["mode"],
                    "selection_metric_value": None if best_metric is None else float(best_metric),
                    "component_selection_metrics": selected_history_row,
                },
                logs_dir / "selected_checkpoint_selection_metrics.json",
            )
            LOGGER.info(
                "Selected checkpoint epoch=%s metric=%s value=%.6f components=%s",
                best_epoch,
                selection_metric_cfg["metric"],
                float(best_metric),
                selected_history_row,
            )
        peak_gpu_memory = _log_peak_gpu_memory(LOGGER, "final")
        if peak_gpu_memory is not None:
            save_json(peak_gpu_memory, logs_dir / "peak_gpu_memory.json")
        if selection_enabled and _save_last_checkpoint(args.save_strategy):
            if last_dir.exists():
                shutil.rmtree(last_dir)
            save_adapter_and_processor(unwrapped, processor, last_dir, config=config)
        if not selection_enabled:
            if last_dir.exists():
                shutil.rmtree(last_dir)
            save_adapter_and_processor(unwrapped, processor, last_dir, config=config)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(last_dir, best_dir)
            best_epoch = completed_epochs
        run_final_eval_in_train = bool(config["training"].get("run_final_eval_in_train", False))
        if partition_plan["cv_protocol"] == CV_PROTOCOL_TRAIN_VAL:
            if run_final_eval_in_train:
                LOGGER.info(
                    "Skipping final held-out test evaluation because split.cv_protocol=%s "
                    "uses the outer fold as validation.",
                    CV_PROTOCOL_TRAIN_VAL,
                )
            run_final_eval_in_train = False
        if selection_enabled and run_final_eval_in_train and not _save_best_checkpoint(args.save_strategy):
            LOGGER.info(
                "Skipping final held-out evaluation inside training because save_strategy=%s does not keep best checkpoints.",
                args.save_strategy,
            )
            run_final_eval_in_train = False
        if run_final_eval_in_train:
            ensure_dir(eval_dir)
            LOGGER.info(
                "Final held-out evaluation backend: %s | aggregation_level=%s | protocol=%s",
                sample_prediction_mode,
                aggregation_level,
                evaluation_protocol_name(sample_prediction_mode),
            )
            evaluate_last_checkpoint = bool(
                config.get("evaluation", {}).get("evaluate_last_checkpoint", True)
            )
            last_metrics = None
            if evaluate_last_checkpoint:
                LOGGER.info("Starting final held-out evaluation for last_checkpoint")
                prepare_model_for_evaluation(unwrapped, config)
                last_metrics = evaluate_examples(
                    unwrapped,
                    processor,
                    final_eval_examples,
                    config,
                    eval_dir / "last_checkpoint",
                    checkpoint_name="last_checkpoint",
                    sample_prediction_mode=sample_prediction_mode,
                )
            if selection_enabled and best_epoch != completed_epochs:
                unwrapped.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                best_model = load_model_for_inference(model_name_or_path, best_dir, config)
                best_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_model.to(best_device)
                best_processor = load_processor(best_dir, config)
                LOGGER.info("Starting final held-out evaluation for best_checkpoint")
                best_metrics = evaluate_examples(
                    best_model,
                    best_processor,
                    final_eval_examples,
                    config,
                    eval_dir / "best_checkpoint",
                    checkpoint_name="best_checkpoint",
                    sample_prediction_mode=sample_prediction_mode,
                )
                selected_best_model = best_model
                selected_best_processor = best_processor
            else:
                selected_best_model = unwrapped
                selected_best_processor = processor
                if evaluate_last_checkpoint and last_metrics is not None:
                    best_metrics = last_metrics
                    if not (eval_dir / "best_checkpoint").exists():
                        shutil.copytree(eval_dir / "last_checkpoint", eval_dir / "best_checkpoint")
                else:
                    LOGGER.info("Starting the only outer-holdout evaluation for selected best_checkpoint")
                    prepare_model_for_evaluation(unwrapped, config)
                    best_metrics = evaluate_examples(
                        unwrapped,
                        processor,
                        final_eval_examples,
                        config,
                        eval_dir / "best_checkpoint",
                        checkpoint_name="best_checkpoint",
                        sample_prediction_mode=sample_prediction_mode,
                    )

            final_checkpoint_payload = {
                "split_mode": split_mode,
                "prediction_backend": sample_prediction_mode,
                "aggregation_level": aggregation_level,
                "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
                "evaluate_last_checkpoint": evaluate_last_checkpoint,
                "selected_best_checkpoint": {
                    "epoch": best_epoch,
                    "active_backend_metrics": best_metrics["backend_results"][sample_prediction_mode]["headline_metrics"],
                },
            }
            if last_metrics is not None:
                final_checkpoint_payload["last_checkpoint"] = {
                    "epoch": completed_epochs,
                    "active_backend_metrics": last_metrics["backend_results"][sample_prediction_mode]["headline_metrics"],
                }
            save_json(final_checkpoint_payload, eval_dir / "best_vs_last_checkpoint_metrics.json")
            joint_final_eval_payload: dict[str, Any] = {}
            for group in joint_final_eval_groups:
                group_eval_dir = eval_dir / "best_checkpoint" / f"{str(group['dataset']).lower()}_fold_{group['fold']}_holdout"
                LOGGER.info(
                    "Starting joint final held-out evaluation dataset=%s fold=%s | aggregation_level=%s | samples=%s subjects=%s",
                    group["dataset"],
                    group["fold"],
                    resolve_aggregation_level(group["config"]),
                    len(group["examples"]),
                    len(group["subject_ids"]),
                )
                group_metrics = evaluate_examples(
                    selected_best_model,
                    selected_best_processor,
                    group["examples"],
                    group["config"],
                    group_eval_dir,
                    checkpoint_name="best_checkpoint",
                    sample_prediction_mode=sample_prediction_mode,
                )
                joint_final_eval_payload[f"{str(group['dataset']).lower()}_fold_{group['fold']}_holdout"] = {
                    "dataset": group["dataset"],
                    "fold": int(group["fold"]),
                    "split_name": group["split_name"],
                    "aggregation_level": resolve_aggregation_level(group["config"]),
                    "active_backend_metrics": group_metrics["backend_results"][sample_prediction_mode]["headline_metrics"],
                    "eval_dir": str(group_eval_dir),
                }
            if joint_final_eval_payload:
                save_json(joint_final_eval_payload, eval_dir / "joint_final_eval_metrics.json")
        else:
            LOGGER.info(
                "Skipping final held-out evaluation inside training to avoid multi-GPU NCCL timeout. "
                "Run scripts/run_eval_slurm.sh separately on best_model if you need a standalone held-out evaluation job."
            )
        if tracking_context is not None:
            _finalize_tracking_artifacts(args, tracking_context, run_root, run_config)
        _write_trial_result(
            args.trial_result_file,
            {
                "status": "completed",
                "split_mode": split_mode,
                "cv_protocol": partition_plan["cv_protocol"],
                "fold": int(args.fold),
                "run_name": args.run_name,
                "run_root": str(run_root),
                "save_strategy": args.save_strategy,
                "metric_name": selection_metric_cfg["metric"] if selection_enabled else None,
                "best_metric": None if best_metric is None else float(best_metric),
                "best_epoch": int(best_epoch),
                "completed_epochs": int(completed_epochs),
                "history_path": str(logs_dir / "training_history.json"),
                "best_model_dir": str(best_dir) if best_dir.exists() else None,
                "last_model_dir": str(last_dir) if last_dir.exists() else None,
                "stopped_early": bool(stopped_early),
                "stop_epoch": int(stop_epoch) if stop_epoch is not None else None,
                "stop_reason": stop_reason,
                "config_overrides": list(args.config_overrides),
                "sample_prediction_mode": sample_prediction_mode,
                "aggregation_level": aggregation_level,
                "input_modality": input_modality,
                "audio_adapter": audio_adapter_cfg,
                "lora_resolution": lora_layer_selection,
                "selection_protocol": run_config["selection_protocol"],
                "final_eval_protocol": run_config["final_eval_protocol"],
                "history": history,
            },
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
