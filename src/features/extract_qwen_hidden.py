from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
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
    build_examples,
    filter_rows_by_subjects,
    load_manifest_rows,
    render_joint_packed30_bundle,
)
from src.daic_chunking import build_joint_epoch_schedule
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator, load_prompt_audio
from src.model.runtime import load_model_for_inference, load_processor, resolve_processor_sampling_rate
from src.utils import read_json, save_json, sha256_file, sha256_jsonl_rows, sha256_text, write_jsonl


SUPPORTED_HIDDEN_SIZES = {3584, 4096}
POOLING_NAME = "last_valid_prompt_token"
CONDITION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
CACHE_SCHEMA_VERSION = "qwen_hidden_cache.v2"
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


def _decoder_hidden_size(model) -> int:
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
            if hidden_size not in SUPPORTED_HIDDEN_SIZES:
                raise ValueError(
                    f"Unexpected decoder hidden size {hidden_size}; expected one of {sorted(SUPPORTED_HIDDEN_SIZES)}."
                )
            return hidden_size
    raise ValueError("Could not resolve the decoder hidden size from the loaded model configuration.")


def _is_joint_packed30_recipe(config: dict[str, Any]) -> bool:
    return (
        str(config.get("dataset", "")).lower() == "daic"
        and str(config.get("data", {}).get("sample_mode", "")).strip().lower()
        == JOINT_PACKED30_MODE
        and str(config.get("data", {}).get("train_chunk_policy", "")).strip().lower()
        == "joint_random_k"
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
        is_daic_chunking = (
            str(config.get("dataset", "")).lower() == "daic"
            and str(
                config.get("evaluation", {}).get(
                    "subject_score_aggregation", ""
                )
            ).lower()
            == "mean_score"
        )
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
    collator = PromptOnlyExtractionCollator(processor)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    mask_sources: dict[str, int] = {}
    determinism_max_abs_diff: float | None = None
    selected = examples[:max_examples] if max_examples else examples
    seen_samples: set[str] = set()
    subject_labels: dict[str, int] = {}
    for index, raw_example in enumerate(selected, start=1):
        example = load_prompt_audio(raw_example, sampling_rate, bool(config["data"].get("silence_audio", False)))
        model_inputs, metadata_rows = collator([example])
        metadata = metadata_rows[0]
        prompt_text = metadata.pop("prompt_text")
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        if "labels" in model_inputs:
            raise AssertionError("Gold labels must never be passed to Qwen during extraction.")
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
        metadata.update({
            "prompt_sha256": sha256_text(prompt_text),
            "hidden_layer": "final",
            "pooling": POOLING_NAME,
            "vector_dimension": expected_hidden_size,
            "vector_dtype": "float32",
            "mask_source": mask_source,
            "checkpoint": str(checkpoint_dir),
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
    return {
        "rows": len(rows),
        "subjects": len(subject_labels),
        "mask_sources": mask_sources,
        "determinism_rtol": 1e-5,
        "determinism_atol": 1e-5,
        "determinism_max_abs_diff": determinism_max_abs_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract prompt-only final Qwen2 hidden vectors.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--manifest-path", type=Path, help="Relocated copy of the saved manifest; hash must match.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--condition", help="Unique experiment condition used in metadata and output grouping.")
    parser.add_argument("--emotion-source", help="Predeclared source label for an emotion cache.")
    parser.add_argument("--emotion-language", help="Predeclared language label for emotion captions.")
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
        saved, config, partition_subject_ids, fold, train_source_count
    )
    manifest_path = args.manifest_path.resolve() if args.manifest_path else _saved_path(saved["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest is unavailable: {manifest_path}")
    manifest_rows = load_manifest_rows(manifest_path)
    canonical_manifest_hash = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and canonical_manifest_hash != saved["manifest_hash"]:
        raise ValueError("Current manifest hash does not match the checkpoint's saved manifest hash.")
    condition = resolve_condition(args.condition, saved.get("input_modality"), use_emotion(config))
    cache_config = {
        "schema_version": CACHE_SCHEMA_VERSION,
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
    }
    cache_config_sha256 = sha256_text(
        json.dumps(cache_config, sort_keys=True, separators=(",", ":"))
    )
    if output_dir.exists() and any(output_dir.iterdir()):
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
        print(
            json.dumps(
                {
                    "status": "skipped_compatible_complete_cache",
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
    examples, joint_head_fit_provenance = _partition_examples(
        manifest_rows, config, partition_subject_ids, fold, checkpoint_dir=checkpoint_dir
    )
    model_name = args.model_name_or_path or saved.get("resolved_model_name_or_path") or config["model_name_or_path"]
    processor = load_processor(checkpoint_dir, config)
    model = load_model_for_inference(str(model_name), checkpoint_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and bool(config.get("training", {}).get("bf16", False)) else None
    model.to(device=device, dtype=dtype)
    model.eval()
    expected_hidden_size = _decoder_hidden_size(model)
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
    save_json(metadata, output_dir / "extraction_metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
