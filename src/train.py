from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.build_manifest import build_for_config
from src.data.runtime import (
    AudioTextDataset,
    build_examples,
    build_subject_label_map,
    filter_rows_by_subjects,
    load_manifest_rows,
    save_partition_subjects,
)
from src.data.split_utils import (
    SPLIT_MODE_CV,
    SPLIT_MODE_FIXED,
    SPLIT_MODE_FULL_TRAIN,
    deterministic_inner_split,
    read_fold_payload,
    resolve_dev_pool_partitions,
    resolve_requested_split_mode,
    resolve_split_mode,
    subject_ids_for_partitions,
)
from src.evaluate import evaluate_examples
from src.model.collator import Qwen2AudioSFTCollator
from src.model.qwen2audio_lora import (
    load_model_for_inference,
    load_model_for_training,
    load_processor,
    prepare_model_for_evaluation,
    resolved_lora_layer_selection,
    resolve_audio_adapter_config,
    save_adapter_and_processor,
)
from src.utils import (
    configure_logging,
    ensure_dir,
    evaluation_protocol_name,
    get_logger,
    load_yaml_with_overrides,
    log_resolved_config,
    read_json,
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


LOGGER = get_logger(__name__)


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
    if split_mode == SPLIT_MODE_FULL_TRAIN:
        return {
            "split_mode": split_mode,
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

    selection_partition = str(config["split"].get("selection_partition", "")).strip()
    if split_mode == SPLIT_MODE_FIXED and selection_partition:
        selection_subject_ids = outer_partitions.get("selection_subject_ids") or []
        if not selection_subject_ids:
            raise ValueError(
                "split.selection_partition was configured, but no subject ids were resolved for that partition."
            )
        return {
            "split_mode": split_mode,
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


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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


def _selection_metric_values(metric_value: float, headline_metrics: dict[str, Any], selection_loss: float) -> dict[str, float]:
    values = {
        "selection_positive_f1": metric_value,
        "selection_macro_f1": float(headline_metrics["macro_f1"]),
        "selection_accuracy": float(headline_metrics["accuracy"]),
        "selection_precision": float(headline_metrics["precision"]),
        "selection_recall": float(headline_metrics["recall"]),
        "selection_loss": selection_loss,
    }
    values.update(
        {
            "inner_val_positive_f1": values["selection_positive_f1"],
            "inner_val_macro_f1": values["selection_macro_f1"],
            "inner_val_accuracy": values["selection_accuracy"],
            "inner_val_precision": values["selection_precision"],
            "inner_val_recall": values["selection_recall"],
            "inner_val_loss": values["selection_loss"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe Qwen2-Audio depression detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--run_name", default="reproduction")
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
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml_with_overrides(args.config, args.config_overrides)
    log_resolved_config(
        LOGGER,
        base_config_path=args.config,
        config_overrides=args.config_overrides,
        resolved_config=config,
    )
    set_seed(int(config["seed"]))
    audio_adapter_cfg = resolve_audio_adapter_config(config)
    sample_prediction_mode = resolve_prediction_mode(config)
    aggregation_level = resolve_aggregation_level(config)
    metadata = _load_metadata_or_build(args.config, config, args.config_overrides)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    subject_labels = build_subject_label_map(manifest_rows)
    partition_plan = _resolve_training_subject_splits(config, metadata, subject_labels, args.fold)
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
    processor = load_processor(model_name_or_path)
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

    train_dataset = AudioTextDataset(
        train_examples,
        processor_sampling_rate=processor.feature_extractor.sampling_rate,
        silence_audio=bool(config["data"].get("silence_audio", False)),
    )
    selection_dataset = None
    if selection_enabled:
        selection_dataset = AudioTextDataset(
            selection_examples,
            processor_sampling_rate=processor.feature_extractor.sampling_rate,
            silence_audio=bool(config["data"].get("silence_audio", False)),
        )
    collator = Qwen2AudioSFTCollator(processor=processor, debug=args.label_mask_debug)
    if args.label_mask_debug:
        _emit_label_mask_debug(train_dataset, collator, processor, logs_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["per_device_train_batch_size"]),
        shuffle=True,
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

    model = load_model_for_training(model_name_or_path, config)
    lora_layer_selection = resolved_lora_layer_selection(model)
    optimizer = AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = max(
        1,
        math.ceil(len(train_loader) / int(config["training"]["gradient_accumulation_steps"])) * int(config["training"]["num_train_epochs"]),
    )
    warmup_steps = int(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(config["training"].get("bf16", False)) else "no",
        kwargs_handlers=[ddp_kwargs],
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    run_config = {
        "config": config,
        "base_config_path": str(Path(args.config)),
        "config_overrides": list(args.config_overrides),
        "evaluation": {
            "sample_prediction_mode": sample_prediction_mode,
            "aggregation_level": aggregation_level,
            "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
        },
        "audio_adapter": audio_adapter_cfg,
        "lora_resolution": lora_layer_selection,
        "resolved_model_name_or_path": model_name_or_path,
        "manifest_path": metadata["manifest_path"],
        "manifest_hash": metadata["manifest_hash"],
        "split_metadata_path": active_split_metadata_path,
        "split_metadata_hash": sha256_file(active_split_metadata_path),
        "split_mode": split_mode,
        "fold": int(args.fold),
        "save_strategy": args.save_strategy,
        "selection_protocol": {
            "mode": partition_plan["selection_mode"],
            "metric_name": "selection_positive_f1" if selection_enabled else None,
            "selection_split_name": partition_plan["selection_split_name"],
            "selection_subject_count": len(partition_plan["selection_subject_ids"]),
            "selection_sample_count": len(selection_examples),
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
            "run_final_eval_in_train": bool(config["training"].get("run_final_eval_in_train", False)),
        },
    }
    if accelerator.is_main_process:
        save_yaml(run_config, run_root / "run_config.yaml")

    best_metric: float | None = float("-inf") if selection_enabled else None
    best_epoch = -1 if selection_enabled else 0
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
    for epoch in range(1, int(config["training"]["num_train_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_losses.append(float(loss.detach().item()))
            if accelerator.is_main_process and step % int(config["training"]["logging_steps"]) == 0:
                LOGGER.info("epoch=%s step=%s loss=%.6f", epoch, step, sum(epoch_losses) / len(epoch_losses))

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if selection_enabled:
                selection_eval_dir = ensure_dir(logs_dir / f"{partition_plan['selection_log_dir_name']}_epoch_{epoch}")
                selection_loss = _compute_dataset_loss(unwrapped, selection_loss_loader)
                LOGGER.info(
                    "Selection evaluation split=%s | backend=%s | aggregation_level=%s | protocol=%s",
                    partition_plan["selection_split_name"],
                    sample_prediction_mode,
                    aggregation_level,
                    evaluation_protocol_name(sample_prediction_mode),
                )
                metrics = evaluate_examples(
                    unwrapped,
                    processor,
                    selection_examples,
                    config,
                    selection_eval_dir,
                    checkpoint_name=f"epoch_{epoch}",
                    sample_prediction_mode=sample_prediction_mode,
                )
                headline_metrics = metrics["backend_results"][sample_prediction_mode]["headline_metrics"]
                metric_value = float(headline_metrics["positive_f1"])
                metric_values = _selection_metric_values(metric_value, headline_metrics, selection_loss)
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
                    **metric_values,
                }
                history.append(history_row)
                LOGGER.info(
                    "Selection epoch=%s | split=%s | aggregation_level=%s | loss=%.6f ACC=%.6f F1=%.6f Precision=%.6f Recall=%.6f",
                    epoch,
                    partition_plan["selection_split_name"],
                    aggregation_level,
                    selection_loss,
                    float(headline_metrics["accuracy"]),
                    float(headline_metrics["positive_f1"]),
                    float(headline_metrics["precision"]),
                    float(headline_metrics["recall"]),
                )
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_epoch = epoch
                    if _save_best_checkpoint(args.save_strategy):
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        save_adapter_and_processor(unwrapped, processor, best_dir, config=config)
                _write_trial_progress(
                    args.trial_progress_file,
                    epoch=epoch,
                    metric_name="selection_positive_f1",
                    metric_value=metric_value,
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

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        completed_epochs = len(history)
        save_json(history, logs_dir / "training_history.json")
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
            LOGGER.info("Starting final held-out evaluation for last_checkpoint")
            prepare_model_for_evaluation(unwrapped)
            last_metrics = evaluate_examples(
                unwrapped,
                processor,
                final_eval_examples,
                config,
                eval_dir / "last_checkpoint",
                checkpoint_name="last_checkpoint",
                sample_prediction_mode=sample_prediction_mode,
            )
            if not selection_enabled or best_epoch == completed_epochs:
                best_metrics = last_metrics
                if not (eval_dir / "best_checkpoint").exists():
                    shutil.copytree(eval_dir / "last_checkpoint", eval_dir / "best_checkpoint")
            else:
                unwrapped.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                best_model = load_model_for_inference(model_name_or_path, best_dir)
                best_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_model.to(best_device)
                best_processor = load_processor(best_dir)
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

            save_json(
                {
                    "split_mode": split_mode,
                    "prediction_backend": sample_prediction_mode,
                    "aggregation_level": aggregation_level,
                    "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
                    "selected_best_checkpoint": {
                        "epoch": best_epoch,
                        "active_backend_metrics": best_metrics["backend_results"][sample_prediction_mode]["headline_metrics"],
                    },
                    "last_checkpoint": {
                        "epoch": completed_epochs,
                        "active_backend_metrics": last_metrics["backend_results"][sample_prediction_mode]["headline_metrics"],
                    },
                },
                eval_dir / "best_vs_last_checkpoint_metrics.json",
            )
        else:
            LOGGER.info(
                "Skipping final held-out evaluation inside training to avoid multi-GPU NCCL timeout. "
                "Run scripts/run_eval_slurm.sh separately on best_model and last_model."
            )
        _write_trial_result(
            args.trial_result_file,
            {
                "status": "completed",
                "split_mode": split_mode,
                "fold": int(args.fold),
                "run_name": args.run_name,
                "run_root": str(run_root),
                "save_strategy": args.save_strategy,
                "metric_name": "selection_positive_f1" if selection_enabled else None,
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
