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
from accelerate import Accelerator
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
from src.data.split_utils import deterministic_inner_split
from src.evaluate import evaluate_examples
from src.model.collator import Qwen2AudioSFTCollator
from src.model.qwen2audio_lora import (
    load_model_for_inference,
    load_model_for_training,
    load_processor,
    save_adapter_and_processor,
)
from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    load_yaml,
    read_json,
    resolve_metadata_paths,
    resolve_model_name_or_path,
    resolve_project_path,
    save_json,
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


def _load_metadata_or_build(config_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = resolve_project_path(Path(config["output_dirs"]["split_dir"]) / f"{config['dataset']}_manifest_metadata.json")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if metadata_path.exists():
        metadata = resolve_metadata_paths(read_json(metadata_path))
        usable, reason = _metadata_artifacts_are_usable(metadata)
        if usable:
            return metadata
        LOGGER.warning("Refreshing stale metadata for %s: %s", config["dataset"], reason)
    if local_rank == 0:
        build_for_config(config_path)
    return _wait_for_usable_metadata(metadata_path)


def _resolve_outer_partitions(config: dict[str, Any], metadata: dict[str, Any], fold: int) -> dict[str, list[str]]:
    dataset_name = str(config["dataset"]).lower()
    if dataset_name == "daic":
        partition_rows = read_json(metadata["subject_partition_path"])
        train_partition = str(config["split"]["train_partition"])
        final_eval_partition = str(config["split"]["final_eval_partition"])
        outer_train_subject_ids = sorted([row["subject_id"] for row in partition_rows if row["partition"] == train_partition])
        final_eval_subject_ids = sorted([row["subject_id"] for row in partition_rows if row["partition"] == final_eval_partition])
        return {
            "outer_train_subject_ids": outer_train_subject_ids,
            "final_eval_subject_ids": final_eval_subject_ids,
        }
    folds = read_json(metadata["folds_path"])
    fold_payload = folds[str(fold)] if str(fold) in folds else folds[fold]
    return {
        "outer_train_subject_ids": sorted(fold_payload["outer_train_subject_ids"]),
        "final_eval_subject_ids": sorted(fold_payload["final_eval_subject_ids"]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe Qwen2-Audio depression detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--run_name", default="reproduction")
    parser.add_argument("--label_mask_debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    metadata = _load_metadata_or_build(args.config, config)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    subject_labels = build_subject_label_map(manifest_rows)
    outer_partitions = _resolve_outer_partitions(config, metadata, args.fold)
    inner_split = deterministic_inner_split(
        subject_labels,
        outer_partitions["outer_train_subject_ids"],
        seed=int(config["split"]["seed"]) + int(args.fold),
        val_ratio=float(config["split"]["inner_val_ratio"]),
    )

    run_root = Path(config["output_dirs"]["run_root"]) / args.run_name / f"fold_{args.fold}"
    logs_dir = ensure_dir(run_root / "logs")
    eval_dir = ensure_dir(run_root / "eval")
    best_dir = ensure_dir(run_root / "best_model")
    last_dir = ensure_dir(run_root / "last_model")

    split_payload = save_partition_subjects(
        logs_dir / "split_used.json",
        train_inner_subject_ids=inner_split["train_inner_subject_ids"],
        val_inner_subject_ids=inner_split["val_inner_subject_ids"],
        final_eval_subject_ids=outer_partitions["final_eval_subject_ids"],
        subject_labels=subject_labels,
    )
    _print_partition_counts(split_payload)

    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, config)
    processor = load_processor(model_name_or_path)
    train_examples = build_examples(
        filter_rows_by_subjects(manifest_rows, inner_split["train_inner_subject_ids"]),
        config,
        partition_name="train_inner",
        truncation_log_path=logs_dir / "train_truncation.jsonl",
    )
    val_examples = build_examples(
        filter_rows_by_subjects(manifest_rows, inner_split["val_inner_subject_ids"]),
        config,
        partition_name="val_inner",
        truncation_log_path=logs_dir / "val_truncation.jsonl",
    )
    final_eval_examples = build_examples(
        filter_rows_by_subjects(manifest_rows, outer_partitions["final_eval_subject_ids"]),
        config,
        partition_name="final_eval",
        truncation_log_path=logs_dir / "final_eval_truncation.jsonl",
    )
    sample_count_payload = {
        "train_inner": _sample_partition_counts(train_examples),
        "val_inner": _sample_partition_counts(val_examples),
        "final_eval": _sample_partition_counts(final_eval_examples),
    }
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

    model = load_model_for_training(model_name_or_path, config)
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
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(config["training"].get("bf16", False)) else "no",
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    run_config = {
        "config": config,
        "resolved_model_name_or_path": model_name_or_path,
        "manifest_path": metadata["manifest_path"],
        "manifest_hash": metadata["manifest_hash"],
        "split_metadata_path": metadata.get("folds_path") or metadata.get("subject_partition_path"),
        "split_metadata_hash": sha256_file(metadata.get("folds_path") or metadata.get("subject_partition_path")),
        "fold": int(args.fold),
    }
    if accelerator.is_main_process:
        save_yaml(run_config, run_root / "run_config.yaml")

    best_metric = float("-inf")
    best_epoch = -1
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
            inner_eval_dir = ensure_dir(logs_dir / f"inner_val_epoch_{epoch}")
            metrics = evaluate_examples(
                unwrapped,
                processor,
                val_examples,
                config,
                inner_eval_dir,
                checkpoint_name=f"epoch_{epoch}",
                run_generation=False,
            )
            metric_value = float(metrics["likelihood"]["subject_metrics"]["positive_f1"])
            history_row = {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                "inner_val_likelihood_positive_f1": metric_value,
                "inner_val_macro_f1": float(metrics["likelihood"]["subject_metrics"]["macro_f1"]),
                "inner_val_accuracy": float(metrics["likelihood"]["subject_metrics"]["accuracy"]),
            }
            history.append(history_row)
            if metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                save_adapter_and_processor(unwrapped, processor, best_dir)
            LOGGER.info("Finished epoch=%s | best_epoch=%s best_metric=%.6f", epoch, best_epoch, best_metric)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if last_dir.exists():
            shutil.rmtree(last_dir)
        save_adapter_and_processor(unwrapped, processor, last_dir)
        save_json(history, logs_dir / "training_history.json")

        last_metrics = evaluate_examples(unwrapped, processor, final_eval_examples, config, eval_dir / "last_checkpoint", checkpoint_name="last_checkpoint")
        if best_epoch == int(config["training"]["num_train_epochs"]):
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
            best_metrics = evaluate_examples(
                best_model,
                best_processor,
                final_eval_examples,
                config,
                eval_dir / "best_checkpoint",
                checkpoint_name="best_checkpoint",
            )

        save_json(
            {
                "selected_best_checkpoint": {
                    "epoch": best_epoch,
                    "likelihood_metrics": best_metrics["likelihood"]["subject_metrics"],
                    "generation_metrics": best_metrics["generation"]["subject_metrics"],
                },
                "last_checkpoint": {
                    "epoch": int(config["training"]["num_train_epochs"]),
                    "likelihood_metrics": last_metrics["likelihood"]["subject_metrics"],
                    "generation_metrics": last_metrics["generation"]["subject_metrics"],
                },
            },
            eval_dir / "best_vs_last_checkpoint_metrics.json",
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
