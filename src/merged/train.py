from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data.runtime import AudioTextDataset
from src.evaluate import evaluate_examples
from src.merged.protocol import (
    DATASETS,
    build_dataset_aware_schedule,
    compute_hierarchical_example_weights,
)
from src.merged.configuration import model_config
from src.merged.provenance import write_slurm_provenance
from src.merged.runtime import (
    limit_grouped_subjects,
    load_merged_config,
    load_records_and_protocol,
    make_final_partitions,
    make_fold_partitions,
)
from src.model.collator import Qwen2AudioSFTCollator
from src.model.runtime import (
    load_model_for_training,
    load_processor,
    resolve_processor_sampling_rate,
    restore_model_for_training,
    save_adapter_and_processor,
)
from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    resolve_model_name_or_path,
    resolve_project_path,
    save_json,
    set_seed,
    sha256_file,
)


LOGGER = get_logger(__name__)


_model_config = model_config


def _limit_subjects_per_class(
    examples: list[dict[str, Any]], *, subjects_per_class: int, seed: int
) -> tuple[list[dict[str, Any]], list[str]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, int] = {}
    for example in examples:
        subject = str(example["subject_id"])
        by_subject.setdefault(subject, []).append(example)
        labels[subject] = int(example["label"])
    selected: list[str] = []
    for label in (0, 1):
        candidates = sorted(subject for subject, value in labels.items() if value == label)
        # The protocol uses deterministic lexicographic selection after the
        # split has already been seeded. The seed is recorded for an explicit
        # smoke identity without changing the component fold assignment.
        selected.extend(candidates[: int(subjects_per_class)])
    selected_set = set(selected)
    return [example for example in examples if str(example["subject_id"]) in selected_set], sorted(selected)


def _component_examples(partitions: dict[str, Any], partition: str) -> dict[str, list[dict[str, Any]]]:
    return partitions["examples"][partition]


def _write_composition(
    path: Path,
    *,
    stage: str,
    fold: int,
    train_examples: list[dict[str, Any]],
    selection_examples: dict[str, list[dict[str, Any]]],
    outer_train_subjects: dict[str, list[str]],
    selection_subjects: dict[str, list[str]],
    holdout_subjects: dict[str, list[str]],
    weighting_audit: dict[str, Any],
    smoke_subject_ids: list[str] | None,
) -> None:
    save_json(
        {
            "schema_version": "symmetric_merged_composition.v1",
            "stage": stage,
            "fold": int(fold),
            "datasets": list(DATASETS),
            "train_example_count": len(train_examples),
            "train_dataset_counts": dict(sorted(Counter(str(row["dataset"]) for row in train_examples).items())),
            "train_subject_counts": {
                dataset: len(values) for dataset, values in sorted(outer_train_subjects.items())
            },
            "qwen_selection_subject_counts": {
                dataset: len(values) for dataset, values in sorted(selection_subjects.items())
            },
            "outer_holdout_subject_counts": {
                dataset: len(values) for dataset, values in sorted(holdout_subjects.items())
            },
            "selection_example_counts": {
                dataset: len(values) for dataset, values in sorted(selection_examples.items())
            },
            "smoke_subject_ids": smoke_subject_ids,
            "weighting_audit": weighting_audit,
            "exhaustive_training": {
                "every_eligible_example_once_per_epoch": True,
                "oversampling": False,
                "undersampling": False,
                "duplication": False,
                "class_rebalancing": False,
            },
        },
        path,
    )


def train_merged_fold(
    config_path: str | Path,
    *,
    stage: str,
    fold: int,
    run_id: str,
    epochs_override: int | None = None,
    subjects_per_class: int | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(f"Unsupported merged training stage: {stage}")
    merged_config = load_merged_config(config_path)
    records, protocol = load_records_and_protocol(merged_config)
    model_config = _model_config(merged_config, records)
    resolved_config_path = resolve_project_path(config_path)
    set_seed(int(merged_config.get("seed", 1337)), deterministic=True)

    if stage == "final":
        partitions = make_final_partitions(records)
        train_examples = list(partitions["flat_examples"]["train"])
        selection_examples: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASETS}
        outer_train_subjects = {
            dataset: list(partitions["subjects"][dataset]) for dataset in DATASETS
        }
        selection_subjects = {dataset: [] for dataset in DATASETS}
        holdout_subjects = {dataset: list(partitions["subjects"].get("daic_official_test", [])) if dataset == "daic" else [] for dataset in DATASETS}
        resolved_epochs = int(epochs_override or merged_config["training"].get("final_epoch_count", 0))
        if resolved_epochs <= 0:
            raise ValueError("Final training requires --epochs or training.final_epoch_count from the median CV selection.")
    else:
        partitions = make_fold_partitions(records, protocol, fold)
        train_examples = list(partitions["flat_examples"]["qwen_train"])
        selection_examples = _component_examples(partitions, "inner_val")
        outer_train_subjects = partitions["subjects"]["outer_train"]
        selection_subjects = partitions["subjects"]["inner_val"]
        holdout_subjects = partitions["subjects"]["outer_holdout"]
        resolved_epochs = int(epochs_override or merged_config["training"].get("num_train_epochs", 20))
        if resolved_epochs > 20:
            raise ValueError("The merged Qwen protocol caps training at 20 epochs.")

    smoke_subject_ids: list[str] | None = None
    if subjects_per_class is not None:
        train_examples, selected_train = _limit_subjects_per_class(
            train_examples, subjects_per_class=int(subjects_per_class), seed=int(merged_config.get("seed", 1337))
        )
        smoke_subject_ids = list(selected_train)
        selection_examples, selected = limit_grouped_subjects(
            selection_examples,
            subjects_per_class=int(subjects_per_class),
        )
        smoke_subject_ids.extend(selected)
    weighted_examples, weighting_audit = compute_hierarchical_example_weights(
        train_examples, expected_datasets=DATASETS
    )

    run_root = Path(merged_config["output_dirs"]["run_root"]) / run_id / stage / f"fold_{int(fold)}"
    logs_dir = ensure_dir(run_root / "logs")
    best_dir = run_root / "best_model"
    complete_path = run_root / "training_complete.json"
    identity = {
        "schema_version": "symmetric_merged_training_identity.v1",
        "config_name": merged_config.get("name"),
        "stage": stage,
        "fold": int(fold),
        "run_id": run_id,
        "protocol_split_hash": protocol.get("protocol", {}).get("split_hash"),
        "manifest_hash": protocol.get("manifest", {}).get("manifest_hash"),
        "merged_config_sha256": sha256_file(resolved_config_path),
        "fold_hash": protocol.get("protocol", {}).get("folds", {}).get(str(int(fold)), {}).get("fold_hash"),
        "epochs": int(resolved_epochs),
        "subjects_per_class": subjects_per_class,
    }
    if complete_path.is_file() and best_dir.is_dir():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"Incompatible completed merged training output: {run_root}")
        return {"status": "skipped_compatible_complete", "run_root": str(run_root), **existing}
    if run_root.exists() and any(run_root.iterdir()) and not complete_path.is_file():
        raise ValueError(f"Refusing to overwrite an incomplete merged training output: {run_root}")
    ensure_dir(run_root)
    save_json(identity, run_root / "training_identity.json")
    write_slurm_provenance(
        run_root / "slurm_provenance.json",
        worker="src.merged.train",
        stage=stage,
        fold=int(fold),
        run_id=run_id,
        config_name=merged_config.get("name"),
        modality=merged_config.get("modality"),
        protocol_split_hash=protocol["protocol"]["split_hash"],
    )
    _write_composition(
        logs_dir / "composition.json",
        stage=stage,
        fold=fold,
        train_examples=weighted_examples,
        selection_examples=selection_examples,
        outer_train_subjects=outer_train_subjects,
        selection_subjects=selection_subjects,
        holdout_subjects=holdout_subjects,
        weighting_audit=weighting_audit,
        smoke_subject_ids=smoke_subject_ids,
    )
    save_json(weighting_audit, logs_dir / "weighting_audit.json")

    model_name = str(resolve_model_name_or_path(None, model_config))
    processor = load_processor(model_name, model_config)
    sampling_rate = resolve_processor_sampling_rate(processor)
    train_dataset = AudioTextDataset(
        weighted_examples,
        processor_sampling_rate=sampling_rate,
        silence_audio=bool(model_config.get("data", {}).get("silence_audio", False)),
        chunk_sampling="deterministic",
    )
    collator = Qwen2AudioSFTCollator(processor=processor, debug=False)
    model = load_model_for_training(model_name, model_config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(model_config["training"].get("learning_rate", 2.0e-4)),
        weight_decay=float(model_config["training"].get("weight_decay", 0.0)),
    )
    accumulation_steps = int(model_config["training"].get("gradient_accumulation_steps", 32))
    schedules = [
        build_dataset_aware_schedule(
            weighted_examples,
            seed=int(merged_config.get("seed", 1337)),
            epoch=epoch,
            accumulation_steps=accumulation_steps,
        )
        for epoch in range(1, resolved_epochs + 1)
    ]
    save_json(
        {"epochs": [schedule["audit"] for schedule in schedules]},
        logs_dir / "schedule_audit.json",
    )
    total_steps = sum(len(schedule["blocks"]) for schedule in schedules)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(model_config["training"].get("warmup_ratio", 0.03))),
        num_training_steps=max(1, total_steps),
    )
    accelerator = Accelerator(
        mixed_precision="bf16" if bool(model_config["training"].get("bf16", False)) else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    bad_epochs = 0
    patience = int(merged_config.get("training", {}).get("early_stopping", {}).get("patience", 3))
    use_selection = stage != "final"
    for epoch_index, schedule in enumerate(schedules, start=1):
        model.train()
        restore_model_for_training(accelerator.unwrap_model(model), model_config)
        block_rows: list[dict[str, Any]] = []
        for block in schedule["blocks"]:
            optimizer.zero_grad()
            block_indices = list(block["example_indices"])
            global_weight = float(block["example_weight_total"])
            process_index = int(accelerator.process_index)
            process_count = int(accelerator.num_processes)
            actual_local_indices = block_indices[process_index::process_count]
            # Every rank must execute the same number of DDP backward calls.
            # Pad short tail ranks with a zero-weight dummy; it is not part of
            # the schedule and is never logged as an eligible example.
            max_local_count = (len(block_indices) + process_count - 1) // process_count
            local_items: list[tuple[int, bool]] = [
                (example_index, False) for example_index in actual_local_indices
            ]
            while len(local_items) < max_local_count:
                local_items.append((block_indices[0], True))
            local_loss_numerator = 0.0
            local_loss_denominator = 0.0
            for local_position, (example_index, dummy) in enumerate(local_items):
                item = train_dataset[example_index]
                batch = collator([item])
                device = accelerator.device
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                batch.pop("loss_weight", None)
                outputs = model(**batch)
                weight = 0.0 if dummy else float(weighted_examples[example_index]["loss_weight"])
                local_loss_numerator += float(outputs.loss.detach().item()) * weight
                local_loss_denominator += weight
                scale = weight / global_weight * process_count
                context = (
                    accelerator.no_sync(model)
                    if local_position < len(local_items) - 1
                    else torch.enable_grad()
                )
                with context:
                    accelerator.backward(outputs.loss * float(scale))
            loss_stats = torch.tensor(
                [local_loss_numerator, local_loss_denominator],
                dtype=torch.float64,
                device=accelerator.device,
            )
            gathered_loss_stats = accelerator.gather(loss_stats.reshape(1, 2))
            global_loss_denominator = float(gathered_loss_stats[:, 1].sum().item())
            global_loss_numerator = float(gathered_loss_stats[:, 0].sum().item())
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(model_config["training"].get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            block_rows.append({
                "block_index": int(block["block_index"]),
                "example_count": len(block_indices),
                "dataset_weight_contributions": block["dataset_weight_contributions"],
                "normalized_loss_denominator": global_weight,
                "normalized_loss": global_loss_numerator / global_loss_denominator if global_loss_denominator else 0.0,
            })
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            row: dict[str, Any] = {
                "epoch": epoch_index,
                "train_loss": float(sum(item["normalized_loss"] for item in block_rows) / max(1, len(block_rows))),
                "realized_dataset_contributions": schedule["audit"]["realized_dataset_weight_contributions"],
                "schedule_hash": schedule["audit"]["schedule_hash"],
            }
            if use_selection:
                component_metrics: dict[str, Any] = {}
                selection_values: list[float] = []
                for dataset in DATASETS:
                    eval_dir = ensure_dir(logs_dir / "selection" / f"epoch_{epoch_index}" / dataset)
                    metrics = evaluate_examples(
                        accelerator.unwrap_model(model),
                        processor,
                        selection_examples[dataset],
                        records[[str(record["dataset"]).lower() for record in records].index(dataset)]["config"],
                        eval_dir,
                        checkpoint_name=f"epoch_{epoch_index}",
                        sample_prediction_mode="original_teacher_forced",
                    )
                    headline = metrics["backend_results"]["original_teacher_forced"]["headline_metrics"]
                    component_metrics[dataset] = headline
                    selection_values.append(float(headline["macro_f1"]))
                mean_macro = float(sum(selection_values) / len(selection_values))
                row["component_selection_metrics"] = component_metrics
                row["mean_dataset_macro_f1"] = mean_macro
                improved = mean_macro > best_metric
                if improved:
                    best_metric = mean_macro
                    best_epoch = epoch_index
                    bad_epochs = 0
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    save_adapter_and_processor(accelerator.unwrap_model(model), processor, best_dir, config=model_config)
                else:
                    bad_epochs += 1
                LOGGER.info(
                    "Merged selection epoch=%s mean_dataset_macro_f1=%.6f best_epoch=%s bad_epochs=%s",
                    epoch_index,
                    mean_macro,
                    best_epoch,
                    bad_epochs,
                )
                if bad_epochs >= patience:
                    row["stopped_early"] = True
                    history.append(row)
                    break
            else:
                row["stopped_early"] = False
            history.append(row)
            save_json(history, logs_dir / "training_history.json")
        stop_tensor = torch.tensor(0, dtype=torch.int32, device=accelerator.device)
        if accelerator.is_main_process and use_selection and history and history[-1].get("stopped_early"):
            stop_tensor.fill_(1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_tensor, src=0)
        accelerator.wait_for_everyone()
        if int(stop_tensor.item()) == 1:
            break
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if not use_selection:
            best_epoch = len(history)
            best_metric = float("nan")
            if best_dir.exists():
                shutil.rmtree(best_dir)
            save_adapter_and_processor(accelerator.unwrap_model(model), processor, best_dir, config=model_config)
        if best_epoch < 0:
            raise RuntimeError("Merged training completed without selecting a checkpoint.")
        save_json(
            {
                "selected_epoch": int(best_epoch),
                "selection_metric": "mean_dataset_macro_f1" if use_selection else None,
                "selection_metric_value": None if math.isnan(best_metric) else float(best_metric),
                "history_path": str(logs_dir / "training_history.json"),
                "protocol_split_hash": protocol["protocol"]["split_hash"],
            },
            logs_dir / "selected_checkpoint.json",
        )
        complete = {
            "status": "completed",
            "identity": identity,
            "best_model_dir": str(best_dir),
            "selected_epoch": int(best_epoch),
            "completed_epochs": len(history),
            "selection_metric_value": None if math.isnan(best_metric) else float(best_metric),
        }
        save_json(complete, complete_path)
    accelerator.wait_for_everyone()
    return {"status": "completed", "run_root": str(run_root), "fold": int(fold), "stage": stage}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one symmetric merged Qwen stage/fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--subjects-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = train_merged_fold(
        args.config,
        stage=args.stage,
        fold=args.fold,
        run_id=args.run_id,
        epochs_override=args.epochs,
        subjects_per_class=args.subjects_per_class,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
