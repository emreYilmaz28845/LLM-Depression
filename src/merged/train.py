from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import timedelta
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
    limit_examples_by_dataset_subjects_per_class,
)
from src.merged.configuration import model_config
from src.merged.provenance import source_commits_match, write_slurm_provenance
from src.merged.runtime import (
    limit_grouped_subjects,
    load_merged_config,
    load_records_and_protocol,
    make_final_partitions,
    make_fold_partitions,
    merged_fold_root,
)
from src.model.runtime import (
    build_collator,
    fsdp_wrap_policy_names,
    load_model_for_training,
    load_processor,
    prepare_backend_examples,
    resolve_processor_sampling_rate,
    restore_model_for_training,
)
from src.training_strategy import (
    TRAINING_STRATEGY_FSDP,
    _force_gradient_sync_in_accumulation,
    activation_offload_context,
    align_fsdp_model_dtypes,
    build_fsdp_plugin,
    broadcast_flag,
    resolve_training_strategy,
    save_training_checkpoint,
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


_MERGED_TRACKING_FILES = frozenset(
    {
        "run_config.yaml",
        "metadata.json",
        "status.json",
        "jobs.jsonl",
        "artifacts.json",
        "evaluations.json",
    }
)
_MANAGED_CHILD_ATTEMPT_FILES = _MERGED_TRACKING_FILES


def _is_managed_child_attempt(path: Path) -> bool:
    """Recognize a child head initialized by the managed v2 orchestration."""

    return path.is_dir() and all(
        (path / name).is_file() for name in _MANAGED_CHILD_ATTEMPT_FILES
    )


def _unexpected_incomplete_output_entries(run_root: Path) -> list[Path]:
    """Return entries that make an incomplete merged output unsafe to resume.

    Managed merged head attempts live below the training fold root so their
    parent checkpoint can be recorded in the same output tree.  They are safe
    to coexist with the parent's tracking sidecars; unrelated files and
    directories remain collision errors.
    """

    if not run_root.exists():
        return []
    return [
        path
        for path in run_root.iterdir()
        if path.name not in _MERGED_TRACKING_FILES
        and not _is_managed_child_attempt(path)
    ]


def _relaunch_multi_gpu_slurm_worker() -> None:
    """Guard against a stale plain-Python batch-script submission.

    ``sbatch`` stores the script body at submission time.  If a submission still
    invokes this module with plain ``python`` while holding a multi-GPU
    allocation, relaunch the same arguments through a local ``torchrun`` process
    group sized by ``NPROC_PER_NODE`` (or by the Slurm allocation). Torchrun
    children expose ``LOCAL_RANK`` and therefore pass through without
    recursively spawning another group.
    """

    if os.environ.get("LOCAL_RANK") is not None:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        return
    raw_gpu_count = (
        os.environ.get("SLURM_GPUS_ON_NODE")
        or os.environ.get("SLURM_GPUS_PER_NODE")
        or ""
    )
    try:
        allocated_gpu_count = int(str(raw_gpu_count).split("(", 1)[0].split(";", 1)[0])
    except ValueError:
        allocated_gpu_count = 0
    raw_ranks = str(os.environ.get("NPROC_PER_NODE", "")).strip()
    if raw_ranks:
        try:
            process_count = int(raw_ranks)
        except ValueError as exc:
            raise RuntimeError(f"Invalid NPROC_PER_NODE={raw_ranks!r}.") from exc
    else:
        process_count = allocated_gpu_count
    if process_count <= 1:
        return
    if allocated_gpu_count and process_count > allocated_gpu_count:
        raise RuntimeError(
            f"NPROC_PER_NODE={process_count} exceeds the Slurm allocation of "
            f"{allocated_gpu_count} GPU(s)."
        )
    # Launch through the current interpreter's torch.distributed.run: a bare
    # torchrun resolves through PATH and can belong to a different environment
    # (the Qwen3 overlay venvs ship no console scripts of their own, and a
    # wrong-environment torchrun silently ran transformers 4.55.0 once).
    if not sys.executable:
        raise RuntimeError("sys.executable is unavailable; cannot relaunch the process group.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={process_count}",
        "-m",
        "src.merged.train",
        *sys.argv[1:],
    ]
    LOGGER.warning(
        "Direct multi-GPU Slurm invocation detected; relaunching under torch.distributed.run: %s",
        " ".join(command),
    )
    os.execvpe(sys.executable, command, os.environ.copy())


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
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(f"Unsupported merged training stage: {stage}")
    merged_config = load_merged_config(config_path, overrides)
    records, protocol = load_records_and_protocol(merged_config)
    model_config = _model_config(merged_config, records)
    evaluation_contract = model_config.get("merged_evaluation_contract") or {}
    prediction_mode = str(evaluation_contract.get("sample_prediction_mode") or "") or "original_teacher_forced"
    evaluation_view = str(evaluation_contract.get("evaluation_view") or "")
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
        train_examples, selected_train = limit_examples_by_dataset_subjects_per_class(
            train_examples, subjects_per_class=int(subjects_per_class)
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

    strategy = resolve_training_strategy(model_config)
    dist_timeout_minutes = int(merged_config.get("training", {}).get("dist_timeout_minutes", 30) or 30)
    if dist_timeout_minutes > 0 and not torch.distributed.is_initialized():
        # A slow collective (the five-dataset selection evaluation or an FSDP
        # all-gather) can hold the other ranks at the next collective longer
        # than torch's default NCCL watchdog timeout. Pre-initialize the process
        # group with a longer timeout; Accelerate reuses an initialized group.
        import datetime as _datetime

        torch.distributed.init_process_group(
            backend="nccl", timeout=_datetime.timedelta(minutes=dist_timeout_minutes)
        )
    model_name = str(resolve_model_name_or_path(None, model_config))
    processor = load_processor(model_name, model_config)
    sampling_rate = resolve_processor_sampling_rate(processor)
    # The FSDP plugin resolves the wrap policy from the loaded model, so the
    # model is created before the Accelerator.
    model = load_model_for_training(model_name, model_config)
    fsdp_plugin = None
    training_kwargs_handlers: list[Any] = []
    if strategy == TRAINING_STRATEGY_FSDP:
        if bool(model_config["training"].get("run_final_eval_in_train", False)):
            raise ValueError(
                "training.run_final_eval_in_train must be false under the fsdp strategy: the "
                "merged selection evaluation runs through the sharded model."
            )
        align_fsdp_model_dtypes(
            model,
            dtype=(
                torch.bfloat16
                if bool(model_config["training"].get("bf16", False)) and torch.cuda.is_available()
                else None
            ),
        )
        fsdp_plugin = build_fsdp_plugin(
            model_config,
            model,
            wrap_policy_names=lambda wrapped_model: fsdp_wrap_policy_names(model_config, wrapped_model),
        )
    else:
        training_kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator = Accelerator(
        mixed_precision=(
            "no"
            if strategy == TRAINING_STRATEGY_FSDP
            else ("bf16" if bool(model_config["training"].get("bf16", False)) else "no")
        ),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=training_kwargs_handlers,
    )
    if strategy == TRAINING_STRATEGY_FSDP:
        # FSDP's no_sync keeps unsharded gradients during accumulation; the
        # merged loop runs one backward per microbatch, so force the
        # reduce-scatter on every microbatch exactly like the standalone FSDP
        # recipe does.
        _force_gradient_sync_in_accumulation(accelerator)
    accelerator.wait_for_everyone()

    run_root = merged_fold_root(
        merged_config, run_id=run_id, stage=stage, fold=int(fold)
    )
    logs_dir = run_root / "logs"
    best_dir = run_root / "best_model"
    complete_path = run_root / "training_complete.json"
    is_local_main_process = accelerator.is_main_process
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
        "model_backend": str(model_config.get("model_backend") or ""),
        "sample_prediction_mode": prediction_mode,
        "evaluation_view": evaluation_view or None,
    }
    if complete_path.is_file() and best_dir.is_dir():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            # Historical Qwen outputs predate the model_backend identity
            # field; treat a missing field as the Qwen default.
            existing_identity = dict(existing.get("identity") or {})
            existing_identity.setdefault("model_backend", "")
            existing_identity.setdefault("sample_prediction_mode", "original_teacher_forced")
            existing_identity.setdefault("evaluation_view", None)
            if existing_identity != identity:
                raise ValueError(f"Incompatible completed merged training output: {run_root}")
        expected_source_commit = (
            os.environ.get("SOURCE_COMMIT")
            or os.environ.get("SYMMETRIC_MERGED_SOURCE_COMMIT")
        )
        provenance_path = run_root / "slurm_provenance.json"
        existing_source_commit: str | None = None
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            existing_source_commit = provenance.get("source_commit")
        if not expected_source_commit or source_commits_match(existing_source_commit, expected_source_commit):
            return {"status": "skipped_compatible_complete", "run_root": str(run_root), **existing}

    expected_source_commit = (
        os.environ.get("SOURCE_COMMIT")
        or os.environ.get("SYMMETRIC_MERGED_SOURCE_COMMIT")
    )
    provenance_path = run_root / "slurm_provenance.json"
    existing_source_commit: str | None = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        existing_source_commit = provenance.get("source_commit")
    stale_source_output = bool(
        expected_source_commit
        and not source_commits_match(existing_source_commit, expected_source_commit)
    )
    if stale_source_output and run_root.exists() and any(run_root.iterdir()):
        source_label = "".join(
            character if character.isalnum() else "_"
            for character in str(existing_source_commit or "missing")
        )[:16]
        job_label = os.environ.get("SLURM_JOB_ID", "retry")
        archive_root = run_root.with_name(
            f"{run_root.name}.stale_{source_label}_{job_label}"
        )
        suffix = 1
        while archive_root.exists():
            archive_root = run_root.with_name(
                f"{run_root.name}.stale_{source_label}_{job_label}_{suffix}"
            )
            suffix += 1
        if is_local_main_process:
            run_root.rename(archive_root)
            LOGGER.warning(
                "Archived stale merged training output source_commit=%s expected=%s path=%s",
                existing_source_commit,
                expected_source_commit,
                archive_root,
            )
        accelerator.wait_for_everyone()
    # The incomplete-output guard must run before any rank can observe the main
    # process's own fresh writes. A per-rank check races the main process and can
    # misread this job's identity/config files as a previous incomplete attempt.
    # Decide once on the main process and broadcast the decision; no rank writes
    # before the broadcast returns.
    incomplete_output_ok: bool | None = None
    if is_local_main_process:
        incomplete_output_ok = not (
            _unexpected_incomplete_output_entries(run_root) and not complete_path.is_file()
        )
    if not broadcast_flag(accelerator, bool(incomplete_output_ok)):
        raise ValueError(f"Refusing to overwrite an incomplete merged training output: {run_root}")
    ensure_dir(run_root)
    ensure_dir(logs_dir)
    if is_local_main_process:
        save_json(identity, run_root / "training_identity.json")
        save_json(merged_config, run_root / "resolved_merged_config.json")
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
    accelerator.wait_for_everyone()

    # Backend-dispatched prompt preparation: Gemma re-renders the prompt from
    # the raw system/user fields through its pinned chat template; Qwen is a
    # no-op. The collator factory below dispatches the same way.
    train_examples_prepared = prepare_backend_examples(
        weighted_examples, model_config, processor
    )
    selection_examples_prepared = {
        dataset: prepare_backend_examples(examples, model_config, processor)
        for dataset, examples in selection_examples.items()
    }
    # Per-epoch stochastic K-chunk resampling for subject_audio components
    # (currently DAIC only). The sampling guard in AudioTextDataset requires
    # subject_chunk_paths + chunks_per_subject, which only subject_audio
    # examples carry; response/segment components fall through to their baked
    # audio paths, so the merged "every example once per epoch" invariant and
    # the other datasets' exposure are unchanged. This restores the standalone
    # DAIC recipe's combinatorial K-view augmentation.
    train_dataset = AudioTextDataset(
        train_examples_prepared,
        processor_sampling_rate=sampling_rate,
        silence_audio=bool(model_config.get("data", {}).get("silence_audio", False)),
        chunk_sampling="random",
    )
    collator = build_collator(model_config, processor, debug=False, require_unit_range=False)
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
    if is_local_main_process:
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
                weight = 0.0 if dummy else float(weighted_examples[example_index]["loss_weight"])
                scale = weight / global_weight * process_count
                context = (
                    accelerator.no_sync(model)
                    if local_position < len(local_items) - 1
                    else torch.enable_grad()
                )
                with activation_offload_context(model_config), context:
                    # DDP observes the no-sync flag during the forward pass;
                    # entering it only around backward still performs an
                    # all-reduce for every microbatch. The merged loop applies
                    # the config's activation offload exactly like the
                    # standalone FSDP recipe does; skipping it doubles the
                    # per-rank activation memory.
                    outputs = model(**batch)
                    accelerator.backward(outputs.loss * float(scale))
                local_loss_numerator += float(outputs.loss.detach().item()) * weight
                local_loss_denominator += weight
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
        # The epoch-end selection evaluation runs on every rank: FSDP needs all
        # ranks inside the forward collectives, and the sharded model must be
        # evaluated through the prepared module (a direct call on the unwrapped
        # tree sees flattened 1-D weights with use_orig_params). Only the main
        # process writes evidence and owns the best-epoch bookkeeping; the save
        # decision is broadcast so every rank joins the checkpoint gather.
        component_metrics: dict[str, Any] = {}
        mean_macro: float | None = None
        if use_selection:
            selection_values: list[float] = []
            for dataset in DATASETS:
                eval_dir = ensure_dir(logs_dir / "selection" / f"epoch_{epoch_index}" / dataset)
                metrics = evaluate_examples(
                    model,
                    processor,
                    selection_examples_prepared[dataset],
                    records[[str(record["dataset"]).lower() for record in records].index(dataset)]["config"],
                    eval_dir,
                    checkpoint_name=f"epoch_{epoch_index}",
                    sample_prediction_mode=prediction_mode,
                    write_artifacts=accelerator.is_main_process,
                )
                active_backend = str(metrics["active_backend"])
                if active_backend != prediction_mode:
                    raise ValueError(
                        f"Merged selection resolved prediction backend {active_backend!r} for "
                        f"{dataset}, expected {prediction_mode!r}."
                    )
                headline = metrics["backend_results"][active_backend]["headline_metrics"]
                component_metrics[dataset] = headline
                selection_values.append(float(headline["macro_f1"]))
            mean_macro = float(sum(selection_values) / len(selection_values))
        save_best_selected = False
        if accelerator.is_main_process:
            row: dict[str, Any] = {
                "epoch": epoch_index,
                "train_loss": float(sum(item["normalized_loss"] for item in block_rows) / max(1, len(block_rows))),
                "realized_dataset_contributions": schedule["audit"]["realized_dataset_weight_contributions"],
                "schedule_hash": schedule["audit"]["schedule_hash"],
                "sample_prediction_mode": prediction_mode,
            }
            if use_selection:
                assert mean_macro is not None
                row["component_selection_metrics"] = component_metrics
                row["mean_dataset_macro_f1"] = mean_macro
                improved = mean_macro > best_metric
                if improved:
                    best_metric = mean_macro
                    best_epoch = epoch_index
                    bad_epochs = 0
                    save_best_selected = True
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
            else:
                row["stopped_early"] = False
            history.append(row)
            save_json(history, logs_dir / "training_history.json")
        # Collective checkpoint save: the main process's decision travels to
        # every rank first, then every rank joins the gather (FSDP) and the main
        # process writes the adapter directory.
        save_best_selected = broadcast_flag(accelerator, save_best_selected)
        if save_best_selected:
            if accelerator.is_main_process and best_dir.exists():
                shutil.rmtree(best_dir)
            accelerator.wait_for_everyone()
            save_training_checkpoint(accelerator, model, processor, best_dir, config=model_config)
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

    save_final_best = False
    if accelerator.is_main_process:
        if not use_selection:
            best_epoch = len(history)
            best_metric = float("nan")
            save_final_best = True
    save_final_best = broadcast_flag(accelerator, save_final_best)
    if save_final_best:
        if accelerator.is_main_process and best_dir.exists():
            shutil.rmtree(best_dir)
        accelerator.wait_for_everyone()
        save_training_checkpoint(accelerator, model, processor, best_dir, config=model_config)
    if accelerator.is_main_process:
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
    # Per-rank memory evidence, in the same format as the standalone trainer.
    # Without it the merged shape decision cannot be checked against the
    # recorded peaks of the standalone audio lane.
    if torch.cuda.is_available():
        LOGGER.info(
            "Per-rank training memory | rank=%s peak_allocated_gb=%.3f peak_reserved_gb=%.3f free_gb=%.3f",
            int(getattr(accelerator, "process_index", 0)),
            torch.cuda.max_memory_allocated() / 1024**3,
            torch.cuda.max_memory_reserved() / 1024**3,
            torch.cuda.mem_get_info()[0] / 1024**3,
        )
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
    parser.add_argument("--override", action="append", default=[])
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
        overrides=args.override,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    _relaunch_multi_gpu_slurm_worker()
    main()
