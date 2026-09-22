#!/usr/bin/env python
"""Measure FSDP throughput and per-rank memory for the Qwen3.8 text-only recipe.

Run inside a Slurm GPU job, after the correctness gates and before the full fold.
It uses the real DAIC examples - the longest participant transcripts - and the
real backend, and compares ``per_device_train_batch_size`` 1, 2 and 4 with the
accumulation that keeps the effective global batch at 128 (32, 16 and 8 on four
ranks). Every option runs the long examples, so a setting is only accepted when
the long examples fit.

It also audits gradient accumulation: whether the accumulate context skips the
per-microbatch gradient sync (no_sync) and what enforcing a sync on every
microbatch costs in step time.

The probe writes one JSON report and never writes checkpoints. It does not change
the model, the LoRA targets, the text handling or any other scientific setting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.runtime import (  # noqa: E402
    AudioTextDataset,
    build_examples,
    load_manifest_rows,
)
from src.model.runtime import (  # noqa: E402
    build_collator,
    fsdp_wrap_policy_names,
    load_model_for_training,
    load_processor,
)
from src.training_strategy import build_accelerator, effective_global_batch_size, resolve_training_strategy  # noqa: E402
from src.utils import load_yaml_with_overrides, resolve_model_name_or_path  # noqa: E402

EFFECTIVE_GLOBAL_BATCH_TARGET = 128


def _memory_state() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "free_gb": round(free_bytes / 1024**3, 3),
        "total_gb": round(total_bytes / 1024**3, 3),
    }


def _longest_subject_rows(manifest_path: Path, limit: int, example_index: int = -1) -> list[dict]:
    rows = load_manifest_rows(manifest_path)
    best_by_subject: dict[str, dict] = {}
    for row in rows:
        subject = str(row["subject_id"])
        transcript = str(row.get("full_participant_transcript") or "")
        current = best_by_subject.get(subject)
        if current is None or len(transcript) > len(str(current.get("full_participant_transcript") or "")):
            best_by_subject[subject] = row
    ordered = sorted(
        best_by_subject.values(),
        key=lambda row: len(str(row.get("full_participant_transcript") or "")),
        reverse=True,
    )
    if example_index >= 0:
        if example_index >= len(ordered):
            raise ValueError(
                f"--example-index {example_index} is out of range for {len(ordered)} subjects"
            )
        return [ordered[example_index]]
    return ordered[:limit]


def _host_ram_gb() -> dict:
    """Host memory from /proc: peak (VmHWM) and current resident (VmRSS)."""
    values = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    values["host_peak_gb"] = round(int(line.split()[1]) / 1024**2, 3)
                elif line.startswith("VmRSS:"):
                    values["host_rss_gb"] = round(int(line.split()[1]) / 1024**2, 3)
    except OSError:
        pass
    return values


def _assert_forced_gradient_sync(accelerator, model) -> dict:
    """Runtime proof that the accumulation window does not use FSDP's no_sync."""
    import contextlib  # noqa: PLC0415

    context = accelerator.no_sync(model)
    forced = isinstance(context, contextlib.nullcontext)
    if not forced:
        raise RuntimeError(
            "FSDP no_sync is active in the accumulation window; gradients would be "
            "kept unsharded. Refusing to measure an unsupported configuration."
        )
    return {"forced_sync_each_microbatch": True, "no_sync_context": type(context).__name__}


def _accumulation_cycle(
    accelerator,
    model,
    optimizer,
    batches: list[dict],
    *,
    skip_sync: bool,
    config: dict | None = None,
) -> dict:
    """One optimizer update over ``batches``; returns wall time and sync mode."""
    from src.training_strategy import activation_offload_context  # noqa: PLC0415

    torch.cuda.synchronize()
    started = time.perf_counter()
    losses: list[float] = []
    with activation_offload_context(config or {}):
        for batch in batches:
            if skip_sync:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    losses.append(float(outputs.loss.detach().item()))
                    accelerator.backward(outputs.loss)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                outputs = model(**batch)
                losses.append(float(outputs.loss.detach().item()))
                accelerator.backward(outputs.loss)
                optimizer.step()
                optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    samples = sum(int(batch["input_ids"].shape[0]) for batch in batches)
    return {
        "batch_count": len(batches),
        "samples": samples,
        "seconds": round(elapsed, 3),
        "samples_per_second": round(samples / elapsed, 4) if elapsed > 0 else None,
        "seconds_per_microbatch": round(elapsed / max(1, len(batches)), 4),
        "losses": [round(value, 6) for value in losses],
    }


def _fast_path_evidence(model) -> dict:
    """Prove which delta-rule kernel the live model actually uses.

    transformers picks the fla kernels at module import time
    (``self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule``),
    so the module of the bound callable is the evidence, not a version string.
    """
    evidence: dict = {}
    try:
        from transformers.utils.import_utils import (  # noqa: PLC0415
            _is_package_available,
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        evidence["fla_distribution"] = list(_is_package_available("fla", return_version=True))
        evidence["transformers_fla_available"] = bool(is_flash_linear_attention_available())
        evidence["causal_conv1d_available"] = bool(is_causal_conv1d_available())
    except Exception as exc:  # noqa: BLE001 - evidence only
        evidence["import_error"] = f"{type(exc).__name__}: {exc}"

    layer = None
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "recurrent_gated_delta_rule"):
            layer = module
            break
    if layer is None:
        evidence["layer_found"] = False
        return evidence
    evidence["layer_found"] = True
    chunk_kernel = getattr(layer, "chunk_gated_delta_rule", None)
    evidence["chunk_kernel_module"] = getattr(chunk_kernel, "__module__", "")
    evidence["chunk_kernel_name"] = getattr(chunk_kernel, "__name__", "")
    evidence["causal_conv1d_fn_used"] = getattr(layer, "causal_conv1d_fn", None) is not None
    norm = getattr(layer, "norm", None)
    evidence["norm_class"] = type(norm).__name__ if norm is not None else None
    evidence["fast_path"] = str(evidence.get("chunk_kernel_module", "")).startswith("fla")
    return evidence


def _measure_batch_size(
    base_config: dict,
    examples: list[dict],
    processor,
    batch_size: int,
    accumulation: int,
    steps: int,
    rank: int,
    report_evidence: dict,
    offload: str = "none",
) -> dict:
    config = json.loads(json.dumps(base_config))
    config["training"]["per_device_train_batch_size"] = batch_size
    config["training"]["gradient_accumulation_steps"] = accumulation
    config["training"]["activation_offload"] = "cpu" if offload == "both" else offload
    model_name_or_path = resolve_model_name_or_path(None, config)

    dataloader = DataLoader(
        AudioTextDataset(
            examples,
            processor_sampling_rate=None,
            silence_audio=bool(config["data"].get("silence_audio", False)),
            chunk_sampling="deterministic",
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collator(config, processor),
        num_workers=0,
    )
    model = load_model_for_training(model_name_or_path, config)
    if rank == 0 and not report_evidence.get("fast_path_evidence"):
        report_evidence["fast_path_evidence"] = _fast_path_evidence(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    accelerator = build_accelerator(
        config,
        model=model,
        wrap_policy_names=lambda wrapped: fsdp_wrap_policy_names(config, wrapped),
    )
    # The loader stays out of accelerator.prepare: the probe must feed the same
    # long examples to every rank, not a DistributedSampler shard of them.
    model, optimizer = accelerator.prepare(model, optimizer)
    if rank == 0:
        report_evidence[f"memory_after_prepare_bs{batch_size}"] = _memory_state()
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: PLC0415

            modules = FSDP.fsdp_modules(model)
            flat_params = [m._flat_param for m in modules if getattr(m, "_flat_param", None) is not None]
            report_evidence["fsdp_units"] = len(modules)
            report_evidence["flat_param_shard_numel"] = int(sum(int(p.numel()) for p in flat_params))
            report_evidence["flat_param_dtypes"] = sorted({str(p.dtype) for p in flat_params})
            report_evidence["sharded_parameter_bytes_gb"] = round(
                sum(int(p.numel()) * p.element_size() for p in flat_params) / 1024**3, 3
            )
        except Exception as exc:  # noqa: BLE001 - evidence only
            report_evidence["fsdp_introspection_error"] = f"{type(exc).__name__}: {exc}"

    torch.cuda.reset_peak_memory_stats()
    batches = []
    for index, batch in enumerate(dataloader):
        batches.append(batch)
        if index + 1 >= max(1, steps):
            break

    result: dict = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "activation_offload": offload,
        "effective_global_batch_size": effective_global_batch_size(
            config, int(getattr(accelerator, "num_processes", 1))
        ),
        "rank": rank,
        "steps": len(batches),
        "long_example_batches": len(batches),
        "gradient_sync": _assert_forced_gradient_sync(accelerator, model),
        "host_ram_before": _host_ram_gb(),
    }

    # Steady-state timing: accumulation with the accumulate context (the production
    # path; gradients still sync every microbatch because no_sync is disabled),
    # then the same microbatches outside the context.
    result["accumulate_context"] = _accumulation_cycle(
        accelerator, model, optimizer, batches, skip_sync=True, config=config
    )
    result["sync_every_microbatch"] = _accumulation_cycle(
        accelerator, model, optimizer, batches, skip_sync=False, config=config
    )
    if offload == "both":
        # Same microbatches, same order: the loss lists are the equality evidence,
        # and the second cycle's timing is the offload cost.
        reference_config = json.loads(json.dumps(config))
        reference_config["training"]["activation_offload"] = "none"
        result["offload_none"] = _accumulation_cycle(
            accelerator, model, optimizer, batches, skip_sync=True, config=reference_config
        )
        torch.cuda.reset_peak_memory_stats()
        result["offload_cpu"] = _accumulation_cycle(
            accelerator, model, optimizer, batches, skip_sync=True, config=config
        )
        result["memory_after_offload_cpu"] = _memory_state()
    result["memory_after_steps"] = _memory_state()
    result["host_ram_after"] = _host_ram_gb()
    result["no_sync_available"] = bool(hasattr(model, "no_sync"))
    result["distributed_type"] = str(getattr(accelerator.state, "distributed_type", ""))
    del model, optimizer, dataloader
    import gc  # noqa: PLC0415

    gc.collect()
    torch.cuda.empty_cache()
    return result


def _build_report(args) -> dict:
    base_config = load_yaml_with_overrides(REPO_ROOT / args.config, args.overrides or None)
    strategy = resolve_training_strategy(base_config)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    report: dict = {
        "schema": "audiollm.qwen38_fsdp_perf_probe.v1",
        "config": str(args.config),
        "strategy": strategy,
        "world_size": world_size,
        "rank": rank,
        "manifest_path": str(args.manifest),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "results": [],
        "fast_path_evidence": {},
        "error": None,
    }
    processor = load_processor(resolve_model_name_or_path(None, base_config), base_config)
    if int(args.spread) > 0:
        # Evenly spaced over the length-sorted subjects, so one sweep covers the
        # real length distribution including the longest example.
        ordered = _longest_subject_rows(Path(args.manifest), 10**9)
        count = min(int(args.spread), len(ordered))
        if count == 1:
            rows = [ordered[0]]
        else:
            indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
            rows = [ordered[index] for index in sorted(set(indices))]
    else:
        rows = _longest_subject_rows(Path(args.manifest), int(args.longest), int(args.example_index))
    report["selected_subjects"] = [str(row["subject_id"]) for row in rows]
    report["selected_transcript_chars"] = [
        len(str(row.get("full_participant_transcript") or "")) for row in rows
    ]
    examples = build_examples(rows, base_config, partition_name="train")
    if not examples:
        raise RuntimeError("No examples were built from the manifest rows.")
    report["example_count"] = len(examples)

    for batch_size in report["batch_sizes"]:
        accumulation = EFFECTIVE_GLOBAL_BATCH_TARGET // max(1, batch_size * world_size)
        try:
            report["results"].append(
                _measure_batch_size(
                    base_config,
                    examples,
                    processor,
                    batch_size,
                    accumulation,
                    int(args.steps),
                    rank,
                    report,
                    offload=str(args.offload),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one option failing must not hide the others
            import traceback as _traceback  # noqa: PLC0415

            report["results"].append(
                {
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": accumulation,
                    "rank": rank,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": _traceback.format_exc(),
                }
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--longest", type=int, default=4)
    parser.add_argument(
        "--example-index",
        type=int,
        default=-1,
        help="pick exactly the N-th longest subject (0 = longest); overrides --longest",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--offload",
        choices=["none", "cpu", "both"],
        default="none",
        help="activation offload for the training step; 'both' runs none then cpu on the same batches",
    )
    parser.add_argument(
        "--spread",
        type=int,
        default=0,
        help="use this many subjects evenly spaced over the length distribution (0 = use --longest)",
    )
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Every rank writes its own report; a shared path would race and the last
    # writer would hide the other ranks' evidence.
    rank = os.environ.get("RANK", "0")
    if rank != "0":
        output_path = output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device visible; run this probe inside a Slurm GPU job.")
        report = _build_report(args)
    except Exception as exc:  # noqa: BLE001 - the report must always carry evidence
        report = {
            "schema": "audiollm.qwen38_fsdp_perf_probe.v1",
            "config": str(args.config),
            "results": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("error"):
        print(f"FAILED: {report['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
