#!/usr/bin/env python
"""Diagnose what gradient checkpointing actually does for Qwen3.8 under FSDP.

Runs one example (the median-length DAIC transcript by default) on a 4-GPU FSDP
rank group and records a controlled ladder of memory phases:

1. forward only, gradients disabled;
2. training forward, graph built, before backward;
3. forward + backward, before any optimizer step;
4. the standard checkpointing path (accumulate context over a single microbatch);
5. the same as 4 with gradient checkpointing disabled, recording the OOM point.

It also reports, at runtime: how many decoder layers carry the checkpointing flag,
whether `use_cache` is off, how many layers sit under the checkpoint wrapper class,
the total bytes of tensors saved for backward during the training forward (with a
best-effort attribution to module classes), and the FSDP shard geometry.

Every rank writes its own JSON report. No checkpoints, no training output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.runtime import AudioTextDataset, build_examples, load_manifest_rows  # noqa: E402
from src.model.runtime import (  # noqa: E402
    build_collator,
    fsdp_wrap_policy_names,
    load_model_for_training,
    load_processor,
)
from src.training_strategy import build_accelerator  # noqa: E402
from src.utils import load_yaml_with_overrides, resolve_model_name_or_path  # noqa: E402


def _memory_state() -> dict:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "allocated_gb": round(allocated / 1024**3, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "reserved_gb": round(reserved / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "non_torch_gb": round((total_bytes - free_bytes - reserved) / 1024**3, 3),
        "free_gb": round(free_bytes / 1024**3, 3),
        "total_gb": round(total_bytes / 1024**3, 3),
    }


def _decoder_layers(model):
    for path in ("base_model.model.model.language_model", "model.language_model", "language_model"):
        target = model
        for part in path.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is not None and hasattr(target, "layers"):
            return list(target.layers)
    raise ValueError("Could not locate the language-model decoder layers.")


def _checkpointing_state(model) -> dict:
    layers = _decoder_layers(model)
    flagged = [index for index, layer in enumerate(layers) if bool(getattr(layer, "gradient_checkpointing", False))]
    class_names = Counter(type(layer).__name__ for layer in layers)
    mro_names = sorted({cls.__name__ for layer in layers for cls in type(layer).__mro__})
    return {
        "decoder_layer_count": len(layers),
        "layers_with_checkpointing_flag": len(flagged),
        "layer_classes": dict(class_names),
        "uses_gradient_checkpointing_layer_base": "GradientCheckpointingLayer" in mro_names,
        "use_cache_model": bool(getattr(getattr(model, "config", None), "use_cache", None)),
        "is_gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "wrapped_forward_uses_checkpoint_wrapper": bool(
            getattr(layers[0], "_gradient_checkpointing_func", None) is not None
        ),
    }


class _ModuleStack:
    """Track the innermost module class while the forward runs."""

    def __init__(self, model) -> None:
        self.stack: list[str] = []
        self.handles = []
        for name, module in model.named_modules():
            if not list(module.children()):
                continue
            self.handles.append(
                module.register_forward_pre_hook(self._pre(type(module).__name__))
            )
            self.handles.append(module.register_forward_hook(self._post))

    def _pre(self, name: str):
        def hook(module, args):
            self.stack.append(name)

        return hook

    def _post(self, module, args, output):
        if self.stack:
            self.stack.pop()

    def current(self) -> str:
        return self.stack[-1] if self.stack else "unknown"

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _saved_tensor_accounting(model, batch) -> dict:
    """Total bytes saved for backward during the training forward, by module class."""
    totals: Counter = Counter()
    counts: Counter = Counter()
    module_stack = _ModuleStack(model)

    def pack(tensor):
        module_stack_current = module_stack.current()
        size = int(tensor.numel()) * int(tensor.element_size())
        totals[module_stack_current] += size
        counts[module_stack_current] += 1
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(**batch)
        gpu_bytes = sum(totals.values())
    finally:
        module_stack.close()
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:12]
    return {
        "saved_bytes_total_gb": round(gpu_bytes / 1024**3, 3),
        "saved_tensor_count": int(sum(counts.values())),
        "top_modules_gb": {name: round(size / 1024**3, 3) for name, size in top},
    }


def _phase(label: str, fn, record: dict, rank: int) -> None:
    """Run one phase and record its memory; a failure ends the ladder on every rank.

    FSDP collectives run inside every phase, so a rank that swallowed an OOM and
    continued would leave the other ranks inside a collective. The failure is
    recorded and re-raised: torchrun tears the job down, and each rank's report is
    still written by the caller's finally block.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        fn()
    except Exception as exc:
        entry = {
            "seconds": round(time.perf_counter() - started, 3),
            **_memory_state(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        record.setdefault("phases", {})[label] = entry
        print(f"[rank {rank}] {label}: FAILED {entry['error']}")
        raise
    torch.cuda.synchronize()
    entry = {"seconds": round(time.perf_counter() - started, 3), **_memory_state()}
    record.setdefault("phases", {})[label] = entry
    print(
        f"[rank {rank}] {label}: peak={entry['peak_allocated_gb']} free={entry['free_gb']} "
        f"non_torch={entry['non_torch_gb']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-index", type=int, default=94, help="0 = longest subject")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rank != 0:
        output_path = output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")

    report: dict = {"schema": "audiollm.qwen38_checkpoint_diag.v1", "rank": rank, "phases": {}, "error": None}
    try:
        config = load_yaml_with_overrides(REPO_ROOT / args.config, args.overrides or None)
        processor = load_processor(resolve_model_name_or_path(None, config), config)
        rows = load_manifest_rows(args.manifest)
        by_subject: dict[str, dict] = {}
        for row in rows:
            subject = str(row["subject_id"])
            transcript = str(row.get("full_participant_transcript") or "")
            if subject not in by_subject or len(transcript) > len(
                str(by_subject[subject].get("full_participant_transcript") or "")
            ):
                by_subject[subject] = row
        ordered = sorted(
            by_subject.values(),
            key=lambda row: len(str(row.get("full_participant_transcript") or "")),
            reverse=True,
        )
        chosen = ordered[int(args.example_index)]
        report["subject"] = str(chosen["subject_id"])
        report["transcript_chars"] = len(str(chosen.get("full_participant_transcript") or ""))
        examples = build_examples([chosen], config, partition_name="train")
        loader = DataLoader(
            AudioTextDataset(examples, processor_sampling_rate=None, silence_audio=False, chunk_sampling="deterministic"),
            batch_size=1,
            shuffle=False,
            collate_fn=build_collator(config, processor),
            num_workers=0,
        )
        batch = next(iter(loader))
        report["sequence_tokens"] = int(batch["input_ids"].shape[1])

        model = load_model_for_training(resolve_model_name_or_path(None, config), config)
        report["checkpointing_before_prepare"] = _checkpointing_state(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config["training"]["learning_rate"]),
        )
        accelerator = build_accelerator(
            config, model=model, wrap_policy_names=lambda wrapped: fsdp_wrap_policy_names(config, wrapped)
        )
        model, optimizer = accelerator.prepare(model, optimizer)
        device = next(model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}

        unwrapped = accelerator.unwrap_model(model)
        report["checkpointing_after_prepare"] = _checkpointing_state(unwrapped)
        if rank == 0:
            report["memory_after_prepare"] = _memory_state()

        model.train()
        with torch.no_grad():
            _phase("1_forward_no_grad", lambda: model(**batch), report, rank)

        def forward_with_graph():
            nonlocal_loss = model(**batch)
            report["forward_loss"] = float(nonlocal_loss.loss.detach().item())

        _phase("2_forward_training", forward_with_graph, report, rank)

        if rank == 0:
            report["saved_tensors"] = _saved_tensor_accounting(model, batch)

        def forward_backward():
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()

        _phase("3_forward_backward", forward_backward, report, rank)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        def standard_step():
            optimizer.zero_grad(set_to_none=True)
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        _phase("4_checkpointing_on_full_step", standard_step, report, rank)
        torch.cuda.empty_cache()

        # Phase 5: the same step with checkpointing disabled.
        if hasattr(unwrapped, "gradient_checkpointing_disable"):
            unwrapped.gradient_checkpointing_disable()
        report["checkpointing_disabled_state"] = _checkpointing_state(unwrapped)
        _phase("5_checkpointing_off_full_step", standard_step, report, rank)

        report["fsdp_units"] = _fsdp_units(model)
    except Exception as exc:  # noqa: BLE001 - always write evidence
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if rank == 0:
        print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=2))
    return 1 if report.get("error") else 0


def _fsdp_units(model) -> dict:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: PLC0415

        modules = FSDP.fsdp_modules(model)
        flat_params = [m._flat_param for m in modules if getattr(m, "_flat_param", None) is not None]
        return {
            "units": len(modules),
            "flat_param_shard_numel": int(sum(int(p.numel()) for p in flat_params)),
            "flat_param_dtypes": sorted({str(p.dtype) for p in flat_params}),
            "sharded_parameter_bytes_gb": round(
                sum(int(p.numel()) * p.element_size() for p in flat_params) / 1024**3, 3
            ),
        }
    except Exception as exc:  # noqa: BLE001 - evidence only
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
