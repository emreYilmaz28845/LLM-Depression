#!/usr/bin/env python
"""Measure the Qwen3.8-27B per-GPU memory footprint of the text-only backend.

Run this only inside a Slurm job on a GPU node; it loads the pinned 27B
checkpoint in BF16. The probe exercises the real code path:

1. ``load_model_for_training`` — config validation, architecture resolution,
   LoRA target regex and the LoRA audit;
2. one training step (forward + backward + AdamW step) under the config's own
   gradient-checkpointing setting;
3. one eval-style likelihood pass over the prompt plus the label span.

Each phase records CUDA peak memory, so the numbers answer the open question:
does the existing 4-GPU DDP training lane and the single-GPU evaluation lane fit
the model, or does the repository need an FSDP / multi-GPU evaluation change?

The probe writes no checkpoints and no training output. It always writes the
JSON report, including a captured error when a phase fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.model import qwen38_lora  # noqa: E402
from src.model.runtime import load_processor  # noqa: E402
from src.utils import load_yaml_with_overrides, resolve_model_name_or_path  # noqa: E402

FILLER_SENTENCE = (
    " The subject describes sleeping badly, feeling tired during the day, and losing "
    "interest in activities that used to matter to them."
)


def _memory_state(device: torch.device) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
        "device_free_gb": round(free_bytes / 1024**3, 3),
        "device_total_gb": round(total_bytes / 1024**3, 3),
    }


def _rename(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def _tokenize(processor, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    batch = processor(text=text, return_tensors="pt", padding=False)
    return {key: value.to(device) for key, value in batch.items()}


def _example(processor, config: dict, base_user_text: str, prompt_tokens: int) -> dict:
    """Render one text-only example whose prompt reaches roughly ``prompt_tokens``."""
    user_text = base_user_text
    example = {
        "sample_id": "memory-probe",
        "subject_id": "0",
        "label": 1,
        "internal_label_text": "Depressed",
        "prompt_system_text": str(config["prompt"]["system"]),
        "prompt_user_text": user_text,
    }
    for _ in range(64):
        rendered = qwen38_lora.render_qwen38_training_text(
            processor, example["prompt_system_text"], example["prompt_user_text"],
            example["internal_label_text"],
        )
        token_count = len(processor(text=rendered, return_tensors=None)["input_ids"])
        if token_count >= prompt_tokens:
            return example
        example["prompt_user_text"] += FILLER_SENTENCE * 8
    raise RuntimeError(f"Could not reach {prompt_tokens} prompt tokens by padding.")


def _build_report(args) -> dict:
    config = load_yaml_with_overrides(REPO_ROOT / args.config, args.overrides or None)
    model_name_or_path = resolve_model_name_or_path(None, config)
    report: dict = {
        "schema": "audiollm.qwen38_memory_probe.v1",
        "config": str(args.config),
        "model_name_or_path": model_name_or_path,
        "prompt_tokens_target": int(args.prompt_tokens),
        "phases": {},
        "error": None,
    }
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": properties.name,
        "total_memory_gb": round(properties.total_memory / 1024**3, 2),
        "device_count": torch.cuda.device_count(),
    }
    report["env"] = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import transformers

        report["env"]["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover - diagnostics only
        report["env"]["transformers"] = None
    try:
        import peft

        report["env"]["peft"] = peft.__version__
    except Exception:  # pragma: no cover - diagnostics only
        report["env"]["peft"] = None

    processor = load_processor(model_name_or_path, config)

    try:
        # Render and tokenize before the 27B load: a tokenization failure must not
        # cost GPU minutes.
        example = _example(
            processor,
            config,
            "The transcript of the subject's speech is:\nHello, thanks for having me.",
            int(args.prompt_tokens),
        )
        prepared = qwen38_lora.prepare_qwen38_examples([example], config, processor)[0]
        prompt_ids = _tokenize(processor, prepared["prompt_text"], device)
        prompt_len = int(prompt_ids["input_ids"].shape[1])
        training_ids = _tokenize(processor, prepared["training_text"], device)
        labels = training_ids["input_ids"].clone()
        labels[:, :prompt_len] = -100
        report["sequence"] = {
            "prompt_tokens": prompt_len,
            "label_tokens": int(training_ids["input_ids"].shape[1]) - prompt_len,
            "total_tokens": int(training_ids["input_ids"].shape[1]),
        }

        _rename(device)
        model = qwen38_lora.load_model_for_training(model_name_or_path, config)
        # Accelerate moves the prepared model to the device in the training path;
        # the probe does the same move explicitly.
        model.to(device)
        torch.cuda.synchronize(device)
        report["phases"]["load_lora_bf16"] = _memory_state(device)
        report["trainable_params"] = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        selection = getattr(model, "_resolved_lora_layer_selection", {})
        report["lora_layer_selection"] = {
            "decoder_hidden_layers": selection.get("decoder_hidden_layer_count"),
            "layers_to_transform": selection.get("layers_to_transform"),
        }

        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["training"]["learning_rate"]),
        )
        _rename(device)
        outputs = model(
            input_ids=training_ids["input_ids"],
            attention_mask=training_ids["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        report["phases"]["train_step"] = _memory_state(device)
        report["phases"]["train_step"]["loss"] = float(loss.item())

        qwen38_lora.prepare_model_for_evaluation(model)
        full_text = prepared["prompt_text"] + example["internal_label_text"]
        _rename(device)
        with torch.inference_mode():
            full_ids = _tokenize(processor, full_text, device)
            logits = model(
                input_ids=full_ids["input_ids"], attention_mask=full_ids["attention_mask"]
            ).logits
            selected = logits[0, prompt_len - 1 : full_ids["input_ids"].shape[1] - 1]
            target = full_ids["input_ids"][0, prompt_len:]
            token_log_probs = (
                torch.log_softmax(selected, dim=-1)
                .gather(-1, target.unsqueeze(-1))
                .squeeze(-1)
            )
            score = float(token_log_probs.mean().item())
        torch.cuda.synchronize(device)
        report["phases"]["eval_likelihood"] = _memory_state(device)
        report["phases"]["eval_likelihood"]["label_log_prob"] = score
    except Exception as exc:  # noqa: BLE001 - keep the phases measured so far
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device visible; run this probe inside a Slurm GPU job.")
        report = _build_report(args)
    except Exception as exc:  # noqa: BLE001 - the report must always carry evidence
        report = {
            "schema": "audiollm.qwen38_memory_probe.v1",
            "config": str(args.config),
            "phases": {},
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("error"):
        print(f"FAILED: {report['error']}", file=sys.stderr)
        print(f"partial report written to {output_path}", file=sys.stderr)
        return 1
    print(f"report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
