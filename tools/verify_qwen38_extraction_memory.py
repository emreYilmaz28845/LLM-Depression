#!/usr/bin/env python
"""Verify the Qwen3.8 memory-reduced extraction path on the real adapter.

Runs one short text-only forward through both the existing wrapper path and the
new base-decoder path, then asserts that the adapter-active decoder is used,
the final hidden state has width 5120, no logits are produced, and the pooled
vectors are equivalent. This is the operational gate for the extraction memory
fix; it needs one GPU and the parent checkpoint of a Qwen3.8 text route.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_qwen_hidden import (  # noqa: E402
    BACKEND_HIDDEN_SIZES,
    _forward_final_hidden_states,
    _load_saved_run,
    _place_model_for_extraction,
    _qwen38_base_decoder,
)
from src.features.pooling import aligned_attention_mask, last_valid_token  # noqa: E402
from src.model.runtime import load_model_for_inference, load_processor  # noqa: E402
from src.utils import MODEL_BACKEND_QWEN38, resolve_model_backend  # noqa: E402

VERIFICATION_TEXT = "This is a short verification transcript about mood and sleep."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    saved, config, run_config_path, _split_path = _load_saved_run(checkpoint_dir)
    backend = resolve_model_backend(config)
    if backend != MODEL_BACKEND_QWEN38:
        raise SystemExit(f"expected qwen38 backend, got {backend}")
    model_name = saved.get("resolved_model_name_or_path") or config["model_name_or_path"]
    processor = load_processor(checkpoint_dir, config)
    model = load_model_for_inference(str(model_name), checkpoint_dir, config)
    # Mirror the extractor: the loader returns the model on CPU and the
    # extraction path moves it onto the GPU before any forward pass.
    _place_model_for_extraction(model, config)

    decoder = _qwen38_base_decoder(model)
    adapter_active = bool(
        getattr(model, "active_peft_config", None) is not None
        or getattr(model, "peft_config", None)
    )
    device = next(model.parameters()).device

    inputs = processor(text=[VERIFICATION_TEXT], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}

    with torch.inference_mode():
        base_out = decoder(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    base_hidden = getattr(base_out, "last_hidden_state", None)
    no_logits = not hasattr(base_out, "logits")

    new_hidden, new_output_mask = _forward_final_hidden_states(model, inputs, MODEL_BACKEND_QWEN38)
    with torch.inference_mode():
        old_out = model(
            **inputs,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    old_hidden = old_out.hidden_states[-1]

    expected_hidden = int(next(iter(BACKEND_HIDDEN_SIZES[MODEL_BACKEND_QWEN38])))
    shape_new = [int(size) for size in new_hidden.shape]
    shape_old = [int(size) for size in old_hidden.shape]
    max_abs_diff = float((old_hidden.float() - new_hidden.float()).abs().max().item())
    equivalent = bool(torch.allclose(old_hidden, new_hidden, rtol=1e-3, atol=1e-3))

    mask_new, _ = aligned_attention_mask(new_hidden, inputs["attention_mask"], new_output_mask)
    mask_old, _ = aligned_attention_mask(old_hidden, inputs["attention_mask"], None)
    mask_equal = bool(torch.equal(mask_new, mask_old))
    vector_new = last_valid_token(new_hidden, mask_new).cpu().numpy()[0]
    vector_old = last_valid_token(old_hidden, mask_old).cpu().numpy()[0]
    pooled_max_abs_diff = float(abs(vector_new - vector_old).max())
    pooled_equal = bool(pooled_max_abs_diff <= 1e-3)

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_config": str(run_config_path),
        "model_name": str(model_name),
        "model_class": type(model).__name__,
        "decoder_class": type(decoder).__name__ if decoder is not None else None,
        "adapter_active": adapter_active,
        "no_logits": no_logits,
        "shape_new": shape_new,
        "shape_old": shape_old,
        "expected_hidden": expected_hidden,
        "equivalent": equivalent,
        "max_abs_diff": max_abs_diff,
        "mask_equal": mask_equal,
        "pooled_equal": pooled_equal,
        "pooled_max_abs_diff": pooled_max_abs_diff,
        "ok": bool(
            decoder is not None
            and type(decoder).__name__ == "Qwen3_5Model"
            and adapter_active
            and no_logits
            and shape_new[-1] == expected_hidden
            and shape_new == shape_old
            and equivalent
            and mask_equal
            and pooled_equal
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
