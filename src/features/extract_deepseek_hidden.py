"""Frozen DeepSeek-V4-Flash text features in the Qwen hidden-cache format.

For each route the script reuses a Qwen3.8 text-only reference run (config,
saved split, manifest) exactly like ``extract_qwen_hidden.py --base-only``. It
builds the same examples, but renders the stored ``prompt_system_text`` and
``prompt_user_text`` with the DeepSeek chat encoding and keeps the final
post-norm hidden state of the last prompt token (4096 values). Rows, partitions
and metadata follow the Qwen cache so ``baselines/frozen_standalone_heads.py``
reads the result unchanged.

Run under ``torchrun --nproc-per-node 4`` against the model-parallel (MP=4)
checkpoint made by the reference ``convert.py``. All ranks build the same
examples; rank 0 writes the files. One prompt per forward pass, no padding,
no labels in the model input.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist

from src.data.runtime import load_manifest_rows
from src.features.extract_qwen_hidden import (
    POOLING_NAME,
    _load_saved_run,
    _partition_examples,
    _saved_path,
    _resolve_subject_partitions,
    _validate_saved_split,
    ensure_hidden_extraction_supported,
    resolve_condition,
)
from src.data.emotion import use_emotion
from src.utils import (
    read_json,
    save_json,
    sha256_file,
    sha256_jsonl_rows,
    sha256_text,
    write_jsonl,
)

SCHEMA_VERSION = "deepseek_v4flash_hidden_cache.v1"
BACKEND_NAME = "deepseek_v4_flash"
HIDDEN_SIZE = 4096
PARTITIONS = ("outer_train", "final_eval")
CACHE_FILES = (
    "outer_train.npz",
    "outer_train_rows.jsonl",
    "final_eval.npz",
    "final_eval_rows.jsonl",
    "extraction_metadata.json",
)
# Same external metadata keys as the Qwen prompt-only collator.
OPTIONAL_ROW_KEYS = (
    "response_id", "question_condition", "turn_key", "question_id", "window_id",
    "prompt_id", "segment_index", "num_segments", "chunk_id", "bundle_id",
    "bundle_chunk_ids", "bundle_coverage_count", "effective_k",
    "chunk_schedule_epoch", "protocol_id", "chunk_index", "num_chunks",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", required=True, type=Path,
                   help="JSON list of {name, reference_checkpoint_dir, output_dir, final_eval_source?}.")
    p.add_argument("--model-dir", required=True, type=Path, help="HF checkpoint dir (config, tokenizer, inference/).")
    p.add_argument("--mp-ckpt-dir", required=True, type=Path, help="Converted model{rank}-mp{N}.safetensors dir.")
    p.add_argument("--max-examples", type=int, help="Smoke only: first N examples per partition.")
    p.add_argument("--max-seq-len", type=int, help="Override the KV length (default: longest prompt + 64).")
    p.add_argument("--plan-only", action="store_true", help="Build prompts and report token counts; no model.")
    return p.parse_args()


def _dist_info() -> tuple[int, int, int]:
    return int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))


def _row_metadata(example: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    row = {
        "dataset": str(example["dataset"]),
        "sample_id": str(example["sample_id"]),
        "subject_id": str(example["subject_id"]),
        "label": int(example["label"]),
        "partition": str(example["partition"]),
        "fold": int(example["fold"]),
    }
    for key in OPTIONAL_ROW_KEYS:
        if key in example:
            row[key] = example[key]
    row.update({
        "prompt_sha256": sha256_text(prompt_text),
        "hidden_layer": "final",
        "pooling": POOLING_NAME,
        "vector_dimension": HIDDEN_SIZE,
        "vector_dtype": "float32",
        "mask_source": "batch1_unpadded",
    })
    return row


def build_route(route: dict[str, Any], encode, tokenizer, max_examples: int | None) -> dict[str, Any]:
    """Examples, token ids and provenance for one route (CPU, deterministic)."""
    ref = Path(route["reference_checkpoint_dir"]).resolve()
    saved, config, run_config_path, split_path = _load_saved_run(ref)
    ensure_hidden_extraction_supported(config, run_config_path)
    config = json.loads(json.dumps(config))
    fold = int(saved["fold"])
    split_payload = read_json(split_path)
    final_source = route.get("final_eval_source", "saved")
    partition_ids, evaluation_provenance = _resolve_subject_partitions(
        saved, config, split_payload, final_eval_source=final_source)
    cv_protocol = str(saved.get("cv_protocol") or config.get("split", {}).get("cv_protocol") or "")
    train_sources = 1 if cv_protocol == "train_val" else 2
    split_metadata_path = _validate_saved_split(
        saved, config, partition_ids, fold, train_sources,
        split_payload=split_payload, final_eval_source=final_source)
    manifest_path = _saved_path(saved["manifest_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_hash = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and manifest_hash != saved["manifest_hash"]:
        raise ValueError("Manifest hash differs from the reference run.")
    examples, _ = _partition_examples(manifest_rows, config, partition_ids, fold, checkpoint_dir=ref)
    if saved.get("input_modality") not in (None, "text_only", "text"):
        raise ValueError(f"DeepSeek extraction is text-only; got {saved.get('input_modality')!r}.")
    out: dict[str, Any] = {"partitions": {}}
    for partition in PARTITIONS:
        items = examples[partition][:max_examples] if max_examples else examples[partition]
        rows, token_ids = [], []
        for example in items:
            system_text = example.get("prompt_system_text")
            user_text = example.get("prompt_user_text")
            if not isinstance(user_text, str) or not user_text.strip():
                raise ValueError(f"{example.get('sample_id')}: missing prompt_user_text.")
            messages = ([{"role": "system", "content": system_text}] if system_text else []) + [
                {"role": "user", "content": user_text}]
            prompt_text = encode(messages, thinking_mode="chat")
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            rows.append(_row_metadata(example, prompt_text))
            token_ids.append(ids)
        out["partitions"][partition] = {"rows": rows, "ids": token_ids}
    out["provenance"] = {
        "dataset": config["dataset"],
        "condition": resolve_condition(None, saved.get("input_modality"), use_emotion(config)),
        "input_modality": saved.get("input_modality"),
        "protocol_id": config.get("protocol_id", ""),
        "fold": fold,
        "reference_checkpoint_dir": str(ref),
        "saved_run_config": str(run_config_path),
        "saved_run_config_sha256": sha256_file(run_config_path),
        "saved_split": str(split_path),
        "saved_split_sha256": sha256_file(split_path),
        "evaluation_provenance": evaluation_provenance,
        "evaluation_view": {
            "sample_mode": config.get("data", {}).get("sample_mode"),
            "eval_chunk_policy": config.get("data", {}).get("eval_chunk_policy"),
            "subject_score_aggregation": config.get("evaluation", {}).get("subject_score_aggregation"),
        },
        "split_metadata": str(split_metadata_path),
        "split_metadata_sha256": sha256_file(split_metadata_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "final_eval_source": final_source,
    }
    return out


@torch.inference_mode()
def final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Post-norm hidden state of the last position, before the vocabulary head.

    Mirrors ``Transformer.forward`` of the reference code up to the logits, so
    the vector is exactly what the language-model head would receive.
    """
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    for layer in model.layers:
        h = layer(h, 0, input_ids)
    x = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
    return model.norm(x)[:, -1].float()


def main() -> None:
    args = parse_args()
    world_size, rank, local_rank = _dist_info()
    inference_dir = args.model_dir / "inference"
    sys.path.insert(0, str(inference_dir))
    sys.path.insert(0, str(args.model_dir / "encoding"))
    from encoding_dsv4 import encode_messages  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    routes = json.loads(args.routes.read_text())
    todo = []
    for route in routes:
        out_dir = Path(route["output_dir"])
        if (out_dir / "extraction_metadata.json").exists():
            print(f"[skip] {route['name']}: complete cache exists", flush=True)
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError(f"{out_dir} is non-empty without extraction_metadata.json; refuse to overwrite.")
        todo.append(route)
    built = {r["name"]: build_route(r, encode_messages, tokenizer, args.max_examples) for r in todo}
    lengths = [len(ids) for b in built.values() for p in b["partitions"].values() for ids in p["ids"]]
    summary = {"routes": len(built), "examples": len(lengths),
               "tokens_max": max(lengths, default=0), "tokens_mean": float(np.mean(lengths)) if lengths else 0,
               "tokens_total": int(sum(lengths))}
    if rank == 0:
        print("PLAN", json.dumps(summary), flush=True)
    if args.plan_only or not built:
        return

    from model import ModelArgs, Transformer  # noqa: PLC0415
    from safetensors.torch import load_model  # noqa: PLC0415

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(inference_dir / "config.json") as handle:
        margs = ModelArgs(**json.load(handle))
    margs.max_batch_size = 1
    margs.max_seq_len = int(args.max_seq_len or summary["tokens_max"] + 64)
    if summary["tokens_max"] > margs.max_seq_len:
        raise ValueError(f"Longest prompt {summary['tokens_max']} exceeds max_seq_len {margs.max_seq_len}.")
    with torch.device("cuda"):
        model = Transformer(margs)
    load_model(model, str(args.mp_ckpt_dir / f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    model.eval()

    for name, route_data in built.items():
        out_dir = Path(next(r for r in todo if r["name"] == name)["output_dir"])
        part_summaries: dict[str, Any] = {}
        for partition, payload in route_data["partitions"].items():
            vectors, max_diff = [], None
            for i, ids in enumerate(payload["ids"]):
                tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
                vec = final_hidden(model, tokens)
                if i == 0:
                    again = final_hidden(model, tokens)
                    max_diff = float((vec - again).abs().max())
                    if not torch.allclose(vec, again, rtol=1e-5, atol=1e-5):
                        raise ValueError(f"{name}: determinism check failed, max_abs_diff={max_diff}")
                    # The vector must reproduce the model's own logits on this rank's vocabulary shard.
                    # inference_mode: head.weight requires grad, and vec is an inference tensor
                    # (smoke job 48383325 failed here without it).
                    with torch.inference_mode():
                        local = torch.nn.functional.linear(vec, model.head.weight)
                        full = model(tokens, 0)
                    shard = full[:, rank * local.shape[-1]:(rank + 1) * local.shape[-1]]
                    if not torch.allclose(local, shard.float(), rtol=1e-3, atol=1e-3):
                        raise ValueError(f"{name}: hidden vector does not reproduce model logits.")
                v = vec[0].cpu().numpy().astype(np.float32, copy=False)
                if v.shape != (HIDDEN_SIZE,) or not np.isfinite(v).all():
                    raise ValueError(f"{name}: bad vector for {payload['rows'][i]['sample_id']}.")
                vectors.append(v)
                if rank == 0 and ((i + 1) % 10 == 0 or i + 1 == len(payload["ids"])):
                    print(f"{name}/{partition}: {i + 1}/{len(payload['ids'])}", flush=True)
            if rank == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                matrix = np.stack(vectors) if vectors else np.empty((0, HIDDEN_SIZE), np.float32)
                np.savez_compressed(out_dir / f"{partition}.npz", vectors=matrix)
                write_jsonl(payload["rows"], out_dir / f"{partition}_rows.jsonl")
            part_summaries[partition] = {
                "rows": len(payload["rows"]),
                "subjects": len({r["subject_id"] for r in payload["rows"]}),
                "tokens_max": max((len(x) for x in payload["ids"]), default=0),
                "determinism_rtol": 1e-5, "determinism_atol": 1e-5, "determinism_max_abs_diff": max_diff,
            }
        if rank == 0:
            prov = route_data["provenance"]
            metadata = {
                **{k: prov[k] for k in ("dataset", "condition", "input_modality", "protocol_id", "fold")},
                "schema_version": SCHEMA_VERSION,
                "model_backend": BACKEND_NAME,
                "checkpoint_type": "base_model",
                "weights_source": "base_model_no_adapter",
                "base_model": "deepseek-ai/DeepSeek-V4-Flash",
                "base_model_path": str(args.model_dir),
                "mp_checkpoint_dir": str(args.mp_ckpt_dir),
                "world_size": world_size,
                "weights_dtype": "fp4_experts_fp8_other",
                "prompt_encoding": "encoding_dsv4.encode_messages(thinking_mode='chat')",
                "hidden_layer": "final",
                "hidden_definition": "post hc_head and final RMSNorm, last prompt token, before the vocabulary head",
                "pooling": POOLING_NAME,
                "vector_dimension": HIDDEN_SIZE,
                "vector_dtype": "float32",
                "max_seq_len": margs.max_seq_len,
                **{k: v for k, v in prov.items() if k not in ("dataset", "condition", "input_modality", "protocol_id", "fold")},
                "gold_label_protection": {"input_field": "prompt_system_text+prompt_user_text",
                                          "generation_used": False, "labels_passed_to_model": False},
                "partitions": part_summaries,
                "note": "Reference run supplies config, split and manifest only; no adapter is loaded.",
                "versions": {"python": platform.python_version(), "torch": torch.__version__},
            }
            save_json(metadata, out_dir / "extraction_metadata.json")
            print(f"[done] {name}", flush=True)
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
