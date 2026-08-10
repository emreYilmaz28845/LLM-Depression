#!/usr/bin/env python3
"""Extract Androids final-layer prompt-token hidden vectors.

This is deliberately separate from the older cross-dataset extractor.  Its
cache schema records the Androids turn/window contract and cannot be consumed
by the generic majority-vote heads or by the excluded full-turn condition.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.runtime import load_manifest_rows
from src.features.androids_hidden_policy import (
    ANDROID_AGGREGATION_POLICY,
    ANDROID_DATASET,
    ANDROID_HIDDEN_CACHE_SCHEMA,
    ANDROID_MANIFEST_HASH,
    ANDROID_SPLIT_HASH,
    ANDROID_TEXT_AGGREGATION_POLICY,
    CACHE_ARTIFACT_NAMES,
    cache_identity,
    file_sha256,
    modality_policy,
    read_json,
    validate_androids_cache_metadata,
    write_jsonl,
)
from src.features.extract_qwen_hidden import (
    _decoder_hidden_size,
    _git_commit,
    _is_daic_chunking,
    _load_saved_run,
    _package_version,
    _partition_examples,
    _resolve_subject_partitions,
    _saved_path,
    _validate_saved_split,
)
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator, load_prompt_audio
from src.model.runtime import load_model_for_inference, load_processor, resolve_processor_sampling_rate
from src.utils import read_json, save_json, sha256_file, sha256_jsonl_rows, sha256_text


MODALITY_TO_INPUT = {
    "audio_only": "audio_only",
    "audio_text": "audio_text",
    "text_only": "text_only",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=tuple(MODALITY_TO_INPUT))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_metadata(
    *,
    metadata: dict[str, Any],
    source_rows_by_sample: dict[str, dict[str, Any]],
    source_rows_by_subject: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    subject_id = str(metadata["subject_id"])
    sample_id = str(metadata["sample_id"])
    if metadata["input_modality"] == "text_only":
        rows = sorted(
            source_rows_by_subject[subject_id],
            key=lambda row: (int(row.get("turn_id", 0)), int(row.get("window_index", 0))),
        )
        return {
            "source_turn_count": len({str(row["response_id"]) for row in rows}),
            "source_window_count": len(rows),
            "source_turn_ids": sorted({str(row["response_id"]) for row in rows}),
            "source_window_ids": [str(row["window_id"]) for row in rows],
            "source_window_inventory_sha256": sha256_text(
                json.dumps(
                    [
                        {
                            "window_id": row["window_id"],
                            "response_id": row["response_id"],
                            "window_index": int(row["window_index"]),
                            "start_time": float(row["start_time"]),
                            "end_time": float(row["end_time"]),
                        }
                        for row in rows
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
    source = source_rows_by_sample.get(sample_id)
    if source is None:
        raise ValueError(f"No canonical Androids manifest row for hidden sample {sample_id}.")
    required = (
        "recording_id",
        "turn_id",
        "turn_key",
        "response_id",
        "window_id",
        "window_index",
        "num_windows",
        "num_segments",
        "segment_index",
        "start_time",
        "end_time",
        "segment_duration",
        "turn_duration",
    )
    return {key: source[key] for key in required}


def _extract_partition(
    *,
    model: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    output_dir: Path,
    partition: str,
    max_examples: int | None,
    expected_hidden_size: int,
    source_rows_by_sample: dict[str, dict[str, Any]],
    source_rows_by_subject: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    sampling_rate = resolve_processor_sampling_rate(processor)
    collator = PromptOnlyExtractionCollator(processor)
    selected = examples[:max_examples] if max_examples else examples
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    labels_by_subject: dict[str, int] = {}
    mask_sources: dict[str, int] = {}
    determinism_max_abs_diff: float | None = None
    for index, raw_example in enumerate(selected, start=1):
        example = load_prompt_audio(
            raw_example,
            sampling_rate,
            bool(config.get("data", {}).get("silence_audio", False)),
        )
        model_inputs, metadata_rows = collator([example])
        metadata = dict(metadata_rows[0])
        prompt_text = metadata.pop("prompt_text")
        metadata["input_modality"] = str(raw_example["input_modality"])
        metadata.update(
            _source_metadata(
                metadata=metadata,
                source_rows_by_sample=source_rows_by_sample,
                source_rows_by_subject=source_rows_by_subject,
            )
        )
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        if "labels" in model_inputs:
            raise AssertionError("Gold labels must never be passed to Qwen during Androids extraction.")
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                labels=None,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        mask, mask_source = aligned_attention_mask(
            hidden,
            model_inputs["attention_mask"],
            getattr(outputs, "attention_mask", None),
        )
        vector = last_valid_token(hidden, mask).cpu().numpy()[0].astype(np.float32, copy=False)
        if vector.shape != (expected_hidden_size,) or not bool(np.isfinite(vector).all()):
            raise ValueError(f"Invalid Androids hidden vector for {metadata['sample_id']}.")
        if index == 1:
            with torch.inference_mode():
                repeated = model(
                    **model_inputs,
                    labels=None,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            repeated_mask, _ = aligned_attention_mask(
                repeated.hidden_states[-1],
                model_inputs["attention_mask"],
                getattr(repeated, "attention_mask", None),
            )
            repeated_vector = (
                last_valid_token(repeated.hidden_states[-1], repeated_mask)
                .cpu()
                .numpy()[0]
                .astype(np.float32, copy=False)
            )
            determinism_max_abs_diff = float(np.max(np.abs(vector - repeated_vector)))
            if not np.allclose(vector, repeated_vector, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"Androids extraction determinism check failed for {metadata['sample_id']}: "
                    f"max_abs_diff={determinism_max_abs_diff}"
                )
        sample_id = str(metadata["sample_id"])
        subject_id = str(metadata["subject_id"])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate Androids hidden sample ID: {sample_id}")
        seen_samples.add(sample_id)
        label = int(metadata["label"])
        if subject_id in labels_by_subject and labels_by_subject[subject_id] != label:
            raise ValueError(f"Androids subject {subject_id} has inconsistent labels.")
        labels_by_subject[subject_id] = label
        metadata.update(
            {
                "prompt_sha256": sha256_text(prompt_text),
                "hidden_layer": "final",
                "pooling": "last_valid_prompt_token",
                "vector_dimension": expected_hidden_size,
                "vector_dtype": "float32",
                "mask_source": mask_source,
                "checkpoint": str(checkpoint_dir),
            }
        )
        rows.append(metadata)
        vectors.append(vector)
        mask_sources[mask_source] = mask_sources.get(mask_source, 0) + 1
        if index % 25 == 0 or index == len(selected):
            print(f"{partition}: extracted {index}/{len(selected)}", flush=True)
    matrix = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, expected_hidden_size), dtype=np.float32)
    )
    np.savez_compressed(output_dir / f"{partition}.npz", vectors=matrix)
    write_jsonl(rows, output_dir / f"{partition}_rows.jsonl")
    return {
        "rows": len(rows),
        "subjects": len(labels_by_subject),
        "mask_sources": mask_sources,
        "determinism_max_abs_diff": determinism_max_abs_diff,
    }


def _validate_source(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    checkpoint_dir = args.checkpoint_dir.resolve()
    if checkpoint_dir.name != "best_model":
        raise ValueError("Androids hidden extraction requires a fold-specific best_model checkpoint.")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint artifact: {checkpoint_dir / name}")
    saved, config, run_config_path, split_path = _load_saved_run(checkpoint_dir)
    if str(config.get("dataset")) != ANDROID_DATASET:
        raise ValueError("Androids hidden extraction received a non-Androids checkpoint.")
    if str(saved.get("input_modality")) != MODALITY_TO_INPUT[args.modality]:
        raise ValueError(
            f"Checkpoint input modality {saved.get('input_modality')!r} does not match {args.modality!r}."
        )
    if args.modality == "audio_text" and config.get("data", {}).get("audio_text_transcript_scope") != "segment_aligned":
        raise ValueError("The Androids hidden Audio + Text source must be segment_aligned.")
    if args.modality != "audio_text" and config.get("data", {}).get("audio_text_transcript_scope") == "full_turn":
        raise ValueError("Full-turn Androids input is excluded from hidden classifiers.")
    if int(saved.get("fold", -1)) not in range(5):
        raise ValueError(f"Unexpected Androids outer fold: {saved.get('fold')!r}")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive when supplied.")
    return saved, config, run_config_path, split_path


def main() -> None:
    args = _parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    saved, config, run_config_path, split_path = _validate_source(args)
    fold = int(saved["fold"])
    split_payload = read_json(split_path)
    partitions, evaluation_provenance = _resolve_subject_partitions(saved, config, split_payload)
    cv_protocol = str(
        saved.get("cv_protocol") or config.get("split", {}).get("cv_protocol") or ""
    )
    train_source_count = 1 if (cv_protocol == "train_val" or _is_daic_chunking(config)) else 2
    split_metadata_path = _validate_saved_split(saved, config, partitions, fold, train_source_count)
    manifest_path = args.manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Androids manifest: {manifest_path}")
    manifest_rows = load_manifest_rows(manifest_path)
    canonical_manifest_hash = sha256_jsonl_rows(manifest_rows)
    if canonical_manifest_hash != ANDROID_MANIFEST_HASH or saved.get("manifest_hash") != canonical_manifest_hash:
        raise ValueError("Androids manifest hash does not match the frozen production manifest.")
    if saved.get("split_metadata_hash") != ANDROID_SPLIT_HASH or sha256_file(split_metadata_path) != ANDROID_SPLIT_HASH:
        raise ValueError("Androids official split hash does not match the frozen production split.")

    source_rows_by_sample = {str(row["sample_id"]): row for row in manifest_rows}
    source_rows_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        source_rows_by_subject[str(row["subject_id"])].append(row)
    examples = _partition_examples(manifest_rows, config, partitions, fold)
    input_modality = MODALITY_TO_INPUT[args.modality]
    scope = config.get("data", {}).get("audio_text_transcript_scope")
    cache_config = {
        "schema_version": ANDROID_HIDDEN_CACHE_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": args.modality,
        "input_modality": input_modality,
        "audio_text_transcript_scope": scope,
        "fold": fold,
        "source_run_id": args.source_run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_sha256": file_sha256(checkpoint_dir / "adapter_config.json"),
        "adapter_sha256": file_sha256(checkpoint_dir / "adapter_model.safetensors"),
        "saved_run_config_sha256": file_sha256(run_config_path),
        "saved_split_sha256": file_sha256(split_path),
        "split_metadata_sha256": file_sha256(split_metadata_path),
        "manifest_sha256": canonical_manifest_hash,
        "source_commit": args.source_commit,
        "max_examples": args.max_examples,
        "aggregation_policy": modality_policy(args.modality),
    }
    cache_config_sha256 = __import__("hashlib").sha256(
        json.dumps(cache_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if output_dir.exists() and any(output_dir.iterdir()):
        metadata_path = output_dir / "extraction_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"Refusing to overwrite a partial Androids cache: {output_dir}")
        existing = read_json(metadata_path)
        if existing.get("cache_config") != cache_config or existing.get("cache_config_sha256") != cache_config_sha256:
            raise ValueError(f"Refusing incompatible Androids cache collision: {output_dir}")
        if not all((output_dir / name).is_file() for name in CACHE_ARTIFACT_NAMES):
            raise ValueError(f"Androids cache is incomplete: {output_dir}")
        validate_androids_cache_metadata(existing, modality=args.modality, fold=fold, source_commit=args.source_commit)
        print(json.dumps({"status": "skipped_compatible_complete_cache", "output_dir": str(output_dir)}, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name_or_path or saved.get("resolved_model_name_or_path") or config["model_name_or_path"]
    processor = load_processor(checkpoint_dir, config)
    model = load_model_for_inference(str(model_name), checkpoint_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and bool(config.get("training", {}).get("bf16", False)) else None
    model.to(device=device, dtype=dtype)
    model.eval()
    hidden_size = _decoder_hidden_size(model)
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
            expected_hidden_size=hidden_size,
            source_rows_by_sample=source_rows_by_sample,
            source_rows_by_subject=source_rows_by_subject,
        )
    metadata = {
        "schema_version": ANDROID_HIDDEN_CACHE_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": args.modality,
        "input_modality": input_modality,
        "audio_text_transcript_scope": scope,
        "fold": fold,
        "source_run_id": args.source_run_id,
        "checkpoint_type": "best_model",
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_sha256": cache_config["adapter_config_sha256"],
        "adapter_sha256": cache_config["adapter_sha256"],
        "base_model": str(model_name),
        "saved_run_config": str(run_config_path),
        "saved_run_config_sha256": cache_config["saved_run_config_sha256"],
        "saved_split": str(split_path),
        "saved_split_sha256": cache_config["saved_split_sha256"],
        "split_metadata": str(split_metadata_path),
        "split_metadata_sha256": cache_config["split_metadata_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": canonical_manifest_hash,
        "manifest_file_sha256": file_sha256(manifest_path),
        "source_commit": args.source_commit,
        "evaluation_provenance": evaluation_provenance,
        "hidden_layer": "final",
        "pooling": "last_valid_prompt_token",
        "vector_dimension": hidden_size,
        "vector_dtype": "float32",
        "aggregation_policy": modality_policy(args.modality),
        "cache_config": cache_config,
        "cache_config_sha256": cache_config_sha256,
        "max_examples": args.max_examples,
        "gold_label_protection": {
            "input_field": "prompt_text",
            "generation_used": False,
            "labels_passed_to_model": False,
        },
        "partitions": partition_summaries,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "project_git": _git_commit(),
        },
    }
    validate_androids_cache_metadata(metadata, modality=args.modality, fold=fold, source_commit=args.source_commit)
    _write_json(output_dir / "extraction_metadata.json", metadata)
    checksum_path = output_dir / "cache_sha256.tsv"
    rows = []
    for name in CACHE_ARTIFACT_NAMES:
        rows.append(f"{file_sha256(output_dir / name)}\t{name}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
