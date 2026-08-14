from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator, load_prompt_audio
from src.evaluate import evaluate_examples
from src.merged.protocol import DATASETS, canonical_sha256
from src.merged.runtime import (
    limit_grouped_subjects,
    load_merged_config,
    load_records_and_protocol,
    make_final_partitions,
    make_fold_partitions,
)
from src.merged.configuration import model_config as _model_config
from src.merged.provenance import write_slurm_provenance
from src.model.runtime import (
    load_model_for_inference,
    load_processor,
    prepare_backend_examples,
    resolve_processor_sampling_rate,
)
from src.utils import (
    configure_logging,
    ensure_dir,
    read_json,
    resolve_project_path,
    save_json,
    sha256_file,
    write_jsonl,
)


def _file_hash_if_present(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _checkpoint_hashes(checkpoint_dir: Path) -> dict[str, str | None]:
    return {
        "adapter_config_sha256": _file_hash_if_present(checkpoint_dir / "adapter_config.json"),
        "adapter_model_sha256": _file_hash_if_present(checkpoint_dir / "adapter_model.safetensors"),
        "adapter_model_bin_sha256": _file_hash_if_present(checkpoint_dir / "adapter_model.bin"),
    }


def _prepare_examples(
    grouped: dict[str, list[dict[str, Any]]], *, fold: int, partition: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        result[dataset] = []
        for example in grouped.get(dataset, []):
            item = dict(example)
            item["fold"] = int(fold)
            item["partition"] = partition
            result[dataset].append(item)
    return result


def _extract_partition(
    model,
    processor,
    examples: list[dict[str, Any]],
    *,
    output_dir: Path,
    partition: str,
    sampling_rate: int | None,
    silence_audio: bool,
    gemma_backend: bool = False,
    expected_hidden_size: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if gemma_backend:
        from src.features.gemma4_hidden_collator import Gemma4PromptOnlyExtractionCollator

        collator = Gemma4PromptOnlyExtractionCollator(processor)
    else:
        collator = PromptOnlyExtractionCollator(processor)
    device = next(model.parameters()).device
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for index, example in enumerate(examples, start=1):
        loaded = load_prompt_audio(example, sampling_rate, silence_audio)
        inputs, metadata_rows = collator([loaded])
        metadata = dict(metadata_rows[0])
        prompt_text = metadata.pop("prompt_text")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if "labels" in inputs:
            raise AssertionError("Gold labels must not enter hidden feature extraction.")
        with torch.inference_mode():
            outputs = model(
                **inputs,
                labels=None,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        mask, mask_source = aligned_attention_mask(
            hidden,
            inputs["attention_mask"],
            getattr(outputs, "attention_mask", None),
        )
        vector = last_valid_token(hidden, mask).cpu().numpy()[0].astype(np.float32, copy=False)
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite hidden vector for {metadata['sample_id']}.")
        if expected_hidden_size is not None and vector.shape[0] != expected_hidden_size:
            raise ValueError(
                f"Expected {expected_hidden_size} hidden features for the "
                f"{'gemma4' if gemma_backend else 'qwen'} backend, got {vector.shape[0]}."
            )
        sample_id = str(metadata["sample_id"])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate hidden feature sample identity: {sample_id}")
        seen_samples.add(sample_id)
        metadata.update(
            {
                "partition": partition,
                "fold": int(example.get("fold", 0)),
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "hidden_layer": "final",
                "pooling": "last_valid_prompt_token",
                "vector_dimension": int(vector.shape[0]),
                "vector_dtype": "float32",
                "mask_source": mask_source,
                "checkpoint": str(output_dir),
                "component_subject_id": example.get("component_subject_id", ""),
                "component_sample_id": example.get("component_sample_id", ""),
            }
        )
        rows.append(metadata)
        vectors.append(vector)
        if index % 25 == 0 or index == len(examples):
            print(f"{partition}: extracted {index}/{len(examples)}", flush=True)
    dimension = int(vectors[0].shape[0]) if vectors else 0
    matrix = np.stack(vectors).astype(np.float32, copy=False) if vectors else np.empty((0, dimension), dtype=np.float32)
    np.savez_compressed(output_dir / f"{partition}.npz", vectors=matrix)
    write_jsonl(rows, output_dir / f"{partition}_rows.jsonl")
    return matrix, rows, {
        "row_count": len(rows),
        "subject_count": len({str(row["subject_id"]) for row in rows}),
        "vector_dimension": dimension,
        "mask_sources": dict(sorted({source: sum(1 for row in rows if row.get("mask_source") == source) for source in {str(row.get("mask_source")) for row in rows}}.items())),
    }


def postprocess_merged_fold(
    config_path: str | Path,
    *,
    stage: str,
    fold: int,
    run_id: str,
    checkpoint_dir: str | Path,
    subjects_per_class: int | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(f"Unsupported merged postprocess stage: {stage}")
    merged_config = load_merged_config(config_path)
    records, protocol = load_records_and_protocol(merged_config)
    model_config = _model_config(merged_config, records)
    resolved_config_path = resolve_project_path(config_path)
    checkpoint = Path(checkpoint_dir).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing merged checkpoint directory: {checkpoint}")
    if stage == "final":
        partitions = make_final_partitions(records)
        train_grouped = partitions["examples"]["train"]
        holdout_grouped = partitions["examples"]["daic_official_test"]
        expected_holdouts = ("daic",)
    else:
        partitions = make_fold_partitions(records, protocol, fold)
        train_grouped = partitions["examples"]["outer_train"]
        holdout_grouped = partitions["examples"]["outer_holdout"]
        expected_holdouts = DATASETS
    if stage == "smoke":
        smoke_limit = int(
            subjects_per_class
            or merged_config.get("execution", {}).get("smoke_subjects_per_class", 2)
        )
        train_grouped, _ = limit_grouped_subjects(
            train_grouped, subjects_per_class=smoke_limit
        )
        holdout_grouped, _ = limit_grouped_subjects(
            holdout_grouped, subjects_per_class=smoke_limit
        )
    train_grouped = _prepare_examples(train_grouped, fold=fold, partition="outer_train")
    holdout_grouped = _prepare_examples(holdout_grouped, fold=fold, partition="outer_holdout")

    output_root = Path(merged_config["output_dirs"]["merged_root"]) / run_id / stage / f"fold_{int(fold)}"
    features_dir = output_root / "features"
    identity = {
        "schema_version": "symmetric_merged_postprocess_identity.v1",
        "config_name": merged_config.get("name"),
        "modality": merged_config.get("modality"),
        "stage": stage,
        "fold": int(fold),
        "run_id": run_id,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_hashes": _checkpoint_hashes(checkpoint),
        "manifest_hash": protocol["manifest"]["manifest_hash"],
        "split_hash": protocol["protocol"]["split_hash"],
        "merged_config_sha256": sha256_file(resolved_config_path),
        "fold_hash": protocol["protocol"].get("folds", {}).get(str(int(fold)), {}).get("fold_hash"),
        "expected_holdout_datasets": list(expected_holdouts),
        "subjects_per_class": subjects_per_class if stage == "smoke" else None,
        "model_backend": str(model_config.get("model_backend") or ""),
    }
    identity_path = output_root / "postprocess_identity.json"
    complete_path = output_root / "postprocess_complete.json"
    if complete_path.is_file() and identity_path.is_file():
        existing = read_json(identity_path)
        # Historical Qwen outputs predate the model_backend identity field;
        # treat a missing field as the Qwen default so they stay readable.
        existing.setdefault("model_backend", "")
        if existing != identity:
            raise ValueError(f"Incompatible merged postprocess output: {output_root}")
        return {"status": "skipped_compatible_complete", "output_root": str(output_root)}
    if output_root.exists() and any(output_root.iterdir()) and not identity_path.is_file():
        raise ValueError(f"Refusing to overwrite incomplete merged postprocess output: {output_root}")
    ensure_dir(output_root)
    ensure_dir(features_dir)
    save_json(identity, identity_path)
    save_json(merged_config, output_root / "resolved_merged_config.json")
    write_slurm_provenance(
        output_root / "slurm_provenance.json",
        worker="src.merged.postprocess",
        stage=stage,
        fold=int(fold),
        run_id=run_id,
        config_name=merged_config.get("name"),
        modality=merged_config.get("modality"),
        checkpoint_hashes=identity["checkpoint_hashes"],
        manifest_hash=identity["manifest_hash"],
        split_hash=identity["split_hash"],
    )

    model_name = str(model_config["model_name_or_path"])
    processor = load_processor(checkpoint, model_config)
    model = load_model_for_inference(model_name, checkpoint, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and bool(model_config["training"].get("bf16", False)) else None
    model.to(device=device, dtype=dtype)
    model.eval()
    sampling_rate = resolve_processor_sampling_rate(processor)
    # Backend-dispatched prompt preparation (Gemma re-renders the prompt from
    # the raw system/user fields; Qwen is a no-op).
    train_grouped = {
        dataset: prepare_backend_examples(examples, model_config, processor)
        for dataset, examples in train_grouped.items()
    }
    holdout_grouped = {
        dataset: prepare_backend_examples(examples, model_config, processor)
        for dataset, examples in holdout_grouped.items()
    }
    gemma_backend = str(model_config.get("model_backend") or "") == "gemma4"
    from src.features.extract_qwen_hidden import BACKEND_HIDDEN_SIZES

    expected_hidden_size = next(
        iter(BACKEND_HIDDEN_SIZES.get(str(model_config.get("model_backend") or ""), {0}))
    ) if gemma_backend else None
    eval_subdir = "gemma4" if gemma_backend else "qwen"

    qwen_summary: dict[str, Any] = {}
    for dataset in expected_holdouts:
        examples = holdout_grouped.get(dataset, [])
        if not examples:
            raise ValueError(f"Merged postprocess has no {dataset} outer holdout examples.")
        eval_dir = ensure_dir(output_root / eval_subdir / dataset)
        component_config = next(record["config"] for record in records if str(record["dataset"]).lower() == dataset)
        metrics = evaluate_examples(
            model,
            processor,
            examples,
            component_config,
            eval_dir,
            checkpoint_name="selected_checkpoint",
            sample_prediction_mode="original_teacher_forced",
        )
        qwen_summary[dataset] = {
            "metrics": metrics["backend_results"]["original_teacher_forced"]["headline_metrics"],
            "output_dir": str(eval_dir),
            "subject_count": len({str(example["subject_id"]) for example in examples}),
            "sample_count": len(examples),
        }
    save_json(qwen_summary, output_root / eval_subdir / "summary.json")

    train_examples = [example for dataset in DATASETS for example in train_grouped.get(dataset, [])]
    holdout_examples = [example for dataset in expected_holdouts for example in holdout_grouped.get(dataset, [])]
    _, train_rows, train_summary = _extract_partition(
        model,
        processor,
        train_examples,
        output_dir=features_dir,
        partition="outer_train",
        sampling_rate=sampling_rate,
        silence_audio=bool(model_config["data"].get("silence_audio", False)),
        gemma_backend=gemma_backend,
        expected_hidden_size=expected_hidden_size,
    )
    _, holdout_rows, holdout_summary = _extract_partition(
        model,
        processor,
        holdout_examples,
        output_dir=features_dir,
        partition="outer_holdout",
        sampling_rate=sampling_rate,
        silence_audio=bool(model_config["data"].get("silence_audio", False)),
        gemma_backend=gemma_backend,
        expected_hidden_size=expected_hidden_size,
    )
    dimensions = {int(row["vector_dimension"]) for row in train_rows + holdout_rows}
    if len(dimensions) > 1:
        raise ValueError(f"Merged hidden feature dimensions disagree: {sorted(dimensions)}")
    feature_metadata = {
        "schema_version": "symmetric_merged_hidden_features.v1",
        "stage": stage,
        "fold": int(fold),
        "modality": merged_config["modality"],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_hashes": _checkpoint_hashes(checkpoint),
        "config_identity": identity["config_name"],
        "manifest_hash": protocol["manifest"]["manifest_hash"],
        "split_hash": protocol["protocol"]["split_hash"],
        "feature_dimension": next(iter(dimensions)) if dimensions else 0,
        "merged_config_sha256": identity["merged_config_sha256"],
        "fold_hash": identity["fold_hash"],
        "pooling": "last_valid_prompt_token",
        "model_backend": str(model_config.get("model_backend") or ""),
        "partitions": {"outer_train": train_summary, "outer_holdout": holdout_summary},
        "gold_label_protection": {"labels_passed_to_model": False, "generation_used": False},
        "row_hashes": {
            "outer_train": canonical_sha256(train_rows),
            "outer_holdout": canonical_sha256(holdout_rows),
        },
    }
    save_json(feature_metadata, features_dir / "feature_metadata.json")
    save_json(
        {
            "status": "completed",
            "identity": identity,
            "qwen_summary": qwen_summary,
            "feature_metadata": str(features_dir / "feature_metadata.json"),
            "feature_dimension": feature_metadata["feature_dimension"],
        },
        complete_path,
    )
    return {"status": "completed", "output_root": str(output_root), "feature_dimension": feature_metadata["feature_dimension"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and extract merged Qwen outputs with one checkpoint load.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--subjects-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = postprocess_merged_fold(
        args.config,
        stage=args.stage,
        fold=args.fold,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        subjects_per_class=args.subjects_per_class,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
