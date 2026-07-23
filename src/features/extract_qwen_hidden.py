from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.emotion import load_emotion_cache, report_cache_coverage, resolve_missing_policy, use_emotion
from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator, load_prompt_audio
from src.model.runtime import load_model_for_inference, load_processor, resolve_processor_sampling_rate
from src.utils import read_json, save_json, sha256_file, sha256_jsonl_rows, sha256_text, write_jsonl


SUPPORTED_HIDDEN_SIZES = {3584, 4096}
POOLING_NAME = "last_valid_prompt_token"
CONDITION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def resolve_condition(value: str | None, input_modality: str | None, emotion_enabled: bool) -> str:
    condition = str(value or input_modality or "").strip()
    if not CONDITION_PATTERN.fullmatch(condition):
        raise ValueError(f"Invalid experiment condition: {condition!r}.")
    if emotion_enabled and condition == str(input_modality):
        raise ValueError("Emotion runs require an explicit condition distinct from input_modality.")
    return condition


def _emotion_provenance(
    config: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    partition_subject_ids: dict[str, list[str]],
    *,
    source: str | None,
    language: str | None,
) -> dict[str, Any] | None:
    if not use_emotion(config):
        if source or language:
            raise ValueError("Emotion source/language metadata was supplied for a non-emotion run.")
        return None
    if not source or not language:
        raise ValueError("Emotion runs require explicit source and language provenance.")
    data = config["data"]
    cache_path = _saved_path(data["emotion_cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(f"Saved emotion cache is unavailable: {cache_path}")
    caption_field = str(data.get("emotion_caption_field", "emotion_en"))
    cache = load_emotion_cache(cache_path, caption_field=caption_field)
    train_ids = set(partition_subject_ids["outer_train"])
    heldout_ids = set(partition_subject_ids["final_eval"])

    def coverage(subject_ids: set[str]) -> dict[str, int]:
        sample_ids = [
            str(row["sample_id"])
            for row in manifest_rows
            if str(row["subject_id"]) in subject_ids
        ]
        return report_cache_coverage(cache, sample_ids)

    policy = resolve_missing_policy(config)
    partition_coverage = {
        "outer_train": coverage(train_ids),
        "final_eval": coverage(heldout_ids),
    }
    missing_total = sum(
        stats["missing_row"] + stats["null_caption"] for stats in partition_coverage.values()
    )
    return {
        "enabled": True,
        "source": source,
        "language": language,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "caption_field": caption_field,
        "missing_policy": policy,
        "partition_coverage": partition_coverage,
        "fallback_caption_count": missing_total if policy == "neutral_fallback" else 0,
        "dropped_caption_count": missing_total if policy == "drop_emotion_line" else 0,
    }


def _saved_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    marker = "LLM-Depression/"
    text = str(path)
    if marker in text:
        candidate = PROJECT_ROOT / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return path


def _load_saved_run(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    fold_dir = checkpoint_dir.parent
    run_config_path = fold_dir / "run_config.yaml"
    split_path = fold_dir / "logs" / "split_used.json"
    if not run_config_path.exists() or not split_path.exists():
        raise FileNotFoundError("Checkpoint requires sibling run_config.yaml and logs/split_used.json.")
    import yaml

    saved = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise ValueError(f"Invalid saved run configuration: {run_config_path}")
    return saved, saved["config"], run_config_path, split_path


def _git_commit() -> str:
    # Cluster copies are synchronized without .git; an old checkout may still
    # leave a stale .git directory behind, so the captured sync provenance wins.
    provenance = PROJECT_ROOT / ".provenance" / "git_commit.txt"
    if provenance.exists():
        return provenance.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _decoder_hidden_size(model) -> int:
    configs = [getattr(model, "config", None)]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        configs.append(getattr(base_model, "config", None))
        configs.append(getattr(getattr(base_model, "model", None), "config", None))
    for config in configs:
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if hidden_size is not None:
            hidden_size = int(hidden_size)
            if hidden_size not in SUPPORTED_HIDDEN_SIZES:
                raise ValueError(
                    f"Unexpected decoder hidden size {hidden_size}; expected one of {sorted(SUPPORTED_HIDDEN_SIZES)}."
                )
            return hidden_size
    raise ValueError("Could not resolve the decoder hidden size from the loaded model configuration.")


def _partition_examples(
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    partition_subject_ids: dict[str, list[str]],
    fold: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for partition in ("outer_train", "final_eval"):
        ids = partition_subject_ids[partition]
        rows = filter_rows_by_subjects(manifest_rows, ids)
        examples = build_examples(rows, config, partition_name=partition, truncation_log_path=None)
        for example in examples:
            example["partition"] = partition
            example["fold"] = fold
        example_subjects = {str(item["subject_id"]) for item in examples}
        if example_subjects != set(ids):
            raise ValueError(
                f"{partition} example subjects differ from saved split: "
                f"missing={sorted(set(ids)-example_subjects)[:10]} extra={sorted(example_subjects-set(ids))[:10]}"
            )
        result[partition] = examples
    return result


def _resolve_subject_partitions(
    saved: dict[str, Any],
    config: dict[str, Any],
    split_payload: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    cv_protocol = str(
        saved.get("cv_protocol")
        or config.get("split", {}).get("cv_protocol")
        or ""
    )
    if cv_protocol == "train_val":
        train_sources = ("train_subject_ids",)
        heldout_source = "selection_subject_ids"
        evaluation_protocol = "table_aligned_outer_validation"
    else:
        train_sources = ("train_subject_ids", "selection_subject_ids")
        heldout_source = "final_eval_subject_ids"
        evaluation_protocol = "saved_final_evaluation"

    train_ids = sorted({
        str(subject_id)
        for source in train_sources
        for subject_id in split_payload.get(source, [])
    })
    heldout_ids = sorted({str(subject_id) for subject_id in split_payload.get(heldout_source, [])})
    if not train_ids:
        raise ValueError("Resolved outer_train subject set is empty.")
    if not heldout_ids:
        raise ValueError(
            f"Resolved final_eval subject set is empty (source={heldout_source}, cv_protocol={cv_protocol!r})."
        )
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise ValueError(f"Training/held-out subject overlap: {overlap[:10]}")
    partitions = {"outer_train": train_ids, "final_eval": heldout_ids}
    provenance = {
        "evaluation_protocol": evaluation_protocol,
        "saved_cv_protocol": cv_protocol or None,
        "partition_sources": {
            "outer_train": list(train_sources),
            "final_eval": [heldout_source],
        },
        "compatibility_note": (
            "The final_eval cache contains the saved outer-fold validation subjects."
            if cv_protocol == "train_val"
            else None
        ),
    }
    return partitions, provenance


def _validate_saved_split(
    saved: dict[str, Any],
    config: dict[str, Any],
    partition_subject_ids: dict[str, list[str]],
    fold: int,
) -> Path:
    split_metadata_path = _saved_path(saved["split_metadata_path"])
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Saved split metadata is unavailable: {split_metadata_path}")
    if saved.get("split_metadata_hash") and sha256_file(split_metadata_path) != saved["split_metadata_hash"]:
        raise ValueError("Current split metadata hash does not match the checkpoint's saved split hash.")
    train_ids = set(partition_subject_ids["outer_train"])
    heldout_ids = set(partition_subject_ids["final_eval"])
    split_metadata = read_json(split_metadata_path)
    if str(saved.get("split_mode", "")) == "cv":
        fold_payload = split_metadata[str(fold)] if str(fold) in split_metadata else split_metadata[fold]
        expected_train = set(fold_payload["outer_train_subject_ids"])
        expected_heldout = set(fold_payload["final_eval_subject_ids"])
    else:
        dev_partitions = set(config["split"].get("dev_pool_partitions") or [
            config["split"]["train_partition"],
            config["split"]["selection_partition"],
        ])
        final_partition = str(config["split"]["final_eval_partition"])
        expected_train = {
            str(row["subject_id"]) for row in split_metadata if str(row["partition"]) in dev_partitions
        }
        expected_heldout = {
            str(row["subject_id"]) for row in split_metadata if str(row["partition"]) == final_partition
        }
    if train_ids != expected_train or heldout_ids != expected_heldout:
        raise ValueError("Checkpoint split_used.json does not match its hashed split metadata.")
    return split_metadata_path


def _extract_partition(
    *,
    model,
    processor,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    output_dir: Path,
    partition: str,
    max_examples: int | None,
    expected_hidden_size: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    sampling_rate = resolve_processor_sampling_rate(processor)
    collator = PromptOnlyExtractionCollator(processor)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    mask_sources: dict[str, int] = {}
    determinism_max_abs_diff: float | None = None
    selected = examples[:max_examples] if max_examples else examples
    seen_samples: set[str] = set()
    subject_labels: dict[str, int] = {}
    for index, raw_example in enumerate(selected, start=1):
        example = load_prompt_audio(raw_example, sampling_rate, bool(config["data"].get("silence_audio", False)))
        model_inputs, metadata_rows = collator([example])
        metadata = metadata_rows[0]
        prompt_text = metadata.pop("prompt_text")
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        if "labels" in model_inputs:
            raise AssertionError("Gold labels must never be passed to Qwen during extraction.")
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                labels=None,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        output_mask = getattr(outputs, "attention_mask", None)
        mask, mask_source = aligned_attention_mask(hidden, model_inputs["attention_mask"], output_mask)
        vector = last_valid_token(hidden, mask).cpu().numpy()[0].astype(np.float32, copy=False)
        if vector.shape != (expected_hidden_size,):
            raise ValueError(f"Expected {expected_hidden_size} features, got {vector.shape}.")
        if not bool(np.isfinite(vector).all()):
            raise ValueError(f"Non-finite vector for sample {metadata['sample_id']}.")
        if index == 1:
            with torch.inference_mode():
                repeated_outputs = model(
                    **model_inputs,
                    labels=None,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            repeated_hidden = repeated_outputs.hidden_states[-1]
            repeated_mask, _ = aligned_attention_mask(
                repeated_hidden,
                model_inputs["attention_mask"],
                getattr(repeated_outputs, "attention_mask", None),
            )
            repeated_vector = (
                last_valid_token(repeated_hidden, repeated_mask).cpu().numpy()[0].astype(np.float32, copy=False)
            )
            determinism_max_abs_diff = float(np.max(np.abs(vector - repeated_vector)))
            if not np.allclose(vector, repeated_vector, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"Determinism check failed for {metadata['sample_id']}: "
                    f"max_abs_diff={determinism_max_abs_diff}"
                )
        sample_id = metadata["sample_id"]
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample ID within {partition}: {sample_id}")
        seen_samples.add(sample_id)
        subject_id = metadata["subject_id"]
        label = int(metadata["label"])
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        subject_labels[subject_id] = label
        metadata.update({
            "prompt_sha256": sha256_text(prompt_text),
            "hidden_layer": "final",
            "pooling": POOLING_NAME,
            "vector_dimension": expected_hidden_size,
            "vector_dtype": "float32",
            "mask_source": mask_source,
            "checkpoint": str(checkpoint_dir),
        })
        rows.append(metadata)
        vectors.append(vector)
        mask_sources[mask_source] = mask_sources.get(mask_source, 0) + 1
        if index % 25 == 0 or index == len(selected):
            print(f"{partition}: extracted {index}/{len(selected)}", flush=True)
    matrix = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, expected_hidden_size), np.float32)
    )
    np.savez_compressed(output_dir / f"{partition}.npz", vectors=matrix)
    write_jsonl(rows, output_dir / f"{partition}_rows.jsonl")
    return {
        "rows": len(rows),
        "subjects": len(subject_labels),
        "mask_sources": mask_sources,
        "determinism_rtol": 1e-5,
        "determinism_atol": 1e-5,
        "determinism_max_abs_diff": determinism_max_abs_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract prompt-only final Qwen2 hidden vectors.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--manifest-path", type=Path, help="Relocated copy of the saved manifest; hash must match.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--condition", help="Unique experiment condition used in metadata and output grouping.")
    parser.add_argument("--emotion-source", help="Predeclared source label for an emotion cache.")
    parser.add_argument("--emotion-language", help="Predeclared language label for emotion captions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved, config, run_config_path, split_path = _load_saved_run(checkpoint_dir)
    fold = int(saved["fold"])
    if checkpoint_dir.name != "best_model":
        raise ValueError("Primary experiment requires the fold-specific best_model checkpoint.")
    split_payload = read_json(split_path)
    partition_subject_ids, evaluation_provenance = _resolve_subject_partitions(
        saved, config, split_payload
    )
    split_metadata_path = _validate_saved_split(
        saved, config, partition_subject_ids, fold
    )
    manifest_path = args.manifest_path.resolve() if args.manifest_path else _saved_path(saved["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest is unavailable: {manifest_path}")
    manifest_rows = load_manifest_rows(manifest_path)
    canonical_manifest_hash = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and canonical_manifest_hash != saved["manifest_hash"]:
        raise ValueError("Current manifest hash does not match the checkpoint's saved manifest hash.")
    condition = resolve_condition(args.condition, saved.get("input_modality"), use_emotion(config))
    emotion_provenance = _emotion_provenance(
        config,
        manifest_rows,
        partition_subject_ids,
        source=args.emotion_source,
        language=args.emotion_language,
    )
    examples = _partition_examples(manifest_rows, config, partition_subject_ids, fold)
    model_name = args.model_name_or_path or saved.get("resolved_model_name_or_path") or config["model_name_or_path"]
    processor = load_processor(checkpoint_dir, config)
    model = load_model_for_inference(str(model_name), checkpoint_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and bool(config.get("training", {}).get("bf16", False)) else None
    model.to(device=device, dtype=dtype)
    model.eval()
    expected_hidden_size = _decoder_hidden_size(model)
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
            expected_hidden_size=expected_hidden_size,
        )
    metadata = {
        "dataset": config["dataset"],
        "condition": condition,
        "input_modality": saved.get("input_modality"),
        "fold": fold,
        "checkpoint_type": "best_model",
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_sha256": sha256_file(checkpoint_dir / "adapter_config.json"),
        "adapter_sha256": sha256_file(checkpoint_dir / "adapter_model.safetensors"),
        "base_model": str(model_name),
        "saved_run_config": str(run_config_path),
        "saved_run_config_sha256": sha256_file(run_config_path),
        "saved_split": str(split_path),
        "saved_split_sha256": sha256_file(split_path),
        "evaluation_provenance": evaluation_provenance,
        "split_metadata": str(split_metadata_path),
        "split_metadata_sha256": sha256_file(split_metadata_path),
        "manifest": str(manifest_path),
        "manifest_sha256": canonical_manifest_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "emotion_provenance": emotion_provenance,
        "hidden_layer": "final",
        "pooling": POOLING_NAME,
        "vector_dimension": expected_hidden_size,
        "vector_dtype": "float32",
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
    save_json(metadata, output_dir / "extraction_metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
