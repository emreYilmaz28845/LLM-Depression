from __future__ import annotations

import argparse
import csv
import json
import sys
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.aggregate import aggregate_generation_predictions, aggregate_likelihood_predictions, parse_generation_label
from src.data.build_manifest import build_for_config
from src.data.runtime import (
    build_examples,
    build_subject_label_map,
    filter_rows_by_subjects,
    load_audio_array,
    load_manifest_rows,
)
from src.data.split_utils import deterministic_inner_split
from src.model.qwen2audio_lora import load_model_for_inference, load_processor
from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    label_text_from_int,
    load_yaml,
    read_json,
    resolve_metadata_paths,
    resolve_model_name_or_path,
    resolve_project_path,
    save_json,
)


LOGGER = get_logger(__name__)


def _metadata_artifacts_are_usable(metadata: dict[str, Any]) -> tuple[bool, str]:
    manifest_path = Path(metadata["manifest_path"])
    if not manifest_path.exists():
        return False, f"manifest_missing:{manifest_path}"
    for key, value in metadata.items():
        if key.endswith("_path") and key != "manifest_path" and isinstance(value, str) and value:
            if not Path(value).exists():
                return False, f"artifact_missing:{key}:{value}"
    preview_rows = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                preview_rows.append(json.loads(line))
                if len(preview_rows) >= 3:
                    break
    except Exception as exc:
        return False, f"manifest_read_error:{exc}"
    if not preview_rows:
        return False, "manifest_empty"
    for row in preview_rows:
        audio_paths = row.get("audio_paths") or ([row["audio_path"]] if row.get("audio_path") else [])
        for audio_path in audio_paths:
            if audio_path and not Path(audio_path).exists():
                return False, f"stale_audio_path:{audio_path}"
    return True, "ok"


def _wait_for_usable_metadata(metadata_path: Path, timeout_seconds: int = 600) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_reason = "metadata_not_ready"
    while time.time() < deadline:
        if metadata_path.exists():
            metadata = resolve_metadata_paths(read_json(metadata_path))
            usable, reason = _metadata_artifacts_are_usable(metadata)
            if usable:
                return metadata
            last_reason = reason
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for usable metadata at {metadata_path}. Last reason: {last_reason}")


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_example_audio(example: dict[str, Any], sampling_rate: int, silence_audio: bool) -> list:
    return [
        load_audio_array(audio_path, sampling_rate, max_seconds, silence_audio)
        for audio_path, max_seconds in zip(example["audio_paths"], example["audio_clip_seconds"])
    ]


def _processor_inputs(processor, example: dict[str, Any], text: str, device: torch.device, silence_audio: bool):
    audio_arrays = _load_example_audio(example, processor.feature_extractor.sampling_rate, silence_audio)
    audio = audio_arrays if audio_arrays else None
    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=False)
    return {key: value.to(device) for key, value in inputs.items()}


def score_candidate_label(
    model,
    processor,
    example: dict[str, Any],
    candidate_label: str,
    device: torch.device,
    silence_audio: bool,
) -> float:
    prompt_inputs = _processor_inputs(processor, example, example["prompt_text"], device, silence_audio)
    full_text = example["prompt_text"] + candidate_label
    full_inputs = _processor_inputs(processor, example, full_text, device, silence_audio)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    target_ids = full_inputs["input_ids"][0, prompt_len:]
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits[0]
        selected_logits = logits[prompt_len - 1 : full_inputs["input_ids"].shape[1] - 1]
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def generate_label_text(
    model,
    processor,
    example: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    silence_audio: bool,
) -> str:
    inputs = _processor_inputs(processor, example, example["prompt_text"], device, silence_audio)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(config["evaluation"]["generation_max_new_tokens"]),
            num_beams=int(config["evaluation"]["num_beams"]),
            do_sample=bool(config["evaluation"]["do_sample"]),
        )
    continuation = generated[0, input_len:]
    return processor.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def evaluate_examples(
    model,
    processor,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path,
    checkpoint_name: str,
    run_generation: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    device = next(model.parameters()).device
    silence_audio = bool(config["data"].get("silence_audio", False))
    sample_rows: list[dict[str, Any]] = []
    invalid_generation_rows: list[dict[str, Any]] = []

    for example in examples:
        dep_score = score_candidate_label(model, processor, example, "Depressed", device, silence_audio)
        non_score = score_candidate_label(model, processor, example, "Non-depressed", device, silence_audio)
        likelihood_pred = 1 if dep_score > non_score else 0
        if run_generation:
            generation_text = generate_label_text(model, processor, example, config, device, silence_audio)
            parsed_generation = parse_generation_label(generation_text)
        else:
            generation_text = ""
            parsed_generation = None
        row = {
            "checkpoint_name": checkpoint_name,
            "subject_id": example["subject_id"],
            "sample_id": example["sample_id"],
            "label": int(example["label"]),
            "label_text": example["label_text"],
            "likelihood_prediction": likelihood_pred,
            "likelihood_prediction_text": label_text_from_int(likelihood_pred),
            "dep_score": dep_score,
            "non_score": non_score,
            "generation_text": generation_text,
            "parsed_prediction": parsed_generation if parsed_generation is not None else "",
            "generation_prediction_text": label_text_from_int(parsed_generation) if parsed_generation in (0, 1) else "INVALID",
        }
        sample_rows.append(row)
        if run_generation and parsed_generation is None:
            invalid_generation_rows.append(row)

    likelihood_subject_rows, likelihood_metrics = aggregate_likelihood_predictions(sample_rows)
    if run_generation:
        generation_subject_rows, generation_metrics = aggregate_generation_predictions(sample_rows)
    else:
        generation_subject_rows = [
            {
                "subject_id": row["subject_id"],
                "prediction": "",
                "prediction_text": "",
                "num_valid_predictions": 0,
            }
            for row in likelihood_subject_rows
        ]
        generation_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "positive_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "support_negative": 0,
            "support_positive": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "num_subjects": len(likelihood_subject_rows),
            "invalid_subjects": 0,
            "invalid_generations": 0,
        }

    merged_subject_rows: list[dict[str, Any]] = []
    generation_by_subject = {row["subject_id"]: row for row in generation_subject_rows}
    for likelihood_row in likelihood_subject_rows:
        subject_id = likelihood_row["subject_id"]
        generation_row = generation_by_subject[subject_id]
        merged_subject_rows.append(
            {
                **likelihood_row,
                "likelihood_prediction_text": likelihood_row["prediction_text"],
                "generation_prediction": generation_row["prediction"],
                "generation_prediction_text": generation_row["prediction_text"],
                "generation_num_valid_predictions": generation_row["num_valid_predictions"],
            }
        )

    _write_csv(sample_rows, output_dir / "predictions_sample_level.csv")
    _write_csv(merged_subject_rows, output_dir / "predictions_subject_level.csv")
    save_json(likelihood_metrics, output_dir / "metrics_likelihood.json")
    save_json(generation_metrics, output_dir / "metrics_generation.json")
    save_json(
        {
            "likelihood": likelihood_metrics["confusion_matrix"],
            "generation": generation_metrics["confusion_matrix"],
        },
        output_dir / "confusion_matrix.json",
    )
    with (output_dir / "invalid_generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in invalid_generation_rows:
            handle.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")

    return {
        "sample_rows": sample_rows,
        "subject_rows": merged_subject_rows,
        "likelihood": {
            "subject_metrics": likelihood_metrics,
        },
        "generation": {
            "subject_metrics": generation_metrics,
        },
    }


def _load_metadata_or_build(config_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = resolve_project_path(Path(config["output_dirs"]["split_dir"]) / f"{config['dataset']}_manifest_metadata.json")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if metadata_path.exists():
        metadata = resolve_metadata_paths(read_json(metadata_path))
        usable, reason = _metadata_artifacts_are_usable(metadata)
        if usable:
            return metadata
        LOGGER.warning("Refreshing stale metadata for %s: %s", config["dataset"], reason)
    if local_rank == 0:
        build_for_config(config_path)
    return _wait_for_usable_metadata(metadata_path)


def _resolve_final_eval_subject_ids(config: dict[str, Any], metadata: dict[str, Any], fold: int) -> list[str]:
    dataset_name = str(config["dataset"]).lower()
    if dataset_name == "daic":
        partition_rows = read_json(metadata["subject_partition_path"])
        return sorted([row["subject_id"] for row in partition_rows if row["partition"] == config["split"]["final_eval_partition"]])
    folds = read_json(metadata["folds_path"])
    fold_payload = folds[str(fold)] if str(fold) in folds else folds[fold]
    return sorted(fold_payload["final_eval_subject_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Qwen2-Audio LoRA adapter checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    metadata = _load_metadata_or_build(args.config, config)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    final_eval_subject_ids = _resolve_final_eval_subject_ids(config, metadata, args.fold)
    final_eval_rows = filter_rows_by_subjects(manifest_rows, final_eval_subject_ids)
    examples = build_examples(final_eval_rows, config, partition_name="final_eval", truncation_log_path=None)

    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, config)
    processor = load_processor(args.checkpoint_dir)
    model = load_model_for_inference(model_name_or_path, args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output_dir = args.output_dir or (Path(args.checkpoint_dir) / "standalone_eval")
    metrics = evaluate_examples(model, processor, examples, config, output_dir, checkpoint_name=Path(args.checkpoint_dir).name)
    LOGGER.info("Standalone evaluation complete: %s", metrics["likelihood"]["subject_metrics"])


if __name__ == "__main__":
    main()
