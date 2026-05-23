from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.aggregate import (
    aggregate_generation_predictions,
    aggregate_likelihood_predictions,
    aggregate_original_teacher_forced_predictions,
)
from src.data.build_manifest import build_for_config
from src.data.runtime import (
    build_examples,
    filter_rows_by_subjects,
    load_audio_array,
    load_manifest_rows,
)
from src.model.qwen2audio_lora import (
    build_generation_config,
    load_model_for_inference,
    load_processor,
    prepare_model_for_evaluation,
)
from src.utils import (
    PREDICTION_MODE_GENERATION,
    PREDICTION_MODE_LIKELIHOOD,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    configure_logging,
    ensure_dir,
    evaluation_protocol_name,
    get_logger,
    internal_label_text_from_int,
    label_text_from_int,
    load_yaml_with_overrides,
    log_resolved_config,
    parse_internal_label_text,
    read_json,
    resolve_label_config,
    resolve_metadata_paths,
    resolve_model_name_or_path,
    resolve_prediction_mode,
    resolve_project_path,
    save_json,
    save_yaml,
    write_jsonl,
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
    processor_kwargs = {
        "text": text,
        "audio": audio,
        "return_tensors": "pt",
        "padding": False,
    }
    if audio is not None:
        processor_kwargs["sampling_rate"] = int(processor.feature_extractor.sampling_rate)
    inputs = processor(**processor_kwargs)
    return {key: value.to(device) for key, value in inputs.items()}


def _base_sample_row(example: dict[str, Any], checkpoint_name: str, backend_name: str) -> dict[str, Any]:
    return {
        "checkpoint_name": checkpoint_name,
        "prediction_backend": backend_name,
        "evaluation_protocol_name": evaluation_protocol_name(backend_name),
        "subject_id": example["subject_id"],
        "sample_id": example["sample_id"],
        "label": int(example["label"]),
        "label_text": example["label_text"],
        "internal_label_text": example["internal_label_text"],
    }


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
    generation_config = build_generation_config(config)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    continuation = generated[0, input_len:]
    return processor.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _predict_sample_likelihood(model, processor, example: dict[str, Any], device: torch.device, silence_audio: bool, checkpoint_name: str) -> dict[str, Any]:
    dep_score = score_candidate_label(
        model,
        processor,
        example,
        internal_label_text_from_int(example["config"], 1),
        device,
        silence_audio,
    )
    non_score = score_candidate_label(
        model,
        processor,
        example,
        internal_label_text_from_int(example["config"], 0),
        device,
        silence_audio,
    )
    likelihood_pred = 1 if dep_score > non_score else 0
    return {
        **_base_sample_row(example, checkpoint_name, PREDICTION_MODE_LIKELIHOOD),
        "likelihood_prediction": likelihood_pred,
        "likelihood_prediction_text": label_text_from_int(likelihood_pred),
        "dep_score": dep_score,
        "non_score": non_score,
    }


def _predict_sample_generation(
    model,
    processor,
    example: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    silence_audio: bool,
    checkpoint_name: str,
) -> dict[str, Any]:
    generation_text = generate_label_text(model, processor, example, config, device, silence_audio)
    parsed_generation = parse_internal_label_text(generation_text, config)
    return {
        **_base_sample_row(example, checkpoint_name, PREDICTION_MODE_GENERATION),
        "generation_text": generation_text,
        "parsed_prediction": parsed_generation if parsed_generation is not None else "",
        "generation_prediction_text": label_text_from_int(parsed_generation) if parsed_generation in (0, 1) else "INVALID",
    }


def _predict_sample_original_teacher_forced(
    model,
    processor,
    example: dict[str, Any],
    device: torch.device,
    silence_audio: bool,
    checkpoint_name: str,
) -> dict[str, Any]:
    prompt_inputs = _processor_inputs(processor, example, example["prompt_text"], device, silence_audio)
    full_text = example["prompt_text"] + example["internal_label_text"]
    full_inputs = _processor_inputs(processor, example, full_text, device, silence_audio)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    target_ids = full_inputs["input_ids"][0, prompt_len:]
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits[0]
        selected_logits = logits[prompt_len - 1 : full_inputs["input_ids"].shape[1] - 1]
        predicted_token_ids = torch.argmax(selected_logits, dim=-1)
    used_len = int(min(target_ids.shape[0], predicted_token_ids.shape[0]))
    gold_label_ids = target_ids[:used_len]
    predicted_label_ids = predicted_token_ids[:used_len]
    gold_label_text = processor.decode(gold_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    predicted_label_text = processor.decode(
        predicted_label_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    parsed_prediction = parse_internal_label_text(predicted_label_text, example["config"])
    return {
        **_base_sample_row(example, checkpoint_name, PREDICTION_MODE_ORIGINAL_TEACHER_FORCED),
        "teacher_forced_gold_text": gold_label_text,
        "teacher_forced_decoded_text": predicted_label_text,
        "teacher_forced_prediction": parsed_prediction if parsed_prediction is not None else "",
        "teacher_forced_prediction_text": (
            label_text_from_int(parsed_prediction) if parsed_prediction in (0, 1) else "INVALID"
        ),
        "teacher_forced_valid": parsed_prediction in (0, 1),
    }


def _prediction_backend(mode: str):
    if mode == PREDICTION_MODE_LIKELIHOOD:
        return _predict_sample_likelihood
    if mode == PREDICTION_MODE_GENERATION:
        return _predict_sample_generation
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return _predict_sample_original_teacher_forced
    raise ValueError(f"Unsupported prediction backend: {mode}")


def _aggregate_predictions(mode: str, sample_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == PREDICTION_MODE_LIKELIHOOD:
        return aggregate_likelihood_predictions(sample_rows)
    if mode == PREDICTION_MODE_GENERATION:
        return aggregate_generation_predictions(sample_rows)
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return aggregate_original_teacher_forced_predictions(sample_rows)
    raise ValueError(f"Unsupported prediction backend: {mode}")


def _metrics_filename_for_mode(mode: str) -> str:
    if mode == PREDICTION_MODE_LIKELIHOOD:
        return "metrics_likelihood.json"
    if mode == PREDICTION_MODE_GENERATION:
        return "metrics_generation.json"
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return "metrics_original_teacher_forced.json"
    raise ValueError(f"Unsupported prediction backend: {mode}")


def _log_invalid_prediction_preview(mode: str, subject_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    invalid_subjects = [row for row in subject_rows if row.get("prediction_text") == "INVALID"]
    if invalid_subjects:
        preview_subjects = ", ".join(
            f"{row['subject_id']}({row['label_text']})" for row in invalid_subjects[:12]
        )
        LOGGER.info(
            "Invalid subject preview backend=%s | count=%s | subjects=%s",
            mode,
            len(invalid_subjects),
            preview_subjects,
        )

    if mode != PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return

    invalid_samples = [row for row in sample_rows if row.get("teacher_forced_prediction_text") == "INVALID"]
    if not invalid_samples:
        return

    preview_chunks = []
    for row in invalid_samples[:20]:
        preview_chunks.append(
            f"{row['sample_id']} gold={row['label_text']} decoded={row.get('teacher_forced_decoded_text', '')!r}"
        )
    LOGGER.info(
        "Invalid teacher-forced sample preview backend=%s | count=%s | samples=%s",
        mode,
        len(invalid_samples),
        " || ".join(preview_chunks),
    )


def _invalid_sample_rows(mode: str, sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if mode == PREDICTION_MODE_GENERATION:
        return [row for row in sample_rows if row.get("generation_prediction_text") == "INVALID"]
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return [row for row in sample_rows if row.get("teacher_forced_prediction_text") == "INVALID"]
    return []


def evaluate_examples(
    model,
    processor,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path,
    checkpoint_name: str,
    sample_prediction_mode: str | None = None,
) -> dict[str, Any]:
    mode = resolve_prediction_mode(config, sample_prediction_mode)
    protocol_name = evaluation_protocol_name(mode)
    output_dir = ensure_dir(output_dir)
    prepare_model_for_evaluation(model)
    device = next(model.parameters()).device
    silence_audio = bool(config["data"].get("silence_audio", False))
    sample_rows: list[dict[str, Any]] = []
    total_examples = len(examples)
    progress_interval = max(1, int(config["evaluation"].get("progress_log_interval", 100)))
    predict_sample = _prediction_backend(mode)

    LOGGER.info(
        "Starting evaluation checkpoint=%s | samples=%s | backend=%s | protocol=%s",
        checkpoint_name,
        total_examples,
        mode,
        protocol_name,
    )
    for example_index, example in enumerate(examples, start=1):
        example = dict(example)
        example["config"] = config
        if mode == PREDICTION_MODE_LIKELIHOOD:
            row = predict_sample(model, processor, example, device, silence_audio, checkpoint_name)
        elif mode == PREDICTION_MODE_GENERATION:
            row = predict_sample(model, processor, example, config, device, silence_audio, checkpoint_name)
        else:
            row = predict_sample(model, processor, example, device, silence_audio, checkpoint_name)
        sample_rows.append(row)
        if example_index % progress_interval == 0 or example_index == total_examples:
            LOGGER.info(
                "Evaluation progress checkpoint=%s | backend=%s | %s/%s samples processed",
                checkpoint_name,
                mode,
                example_index,
                total_examples,
            )

    subject_rows, subject_metrics = _aggregate_predictions(mode, sample_rows)
    metrics_payload = dict(subject_metrics)
    metrics_payload["checkpoint_name"] = checkpoint_name

    sample_csv_path = output_dir / "predictions_sample_level.csv"
    subject_csv_path = output_dir / "predictions_subject_level.csv"
    sample_jsonl_path = output_dir / "predictions_sample_level.jsonl"
    invalid_jsonl_path = output_dir / "predictions_invalid_sample_level.jsonl"
    _write_csv(sample_rows, sample_csv_path)
    _write_csv(subject_rows, subject_csv_path)
    write_jsonl(sample_rows, sample_jsonl_path)
    invalid_rows = _invalid_sample_rows(mode, sample_rows)
    write_jsonl(invalid_rows, invalid_jsonl_path)
    save_json(metrics_payload, output_dir / _metrics_filename_for_mode(mode))
    save_json(
        {
            "prediction_backend": mode,
            "evaluation_protocol_name": protocol_name,
            "aggregation_level": "subject",
            "binary_strict_confusion_matrix": metrics_payload["binary_strict_confusion_matrix"],
            "diagnostic_three_class_labels": metrics_payload["diagnostic_three_class_labels"],
            "diagnostic_three_class_confusion_matrix": metrics_payload["diagnostic_three_class_confusion_matrix"],
        },
        output_dir / "confusion_matrix.json",
    )

    LOGGER.info(
        "Finished evaluation checkpoint=%s | backend=%s | ACC=%.6f F1=%.6f Precision=%.6f Recall=%.6f",
        checkpoint_name,
        mode,
        float(metrics_payload["accuracy"]),
        float(metrics_payload["positive_f1"]),
        float(metrics_payload["precision"]),
        float(metrics_payload["recall"]),
    )
    LOGGER.info(
        "Validation subject counts checkpoint=%s | backend=%s | predicted_depressed=%s predicted_non_depressed=%s predicted_invalid=%s",
        checkpoint_name,
        mode,
        int(metrics_payload["predicted_depressed_subjects"]),
        int(metrics_payload["predicted_non_depressed_subjects"]),
        int(metrics_payload["predicted_invalid_subjects"]),
    )
    LOGGER.info(
        "Validation true subject counts checkpoint=%s | backend=%s | true_depressed=%s true_non_depressed=%s",
        checkpoint_name,
        mode,
        int(metrics_payload["true_depressed_subjects"]),
        int(metrics_payload["true_non_depressed_subjects"]),
    )
    LOGGER.info(
        "Validation output files checkpoint=%s | backend=%s | subject_csv=%s sample_csv=%s sample_jsonl=%s invalid_sample_jsonl=%s invalid_sample_count=%s",
        checkpoint_name,
        mode,
        subject_csv_path,
        sample_csv_path,
        sample_jsonl_path,
        invalid_jsonl_path,
        len(invalid_rows),
    )
    if int(metrics_payload.get("invalid_subjects", 0)) > 0:
        LOGGER.info(
            "Validation valid-only metrics checkpoint=%s | backend=%s | valid_subjects=%s ACC=%.6f F1=%.6f Precision=%.6f Recall=%.6f",
            checkpoint_name,
            mode,
            int(metrics_payload["num_valid_subject_predictions"]),
            float(metrics_payload["valid_only_accuracy"]),
            float(metrics_payload["valid_only_positive_f1"]),
            float(metrics_payload["valid_only_precision"]),
            float(metrics_payload["valid_only_recall"]),
        )
    _log_invalid_prediction_preview(mode, subject_rows, sample_rows)
    return {
        "active_backend": mode,
        "evaluation_protocol_name": protocol_name,
        "sample_rows": sample_rows,
        "subject_rows": subject_rows,
        "backend_results": {
            mode: {
                "subject_rows": subject_rows,
                "subject_metrics": metrics_payload,
            }
        },
    }


def _load_metadata_or_build(config_path: str | Path, config: dict[str, Any], config_overrides: list[str] | None = None) -> dict[str, Any]:
    metadata_path = resolve_project_path(Path(config["output_dirs"]["split_dir"]) / f"{config['dataset']}_manifest_metadata.json")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if metadata_path.exists():
        metadata = resolve_metadata_paths(read_json(metadata_path))
        usable, reason = _metadata_artifacts_are_usable(metadata)
        if usable:
            return metadata
        LOGGER.warning("Refreshing stale metadata for %s: %s", config["dataset"], reason)
    if local_rank == 0:
        build_for_config(config_path, config_overrides)
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
    parser.add_argument("--sample_prediction_mode", default=None)
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE, using dot paths for nested keys.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml_with_overrides(args.config, args.config_overrides)
    log_resolved_config(
        LOGGER,
        base_config_path=args.config,
        config_overrides=args.config_overrides,
        resolved_config=config,
    )
    sample_prediction_mode = resolve_prediction_mode(config, args.sample_prediction_mode)
    LOGGER.info(
        "Standalone evaluation backend selected: %s | protocol=%s",
        sample_prediction_mode,
        evaluation_protocol_name(sample_prediction_mode),
    )
    metadata = _load_metadata_or_build(args.config, config, args.config_overrides)
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
    save_yaml(
        {
            "base_config_path": str(Path(args.config)),
            "config_overrides": list(args.config_overrides),
            "sample_prediction_mode": sample_prediction_mode,
            "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
            "resolved_model_name_or_path": model_name_or_path,
            "checkpoint_dir": str(Path(args.checkpoint_dir)),
            "config": config,
        },
        Path(output_dir) / "eval_config.yaml",
    )
    metrics = evaluate_examples(
        model,
        processor,
        examples,
        config,
        output_dir,
        checkpoint_name=Path(args.checkpoint_dir).name,
        sample_prediction_mode=sample_prediction_mode,
    )
    active_backend = metrics["active_backend"]
    LOGGER.info("Standalone evaluation complete: %s", metrics["backend_results"][active_backend]["subject_metrics"])


if __name__ == "__main__":
    main()
