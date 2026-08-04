from __future__ import annotations

import argparse
import csv
import hashlib
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
    aggregate_margin_predictions,
    aggregate_predictions,
    aggregate_response_subject_predictions,
)
from src.data.build_manifest import build_for_config, manifest_build_signature
from src.data.runtime import (
    build_examples,
    filter_rows_by_subjects,
    load_audio_array,
    load_manifest_rows,
)
from src.data.split_utils import (
    CV_PROTOCOL_TRAIN_VAL,
    SPLIT_MODE_CV,
    SPLIT_MODE_FIXED,
    SPLIT_MODE_FULL_TRAIN,
    read_fold_payload,
    resolve_cv_protocol,
    resolve_requested_split_mode,
    resolve_split_mode,
    subject_ids_for_partitions,
)
from src.model.runtime import (
    build_generation_config,
    load_model_for_inference,
    load_processor,
    resolve_processor_sampling_rate,
    prepare_model_for_evaluation,
)
from src.model.lora_common import resolve_lora_layer_selection
from src.utils import (
    AGGREGATION_LEVEL_SUBJECT,
    AGGREGATION_LEVEL_RESPONSE_SUBJECT,
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
    parse_generated_label_text,
    parse_internal_label_text,
    read_json,
    resolve_label_config,
    resolve_input_modality,
    resolve_metadata_paths,
    resolve_aggregation_level,
    resolve_model_name_or_path,
    resolve_prediction_mode,
    resolve_project_path,
    save_json,
    save_yaml,
    write_jsonl,
)


LOGGER = get_logger(__name__)


def _load_best_validation_result(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Load the validation metrics that selected ``checkpoint_dir``.

    Training stores these metrics at the fold level rather than inside the
    adapter directory. Evaluation must remain usable for external/legacy
    checkpoints, so missing or incomplete selection metadata is non-fatal.
    """
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path.parent / "logs" / "selected_checkpoint_selection_metrics.json"
    if not metadata_path.exists():
        LOGGER.warning(
            "Best-validation metadata not found for checkpoint=%s (expected %s).",
            checkpoint_path,
            metadata_path,
        )
        return None

    payload = read_json(metadata_path)
    component_metrics = payload.get("component_selection_metrics") or {}
    if not isinstance(component_metrics, dict):
        component_metrics = {}

    def metric(name: str) -> float | None:
        value = component_metrics.get(f"inner_val_{name}")
        if value is None:
            value = component_metrics.get(f"selection_{name}")
        return None if value is None else float(value)

    return {
        "source_path": str(metadata_path),
        "selected_epoch": int(payload.get("selected_epoch", component_metrics.get("epoch", -1))),
        "selection_split_name": component_metrics.get("selection_split_name", "validation"),
        "selection_metric": payload.get("selection_metric"),
        "selection_metric_mode": payload.get("selection_metric_mode"),
        "selection_metric_value": payload.get("selection_metric_value"),
        "prediction_backend": component_metrics.get("selection_prediction_backend"),
        "aggregation_level": component_metrics.get("selection_aggregation_level"),
        "loss": metric("loss"),
        "accuracy": metric("accuracy"),
        "positive_f1": metric("positive_f1"),
        "macro_f1": metric("macro_f1"),
        "precision": metric("precision"),
        "recall": metric("recall"),
    }


def _log_metric_summary(label: str, summary: dict[str, Any]) -> None:
    LOGGER.info(
        "%s | split=%s epoch=%s backend=%s aggregation=%s | "
        "ACC=%s positive_f1=%s macro_f1=%s Precision=%s Recall=%s loss=%s",
        label,
        summary.get("split_name", summary.get("selection_split_name", "unknown")),
        summary.get("selected_epoch", "n/a"),
        summary.get("prediction_backend", "unknown"),
        summary.get("aggregation_level", "unknown"),
        "n/a" if summary.get("accuracy") is None else f"{float(summary['accuracy']):.6f}",
        "n/a" if summary.get("positive_f1") is None else f"{float(summary['positive_f1']):.6f}",
        "n/a" if summary.get("macro_f1") is None else f"{float(summary['macro_f1']):.6f}",
        "n/a" if summary.get("precision") is None else f"{float(summary['precision']):.6f}",
        "n/a" if summary.get("recall") is None else f"{float(summary['recall']):.6f}",
        "n/a" if summary.get("loss") is None else f"{float(summary['loss']):.6f}",
    )


def _recommended_generation_max_new_tokens(processor, config: dict[str, Any]) -> int:
    labels_cfg = resolve_label_config(config)
    label_texts = (
        labels_cfg["internal_positive_label"],
        labels_cfg["internal_negative_label"],
    )
    candidate_prefixes = (
        "",
        "Answer: ",
        "Label: ",
        "Prediction: ",
        "The subject is ",
    )
    token_lengths: list[int] = []
    for prefix in candidate_prefixes:
        for label_text in label_texts:
            token_ids = processor.tokenizer(
                f"{prefix}{label_text}",
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            token_lengths.append(len(token_ids))
    return max(token_lengths, default=0) + 1


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


def _required_split_metadata_keys(config: dict[str, Any]) -> list[str]:
    requested_mode = resolve_requested_split_mode(config)
    if requested_mode == SPLIT_MODE_CV:
        return ["folds_path"]
    if requested_mode in {SPLIT_MODE_FIXED, SPLIT_MODE_FULL_TRAIN}:
        return ["subject_partition_path"]
    split_cfg = config.get("split", {})
    if any(key in split_cfg for key in ("train_partition", "train_partitions", "selection_partition", "dev_pool_partitions")):
        return ["subject_partition_path"]
    return ["folds_path"]


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
    start_times = list(example.get("audio_start_times") or [None] * len(example["audio_paths"]))
    end_times = list(example.get("audio_end_times") or [None] * len(example["audio_paths"]))
    return [
        load_audio_array(audio_path, sampling_rate, max_seconds, silence_audio, start_time, end_time)
        for audio_path, max_seconds, start_time, end_time in zip(
            example["audio_paths"], example["audio_clip_seconds"], start_times, end_times
        )
    ]


def _processor_inputs(
    processor,
    example: dict[str, Any],
    text: str | list[str],
    device: torch.device,
    silence_audio: bool,
    *,
    audio_arrays: list | None = None,
    repeat_audio: int = 1,
    padding: bool = False,
):
    sampling_rate = resolve_processor_sampling_rate(processor)
    if audio_arrays is None and example["audio_paths"]:
        if sampling_rate is None:
            raise ValueError("Audio examples require a processor sampling rate.")
        audio_arrays = _load_example_audio(example, sampling_rate, silence_audio)
    audio_arrays = list(audio_arrays or [])
    if repeat_audio > 1:
        audio_arrays = [array for _ in range(int(repeat_audio)) for array in audio_arrays]
    audio = audio_arrays if audio_arrays else None
    processor_kwargs = {
        "text": text,
        "return_tensors": "pt",
        "padding": padding,
    }
    if audio is not None:
        processor_kwargs["audio"] = audio
        processor_kwargs["sampling_rate"] = int(sampling_rate)
    inputs = processor(**processor_kwargs)
    return {key: value.to(device) for key, value in inputs.items()}


def _base_sample_row(example: dict[str, Any], checkpoint_name: str, backend_name: str) -> dict[str, Any]:
    prompt_text = str(example.get("prompt_text", ""))
    transcript = str(example.get("transcript", ""))
    transcript_block = (
        f"The transcript of the subject's speech is:\n{transcript}\n\n"
    )
    prompt_context = prompt_text.replace(
        transcript_block,
        "The transcript of the subject's speech is:\n<TRANSCRIPT>\n\n",
        1,
    )
    config = example.get("config", {})
    row = {
        "checkpoint_name": checkpoint_name,
        "prediction_backend": backend_name,
        "evaluation_protocol_name": evaluation_protocol_name(backend_name),
        "subject_id": example["subject_id"],
        "sample_id": example["sample_id"],
        "label": int(example["label"]),
        "label_text": example["label_text"],
        "internal_label_text": example["internal_label_text"],
        "response_id": example.get("response_id", ""),
        "prompt_id": example.get("prompt_id", example.get("question_id", "")),
        "segment_index": example.get("segment_index", 0),
        "num_segments": example.get("num_segments", 1),
        "start_time": example.get("start_time", ""),
        "end_time": example.get("end_time", ""),
        "segment_duration": example.get("segment_duration", ""),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_context_sha256": hashlib.sha256(
            prompt_context.encode("utf-8")
        ).hexdigest(),
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "transcript_chars": len(transcript),
        "audio_text_transcript_scope": config.get("data", {}).get(
            "audio_text_transcript_scope", ""
        ),
    }
    evaluation = config.get("evaluation", {})
    if evaluation.get("subject_score_aggregation"):
        row["subject_score_aggregation"] = evaluation["subject_score_aggregation"]
    for key in ("chunk_id", "bundle_id", "bundle_chunk_ids", "bundle_coverage_count"):
        if key in example:
            row[key] = example[key]
    return row


def score_candidate_label(
    model,
    processor,
    example: dict[str, Any],
    candidate_label: str,
    device: torch.device,
    silence_audio: bool,
    audio_arrays: list | None = None,
) -> float:
    if audio_arrays is None:
        sampling_rate = resolve_processor_sampling_rate(processor)
        audio_arrays = (
            _load_example_audio(example, int(sampling_rate), silence_audio)
            if example["audio_paths"] and sampling_rate is not None
            else []
        )
    prompt_inputs = _processor_inputs(
        processor, example, example["prompt_text"], device, silence_audio,
        audio_arrays=audio_arrays,
    )
    full_text = example["prompt_text"] + candidate_label
    full_inputs = _processor_inputs(
        processor, example, full_text, device, silence_audio,
        audio_arrays=audio_arrays,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    target_ids = full_inputs["input_ids"][0, prompt_len:]
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits[0]
        selected_logits = logits[prompt_len - 1 : full_inputs["input_ids"].shape[1] - 1]
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def score_candidate_pair(
    model,
    processor,
    example: dict[str, Any],
    device: torch.device,
    silence_audio: bool,
) -> dict[str, Any]:
    """Score both labels in one forward and retain the gold-span argmax.

    Raw audio is decoded once. The processor expands the prompt once to obtain
    the label boundary and expands the paired full texts together. Qwen maps the
    flattened audio list to audio placeholders in text order.
    """
    sampling_rate = resolve_processor_sampling_rate(processor)
    if example["audio_paths"] and sampling_rate is None:
        raise ValueError("Audio examples require a processor sampling rate.")
    audio_arrays = (
        _load_example_audio(example, int(sampling_rate), silence_audio)
        if example["audio_paths"]
        else []
    )
    prompt_inputs = _processor_inputs(
        processor, example, example["prompt_text"], device, silence_audio,
        audio_arrays=audio_arrays,
    )
    prompt_len = int(prompt_inputs["attention_mask"].sum().item())
    labels = [
        internal_label_text_from_int(example["config"], 1),
        internal_label_text_from_int(example["config"], 0),
    ]
    full_inputs = _processor_inputs(
        processor,
        example,
        [example["prompt_text"] + label for label in labels],
        device,
        silence_audio,
        audio_arrays=audio_arrays,
        repeat_audio=2,
        padding=True,
    )
    with torch.inference_mode():
        logits = model(**full_inputs).logits
    scores: list[float] = []
    predicted_ids: list[torch.Tensor] = []
    target_ids: list[torch.Tensor] = []
    attention_mask = full_inputs["attention_mask"]
    for row_index in range(2):
        nonzero = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
        if nonzero.numel() < prompt_len:
            raise ValueError("Candidate input is shorter than its expanded prompt.")
        first = int(nonzero[0].item())
        full_len = int(nonzero.numel())
        target_start = first + prompt_len
        target_end = first + full_len
        row_target = full_inputs["input_ids"][row_index, target_start:target_end]
        selected = logits[row_index, target_start - 1:target_end - 1]
        log_probs = torch.log_softmax(selected, dim=-1)
        token_log_probs = log_probs.gather(-1, row_target.unsqueeze(-1)).squeeze(-1)
        scores.append(float(token_log_probs.mean().item()))
        predicted_ids.append(torch.argmax(selected, dim=-1))
        target_ids.append(row_target)
    gold_index = 0 if int(example["label"]) == 1 else 1
    return {
        "dep_score": scores[0],
        "non_score": scores[1],
        "gold_target_ids": target_ids[gold_index],
        "gold_predicted_ids": predicted_ids[gold_index],
        "audio_file_loads": len(audio_arrays),
        "model_forwards": 1,
    }


def _candidate_batching(config: dict[str, Any]) -> str:
    value = str(config.get("evaluation", {}).get("candidate_batching", "sequential")).strip().lower()
    if value not in {"sequential", "paired"}:
        raise ValueError("evaluation.candidate_batching must be sequential or paired.")
    return value


def _score_sequential_original(
    model, processor, example: dict[str, Any], device: torch.device, silence_audio: bool,
) -> dict[str, Any]:
    dep_score = score_candidate_label(
        model, processor, example, internal_label_text_from_int(example["config"], 1),
        device, silence_audio,
    )
    non_score = score_candidate_label(
        model, processor, example, internal_label_text_from_int(example["config"], 0),
        device, silence_audio,
    )
    prompt_inputs = _processor_inputs(processor, example, example["prompt_text"], device, silence_audio)
    full_inputs = _processor_inputs(
        processor, example, example["prompt_text"] + example["internal_label_text"],
        device, silence_audio,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    target_ids = full_inputs["input_ids"][0, prompt_len:]
    with torch.no_grad():
        logits = model(**full_inputs).logits[0]
    predicted = torch.argmax(
        logits[prompt_len - 1:full_inputs["input_ids"].shape[1] - 1], dim=-1
    )
    return {
        "dep_score": dep_score, "non_score": non_score,
        "gold_target_ids": target_ids, "gold_predicted_ids": predicted,
        "audio_file_loads": len(example.get("audio_paths") or []) * 6,
        "model_forwards": 3,
    }


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
    generation_config.max_new_tokens = max(
        int(generation_config.max_new_tokens),
        _recommended_generation_max_new_tokens(processor, config),
    )
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
    if _candidate_batching(example["config"]) == "paired":
        paired = score_candidate_pair(model, processor, example, device, silence_audio)
    else:
        dep_score = score_candidate_label(
            model, processor, example, internal_label_text_from_int(example["config"], 1), device, silence_audio
        )
        non_score = score_candidate_label(
            model, processor, example, internal_label_text_from_int(example["config"], 0), device, silence_audio
        )
        paired = {
            "dep_score": dep_score, "non_score": non_score, "model_forwards": 2,
            "audio_file_loads": len(example.get("audio_paths") or []) * 4,
        }
    dep_score, non_score = paired["dep_score"], paired["non_score"]
    likelihood_pred = 1 if dep_score > non_score else 0
    return {
        **_base_sample_row(example, checkpoint_name, PREDICTION_MODE_LIKELIHOOD),
        "likelihood_prediction": likelihood_pred,
        "likelihood_prediction_text": label_text_from_int(likelihood_pred),
        "dep_score": dep_score,
        "non_score": non_score,
        "inference_model_forwards": paired["model_forwards"],
        "inference_audio_file_loads": paired["audio_file_loads"],
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
    parsed_generation = parse_generated_label_text(generation_text, config)
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
    paired = (
        score_candidate_pair(model, processor, example, device, silence_audio)
        if _candidate_batching(example["config"]) == "paired"
        else _score_sequential_original(model, processor, example, device, silence_audio)
    )
    dep_score, non_score = paired["dep_score"], paired["non_score"]
    target_ids = paired["gold_target_ids"]
    predicted_token_ids = paired["gold_predicted_ids"]
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
        "dep_score": dep_score,
        "non_score": non_score,
        "teacher_forced_margin": dep_score - non_score,
        "inference_model_forwards": paired["model_forwards"],
        "inference_audio_file_loads": paired["audio_file_loads"],
    }


def _prediction_backend(mode: str):
    if mode == PREDICTION_MODE_LIKELIHOOD:
        return _predict_sample_likelihood
    if mode == PREDICTION_MODE_GENERATION:
        return _predict_sample_generation
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return _predict_sample_original_teacher_forced
    raise ValueError(f"Unsupported prediction backend: {mode}")


def _metrics_filename_for_mode(mode: str) -> str:
    if mode == PREDICTION_MODE_LIKELIHOOD:
        return "metrics_likelihood.json"
    if mode == PREDICTION_MODE_GENERATION:
        return "metrics_generation.json"
    if mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED:
        return "metrics_original_teacher_forced.json"
    raise ValueError(f"Unsupported prediction backend: {mode}")


def _subject_metrics_filename_for_mode(mode: str) -> str:
    return f"metrics_subject_level_{mode}.json"


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
    aggregation_level = resolve_aggregation_level(config)
    protocol_name = evaluation_protocol_name(mode)
    output_dir = ensure_dir(output_dir)
    prepare_model_for_evaluation(model, config)
    device = next(model.parameters()).device
    silence_audio = bool(config["data"].get("silence_audio", False))
    sample_rows: list[dict[str, Any]] = []
    total_examples = len(examples)
    progress_interval = max(1, int(config["evaluation"].get("progress_log_interval", 100)))
    predict_sample = _prediction_backend(mode)

    LOGGER.info(
        "Starting evaluation checkpoint=%s | samples=%s | backend=%s | aggregation_level=%s | protocol=%s",
        checkpoint_name,
        total_examples,
        mode,
        aggregation_level,
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

    response_rows: list[dict[str, Any]] = []
    response_metrics: dict[str, Any] | None = None
    if aggregation_level == AGGREGATION_LEVEL_RESPONSE_SUBJECT:
        prediction_field = {
            PREDICTION_MODE_LIKELIHOOD: "likelihood_prediction",
            PREDICTION_MODE_GENERATION: "parsed_prediction",
            PREDICTION_MODE_ORIGINAL_TEACHER_FORCED: "teacher_forced_prediction",
        }[mode]
        response_rows, response_metrics, subject_rows, subject_metrics = (
            aggregate_response_subject_predictions(
                sample_rows,
                prediction_field=prediction_field,
                backend_name=mode,
                invalid_as_wrong=mode == PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                score_average=(
                    str(
                        config.get("evaluation", {}).get(
                            "hierarchical_score_aggregation", ""
                        )
                    ).lower()
                    == "mean"
                ),
            )
        )
        headline_rows, headline_metrics = subject_rows, subject_metrics
    else:
        headline_rows, headline_metrics, subject_rows, subject_metrics = aggregate_predictions(
            sample_rows,
            mode=mode,
            aggregation_level=aggregation_level,
        )
    headline_metrics_payload = dict(headline_metrics)
    headline_metrics_payload["checkpoint_name"] = checkpoint_name
    subject_metrics_payload = dict(subject_metrics)
    subject_metrics_payload["checkpoint_name"] = checkpoint_name

    secondary_aggregations: dict[str, Any] = {}
    requested_secondary = config.get("evaluation", {}).get("secondary_aggregations") or []
    if requested_secondary and sample_rows and all(
        row.get("dep_score") is not None and row.get("non_score") is not None
        for row in sample_rows
    ):
        expected_subjects = {str(row["subject_id"]) for row in sample_rows}
        for method in requested_secondary:
            method = str(method)
            secondary_rows, secondary_metrics = aggregate_margin_predictions(sample_rows, method)
            observed_subjects = {str(row["subject_id"]) for row in secondary_rows}
            if observed_subjects != expected_subjects or len(secondary_rows) != len(observed_subjects):
                raise ValueError(
                    f"Secondary aggregation {method!r} did not produce exactly one row per subject."
                )
            secondary_metrics = dict(secondary_metrics)
            secondary_metrics["checkpoint_name"] = checkpoint_name
            secondary_metrics["aggregation_method"] = method
            write_jsonl(
                secondary_rows,
                output_dir / f"predictions_subject_level_{method}.jsonl",
            )
            save_json(secondary_metrics, output_dir / f"metrics_secondary_{method}.json")
            secondary_aggregations[method] = {
                "metrics": secondary_metrics,
                "prediction_path": str(output_dir / f"predictions_subject_level_{method}.jsonl"),
            }
    elif requested_secondary:
        secondary_aggregations = {"status": "not_applicable"}
    if requested_secondary:
        save_json(
            {
                "requested": [str(method) for method in requested_secondary],
                "results": secondary_aggregations,
            },
            output_dir / "secondary_aggregations.json",
        )

    sample_csv_path = output_dir / "predictions_sample_level.csv"
    headline_csv_path = output_dir / "predictions_headline_level.csv"
    subject_csv_path = output_dir / "predictions_subject_level.csv"
    response_csv_path = output_dir / "predictions_response_level.csv"
    sample_jsonl_path = output_dir / "predictions_sample_level.jsonl"
    invalid_jsonl_path = output_dir / "predictions_invalid_sample_level.jsonl"
    _write_csv(sample_rows, sample_csv_path)
    _write_csv(headline_rows, headline_csv_path)
    _write_csv(subject_rows, subject_csv_path)
    if response_rows:
        _write_csv(response_rows, response_csv_path)
    write_jsonl(sample_rows, sample_jsonl_path)
    if response_rows:
        write_jsonl(response_rows, output_dir / "predictions_response_level.jsonl")
    invalid_rows = _invalid_sample_rows(mode, sample_rows)
    write_jsonl(invalid_rows, invalid_jsonl_path)
    save_json(headline_metrics_payload, output_dir / _metrics_filename_for_mode(mode))
    if response_metrics is not None:
        save_json(response_metrics, output_dir / f"metrics_response_level_{mode}.json")
    if aggregation_level != AGGREGATION_LEVEL_SUBJECT:
        save_json(subject_metrics_payload, output_dir / _subject_metrics_filename_for_mode(mode))
    save_json(
        {
            "prediction_backend": mode,
            "evaluation_protocol_name": protocol_name,
            "aggregation_level": aggregation_level,
            "binary_strict_confusion_matrix": headline_metrics_payload["binary_strict_confusion_matrix"],
            "diagnostic_three_class_labels": headline_metrics_payload["diagnostic_three_class_labels"],
            "diagnostic_three_class_confusion_matrix": headline_metrics_payload["diagnostic_three_class_confusion_matrix"],
        },
        output_dir / "confusion_matrix.json",
    )

    LOGGER.info(
        "Finished evaluation checkpoint=%s | backend=%s | aggregation_level=%s | ACC=%.6f F1=%.6f Precision=%.6f Recall=%.6f",
        checkpoint_name,
        mode,
        aggregation_level,
        float(headline_metrics_payload["accuracy"]),
        float(headline_metrics_payload["positive_f1"]),
        float(headline_metrics_payload["precision"]),
        float(headline_metrics_payload["recall"]),
    )
    unit_suffix = (
        "subjects"
        if aggregation_level in {AGGREGATION_LEVEL_SUBJECT, AGGREGATION_LEVEL_RESPONSE_SUBJECT}
        else "segments"
    )
    LOGGER.info(
        "Validation predicted counts checkpoint=%s | backend=%s | aggregation_level=%s | predicted_depressed=%s predicted_non_depressed=%s predicted_invalid=%s",
        checkpoint_name,
        mode,
        aggregation_level,
        int(headline_metrics_payload[f"predicted_depressed_{unit_suffix}"]),
        int(headline_metrics_payload[f"predicted_non_depressed_{unit_suffix}"]),
        int(headline_metrics_payload[f"predicted_invalid_{unit_suffix}"]),
    )
    LOGGER.info(
        "Validation true counts checkpoint=%s | backend=%s | aggregation_level=%s | true_depressed=%s true_non_depressed=%s",
        checkpoint_name,
        mode,
        aggregation_level,
        int(headline_metrics_payload[f"true_depressed_{unit_suffix}"]),
        int(headline_metrics_payload[f"true_non_depressed_{unit_suffix}"]),
    )
    LOGGER.info(
        "Validation output files checkpoint=%s | backend=%s | aggregation_level=%s | headline_csv=%s subject_csv=%s sample_csv=%s sample_jsonl=%s invalid_sample_jsonl=%s invalid_sample_count=%s",
        checkpoint_name,
        mode,
        aggregation_level,
        headline_csv_path,
        subject_csv_path,
        sample_csv_path,
        sample_jsonl_path,
        invalid_jsonl_path,
        len(invalid_rows),
    )
    invalid_units_key = f"invalid_{unit_suffix}"
    valid_units_key = f"num_valid_{aggregation_level}_predictions"
    if int(headline_metrics_payload.get(invalid_units_key, 0)) > 0:
        LOGGER.info(
            "Validation valid-only metrics checkpoint=%s | backend=%s | aggregation_level=%s | valid_units=%s ACC=%.6f F1=%.6f Precision=%.6f Recall=%.6f",
            checkpoint_name,
            mode,
            aggregation_level,
            int(headline_metrics_payload[valid_units_key]),
            float(headline_metrics_payload["valid_only_accuracy"]),
            float(headline_metrics_payload["valid_only_positive_f1"]),
            float(headline_metrics_payload["valid_only_precision"]),
            float(headline_metrics_payload["valid_only_recall"]),
        )
    _log_invalid_prediction_preview(mode, subject_rows, sample_rows)
    return {
        "active_backend": mode,
        "active_aggregation_level": aggregation_level,
        "evaluation_protocol_name": protocol_name,
        "sample_rows": sample_rows,
        "headline_rows": headline_rows,
        "headline_metrics": headline_metrics_payload,
        "subject_rows": subject_rows,
        "subject_metrics": subject_metrics_payload,
        "response_rows": response_rows,
        "response_metrics": response_metrics,
        "secondary_aggregations": secondary_aggregations,
        "backend_results": {
            mode: {
                "headline_rows": headline_rows,
                "headline_metrics": headline_metrics_payload,
                "subject_rows": subject_rows,
                "subject_metrics": subject_metrics_payload,
            }
        },
    }


def _load_metadata_or_build(config_path: str | Path, config: dict[str, Any], config_overrides: list[str] | None = None) -> dict[str, Any]:
    metadata_path = resolve_project_path(Path(config["output_dirs"]["split_dir"]) / f"{config['dataset']}_manifest_metadata.json")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if metadata_path.exists():
        metadata = resolve_metadata_paths(read_json(metadata_path))
        usable, reason = _metadata_artifacts_are_usable(metadata)
        missing_split_keys = [key for key in _required_split_metadata_keys(config) if not metadata.get(key)]
        if usable and missing_split_keys:
            usable = False
            reason = f"split_metadata_missing:{','.join(missing_split_keys)}"
        if usable and metadata.get("build_signature") != manifest_build_signature(config):
            usable = False
            reason = "build_signature_mismatch"
        if usable:
            return metadata
        LOGGER.warning("Refreshing stale metadata for %s: %s", config["dataset"], reason)
    if local_rank == 0:
        build_for_config(config_path, config_overrides)
    return _wait_for_usable_metadata(metadata_path)


def _resolve_final_eval_subject_ids(config: dict[str, Any], metadata: dict[str, Any], fold: int) -> list[str]:
    split_mode = resolve_split_mode(config, metadata)
    if split_mode == SPLIT_MODE_CV:
        fold_payload = read_fold_payload(metadata, fold)
        return sorted(fold_payload["final_eval_subject_ids"])
    if not metadata.get("subject_partition_path"):
        raise ValueError("Split metadata does not include subject_partition_path for fixed/full_train evaluation.")
    partition_rows = read_json(metadata["subject_partition_path"])
    return subject_ids_for_partitions(partition_rows, [str(config["split"]["final_eval_partition"])])


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
    aggregation_level = resolve_aggregation_level(config)
    input_modality = resolve_input_modality(config)
    LOGGER.info(
        "Standalone evaluation backend selected: %s | aggregation_level=%s | protocol=%s | input_modality=%s",
        sample_prediction_mode,
        aggregation_level,
        evaluation_protocol_name(sample_prediction_mode),
        input_modality,
    )
    metadata = _load_metadata_or_build(args.config, config, args.config_overrides)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    split_mode = resolve_split_mode(config, metadata)
    cv_protocol = resolve_cv_protocol(config) if split_mode == SPLIT_MODE_CV else None
    final_eval_subject_ids = _resolve_final_eval_subject_ids(config, metadata, args.fold)
    if int(config.get("split", {}).get("smoke_subject_limit", 0) or 0) > 0:
        saved_split_path = Path(args.checkpoint_dir).parent / "logs" / "split_used.json"
        if not saved_split_path.is_file():
            raise FileNotFoundError(
                f"Smoke evaluation requires the checkpoint's saved split: {saved_split_path}"
            )
        saved_split = read_json(saved_split_path)
        final_eval_subject_ids = sorted(saved_split["final_eval_subject_ids"])
    final_eval_rows = filter_rows_by_subjects(manifest_rows, final_eval_subject_ids)
    evaluation_role = "fold_validation" if cv_protocol == CV_PROTOCOL_TRAIN_VAL else "final_eval"
    examples = build_examples(final_eval_rows, config, partition_name=evaluation_role, truncation_log_path=None)

    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, config)
    processor = load_processor(args.checkpoint_dir, config)
    model = load_model_for_inference(model_name_or_path, args.checkpoint_dir, config)
    lora_layer_selection = resolve_lora_layer_selection(config, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output_dir = args.output_dir or (Path(args.checkpoint_dir) / "standalone_eval")
    save_yaml(
        {
            "base_config_path": str(Path(args.config)),
            "config_overrides": list(args.config_overrides),
            "split_mode": split_mode,
            "cv_protocol": cv_protocol,
            "evaluation_role": evaluation_role,
            "sample_prediction_mode": sample_prediction_mode,
            "aggregation_level": aggregation_level,
            "evaluation_protocol_name": evaluation_protocol_name(sample_prediction_mode),
            "input_modality": input_modality,
            "lora_resolution": lora_layer_selection,
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
    headline_metrics = metrics["backend_results"][active_backend]["headline_metrics"]
    final_result = {
        "split_name": str(config["split"]["final_eval_partition"]),
        "prediction_backend": active_backend,
        "aggregation_level": aggregation_level,
        "loss": None,
        "accuracy": headline_metrics.get("accuracy"),
        "positive_f1": headline_metrics.get("positive_f1"),
        "macro_f1": headline_metrics.get("macro_f1"),
        "precision": headline_metrics.get("precision"),
        "recall": headline_metrics.get("recall"),
        "headline_metrics": headline_metrics,
    }
    best_validation_result = _load_best_validation_result(args.checkpoint_dir)

    LOGGER.info("Standalone evaluation complete: %s", headline_metrics)
    _log_metric_summary("FINAL EVALUATION RESULT", final_result)
    if best_validation_result is not None:
        _log_metric_summary("BEST VALIDATION RESULT (checkpoint-selection score)", best_validation_result)
    save_json(
        {
            "final_evaluation": final_result,
            "best_validation": best_validation_result,
        },
        Path(output_dir) / "final_and_best_validation_metrics.json",
    )


if __name__ == "__main__":
    main()
