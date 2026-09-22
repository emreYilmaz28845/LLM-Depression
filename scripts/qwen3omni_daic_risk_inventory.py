#!/usr/bin/env python3
"""Model-free DAIC risk inventory for the Qwen3-Omni pilot.

Builds one deterministic inventory over the resolved DAIC manifest and selects
the examples that stress the processor and the model most: the shortest and
longest audio, the shortest and longest rendered prompt, the largest combined
processor footprint, and — when they are different examples — each maximum
separately. The probes and the memory gate then reuse exactly this selection.

Nothing here loads model weights. With ``--with-processor`` the real
``Qwen3OmniMoeProcessor`` is used to measure the rendered token counts and the
audio-feature shapes; without it the script stays purely model-free and records
the audio durations and span counts only.

Privacy: the audit records durations, token/frame counts, shapes, dtypes and
hashes. It never writes transcripts, subject identifiers, or audio paths; each
selected example is identified by ``example_ref = sha256(sample_id)[:16]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.daic import PACKED30_SAMPLE_RATE
from src.data.runtime import build_examples, load_manifest_rows
from src.utils import (
    get_logger,
    normalize_config_overrides,
    resolve_input_modality,
    resolve_model_name_or_path,
    resolve_project_path,
    save_json,
)
from src.utils import load_yaml_with_overrides

LOGGER = get_logger(__name__)

SCHEMA_VERSION = "audiollm.qwen3omni_risk_inventory.v1"
AUDIO_TOKENS_PER_MEL_FRAME = 0.25  # Whisper-style two stride-2 convolutions.


def example_ref(sample_id: str) -> str:
    """Stable, non-identifying reference for one manifest example."""
    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]


def _audio_seconds(row: dict[str, Any]) -> float:
    spans = list(row.get("audio_spans") or [])
    frames = sum(int(span["end_frame"]) - int(span["start_frame"]) for span in spans)
    return frames / float(PACKED30_SAMPLE_RATE)


def _load_manifest_rows(config: dict[str, Any], config_path: Path, overrides: list[str]) -> list[dict[str, Any]]:
    from src.evaluate import _load_metadata_or_build

    metadata = _load_metadata_or_build(config_path, config, overrides or None)
    rows = load_manifest_rows(metadata["manifest_path"])
    config["_manifest_path"] = metadata["manifest_path"]
    config["_manifest_hash"] = metadata.get("manifest_hash")
    return rows


def _processor_measurements(
    processor, config: dict[str, Any], examples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Rendered token counts and audio-feature shapes per example."""
    from src.data.runtime import AudioTextDataset
    from src.model.collator import Qwen2AudioSFTCollator
    from src.model.runtime import resolve_processor_sampling_rate

    sampling_rate = resolve_processor_sampling_rate(processor)
    collator = Qwen2AudioSFTCollator(processor=processor)
    dataset = AudioTextDataset(examples, processor_sampling_rate=sampling_rate, silence_audio=False)
    measurements: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        item = dataset[index]
        batch = collator([item])
        tokens = int(batch["input_ids"].shape[-1])
        features = batch.get("input_features")
        mask = batch.get("feature_attention_mask")
        mel_frames = int(mask.sum().item()) if mask is not None else 0
        measurements[example_ref(example["sample_id"])] = {
            "rendered_tokens": tokens,
            "audio_seconds": round(_audio_seconds(example), 4),
            "mel_frames": mel_frames,
            "estimated_audio_tokens": int(round(mel_frames * AUDIO_TOKENS_PER_MEL_FRAME)),
            "input_features_shape": list(features.shape) if features is not None else None,
            "input_features_dtype": str(features.dtype) if features is not None else None,
            "feature_attention_mask_dtype": str(mask.dtype) if mask is not None else None,
        }
    return measurements


def _select(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic max-risk selection over the measured records.

    The pseudonymous subject reference is used only to answer whether the audio
    and token maxima belong to different participants; it is never published.
    """
    subject_by_ref = {record["example_ref"]: record.pop("subject_ref", None) for record in records}
    by_duration = sorted(records, key=lambda r: (r["audio_seconds"], r["example_ref"]))
    by_tokens = sorted(records, key=lambda r: (r["rendered_tokens"], r["example_ref"]))
    by_footprint = sorted(
        records,
        key=lambda r: (r["rendered_tokens"] + r["estimated_audio_tokens"], r["example_ref"]),
    )
    longest_audio = by_duration[-1]
    longest_tokens = by_tokens[-1]
    largest_footprint = by_footprint[-1]
    return {
        "shortest_audio": by_duration[0],
        "longest_audio": longest_audio,
        "shortest_prompt": by_tokens[0],
        "longest_prompt": longest_tokens,
        "largest_combined_footprint": largest_footprint,
        "audio_and_token_maxima_differ": longest_audio["example_ref"] != longest_tokens["example_ref"],
        "audio_and_token_maxima_in_different_subjects": (
            subject_by_ref.get(longest_audio["example_ref"])
            != subject_by_ref.get(longest_tokens["example_ref"])
        ),
        "footprint_maximum_differs_from_both": (
            largest_footprint["example_ref"] not in {longest_audio["example_ref"], longest_tokens["example_ref"]}
        ),
    }


def _per_modality(
    config: dict[str, Any], rows: list[dict[str, Any]], manifests: dict[str, Any]
) -> dict[str, Any]:
    import copy

    result: dict[str, Any] = {}
    for modality_overrides in (
        {"use_audio": True, "use_text": False},
        {"use_audio": True, "use_text": True},
    ):
        modality_config = copy.deepcopy(config)
        modality_config["data"].update(modality_overrides)
        modality_config["data"].pop("audio_text_transcript_scope", None)
        if modality_overrides["use_text"]:
            modality_config["data"]["audio_text_transcript_scope"] = "full_participant"
        modality = resolve_input_modality(modality_config)
        examples = build_examples(rows, modality_config, partition_name="risk_inventory")
        records = [
            {
                "example_ref": example_ref(example["sample_id"]),
                "subject_ref": example_ref(f"subject:{example['subject_id']}"),
                "partition": str(example.get("partition_name", "")),
                "audio_seconds": round(_audio_seconds(example), 4),
                "span_count": len(example.get("audio_spans") or []),
                "rendered_tokens": len(str(example["prompt_text"])),
                "mel_frames": 0,
                "estimated_audio_tokens": 0,
                "input_features_shape": None,
                "input_features_dtype": None,
                "feature_attention_mask_dtype": None,
            }
            for example in examples
        ]
        result[modality] = {"examples": len(records), "records": records}
        manifests[modality] = examples
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--with-processor", action="store_true")
    parser.add_argument("--processor-dir", default=None)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override in the repository --set form (repeatable)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    overrides = normalize_config_overrides(getattr(args, "set_overrides", []) or [])
    config = load_yaml_with_overrides(config_path, overrides)

    rows = _load_manifest_rows(config, config_path, overrides)
    LOGGER.info("DAIC manifest rows: %s", len(rows))

    manifests: dict[str, Any] = {}
    per_modality = _per_modality(config, rows, manifests)

    processor_summary: dict[str, Any] = {}
    if args.with_processor:
        from src.model.runtime import load_processor

        processor_dir = args.processor_dir or resolve_model_name_or_path(None, config)
        processor = load_processor(processor_dir, config)
        for modality, payload in per_modality.items():
            examples = manifests[modality]
            measurements = _processor_measurements(processor, config, examples)
            for record in payload["records"]:
                record.update(measurements.get(record["example_ref"], {}))
            processor_summary[modality] = {
                "processor_dir": str(processor_dir),
                "measured_examples": len(measurements),
            }

    audit = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_dataset": config["dataset"],
        "prompt_version": (config.get("prompt") or {}).get("version"),
        "manifest_path": config.get("_manifest_path"),
        "manifest_sha256": config.get("_manifest_hash"),
        "protocol_id": config.get("protocol_id"),
        "sample_rate": int(PACKED30_SAMPLE_RATE),
        "with_processor": bool(args.with_processor),
        "processor": processor_summary,
        "modalities": {},
    }
    for modality, payload in per_modality.items():
        records = payload["records"]
        audit["modalities"][modality] = {
            "examples": payload["examples"],
            "selection": _select(records) if records else None,
            "records": records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(audit, args.output)
    LOGGER.info("Wrote risk inventory to %s", args.output)
    for modality, payload in audit["modalities"].items():
        selection = payload["selection"]
        if not selection:
            continue
        LOGGER.info(
            "%s | longest_audio=%.2fs (ref %s) | longest_prompt=%s tokens (ref %s) | maxima_differ=%s",
            modality,
            selection["longest_audio"]["audio_seconds"],
            selection["longest_audio"]["example_ref"],
            selection["longest_prompt"]["rendered_tokens"],
            selection["longest_prompt"]["example_ref"],
            selection["audio_and_token_maxima_differ"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
