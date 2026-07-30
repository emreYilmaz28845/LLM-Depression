#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.androids import build_androids_interview_manifest
from src.data.runtime import build_examples
from src.utils import load_yaml, save_json, sha256_jsonl_rows


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "androids_interview"
CONFIG_PATHS = [
    CONFIG_DIR / "androids_interview_audio_only.yaml",
    CONFIG_DIR / "androids_interview_audio_text_segment_aligned.yaml",
    CONFIG_DIR / "androids_interview_audio_text_full_turn.yaml",
    CONFIG_DIR / "androids_interview_text_only.yaml",
]


def audit(args: argparse.Namespace) -> dict:
    condition_reports = []
    manifest_hashes: set[str] = set()
    fold_hashes: set[str] = set()
    sample_ids_by_audio_condition: list[list[str]] = []
    interval_ids_by_audio_condition: list[list[tuple[str, float, float]]] = []
    prompts_by_condition: dict[str, dict[str, tuple[str, str]]] = {}
    for config_path in CONFIG_PATHS:
        config = load_yaml(config_path)
        config["dataset_root"] = str(args.dataset_root)
        config["full_transcript_path"] = str(args.full_transcripts)
        config["segment_transcript_path"] = str(args.segment_transcripts)
        result = build_androids_interview_manifest(config, {})
        rows = result["manifest_rows"]
        manifest_hash = sha256_jsonl_rows(rows)
        manifest_hashes.add(manifest_hash)
        fold_hashes.add(result["fold_hash"])
        examples = build_examples(rows, config, "input_audit")
        condition = config_path.stem.removeprefix("androids_interview_")
        max_audio_seconds = max(
            (float(row["segment_duration"]) for row in rows), default=0.0
        )
        truncated = sum(
            bool(config["data"].get("transcript_max_chars"))
            and len(example["transcript"])
            >= int(config["data"]["transcript_max_chars"])
            for example in examples
        )
        condition_reports.append(
            {
                "condition": condition,
                "manifest_hash": manifest_hash,
                "fold_hash": result["fold_hash"],
                "manifest_rows": len(rows),
                "runtime_examples": len(examples),
                "max_audio_interval_seconds": max_audio_seconds,
                "transcript_cap": int(
                    config["data"].get("transcript_max_chars", 0) or 0
                ),
                "transcripts_at_cap": truncated,
            }
        )
        prompts_by_condition[condition] = {
            example["sample_id"]: (example["prompt_text"], example["transcript"])
            for example in examples
        }
        if config["data"]["use_audio"]:
            sample_ids_by_audio_condition.append(
                [example["sample_id"] for example in examples]
            )
            interval_ids_by_audio_condition.append(
                [
                    (
                        example["sample_id"],
                        float(example["start_time"]),
                        float(example["end_time"]),
                    )
                    for example in examples
                ]
            )
    if len(manifest_hashes) != 1 or len(fold_hashes) != 1:
        raise ValueError("The four conditions do not share manifest/fold hashes.")
    if len({tuple(values) for values in sample_ids_by_audio_condition}) != 1:
        raise ValueError("Audio conditions do not use identical window IDs.")
    if len({tuple(values) for values in interval_ids_by_audio_condition}) != 1:
        raise ValueError("Audio conditions do not use identical intervals.")
    if any(report["max_audio_interval_seconds"] > 30.0 + 1e-6 for report in condition_reports):
        raise ValueError("A Qwen2-Audio input exceeds the 30-second budget.")
    if any(report["transcripts_at_cap"] for report in condition_reports):
        raise ValueError("The current corpus incurs transcript truncation.")
    aligned = prompts_by_condition["audio_text_segment_aligned"]
    full = prompts_by_condition["audio_text_full_turn"]
    if set(aligned) != set(full) or not any(
        aligned[sample_id][0] != full[sample_id][0] for sample_id in aligned
    ):
        raise ValueError("The two audio+text conditions did not resolve distinct prompts.")
    for sample_id in aligned:
        aligned_prompt, aligned_transcript = aligned[sample_id]
        full_prompt, full_transcript = full[sample_id]
        aligned_block = (
            "The transcript of the subject's speech is:\n"
            f"{aligned_transcript}\n\n"
        )
        full_block = (
            "The transcript of the subject's speech is:\n"
            f"{full_transcript}\n\n"
        )
        if (
            aligned_prompt.replace(
                aligned_block,
                "The transcript of the subject's speech is:\n<TRANSCRIPT>\n\n",
                1,
            )
            != full_prompt.replace(
                full_block,
                "The transcript of the subject's speech is:\n<TRANSCRIPT>\n\n",
                1,
            )
        ):
            raise ValueError(
                "Audio+text prompts differ outside transcript scope for "
                f"{sample_id}."
            )
    report = {
        "schema_version": "androids_interview_input_audit.v1",
        "status": "passed",
        "manifest_hash": next(iter(manifest_hashes)),
        "fold_hash": next(iter(fold_hashes)),
        "full_transcript_sha256": hashlib.sha256(
            args.full_transcripts.read_bytes()
        ).hexdigest(),
        "segment_transcript_sha256": hashlib.sha256(
            args.segment_transcripts.read_bytes()
        ).hexdigest(),
        "conditions": condition_reports,
    }
    save_json(report, args.out)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus"
        ),
    )
    parser.add_argument("--full-transcripts", type=Path)
    parser.add_argument("--segment-transcripts", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "androids_interview_input_audit.json",
    )
    args = parser.parse_args()
    if args.full_transcripts is None:
        args.full_transcripts = (
            args.dataset_root / "interview_transcripts_qwen3_asr_italian.jsonl"
        )
    if args.segment_transcripts is None:
        args.segment_transcripts = (
            args.dataset_root
            / "interview_transcripts_qwen3_asr_italian_segments.jsonl"
        )
    return args


if __name__ == "__main__":
    payload = audit(parse_args())
    print(json.dumps(payload, indent=2))
