#!/usr/bin/env python3
"""Create a small deterministic Androids-shaped cache for Slurm smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.features.androids_hidden_policy import (
    ANDROID_AGGREGATION_POLICY,
    ANDROID_DATASET,
    ANDROID_HIDDEN_CACHE_SCHEMA,
    ANDROID_MANIFEST_HASH,
    ANDROID_SPLIT_HASH,
    CACHE_ARTIFACT_NAMES,
    file_sha256,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite synthetic cache: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1337)
    dimension = 8

    def make_rows(subjects: list[tuple[str, int]], partition: str) -> tuple[np.ndarray, list[dict[str, object]]]:
        vectors: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for subject_index, (subject_id, label) in enumerate(subjects):
            for turn_id, window_count in ((1, 1), (2, 2)):
                response_id = f"{subject_id}_turn{turn_id}"
                for window_index in range(window_count):
                    sample_id = f"{response_id}_w{window_index:02d}"
                    signal = 2.5 if label else -2.5
                    vector = rng.normal(0.0, 0.2, dimension).astype(np.float32)
                    vector[0] += signal + subject_index * 0.01
                    vectors.append(vector)
                    rows.append(
                        {
                            "dataset": ANDROID_DATASET,
                            "modality": "audio_only",
                            "input_modality": "audio_only",
                            "partition": partition,
                            "fold": 0,
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "label": label,
                            "recording_id": subject_id,
                            "turn_id": turn_id,
                            "turn_key": response_id,
                            "response_id": response_id,
                            "window_id": sample_id,
                            "window_index": window_index,
                            "num_windows": window_count,
                            "num_segments": window_count,
                            "segment_index": window_index,
                            "start_time": float(window_index * 10.0),
                            "end_time": float((window_index + 1) * 10.0),
                            "segment_duration": 10.0,
                            "turn_duration": float(window_count * 10.0),
                        }
                    )
        return np.stack(vectors).astype(np.float32), rows

    train_subjects = [(f"train_{index:02d}", index % 2) for index in range(8)]
    eval_subjects = [(f"eval_{index:02d}", index % 2) for index in range(6)]
    train_x, train_rows = make_rows(train_subjects, "outer_train")
    eval_x, eval_rows = make_rows(eval_subjects, "final_eval")
    np.savez_compressed(args.output_dir / "outer_train.npz", vectors=train_x)
    np.savez_compressed(args.output_dir / "final_eval.npz", vectors=eval_x)
    write_jsonl(train_rows, args.output_dir / "outer_train_rows.jsonl")
    write_jsonl(eval_rows, args.output_dir / "final_eval_rows.jsonl")
    metadata = {
        "schema_version": ANDROID_HIDDEN_CACHE_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": "audio_only",
        "input_modality": "audio_only",
        "audio_text_transcript_scope": None,
        "fold": 0,
        "source_run_id": "synthetic_smoke",
        "source_commit": args.source_commit,
        "manifest_sha256": ANDROID_MANIFEST_HASH,
        "split_metadata_sha256": ANDROID_SPLIT_HASH,
        "adapter_config_sha256": "synthetic-adapter-config",
        "adapter_sha256": "synthetic-adapter",
        "aggregation_policy": ANDROID_AGGREGATION_POLICY,
        "vector_dimension": dimension,
        "vector_dtype": "float32",
        "max_examples": "synthetic",
        "gold_label_protection": {"labels_passed_to_model": False},
        "cache_config": {"schema_version": ANDROID_HIDDEN_CACHE_SCHEMA, "synthetic": True},
        "partitions": {
            "outer_train": {"rows": len(train_rows), "subjects": len(train_subjects)},
            "final_eval": {"rows": len(eval_rows), "subjects": len(eval_subjects)},
        },
    }
    (args.output_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_rows = [f"{file_sha256(args.output_dir / name)}\t{name}" for name in CACHE_ARTIFACT_NAMES]
    (args.output_dir / "cache_sha256.tsv").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "train_rows": len(train_rows), "eval_rows": len(eval_rows)}, indent=2))


if __name__ == "__main__":
    main()
