#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_json, save_json, write_jsonl


SCHEMA_VERSION = "d3tec_hidden_synthetic_smoke.v1"
ARTIFACTS = (
    "outer_train.npz",
    "outer_train_rows.jsonl",
    "final_eval.npz",
    "final_eval_rows.jsonl",
    "extraction_metadata.json",
)


def _rows(subject_ids: list[str], *, dimension: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors = []
    rows = []
    for subject_index, subject_id in enumerate(subject_ids):
        label = subject_index % 2
        for prompt_id in range(27):
            count = 1 + (prompt_id % 3)
            for segment_index in range(count):
                vector = np.zeros(dimension, dtype=np.float32)
                vector[0] = 2.0 if label else -2.0
                vector[1] = prompt_id / 27
                vector[2] = segment_index / count
                vectors.append(vector)
                rows.append(
                    {
                        "dataset": "d3tec",
                        "sample_id": f"{subject_id}_p{prompt_id}_s{segment_index}",
                        "subject_id": subject_id,
                        "response_id": f"{subject_id}_p{prompt_id}",
                        "prompt_id": prompt_id,
                        "segment_index": segment_index,
                        "num_segments": count,
                        "label": label,
                    }
                )
    return np.asarray(vectors, dtype=np.float32), rows


def create(output_dir: Path, dimension: int = 8) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dimension": dimension,
        "outer_train_subjects": [f"train-{index}" for index in range(6)],
        "final_eval_subjects": ["eval-0", "eval-1"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output_dir.exists() and any(output_dir.iterdir()):
        metadata_path = output_dir / "extraction_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"Partial synthetic smoke cache: {output_dir}.")
        metadata = read_json(metadata_path)
        if metadata.get("smoke_identity_sha256") != digest:
            raise ValueError(f"Incompatible synthetic smoke cache: {output_dir}.")
        if not all((output_dir / name).is_file() for name in ARTIFACTS):
            raise ValueError(f"Partial synthetic smoke cache: {output_dir}.")
        return "skipped_compatible_complete"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_rows = _rows(identity["outer_train_subjects"], dimension=dimension)
    final_x, final_rows = _rows(identity["final_eval_subjects"], dimension=dimension)
    np.savez_compressed(output_dir / "outer_train.npz", vectors=train_x)
    np.savez_compressed(output_dir / "final_eval.npz", vectors=final_x)
    write_jsonl(train_rows, output_dir / "outer_train_rows.jsonl")
    write_jsonl(final_rows, output_dir / "final_eval_rows.jsonl")
    save_json(
        {
            "dataset": "d3tec",
            "condition": "audio_text_normalized",
            "input_modality": "audio_text",
            "fold": 0,
            "checkpoint_dir": "synthetic/best_model",
            "adapter_config_sha256": "0" * 64,
            "adapter_sha256": "1" * 64,
            "saved_split_sha256": "2" * 64,
            "split_metadata_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "smoke_identity": identity,
            "smoke_identity_sha256": digest,
        },
        output_dir / "extraction_metadata.json",
    )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a repeated-response D3TEC smoke cache.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=8)
    args = parser.parse_args()
    print(create(args.output_dir, args.dimension))


if __name__ == "__main__":
    main()
