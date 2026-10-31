"""Frozen wav2vec2 window features for the audio baseline.

Uses a Qwen3-Omni frozen-feature reference run (fold 0) for the dataset
config, saved split and manifest, so the subjects, partitions and natural
units match the Omni ``audio_only`` frozen + new head route. Only the window
length changes (``--segment-seconds``):

* D3TEC / ANDROIDS Interview: manifest windows re-split per natural unit;
* CMDC / Turkish: ``data.segment_seconds`` (windows are cut at runtime);
* DAIC: participant-only packed stream re-packed at ``segment_seconds``.

Each window goes through the frozen model in FP32 (batch 1, no padding). The
last transformer hidden state is pooled over time two ways, written as two
caches in the layout ``baselines/frozen_standalone_heads.py`` reads:

* ``<output-root>/meanvar``: time mean + population variance (2 x hidden);
* ``<output-root>/mean``: time mean (hidden).

Partitions follow the Omni extraction (``outer_train`` / ``final_eval``);
``--final-eval-source selection`` writes the DAIC official-dev cache.

``--audio-path-map`` runs the extraction on another machine: a JSON object
mapping every manifest audio path (as written on MN5) to a local file whose
content hash was checked against the MN5 file. Paths are remapped only after
the reference manifest hash is verified; an unmapped path is fatal.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
from src.features.extract_qwen_hidden import (
    _git_commit,
    _load_saved_run,
    _package_version,
    _resolve_subject_partitions,
    _saved_path,
)
from src.features.qwen_hidden_collator import load_prompt_audio
from src.features.wav2vec2_windows import (
    HARMONIZED_REWINDOW_DATASETS,
    RUNTIME_WINDOW_DATASETS,
    repack_daic_packed_rows,
    rewindow_harmonized_rows,
)
from src.utils import read_json, save_json, sha256_file, sha256_jsonl_rows, write_jsonl

SCHEMA_VERSION = "wav2vec2_window_cache.v1"
MODEL_BACKEND = "wav2vec2"
SAMPLING_RATE = 16000
POOLINGS = {"meanvar": "time_mean_population_variance", "mean": "time_mean"}
PARTITIONS = ("outer_train", "final_eval")
MODEL_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors", "CONVERSION.json")
DETERMINISM_ATOL = 1e-5


def window_rows(manifest_rows: list[dict[str, Any]], config: dict[str, Any], segment_seconds: float):
    """Return (rows, config) re-windowed at ``segment_seconds`` plus the policy name."""
    dataset = str(config["dataset"]).lower()
    config = json.loads(json.dumps(config))
    data = config.setdefault("data", {})
    sample_mode = str(data.get("sample_mode", ""))
    if dataset == "daic":
        if sample_mode != "participant_speech_packed30":
            raise ValueError(f"DAIC reference run must use participant_speech_packed30, got {sample_mode!r}.")
        chunk_samples = int(round(segment_seconds * SAMPLING_RATE))
        data["participant_chunk_samples"] = chunk_samples
        return repack_daic_packed_rows(manifest_rows, chunk_samples), config, f"daic_repack_{chunk_samples}_samples"
    if sample_mode != "harmonized_response_windows":
        raise ValueError(f"{dataset} reference run must use harmonized_response_windows, got {sample_mode!r}.")
    data["segment_seconds"] = float(segment_seconds)
    if dataset in HARMONIZED_REWINDOW_DATASETS:
        return rewindow_harmonized_rows(manifest_rows, dataset, segment_seconds), config, "manifest_rewindow_equal_duration"
    if dataset in RUNTIME_WINDOW_DATASETS:
        return list(manifest_rows), config, "runtime_equal_duration"
    raise ValueError(f"Unsupported dataset {dataset!r}.")


def _git_head() -> str | None:
    """HEAD of a real checkout; ``_git_commit`` may return a stale deploy-time value."""
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def remap_audio_paths(rows: list[dict[str, Any]], path_map: dict[str, str]) -> list[dict[str, Any]]:
    """Replace every audio path through ``path_map``; any unmapped path is an error."""
    missing: set[str] = set()

    def _map(path: str) -> str:
        if path not in path_map:
            missing.add(path)
            return path
        return path_map[path]

    output = []
    for row in rows:
        new = dict(row)
        if new.get("audio_path"):
            new["audio_path"] = _map(str(new["audio_path"]))
        if new.get("audio_paths"):
            new["audio_paths"] = [_map(str(p)) for p in new["audio_paths"]]
        output.append(new)
    if missing:
        raise ValueError(f"{len(missing)} audio paths are not in the verified path map, e.g. {sorted(missing)[:3]}")
    return output


def pool_hidden(hidden: torch.Tensor) -> dict[str, np.ndarray]:
    """``hidden`` is one unpadded (1, T, H) sequence; pool over time in float64."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1:
        raise ValueError(f"Expected one unpadded (1, T, H) sequence, got {tuple(hidden.shape)}.")
    h = hidden[0].detach().to("cpu", torch.float64)
    mean = h.mean(dim=0)
    variance = h.var(dim=0, unbiased=False)
    return {
        "mean": mean.numpy().astype(np.float32),
        "meanvar": torch.cat([mean, variance]).numpy().astype(np.float32),
    }


def _model_identity(model_path: Path, revision: str) -> dict[str, Any]:
    files = {name: sha256_file(model_path / name) for name in MODEL_FILES if (model_path / name).is_file()}
    if "config.json" not in files or not ({"pytorch_model.bin", "model.safetensors"} & set(files)):
        raise FileNotFoundError(f"{model_path} lacks config.json or model weights.")
    return {"path": str(model_path), "revision": revision, "file_sha256": files}


def _row_record(example: dict[str, Any], partition: str, fold: int, num_samples: int, num_frames: int) -> dict[str, Any]:
    record = {
        "dataset": example.get("dataset"),
        "sample_id": str(example["sample_id"]),
        "subject_id": str(example["subject_id"]),
        "label": int(example["label"]),
        "partition": partition,
        "fold": fold,
        "num_audio_samples": int(num_samples),
        "num_frames": int(num_frames),
    }
    for key in ("response_id", "question_condition", "question_id", "segment_index", "num_segments",
                "chunk_index", "num_chunks", "protocol_id"):
        if key in example:
            record[key] = example[key]
    for key in ("start_time", "end_time"):
        if key in example:
            record[key] = example[key]
        elif example.get(f"audio_{key}s"):
            record[key] = example[f"audio_{key}s"][0]
    return record


def _load_window_audio(example: dict[str, Any]) -> np.ndarray:
    arrays = load_prompt_audio(example, SAMPLING_RATE, False)["audio_arrays"]
    if len(arrays) != 1:
        raise ValueError(f"{example['sample_id']}: expected one audio array, got {len(arrays)}.")
    audio = np.asarray(arrays[0], dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"{example['sample_id']}: invalid audio array {audio.shape}.")
    return audio


@torch.inference_mode()
def _embed(model, feature_extractor, audio: np.ndarray, device: torch.device) -> tuple[dict[str, np.ndarray], int]:
    inputs = feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    values = inputs["input_values"].to(device=device, dtype=torch.float32)
    hidden = model(input_values=values).last_hidden_state
    return pool_hidden(hidden), int(hidden.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-fold-dir", required=True, type=Path,
                        help="Omni reference fold dir with run_config.yaml and logs/split_used.json.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--segment-seconds", required=True, type=float)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path, help="Override the saved manifest path.")
    parser.add_argument("--final-eval-source", choices=("saved", "selection"), default="saved")
    parser.add_argument("--smoke-subjects", type=int, default=0,
                        help="Smoke only: keep this many subjects per partition (non-reportable).")
    parser.add_argument("--audio-path-map", type=Path,
                        help="JSON {mn5_path: local_path} of content-hash-verified audio files.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.segment_seconds <= 30.0:
        raise ValueError("--segment-seconds must be in (0, 30].")
    fold_dir = args.reference_fold_dir.resolve()
    saved, config, run_config_path, split_path = _load_saved_run(fold_dir / "best_model")
    if str(saved.get("input_modality")) != "audio_only":
        raise ValueError("The reference run must be the Omni audio_only route.")
    split_payload = read_json(split_path)
    partitions, evaluation_provenance = _resolve_subject_partitions(
        saved, config, split_payload, final_eval_source=args.final_eval_source
    )
    if args.smoke_subjects > 0:
        partitions = {name: ids[: args.smoke_subjects] for name, ids in partitions.items()}

    manifest_path = args.manifest_path.resolve() if args.manifest_path else _saved_path(saved["manifest_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    source_manifest_sha256 = sha256_jsonl_rows(manifest_rows)
    if saved.get("manifest_hash") and source_manifest_sha256 != saved["manifest_hash"]:
        raise ValueError("Reference manifest hash differs from the reference run's saved manifest hash.")
    rows, config, window_policy = window_rows(manifest_rows, config, args.segment_seconds)
    window_manifest_sha256 = sha256_jsonl_rows(rows)
    path_map_identity = None
    if args.audio_path_map:
        rows = remap_audio_paths(rows, read_json(args.audio_path_map))
        path_map_identity = {"path": str(args.audio_path_map.resolve()), "sha256": sha256_file(args.audio_path_map)}

    output_root = args.output_root.resolve()
    out_dirs = {name: output_root / name for name in POOLINGS}
    for path in out_dirs.values():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing cache: {path}")

    examples: dict[str, list[dict[str, Any]]] = {}
    fold = int(saved["fold"])
    for partition in PARTITIONS:
        ids = partitions[partition]
        built = build_examples(filter_rows_by_subjects(rows, ids), config, partition_name=partition,
                               truncation_log_path=None)
        got = {str(e["subject_id"]) for e in built}
        if got != set(ids):
            raise ValueError(f"{partition}: example subjects differ from the saved split "
                             f"(missing={sorted(set(ids) - got)[:5]}, extra={sorted(got - set(ids))[:5]}).")
        examples[partition] = sorted(built, key=lambda e: (str(e["subject_id"]), str(e["sample_id"])))

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = torch.device(args.device)
    model_identity = _model_identity(args.model_path.resolve(), args.model_revision)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model_path))
    if int(feature_extractor.sampling_rate) != SAMPLING_RATE:
        raise ValueError("Feature extractor sampling rate is not 16 kHz.")
    model = Wav2Vec2Model.from_pretrained(str(args.model_path), torch_dtype=torch.float32).to(device).eval()
    hidden_size = int(model.config.hidden_size)

    summaries: dict[str, Any] = {}
    staged = {name: path.with_name(path.name + ".partial") for name, path in out_dirs.items()}
    for path in staged.values():
        if path.exists():
            raise FileExistsError(f"Partial output from an earlier run exists: {path}")
        path.mkdir(parents=True)
    for partition in PARTITIONS:
        vectors = {name: [] for name in POOLINGS}
        records = []
        durations = []
        for example in examples[partition]:
            audio = _load_window_audio(example)
            pooled, frames = _embed(model, feature_extractor, audio, device)
            for name in POOLINGS:
                vectors[name].append(pooled[name])
            records.append(_row_record(example, partition, fold, audio.size, frames))
            durations.append(audio.size / SAMPLING_RATE)
        first = examples[partition][0]
        repeat, _ = _embed(model, feature_extractor, _load_window_audio(first), device)
        determinism = float(np.max(np.abs(repeat["meanvar"] - vectors["meanvar"][0])))
        if determinism > DETERMINISM_ATOL:
            raise RuntimeError(f"{partition}: repeated forward differs by {determinism}.")
        for name in POOLINGS:
            matrix = np.stack(vectors[name]).astype(np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError(f"{partition}/{name}: non-finite features.")
            np.savez(staged[name] / f"{partition}.npz", vectors=matrix)
            write_jsonl([{**r, "pooling": POOLINGS[name], "vector_dimension": int(matrix.shape[1])} for r in records],
                        staged[name] / f"{partition}_rows.jsonl")
        summaries[partition] = {
            "rows": len(records),
            "subjects": len({r["subject_id"] for r in records}),
            "window_seconds": {"min": float(min(durations)), "max": float(max(durations)),
                               "total": float(sum(durations))},
            "frames_min": int(min(r["num_frames"] for r in records)),
            "determinism_max_abs_diff": determinism,
        }

    for name, path in out_dirs.items():
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset": config["dataset"],
            "condition": f"audio_only_frozen_wav2vec2_{int(round(args.segment_seconds))}s",
            "input_modality": "audio_only",
            "fold": fold,
            "model_backend": MODEL_BACKEND,
            "weights_source": "base_model_no_adapter",
            "base_model": model_identity["path"],
            "base_model_revision": args.model_revision,
            "base_model_files_sha256": model_identity["file_sha256"],
            "hidden_layer": "last_hidden_state",
            "pooling": POOLINGS[name],
            "vector_dimension": hidden_size * (2 if name == "meanvar" else 1),
            "vector_dtype": "float32",
            "inference_dtype": "float32",
            "feature_extractor": {"do_normalize": bool(feature_extractor.do_normalize),
                                  "sampling_rate": SAMPLING_RATE},
            "segment_seconds": float(args.segment_seconds),
            "window_policy": window_policy,
            "manifest_sha256": window_manifest_sha256,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": source_manifest_sha256,
            "audio_path_map": path_map_identity,
            "reference_fold_dir": str(fold_dir),
            "reference_run_config_sha256": sha256_file(run_config_path),
            "reference_split": str(split_path),
            "reference_split_sha256": sha256_file(split_path),
            "evaluation_provenance": evaluation_provenance,
            "final_eval_source": args.final_eval_source,
            "partition_subject_ids": partitions,
            "smoke": bool(args.smoke_subjects > 0),
            "partitions": summaries,
            "gold_label_protection": {"labels_passed_to_model": False},
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": _package_version("transformers"),
                "project_git": _git_commit(),
                "git_head": _git_head(),
            },
        }
        save_json(metadata, staged[name] / "extraction_metadata.json")
        staged[name].rename(path)
    print(json.dumps({"output_root": str(output_root), "partitions": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
