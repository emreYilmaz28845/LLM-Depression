#!/usr/bin/env python3
"""Frozen WavLM Base+ subject-level baseline for the fixed DAIC split.

The scientific unit is always one participant.  Exactly four chunks are selected
per participant using numeric chunk order and evenly spaced indices.  WavLM is
frozen and transformer layers 6, 7, and 8 are mean-pooled over time, then
concatenated immediately into one vector per selected chunk; frame-level
representations are never written to disk.  A subject vector is the concatenation
of the mean and standard deviation of its four chunk vectors.

The fixed train/validation/test split is used as follows:

* train selects the regularization strength by validation log loss;
* train+validation is refit once with that strength;
* test is evaluated once;
* majority and within-partition shuffled-audio controls are reported.

The shuffled-audio control deranges complete K=4 subject bundles within each
partition while leaving target subject IDs and labels unchanged.  It therefore
breaks acoustic/label alignment without crossing split boundaries or reintroducing
the DAIC chunk-count shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Must be set before CUDA context initialization for deterministic CUDA GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, WavLMModel, __version__ as transformers_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import binary_auroc, classification_metrics


DEFAULT_MODEL_ID = "microsoft/wavlm-base-plus"
DEFAULT_MODEL_DIR = Path("/home/emre/models/WavLM-Base-Plus")
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/manifests/daic_manifest.jsonl"
DEFAULT_PARTITIONS = PROJECT_ROOT / "outputs/splits/daic_subject_partitions.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/baselines/e1b_wavlm_base_plus_daic"
PARTITIONS = ("train", "val", "test")
CHUNK_NUMBER = re.compile(r"(\d+)$")
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
WAVLM_LAYER_NUMBERS = (6, 7, 8)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, values, allow_pickle=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Match the repository's deterministic even-spacing rule."""

    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / (count - 1)
    indices = [int(round(index * step)) for index in range(count)]
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    fallback = 0
    while len(deduped) < count and fallback < total:
        if fallback not in deduped:
            deduped.append(fallback)
        fallback += 1
    return sorted(deduped)


def numeric_chunk_key(row: dict[str, Any]) -> tuple[int, str]:
    for value in (row.get("chunk_id", ""), row.get("sample_id", "")):
        match = CHUNK_NUMBER.search(str(value).strip())
        if match:
            return int(match.group(1)), str(row["sample_id"])
    raise ValueError(f"Cannot parse numeric chunk order for sample {row.get('sample_id')!r}")


def load_fixed_daic_rows(
    manifest_path: Path,
    partitions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_rows = _read_jsonl(manifest_path)
    partition_rows = json.loads(partitions_path.read_text(encoding="utf-8"))
    partition_by_subject: dict[str, dict[str, Any]] = {}
    for row in partition_rows:
        subject_id = str(row["subject_id"])
        if subject_id in partition_by_subject:
            raise ValueError(f"Duplicate partition assignment for subject {subject_id}")
        partition = str(row["partition"])
        if partition not in PARTITIONS:
            raise ValueError(f"Unexpected DAIC partition {partition!r} for {subject_id}")
        partition_by_subject[subject_id] = {
            "partition": partition,
            "label": int(row["label"]),
        }

    seen_samples: set[str] = set()
    seen_subjects: set[str] = set()
    subject_labels: dict[str, set[int]] = defaultdict(set)
    normalized: list[dict[str, Any]] = []
    for original in manifest_rows:
        row = dict(original)
        sample_id = str(row["sample_id"])
        subject_id = str(row["subject_id"])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen_samples.add(sample_id)
        if subject_id not in partition_by_subject:
            raise ValueError(f"Manifest subject missing fixed partition: {subject_id}")
        expected = partition_by_subject[subject_id]
        label = int(row["label"])
        if label != expected["label"]:
            raise ValueError(f"Label mismatch for subject {subject_id}")
        manifest_partition = str(row.get("split_original", ""))
        if manifest_partition != expected["partition"]:
            raise ValueError(
                f"Manifest/fixed partition mismatch for {subject_id}: "
                f"{manifest_partition!r} != {expected['partition']!r}"
            )
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        row["subject_id"] = subject_id
        row["sample_id"] = sample_id
        row["label"] = label
        row["partition"] = expected["partition"]
        row["audio_path"] = str(audio_path)
        normalized.append(row)
        seen_subjects.add(subject_id)
        subject_labels[subject_id].add(label)

    if seen_subjects != set(partition_by_subject):
        missing = sorted(set(partition_by_subject) - seen_subjects)
        raise ValueError(f"Fixed-partition subjects missing manifest rows: {missing[:10]}")
    mixed = sorted(subject_id for subject_id, labels in subject_labels.items() if len(labels) != 1)
    if mixed:
        raise ValueError(f"Mixed-label subjects: {mixed[:10]}")
    return normalized, partition_by_subject


def select_fixed_k_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    chunks_per_subject: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["subject_id"])].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    for subject_id in sorted(grouped):
        rows = sorted(grouped[subject_id], key=numeric_chunk_key)
        if len(rows) < chunks_per_subject:
            raise ValueError(
                f"Subject {subject_id} has {len(rows)} chunks; fixed K={chunks_per_subject} required"
            )
        indices = evenly_spaced_indices(len(rows), chunks_per_subject)
        subject_rows: list[dict[str, Any]] = []
        for selected_position, index in enumerate(indices):
            row = dict(rows[index])
            row["selected_position"] = selected_position
            row["numeric_chunk_number"] = numeric_chunk_key(row)[0]
            subject_rows.append(row)
        selected.extend(subject_rows)
        selected_ids[subject_id] = [str(row["sample_id"]) for row in subject_rows]
    return selected, selected_ids


def select_smoke_subjects(
    rows: list[dict[str, Any]],
    *,
    per_class_per_partition: int,
) -> list[dict[str, Any]]:
    if per_class_per_partition <= 0:
        return rows
    subjects: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: (item["partition"], item["subject_id"])):
        subject_id = str(row["subject_id"])
        if subject_id in seen:
            continue
        seen.add(subject_id)
        subjects[(str(row["partition"]), int(row["label"]))].append(subject_id)
    keep: set[str] = set()
    for partition in PARTITIONS:
        for label in (0, 1):
            candidates = subjects[(partition, label)]
            if len(candidates) < per_class_per_partition:
                raise ValueError(f"Not enough {partition}/label={label} subjects for smoke run")
            keep.update(candidates[:per_class_per_partition])
    return [row for row in rows if str(row["subject_id"]) in keep]


def _model_file_hashes(model_dir: Path) -> dict[str, str]:
    allowed = {
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    }
    paths = [path for path in model_dir.iterdir() if path.is_file() and path.name in allowed]
    if not any(path.name in {"pytorch_model.bin", "model.safetensors"} for path in paths):
        raise FileNotFoundError(f"No WavLM weight file found under {model_dir}")
    return {path.name: _sha256_file(path) for path in sorted(paths)}


def download_model(model_id: str, model_dir: Path) -> dict[str, Any]:
    info = HfApi().model_info(model_id)
    revision = str(info.sha)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        allow_patterns=(
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "README.md",
            "LICENSE*",
        ),
    )
    payload = {
        "downloaded_at_utc": _utc_now(),
        "model_id": model_id,
        "resolved_revision": revision,
        "model_dir": str(model_dir.resolve()),
        "file_sha256": _model_file_hashes(model_dir),
    }
    _atomic_json(model_dir / "download_provenance.json", payload)
    return payload


def convert_model_to_safetensors(model_id: str, model_dir: Path) -> dict[str, Any]:
    """Convert a pinned official PyTorch checkpoint under a CVE-safe Torch.

    This action intentionally refuses Torch versions older than 2.6.  It performs
    exact tensor-key, shape, dtype, and value parity checks before atomically
    publishing the derived safetensors file.
    """

    from packaging.version import Version
    from safetensors import __version__ as safetensors_version
    from safetensors.torch import load_file, save_file
    from transformers import __version__ as transformers_version

    torch_base_version = torch.__version__.split("+", maxsplit=1)[0]
    if Version(torch_base_version) < Version("2.6"):
        raise RuntimeError(
            "Safetensors conversion requires Torch >=2.6 because the source is a "
            "pickle-based pytorch_model.bin checkpoint."
        )
    source_path = model_dir / "pytorch_model.bin"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    provenance_path = model_dir / "download_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Pinned download provenance is required before conversion: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("model_id") != model_id:
        raise ValueError("Model ID does not match pinned download provenance")
    source_sha256 = _sha256_file(source_path)
    recorded_source_sha256 = provenance.get("file_sha256", {}).get("pytorch_model.bin")
    if recorded_source_sha256 != source_sha256:
        raise ValueError("Source checkpoint hash does not match download provenance")

    source_state = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(source_state, dict) or not source_state:
        raise TypeError("Expected a non-empty state dictionary in pytorch_model.bin")
    non_tensors = sorted(key for key, value in source_state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise TypeError(f"Checkpoint contains non-tensor values: {non_tensors[:10]}")
    source_state = {str(key): value.contiguous() for key, value in source_state.items()}

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".safetensors", dir=model_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(source_state, str(temporary), metadata={"format": "pt"})
        converted_state = load_file(str(temporary), device="cpu")
        source_keys = set(source_state)
        converted_keys = set(converted_state)
        keys_match = source_keys == converted_keys
        shapes_match = keys_match and all(
            source_state[key].shape == converted_state[key].shape for key in source_keys
        )
        dtypes_match = keys_match and all(
            source_state[key].dtype == converted_state[key].dtype for key in source_keys
        )
        values_match = keys_match and all(
            torch.equal(source_state[key], converted_state[key]) for key in source_keys
        )
        parity = {
            "tensor_keys_match": bool(keys_match),
            "tensor_shapes_match": bool(shapes_match),
            "tensor_dtypes_match": bool(dtypes_match),
            "tensor_values_match": bool(values_match),
            "tensor_count": len(source_state),
        }
        if not all(
            parity[key]
            for key in (
                "tensor_keys_match",
                "tensor_shapes_match",
                "tensor_dtypes_match",
                "tensor_values_match",
            )
        ):
            raise RuntimeError(f"Safetensors parity verification failed: {parity}")
        destination = model_dir / "model.safetensors"
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    conversion = {
        "converted_at_utc": _utc_now(),
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "output_file": destination.name,
        "output_sha256": _sha256_file(destination),
        "pinned_huggingface_revision": provenance.get("resolved_revision"),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers_version,
            "safetensors": safetensors_version,
        },
        "parity": parity,
    }
    provenance["file_sha256"] = _model_file_hashes(model_dir)
    provenance["safetensors_conversion"] = conversion
    _atomic_json(provenance_path, provenance)
    return conversion


def _cache_signature(
    *,
    manifest_path: Path,
    partitions_path: Path,
    model_id: str,
    model_dir: Path,
    sample_rate: int,
    max_seconds: float,
    chunks_per_subject: int,
) -> dict[str, Any]:
    provenance_path = model_dir / "download_provenance.json"
    model_download = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {"model_id": model_id, "resolved_revision": "unknown"}
    )
    return {
        "schema_version": 1,
        "dataset": "daic",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "partitions_path": str(partitions_path.resolve()),
        "partitions_sha256": _sha256_file(partitions_path),
        "model_id": model_id,
        "model_dir": str(model_dir.resolve()),
        "model_revision": model_download.get("resolved_revision", "unknown"),
        "model_file_sha256": _model_file_hashes(model_dir),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "sample_rate": sample_rate,
        "max_seconds": max_seconds,
        "chunks_per_subject": chunks_per_subject,
        "chunk_order": "numeric trailing integer",
        "chunk_selection": "evenly spaced indices matching src.data.runtime, repaired numeric order",
        "wavlm_transformer_layer_numbers_1_based": list(WAVLM_LAYER_NUMBERS),
        "hidden_states_tuple_indices": list(WAVLM_LAYER_NUMBERS),
        "chunk_pooling": (
            "arithmetic mean over time independently for transformer layers 6, 7, 8; "
            "layer vectors concatenated in ascending order"
        ),
        "chunk_cache_dtype": "float32",
        "audio_integrity": "SHA-256 recorded per selected WAV and validated on cache reuse",
        "frame_level_cache": False,
        "batch_size": 1,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
    }


def prepare_cache(output_root: Path, signature: dict[str, Any]) -> Path:
    cache_root = output_root / "chunk_vectors"
    cache_root.mkdir(parents=True, exist_ok=True)
    signature_path = output_root / "extraction_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        existing_signature = existing.get("signature", existing)
        if existing_signature != signature:
            raise ValueError(
                "Existing WavLM cache signature differs from this run. "
                "Use a new output root; stale vectors will not be reused."
            )
    else:
        _atomic_json(
            signature_path,
            {"created_at_utc": _utc_now(), "signature": signature},
        )
    return cache_root


def _safe_component(value: str) -> str:
    return SAFE_NAME.sub("_", value)


def vector_path(cache_root: Path, row: dict[str, Any]) -> Path:
    return (
        cache_root
        / _safe_component(str(row["partition"]))
        / _safe_component(str(row["subject_id"]))
        / f"{_safe_component(str(row['sample_id']))}.npy"
    )


def vector_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _load_audio(path: Path, *, sample_rate: int, max_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
    values, original_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Unexpected audio shape for {path}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite audio values in {path}")
    original_samples = int(values.shape[0])
    if int(original_rate) != sample_rate:
        values = librosa.resample(
            values,
            orig_sr=int(original_rate),
            target_sr=sample_rate,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
    cap = int(round(sample_rate * max_seconds))
    values = np.ascontiguousarray(values[:cap], dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return values, {
        "original_sample_rate": int(original_rate),
        "original_samples": original_samples,
        "processed_sample_rate": sample_rate,
        "processed_samples": int(values.shape[0]),
        "processed_seconds": float(values.shape[0] / sample_rate),
    }


def _extract_vector(
    values: np.ndarray,
    *,
    feature_extractor: AutoFeatureExtractor,
    model: WavLMModel,
    device: torch.device,
) -> np.ndarray:
    inputs = feature_extractor(
        values,
        sampling_rate=int(feature_extractor.sampling_rate),
        return_tensors="pt",
        padding=False,
    )
    model_inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= max(WAVLM_LAYER_NUMBERS):
            raise ValueError(
                f"WavLM returned {0 if hidden_states is None else len(hidden_states)} hidden "
                f"states; layers {WAVLM_LAYER_NUMBERS} are required"
            )
        pooled_layers = [
            hidden_states[layer_number].mean(dim=1).squeeze(0)
            for layer_number in WAVLM_LAYER_NUMBERS
        ]
        vector = torch.cat(pooled_layers, dim=0)
    result = vector.detach().to(device="cpu", dtype=torch.float32).numpy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"Invalid pooled WavLM vector shape/values: {result.shape}")
    return result


def extract_chunk_vectors(
    rows: list[dict[str, Any]],
    *,
    model_dir: Path,
    cache_root: Path,
    device_name: str,
    sample_rate: int,
    max_seconds: float,
    verify_repeat: bool,
    cpu_threads: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    extraction_started = time.perf_counter()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
    torch.set_num_threads(max(1, cpu_threads))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model_load_started = time.perf_counter()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    if int(feature_extractor.sampling_rate) != sample_rate:
        raise ValueError(
            f"Feature extractor sampling rate {feature_extractor.sampling_rate} != {sample_rate}"
        )
    model = WavLMModel.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    model_load_seconds = time.perf_counter() - model_load_started
    expected_vector_dim = int(model.config.hidden_size) * len(WAVLM_LAYER_NUMBERS)

    vectors: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, Any]] = []
    extracted = reused = 0
    repeat_max_abs_diff: float | None = None
    processed_audio_seconds = 0.0
    chunk_loop_started = time.perf_counter()
    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row["partition"]), str(row["subject_id"]), int(row["selected_position"])),
    )
    for position, row in enumerate(ordered_rows, start=1):
        cache_path = vector_path(cache_root, row)
        metadata_path = vector_metadata_path(cache_path)
        audio_path = Path(str(row["audio_path"]))
        audio_sha256 = _sha256_file(audio_path)
        audio_info: dict[str, Any]
        if cache_path.exists():
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Cached vector metadata is missing: {metadata_path}")
            cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cache_metadata.get("sample_id") != str(row["sample_id"]):
                raise ValueError(f"Cached sample ID mismatch: {metadata_path}")
            if cache_metadata.get("audio_sha256") != audio_sha256:
                raise ValueError(
                    f"Selected WAV changed after caching; refusing stale vector: {audio_path}"
                )
            vector = np.load(cache_path, allow_pickle=False)
            if (
                vector.shape != (expected_vector_dim,)
                or vector.dtype != np.float32
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"Invalid cached vector: {cache_path}")
            vector_sha256 = _sha256_file(cache_path)
            if cache_metadata.get("vector_sha256") != vector_sha256:
                raise ValueError(f"Cached vector hash mismatch: {cache_path}")
            info = sf.info(audio_path)
            processed_samples = min(int(info.frames), int(round(max_seconds * info.samplerate)))
            audio_info = {
                "original_sample_rate": int(info.samplerate),
                "original_samples": int(info.frames),
                "processed_sample_rate": sample_rate,
                "processed_samples": int(round(processed_samples * sample_rate / info.samplerate)),
                "processed_seconds": float(processed_samples / info.samplerate),
            }
            reused += 1
        else:
            values, audio_info = _load_audio(
                audio_path,
                sample_rate=sample_rate,
                max_seconds=max_seconds,
            )
            vector = _extract_vector(
                values,
                feature_extractor=feature_extractor,
                model=model,
                device=device,
            )
            if vector.shape != (expected_vector_dim,):
                raise ValueError(
                    f"Expected WavLM vector dimension {expected_vector_dim}, got {vector.shape}"
                )
            if verify_repeat and repeat_max_abs_diff is None:
                repeated = _extract_vector(
                    values,
                    feature_extractor=feature_extractor,
                    model=model,
                    device=device,
                )
                repeat_max_abs_diff = float(np.max(np.abs(vector - repeated)))
                if not np.array_equal(vector, repeated):
                    raise RuntimeError(
                        f"Repeated deterministic extraction differed; max_abs_diff={repeat_max_abs_diff}"
                    )
            _atomic_npy(cache_path, vector.astype(np.float32, copy=False))
            vector_sha256 = _sha256_file(cache_path)
            _atomic_json(
                metadata_path,
                {
                    "audio_path": str(audio_path),
                    "audio_sha256": audio_sha256,
                    "sample_id": str(row["sample_id"]),
                    "vector_dim": expected_vector_dim,
                    "vector_dtype": "float32",
                    "vector_sha256": vector_sha256,
                },
            )
            extracted += 1
        processed_audio_seconds += float(audio_info["processed_seconds"])
        vectors[str(row["sample_id"])] = vector
        index_rows.append(
            {
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha256,
                "cache_path": str(cache_path.relative_to(cache_root.parent)),
                "cache_metadata_path": str(metadata_path.relative_to(cache_root.parent)),
                "label": int(row["label"]),
                "numeric_chunk_number": int(row["numeric_chunk_number"]),
                "partition": str(row["partition"]),
                "sample_id": str(row["sample_id"]),
                "selected_position": int(row["selected_position"]),
                "subject_id": str(row["subject_id"]),
                "vector_dim": int(vector.shape[0]),
                "vector_sha256": vector_sha256,
                **audio_info,
            }
        )
        if position == 1 or position % 25 == 0 or position == len(ordered_rows):
            print(
                f"[wavlm] {position}/{len(ordered_rows)} "
                f"extracted={extracted} reused={reused} sample={row['sample_id']}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    chunk_loop_seconds = time.perf_counter() - chunk_loop_started
    cuda_peak_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    cuda_peak_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vectors, index_rows, {
        "device": str(device),
        "extracted_vectors": extracted,
        "reused_vectors": reused,
        "repeat_max_abs_diff": repeat_max_abs_diff,
        "model_load_seconds": model_load_seconds,
        "chunk_loop_seconds": chunk_loop_seconds,
        "elapsed_seconds": time.perf_counter() - extraction_started,
        "processed_audio_seconds": processed_audio_seconds,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }


def build_subject_features(
    selected_rows: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    *,
    chunks_per_subject: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["subject_id"])].append(row)
    subjects: dict[str, dict[str, Any]] = {}
    for subject_id in sorted(grouped):
        rows = sorted(grouped[subject_id], key=lambda row: int(row["selected_position"]))
        if len(rows) != chunks_per_subject:
            raise ValueError(f"Subject {subject_id} does not have exactly K={chunks_per_subject}")
        matrix = np.stack([vectors[str(row["sample_id"])] for row in rows]).astype(np.float64)
        feature = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0, ddof=0)])
        labels = {int(row["label"]) for row in rows}
        partitions = {str(row["partition"]) for row in rows}
        if len(labels) != 1 or len(partitions) != 1:
            raise ValueError(f"Inconsistent subject metadata for {subject_id}")
        subjects[subject_id] = {
            "feature": feature,
            "label": labels.pop(),
            "partition": partitions.pop(),
            "sample_ids": [str(row["sample_id"]) for row in rows],
        }
    return subjects


def _partition_arrays(
    subjects: dict[str, dict[str, Any]],
    partition: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    subject_ids = sorted(
        subject_id
        for subject_id, row in subjects.items()
        if str(row["partition"]) == partition
    )
    features = np.stack([subjects[subject_id]["feature"] for subject_id in subject_ids])
    labels = np.asarray([subjects[subject_id]["label"] for subject_id in subject_ids], dtype=np.int64)
    return subject_ids, features, labels


def _classifier(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    penalty="l2",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    clipped_scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    predictions = (clipped_scores >= 0.5).astype(np.int64)
    result = classification_metrics(y_true.tolist(), predictions.tolist())
    has_both_labels = len(set(y_true.tolist())) == 2
    result["auroc"] = (
        binary_auroc(y_true.tolist(), clipped_scores.tolist()) if has_both_labels else None
    )
    result["average_precision"] = (
        float(average_precision_score(y_true, clipped_scores)) if np.any(y_true == 1) else None
    )
    # The repository metric already treats an absent class as zero recall, which
    # keeps bootstrap resamples defined without sklearn's single-class warning.
    result["balanced_accuracy"] = float(result["macro_recall"])
    result["log_loss"] = float(log_loss(y_true, clipped_scores, labels=[0, 1]))
    result["brier_score"] = float(np.mean((clipped_scores - y_true) ** 2))
    return result


def _fit_select_refit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, Any], Pipeline, np.ndarray, np.ndarray]:
    candidates: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_c: float | None = None
    best_val_scores: np.ndarray | None = None
    for c_value in sorted(c_grid):
        model = _classifier(c_value, seed)
        model.fit(train_x, train_y)
        scores = model.predict_proba(val_x)[:, 1]
        metrics = _metrics(val_y, scores)
        candidates.append({"c": float(c_value), "metrics": metrics})
        key = (float(metrics["log_loss"]), float(c_value))
        if best_key is None or key < best_key:
            best_key = key
            best_c = float(c_value)
            best_val_scores = scores
    assert best_c is not None and best_val_scores is not None
    train_val_x = np.concatenate([train_x, val_x], axis=0)
    train_val_y = np.concatenate([train_y, val_y], axis=0)
    final_model = _classifier(best_c, seed)
    final_model.fit(train_val_x, train_val_y)
    test_scores = final_model.predict_proba(test_x)[:, 1]
    return (
        {
            "selection_metric": "minimum validation log loss, then smaller C",
            "selected_c": best_c,
            "validation_metrics": _metrics(val_y, best_val_scores),
            "test_metrics": _metrics(test_y, test_scores),
            "candidates": candidates,
        },
        final_model,
        best_val_scores,
        test_scores,
    )


def derangement(subject_ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    if len(subject_ids) < 2:
        raise ValueError("At least two subjects are required for shuffled-audio control")
    original = np.asarray(subject_ids, dtype=object)
    for _ in range(10000):
        permuted = original[rng.permutation(len(original))]
        if bool(np.all(permuted != original)):
            return {str(target): str(source) for target, source in zip(original, permuted)}
    # A deterministic one-position rotation is always a derangement for n >= 2.
    permuted = np.roll(original, 1)
    return {str(target): str(source) for target, source in zip(original, permuted)}


def shuffled_subject_features(
    subjects: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    rng = np.random.default_rng(seed)
    shuffled: dict[str, dict[str, Any]] = {}
    mappings: dict[str, dict[str, str]] = {}
    for partition in PARTITIONS:
        subject_ids = sorted(
            subject_id
            for subject_id, row in subjects.items()
            if str(row["partition"]) == partition
        )
        mapping = derangement(subject_ids, rng)
        mappings[partition] = mapping
        for target, source in mapping.items():
            shuffled[target] = {
                "feature": subjects[source]["feature"].copy(),
                "label": int(subjects[target]["label"]),
                "partition": partition,
                "sample_ids": list(subjects[source]["sample_ids"]),
                "source_subject_id": source,
            }
            if target == source:
                raise AssertionError("Shuffled-audio control retained an aligned subject")
    return shuffled, mappings


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_subject_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile intervals from resampling test subjects with replacement."""

    if repeats <= 0:
        return {"repeats": 0, "seed": seed, "unit": "subject", "intervals": {}}
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "positive_f1",
        "macro_f1",
        "auroc",
        "average_precision",
        "log_loss",
        "brier_score",
    )
    samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = _metrics(y_true[indices], scores[indices])
        for metric in metric_names:
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                samples[metric].append(float(value))
    intervals: dict[str, dict[str, Any]] = {}
    for metric, values in samples.items():
        if values:
            intervals[metric] = {
                "lower_2.5pct": float(np.quantile(values, 0.025)),
                "upper_97.5pct": float(np.quantile(values, 0.975)),
                "valid_replicates": len(values),
            }
    return {
        "repeats": repeats,
        "seed": seed,
        "unit": "subject",
        "method": "percentile bootstrap with replacement",
        "intervals": intervals,
    }


def evaluate_subject_baseline(
    subjects: dict[str, dict[str, Any]],
    *,
    c_grid: tuple[float, ...],
    seed: int,
    shuffle_repeats: int,
    bootstrap_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    arrays = {partition: _partition_arrays(subjects, partition) for partition in PARTITIONS}
    train_ids, train_x, train_y = arrays["train"]
    val_ids, val_x, val_y = arrays["val"]
    test_ids, test_x, test_y = arrays["test"]
    primary, final_model, val_scores, test_scores = _fit_select_refit(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        c_grid=c_grid,
        seed=seed,
    )

    train_val_y = np.concatenate([train_y, val_y])
    train_prevalence = float(train_y.mean())
    train_val_prevalence = float(train_val_y.mean())
    train_majority = int(train_prevalence >= 0.5)
    train_val_majority = int(train_val_prevalence >= 0.5)
    majority = {
        "description": (
            "Constant development-set positive prevalence; displayed label and metric "
            "prediction both use probability >= 0.5."
        ),
        "validation_label_from_train": int(train_majority),
        "validation_probability_from_train_prevalence": train_prevalence,
        "validation_metrics": _metrics(
            val_y,
            np.full(val_y.shape, train_prevalence, dtype=np.float64),
        ),
        "test_label_from_train_plus_validation": int(train_val_majority),
        "test_probability_from_train_plus_validation_prevalence": train_val_prevalence,
        "test_metrics": _metrics(
            test_y,
            np.full(test_y.shape, train_val_prevalence, dtype=np.float64),
        ),
    }

    shuffle_rows: list[dict[str, Any]] = []
    first_mapping: dict[str, dict[str, str]] | None = None
    for repeat in range(shuffle_repeats):
        repeat_seed = seed + 1009 * (repeat + 1)
        shuffled, mappings = shuffled_subject_features(subjects, seed=repeat_seed)
        if first_mapping is None:
            first_mapping = mappings
        shuffled_arrays = {
            partition: _partition_arrays(shuffled, partition) for partition in PARTITIONS
        }
        _, shuffled_train_x, shuffled_train_y = shuffled_arrays["train"]
        _, shuffled_val_x, shuffled_val_y = shuffled_arrays["val"]
        _, shuffled_test_x, shuffled_test_y = shuffled_arrays["test"]
        result, _, _, _ = _fit_select_refit(
            shuffled_train_x,
            shuffled_train_y,
            shuffled_val_x,
            shuffled_val_y,
            shuffled_test_x,
            shuffled_test_y,
            c_grid=c_grid,
            seed=repeat_seed,
        )
        shuffle_rows.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "selected_c": result["selected_c"],
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
            }
        )

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "positive_f1",
        "macro_f1",
        "auroc",
        "average_precision",
        "log_loss",
        "brier_score",
    )
    lower_is_better = {"log_loss", "brier_score"}
    shuffle_summary: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        split_summary: dict[str, Any] = {}
        primary_metrics = primary[f"{split_name}_metrics"]
        for metric in metric_names:
            values = [float(row[f"{split_name}_metrics"][metric]) for row in shuffle_rows]
            observed = float(primary_metrics[metric])
            as_good_or_better = (
                sum(value <= observed for value in values)
                if metric in lower_is_better
                else sum(value >= observed for value in values)
            )
            split_summary[metric] = {
                **_mean_std(values),
                "empirical_p_as_good_or_better": float(
                    (1 + as_good_or_better) / (1 + len(values))
                ),
                "tail": "less_or_equal" if metric in lower_is_better else "greater_or_equal",
            }
        shuffle_summary[split_name] = split_summary

    predictions: list[dict[str, Any]] = []
    for subject_id, gold, score in zip(val_ids, val_y, val_scores):
        predictions.append(
            {
                "partition": "val",
                "role": "selection",
                "subject_id": subject_id,
                "label": int(gold),
                "score": float(score),
                "prediction": int(score >= 0.5),
            }
        )
    for subject_id, gold, score in zip(test_ids, test_y, test_scores):
        predictions.append(
            {
                "partition": "test",
                "role": "final",
                "subject_id": subject_id,
                "label": int(gold),
                "score": float(score),
                "prediction": int(score >= 0.5),
            }
        )

    scaler = final_model.named_steps["scale"]
    logistic = final_model.named_steps["logistic"]
    model_arrays = {
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": logistic.coef_.astype(np.float64),
        "intercept": logistic.intercept_.astype(np.float64),
    }
    result = {
        "primary": primary,
        "test_bootstrap_95pct": bootstrap_subject_intervals(
            test_y,
            test_scores,
            repeats=bootstrap_repeats,
            seed=seed + 1,
        ),
        "majority_control": majority,
        "shuffled_audio_control": {
            "description": (
                "Complete K=4 subject audio bundles are deranged within each partition; "
                "target subject IDs and labels remain unchanged."
            ),
            "repeats": shuffle_repeats,
            "first_repeat_mapping": first_mapping,
            "summary": shuffle_summary,
            "rows": shuffle_rows,
        },
    }
    return result, predictions, model_arrays


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    partitions_path = Path(args.partitions).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.download_if_missing and not (model_dir / "config.json").exists():
        download_model(args.model_id, model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    rows, partition_by_subject = load_fixed_daic_rows(manifest_path, partitions_path)
    selected_rows, selected_ids = select_fixed_k_rows(
        rows,
        chunks_per_subject=args.chunks_per_subject,
    )
    selected_rows = select_smoke_subjects(
        selected_rows,
        per_class_per_partition=args.smoke_subjects_per_class,
    )
    selected_subjects = sorted({str(row["subject_id"]) for row in selected_rows})
    selected_ids = {
        subject_id: selected_ids[subject_id]
        for subject_id in selected_subjects
    }

    signature = _cache_signature(
        manifest_path=manifest_path,
        partitions_path=partitions_path,
        model_id=args.model_id,
        model_dir=model_dir,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        chunks_per_subject=args.chunks_per_subject,
    )
    cache_root = prepare_cache(output_root, signature)
    run_dir = output_root / "runs" / _safe_component(args.run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_payload = {
        "chunks_per_subject": args.chunks_per_subject,
        "ordering": "numeric trailing chunk number",
        "selection": "evenly spaced indices",
        "num_subjects": len(selected_subjects),
        "num_chunks": len(selected_rows),
        "sample_ids_by_subject": selected_ids,
    }
    _atomic_json(run_dir / "selected_chunks.json", selection_payload)

    vectors, index_rows, extraction = extract_chunk_vectors(
        selected_rows,
        model_dir=model_dir,
        cache_root=cache_root,
        device_name=args.device,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        verify_repeat=args.verify_repeat,
        cpu_threads=args.cpu_threads,
    )
    _atomic_jsonl(run_dir / "chunk_index.jsonl", index_rows)
    subjects = build_subject_features(
        selected_rows,
        vectors,
        chunks_per_subject=args.chunks_per_subject,
    )
    counts = {
        partition: {
            "subjects": sum(row["partition"] == partition for row in subjects.values()),
            "negative": sum(
                row["partition"] == partition and int(row["label"]) == 0
                for row in subjects.values()
            ),
            "positive": sum(
                row["partition"] == partition and int(row["label"]) == 1
                for row in subjects.values()
            ),
        }
        for partition in PARTITIONS
    }
    evaluation, predictions, model_arrays = evaluate_subject_baseline(
        subjects,
        c_grid=tuple(float(value) for value in args.c_grid),
        seed=args.seed,
        shuffle_repeats=args.shuffle_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    _atomic_jsonl(run_dir / "subject_predictions.jsonl", predictions)
    np.savez_compressed(run_dir / "linear_model.npz", **model_arrays)
    subject_ids = sorted(subjects)
    np.savez_compressed(
        run_dir / "subject_features.npz",
        subject_ids=np.asarray(subject_ids),
        features=np.stack([subjects[subject_id]["feature"] for subject_id in subject_ids]),
        labels=np.asarray([subjects[subject_id]["label"] for subject_id in subject_ids]),
        partitions=np.asarray([subjects[subject_id]["partition"] for subject_id in subject_ids]),
    )
    result = {
        "schema_version": 1,
        "completed_at_utc": _utc_now(),
        "run_label": args.run_label,
        "smoke_subjects_per_class": args.smoke_subjects_per_class,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "partitions_path": str(partitions_path),
        "partitions_sha256": _sha256_file(partitions_path),
        "fixed_partition_counts": counts,
        "fixed_partition_subjects_total": len(partition_by_subject),
        "selected_subjects": len(subjects),
        "selected_chunks": len(selected_rows),
        "selected_sample_ids_sha256": _sha256_strings(
            str(row["sample_id"])
            for row in sorted(selected_rows, key=lambda item: str(item["sample_id"]))
        ),
        "representation": {
            "model_id": args.model_id,
            "model_dir": str(model_dir),
            "chunks_per_subject": args.chunks_per_subject,
            "wavlm_transformer_layer_numbers_1_based": list(WAVLM_LAYER_NUMBERS),
            "hidden_states_tuple_indices": list(WAVLM_LAYER_NUMBERS),
            "chunk_pooling": (
                "per-layer time mean for transformer layers 6, 7, 8; "
                "concatenated in ascending layer order"
            ),
            "subject_pooling": "chunk-vector mean concatenated with chunk-vector population std",
            "feature_dimension": int(next(iter(subjects.values()))["feature"].shape[0]),
            "batch_size": 1,
            "numeric_chunk_order": True,
            "frame_level_cache": False,
        },
        "extraction": extraction,
        "evaluation": evaluation,
        "provenance": {
            **_git_provenance(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers_model_revision": signature["model_revision"],
            "model_file_sha256": signature["model_file_sha256"],
        },
    }
    _atomic_json(run_dir / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--convert-to-safetensors", action="store_true")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--partitions", default=str(DEFAULT_PARTITIONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-label", default="full_numeric_k4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--chunks-per-subject", type=int, default=4)
    parser.add_argument("--smoke-subjects-per-class", type=int, default=0)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    args = parser.parse_args()
    if args.chunks_per_subject != 4:
        parser.error("E1b requires exactly --chunks-per-subject 4")
    if args.shuffle_repeats < 1:
        parser.error("--shuffle-repeats must be at least 1")
    if args.bootstrap_repeats < 0:
        parser.error("--bootstrap-repeats cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if args.download_only:
        payload = download_model(args.model_id, model_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.convert_to_safetensors:
        payload = convert_model_to_safetensors(args.model_id, model_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    run(args)


if __name__ == "__main__":
    main()
