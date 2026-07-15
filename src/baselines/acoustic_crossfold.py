"""Frozen N0/N1 DAIC acoustic protocol and nested cross-fold linear ceilings.

This module implements the immediate actions in
``docs/NEXT_LOCAL_EXPERIMENT_PLAN_2026-07-15.md``.  It deliberately does not
evaluate the official DAIC test partition: N1 produces development out-of-fold
(OOF) predictions only.  Test evaluation remains gated on the later MIL and
analysis freeze.

The implementation consumes the existing fixed-K eGeMAPSv02 and frozen WavLM
chunk caches.  It validates every selected chunk, cache signature, vector, and
audio hash before fitting a model.  All scientific splits operate at subject
level.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.baselines.egemaps_ceiling import (
    EXPECTED_EGEMAPS_DIMENSION,
    natural_key,
    opensmile_config_tree,
    select_equal_chunks,
    validate_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/manifests/daic_manifest.jsonl"
DEFAULT_PARTITIONS = PROJECT_ROOT / "outputs/splits/daic_subject_partitions.json"
DEFAULT_EGEMAPS_CACHE = PROJECT_ROOT / "outputs/baselines/daic_egemaps_v02_fixedk4"
DEFAULT_WAVLM_CACHE = (
    PROJECT_ROOT / "outputs/baselines/e1b_wavlm_base_plus_daic_layers678_full_final"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/baselines/daic_acoustic_crossfold_n0n1_20260715"
WAVLM_RUN_RELATIVE = Path("runs/full_numeric_k4_layers678")
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"

OUTER_FOLDS = 5
INNER_FOLDS = 4
CHUNKS_PER_SUBJECT = 4
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
SHUFFLE_REPEATS = 100
BOOTSTRAP_REPEATS = 10_000
OUTER_SEED = 20260715
INNER_SEED = 20260716
SHUFFLE_SEED = 20260717
BOOTSTRAP_SEED = 20260718
MODEL_SEED = 20260719
THRESHOLD = 0.5

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "positive_f1",
    "macro_f1",
    "auroc",
    "auprc",
    "log_loss",
    "brier_score",
)
PRIMARY_COMPARISON_METRICS = (
    "auroc",
    "auprc",
    "log_loss",
    "balanced_accuracy",
    "macro_f1",
    "positive_f1",
)
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "positive_f1",
    "macro_f1",
    "auroc",
    "auprc",
}
CHUNK_SUFFIX = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class CachePaths:
    manifest: Path
    partitions: Path
    egemaps: Path
    wavlm: Path
    output: Path


@dataclass
class FeatureBundle:
    family: str
    chunk_dimension: int
    feature_names: list[str]
    vectors: dict[str, np.ndarray]
    validation: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_chunk_number(sample_id: str) -> int:
    match = CHUNK_SUFFIX.search(str(sample_id).strip())
    if match is None:
        raise ValueError(f"Cannot parse trailing numeric chunk number: {sample_id!r}")
    return int(match.group(1))


def sample_kind(sample_id: str) -> str:
    if "_random_segment_" in sample_id:
        return "random_segment"
    if "_segment_" in sample_id:
        return "segment"
    return "other"


def load_selected_protocol_rows(
    manifest_path: Path,
    partitions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    partitions_payload = read_json(partitions_path)
    normalized, partitions = validate_inputs(manifest, partitions_payload, require_audio=True)
    selected, selection_audit = select_equal_chunks(normalized, CHUNKS_PER_SUBJECT)
    selected_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_subject[str(row["subject_id"])].append(row)
    for subject_id, rows in selected_by_subject.items():
        rows.sort(key=lambda row: numeric_chunk_number(str(row["sample_id"])))
        if len(rows) != CHUNKS_PER_SUBJECT:
            raise ValueError(f"Subject {subject_id} does not have exactly K=4 selected chunks")
        numbers = [numeric_chunk_number(str(row["sample_id"])) for row in rows]
        expected = [1, 4, 7, 10] if len(
            [item for item in normalized if str(item["subject_id"]) == subject_id]
        ) == 10 else [1, 6, 10, 15]
        if numbers != expected:
            raise ValueError(
                f"Numeric K=4 selection mismatch for {subject_id}: {numbers} != {expected}"
            )
        for position, row in enumerate(rows):
            row["selected_position"] = position
            row["numeric_chunk_number"] = numbers[position]
    selected = [
        row
        for subject_id in sorted(selected_by_subject, key=natural_key)
        for row in selected_by_subject[subject_id]
    ]
    development = [meta for meta in partitions.values() if meta["partition"] in {"train", "val"}]
    locked_test = [meta for meta in partitions.values() if meta["partition"] == "test"]
    if len(development) != 142 or len(locked_test) != 47:
        raise ValueError(
            f"Locked DAIC protocol requires 142 development and 47 test subjects; "
            f"found {len(development)} and {len(locked_test)}"
        )
    return selected, partitions, selection_audit


def selected_sample_payload(
    selected_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    subjects: dict[str, dict[str, Any]] = {}
    selected_by_id: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        audio_path = Path(str(row["audio_path"])).resolve()
        audio_hash = sha256_file(audio_path)
        sample_id = str(row["sample_id"])
        record = {
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "audio_sha256": audio_hash,
            "numeric_chunk_number": int(row["numeric_chunk_number"]),
            "selected_position": int(row["selected_position"]),
            "sample_kind": sample_kind(sample_id),
        }
        selected_by_id[sample_id] = {
            **record,
            "subject_id": str(row["subject_id"]),
            "partition": str(row["split_original"]),
            "label": int(row["label"]),
        }
        subject_id = str(row["subject_id"])
        subject = subjects.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "original_partition": str(row["split_original"]),
                "label": int(row["label"]),
                "samples": [],
            },
        )
        if (
            subject["label"] != int(row["label"])
            or subject["original_partition"] != str(row["split_original"])
        ):
            raise ValueError(f"Inconsistent selected metadata for subject {subject_id}")
        subject["samples"].append(record)
    if len(selected_by_id) != len(selected_rows):
        raise ValueError("Selected sample IDs are duplicated")
    for subject in subjects.values():
        subject["samples"].sort(key=lambda row: int(row["selected_position"]))
    payload = {
        "schema_version": 1,
        "chunks_per_subject": CHUNKS_PER_SUBJECT,
        "ordering": "trailing numeric suffix; ordinal only, not verified chronology",
        "selection": "evenly spaced inclusive endpoints",
        "subject_count": len(subjects),
        "sample_count": len(selected_by_id),
        "subjects": {
            subject_id: subjects[subject_id]
            for subject_id in sorted(subjects, key=natural_key)
        },
    }
    return payload, selected_by_id


def _assert_exact_ids(observed: Sequence[str], expected: set[str], context: str) -> None:
    counts = Counter(observed)
    duplicated = sorted((sample for sample, count in counts.items() if count != 1), key=natural_key)
    if duplicated:
        raise ValueError(f"{context} has duplicated IDs: {duplicated[:10]}")
    observed_set = set(observed)
    if observed_set != expected:
        raise ValueError(
            f"{context} sample IDs differ from frozen K=4 selection: "
            f"missing={sorted(expected - observed_set, key=natural_key)[:10]}, "
            f"extra={sorted(observed_set - expected, key=natural_key)[:10]}"
        )


def validate_egemaps_cache(
    cache_root: Path,
    selected_by_id: Mapping[str, dict[str, Any]],
    *,
    manifest_sha256: str,
    partitions_sha256: str,
) -> FeatureBundle:
    chunk_path = cache_root / "chunk_features.npz"
    metadata_path = cache_root / "chunk_feature_metadata.jsonl"
    provenance_path = cache_root / "provenance.json"
    run_config_path = cache_root / "run_config.json"
    for path in (chunk_path, metadata_path, provenance_path, run_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = read_json(provenance_path)
    if provenance.get("inputs", {}).get("manifest", {}).get("sha256") != manifest_sha256:
        raise ValueError("eGeMAPS cache manifest hash is stale")
    if provenance.get("inputs", {}).get("partitions", {}).get("sha256") != partitions_sha256:
        raise ValueError("eGeMAPS cache partition hash is stale")
    opensmile = provenance.get("opensmile", {})
    binary = Path(str(opensmile.get("binary", "")))
    config = Path(str(opensmile.get("config", "")))
    if not binary.is_file() or sha256_file(binary) != opensmile.get("binary_sha256"):
        raise ValueError("eGeMAPS openSMILE binary is missing or hash-mismatched")
    if not config.is_file() or sha256_file(config) != opensmile.get("config_file_sha256"):
        raise ValueError("eGeMAPS openSMILE config is missing or hash-mismatched")
    config_tree_hash, _, config_tree_count = opensmile_config_tree(config)
    if (
        config_tree_hash != opensmile.get("config_tree_sha256")
        or config_tree_count != int(opensmile.get("config_tree_file_count", -1))
    ):
        raise ValueError("eGeMAPS openSMILE config tree is stale")

    with np.load(chunk_path, allow_pickle=False) as cache:
        required = {"sample_ids", "subject_ids", "partitions", "labels", "feature_names", "features"}
        if set(cache.files) != required:
            raise ValueError(f"Unexpected eGeMAPS NPZ fields: {cache.files}")
        sample_ids = [str(value) for value in cache["sample_ids"].tolist()]
        subject_ids = [str(value) for value in cache["subject_ids"].tolist()]
        partitions = [str(value) for value in cache["partitions"].tolist()]
        labels = np.asarray(cache["labels"], dtype=np.int64)
        feature_names = [str(value) for value in cache["feature_names"].tolist()]
        features = np.asarray(cache["features"], dtype=np.float64)
    expected_ids = set(selected_by_id)
    _assert_exact_ids(sample_ids, expected_ids, "eGeMAPS chunk feature cache")
    if len(feature_names) != EXPECTED_EGEMAPS_DIMENSION or len(set(feature_names)) != len(feature_names):
        raise ValueError("eGeMAPS feature schema is missing, duplicated, or dimensionally inconsistent")
    if features.shape != (len(expected_ids), EXPECTED_EGEMAPS_DIMENSION):
        raise ValueError(f"Invalid eGeMAPS feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("eGeMAPS feature cache contains non-finite values")

    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    vectors: dict[str, np.ndarray] = {}
    metadata_rows = read_jsonl(metadata_path)
    metadata_ids = [str(row.get("sample_id", "")) for row in metadata_rows]
    _assert_exact_ids(metadata_ids, expected_ids, "eGeMAPS metadata")
    extraction_signature = str(opensmile.get("extraction_signature_sha256", ""))
    for metadata in metadata_rows:
        sample_id = str(metadata["sample_id"])
        expected = selected_by_id[sample_id]
        position = index[sample_id]
        if (
            str(subject_ids[position]) != expected["subject_id"]
            or str(partitions[position]) != expected["partition"]
            or int(labels[position]) != expected["label"]
        ):
            raise ValueError(f"eGeMAPS metadata mismatch for {sample_id}")
        if (
            str(metadata.get("audio_sha256")) != expected["audio_sha256"]
            or str(metadata.get("subject_id")) != expected["subject_id"]
            or str(metadata.get("partition")) != expected["partition"]
            or int(metadata.get("label")) != expected["label"]
            or int(metadata.get("nonfinite_count", -1)) != 0
        ):
            raise ValueError(f"eGeMAPS per-chunk provenance mismatch for {sample_id}")
        per_chunk_path = Path(str(metadata.get("cache_path", "")))
        if not per_chunk_path.is_file():
            raise FileNotFoundError(per_chunk_path)
        with np.load(per_chunk_path, allow_pickle=False) as per_chunk:
            if str(per_chunk["audio_sha256"].item()) != expected["audio_sha256"]:
                raise ValueError(f"eGeMAPS cached audio hash mismatch for {sample_id}")
            if str(per_chunk["extraction_signature_sha256"].item()) != extraction_signature:
                raise ValueError(f"eGeMAPS extraction signature mismatch for {sample_id}")
            cached_names = [str(value) for value in per_chunk["feature_names"].tolist()]
            cached_vector = np.asarray(per_chunk["features"], dtype=np.float64)
        if cached_names != feature_names or cached_vector.shape != (EXPECTED_EGEMAPS_DIMENSION,):
            raise ValueError(f"eGeMAPS per-chunk schema mismatch for {sample_id}")
        if not np.isfinite(cached_vector).all() or not np.array_equal(cached_vector, features[position]):
            raise ValueError(f"eGeMAPS container/per-chunk feature mismatch for {sample_id}")
        vectors[sample_id] = cached_vector

    extraction_code = PROJECT_ROOT / "src/baselines/egemaps_ceiling.py"
    return FeatureBundle(
        family="egemaps",
        chunk_dimension=EXPECTED_EGEMAPS_DIMENSION,
        feature_names=feature_names,
        vectors=vectors,
        validation={
            "status": "passed",
            "family": "eGeMAPSv02",
            "sample_count": len(vectors),
            "chunk_dimension": EXPECTED_EGEMAPS_DIMENSION,
            "subject_dimension": EXPECTED_EGEMAPS_DIMENSION * 2,
            "manifest_sha256": manifest_sha256,
            "partitions_sha256": partitions_sha256,
            "feature_container_sha256": sha256_file(chunk_path),
            "metadata_sha256": sha256_file(metadata_path),
            "provenance_sha256": sha256_file(provenance_path),
            "run_config_sha256": sha256_file(run_config_path),
            "extraction_code_path": str(extraction_code),
            "extraction_code_sha256": sha256_file(extraction_code),
            "extraction_code_hash_note": (
                "The legacy eGeMAPS provenance did not store its Python script hash. N0 registers "
                "the current committed extractor after exact reconciliation of all container vectors "
                "against audio-hash- and openSMILE-signature-keyed per-chunk caches."
            ),
            "opensmile_version": opensmile.get("version"),
            "opensmile_binary_sha256": opensmile.get("binary_sha256"),
            "opensmile_config_tree_sha256": opensmile.get("config_tree_sha256"),
            "extraction_signature_sha256": extraction_signature,
            "finite": True,
            "unique_sample_ids": True,
        },
    )


def validate_wavlm_cache(
    cache_root: Path,
    selected_by_id: Mapping[str, dict[str, Any]],
    *,
    manifest_sha256: str,
    partitions_sha256: str,
) -> FeatureBundle:
    signature_path = cache_root / "extraction_signature.json"
    run_root = cache_root / WAVLM_RUN_RELATIVE
    selected_path = run_root / "selected_chunks.json"
    index_path = run_root / "chunk_index.jsonl"
    for path in (signature_path, selected_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    signature = read_json(signature_path).get("signature", {})
    expected_signature_values = {
        "manifest_sha256": manifest_sha256,
        "partitions_sha256": partitions_sha256,
        "model_id": "microsoft/wavlm-base-plus",
        "model_revision": WAVLM_REVISION,
        "chunks_per_subject": CHUNKS_PER_SUBJECT,
    }
    for key, expected in expected_signature_values.items():
        if signature.get(key) != expected:
            raise ValueError(f"WavLM cache signature mismatch for {key}: {signature.get(key)!r}")
    if signature.get("hidden_states_tuple_indices") != [6, 7, 8]:
        raise ValueError("WavLM cache does not contain the locked layers 6, 7, and 8")
    extraction_code = PROJECT_ROOT / "baselines/wavlm_frozen_subject_baseline.py"
    current_code_hash = sha256_file(extraction_code)
    if signature.get("script_sha256") != current_code_hash:
        raise ValueError("WavLM extraction-code hash is stale")
    model_dir = Path(str(signature.get("model_dir", "")))
    for filename, expected_hash in signature.get("model_file_sha256", {}).items():
        model_file = model_dir / filename
        if not model_file.is_file() or sha256_file(model_file) != expected_hash:
            raise ValueError(f"WavLM model file is missing or hash-mismatched: {model_file}")

    selected_payload = read_json(selected_path)
    selected_ids = selected_payload.get("sample_ids_by_subject", {})
    flattened_selected = [
        str(sample_id)
        for subject_id in sorted(selected_ids, key=natural_key)
        for sample_id in selected_ids[subject_id]
    ]
    expected_ids = set(selected_by_id)
    _assert_exact_ids(flattened_selected, expected_ids, "WavLM selected-chunk artifact")
    for subject_id, sample_ids in selected_ids.items():
        expected_subject_ids = [
            sample_id
            for sample_id, row in selected_by_id.items()
            if row["subject_id"] == str(subject_id)
        ]
        expected_subject_ids.sort(key=numeric_chunk_number)
        if [str(value) for value in sample_ids] != expected_subject_ids:
            raise ValueError(f"WavLM numeric K=4 selection mismatch for subject {subject_id}")

    index_rows = read_jsonl(index_path)
    index_ids = [str(row.get("sample_id", "")) for row in index_rows]
    _assert_exact_ids(index_ids, expected_ids, "WavLM chunk index")
    vectors: dict[str, np.ndarray] = {}
    for row in index_rows:
        sample_id = str(row["sample_id"])
        expected = selected_by_id[sample_id]
        if (
            str(row.get("subject_id")) != expected["subject_id"]
            or str(row.get("partition")) != expected["partition"]
            or int(row.get("label")) != expected["label"]
            or int(row.get("numeric_chunk_number")) != expected["numeric_chunk_number"]
            or int(row.get("selected_position")) != expected["selected_position"]
            or str(row.get("audio_sha256")) != expected["audio_sha256"]
            or int(row.get("vector_dim", -1)) != 2304
        ):
            raise ValueError(f"WavLM chunk-index metadata mismatch for {sample_id}")
        vector_path = cache_root / str(row["cache_path"])
        metadata_path = cache_root / str(row["cache_metadata_path"])
        if not vector_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing WavLM vector or sidecar for {sample_id}")
        vector_hash = sha256_file(vector_path)
        if vector_hash != str(row.get("vector_sha256")):
            raise ValueError(f"WavLM vector hash mismatch for {sample_id}")
        metadata = read_json(metadata_path)
        if (
            metadata.get("sample_id") != sample_id
            or metadata.get("audio_sha256") != expected["audio_sha256"]
            or metadata.get("vector_sha256") != vector_hash
            or metadata.get("vector_dtype") != "float32"
            or int(metadata.get("vector_dim", -1)) != 2304
        ):
            raise ValueError(f"WavLM vector sidecar mismatch for {sample_id}")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.shape != (2304,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid WavLM vector for {sample_id}: {vector.shape}, {vector.dtype}")
        vectors[sample_id] = vector.astype(np.float64)

    return FeatureBundle(
        family="wavlm",
        chunk_dimension=2304,
        feature_names=[f"wavlm_layers678_{index}" for index in range(2304)],
        vectors=vectors,
        validation={
            "status": "passed",
            "family": "frozen_wavlm_base_plus_layers_6_7_8",
            "sample_count": len(vectors),
            "chunk_dimension": 2304,
            "subject_dimension": 4608,
            "manifest_sha256": manifest_sha256,
            "partitions_sha256": partitions_sha256,
            "model_id": signature["model_id"],
            "model_revision": signature["model_revision"],
            "model_file_sha256": signature.get("model_file_sha256", {}),
            "signature_sha256": sha256_file(signature_path),
            "selected_chunks_sha256": sha256_file(selected_path),
            "chunk_index_sha256": sha256_file(index_path),
            "extraction_code_path": str(extraction_code),
            "extraction_code_sha256": current_code_hash,
            "finite": True,
            "unique_sample_ids": True,
        },
    )


def build_fold_assignments(
    subject_metadata: Mapping[str, dict[str, Any]],
    *,
    outer_seed: int = OUTER_SEED,
    inner_seed: int = INNER_SEED,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    development_ids = sorted(
        (
            subject_id
            for subject_id, row in subject_metadata.items()
            if row["original_partition"] in {"train", "val"}
        ),
        key=natural_key,
    )
    test_ids = sorted(
        (
            subject_id
            for subject_id, row in subject_metadata.items()
            if row["original_partition"] == "test"
        ),
        key=natural_key,
    )
    labels = np.asarray([subject_metadata[subject_id]["label"] for subject_id in development_ids])
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    outer_folds: list[dict[str, Any]] = []
    assignment: dict[str, int] = {}
    dummy = np.zeros(len(development_ids), dtype=np.int8)
    for outer_fold, (train_indices, holdout_indices) in enumerate(outer.split(dummy, labels)):
        train_ids = [development_ids[index] for index in train_indices]
        holdout_ids = [development_ids[index] for index in holdout_indices]
        for subject_id in holdout_ids:
            if subject_id in assignment:
                raise RuntimeError(f"Duplicate outer holdout assignment: {subject_id}")
            assignment[subject_id] = outer_fold
        inner_labels = np.asarray([subject_metadata[subject_id]["label"] for subject_id in train_ids])
        resolved_inner_seed = inner_seed + outer_fold
        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=resolved_inner_seed,
        )
        inner_dummy = np.zeros(len(train_ids), dtype=np.int8)
        inner_folds: list[dict[str, Any]] = []
        for inner_fold, (inner_train_indices, validation_indices) in enumerate(
            inner.split(inner_dummy, inner_labels)
        ):
            inner_folds.append(
                {
                    "inner_fold": inner_fold,
                    "train_subject_ids": [train_ids[index] for index in inner_train_indices],
                    "validation_subject_ids": [train_ids[index] for index in validation_indices],
                }
            )
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "train_subject_ids": train_ids,
                "holdout_subject_ids": holdout_ids,
                "inner_seed": resolved_inner_seed,
                "inner_folds": inner_folds,
            }
        )
    payload = {
        "schema_version": 1,
        "unit": "subject",
        "outer_method": "StratifiedKFold(shuffle=True)",
        "outer_folds": OUTER_FOLDS,
        "outer_seed": outer_seed,
        "inner_method": "StratifiedKFold(shuffle=True) within each outer training partition",
        "inner_folds": INNER_FOLDS,
        "inner_seed_policy": f"{inner_seed} + outer_fold",
        "development_subject_ids": development_ids,
        "locked_test_subject_ids": test_ids,
        "outer_assignment_by_subject": {
            subject_id: assignment[subject_id] for subject_id in sorted(assignment, key=natural_key)
        },
        "folds": outer_folds,
    }
    validate_fold_assignments(payload, subject_metadata)
    return payload


def validate_fold_assignments(
    folds: Mapping[str, Any],
    subject_metadata: Mapping[str, dict[str, Any]],
) -> None:
    development = set(folds["development_subject_ids"])
    locked_test = set(folds["locked_test_subject_ids"])
    if development & locked_test:
        raise ValueError("Development and locked-test subject sets overlap")
    observed_holdouts: list[str] = []
    if len(folds["folds"]) != OUTER_FOLDS:
        raise ValueError("Exactly five outer folds are required")
    for outer in folds["folds"]:
        train = set(outer["train_subject_ids"])
        holdout = set(outer["holdout_subject_ids"])
        if train & holdout or train | holdout != development:
            raise ValueError(f"Invalid outer subject isolation in fold {outer['outer_fold']}")
        if (train | holdout) & locked_test:
            raise ValueError("Locked-test subject entered an outer development fold")
        if {subject_metadata[subject_id]["label"] for subject_id in holdout} != {0, 1}:
            raise ValueError(f"Outer fold {outer['outer_fold']} is not stratified across both labels")
        observed_holdouts.extend(outer["holdout_subject_ids"])
        inner_validation: list[str] = []
        if len(outer["inner_folds"]) != INNER_FOLDS:
            raise ValueError("Exactly four inner folds are required")
        for inner in outer["inner_folds"]:
            inner_train = set(inner["train_subject_ids"])
            validation = set(inner["validation_subject_ids"])
            if inner_train & validation or inner_train | validation != train:
                raise ValueError(
                    f"Invalid inner subject isolation in outer={outer['outer_fold']}, "
                    f"inner={inner['inner_fold']}"
                )
            if (inner_train | validation) & (holdout | locked_test):
                raise ValueError("Outer holdout or locked test entered inner model selection")
            inner_validation.extend(inner["validation_subject_ids"])
        if Counter(inner_validation) != Counter({subject_id: 1 for subject_id in train}):
            raise ValueError(f"Inner OOF coverage is not exactly once in outer fold {outer['outer_fold']}")
    if Counter(observed_holdouts) != Counter({subject_id: 1 for subject_id in development}):
        raise ValueError("Development outer OOF coverage is not exactly once")


def _subject_metadata_from_selection(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(subject_id): {
            "subject_id": str(subject_id),
            "original_partition": str(row["original_partition"]),
            "label": int(row["label"]),
            "samples": list(row["samples"]),
        }
        for subject_id, row in selection["subjects"].items()
    }


def freeze_protocol(paths: CachePaths) -> dict[str, Any]:
    freeze_path = paths.output / "protocol_freeze.json"
    if freeze_path.exists():
        verified = verify_frozen_protocol(paths)
        return {
            "status": "already_frozen_and_verified",
            "experiment_spec_sha256": verified["freeze"]["experiment_spec_sha256"],
            "official_test_predictions_created": False,
        }
    paths.output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in paths.output.iterdir() if not path.name.startswith(".")]
    if unexpected:
        raise FileExistsError(
            f"Refusing to freeze into non-empty output directory without a freeze marker: {paths.output}"
        )

    manifest_hash = sha256_file(paths.manifest)
    partitions_hash = sha256_file(paths.partitions)
    selected_rows, _, selection_audit = load_selected_protocol_rows(paths.manifest, paths.partitions)
    selection, selected_by_id = selected_sample_payload(selected_rows)
    subject_metadata = _subject_metadata_from_selection(selection)
    folds = build_fold_assignments(subject_metadata)
    print("N0 validating all eGeMAPS cache entries...", flush=True)
    egemaps = validate_egemaps_cache(
        paths.egemaps,
        selected_by_id,
        manifest_sha256=manifest_hash,
        partitions_sha256=partitions_hash,
    )
    print("N0 validating all WavLM cache entries...", flush=True)
    wavlm = validate_wavlm_cache(
        paths.wavlm,
        selected_by_id,
        manifest_sha256=manifest_hash,
        partitions_sha256=partitions_hash,
    )

    selection_path = paths.output / "selected_k4_samples.json"
    folds_path = paths.output / "fold_assignments.json"
    validation_path = paths.output / "feature_cache_validation.json"
    atomic_json(selection_path, selection)
    atomic_json(folds_path, folds)
    feature_validation = {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "passed",
        "fail_closed_checks": [
            "missing features",
            "duplicate sample IDs",
            "non-finite values",
            "dimensional inconsistency",
            "manifest/partition hash mismatch",
            "audio hash mismatch",
            "feature/model signature mismatch",
            "extraction-code hash mismatch where recorded upstream",
        ],
        "families": {
            "egemaps": egemaps.validation,
            "wavlm": wavlm.validation,
        },
    }
    atomic_json(validation_path, feature_validation)

    analysis_code = Path(__file__).resolve()
    spec = {
        "schema_version": 1,
        "experiment_id": "daic_acoustic_crossfold_n0n1_20260715",
        "status": "frozen",
        "frozen_at_utc": utc_now(),
        "scope": ["N0", "N1_development_OOF"],
        "official_test_policy": {
            "status": "locked",
            "subject_count": 47,
            "predictions_permitted_in_n1": False,
            "unlock_condition": (
                "Freeze N2 model classes/seeds and N3 winner/analysis protocol before the one-time N4 run."
            ),
        },
        "inputs": {
            "manifest": {"path": str(paths.manifest), "sha256": manifest_hash},
            "partitions": {"path": str(paths.partitions), "sha256": partitions_hash},
        },
        "subjects": {
            "development_count": 142,
            "development_ids": folds["development_subject_ids"],
            "locked_test_count": 47,
            "locked_test_ids": folds["locked_test_subject_ids"],
        },
        "audio_sampling": {
            "chunks_per_subject": CHUNKS_PER_SUBJECT,
            "numeric_suffix_order_is_ordinal_only": True,
            "ten_chunk_zero_based_positions": [0, 3, 6, 9],
            "fifteen_chunk_zero_based_positions": [0, 5, 9, 14],
            "selection_artifact": selection_path.name,
            "selection_artifact_sha256": sha256_file(selection_path),
            "sample_ids_by_subject": {
                subject_id: [sample["sample_id"] for sample in row["samples"]]
                for subject_id, row in selection["subjects"].items()
            },
            "known_limitation": (
                "random_segment/segment preprocessing kind remains perfectly associated with label."
            ),
            "selection_audit": selection_audit,
        },
        "folds": {
            "outer_folds": OUTER_FOLDS,
            "outer_seed": OUTER_SEED,
            "inner_folds": INNER_FOLDS,
            "inner_seed_base": INNER_SEED,
            "assignment_artifact": folds_path.name,
            "assignment_artifact_sha256": sha256_file(folds_path),
        },
        "features": {
            "egemaps": {
                "chunk_dimension": 88,
                "subject_pooling": "chunk mean + population standard deviation",
                "subject_dimension": 176,
            },
            "wavlm": {
                "model_id": "microsoft/wavlm-base-plus",
                "model_revision": WAVLM_REVISION,
                "layers": [6, 7, 8],
                "chunk_pooling": "per-layer time mean, then concatenate",
                "chunk_dimension": 2304,
                "subject_pooling": "chunk-vector mean + population standard deviation",
                "subject_dimension": 4608,
            },
            "validation_artifact": validation_path.name,
            "validation_artifact_sha256": sha256_file(validation_path),
        },
        "linear_model": {
            "pipeline": ["StandardScaler", "L2 LogisticRegression"],
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 5000,
            "c_grid": list(C_GRID),
            "selection": "minimum pooled inner-OOF log loss; ties use smaller C",
            "model_seed": MODEL_SEED,
        },
        "controls": {
            "constant_probability": "outer-training positive prevalence",
            "majority_threshold": THRESHOLD,
            "shuffle_repeats": SHUFFLE_REPEATS,
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_unit": "complete K=4 subject bundle",
            "shuffle_partitioning": (
                "independent derangement inside every inner-train, inner-validation, "
                "outer-train, and outer-holdout partition"
            ),
            "shuffle_selection": "repeat the complete nested C-selection protocol per derangement",
        },
        "analysis": {
            "threshold": THRESHOLD,
            "metrics": list(METRIC_NAMES),
            "fold_reporting": "fold values, population mean/std, and pooled OOF separately",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_method": (
                "paired percentile subject bootstrap; each replicate samples one deterministic "
                "shuffle repeat to include derangement uncertainty"
            ),
            "permutation_p_value": "(1 + shuffled results as-good-or-better) / 101",
        },
        "gate": {
            "status_after_n1": "pending_N2_N3_and_locked_test",
            "criteria": [
                "pooled development OOF AUROC > 0.5 and real-minus-shuffled AUROC CI excludes 0",
                "pooled development OOF balanced accuracy exceeds shuffled with CI excluding 0",
                "positive direction in at least four of five outer folds",
                "MIL result is not driven by one selected seed",
                "locked official-test direction is positive against shuffled audio",
                "provenance and leakage checks remain clean",
            ],
        },
        "provenance": {
            "analysis_code_path": str(analysis_code),
            "analysis_code_sha256": sha256_file(analysis_code),
        },
    }
    spec_path = paths.output / "experiment_spec.json"
    atomic_json(spec_path, spec)
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_spec": spec_path.name,
        "experiment_spec_sha256": sha256_file(spec_path),
        "fold_assignments_sha256": sha256_file(folds_path),
        "selected_k4_samples_sha256": sha256_file(selection_path),
        "feature_cache_validation_sha256": sha256_file(validation_path),
        "analysis_code_sha256": sha256_file(analysis_code),
        "official_test_predictions_created": False,
    }
    atomic_json(freeze_path, freeze)
    return {
        "status": "frozen",
        "experiment_spec_sha256": freeze["experiment_spec_sha256"],
        "fold_assignments_sha256": freeze["fold_assignments_sha256"],
        "selected_k4_samples_sha256": freeze["selected_k4_samples_sha256"],
        "feature_cache_validation_sha256": freeze["feature_cache_validation_sha256"],
        "official_test_predictions_created": False,
    }


def verify_frozen_protocol(paths: CachePaths) -> dict[str, Any]:
    freeze_path = paths.output / "protocol_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Protocol is not frozen: {freeze_path}")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "frozen" or freeze.get("official_test_predictions_created") is not False:
        raise ValueError("Protocol freeze marker is invalid or the locked test has been touched")
    artifacts = {
        "experiment_spec.json": freeze["experiment_spec_sha256"],
        "fold_assignments.json": freeze["fold_assignments_sha256"],
        "selected_k4_samples.json": freeze["selected_k4_samples_sha256"],
        "feature_cache_validation.json": freeze["feature_cache_validation_sha256"],
    }
    for filename, expected_hash in artifacts.items():
        path = paths.output / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen protocol artifact changed: {path}")
    analysis_hash = sha256_file(Path(__file__).resolve())
    if analysis_hash != freeze.get("analysis_code_sha256"):
        raise ValueError("Analysis code changed after protocol freeze; create a new experiment ID")
    spec = read_json(paths.output / "experiment_spec.json")
    if sha256_file(paths.manifest) != spec["inputs"]["manifest"]["sha256"]:
        raise ValueError("Manifest changed after protocol freeze")
    if sha256_file(paths.partitions) != spec["inputs"]["partitions"]["sha256"]:
        raise ValueError("Partitions changed after protocol freeze")
    selection = read_json(paths.output / "selected_k4_samples.json")
    folds = read_json(paths.output / "fold_assignments.json")
    validate_fold_assignments(folds, _subject_metadata_from_selection(selection))
    return {"status": "verified", "freeze": freeze, "spec": spec}


def subject_features(
    bundle: FeatureBundle,
    selection: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    expected_dimension = bundle.chunk_dimension * 2
    for subject_id, subject in selection["subjects"].items():
        samples = sorted(subject["samples"], key=lambda row: int(row["selected_position"]))
        if len(samples) != CHUNKS_PER_SUBJECT:
            raise ValueError(f"Subject {subject_id} does not have exactly K=4 samples")
        try:
            matrix = np.stack([bundle.vectors[str(row["sample_id"])] for row in samples])
        except KeyError as exc:
            raise ValueError(f"Missing {bundle.family} feature for {exc.args[0]}") from exc
        if matrix.shape != (CHUNKS_PER_SUBJECT, bundle.chunk_dimension):
            raise ValueError(f"Invalid {bundle.family} chunk matrix for {subject_id}: {matrix.shape}")
        pooled = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0, ddof=0)])
        if pooled.shape != (expected_dimension,) or not np.isfinite(pooled).all():
            raise ValueError(f"Invalid {bundle.family} subject feature for {subject_id}")
        features[str(subject_id)] = pooled
    return features


def evaluate_binary(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.shape != y.shape or len(y) == 0:
        raise ValueError("Targets/probabilities must be non-empty 1D arrays of equal length")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities are non-finite or outside [0, 1]")
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    prediction = (clipped >= THRESHOLD).astype(np.int64)
    tn = int(np.count_nonzero((y == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((y == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((y == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((y == 1) & (prediction == 1)))

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    positive_f1 = divide(2 * precision * recall, precision + recall)
    negative_precision = divide(tn, tn + fn)
    negative_f1 = divide(
        2 * negative_precision * specificity,
        negative_precision + specificity,
    )
    has_both = len(set(y.tolist())) == 2
    return {
        "accuracy": divide(tp + tn, len(y)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "positive_f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "auroc": float(roc_auc_score(y, clipped)) if has_both else None,
        "auprc": float(average_precision_score(y, clipped)) if np.any(y == 1) else None,
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "brier_score": float(np.mean((clipped - y) ** 2)),
        "support_negative": int(np.count_nonzero(y == 0)),
        "support_positive": int(np.count_nonzero(y == 1)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def build_classifier(c_value: float, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _matrix(features: Mapping[str, np.ndarray], target_ids: Sequence[str], source_ids: Sequence[str] | None = None) -> np.ndarray:
    if source_ids is None:
        source_ids = target_ids
    if len(target_ids) != len(source_ids):
        raise ValueError("Target/source feature lists have different lengths")
    matrix = np.stack([features[str(subject_id)] for subject_id in source_ids])
    if not np.isfinite(matrix).all():
        raise ValueError("Model matrix contains non-finite values")
    return matrix


def _labels(metadata: Mapping[str, dict[str, Any]], subject_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(metadata[str(subject_id)]["label"]) for subject_id in subject_ids])


def derived_seed(base: int, *coordinates: int) -> int:
    sequence = np.random.SeedSequence([int(base), *(int(value) for value in coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def derangement(subject_ids: Sequence[str], seed: int) -> dict[str, str]:
    ids = [str(subject_id) for subject_id in subject_ids]
    if len(ids) < 2:
        raise ValueError("At least two subjects are needed for bundle derangement")
    rng = np.random.default_rng(seed)
    original = np.asarray(ids, dtype=object)
    for _ in range(10_000):
        shuffled = original[rng.permutation(len(original))]
        if bool(np.all(shuffled != original)):
            mapping = {target: str(source) for target, source in zip(ids, shuffled)}
            validate_derangement(ids, mapping)
            return mapping
    rotated = np.roll(original, 1)
    mapping = {target: str(source) for target, source in zip(ids, rotated)}
    validate_derangement(ids, mapping)
    return mapping


def validate_derangement(subject_ids: Sequence[str], mapping: Mapping[str, str]) -> None:
    expected = {str(subject_id) for subject_id in subject_ids}
    if set(mapping) != expected or set(mapping.values()) != expected:
        raise ValueError("Derangement is not a bijection over the requested partition")
    if any(str(target) == str(source) for target, source in mapping.items()):
        raise ValueError("Derangement retained an aligned subject bundle")


def select_c_nested(
    features: Mapping[str, np.ndarray],
    metadata: Mapping[str, dict[str, Any]],
    outer: Mapping[str, Any],
    *,
    shuffle_repeat: int | None,
) -> tuple[float, list[dict[str, Any]]]:
    candidate_scores: dict[float, list[tuple[str, int, float, int]]] = {
        c_value: [] for c_value in C_GRID
    }
    outer_fold = int(outer["outer_fold"])
    from sklearn.exceptions import ConvergenceWarning

    for inner in outer["inner_folds"]:
        inner_fold = int(inner["inner_fold"])
        train_ids = [str(value) for value in inner["train_subject_ids"]]
        validation_ids = [str(value) for value in inner["validation_subject_ids"]]
        if shuffle_repeat is None:
            train_sources = train_ids
            validation_sources = validation_ids
        else:
            train_map = derangement(
                train_ids,
                derived_seed(SHUFFLE_SEED, shuffle_repeat, outer_fold, inner_fold, 0),
            )
            validation_map = derangement(
                validation_ids,
                derived_seed(SHUFFLE_SEED, shuffle_repeat, outer_fold, inner_fold, 1),
            )
            train_sources = [train_map[target] for target in train_ids]
            validation_sources = [validation_map[target] for target in validation_ids]
        train_x = _matrix(features, train_ids, train_sources)
        train_y = _labels(metadata, train_ids)
        validation_x = _matrix(features, validation_ids, validation_sources)
        validation_y = _labels(metadata, validation_ids)
        for c_index, c_value in enumerate(C_GRID):
            model = build_classifier(
                c_value,
                derived_seed(
                    MODEL_SEED,
                    outer_fold,
                    inner_fold,
                    c_index,
                    0 if shuffle_repeat is None else shuffle_repeat + 1,
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train_x, train_y)
            probabilities = model.predict_proba(validation_x)[:, 1]
            candidate_scores[c_value].extend(
                (
                    subject_id,
                    int(label),
                    float(probability),
                    inner_fold,
                )
                for subject_id, label, probability in zip(
                    validation_ids, validation_y, probabilities
                )
            )
    records: list[dict[str, Any]] = []
    outer_train = set(str(value) for value in outer["train_subject_ids"])
    for c_value in C_GRID:
        rows = candidate_scores[c_value]
        validate_oof_coverage(
            rows,
            expected_subject_ids=outer_train,
            id_index=0,
            context=f"inner OOF outer={outer_fold}, C={c_value}",
        )
        rows.sort(key=lambda row: natural_key(row[0]))
        metrics = evaluate_binary(
            [row[1] for row in rows],
            [row[2] for row in rows],
        )
        records.append(
            {
                "C": c_value,
                "pooled_inner_oof_log_loss": metrics["log_loss"],
                "pooled_inner_oof_metrics": metrics,
            }
        )
    selected = min(records, key=lambda row: (row["pooled_inner_oof_log_loss"], row["C"]))
    return float(selected["C"]), records


def validate_oof_coverage(
    rows: Sequence[Any],
    *,
    expected_subject_ids: set[str],
    id_index: int | None = None,
    id_key: str = "subject_id",
    context: str,
) -> None:
    if id_index is None:
        ids = [str(row[id_key]) for row in rows]
    else:
        ids = [str(row[id_index]) for row in rows]
    counts = Counter(ids)
    if set(counts) != expected_subject_ids or any(count != 1 for count in counts.values()):
        raise ValueError(
            f"{context} does not contain exactly one prediction per subject: "
            f"expected={len(expected_subject_ids)}, observed={len(ids)}, unique={len(counts)}"
        )


def fit_outer_fold(
    features: Mapping[str, np.ndarray],
    metadata: Mapping[str, dict[str, Any]],
    outer: Mapping[str, Any],
    *,
    shuffle_repeat: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outer_fold = int(outer["outer_fold"])
    selected_c, candidates = select_c_nested(
        features,
        metadata,
        outer,
        shuffle_repeat=shuffle_repeat,
    )
    train_ids = [str(value) for value in outer["train_subject_ids"]]
    holdout_ids = [str(value) for value in outer["holdout_subject_ids"]]
    if shuffle_repeat is None:
        train_sources = train_ids
        holdout_sources = holdout_ids
    else:
        train_mapping = derangement(
            train_ids,
            derived_seed(SHUFFLE_SEED, shuffle_repeat, outer_fold, 90),
        )
        holdout_mapping = derangement(
            holdout_ids,
            derived_seed(SHUFFLE_SEED, shuffle_repeat, outer_fold, 91),
        )
        train_sources = [train_mapping[target] for target in train_ids]
        holdout_sources = [holdout_mapping[target] for target in holdout_ids]
    train_x = _matrix(features, train_ids, train_sources)
    train_y = _labels(metadata, train_ids)
    holdout_x = _matrix(features, holdout_ids, holdout_sources)
    holdout_y = _labels(metadata, holdout_ids)
    model = build_classifier(
        selected_c,
        derived_seed(
            MODEL_SEED,
            outer_fold,
            999,
            0 if shuffle_repeat is None else shuffle_repeat + 1,
        ),
    )
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_x, train_y)
    probabilities = model.predict_proba(holdout_x)[:, 1]
    rows = [
        {
            "subject_id": target,
            "source_subject_id": source,
            "label": int(label),
            "probability": float(probability),
            "prediction": int(probability >= THRESHOLD),
            "outer_fold": outer_fold,
            "selected_C": selected_c,
        }
        for target, source, label, probability in zip(
            holdout_ids, holdout_sources, holdout_y, probabilities
        )
    ]
    return rows, {
        "outer_fold": outer_fold,
        "selected_C": selected_c,
        "inner_selection": candidates,
        "holdout_metrics": evaluate_binary(holdout_y, probabilities),
    }


def bootstrap_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in PRIMARY_COMPARISON_METRICS}
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        metrics = evaluate_binary(y[indices], scores[indices])
        for metric in PRIMARY_COMPARISON_METRICS:
            value = metrics.get(metric)
            if value is not None and math.isfinite(float(value)):
                samples[metric].append(float(value))
    return {
        "unit": "subject",
        "method": "percentile bootstrap with replacement",
        "repeats": repeats,
        "seed": seed,
        "intervals": {
            metric: {
                "lower_2.5pct": float(np.quantile(values, 0.025)),
                "upper_97.5pct": float(np.quantile(values, 0.975)),
                "valid_replicates": len(values),
            }
            for metric, values in samples.items()
            if values
        },
    }


def paired_bootstrap_differences(
    y: np.ndarray,
    real_scores: np.ndarray,
    shuffled_scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if shuffled_scores.ndim != 2 or shuffled_scores.shape[1] != len(y):
        raise ValueError("Shuffled score matrix must be repeats by subjects")
    real_metrics = evaluate_binary(y, real_scores)
    shuffled_metrics = [evaluate_binary(y, scores) for scores in shuffled_scores]
    point_differences: dict[str, float] = {}
    for metric in PRIMARY_COMPARISON_METRICS:
        real_value = real_metrics.get(metric)
        values = [row.get(metric) for row in shuffled_metrics if row.get(metric) is not None]
        if real_value is not None and values:
            point_differences[metric] = float(real_value - np.mean(values))

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in PRIMARY_COMPARISON_METRICS}
    correct_score_samples: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        shuffled_index = int(rng.integers(0, shuffled_scores.shape[0]))
        real = evaluate_binary(y[indices], real_scores[indices])
        shuffled = evaluate_binary(y[indices], shuffled_scores[shuffled_index, indices])
        for metric in PRIMARY_COMPARISON_METRICS:
            real_value = real.get(metric)
            shuffled_value = shuffled.get(metric)
            if real_value is not None and shuffled_value is not None:
                difference = float(real_value) - float(shuffled_value)
                if math.isfinite(difference):
                    samples[metric].append(difference)
        signed = 2 * y[indices] - 1
        correct_score_samples.append(
            float(
                np.mean(
                    signed
                    * (real_scores[indices] - shuffled_scores[shuffled_index, indices])
                )
            )
        )
    signed = 2 * y - 1
    point_correct_score = float(
        np.mean(
            signed * (real_scores - shuffled_scores.mean(axis=0))
        )
    )
    return {
        "unit": "subject",
        "method": (
            "paired percentile subject bootstrap; one of 100 deterministic shuffled-control "
            "runs is sampled per bootstrap replicate"
        ),
        "repeats": repeats,
        "seed": seed,
        "difference_definition": "real metric minus shuffled metric (raw orientation)",
        "point_difference_vs_mean_shuffled_metric": point_differences,
        "intervals": {
            metric: {
                "lower_2.5pct": float(np.quantile(values, 0.025)),
                "upper_97.5pct": float(np.quantile(values, 0.975)),
                "valid_replicates": len(values),
            }
            for metric, values in samples.items()
            if values
        },
        "mean_correct_class_probability_difference": {
            "point": point_correct_score,
            "lower_2.5pct": float(np.quantile(correct_score_samples, 0.025)),
            "upper_97.5pct": float(np.quantile(correct_score_samples, 0.975)),
            "valid_replicates": len(correct_score_samples),
        },
    }


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_folds(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    summary: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in metrics if row.get(metric) is not None]
        if values:
            summary[metric] = _mean_std(values)
    return summary


def _score_arrays(
    rows: Sequence[dict[str, Any]],
    development_ids: Sequence[str],
    metadata: Mapping[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    by_subject = {str(row["subject_id"]): row for row in rows}
    if len(by_subject) != len(rows) or set(by_subject) != set(development_ids):
        raise ValueError("Pooled OOF rows do not cover development subjects exactly once")
    y = _labels(metadata, development_ids)
    scores = np.asarray([float(by_subject[subject_id]["probability"]) for subject_id in development_ids])
    return y, scores


def run_feature_family(
    family: str,
    features: Mapping[str, np.ndarray],
    metadata: Mapping[str, dict[str, Any]],
    folds: Mapping[str, Any],
    paths: CachePaths,
    cache_validation: Mapping[str, Any],
) -> dict[str, Any]:
    final_dir = paths.output / "n1" / family
    complete_path = final_dir / "complete.json"
    if final_dir.exists():
        if not complete_path.is_file():
            raise FileExistsError(f"Incomplete immutable N1 directory exists: {final_dir}")
        completion = read_json(complete_path)
        for relative, expected_hash in completion.get("artifact_sha256", {}).items():
            path = final_dir / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Completed N1 artifact changed: {path}")
        return read_json(final_dir / "metrics.json")

    staging_parent = paths.output / "n1"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{family}.", dir=staging_parent))
    try:
        development_ids = [str(value) for value in folds["development_subject_ids"]]
        real_rows: list[dict[str, Any]] = []
        real_fold_records: list[dict[str, Any]] = []
        print(f"N1 {family}: fitting five real outer folds...", flush=True)
        for outer in folds["folds"]:
            rows, record = fit_outer_fold(
                features,
                metadata,
                outer,
                shuffle_repeat=None,
            )
            for row in rows:
                row.update(feature_family=family, condition="real")
            real_rows.extend(rows)
            real_fold_records.append(record)
        validate_oof_coverage(
            real_rows,
            expected_subject_ids=set(development_ids),
            context=f"{family} real pooled OOF",
        )

        shuffled_rows: list[dict[str, Any]] = []
        shuffled_selection: list[dict[str, Any]] = []
        for repeat in range(SHUFFLE_REPEATS):
            if repeat == 0 or (repeat + 1) % 10 == 0:
                print(f"N1 {family}: shuffled nested CV {repeat + 1}/{SHUFFLE_REPEATS}", flush=True)
            repeat_rows: list[dict[str, Any]] = []
            selected_cs: list[dict[str, Any]] = []
            for outer in folds["folds"]:
                rows, record = fit_outer_fold(
                    features,
                    metadata,
                    outer,
                    shuffle_repeat=repeat,
                )
                for row in rows:
                    row.update(
                        feature_family=family,
                        condition="shuffled_bundle",
                        shuffle_repeat=repeat,
                    )
                repeat_rows.extend(rows)
                selected_cs.append(
                    {
                        "outer_fold": int(outer["outer_fold"]),
                        "selected_C": record["selected_C"],
                    }
                )
            validate_oof_coverage(
                repeat_rows,
                expected_subject_ids=set(development_ids),
                context=f"{family} shuffled OOF repeat={repeat}",
            )
            shuffled_rows.extend(repeat_rows)
            shuffled_selection.append(
                {
                    "shuffle_repeat": repeat,
                    "outer_selected_C": selected_cs,
                }
            )

        real_rows.sort(key=lambda row: natural_key(row["subject_id"]))
        shuffled_rows.sort(key=lambda row: (int(row["shuffle_repeat"]), natural_key(row["subject_id"])))
        y, real_scores = _score_arrays(real_rows, development_ids, metadata)
        shuffled_matrix = np.empty((SHUFFLE_REPEATS, len(development_ids)), dtype=np.float64)
        shuffled_by_repeat: list[dict[str, Any]] = []
        for repeat in range(SHUFFLE_REPEATS):
            repeat_rows = [row for row in shuffled_rows if int(row["shuffle_repeat"]) == repeat]
            repeat_y, repeat_scores = _score_arrays(repeat_rows, development_ids, metadata)
            if not np.array_equal(y, repeat_y):
                raise ValueError("Shuffled OOF labels changed")
            shuffled_matrix[repeat] = repeat_scores
            shuffled_by_repeat.append(
                {"shuffle_repeat": repeat, "pooled_oof_metrics": evaluate_binary(y, repeat_scores)}
            )

        pooled_real_metrics = evaluate_binary(y, real_scores)
        real_fold_metrics: list[dict[str, Any]] = []
        constant_rows: list[dict[str, Any]] = []
        for outer in folds["folds"]:
            fold = int(outer["outer_fold"])
            holdout_ids = [str(value) for value in outer["holdout_subject_ids"]]
            fold_rows = [row for row in real_rows if int(row["outer_fold"]) == fold]
            fold_y, fold_scores = _score_arrays(fold_rows, holdout_ids, metadata)
            real_fold_metrics.append(
                {"outer_fold": fold, "metrics": evaluate_binary(fold_y, fold_scores)}
            )
            prevalence = float(_labels(metadata, outer["train_subject_ids"]).mean())
            for subject_id in holdout_ids:
                constant_rows.append(
                    {
                        "subject_id": subject_id,
                        "outer_fold": fold,
                        "label": int(metadata[subject_id]["label"]),
                        "probability": prevalence,
                        "prediction": int(prevalence >= THRESHOLD),
                    }
                )
        _, constant_scores = _score_arrays(constant_rows, development_ids, metadata)
        constant_fold_metrics = []
        for fold in range(OUTER_FOLDS):
            rows = [row for row in constant_rows if int(row["outer_fold"]) == fold]
            ids = [str(row["subject_id"]) for row in rows]
            fold_y, fold_scores = _score_arrays(rows, ids, metadata)
            constant_fold_metrics.append(
                {"outer_fold": fold, "metrics": evaluate_binary(fold_y, fold_scores)}
            )

        shuffle_summary: dict[str, Any] = {}
        for metric in METRIC_NAMES:
            values = [
                float(row["pooled_oof_metrics"][metric])
                for row in shuffled_by_repeat
                if row["pooled_oof_metrics"].get(metric) is not None
            ]
            if not values:
                continue
            observed = pooled_real_metrics.get(metric)
            summary = {
                **_mean_std(values),
                "lower_2.5pct": float(np.quantile(values, 0.025)),
                "upper_97.5pct": float(np.quantile(values, 0.975)),
            }
            if observed is not None:
                if metric in HIGHER_IS_BETTER:
                    as_good = sum(value >= float(observed) for value in values)
                    tail = "shuffled_greater_or_equal"
                else:
                    as_good = sum(value <= float(observed) for value in values)
                    tail = "shuffled_less_or_equal"
                summary["empirical_p_value"] = float((1 + as_good) / (1 + len(values)))
                summary["tail"] = tail
            shuffle_summary[metric] = summary

        fold_directions: dict[str, Any] = {}
        for metric in ("auroc", "balanced_accuracy"):
            records = []
            for fold in range(OUTER_FOLDS):
                real_value = float(real_fold_metrics[fold]["metrics"][metric])
                values = []
                for repeat in range(SHUFFLE_REPEATS):
                    repeat_rows = [
                        row
                        for row in shuffled_rows
                        if int(row["shuffle_repeat"]) == repeat and int(row["outer_fold"]) == fold
                    ]
                    ids = [str(row["subject_id"]) for row in repeat_rows]
                    fold_y, fold_scores = _score_arrays(repeat_rows, ids, metadata)
                    value = evaluate_binary(fold_y, fold_scores).get(metric)
                    if value is not None:
                        values.append(float(value))
                shuffled_mean = float(np.mean(values))
                records.append(
                    {
                        "outer_fold": fold,
                        "real": real_value,
                        "mean_shuffled": shuffled_mean,
                        "real_minus_mean_shuffled": real_value - shuffled_mean,
                        "positive": real_value > shuffled_mean,
                    }
                )
            fold_directions[metric] = {
                "positive_fold_count": sum(row["positive"] for row in records),
                "records": records,
            }

        paired = paired_bootstrap_differences(
            y,
            real_scores,
            shuffled_matrix,
            repeats=BOOTSTRAP_REPEATS,
            seed=derived_seed(BOOTSTRAP_SEED, 0 if family == "egemaps" else 1, 1),
        )
        real_bootstrap = bootstrap_metrics(
            y,
            real_scores,
            repeats=BOOTSTRAP_REPEATS,
            seed=derived_seed(BOOTSTRAP_SEED, 0 if family == "egemaps" else 1, 0),
        )
        auc_interval = paired["intervals"].get("auroc", {})
        balanced_interval = paired["intervals"].get("balanced_accuracy", {})
        development_gate = {
            "not_a_final_gate_decision": True,
            "pooled_auroc_above_0.5": bool(pooled_real_metrics["auroc"] > 0.5),
            "auroc_real_minus_shuffled_ci_excludes_zero_positive": bool(
                auc_interval and auc_interval["lower_2.5pct"] > 0
            ),
            "balanced_accuracy_real_minus_shuffled_ci_excludes_zero_positive": bool(
                balanced_interval and balanced_interval["lower_2.5pct"] > 0
            ),
            "auroc_positive_in_at_least_four_folds": bool(
                fold_directions["auroc"]["positive_fold_count"] >= 4
            ),
            "balanced_accuracy_positive_in_at_least_four_folds": bool(
                fold_directions["balanced_accuracy"]["positive_fold_count"] >= 4
            ),
            "remaining_required_phases": ["N2 MIL across five seeds", "N3 winner freeze", "N4 locked test"],
        }
        metrics = {
            "schema_version": 1,
            "feature_family": family,
            "unit": "subject",
            "development_subjects": len(development_ids),
            "threshold": THRESHOLD,
            "official_test_status": "locked; no test prediction was computed",
            "real": {
                "pooled_oof_metrics": pooled_real_metrics,
                "fold_metrics": real_fold_metrics,
                "fold_mean_std": _summarize_folds(real_fold_metrics),
                "subject_bootstrap_95pct": real_bootstrap,
            },
            "constant_prevalence_majority_control": {
                "description": (
                    "Each outer holdout receives its outer-training positive prevalence; "
                    "hard label uses probability >= 0.5."
                ),
                "pooled_oof_metrics": evaluate_binary(y, constant_scores),
                "fold_metrics": constant_fold_metrics,
                "fold_mean_std": _summarize_folds(constant_fold_metrics),
            },
            "shuffled_bundle_control": {
                "repeats": SHUFFLE_REPEATS,
                "summary": shuffle_summary,
                "per_repeat": shuffled_by_repeat,
                "fold_directions": fold_directions,
            },
            "real_minus_shuffled_paired_bootstrap_95pct": paired,
            "development_gate_components": development_gate,
        }

        for row in real_rows:
            row["sample_ids"] = [
                sample["sample_id"] for sample in metadata[row["subject_id"]]["samples"]
            ]
        for row in shuffled_rows:
            row["source_sample_ids"] = [
                sample["sample_id"] for sample in metadata[row["source_subject_id"]]["samples"]
            ]
        atomic_jsonl(staging / "oof_predictions.jsonl", real_rows)
        fold_dir = staging / "fold_predictions"
        for fold in range(OUTER_FOLDS):
            atomic_jsonl(
                fold_dir / f"fold_{fold}.jsonl",
                [row for row in real_rows if int(row["outer_fold"]) == fold],
            )
        atomic_jsonl(staging / "shuffled_oof_predictions.jsonl", shuffled_rows)
        atomic_json(
            staging / "selected_hyperparameters.json",
            {
                "real_outer_folds": real_fold_records,
                "shuffled_outer_folds": shuffled_selection,
            },
        )
        atomic_json(staging / "metrics.json", metrics)
        # The official test cache is validated in N0, but N1 artifacts contain
        # development subjects only so the locked partition is not exposed as
        # an analysis table or model input.
        subject_ids = list(development_ids)
        np.savez_compressed(
            staging / "subject_features.npz",
            subject_ids=np.asarray(subject_ids),
            labels=_labels(metadata, subject_ids),
            original_partitions=np.asarray(
                [metadata[subject_id]["original_partition"] for subject_id in subject_ids]
            ),
            features=np.stack([features[subject_id] for subject_id in subject_ids]),
        )
        provenance = {
            "schema_version": 1,
            "completed_at_utc": utc_now(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "feature_family": family,
            "protocol_spec_sha256": sha256_file(paths.output / "experiment_spec.json"),
            "fold_assignments_sha256": sha256_file(paths.output / "fold_assignments.json"),
            "selected_k4_samples_sha256": sha256_file(paths.output / "selected_k4_samples.json"),
            "feature_cache_validation": cache_validation,
            "analysis_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "repository": git_provenance(),
            "environment": dependency_versions(),
            "platform": platform.platform(),
            "leakage_guards": {
                "subject_disjoint_outer_folds": True,
                "subject_disjoint_inner_folds": True,
                "one_real_oof_prediction_per_development_subject": True,
                "one_shuffled_oof_prediction_per_subject_per_repeat": True,
                "official_test_used": False,
                "fixed_chunks_per_subject": CHUNKS_PER_SUBJECT,
                "shuffle_deranged_within_each_model_partition": True,
            },
        }
        atomic_json(staging / "provenance.json", provenance)
        artifact_paths = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(staging)),
        )
        completion = {
            "schema_version": 1,
            "status": "complete_immutable",
            "feature_family": family,
            "official_test_predictions_created": False,
            "artifact_sha256": {
                str(path.relative_to(staging)): sha256_file(path) for path in artifact_paths
            },
        }
        atomic_json(staging / "complete.json", completion)
        os.replace(staging, final_dir)
        return metrics
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def dependency_versions() -> dict[str, str | None]:
    packages = ("numpy", "scipy", "scikit-learn", "joblib")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_provenance() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        status = command("status", "--porcelain").splitlines()
        return {
            "commit": command("rev-parse", "HEAD"),
            "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "status_porcelain": []}


def validate_caches_from_frozen(paths: CachePaths) -> tuple[dict[str, Any], FeatureBundle, FeatureBundle]:
    verified = verify_frozen_protocol(paths)
    spec = verified["spec"]
    selection = read_json(paths.output / "selected_k4_samples.json")
    selected_by_id: dict[str, dict[str, Any]] = {}
    for subject_id, subject in selection["subjects"].items():
        for sample in subject["samples"]:
            selected_by_id[str(sample["sample_id"])] = {
                **sample,
                "subject_id": str(subject_id),
                "partition": str(subject["original_partition"]),
                "label": int(subject["label"]),
            }
    print("Revalidating frozen eGeMAPS cache...", flush=True)
    egemaps = validate_egemaps_cache(
        paths.egemaps,
        selected_by_id,
        manifest_sha256=spec["inputs"]["manifest"]["sha256"],
        partitions_sha256=spec["inputs"]["partitions"]["sha256"],
    )
    print("Revalidating frozen WavLM cache...", flush=True)
    wavlm = validate_wavlm_cache(
        paths.wavlm,
        selected_by_id,
        manifest_sha256=spec["inputs"]["manifest"]["sha256"],
        partitions_sha256=spec["inputs"]["partitions"]["sha256"],
    )
    frozen_validation = read_json(paths.output / "feature_cache_validation.json")
    for bundle in (egemaps, wavlm):
        if bundle.validation != frozen_validation["families"][bundle.family]:
            raise ValueError(f"{bundle.family} cache validation changed after protocol freeze")
    return verified, egemaps, wavlm


def run_n1(paths: CachePaths, families: Sequence[str]) -> dict[str, Any]:
    verified, egemaps, wavlm = validate_caches_from_frozen(paths)
    selection = read_json(paths.output / "selected_k4_samples.json")
    folds = read_json(paths.output / "fold_assignments.json")
    metadata = _subject_metadata_from_selection(selection)
    bundles = {"egemaps": egemaps, "wavlm": wavlm}
    results: dict[str, Any] = {}
    for family in families:
        if family not in bundles:
            raise ValueError(f"Unknown feature family: {family}")
        bundle = bundles[family]
        pooled = subject_features(bundle, selection)
        results[family] = run_feature_family(
            family,
            pooled,
            metadata,
            folds,
            paths,
            bundle.validation,
        )
    render_report(paths, results)
    return {
        "status": "N0_N1_complete",
        "official_test_status": "locked",
        "families": list(results),
        "experiment_spec_sha256": verified["freeze"]["experiment_spec_sha256"],
    }


def render_report(paths: CachePaths, current_results: Mapping[str, Any] | None = None) -> None:
    results: dict[str, Any] = dict(current_results or {})
    for family in ("egemaps", "wavlm"):
        path = paths.output / "n1" / family / "metrics.json"
        if family not in results and path.is_file():
            results[family] = read_json(path)
    if not results:
        return
    lines = [
        "# DAIC N0/N1 acoustic cross-fold results",
        "",
        f"Generated: {utc_now()}",
        "",
        "The N0 protocol is frozen and all selected eGeMAPS/WavLM cache entries passed "
        "manifest, sample-ID, audio-hash, signature, dimension, uniqueness, and finiteness checks.",
        "",
        "The official 47-subject test set remains locked. These development results cannot pass "
        "or fail the complete acoustic gate until N2/N3 are frozen and the one-time N4 check runs.",
        "",
        "| Feature | OOF AUROC | OOF AUPRC | OOF balanced acc. | OOF log loss | AUROC delta vs shuffle (95% CI) | Balanced-acc. delta (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("egemaps", "wavlm"):
        if family not in results:
            continue
        metrics = results[family]
        real = metrics["real"]["pooled_oof_metrics"]
        paired = metrics["real_minus_shuffled_paired_bootstrap_95pct"]
        point = paired["point_difference_vs_mean_shuffled_metric"]
        intervals = paired["intervals"]
        auc_ci = intervals["auroc"]
        bal_ci = intervals["balanced_accuracy"]
        lines.append(
            f"| {family} | {real['auroc']:.4f} | {real['auprc']:.4f} | "
            f"{real['balanced_accuracy']:.4f} | {real['log_loss']:.4f} | "
            f"{point['auroc']:.4f} [{auc_ci['lower_2.5pct']:.4f}, {auc_ci['upper_97.5pct']:.4f}] | "
            f"{point['balanced_accuracy']:.4f} "
            f"[{bal_ci['lower_2.5pct']:.4f}, {bal_ci['upper_97.5pct']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Interpretation is limited to the fixed-K preprocessed DAIC audio protocol. The "
            "perfect label association of `random_segment` versus `segment` preprocessing kind remains.",
            "",
        ]
    )
    report_path = paths.output / "N0_N1_RESULTS.md"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=report_path.parent, prefix=f".{report_path.name}.", delete=False
    ) as handle:
        handle.write("\n".join(lines))
        temporary = Path(handle.name)
    os.replace(temporary, report_path)


def verify_outputs(paths: CachePaths) -> dict[str, Any]:
    verified, _, _ = validate_caches_from_frozen(paths)
    completed: list[str] = []
    for family in ("egemaps", "wavlm"):
        final_dir = paths.output / "n1" / family
        complete = final_dir / "complete.json"
        if not complete.exists():
            continue
        payload = read_json(complete)
        if payload.get("official_test_predictions_created") is not False:
            raise ValueError(f"Unexpected locked-test state in {complete}")
        for relative, expected_hash in payload["artifact_sha256"].items():
            path = final_dir / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"N1 immutable artifact changed: {path}")
        predictions = read_jsonl(final_dir / "oof_predictions.jsonl")
        validate_oof_coverage(
            predictions,
            expected_subject_ids=set(verified["spec"]["subjects"]["development_ids"]),
            context=f"verified {family} OOF",
        )
        shuffled = read_jsonl(final_dir / "shuffled_oof_predictions.jsonl")
        for repeat in range(SHUFFLE_REPEATS):
            validate_oof_coverage(
                [row for row in shuffled if int(row["shuffle_repeat"]) == repeat],
                expected_subject_ids=set(verified["spec"]["subjects"]["development_ids"]),
                context=f"verified {family} shuffle repeat={repeat}",
            )
        completed.append(family)
    return {
        "status": "verified",
        "protocol": "frozen",
        "completed_n1_families": completed,
        "official_test_predictions_created": False,
    }


def cache_paths_from_args(args: argparse.Namespace) -> CachePaths:
    return CachePaths(
        manifest=Path(args.manifest).expanduser().resolve(),
        partitions=Path(args.partitions).expanduser().resolve(),
        egemaps=Path(args.egemaps_cache).expanduser().resolve(),
        wavlm=Path(args.wavlm_cache).expanduser().resolve(),
        output=Path(args.output_dir).expanduser().resolve(),
    )


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--partitions", default=str(DEFAULT_PARTITIONS))
    parser.add_argument("--egemaps-cache", default=str(DEFAULT_EGEMAPS_CACHE))
    parser.add_argument("--wavlm-cache", default=str(DEFAULT_WAVLM_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze and validate the N0 protocol")
    add_path_arguments(freeze)
    run = subparsers.add_parser("run", help="Run development-only N1 nested CV")
    add_path_arguments(run)
    run.add_argument(
        "--families",
        nargs="+",
        choices=("egemaps", "wavlm"),
        default=["egemaps", "wavlm"],
    )
    all_parser = subparsers.add_parser("all", help="Freeze N0 and run development-only N1")
    add_path_arguments(all_parser)
    all_parser.add_argument(
        "--families",
        nargs="+",
        choices=("egemaps", "wavlm"),
        default=["egemaps", "wavlm"],
    )
    verify = subparsers.add_parser("verify", help="Revalidate frozen protocol, caches, and N1 outputs")
    add_path_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = cache_paths_from_args(args)
    if args.command == "freeze":
        result = freeze_protocol(paths)
    elif args.command == "run":
        result = run_n1(paths, args.families)
    elif args.command == "all":
        freeze_protocol(paths)
        result = run_n1(paths, args.families)
    elif args.command == "verify":
        result = verify_outputs(paths)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
