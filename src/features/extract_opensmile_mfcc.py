"""Extract openSMILE MFCC functionals into the hidden-classifier cache layout.

One vector is computed for every harmonized example: the same window of
audio the LLM receives (``build_examples`` + ``load_prompt_audio``), at the
same 16 kHz rate. Each fold gets ``outer_train`` / ``final_eval`` caches that
``baselines/qwen_hidden_classifier.py`` and ``baselines/qwen_hidden_xgb_optuna.py``
read unchanged. Any extraction failure stops the run; nothing is filled in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.aggregate import HIERARCHICAL_MEAN_CLASSIFIER_POLICY
from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
from src.utils import resolve_input_modality
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    resolve_metadata_paths,
    resolve_project_path,
    save_json,
    sha256_file,
    write_jsonl,
)

SCHEMA_VERSION = "opensmile_mfcc_cache.v1"
MODEL_BACKEND = "opensmile_mfcc"
SAMPLING_RATE = 16000
PARTITIONS = ("outer_train", "final_eval")
INPUT_PLACEHOLDER = "__INPUT_FILE_PLACEHOLDER__"
OUTPUT_PLACEHOLDER = "__OUTPUT_CSV_PLACEHOLDER__"
NON_FEATURE_COLUMNS = {"name", "frameIndex", "frameTime"}
ROW_METADATA_KEYS = (
    "response_id",
    "question_condition",
    "turn_key",
    "question_id",
    "window_id",
    "prompt_id",
    "segment_index",
    "num_segments",
    "chunk_id",
    "protocol_id",
    "chunk_index",
    "num_chunks",
)
AUDIO_PATH_KEYS = ("audio_path", "audio_paths")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    maps = []
    for value in values:
        old, sep, new = value.partition("=")
        if not sep or not old or not new:
            raise ValueError(f"--path-map must be OLD=NEW, got {value!r}.")
        maps.append((old, new))
    return maps


def remap_path(path: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def remap_manifest_rows(rows: list[dict[str, Any]], maps: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not maps:
        return rows
    output = []
    for row in rows:
        row = dict(row)
        for key in AUDIO_PATH_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value:
                row[key] = remap_path(value, maps)
            elif isinstance(value, list):
                row[key] = [remap_path(str(item), maps) for item in value]
        output.append(row)
    return output


def load_locked_folds(path: Path, expected_sha256: str, subject_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """Read a pinned held-out assignment ({"0": {"final_eval_subject_ids": [...]}, ...}).

    Same contract as the Androids locked-fold builder: every subject is held
    out exactly once and outer_train is every other subject.
    """
    actual = sha256_file(path)
    if actual != expected_sha256.strip().lower():
        raise ValueError(f"Locked folds {path} have sha256 {actual}, expected {expected_sha256}.")
    payload = read_json(path)
    observed: set[str] = set()
    folds: dict[str, dict[str, list[str]]] = {}
    for key in sorted(payload, key=int):
        heldout = sorted(str(item) for item in payload[key]["final_eval_subject_ids"])
        if set(heldout) - subject_ids:
            raise ValueError(f"Locked fold {key} names subjects outside the manifest.")
        if observed & set(heldout):
            raise ValueError(f"Locked fold {key} overlaps an earlier held-out fold.")
        observed.update(heldout)
        folds[key] = {
            "outer_train_subject_ids": sorted(subject_ids - set(heldout)),
            "final_eval_subject_ids": heldout,
        }
    if observed != subject_ids:
        raise ValueError("Locked folds do not hold out every manifest subject exactly once.")
    return folds


def resolve_partitions(
    config: dict[str, Any],
    metadata: dict[str, Any],
    fold: int,
    locked_folds: dict[str, dict[str, list[str]]] | None,
) -> dict[str, list[str]]:
    """Return the subjects the LLM trains on and is scored on for this fold.

    CV datasets use the outer fold (train + inner selection subjects fit the
    head; the held-out fold is scored). DAIC uses the fixed official split
    with the configured train partition, like the DAIC hidden heads.
    """
    if locked_folds is not None:
        payload = locked_folds[str(fold)]
    else:
        from src.train import _resolve_outer_partitions

        payload = _resolve_outer_partitions(config, metadata, fold)
    partitions = {
        "outer_train": sorted(str(item) for item in payload["outer_train_subject_ids"]),
        "final_eval": sorted(str(item) for item in payload["final_eval_subject_ids"]),
    }
    if not partitions["outer_train"] or not partitions["final_eval"]:
        raise ValueError(f"Fold {fold} has an empty partition.")
    if set(partitions["outer_train"]) & set(partitions["final_eval"]):
        raise ValueError(f"Fold {fold} train and held-out subjects overlap.")
    return partitions


def audio_identity(example: dict[str, Any]) -> str:
    """Hash of everything that decides which samples the window contains."""
    keys = (
        "audio_path",
        "audio_paths",
        "audio_spans",
        "audio_span_groups",
        "audio_start_times",
        "audio_end_times",
        "audio_clip_seconds",
        "participant_sample_count",
    )
    spec = {key: example.get(key) for key in keys if example.get(key) not in (None, [], "")}
    return _sha256_text(json.dumps(spec, sort_keys=True, default=str))


def load_window_audio(example: dict[str, Any]) -> np.ndarray:
    from src.features.qwen_hidden_collator import load_prompt_audio

    loaded = load_prompt_audio(example, SAMPLING_RATE, False)
    arrays = loaded.get("audio_arrays") or []
    if len(arrays) != 1:
        raise ValueError(
            f"Sample {example['sample_id']} has {len(arrays)} audio arrays; harmonized input needs exactly one."
        )
    audio = np.asarray(arrays[0], dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Sample {example['sample_id']} has empty or non-mono audio.")
    return audio


def run_smile(
    audio: np.ndarray,
    *,
    smile_bin: str,
    config_text: str,
    sample_id: str,
) -> tuple[list[str], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="mfcc_") as tmp:
        wav_path = Path(tmp) / "window.wav"
        csv_path = Path(tmp) / "features.csv"
        conf_path = Path(tmp) / "extract.conf"
        sf.write(wav_path, np.clip(audio, -1.0, 1.0), SAMPLING_RATE, subtype="PCM_16")
        conf_path.write_text(
            config_text.replace(INPUT_PLACEHOLDER, str(wav_path)).replace(OUTPUT_PLACEHOLDER, str(csv_path)),
            encoding="utf-8",
        )
        result = subprocess.run(
            [smile_bin, "-C", str(conf_path), "-loglevel", "1"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SMILExtract failed for {sample_id}: {result.stderr.strip()[-500:]}")
        if not csv_path.is_file():
            raise RuntimeError(f"SMILExtract wrote no output for {sample_id}.")
        with csv_path.open(encoding="utf-8") as handle:
            table = list(csv.reader(handle, delimiter=";"))
    if len(table) != 2:
        raise RuntimeError(f"Expected one header and one functional row for {sample_id}, got {len(table)} rows.")
    header, values = table
    keep = [index for index, name in enumerate(header) if name not in NON_FEATURE_COLUMNS]
    names = [header[index] for index in keep]
    vector = np.asarray([float(values[index]) for index in keep], dtype=np.float64)
    if not np.isfinite(vector).all():
        bad = [names[index] for index in np.flatnonzero(~np.isfinite(vector))[:5]]
        raise RuntimeError(f"Non-finite MFCC functionals for {sample_id}: {bad}")
    return names, vector


def smile_version(smile_bin: str) -> str:
    result = subprocess.run([smile_bin, "-h"], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if "openSMILE version" in line:
            return line.strip()
    return "unknown"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=resolve_project_path(".")
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=resolve_project_path("."),
    )
    return bool(result.stdout.strip())


def row_metadata(example: dict[str, Any], partition: str, fold: int) -> dict[str, Any]:
    row = {
        "dataset": str(example["dataset"]),
        "sample_id": str(example["sample_id"]),
        "subject_id": str(example["subject_id"]),
        "label": int(example["label"]),
        "partition": partition,
        "fold": int(fold),
    }
    for key in ROW_METADATA_KEYS:
        if key in example:
            row[key] = example[key]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Harmonized audio_only config whose examples define the windows.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--manifest-metadata", required=True, type=Path, help="<dataset>_manifest_metadata.json of the LLM run.")
    parser.add_argument("--inputs-root", type=Path, help="Checkout that relative metadata paths are relative to.")
    parser.add_argument("--manifest-path", type=Path, help="Use this manifest instead of metadata.manifest_path.")
    parser.add_argument("--folds-path", type=Path, help="Use this folds file instead of metadata.folds_path.")
    parser.add_argument("--locked-folds", type=Path, help="Pinned held-out folds JSON (e.g. Androids merged folds).")
    parser.add_argument("--locked-folds-sha256")
    parser.add_argument("--path-map", action="append", default=[], help="OLD=NEW audio path prefix rewrite.")
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="Dataset output root; fold_<n>/cache is written below it.")
    parser.add_argument("--smile-bin", default="SMILExtract")
    parser.add_argument("--smile-config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.locked_folds and not args.locked_folds_sha256:
        raise SystemExit("--locked-folds requires --locked-folds-sha256.")
    config_path = resolve_project_path(args.config)
    config = load_yaml_with_overrides(config_path, args.overrides)
    if resolve_input_modality(config) != "audio_only":
        raise SystemExit("Use the audio_only config so examples carry audio and no transcript.")
    metadata = read_json(args.manifest_metadata)
    if args.inputs_root:
        root = args.inputs_root.resolve()
        metadata = {
            key: str(root / value)
            if key.endswith("_path") and isinstance(value, str) and value and not Path(value).is_absolute()
            else value
            for key, value in metadata.items()
        }
    metadata = resolve_metadata_paths(metadata)
    if args.manifest_path:
        metadata["manifest_path"] = str(args.manifest_path.resolve())
    if args.folds_path:
        metadata["folds_path"] = str(args.folds_path.resolve())
    manifest_path = Path(metadata["manifest_path"])
    path_maps = parse_path_maps(args.path_map)
    manifest_rows = remap_manifest_rows(load_manifest_rows(manifest_path), path_maps)
    subject_ids = {str(row["subject_id"]) for row in manifest_rows}
    locked = (
        load_locked_folds(args.locked_folds, args.locked_folds_sha256, subject_ids)
        if args.locked_folds
        else None
    )
    smile_config_text = args.smile_config.read_text(encoding="utf-8")
    for placeholder in (INPUT_PLACEHOLDER, OUTPUT_PLACEHOLDER):
        if placeholder not in smile_config_text:
            raise SystemExit(f"openSMILE config has no {placeholder}.")

    plans: dict[int, dict[str, list[dict[str, Any]]]] = {}
    split_identity: dict[int, dict[str, list[str]]] = {}
    for fold in args.folds:
        partitions = resolve_partitions(config, metadata, fold, locked)
        split_identity[fold] = partitions
        plans[fold] = {}
        for partition, ids in partitions.items():
            examples = build_examples(
                filter_rows_by_subjects(manifest_rows, ids), config, partition_name=partition, truncation_log_path=None
            )
            if {str(item["subject_id"]) for item in examples} != set(ids):
                raise ValueError(f"Fold {fold} {partition} examples do not cover the split subjects.")
            plans[fold][partition] = examples

    unique: dict[str, dict[str, Any]] = {}
    for fold_plan in plans.values():
        for examples in fold_plan.values():
            for example in examples:
                unique.setdefault(audio_identity(example), example)
    print(f"{len(unique)} distinct audio windows across folds {args.folds}", flush=True)

    def extract(item: tuple[str, dict[str, Any]]) -> tuple[str, list[str], np.ndarray, float]:
        key, example = item
        audio = load_window_audio(example)
        names, vector = run_smile(
            audio, smile_bin=args.smile_bin, config_text=smile_config_text, sample_id=str(example["sample_id"])
        )
        return key, names, vector, float(audio.size / SAMPLING_RATE)

    store: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    feature_names: list[str] | None = None
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, (key, names, vector, seconds) in enumerate(pool.map(extract, sorted(unique.items())), start=1):
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("openSMILE feature names changed between windows.")
            store[key] = vector
            durations[key] = seconds
            if index % 200 == 0 or index == len(unique):
                print(f"extracted {index}/{len(unique)}", flush=True)
    assert feature_names is not None

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "model_backend": MODEL_BACKEND,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_overrides": list(args.overrides),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_metadata_path": str(args.manifest_metadata.resolve()),
        "folds_path": None if locked else metadata.get("folds_path"),
        "folds_sha256": None if locked or not metadata.get("folds_path") else sha256_file(metadata["folds_path"]),
        "subject_partition_path": metadata.get("subject_partition_path"),
        "locked_folds_path": str(args.locked_folds.resolve()) if locked else None,
        "locked_folds_sha256": sha256_file(args.locked_folds) if locked else None,
        "path_maps": [{"old": old, "new": new} for old, new in path_maps],
        "opensmile": {
            "binary": args.smile_bin,
            "version": smile_version(args.smile_bin),
            "config_path": str(args.smile_config.resolve()),
            "config_sha256": sha256_file(args.smile_config),
            "sampling_rate": SAMPLING_RATE,
            "wav_subtype": "PCM_16",
            "feature_dimension": len(feature_names),
            "feature_names_sha256": _sha256_text("\n".join(feature_names)),
        },
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    for fold, fold_plan in plans.items():
        cache_dir = args.output_dir / f"fold_{fold}" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for partition in PARTITIONS:
            examples = fold_plan[partition]
            rows = [row_metadata(example, partition, fold) for example in examples]
            if len({row["sample_id"] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate sample ids in fold {fold} {partition}.")
            for row, example in zip(rows, examples):
                row["audio_seconds"] = durations[audio_identity(example)]
            matrix = np.stack([store[audio_identity(example)] for example in examples]).astype(np.float32)
            np.savez_compressed(cache_dir / f"{partition}.npz", vectors=matrix)
            write_jsonl(rows, cache_dir / f"{partition}_rows.jsonl")
        partitions = split_identity[fold]
        save_json(
            {
                **provenance,
                "dataset": str(config["dataset"]).lower(),
                "dataset_variant": str(config.get("dataset_variant") or ""),
                "fold": int(fold),
                "condition": "audio_only",
                "input_modality": "audio_only",
                "checkpoint_dir": None,
                "protocol_id": config.get("protocol_id"),
                "classifier_aggregation_policy": HIERARCHICAL_MEAN_CLASSIFIER_POLICY,
                "evaluation_provenance": {
                    "evaluation_protocol": (
                        "daic_official_train_fit_locked_test_evaluation"
                        if str(config.get("split", {}).get("mode")) == "fixed"
                        else "saved_final_evaluation"
                    ),
                    "split_mode": str(config.get("split", {}).get("mode")),
                    "cv_protocol": config.get("split", {}).get("cv_protocol"),
                },
                "partition_subject_counts": {name: len(ids) for name, ids in partitions.items()},
                "partition_subject_ids_sha256": {
                    name: _sha256_text("\n".join(ids)) for name, ids in partitions.items()
                },
                "partition_row_counts": {name: len(fold_plan[name]) for name in PARTITIONS},
            },
            cache_dir / "extraction_metadata.json",
        )
        print(f"fold {fold}: wrote {cache_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
