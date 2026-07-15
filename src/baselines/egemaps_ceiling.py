from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


OPENSMILE_VERSION = "3.0.1"
OPENSMILE_BUNDLE_NAME = f"opensmile-{OPENSMILE_VERSION}-linux-x64"
OPENSMILE_RELEASE_URL = (
    "https://github.com/audeering/opensmile/releases/download/"
    f"v{OPENSMILE_VERSION}/{OPENSMILE_BUNDLE_NAME}.tar.xz"
)
OPENSMILE_RELEASE_SHA256 = "ba2cc5a271ee9fb5cebdf372e60457fd512fad5f502b45e758ccfea1f3488e1a"
EGEMAPS_CONFIG_RELATIVE = Path("config/egemaps/v02/eGeMAPSv02.conf")
EXPECTED_EGEMAPS_DIMENSION = 88
SCALAR_METRICS = (
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def opensmile_config_tree(path: str | Path) -> tuple[str, Path, int]:
    """Hash the complete config tree containing the selected top-level config.

    openSMILE configs can resolve command-line macro includes dynamically. Hashing
    the full official config tree is deliberately conservative and avoids stale
    caches if any included fragment changes.
    """
    top = Path(path).resolve()
    config_root = next(
        (candidate for candidate in (top.parent, *top.parents) if candidate.name == "config"),
        top.parent,
    )
    files = sorted(
        (candidate for candidate in config_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: str(candidate.relative_to(config_root)),
    )
    if not files:
        raise FileNotFoundError(f"No files found in openSMILE config tree: {config_root}")
    digest = hashlib.sha256()
    for candidate in files:
        label = str(candidate.relative_to(config_root))
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
        digest.update(b"\0")
    return digest.hexdigest(), config_root, len(files)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def save_npz_atomic(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def validate_inputs(
    manifest_rows: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
    *,
    require_audio: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate canonical DAIC rows and return normalized rows and subject metadata."""
    if not manifest_rows:
        raise ValueError("The DAIC manifest is empty.")
    if not partition_rows:
        raise ValueError("The DAIC subject partition file is empty.")

    partitions: dict[str, dict[str, Any]] = {}
    for raw in partition_rows:
        subject_id = str(raw.get("subject_id", "")).strip()
        partition = str(raw.get("partition", "")).strip().lower()
        if not subject_id or partition not in {"train", "val", "test"}:
            raise ValueError(f"Invalid subject partition row: {raw!r}")
        label = int(raw["label"])
        if label not in {0, 1}:
            raise ValueError(f"Subject {subject_id} has non-binary label {label}.")
        if subject_id in partitions:
            raise ValueError(
                f"Subject {subject_id} occurs more than once in the partition file; "
                "participant-disjoint membership cannot be certified."
            )
        partitions[subject_id] = {
            "subject_id": subject_id,
            "partition": partition,
            "label": label,
        }

    normalized: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    subject_observations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for raw in manifest_rows:
        subject_id = str(raw.get("subject_id", "")).strip()
        sample_id = str(raw.get("sample_id", "")).strip()
        raw_audio_path = str(raw.get("audio_path", "")).strip()
        audio_path = Path(raw_audio_path).expanduser()
        partition = str(raw.get("split_original", "")).strip().lower()
        if not subject_id or not sample_id or not raw_audio_path:
            raise ValueError(f"Manifest row is missing subject/sample/audio identity: {raw!r}")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in DAIC manifest: {sample_id}")
        sample_ids.add(sample_id)
        label = int(raw["label"])
        if label not in {0, 1}:
            raise ValueError(f"Sample {sample_id} has non-binary label {label}.")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Sample {sample_id} has invalid split_original={partition!r}.")
        if require_audio and not audio_path.is_file():
            raise FileNotFoundError(f"Canonical audio is missing for {sample_id}: {audio_path}")
        subject_observations[subject_id].add((partition, label))
        row = dict(raw)
        row.update(
            subject_id=subject_id,
            sample_id=sample_id,
            audio_path=str(audio_path.resolve() if audio_path.exists() else audio_path),
            label=label,
            split_original=partition,
        )
        normalized.append(row)

    manifest_subjects = set(subject_observations)
    partition_subjects = set(partitions)
    if manifest_subjects != partition_subjects:
        missing_manifest = sorted(partition_subjects - manifest_subjects, key=natural_key)
        missing_partitions = sorted(manifest_subjects - partition_subjects, key=natural_key)
        raise ValueError(
            "Manifest/partition subject sets differ: "
            f"partition_only={missing_manifest[:10]}, manifest_only={missing_partitions[:10]}"
        )

    for subject_id, observations in subject_observations.items():
        if len(observations) != 1:
            raise ValueError(
                f"Subject {subject_id} has inconsistent manifest split/label values: "
                f"{sorted(observations)}"
            )
        partition, label = next(iter(observations))
        expected = partitions[subject_id]
        if (partition, label) != (expected["partition"], expected["label"]):
            raise ValueError(
                f"Subject {subject_id} disagrees between manifest and partition file: "
                f"manifest={(partition, label)}, partition="
                f"{(expected['partition'], expected['label'])}"
            )

    for partition in ("train", "val", "test"):
        labels = {
            int(meta["label"])
            for meta in partitions.values()
            if meta["partition"] == partition
        }
        if labels != {0, 1}:
            raise ValueError(
                f"Partition {partition!r} must contain both labels for evaluation; found {labels}."
            )

    normalized.sort(key=lambda row: (natural_key(row["subject_id"]), natural_key(row["sample_id"])))
    return normalized, partitions


def _sample_kind(sample_id: str) -> str:
    if "_random_segment_" in sample_id:
        return "random_segment"
    if "_segment_" in sample_id:
        return "segment"
    return "other"


def select_equal_chunks(
    manifest_rows: Sequence[dict[str, Any]],
    requested_chunks_per_subject: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the same deterministic number of chunks for every subject.

    Chunks are sorted by their numeric/chronological sample ID and selected at
    evenly spaced positions including both endpoints. Fixed K=4 is the primary
    protocol because current DAIC preprocessing has label-revealing 10-vs-15
    chunk counts.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["subject_id"])].append(dict(row))
    if not grouped:
        raise ValueError("Cannot select chunks from an empty manifest.")
    original_counts = {subject_id: len(rows) for subject_id, rows in grouped.items()}
    global_minimum = min(original_counts.values())
    resolved = requested_chunks_per_subject
    if resolved <= 0:
        raise ValueError("chunks_per_subject must be a positive integer (the primary protocol uses 4).")
    undersized = {
        subject_id: count
        for subject_id, count in original_counts.items()
        if count < resolved
    }
    if undersized:
        raise ValueError(
            f"{len(undersized)} subjects have fewer than {resolved} chunks: "
            f"{dict(list(sorted(undersized.items(), key=lambda item: natural_key(item[0])))[:10])}"
        )

    selected: list[dict[str, Any]] = []
    selected_positions: dict[str, list[int]] = {}
    for subject_id in sorted(grouped, key=natural_key):
        rows = sorted(grouped[subject_id], key=lambda row: natural_key(row["sample_id"]))
        positions = np.floor(np.linspace(0, len(rows) - 1, resolved) + 0.5).astype(int).tolist()
        if len(set(positions)) != resolved:
            raise RuntimeError(
                f"Evenly spaced selection produced duplicate positions for subject {subject_id}: {positions}"
            )
        selected_positions[subject_id] = positions
        selected.extend(rows[position] for position in positions)

    label_count_patterns: Counter[tuple[int, int]] = Counter()
    kind_by_label: Counter[tuple[int, str]] = Counter()
    split_subjects: Counter[str] = Counter()
    for subject_id, rows in grouped.items():
        label_count_patterns[(int(rows[0]["label"]), len(rows))] += 1
        split_subjects[str(rows[0]["split_original"])] += 1
        for row in rows:
            kind_by_label[(int(row["label"]), _sample_kind(str(row["sample_id"])))] += 1

    warnings: list[str] = []
    counts_by_label = {
        label: sorted({count for (observed_label, count), _ in label_count_patterns.items() if observed_label == label})
        for label in (0, 1)
    }
    if counts_by_label[0] != counts_by_label[1]:
        warnings.append(
            "Original chunk count is label-associated; equal-count selection prevents direct count leakage."
        )
    kinds_by_label = {
        label: sorted(
            kind
            for (observed_label, kind), count in kind_by_label.items()
            if observed_label == label and count
        )
        for label in (0, 1)
    }
    if set(kinds_by_label[0]).isdisjoint(kinds_by_label[1]):
        warnings.append(
            "Chunk sampling kind is perfectly label-associated in sample IDs; acoustic results may reflect "
            "the preprocessing policy as well as depression-related speech."
        )

    audit = {
        "selection_policy": "numeric_chronological_order_evenly_spaced_inclusive_endpoints",
        "requested_chunks_per_subject": requested_chunks_per_subject,
        "resolved_chunks_per_subject": resolved,
        "global_minimum_chunks": global_minimum,
        "selected_zero_based_positions_by_original_chunk_count": {
            str(count): np.floor(np.linspace(0, count - 1, resolved) + 0.5).astype(int).tolist()
            for count in sorted(set(original_counts.values()))
        },
        "subject_count": len(grouped),
        "selected_chunk_count": len(selected),
        "original_chunk_count_distribution": dict(sorted(Counter(original_counts.values()).items())),
        "original_subject_count_by_label_and_chunk_count": {
            f"label_{label}_chunks_{count}": frequency
            for (label, count), frequency in sorted(label_count_patterns.items())
        },
        "original_chunk_kind_count_by_label": {
            f"label_{label}_{kind}": count
            for (label, kind), count in sorted(kind_by_label.items())
        },
        "subject_count_by_split": dict(sorted(split_subjects.items())),
        "warnings": warnings,
    }
    return selected, audit


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, mode="r:xz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if os.path.commonpath([destination_resolved, target]) != str(destination_resolved):
                raise RuntimeError(f"Unsafe path in openSMILE release archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link in openSMILE release archive: {member.name}")
        tar.extractall(destination, members=members)


def provision_opensmile(bundle_dir: str | Path) -> dict[str, Any]:
    """Download the pinned official release into a local experiment tools directory."""
    target = Path(bundle_dir).expanduser().resolve()
    binary = target / "bin/SMILExtract"
    config = target / EGEMAPS_CONFIG_RELATIVE
    if binary.is_file() and config.is_file():
        return {
            "bundle_dir": str(target),
            "binary": str(binary),
            "config": str(config),
            "downloaded": False,
        }
    if target.exists():
        raise RuntimeError(
            f"Incomplete openSMILE bundle already exists at {target}; move it aside or choose another path."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".opensmile-provision-", dir=target.parent) as raw_temp:
        temp_dir = Path(raw_temp)
        archive = temp_dir / f"{OPENSMILE_BUNDLE_NAME}.tar.xz"
        request = urllib.request.Request(
            OPENSMILE_RELEASE_URL,
            headers={"User-Agent": "LLM-Depression-eGeMAPS-baseline/1"},
        )
        print(f"Downloading pinned openSMILE {OPENSMILE_VERSION} release...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        observed_hash = sha256_file(archive)
        if observed_hash != OPENSMILE_RELEASE_SHA256:
            raise RuntimeError(
                "openSMILE release checksum mismatch: "
                f"expected {OPENSMILE_RELEASE_SHA256}, observed {observed_hash}"
            )
        extraction_root = temp_dir / "extracted"
        extraction_root.mkdir()
        _safe_extract_tar(archive, extraction_root)
        extracted_bundle = extraction_root / OPENSMILE_BUNDLE_NAME
        if not (extracted_bundle / "bin/SMILExtract").is_file():
            raise RuntimeError("Pinned openSMILE archive did not contain the expected binary.")
        shutil.move(str(extracted_bundle), str(target))

    if not binary.is_file() or not config.is_file():
        raise RuntimeError(f"Provisioned openSMILE bundle is incomplete at {target}.")
    marker = {
        "version": OPENSMILE_VERSION,
        "release_url": OPENSMILE_RELEASE_URL,
        "release_sha256": OPENSMILE_RELEASE_SHA256,
        "binary": str(binary),
        "config": str(config),
    }
    write_json_atomic(target / ".provision.json", marker)
    return {"bundle_dir": str(target), "binary": str(binary), "config": str(config), "downloaded": True}


def opensmile_version(binary: str | Path) -> str:
    result = subprocess.run(
        [str(binary), "-h"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    match = re.search(r"openSMILE version\s+([^\s]+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not identify openSMILE version from {binary}: {result.stdout[:1000]}")
    return match.group(1)


def resolve_opensmile(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = Path(args.opensmile_bundle_dir).expanduser().resolve()
    provision: dict[str, Any] | None = None
    if args.provision_opensmile:
        provision = provision_opensmile(bundle_dir)

    binary_value = args.opensmile_bin
    if not binary_value and provision:
        binary_value = provision["binary"]
    if not binary_value:
        binary_value = shutil.which("SMILExtract")
    if not binary_value:
        raise FileNotFoundError(
            "SMILExtract was not found. Pass --provision-opensmile or --opensmile-bin PATH."
        )
    binary = Path(binary_value).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"SMILExtract binary does not exist: {binary}")

    config_value = args.opensmile_config
    if not config_value and provision:
        config_value = provision["config"]
    candidates = [
        bundle_dir / EGEMAPS_CONFIG_RELATIVE,
        binary.parent.parent / EGEMAPS_CONFIG_RELATIVE,
    ]
    if not config_value:
        config_value = next((str(path) for path in candidates if path.is_file()), None)
    if not config_value:
        raise FileNotFoundError(
            "eGeMAPSv02.conf was not found. Pass --provision-opensmile (recommended) or "
            "--opensmile-config PATH."
        )
    config = Path(config_value).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"eGeMAPSv02 config does not exist: {config}")

    version = opensmile_version(binary)
    if version != OPENSMILE_VERSION and not args.allow_opensmile_version_mismatch:
        raise RuntimeError(
            f"SMILExtract is version {version}, but this pipeline pins {OPENSMILE_VERSION}. "
            "Pass --allow-opensmile-version-mismatch only for an explicitly documented deviation."
        )
    binary_hash = sha256_file(binary)
    config_file_hash = sha256_file(config)
    config_tree_hash, config_tree_root, config_tree_file_count = opensmile_config_tree(config)
    extraction_signature = hashlib.sha256(
        f"binary={binary_hash}\nconfig_tree={config_tree_hash}\n".encode("ascii")
    ).hexdigest()
    return {
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "version": version,
        "config": str(config),
        "config_file_sha256": config_file_hash,
        "config_tree_root": str(config_tree_root),
        "config_tree_sha256": config_tree_hash,
        "config_tree_file_count": config_tree_file_count,
        "extraction_signature_sha256": extraction_signature,
        "provision": provision,
    }


def parse_opensmile_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter=";") if row]
    if len(rows) != 2:
        raise ValueError(f"Expected one eGeMAPS functional row in {path}, found {max(0, len(rows) - 1)}.")
    header, values = rows
    if len(header) != len(values) or header[:2] != ["name", "frameTime"]:
        raise ValueError(
            f"Unexpected openSMILE CSV schema in {path}: first columns={header[:2]}, "
            f"header={len(header)}, values={len(values)}"
        )
    feature_names = header[2:]
    if len(feature_names) != EXPECTED_EGEMAPS_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_EGEMAPS_DIMENSION} eGeMAPSv02 functionals, "
            f"found {len(feature_names)} in {path}."
        )
    try:
        vector = np.asarray([float(value) for value in values[2:]], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Non-numeric eGeMAPS value in {path}: {exc}") from exc
    vector[~np.isfinite(vector)] = np.nan
    return feature_names, vector


def _cache_path(
    cache_dir: Path,
    sample_id: str,
    audio_sha256: str,
    extraction_signature_sha256: str,
) -> Path:
    safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return cache_dir / (
        f"{safe_sample}.{audio_sha256[:12]}.{extraction_signature_sha256[:12]}.npz"
    )


def _load_cached_feature(
    path: Path,
    *,
    audio_sha256: str,
    extraction_signature_sha256: str,
) -> tuple[list[str], np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["audio_sha256"].item()) != audio_sha256:
                return None
            if (
                str(cache["extraction_signature_sha256"].item())
                != extraction_signature_sha256
            ):
                return None
            names = [str(value) for value in cache["feature_names"].tolist()]
            vector = np.asarray(cache["features"], dtype=np.float64)
    except (OSError, ValueError, KeyError):
        return None
    if vector.shape != (EXPECTED_EGEMAPS_DIMENSION,) or len(names) != EXPECTED_EGEMAPS_DIMENSION:
        return None
    return names, vector


def extract_one(
    row: dict[str, Any],
    *,
    binary: str | Path,
    config: str | Path,
    extraction_signature_sha256: str,
    cache_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"]))
    audio_hash = sha256_file(audio_path)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        cache_root,
        str(row["sample_id"]),
        audio_hash,
        extraction_signature_sha256,
    )
    cached = None if force else _load_cached_feature(
        cache_path,
        audio_sha256=audio_hash,
        extraction_signature_sha256=extraction_signature_sha256,
    )
    if cached is not None:
        names, vector = cached
        return {
            "row": row,
            "feature_names": names,
            "features": vector,
            "audio_sha256": audio_hash,
            "cache_path": str(cache_path),
            "cache_hit": True,
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(vector))),
        }

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        prefix=f".{row['sample_id']}.",
        dir=cache_root,
        delete=False,
    ) as handle:
        output_csv = Path(handle.name)
    output_csv.unlink(missing_ok=True)
    command = [
        str(binary),
        "-C",
        str(config),
        "-I",
        str(audio_path),
        "-csvoutput",
        str(output_csv),
        "-l",
        "0",
        "-noconsoleoutput",
        "1",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SMILExtract failed for {row['sample_id']} (exit {result.returncode}): "
                f"{result.stdout[-3000:]}"
            )
        if not output_csv.is_file():
            raise RuntimeError(f"SMILExtract produced no CSV for {row['sample_id']}.")
        names, vector = parse_opensmile_csv(output_csv)
    finally:
        output_csv.unlink(missing_ok=True)

    save_npz_atomic(
        cache_path,
        audio_sha256=np.asarray(audio_hash),
        extraction_signature_sha256=np.asarray(extraction_signature_sha256),
        feature_names=np.asarray(names),
        features=vector,
    )
    return {
        "row": row,
        "feature_names": names,
        "features": vector,
        "audio_sha256": audio_hash,
        "cache_path": str(cache_path),
        "cache_hit": False,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(vector))),
    }


def extract_selected_features(
    rows: Sequence[dict[str, Any]],
    *,
    opensmile: dict[str, Any],
    cache_dir: str | Path,
    jobs: int,
    force: bool = False,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    if jobs <= 0:
        raise ValueError("jobs must be positive.")
    records: list[dict[str, Any] | None] = [None] * len(rows)

    def task(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return index, extract_one(
            row,
            binary=opensmile["binary"],
            config=opensmile["config"],
            extraction_signature_sha256=opensmile["extraction_signature_sha256"],
            cache_dir=cache_dir,
            force=force,
        )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(task, index, row) for index, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, record = future.result()
            records[index] = record
            if completed == 1 or completed % 100 == 0 or completed == len(rows):
                print(f"eGeMAPS extraction: {completed}/{len(rows)} chunks", flush=True)

    concrete = [record for record in records if record is not None]
    if len(concrete) != len(rows):
        raise RuntimeError("Internal extraction error: one or more feature records are missing.")
    reference_names = concrete[0]["feature_names"]
    for record in concrete[1:]:
        if record["feature_names"] != reference_names:
            raise ValueError(f"Feature schema changed at sample {record['row']['sample_id']}.")
    matrix = np.vstack([record["features"] for record in concrete])
    return list(reference_names), matrix, concrete


def _finite_mean_std(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0)
    total = np.where(finite, matrix, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.full(matrix.shape[1], np.nan), where=count > 0)
    squared = np.where(finite, (matrix - mean) ** 2, 0.0).sum(axis=0)
    variance = np.divide(squared, count, out=np.full(matrix.shape[1], np.nan), where=count > 0)
    return mean, np.sqrt(variance)


def aggregate_subject_features(
    rows: Sequence[dict[str, Any]],
    chunk_features: np.ndarray,
    chunk_feature_names: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    if len(rows) != len(chunk_features):
        raise ValueError("Rows and chunk feature matrix have different lengths.")
    if chunk_features.ndim != 2 or chunk_features.shape[1] != len(chunk_feature_names):
        raise ValueError("Chunk feature matrix does not match the feature-name schema.")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices[str(row["subject_id"])].append(index)

    subject_rows: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    for subject_id in sorted(indices, key=natural_key):
        subject_indices = indices[subject_id]
        subject_chunks = chunk_features[subject_indices]
        mean, std = _finite_mean_std(subject_chunks)
        first = rows[subject_indices[0]]
        observed = {
            (str(rows[index]["split_original"]), int(rows[index]["label"]))
            for index in subject_indices
        }
        if len(observed) != 1:
            raise ValueError(f"Inconsistent split/label while pooling subject {subject_id}: {observed}")
        subject_rows.append(
            {
                "subject_id": subject_id,
                "partition": str(first["split_original"]),
                "label": int(first["label"]),
                "selected_chunk_count": len(subject_indices),
            }
        )
        pooled.append(np.concatenate([mean, std]))
    names = [f"chunk_mean__{name}" for name in chunk_feature_names] + [
        f"chunk_std__{name}" for name in chunk_feature_names
    ]
    return subject_rows, np.vstack(pooled), names


def evaluate_binary(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.shape != y.shape or len(y) == 0:
        raise ValueError("Binary targets/probabilities must be non-empty one-dimensional arrays of equal size.")
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    prediction = (probability >= 0.5).astype(np.int64)
    tn = int(np.count_nonzero((y == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((y == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((y == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((y == 1) & (prediction == 1)))

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision_pos = safe_divide(tp, tp + fp)
    recall_pos = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_pos = safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = safe_divide(tn, tn + fn)
    recall_neg = specificity
    f1_neg = safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    log_loss_value = float(
        -np.mean(y * np.log(probability) + (1 - y) * np.log(1.0 - probability))
    )
    has_both_labels = len(set(y.tolist())) == 2
    auroc = float(roc_auc_score(y, probability)) if has_both_labels else None
    auprc = float(average_precision_score(y, probability)) if np.any(y == 1) else None
    return {
        "accuracy": safe_divide(tp + tn, len(y)),
        "balanced_accuracy": (recall_pos + specificity) / 2.0,
        "precision": precision_pos,
        "recall": recall_pos,
        "specificity": specificity,
        "positive_f1": f1_pos,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "auroc": auroc,
        "auprc": auprc,
        "log_loss": log_loss_value,
        "brier_score": float(np.mean((probability - y) ** 2)),
        "support_negative": int(np.count_nonzero(y == 0)),
        "support_positive": int(np.count_nonzero(y == 1)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _build_linear_pipeline(c_value: float, seed: int) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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


def select_regularization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("The regularization C grid must contain positive values.")
    records: list[dict[str, Any]] = []
    for c_value in sorted(set(float(value) for value in c_grid)):
        model = _build_linear_pipeline(c_value, seed)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)[:, 1]
        records.append(
            {
                "C": c_value,
                "selection_objective": "minimum_validation_log_loss",
                "validation_metrics": evaluate_binary(y_val, probabilities),
            }
        )
    best = min(records, key=lambda record: (record["validation_metrics"]["log_loss"], record["C"]))
    return float(best["C"]), records


def fit_final_linear_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[Any, float, list[dict[str, Any]]]:
    selected_c, records = select_regularization(
        x_train,
        y_train,
        x_val,
        y_val,
        c_grid=c_grid,
        seed=seed,
    )
    final = _build_linear_pipeline(selected_c, seed)
    final.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))
    return final, selected_c, records


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        return {"repeats": 0, "seed": seed, "intervals": {}}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in SCALAR_METRICS}
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = evaluate_binary(y_true[indices], probabilities[indices])
        for metric in SCALAR_METRICS:
            value = metrics.get(metric)
            if value is not None and math.isfinite(float(value)):
                samples[metric].append(float(value))
    intervals = {}
    for metric, values in samples.items():
        if not values:
            continue
        intervals[metric] = {
            "lower_2.5pct": float(np.quantile(values, 0.025)),
            "upper_97.5pct": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
    return {"repeats": repeats, "seed": seed, "unit": "subject", "intervals": intervals}


def run_shuffled_audio_control(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    c_grid: Sequence[float],
    repeats: int,
    seed: int,
    real_metrics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Permute subject-level audio features independently within each split."""
    if repeats <= 0:
        return [], {
            "repeats": 0,
            "seed": seed,
            "permutation_unit": "subject_feature_vector_within_partition",
            "metrics": {},
        }
    children = np.random.SeedSequence(seed).spawn(repeats)
    records: list[dict[str, Any]] = []
    for repeat_index, child in enumerate(children):
        repeat_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(child)
        shuffled_train = x_train[rng.permutation(len(x_train))]
        shuffled_val = x_val[rng.permutation(len(x_val))]
        shuffled_test = x_test[rng.permutation(len(x_test))]
        model, selected_c, _ = fit_final_linear_model(
            shuffled_train,
            y_train,
            shuffled_val,
            y_val,
            c_grid=c_grid,
            seed=repeat_seed,
        )
        probability = model.predict_proba(shuffled_test)[:, 1]
        records.append(
            {
                "repeat": repeat_index,
                "seed": repeat_seed,
                "selected_C": selected_c,
                "test_metrics": evaluate_binary(y_test, probability),
            }
        )

    summary_metrics: dict[str, Any] = {}
    for metric in SCALAR_METRICS:
        values = [
            float(record["test_metrics"][metric])
            for record in records
            if record["test_metrics"].get(metric) is not None
        ]
        if not values:
            continue
        summary: dict[str, Any] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "lower_2.5pct": float(np.quantile(values, 0.025)),
            "upper_97.5pct": float(np.quantile(values, 0.975)),
        }
        if real_metrics and real_metrics.get(metric) is not None:
            observed = float(real_metrics[metric])
            if metric in HIGHER_IS_BETTER:
                summary["empirical_p_value"] = (1 + sum(value >= observed for value in values)) / (1 + len(values))
                summary["tail"] = "shuffle_greater_or_equal"
            else:
                summary["empirical_p_value"] = (1 + sum(value <= observed for value in values)) / (1 + len(values))
                summary["tail"] = "shuffle_less_or_equal"
        summary_metrics[metric] = summary
    return records, {
        "repeats": repeats,
        "seed": seed,
        "permutation_unit": "subject_feature_vector_within_partition",
        "cross_partition_feature_movement": False,
        "metrics": summary_metrics,
    }


def _partition_arrays(
    subject_rows: Sequence[dict[str, Any]],
    subject_features: np.ndarray,
    partition: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    indices = [index for index, row in enumerate(subject_rows) if row["partition"] == partition]
    subject_ids = [str(subject_rows[index]["subject_id"]) for index in indices]
    labels = np.asarray([int(subject_rows[index]["label"]) for index in indices], dtype=np.int64)
    return subject_ids, subject_features[indices], labels


def _majority_baseline(y_development: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(y_development, minlength=2)
    majority_label = int(np.flatnonzero(counts == counts.max())[0])
    prevalence = float(np.mean(y_development))
    probabilities = np.full(len(y_test), prevalence, dtype=np.float64)
    metrics = evaluate_binary(y_test, probabilities)
    metrics.update(
        majority_label=majority_label,
        development_positive_prevalence=prevalence,
        probability_policy="development_positive_prevalence",
    )
    return probabilities, metrics


def _write_predictions(
    path: Path,
    subject_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    majority_probabilities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "label",
                "probability_depressed",
                "prediction",
                "majority_probability_depressed",
                "majority_prediction",
            ],
        )
        writer.writeheader()
        for subject_id, label, probability, majority_probability in zip(
            subject_ids, labels, probabilities, majority_probabilities
        ):
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "label": int(label),
                    "probability_depressed": f"{float(probability):.12g}",
                    "prediction": int(probability >= 0.5),
                    "majority_probability_depressed": f"{float(majority_probability):.12g}",
                    "majority_prediction": int(majority_probability >= 0.5),
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(), "numpy": np.__version__}
    for module_name in ("sklearn", "scipy", "joblib"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = None
    return versions


def _structural_counts(
    subject_rows: Sequence[dict[str, Any]],
    chunk_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    subject_counts = Counter((row["partition"], int(row["label"])) for row in subject_rows)
    chunk_counts = Counter(
        (record["row"]["split_original"], int(record["row"]["label"]))
        for record in chunk_records
    )
    return {
        "subjects_by_partition_and_label": {
            f"{partition}_label_{label}": count
            for (partition, label), count in sorted(subject_counts.items())
        },
        "selected_chunks_by_partition_and_label": {
            f"{partition}_label_{label}": count
            for (partition, label), count in sorted(chunk_counts.items())
        },
        "chunk_cache_hits": sum(bool(record["cache_hit"]) for record in chunk_records),
        "chunk_cache_misses": sum(not bool(record["cache_hit"]) for record in chunk_records),
        "chunks_with_nonfinite_features": sum(record["nonfinite_count"] > 0 for record in chunk_records),
        "total_nonfinite_feature_values": sum(record["nonfinite_count"] for record in chunk_records),
    }


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Extract eGeMAPSv02 functionals and fit a subject-level, L2-regularized "
            "linear DAIC acoustic ceiling with majority and shuffled-audio controls."
        )
    )
    parser.add_argument("--manifest", default=str(root / "outputs/manifests/daic_manifest.jsonl"))
    parser.add_argument(
        "--partitions", default=str(root / "outputs/splits/daic_subject_partitions.json")
    )
    parser.add_argument(
        "--output-dir", default=str(root / "outputs/baselines/daic_egemaps_v02_fixedk4")
    )
    parser.add_argument(
        "--chunks-per-subject",
        type=int,
        default=4,
        help=(
            "Equal, evenly spaced chunks used per subject (default: 4; the preregistered primary "
            "protocol)."
        ),
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    )
    parser.add_argument("--opensmile-bin")
    parser.add_argument("--opensmile-config")
    parser.add_argument(
        "--opensmile-bundle-dir",
        default=str(root / f"outputs/tools/{OPENSMILE_BUNDLE_NAME}"),
    )
    parser.add_argument(
        "--provision-opensmile",
        action="store_true",
        help="Download and verify the official pinned 3.0.1 Linux bundle under outputs/tools.",
    )
    parser.add_argument("--allow-opensmile-version-mismatch", action="store_true")
    parser.add_argument("--force-extract", action="store_true", help="Ignore valid per-chunk caches.")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after saving chunk and subject features (useful for staged runs).",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import joblib

    root = project_root()
    manifest_path = Path(args.manifest).expanduser().resolve()
    partition_path = Path(args.partitions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache/chunks"

    manifest_rows = _read_jsonl(manifest_path)
    raw_partitions = _read_json(partition_path)
    if not isinstance(raw_partitions, list):
        raise ValueError(f"Expected a list in {partition_path}, found {type(raw_partitions).__name__}.")
    normalized_rows, _ = validate_inputs(manifest_rows, raw_partitions, require_audio=True)
    selected_rows, selection_audit = select_equal_chunks(
        normalized_rows, args.chunks_per_subject
    )
    for warning in selection_audit["warnings"]:
        print(f"AUDIT WARNING: {warning}", file=sys.stderr, flush=True)

    opensmile = resolve_opensmile(args)
    run_config = {
        "status": "extracting",
        "command": shlex.join([sys.executable, *sys.argv]),
        "arguments": vars(args),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "partitions": {"path": str(partition_path), "sha256": sha256_file(partition_path)},
        "opensmile": opensmile,
        "feature_set": "eGeMAPSv02",
        "feature_level": "Functionals",
        "subject_pooling": ["mean", "population_standard_deviation"],
        "selection_audit": selection_audit,
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    chunk_names, chunk_matrix, chunk_records = extract_selected_features(
        selected_rows,
        opensmile=opensmile,
        cache_dir=cache_dir,
        jobs=args.jobs,
        force=args.force_extract,
    )
    subject_rows, subject_matrix, subject_names = aggregate_subject_features(
        selected_rows, chunk_matrix, chunk_names
    )
    save_npz_atomic(
        output_dir / "chunk_features.npz",
        sample_ids=np.asarray([row["sample_id"] for row in selected_rows]),
        subject_ids=np.asarray([row["subject_id"] for row in selected_rows]),
        partitions=np.asarray([row["split_original"] for row in selected_rows]),
        labels=np.asarray([row["label"] for row in selected_rows], dtype=np.int64),
        feature_names=np.asarray(chunk_names),
        features=chunk_matrix,
    )
    write_jsonl_atomic(
        output_dir / "chunk_feature_metadata.jsonl",
        (
            {
                "sample_id": record["row"]["sample_id"],
                "subject_id": record["row"]["subject_id"],
                "partition": record["row"]["split_original"],
                "label": record["row"]["label"],
                "audio_path": record["row"]["audio_path"],
                "audio_sha256": record["audio_sha256"],
                "cache_path": record["cache_path"],
                "cache_hit": record["cache_hit"],
                "nonfinite_count": record["nonfinite_count"],
            }
            for record in chunk_records
        ),
    )
    save_npz_atomic(
        output_dir / "subject_features.npz",
        subject_ids=np.asarray([row["subject_id"] for row in subject_rows]),
        partitions=np.asarray([row["partition"] for row in subject_rows]),
        labels=np.asarray([row["label"] for row in subject_rows], dtype=np.int64),
        selected_chunk_counts=np.asarray(
            [row["selected_chunk_count"] for row in subject_rows], dtype=np.int64
        ),
        feature_names=np.asarray(subject_names),
        features=subject_matrix,
    )

    structural_counts = _structural_counts(subject_rows, chunk_records)
    if args.extract_only:
        summary = {
            "status": "features_complete",
            "output_dir": str(output_dir),
            "chunk_feature_shape": list(chunk_matrix.shape),
            "subject_feature_shape": list(subject_matrix.shape),
            "counts": structural_counts,
        }
        write_json_atomic(output_dir / "run_summary.json", summary)
        run_config["status"] = "features_complete"
        write_json_atomic(output_dir / "run_config.json", run_config)
        return summary

    train_ids, x_train, y_train = _partition_arrays(subject_rows, subject_matrix, "train")
    val_ids, x_val, y_val = _partition_arrays(subject_rows, subject_matrix, "val")
    test_ids, x_test, y_test = _partition_arrays(subject_rows, subject_matrix, "test")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError("Participant overlap detected after subject aggregation.")

    model, selected_c, selection_records = fit_final_linear_model(
        x_train,
        y_train,
        x_val,
        y_val,
        c_grid=args.c_grid,
        seed=args.seed,
    )
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_probabilities)
    majority_probabilities, majority_metrics = _majority_baseline(
        np.concatenate([y_train, y_val]), y_test
    )
    bootstrap = bootstrap_intervals(
        y_test,
        test_probabilities,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1,
    )
    shuffle_records, shuffle_summary = run_shuffled_audio_control(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        c_grid=args.c_grid,
        repeats=args.shuffle_repeats,
        seed=args.seed + 2,
        real_metrics=test_metrics,
    )

    write_json_atomic(
        output_dir / "validation_selection.json",
        {
            "objective": "minimum_validation_log_loss",
            "selected_C": selected_c,
            "records": selection_records,
            "final_refit_partitions": ["train", "val"],
            "test_used_during_selection": False,
        },
    )
    metrics_payload = {
        "unit": "subject",
        "threshold": 0.5,
        "model": test_metrics,
        "majority": majority_metrics,
        "model_bootstrap_95pct": bootstrap,
        "shuffled_audio": shuffle_summary,
    }
    write_json_atomic(output_dir / "test_metrics.json", metrics_payload)
    write_jsonl_atomic(output_dir / "shuffled_audio_runs.jsonl", shuffle_records)
    _write_predictions(
        output_dir / "test_predictions.csv",
        test_ids,
        y_test,
        test_probabilities,
        majority_probabilities,
    )
    temporary_model = output_dir / ".model.joblib.tmp"
    joblib.dump(model, temporary_model)
    os.replace(temporary_model, output_dir / "model.joblib")

    provenance = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": _dependency_versions(),
        "git": _git_provenance(root),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "partitions": {"path": str(partition_path), "sha256": sha256_file(partition_path)},
        },
        "opensmile": opensmile,
        "randomness": {
            "primary_seed": args.seed,
            "bootstrap_seed": args.seed + 1,
            "shuffle_seed": args.seed + 2,
        },
        "leakage_guards": {
            "participant_disjoint": True,
            "one_prediction_per_subject": True,
            "test_used_during_model_selection": False,
            "equal_chunks_per_subject": selection_audit["resolved_chunks_per_subject"],
            "chunk_count_is_model_feature": False,
            "shuffle_is_within_partition": True,
        },
        "selection_audit": selection_audit,
        "counts": structural_counts,
        "feature_shapes": {
            "chunk": list(chunk_matrix.shape),
            "subject": list(subject_matrix.shape),
        },
    }
    write_json_atomic(output_dir / "provenance.json", provenance)

    run_config["status"] = "complete"
    run_config["selected_C"] = selected_c
    write_json_atomic(output_dir / "run_config.json", run_config)
    summary = {
        "status": "complete",
        "output_dir": str(output_dir),
        "selected_C": selected_c,
        "test_metrics": test_metrics,
        "majority_metrics": majority_metrics,
        "shuffled_audio": shuffle_summary,
        "counts": structural_counts,
    }
    write_json_atomic(output_dir / "run_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
