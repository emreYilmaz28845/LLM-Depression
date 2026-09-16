#!/usr/bin/env python3
"""Build and verify MN5-native harmonized manifests and merged protocols."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.merged.protocol import load_component_records, save_protocol_artifacts
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)


COMPONENT_CONFIGS = (
    "configs/main/d3tec_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/turkish_pos_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml",
    "configs/main/androids_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/cmdc_audio_text_harmonized_selmacrof1_tf.yaml",
)
MERGED_CONFIGS = (
    "configs/experiments/merged/symmetric_merged_harmonized_audio_text.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_audio_only.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_text_only.yaml",
)

# The Turkish pooled input is not produced by build_for_config: it is the
# concatenation of four already-validated source pairs, built by
# scripts/build_turkish_pooled_manifest.py. The constants below pin the published
# pooled campaign contract (group_id, production preflight audit, status passed).
# Source hashes were re-verified byte-for-byte against that audit; the output
# hashes are enforced so that a rebuild which fails to reproduce the published
# pooled input stops the preflight instead of silently swapping the Turkish
# dataset under the merged model.
POOLED_TURKISH_GROUP_ID = "turkish-pooled-qcond-clean-v1-20260903"
POOLED_TURKISH_COMPONENT_CONFIG = (
    "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml"
)
POOLED_TURKISH_COMPONENT_CONFIG_EN = (
    "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml"
)
POOLED_TURKISH_SOURCE_FILES = {
    ("pos_only_t17", "native"): (
        "manifests/pos_native/turkish_manifest.jsonl",
        "b26cce5e1d3b2ab9b955b28159eb28ebebec3383dc548016ee2dfa03c27b3e2d",
        1051,
    ),
    ("pos_only_t17", "english"): (
        "manifests/pos_english/turkish_manifest.jsonl",
        "c66d781e6cdc8229246c7ce36568f20fdb33f86adf6b0892f0024b2e13e93a7b",
        1051,
    ),
    ("negative_only_t17", "native"): (
        "manifests/neg_native/turkish_manifest.jsonl",
        "f655b1b76c0fe855144507b6b69521d4ff4236be816fa187878675573aea7099",
        1170,
    ),
    ("negative_only_t17", "english"): (
        "manifests/neg_english/turkish_manifest.jsonl",
        "0d19d00ed3a5e5821a40222bef182cc088667181d7ed4a4ce3d4b1698253f200",
        1170,
    ),
}
POOLED_TURKISH_SOURCE_SPLIT_SHA256 = (
    "2ae3d417debd07cb72ee5881a2ffb6d7202c21e5a2c0be8cd0b059b250a97813"
)
POOLED_TURKISH_SOURCE_SPLIT_FILES = {
    ("pos_only_t17", "native"): "splits/pos_native/turkish_folds.json",
    ("pos_only_t17", "english"): "splits/pos_english/turkish_folds.json",
    ("negative_only_t17", "native"): "splits/neg_native/turkish_folds.json",
    ("negative_only_t17", "english"): "splits/neg_english/turkish_folds.json",
}
POOLED_TURKISH_EXPECTED = {
    "native": {
        "manifest_sha256": "a0daf6658f111f113bcf17231a6abbc0b8a472a3d60d9ddf76cd9ff25eb4e089",
        "manifest_hash": "37e991526986d9693c9620682719a6b54c7d30ec86b53152ae23dde167271b70",
        "folds_sha256": "3262a009db52c6d049e223947a9be6ce119a31e816b5c3072ce84b3ad92ecd58",
        "fold_hash": "7854f7300e60fa50cb6556c627ebc936380bc3de8fc65c2f3f9b4c261d17e97a",
    },
    "english": {
        "manifest_sha256": "cdbc113f97c204bb1eb487a4fd06867e7f1ed6a95ffba28df26c166ab97734fe",
        "manifest_hash": "5c4578e76042ce7b5d3def8258bb3d6efbd3433a79cc0e9e40fc4ab91e2630b4",
        "folds_sha256": "3262a009db52c6d049e223947a9be6ce119a31e816b5c3072ce84b3ad92ecd58",
        "fold_hash": "7854f7300e60fa50cb6556c627ebc936380bc3de8fc65c2f3f9b4c261d17e97a",
    },
}
POOLED_TURKISH_EXPECTED_ROWS = 2221
POOLED_TURKISH_EXPECTED_SUBJECTS = 120
POOLED_TURKISH_EXPECTED_CONDITIONS = {"pos_only_t17": 1051, "negative_only_t17": 1170}
POOLED_MERGED_CONFIGS = (
    "configs/experiments/merged/symmetric_merged_harmonized_pooled_t17_audio_text.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_pooled_t17_audio_only.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_pooled_t17_text_only.yaml",
)
POOLED_GEMMA_MERGED_CONFIGS = (
    "configs/experiments/merged/symmetric_merged_harmonized_gemma4_pooled_t17_audio_text.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_gemma4_pooled_t17_audio_only.yaml",
    "configs/experiments/merged/symmetric_merged_harmonized_gemma4_pooled_t17_text_only.yaml",
)


def pooled_component_configs() -> tuple[str, ...]:
    """The component set whose Turkish member is the pooled question-conditioned input."""
    non_turkish = tuple(path for path in COMPONENT_CONFIGS if "/turkish_" not in path)
    return (*non_turkish, POOLED_TURKISH_COMPONENT_CONFIG)


def _pooled_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml_with_overrides(config_path, [])
    return (
        resolve_project_path(config["output_dirs"]["manifest_dir"]),
        resolve_project_path(config["output_dirs"]["split_dir"]),
    )


def _sha256_or_raise(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path} has {actual}, expected {expected}")
    return actual


def verify_pooled_turkish_source(source_root: Path) -> dict[str, Any]:
    """Verify the four source pairs against the hashes recorded by the published campaign."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Pooled Turkish source root is missing: {source_root}")
    manifests: dict[str, Any] = {}
    for (condition, language), (relative, expected, rows) in sorted(
        POOLED_TURKISH_SOURCE_FILES.items()
    ):
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pooled Turkish source manifest: {path}")
        _sha256_or_raise(path, expected, f"pooled source manifest {condition}/{language}")
        payload = read_jsonl(path)
        if len(payload) != rows:
            raise ValueError(
                f"pooled source manifest {condition}/{language} has {len(payload)} rows, "
                f"expected {rows}: {path}"
            )
        manifests[f"{condition}/{language}"] = {
            "path": str(path),
            "sha256": expected,
            "rows": rows,
        }
    for label, relative in POOLED_TURKISH_SOURCE_SPLIT_FILES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pooled Turkish source split: {path}")
        _sha256_or_raise(
            path,
            POOLED_TURKISH_SOURCE_SPLIT_SHA256,
            f"pooled source split {label[0]}/{label[1]}",
        )
    return {
        "group_id": POOLED_TURKISH_GROUP_ID,
        "root": str(source_root),
        "manifests": manifests,
        "split_sha256": POOLED_TURKISH_SOURCE_SPLIT_SHA256,
        "split_files": sorted(POOLED_TURKISH_SOURCE_SPLIT_FILES.values()),
    }


def verify_pooled_turkish_outputs() -> dict[str, Any]:
    """Enforce the published pooled manifest/fold hashes on the built outputs."""
    records: dict[str, Any] = {}
    for language, config_rel in (
        ("native", POOLED_TURKISH_COMPONENT_CONFIG),
        ("english", POOLED_TURKISH_COMPONENT_CONFIG_EN),
    ):
        expected = POOLED_TURKISH_EXPECTED[language]
        manifest_dir, split_dir = _pooled_output_dirs(resolve_project_path(config_rel))
        metadata_path = split_dir / "turkish_manifest_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing pooled Turkish metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        if metadata.get("dataset_variant") != "pooled_t17":
            raise ValueError(f"pooled {language} metadata is not pooled_t17: {metadata_path}")
        manifest_path = resolve_project_path(metadata["manifest_path"])
        folds_path = resolve_project_path(metadata["folds_path"])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing pooled {language} manifest: {manifest_path}")
        if not folds_path.is_file():
            raise FileNotFoundError(f"Missing pooled {language} folds: {folds_path}")
        _sha256_or_raise(
            manifest_path, expected["manifest_sha256"], f"pooled {language} manifest"
        )
        _sha256_or_raise(folds_path, expected["folds_sha256"], f"pooled {language} folds")
        if metadata.get("manifest_hash") != expected["manifest_hash"]:
            raise ValueError(
                f"pooled {language} canonical manifest hash mismatch: "
                f"{metadata.get('manifest_hash')} != {expected['manifest_hash']}"
            )
        if metadata.get("fold_hash") != expected["fold_hash"]:
            raise ValueError(
                f"pooled {language} canonical fold hash mismatch: "
                f"{metadata.get('fold_hash')} != {expected['fold_hash']}"
            )
        rows = read_jsonl(manifest_path)
        if len(rows) != POOLED_TURKISH_EXPECTED_ROWS:
            raise ValueError(
                f"pooled {language} manifest has {len(rows)} rows, "
                f"expected {POOLED_TURKISH_EXPECTED_ROWS}"
            )
        subjects = {str(row["subject_id"]) for row in rows}
        if len(subjects) != POOLED_TURKISH_EXPECTED_SUBJECTS:
            raise ValueError(
                f"pooled {language} manifest has {len(subjects)} subjects, "
                f"expected {POOLED_TURKISH_EXPECTED_SUBJECTS}"
            )
        conditions = Counter(str(row.get("dataset_variant", "")) for row in rows)
        if dict(conditions) != POOLED_TURKISH_EXPECTED_CONDITIONS:
            raise ValueError(
                f"pooled {language} condition counts mismatch: {dict(conditions)}"
            )
        records[language] = {
            "config": config_rel,
            "manifest_dir": str(manifest_dir),
            "split_dir": str(split_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": expected["manifest_sha256"],
            "manifest_hash": expected["manifest_hash"],
            "metadata_path": str(metadata_path),
            "folds_path": str(folds_path),
            "folds_sha256": expected["folds_sha256"],
            "fold_hash": expected["fold_hash"],
            "rows": len(rows),
            "subjects": len(subjects),
            "condition_counts": {key: int(value) for key, value in sorted(conditions.items())},
        }
    if records["native"]["manifest_sha256"] == records["english"]["manifest_sha256"]:
        raise ValueError("native and English pooled manifest hashes unexpectedly match")
    return records


def build_pooled_turkish(source_root: Path, *, audit_path: Path | None = None) -> dict[str, Any]:
    """Rebuild the pooled Turkish manifest from its sources, then enforce the published hashes."""
    from scripts.build_turkish_pooled_manifest import main as build_pooled_manifest

    native_manifest_dir, native_split_dir = _pooled_output_dirs(
        resolve_project_path(POOLED_TURKISH_COMPONENT_CONFIG)
    )
    english_manifest_dir, english_split_dir = _pooled_output_dirs(
        resolve_project_path(POOLED_TURKISH_COMPONENT_CONFIG_EN)
    )
    source_audit = verify_pooled_turkish_source(source_root)

    def source(relative: str) -> str:
        return str(source_root / relative)

    argv = [
        "--positive-native-manifest",
        source(POOLED_TURKISH_SOURCE_FILES[("pos_only_t17", "native")][0]),
        "--positive-native-split",
        source(POOLED_TURKISH_SOURCE_SPLIT_FILES[("pos_only_t17", "native")]),
        "--negative-native-manifest",
        source(POOLED_TURKISH_SOURCE_FILES[("negative_only_t17", "native")][0]),
        "--negative-native-split",
        source(POOLED_TURKISH_SOURCE_SPLIT_FILES[("negative_only_t17", "native")]),
        "--positive-english-manifest",
        source(POOLED_TURKISH_SOURCE_FILES[("pos_only_t17", "english")][0]),
        "--positive-english-split",
        source(POOLED_TURKISH_SOURCE_SPLIT_FILES[("pos_only_t17", "english")]),
        "--negative-english-manifest",
        source(POOLED_TURKISH_SOURCE_FILES[("negative_only_t17", "english")][0]),
        "--negative-english-split",
        source(POOLED_TURKISH_SOURCE_SPLIT_FILES[("negative_only_t17", "english")]),
        "--native-output-dir",
        str(native_manifest_dir),
        "--english-output-dir",
        str(english_manifest_dir),
        "--native-split-output-dir",
        str(native_split_dir),
        "--english-split-output-dir",
        str(english_split_dir),
        "--audit-output",
        str(
            audit_path
            or PROJECT_ROOT / "outputs/turkish_pooled_preflight" / "manifest_audit.json"
        ),
        "--native-config",
        str(resolve_project_path(POOLED_TURKISH_COMPONENT_CONFIG)),
        "--english-config",
        str(resolve_project_path(POOLED_TURKISH_COMPONENT_CONFIG_EN)),
    ]
    build_pooled_manifest(argv)
    return {"source": source_audit, "outputs": verify_pooled_turkish_outputs()}


def _path_strings(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _path_strings(child_value, key=str(child_key))
    elif isinstance(value, list):
        if key.endswith("paths"):
            for item in value:
                if isinstance(item, str):
                    yield item
        else:
            for item in value:
                yield from _path_strings(item, key=key)
    elif isinstance(value, str) and key.endswith("path"):
        yield value


def validate_component(
    config_path: Path, *, required_path_prefix: Path | None
) -> dict[str, Any]:
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing harmonized metadata for {dataset}: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("build_signature") != manifest_build_signature(config):
        raise ValueError(f"Stale build signature for {dataset}: {metadata_path}")
    manifest_path = resolve_project_path(metadata["manifest_path"])
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Empty harmonized manifest for {dataset}: {manifest_path}")
    if int(metadata.get("manifest_row_count", -1)) != len(rows):
        raise ValueError(f"Manifest row-count mismatch for {dataset}: {manifest_path}")

    prefix = required_path_prefix.resolve() if required_path_prefix else None
    checked_paths: set[str] = set()
    for row in rows:
        for text in _path_strings(row):
            if not text or text.startswith("${PROJECT_ROOT}"):
                continue
            path = Path(text)
            if not path.is_absolute():
                continue
            resolved = path.resolve()
            if prefix is not None and resolved != prefix and prefix not in resolved.parents:
                raise ValueError(
                    f"{dataset} manifest contains a non-MN5 dataset path: {resolved}"
                )
            if not resolved.exists():
                raise FileNotFoundError(f"{dataset} manifest path is missing: {resolved}")
            checked_paths.add(str(resolved))
    if not checked_paths:
        raise ValueError(f"No source paths were verified for {dataset}: {manifest_path}")
    return {
        "dataset": dataset,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "metadata_path": str(metadata_path),
        "split_metadata_sha256": sha256_file(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_hash": metadata.get("manifest_hash"),
        "rows": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "verified_source_paths": len(checked_paths),
    }


def prepare(
    *, run_id: str, build: bool, required_path_prefix: Path | None,
    build_merged: bool = True, pooled_turkish: bool = False,
    pooled_source_root: Path | None = None,
) -> dict[str, Any]:
    component_paths = [
        resolve_project_path(path)
        for path in (pooled_component_configs() if pooled_turkish else COMPONENT_CONFIGS)
    ]
    pooled_component_path = resolve_project_path(POOLED_TURKISH_COMPONENT_CONFIG)
    pooled: dict[str, Any] = {}
    if pooled_turkish and build:
        if pooled_source_root is None:
            raise ValueError("pooled_turkish builds require a pooled source root")
        print(f"Rebuilding pooled Turkish manifest from {pooled_source_root}", flush=True)
        pooled = build_pooled_turkish(pooled_source_root)
    if build:
        for config_path in component_paths:
            if pooled_turkish and config_path == pooled_component_path:
                continue
            print(f"Building MN5 harmonized component: {config_path}", flush=True)
            build_for_config(config_path, [])
    elif pooled_turkish:
        pooled = {"outputs": verify_pooled_turkish_outputs()}
    components = [
        validate_component(path, required_path_prefix=required_path_prefix)
        for path in component_paths
    ]

    merged_configs = POOLED_MERGED_CONFIGS if pooled_turkish else MERGED_CONFIGS
    merged: list[dict[str, Any]] = []
    if build_merged:
        for raw_path in merged_configs:
            config_path = resolve_project_path(raw_path)
            config = load_yaml_with_overrides(config_path, [])
            records = load_component_records(config, require_files=True)
            output_dir = resolve_project_path(config["output_dirs"]["merged_root"])
            payload = save_protocol_artifacts(
                config,
                records,
                output_dir,
                seed=int(config.get("seed", 1337)),
                inner_val_ratio=float(config["protocol_settings"]["inner_val_ratio"]),
            )
            if payload["split_audit"].get("status") != "passed":
                raise ValueError(f"Merged protocol audit failed for {config_path}")
            merged.append(
                {
                    "modality": config["modality"],
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "manifest_path": payload["manifest_path"],
                    "manifest_file_sha256": payload["manifest_file_sha256"],
                    "manifest_hash": payload["manifest"]["manifest_hash"],
                    "split_hash": payload["protocol"]["split_hash"],
                    "artifact_hash": payload["artifact_hash"],
                }
            )

    return {
        "schema_version": "harmonized_mn5_preflight.v1",
        "status": "passed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("HARMONIZED_SOURCE_COMMIT"),
        "source_branch": os.environ.get("HARMONIZED_SOURCE_BRANCH"),
        "required_path_prefix": str(required_path_prefix) if required_path_prefix else None,
        "components": components,
        "merged": merged,
        "pooled_turkish": pooled or None,
        "optuna_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-merged", action="store_true")
    parser.add_argument("--required-path-prefix", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument(
        "--pooled-turkish",
        action="store_true",
        help="use the pooled question-conditioned Turkish input and the pooled merged family",
    )
    parser.add_argument(
        "--pooled-source-root",
        type=Path,
        help="directory holding manifests/{pos,neg}_{native,english} and splits/ for the pooled rebuild",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = prepare(
        run_id=args.run_id,
        build=not args.validate_only,
        required_path_prefix=args.required_path_prefix,
        build_merged=not args.skip_merged,
        pooled_turkish=args.pooled_turkish,
        pooled_source_root=args.pooled_source_root,
    )
    audit_path = resolve_project_path(
        args.audit_path
        or PROJECT_ROOT / "outputs/harmonized_mn5_preflight" / args.run_id / "audit.json"
    )
    save_json(audit, audit_path)
    print(json.dumps({"status": "passed", "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
