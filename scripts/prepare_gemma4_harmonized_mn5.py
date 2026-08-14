#!/usr/bin/env python3
"""Model-free Gemma harmonized MN5 preflight.

Runs entirely on CPU in the Gemma environment (no model weights):

1. Builds and validates the shared harmonized manifests (byte-identical to the
   Qwen campaign: same build signature, same manifest/split files).
2. Validates every Gemma config in the selected matrix through
   ``validate_gemma4_config``.
3. For every config, builds fold-0 examples model-free and audits:
   - subject partition equality with the saved split and disjointness;
   - label/class counts and modality flags;
   - one audio window per audio prompt;
   - maximum window length at or below 30 s (480,000 samples at 16 kHz);
   - hierarchical subject/unit/window weights (equal subject totals);
   - transcript condition (native completeness, or English overlay with no
     native fallback for the English matrix);
   - worst-case rendered Gemma prompt length against the pinned model context
     (real Gemma processor, tokenize only);
   - no raw transcript or subject content in the audit payload (hashes only).
4. Records the exact production job scope: native 60 train + 30 eval + 60
   hidden = 150; English 40 + 20 + 40 = 100. Optuna is never enabled here.

The audit must report ``status: passed`` with an empty failure list before
any GPU job is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    build_examples,
    resolve_input_modality,
)
from src.data.androids import apply_hierarchical_training_weights
from src.features.extract_qwen_hidden import _resolve_subject_partitions
from src.model.gemma4_io import (
    GEMMA4_MAX_AUDIO_SAMPLES,
    GEMMA4_MODEL_REVISION,
    prepare_gemma4_examples,
    validate_gemma4_config,
)
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)

NATIVE_MATRIX = "configs/experiments/harmonized/gemma4_standalone_matrix.yaml"
EN_MATRIX = "configs/experiments/harmonized/gemma4_english_translation_matrix.yaml"
GEMMA_MODEL_PATH = (
    "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/"
    "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
)
AUDIO_SAMPLING_RATE = 16000
TOKENIZE_EXAMPLES_PER_CONFIG = 32


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


def validate_manifest(config_path: Path, required_path_prefix: Path | None) -> dict[str, Any]:
    from scripts.prepare_harmonized_mn5 import validate_component

    return validate_component(config_path, required_path_prefix=required_path_prefix)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    keys = ("subject_id", "label", "sample_id", "audio_path", "start_time", "end_time")
    return tuple(str(row.get(key, "")) for key in keys)


def _coverage_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_seconds = 0.0
    for row in rows:
        start = float(row.get("start_time", 0.0) or 0.0)
        end = float(row["end_time"]) if row.get("end_time") not in (None, "") else None
        total_seconds += float(end - start if end is not None else 0.0)
    return {
        "rows": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "total_window_seconds": round(total_seconds, 3),
    }


def _subject_fold_map(metadata: dict[str, Any]) -> dict[str, int]:
    folds = metadata.get("folds") or metadata.get("fold_assignments") or {}
    return {str(subject): int(fold) for subject, fold in folds.items()}


def equivalence_audit(
    native_meta: dict[str, Any],
    native_rows: list[dict[str, Any]],
    en_meta: dict[str, Any],
    en_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    native_by_id = {str(row["sample_id"]): row for row in native_rows}
    en_by_id = {str(row["sample_id"]): row for row in en_rows}
    if set(native_by_id) != set(en_by_id):
        failures.append("sample_id sets differ between native and English")
        return {"failures": failures, "details": {}}
    native_subjects = {str(row["subject_id"]): row.get("label") for row in native_rows}
    en_subjects = {str(row["subject_id"]): row.get("label") for row in en_rows}
    if native_subjects != en_subjects:
        failures.append("subject set or per-subject labels differ")
    if len(native_rows) != len(en_rows):
        failures.append("row count differs between native and English")
    for sample_id, native_row in native_by_id.items():
        if _row_identity(native_row) != _row_identity(en_by_id[sample_id]):
            failures.append(f"row identity differs for {sample_id}")
            break
    if _coverage_totals(native_rows) != _coverage_totals(en_rows):
        failures.append("coverage totals differ between native and English")
    if _subject_fold_map(native_meta) != _subject_fold_map(en_meta):
        failures.append("per-subject fold assignment differs")
    transcript_identical = 0
    for sample_id, native_row in native_by_id.items():
        en_row = en_by_id[sample_id]
        if str(en_row.get("language", "")) != "en":
            failures.append(f"row without language=en: {sample_id}")
            break
        if str(en_row.get("transcript_variant", "")) != "english":
            failures.append(f"row without transcript_variant=english: {sample_id}")
            break
        if str(native_row.get("transcript", "")) == str(en_row.get("transcript", "")):
            transcript_identical += 1
    if transcript_identical:
        failures.append(
            f"{transcript_identical} rows kept byte-identical transcript text "
            "(native fallback suspicion)"
        )
    return {
        "failures": failures,
        "details": {
            "subjects": len(en_subjects),
            "rows": len(en_rows),
            "native_manifest_hash": native_meta.get("manifest_hash"),
            "en_manifest_hash": en_meta.get("manifest_hash"),
            "transcript_identical_rows": transcript_identical,
        },
    }


def _fold0_subjects(split_metadata: dict[str, Any]) -> dict[str, list[str]]:
    fold_payload = split_metadata.get("0", split_metadata.get(0))
    if fold_payload is None:
        raise ValueError("Split metadata has no fold-0 payload.")
    return {
        "outer_train": [str(s) for s in fold_payload.get("outer_train_subject_ids", [])],
        "final_eval": [str(s) for s in fold_payload.get("final_eval_subject_ids", [])],
    }


def check_gemma_config(
    config_path: Path,
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    native_meta: dict[str, Any] | None,
    native_rows: list[dict[str, Any]] | None,
    processor,
    context_limit: int,
    required_path_prefix: Path | None,
    english: bool,
) -> dict[str, Any]:
    config = load_yaml_with_overrides(config_path, [])
    validate_gemma4_config(config)
    failures: list[str] = []
    dataset = str(config["dataset"]).lower()
    modality = resolve_input_modality(config)
    use_audio = bool(config["data"].get("use_audio", False))
    use_text = bool(config["data"].get("use_text", False))

    if not rows:
        failures.append("empty manifest rows")
        return {"config": str(config_path), "failures": failures, "details": {}}

    if english:
        if native_meta is None or native_rows is None:
            failures.append("missing native manifest for English equivalence audit")
        else:
            audit = equivalence_audit(native_meta, native_rows, metadata, rows)
            failures.extend(audit["failures"])
    else:
        if use_text:
            empty_transcripts = sum(1 for row in rows if not str(row.get("transcript", "")).strip())
            if empty_transcripts:
                failures.append(f"{empty_transcripts} rows have empty transcripts")

    split_metadata_path = resolve_project_path(
        metadata.get("split_metadata_path") or metadata.get("folds_path")
    )
    split_metadata_hash = metadata.get("split_metadata_sha256") or metadata.get("fold_hash")
    if not split_metadata_path.is_file():
        failures.append(f"missing split metadata: {split_metadata_path}")
        split_metadata: dict[str, Any] = {}
    else:
        split_metadata = read_json(split_metadata_path)
        if split_metadata_hash and sha256_file(split_metadata_path) != str(split_metadata_hash):
            failures.append("split metadata hash mismatch")

    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    dataset_meta_path = split_dir / f"{dataset}_manifest_metadata.json"

    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[str(row.get("label", ""))] = class_counts.get(str(row.get("label", "")), 0) + 1

    examples: list[dict[str, Any]] = []
    try:
        if split_metadata:
            fold0 = _fold0_subjects(split_metadata)
            for partition, subject_ids in fold0.items():
                partition_rows = [
                    row for row in rows if str(row["subject_id"]) in set(subject_ids)
                ]
                partition_examples = build_examples(
                    partition_rows, config, partition_name=partition, truncation_log_path=None
                )
                example_subjects = {str(item["subject_id"]) for item in partition_examples}
                if example_subjects != set(subject_ids):
                    failures.append(
                        f"{partition} example subjects differ from saved split "
                        f"(missing={sorted(set(subject_ids) - example_subjects)[:5]})"
                    )
                examples.extend(partition_examples)
    except Exception as error:  # noqa: BLE001
        failures.append(f"example build failed: {error}")

    one_window_violations = 0
    max_window = 0.0
    for example in examples:
        if use_audio:
            placeholders = str(example.get("prompt_text", "")).count(AUDIO_PLACEHOLDER)
            if placeholders != 1:
                one_window_violations += 1
            start = float(example.get("start_time", 0.0) or 0.0)
            end = example.get("end_time")
            if end not in (None, ""):
                max_window = max(max_window, float(end) - start)
    if one_window_violations:
        failures.append(f"{one_window_violations} audio examples without exactly one window")
    if max_window > 30.0:
        failures.append(
            f"max window {max_window}s exceeds the 30 s / {GEMMA4_MAX_AUDIO_SAMPLES}-sample contract"
        )

    weight_audit: dict[str, Any] = {}
    try:
        train_examples = [item for item in examples if item.get("partition") == "outer_train"]
        if train_examples and use_text:
            weighted, weight_audit = apply_hierarchical_training_weights(train_examples)
            totals: dict[str, float] = {}
            for item in weighted:
                totals[str(item["subject_id"])] = totals.get(str(item["subject_id"]), 0.0) + float(
                    item["raw_loss_weight"]
                )
            spread = max(totals.values()) - min(totals.values()) if totals else 0.0
            if spread > 1e-6:
                failures.append(f"hierarchical weights do not equalize subjects (spread={spread:.6f})")
    except Exception as error:  # noqa: BLE001
        failures.append(f"weight audit failed: {error}")

    tokenizer_stats: dict[str, Any] = {"checked_examples": 0, "max_tokens": 0}
    try:
        prepared = prepare_gemma4_examples(examples, config, processor)
        prepared.sort(key=lambda item: len(item["training_text"]), reverse=True)
        checked = prepared[:TOKENIZE_EXAMPLES_PER_CONFIG]
        for example in checked:
            token_count = len(processor.tokenizer.encode(example["training_text"]))
            tokenizer_stats["checked_examples"] += 1
            tokenizer_stats["max_tokens"] = max(tokenizer_stats["max_tokens"], int(token_count))
            tokenizer_stats.setdefault("prompt_sha256", _sha256_text(example["prompt_text"]))
        if tokenizer_stats["max_tokens"] > context_limit:
            failures.append(
                f"max rendered prompt {tokenizer_stats['max_tokens']} tokens exceeds "
                f"context limit {context_limit}"
            )
    except Exception as error:  # noqa: BLE001
        failures.append(f"Gemma tokenizer check failed: {error}")

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
                failures.append(f"non-MN5 dataset path: {resolved}")
            if not resolved.exists():
                failures.append(f"missing manifest path: {resolved}")
            checked_paths.add(str(resolved))
    if not checked_paths:
        failures.append("no source paths were verified")

    details: dict[str, Any] = {
        "dataset": dataset,
        "modality": modality,
        "manifest_hash": metadata.get("manifest_hash"),
        "manifest_file_sha256": metadata.get("manifest_file_sha256"),
        "split_metadata_sha256": metadata.get("split_metadata_sha256"),
        "rows": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "class_counts": class_counts,
        "example_count_fold0": len(examples),
        "max_window_seconds": max_window,
        "weight_audit": {
            "unit_totals": (weight_audit or {}).get("raw_source_unit_weight_totals"),
            "subject_totals_ok": not any("weights" in failure for failure in failures),
        },
        "tokenizer": tokenizer_stats,
        "model_revision": GEMMA4_MODEL_REVISION,
        "verified_source_paths": len(checked_paths),
    }
    return {"config": str(config_path), "failures": failures, "details": details}


def _load_manifest(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    metadata = read_json(metadata_path)
    manifest_path = resolve_project_path(metadata["manifest_path"])
    return metadata, read_jsonl(manifest_path)


def prepare(
    *,
    run_id: str,
    build: bool,
    required_path_prefix: Path | None,
    english: bool,
    model_path: str,
) -> dict[str, Any]:
    from scripts.prepare_harmonized_mn5 import COMPONENT_CONFIGS, MERGED_CONFIGS

    failures: list[str] = []
    matrix_path = resolve_project_path(EN_MATRIX if english else NATIVE_MATRIX)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing Gemma matrix: {matrix_path}")
    matrix = load_yaml_with_overrides(matrix_path, [])

    if english:
        config_names = list(dict.fromkeys(item["config"] for item in matrix["experiments"]))
        build_paths = [resolve_project_path(path) for path in config_names]
    else:
        config_names = [item["config"] for item in matrix["experiments"]]
        build_paths = [resolve_project_path(path) for path in COMPONENT_CONFIGS]

    if build:
        for config_path in build_paths:
            print(f"Building MN5 Gemma harmonized component: {config_path}", flush=True)
            from src.data.build_manifest import build_for_config

            build_for_config(config_path, [])

    components = [
        validate_manifest(path, required_path_prefix=required_path_prefix)
        for path in COMPONENT_CONFIGS
    ]

    merged: list[dict[str, Any]] = []
    if not english:
        try:
            for raw_path in MERGED_CONFIGS:
                config_path = resolve_project_path(raw_path)
                config = load_yaml_with_overrides(config_path, [])
                from src.merged.protocol import (
                    load_component_records,
                    save_protocol_artifacts,
                )

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
                    failures.append(f"merged protocol audit failed for {config_path}")
                merged.append(
                    {
                        "modality": config["modality"],
                        "config": str(config_path),
                        "manifest_file_sha256": payload["manifest_file_sha256"],
                        "split_hash": payload["protocol"]["split_hash"],
                        "artifact_hash": payload["artifact_hash"],
                    }
                )
        except Exception as error:  # noqa: BLE001
            failures.append(f"merged protocol preparation failed: {error}")

    processor = None
    context_limit = 0
    try:
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        text_config = getattr(config, "text_config", None)
        context_limit = int(
            getattr(text_config, "max_position_embeddings", 0)
            or getattr(config, "max_position_embeddings", 0)
            or 0
        )
        if context_limit <= 0:
            raise ValueError("model config declares no max_position_embeddings")
    except Exception as error:  # noqa: BLE001
        failures.append(f"Gemma processor/context resolution failed: {error}")

    gemma_checks: list[dict[str, Any]] = []
    for config_name in config_names:
        config_path = resolve_project_path(config_name)
        config = load_yaml_with_overrides(config_path, [])
        dataset = str(config["dataset"]).lower()
        try:
            metadata, rows = _load_manifest(config_path)
        except Exception as error:  # noqa: BLE001
            gemma_checks.append(
                {
                    "config": str(config_path),
                    "failures": [f"manifest/metadata resolution failed: {error}"],
                    "details": {"dataset": dataset},
                }
            )
            failures.extend(gemma_checks[-1]["failures"])
            continue
        native_meta = native_rows = None
        if english:
            try:
                native_config_path = None
                for raw_path in COMPONENT_CONFIGS:
                    candidate = load_yaml_with_overrides(resolve_project_path(raw_path), [])
                    if str(candidate["dataset"]).lower() == dataset:
                        native_config_path = resolve_project_path(raw_path)
                        break
                if native_config_path is None:
                    raise ValueError(f"no native component config for {dataset}")
                native_meta, native_rows = _load_manifest(native_config_path)
            except Exception as error:  # noqa: BLE001
                failures.append(f"native manifest resolution failed for {dataset}: {error}")
        gemma_checks.append(
            check_gemma_config(
                config_path,
                metadata=metadata,
                rows=rows,
                native_meta=native_meta,
                native_rows=native_rows,
                processor=processor,
                context_limit=context_limit,
                required_path_prefix=required_path_prefix,
                english=english,
            )
        )
        failures.extend(gemma_checks[-1]["failures"])

    train_folds = sum(len(item["folds"]) for item in matrix["experiments"])
    eval_folds = sum(
        len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"]
    )
    job_scope = {
        "train_jobs": train_folds,
        "separate_eval_jobs": eval_folds,
        "hidden_jobs": train_folds,
        "total_jobs": train_folds * 2 + eval_folds,
    }
    expected_scope = {
        "train_jobs": 60 if not english else 40,
        "separate_eval_jobs": 30 if not english else 20,
        "hidden_jobs": 60 if not english else 40,
        "total_jobs": 150 if not english else 100,
    }
    if job_scope != expected_scope:
        failures.append(f"job scope mismatch: {job_scope} != {expected_scope}")

    return {
        "schema_version": "gemma4_harmonized_mn5_preflight.v1",
        "status": "passed" if not failures else "failed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("HARMONIZED_SOURCE_COMMIT"),
        "source_branch": os.environ.get("HARMONIZED_SOURCE_BRANCH"),
        "english": english,
        "required_path_prefix": str(required_path_prefix) if required_path_prefix else None,
        "components": components,
        "merged": merged,
        "gemma_checks": gemma_checks,
        "job_scope": job_scope,
        "optuna_enabled": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--english", action="store_true")
    parser.add_argument("--required-path-prefix", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("GEMMA4_MODEL_PATH", GEMMA_MODEL_PATH),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = prepare(
        run_id=args.run_id,
        build=not args.validate_only,
        required_path_prefix=args.required_path_prefix,
        english=args.english,
        model_path=args.model_path,
    )
    audit_path = resolve_project_path(
        args.audit_path
        or PROJECT_ROOT
        / ("outputs/gemma4_en_mn5_preflight" if args.english else "outputs/gemma4_mn5_preflight")
        / args.run_id
        / "audit.json"
    )
    save_json(audit, audit_path)
    print(json.dumps({"status": audit["status"], "audit": str(audit_path)}, indent=2))
    if audit["failures"]:
        print("\n".join(f"FAILURE: {failure}" for failure in audit["failures"][:40]), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
