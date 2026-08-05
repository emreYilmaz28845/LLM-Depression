from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cmdc import build_cmdc_manifest
from src.data.androids import build_androids_interview_manifest
from src.data.daic import build_daic_manifest
from src.data.d3tec import build_d3tec_manifest
from src.data.edaic import build_edaic_manifest
from src.data.eatd import build_eatd_manifest
from src.data.turkish import build_turkish_manifest
from src.data.validation import (
    assert_audio_exists,
    assert_clean_labels,
    assert_transcripts,
    load_quarantine,
    print_class_counts,
    print_random_rows,
)
from src.translation.overlay import apply_overlay
from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    load_yaml_with_overrides,
    log_resolved_config,
    read_jsonl,
    save_json,
    sha256_file,
    sha256_jsonl_rows,
    resolve_project_path,
    write_jsonl,
)
from src.utils import serialize_project_path


LOGGER = get_logger(__name__)


def manifest_build_signature(config: dict[str, Any]) -> dict[str, Any]:
    excluded_top_level = {
        "audio_adapter",
        "data",
        "evaluation",
        "joint_train",
        "labels",
        "lora",
        "model_name_or_path",
        "output_dirs",
        "prompt",
        "quarantine_path",
        "training",
    }
    if str((config.get("transcripts") or {}).get("variant", "original")).strip().lower() == "original":
        excluded_top_level.add("transcripts")
    builder_options = {
        key: value
        for key, value in config.items()
        if key not in excluded_top_level and key != "split"
    }
    data_options = {
        key: value
        for key, value in (config.get("data") or {}).items()
        if key in {"segment_seconds", "segment_partition"}
    }
    split_options = {
        key: value
        for key, value in (config.get("split") or {}).items()
        if key not in {"cv_protocol", "fixed_protocol", "smoke_subject_limit"}
    }
    return {
        "builder_options": builder_options,
        "data_options": data_options,
        "split_options": split_options,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_common_outputs(config: dict[str, Any], manifest_rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Path]:
    output_dir = ensure_dir(config["output_dirs"]["manifest_dir"])
    manifest_jsonl = output_dir / f"{dataset_name}_manifest.jsonl"
    manifest_csv = output_dir / f"{dataset_name}_manifest.csv"
    write_jsonl(manifest_rows, manifest_jsonl)
    _write_csv(manifest_rows, manifest_csv)
    return {
        "manifest_jsonl": manifest_jsonl,
        "manifest_csv": manifest_csv,
    }


def _apply_transcript_overlay(
    config: dict[str, Any], dataset_name: str, manifest_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    transcripts_cfg = config.get("transcripts") or {}
    variant = str(transcripts_cfg.get("variant", "original")).strip().lower()
    if variant == "original":
        return manifest_rows, None
    if variant != "english":
        raise ValueError(
            f"Unsupported transcripts.variant={variant!r}. "
            f"Expected 'original' or 'english'."
        )
    cache_path = resolve_project_path(transcripts_cfg["cache_path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"Translation accepted cache not found: {cache_path}")
    accepted_rows = read_jsonl(cache_path)
    minimum_status = str(transcripts_cfg.get("minimum_status", "automatic_high"))
    require_complete = bool(transcripts_cfg.get("require_complete", True))
    include_failed = bool(transcripts_cfg.get("include_failed", False))
    rejected_rows: list[dict[str, Any]] = []
    if include_failed:
        rejected_path = resolve_project_path(transcripts_cfg["rejected_path"])
        if not rejected_path.is_file():
            raise FileNotFoundError(
                f"transcripts.include_failed requires transcripts.rejected_path; "
                f"not found: {rejected_path}"
            )
        rejected_rows = read_jsonl(rejected_path)
    LOGGER.info(
        "Applying English transcript overlay: dataset=%s cache=%s minimum_status=%s "
        "require_complete=%s include_failed=%s",
        dataset_name,
        cache_path,
        minimum_status,
        require_complete,
        include_failed,
    )
    overlaid_rows, audit = apply_overlay(
        manifest_rows,
        dataset_name,
        accepted_rows,
        minimum_status=minimum_status,
        require_complete=require_complete,
        include_failed=include_failed,
        rejected_rows=rejected_rows,
    )
    audit["accepted_cache_path"] = serialize_project_path(cache_path)
    audit["accepted_cache_sha256"] = sha256_file(cache_path)
    audit["accepted_cache_record_count"] = len(accepted_rows)
    if include_failed:
        audit["rejected_cache_path"] = serialize_project_path(
            resolve_project_path(transcripts_cfg["rejected_path"])
        )
        audit["rejected_cache_record_count"] = len(rejected_rows)
    return overlaid_rows, audit


def build_for_config(config_path: str | Path, config_overrides: list[str] | None = None) -> None:
    config = load_yaml_with_overrides(config_path, config_overrides)
    log_resolved_config(
        LOGGER,
        base_config_path=config_path,
        config_overrides=config_overrides,
        resolved_config=config,
    )
    dataset_name = str(config["dataset"]).lower()
    quarantine = load_quarantine(config.get("quarantine_path"))

    if dataset_name == "androids_interview":
        result = build_androids_interview_manifest(config, quarantine)
    elif dataset_name == "daic":
        result = build_daic_manifest(config, quarantine)
    elif dataset_name == "d3tec":
        result = build_d3tec_manifest(config, quarantine)
    elif dataset_name == "edaic":
        result = build_edaic_manifest(config, quarantine)
    elif dataset_name == "cmdc":
        result = build_cmdc_manifest(config, quarantine)
    elif dataset_name == "eatd":
        result = build_eatd_manifest(config, quarantine)
    elif dataset_name == "turkish":
        result = build_turkish_manifest(config, quarantine)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    manifest_rows, overlay_audit = _apply_transcript_overlay(config, dataset_name, result["manifest_rows"])
    assert_clean_labels(manifest_rows)
    assert_audio_exists(manifest_rows)
    assert_transcripts(manifest_rows)
    print_class_counts(manifest_rows, dataset_name, partition_field="split_original")
    print_random_rows(manifest_rows, dataset_name, seed=int(config["seed"]), limit=10)
    output_paths = _save_common_outputs(config, manifest_rows, dataset_name)

    split_dir = ensure_dir(config["output_dirs"]["split_dir"])
    manifest_hash = sha256_jsonl_rows(manifest_rows)
    metadata = {
        "dataset": dataset_name,
        "manifest_path": serialize_project_path(output_paths["manifest_jsonl"]),
        "manifest_hash": manifest_hash,
        "build_signature": manifest_build_signature(config),
        "manifest_row_count": len(manifest_rows),
        "manifest_subject_count": len({str(row["subject_id"]) for row in manifest_rows}),
    }
    if dataset_name == "daic":
        subject_rows_by_id: dict[str, list[dict[str, Any]]] = {}
        for row in manifest_rows:
            subject_rows_by_id.setdefault(str(row["subject_id"]), []).append(row)
        metadata["manifest_subject_counts_by_label"] = {
            str(label): sum(
                1
                for subject_rows in subject_rows_by_id.values()
                if int(subject_rows[0]["label"]) == label
            )
            for label in (0, 1)
        }
        metadata["manifest_chunk_counts_by_label"] = {
            str(label): sorted(
                {
                    len(subject_rows)
                    for subject_rows in subject_rows_by_id.values()
                    if int(subject_rows[0]["label"]) == label
                }
            )
            for label in (0, 1)
        }
        metadata["split_seed"] = int(config.get("split", {}).get("seed", 1337))
        metadata["fold_count"] = len(result.get("folds", {})) if result.get("folds") else 0
    if "subject_partition_rows" in result:
        partition_path = split_dir / f"{dataset_name}_subject_partitions.json"
        save_json(result["subject_partition_rows"], partition_path)
        metadata["subject_partition_path"] = serialize_project_path(partition_path)
    if "join_audit_rows" in result:
        join_audit_path = split_dir / f"{dataset_name}_join_audit.csv"
        _write_csv(result["join_audit_rows"], join_audit_path)
        metadata["join_audit_path"] = serialize_project_path(join_audit_path)
    if "folds" in result:
        folds_path = split_dir / f"{dataset_name}_folds.json"
        save_json(result["folds"], folds_path)
        metadata["folds_path"] = serialize_project_path(folds_path)
    if "fold_report" in result:
        fold_report_path = split_dir / f"{dataset_name}_fold_report.json"
        save_json(result["fold_report"], fold_report_path)
        metadata["fold_report_path"] = serialize_project_path(fold_report_path)
    if "workbook_fold_report" in result:
        workbook_fold_report_path = split_dir / f"{dataset_name}_workbook_fold_report.json"
        save_json(result["workbook_fold_report"], workbook_fold_report_path)
        metadata["workbook_fold_report_path"] = serialize_project_path(workbook_fold_report_path)
    if "split_source" in result:
        metadata["split_source"] = result["split_source"]
        metadata["split_source_notes"] = result.get("split_source_notes", "")
    if "subject_rows" in result:
        subject_rows_path = split_dir / f"{dataset_name}_subjects.json"
        save_json(result["subject_rows"], subject_rows_path)
        metadata["subject_rows_path"] = serialize_project_path(subject_rows_path)
    if "extra_file_audit" in result and result["extra_file_audit"]:
        extra_path = split_dir / f"{dataset_name}_extra_file_audit.json"
        save_json(result["extra_file_audit"], extra_path)
        metadata["extra_file_audit_path"] = serialize_project_path(extra_path)
    for result_key, filename in (
        ("chunk_window_audit", f"{dataset_name}_chunk_window_audit.json"),
        ("label_source_audit", f"{dataset_name}_label_source_audit.json"),
    ):
        if result_key in result:
            artifact_path = split_dir / filename
            save_json(result[result_key], artifact_path)
            metadata[f"{result_key}_path"] = serialize_project_path(artifact_path)
    if "fold_hash" in result:
        metadata["fold_hash"] = result["fold_hash"]
    if "transcript_paths" in result:
        metadata["transcript_paths"] = result["transcript_paths"]
    if "source_hashes" in result:
        metadata["source_hashes"] = result["source_hashes"]
    if overlay_audit is not None:
        metadata["transcript_overlay"] = overlay_audit

    metadata["artifact_hashes"] = {
        f"{key}_sha256": sha256_file(resolve_project_path(value))
        for key, value in metadata.items()
        if key.endswith("_path")
        and isinstance(value, str)
        and value
        and resolve_project_path(value).is_file()
    }

    metadata_path = split_dir / f"{dataset_name}_manifest_metadata.json"
    save_json(metadata, metadata_path)
    LOGGER.info("Saved %s manifest to %s", dataset_name, output_paths["manifest_jsonl"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifests and split metadata for the depression pipeline.")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more dataset config paths.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE, using dot paths for nested keys.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    for config_path in args.config:
        build_for_config(config_path, args.config_overrides)


if __name__ == "__main__":
    main()
