#!/usr/bin/env python3
"""Build and verify MN5-native harmonized English manifests, translation
completeness, native/English input equivalence, tokenizer/context fit, and
the exact 100-job production scope.

CPU-only: loads the processor/tokenizer for the context audit, never the model.
Writes a machine-readable audit; GPU submission must wait for status "passed"
with an empty failure list.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.data.runtime import _harmonized_subject_transcripts, qwen2audio_audio_token_length
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    save_json,
    sha256_file,
)

EN_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1"

# One audio_text config per dataset builds the shared manifest.
BUILD_CONFIGS = (
    "configs/main/d3tec_audio_text_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/androids_audio_text_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/cmdc_audio_text_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/turkish_pos_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
)
NATIVE_CONFIGS = (
    "configs/main/d3tec_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/androids_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/cmdc_audio_text_harmonized_selmacrof1_tf.yaml",
    "configs/main/turkish_pos_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml",
)
ALL_EN_CONFIGS = BUILD_CONFIGS + (
    "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/androids_text_only_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en.yaml",
    "configs/main/turkish_pos_only_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
)
MATRIX = "configs/experiments/harmonized/english_translation_matrix.yaml"

EXPECTED_ACCEPTED = {"d3tec": 3677, "androids_interview": 2176, "cmdc": 923, "turkish": 1051}
CACHE_DATASET = {"d3tec": "d3tec", "androids_interview": "androids_interview", "cmdc": "cmdc", "turkish": "turkish"}
FULL_SCOPE_FIELD = {
    "d3tec": "full_response_transcript",
    "androids_interview": "full_turn_transcript",
    "cmdc": "transcript",
    "turkish": "transcript",
}
NATURAL_KEY_RE = re.compile(r"(\d+)")

def natural_key(value: Any) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in NATURAL_KEY_RE.split(str(value)))


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


def validate_component(config_path: Path, *, required_path_prefix: Path | None) -> dict[str, Any]:
    config = load_yaml_with_overrides(config_path, [])
    dataset = str(config["dataset"]).lower()
    split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
    metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing harmonized English metadata for {dataset}: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("build_signature") != manifest_build_signature(config):
        raise ValueError(f"Stale build signature for {dataset}: {metadata_path}")
    manifest_path = resolve_project_path(metadata["manifest_path"])
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Empty harmonized English manifest for {dataset}: {manifest_path}")
    if int(metadata.get("manifest_row_count", -1)) != len(rows):
        raise ValueError(f"Manifest row-count mismatch for {dataset}: {manifest_path}")

    overlay = (metadata.get("transcript_overlay") or {})
    for key in ("missing_units", "below_status_units", "native_rows_kept"):
        if overlay.get(key):
            raise ValueError(f"Overlay {key} not empty for {dataset}: {overlay[key]}")
    if overlay.get("failed_included_rows"):
        raise ValueError(f"Overlay included failed rows for {dataset}: {overlay['failed_included_rows']}")
    if overlay.get("variant") != "english":
        raise ValueError(f"Overlay variant not english for {dataset}")
    if not overlay.get("accepted_cache_sha256"):
        raise ValueError(f"Overlay cache hash missing for {dataset}")

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
                raise ValueError(f"{dataset} manifest contains a non-MN5 dataset path: {resolved}")
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


def audit_translation_cache(dataset: str) -> dict[str, Any]:
    root = Path(os.environ.get(
        "TRANSLATION_ROOT",
        "/gpfs/projects/etur92/ozu647717/AudioLLM/translations",
    ))
    cache_dir = root / "harmonized_en_complete_v1" / CACHE_DATASET[dataset]
    expected = EXPECTED_ACCEPTED[dataset]
    failures: list[str] = []
    hashes: dict[str, str] = {}
    for name in ("units.jsonl", "candidates.jsonl", "accepted.jsonl", "rejected.jsonl", "audit.json", "repair_provenance.json"):
        path = cache_dir / name
        if not path.is_file():
            failures.append(f"translation cache file missing: {path}")
        else:
            hashes[name] = sha256_file(path)
    accepted = read_jsonl(cache_dir / "accepted.jsonl") if (cache_dir / "accepted.jsonl").is_file() else []
    rejected = read_jsonl(cache_dir / "rejected.jsonl") if (cache_dir / "rejected.jsonl").is_file() else []
    if len(accepted) != expected:
        failures.append(f"translation cache accepted count {len(accepted)} != expected {expected}")
    if rejected:
        failures.append(f"translation cache rejected rows: {len(rejected)}")
    statuses = {str(row.get("status", "")) for row in accepted}
    if statuses - {"automatic_high", "automatic_medium", "automatic_low", "human_verified"}:
        failures.append(f"unexpected accepted statuses: {sorted(statuses)}")
    return {
        "dataset": dataset,
        "cache_root": str(cache_dir),
        "expected_accepted": expected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "statuses": sorted(statuses),
        "file_hashes": hashes,
        "failures": failures,
    }


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    dataset = str(row["dataset"]).lower()
    base: list[Any] = [
        str(row.get("sample_id", "")),
        str(row.get("audio_path", "")),
        str(row.get("start_time", "")),
        str(row.get("end_time", "")),
    ]
    if dataset in ("d3tec", "androids_interview"):
        base.append(str(row.get("response_id", "")))
    if dataset == "androids_interview":
        base.extend([str(row.get("recording_id", "")), str(row.get("turn_id", ""))])
    if dataset in ("cmdc", "turkish"):
        base.append(str(row.get("question_id", "")))
    return tuple(base)


def _coverage_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["segment_duration"]) for row in rows
                 if row.get("segment_duration") not in (None, "") and _is_number(row.get("segment_duration"))]
    spans = []
    for row in rows:
        start, end = row.get("start_time"), row.get("end_time")
        if _is_number(start) and _is_number(end):
            spans.append(max(0.0, float(end) - float(start)))
    return {
        "rows": len(rows),
        "segment_duration_sum": round(sum(durations), 6),
        "window_span_sum": round(sum(spans), 6),
        "audio_paths": len({str(row.get("audio_path", "")) for row in rows}),
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _subject_fold_map(metadata: dict[str, Any]) -> dict[str, int]:
    folds_path = resolve_project_path(metadata["folds_path"])
    folds = read_json(folds_path)
    mapping: dict[str, int] = {}
    for fold, partitions in folds.items():
        for partition, subjects in partitions.items():
            for subject in subjects:
                mapping[str(subject)] = int(fold)
    return mapping


def equivalence_audit(native_meta: dict[str, Any], native_rows: list[dict[str, Any]],
                      en_meta: dict[str, Any], en_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    dataset = str(en_rows[0]["dataset"]).lower() if en_rows else ""
    native_by_id = {str(row.get("sample_id", "")): row for row in native_rows}
    en_by_id = {str(row.get("sample_id", "")): row for row in en_rows}
    if set(native_by_id) != set(en_by_id):
        failures.append(f"sample_id sets differ (native {len(native_by_id)} vs en {len(en_by_id)})")
        return {"failures": failures, "details": {}}

    native_subjects = {str(row["subject_id"]): row.get("label") for row in native_rows}
    en_subjects = {str(row["subject_id"]): row.get("label") for row in en_rows}
    if native_subjects != en_subjects:
        failures.append("subject set or per-subject labels differ")
    if len(native_rows) != len(en_rows):
        failures.append(f"row count differs: native {len(native_rows)} vs en {len(en_rows)}")
    if native_meta.get("manifest_row_count") != en_meta.get("manifest_row_count"):
        failures.append("manifest_row_count differs between native and English metadata")

    for sample_id, native_row in native_by_id.items():
        en_row = en_by_id[sample_id]
        if _row_identity(native_row) != _row_identity(en_row):
            failures.append(f"row identity differs for {sample_id}")
            break
    native_coverage = _coverage_totals(native_rows)
    en_coverage = _coverage_totals(en_rows)
    if native_coverage != en_coverage:
        failures.append(f"coverage totals differ: {native_coverage} vs {en_coverage}")

    native_folds = _subject_fold_map(native_meta)
    en_folds = _subject_fold_map(en_meta)
    if native_folds != en_folds:
        failures.append("per-subject fold assignment differs")
    if native_meta.get("fold_hash") != en_meta.get("fold_hash"):
        failures.append("fold file hash differs between native and English")

    if native_meta.get("build_signature", {}).get("split_options") != en_meta.get("build_signature", {}).get("split_options"):
        failures.append("split protocol options differ between native and English")

    fallback_rows = 0
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
        if not str(en_row.get("translation_sha256", "")):
            fallback_rows += 1
    if transcript_identical:
        failures.append(f"{transcript_identical} rows kept byte-identical transcript text (native fallback suspicion)")
    if fallback_rows:
        failures.append(f"{fallback_rows} rows lack translation_sha256 (native fallback)")

    details = {
        "dataset": dataset,
        "subjects": len(en_subjects),
        "rows": len(en_rows),
        "native_manifest_hash": native_meta.get("manifest_hash"),
        "en_manifest_hash": en_meta.get("manifest_hash"),
        "native_fold_hash": native_meta.get("fold_hash"),
        "en_fold_hash": en_meta.get("fold_hash"),
        "native_coverage": native_coverage,
        "en_coverage": en_coverage,
        "transcript_identical_rows": transcript_identical,
    }
    return {"failures": failures, "details": details}


def _join_full_subject_transcripts(rows: list[dict[str, Any]]) -> dict[str, str]:
    # The harmonized runtime assembles the model-visible full-subject text by
    # deduplicating natural units (response/turn/question) and joining them in
    # unit order. The context audit must measure exactly that text; joining raw
    # manifest rows would duplicate full-response/turn text across segments.
    dataset = str(rows[0]["dataset"]).lower()
    return _harmonized_subject_transcripts(rows, dataset)


def context_fit(en_rows_by_dataset: dict[str, list[dict[str, Any]]],
                model_path: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoProcessor
    from scripts.audit_full_transcript_context import audit_dataset

    failures: list[str] = []
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = processor.tokenizer
    audio_token_id = int(tokenizer.convert_tokens_to_ids("<|AUDIO|>"))
    if tokenizer.unk_token_id is not None and audio_token_id == int(tokenizer.unk_token_id):
        raise RuntimeError("The selected processor does not recognize Qwen2-Audio's <|AUDIO|> token.")
    context_limit = int(config.text_config.max_position_embeddings)
    audio_tokens = qwen2audio_audio_token_length(3000)

    summaries: list[dict[str, Any]] = []
    for dataset, rows in en_rows_by_dataset.items():
        transcripts = _join_full_subject_transcripts(rows)
        subjects = {str(row["subject_id"]) for row in rows}
        details, summary = audit_dataset(
            dataset,
            transcripts,
            subjects,
            tokenizer,
            audio_token_id,
            audio_tokens,
            context_limit,
            safety_margin=128,
        )
        over_limit_subjects = [
            {
                "subject_id": row["subject_id"],
                "effective_multimodal_tokens": row["effective_multimodal_tokens"],
                "transcript_token_count": row["transcript_token_count"],
            }
            for row in details
            if not row["fits"]
        ]
        summary["over_limit_subjects"] = over_limit_subjects
        if summary["subjects_over_limit"]:
            failures.append(
                f"{dataset}: {summary['subjects_over_limit']} subjects over context limit"
            )
        if summary["audited_subjects"] != len(subjects):
            failures.append(f"{dataset}: audited subjects {summary['audited_subjects']} != {len(subjects)}")
        summaries.append(summary)
    return {
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "context_limit": context_limit,
        "audio_embedding_tokens_30sec": audio_tokens,
        "datasets": summaries,
        "failures": failures,
    }


def recipe_and_scope_audit() -> dict[str, Any]:
    failures: list[str] = []
    matrix_path = resolve_project_path(MATRIX)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("fixed_heads") != ["logreg_raw", "xgb_raw"]:
        failures.append("matrix fixed_heads must be [logreg_raw, xgb_raw]")
    if matrix.get("max_epochs") != 20 or matrix.get("checkpoint_selection") != "inner_val_macro_f1":
        failures.append("matrix recipe fields differ from the plan")
    if matrix.get("optuna") is not False:
        failures.append("matrix optuna must be false")
    allowed_datasets = {"d3tec", "androids_interview", "cmdc", "turkish"}
    train_folds = 0
    eval_folds = 0
    for item in matrix["experiments"]:
        config_path = resolve_project_path(item["config"])
        config = load_yaml_with_overrides(config_path, [])
        dataset = str(config["dataset"]).lower()
        if dataset not in allowed_datasets:
            failures.append(f"disallowed dataset in matrix: {dataset}")
        use_audio = bool(config["data"].get("use_audio"))
        use_text = bool(config["data"].get("use_text"))
        if not (use_audio and use_text) and not (use_text and not use_audio):
            failures.append(f"audio-only or invalid modality in matrix: {config_path}")
        if config.get("recipe_id") != EN_RECIPE:
            failures.append(f"wrong recipe_id: {config_path}")
        if int(config["training"]["num_train_epochs"]) != 20:
            failures.append(f"expected 20 epochs: {config_path}")
        if config["training"]["selection_metric"] != "inner_val_macro_f1" or config["training"]["selection_metric_mode"] != "max":
            failures.append(f"expected macro-F1 max selection: {config_path}")
        if config["training"]["early_stopping"]["patience"] != 3:
            failures.append(f"expected patience 3: {config_path}")
        if config["evaluation"]["sample_prediction_mode"] != "original_teacher_forced" or config["evaluation"]["headline_mode"] != "original_teacher_forced":
            failures.append(f"expected teacher-forced evaluation: {config_path}")
        audio_adapter = config.get("audio_adapter") or {}
        if audio_adapter.get("enabled") or audio_adapter.get("train_projector"):
            failures.append(f"audio encoder not frozen: {config_path}")
        transcripts = config.get("transcripts") or {}
        if (transcripts.get("variant"), transcripts.get("minimum_status"),
                transcripts.get("require_complete"), transcripts.get("include_failed")) != (
                "english", "automatic_low", True, False):
            failures.append(f"transcripts policy differs: {config_path}")
        if "optuna" in config:
            failures.append(f"optuna key present in config: {config_path}")
        train_folds += len(item["folds"])
        eval_folds += len(item["folds"]) if item.get("separate_eval") else 0
    if train_folds != 40 or eval_folds != 20:
        failures.append(f"matrix fold counts differ: train={train_folds} eval={eval_folds}")

    if len(matrix["experiments"]) != 8:
        failures.append("matrix must contain exactly eight experiments")
    return {
        "matrix": str(MATRIX),
        "experiments": len(matrix["experiments"]),
        "train_folds": train_folds,
        "eval_folds": eval_folds,
        "hidden_folds": train_folds,
        "total_jobs": train_folds + eval_folds + train_folds,
        "failures": failures,
    }


def prepare(*, run_id: str, build: bool, required_path_prefix: Path | None,
            model_path: Path, github_issue: int, github_pr: int,
            skip_context_fit: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    components: list[dict[str, Any]] = []
    en_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    equivalences: list[dict[str, Any]] = []
    translations: list[dict[str, Any]] = []

    for config_path, native_path in zip(
        [resolve_project_path(p) for p in BUILD_CONFIGS],
        [resolve_project_path(p) for p in NATIVE_CONFIGS],
    ):
        dataset = str(load_yaml_with_overrides(config_path, [])["dataset"]).lower()
        if build:
            print(f"Building MN5 harmonized English component: {config_path}", flush=True)
            build_for_config(config_path, [])
        component = validate_component(config_path, required_path_prefix=required_path_prefix)
        components.append(component)
        manifest_path = resolve_project_path(component["manifest_path"])
        en_rows = read_jsonl(manifest_path)
        en_rows_by_dataset[dataset] = en_rows
        translations.append(audit_translation_cache(dataset))

        native_config = load_yaml_with_overrides(native_path, [])
        native_split_dir = resolve_project_path(native_config["output_dirs"]["split_dir"])
        native_metadata_path = native_split_dir / f"{dataset}_manifest_metadata.json"
        if not native_metadata_path.is_file():
            failures.append(f"missing native metadata for equivalence: {native_metadata_path}")
            equivalences.append({"dataset": dataset, "failures": [f"native metadata missing: {native_metadata_path}"], "details": {}})
            continue
        native_meta = read_json(native_metadata_path)
        native_manifest = resolve_project_path(native_meta["manifest_path"])
        if not native_manifest.is_file():
            failures.append(f"missing native manifest for equivalence: {native_manifest}")
            equivalences.append({"dataset": dataset, "failures": [f"native manifest missing: {native_manifest}"], "details": {}})
            continue
        native_rows = read_jsonl(native_manifest)
        equivalences.append(equivalence_audit(native_meta, native_rows, read_json(component["metadata_path"]), en_rows))

    recipe_audit = recipe_and_scope_audit()
    failures.extend(recipe_audit["failures"])
    for translation in translations:
        failures.extend(translation["failures"])
    for equivalence in equivalences:
        failures.extend(equivalence["failures"])

    context = None
    if skip_context_fit:
        context = {"skipped": True, "reason": "local validation without model copy"}
    elif model_path.is_dir():
        context = context_fit(en_rows_by_dataset, model_path)
        failures.extend(context["failures"])
    else:
        failures.append(f"model path missing for context audit: {model_path}")

    audit = {
        "schema_version": "harmonized_en_mn5_preflight.v1",
        "status": "passed" if not failures else "failed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("HARMONIZED_EN_SOURCE_COMMIT"),
        "source_branch": os.environ.get("HARMONIZED_EN_SOURCE_BRANCH"),
        "research": {"github_issue": github_issue, "github_pr": github_pr},
        "required_path_prefix": str(required_path_prefix) if required_path_prefix else None,
        "recipe_id": EN_RECIPE,
        "components": components,
        "translations": translations,
        "equivalence": equivalences,
        "context_fit": context,
        "job_scope": {key: recipe_audit[key] for key in ("train_folds", "eval_folds", "hidden_folds", "total_jobs")},
        "optuna_enabled": False,
        "failures": failures,
    }
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--required-path-prefix", type=Path)
    parser.add_argument("--model-path", type=Path,
                        default=Path(os.environ.get(
                            "MODEL_PATH",
                            "/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct",
                        )))
    parser.add_argument("--github-issue", type=int, required=True)
    parser.add_argument("--github-pr", type=int, required=True)
    parser.add_argument("--skip-context-fit", action="store_true",
                        help="Skip the processor/tokenizer context audit (local validation without a model copy).")
    parser.add_argument("--audit-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = prepare(
        run_id=args.run_id,
        build=not args.validate_only,
        required_path_prefix=args.required_path_prefix,
        model_path=args.model_path,
        github_issue=args.github_issue,
        github_pr=args.github_pr,
        skip_context_fit=args.skip_context_fit,
    )
    audit_path = resolve_project_path(
        args.audit_path
        or PROJECT_ROOT / "outputs/harmonized_en_mn5_preflight" / args.run_id / "audit.json"
    )
    save_json(audit, audit_path)
    print(json.dumps({"status": audit["status"], "audit": str(audit_path),
                      "failures": audit["failures"]}, indent=2))


if __name__ == "__main__":
    main()
