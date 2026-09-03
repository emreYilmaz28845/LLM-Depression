#!/usr/bin/env python3
"""Fail-closed, model-free preflight for the Turkish question-condition study.

The preflight creates four isolated manifest/split pairs (pos_only/negative-only ×
native/English), audits their identity and translation provenance, and checks
the exact data contract before any model or Slurm job is started.  It never
loads a model or trains a classifier.  ``--require-models`` is used on MN5 to
turn the model-snapshot check into a hard gate; local development can omit it
when the local machine intentionally has no Qwen-Audio or Gemma snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import build_for_config
from src.turkish_question_condition import GROUP_ID, load_cells
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
)


SCHEMA_VERSION = "audiollm.turkish_question_condition_preflight.v1"
EXPECTED_SUBJECTS = 120
EXPECTED_LABELS = {0: 37, 1: 83}
EXPECTED_ROWS = {"pos_only": 1051, "negative_only": 1170}
EXPECTED_EXCLUDED_WAVS = 145
EXPECTED_TRANSLATION_STATUS = {
    "automatic_high",
    "automatic_medium",
    "automatic_low",
    "human_verified",
}
REQUIRED_IMPORTS = ("torch", "transformers", "accelerate", "peft", "optuna", "xgboost", "sklearn", "yaml")


class PreflightError(ValueError):
    """Raised when a preflight input or output contract is unsafe."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_yaml(path: Path, overrides: list[str]) -> dict[str, Any]:
    value = load_yaml_with_overrides(path, overrides)
    if not isinstance(value, dict):
        raise PreflightError(f"resolved config is not an object: {path}")
    return value


def _manifest_location(metadata_path: Path) -> tuple[Path, dict[str, Any]]:
    if not metadata_path.is_file():
        raise PreflightError(f"manifest metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    manifest_path = resolve_project_path(metadata.get("manifest_path"))
    if not manifest_path.is_file():
        raise PreflightError(f"manifest is missing: {manifest_path}")
    return manifest_path, metadata


def _subject_contract(rows: list[dict[str, Any]], *, condition: str) -> dict[str, Any]:
    subjects: dict[str, int] = {}
    sample_ids: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        subject_id = str(row.get("subject_id", ""))
        if not sample_id or not subject_id:
            raise PreflightError("manifest row is missing sample_id or subject_id")
        sample_ids.append(sample_id)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise PreflightError(f"non-binary label in {condition} manifest: {row}")
        previous = subjects.setdefault(subject_id, label)
        if previous != label:
            raise PreflightError(f"subject has inconsistent labels in {condition}: {subject_id}")
    duplicate_sample_ids = len(sample_ids) != len(set(sample_ids))
    if duplicate_sample_ids:
        raise PreflightError(f"duplicate sample IDs in {condition} manifest")
    labels = Counter(subjects.values())
    return {
        "rows": len(rows),
        "subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "subject_labels": {str(key): int(value) for key, value in sorted(labels.items())},
        "sample_ids": sorted(sample_ids),
        "sample_id_sha256": _canonical_sha(sorted(sample_ids)),
    }


def _fold_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    folds_path = metadata.get("folds_path")
    if not folds_path:
        raise PreflightError("manifest metadata has no folds_path")
    path = resolve_project_path(folds_path)
    if not path.is_file():
        raise PreflightError(f"fold file is missing: {path}")
    folds = read_json(path)
    normalized = json.loads(json.dumps(folds, sort_keys=True, ensure_ascii=True))
    return {
        "fold_hash": metadata.get("fold_hash"),
        "folds_path": str(path),
        "folds_sha256": sha256_file(path),
        "folds": normalized,
    }


def _translation_audit(config: dict[str, Any], condition: str, expected_rows: int) -> dict[str, Any]:
    transcript_cfg = config.get("transcripts") or {}
    cache_value = transcript_cfg.get("cache_path")
    if not cache_value:
        raise PreflightError(f"English config has no translation cache: {condition}")
    accepted_path = resolve_project_path(cache_value)
    rejected_path = accepted_path.with_name("rejected.jsonl")
    if not accepted_path.is_file():
        raise PreflightError(f"translation accepted cache is missing: {accepted_path}")
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path) if rejected_path.is_file() else []
    failures: list[str] = []
    if len(accepted) != expected_rows:
        failures.append(f"accepted translation rows {len(accepted)} != {expected_rows}")
    if rejected:
        failures.append(f"translation rejected cache is not empty: {len(rejected)} rows")
    keys: set[tuple[str, str, int]] = set()
    for row in accepted:
        key = (str(row.get("unit_id")), str(row.get("field")), int(row.get("part_index", 0)))
        if key in keys:
            failures.append(f"duplicate translation key: {key}")
        keys.add(key)
        status = str(row.get("status", "")).lower()
        if status not in EXPECTED_TRANSLATION_STATUS:
            failures.append(f"unaccepted translation status {status!r}")
        if bool(row.get("fallback")) or bool(row.get("used_native_fallback")):
            failures.append(f"translation fallback flag present for {key}")
        if not str(row.get("translation_sha256", "")):
            failures.append(f"translation hash missing for {key}")
    return {
        "accepted_path": str(accepted_path),
        "accepted_sha256": sha256_file(accepted_path),
        "accepted_rows": len(accepted),
        "rejected_path": str(rejected_path),
        "rejected_rows": len(rejected),
        "status_counts": dict(sorted(Counter(str(row.get("status", "")).lower() for row in accepted).items())),
        "failures": failures,
    }


def _model_audit(config_paths: list[Path], *, require_models: bool) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for path in sorted(set(config_paths)):
        config = _read_yaml(path, [])
        model_path = resolve_project_path(config.get("model_name_or_path"))
        backend = str(config.get("model_backend") or "qwen_audio").lower()
        exists = model_path.is_dir()
        record = {
            "config": str(path.relative_to(PROJECT_ROOT)),
            "backend": backend,
            "path": str(model_path),
            "exists": exists,
        }
        if exists and (model_path / "config.json").is_file():
            record["config_sha256"] = sha256_file(model_path / "config.json")
        if require_models and not exists:
            failures.append(f"missing model snapshot for {path}: {model_path}")
        models[str(path)] = record
    return {"required": require_models, "models": models, "failures": failures}


def _environment_audit() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    failures: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "imported"))
        except Exception as exc:  # pragma: no cover - interpreter dependent
            versions[name] = None
            failures.append(f"import {name} failed: {exc}")
    return {"python": platform.python_version(), "versions": versions, "failures": failures}


def _build_pair(
    *,
    config_path: Path,
    condition: str,
    language: str,
    output_root: Path,
    dataset_root: Path | None,
    translation_root: Path | None,
    reuse_existing: bool,
) -> dict[str, Any]:
    manifest_dir = output_root / "manifests" / condition / language
    split_dir = output_root / "splits" / condition / language
    existing_pair = manifest_dir.exists() or split_dir.exists()
    if existing_pair and not reuse_existing:
        raise PreflightError(f"refusing to overwrite existing preflight outputs: {manifest_dir} / {split_dir}")
    if existing_pair and (not manifest_dir.is_dir() or not split_dir.is_dir()):
        raise PreflightError(f"existing preflight pair is not two directories: {manifest_dir} / {split_dir}")
    overrides = [
        f"--set=output_dirs.manifest_dir={manifest_dir}",
        f"--set=output_dirs.split_dir={split_dir}",
        "--set=seed=1337",
    ]
    if dataset_root is not None:
        overrides.append(f"--set=dataset_root={dataset_root}")
    if translation_root is not None and language == "english":
        cache_family = "harmonized_en_complete_v1" if condition == "pos_only" else "harmonized_en_complete_v3"
        cache_dataset = "turkish_pos_only_t17" if condition == "pos_only" else "turkish_negative_only_t17"
        overrides.append(
            f"--set=transcripts.cache_path={translation_root / cache_family / cache_dataset / 'accepted.jsonl'}"
        )
    config = _read_yaml(config_path, overrides)
    if not existing_pair:
        build_for_config(config_path, overrides)
    metadata_path = split_dir / "turkish_manifest_metadata.json"
    manifest_path, metadata = _manifest_location(metadata_path)
    rows = read_jsonl(manifest_path)
    contract = _subject_contract(rows, condition=condition)
    if contract["rows"] != EXPECTED_ROWS[condition]:
        raise PreflightError(f"{condition}/{language} rows {contract['rows']} != {EXPECTED_ROWS[condition]}")
    if contract["subjects"] != EXPECTED_SUBJECTS:
        raise PreflightError(f"{condition}/{language} subjects {contract['subjects']} != {EXPECTED_SUBJECTS}")
    if {int(key): value for key, value in contract["subject_labels"].items()} != EXPECTED_LABELS:
        raise PreflightError(f"{condition}/{language} subject labels do not match {EXPECTED_LABELS}")
    if language == "english":
        if any(str(row.get("language")) != "en" for row in rows):
            raise PreflightError(f"English manifest contains a non-English row: {manifest_path}")
        if any(str(row.get("transcript_variant")) != "english" for row in rows):
            raise PreflightError(f"English manifest contains a non-English transcript variant: {manifest_path}")
        if any(not str(row.get("translation_sha256", "")) for row in rows):
            raise PreflightError(f"English manifest contains a row without translation hash: {manifest_path}")
    return {
        "condition": condition,
        "language": language,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "resolved_dataset_root": str(config.get("dataset_root")),
        "manifest": str(manifest_path),
        "metadata": str(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_manifest_hash": metadata.get("manifest_hash"),
        "contract": contract,
        "folds": _fold_identity(metadata),
        "translation": _translation_audit(config, condition, EXPECTED_ROWS[condition]) if language == "english" else None,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    cells = load_cells(PROJECT_ROOT)
    by_id = {cell.cell_id: cell for cell in cells}
    representative_ids = {"pos_only": {"native": "P02", "english": "P03"}, "negative_only": {"native": "N02", "english": "N03"}}
    dataset_roots = {"pos_only": args.pos_only_root, "negative_only": args.negative_root}
    output_root = args.output_root.resolve()
    if output_root.exists():
        existing_files = [path for path in output_root.rglob("*") if path.is_file()]
        if existing_files and not args.reuse_existing:
            raise PreflightError(
                f"preflight output root already contains files: {existing_files[:3]}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    for condition, languages in representative_ids.items():
        for language, cell_id in languages.items():
            try:
                pairs.append(
                    _build_pair(
                        config_path=PROJECT_ROOT / by_id[cell_id].config,
                        condition=condition,
                        language=language,
                        output_root=output_root,
                        dataset_root=dataset_roots[condition],
                        translation_root=args.translation_root,
                        reuse_existing=args.reuse_existing,
                    )
                )
            except Exception as exc:
                failures.append(f"{condition}/{language}: {exc}")

    pair_map = {(item["condition"], item["language"]): item for item in pairs}
    for condition in ("pos_only", "negative_only"):
        native = pair_map.get((condition, "native"))
        english = pair_map.get((condition, "english"))
        if not native or not english:
            continue
        if native["contract"]["sample_ids"] != english["contract"]["sample_ids"]:
            failures.append(f"{condition}: native and English sample IDs differ")
        if native["contract"]["subject_ids"] != english["contract"]["subject_ids"]:
            failures.append(f"{condition}: native and English subject IDs differ")
        if native["folds"]["fold_hash"] != english["folds"]["fold_hash"]:
            failures.append(f"{condition}: native and English fold hashes differ")

    for condition, root in dataset_roots.items():
        if root is None:
            continue
        audio_dir = root / "all-files" if (root / "all-files").is_dir() else root
        wav_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.is_dir() else 0
        if condition == "negative_only" and wav_count - EXPECTED_ROWS[condition] != EXPECTED_EXCLUDED_WAVS:
            failures.append(
                f"negative_only: all-files WAV count {wav_count} minus selected rows {EXPECTED_ROWS[condition]} "
                f"does not equal excluded count {EXPECTED_EXCLUDED_WAVS}"
            )

    configs = [PROJECT_ROOT / cell.config for cell in cells]
    model_audit = _model_audit(configs, require_models=args.require_models)
    failures.extend(model_audit["failures"])
    environment = _environment_audit() if args.require_environment else {"skipped": True, "failures": []}
    failures.extend(environment.get("failures", []))
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "group_id": GROUP_ID,
        "status": "passed" if not failures else "failed",
        "stage": args.stage,
        "require_models": bool(args.require_models),
        "require_environment": bool(args.require_environment),
        "output_root": str(output_root),
        "expected": {
            "subjects": EXPECTED_SUBJECTS,
            "subject_labels": {str(key): value for key, value in EXPECTED_LABELS.items()},
            "rows": EXPECTED_ROWS,
            "negative_only_excluded_wav_count": EXPECTED_EXCLUDED_WAVS,
        },
        "pairs": pairs,
        "models": model_audit,
        "environment": environment,
        "failures": failures,
    }
    audit["audit_sha256"] = _canonical_sha({key: value for key, value in audit.items() if key != "audit_sha256"})
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "production"), default="production")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pos-only-root", type=Path)
    parser.add_argument("--negative-root", type=Path)
    parser.add_argument("--translation-root", type=Path)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-environment", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="re-audit existing manifest/split pairs without overwriting them")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing preflight audit: {args.output}")
    try:
        audit = run_preflight(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failures": [str(exc)]}, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(args.output), "audit_sha256": audit["audit_sha256"], "failures": audit["failures"]}, indent=2, sort_keys=True))
    raise SystemExit(0 if audit["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
