#!/usr/bin/env python3
"""Fail-closed, model-free preflight for the Turkish pooled study.

The two pooled manifest/split pairs (native + English, 2,221 rows each) are
built once by scripts/build_turkish_pooled_manifest.py.  This preflight
audits the prebuilt pairs without rebuilding them: row counts, variant mix,
subject/label contract, per-row translation hashes (EN), fold identity,
native/EN parity, and (on MN5) model snapshots and environment.  It never
loads a model or trains a classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.turkish_pooled_qcond import GROUP_ID, load_cells
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
)


SCHEMA_VERSION = "audiollm.turkish_pooled_qcond_preflight.v1"
EXPECTED_SUBJECTS = 120
EXPECTED_LABELS = {0: 37, 1: 83}
EXPECTED_ROWS = 2221
EXPECTED_VARIANTS = {"pos_only_t17": 1051, "negative_only_t17": 1170}
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


def _subject_contract(rows: list[dict[str, Any]], *, language: str) -> dict[str, Any]:
    subjects: dict[str, int] = {}
    sample_ids: list[str] = []
    variants: Counter[str] = Counter()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        subject_id = str(row.get("subject_id", ""))
        if not sample_id or not subject_id:
            raise PreflightError("manifest row is missing sample_id or subject_id")
        sample_ids.append(sample_id)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise PreflightError(f"non-binary label in pooled/{language} manifest: {row}")
        previous = subjects.setdefault(subject_id, label)
        if previous != label:
            raise PreflightError(f"subject has inconsistent labels in pooled/{language}: {subject_id}")
        variant = str(row.get("dataset_variant", "")).strip()
        if variant not in EXPECTED_VARIANTS:
            raise PreflightError(f"unexpected dataset_variant {variant!r} in pooled/{language}")
        variants[variant] += 1
    if len(sample_ids) != len(set(sample_ids)):
        raise PreflightError(f"duplicate sample IDs in pooled/{language} manifest")
    if dict(variants) != dict(EXPECTED_VARIANTS):
        raise PreflightError(f"pooled/{language} variant mix {dict(variants)} != {EXPECTED_VARIANTS}")
    labels = Counter(subjects.values())
    return {
        "rows": len(rows),
        "subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "subject_labels": {str(key): int(value) for key, value in sorted(labels.items())},
        "sample_ids": sorted(sample_ids),
        "sample_id_sha256": _canonical_sha(sorted(sample_ids)),
        "variant_counts": dict(sorted(variants.items())),
    }


def _fold_identity(split_dir: Path) -> dict[str, Any]:
    metadata_path = split_dir / "turkish_manifest_metadata.json"
    if not metadata_path.is_file():
        raise PreflightError(f"manifest metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    manifest_path = resolve_project_path(metadata.get("manifest_path"))
    if not manifest_path.is_file():
        raise PreflightError(f"manifest is missing: {manifest_path}")
    folds_path = metadata.get("folds_path")
    if not folds_path:
        raise PreflightError("manifest metadata has no folds_path")
    path = resolve_project_path(folds_path)
    if not path.is_file():
        raise PreflightError(f"fold file is missing: {path}")
    folds = read_json(path)
    normalized = json.loads(json.dumps(folds, sort_keys=True, ensure_ascii=True))
    return {
        "manifest": str(manifest_path),
        "metadata": str(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_manifest_hash": metadata.get("manifest_hash"),
        "fold_hash": metadata.get("fold_hash"),
        "folds_path": str(path),
        "folds_sha256": sha256_file(path),
        "folds": normalized,
    }


def _translation_audit(manifest_path: Path, expected_rows: int) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"EN manifest rows {len(rows)} != {expected_rows}")
    for row in rows:
        if str(row.get("language", "")).strip() not in ("en", ""):
            pass
        if not str(row.get("translation_sha256", "")).strip():
            failures.append(f"row without translation hash: {row.get('sample_id')}")
            break
    return {
        "manifest": str(manifest_path),
        "rows": len(rows),
        "failures": failures,
    }


def _model_audit(config_paths: list[Path], *, require_models: bool) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for path in sorted(set(config_paths)):
        config = load_yaml_with_overrides(path, [])
        if not isinstance(config, dict):
            raise PreflightError(f"resolved config is not an object: {path}")
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


def _audit_pair(*, language: str, manifest_dir: Path, split_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_dir / "turkish_manifest.jsonl"
    if not manifest_path.is_file():
        raise PreflightError(f"pooled manifest is missing: {manifest_path}")
    rows = read_jsonl(manifest_path)
    contract = _subject_contract(rows, language=language)
    if contract["rows"] != EXPECTED_ROWS:
        raise PreflightError(f"pooled/{language} rows {contract['rows']} != {EXPECTED_ROWS}")
    if contract["subjects"] != EXPECTED_SUBJECTS:
        raise PreflightError(f"pooled/{language} subjects {contract['subjects']} != {EXPECTED_SUBJECTS}")
    if {int(key): value for key, value in contract["subject_labels"].items()} != EXPECTED_LABELS:
        raise PreflightError(f"pooled/{language} subject labels do not match {EXPECTED_LABELS}")
    folds = _fold_identity(split_dir)
    translation = _translation_audit(manifest_path, EXPECTED_ROWS) if language == "english" else None
    return {
        "condition": "pooled",
        "language": language,
        "manifest": str(manifest_path),
        "metadata": folds["metadata"],
        "manifest_sha256": folds["manifest_sha256"],
        "metadata_sha256": folds["metadata_sha256"],
        "metadata_manifest_hash": folds["metadata_manifest_hash"],
        "contract": contract,
        "folds": folds,
        "translation": translation,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    cells = load_cells(PROJECT_ROOT)
    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    pairs: list[dict[str, Any]] = []
    for language in ("native", "english"):
        manifest_dir = manifest_root / ("pooled_en" if language == "english" else "pooled")
        split_dir = split_root / ("pooled_en" if language == "english" else "pooled")
        try:
            pairs.append(_audit_pair(language=language, manifest_dir=manifest_dir, split_dir=split_dir))
        except Exception as exc:
            failures.append(f"pooled/{language}: {exc}")
    pair_map = {(item["condition"], item["language"]): item for item in pairs}
    native = pair_map.get(("pooled", "native"))
    english = pair_map.get(("pooled", "english"))
    if native and english:
        if native["contract"]["sample_ids"] != english["contract"]["sample_ids"]:
            failures.append("pooled: native and English sample IDs differ")
        if native["contract"]["subject_ids"] != english["contract"]["subject_ids"]:
            failures.append("pooled: native and English subject IDs differ")
        if native["folds"]["folds_sha256"] != english["folds"]["folds_sha256"]:
            failures.append("pooled: native and English fold files differ")
    configs = [PROJECT_ROOT / cell.config for cell in cells]
    model_audit = _model_audit(configs, require_models=args.require_models)
    failures.extend(model_audit["failures"])
    for pair in pairs:
        failures.extend((pair.get("translation") or {}).get("failures", []))
    environment = _environment_audit() if args.require_environment else {"skipped": True, "failures": []}
    failures.extend(environment.get("failures", []))
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "group_id": GROUP_ID,
        "status": "passed" if not failures else "failed",
        "stage": args.stage,
        "require_models": bool(args.require_models),
        "require_environment": bool(args.require_environment),
        "manifest_root": str(manifest_root),
        "split_root": str(split_root),
        "expected": {
            "subjects": EXPECTED_SUBJECTS,
            "subject_labels": {str(key): value for key, value in EXPECTED_LABELS.items()},
            "rows": EXPECTED_ROWS,
            "variant_counts": EXPECTED_VARIANTS,
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
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-environment", action="store_true")
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
