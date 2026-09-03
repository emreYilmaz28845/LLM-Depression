#!/usr/bin/env python3
"""Model-free preflight for the Turkish pooled question-conditioned campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_turkish_pooled_manifest import ManifestError, main as build_pooled_manifest
from src.turkish_pooled_qcond import GROUP_ID, load_cells
from src.utils import load_yaml_with_overrides, read_json, read_jsonl, resolve_project_path, sha256_file


SCHEMA_VERSION = "audiollm.turkish_pooled_qcond_preflight.v1"
REQUIRED_IMPORTS = ("torch", "transformers", "accelerate", "peft", "optuna", "xgboost", "sklearn", "yaml")


class PreflightError(ValueError):
    """Raised when a preflight contract fails."""


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _builder_args(args: argparse.Namespace, output_root: Path, audit_path: Path) -> list[str]:
    values = [
        "--positive-native-manifest", str(args.positive_native_manifest),
        "--positive-native-split", str(args.positive_native_split),
        "--negative-native-manifest", str(args.negative_native_manifest),
        "--negative-native-split", str(args.negative_native_split),
        "--positive-english-manifest", str(args.positive_english_manifest),
        "--positive-english-split", str(args.positive_english_split),
        "--negative-english-manifest", str(args.negative_english_manifest),
        "--negative-english-split", str(args.negative_english_split),
        "--native-output-dir", str(output_root / "manifests" / "native"),
        "--english-output-dir", str(output_root / "manifests" / "english"),
        "--native-split-output-dir", str(output_root / "splits" / "native"),
        "--english-split-output-dir", str(output_root / "splits" / "english"),
        "--audit-output", str(output_root / "preflight" / "manifest_audit.json"),
        "--native-config", str(PROJECT_ROOT / "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml"),
        "--english-config", str(PROJECT_ROOT / "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml"),
    ]
    return values


def _environment_audit(required: bool) -> dict[str, Any]:
    if not required:
        return {"required": False, "skipped": True, "failures": []}
    versions: dict[str, str | None] = {}
    failures: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "imported"))
        except Exception as exc:  # pragma: no cover - environment-dependent
            versions[name] = None
            failures.append(f"import {name} failed: {exc}")
    return {"required": True, "python": platform.python_version(), "versions": versions, "failures": failures}


def _model_audit(require_models: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for cell in load_cells(PROJECT_ROOT):
        config_path = PROJECT_ROOT / cell.config
        config = load_yaml_with_overrides(config_path, [])
        model_path = resolve_project_path(config["model_name_or_path"])
        exists = model_path.is_dir()
        record = {
            "cell_id": cell.cell_id,
            "config": cell.config,
            "model_backend": config.get("model_backend", "qwen"),
            "model_path": str(model_path),
            "exists": exists,
        }
        if exists and (model_path / "config.json").is_file():
            record["config_sha256"] = sha256_file(model_path / "config.json")
        if require_models and not exists:
            failures.append(f"missing model snapshot for {cell.cell_id}: {model_path}")
        records.append(record)
    return {"required": require_models, "models": records, "failures": failures}


def _check_outputs(output_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for language in ("native", "english"):
        manifest_metadata = output_root / "splits" / language / "turkish_manifest_metadata.json"
        if not manifest_metadata.is_file():
            raise PreflightError(f"pooled {language} metadata is missing: {manifest_metadata}")
        metadata = read_json(manifest_metadata)
        manifest = Path(str(metadata["manifest_path"]))
        folds = Path(str(metadata["folds_path"]))
        rows = read_jsonl(manifest)
        if len(rows) != 2221 or len({str(row["subject_id"]) for row in rows}) != 120:
            raise PreflightError(f"pooled {language} output has the wrong row or subject count")
        conditions = {str(row.get("dataset_variant", "")) for row in rows}
        if conditions != {"pos_only_t17", "negative_only_t17"}:
            raise PreflightError(f"pooled {language} output has unexpected conditions: {sorted(conditions)}")
        records[language] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "manifest_hash": metadata.get("manifest_hash"),
            "metadata": str(manifest_metadata),
            "metadata_sha256": sha256_file(manifest_metadata),
            "folds": str(folds),
            "folds_sha256": sha256_file(folds),
            "fold_hash": metadata.get("fold_hash"),
            "row_count": len(rows),
            "subject_count": len({str(row["subject_id"]) for row in rows}),
            "condition_counts": {
                condition: sum(str(row.get("dataset_variant")) == condition for row in rows)
                for condition in ("pos_only_t17", "negative_only_t17")
            },
        }
    if records["native"]["manifest_hash"] == records["english"]["manifest_hash"]:
        raise PreflightError("native and English pooled manifest hashes unexpectedly match")
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "production"), default="production")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="preflight audit JSON path")
    for name in (
        "positive-native-manifest", "positive-native-split", "negative-native-manifest", "negative-native-split",
        "positive-english-manifest", "positive-english-split", "negative-english-manifest", "negative-english-split",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-environment", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="validate existing pooled outputs without building them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    audit_path = args.output.resolve()
    if not args.check_only:
        output_root.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        build_pooled_manifest(_builder_args(args, output_root, audit_path))
    outputs = _check_outputs(output_root)
    model_audit = _model_audit(args.require_models)
    environment = _environment_audit(args.require_environment)
    failures = [*model_audit["failures"], *environment["failures"]]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "group_id": GROUP_ID,
        "stage": args.stage,
        "status": "passed" if not failures else "failed",
        "output_root": str(output_root),
        "outputs": outputs,
        "models": model_audit,
        "environment": environment,
        "failures": failures,
    }
    audit["audit_sha256"] = _canonical_sha(audit)
    if audit_path.exists() and audit_path.read_bytes() != (json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"):
        raise PreflightError(f"refusing to overwrite incompatible preflight audit: {audit_path}")
    if not args.check_only and not audit_path.exists():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(audit_path), "audit_sha256": audit["audit_sha256"], "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, PreflightError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
