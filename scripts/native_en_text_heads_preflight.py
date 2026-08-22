#!/usr/bin/env python3
"""Model-free preflight for the v2 native/complete-English text-head study.

The script is intentionally fail-closed.  It audits the current translation
cache records, paired manifests/splits, all study configs, model snapshots,
runtime imports, and the exact dry-run matrix.  It never trains or writes into
an immutable deployment tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import manifest_build_signature
from src.merged.protocol import canonical_sha256
from src.native_en_text_heads import (
    BACKBONES,
    CONDITIONS,
    GROUP_ID,
    MERGED_CONFIGS,
    SPLIT_SEED,
    STANDALONE_CONFIGS,
    matrix_payload,
    validate_configs,
)
from src.utils import load_yaml_with_overrides, read_json, read_jsonl, resolve_project_path, sha256_file


EXPECTED_TRANSLATION_COUNTS = {
    "d3tec": 3677,
    "androids_interview": 2176,
    "cmdc": 923,
    "turkish": 1051,
}
DATASET_CACHE_NAMES = {
    "d3tec": "d3tec",
    "androids_interview": "androids_interview",
    "cmdc": "cmdc",
    "turkish": "turkish",
}


def _load_path_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest map must be a JSON object")
    return {str(key): dict(value) for key, value in payload.items()}


def _mapped_paths(
    *,
    condition: str,
    backbone: str,
    dataset: str,
    config: dict[str, Any],
    path_map: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    keys = (
        f"{condition}/{backbone}/{dataset}",
        str(config.get("dataset", dataset)).lower(),
    )
    for key in keys:
        item = path_map.get(key)
        if item:
            return resolve_project_path(item["manifest"]), resolve_project_path(item["metadata"])
    output = config.get("output_dirs") or {}
    dataset_name = str(config["dataset"]).lower()
    manifest_dir = resolve_project_path(output["manifest_dir"])
    split_dir = resolve_project_path(output["split_dir"])
    return manifest_dir / f"{dataset_name if dataset_name != 'androids_interview' else 'androids_interview'}_manifest.jsonl", split_dir / f"{dataset_name}_manifest_metadata.json"


def _manifest_record(
    *,
    config_path: Path,
    manifest_path: Path,
    metadata_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    if not metadata_path.is_file():
        return {}, [], [f"missing metadata: {metadata_path}"]
    if not manifest_path.is_file():
        return {}, [], [f"missing manifest: {manifest_path}"]
    metadata = read_json(metadata_path)
    rows = read_jsonl(manifest_path)
    if not rows:
        failures.append(f"empty manifest: {manifest_path}")
    if int(metadata.get("manifest_row_count", -1)) != len(rows):
        failures.append(f"manifest row count mismatch: {manifest_path}")
    if metadata.get("build_signature") != manifest_build_signature(
        load_yaml_with_overrides(config_path, [])
    ):
        failures.append(f"manifest build signature mismatch: {manifest_path}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append(f"duplicate sample IDs: {manifest_path}")
    labels: dict[str, int] = {}
    for row in rows:
        subject = str(row.get("subject_id", ""))
        label = int(row.get("label", -1))
        if subject in labels and labels[subject] != label:
            failures.append(f"inconsistent subject label {subject} in {manifest_path}")
        labels[subject] = label
    return metadata, rows, failures


def _paired_manifest_audit(
    native_config_path: Path,
    english_config_path: Path,
    native_paths: tuple[Path, Path],
    english_paths: tuple[Path, Path],
) -> dict[str, Any]:
    native_meta, native_rows, failures = _manifest_record(
        config_path=native_config_path,
        manifest_path=native_paths[0],
        metadata_path=native_paths[1],
    )
    english_meta, english_rows, english_failures = _manifest_record(
        config_path=english_config_path,
        manifest_path=english_paths[0],
        metadata_path=english_paths[1],
    )
    failures.extend(english_failures)
    native_ids = {str(row.get("sample_id", "")) for row in native_rows}
    english_ids = {str(row.get("sample_id", "")) for row in english_rows}
    if native_ids != english_ids:
        failures.append(
            f"native/English sample sets differ: {len(native_ids)} vs {len(english_ids)}"
        )
    native_labels = {str(row.get("subject_id")): int(row.get("label", -1)) for row in native_rows}
    english_labels = {str(row.get("subject_id")): int(row.get("label", -1)) for row in english_rows}
    if native_labels != english_labels:
        failures.append("native/English subject labels differ")
    native_folds = native_meta.get("fold_hash")
    english_folds = english_meta.get("fold_hash")
    if native_folds != english_folds:
        failures.append("native/English fold hashes differ")
    if int(load_yaml_with_overrides(native_config_path, []).get("split", {}).get("seed", -1)) != SPLIT_SEED:
        failures.append(f"native split seed is not {SPLIT_SEED}: {native_config_path}")
    if int(load_yaml_with_overrides(english_config_path, []).get("split", {}).get("seed", -1)) != SPLIT_SEED:
        failures.append(f"English split seed is not {SPLIT_SEED}: {english_config_path}")
    english_rows_without_sha = sum(not str(row.get("translation_sha256", "")) for row in english_rows)
    english_not_en = sum(str(row.get("language", "")) != "en" for row in english_rows)
    english_not_variant = sum(str(row.get("transcript_variant", "")) != "english" for row in english_rows)
    if english_rows_without_sha:
        failures.append(f"English rows without translation_sha256: {english_rows_without_sha}")
    if english_not_en or english_not_variant:
        failures.append(f"English rows with wrong language/variant: {english_not_en}/{english_not_variant}")
    return {
        "native": {
            "manifest": str(native_paths[0]),
            "metadata": str(native_paths[1]),
            "rows": len(native_rows),
            "sha256": sha256_file(native_paths[0]) if native_paths[0].is_file() else None,
            "manifest_hash": native_meta.get("manifest_hash"),
            "metadata_sha256": sha256_file(native_paths[1]) if native_paths[1].is_file() else None,
            "fold_hash": native_meta.get("fold_hash"),
        },
        "english": {
            "manifest": str(english_paths[0]),
            "metadata": str(english_paths[1]),
            "rows": len(english_rows),
            "sha256": sha256_file(english_paths[0]) if english_paths[0].is_file() else None,
            "manifest_hash": english_meta.get("manifest_hash"),
            "metadata_sha256": sha256_file(english_paths[1]) if english_paths[1].is_file() else None,
            "fold_hash": english_meta.get("fold_hash"),
        },
        "failures": failures,
    }


def _translation_audits() -> list[dict[str, Any]]:
    from scripts.prepare_harmonized_en_mn5 import audit_translation_cache

    audits: list[dict[str, Any]] = []
    for dataset in DATASET_CACHE_NAMES:
        audit = audit_translation_cache(dataset)
        cache_dir = Path(audit["cache_root"])
        accepted_path = cache_dir / "accepted.jsonl"
        accepted = read_jsonl(accepted_path) if accepted_path.is_file() else []
        extra_failures: list[str] = []
        for row in accepted:
            status = str(row.get("status", "")).lower()
            if status not in {"automatic_high", "automatic_medium", "automatic_low", "human_verified"}:
                extra_failures.append(f"{dataset}: unacceptable accepted status {status!r}")
            if bool(row.get("fallback")) or bool(row.get("used_native_fallback")):
                extra_failures.append(f"{dataset}: fallback translation record present")
            if str(row.get("translation_sha256", "")) == "":
                extra_failures.append(f"{dataset}: accepted record lacks translation_sha256")
        audit["failures"] = list(audit.get("failures", [])) + extra_failures
        audits.append(audit)
    return audits


def _context_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run tokenizer checks inside the interpreter matching one model backend."""
    failures: list[str] = []
    models: dict[str, dict[str, Any]] = {}
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"failures": [f"transformers import failed: {exc}"], "models": {}}
    for item in payload.get("items", []):
        config_path = str(item["config"])
        model_path = Path(str(item["model_path"]))
        backend = str(item["backend"])
        entry: dict[str, Any] = {"backend": backend, "path": str(model_path), "config": config_path}
        if not model_path.is_dir():
            failures.append(f"missing {backend} text model snapshot: {model_path}")
            models[config_path] = entry
            continue
        try:
            model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
            text_config = getattr(model_config, "text_config", model_config)
            limit = int(getattr(text_config, "max_position_embeddings"))
            texts = [str(text) for text in item.get("subject_texts", [])]
            counts = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
            maximum = max(counts, default=0)
            entry.update({
                "config_sha256": sha256_file(model_path / "config.json"),
                "context_limit": limit,
                "max_subject_transcript_tokens": maximum,
                "subjects": len(counts),
                "interpreter": sys.executable,
            })
            if maximum + 128 > limit:
                failures.append(f"{config_path}: transcript context {maximum}+128 exceeds {limit}")
        except Exception as exc:  # pragma: no cover - environment-specific
            failures.append(f"{config_path}: tokenizer/config context audit failed: {exc}")
        models[config_path] = entry
    return {"failures": failures, "models": models}


def _context_audit(config_paths: list[Path], manifest_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    failures: list[str] = []
    models: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {"qwen": [], "gemma4": []}
    for config_path in sorted(set(config_paths)):
        config = load_yaml_with_overrides(config_path, [])
        backend = "gemma4" if str(config.get("model_backend", "")).lower() == "gemma4" else "qwen"
        model_path = resolve_project_path(config["model_name_or_path"])
        entry: dict[str, Any] = {"backend": backend, "path": str(model_path), "config": str(config_path)}
        if str(config.get("protocol", "")) == "symmetric_merged":
            # The merged YAML has no single dataset transcript to tokenize;
            # its component configs are audited below through the standalone
            # matrix, while the merged protocol itself is checked separately.
            entry["skipped"] = "merged config; component context audit"
            models[str(config_path)] = entry
            continue
        dataset = str(config["dataset"]).lower()
        subjects: dict[str, list[str]] = {}
        for row in manifest_rows.get(dataset, []):
            subjects.setdefault(str(row["subject_id"]), []).append(str(row.get("transcript", "")))
        grouped[backend].append({
            "backend": backend,
            "config": str(config_path),
            "model_path": str(model_path),
            "subject_texts": ["\n".join(values) for values in subjects.values()],
        })

    interpreters = {
        "qwen": os.environ.get("QWEN_PYTHON") or sys.executable,
        "gemma4": os.environ.get("GEMMA_PYTHON") or str(Path(os.environ.get("GEMMA_ENV", "")) / "bin" / "python"),
    }
    for backend, items in grouped.items():
        if not items:
            continue
        interpreter = Path(interpreters[backend])
        if not interpreter.is_file():
            failures.append(f"{backend} context interpreter missing: {interpreter}")
            continue
        try:
            result = subprocess.run(
                [str(interpreter), str(Path(__file__).resolve()), "--context-worker"],
                input=json.dumps({"items": items}),
                capture_output=True,
                text=True,
                check=False,
                timeout=6 * 60 * 60,
                env=dict(os.environ),
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            failures.append(f"{backend} context worker failed to start: {exc}")
            continue
        if result.returncode != 0:
            failures.append(f"{backend} context worker exited {result.returncode}: {result.stderr.strip()}")
            continue
        try:
            worker = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"{backend} context worker returned invalid JSON: {exc}")
            continue
        failures.extend(str(value) for value in worker.get("failures", []))
        models.update({str(key): dict(value) for key, value in worker.get("models", {}).items()})
    return {"failures": failures, "models": models}


def _environment_audit() -> dict[str, Any]:
    failures: list[str] = []
    modules = ("torch", "transformers", "accelerate", "peft", "optuna", "xgboost", "sklearn", "yaml")
    versions: dict[str, str | None] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "imported"))
        except Exception as exc:
            versions[name] = None
            failures.append(f"import {name} failed: {exc}")
    return {"python": platform.python_version(), "versions": versions, "failures": failures}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    try:
        validate_configs(PROJECT_ROOT)
    except Exception as exc:
        failures.append(f"config validation failed: {exc}")
    path_map = _load_path_map(args.manifest_map)
    pairs: list[dict[str, Any]] = []
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for backbone in BACKBONES:
        for dataset in ("d3tec", "androids_interview", "cmdc", "turkish"):
            native = STANDALONE_CONFIGS[("native", backbone, dataset)]
            english = STANDALONE_CONFIGS[("english", backbone, dataset)]
            native_path = PROJECT_ROOT / native
            english_path = PROJECT_ROOT / english
            native_cfg = load_yaml_with_overrides(native_path, [])
            english_cfg = load_yaml_with_overrides(english_path, [])
            native_locations = _mapped_paths(condition="native", backbone=backbone, dataset=dataset, config=native_cfg, path_map=path_map)
            english_locations = _mapped_paths(condition="english", backbone=backbone, dataset=dataset, config=english_cfg, path_map=path_map)
            pair = _paired_manifest_audit(native_path, english_path, native_locations, english_locations)
            pair["dataset"] = dataset
            pair["backbone"] = backbone
            pairs.append(pair)
            for path in (native_locations[0], english_locations[0]):
                if path.is_file():
                    rows = read_jsonl(path)
                    manifest_rows.setdefault(dataset, rows)
    translations = _translation_audits()
    for audit in translations:
        failures.extend(audit.get("failures", []))
    for pair in pairs:
        failures.extend(pair.get("failures", []))
    environment = _environment_audit()
    failures.extend(environment["failures"])
    context = {"skipped": True, "reason": "requested"}
    if not args.skip_context_fit:
        context = _context_audit(
            [PROJECT_ROOT / value for value in STANDALONE_CONFIGS.values()]
            + [PROJECT_ROOT / value for value in MERGED_CONFIGS.values()],
            manifest_rows,
        )
        failures.extend(context["failures"])
    matrix = matrix_payload(args.stage)
    merged = []
    for (condition, backbone), relative in sorted(MERGED_CONFIGS.items()):
        config = load_yaml_with_overrides(PROJECT_ROOT / relative, [])
        merged_record = {
            "condition": condition,
            "backbone": backbone,
            "config": relative,
            "components": [str(item["name"]) for item in config.get("components", [])],
            "split_seed": config.get("protocol_settings", {}).get("split_seed"),
            "head_seed": config.get("heads", {}).get("fixed_seed"),
            "optuna_trials": config.get("heads", {}).get("optuna", {}).get("target_trials"),
        }
        mapped = path_map.get(f"merged/{condition}/{backbone}") or {}
        protocol_root = resolve_project_path(mapped["merged_root"]) if mapped.get("merged_root") else resolve_project_path(config["output_dirs"]["merged_root"])
        protocol_path = protocol_root / "merged_protocol.json"
        if protocol_path.is_file():
            protocol = read_json(protocol_path)
            if protocol.get("protocol", {}).get("split_seed") not in (None, SPLIT_SEED):
                failures.append(f"merged protocol split seed mismatch: {protocol_path}")
            merged_record["protocol"] = {
                "path": str(protocol_path),
                "manifest_hash": protocol.get("manifest", {}).get("manifest_hash"),
                "split_hash": protocol.get("protocol", {}).get("split_hash"),
                "protocol_sha256": sha256_file(protocol_path),
            }
        else:
            merged_record["protocol"] = {
                "path": str(protocol_path),
                "manifest_hash": None,
                "split_hash": None,
                "protocol_sha256": None,
            }
            failures.append(f"missing merged protocol: {protocol_path}")
        merged.append(merged_record)
    audit = {
        "schema_version": "native_en_text_heads_v2_preflight.v1",
        "status": "passed" if not failures else "failed",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": GROUP_ID,
        "deployment_id": args.deployment_id,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_commit": os.environ.get("NATIVE_EN_TEXT_HEADS_SOURCE_COMMIT"),
        "environment": environment,
        "translations": translations,
        "paired_manifests": pairs,
        "merged_configs": merged,
        "context_fit": context,
        "matrix": matrix,
        "failures": failures,
    }
    audit["audit_sha256"] = canonical_sha256({key: value for key, value in audit.items() if key != "audit_sha256"})
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("smoke", "production"), default="production")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deployment-id")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--manifest-map", type=Path)
    parser.add_argument("--skip-context-fit", action="store_true")
    parser.add_argument("--context-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_worker:
        print(json.dumps(_context_worker(json.load(sys.stdin)), sort_keys=True))
        return
    if not args.run_id or args.output is None:
        raise SystemExit("--run-id and --output are required unless --context-worker is used")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing preflight audit: {output}")
    audit = run_preflight(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(output), "audit_sha256": audit["audit_sha256"], "failures": audit["failures"]}, indent=2))
    raise SystemExit(0 if audit["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
