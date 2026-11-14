#!/usr/bin/env python3
"""Deterministic read-only regression audit for merged component outer folds.

Rebuilds each merged route protocol with the current lane code (production path
resolution: ``--input-root`` rewrites relative component manifest/metadata paths,
``--pooled-runtime-root`` repoints the pooled Turkish component exactly like the
submitter) and checks:

* ``androids_interview``: exact official outer membership (the corrected
  baseline), with official sizes 24/23/23/23/23 over 116 subjects.
* ``cmdc``, ``d3tec``, ``turkish``: exact official outer membership preserved.
* ``daic``: the intentional fixed-development-pool signature preserved.
* When a historical protocol directory is supplied, non-Androids component
  memberships must equal the historical protocol memberships exactly (no
  semantic split change); Androids is expected to differ (the correction).

Missing components can be skipped with ``--allow-missing-components`` for local
partial audits; strict mode (default) fails on any missing component and is the
mandatory pre-production check on the resolved deployment. Writes a JSON audit
and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROUTES = {
    "native_text_only": "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_text_only.yaml",
    "native_audio_only": "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_audio_only.yaml",
    "native_audio_text": "configs/experiments/merged/symmetric_merged_qwen3_pooled_native_audio_text.yaml",
    "english_text_only": "configs/experiments/merged/symmetric_merged_qwen3_pooled_english_text_only.yaml",
    "english_audio_text": "configs/experiments/merged/symmetric_merged_qwen3_pooled_english_audio_text.yaml",
}
DATASETS = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")
ANDROIDS_OFFICIAL_SIZES = [24, 23, 23, 23, 23]
DAIC_FIXED_POOL_SIZES = [29, 29, 28, 28, 28]
HISTORICAL_CAMPAIGN = {
    "native": "qwen3_pooled_native",
    "english": "qwen3_pooled_english",
}
HISTORICAL_MODALITY = {
    "text_only": "text_only",
    "audio_only": "audio_only",
    "audio_text": "audio_text",
}


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def strip_namespace(values) -> set[str]:
    return {
        str(value).split("::", 1)[1] if "::" in str(value) else str(value) for value in values
    }


def load_official_folds(path: Path) -> dict[int, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds: dict[int, set[str]] = {}
    for fold, item in payload.items():
        holdout = item.get("final_eval_subject_ids") or item.get("outer_holdout_subject_ids")
        folds[int(fold)] = strip_namespace(holdout or [])
    return folds


def component_new_membership(protocol: dict, dataset: str) -> dict[int, set[str]]:
    folds = protocol["components"][dataset]["folds"]
    out: dict[int, set[str]] = {}
    for fold in range(5):
        payload = folds[str(fold)]
        raw = payload.get("component_outer_holdout_subject_ids")
        out[fold] = strip_namespace(raw if raw is not None else payload["outer_holdout_subject_ids"])
    return out


def component_historical_membership(
    protocol_payload: dict, dataset: str
) -> dict[int, set[str]] | None:
    components = (protocol_payload.get("protocol") or {}).get("components") or {}
    if dataset not in components:
        return None
    folds = components[dataset]["folds"]
    out: dict[int, set[str]] = {}
    for fold in range(5):
        payload = folds[str(fold)]
        raw = payload.get("component_outer_holdout_subject_ids")
        out[fold] = strip_namespace(raw if raw is not None else payload["outer_holdout_subject_ids"])
    return out


def rewrite_component_paths(
    config: dict, input_root: Path, pooled_runtime_root: Path | None
) -> dict:
    """Replicate the submitter's read-only input repointing (dry, no mutation)."""

    payload = json.loads(json.dumps(config))
    for component in payload.get("components") or []:
        for field in ("manifest_path", "metadata_path"):
            raw = str(component.get(field) or "")
            if raw and not Path(raw).is_absolute():
                component[field] = str(input_root / raw)
        if pooled_runtime_root is not None and str(component.get("name")) == "turkish":
            english = "harmonized_en" in str(component.get("manifest_path", ""))
            manifest_root = pooled_runtime_root / ("manifests_en" if english else "manifests")
            split_root = pooled_runtime_root / ("splits_en" if english else "splits")
            component["manifest_path"] = str(
                manifest_root / "turkish" / "turkish_manifest.jsonl"
            )
            component["metadata_path"] = str(
                split_root / "turkish" / "turkish_manifest_metadata.json"
            )
    return payload


def component_config_path(value: str) -> Path:
    """Component configs ship with the audited code, not with the input root."""

    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pooled-runtime-root", default=None)
    parser.add_argument("--historical-protocol-dir", default=None)
    parser.add_argument("--routes", nargs="*", default=list(ROUTES))
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing-components", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    os.environ["PROJECT_ROOT"] = str(input_root)

    from src.utils import load_yaml_with_overrides, read_json, read_jsonl
    from src.merged.protocol import (
        _label_map,
        _resolve_component_folds_path,
        _resolve_component_path,
        build_component_outer_folds,
    )

    failures: list[str] = []
    audit: dict = {
        "schema_version": "audiollm.merged_official_folds_regression_audit.v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(input_root),
        "pooled_runtime_root": args.pooled_runtime_root,
        "historical_protocol_dir": args.historical_protocol_dir,
        "allow_missing_components": bool(args.allow_missing_components),
        "routes": {},
        "failures": failures,
    }
    try:
        audit["lane_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - informational only
        audit["lane_commit"] = None

    for route in args.routes:
        if route not in ROUTES:
            failures.append(f"unknown route {route}")
            continue
        config = load_yaml_with_overrides(REPO_ROOT / ROUTES[route], [])
        config = rewrite_component_paths(
            config,
            input_root,
            Path(args.pooled_runtime_root) if args.pooled_runtime_root else None,
        )
        entry: dict = {"components": {}, "status": "passed"}
        components = list(config.get("components") or [])
        missing = [
            str(component.get("name"))
            for component in components
            if not Path(str(component.get("manifest_path"))).is_file()
        ]
        entry["skipped_components"] = sorted(set(missing))
        if missing and not args.allow_missing_components:
            failures.append(f"{route}: missing component inputs: {sorted(set(missing))}")
            entry["status"] = "failed"
            audit["routes"][route] = entry
            continue
        available = [
            component
            for component in components
            if Path(str(component.get("manifest_path"))).is_file()
        ]
        if not available:
            failures.append(f"{route}: no component inputs available")
            entry["status"] = "failed"
            audit["routes"][route] = entry
            continue

        for component in available:
            dataset = str(component.get("name")).lower()
            detail: dict = {}
            metadata_path = _resolve_component_path(component["metadata_path"])
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            declared = metadata.get("folds_path")
            resolved = _resolve_component_folds_path(declared, metadata_path) if declared else None
            folds = read_json(resolved) if resolved is not None else {}
            rows = read_jsonl(_resolve_component_path(component["manifest_path"]))
            official_test = {
                str(row["subject_id"])
                for row in rows
                if str(row.get("split_original", "")).lower() in {"test", "official_test"}
            }
            official_test.update(
                str(value) for value in component.get("official_test_subject_ids", [])
            )
            if metadata.get("subject_partition_path"):
                partition_path = _resolve_component_path(metadata["subject_partition_path"])
                if partition_path.is_file():
                    official_test.update(
                        str(row["subject_id"])
                        for row in read_json(partition_path)
                        if str(row.get("partition", "")).lower() == "test"
                    )
            record = {
                "dataset": dataset,
                "config_path": str(component.get("config")),
                "config": load_yaml_with_overrides(
                    component_config_path(str(component.get("config"))), []
                ),
                "manifest_hash": None,
                "rows": rows,
                "labels": _label_map(rows),
                "folds": folds,
                "folds_path": str(resolved) if resolved else None,
                "folds_path_declared": str(declared) if declared else None,
                "official_test_subject_ids": sorted(official_test),
            }
            detail["folds_path_declared"] = record["folds_path_declared"]
            detail["folds_path_resolved"] = record["folds_path"]
            detail["official_test_count"] = len(official_test)
            try:
                new_folds = build_component_outer_folds(record, seed=1337)
            except ValueError as exc:
                detail["status"] = "failed"
                detail["error"] = str(exc)
                failures.append(f"{route}/{dataset}: {exc}")
                entry["components"][dataset] = detail
                entry["status"] = "failed"
                continue
            new_membership = {
                fold: set(payload["final_eval_subject_ids"])
                for fold, payload in new_folds.items()
            }
            declared_value = record.get("folds_path_declared")
            official_path = None
            if declared_value:
                candidate = Path(declared_value)
                if not candidate.is_absolute():
                    candidate = input_root / candidate
                if candidate.is_file():
                    official_path = candidate
            detail["official_folds_path"] = str(official_path) if official_path else None
            detail["official_folds_sha256"] = sha256(official_path) if official_path else None
            if official_path is not None:
                official = load_official_folds(official_path)
                detail["official_sizes"] = [len(official[fold]) for fold in range(5)]
                detail["new_sizes"] = [len(new_membership[fold]) for fold in range(5)]
                detail["per_fold_intersection"] = [
                    len(new_membership[fold] & official[fold]) for fold in range(5)
                ]
                detail["per_fold_symmetric_diff"] = [
                    len(new_membership[fold] ^ official[fold]) for fold in range(5)
                ]
                if dataset == "daic":
                    if detail["new_sizes"] != DAIC_FIXED_POOL_SIZES:
                        failures.append(f"{route}/daic: fixed-pool sizes {detail['new_sizes']}")
                        detail["status"] = "failed"
                    else:
                        detail["status"] = "passed"
                        detail["note"] = "intentional fixed-development-pool folds"
                elif dataset == "androids_interview":
                    if detail["official_sizes"] != ANDROIDS_OFFICIAL_SIZES:
                        failures.append(
                            f"{route}/androids_interview: official sizes {detail['official_sizes']}"
                        )
                        detail["status"] = "failed"
                    elif any(new_membership[fold] != official[fold] for fold in range(5)):
                        failures.append(f"{route}/androids_interview: membership mismatch")
                        detail["status"] = "failed"
                    else:
                        detail["status"] = "passed"
                else:
                    if any(new_membership[fold] != official[fold] for fold in range(5)):
                        failures.append(f"{route}/{dataset}: official membership mismatch")
                        detail["status"] = "failed"
                    else:
                        detail["status"] = "passed"
            else:
                detail["status"] = "unverified_no_official_file"
                failures.append(
                    f"{route}/{dataset}: official folds file not found ({declared_value})"
                )
            if args.historical_protocol_dir:
                camp = HISTORICAL_CAMPAIGN["native" if route.startswith("native") else "english"]
                modality = HISTORICAL_MODALITY[
                    "text_only" if route.endswith("text_only") else ("audio_only" if route.endswith("audio_only") else "audio_text")
                ]
                historical_path = (
                    Path(args.historical_protocol_dir) / camp / modality / "merged_protocol.json"
                )
                if historical_path.is_file():
                    historical_payload = json.loads(
                        historical_path.read_text(encoding="utf-8")
                    )
                    historical = component_historical_membership(historical_payload, dataset)
                    if historical is not None:
                        same = all(
                            new_membership[fold] == historical[fold] for fold in range(5)
                        )
                        detail["historical_protocol_path"] = str(historical_path)
                        detail["historical_sha256"] = sha256(historical_path)
                        detail["historical_identical"] = same
                        if dataset == "androids_interview":
                            if same:
                                failures.append(
                                    f"{route}/androids_interview: membership unexpectedly "
                                    "equals historical regenerated folds"
                                )
                                detail["status"] = "failed"
                        elif not same:
                            failures.append(
                                f"{route}/{dataset}: membership changed vs historical protocol"
                            )
                            detail["status"] = "failed"
            entry["components"][dataset] = detail
        audit["routes"][route] = entry

    audit["status"] = "passed" if not failures else "failed"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": audit["status"], "failures": failures[:10], "out": str(out_path)},
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
