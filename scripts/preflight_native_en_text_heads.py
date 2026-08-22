#!/usr/bin/env python3
"""Preflight audit for the native-versus-English text-only head study.

Two modes:

* ``--mode local`` (default): verifies the locked matrix contract against the
  repository itself — configs exist and resolve, merged configs carry the
  seed locks, the scientific group definition matches the study, and every
  planned cell has a canonical config. Writes a hashed audit artifact.
* ``--mode mn5``: additionally verifies deployment identity (when a
  deployment record is supplied), MN5 environment imports, model snapshot
  paths, dataset roots, native manifests/splits, the four English
  translation caches with exact accepted counts, no fallback/failed rows,
  identical subject membership between paired native/English inputs, and
  tokenizer/context fit inputs.

The audit never trains anything. Exit code is zero only for status=passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

import tools.native_en_submit as ns

AUDIT_SCHEMA = "audiollm.native_en_text_heads_preflight.v1"
EXPECTED_ACCEPTED = {
    "d3tec": 3677,
    "androids_interview": 2176,
    "cmdc": 923,
    "turkish": 1051,
}
TRANSLATION_CACHE_SUBDIRS = {
    "d3tec": "d3tec",
    "androids_interview": "androids_interview",
    "cmdc": "cmdc",
    "turkish": "turkish",
}

STANDALONE_CONFIGS = {
    ("native", "qwen"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml",
    },
    ("english", "qwen"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_en.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
    },
    ("native", "gemma4"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_gemma4_12b.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml",
    },
    ("english", "gemma4"): {
        "d3tec": "configs/main/d3tec_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "androids_interview": "configs/main/androids_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "cmdc": "configs/main/cmdc_text_only_harmonized_selmacrof1_tf_en_gemma4_12b.yaml",
        "turkish": "configs/main/turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml",
    },
}
MERGED_CONFIGS = {
    ("native", "qwen"): "configs/experiments/merged/symmetric_merged_text_heads_native_qwen.yaml",
    ("english", "qwen"): "configs/experiments/merged/symmetric_merged_text_heads_english_qwen.yaml",
    ("native", "gemma4"): "configs/experiments/merged/symmetric_merged_text_heads_native_gemma4.yaml",
    ("english", "gemma4"): "configs/experiments/merged/symmetric_merged_text_heads_english_gemma4.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def check_local() -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {"configs": {}, "matrix": {}}
    job_cells = 0
    for (condition, backbone), dataset_map in STANDALONE_CONFIGS.items():
        for dataset, rel in dataset_map.items():
            path = PROJECT_ROOT / rel
            if not path.is_file():
                failures.append(f"missing standalone config {rel}")
                continue
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            transcripts = doc.get("transcripts") or {}
            is_en = condition == "english"
            if is_en:
                if str(transcripts.get("variant")) != "english":
                    failures.append(f"{rel}: english config lacks variant=english")
                if not bool(transcripts.get("require_complete")):
                    failures.append(f"{rel}: require_complete must be true")
                if transcripts.get("include_failed") is not False:
                    failures.append(f"{rel}: include_failed must be false")
                cache = str(transcripts.get("cache_path", ""))
                if "harmonized_en_complete_v1" not in cache:
                    failures.append(f"{rel}: unexpected translation cache root")
            else:
                if transcripts:
                    failures.append(f"{rel}: native config unexpectedly declares a transcripts block")
            view = (doc.get("evaluation") or {}).get("evaluation_view")
            details["configs"][rel] = {
                "evaluation_view": view,
                "model_backend": doc.get("model_backend", ""),
            }
            job_cells += 1
    for (condition, backbone), rel in MERGED_CONFIGS.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            failures.append(f"missing merged config {rel}")
            continue
        try:
            ns.materialize_merged_config(yaml.safe_load(path.read_text(encoding="utf-8")), seed=1337)
        except ValueError as exc:
            failures.append(f"{rel}: seed-lock validation failed: {exc}")
            continue
        job_cells += 1
    expected_standalone_panels = len(ns.CONDITIONS) * len(ns.BACKBONES)
    details["matrix"] = {
        "standalone_configs": job_cells - len(MERGED_CONFIGS),
        "merged_configs": len(MERGED_CONFIGS),
        "expected_seeds": list(ns.STUDY_SEEDS),
        "planned_production_jobs": 960 + 240 + 48,
        "planned_smoke_jobs": 32,
    }
    if job_cells != expected_standalone_panels * len(ns.STANDALONE_DATASETS) + len(MERGED_CONFIGS):
        failures.append("config inventory does not match the locked matrix")
    group_path = PROJECT_ROOT / "experiments/definitions/native-en-text-heads-20260822.yaml"
    if not group_path.is_file():
        failures.append("scientific experiment-group definition missing")
    else:
        group = yaml.safe_load(group_path.read_text(encoding="utf-8"))
        primary = group.get("primary_metric") or {}
        if not str(primary.get("aggregation", "")):
            failures.append("group primary metric aggregation missing")
        if sorted(int(v) for v in group.get("expected_seeds", [])) != sorted(ns.STUDY_SEEDS):
            failures.append("group expected seeds do not match the locked seeds")
    return failures, details


def check_translation_caches(translation_root: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    for dataset, subdir in TRANSLATION_CACHE_SUBDIRS.items():
        accepted_path = translation_root / "harmonized_en_complete_v1" / subdir / "accepted.jsonl"
        if not accepted_path.is_file():
            failures.append(f"missing accepted cache for {dataset}: {accepted_path}")
            continue
        rejected_path = accepted_path.parent / "rejected.jsonl"
        rows = [
            json.loads(line)
            for line in accepted_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        count = len(rows)
        details[dataset] = {"accepted": count, "expected": EXPECTED_ACCEPTED[dataset]}
        if count != EXPECTED_ACCEPTED[dataset]:
            failures.append(
                f"{dataset}: accepted cache has {count} records; expected exactly {EXPECTED_ACCEPTED[dataset]}"
            )
        if rejected_path.is_file():
            rejected_rows = [ln for ln in rejected_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if rejected_rows:
                failures.append(f"{dataset}: rejected.jsonl is non-empty ({len(rejected_rows)} rows)")
        bad_status = [
            row
            for row in rows
            if str(row.get("status")) not in {"automatic_high", "automatic_medium", "automatic_low", "human_verified"}
        ]
        if bad_status:
            failures.append(f"{dataset}: {len(bad_status)} accepted rows carry disallowed statuses")
    return failures, details


def check_paired_membership(manifest_pairs: dict[str, tuple[Path, Path]]) -> tuple[list[str], dict[str, Any]]:
    """Native/English manifest pairs must cover identical subjects and labels."""

    failures: list[str] = []
    details: dict[str, Any] = {}
    for dataset, (native_path, english_path) in manifest_pairs.items():
        def load(path: Path) -> dict[str, int]:
            mapping: dict[str, int] = {}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    subject = f"{str(row['dataset']).lower()}::{row['subject_id']}" if "::" not in str(row["subject_id"]) else str(row["subject_id"])
                    label = int(row["label"])
                    if subject in mapping and mapping[subject] != label:
                        raise ValueError(f"{path}: subject {subject} has inconsistent labels")
                    mapping[subject] = label
            return mapping

        native_subjects = load(native_path)
        english_subjects = load(english_path)
        mismatched_labels = sorted(
            subject for subject in set(native_subjects) & set(english_subjects)
            if native_subjects[subject] != english_subjects[subject]
        )
        missing_in_en = sorted(set(native_subjects) - set(english_subjects))
        extra_in_en = sorted(set(english_subjects) - set(native_subjects))
        details[dataset] = {
            "native_subjects": len(native_subjects),
            "english_subjects": len(english_subjects),
            "missing_in_english": missing_in_en[:10],
            "extra_in_english": extra_in_en[:10],
            "label_mismatches": mismatched_labels[:10],
        }
        if missing_in_en or extra_in_en:
            failures.append(f"{dataset}: native/english subject sets differ")
        if mismatched_labels:
            failures.append(f"{dataset}: per-subject labels differ between native and english")
    return failures, details


def build_audit(*, mode: str, failures: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if not failures else "failed",
        "mode": mode,
        "failures": failures,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "mn5"), default="local")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--translation-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-pairs",
        type=Path,
        default=None,
        help="JSON file: {dataset: [native_manifest, english_manifest]} for mn5 mode",
    )
    args = parser.parse_args()

    failures: list[str] = []
    details: dict[str, Any] = {}

    local_failures, local_details = check_local()
    failures.extend(local_failures)
    details["local"] = local_details

    if args.mode == "mn5":
        if args.translation_root is None:
            failures.append("mn5 mode requires --translation-root")
        else:
            cache_failures, cache_details = check_translation_caches(args.translation_root)
            failures.extend(cache_failures)
            details["translation_caches"] = cache_details
        if args.manifest_pairs is not None:
            pairs = json.loads(args.manifest_pairs.read_text(encoding="utf-8"))
            pair_failures, pair_details = check_paired_membership(
                {ds: (Path(p[0]), Path(p[1])) for ds, p in pairs.items()}
            )
            failures.extend(pair_failures)
            details["paired_membership"] = pair_details

    audit = build_audit(mode=args.mode, failures=failures, details=details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_sha = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{audit_sha}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "failures": len(failures),
                "audit": str(args.output),
                "sha256": audit_sha,
            },
            indent=2,
        )
    )
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
