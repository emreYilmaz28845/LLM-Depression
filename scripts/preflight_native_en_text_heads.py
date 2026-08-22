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
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--project-root", type=Path,
                        default=Path("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression"))
    parser.add_argument("--qwen-model-path", type=Path,
                        default=Path("/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"))
    parser.add_argument("--gemma-model-path", type=Path,
                        default=Path("/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it"))
    parser.add_argument("--gemma-revision",
                        default="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
    parser.add_argument("--dataset-roots", nargs="*", default=None)
    parser.add_argument("--run-names", nargs="*", default=None)
    parser.add_argument("--merged-run-ids", nargs="*", default=None)
    parser.add_argument("--with-tokenizer", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    details: dict[str, Any] = {}

    local_failures, local_details = check_local()
    failures.extend(local_failures)
    details["local"] = local_details

    if args.mode == "mn5":
        dep_failures, dep_details = check_deployment_identity(
            expected_commit=args.expected_commit
        )
        failures.extend(dep_failures)
        details["deployment_identity"] = dep_details

        env_failures, env_details = check_environment_imports()
        failures.extend(env_failures)
        details["environment_imports"] = env_details

        model_failures, model_details = check_model_snapshots(
            qwen_path=args.qwen_model_path,
            gemma_path=args.gemma_model_path,
            gemma_revision=args.gemma_revision,
        )
        failures.extend(model_failures)
        details["model_snapshots"] = model_details

        root_failures, root_details = check_dataset_roots(args.dataset_roots or [])
        failures.extend(root_failures)
        details["dataset_roots"] = root_details

        manifest_failures, manifest_details = check_manifests_and_splits(
            project_root=args.project_root
        )
        failures.extend(manifest_failures)
        details["native_manifests"] = manifest_details

        merged_failures, merged_details = check_merged_protocols(
            project_root=args.project_root
        )
        failures.extend(merged_failures)
        details["merged_protocols"] = merged_details

        matrix_failures, matrix_details = check_job_matrix()
        failures.extend(matrix_failures)
        details["job_matrix"] = matrix_details

        collision_failures, collision_details = check_output_collisions(
            project_root=args.project_root,
            run_names=args.run_names or [],
            merged_run_ids=args.merged_run_ids or [],
        )
        failures.extend(collision_failures)
        details["collision_scan"] = collision_details

        qualifier_failures, qualifier_details = check_qualifiers()
        failures.extend(qualifier_failures)
        details["qualifiers"] = qualifier_details

        if args.with_tokenizer:
            tok_failures, tok_details = check_tokenizer_fit(
                project_root=args.project_root,
                qwen_path=args.qwen_model_path,
                gemma_path=args.gemma_model_path,
            )
            failures.extend(tok_failures)
            details["tokenizer_fit"] = tok_details

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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_deployment_identity(*, expected_commit: str | None) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    provenance_dir = PROJECT_ROOT / ".provenance"
    commit_path = provenance_dir / "git_commit.txt"
    manifest_path = provenance_dir / "source_manifest.json"
    if not commit_path.is_file():
        failures.append(f"deployment provenance missing: {commit_path}")
        return failures, details
    commit = commit_path.read_text(encoding="utf-8").strip()
    details["git_commit"] = commit
    if not manifest_path.is_file():
        failures.append(f"deployment source manifest missing: {manifest_path}")
    else:
        payload = _read_json(manifest_path)
        details["source_manifest_files"] = len(payload.get("files", []))
    if expected_commit and commit != expected_commit:
        failures.append(f"deployed commit {commit} != expected {expected_commit}")
    return failures, details


def check_environment_imports() -> tuple[list[str], dict[str, Any]]:
    from src.features import optuna100_policy as policy

    failures: list[str] = []
    details: dict[str, Any] = {}
    try:
        import optuna

        details["optuna"] = optuna.__version__
        if str(optuna.__version__) != policy.OPTUNA_VERSION:
            failures.append(f"optuna {optuna.__version__} != pinned {policy.OPTUNA_VERSION}")
    except Exception as exc:
        failures.append(f"optuna import failed: {exc}")
    try:
        import xgboost

        details["xgboost"] = xgboost.__version__
        if str(xgboost.__version__) != policy.XGBOOST_VERSION:
            failures.append(f"xgboost {xgboost.__version__} != pinned {policy.XGBOOST_VERSION}")
    except Exception as exc:
        failures.append(f"xgboost import failed: {exc}")
    try:
        import sklearn

        details["sklearn"] = sklearn.__version__
    except Exception as exc:
        failures.append(f"sklearn import failed: {exc}")
    return failures, details


def check_model_snapshots(
    *, qwen_path: Path, gemma_path: Path, gemma_revision: str
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    gemma_snapshot = gemma_path / gemma_revision
    for label, path in (("qwen2_7b_instruct", qwen_path), ("gemma4_revision", gemma_snapshot)):
        config = path / "config.json"
        if not path.is_dir():
            failures.append(f"{label} snapshot dir missing: {path}")
        elif not config.is_file():
            failures.append(f"{label} snapshot has no config.json: {config}")
    details = {
        "qwen": str(qwen_path),
        "gemma": str(gemma_snapshot),
        "gemma_revision_expected": gemma_revision,
    }
    return failures, details


def check_dataset_roots(entries: list[str]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    import os

    canonical = {
        "DAIC_DATASET_ROOT",
        "D3TEC_DATASET_ROOT",
        "CMDC_DATASET_ROOT",
        "TURKISH_DATASET_ROOT",
    }
    provided = {}
    for entry in entries:
        key, _, value = entry.partition("=")
        if key in canonical:
            provided[key] = value
    for key in sorted(canonical):
        value = provided.get(key) or os.environ.get(key)
        if not value:
            failures.append(f"dataset root {key} is neither flagged nor set in the environment")
            details[key] = None
            continue
        details[key] = value
        if not Path(value).is_dir():
            failures.append(f"{key}={value} does not exist on this filesystem")
    return failures, details


_STANDALONE_MANIFEST_DIRS = {
    ("d3tec", False): "outputs/manifests_harmonized/d3tec/d3tec_manifest.jsonl",
    ("d3tec", True): "outputs/manifests_harmonized_en/d3tec/d3tec_manifest.jsonl",
    ("androids_interview", False): "outputs/manifests_harmonized/androids/androids_interview_manifest.jsonl",
    ("androids_interview", True): "outputs/manifests_harmonized_en/androids/androids_interview_manifest.jsonl",
    ("cmdc", False): "outputs/manifests_harmonized/cmdc/cmdc_manifest.jsonl",
    ("cmdc", True): "outputs/manifests_harmonized_en/cmdc/cmdc_manifest.jsonl",
    ("turkish", False): "outputs/manifests_harmonized/turkish_t17_qwen3asr/turkish_manifest.jsonl",
    ("turkish", True): "outputs/manifests_harmonized_en/turkish_t17_qwen3asr/turkish_manifest.jsonl",
}


def check_manifests_and_splits(project_root: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    for (dataset, english), rel in sorted(_STANDALONE_MANIFEST_DIRS.items()):
        manifest = project_root / rel
        if not manifest.is_file():
            failures.append(f"missing manifest {rel}")
            continue
        metadata_rel = rel.replace("manifests_harmonized", "splits_harmonized").rsplit("/", 1)[0]
        metadata_name = manifest.stem.replace("_manifest", "") + "_manifest_metadata.json"
        metadata = project_root / metadata_rel / metadata_name
        if not metadata.is_file():
            failures.append(f"missing split metadata {metadata.name} for {dataset}{' EN' if english else ''}")
            continue
        payload = _read_json(metadata)
        split_seed = payload.get("split_seed")
        row_count = int(payload.get("manifest_row_count", -1))
        rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
        entry = {
            "rows": rows,
            "metadata_row_count": row_count,
            "split_seed": split_seed,
            "build_signature_present": bool(payload.get("build_signature")),
        }
        details[f"{dataset}{'_en' if english else ''}"] = entry
        if row_count != rows:
            failures.append(f"{rel}: row count {rows} != metadata {row_count}")
        if split_seed is None or int(split_seed) != 1337:
            failures.append(f"{dataset}{' EN' if english else ''}: split_seed must be 1337")
        if english:
            sample_rows = [json.loads(line) for _, line in zip(range(5), manifest.open(encoding="utf-8"))]
            bad_variant = [r for r in sample_rows if str(r.get("transcript_variant") or "english") != "english"]
            missing_hash = [r for r in sample_rows if not r.get("translation_sha256")]
            if bad_variant or missing_hash:
                failures.append(
                    f"{rel}: english rows must carry transcript_variant=english and translation_sha256"
                )
    return failures, details


def check_merged_protocols(project_root: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    expected_components = {"daic", "cmdc", "turkish", "d3tec", "androids_interview"}
    for (condition, backbone) in sorted(ns.CAMPAIGN_BY_CONDITION_BACKBONE):
        variant = f"{condition}_{backbone}"
        merged_root = project_root / (
            f"outputs/symmetric_merged/native_en_text_heads_v1/{variant}_text_only"
        )
        protocol_path = merged_root / "merged_protocol.json"
        if not protocol_path.is_file():
            failures.append(f"missing merged protocol artifact for {variant}: build it first")
            continue
        payload = _read_json(protocol_path)
        if str(payload.get("schema_version")) != "symmetric_merged_protocol.v1":
            failures.append(f"{variant}: unsupported merged protocol schema")
            continue
        audit_status = (payload.get("split_audit") or {}).get("status")
        if audit_status != "passed":
            failures.append(f"{variant}: merged split audit status={audit_status!r}")
        seed = (payload.get("protocol") or {}).get("seed") or payload.get("seed")
        if seed is None or int(seed) != 1337:
            failures.append(f"{variant}: merged protocol split seed must be 1337")
        components = set((payload.get("manifest") or {}).get("component_manifest_hashes") or {})
        if components != expected_components:
            failures.append(f"{variant}: merged components {sorted(components)} != five locked datasets")
        details[variant] = {
            "split_hash": (payload.get("protocol") or {}).get("split_hash"),
            "components": sorted(components),
            "seed": seed,
        }
    return failures, details


def check_job_matrix() -> tuple[list[str], dict[str, int | list[int]], dict[str, int]] | tuple[list[str], dict[str, Any]]:
    panels = len(ns.CONDITIONS) * len(ns.BACKBONES) * len(ns.STUDY_SEEDS)
    planned = {
        "standalone_jobs": panels * len(ns.STANDALONE_DATASETS) * 5 * 4,
        "merged_cv_jobs": panels * 5 * 4,
        "merged_final_jobs": panels * 4,
        "smoke_jobs": 32,
    }
    failures: list[str] = []
    if planned["standalone_jobs"] != 960:
        failures.append(f"standalone expansion {planned['standalone_jobs']} != locked 960")
    if planned["merged_cv_jobs"] != 240:
        failures.append(f"merged CV expansion {planned['merged_cv_jobs']} != locked 240")
    if planned["merged_final_jobs"] != 48:
        failures.append(f"final expansion {planned['merged_final_jobs']} != locked 48")
    return failures, {"planned": planned, "panels": panels}


def run_name_for(condition: str, backbone: str, dataset: str, seed: int) -> str:
    cond = "nat" if condition == "native" else "en"
    return f"tnh-{cond}-{backbone}-{dataset}-s{int(seed)}"


def merged_run_id_for(condition: str, backbone: str, seed: int) -> str:
    cond = "nat" if condition == "native" else "en"
    return f"tmh-{cond}-{backbone}-s{int(seed)}"


def check_output_collisions(
    *,
    project_root: Path,
    run_names: list[str],
    merged_run_ids: list[str],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    existing: list[str] = []
    for run_name in run_names:
        base = project_root / "output_model"
        matches = [
            str(path)
            for campaign in ns.CAMPAIGN_BY_CONDITION_BACKBONE.values()
            for dataset in ns.STANDALONE_DATASETS
            for path in [(base / campaign / "text_only" / dataset / run_name)]
            if path.exists()
        ]
        existing.extend(matches)
    for run_id in merged_run_ids:
        for condition in ns.CONDITIONS:
            for backbone in ns.BACKBONES:
                root = project_root / (
                    f"output_model/symmetric_merged/native_en_text_heads_v1/"
                    f"{condition}_{backbone}_text_only/{run_id}"
                )
                if root.exists():
                    existing.append(str(root))
    if existing:
        failures.append(f"output collision(s): {existing[:10]}")
    return failures, {"checked_run_names": len(run_names), "checked_merged_run_ids": len(merged_run_ids), "existing": existing[:10]}


def check_qualifiers() -> tuple[list[str], dict[str, Any]]:
    """Every production submission must resolve an explicit evaluation view."""

    failures: list[str] = []
    details: dict[str, Any] = {"need_view_override": [], "carry_view": []}
    for (condition, backbone), dataset_map in STANDALONE_CONFIGS.items():
        for dataset, rel in dataset_map.items():
            doc = yaml.safe_load((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
            view = (doc.get("evaluation") or {}).get("evaluation_view")
            if view:
                details["carry_view"].append(rel)
                if view != "harmonized_all_windows_full_coverage":
                    failures.append(f"{rel}: unexpected view {view!r}")
            else:
                # Managed submissions pass --set evaluation.evaluation_view=…
                details["need_view_override"].append(rel)
    return failures, details


def check_tokenizer_fit(
    *, project_root: Path, qwen_path: Path, gemma_path: Path
) -> tuple[list[str], dict[str, Any]]:
    """Tokenize the longest sampled transcript per dataset with each backbone tokenizer."""

    failures: list[str] = []
    details: dict[str, Any] = {}
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        failures.append(f"transformers unavailable for tokenizer fit: {exc}")
        return failures, details
    tokenizers = {}
    for name, path in (("qwen", qwen_path), ("gemma4", gemma_path / "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")):
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        except Exception as exc:
            failures.append(f"{name} tokenizer load failed: {exc}")
    if len(tokenizers) != 2:
        return failures, details
    limit = 8192 - 512  # prompt-template + label budget
    for (dataset, english), rel in sorted(_STANDALONE_MANIFEST_DIRS.items()):
        if not english:
            continue
        manifest = project_root / rel
        longest = ""
        scanned = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = str(row.get("transcript") or "")
                if len(text) > len(longest):
                    longest = text
                scanned += 1
                if scanned >= 400:
                    break
        if not longest:
            continue
        per_backend = {}
        for name, tok in tokenizers.items():
            n_tokens = len(tok(longest, add_special_tokens=False)["input_ids"])
            per_backend[name] = n_tokens
            if n_tokens > limit:
                failures.append(
                    f"{dataset}: longest sampled transcript needs {n_tokens} tokens "
                    f"for {name}; budget {limit}"
                )
        details[dataset] = {"sampled_rows": scanned, **per_backend}
    return failures, details
