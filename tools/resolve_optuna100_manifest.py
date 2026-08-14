#!/usr/bin/env python3
"""Resolve the exact paired Optuna-100 study manifest from qualified caches.

For every logical cell of a tracked matrix (native / english / officialdev),
for every backend (qwen, gemma4), this tool discovers and qualifies the exact
hidden-feature cache: extraction metadata identity (dataset, modality, fold,
condition, backend), no smoke-cache marker, unambiguous production run, exact
parent attempt/checkpoint identity, and evaluation qualifiers. It refuses
duplicate logical identities, missing required caches in ``--require-caches``
mode, and ambiguous parents.

Expected production counts per matrix (dry-run): native 126 (63 per backend),
english 80 (40 per backend), officialdev 6 (3 per backend). Any other count
is a hard stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hidden_classifier_policy import cache_identity, canonical_sha256  # noqa: E402
from src.utils import load_yaml_with_overrides, read_json, read_jsonl, save_json  # noqa: E402

from src.features import optuna100_policy as policy  # noqa: E402

EXPERIMENT_ID = policy.EXPERIMENT_ID
MATRICES = {
    "native": "configs/experiments/optuna100/native_matrix.yaml",
    "english": "configs/experiments/optuna100/english_matrix.yaml",
    "officialdev": "configs/experiments/optuna100/officialdev_matrix.yaml",
}
EXPECTED_COUNTS = {"native": 126, "english": 80, "officialdev": 6}

QWEN_FEATURES_ROOTS = {
    "native": PROJECT_ROOT / "outputs/hidden_features/harmonized_v1",
    "english": PROJECT_ROOT / "outputs/hidden_features/harmonized_v1_en",
}
GEMMA_FEATURES_ROOTS = {
    "native": PROJECT_ROOT / "outputs/hidden_features/harmonized_v1_gemma4",
    "english": PROJECT_ROOT / "outputs/hidden_features/harmonized_v1_en_gemma4",
}
HEADS_ROOTS = {
    "qwen": {
        "officialdev": PROJECT_ROOT / "output_model/harmonized_v1_officialdev_heads",
    },
    "gemma4": {
        "officialdev": PROJECT_ROOT / "output_model/harmonized_v1_gemma4_officialdev_heads",
        "native": PROJECT_ROOT / "output_model/harmonized_v1_gemma4_heads",
    },
}
RUN_NAME_PREFIXES = {
    ("native", "qwen"): "harmonized_v1_optuna100",
    ("native", "gemma4"): "gemma4_harmonized_v1_optuna100",
    ("english", "qwen"): "harmonized_v1_en_optuna100",
    ("english", "gemma4"): "gemma4_harmonized_v1_en_optuna100",
    ("officialdev", "qwen"): "harmonized_v1_officialdev_optuna100",
    ("officialdev", "gemma4"): "gemma4_harmonized_v1_officialdev_optuna100",
}
ATTEMPT_ROOTS = {
    ("native", "qwen"): "output_model/harmonized_v1_optuna100",
    ("native", "gemma4"): "output_model/harmonized_v1_gemma4_optuna100",
    ("english", "qwen"): "output_model/harmonized_v1_en_optuna100",
    ("english", "gemma4"): "output_model/harmonized_v1_en_gemma4_optuna100",
    ("officialdev", "qwen"): "output_model/harmonized_v1_officialdev_optuna100",
    ("officialdev", "gemma4"): "output_model/harmonized_v1_gemma4_officialdev_optuna100",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "extraction_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"cache has no extraction_metadata.json: {cache_dir}")
    return read_json(path)


def _parent_from_checkpoint_dir(metadata: dict[str, Any]) -> dict[str, Any] | None:
    cache_config = metadata.get("cache_config") or {}
    checkpoint = str(cache_config.get("checkpoint_dir") or "")
    if not checkpoint:
        return None
    marker = "LLM-Depression/"
    if marker in checkpoint:
        checkpoint = str(PROJECT_ROOT / checkpoint.split(marker, 1)[1])
    parent_fold = Path(checkpoint).parent
    metadata_path = parent_fold / "metadata.json"
    parent_attempt = str(cache_config.get("parent_attempt_id") or "")
    if metadata_path.is_file():
        actual = str(read_json(metadata_path).get("attempt_id") or "")
        if parent_attempt and actual and actual != parent_attempt:
            raise ValueError(
                f"cache parent metadata attempt {actual} contradicts cache "
                f"identity {parent_attempt} at {parent_fold}"
            )
        parent_attempt = actual or parent_attempt
    return {
        "parent_attempt_id": parent_attempt or None,
        "parent_fold_dir": _rel(parent_fold),
        "parent_checkpoint_path": _rel(parent_fold / "best_model"),
        "adapter_config_sha256": cache_config.get("adapter_config_sha256"),
        "adapter_sha256": cache_config.get("adapter_sha256"),
    }


def _evaluation_qualifiers(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    provenance = metadata.get("evaluation_provenance") or {}
    support: int | None = None
    rows_path = cache_dir / "final_eval_rows.jsonl"
    if rows_path.is_file():
        support = len(
            {
                str(row.get("subject_id"))
                for row in read_jsonl(rows_path)
            }
        )
    return {
        "dataset": metadata["dataset"],
        "split_name": str(provenance.get("split_name") or ""),
        "split_protocol": str(provenance.get("evaluation_protocol") or ""),
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "subject_level",
        "metric_namespace": "headline/binary_strict",
        "support": support,
    }


def _discover_features_cache(
    root: Path, dataset: str, modality: str, fold: int, backend: str
) -> tuple[Path, dict[str, Any]] | None:
    """Discover the unique production cache under a features root.

    Run dirs named with ``smoke`` are excluded. Exactly one production cache
    per (dataset, modality, fold) may exist; duplicates are ambiguous.
    """
    dataset_root = root / dataset
    if not dataset_root.is_dir():
        return None
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for cache_dir in sorted(dataset_root.glob("*/fold_*/")):
        if "smoke" in cache_dir.parts[-2]:
            continue
        try:
            metadata = _read_cache_metadata(cache_dir)
        except FileNotFoundError:
            continue
        if str(metadata.get("dataset", "")).lower() != dataset.lower():
            continue
        if str(metadata.get("input_modality", "")) != modality:
            continue
        if int(metadata.get("fold", -1)) != fold:
            continue
        cache_backend = str(metadata.get("model_backend") or "").lower()
        if backend == "gemma4" and cache_backend != "gemma4":
            continue
        if backend == "qwen" and cache_backend == "gemma4":
            continue
        candidates.append((cache_dir, metadata))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous caches for {dataset}/{modality}/fold {fold} backend {backend}: "
            + ", ".join(str(path) for path, _ in candidates)
        )
    return candidates[0]


def _discover_heads_cache(
    root: Path, modality: str, dataset: str, fold: int, backend: str
) -> tuple[Path, dict[str, Any]] | None:
    modality_root = root / modality / dataset
    if not modality_root.is_dir():
        return None
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for attempt_dir in sorted(modality_root.glob("*/fold_*/")):
        if "smoke" in str(attempt_dir):
            continue
        status_path = attempt_dir / "status.json"
        if not status_path.is_file():
            continue
        status = read_json(status_path)
        if str(status.get("state", "")) != "REPORTABLE":
            continue
        cache_dir = attempt_dir / "hidden_features"
        try:
            metadata = _read_cache_metadata(cache_dir)
        except FileNotFoundError:
            continue
        if str(metadata.get("dataset", "")).lower() != dataset.lower():
            continue
        if str(metadata.get("input_modality", "")) != modality:
            continue
        if int(metadata.get("fold", -1)) != fold:
            continue
        candidates.append((cache_dir, metadata))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous head-attempt caches for {dataset}/{modality}/fold {fold}: "
            + ", ".join(str(path) for path, _ in candidates)
        )
    return candidates[0]


def resolve_study(
    *,
    matrix: dict[str, Any],
    family: str,
    backend: str,
    dataset: str,
    modality: str,
    fold: int,
    run_id: str,
    merged_sha: str,
    branch: str,
    github_issue: int | None,
    pr: int | None,
    require_caches: bool,
) -> dict[str, Any]:
    cache_dir: Path | None = None
    metadata: dict[str, Any] | None = None
    if dataset == "daic" and family == "officialdev":
        root = HEADS_ROOTS[backend].get("officialdev")
        if root is not None:
            discovered = _discover_heads_cache(root, modality, dataset, fold, backend)
            if discovered is not None:
                cache_dir, metadata = discovered
    elif dataset == "daic" and family == "native" and backend == "gemma4":
        root = HEADS_ROOTS["gemma4"].get("native")
        if root is not None:
            discovered = _discover_heads_cache(root, modality, dataset, fold, backend)
            if discovered is not None:
                cache_dir, metadata = discovered
    if cache_dir is None or metadata is None:
        root = (
            QWEN_FEATURES_ROOTS[family]
            if backend == "qwen"
            else GEMMA_FEATURES_ROOTS[family]
        )
        discovered = _discover_features_cache(root, dataset, modality, fold, backend)
        if discovered is not None:
            cache_dir, metadata = discovered

    if cache_dir is None or metadata is None:
        if require_caches:
            raise FileNotFoundError(
                f"missing qualified cache for {family}/{backend}/{dataset}/{modality}/fold {fold}"
            )
        return {
            "schema_version": "audiollm.posthoc_head_task.v1",
            "dataset": dataset,
            "modality": modality,
            "condition": modality,
            "fold": fold,
            "seed": 1337,
            "family": family,
            "backend": backend,
            "cache_dir": None,
            "cache_missing": True,
            "experiment_id": EXPERIMENT_ID,
            "objective": matrix.get("objective", "macro_f1"),
            "target_trials": int(matrix.get("target_trials", 100)),
            "group_id": f"{family}-optuna100-{run_id}",
            "logical_run_name": (
                f"{RUN_NAME_PREFIXES[(family, backend)]}_{dataset}_{modality}_seed1337"
            ),
            "run_name": (
                f"{RUN_NAME_PREFIXES[(family, backend)]}_{run_id}_{dataset}_{modality}"
            ),
            "attempt_dir": None,
            "branch": branch,
            "merged_sha": merged_sha,
            "github_issue": github_issue,
            "pr": pr,
        }

    cache_config = metadata.get("cache_config") or {}
    parent = _parent_from_checkpoint_dir(metadata)
    if parent is None:
        raise ValueError(
            f"cache without checkpoint parent identity: {cache_dir} "
            "(cache_config.checkpoint_dir missing)"
        )
    if parent["parent_attempt_id"] is None:
        raise ValueError(
            f"cache parent attempt identity unresolved: {cache_dir} "
            "(no parent metadata.json and no cache_config.parent_attempt_id)"
        )
    qualifiers = _evaluation_qualifiers(cache_dir, metadata)
    return {
        "schema_version": "audiollm.posthoc_head_task.v1",
        "dataset": dataset,
        "modality": modality,
        "condition": str(metadata.get("condition") or modality),
        "fold": fold,
        "seed": 1337,
        "family": family,
        "backend": backend,
        "cache_dir": _rel(cache_dir),
        "cache_identity_sha256": canonical_sha256(cache_identity(cache_dir)),
        "cache_missing": False,
        "parent": parent,
        "evaluation_qualifiers": qualifiers,
        "experiment_id": EXPERIMENT_ID,
        "objective": matrix.get("objective", "macro_f1"),
        "target_trials": int(matrix.get("target_trials", 100)),
        "group_id": f"{family}-optuna100-{run_id}",
        "logical_run_name": (
            f"{RUN_NAME_PREFIXES[(family, backend)]}_{dataset}_{modality}_seed1337"
        ),
        "run_name": f"{RUN_NAME_PREFIXES[(family, backend)]}_{run_id}_{dataset}_{modality}",
        "attempt_dir": _rel(
            PROJECT_ROOT
            / ATTEMPT_ROOTS[(family, backend)]
            / modality
            / dataset
            / f"{RUN_NAME_PREFIXES[(family, backend)]}_{run_id}_{dataset}_{modality}"
            / f"fold_{fold}"
            / EXPERIMENT_ID
        ),
        "branch": branch,
        "merged_sha": merged_sha,
        "github_issue": github_issue,
        "pr": pr,
        "checkpoint_hashes": {
            "adapter_config_sha256": cache_config.get("adapter_config_sha256"),
            "adapter_sha256": cache_config.get("adapter_sha256"),
            "saved_run_config_sha256": cache_config.get("saved_run_config_sha256"),
            "saved_split_sha256": cache_config.get("saved_split_sha256"),
            "manifest_sha256": cache_config.get("manifest_sha256"),
            "split_metadata_sha256": cache_config.get("split_metadata_sha256"),
        },
    }


def resolve(
    *,
    family: str,
    run_id: str,
    merged_sha: str,
    branch: str,
    github_issue: int | None,
    pr: int | None,
    require_caches: bool,
) -> dict[str, Any]:
    matrix_path = PROJECT_ROOT / MATRICES[family]
    matrix = load_yaml_with_overrides(matrix_path, [])
    studies: list[dict[str, Any]] = []
    for backend in ("qwen", "gemma4"):
        for cell in matrix["studies"]:
            for fold in cell["folds"]:
                studies.append(
                    resolve_study(
                        matrix=matrix,
                        family=family,
                        backend=backend,
                        dataset=cell["dataset"],
                        modality=cell["modality"],
                        fold=int(fold),
                        run_id=run_id,
                        merged_sha=merged_sha,
                        branch=branch,
                        github_issue=github_issue,
                        pr=pr,
                        require_caches=require_caches,
                    )
                )
    per_backend = {"qwen": 0, "gemma4": 0}
    for study in studies:
        per_backend[study["backend"]] += 1
    expected = EXPECTED_COUNTS[family]
    if len(studies) != expected or per_backend["qwen"] != expected // 2 or per_backend["gemma4"] != expected // 2:
        raise SystemExit(
            f"Optuna-100 {family} matrix must resolve to exactly {expected} "
            f"studies ({expected // 2} per backend); got {len(studies)} "
            f"qwen={per_backend['qwen']} gemma4={per_backend['gemma4']}"
        )
    if require_caches:
        missing = [study for study in studies if study.get("cache_missing")]
        if missing:
            raise SystemExit(
                f"{len(missing)} required caches are missing; refusing to resolve. "
                f"First: {missing[0]['dataset']}/{missing[0]['modality']}/{missing[0]['backend']}"
            )
    return {
        "schema_version": "optuna100_resolved_manifest.v1",
        "family": family,
        "run_id": run_id,
        "matrix": str(matrix_path),
        "protocol_profile": matrix.get("protocol_profile"),
        "objective": matrix.get("objective"),
        "target_trials": int(matrix.get("target_trials", 0)),
        "merged_sha": merged_sha,
        "branch": branch,
        "github_issue": github_issue,
        "pr": pr,
        "studies": studies,
        "study_count": len(studies),
        "per_backend": per_backend,
        "missing_cache_count": sum(1 for study in studies if study.get("cache_missing")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("native", "english", "officialdev"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--github-issue", type=int)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--require-caches", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = resolve(
        family=args.family,
        run_id=args.run_id,
        merged_sha=args.merged_sha,
        branch=args.branch,
        github_issue=args.github_issue,
        pr=args.pr,
        require_caches=args.require_caches,
    )
    if args.output:
        save_json(manifest, args.output)
        print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "family": manifest["family"],
                "study_count": manifest["study_count"],
                "per_backend": manifest["per_backend"],
                "missing_cache_count": manifest["missing_cache_count"],
                "require_caches": args.require_caches,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
