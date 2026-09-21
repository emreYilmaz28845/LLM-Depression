#!/usr/bin/env python3
"""Resolve and submit the prompt-context experiment matrix.

The matrix is declared in ``configs/experiments/promptcontext/matrix.yaml``.
This tool turns it into concrete work: one managed submission per standalone
fold (``tools/exp.py submit``) and one merged stage submission per backbone
(``scripts/submit_symmetric_merged.py``).

``--audit`` prints the resolved plan and fails when a declared config is
missing. ``--dry-run`` prints the exact commands. ``--execute`` runs them.

Standalone submission uses ``--manifest-policy prebuilt`` for the pooled
Turkish cells because their manifest is source-manifest-only; every other cell
builds its manifest in the runtime directory from the dataset root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "configs/experiments/promptcontext/matrix.yaml"

STANDALONE_BACKENDS = ("qwen", "gemma4")
MERGED_STAGES = ("smoke", "cv", "final")
MODALITY_ORDER = ("audio_only", "text_only", "audio_text")

# Datasets whose manifest is built by the pooled source-manifest script instead
# of the raw dataset-root builder.
PREBUILT_DATASET_VARIANTS = {"pooled_t17"}


class MatrixError(RuntimeError):
    """Raised when the matrix or a derived job cannot be resolved safely."""


def load_matrix() -> dict[str, Any]:
    try:
        matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MatrixError(f"cannot read {MATRIX_PATH}: {exc}") from exc
    if not isinstance(matrix, dict):
        raise MatrixError(f"{MATRIX_PATH} is not an object")
    return matrix


def standalone_cells(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cells = list(matrix.get("standalone") or [])
    if len(cells) != matrix["counts"]["standalone_cells"]:
        raise MatrixError("standalone cell count does not match the declared count")
    return cells


def merged_cells(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cells = list(matrix.get("merged") or [])
    if len(cells) != matrix["counts"]["merged_cells"]:
        raise MatrixError("merged cell count does not match the declared count")
    return cells


def run_name(cell: dict[str, Any], fold: int) -> str:
    return (
        f"pctx_v1_{cell['backbone']}_{cell['dataset']}_{cell['modality']}"
        f"_s1337_f{fold}"
    )


def campaign(cell: dict[str, Any]) -> str:
    return (
        "promptcontext_v1_gemma4_likelihood"
        if cell["backbone"] == "gemma4"
        else "promptcontext_v1_likelihood"
    )


def manifest_policy(cell: dict[str, Any]) -> str:
    config_path = PROJECT_ROOT / cell["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variant = str(config.get("dataset_variant", "")).strip()
    return "prebuilt" if variant in PREBUILT_DATASET_VARIANTS else "build"


def standalone_jobs(
    cells: list[dict[str, Any]],
    *,
    run_name_prefix: str | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in cells:
        folds = list(cell["folds"])
        if int(cell.get("fold_count_cap", 0) or 0):
            folds = folds[: int(cell["fold_count_cap"])]
        for fold in folds:
            name = run_name(cell, fold)
            if run_name_prefix:
                name = f"{run_name_prefix}{name}"
            jobs.append(
                {
                    "kind": "standalone",
                    "dataset": cell["dataset"],
                    "modality": cell["modality"],
                    "backbone": cell["backbone"],
                    "fold": fold,
                    "config": cell["config"],
                    "campaign": campaign(cell),
                    "run_name": name,
                    "manifest_policy": manifest_policy(cell),
                }
            )
    return jobs


def merged_jobs(cells: list[dict[str, Any]], *, stage: str, backend: str) -> list[dict[str, Any]]:
    if stage not in MERGED_STAGES:
        raise MatrixError(f"unsupported merged stage {stage!r}")
    selected = [cell for cell in cells if cell["backbone"] == backend]
    if len(selected) != 3:
        raise MatrixError(f"merged {backend} must declare three modalities")
    if stage not in ("smoke", "cv", "final"):
        raise MatrixError(f"unsupported stage {stage!r}")
    return selected


def merged_run_id(backend: str, stage: str) -> str:
    return f"promptcontext_v1_{backend}_{stage}"


def describe(matrix: dict[str, Any]) -> dict[str, Any]:
    standalone = standalone_cells(matrix)
    merged = merged_cells(matrix)
    jobs = standalone_jobs(standalone)
    counts = matrix["counts"]
    if len(jobs) != counts["standalone_training_folds"]:
        raise MatrixError(
            f"resolved standalone folds {len(jobs)} != declared {counts['standalone_training_folds']}"
        )
    # Each merged cell runs the protocol's five outer folds for CV and one final
    # fit per modality; the planner fixes those shapes.
    cv_fits = 5 * sum(1 for cell in merged if "cv" in cell["stages"])
    final_fits = sum(1 for cell in merged if "final" in cell["stages"])
    if cv_fits != counts["merged_cv_folds"] or final_fits != counts["merged_final_fits"]:
        raise MatrixError(
            f"resolved merged fits (cv={cv_fits}, final={final_fits}) do not match the declared counts"
        )
    return {
        "recipe_id": matrix["recipe_id"],
        "standalone_cells": len(standalone),
        "merged_cells": len(merged),
        "standalone_folds": len(jobs),
        "merged_cv_folds": cv_fits,
        "merged_final_fits": final_fits,
        "total_training_fits": len(jobs) + cv_fits + final_fits,
        "by_backbone": {
            backbone: {
                dataset: sum(
                    1
                    for job in jobs
                    if job["backbone"] == backbone and job["dataset"] == dataset
                )
                for dataset in sorted({job["dataset"] for job in jobs})
            }
            for backbone in STANDALONE_BACKENDS
        },
    }


def _exp_submit_argv(job: dict[str, Any], *, execute: bool) -> list[str]:
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "exp.py"),
        "submit",
        "prompt-context-qwen-gemma-v1",
        "--config",
        job["config"],
        "--fold",
        str(job["fold"]),
        "--run-name",
        job["run_name"],
        "--campaign",
        job["campaign"],
        "--modality",
        job["modality"],
        "--dataset",
        job["dataset"],
        "--manifest-policy",
        job["manifest_policy"],
    ]
    argv.append("--execute" if execute else "--dry-run")
    return argv


def _merged_submit_argv(
    cells: list[dict[str, Any]], *, stage: str, backend: str, execute: bool
) -> list[str]:
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "submit_symmetric_merged.py"),
        "--stage",
        stage,
        "--run-id",
        merged_run_id(backend, stage),
        "--smoke-trials",
        "0",
    ]
    for cell in cells:
        argv += ["--config", cell["config"]]
    argv.append("--execute" if execute else "--dry-run")
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="print the resolved matrix plan")
    parser.add_argument(
        "--submit",
        choices=("standalone", "merged"),
        default=None,
        help="submit standalone folds or one merged stage",
    )
    parser.add_argument("--backend", choices=STANDALONE_BACKENDS, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--modality", choices=MODALITY_ORDER, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--stage", choices=MERGED_STAGES, default="smoke")
    parser.add_argument("--limit", type=int, default=None, help="submit at most N resolved jobs")
    parser.add_argument("--run-name-prefix", default=None)
    parser.add_argument("--execute", action="store_true", help="run the resolved commands")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved commands")
    parser.add_argument("--output", type=Path, default=None, help="write the resolved plan as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = load_matrix()
    try:
        plan = describe(matrix)
    except MatrixError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for cell in standalone_cells(matrix) + merged_cells(matrix):
        if not (PROJECT_ROOT / cell["config"]).is_file():
            print(f"ERROR: declared config is missing: {cell['config']}", file=sys.stderr)
            return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.submit is None:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.submit == "standalone":
        cells = standalone_cells(matrix)
        if args.backend:
            cells = [cell for cell in cells if cell["backbone"] == args.backend]
        if args.dataset:
            cells = [cell for cell in cells if cell["dataset"] == args.dataset]
        if args.modality:
            cells = [cell for cell in cells if cell["modality"] == args.modality]
        jobs = standalone_jobs(cells, run_name_prefix=args.run_name_prefix)
        if args.fold is not None:
            jobs = [job for job in jobs if job["fold"] == args.fold]
        if args.limit is not None:
            jobs = jobs[: int(args.limit)]
        if not jobs:
            print("ERROR: no standalone job matches the filters", file=sys.stderr)
            return 1
        commands = [_exp_submit_argv(job, execute=args.execute) for job in jobs]
    else:
        if not args.backend:
            print("ERROR: merged submission requires --backend", file=sys.stderr)
            return 1
        cells = merged_jobs(merged_cells(matrix), stage=args.stage, backend=args.backend)
        if args.modality:
            cells = [cell for cell in cells if cell["modality"] == args.modality]
        if not cells:
            print("ERROR: no merged config matches the filters", file=sys.stderr)
            return 1
        commands = [_merged_submit_argv(cells, stage=args.stage, backend=args.backend, execute=args.execute)]

    print(f"resolved {len(commands)} command(s); execute={bool(args.execute)}")
    failures = 0
    for command in commands:
        print("$ " + " ".join(command))
        if not args.execute:
            continue
        completed = subprocess.run(command)
        if completed.returncode != 0:
            failures += 1
            print(f"ERROR: command failed with rc={completed.returncode}", file=sys.stderr)
            break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
