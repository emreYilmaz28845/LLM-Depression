from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


EXPECTED_KINDS = {"train", "evaluation", "hidden", "classical"}


def audit_matrix(matrix: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    tasks = matrix.get("tasks", [])
    task_ids = [str(row.get("task_id")) for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        failures.append("duplicate_task_ids")
    counts = Counter(str(row.get("kind")) for row in tasks)
    if set(counts) != EXPECTED_KINDS:
        failures.append(f"task_kinds:{sorted(counts)}")
    if dict(counts) != matrix.get("kind_counts"):
        failures.append("kind_counts_mismatch")
    cells: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        cells[str(task.get("cell_id"))].add(str(task.get("kind")))
        if int(task.get("overrides", {}).get("split.seed", -1)) != 1337:
            failures.append(f"split_seed_changed:{task.get('task_id')}")
        if not task.get("config_hash"):
            failures.append(f"missing_config_hash:{task.get('task_id')}")
    for cell, kinds in cells.items():
        if kinds != EXPECTED_KINDS:
            failures.append(f"incomplete_cell:{cell}:{sorted(kinds)}")
    expected_cells = int(matrix.get("expected_training_cells", -1))
    if len(cells) != expected_cells:
        failures.append(f"cell_count:{len(cells)}!={expected_cells}")
    if matrix.get("stage") == "core" and (expected_cells != 90 or len(tasks) != 360):
        failures.append("core_count_contract")
    return failures


def audit_schedule(schedule: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for epoch, mean in enumerate(schedule.get("epoch_mean_effective_weights", [])):
        if schedule.get("loss_weight_rescale") == "mean_one" and not math.isclose(float(mean), 1.0, abs_tol=1e-8):
            failures.append(f"mean_effective_weight:{epoch}:{mean}")
    for row in schedule.get("rows", []):
        if "bundle_chunk_ids" in row and not row.get("bundle_chunk_ids"):
            failures.append(f"empty_bundle:{row.get('sample_id')}")
        if "bundle_chunk_ids" not in row and not row.get("chunk_id"):
            failures.append(f"missing_chunk_id:{row.get('sample_id')}")
        if float(row.get("raw_loss_weight", 0.0)) <= 0 or float(row.get("effective_loss_weight", 0.0)) <= 0:
            failures.append(f"bad_weight:{row.get('sample_id')}")
    return failures


def audit_oof_predictions(
    rows: Sequence[dict[str, Any]], *, expected_subject_ids: set[str],
    protocols: set[str], seeds: set[int], folds: set[int],
) -> list[str]:
    failures: list[str] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol_id"]), int(row["seed"]))].append(row)
    for protocol in protocols:
        for seed in seeds:
            cell = grouped.get((protocol, seed), [])
            counts = Counter(str(row["subject_id"]) for row in cell)
            if set(counts) != expected_subject_ids:
                failures.append(f"oof_subject_coverage:{protocol}:{seed}")
            if any(value != 1 for value in counts.values()):
                failures.append(f"oof_duplicate_subject:{protocol}:{seed}")
            if {int(row["fold"]) for row in cell} != folds:
                failures.append(f"oof_fold_coverage:{protocol}:{seed}")
    return failures


def audit_slurm(rows: Sequence[dict[str, Any]], expected_task_ids: set[str]) -> list[str]:
    failures: list[str] = []
    seen = {str(row.get("task_id")) for row in rows}
    if seen != expected_task_ids:
        failures.append("slurm_task_coverage")
    for row in rows:
        if str(row.get("state")) != "COMPLETED" or str(row.get("exit_code")) != "0:0":
            failures.append(f"slurm_failure:{row.get('task_id')}:{row.get('state')}:{row.get('exit_code')}")
    return failures


def audit_test_gate(root: Path, stage: str) -> list[str]:
    if stage == "final":
        marker = root / "FINAL_TEST_AUTHORIZED.json"
        return [] if marker.exists() else ["missing_final_test_authorization"]
    forbidden = [path for path in root.rglob("*") if path.is_file() and "test" in path.name.lower()]
    return [f"test_artifact_before_final:{path.relative_to(root)}" for path in forbidden]
