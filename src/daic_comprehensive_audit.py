from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


EXPECTED_KINDS = {"train", "evaluation", "hidden", "classical"}
DEFAULT_SPLIT_SEED = 1337
DEFAULT_FOLDS = {0, 1, 2, 3, 4}
FORBIDDEN_RUN_IDS = {"daic_k_prod_20260730_204c550"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def audit_manifest_contract(
    rows: Sequence[dict[str, Any]],
    *,
    partition_rows: Sequence[dict[str, Any]] | None = None,
    folds: dict[Any, dict[str, Any]] | None = None,
    expected_subject_count: int | None = 189,
    expected_development_subject_count: int | None = 142,
    expected_test_subject_count: int | None = 47,
    expected_chunks_by_label: dict[int, int] | None = None,
    expected_subjects_by_label: dict[int, int] | None = None,
    split_seed: int | None = DEFAULT_SPLIT_SEED,
    fold_count: int | None = 5,
) -> list[str]:
    """Validate the leakage-sensitive DAIC manifest and split contract.

    The exact 189/142/47 and 10/15 contracts are the defaults for the
    comprehensive study.  Callers auditing a small synthetic fixture can set
    the expected counts to ``None`` while retaining all structural checks.
    """
    failures: list[str] = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_ids: list[str] = []
    for row in rows:
        subject_id = str(row.get("subject_id", ""))
        sample_id = str(row.get("sample_id", ""))
        if not subject_id:
            failures.append("manifest_missing_subject_id")
        if not sample_id:
            failures.append("manifest_missing_sample_id")
        sample_ids.append(sample_id)
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"manifest_invalid_label:{sample_id}")
            continue
        if label not in {0, 1}:
            failures.append(f"manifest_invalid_label:{sample_id}:{label}")
        by_subject[subject_id].append(row)

    duplicate_samples = [sample_id for sample_id, count in Counter(sample_ids).items() if sample_id and count != 1]
    failures.extend(f"manifest_duplicate_sample_id:{sample_id}" for sample_id in sorted(duplicate_samples))
    subject_labels: dict[str, int] = {}
    for subject_id, subject_rows in sorted(by_subject.items()):
        labels: set[int] = set()
        for row in subject_rows:
            try:
                labels.add(int(row["label"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(labels) != 1:
            failures.append(f"manifest_inconsistent_subject_label:{subject_id}")
        else:
            subject_labels[subject_id] = next(iter(labels))

    subject_count = len(by_subject)
    if expected_subject_count is not None and subject_count != int(expected_subject_count):
        failures.append(f"manifest_subject_count:{subject_count}!={int(expected_subject_count)}")
    if expected_chunks_by_label is not None:
        for subject_id, subject_rows in sorted(by_subject.items()):
            label = subject_labels.get(subject_id)
            expected = expected_chunks_by_label.get(label) if label is not None else None
            if expected is not None and len(subject_rows) != int(expected):
                failures.append(
                    f"manifest_chunk_count:{subject_id}:{len(subject_rows)}!={int(expected)}"
                )
    if expected_subjects_by_label is not None:
        observed = Counter(subject_labels.values())
        for label, expected in sorted(expected_subjects_by_label.items()):
            if observed[int(label)] != int(expected):
                failures.append(
                    f"manifest_subject_class_count:{int(label)}:{observed[int(label)]}!={int(expected)}"
                )

    partition_map: dict[str, str] = {}
    if partition_rows is not None:
        for row in partition_rows:
            subject_id = str(row.get("subject_id", ""))
            partition = str(row.get("partition", ""))
            if subject_id in partition_map:
                failures.append(f"split_duplicate_subject:{subject_id}")
            partition_map[subject_id] = partition
            if subject_id not in subject_labels:
                failures.append(f"split_unknown_subject:{subject_id}")
            else:
                try:
                    partition_label = int(row.get("label", subject_labels[subject_id]))
                except (TypeError, ValueError):
                    failures.append(f"split_invalid_label:{subject_id}")
                else:
                    if partition_label != subject_labels[subject_id]:
                        failures.append(f"split_label_mismatch:{subject_id}")
        missing_partition_subjects = sorted(set(subject_labels) - set(partition_map))
        failures.extend(f"split_missing_subject:{subject_id}" for subject_id in missing_partition_subjects)
        development = {
            subject_id
            for subject_id, partition in partition_map.items()
            if partition in {"train", "val", "dev", "development"}
        }
        test = {subject_id for subject_id, partition in partition_map.items() if partition == "test"}
        if development & test:
            failures.append("development_test_overlap")
        if expected_development_subject_count is not None and len(development) != int(expected_development_subject_count):
            failures.append(
                f"development_subject_count:{len(development)}!={int(expected_development_subject_count)}"
            )
        if expected_test_subject_count is not None and len(test) != int(expected_test_subject_count):
            failures.append(f"test_subject_count:{len(test)}!={int(expected_test_subject_count)}")

    if split_seed is not None and int(split_seed) != DEFAULT_SPLIT_SEED:
        failures.append(f"split_seed:{int(split_seed)}!={DEFAULT_SPLIT_SEED}")

    if folds is not None:
        normalized_folds: dict[int, dict[str, Any]] = {}
        for key, payload in folds.items():
            try:
                normalized_folds[int(key)] = payload
            except (TypeError, ValueError):
                failures.append(f"invalid_fold_id:{key}")
        if fold_count is not None and set(normalized_folds) != set(range(int(fold_count))):
            failures.append(f"fold_count:{len(normalized_folds)}!={int(fold_count)}")
        development = {
            subject_id
            for subject_id, partition in partition_map.items()
            if partition in {"train", "val", "dev", "development"}
        }
        test = {subject_id for subject_id, partition in partition_map.items() if partition == "test"}
        heldout_union: set[str] = set()
        for fold, payload in sorted(normalized_folds.items()):
            if not isinstance(payload, dict):
                failures.append(f"invalid_fold_payload:{fold}")
                continue
            heldout = {str(item) for item in payload.get("final_eval_subject_ids", [])}
            outer_train = {str(item) for item in payload.get("outer_train_subject_ids", [])}
            overlap = heldout & outer_train
            if overlap:
                failures.append(f"fold_overlap:{fold}")
            if heldout & test:
                failures.append(f"fold_test_leakage:{fold}")
            if heldout_union & heldout:
                failures.append(f"fold_heldout_reuse:{fold}")
            heldout_union.update(heldout)
            if development and (heldout | outer_train) != development:
                failures.append(f"fold_development_coverage:{fold}")
        if development and heldout_union != development:
            failures.append("fold_heldout_union_mismatch")

    return failures


# Short alias used by callers that do not need to spell out the study contract.
audit_manifest = audit_manifest_contract


def audit_matrix(matrix: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(matrix.get("run_id", "")) in FORBIDDEN_RUN_IDS:
        failures.append("historical_run_id_forbidden")
    for field in ("implementation_commit", "implementation_hash", "spec_hash"):
        if not matrix.get(field):
            failures.append(f"missing_matrix_{field}")
    tasks = matrix.get("tasks", [])
    if not isinstance(tasks, list):
        return ["tasks_not_a_list"]
    if matrix.get("task_count") != len(tasks):
        failures.append(f"task_count:{len(tasks)}!={matrix.get('task_count')}")
    task_ids = [str(row.get("task_id")) for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        failures.append("duplicate_task_ids")
    counts = Counter(str(row.get("kind")) for row in tasks)
    if set(counts) != EXPECTED_KINDS:
        failures.append(f"task_kinds:{sorted(counts)}")
    if dict(counts) != matrix.get("kind_counts"):
        failures.append("kind_counts_mismatch")

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        split_seed = int(matrix.get("split_seed", DEFAULT_SPLIT_SEED))
    except (TypeError, ValueError):
        failures.append("invalid_matrix_split_seed")
        split_seed = DEFAULT_SPLIT_SEED
    for task in tasks:
        task_id = str(task.get("task_id"))
        kind = str(task.get("kind"))
        cell_id = str(task.get("cell_id"))
        cells[cell_id].append(task)
        overrides = task.get("overrides") or {}
        if _safe_int(overrides.get("split.seed", -1)) != split_seed:
            failures.append(f"split_seed_changed:{task_id}")
        if not task.get("config_hash"):
            failures.append(f"missing_config_hash:{task_id}")
        if not task.get("implementation_hash", matrix.get("implementation_hash")):
            failures.append(f"missing_implementation_hash:{task_id}")
        elif task.get("implementation_hash") != matrix.get("implementation_hash"):
            failures.append(f"implementation_hash_mismatch:{task_id}")
        if str(task.get("resource_profile")) not in matrix.get("resources", {}):
            failures.append(f"unknown_resource_profile:{task_id}")
        expected_task_id = f"{kind}__{cell_id}"
        if task_id != expected_task_id:
            failures.append(f"task_id_kind_mismatch:{task_id}")
        if str(task.get("stage")) != str(matrix.get("stage")):
            failures.append(f"task_stage_mismatch:{task_id}")

    expected_kinds = {"train", "evaluation", "hidden", "classical"}
    for cell, cell_tasks in cells.items():
        cell_counts = Counter(str(task.get("kind")) for task in cell_tasks)
        if set(cell_counts) != expected_kinds or any(value != 1 for value in cell_counts.values()):
            failures.append(f"incomplete_cell:{cell}:{dict(cell_counts)}")
        roots = {str(task.get("output_root")) for task in cell_tasks}
        if len(roots) != 1:
            failures.append(f"cell_output_root_mismatch:{cell}")
        config_hashes = {str(task.get("config_hash")) for task in cell_tasks}
        if len(config_hashes) != 1:
            failures.append(f"cell_config_hash_mismatch:{cell}")
        by_kind = {str(task.get("kind")): task for task in cell_tasks}
        train_id = str(by_kind.get("train", {}).get("task_id"))
        hidden_id = str(by_kind.get("hidden", {}).get("task_id"))
        if by_kind.get("train", {}).get("dependencies") not in ([], None):
            failures.append(f"train_has_dependencies:{cell}")
        if by_kind.get("evaluation", {}).get("dependencies") != [train_id]:
            failures.append(f"evaluation_dependency_mismatch:{cell}")
        if by_kind.get("hidden", {}).get("dependencies") != [train_id]:
            failures.append(f"hidden_dependency_mismatch:{cell}")
        if by_kind.get("classical", {}).get("dependencies") != [hidden_id]:
            failures.append(f"classical_dependency_mismatch:{cell}")
        known_task_ids = {str(task.get("task_id")) for task in cell_tasks}
        for task in cell_tasks:
            for dependency in task.get("dependencies") or []:
                if str(dependency) not in known_task_ids:
                    failures.append(f"unknown_dependency:{cell}:{dependency}")

    expected_cells = int(matrix.get("expected_training_cells", -1))
    if len(cells) != expected_cells:
        failures.append(f"cell_count:{len(cells)}!={expected_cells}")
    stage = str(matrix.get("stage", ""))
    if stage == "smoke" and (expected_cells != 7 or len(tasks) != 28):
        failures.append("smoke_count_contract")
    if stage == "core" and (expected_cells != 90 or len(tasks) != 360):
        failures.append("core_count_contract")
    if stage in {"smoke", "core"}:
        expected_protocols = {"jr4", "jt4", "ja4", "ir4", "ian", "iaf"}
        if stage == "smoke":
            expected_protocols.add("qwen_mil")
        observed_protocols = {str(task.get("protocol_id")) for task in tasks}
        if observed_protocols != expected_protocols:
            failures.append(f"protocol_contract:{sorted(observed_protocols)}")
        expected_seeds = {1337} if stage == "smoke" else {1337, 2027, 3407}
        observed_seeds = {_safe_int(task.get("seed")) for task in tasks}
        if observed_seeds != expected_seeds:
            failures.append(f"seed_contract:{sorted(observed_seeds)}")
        expected_folds = {0} if stage == "smoke" else DEFAULT_FOLDS
        observed_folds = {_safe_int(task.get("fold")) for task in tasks}
        if observed_folds != expected_folds:
            failures.append(f"fold_contract:{sorted(observed_folds)}")
    if stage == "final" and len(tasks) != 12:
        failures.append("final_count_contract")
    if stage == "final":
        if {_safe_int(task.get("seed")) for task in tasks} != {1337, 2027, 3407}:
            failures.append("final_seed_contract")
        if {_safe_int(task.get("fold")) for task in tasks} != {0}:
            failures.append("final_fold_contract")
        if len({str(task.get("protocol_id")) for task in tasks}) != 1:
            failures.append("final_protocol_contract")
    if stage == "final" and not matrix.get("test_authorization"):
        failures.append("missing_final_test_authorization")
    return failures


def audit_schedule(schedule: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    try:
        epochs = int(schedule.get("epochs", len(schedule.get("epoch_mean_effective_weights", []))))
    except (TypeError, ValueError):
        failures.append("invalid_epoch_count")
        epochs = 0
    epoch_means = schedule.get("epoch_mean_effective_weights", [])
    if epochs and len(epoch_means) != epochs:
        failures.append(f"epoch_count_mismatch:{len(epoch_means)}!={epochs}")
    rows_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in schedule.get("rows", []):
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            failures.append("schedule_missing_epoch")
            continue
        rows_by_epoch[epoch].append(row)
        if "bundle_chunk_ids" in row:
            members = [str(item) for item in row.get("bundle_chunk_ids", [])]
            if not members:
                failures.append(f"empty_bundle:{row.get('sample_id')}")
            if len(members) != len(set(members)):
                failures.append(f"duplicate_bundle_chunk:{row.get('sample_id')}")
        elif not row.get("chunk_id"):
            failures.append(f"missing_chunk_id:{row.get('sample_id')}")
        for key in ("raw_loss_weight", "effective_loss_weight"):
            if not _finite(row.get(key)) or float(row[key]) <= 0:
                failures.append(f"bad_weight:{row.get('sample_id')}:{key}")
        if "weight_scale" in row and (not _finite(row["weight_scale"]) or float(row["weight_scale"]) <= 0):
            failures.append(f"bad_weight_scale:{row.get('sample_id')}")
    for epoch, epoch_rows in sorted(rows_by_epoch.items()):
        positions: list[int] = []
        for row in epoch_rows:
            if "position" not in row:
                continue
            try:
                positions.append(int(row["position"]))
            except (TypeError, ValueError):
                failures.append(f"invalid_schedule_position:{epoch}")
        if len(positions) != len(set(positions)):
            failures.append(f"duplicate_schedule_position:{epoch}")
        observed_audio = 0.0
        for row in epoch_rows:
            if not _finite(row.get("audio_seconds", 0.0)):
                failures.append(f"bad_audio_seconds:{row.get('sample_id')}")
            else:
                observed_audio += float(row.get("audio_seconds", 0.0))
        declared_audio = schedule.get("epoch_audio_exposure_seconds", [])
        if epoch < len(declared_audio):
            if not _finite(declared_audio[epoch]) or not math.isclose(
                observed_audio, float(declared_audio[epoch]), abs_tol=1e-6
            ):
                failures.append(f"audio_exposure_mismatch:{epoch}")
        if schedule.get("subject_weighting") == "subject_normalized":
            totals = Counter()
            for row in epoch_rows:
                try:
                    totals[str(row.get("subject_id", ""))] += float(row["raw_loss_weight"])
                except (KeyError, TypeError, ValueError):
                    continue
            if totals and max(totals.values()) - min(totals.values()) > 1e-8:
                failures.append(f"subject_weight_mismatch:{epoch}")
        if schedule.get("subject_weighting") == "class_inverse_frequency":
            subjects_by_label: dict[str, int] = {}
            for row in epoch_rows:
                subject_id = str(row.get("subject_id", ""))
                label = _safe_int(row.get("label"), -1)
                if label not in {0, 1}:
                    failures.append(f"missing_schedule_label:{subject_id}")
                    continue
                if subject_id in subjects_by_label and subjects_by_label[subject_id] != label:
                    failures.append(f"schedule_label_mismatch:{subject_id}")
                subjects_by_label[subject_id] = label
            class_counts = Counter(subjects_by_label.values())
            totals = Counter()
            for row in epoch_rows:
                try:
                    totals[str(row.get("subject_id", ""))] += float(row["raw_loss_weight"])
                except (KeyError, TypeError, ValueError):
                    continue
            for subject_id, label in subjects_by_label.items():
                expected_total = 1.0 / class_counts[label] if class_counts[label] else 0.0
                if not math.isclose(totals[subject_id], expected_total, abs_tol=1e-8):
                    failures.append(f"class_weight_mismatch:{epoch}:{subject_id}")
        if schedule.get("policy") in {"joint_balanced_cover", "fixed_count_balanced_joint_cover"}:
            coverage_by_subject: dict[str, set[int]] = defaultdict(set)
            for row in epoch_rows:
                if row.get("bundle_coverage_count") is not None:
                    coverage_by_subject[str(row.get("subject_id"))].add(int(row["bundle_coverage_count"]))
            for subject_id, coverage in coverage_by_subject.items():
                if len(coverage) != 1:
                    failures.append(f"unequal_bundle_coverage:{epoch}:{subject_id}")
    if schedule.get("policy") in {"joint_balanced_cover", "fixed_count_balanced_joint_cover"}:
        for subject_id, counts in (schedule.get("exposure_counts_by_subject") or {}).items():
            values = [int(value) for value in counts.values()]
            if values and len(set(values)) != 1:
                failures.append(f"unequal_chunk_exposure:{subject_id}")
    if schedule.get("policy") == "all":
        expected_exposure = epochs
        for subject_id, counts in (schedule.get("exposure_counts_by_chunk_id") or {}).items():
            values = [int(value) for value in counts.values()]
            if not values or any(value != expected_exposure for value in values):
                failures.append(f"incomplete_all_chunk_exposure:{subject_id}")
        for epoch, epoch_rows in sorted(rows_by_epoch.items()):
            by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in epoch_rows:
                by_subject[str(row.get("subject_id", ""))].append(row)
            for subject_id, subject_rows in by_subject.items():
                chunk_ids = [str(row.get("chunk_id", row.get("sample_id", ""))) for row in subject_rows]
                if len(chunk_ids) != len(set(chunk_ids)):
                    failures.append(f"duplicate_all_chunk:{epoch}:{subject_id}")
    for epoch, mean in enumerate(epoch_means):
        if not _finite(mean):
            failures.append(f"mean_effective_weight_nonfinite:{epoch}")
        if schedule.get("loss_weight_rescale") == "mean_one" and not math.isclose(float(mean), 1.0, abs_tol=1e-8):
            failures.append(f"mean_effective_weight:{epoch}:{mean}")
    return failures


def audit_oof_predictions(
    rows: Sequence[dict[str, Any]], *, expected_subject_ids: set[str],
    protocols: set[str], seeds: set[int], folds: set[int],
    expected_fold_subjects: dict[int, set[str]] | None = None,
    expected_implementation_hash: str | None = None,
    expected_config_hashes: dict[tuple[str, int, int], str] | None = None,
    require_hashes: bool = False,
) -> list[str]:
    failures: list[str] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    hash_cells: dict[tuple[str, int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"implementation": set(), "config": set()}
    )
    for row in rows:
        protocol = str(row.get("protocol_id", ""))
        try:
            seed = int(row["seed"])
            fold = int(row["fold"])
            subject_id = str(row["subject_id"])
            label = int(row["label"])
            prediction = int(row["prediction"])
        except (KeyError, TypeError, ValueError):
            failures.append("oof_malformed_row")
            continue
        if protocol not in protocols or seed not in seeds or fold not in folds:
            failures.append(f"oof_unexpected_cell:{protocol}:{seed}:{fold}")
        if prediction not in {0, 1}:
            failures.append(f"oof_invalid_prediction:{protocol}:{seed}:{subject_id}")
        if "score_margin" not in row or not _finite(row.get("score_margin")):
            failures.append(f"oof_invalid_score_margin:{protocol}:{seed}:{subject_id}")
        cell_key = (protocol, seed, fold)
        implementation_hash = str(row.get("implementation_hash", "")).strip()
        config_hash = str(row.get("config_hash", "")).strip()
        if require_hashes and not implementation_hash:
            failures.append(f"oof_missing_implementation_hash:{protocol}:{seed}:{fold}")
        if require_hashes and not config_hash:
            failures.append(f"oof_missing_config_hash:{protocol}:{seed}:{fold}")
        if implementation_hash:
            hash_cells[cell_key]["implementation"].add(implementation_hash)
        if config_hash:
            hash_cells[cell_key]["config"].add(config_hash)
        if expected_implementation_hash and implementation_hash != expected_implementation_hash:
            failures.append(f"oof_implementation_hash_mismatch:{protocol}:{seed}:{fold}")
        if expected_config_hashes and expected_config_hashes.get(cell_key) and config_hash != expected_config_hashes[cell_key]:
            failures.append(f"oof_config_hash_mismatch:{protocol}:{seed}:{fold}")
        grouped[(protocol, seed)].append(row)
        if label not in {0, 1}:
            failures.append(f"oof_invalid_label:{protocol}:{seed}:{subject_id}")
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
            fold_subjects: dict[int, set[str]] = defaultdict(set)
            for row in cell:
                fold_subjects[int(row["fold"])].add(str(row["subject_id"]))
            if sum(len(subjects) for subjects in fold_subjects.values()) != len(counts):
                failures.append(f"oof_subject_fold_overlap:{protocol}:{seed}")
            if expected_fold_subjects is not None:
                for fold in folds:
                    if fold_subjects.get(fold, set()) != expected_fold_subjects.get(fold, set()):
                        failures.append(f"oof_fold_membership_mismatch:{protocol}:{seed}:{fold}")
    for cell_key, hashes in sorted(hash_cells.items()):
        protocol, seed, fold = cell_key
        if len(hashes["implementation"]) > 1:
            failures.append(f"oof_mixed_implementation_hashes:{protocol}:{seed}:{fold}")
        if len(hashes["config"]) > 1:
            failures.append(f"oof_mixed_config_hashes:{protocol}:{seed}:{fold}")
    return failures


def audit_slurm(rows: Sequence[dict[str, Any]], expected_task_ids: set[str]) -> list[str]:
    failures: list[str] = []
    seen = [str(row.get("task_id")) for row in rows]
    if len(seen) != len(set(seen)):
        failures.append("slurm_duplicate_task_rows")
    if set(seen) != expected_task_ids:
        failures.append("slurm_task_coverage")
    for row in rows:
        state = str(row.get("state", "")).split()[0]
        exit_code = str(row.get("exit_code", "")).strip()
        if state != "COMPLETED" or exit_code != "0:0":
            failures.append(f"slurm_failure:{row.get('task_id')}:{state}:{exit_code}")
    return failures


def validate_final_test_authorization(
    marker_path: Path,
    *,
    selection_hash: str | None = None,
    implementation_commit: str | None = None,
    spec_hash: str | None = None,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    failures: list[str] = []
    if not marker_path.is_file():
        return False, ["missing_final_test_authorization"], None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"invalid_final_test_authorization:{exc}"], None
    for field in (
        "winner",
        "selection_artifact_sha256",
        "timestamp",
        "implementation_commit",
        "spec_hash",
        "config_hashes",
        "aggregation_view",
        "final_epoch_count",
        "historical_test_exposure",
    ):
        if not payload.get(field):
            failures.append(f"authorization_missing:{field}")
    if payload.get("config_hashes") and not isinstance(payload.get("config_hashes"), dict):
        failures.append("authorization_invalid:config_hashes")
    try:
        if not 1 <= int(payload.get("final_epoch_count", 0)) <= 20:
            failures.append("authorization_invalid:final_epoch_count")
    except (TypeError, ValueError):
        failures.append("authorization_invalid:final_epoch_count")
    if payload.get("historical_test_exposure") is not True:
        failures.append("authorization_historical_exposure_not_declared")
    if selection_hash and payload.get("selection_artifact_sha256") != selection_hash:
        failures.append("authorization_selection_hash_mismatch")
    if implementation_commit and payload.get("implementation_commit") != implementation_commit:
        failures.append("authorization_implementation_mismatch")
    if spec_hash and payload.get("spec_hash") != spec_hash:
        failures.append("authorization_spec_hash_mismatch")
    return not failures, failures, payload


def audit_test_gate(
    root: Path,
    stage: str,
    *,
    authorization_root: Path | None = None,
    selection_hash: str | None = None,
    implementation_commit: str | None = None,
    spec_hash: str | None = None,
) -> list[str]:
    root = Path(root)
    if stage == "final":
        marker = Path(authorization_root or root) / "FINAL_TEST_AUTHORIZED.json"
        _, failures, _ = validate_final_test_authorization(
            marker,
            selection_hash=selection_hash,
            implementation_commit=implementation_commit,
            spec_hash=spec_hash,
        )
        return failures
    if not root.exists():
        return [f"missing_artifact_root:{root}"]
    forbidden: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "FINAL_TEST_AUTHORIZED.json":
            continue
        lowered = path.name.lower()
        if (
            lowered.startswith("test_")
            or lowered.startswith("test-")
            or "_test_" in lowered
            or lowered.endswith("_test.json")
            or lowered.endswith("_test.csv")
            or "official_test" in lowered
        ):
            forbidden.append(path)
    return [f"test_artifact_before_final:{path.relative_to(root)}" for path in forbidden]
