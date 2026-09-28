from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.split_utils import subject_fold_report
from src.data.turkish import build_turkish_manifest, verify_turkish_split_integrity
from src.utils import label_text_from_int, sha256_file

LOCKED_FOLDS_PATH_KEY = "locked_original_folds_path"
LOCKED_FOLDS_SHA256_KEY = "locked_original_folds_sha256"
NEW_COHORT_ASSIGNMENT_RULE = "label_stratified_balanced_round_robin_v1"


class CombinedTurkishError(ValueError):
    """Raised when the four-source Turkish manifest contract is violated."""


def _canonical_mapping_sha256(mapping: dict[str, int]) -> str:
    """Hash a subject-to-fold mapping the way the pooled builder hashes its split."""
    canonical = sorted((str(subject), int(fold)) for subject, fold in mapping.items())
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_locked_original_folds(
    path: Path, expected_sha256: str
) -> tuple[dict[str, int], dict[str, Any]]:
    """Load the canonical baseline fold mapping and verify its exact bytes."""
    if not path.is_file():
        raise CombinedTurkishError(
            f"Turkish combined split needs the locked baseline folds file: {path}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise CombinedTurkishError(
            f"locked baseline folds hash mismatch for {path}: "
            f"{actual_sha256} != declared {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CombinedTurkishError(f"cannot read locked baseline folds {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise CombinedTurkishError(f"locked baseline folds file is not a non-empty object: {path}")
    mapping: dict[str, int] = {}
    for raw_fold, fold_payload in payload.items():
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as exc:
            raise CombinedTurkishError(f"non-integer fold in {path}: {raw_fold!r}") from exc
        subjects = (fold_payload or {}).get("final_eval_subject_ids")
        if not isinstance(subjects, list) or not subjects:
            raise CombinedTurkishError(
                f"locked baseline fold {fold} has no final_eval_subject_ids in {path}"
            )
        for subject in subjects:
            subject_id = str(subject)
            if subject_id in mapping:
                raise CombinedTurkishError(
                    f"locked baseline folds hold out the same subject twice: {subject_id!r}"
                )
            mapping[subject_id] = fold
    if sorted(set(mapping.values())) != list(range(len(payload))):
        raise CombinedTurkishError(
            f"locked baseline folds are not a contiguous 0..{len(payload) - 1} partition: {path}"
        )
    record = {
        "path": str(path),
        "file_sha256": actual_sha256,
        "subject_count": len(mapping),
        "fold_count": len(set(mapping.values())),
        "canonical_mapping_sha256": _canonical_mapping_sha256(mapping),
    }
    return mapping, record


def _assign_new_subjects(
    labels: dict[str, int], new_subjects: list[str], n_splits: int
) -> dict[str, int]:
    """Place new subjects deterministically, balancing every label across folds."""
    assignment: dict[str, int] = {}
    label_counts: dict[int, Counter] = {fold: Counter() for fold in range(n_splits)}
    for label in sorted({int(labels[subject]) for subject in new_subjects}):
        stratum = sorted(subject for subject in new_subjects if int(labels[subject]) == label)
        for subject in stratum:
            fold = min(range(n_splits), key=lambda candidate: (label_counts[candidate][label], candidate))
            assignment[subject] = fold
            label_counts[fold][label] += 1
    return assignment


def build_turkish_combined_manifest(
    config: dict[str, Any], quarantine: dict[str, Any]
) -> dict[str, Any]:
    """Join question sets by patient within each cohort, then lock patient folds.

    The original cohort keeps the canonical baseline fold assignment byte-for-byte
    (verified against a declared sha256), and the new cohort is placed
    deterministically so that every subject is held out exactly once.
    """
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise CombinedTurkishError("Turkish combined manifest requires at least two sources.")
    if config.get("score_authority") != "metadata_csv":
        raise CombinedTurkishError("Turkish combined manifest requires score_authority=metadata_csv.")

    rows: list[dict[str, Any]] = []
    join_audit: list[dict[str, Any]] = []
    extra_audit: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    labels: dict[str, int] = {}
    scores: dict[str, float] = {}
    cohort_members: dict[str, dict[str, set[str]]] = defaultdict(dict)
    source_ids: set[str] = set()
    for source in sources:
        source_id = str(source["id"])
        cohort = str(source["cohort"])
        condition = str(source["question_condition"])
        if not source_id or source_id in source_ids:
            raise CombinedTurkishError(f"Duplicate or empty Turkish source id: {source_id!r}")
        source_ids.add(source_id)
        if condition not in {"pos_only_t17", "negative_only_t17"}:
            raise CombinedTurkishError(f"Unknown Turkish question condition: {condition!r}")
        if condition in cohort_members[cohort]:
            raise CombinedTurkishError(f"Duplicate Turkish question condition in {cohort}: {condition}")
        local_config = {
            **config,
            **source,
            "dataset_variant": condition,
            "subject_namespace": cohort if cohort != "original" else "",
            "sample_namespace": cohort if cohort != "original" else "",
        }
        local_result = build_turkish_manifest(local_config, quarantine)
        cohort_members[cohort][condition] = {
            str(row["subject_id"]) for row in local_result["manifest_rows"]
        }
        root = Path(source["dataset_root"])
        source_hashes[source_id] = {
            "metadata_csv_sha256": sha256_file(root / source["metadata_csv"]),
            "transcript_jsonl_sha256": sha256_file(root / source["transcript_file"]),
        }
        for row in local_result["manifest_rows"]:
            subject = str(row["subject_id"])
            label = int(row["label"])
            score = float(row["score"])
            if subject in labels and (labels[subject] != label or scores[subject] != score):
                raise CombinedTurkishError(
                    f"Turkish subject has conflicting BDO scores or labels: {subject}"
                )
            labels[subject] = label
            scores[subject] = score
            rows.append({**row, "source_cohort": cohort, "source_id": source_id})
        join_audit.extend({**row, "source_id": source_id} for row in local_result["join_audit_rows"])
        extra_audit.extend({**row, "source_id": source_id} for row in local_result["extra_file_audit"])

    original_subjects: set[str] = set()
    new_subjects: set[str] = set()
    for cohort, conditions in cohort_members.items():
        if set(conditions) != {"pos_only_t17", "negative_only_t17"}:
            raise CombinedTurkishError(f"Turkish cohort {cohort!r} needs both question sets.")
        if conditions["pos_only_t17"] != conditions["negative_only_t17"]:
            raise CombinedTurkishError(
                f"Turkish cohort {cohort!r} has different patients across question sets."
            )
        members = set(conditions["pos_only_t17"])
        if cohort == "original":
            original_subjects |= members
        else:
            new_subjects |= members

    split_cfg = config.get("split", {})
    n_splits = int(split_cfg.get("outer_folds", 5))
    seed = int(split_cfg.get("seed", config.get("seed", 1337)))
    inner_val_ratio = float(split_cfg.get("inner_val_ratio", 0.2))

    locked_path_value = str(split_cfg.get(LOCKED_FOLDS_PATH_KEY, "") or "").strip()
    locked_sha256_value = str(split_cfg.get(LOCKED_FOLDS_SHA256_KEY, "") or "").strip()
    if not locked_path_value or not locked_sha256_value:
        raise CombinedTurkishError(
            "four-source Turkish configs must declare split."
            f"{LOCKED_FOLDS_PATH_KEY} and split.{LOCKED_FOLDS_SHA256_KEY}: the original "
            "cohort has to keep the canonical baseline folds"
        )
    locked_mapping, locked_record = _load_locked_original_folds(
        Path(locked_path_value), locked_sha256_value
    )
    if sorted(set(locked_mapping.values())) != list(range(n_splits)):
        raise CombinedTurkishError(
            f"locked baseline folds use folds {sorted(set(locked_mapping.values()))} but the "
            f"config declares split.outer_folds={n_splits}"
        )

    all_subjects = set(labels)
    if not original_subjects:
        raise CombinedTurkishError("Turkish combined manifest needs an 'original' cohort.")
    if original_subjects != set(locked_mapping):
        missing = sorted(set(locked_mapping) - original_subjects)
        extra = sorted(original_subjects - set(locked_mapping))
        raise CombinedTurkishError(
            "original cohort does not match the locked baseline folds "
            f"(locked={len(locked_mapping)}, built={len(original_subjects)}, "
            f"missing={missing[:5]}, unexpected={extra[:5]})"
        )
    if original_subjects & new_subjects:
        raise CombinedTurkishError(
            f"new cohort reuses locked original subjects: {sorted(original_subjects & new_subjects)[:5]}"
        )
    if original_subjects | new_subjects != all_subjects:
        raise CombinedTurkishError("Turkish combined manifest has subjects outside both cohorts.")

    new_assignment = _assign_new_subjects(labels, sorted(new_subjects), n_splits)
    fold_of_subject = {**{str(s): int(f) for s, f in locked_mapping.items()}, **new_assignment}
    folds: dict[int, dict[str, list[str]]] = {}
    for fold in range(n_splits):
        heldout = sorted(subject for subject, value in fold_of_subject.items() if value == fold)
        folds[fold] = {
            "outer_train_subject_ids": sorted(all_subjects - set(heldout)),
            "final_eval_subject_ids": heldout,
        }
    verify_turkish_split_integrity(rows, folds, inner_val_ratio, seed)

    for subject in original_subjects:
        if fold_of_subject[subject] != int(locked_mapping[subject]):
            raise CombinedTurkishError(
                f"locked original subject moved folds: {subject!r}"
            )

    sample_counts = Counter(str(row["subject_id"]) for row in rows)
    subject_rows = [
        {
            "subject_id": subject,
            "label": labels[subject],
            "label_text": label_text_from_int(labels[subject]),
            "score": scores[subject],
            "num_samples": sample_counts[subject],
            "source_cohort": "geriatri" if subject.startswith("geriatri:") else "original",
            "fold": fold_of_subject[subject],
        }
        for subject in sorted(labels)
    ]
    fold_lock = {
        "new_cohort_assignment_rule": NEW_COHORT_ASSIGNMENT_RULE,
        "locked_original_folds": locked_record,
        "original_subject_count": len(original_subjects),
        "new_subject_count": len(new_subjects),
        "per_fold_locked_original_subject_counts": {
            str(fold): sum(1 for subject in original_subjects if fold_of_subject[subject] == fold)
            for fold in range(n_splits)
        },
        "per_fold_new_subject_counts": {
            str(fold): sum(1 for subject in new_subjects if fold_of_subject[subject] == fold)
            for fold in range(n_splits)
        },
        "per_fold_label_counts": {
            str(fold): dict(
                sorted(
                    Counter(
                        int(labels[subject])
                        for subject in fold_of_subject
                        if fold_of_subject[subject] == fold
                    ).items()
                )
            )
            for fold in range(n_splits)
        },
        "combined_mapping_canonical_sha256": _canonical_mapping_sha256(fold_of_subject),
        "new_assignment_canonical_sha256": _canonical_mapping_sha256(new_assignment),
    }
    return {
        "manifest_rows": rows,
        "subject_rows": subject_rows,
        "folds": folds,
        "fold_report": subject_fold_report(folds, labels),
        "join_audit_rows": join_audit,
        "extra_file_audit": extra_audit,
        "source_hashes": source_hashes,
        "split_source": "locked_original_folds_plus_stratified_new_cohort",
        "split_source_notes": (
            f"original cohort folds locked to the canonical baseline mapping "
            f"({locked_record['file_sha256']}) for {len(original_subjects)} subjects; "
            f"{len(new_subjects)} new subjects assigned by {NEW_COHORT_ASSIGNMENT_RULE} "
            f"within label strata; seed={seed}"
        ),
        "fold_lock": fold_lock,
        "fold_hash": fold_lock["combined_mapping_canonical_sha256"],
    }
