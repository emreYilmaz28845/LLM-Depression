from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.split_utils import assign_stratified_group_folds, subject_fold_report
from src.data.turkish import build_turkish_manifest, verify_turkish_split_integrity
from src.utils import label_text_from_int, sha256_file


def build_turkish_combined_manifest(
    config: dict[str, Any], quarantine: dict[str, Any]
) -> dict[str, Any]:
    """Join question sets by patient within each cohort, then split patients."""
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("Turkish combined manifest requires at least two sources.")
    if config.get("score_authority") != "metadata_csv":
        raise ValueError("Turkish combined manifest requires score_authority=metadata_csv.")

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
            raise ValueError(f"Duplicate or empty Turkish source id: {source_id!r}")
        source_ids.add(source_id)
        if condition not in {"pos_only_t17", "negative_only_t17"}:
            raise ValueError(f"Unknown Turkish question condition: {condition!r}")
        if condition in cohort_members[cohort]:
            raise ValueError(f"Duplicate Turkish question condition in {cohort}: {condition}")
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
                raise ValueError(f"Turkish subject has conflicting BDO scores or labels: {subject}")
            labels[subject] = label
            scores[subject] = score
            rows.append({**row, "source_cohort": cohort, "source_id": source_id})
        join_audit.extend({**row, "source_id": source_id} for row in local_result["join_audit_rows"])
        extra_audit.extend({**row, "source_id": source_id} for row in local_result["extra_file_audit"])

    for cohort, conditions in cohort_members.items():
        if set(conditions) != {"pos_only_t17", "negative_only_t17"}:
            raise ValueError(f"Turkish cohort {cohort!r} needs both question sets.")
        if conditions["pos_only_t17"] != conditions["negative_only_t17"]:
            raise ValueError(f"Turkish cohort {cohort!r} has different patients across question sets.")

    split_cfg = config.get("split", {})
    n_splits = int(split_cfg.get("outer_folds", 5))
    seed = int(split_cfg.get("seed", config.get("seed", 1337)))
    inner_val_ratio = float(split_cfg.get("inner_val_ratio", 0.2))
    folds = assign_stratified_group_folds(labels, n_splits=n_splits, seed=seed)
    verify_turkish_split_integrity(rows, folds, inner_val_ratio, seed)
    sample_counts = Counter(str(row["subject_id"]) for row in rows)
    subject_rows = [
        {
            "subject_id": subject,
            "label": labels[subject],
            "label_text": label_text_from_int(labels[subject]),
            "score": scores[subject],
            "num_samples": sample_counts[subject],
            "source_cohort": "geriatri" if subject.startswith("geriatri:") else "original",
        }
        for subject in sorted(labels)
    ]
    return {
        "manifest_rows": rows,
        "subject_rows": subject_rows,
        "folds": folds,
        "fold_report": subject_fold_report(folds, labels),
        "join_audit_rows": join_audit,
        "extra_file_audit": extra_audit,
        "source_hashes": source_hashes,
        "split_source": "stratified_group_cv",
        "split_source_notes": (
            f"{n_splits}-fold patient-level CV across original and geriatri cohorts; "
            f"each cohort's question sets share patient folds; seed={seed}"
        ),
    }
