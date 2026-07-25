from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SAMPLING_MODE_NONE = "none"
SAMPLING_MODE_SUBJECT_OVERSAMPLE = "minority_subject_oversample"
SUPPORTED_OVERSAMPLING_RATIOS = (0.75, 1.0)


@dataclass(frozen=True)
class SubjectOversamplingResult:
    indices: tuple[int, ...]
    audit: dict[str, Any]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_identity(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "sample_id": str(row.get("sample_id", index)),
        "subject_id": str(row["subject_id"]),
        "label": int(row["label"]),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    if not rows:
        raise ValueError("Subject oversampling requires at least one training row.")

    subject_labels: dict[str, int] = {}
    subject_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        subject_id = str(row.get("subject_id", "")).strip()
        if not subject_id:
            raise ValueError(f"Training row {index} has no subject_id.")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"Subject oversampling requires binary labels; found {label}.")
        prior = subject_labels.setdefault(subject_id, label)
        if prior != label:
            raise ValueError(
                f"Inconsistent labels for subject {subject_id!r}: {prior} and {label}."
            )
        subject_rows[subject_id].append(index)

    present_labels = set(subject_labels.values())
    if present_labels != {0, 1}:
        raise ValueError(
            "Subject oversampling requires a two-class training partition; "
            f"found labels {sorted(present_labels)}."
        )
    return subject_labels, dict(subject_rows)


def validate_oversampling_ratio(ratio: float | None) -> float:
    if ratio is None:
        raise ValueError("oversampling_ratio is required for minority subject oversampling.")
    resolved = float(ratio)
    if not any(math.isclose(resolved, item, abs_tol=1e-12) for item in SUPPORTED_OVERSAMPLING_RATIOS):
        supported = ", ".join(str(item) for item in SUPPORTED_OVERSAMPLING_RATIOS)
        raise ValueError(f"Unsupported oversampling_ratio={resolved}; expected one of {supported}.")
    return resolved


def build_subject_oversampling(
    rows: Sequence[Mapping[str, Any]],
    *,
    ratio: float | None,
    seed: int,
    expected_minority_label: int | None = 0,
    validation_rows: Sequence[Mapping[str, Any]] | None = None,
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> SubjectOversamplingResult:
    resolved_ratio = validate_oversampling_ratio(ratio)
    subject_labels, subject_rows = _validate_rows(rows)
    subject_counts = Counter(subject_labels.values())
    if subject_counts[0] == subject_counts[1]:
        raise ValueError(
            "Subject oversampling requires a unique minority class; subject counts are tied."
        )
    minority_label = min(subject_counts, key=subject_counts.get)
    majority_label = max(subject_counts, key=subject_counts.get)
    if expected_minority_label is not None and minority_label != int(expected_minority_label):
        raise ValueError(
            f"Expected minority label {expected_minority_label}, detected {minority_label}."
        )

    minority_subjects = sorted(
        subject_id
        for subject_id, label in subject_labels.items()
        if label == minority_label
    )
    target_occurrences = int(math.ceil(resolved_ratio * subject_counts[majority_label]))
    if target_occurrences < subject_counts[minority_label]:
        raise ValueError(
            "Requested ratio would undersample the detected minority class: "
            f"target={target_occurrences}, existing={subject_counts[minority_label]}."
        )

    rng = random.Random(int(seed))
    additional_subjects = [
        rng.choice(minority_subjects)
        for _ in range(target_occurrences - subject_counts[minority_label])
    ]
    multiplicity = {subject_id: 1 for subject_id in sorted(subject_labels)}
    for subject_id in additional_subjects:
        multiplicity[subject_id] += 1

    indices = list(range(len(rows)))
    for subject_id in additional_subjects:
        indices.extend(subject_rows[subject_id])

    original_rows_by_label = Counter(int(row["label"]) for row in rows)
    final_rows_by_label = Counter(int(rows[index]["label"]) for index in indices)
    final_subject_occurrences = dict(subject_counts)
    final_subject_occurrences[minority_label] = target_occurrences
    validation_identity = [
        _row_identity(row, index) for index, row in enumerate(validation_rows or [])
    ]
    evaluation_identity = [
        _row_identity(row, index) for index, row in enumerate(evaluation_rows or [])
    ]
    source_row_identity = [_row_identity(row, index) for index, row in enumerate(rows)]
    source_subject_identity = [
        {"subject_id": subject_id, "label": subject_labels[subject_id]}
        for subject_id in sorted(subject_labels)
    ]
    audit = {
        "schema_version": "subject_oversampling_audit.v1",
        "strategy": SAMPLING_MODE_SUBJECT_OVERSAMPLE,
        "requested_ratio": resolved_ratio,
        "sampling_seed": int(seed),
        "detected_minority_label": int(minority_label),
        "detected_majority_label": int(majority_label),
        "original_subject_occurrence_counts_by_class": {
            str(label): int(subject_counts[label]) for label in (0, 1)
        },
        "final_subject_occurrence_counts_by_class": {
            str(label): int(final_subject_occurrences[label]) for label in (0, 1)
        },
        "original_row_counts_by_class": {
            str(label): int(original_rows_by_label[label]) for label in (0, 1)
        },
        "final_row_counts_by_class": {
            str(label): int(final_rows_by_label[label]) for label in (0, 1)
        },
        "target_minority_occurrences": int(target_occurrences),
        "additional_minority_subject_occurrences": len(additional_subjects),
        "sampled_additional_subject_ids": additional_subjects,
        "duplicate_multiplicity_by_subject": multiplicity,
        "source_row_assignments_sha256": _canonical_hash(source_row_identity),
        "source_subject_assignments_sha256": _canonical_hash(source_subject_identity),
        "expanded_index_multiset_sha256": _canonical_hash(sorted(indices)),
        "validation_indices_untouched": True,
        "evaluation_indices_untouched": True,
        "validation_row_assignments_sha256": _canonical_hash(validation_identity),
        "evaluation_row_assignments_sha256": _canonical_hash(evaluation_identity),
    }
    return SubjectOversamplingResult(indices=tuple(indices), audit=audit)


def build_no_sampling_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    validation_rows: Sequence[Mapping[str, Any]] | None = None,
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> SubjectOversamplingResult:
    subject_labels, _ = _validate_rows(rows)
    subject_counts = Counter(subject_labels.values())
    row_counts = Counter(int(row["label"]) for row in rows)
    source_row_identity = [_row_identity(row, index) for index, row in enumerate(rows)]
    source_subject_identity = [
        {"subject_id": subject_id, "label": subject_labels[subject_id]}
        for subject_id in sorted(subject_labels)
    ]
    indices = tuple(range(len(rows)))
    audit = {
        "schema_version": "subject_oversampling_audit.v1",
        "strategy": SAMPLING_MODE_NONE,
        "requested_ratio": None,
        "sampling_seed": int(seed),
        "detected_minority_label": int(min(subject_counts, key=subject_counts.get)),
        "detected_majority_label": int(max(subject_counts, key=subject_counts.get)),
        "original_subject_occurrence_counts_by_class": {
            str(label): int(subject_counts[label]) for label in (0, 1)
        },
        "final_subject_occurrence_counts_by_class": {
            str(label): int(subject_counts[label]) for label in (0, 1)
        },
        "original_row_counts_by_class": {
            str(label): int(row_counts[label]) for label in (0, 1)
        },
        "final_row_counts_by_class": {
            str(label): int(row_counts[label]) for label in (0, 1)
        },
        "target_minority_occurrences": int(min(subject_counts.values())),
        "additional_minority_subject_occurrences": 0,
        "sampled_additional_subject_ids": [],
        "duplicate_multiplicity_by_subject": {
            subject_id: 1 for subject_id in sorted(subject_labels)
        },
        "source_row_assignments_sha256": _canonical_hash(source_row_identity),
        "source_subject_assignments_sha256": _canonical_hash(source_subject_identity),
        "expanded_index_multiset_sha256": _canonical_hash(indices),
        "validation_indices_untouched": True,
        "evaluation_indices_untouched": True,
        "validation_row_assignments_sha256": _canonical_hash(
            [_row_identity(row, index) for index, row in enumerate(validation_rows or [])]
        ),
        "evaluation_row_assignments_sha256": _canonical_hash(
            [_row_identity(row, index) for index, row in enumerate(evaluation_rows or [])]
        ),
    }
    return SubjectOversamplingResult(indices=indices, audit=audit)
