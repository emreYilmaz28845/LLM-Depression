"""Deterministic training-window coverage capping.

Worker 3 campaign: reduce the number of *training* windows per subject to a
fraction of the baseline set, using one reproducible pseudorandom permutation
per subject. Evaluation and selection partitions are untouched.

Frozen algorithm (``sha256-subject-permutation-v1``):

1. Group the baseline *training* examples by ``subject_id``. Require one label
   per subject and unique ``sample_id`` values within a subject.
2. Canonically order the unique sample ids (lexicographic) before hashing, so
   input example order cannot change the permutation.
3. Derive a per-subject digest:
   ``sha256(b"window-cap|<version>|<sampling_seed>|<subject_id>")``.
4. Permute the subject's sample ids by
   ``sha256(subject_digest + b"|" + sample_id)`` (ascending digest, sample id
   as tie-break).
5. Keep ``k = max(1, ceil(fraction * n))`` prefix samples. Prefixes for
   25/50/75 percent are nested by construction and fixed across epochs and
   training seeds.

No labels, model seeds, outcomes, ``hash()`` or process randomness take part.
The mask artifact holds the private membership (subject -> selected sample
ids), its selection hash and the algorithm version. Weight handling preserves
each subject's baseline total raw mass before the usual global mean-one
rescale, per the campaign contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

ALGORITHM_VERSION = "sha256-subject-permutation-v1"
MASK_SCHEMA_VERSION = "audiollm.window_cap_mask.v1"
AUDIT_SCHEMA_VERSION = "audiollm.window_cap_audit.v1"


class WindowCapError(ValueError):
    """Raised when the cap cannot be applied safely."""


def _subject_digest(sampling_seed: int, subject_id: str) -> bytes:
    payload = f"window-cap|{ALGORITHM_VERSION}|{int(sampling_seed)}|{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _sample_key(subject_digest: bytes, sample_id: str) -> bytes:
    return hashlib.sha256(subject_digest + b"|" + sample_id.encode("utf-8")).digest()


def permute_subject_samples(
    subject_id: str, sample_ids: list[str], sampling_seed: int
) -> list[str]:
    """Return the subject's unique sample ids in the frozen permutation order."""
    unique = sorted({str(sample_id) for sample_id in sample_ids})
    if not unique or any(not sample_id for sample_id in unique):
        raise WindowCapError(f"subject {subject_id!r} has empty or missing sample ids")
    digest = _subject_digest(sampling_seed, str(subject_id))
    return sorted(unique, key=lambda sample_id: (_sample_key(digest, sample_id), sample_id))


def selections_for_fraction(
    examples: list[dict[str, Any]], fraction: float, sampling_seed: int
) -> dict[str, list[str]]:
    """Selected sample ids per subject for one fraction of the baseline set."""
    if not 0.0 < float(fraction) <= 1.0:
        raise WindowCapError(f"fraction must be in (0, 1], got {fraction!r}")
    by_subject: dict[str, list[str]] = defaultdict(list)
    labels_by_subject: dict[str, set[int]] = defaultdict(set)
    for example in examples:
        subject_id = str(example.get("subject_id", "")).strip()
        sample_id = str(example.get("sample_id", "")).strip()
        if not subject_id or not sample_id:
            raise WindowCapError("every training example needs subject_id and sample_id")
        by_subject[subject_id].append(sample_id)
        labels_by_subject[subject_id].add(int(example["label"]))
    for subject_id, labels in labels_by_subject.items():
        if len(labels) != 1:
            raise WindowCapError(
                f"window cap requires one label per subject; {subject_id!r} has {sorted(labels)}"
            )
    for subject_id, sample_ids in by_subject.items():
        if len(sample_ids) != len(set(sample_ids)):
            raise WindowCapError(
                f"window cap requires unique sample ids per subject; {subject_id!r} has duplicates"
            )
    selections: dict[str, list[str]] = {}
    for subject_id, sample_ids in sorted(by_subject.items()):
        ordered = permute_subject_samples(subject_id, sample_ids, sampling_seed)
        keep = max(1, math.ceil(float(fraction) * len(ordered)))
        selections[subject_id] = ordered[:keep]
    return selections


def compute_selection_sha256(
    algorithm_version: str, sampling_seed: int, fraction: float, subjects: dict[str, Any]
) -> str:
    """Canonical selection hash over the exact membership payload.

    This is the single definition used by both the training hook (``build_mask``)
    and the head-side mask validation, so a mutated membership payload cannot
    keep a stale accepted hash: the digest is always recomputed from
    ``algorithm_version``, ``sampling_seed``, ``fraction`` and ``subjects``.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "algorithm_version": str(algorithm_version),
                "sampling_seed": int(sampling_seed),
                "fraction": float(fraction),
                "subjects": subjects,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_mask(
    examples: list[dict[str, Any]],
    *,
    fraction: float,
    sampling_seed: int,
    baseline_input_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the private mask artifact (subject -> selected sample ids)."""
    selections = selections_for_fraction(examples, fraction, sampling_seed)
    available = defaultdict(int)
    selected = defaultdict(int)
    for example in examples:
        available[str(example["subject_id"])] += 1
    for subject_id, ids in selections.items():
        selected[subject_id] = len(ids)
    mask: dict[str, Any] = {
        "schema_version": MASK_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "sampling_seed": int(sampling_seed),
        "fraction": float(fraction),
        "baseline_input_sha256": baseline_input_sha256,
        "subjects": {subject_id: ids for subject_id, ids in sorted(selections.items())},
        "available_counts": {subject_id: available[subject_id] for subject_id in sorted(available)},
        "selected_counts": {subject_id: selected[subject_id] for subject_id in sorted(selections)},
        "total_available": len(examples),
        "total_selected": sum(selected.values()),
    }
    mask["selection_sha256"] = compute_selection_sha256(
        mask["algorithm_version"], mask["sampling_seed"], mask["fraction"], mask["subjects"]
    )
    return mask


def apply_selection(
    examples: list[dict[str, Any]], selections: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Filter examples to the selected sample ids, preserving input order."""
    selected_by_subject = {
        subject_id: set(ids) for subject_id, ids in selections.items()
    }
    returned: list[dict[str, Any]] = []
    for example in examples:
        subject_id = str(example["subject_id"])
        if str(example["sample_id"]) in selected_by_subject.get(subject_id, set()):
            returned.append(example)
    return returned


def preserve_subject_weight_mass(
    baseline_examples: list[dict[str, Any]],
    selected_examples: list[dict[str, Any]],
    *,
    mean_one_rescale: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep each subject's baseline total raw mass on the selected rows.

    The baseline per-window ``raw_loss_weight`` values are kept; within each
    subject every selected row is multiplied by
    ``baseline_subject_total / selected_subject_total``. When the baseline
    recipe rescales to mean one, the selected set receives the same global
    mean-one rescale. The audit records mass before the global rescale and the
    relative subject contribution after it.
    """
    if not selected_examples:
        raise WindowCapError("cannot weight an empty capped training partition")
    for name, rows in (("baseline", baseline_examples), ("selected", selected_examples)):
        for example in rows:
            if "raw_loss_weight" not in example:
                raise WindowCapError(
                    f"{name} example {example.get('sample_id')!r} lacks raw_loss_weight; "
                    "apply the cap only after baseline weights are computed"
                )
    baseline_totals: dict[str, float] = defaultdict(float)
    for example in baseline_examples:
        baseline_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    selected_totals: dict[str, float] = defaultdict(float)
    for example in selected_examples:
        selected_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    scaled: list[dict[str, Any]] = []
    for example in selected_examples:
        subject_id = str(example["subject_id"])
        if selected_totals[subject_id] <= 0.0:
            raise WindowCapError(f"selected raw mass is not positive for subject {subject_id!r}")
        factor = baseline_totals[subject_id] / selected_totals[subject_id]
        raw = float(example["raw_loss_weight"]) * factor
        scaled.append({**example, "raw_loss_weight": raw, "loss_weight": raw})
    pre_rescale_subject_totals: dict[str, float] = defaultdict(float)
    for example in scaled:
        pre_rescale_subject_totals[str(example["subject_id"])] += float(example["raw_loss_weight"])
    global_scale = 1.0
    if mean_one_rescale:
        total_raw = sum(float(example["raw_loss_weight"]) for example in scaled)
        global_scale = len(scaled) / total_raw
        scaled = [
            {**example, "loss_weight": float(example["raw_loss_weight"]) * global_scale}
            for example in scaled
        ]
    post_subject_totals: dict[str, float] = defaultdict(float)
    for example in scaled:
        post_subject_totals[str(example["subject_id"])] += float(example["loss_weight"])
    total_post = sum(post_subject_totals.values())
    relative_contribution = {
        subject_id: (total / total_post if total_post else 0.0)
        for subject_id, total in sorted(post_subject_totals.items())
    }
    mean_loss_weight = (
        sum(float(example["loss_weight"]) for example in scaled) / len(scaled)
    )
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "baseline_subject_raw_totals": {k: v for k, v in sorted(baseline_totals.items())},
        "selected_subject_raw_totals_before_rescale": {
            k: v for k, v in sorted(pre_rescale_subject_totals.items())
        },
        "selected_subject_weight_totals_after_rescale": {
            k: v for k, v in sorted(post_subject_totals.items())
        },
        "relative_subject_contribution": relative_contribution,
        "global_rescale_factor": float(global_scale),
        "mean_loss_weight": float(mean_loss_weight),
        "subject_count": len(pre_rescale_subject_totals),
        "selected_example_count": len(scaled),
    }
    for subject_id, total in pre_rescale_subject_totals.items():
        if not math.isclose(
            total, baseline_totals[subject_id], rel_tol=0.0, abs_tol=1e-9
        ):
            raise WindowCapError(
                f"subject {subject_id!r} raw mass not preserved before global rescale: "
                f"{total} != {baseline_totals[subject_id]}"
            )
    if mean_one_rescale and not math.isclose(
        mean_loss_weight, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise WindowCapError(f"capped weights do not average one: {mean_loss_weight}")
    return scaled, audit


def apply_training_window_cap(
    examples: list[dict[str, Any]],
    *,
    fraction: float,
    sampling_seed: int,
    mean_one_rescale: bool = True,
    baseline_input_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """One-stop cap application used by the training hook.

    Returns ``(selected_weighted_examples, audit, mask)``. The input examples
    must already carry the baseline ``raw_loss_weight`` values.
    """
    mask = build_mask(
        examples,
        fraction=fraction,
        sampling_seed=sampling_seed,
        baseline_input_sha256=baseline_input_sha256,
    )
    selected = apply_selection(examples, mask["subjects"])
    weighted, audit = preserve_subject_weight_mass(
        examples, selected, mean_one_rescale=mean_one_rescale
    )
    omitted = [example for example in examples if str(example["sample_id"]) not in {
        sample_id
        for ids in mask["subjects"].values()
        for sample_id in ids
    }]
    omitted_subjects = defaultdict(int)
    omitted_responses: set[str] = set()
    for example in omitted:
        omitted_subjects[str(example["subject_id"])] += 1
        response_id = example.get("response_id")
        if response_id is not None:
            omitted_responses.add(str(response_id))
    audit.update(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "fraction": float(fraction),
            "sampling_seed": int(sampling_seed),
            "selection_sha256": mask["selection_sha256"],
            "available_example_count": len(examples),
            "omitted_example_count": len(omitted),
            "omitted_examples_per_subject": {
                k: v for k, v in sorted(omitted_subjects.items())
            },
            "omitted_response_count": len(omitted_responses),
            "effective_fraction": len(weighted) / len(examples) if examples else 0.0,
        }
    )
    return weighted, audit, mask
