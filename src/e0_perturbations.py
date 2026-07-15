#!/usr/bin/env python3
"""Checkpoint-once E0 modality perturbations for the DAIC K=4 audit.

This module never edits manifests or audio.  It builds the checkpoint's baked,
deterministic subject examples once, optionally materializes an independent
eight-view numeric-ordinal family from the full manifest rows, derives immutable
donor mappings, and scores copied examples with one prompt-only forward per
condition/subject.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch requires this to be present before the first CUDA/cuBLAS handle is
# created when deterministic algorithms are enabled.  Keeping the default here
# makes direct and ``python -m`` invocations equally reproducible, while still
# allowing an explicit caller-provided choice.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from src.data.runtime import (
    build_examples,
    build_prompt_text,
    build_training_text,
    filter_rows_by_subjects,
    load_audio_array,
    load_manifest_rows,
    render_user_prompt_text,
    resolve_audio_placeholder,
)
from src.metrics import binary_auroc, classification_metrics
from src.utils import (
    read_json,
    resolve_label_config,
    resolve_project_path,
    set_seed,
    sha256_file,
    sha256_jsonl_rows,
    sha256_text,
)


SCHEMA_VERSION = 1
SCORER_PROTOCOL = "prompt_only_restricted_legacy_first_label_token_logit_margin_v1"
LEGACY_VIEW_ID = "legacy_deterministic_k4"
LEGACY_VIEW_FAMILY = "legacy"
NUMERIC_BALANCED_VIEW_FAMILY = "numeric_balanced_k4"
NUMERIC_BALANCED_VIEW_COUNT = 8
NUMERIC_BALANCED_POLICY_VERSION = "numeric_ordinal_transposed_cycle_v1"
LEGACY_POSITIVE_LABEL = "Depressed"
LEGACY_NEGATIVE_LABEL = "Non-depressed"


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    audio_source: str
    transcript_source: str
    silence_audio: bool = False


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "real": ConditionSpec("real", "recipient", "recipient"),
    "silence": ConditionSpec("silence", "recipient", "recipient", silence_audio=True),
    "audio_shuffle": ConditionSpec("audio_shuffle", "across_subject", "recipient"),
    "audio_shuffle_same_class": ConditionSpec(
        "audio_shuffle_same_class", "same_class", "recipient"
    ),
    "transcript_shuffle": ConditionSpec(
        "transcript_shuffle", "recipient", "across_subject"
    ),
    "audio_only_real": ConditionSpec("audio_only_real", "recipient", "none"),
    "audio_only_silence": ConditionSpec(
        "audio_only_silence", "recipient", "none", silence_audio=True
    ),
    "audio_only_shuffle": ConditionSpec(
        "audio_only_shuffle", "across_subject", "none"
    ),
}
DEFAULT_CONDITIONS = tuple(CONDITION_SPECS)


def _stable_namespace_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def deterministic_derangement(
    subject_ids: Iterable[str],
    *,
    seed: int,
    namespace: str,
) -> dict[str, str]:
    """Return a deterministic one-to-one, no-fixed-point donor assignment."""
    ordered = sorted(str(subject_id) for subject_id in subject_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"Duplicate subject IDs in derangement namespace={namespace!r}.")
    if len(ordered) < 2:
        raise ValueError(
            f"Derangement namespace={namespace!r} requires at least two subjects; "
            f"received {len(ordered)}."
        )
    shuffled = list(ordered)
    random.Random(_stable_namespace_seed(seed, namespace)).shuffle(shuffled)
    mapping = {
        recipient: shuffled[(index + 1) % len(shuffled)]
        for index, recipient in enumerate(shuffled)
    }
    if set(mapping) != set(ordered) or set(mapping.values()) != set(ordered):
        raise AssertionError("Derangement must be a bijection over the input subjects.")
    if any(recipient == donor for recipient, donor in mapping.items()):
        raise AssertionError("Derangement unexpectedly contains a fixed point.")
    return dict(sorted(mapping.items()))


def build_perturbation_plan(
    examples: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, str]]:
    examples_by_subject = _examples_by_subject(examples)
    subject_ids = sorted(examples_by_subject)
    across_audio = deterministic_derangement(
        subject_ids,
        seed=seed,
        namespace="across_subject_audio_bundle",
    )
    transcript = deterministic_derangement(
        subject_ids,
        seed=seed,
        namespace="across_subject_transcript",
    )

    same_class: dict[str, str] = {}
    labels = sorted({int(example["label"]) for example in examples})
    for label in labels:
        group = [
            subject_id
            for subject_id, example in examples_by_subject.items()
            if int(example["label"]) == label
        ]
        same_class.update(
            deterministic_derangement(
                group,
                seed=seed,
                namespace=f"same_class_audio_bundle_label_{label}",
            )
        )
    same_class = dict(sorted(same_class.items()))
    for recipient, donor in same_class.items():
        if int(examples_by_subject[recipient]["label"]) != int(
            examples_by_subject[donor]["label"]
        ):
            raise AssertionError("Same-class donor mapping crossed a class boundary.")

    return {
        "across_subject_audio": across_audio,
        "same_class_audio": same_class,
        "transcript": transcript,
    }


def _examples_by_subject(
    examples: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_subject: dict[str, dict[str, Any]] = {}
    for example in examples:
        subject_id = str(example["subject_id"])
        if subject_id in by_subject:
            raise ValueError(f"E0 requires one example per subject; duplicate {subject_id!r}.")
        by_subject[subject_id] = example
    if not by_subject:
        raise ValueError("E0 received no subject examples.")
    return by_subject


def validate_legacy_view(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    expected_k: int,
    view_index: int,
) -> dict[str, Any]:
    if view_index != 0:
        raise ValueError(
            "Only the checkpoint's baked legacy view_index=0 exists. "
            f"Use --view-family {NUMERIC_BALANCED_VIEW_FAMILY} for the separate "
            "eight-view materialization."
        )
    configured_k = int(config.get("data", {}).get("chunks_per_subject", 0))
    if configured_k != expected_k:
        raise ValueError(
            f"Checkpoint config has chunks_per_subject={configured_k}, expected {expected_k}."
        )
    bad_subjects = [
        str(example["subject_id"])
        for example in examples
        if len(example.get("audio_paths", [])) != expected_k
        or len(example.get("audio_clip_seconds", [])) != expected_k
    ]
    if bad_subjects:
        raise ValueError(
            f"Legacy view requires exactly K={expected_k} audio paths and clip limits per subject; "
            f"invalid subjects={bad_subjects[:10]}."
        )
    return {
        "view_family": LEGACY_VIEW_FAMILY,
        "view_id": LEGACY_VIEW_ID,
        "view_index": 0,
        "available_views": 1,
        "k": expected_k,
        "selection": "checkpoint_config_evenly_spaced_subject_audio_bundle",
        "extension_contract": (
            "The legacy anchor remains a one-view family and is never pooled with the "
            "separately materialized numeric-balanced family."
        ),
    }


def _numeric_chunk_ordinal(row: dict[str, Any]) -> int:
    """Return a deterministic numeric ordinal without consulting labels/content.

    DAIC's audited manifests have values such as ``random_segment_10``.  The
    runtime legacy builder sorts sample IDs lexically, which places segment 10
    between segments 1 and 2.  The multi-view family instead parses the trailing
    integer.  This is an ordinal convention only; it is not evidence that the
    suffix is a verified interview timestamp.
    """
    for field in ("chunk_id", "sample_id"):
        value = str(row.get(field, "")).strip()
        match = re.search(r"(\d+)$", value)
        if match is not None:
            return int(match.group(1))
    raise ValueError(
        "numeric_balanced_k4 requires a trailing integer in manifest chunk_id "
        f"or sample_id; subject={row.get('subject_id')!r} sample={row.get('sample_id')!r}."
    )


def _numeric_order_subject_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort one subject's rows by numeric suffix and reject ambiguous pools."""
    if not rows:
        raise ValueError("Cannot materialize an E0 view from an empty subject chunk pool.")
    subject_ids = {str(row.get("subject_id")) for row in rows}
    if len(subject_ids) != 1:
        raise ValueError(f"Expected rows for one subject, received {sorted(subject_ids)}.")

    keyed = [(_numeric_chunk_ordinal(row), str(row["sample_id"]), row) for row in rows]
    ordinals = [item[0] for item in keyed]
    sample_ids = [item[1] for item in keyed]
    audio_paths = [str(item[2]["audio_path"]) for item in keyed]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError(
            f"Duplicate numeric chunk ordinals for subject={next(iter(subject_ids))!r}: "
            f"{ordinals}."
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            f"Duplicate sample IDs for subject={next(iter(subject_ids))!r}."
        )
    if len(audio_paths) != len(set(audio_paths)):
        raise ValueError(
            f"Duplicate audio paths for subject={next(iter(subject_ids))!r}."
        )
    return [item[2] for item in sorted(keyed, key=lambda item: (item[0], item[1]))]


def _circular_gaps(selection: tuple[int, ...], num_chunks: int) -> tuple[int, ...]:
    ordered = tuple(sorted(selection))
    return tuple(
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
    ) + (num_chunks + ordered[0] - ordered[-1],)


def _numeric_balanced_schedule(
    num_chunks: int,
    *,
    k: int = 4,
    view_count: int = NUMERIC_BALANCED_VIEW_COUNT,
) -> dict[str, Any]:
    """Build eight unique, exposure-balanced, content-independent K-sets.

    The 32 view slots are a transposed traversal of a modular cycle.  A step
    coprime to the pool size makes the traversal a permutation, so every chunk
    receives either floor(32/N) or ceil(32/N) exposures.  Among valid cycles we
    deterministically select the one with the best worst circular separation,
    then the lowest total gap imbalance and finally the smallest step.
    """
    if view_count != NUMERIC_BALANCED_VIEW_COUNT:
        raise ValueError(
            f"{NUMERIC_BALANCED_VIEW_FAMILY} supports exactly "
            f"{NUMERIC_BALANCED_VIEW_COUNT} views, received {view_count}."
        )
    if k != 4:
        raise ValueError(
            f"{NUMERIC_BALANCED_VIEW_FAMILY} is pinned to K=4, received K={k}."
        )
    if num_chunks < k:
        raise ValueError(
            f"A K={k} view requires at least {k} available chunks; received {num_chunks}."
        )

    candidates: list[tuple[tuple[Any, ...], int, tuple[tuple[int, ...], ...]]] = []
    total_slots = view_count * k
    for step in range(1, num_chunks):
        if math.gcd(step, num_chunks) != 1:
            continue
        schedule = tuple(
            tuple(
                sorted(
                    ((view_index + slot_index * view_count) * step) % num_chunks
                    for slot_index in range(k)
                )
            )
            for view_index in range(view_count)
        )
        if any(len(set(selection)) != k for selection in schedule):
            continue
        signatures = [frozenset(selection) for selection in schedule]
        if len(set(signatures)) != view_count:
            continue
        exposure_counts = tuple(
            sum(ordinal in selection for selection in schedule)
            for ordinal in range(num_chunks)
        )
        low, remainder = divmod(total_slots, num_chunks)
        high = low + int(remainder > 0)
        if any(count not in {low, high} for count in exposure_counts):
            continue
        if sum(count == high for count in exposure_counts) != remainder:
            continue
        gaps = tuple(
            gap
            for selection in schedule
            for gap in _circular_gaps(selection, num_chunks)
        )
        minimum_gap = min(gaps)
        gap_imbalance = sum((gap * k - num_chunks) ** 2 for gap in gaps)
        rank = (-minimum_gap, gap_imbalance, step, schedule)
        candidates.append((rank, step, schedule))

    if not candidates:
        raise ValueError(
            "Could not construct eight distinct, exposure-balanced numeric K=4 views "
            f"for a subject with {num_chunks} chunks."
        )
    _, step, schedule = min(candidates, key=lambda item: item[0])
    exposure_counts = [
        sum(ordinal in selection for selection in schedule)
        for ordinal in range(num_chunks)
    ]
    payload: dict[str, Any] = {
        "policy_version": NUMERIC_BALANCED_POLICY_VERSION,
        "num_chunks": num_chunks,
        "k": k,
        "view_count": view_count,
        "modular_step": step,
        "view_ordinal_indices_zero_based": [list(selection) for selection in schedule],
        "exposure_counts_by_ordinal": exposure_counts,
        "minimum_exposure": min(exposure_counts),
        "maximum_exposure": max(exposure_counts),
    }
    payload["schedule_sha256"] = sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return payload


def materialize_numeric_balanced_view(
    legacy_examples: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    expected_k: int,
    view_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize one member of the independent eight-view numeric family."""
    if not 0 <= view_index < NUMERIC_BALANCED_VIEW_COUNT:
        raise ValueError(
            f"{NUMERIC_BALANCED_VIEW_FAMILY} view_index must be in "
            f"[0, {NUMERIC_BALANCED_VIEW_COUNT - 1}], received {view_index}."
        )
    if expected_k != 4:
        raise ValueError(
            f"{NUMERIC_BALANCED_VIEW_FAMILY} requires expected_k=4, received {expected_k}."
        )
    # The checkpoint-built examples remain the authoritative source for every
    # non-audio field and for the per-chunk clip budget.
    validate_legacy_view(
        legacy_examples,
        config,
        expected_k=expected_k,
        view_index=0,
    )

    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        rows_by_subject.setdefault(str(row["subject_id"]), []).append(row)
    examples_by_subject = _examples_by_subject(legacy_examples)
    if set(rows_by_subject) != set(examples_by_subject):
        raise ValueError(
            "Numeric-view manifest rows and checkpoint-built examples must contain "
            "exactly the same subjects."
        )

    source_hash = sha256_jsonl_rows(legacy_examples)
    materialized: list[dict[str, Any]] = []
    schedule_catalog: dict[str, dict[str, Any]] = {}
    selected_sample_ids_by_subject: dict[str, list[str]] = {}
    all_view_sample_ids_by_subject: dict[str, list[list[str]]] = {}
    exposure_counts_by_subject: dict[str, dict[str, int]] = {}
    immutable_fields = (
        "dataset",
        "subject_id",
        "sample_id",
        "label",
        "label_text",
        "internal_label_text",
        "transcript",
        "input_modality",
        "prompt_text",
        "training_text",
        "question_id",
    )

    for legacy_example in legacy_examples:
        subject_id = str(legacy_example["subject_id"])
        ordered_rows = _numeric_order_subject_rows(rows_by_subject[subject_id])
        num_chunks = len(ordered_rows)
        schedule = _numeric_balanced_schedule(num_chunks, k=expected_k)
        schedule_key = str(num_chunks)
        if schedule_key in schedule_catalog and schedule_catalog[schedule_key] != schedule:
            raise AssertionError("Numeric schedule changed for an identical pool size.")
        schedule_catalog[schedule_key] = schedule

        all_selections: list[list[str]] = []
        for ordinal_selection in schedule["view_ordinal_indices_zero_based"]:
            all_selections.append(
                [str(ordered_rows[ordinal]["sample_id"]) for ordinal in ordinal_selection]
            )
        signatures = [frozenset(selection) for selection in all_selections]
        if len(set(signatures)) != NUMERIC_BALANCED_VIEW_COUNT:
            raise ValueError(
                "Duplicate chunk bundles are masquerading as distinct views for "
                f"subject={subject_id}."
            )

        selected_ordinals = schedule["view_ordinal_indices_zero_based"][view_index]
        selected_rows = [ordered_rows[ordinal] for ordinal in selected_ordinals]
        selected_sample_ids = [str(row["sample_id"]) for row in selected_rows]
        selected_audio_paths = [str(row["audio_path"]) for row in selected_rows]
        selected_numeric_ordinals = [_numeric_chunk_ordinal(row) for row in selected_rows]
        if selected_numeric_ordinals != sorted(selected_numeric_ordinals):
            raise AssertionError("A numeric-balanced view is not in ascending ordinal order.")

        clip_seconds = list(legacy_example["audio_clip_seconds"])
        if not clip_seconds or any(value != clip_seconds[0] for value in clip_seconds):
            raise ValueError(
                "Numeric view materialization requires the checkpoint's K legacy chunks "
                f"to share one clip budget; subject={subject_id} has {clip_seconds}."
            )
        changed = copy.deepcopy(legacy_example)
        changed["audio_paths"] = selected_audio_paths
        changed["audio_clip_seconds"] = [clip_seconds[0]] * expected_k
        changed["subject_chunk_paths"] = [str(row["audio_path"]) for row in ordered_rows]
        changed["chunks_per_subject"] = expected_k
        for field in immutable_fields:
            if changed.get(field) != legacy_example.get(field):
                raise AssertionError(
                    f"Numeric view unexpectedly changed immutable field={field!r} "
                    f"for subject={subject_id}."
                )
        if len(changed["audio_paths"]) != expected_k:
            raise AssertionError("Numeric view did not preserve fixed K=4.")
        materialized.append(changed)
        selected_sample_ids_by_subject[subject_id] = selected_sample_ids
        all_view_sample_ids_by_subject[subject_id] = all_selections
        exposure_counts_by_subject[subject_id] = {
            str(row["sample_id"]): int(count)
            for row, count in zip(
                ordered_rows,
                schedule["exposure_counts_by_ordinal"],
            )
        }

    selection_payload = {
        "schedule_catalog_by_num_chunks": schedule_catalog,
        "all_view_selected_sample_ids_by_subject": all_view_sample_ids_by_subject,
    }
    selection_schedule_sha256 = sha256_text(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":"))
    )
    view_spec = {
        "view_family": NUMERIC_BALANCED_VIEW_FAMILY,
        "view_id": (
            f"{NUMERIC_BALANCED_VIEW_FAMILY}_view_{view_index}_of_"
            f"{NUMERIC_BALANCED_VIEW_COUNT}"
        ),
        "view_index": view_index,
        "available_views": NUMERIC_BALANCED_VIEW_COUNT,
        "k": expected_k,
        "selection_policy_version": NUMERIC_BALANCED_POLICY_VERSION,
        "selection": (
            "content-independent transposed modular-cycle schedule over full per-subject "
            "manifest rows sorted by trailing numeric chunk ordinal"
        ),
        "labels_or_content_used_for_selection": False,
        "numeric_ordinal_source": (
            "trailing integer in chunk_id, falling back to trailing integer in sample_id"
        ),
        "chronology_caveat": (
            "Numeric suffix order is a deterministic ordinal convention; the manifest does "
            "not provide timestamps proving that it is interview chronology."
        ),
        "aggregation_boundary": (
            "This eight-view family is independent of the checkpoint-baked legacy lexical "
            "replication anchor; legacy results must not be pooled into its aggregate."
        ),
        "schedule_catalog_by_num_chunks": schedule_catalog,
        "selection_schedule_sha256": selection_schedule_sha256,
        "selected_sample_ids_by_subject": selected_sample_ids_by_subject,
        "all_view_selected_sample_ids_by_subject": all_view_sample_ids_by_subject,
        "exposure_counts_by_subject": exposure_counts_by_subject,
        "source_legacy_examples_sha256": source_hash,
        "materialized_examples_sha256": sha256_jsonl_rows(materialized),
        "invariants": {
            "fixed_subject_split_transcript_prompt": True,
            "fixed_k": expected_k,
            "unique_bundle_per_subject_per_view": True,
            "numeric_ascending_audio_order": True,
            "balanced_exposure_floor_or_ceil": True,
        },
    }
    return materialized, view_spec


def _assert_supported_examples(examples: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if bool(config.get("data", {}).get("use_emotion", False)):
        raise ValueError("E0 legacy harness does not support emotion-caption prompt perturbations.")
    if str(config.get("data", {}).get("sample_mode", "")).lower() != "subject_audio":
        raise ValueError("E0 legacy harness requires data.sample_mode=subject_audio.")
    unsupported_fields = {
        "chunk_caption_by_path",
        "emotion_user_text",
        "emotion_system_prompt",
    }
    for example in examples:
        if str(example.get("question_id", "")) != "subject_audio_bundle":
            raise ValueError(
                "E0 legacy prompt reconstruction requires question_id='subject_audio_bundle'; "
                f"subject={example['subject_id']} has {example.get('question_id')!r}."
            )
        present = unsupported_fields.intersection(example)
        if present:
            raise ValueError(
                f"Unsupported emotion prompt fields for subject={example['subject_id']}: "
                f"{sorted(present)}"
            )


def _subject_audio_context(k: int) -> str:
    return (
        f"The subject's speech audio is provided in {k} "
        f"segment{'s' if k != 1 else ''} sampled from the interview."
    )


def _render_condition_prompt(
    example: dict[str, Any],
    config: dict[str, Any],
    *,
    use_text: bool,
) -> tuple[str, str]:
    condition_config = copy.deepcopy(config)
    condition_config.setdefault("data", {})["use_audio"] = True
    condition_config["data"]["use_text"] = bool(use_text)
    transcript = str(example["transcript"]) if use_text else ""
    k = len(example["audio_paths"])
    user_text = render_user_prompt_text(
        condition_config,
        transcript,
        is_subject_bundle=True,
        audio_context_override=_subject_audio_context(k),
    )
    prompt_text = build_prompt_text(
        system_prompt=str(condition_config["prompt"]["system"]),
        user_text=user_text,
        num_audios=k,
        use_audio=True,
        audio_placeholder=resolve_audio_placeholder(condition_config),
    )
    return prompt_text, "audio_text" if use_text else "audio_only"


def build_condition_example(
    recipient: dict[str, Any],
    examples_by_subject: dict[str, dict[str, Any]],
    plan: dict[str, dict[str, str]],
    condition: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if condition not in CONDITION_SPECS:
        raise ValueError(f"Unsupported E0 condition={condition!r}.")
    spec = CONDITION_SPECS[condition]
    recipient_id = str(recipient["subject_id"])

    if spec.audio_source == "recipient":
        audio_source_id = recipient_id
    elif spec.audio_source == "across_subject":
        audio_source_id = plan["across_subject_audio"][recipient_id]
    elif spec.audio_source == "same_class":
        audio_source_id = plan["same_class_audio"][recipient_id]
    else:
        raise AssertionError(f"Unhandled audio source {spec.audio_source!r}.")

    if spec.transcript_source == "recipient":
        transcript_source_id: str | None = recipient_id
        transcript = str(recipient["transcript"])
    elif spec.transcript_source == "across_subject":
        transcript_source_id = plan["transcript"][recipient_id]
        transcript = str(examples_by_subject[transcript_source_id]["transcript"])
    elif spec.transcript_source == "none":
        transcript_source_id = None
        transcript = ""
    else:
        raise AssertionError(f"Unhandled transcript source {spec.transcript_source!r}.")

    audio_donor = examples_by_subject[audio_source_id]
    if len(audio_donor["audio_paths"]) != len(recipient["audio_paths"]):
        raise ValueError(
            f"Audio donor K mismatch recipient={recipient_id} donor={audio_source_id}: "
            f"{len(recipient['audio_paths'])} != {len(audio_donor['audio_paths'])}."
        )
    transformed = copy.deepcopy(recipient)
    transformed["audio_paths"] = copy.deepcopy(audio_donor["audio_paths"])
    transformed["audio_clip_seconds"] = copy.deepcopy(audio_donor["audio_clip_seconds"])
    transformed["subject_chunk_paths"] = copy.deepcopy(
        audio_donor.get("subject_chunk_paths", audio_donor["audio_paths"])
    )
    transformed["chunks_per_subject"] = len(transformed["audio_paths"])
    transformed["transcript"] = transcript
    if spec.transcript_source == "recipient":
        # Audio swaps preserve the recipient's exact prompt bytes and placeholders.
        prompt_text = str(recipient["prompt_text"])
        input_modality = str(recipient["input_modality"])
    else:
        prompt_text, input_modality = _render_condition_prompt(
            transformed,
            config,
            use_text=transcript_source_id is not None,
        )
    transformed["prompt_text"] = prompt_text
    transformed["training_text"] = build_training_text(
        prompt_text,
        str(transformed["internal_label_text"]),
    )
    transformed["input_modality"] = input_modality
    transformed["e0_condition"] = condition
    transformed["e0_audio_source_subject_id"] = audio_source_id
    transformed["e0_transcript_source_subject_id"] = transcript_source_id
    transformed["e0_silence_audio"] = bool(spec.silence_audio)

    if len(transformed["audio_paths"]) != len(transformed["audio_clip_seconds"]):
        raise AssertionError("Audio paths and clip limits must remain an indivisible bundle.")
    if audio_source_id != recipient_id and transformed["audio_paths"] != audio_donor["audio_paths"]:
        raise AssertionError("Audio perturbation did not copy the donor's complete baked bundle.")
    return transformed


class DirectFirstTokenScorer:
    """Restricted two-token classifier from one prompt-only model forward."""

    def __init__(
        self,
        model,
        processor,
        config: dict[str, Any],
        device: torch.device,
        *,
        include_candidate_likelihood: bool = False,
        audio_loader: Callable[[str, int, float | None, bool], Any] = load_audio_array,
    ) -> None:
        labels = resolve_label_config(config)
        if (
            labels["label_vocab_version"] != "legacy_english_labels"
            or labels["internal_positive_label"] != LEGACY_POSITIVE_LABEL
            or labels["internal_negative_label"] != LEGACY_NEGATIVE_LABEL
        ):
            raise ValueError(
                "E0 primary scorer is checkpoint-specific and requires legacy labels "
                f"{LEGACY_POSITIVE_LABEL!r}/{LEGACY_NEGATIVE_LABEL!r}; resolved labels={labels}."
            )
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.positive_label = labels["internal_positive_label"]
        self.negative_label = labels["internal_negative_label"]
        self.include_candidate_likelihood = bool(include_candidate_likelihood)
        self.audio_loader = audio_loader

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _encode_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Unexpected tokenizer input_ids shape={tuple(input_ids.shape)}.")
        return input_ids[0]

    def _first_continuation_token_id(self, prompt_text: str, label_text: str) -> int:
        prompt_ids = self._encode_text(prompt_text)
        full_ids = self._encode_text(prompt_text + label_text)
        prompt_len = int(prompt_ids.shape[0])
        if full_ids.shape[0] <= prompt_len or not torch.equal(full_ids[:prompt_len], prompt_ids):
            raise ValueError(
                f"Tokenizer boundary changed when appending label={label_text!r}; "
                "the prompt-only first-token protocol is undefined for this template/tokenizer."
            )
        return int(full_ids[prompt_len].item())

    def _sampling_rate(self) -> int:
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("Audio E0 scoring requires processor.feature_extractor.sampling_rate.")
        return int(sampling_rate)

    def _audio_arrays(self, example: dict[str, Any], silence_audio: bool) -> list[Any]:
        paths = list(example["audio_paths"])
        clip_seconds = list(example["audio_clip_seconds"])
        if not paths or len(paths) != len(clip_seconds):
            raise ValueError("E0 scoring requires a non-empty, aligned audio bundle.")
        sampling_rate = self._sampling_rate()
        return [
            self.audio_loader(path, sampling_rate, max_seconds, silence_audio)
            for path, max_seconds in zip(paths, clip_seconds)
        ]

    def _processor_inputs(
        self,
        text: str,
        audio_arrays: list[Any],
    ) -> dict[str, torch.Tensor]:
        inputs = self.processor(
            text=text,
            audio=audio_arrays,
            sampling_rate=self._sampling_rate(),
            return_tensors="pt",
            padding=False,
        )
        return dict(inputs)

    def _to_device(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    @staticmethod
    def _validate_processor_continuation(
        prompt_inputs: dict[str, torch.Tensor],
        full_inputs: dict[str, torch.Tensor],
        *,
        expected_first_token_id: int,
        label_text: str,
    ) -> tuple[int, int]:
        prompt_ids = prompt_inputs["input_ids"]
        full_ids = full_inputs["input_ids"]
        if prompt_ids.ndim != 2 or full_ids.ndim != 2 or prompt_ids.shape[0] != 1 or full_ids.shape[0] != 1:
            raise ValueError(
                f"Processor returned invalid input shapes prompt={tuple(prompt_ids.shape)} "
                f"full={tuple(full_ids.shape)} for label={label_text!r}."
            )
        prompt_len = int(prompt_ids.shape[1])
        continuation_len = int(full_ids.shape[1]) - prompt_len
        if continuation_len < 1 or not torch.equal(full_ids[:, :prompt_len], prompt_ids):
            raise ValueError(
                f"Audio processor sequence for label={label_text!r} does not preserve the "
                f"expanded prompt prefix (prompt_len={prompt_len}, full_len={full_ids.shape[1]})."
            )
        processor_first_token_id = int(full_ids[0, prompt_len].item())
        if processor_first_token_id != expected_first_token_id:
            raise ValueError(
                f"Tokenizer/processor first-token disagreement for label={label_text!r}: "
                f"tokenizer={expected_first_token_id}, processor={processor_first_token_id}."
            )
        return prompt_len, continuation_len

    def _candidate_score(
        self,
        prompt_inputs: dict[str, torch.Tensor],
        full_inputs: dict[str, torch.Tensor],
        label_text: str,
    ) -> float:
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        target_ids = full_inputs["input_ids"][0, prompt_len:]
        if target_ids.numel() == 0:
            raise ValueError(f"Candidate label {label_text!r} produced no continuation tokens.")
        with torch.inference_mode():
            outputs = self.model(**full_inputs, use_cache=False)
            logits = outputs.logits[0]
            selected = logits[prompt_len - 1 : full_inputs["input_ids"].shape[1] - 1]
            if int(selected.shape[0]) != int(target_ids.shape[0]):
                raise ValueError(
                    f"Candidate alignment mismatch for label={label_text!r}: "
                    f"selected_logits={selected.shape[0]} targets={target_ids.shape[0]}."
                )
            token_log_probs = torch.log_softmax(selected, dim=-1).gather(
                -1,
                target_ids.unsqueeze(-1),
            )
        return float(token_log_probs.mean().item())

    def score(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(example["prompt_text"])
        positive_token_id = self._first_continuation_token_id(
            prompt_text,
            self.positive_label,
        )
        negative_token_id = self._first_continuation_token_id(
            prompt_text,
            self.negative_label,
        )
        if positive_token_id == negative_token_id:
            raise ValueError("Positive and negative first-label token IDs must be distinct.")

        audio_arrays = self._audio_arrays(example, bool(example["e0_silence_audio"]))
        prompt_inputs_cpu = self._processor_inputs(prompt_text, audio_arrays)
        positive_full_inputs_cpu = self._processor_inputs(
            prompt_text + self.positive_label,
            audio_arrays,
        )
        negative_full_inputs_cpu = self._processor_inputs(
            prompt_text + self.negative_label,
            audio_arrays,
        )
        prompt_len, positive_continuation_len = self._validate_processor_continuation(
            prompt_inputs_cpu,
            positive_full_inputs_cpu,
            expected_first_token_id=positive_token_id,
            label_text=self.positive_label,
        )
        negative_prompt_len, negative_continuation_len = self._validate_processor_continuation(
            prompt_inputs_cpu,
            negative_full_inputs_cpu,
            expected_first_token_id=negative_token_id,
            label_text=self.negative_label,
        )
        if negative_prompt_len != prompt_len:
            raise AssertionError("Expanded prompt length changed between label validations.")
        prompt_inputs = self._to_device(prompt_inputs_cpu)
        with torch.inference_mode():
            outputs = self.model(**prompt_inputs, use_cache=False)
            if int(outputs.logits.shape[1]) != prompt_len:
                raise ValueError(
                    f"Prompt-only model output length={outputs.logits.shape[1]} does not match "
                    f"expanded processor prompt length={prompt_len}."
                )
            next_token_logits = outputs.logits[0, -1]
            positive_logit = float(next_token_logits[positive_token_id].item())
            negative_logit = float(next_token_logits[negative_token_id].item())
        del outputs, next_token_logits
        margin = positive_logit - negative_logit
        prediction = int(margin > 0.0)
        result: dict[str, Any] = {
            "scorer_protocol": SCORER_PROTOCOL,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "positive_first_token_id": positive_token_id,
            "negative_first_token_id": negative_token_id,
            "positive_first_token_text": self.tokenizer.decode([positive_token_id]),
            "negative_first_token_text": self.tokenizer.decode([negative_token_id]),
            "positive_first_token_logit": positive_logit,
            "negative_first_token_logit": negative_logit,
            "first_token_margin": margin,
            "first_token_prediction": prediction,
            "expanded_prompt_input_tokens": prompt_len,
            "positive_label_continuation_tokens": positive_continuation_len,
            "negative_label_continuation_tokens": negative_continuation_len,
            # Compatibility aliases for the existing paired E0 comparison script.
            "dep_score": positive_logit,
            "non_score": negative_logit,
            "likelihood_prediction": prediction,
        }
        if self.include_candidate_likelihood:
            positive_full_inputs = self._to_device(positive_full_inputs_cpu)
            candidate_positive = self._candidate_score(
                prompt_inputs,
                positive_full_inputs,
                self.positive_label,
            )
            del positive_full_inputs
            negative_full_inputs = self._to_device(negative_full_inputs_cpu)
            candidate_negative = self._candidate_score(
                prompt_inputs,
                negative_full_inputs,
                self.negative_label,
            )
            result.update(
                {
                    "candidate_positive_mean_logprob": candidate_positive,
                    "candidate_negative_mean_logprob": candidate_negative,
                    "candidate_likelihood_margin": candidate_positive - candidate_negative,
                    "candidate_likelihood_prediction": int(
                        candidate_positive > candidate_negative
                    ),
                }
            )
        return result


def _atomic_write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E0 artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, payload: Any) -> None:
    _atomic_write_text_new(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    serialized = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _atomic_write_text_new(path, serialized)


def _load_checkpoint_config(config_snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not config_snapshot.is_file():
        raise FileNotFoundError(f"Checkpoint config snapshot not found: {config_snapshot}")
    payload = yaml.safe_load(config_snapshot.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {config_snapshot}.")
    if isinstance(payload.get("config"), dict):
        config = copy.deepcopy(payload["config"])
        envelope = {key: value for key, value in payload.items() if key != "config"}
    else:
        config = copy.deepcopy(payload)
        envelope = {}
    if str(config.get("dataset", "")).lower() != "daic":
        raise ValueError("The authorized E0 harness is limited to the DAIC checkpoint.")
    labels = resolve_label_config(config)
    if (
        labels["label_vocab_version"] != "legacy_english_labels"
        or labels["internal_positive_label"] != LEGACY_POSITIVE_LABEL
        or labels["internal_negative_label"] != LEGACY_NEGATIVE_LABEL
    ):
        raise ValueError(f"Checkpoint snapshot is not the required legacy-label checkpoint: {labels}")
    return config, envelope


def _resolve_input_examples(
    config: dict[str, Any],
    manifest_metadata_path: Path,
    *,
    partition: str,
    expected_k: int,
    view_family: str,
    view_index: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    metadata = read_json(manifest_metadata_path)
    if str(metadata.get("dataset", "")).lower() != "daic":
        raise ValueError(f"Expected DAIC manifest metadata, received {metadata.get('dataset')!r}.")
    manifest_path = resolve_project_path(metadata["manifest_path"])
    partition_path = resolve_project_path(metadata["subject_partition_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    observed_manifest_hash = sha256_jsonl_rows(manifest_rows)
    expected_manifest_hash = str(metadata.get("manifest_hash", ""))
    if expected_manifest_hash and observed_manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"Manifest hash mismatch: observed={observed_manifest_hash} "
            f"expected={expected_manifest_hash}."
        )
    partition_rows = read_json(partition_path)
    subject_ids = sorted(
        str(row["subject_id"])
        for row in partition_rows
        if str(row["partition"]) == partition
    )
    if not subject_ids:
        raise ValueError(f"No subjects found in partition={partition!r}.")
    final_rows = filter_rows_by_subjects(manifest_rows, subject_ids)
    examples = build_examples(
        final_rows,
        config,
        partition_name=f"e0_{partition}",
        truncation_log_path=None,
    )
    if sorted(str(example["subject_id"]) for example in examples) != subject_ids:
        raise ValueError("Built E0 examples do not match the partition subject IDs exactly.")
    _assert_supported_examples(examples, config)
    if view_family == LEGACY_VIEW_FAMILY:
        view_spec = validate_legacy_view(
            examples,
            config,
            expected_k=expected_k,
            view_index=view_index,
        )
    elif view_family == NUMERIC_BALANCED_VIEW_FAMILY:
        examples, view_spec = materialize_numeric_balanced_view(
            examples,
            final_rows,
            config,
            expected_k=expected_k,
            view_index=view_index,
        )
    else:
        raise ValueError(f"Unsupported E0 view_family={view_family!r}.")
    return examples, view_spec, metadata, manifest_path, partition_path


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _artifact_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    unique_paths = sorted({path.resolve() for path in paths if path.is_file()})
    return [_artifact_entry(path) for path in unique_paths]


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short", "--untracked-files=all").splitlines(),
        "tracked_diff_sha256": sha256_text(diff),
        "harness_source": _artifact_entry(Path(__file__)),
    }


def _input_provenance(
    *,
    project_root: Path,
    config_snapshot: Path,
    manifest_metadata_path: Path,
    manifest_path: Path,
    partition_path: Path,
    checkpoint_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    checkpoint_files = [path for path in checkpoint_dir.iterdir() if path.is_file()]
    model_files = [
        path
        for path in model_path.iterdir()
        if path.is_file()
        and (
            path.name == "config.json"
            or path.name == "model.safetensors.index.json"
            or path.suffix == ".safetensors"
        )
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "git": _git_provenance(project_root),
        "inputs": {
            "config_snapshot": _artifact_entry(config_snapshot),
            "manifest_metadata": _artifact_entry(manifest_metadata_path),
            "manifest": _artifact_entry(manifest_path),
            "subject_partitions": _artifact_entry(partition_path),
            "checkpoint_files": _artifact_inventory(checkpoint_files),
            "base_model_files": _artifact_inventory(model_files),
        },
    }


def _model_dtype_metadata(model) -> dict[str, Any]:
    parameter_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()}
    )
    try:
        primary = str(next(model.parameters()).dtype).removeprefix("torch.")
    except StopIteration:
        primary = "unknown"
    return {"model_dtype": primary, "model_parameter_dtypes": parameter_dtypes}


def _runtime_metadata(model, device: torch.device, elapsed_seconds: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": str(device),
        "elapsed_seconds": float(elapsed_seconds),
        **_model_dtype_metadata(model),
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": None,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        payload.update(
            {
                "cuda_device_name": torch.cuda.get_device_properties(device).name,
                "cuda_max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "cuda_max_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
        )
    return payload


def _condition_assignments(
    transformed_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "recipient_subject_id": str(example["subject_id"]),
            "audio_source_subject_id": str(example["e0_audio_source_subject_id"]),
            "transcript_source_subject_id": example["e0_transcript_source_subject_id"],
            "silence_audio": bool(example["e0_silence_audio"]),
        }
        for example in transformed_examples
    ]


def _condition_interpretation(condition: str) -> dict[str, str]:
    if CONDITION_SPECS[condition].transcript_source == "none":
        return {
            "control_role": "text_removed_control",
            "guardrail": (
                "This is not a clean separately trained audio-only model. The checkpoint's "
                "original system prompt still mentions transcript information; only the user "
                "transcript block is removed and its decision basis is changed to audio."
            ),
        }
    return {
        "control_role": "paired_modality_perturbation",
        "guardrail": (
            "Sensitivity under this paired condition is causal evidence about the supplied "
            "input bundle, not by itself evidence of clinically valid acoustic reasoning."
        ),
    }


def _binary_average_precision(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("AUPRC labels and scores must have the same length.")
    positives = sum(int(label) == 1 for label in labels)
    if positives == 0:
        return 0.0
    grouped: dict[float, list[int]] = {}
    for label, score in zip(labels, scores):
        grouped.setdefault(float(score), []).append(int(label))
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positives += sum(label == 1 for label in group)
        false_positives += sum(label == 0 for label in group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["first_token_prediction"]) for row in rows]
    margins = [float(row["first_token_margin"]) for row in rows]
    primary_metrics = classification_metrics(labels, predictions)
    summary: dict[str, Any] = {
        **primary_metrics,
        "balanced_accuracy": float(primary_metrics["macro_recall"]),
        "auroc": binary_auroc(labels, margins),
        "auprc": _binary_average_precision(labels, margins),
        "num_subjects": len(rows),
        "mean_first_token_margin": float(sum(margins) / len(margins)),
        "score_sign_mismatches": sum(
            int(int(row["first_token_prediction"]) != int(float(row["first_token_margin"]) > 0.0))
            for row in rows
        ),
    }
    if all("candidate_likelihood_margin" in row for row in rows):
        candidate_margins = [float(row["candidate_likelihood_margin"]) for row in rows]
        candidate_predictions = [
            int(row["candidate_likelihood_prediction"]) for row in rows
        ]
        candidate_metrics = classification_metrics(labels, candidate_predictions)
        summary["candidate_likelihood_secondary"] = {
            **candidate_metrics,
            "balanced_accuracy": float(candidate_metrics["macro_recall"]),
            "auroc": binary_auroc(labels, candidate_margins),
            "auprc": _binary_average_precision(labels, candidate_margins),
            "mean_margin": float(sum(candidate_margins) / len(candidate_margins)),
            "score_sign_mismatches": sum(
                int(
                    int(row["candidate_likelihood_prediction"])
                    != int(float(row["candidate_likelihood_margin"]) > 0.0)
                )
                for row in rows
            ),
        }
    return summary


def run_condition(
    *,
    condition: str,
    recipient_examples: list[dict[str, Any]],
    examples_by_subject: dict[str, dict[str, Any]],
    plan: dict[str, dict[str, str]],
    scorer: DirectFirstTokenScorer,
    model,
    config: dict[str, Any],
    view_spec: dict[str, Any],
    seed: int,
    output_root: Path,
    input_provenance: dict[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    condition_dir = output_root / condition
    if condition_dir.exists() and any(condition_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite condition directory: {condition_dir}")
    condition_dir.mkdir(parents=True, exist_ok=True)
    transformed_examples = [
        build_condition_example(
            recipient,
            examples_by_subject,
            plan,
            condition,
            config,
        )
        for recipient in recipient_examples
    ]
    if [str(example["subject_id"]) for example in transformed_examples] != [
        str(example["subject_id"]) for example in recipient_examples
    ]:
        raise AssertionError("Condition transform broke paired recipient ordering.")

    if scorer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(scorer.device)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(transformed_examples, start=1):
        scores = scorer.score(example)
        row = {
            "schema_version": SCHEMA_VERSION,
            "condition": condition,
            "view_id": view_spec["view_id"],
            "view_index": view_spec["view_index"],
            "subject_id": str(example["subject_id"]),
            "sample_id": str(example["sample_id"]),
            "label": int(example["label"]),
            "label_text": str(example["label_text"]),
            "audio_source_subject_id": str(example["e0_audio_source_subject_id"]),
            "transcript_source_subject_id": example["e0_transcript_source_subject_id"],
            "silence_audio": bool(example["e0_silence_audio"]),
            **_condition_interpretation(condition),
            "audio_paths": list(example["audio_paths"]),
            "audio_clip_seconds": list(example["audio_clip_seconds"]),
            "transcript_sha256": sha256_text(str(example["transcript"])),
            "prompt_sha256": sha256_text(str(example["prompt_text"])),
            **scores,
        }
        rows.append(row)
        if index % max(1, progress_every) == 0 or index == len(transformed_examples):
            print(
                f"E0 condition={condition} progress={index}/{len(transformed_examples)} "
                f"subject={example['subject_id']} margin={row['first_token_margin']:.6f}",
                flush=True,
            )
    runtime = _runtime_metadata(model, scorer.device, time.monotonic() - started)
    summary = {
        **_condition_summary(rows),
        **_condition_interpretation(condition),
        "runtime": runtime,
    }
    token_pairs = sorted(
        {
            (int(row["positive_first_token_id"]), int(row["negative_first_token_id"]))
            for row in rows
        }
    )
    if len(token_pairs) != 1:
        raise ValueError(f"First-label token IDs changed across subjects: {token_pairs}")

    condition_config = {
        "schema_version": SCHEMA_VERSION,
        "condition": asdict(CONDITION_SPECS[condition]),
        "seed": int(seed),
        "view": view_spec,
        "scorer": {
            "primary_protocol": SCORER_PROTOCOL,
            "positive_label": LEGACY_POSITIVE_LABEL,
            "negative_label": LEGACY_NEGATIVE_LABEL,
            "positive_first_token_id": token_pairs[0][0],
            "negative_first_token_id": token_pairs[0][1],
            "candidate_likelihood_secondary": scorer.include_candidate_likelihood,
            "compatibility_aliases": {
                "dep_score": "positive_first_token_logit",
                "non_score": "negative_first_token_logit",
                "likelihood_prediction": "first_token_prediction",
            },
        },
        "audio_only_prompt_policy": (
            "retain checkpoint system prompt; remove transcript user block; "
            "change user decision basis to audio"
        ),
        "interpretation": _condition_interpretation(condition),
        "assignments": _condition_assignments(transformed_examples),
        "resolved_checkpoint_config": config,
    }
    predictions_path = condition_dir / "predictions_subject_level.jsonl"
    config_path = condition_dir / "condition_config.json"
    summary_path = condition_dir / "condition_summary.json"
    _write_jsonl_new(predictions_path, rows)
    _write_json_new(config_path, condition_config)
    _write_json_new(summary_path, summary)
    provenance = {
        **input_provenance,
        "condition": condition,
        "interpretation": _condition_interpretation(condition),
        "view": view_spec,
        "outputs": {
            "predictions": _artifact_entry(predictions_path),
            "condition_config": _artifact_entry(config_path),
            "condition_summary": _artifact_entry(summary_path),
        },
    }
    _write_json_new(condition_dir / "provenance.json", provenance)
    return {
        "condition": condition,
        "condition_dir": str(condition_dir),
        "summary": summary,
        "predictions_sha256": provenance["outputs"]["predictions"]["sha256"],
    }


def validate_transform_plan(
    examples: list[dict[str, Any]],
    plan: dict[str, dict[str, str]],
    config: dict[str, Any],
    conditions: Iterable[str],
) -> dict[str, Any]:
    examples_by_subject = _examples_by_subject(examples)
    before_hash = sha256_jsonl_rows(examples)
    condition_hashes: dict[str, str] = {}
    for condition in conditions:
        transformed = [
            build_condition_example(example, examples_by_subject, plan, condition, config)
            for example in examples
        ]
        condition_hashes[condition] = sha256_jsonl_rows(transformed)
        for original, changed in zip(examples, transformed):
            if str(original["subject_id"]) != str(changed["subject_id"]):
                raise AssertionError("Condition transform changed recipient identity.")
            if int(original["label"]) != int(changed["label"]):
                raise AssertionError("Condition transform changed recipient diagnosis.")
            if len(changed["audio_paths"]) != len(changed["audio_clip_seconds"]):
                raise AssertionError("Condition transform broke audio bundle alignment.")
            if CONDITION_SPECS[condition].transcript_source == "recipient":
                if changed["prompt_text"] != original["prompt_text"]:
                    raise AssertionError(
                        f"Condition {condition} must preserve recipient prompt bytes."
                    )
            if CONDITION_SPECS[condition].transcript_source == "none":
                if changed["transcript"] or str(original["transcript"]) in changed["prompt_text"]:
                    raise AssertionError(f"Condition {condition} retained recipient transcript text.")
    after_hash = sha256_jsonl_rows(examples)
    if after_hash != before_hash:
        raise AssertionError("E0 transforms mutated the source examples.")
    return {
        "source_examples_sha256": before_hash,
        "condition_example_hashes": condition_hashes,
        "num_subjects": len(examples),
        "across_subject_audio_mapping_sha256": sha256_text(
            json.dumps(plan["across_subject_audio"], sort_keys=True)
        ),
        "same_class_audio_mapping_sha256": sha256_text(
            json.dumps(plan["same_class_audio"], sort_keys=True)
        ),
        "transcript_mapping_sha256": sha256_text(
            json.dumps(plan["transcript"], sort_keys=True)
        ),
    }


def _resolve_model_path(
    cli_path: Path | None,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> Path:
    candidates: list[Path] = []
    if cli_path is not None:
        candidates.append(cli_path)
    environment_path = os.environ.get("MODEL_PATH")
    if environment_path:
        candidates.append(Path(environment_path))
    configured = config.get("model_name_or_path")
    if configured:
        candidates.append(Path(str(configured)))
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if adapter_config_path.is_file():
        adapter_config = read_json(adapter_config_path)
        if adapter_config.get("base_model_name_or_path"):
            candidates.append(Path(str(adapter_config["base_model_name_or_path"])))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        "No local base-model directory found. Pass --model-name-or-path explicitly; "
        f"checked {[str(candidate) for candidate in candidates]}."
    )


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"E0 output directory must be new or empty; refusing to overwrite {output_root}."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        default=None,
        help="Resolved checkpoint eval_config.yaml; defaults to CHECKPOINT/standalone_eval/eval_config.yaml.",
    )
    parser.add_argument(
        "--manifest-metadata",
        type=Path,
        default=Path("outputs/splits/daic_manifest_metadata.json"),
    )
    parser.add_argument("--model-name-or-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument(
        "--view-family",
        choices=(LEGACY_VIEW_FAMILY, NUMERIC_BALANCED_VIEW_FAMILY),
        default=LEGACY_VIEW_FAMILY,
        help=(
            "Keep the exact checkpoint-baked legacy anchor (default), or explicitly "
            "select the independent eight-view numeric-balanced family."
        ),
    )
    parser.add_argument(
        "--view-index",
        type=int,
        default=0,
        help="Legacy accepts only 0; numeric_balanced_k4 accepts 0 through 7.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_SPECS),
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Score a deterministic recipient prefix while deriving donors from the full partition.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--include-candidate-likelihood", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs, mappings, transforms, and prompts without loading a model or writing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not (checkpoint_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint adapter weights not found under {checkpoint_dir}.")
    config_snapshot = (
        args.config_snapshot.expanduser().resolve()
        if args.config_snapshot is not None
        else checkpoint_dir / "standalone_eval" / "eval_config.yaml"
    )
    manifest_metadata_path = resolve_project_path(args.manifest_metadata)
    config, config_envelope = _load_checkpoint_config(config_snapshot)
    examples, view_spec, manifest_metadata, manifest_path, partition_path = (
        _resolve_input_examples(
            config,
            manifest_metadata_path,
            partition=str(args.partition),
            expected_k=int(args.expected_k),
            view_family=str(args.view_family),
            view_index=int(args.view_index),
        )
    )
    plan = build_perturbation_plan(examples, seed=int(args.seed))
    transform_validation = validate_transform_plan(
        examples,
        plan,
        config,
        args.conditions,
    )
    recipients = list(examples)
    if args.max_subjects is not None:
        if int(args.max_subjects) < 1:
            raise ValueError("--max-subjects must be positive when provided.")
        recipients = recipients[: int(args.max_subjects)]
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_dir": str(checkpoint_dir),
        "config_snapshot": str(config_snapshot),
        "manifest_metadata": str(manifest_metadata_path),
        "partition": str(args.partition),
        "seed": int(args.seed),
        "conditions": list(args.conditions),
        "available_subjects": len(examples),
        "selected_subjects": len(recipients),
        "selected_subject_ids": [str(example["subject_id"]) for example in recipients],
        "view": view_spec,
        "transform_validation": transform_validation,
        "manifest_hash": manifest_metadata.get("manifest_hash"),
        "scorer_protocol": SCORER_PROTOCOL,
    }
    if args.validate_only:
        print(json.dumps(validation_report, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-only is used.")

    output_root = args.output_dir.expanduser().resolve()
    _prepare_output_root(output_root)
    model_path = _resolve_model_path(args.model_name_or_path, config, checkpoint_dir)
    set_seed(int(args.seed), deterministic=True)
    input_provenance = _input_provenance(
        project_root=project_root,
        config_snapshot=config_snapshot,
        manifest_metadata_path=manifest_metadata_path,
        manifest_path=manifest_path,
        partition_path=partition_path,
        checkpoint_dir=checkpoint_dir,
        model_path=model_path,
    )
    run_config = {
        **validation_report,
        "model_name_or_path": str(model_path),
        "output_dir": str(output_root),
        "include_candidate_likelihood": bool(args.include_candidate_likelihood),
        "config_snapshot_envelope": config_envelope,
        "resolved_checkpoint_config": config,
        "perturbation_plan": plan,
        "write_policy": "immutable_fail_if_exists",
    }
    _write_json_new(output_root / "run_config.json", run_config)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Lazy import keeps transform-only validation independent of PEFT/model dependencies.
    from src.model.runtime import (
        load_model_for_inference,
        load_processor,
        prepare_model_for_evaluation,
    )

    print(f"Loading E0 checkpoint once from {checkpoint_dir}", flush=True)
    processor = load_processor(checkpoint_dir, config)
    model = load_model_for_inference(
        str(model_path),
        adapter_path=checkpoint_dir,
        config=config,
    )
    model.to(device)
    model.eval()
    prepare_model_for_evaluation(model, config)
    scorer = DirectFirstTokenScorer(
        model,
        processor,
        config,
        device,
        include_candidate_likelihood=bool(args.include_candidate_likelihood),
    )
    examples_by_subject = _examples_by_subject(examples)
    condition_results = []
    for condition in args.conditions:
        condition_results.append(
            run_condition(
                condition=condition,
                recipient_examples=recipients,
                examples_by_subject=examples_by_subject,
                plan=plan,
                scorer=scorer,
                model=model,
                config=config,
                view_spec=view_spec,
                seed=int(args.seed),
                output_root=output_root,
                input_provenance=input_provenance,
                progress_every=int(args.progress_every),
            )
        )
    run_provenance = {
        **input_provenance,
        "run_config": _artifact_entry(output_root / "run_config.json"),
        "conditions": condition_results,
        "checkpoint_load_count": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(output_root / "run_provenance.json", run_provenance)
    print(json.dumps(condition_results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
