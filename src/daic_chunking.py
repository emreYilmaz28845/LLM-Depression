from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Sequence


SUBJECT_AUDIO = "subject_audio"
SUBJECT_CHUNKS = "subject_chunks"
TRAIN_POLICIES = {"random_k", "rotary_k", "all"}
EVAL_POLICIES = {"fixed_k", "balanced_joint_cover", "all"}


def _integer_or_all(value: Any, *, name: str) -> int | str:
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer or 'all'.")
    return resolved


def resolve_chunking_controls(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the new controls while preserving legacy subject_audio K behavior."""
    data = config.get("data", {})
    mode = str(data.get("sample_mode", "response")).strip().lower()
    if mode not in {SUBJECT_AUDIO, SUBJECT_CHUNKS}:
        return {"enabled": False, "sample_mode": mode}
    legacy_k = data.get("chunks_per_subject", 4)
    train_policy = str(data.get("train_chunk_policy", "random_k")).strip().lower()
    eval_policy = str(data.get("eval_chunk_policy", "fixed_k")).strip().lower()
    if train_policy not in TRAIN_POLICIES:
        raise ValueError(f"Unsupported data.train_chunk_policy={train_policy!r}.")
    if eval_policy not in EVAL_POLICIES:
        raise ValueError(f"Unsupported data.eval_chunk_policy={eval_policy!r}.")
    train_k = _integer_or_all(
        data.get("train_chunks_per_subject", legacy_k),
        name="data.train_chunks_per_subject",
    )
    eval_k = _integer_or_all(
        data.get("eval_chunks_per_subject", legacy_k),
        name="data.eval_chunks_per_subject",
    )
    if train_policy == "all":
        train_k = "all"
    if eval_policy == "all":
        eval_k = "all"
    if mode == SUBJECT_AUDIO and train_policy not in {"random_k", "all"}:
        raise ValueError("subject_audio supports train_chunk_policy=random_k or all.")
    if mode == SUBJECT_CHUNKS and train_policy == "random_k":
        raise ValueError("subject_chunks supports train_chunk_policy=rotary_k or all.")
    if mode == SUBJECT_CHUNKS and eval_policy != "all":
        raise ValueError("subject_chunks requires eval_chunk_policy=all.")
    return {
        "enabled": True,
        "sample_mode": mode,
        "train_chunk_policy": train_policy,
        "train_chunks_per_subject": train_k,
        "eval_chunk_policy": eval_policy,
        "eval_chunks_per_subject": eval_k,
        "max_audio_seconds_per_chunk": float(
            data.get("max_audio_seconds_per_chunk", 30.0)
        ),
    }


def subject_permutation(size: int, *, subject_id: str, seed: int) -> list[int]:
    if size < 1:
        return []
    digest = hashlib.sha256(f"{int(seed)}:{subject_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = list(range(size))
    rng.shuffle(indices)
    return indices


def rotary_indices(
    size: int, k: int, *, subject_id: str, seed: int, epoch: int
) -> list[int]:
    if size < 1:
        return []
    k = min(int(k), size)
    permutation = subject_permutation(size, subject_id=subject_id, seed=seed)
    start = (int(epoch) * k) % size
    return [permutation[(start + offset) % size] for offset in range(k)]


def balanced_joint_bundles(
    chunk_ids: Sequence[str], k: int
) -> tuple[list[list[int]], dict[str, Any]]:
    """Minimum cyclic K-bundles with exactly equal per-chunk coverage."""
    n = len(chunk_ids)
    if n < 1:
        raise ValueError("Balanced bundles require at least one chunk.")
    k = min(int(k), n)
    divisor = math.gcd(n, k)
    bundle_count = n // divisor
    bundles = [
        [((bundle_id * k) + offset) % n for offset in range(k)]
        for bundle_id in range(bundle_count)
    ]
    counts = Counter(index for bundle in bundles for index in bundle)
    expected = k // divisor
    if set(counts) != set(range(n)) or set(counts.values()) != {expected}:
        raise AssertionError("Internal balanced bundle construction error.")
    audit_rows = []
    for bundle_id, members in enumerate(bundles):
        for position, index in enumerate(members):
            audit_rows.append(
                {
                    "bundle_id": bundle_id,
                    "member_position": position,
                    "chunk_index": index,
                    "chunk_id": str(chunk_ids[index]),
                    "coverage_count": counts[index],
                }
            )
    return bundles, {
        "num_chunks": n,
        "chunks_per_bundle": k,
        "num_bundles": bundle_count,
        "occurrences_per_chunk": expected,
        "coverage_by_chunk_id": {
            str(chunk_ids[index]): counts[index] for index in range(n)
        },
        "memberships": audit_rows,
    }


def build_independent_epoch_schedule(
    examples: Sequence[dict[str, Any]],
    *,
    policy: str,
    chunks_per_subject: int | str,
    seed: int,
    epochs: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example["subject_id"])].append(dict(example))
    schedules: list[list[dict[str, Any]]] = []
    exposure: dict[str, Counter[str]] = {
        subject_id: Counter() for subject_id in grouped
    }
    weight_totals: dict[str, list[float]] = defaultdict(list)
    for epoch in range(epochs):
        epoch_rows: list[dict[str, Any]] = []
        for subject_id in sorted(grouped):
            rows = sorted(grouped[subject_id], key=lambda row: str(row["sample_id"]))
            n = len(rows)
            if policy == "all":
                selected = list(range(n))
            elif policy == "rotary_k":
                if chunks_per_subject == "all":
                    raise ValueError("rotary_k requires an integer chunks_per_subject.")
                selected = rotary_indices(
                    n, int(chunks_per_subject), subject_id=subject_id, seed=seed, epoch=epoch
                )
            else:
                raise ValueError(f"Unsupported independent policy {policy!r}.")
            weight = 1.0 / len(selected)
            subject_total = 0.0
            for index in selected:
                row = dict(rows[index])
                row["loss_weight"] = weight
                row["chunk_schedule_epoch"] = epoch
                row["chunk_schedule_position"] = len(epoch_rows)
                epoch_rows.append(row)
                exposure[subject_id][str(row["sample_id"])] += 1
                subject_total += weight
            weight_totals[subject_id].append(subject_total)
        schedules.append(epoch_rows)
    return schedules, {
        "schema_version": "daic_independent_schedule.v1",
        "policy": policy,
        "seed": int(seed),
        "epochs": int(epochs),
        "chunks_per_subject": chunks_per_subject,
        "epoch_example_counts": [len(rows) for rows in schedules],
        "exposure_counts_by_subject": {
            subject_id: dict(sorted(counts.items()))
            for subject_id, counts in sorted(exposure.items())
        },
        "subject_epoch_weight_totals": dict(sorted(weight_totals.items())),
        "equal_total_subject_weight": all(
            all(abs(value - 1.0) < 1e-9 for value in values)
            for values in weight_totals.values()
        ),
    }


def gradient_accumulation_for_reference_updates(
    *,
    independent_examples_per_epoch: int,
    reference_subjects: int,
    reference_gradient_accumulation: int,
    world_size: int = 1,
    per_device_batch_size: int = 1,
) -> int:
    """Choose accumulation so independent and joint runs have equal updates/epoch."""
    reference_updates = math.ceil(
        reference_subjects
        / max(1, reference_gradient_accumulation * world_size * per_device_batch_size)
    )
    micro_batches = math.ceil(
        independent_examples_per_epoch / max(1, world_size * per_device_batch_size)
    )
    return max(1, math.ceil(micro_batches / max(1, reference_updates)))
