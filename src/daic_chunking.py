from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Sequence


SUBJECT_AUDIO = "subject_audio"
SUBJECT_CHUNKS = "subject_chunks"
SUBJECT_MIL = "subject_mil"
TRAIN_POLICIES = {
    "random_k",
    "fixed_k",
    "rotary_k",
    "all",
    "joint_random_k",
    "joint_rotary_k",
    "joint_balanced_cover",
}
EVAL_POLICIES = {"fixed_k", "balanced_joint_cover", "fixed_count_balanced_joint_cover", "all", "matched_k"}


def _chunk_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    """Order DAIC chunks numerically while retaining a deterministic fallback."""
    raw_chunk_id = str(row.get("chunk_id", "")).strip()
    match = re.search(r"(\d+)$", raw_chunk_id)
    return (
        0 if match else 1,
        int(match.group(1)) if match else 10**9,
        str(row.get("sample_id", "")),
    )


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
    if mode not in {SUBJECT_AUDIO, SUBJECT_CHUNKS, SUBJECT_MIL}:
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
    if mode == SUBJECT_AUDIO and train_policy not in {
        "random_k", "fixed_k", "all", "joint_random_k", "joint_rotary_k", "joint_balanced_cover"
    }:
        raise ValueError("Unsupported subject_audio train chunk policy.")
    if mode == SUBJECT_AUDIO and eval_policy not in {
        "fixed_k", "balanced_joint_cover", "fixed_count_balanced_joint_cover", "all"
    }:
        raise ValueError(
            "subject_audio requires eval_chunk_policy=fixed_k, balanced_joint_cover, "
            "fixed_count_balanced_joint_cover, or all."
        )
    if mode == SUBJECT_CHUNKS and train_policy not in {"fixed_k", "rotary_k", "all"}:
        raise ValueError("subject_chunks supports train_chunk_policy=fixed_k, rotary_k, or all.")
    if mode == SUBJECT_CHUNKS and eval_policy not in {"all", "matched_k"}:
        raise ValueError("subject_chunks requires eval_chunk_policy=all or matched_k.")
    if mode == SUBJECT_MIL and (train_policy != "all" or eval_policy not in {"all", "matched_k"}):
        raise ValueError("subject_mil requires all-chunk training and all/matched_k evaluation.")
    if train_policy in {"fixed_k", "joint_random_k", "joint_rotary_k", "joint_balanced_cover"} and train_k == "all":
        raise ValueError(f"{train_policy} requires an integer data.train_chunks_per_subject.")
    if eval_policy in {
        "fixed_k", "balanced_joint_cover", "fixed_count_balanced_joint_cover", "matched_k"
    } and eval_k == "all":
        raise ValueError(f"{eval_policy} requires an integer data.eval_chunks_per_subject.")
    loss_weight_rescale = str(data.get("loss_weight_rescale", "none")).strip().lower()
    if loss_weight_rescale not in {"none", "mean_one"}:
        raise ValueError("data.loss_weight_rescale must be 'none' or 'mean_one'.")
    eval_bundles = int(data.get("eval_bundles_per_subject", 15))
    if eval_bundles < 1:
        raise ValueError("data.eval_bundles_per_subject must be positive.")
    return {
        "enabled": True,
        "sample_mode": mode,
        "train_chunk_policy": train_policy,
        "train_chunks_per_subject": train_k,
        "eval_chunk_policy": eval_policy,
        "eval_chunks_per_subject": eval_k,
        "eval_bundles_per_subject": eval_bundles,
        "loss_weight_rescale": loss_weight_rescale,
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


def random_k_indices(size: int, k: int, *, subject_id: str, seed: int, epoch: int) -> list[int]:
    if size < 1:
        return []
    k = min(int(k), size)
    digest = hashlib.sha256(f"{seed}:{subject_id}:random_k:{epoch}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).sample(range(size), k)


def balanced_joint_bundles(
    chunk_ids: Sequence[str], k: int
) -> tuple[list[list[int]], dict[str, Any]]:
    """Minimum cyclic K-bundles with exactly equal per-chunk coverage."""
    n = len(chunk_ids)
    if n < 1:
        raise ValueError("Balanced bundles require at least one chunk.")
    if len({str(chunk_id) for chunk_id in chunk_ids}) != n:
        raise ValueError("Balanced bundles require unique chunk IDs.")
    if int(k) < 1:
        raise ValueError("Balanced bundles require a positive K.")
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


def fixed_count_balanced_joint_bundles(
    chunk_ids: Sequence[str], k: int, bundle_count: int
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build exactly B cyclic bundles, failing unless coverage is equal."""
    n = len(chunk_ids)
    k = int(k)
    bundle_count = int(bundle_count)
    if n < 1 or k < 1 or k > n or bundle_count < 1:
        raise ValueError("Fixed-count bundles require 1 <= K <= N and B >= 1.")
    if len({str(chunk_id) for chunk_id in chunk_ids}) != n:
        raise ValueError("Fixed-count bundles require unique chunk IDs.")
    if (bundle_count * k) % n:
        raise ValueError(
            f"Cannot equally cover {n} chunks with B={bundle_count}, K={k}; B*K must be divisible by N."
        )
    bundles = [
        [((bundle_id * k) + offset) % n for offset in range(k)]
        for bundle_id in range(bundle_count)
    ]
    counts = Counter(index for bundle in bundles for index in bundle)
    expected = bundle_count * k // n
    if set(counts) != set(range(n)) or set(counts.values()) != {expected}:
        raise AssertionError("Internal fixed-count bundle construction error.")
    return bundles, {
        "num_chunks": n,
        "chunks_per_bundle": k,
        "num_bundles": bundle_count,
        "occurrences_per_chunk": expected,
        "coverage_by_chunk_id": {str(chunk_ids[index]): counts[index] for index in range(n)},
    }


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Return a deterministic endpoint-preserving matched-count view."""
    if total < 1 or count < 1:
        return []
    if count > total:
        raise ValueError(f"Cannot choose {count} unique chunks from {total}.")
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]
    chosen = [int(round(i * (total - 1) / (count - 1))) for i in range(count)]
    if len(set(chosen)) != count:
        raise AssertionError("Evenly-spaced construction produced duplicate indices.")
    return chosen


def deterministic_subject_order(subject_ids: Sequence[str], *, seed: int, epoch: int) -> list[str]:
    """Shuffle complete subject blocks independently of label-ordered IDs."""
    ordered = sorted(map(str, subject_ids))
    digest = hashlib.sha256(f"{int(seed)}:subject_order:{int(epoch)}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def _rescale_weights(rows: list[dict[str, Any]], mode: str) -> float:
    mode = str(mode).strip().lower()
    if mode not in {"none", "mean_one"}:
        raise ValueError("loss_weight_rescale must be none or mean_one.")
    if not rows:
        raise ValueError("Cannot rescale an empty chunk schedule.")
    mean_raw = sum(float(row["raw_loss_weight"]) for row in rows) / len(rows)
    if not math.isfinite(mean_raw) or mean_raw <= 0:
        raise ValueError(f"Chunk schedule has invalid mean raw loss weight: {mean_raw!r}.")
    scale = 1.0 if mode == "none" else 1.0 / mean_raw
    for row in rows:
        row["loss_weight"] = float(row["raw_loss_weight"]) * scale
        row["effective_loss_weight"] = row["loss_weight"]
    return scale


def build_independent_epoch_schedule(
    examples: Sequence[dict[str, Any]],
    *,
    policy: str,
    chunks_per_subject: int | str,
    seed: int,
    epochs: int,
    loss_weight_rescale: str = "none",
    equal_row_weight: bool = False,
    class_balance: bool = False,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    policy = str(policy).strip().lower()
    loss_weight_rescale = str(loss_weight_rescale).strip().lower()
    if policy not in {"fixed_k", "rotary_k", "all"}:
        raise ValueError(f"Unsupported independent policy {policy!r}.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example["subject_id"])].append(dict(example))
    if not grouped:
        raise ValueError("Independent schedules require at least one subject.")
    for subject_id, subject_rows in grouped.items():
        if len({int(row["label"]) for row in subject_rows}) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        sample_ids = [str(row.get("sample_id", "")) for row in subject_rows]
        if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Subject {subject_id} has duplicate or missing sample IDs.")
        chunk_ids = [str(row.get("chunk_id", row.get("sample_id", ""))) for row in subject_rows]
        if not all(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"Subject {subject_id} has duplicate or missing chunk IDs.")
    schedules: list[list[dict[str, Any]]] = []
    exposure: dict[str, Counter[str]] = {
        subject_id: Counter() for subject_id in grouped
    }
    exposure_by_chunk: dict[str, Counter[str]] = {
        subject_id: Counter() for subject_id in grouped
    }
    weight_totals: dict[str, list[float]] = defaultdict(list)
    effective_weight_totals: dict[str, list[float]] = defaultdict(list)
    scales: list[float] = []
    audit_rows: list[dict[str, Any]] = []
    class_counts = Counter(int(rows[0]["label"]) for rows in grouped.values())
    epoch_audio_exposure: list[float] = []
    epoch_raw_weight_totals: list[float] = []
    epoch_effective_weight_totals: list[float] = []
    for epoch in range(epochs):
        epoch_rows: list[dict[str, Any]] = []
        epoch_subject_raw: dict[str, float] = {}
        for subject_id in deterministic_subject_order(grouped, seed=seed, epoch=epoch):
            rows = sorted(grouped[subject_id], key=_chunk_sort_key)
            n = len(rows)
            if policy == "all":
                selected = list(range(n))
            elif policy == "fixed_k":
                if chunks_per_subject == "all":
                    raise ValueError("fixed_k requires an integer chunks_per_subject.")
                if n < int(chunks_per_subject):
                    raise ValueError(
                        f"fixed_k requires at least {int(chunks_per_subject)} chunks for "
                        f"subject_id={subject_id}; found {n}."
                    )
                selected = evenly_spaced_indices(n, int(chunks_per_subject))
            elif policy == "rotary_k":
                if chunks_per_subject == "all":
                    raise ValueError("rotary_k requires an integer chunks_per_subject.")
                selected = rotary_indices(
                    n, int(chunks_per_subject), subject_id=subject_id, seed=seed, epoch=epoch
                )
            class_factor = 1.0 / class_counts[int(rows[0]["label"])] if class_balance else 1.0
            weight = class_factor if equal_row_weight else class_factor / len(selected)
            subject_total = 0.0
            for index in selected:
                row = dict(rows[index])
                row["raw_loss_weight"] = weight
                row["chunk_schedule_epoch"] = epoch
                row["chunk_schedule_position"] = len(epoch_rows)
                epoch_rows.append(row)
                exposure[subject_id][str(row["sample_id"])] += 1
                exposure_by_chunk[subject_id][str(row.get("chunk_id", row["sample_id"]))] += 1
                subject_total += weight
            weight_totals[subject_id].append(subject_total)
            epoch_subject_raw[subject_id] = subject_total
        scale = _rescale_weights(epoch_rows, loss_weight_rescale)
        scales.append(scale)
        for subject_id in grouped:
            effective_weight_totals[subject_id].append(epoch_subject_raw[subject_id] * scale)
        for row in epoch_rows:
            clip_seconds = row.get("audio_clip_seconds") or []
            audit_rows.append({
                "epoch": epoch, "position": int(row["chunk_schedule_position"]),
                "subject_id": str(row["subject_id"]), "sample_id": str(row["sample_id"]),
                "label": int(row["label"]),
                "chunk_id": str(row.get("chunk_id", row["sample_id"])),
                "raw_loss_weight": float(row["raw_loss_weight"]),
                "effective_loss_weight": float(row["loss_weight"]),
                "weight_scale": scale,
                "audio_seconds": sum(float(value) for value in clip_seconds if value is not None),
            })
        epoch_audio_exposure.append(
            sum(float(row.get("audio_seconds", 0.0)) for row in audit_rows if int(row["epoch"]) == epoch)
        )
        epoch_raw_weight_totals.append(sum(float(row["raw_loss_weight"]) for row in epoch_rows))
        epoch_effective_weight_totals.append(sum(float(row["loss_weight"]) for row in epoch_rows))
        schedules.append(epoch_rows)
    return schedules, {
        "schema_version": "daic_independent_schedule.v2",
        "policy": policy,
        "seed": int(seed),
        "epochs": int(epochs),
        "chunks_per_subject": chunks_per_subject,
        "loss_weight_rescale": loss_weight_rescale,
        "equal_row_weight": bool(equal_row_weight),
        "class_balance": bool(class_balance),
        "epoch_example_counts": [len(rows) for rows in schedules],
        "epoch_weight_scales": scales,
        "epoch_mean_effective_weights": [
            sum(float(row["loss_weight"]) for row in rows) / len(rows) for rows in schedules
        ],
        "epoch_raw_weight_totals": epoch_raw_weight_totals,
        "epoch_effective_weight_totals": epoch_effective_weight_totals,
        "epoch_audio_exposure_seconds": epoch_audio_exposure,
        "total_audio_exposure_seconds": sum(epoch_audio_exposure),
        "micro_batches_per_epoch": [len(rows) for rows in schedules],
        "rows": audit_rows,
        "exposure_counts_by_subject": {
            subject_id: dict(sorted(counts.items()))
            for subject_id, counts in sorted(exposure.items())
        },
        "exposure_counts_by_chunk_id": {
            subject_id: dict(sorted(counts.items()))
            for subject_id, counts in sorted(exposure_by_chunk.items())
        },
        "subject_epoch_weight_totals": dict(sorted(weight_totals.items())),
        "subject_epoch_effective_weight_totals": dict(sorted(effective_weight_totals.items())),
        "equal_total_subject_weight": (not equal_row_weight) and all(
            max(values) - min(values) < 1e-9
            for values in weight_totals.values()
        ),
        "subject_weighting": (
            "equal_row"
            if equal_row_weight
            else "class_inverse_frequency" if class_balance else "subject_normalized"
        ),
    }


def build_joint_epoch_schedule(
    subjects: Sequence[dict[str, Any]], *, policy: str, k: int, seed: int,
    epochs: int, loss_weight_rescale: str = "mean_one", class_balance: bool = False,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Materialize rotary or minimum-cover joint bundles for every epoch."""
    loss_weight_rescale = str(loss_weight_rescale).strip().lower()
    if int(epochs) < 1:
        raise ValueError("epochs must be positive.")
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive.")
    by_id = {str(row["subject_id"]): dict(row) for row in subjects}
    if not by_id:
        raise ValueError("Joint schedules require at least one subject.")
    if len(by_id) != len(subjects):
        raise ValueError("Joint schedules require one source example per subject.")
    for subject_id, row in by_id.items():
        paths = list(row.get("subject_chunk_paths") or [])
        if not paths:
            raise ValueError(f"Subject {subject_id} has no source chunks.")
        chunk_ids = list(map(str, row.get("subject_chunk_ids", range(len(paths)))))
        if len(chunk_ids) != len(paths):
            raise ValueError(f"Subject {subject_id} chunk IDs and paths have different lengths.")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"Subject {subject_id} has duplicate chunk IDs.")
    class_counts = Counter(int(row["label"]) for row in by_id.values())
    schedules: list[list[dict[str, Any]]] = []
    audit_rows: list[dict[str, Any]] = []
    exposure_by_subject: dict[str, Counter[str]] = {
        subject_id: Counter() for subject_id in by_id
    }
    epoch_audio_exposure: list[float] = []
    epoch_raw_weight_totals: list[float] = []
    epoch_effective_weight_totals: list[float] = []
    for epoch in range(int(epochs)):
        rows: list[dict[str, Any]] = []
        for subject_id in deterministic_subject_order(by_id, seed=seed, epoch=epoch):
            source = by_id[subject_id]
            chunk_paths = list(source["subject_chunk_paths"])
            chunk_ids = list(map(str, source.get("subject_chunk_ids", range(len(chunk_paths)))))
            if policy == "joint_random_k":
                memberships = [random_k_indices(len(chunk_paths), k, subject_id=subject_id, seed=seed, epoch=epoch)]
            elif policy == "joint_rotary_k":
                memberships = [rotary_indices(len(chunk_paths), k, subject_id=subject_id, seed=seed, epoch=epoch)]
            elif policy == "joint_balanced_cover":
                memberships, _ = balanced_joint_bundles(chunk_ids, k)
            else:
                raise ValueError(f"Unsupported joint policy {policy!r}.")
            class_factor = 1.0 / class_counts[int(source["label"])] if class_balance else 1.0
            source_clip_seconds = list(
                source.get("subject_chunk_clip_seconds")
                or source.get("audio_clip_seconds")
                or []
            )
            if not source_clip_seconds:
                source_clip_seconds = [None] * len(chunk_paths)
            elif len(source_clip_seconds) == 1:
                source_clip_seconds = source_clip_seconds * len(chunk_paths)
            elif len(source_clip_seconds) != len(chunk_paths):
                # Subject-level examples historically carried one duration per
                # prompt slot (K), while the schedule also carries the full
                # source-chunk list.  A uniform cap is safe to broadcast; a
                # heterogeneous, mis-sized duration vector is ambiguous.
                if len(set(source_clip_seconds)) == 1:
                    source_clip_seconds = [source_clip_seconds[0]] * len(chunk_paths)
                else:
                    raise ValueError(
                        f"Subject {subject_id} audio duration metadata does not match source chunks."
                    )
            for bundle_id, indices in enumerate(memberships):
                row = dict(source)
                row["sample_id"] = f"{subject_id}__epoch_{epoch:03d}__bundle_{bundle_id:03d}"
                row["audio_paths"] = [chunk_paths[index] for index in indices]
                row["audio_clip_seconds"] = [source_clip_seconds[index] for index in indices]
                row["bundle_id"] = bundle_id
                row["bundle_chunk_ids"] = [chunk_ids[index] for index in indices]
                row["raw_loss_weight"] = class_factor / len(memberships)
                row["chunk_schedule_epoch"] = epoch
                rows.append(row)
                for index in indices:
                    exposure_by_subject[subject_id][chunk_ids[index]] += 1
        scale = _rescale_weights(rows, loss_weight_rescale)
        for position, row in enumerate(rows):
            row["chunk_schedule_position"] = position
            audit_rows.append({
                "epoch": epoch, "position": position, "subject_id": row["subject_id"],
                "sample_id": row["sample_id"], "label": int(row["label"]),
                "bundle_chunk_ids": row["bundle_chunk_ids"],
                "raw_loss_weight": row["raw_loss_weight"],
                "effective_loss_weight": row["loss_weight"], "weight_scale": scale,
                "audio_seconds": sum(
                    float(value)
                    for value in row["audio_clip_seconds"]
                    if value is not None
                ),
            })
        epoch_audio_exposure.append(
            sum(float(row.get("audio_seconds", 0.0)) for row in audit_rows if int(row["epoch"]) == epoch)
        )
        epoch_raw_weight_totals.append(sum(float(row["raw_loss_weight"]) for row in rows))
        epoch_effective_weight_totals.append(sum(float(row["loss_weight"]) for row in rows))
        schedules.append(rows)
    return schedules, {
        "schema_version": "daic_joint_schedule.v2", "policy": policy,
        "seed": int(seed), "epochs": int(epochs), "k": int(k),
        "loss_weight_rescale": loss_weight_rescale,
        "epoch_example_counts": [len(rows) for rows in schedules],
        "epoch_mean_effective_weights": [
            sum(float(row["loss_weight"]) for row in rows) / len(rows) for rows in schedules
        ],
        "epoch_raw_weight_totals": epoch_raw_weight_totals,
        "epoch_effective_weight_totals": epoch_effective_weight_totals,
        "epoch_audio_exposure_seconds": epoch_audio_exposure,
        "total_audio_exposure_seconds": sum(epoch_audio_exposure),
        "optimizer_update_units_per_epoch": [len(rows) for rows in schedules],
        "rows": audit_rows,
        "exposure_counts_by_subject": {
            subject_id: dict(sorted(counts.items()))
            for subject_id, counts in sorted(exposure_by_subject.items())
        },
        "subject_weighting": "class_inverse_frequency" if class_balance else "subject_normalized",
        "equal_total_subject_weight": all(
            math.isclose(
                sum(float(row["raw_loss_weight"]) for row in epoch_rows if str(row["subject_id"]) == subject_id),
                sum(float(row["raw_loss_weight"]) for row in epoch_rows if str(row["subject_id"]) == other_id),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for epoch_rows in schedules[:1]
            for subject_id in by_id
            for other_id in by_id
        ) if not class_balance else False,
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
    if independent_examples_per_epoch < 1 or reference_subjects < 1:
        raise ValueError("Schedule sizes must be positive.")
    if reference_gradient_accumulation < 1 or world_size < 1 or per_device_batch_size < 1:
        raise ValueError("Batch and world-size controls must be positive.")
    reference_updates = math.ceil(
        reference_subjects
        / max(1, reference_gradient_accumulation * world_size * per_device_batch_size)
    )
    micro_batches = math.ceil(
        independent_examples_per_epoch / max(1, world_size * per_device_batch_size)
    )
    return max(1, math.ceil(micro_batches / max(1, reference_updates)))


def matched_k_resamples(
    sample_rows: Sequence[dict[str, Any]], *, k: int = 10, iterations: int = 1000,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Resample cached per-chunk scores; this performs no model inference."""
    if int(k) < 1 or int(iterations) < 1:
        raise ValueError("matched-k resampling requires positive k and iterations.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["subject_id"])].append(dict(row))
    output: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=_chunk_sort_key)
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        if len(rows) < k:
            raise ValueError(f"Subject {subject_id} has only {len(rows)} chunks; matched k={k}.")
        digest = hashlib.sha256(f"{seed}:matched:{subject_id}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        for resample_id in range(iterations):
            chosen = rng.sample(range(len(rows)), k)
            margins = [
                float(row["score_margin"])
                if "score_margin" in row
                else float(row["dep_score"]) - float(row["non_score"])
                for row in (rows[index] for index in chosen)
            ]
            output.append({
                "subject_id": subject_id, "label": int(rows[0]["label"]),
                "resample_id": resample_id,
                "sample_ids": [str(rows[index]["sample_id"]) for index in chosen],
                "score_margin": sum(margins) / k,
                "prediction": int(sum(margins) / k > 0.0),
            })
    return output
