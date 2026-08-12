from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


D3TEC_DATASET = "d3tec"
D3TEC_RESPONSE_COUNT = 27
D3TEC_AGGREGATION_POLICY = (
    "segment_majority_probability_margin_tie_break_then_"
    "equal_response_majority_probability_margin_tie_break_then_subject"
)
D3TEC_WEIGHT_POLICY = "inverse_segments_per_response_rescaled_to_mean_one"
TEXT_WEIGHT_POLICY = "one_vector_per_subject_unweighted"
LEGACY_WEIGHT_POLICY = "uniform_rows"
DAIC_SUBJECT_WEIGHT_POLICY = "inverse_chunks_per_subject_rescaled_to_mean_one"
PACKED30_PROTOCOL_ID = "daic_participant_speech_packed30_v1"
PACKED30_AGGREGATION_POLICY = "mean_depressed_probability_threshold_0_5"


def is_d3tec_audio_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("dataset", "")).lower() == D3TEC_DATASET
        and str(metadata.get("input_modality", "")) in {"audio_only", "audio_text"}
        and bool(rows)
    )


def is_packed30_rows(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("protocol_id", "")).strip() == PACKED30_PROTOCOL_ID


def classifier_aggregation_policy(metadata: dict[str, Any]) -> str:
    if is_packed30_rows(metadata):
        return PACKED30_AGGREGATION_POLICY
    if str(metadata.get("dataset", "")).lower() == D3TEC_DATASET:
        if str(metadata.get("input_modality", "")) == "text_only":
            return "one_prediction_per_subject"
        return D3TEC_AGGREGATION_POLICY
    return "sample_majority_probability_margin_tie_break_then_subject"


def response_normalized_sample_weights(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return fit weights and a complete, independently checkable audit.

    D3TEC audio rows receive inverse response-size weights and are rescaled so
    the mean row weight is one. Since every D3TEC subject has exactly 27
    responses, equal response totals also imply equal subject totals.
    """
    if not rows:
        return np.empty(0, dtype=np.float64), {
            "schema_version": "hidden_classifier_weight_audit.v1",
            "policy": LEGACY_WEIGHT_POLICY,
            "row_count": 0,
            "mean_weight": None,
            "response_count": None,
            "subject_count": 0,
            "equal_response_totals": None,
            "equal_subject_totals": None,
        }

    if not is_d3tec_audio_rows(rows, metadata):
        if str(metadata.get("dataset", "")).lower() == "daic":
            counts = Counter(str(row["subject_id"]) for row in rows)
            raw = np.asarray(
                [1.0 / counts[str(row["subject_id"])] for row in rows],
                dtype=np.float64,
            )
            weights = raw / raw.mean()
            totals: dict[str, float] = defaultdict(float)
            for row, weight in zip(rows, weights.tolist()):
                totals[str(row["subject_id"])] += weight
            equal_subject_totals = len(
                {round(value, 12) for value in totals.values()}
            ) == 1
            if not equal_subject_totals:
                raise AssertionError(
                    "DAIC subjects do not have equal total classifier fit weight."
                )
            return weights, {
                "schema_version": "hidden_classifier_weight_audit.v1",
                "policy": DAIC_SUBJECT_WEIGHT_POLICY,
                "row_count": len(rows),
                "mean_weight": float(weights.mean()),
                "response_count": None,
                "subject_count": len(counts),
                "equal_response_totals": None,
                "equal_subject_totals": equal_subject_totals,
                "subject_weight_totals": dict(sorted(totals.items())),
                "chunks_per_subject": dict(sorted(counts.items())),
            }
        weights = np.ones(len(rows), dtype=np.float64)
        policy = (
            TEXT_WEIGHT_POLICY
            if (
                str(metadata.get("dataset", "")).lower() == D3TEC_DATASET
                and str(metadata.get("input_modality", "")) == "text_only"
            )
            else LEGACY_WEIGHT_POLICY
        )
        return weights, {
            "schema_version": "hidden_classifier_weight_audit.v1",
            "policy": policy,
            "row_count": len(rows),
            "mean_weight": 1.0,
            "response_count": None,
            "subject_count": len({str(row["subject_id"]) for row in rows}),
            "equal_response_totals": None,
            "equal_subject_totals": None,
        }

    grouped: dict[str, list[int]] = defaultdict(list)
    response_subject: dict[str, str] = {}
    declared_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        response_id = str(row.get("response_id", "")).strip()
        if not response_id:
            raise ValueError("D3TEC audio classifier rows require response_id.")
        subject_id = str(row["subject_id"])
        if response_id in response_subject and response_subject[response_id] != subject_id:
            raise ValueError(f"Response {response_id} spans multiple subjects.")
        response_subject[response_id] = subject_id
        grouped[response_id].append(index)
        declared = int(row.get("num_segments", 0))
        if declared < 1:
            raise ValueError(f"Response {response_id} has invalid num_segments={declared}.")
        if response_id in declared_counts and declared_counts[response_id] != declared:
            raise ValueError(f"Response {response_id} has inconsistent num_segments.")
        declared_counts[response_id] = declared

    for response_id, indices in grouped.items():
        if len(indices) != declared_counts[response_id]:
            raise ValueError(
                f"Response {response_id} is incomplete in fit partition: "
                f"declared={declared_counts[response_id]} observed={len(indices)}."
            )

    raw = np.asarray(
        [1.0 / len(grouped[str(row["response_id"])]) for row in rows],
        dtype=np.float64,
    )
    scale = len(rows) / len(grouped)
    weights = raw * scale
    response_totals = {
        response_id: float(weights[indices].sum())
        for response_id, indices in sorted(grouped.items())
    }
    subject_totals: dict[str, float] = defaultdict(float)
    responses_by_subject: Counter[str] = Counter()
    for response_id, total in response_totals.items():
        subject_id = response_subject[response_id]
        subject_totals[subject_id] += total
        responses_by_subject[subject_id] += 1
    response_values = np.asarray(list(response_totals.values()), dtype=np.float64)
    subject_values = np.asarray(list(subject_totals.values()), dtype=np.float64)
    equal_response_totals = bool(np.allclose(response_values, response_values[0]))
    equal_subject_totals = bool(np.allclose(subject_values, subject_values[0]))
    if not np.isclose(weights.mean(), 1.0):
        raise AssertionError("D3TEC response-normalized sample weights do not have mean one.")
    if not equal_response_totals:
        raise AssertionError("D3TEC responses do not have equal total classifier fit weight.")
    if set(responses_by_subject.values()) != {D3TEC_RESPONSE_COUNT}:
        raise ValueError(
            "D3TEC audio classifier fitting requires exactly 27 complete responses "
            f"per subject; found={dict(sorted(responses_by_subject.items()))}."
        )
    if not equal_subject_totals:
        raise AssertionError("D3TEC subjects do not have equal total classifier fit weight.")
    return weights, {
        "schema_version": "hidden_classifier_weight_audit.v1",
        "policy": D3TEC_WEIGHT_POLICY,
        "row_count": len(rows),
        "mean_weight": float(weights.mean()),
        "rescale_factor": float(scale),
        "response_count": len(grouped),
        "subject_count": len(subject_totals),
        "responses_per_subject": dict(sorted(responses_by_subject.items())),
        "response_total_weight": dict(sorted(response_totals.items())),
        "subject_total_weight": dict(sorted(subject_totals.items())),
        "equal_response_totals": equal_response_totals,
        "equal_subject_totals": equal_subject_totals,
    }


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def cache_identity(cache_dir: Path) -> dict[str, Any]:
    names = (
        "outer_train.npz",
        "outer_train_rows.jsonl",
        "final_eval.npz",
        "final_eval_rows.jsonl",
        "extraction_metadata.json",
    )
    return {name: file_identity(cache_dir / name) for name in names}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
