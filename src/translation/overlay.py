"""Config-selected English transcript overlay for native manifests.

Replaces the selected transcript fields of native manifest rows with accepted
English translations and adds provenance fields. ``variant: original`` retains
current behavior exactly; ``variant: english`` with ``require_complete: true``
fails the build when any unit is missing or below the minimum status so that
training prompts never mix native and English text unintentionally.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.utils import get_logger, sha256_text


LOGGER = get_logger(__name__)

SUPPORTED_VARIANTS = ("original", "english")

STATUS_ORDER = ("failed", "automatic_low", "automatic_medium", "automatic_high", "human_verified")
STATUS_RANK = {status: index for index, status in enumerate(STATUS_ORDER)}

# Fields replaced per dataset when the English variant is selected. For D3TEC
# and ANDROIDS, segment-level rows carry the segment text in every field name,
# and full-response/turn units replace the parent-level fields.
FIELD_MAP: dict[str, tuple[str, ...]] = {
    "cmdc": ("transcript",),
    "turkish": ("transcript",),
    "d3tec": ("transcript", "segment_transcript", "full_response_transcript"),
    "androids_interview": ("transcript", "segment_transcript", "full_turn_transcript"),
}

# Mapping of manifest row identity to (unit_id, field) for every replaceable
# field. Full-response/turn units share the unit_id across rows of the same
# response; segment units use sample_id.
FIELD_UNIT_LOOKUP: dict[str, dict[str, tuple[str, str]]] = {
    "cmdc": {"transcript": ("sample_id", "transcript")},
    "turkish": {"transcript": ("sample_id", "transcript")},
    "d3tec": {
        "transcript": ("sample_id", "segment_transcript"),
        "segment_transcript": ("sample_id", "segment_transcript"),
        "full_response_transcript": ("response_id", "full_response_transcript"),
    },
    "androids_interview": {
        "transcript": ("sample_id", "segment_transcript"),
        "segment_transcript": ("sample_id", "segment_transcript"),
        "full_turn_transcript": ("response_id", "full_turn_transcript"),
    },
}


def minimum_status_rank(minimum_status: str) -> int:
    if minimum_status not in STATUS_RANK:
        raise ValueError(
            f"Unsupported minimum_status={minimum_status!r}. "
            f"Expected one of {', '.join(STATUS_ORDER)}."
        )
    return STATUS_RANK[minimum_status]


def index_accepted(accepted_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        key = (str(row["unit_id"]), str(row["field"]))
        index[key].append(row)
    for key, records in index.items():
        index[key] = sorted(records, key=lambda record: int(record.get("part_index", 0)))
        expected_parts = set(int(record.get("part_index", 0)) for record in records)
        if len(records) > 1 and expected_parts != set(range(len(records))):
            raise ValueError(f"Accepted cache has non-contiguous parts for {key}")
    return index


def apply_overlay(
    manifest_rows: list[dict[str, Any]],
    dataset: str,
    accepted_rows: list[dict[str, Any]],
    *,
    minimum_status: str,
    require_complete: bool,
    include_failed: bool = False,
    rejected_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset not in FIELD_MAP:
        raise ValueError(f"Unsupported overlay dataset: {dataset!r}")
    rank = minimum_status_rank(minimum_status)
    accepted_index = index_accepted(accepted_rows)
    rejected_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if include_failed and rejected_rows:
        rejected_index = index_accepted(rejected_rows)

    output_rows: list[dict[str, Any]] = []
    audit = {
        "variant": "english",
        "minimum_status": minimum_status,
        "require_complete": require_complete,
        "include_failed": include_failed,
        "rows": len(manifest_rows),
        "replaced_rows": 0,
        "missing_units": [],
        "below_status_units": [],
        "failed_included_rows": 0,
        "native_rows_kept": [],
    }
    for row in manifest_rows:
        if str(row.get("dataset", "")).lower() != dataset:
            raise ValueError(
                f"Overlay row dataset={row.get('dataset')} does not match {dataset}."
            )
        new_row = dict(row)
        row_ok = True
        for field in FIELD_MAP[dataset]:
            id_key, unit_field = FIELD_UNIT_LOOKUP[dataset][field]
            unit_id = str(row.get(id_key, ""))
            records = accepted_index.get((unit_id, unit_field), [])
            source_status = ""
            if not records and include_failed:
                records = rejected_index.get((unit_id, unit_field), [])
                source_status = "failed"
            if not records:
                if require_complete:
                    raise ValueError(
                        f"Overlay incomplete for {dataset} row {unit_id} field={field}: "
                        "no accepted translation."
                    )
                audit["missing_units"].append(f"{unit_id}:{field}")
                row_ok = False
                continue
            statuses = [str(record.get("status", "failed")) for record in records]
            if not source_status and any(
                STATUS_RANK.get(status, STATUS_RANK["failed"]) < rank for status in statuses
            ):
                if require_complete:
                    raise ValueError(
                        f"Overlay incomplete for {dataset} row {unit_id} field={field}: "
                        f"statuses {statuses} below minimum_status={minimum_status}."
                    )
                audit["below_status_units"].append(f"{unit_id}:{field}:{statuses}")
                row_ok = False
                continue
            joined = " ".join(str(record["translation"]).strip() for record in records).strip()
            if not joined:
                raise ValueError(f"Empty joined translation for {dataset} row {unit_id} field={field}.")
            new_row[field] = joined
            if dataset in {"d3tec", "androids_interview"}:
                new_row.setdefault("transcript_original", {})[field] = str(row.get(field, ""))
            else:
                new_row["transcript_original"] = str(row.get(field, ""))
            new_row["translation_model"] = str(records[0].get("model", ""))
            new_row["translation_status"] = (
                "failed_included" if source_status == "failed" else min(
                    (record.get("status", "failed") for record in records),
                    key=lambda status: STATUS_RANK.get(status, STATUS_RANK["failed"]),
                )
            )
            new_row["translation_sha256"] = sha256_text(joined)
            if source_status == "failed":
                audit["failed_included_rows"] += 1
        if row_ok:
            audit["replaced_rows"] += 1
            if dataset == "cmdc":
                new_row["source_language"] = "zh"
            elif dataset == "turkish":
                new_row["source_language"] = "tr"
            elif dataset == "d3tec":
                new_row["source_language"] = "es"
            elif dataset == "androids_interview":
                new_row["source_language"] = "it"
            new_row["language"] = "en"
            new_row["transcript_variant"] = "english"
        else:
            audit["native_rows_kept"].append(str(row.get("sample_id", "")))
        output_rows.append(new_row)

    audit["accepted_records"] = len(accepted_rows)
    audit["accepted_unit_keys"] = len(accepted_index)
    return output_rows, audit
