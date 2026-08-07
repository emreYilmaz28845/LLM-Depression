from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_sha256, format_utc_timestamp, utc_now
from .constants import LEGACY_ATTEMPT_ID_ALGORITHM_VERSION

_ATTEMPT_ID_PATTERN = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]+-[0-9a-f]{8}-[0-9a-f]{8}$"
)
_SAFE_ID_PATTERN = re.compile(r"^[a-z0-9._-]+$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def sanitize_logical_run_name(logical_run_name: str) -> str:
    return re.sub(r"[^a-z0-9._-]", "-", logical_run_name.lower())


def validate_attempt_id(attempt_id: str) -> bool:
    return isinstance(attempt_id, str) and _ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is not None


def new_attempt_id(logical_run_name: str, git_commit: str, at_utc: datetime | None = None) -> str:
    if not isinstance(git_commit, str) or _GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a 40-character lowercase hex SHA")
    timestamp = at_utc if at_utc is not None else utc_now()
    compact = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{compact}-{sanitize_logical_run_name(logical_run_name)}"
        f"-{git_commit[:8]}-{secrets.token_hex(4)}"
    )


def reserve_attempt_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=False, exist_ok=False)
    return target


def logical_run_id(
    *,
    group_id: str | None,
    logical_run_name: str,
    dataset: str | None,
    modality: str | None,
    method: str | None,
    seed: int | None,
) -> str:
    payload = {
        "group_id": group_id,
        "logical_run_name": logical_run_name,
        "dataset": dataset,
        "modality": modality,
        "method": method,
        "seed": seed,
    }
    return "lr-" + canonical_sha256(payload)[:24]


def artifact_id(
    *,
    attempt_id: str,
    fold: int,
    role: str,
    relative_path: str,
    artifact_sha256: str | None,
) -> str:
    payload = {
        "attempt_id": attempt_id,
        "fold": fold,
        "role": role,
        "relative_path": relative_path,
        "artifact_sha256": artifact_sha256,
    }
    return "art-" + canonical_sha256(payload)[:24]


def evaluation_id(
    *,
    attempt_id: str,
    fold: int,
    dataset: str,
    split_name: str,
    split_protocol: str,
    checkpoint_role: str,
    checkpoint_path: str,
    backend: str,
    evaluation_view: str,
    aggregation: str,
    metric_namespace: str,
    metrics_artifact_sha256: str | None,
) -> str:
    payload = {
        "attempt_id": attempt_id,
        "fold": fold,
        "dataset": dataset,
        "split_name": split_name,
        "split_protocol": split_protocol,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": checkpoint_path,
        "backend": backend,
        "evaluation_view": evaluation_view,
        "aggregation": aggregation,
        "metric_namespace": metric_namespace,
        "metrics_artifact_sha256": metrics_artifact_sha256,
    }
    return "eval-" + canonical_sha256(payload)[:24]


def wandb_run_id(attempt_id: str, fold: int) -> str:
    return f"{attempt_id}-fold{fold}"


def legacy_attempt_id(
    identity_input: Mapping[str, Any],
    algorithm_version: str = LEGACY_ATTEMPT_ID_ALGORITHM_VERSION,
) -> str:
    digest = canonical_sha256(dict(identity_input))
    return f"legacy-{algorithm_version}-{digest[:24]}"


def deployed_source_sha256(file_records: list[Mapping[str, Any]]) -> str:
    ordered = sorted(file_records, key=lambda record: str(record["path"]))
    return canonical_sha256([dict(record) for record in ordered])
