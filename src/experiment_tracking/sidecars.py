from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .canonical import canonical_sha256, read_json, read_jsonl, sha256_file
from .schemas import (
    validate_artifacts,
    validate_evaluations,
    validate_job_event,
    validate_metadata,
    validate_record,
    validate_status,
)

METADATA_FILE = "metadata.json"
STATUS_FILE = "status.json"
JOBS_FILE = "jobs.jsonl"
ARTIFACTS_FILE = "artifacts.json"
EVALUATIONS_FILE = "evaluations.json"

SIDECAR_FILES = (
    METADATA_FILE,
    STATUS_FILE,
    JOBS_FILE,
    ARTIFACTS_FILE,
    EVALUATIONS_FILE,
)


class SidecarValidationError(ValueError):
    """The fold directory carries modern tracking sidecars that are malformed
    or contradictory. Such runs must never be imported as legacy runs."""


@dataclasses.dataclass(frozen=True)
class ModernSidecars:
    fold_dir: str
    metadata: dict[str, Any]
    status: dict[str, Any]
    jobs: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    file_sha256: dict[str, str]

    @property
    def attempt_id(self) -> str:
        return str(self.metadata["attempt_id"])

    @property
    def fold(self) -> int:
        return int(self.metadata["fold"])

    @property
    def state(self) -> str:
        return str(self.status["state"])

    def evidence_sha256(self, run_config_sha256: str | None) -> str:
        records = [{"path": path, "sha256": digest} for path, digest in self.file_sha256.items()]
        if run_config_sha256 is not None:
            records.append({"path": "run_config.yaml", "sha256": run_config_sha256})
        return canonical_sha256(sorted(records, key=lambda record: record["path"]))


def is_modern_tracked(fold_dir: str | Path) -> bool:
    return (Path(fold_dir) / METADATA_FILE).is_file()


def _read_file(path: Path) -> tuple[Any, str]:
    try:
        content = read_json(path) if path.suffix == ".json" else read_jsonl(path)
        return content, sha256_file(path)
    except (ValueError, OSError) as error:
        raise SidecarValidationError(f"unreadable sidecar {path.name}: {error}") from error


def read_modern_sidecars(fold_dir: str | Path) -> ModernSidecars | None:
    """Read and validate the modern tracking sidecars of a fold directory.

    Returns None when the directory carries no modern tracking evidence
    (no metadata.json); the legacy qualification path remains responsible.
    Raises SidecarValidationError when sidecars exist but are malformed or
    contradictory, so callers fail closed instead of importing the run as a
    synthetic legacy attempt.
    """
    target = Path(fold_dir)
    if not is_modern_tracked(target):
        return None

    metadata, metadata_sha = _read_file(target / METADATA_FILE)
    status, status_sha = _read_file(target / STATUS_FILE)
    jobs, jobs_sha = _read_file(target / JOBS_FILE)
    artifacts, artifacts_sha = _read_file(target / ARTIFACTS_FILE)
    evaluations_path = target / EVALUATIONS_FILE
    if evaluations_path.is_file():
        evaluations, evaluations_sha = _read_file(evaluations_path)
    else:
        evaluations, evaluations_sha = [], None

    for name, version, record in (
        (METADATA_FILE, "audiollm.metadata.v1", metadata),
        (STATUS_FILE, "audiollm.status.v1", status),
        (ARTIFACTS_FILE, "audiollm.artifacts.v1", artifacts),
    ):
        ok, errors = validate_record(version, record)
        if not ok:
            raise SidecarValidationError(f"invalid {name}: " + "; ".join(errors))
    if evaluations_path.is_file():
        ok, errors = validate_record("audiollm.evaluations.v1", evaluations)
        if not ok:
            raise SidecarValidationError(
                f"invalid {EVALUATIONS_FILE}: " + "; ".join(errors)
            )
    for index, event in enumerate(jobs):
        ok, errors = validate_job_event(event)
        if not ok:
            raise SidecarValidationError(
                f"invalid jobs.jsonl[{index}]: " + "; ".join(errors)
            )
    if not jobs:
        raise SidecarValidationError("jobs.jsonl is empty for a modern tracked run")

    errors: list[str] = []
    if metadata.get("attempt_id") != status.get("attempt_id"):
        errors.append("metadata.json and status.json attempt_id differ")
    if metadata.get("attempt_id") != artifacts.get("attempt_id"):
        errors.append("metadata.json and artifacts.json attempt_id differ")
    if evaluations_path.is_file() and metadata.get("attempt_id") != evaluations.get("attempt_id"):
        errors.append("metadata.json and evaluations.json attempt_id differ")
    if metadata.get("fold") != status.get("fold"):
        errors.append("metadata.json and status.json fold differ")
    if metadata.get("fold") != artifacts.get("fold"):
        errors.append("metadata.json and artifacts.json fold differ")
    if evaluations_path.is_file() and metadata.get("fold") != evaluations.get("fold"):
        errors.append("metadata.json and evaluations.json fold differ")
    for index, event in enumerate(jobs):
        if event.get("attempt_id") != metadata.get("attempt_id"):
            errors.append(f"jobs.jsonl[{index}] attempt_id differs from metadata.json")
        if event.get("fold") != metadata.get("fold"):
            errors.append(f"jobs.jsonl[{index}] fold differs from metadata.json")
    if errors:
        raise SidecarValidationError("contradictory modern sidecars: " + "; ".join(errors))

    file_sha256 = {
        METADATA_FILE: metadata_sha,
        STATUS_FILE: status_sha,
        JOBS_FILE: jobs_sha,
        ARTIFACTS_FILE: artifacts_sha,
    }
    if evaluations_sha is not None:
        file_sha256[EVALUATIONS_FILE] = evaluations_sha
    return ModernSidecars(
        fold_dir=str(target),
        metadata=metadata,
        status=status,
        jobs=tuple(jobs),
        artifacts=tuple(artifacts["artifacts"] or []),
        evaluations=tuple(evaluations.get("evaluations") or [])
        if isinstance(evaluations, dict)
        else (),
        file_sha256=file_sha256,
    )


def reportable_state_issues(sidecars: ModernSidecars) -> list[str]:
    """Fail-closed gate: a sidecar state of REPORTABLE is only consistent when
    every evaluation and every hashed artifact is locally verified. Returns the
    contradictions, or an empty list when the state is consistent."""
    if sidecars.state != "REPORTABLE":
        return []
    issues: list[str] = []
    for index, artifact in enumerate(sidecars.artifacts):
        if artifact.get("sha256") is not None and artifact.get("locally_verified") is not True:
            issues.append(
                f"artifacts.json[{index}] hashed artifact not locally verified: "
                f"{artifact.get('path')!r}"
            )
    for index, evaluation in enumerate(sidecars.evaluations):
        if evaluation.get("locally_verified") is not True:
            issues.append(f"evaluations.json[{index}] not locally verified")
        if evaluation.get("reportable") is not True:
            issues.append(f"evaluations.json[{index}] not reportable")
        if evaluation.get("warnings"):
            issues.append(f"evaluations.json[{index}] has warnings: {evaluation['warnings']}")
    return issues
