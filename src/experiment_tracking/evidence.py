from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import read_json, sha256_file, write_json_atomic
from .schemas import validate_record
from .sidecars import ARTIFACTS_FILE, EVALUATIONS_FILE, METADATA_FILE


class EvidenceVerificationError(ValueError):
    pass


def _explicitly_invalidated(evaluation: dict[str, Any]) -> bool:
    warnings = evaluation.get("warnings") or []
    return bool(warnings) and all(
        isinstance(warning, str) and warning.startswith("invalidated by ")
        for warning in warnings
    )


def _read_record(fold_dir: Path, filename: str, schema_version: str) -> dict[str, Any]:
    path = fold_dir / filename
    if not path.is_file():
        raise EvidenceVerificationError(f"{filename} not found in {fold_dir}")
    try:
        record = read_json(path)
    except (ValueError, OSError) as error:
        raise EvidenceVerificationError(f"unreadable {filename}: {error}") from error
    ok, errors = validate_record(schema_version, record)
    if not ok:
        raise EvidenceVerificationError(f"invalid {filename}: " + "; ".join(errors))
    return record


def _artifact_verified(fold_dir: Path, artifact: dict[str, Any]) -> bool:
    full = fold_dir / artifact["path"]
    sha256 = artifact.get("sha256")
    if sha256 is not None:
        return full.is_file() and sha256_file(full) == sha256
    return full.is_dir()


def verify_artifacts_locally(fold_dir: str | Path) -> dict[str, Any]:
    """Flip artifacts.json existence/verification flags to match local disk.

    Only ever upgrades flags from false to true, and only when the local file
    hash exactly matches the recorded hash (or the checkpoint directory
    exists). Never rewrites metric or prediction content; never downgrades a
    verified flag.
    """
    target = Path(fold_dir)
    record = _read_record(target, ARTIFACTS_FILE, "audiollm.artifacts.v1")
    changed: list[dict[str, Any]] = []
    for artifact in record["artifacts"]:
        verified = _artifact_verified(target, artifact)
        updates: dict[str, Any] = {}
        if verified and artifact.get("exists_locally") is not True:
            updates["exists_locally"] = True
        if verified and artifact.get("locally_verified") is not True:
            updates["locally_verified"] = True
        if updates:
            artifact.update(updates)
            changed.append({"path": artifact["path"], **updates})
    if changed:
        write_json_atomic(target / ARTIFACTS_FILE, record)
    return {
        "fold_dir": str(target),
        "verified_artifacts": sum(
            1 for artifact in record["artifacts"] if artifact.get("locally_verified") is True
        ),
        "total_artifacts": len(record["artifacts"]),
        "changed": changed,
    }


def verify_evaluations_locally(fold_dir: str | Path) -> dict[str, Any]:
    """Mark evaluation records locally verified / reportable only when their
    metrics and predictions artifacts are locally verified and the record has
    no warnings. Refuses when any required artifact is unverified."""
    target = Path(fold_dir)
    record = _read_record(target, EVALUATIONS_FILE, "audiollm.evaluations.v1")
    artifacts = _read_record(target, ARTIFACTS_FILE, "audiollm.artifacts.v1")
    by_path = {artifact["path"]: artifact for artifact in artifacts["artifacts"]}
    changed: list[dict[str, Any]] = []
    for evaluation in record["evaluations"]:
        # A warning is an explicit disqualification. Preserve the record for
        # audit history, keep it non-reportable, and continue verifying other
        # valid evaluations in the same attempt.
        if _explicitly_invalidated(evaluation):
            updates: dict[str, Any] = {}
            if evaluation.get("locally_verified") is not False:
                updates["locally_verified"] = False
            if evaluation.get("reportable") is not False:
                updates["reportable"] = False
            if updates:
                evaluation.update(updates)
                changed.append({"evaluation_id": evaluation["evaluation_id"], **updates})
            continue
        if evaluation.get("warnings"):
            raise EvidenceVerificationError(
                f"evaluation {evaluation['evaluation_id']} has warnings and cannot be reportable: "
                f"{evaluation['warnings']}"
            )
        required = [
            evaluation.get("metrics_artifact_path"),
            evaluation.get("predictions_artifact_path"),
        ]
        missing = [
            path for path in required if path and by_path.get(path, {}).get("locally_verified") is not True
        ]
        if missing:
            raise EvidenceVerificationError(
                f"evaluation {evaluation['evaluation_id']} requires locally verified artifacts "
                f"first: {sorted(set(missing))}"
            )
        updates: dict[str, Any] = {}
        if evaluation.get("locally_verified") is not True:
            updates["locally_verified"] = True
        if evaluation.get("reportable") is not True:
            updates["reportable"] = True
        if updates:
            evaluation.update(updates)
            changed.append({"evaluation_id": evaluation["evaluation_id"], **updates})
    if changed:
        write_json_atomic(target / EVALUATIONS_FILE, record)
    return {
        "fold_dir": str(target),
        "verified_evaluations": sum(
            1 for evaluation in record["evaluations"] if evaluation.get("locally_verified") is True
        ),
        "reportable_evaluations": sum(
            1 for evaluation in record["evaluations"] if evaluation.get("reportable") is True
        ),
        "total_evaluations": len(record["evaluations"]),
        "changed": changed,
    }


def set_metadata_supersedes(fold_dir: str | Path, supersedes_attempt_id: str) -> dict[str, Any]:
    """Record the attempt this run supersedes in metadata.json.

    Only sets the field when it is currently absent; refuses to change an
    existing value (identity records must not be rewritten)."""
    from .identity import validate_attempt_id

    target = Path(fold_dir)
    record = _read_record(target, METADATA_FILE, "audiollm.metadata.v1")
    if not validate_attempt_id(supersedes_attempt_id):
        raise EvidenceVerificationError(
            f"supersedes_attempt_id must be a valid modern attempt id: {supersedes_attempt_id!r}"
        )
    existing = record.get("supersedes_attempt_id")
    if existing is not None:
        if existing == supersedes_attempt_id:
            return {
                "fold_dir": str(target),
                "supersedes_attempt_id": existing,
                "changed": False,
            }
        raise EvidenceVerificationError(
            f"refusing to overwrite metadata.supersedes_attempt_id "
            f"{existing!r} with {supersedes_attempt_id!r}"
        )
    record["supersedes_attempt_id"] = supersedes_attempt_id
    write_json_atomic(target / METADATA_FILE, record)
    return {
        "fold_dir": str(target),
        "supersedes_attempt_id": supersedes_attempt_id,
        "changed": True,
    }
