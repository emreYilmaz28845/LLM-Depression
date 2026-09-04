from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import normalize_relative_path, read_json, sha256_file, write_json_atomic
from .constants import ARTIFACT_TYPES, SCHEMA_VERSION_ARTIFACTS
from .identity import artifact_id
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


def register_local_artifacts(
    fold_dir: str | Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Register existing local files in an artifacts sidecar.

    This is a deliberately narrow evidence-correction API. It never changes
    an existing artifact's identity or content, and it writes only after every
    requested entry has passed path, file, hash, and sidecar validation. Each
    new record is marked locally verified because its hash was computed from
    the existing local file in this call.

    Entries contain ``path``, ``artifact_type``, and ``role``. The optional
    ``exists_on_mn5`` field records whether the caller has independent proof
    that the same artifact was produced on MN5; it is never inferred here.
    """
    target = Path(fold_dir)
    record = _read_record(target, ARTIFACTS_FILE, SCHEMA_VERSION_ARTIFACTS)
    if not isinstance(entries, list) or not entries:
        raise EvidenceVerificationError("artifact entries must be a non-empty array")

    root = target.resolve()
    existing_by_path: dict[str, dict[str, Any]] = {}
    for artifact in record["artifacts"]:
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            continue
        if path in existing_by_path:
            raise EvidenceVerificationError(
                f"artifacts.json contains duplicate path {path!r}"
            )
        existing_by_path[path] = artifact

    registered: list[str] = []
    already_registered: list[str] = []
    changed_existing: list[str] = []
    requested_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EvidenceVerificationError(f"artifact entry {index} must be an object")
        unknown = sorted(set(entry) - {"path", "artifact_type", "role", "exists_on_mn5"})
        if unknown:
            raise EvidenceVerificationError(
                f"artifact entry {index} has unsupported fields: {unknown}"
            )
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise EvidenceVerificationError(f"artifact entry {index}.path must be a string")
        try:
            relative_path = normalize_relative_path(relative_path)
        except ValueError as error:
            raise EvidenceVerificationError(
                f"artifact entry {index}.path is unsafe: {error}"
            ) from error
        if relative_path in requested_paths:
            raise EvidenceVerificationError(
                f"artifact entries contain duplicate path {relative_path!r}"
            )
        requested_paths.add(relative_path)

        artifact_type = entry.get("artifact_type")
        if not isinstance(artifact_type, str) or artifact_type not in ARTIFACT_TYPES:
            raise EvidenceVerificationError(
                f"artifact entry {index}.artifact_type must be one of {ARTIFACT_TYPES}"
            )
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise EvidenceVerificationError(f"artifact entry {index}.role must be non-empty")
        exists_on_mn5 = entry.get("exists_on_mn5")
        if exists_on_mn5 is not None and not isinstance(exists_on_mn5, bool):
            raise EvidenceVerificationError(
                f"artifact entry {index}.exists_on_mn5 must be a boolean or null"
            )

        full_path = (root / relative_path).resolve()
        try:
            full_path.relative_to(root)
        except ValueError as error:
            raise EvidenceVerificationError(
                f"artifact entry {index}.path resolves outside fold directory: {relative_path!r}"
            ) from error
        if not full_path.is_file():
            raise EvidenceVerificationError(
                f"artifact entry {index}.path is not an existing regular file: {relative_path!r}"
            )
        sha = sha256_file(full_path)
        size_bytes = full_path.stat().st_size
        expected_id = artifact_id(
            attempt_id=record["attempt_id"],
            fold=record["fold"],
            role=role,
            relative_path=relative_path,
            artifact_sha256=sha,
        )
        existing = existing_by_path.get(relative_path)
        if existing is not None:
            expected_fields = {
                "artifact_id": expected_id,
                "artifact_type": artifact_type,
                "role": role,
                "path": relative_path,
                "sha256": sha,
                "size_bytes": size_bytes,
            }
            mismatches = {
                key: {"recorded": existing.get(key), "expected": value}
                for key, value in expected_fields.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise EvidenceVerificationError(
                    f"refusing to overwrite contradictory artifact {relative_path!r}: {mismatches}"
                )
            updates: dict[str, Any] = {}
            if existing.get("exists_locally") is not True:
                updates["exists_locally"] = True
            if existing.get("locally_verified") is not True:
                updates["locally_verified"] = True
            if updates:
                existing.update(updates)
                changed_existing.append(relative_path)
            else:
                already_registered.append(relative_path)
            continue

        artifact = {
            "artifact_id": expected_id,
            "artifact_type": artifact_type,
            "role": role,
            "path": relative_path,
            "sha256": sha,
            "size_bytes": size_bytes,
            "exists_on_mn5": exists_on_mn5,
            "exists_locally": True,
            "locally_verified": True,
        }
        record["artifacts"].append(artifact)
        existing_by_path[relative_path] = artifact
        registered.append(relative_path)

    ok, errors = validate_record(SCHEMA_VERSION_ARTIFACTS, record)
    if not ok:
        raise EvidenceVerificationError(
            "refusing to write corrected artifacts.json: " + "; ".join(errors)
        )
    if registered or changed_existing:
        write_json_atomic(target / ARTIFACTS_FILE, record)
    return {
        "fold_dir": str(target),
        "registered": registered,
        "already_registered": already_registered,
        "changed_existing": changed_existing,
        "total_artifacts": len(record["artifacts"]),
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
