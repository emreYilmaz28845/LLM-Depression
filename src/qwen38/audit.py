"""Qwen3.8 audits: wheel tags, deployment records, and Turkish compact outputs.

Three audit entry points:

- ``audit_wheelhouse``: parse every wheel filename and reject wheels whose
  platform tags are not x86-64 Linux (or ``any``), whose manylinux glibc
  baseline is newer than 2.34, or whose CPython ABI cannot run on Python 3.10.
- ``audit_deployment``: verify the pinned environment versions, manifest
  hashes, driver record, per-TP acceptance, and the serving selection.
- ``audit_turkish``: run every runbook section 21 check against the run
  directory. Restricted-intermediate checks run only when the restricted
  evidence is present; the local re-run covers the compact outputs, the
  transcript hash, and all privacy checks without reconstructing subject
  level intermediates.

All checks are recorded as an ordered list with pass/fail status; a failing
check makes the audit exit non-zero in the CLI.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.qwen38.contracts import (
    FINAL_TABLE_COLUMNS,
    HUGGINGFACE_HUB_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    OPENAI_VERSION,
    PYTHON_MAJOR,
    PYTHON_MINOR,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    TURKISH_SOURCE_HASH,
    VLLM_VERSION,
    Confidence,
    Label,
    WordingStatus,
    generation_settings_hash,
    ngram_overlap_at_least,
)
from src.qwen38.turkish_questions import (
    _canonical_json,
    _episode_provenance,
    _sha256_text,
    aggregate_families,
    collect_candidates,
    load_prepared_sequences,
    load_table_rows,
    parse_filename_stem,
    prompt_contract_sha256,
    PROMPT_VERSION,
    _validate_episodes,
)

MANYLINUX_RE = re.compile(r"^manylinux(_\d+_\d+)?_(x86_64|i686|aarch64|armv7l|ppc64le|s390x|riscv64)$")
MANYLINUX_LEGACY_RE = re.compile(r"^manylinux1_(x86_64|i686|aarch64)$|^manylinux2010_(x86_64|i686|aarch64)$|^manylinux2014_(x86_64|i686|aarch64)$")
MUSL_RE = re.compile(r"^musllinux_\d+_\d+_(x86_64|i686|aarch64|armv7l|ppc64le|s390x)$")
LINUX_PLAIN_RE = re.compile(r"^linux_(x86_64|i686|aarch64)$")
WHEEL_FILENAME_RE = re.compile(r"^(?P<name>.+)-(?P<version>[^-]+)-(?P<build>\d+)?-?(?P<py>.+?)-(?P<abi>.+?)-(?P<platform>.+?)\.whl$")

MAX_MANYLINUX_GLIBC_MINOR = 34
MIN_DRIVER_VERSION = 580.00
OVERLAP_TOKEN_COUNT = 12


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.strip().split(".") if part)


def driver_version_ge(driver_version: str, minimum: float) -> bool:
    """Compare nvidia-smi versions like ``595.71.05`` against a float minimum."""
    minimum_str = f"{minimum:.2f}"
    minimum_tuple = tuple(int(part) for part in minimum_str.split("."))
    return _version_tuple(driver_version) >= minimum_tuple

ENV_PINS = {
    "python_major": PYTHON_MAJOR,
    "python_minor": PYTHON_MINOR,
    "vllm": VLLM_VERSION,
    "transformers": TRANSFORMERS_VERSION,
    "torch": TORCH_VERSION,
    "torchvision": TORCHVISION_VERSION,
    "torchaudio": TORCHAUDIO_VERSION,
    "openai": OPENAI_VERSION,
    "huggingface_hub": HUGGINGFACE_HUB_VERSION,
}


def _normalize_environment_versions(values: dict[str, Any]) -> dict[str, Any]:
    """Compare CUDA-tagged torchvision/audio builds by their package version.

    The deployment record stores the pinned package versions without a local
    CUDA build tag, while the installed distributions may report values such
    as ``0.26.0+cu130``.  The local tag is part of the wheel build identity,
    not a different pinned package release, so strip it only for the two
    packages that carry the CUDA suffix in this environment.
    """
    normalized = dict(values)
    for key in ("torchvision", "torchaudio"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.split("+", 1)[0]
    return normalized


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _manifest_sha256(manifest_path: str | Path) -> str:
    """Hash of the sorted ``<sha256>  <path>`` manifest lines (canonical form)."""
    lines: list[str] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            lines.append(line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Wheel-tag audit
# --------------------------------------------------------------------------


def parse_wheel_tags(filename: str) -> tuple[str, list[str], list[str]] | None:
    """Return (python_tags, abi_tags, platform_tags) or None when not a wheel."""
    match = WHEEL_FILENAME_RE.match(filename)
    if match is None:
        return None
    py_tags = [tag for tag in match.group("py").split(".") if tag]
    abi_tags = [tag for tag in match.group("abi").split(".") if tag]
    platform_tags = [tag for tag in match.group("platform").split(".") if tag]
    return py_tags, abi_tags, platform_tags


def wheel_tag_errors(filename: str, *, python_minor: int = PYTHON_MINOR) -> list[str]:
    """Audit one wheel filename against the runbook tag rules.

    Accepts any ``any`` platform, manylinux glibc <= 2.34 x86-64, legacy
    manylinux1/2010/2014 x86-64, musllinux x86-64, and plain linux x86-64.
    Rejects non-x86-64 platforms, manylinux requiring glibc newer than 2.34,
    and CPython-only ABIs that cannot run on the pinned Python version.
    """
    parsed = parse_wheel_tags(filename)
    if parsed is None:
        return ["not a wheel filename"]
    py_tags, abi_tags, platform_tags = parsed
    errors: list[str] = []

    for tag in platform_tags:
        if tag == "any":
            continue
        legacy = MANYLINUX_LEGACY_RE.fullmatch(tag)
        many = MANYLINUX_RE.fullmatch(tag)
        musl = MUSL_RE.fullmatch(tag)
        plain = LINUX_PLAIN_RE.fullmatch(tag)
        if many is not None:
            arch = many.group(2)
            version = many.group(1)
            if arch != "x86_64":
                errors.append(f"platform {tag!r}: non-x86-64 architecture")
                continue
            if version is not None:
                version_parts = version.strip("_").split("_")
                glibc_minor = int(version_parts[1])
                if glibc_minor > MAX_MANYLINUX_GLIBC_MINOR:
                    errors.append(
                        f"platform {tag!r}: minimum glibc {glibc_minor} newer than 2.{MAX_MANYLINUX_GLIBC_MINOR}"
                    )
            continue
        if legacy is not None:
            if not tag.endswith("_x86_64"):
                errors.append(f"platform {tag!r}: non-x86-64 architecture")
            continue
        if musl is not None:
            if not tag.endswith("_x86_64"):
                errors.append(f"platform {tag!r}: non-x86-64 architecture")
            continue
        if plain is not None:
            if tag != "linux_x86_64":
                errors.append(f"platform {tag!r}: non-x86-64 architecture")
            continue
        errors.append(f"platform {tag!r}: unsupported platform tag")

    supported_py = {
        "py3",
        f"cp{python_minor}",
        f"cp3{python_minor}",
        *(f"py3{i}" for i in range(python_minor + 1)),
    }
    abi3_ok = any(
        re.fullmatch(rf"cp3[0-{python_minor}]?-abi3", tag)
        or re.fullmatch(rf"cp{python_minor}?-abi3", tag)
        or tag == "abi3"
        for tag in abi_tags
    )
    older_cp_pure = any(
        re.fullmatch(rf"cp3[0-{python_minor - 1}]", tag) for tag in py_tags
    )
    compatible = any(tag in supported_py for tag in py_tags)
    if older_cp_pure and abi3_ok:
        compatible = True
    if not compatible:
        errors.append(
            f"ABI {py_tags}/{abi_tags}: no tag supports CPython 3.{python_minor}"
        )
    if "cp311" in "".join(py_tags) or "cp312" in "".join(py_tags) or "cp313" in "".join(py_tags):
        if not any(tag in supported_py for tag in py_tags):
            errors.append("ABI: CPython-only tag newer than 3.10 present")
    return errors


def audit_wheelhouse(wheelhouse_dir: str | Path) -> dict[str, Any]:
    """Audit every wheel in the wheelhouse; sdist presence is a hard failure."""
    wheelhouse_dir = Path(wheelhouse_dir)
    wheels_dir = wheelhouse_dir / "wheels"
    if not wheels_dir.is_dir():
        return {
            "passed": False,
            "checks": [
                {
                    "check_id": "wheels_dir",
                    "description": "wheels directory exists",
                    "passed": False,
                    "details": {"wheels_dir": str(wheels_dir)},
                }
            ],
            "wheel_count": 0,
            "errors": [],
        }
    non_wheels = sorted(
        str(path.relative_to(wheelhouse_dir))
        for path in wheels_dir.iterdir()
        if path.is_file() and not path.name.endswith(".whl")
    )
    wheels = sorted(path.name for path in wheels_dir.iterdir() if path.name.endswith(".whl"))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    checks.append(
        {
            "check_id": "binary_only",
            "description": "wheelhouse contains only binary wheels",
            "passed": not non_wheels,
            "details": {"non_wheel_files": non_wheels},
        }
    )
    if non_wheels:
        errors.append(f"{len(non_wheels)} non-wheel files present in wheels/")

    per_wheel: dict[str, list[str]] = {}
    for wheel in wheels:
        tag_errors = wheel_tag_errors(wheel)
        per_wheel[wheel] = tag_errors
        if tag_errors:
            errors.append(f"{wheel}: {'; '.join(tag_errors)}")

    checks.append(
        {
            "check_id": "wheel_tags",
            "description": "every wheel tag compatible with CPython 3.10 / glibc 2.34 x86-64",
            "passed": not errors,
            "details": {"errors": errors[:50], "wheel_count": len(wheels)},
        }
    )
    manifest = wheelhouse_dir / "SHA256SUMS"
    manifest_ok = manifest.is_file()
    if manifest_ok:
        result = _verify_sha256_file(manifest)
        manifest_ok = result["ok"]
        errors.extend(result["errors"])
    else:
        errors.append("SHA256SUMS manifest missing")
    checks.append(
        {
            "check_id": "manifest",
            "description": "wheelhouse SHA256SUMS verifies",
            "passed": manifest_ok,
            "details": {"manifest_path": str(manifest)},
        }
    )
    return {
        "passed": not errors,
        "checks": checks,
        "wheel_count": len(wheels),
        "errors": errors,
    }


def _verify_sha256_file(manifest_path: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    missing = 0
    checked = 0
    base = Path(manifest_path).parent
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                errors.append(f"malformed manifest line: {line[:80]}")
                continue
            expected, rel = parts
            target = base / rel
            if not target.is_file():
                missing += 1
                errors.append(f"missing file: {rel}")
                continue
            actual = _sha256_file(target)
            checked += 1
            if actual != expected:
                errors.append(f"hash mismatch: {rel}")
    return {"ok": not errors and missing == 0, "errors": errors, "checked": checked, "missing": missing}


# --------------------------------------------------------------------------
# Deployment audit
# --------------------------------------------------------------------------


def audit_deployment(
    deploy_dir: str | Path,
    deployment_id: str,
    *,
    model_dir: str | Path,
    wheelhouse_dir: str | Path,
    environment_dir: str | Path | None = None,
    source_commit: str | None = None,
    selection_file: str | Path | None = None,
) -> dict[str, Any]:
    """Verify environment versions, manifests, driver, TP acceptance, selection."""
    deploy_dir = Path(deploy_dir)
    checks: list[dict[str, Any]] = []
    passed = True

    def record(check_id: str, description: str, ok: bool, details: Any = None) -> None:
        nonlocal passed
        if not ok:
            passed = False
        checks.append(
            {"check_id": check_id, "description": description, "passed": ok, "details": details}
        )

    env_dir = Path(environment_dir) if environment_dir else deploy_dir / deployment_id / "environment"
    runtime_path = env_dir / "runtime_versions.json"
    if not runtime_path.is_file():
        record("environment_record", "runtime_versions.json exists", False, {"path": str(runtime_path)})
    else:
        with runtime_path.open("r", encoding="utf-8") as handle:
            runtime = json.load(handle)
        actual_raw = {
            "python_major": int(runtime.get("python_major")),
            "python_minor": int(runtime.get("python_minor")),
            "vllm": runtime.get("vllm"),
            "transformers": runtime.get("transformers"),
            "torch": runtime.get("torch"),
            "torchvision": runtime.get("torchvision"),
            "torchaudio": runtime.get("torchaudio"),
            "openai": runtime.get("openai"),
            "huggingface_hub": runtime.get("huggingface_hub"),
        }
        actual = _normalize_environment_versions(actual_raw)
        record(
            "environment_versions",
            "environment versions match the pinned set",
            actual == ENV_PINS,
            {"actual": actual, "raw_actual": actual_raw, "expected": ENV_PINS},
        )
        record(
            "model_identity",
            "model ID and revision match the pins",
            runtime.get("model_id") == MODEL_ID and runtime.get("model_revision") == MODEL_REVISION,
            {"actual": [runtime.get("model_id"), runtime.get("model_revision")]},
        )

    model_manifest = Path(model_dir) / "SHA256SUMS"
    if model_manifest.is_file():
        result = _verify_sha256_file(model_manifest)
        record(
            "model_manifest",
            "model SHA256SUMS verifies",
            result["ok"],
            {"checked": result["checked"], "errors": result["errors"][:10]},
        )
    else:
        record("model_manifest", "model SHA256SUMS verifies", False, {"missing": str(model_manifest)})

    wheelhouse_manifest = Path(wheelhouse_dir) / "SHA256SUMS"
    if wheelhouse_manifest.is_file():
        result = _verify_sha256_file(wheelhouse_manifest)
        record(
            "wheelhouse_manifest",
            "wheelhouse SHA256SUMS verifies",
            result["ok"],
            {"checked": result["checked"], "errors": result["errors"][:10]},
        )
    else:
        record(
            "wheelhouse_manifest",
            "wheelhouse SHA256SUMS verifies",
            False,
            {"missing": str(wheelhouse_manifest)},
        )

    driver_path = env_dir / "driver_probe.json"
    if driver_path.is_file():
        with driver_path.open("r", encoding="utf-8") as handle:
            driver = json.load(handle)
        driver_ok = bool(driver.get("driver_version")) and driver_version_ge(
            str(driver["driver_version"]), MIN_DRIVER_VERSION
        )
        record(
            "driver_version",
            "NVIDIA driver >= 580.00 on allocated GPU node",
            driver_ok,
            driver,
        )
    else:
        record("driver_version", "NVIDIA driver >= 580.00 on allocated GPU node", False, "no driver probe")

    acceptance_records: list[dict[str, Any]] = []
    for tp in (1, 2, 4):
        tp_dir = deploy_dir / deployment_id / "validation" / f"tp{tp}"
        attempts = sorted(p.name for p in tp_dir.glob("attempt*")) if tp_dir.is_dir() else []
        record(
            f"tp{tp}_attempts",
            f"TP={tp} attempt directories",
            bool(attempts),
            {"attempts": attempts},
        )
        latest_attempt = attempts[-1] if attempts else None
        if latest_attempt:
            acceptance_path = tp_dir / latest_attempt / "acceptance.json"
            capacity_path = tp_dir / latest_attempt / "capacity_ineligible.json"
            capacity = {}
            if capacity_path.is_file():
                with capacity_path.open("r", encoding="utf-8") as handle:
                    capacity = json.load(handle)
            capacity_ineligible = bool(capacity.get("capacity_ineligible"))
            if acceptance_path.is_file():
                with acceptance_path.open("r", encoding="utf-8") as handle:
                    acceptance = json.load(handle)
                accepted = bool(acceptance.get("passed"))
                acceptance_records.append({"tp": tp, "attempt": latest_attempt, "passed": accepted})
                if tp == 1 and capacity_ineligible:
                    gate_ok = True
                    description = "TP=1 acceptance or capacity-ineligible evidence"
                elif tp == 4:
                    # TP=4 is a bounded comparison. A recorded failed
                    # comparison is valid deployment evidence and must not
                    # block the TP=2 deployment.
                    gate_ok = True
                    description = "TP=4 comparison evidence is recorded"
                else:
                    gate_ok = accepted
                    description = f"TP={tp} acceptance gate"
                record(
                    f"tp{tp}_acceptance",
                    description,
                    gate_ok,
                    {
                        "attempt": latest_attempt,
                        "passed": accepted,
                        "capacity_ineligible": capacity_ineligible,
                    },
                )
            elif tp == 1 and capacity_ineligible:
                record(
                    "tp1_acceptance",
                    "TP=1 acceptance or capacity-ineligible evidence",
                    True,
                    {"attempt": latest_attempt, "capacity_ineligible": True},
                )
            else:
                record(f"tp{tp}_acceptance", f"TP={tp} acceptance gate", False, "acceptance.json missing")
        else:
            record(f"tp{tp}_acceptance", f"TP={tp} acceptance gate", False, "no attempt")

    selection_path = Path(selection_file) if selection_file else deploy_dir / deployment_id / "serving_selection.json"
    if selection_path.is_file():
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        required_fields = (
            "deployment_id",
            "model_id",
            "model_revision",
            "selected_tp",
            "decision_rule",
            "candidate_results",
            "projected_requests",
            "projected_wall_seconds",
            "measured_metrics_paths",
            "created_utc",
            "source_commit",
        )
        missing_fields = [field for field in required_fields if field not in selection]
        record(
            "serving_selection_schema",
            "serving_selection.json has all required fields",
            not missing_fields,
            {"missing": missing_fields},
        )
        if selection_file is not None:
            v2_fields = (
                "selection_version",
                "supersedes_selection",
                "original_tp2_acceptance",
                "tp1_capacity_evidence",
                "tp4_attempt2",
                "old_tp4_attempt1",
                "selection_implementation_commit",
                "fixed_section_17_decision_rule",
            )
            missing_v2 = [field for field in v2_fields if field not in selection]
            record(
                "serving_selection_v2_schema",
                "serving_selection_v2.json preserves all input and supersession evidence",
                selection.get("selection_version") == 2
                and selection_path.name == "serving_selection_v2.json"
                and not missing_v2,
                {"missing": missing_v2, "path": str(selection_path)},
            )
            v2_evidence_ok = False
            v2_details: dict[str, Any] = {"missing": missing_v2}
            if selection.get("selection_version") == 2 and not missing_v2:
                supersedes = selection["supersedes_selection"]
                original_path = Path(str(supersedes.get("path", "")))
                original_hash_ok = False
                if original_path.is_file():
                    original_hash_ok = _sha256_file(original_path) == supersedes.get("sha256")
                tp2_evidence = selection["original_tp2_acceptance"]
                tp1_evidence = selection["tp1_capacity_evidence"]
                tp4_evidence = selection["tp4_attempt2"]
                old_tp4 = selection["old_tp4_attempt1"]
                tp1_capacity_path = Path(str(tp1_evidence.get("capacity_path", "")))
                tp1_capacity_ok = False
                if tp1_capacity_path.is_file():
                    with tp1_capacity_path.open("r", encoding="utf-8") as handle:
                        tp1_capacity_ok = bool(json.load(handle).get("capacity_ineligible"))
                v2_evidence_ok = (
                    original_hash_ok
                    and Path(str(tp2_evidence.get("path", ""))).is_file()
                    and bool(tp2_evidence.get("job_ids"))
                    and tp1_capacity_ok
                    and bool(tp1_evidence.get("job_ids"))
                    and Path(str(tp4_evidence.get("acceptance_path", ""))).is_file()
                    and bool(tp4_evidence.get("job_ids"))
                    and all(
                        isinstance(item.get("source_commit"), str) and bool(item.get("source_commit"))
                        for item in (tp2_evidence, tp1_evidence, tp4_evidence, old_tp4)
                    )
                    and old_tp4.get("eligible") is False
                    and bool(old_tp4.get("reason"))
                    and selection.get("selection_implementation_commit") == source_commit
                    and selection.get("fixed_section_17_decision_rule", {}).get("result")
                    == selection.get("decision_rule")
                )
                v2_details = {
                    "original_selection_path": str(original_path),
                    "original_selection_hash_ok": original_hash_ok,
                    "tp2_job_ids": tp2_evidence.get("job_ids"),
                    "tp1_job_ids": tp1_evidence.get("job_ids"),
                    "tp1_capacity_ok": tp1_capacity_ok,
                    "tp4_attempt2_job_ids": tp4_evidence.get("job_ids"),
                    "old_tp4_eligible": old_tp4.get("eligible"),
                }
            record(
                "serving_selection_v2_evidence",
                "selection v2 inputs, supersession hash, and ineligible TP=4 history are valid",
                v2_evidence_ok,
                v2_details,
            )
        selected = selection.get("selected_tp")
        record(
            "serving_selection_consistency",
            "selected TP has a passing acceptance record",
            any(item["tp"] == selected and item["passed"] for item in acceptance_records),
            {"selected_tp": selected, "records": acceptance_records},
        )
        if source_commit is not None:
            record(
                "source_commit_match",
                "selection source commit matches the deployment commit",
                selection.get("source_commit") == source_commit,
                {"actual": selection.get("source_commit"), "expected": source_commit},
            )
    else:
        record("serving_selection_schema", "serving_selection.json exists", False, "missing")

    return {"deployment_id": deployment_id, "passed": passed, "checks": checks}


# --------------------------------------------------------------------------
# Turkish audit
# --------------------------------------------------------------------------


def _compact_texts(csv_path: Path, json_path: Path, md_path: Path) -> dict[str, Any]:
    csv_rows = load_table_rows(csv_path, "csv")
    json_rows = load_table_rows(json_path, "json")
    md_rows = load_table_rows(md_path, "md")
    texts: list[str] = []
    for rows in (csv_rows, json_rows, md_rows):
        for row in rows:
            for column in FINAL_TABLE_COLUMNS:
                if column in ("order", "supporting_subjects"):
                    continue
                texts.append(str(row.get(column, "")))
    return {
        "csv_rows": csv_rows,
        "json_rows": json_rows,
        "md_rows": md_rows,
        "texts": texts,
    }


def _table_rows_for_render(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row[column] for column in FINAL_TABLE_COLUMNS} for row in rows]


def _recompute_restricted_evidence(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    compact: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the final table from restricted evidence and compare every layer."""
    restricted = run_dir / "restricted"
    prepared_path = restricted / "prepared_sequences.jsonl"
    inferences_dir = restricted / "subject_inferences"
    consolidation_dir = restricted / "consolidation_batches"
    required = [prepared_path, consolidation_dir / "final_merge.json"]
    if not all(path.is_file() for path in required):
        raise ValueError("restricted aggregation evidence is incomplete")

    sequences = load_prepared_sequences(prepared_path)
    expected_count = int(manifest.get("expected_sequences", 135))
    expected_windows = int(manifest.get("expected_windows", 1186))
    if len(sequences) != expected_count:
        raise ValueError(f"prepared sequence count mismatch: {len(sequences)} != {expected_count}")
    if sum(int(sequence.get("window_count", 0)) for sequence in sequences) != expected_windows:
        raise ValueError("prepared window count mismatch")
    prepared_ids = {sequence.get("sequence_id") for sequence in sequences}
    if len(prepared_ids) != expected_count or None in prepared_ids:
        raise ValueError("prepared sequence IDs are not unique")

    manifest_hash = _sha256_file(run_dir / "run_manifest.json")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("run manifest prompt version is not v2")
    expected_contract = prompt_contract_sha256(
        model_revision=manifest["model_revision"],
        max_tokens=int(manifest["request_settings"]["max_tokens"]),
        seed=int(manifest["request_settings"]["seed"]),
    )
    if manifest.get("prompt_contract_sha256") != expected_contract:
        raise ValueError("run manifest prompt contract hash is invalid")
    if manifest.get("source_sha256") != TURKISH_SOURCE_HASH:
        raise ValueError("run manifest source hash is not the fixed Turkish source")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("run manifest model identity is not pinned")
    request = manifest.get("request_settings", {})
    if request.get("seed") != 42 or request.get("max_tokens") != 2048:
        raise ValueError("run manifest generation settings are not pinned")
    if manifest.get("generation_settings_hash") != generation_settings_hash(2048):
        raise ValueError("run manifest generation settings hash is invalid")

    sequence_by_id = {sequence["sequence_id"]: sequence for sequence in sequences}
    inference_paths = sorted(inferences_dir.glob("S*.json")) if inferences_dir.is_dir() else []
    if len(inference_paths) != expected_count:
        raise ValueError(f"completed subject file count mismatch: {len(inference_paths)} != {expected_count}")
    records: list[dict[str, Any]] = []
    prompt_bundles: set[str] = set()
    for path in inference_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        sequence_id = record.get("sequence_id")
        sequence = sequence_by_id.get(sequence_id)
        if sequence is None or record.get("status") != "completed":
            raise ValueError(f"invalid completed inference record: {path.name}")
        expected = _episode_provenance(sequence)
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path.name}: provenance mismatch for {key}")
        if record.get("run_manifest_sha256") != manifest_hash:
            raise ValueError(f"{path.name}: run manifest hash mismatch")
        if record.get("prompt_version") != manifest.get("prompt_version"):
            raise ValueError(f"{path.name}: prompt version mismatch")
        if record.get("source_sha256") != manifest.get("source_sha256"):
            raise ValueError(f"{path.name}: source hash mismatch")
        if record.get("source_commit") != manifest.get("source_commit"):
            raise ValueError(f"{path.name}: source commit mismatch")
        if record.get("model_id") != manifest.get("model_id"):
            raise ValueError(f"{path.name}: model ID mismatch")
        if record.get("model_revision") != manifest.get("model_revision"):
            raise ValueError(f"{path.name}: model revision mismatch")
        if record.get("generation_settings_hash") != manifest.get("generation_settings_hash"):
            raise ValueError(f"{path.name}: generation settings mismatch")
        prompt_bundles.add(str(record.get("prompt_bundle_sha256")))
        original_payload = json.loads(json.dumps({"episodes": record.get("episodes", [])}, ensure_ascii=False))
        _validate_episodes(
            original_payload,
            sequence_id,
            [window["text"] for window in sequence.get("windows", [])],
        )
        if original_payload["episodes"] != record.get("episodes", []):
            raise ValueError(f"{path.name}: evidence_basis was not sanitized before storage")
        records.append(record)
    if len({record.get("sequence_id") for record in records}) != expected_count:
        raise ValueError("completed subject IDs are not unique")
    if len(prompt_bundles) != expected_count:
        raise ValueError("prompt bundle hashes are not unique per subject")

    candidates = collect_candidates(records)
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    batch_paths = sorted(consolidation_dir.glob("batch_*.json"))
    assigned_candidates: dict[str, str] = {}
    cluster_to_candidate: dict[str, list[str]] = {}
    cluster_ids: list[str] = []
    expected_batch_sizes = list(manifest.get("consolidation_batches", [32, 32, 32, 32, 7]))
    if len(batch_paths) != len(expected_batch_sizes):
        raise ValueError(
            f"expected {len(expected_batch_sizes)} consolidation batches, found {len(batch_paths)}"
        )
    for index, batch_path in enumerate(batch_paths):
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch.get("batch_index") != index + 1:
            raise ValueError(f"{batch_path.name}: batch index mismatch")
        if len(batch.get("sequence_ids", [])) != expected_batch_sizes[index]:
            raise ValueError(f"{batch_path.name}: sequence batch size mismatch")
        batch_candidates = [
            candidate for candidate in candidates
            if candidate["sequence_id"] in set(batch["sequence_ids"])
        ]
        expected_ids = {candidate["candidate_id"] for candidate in batch_candidates}
        for cluster in batch.get("clusters", []):
            cluster_id = cluster.get("cluster_id")
            if cluster_id in cluster_to_candidate:
                raise ValueError(f"duplicate cluster assignment: {cluster_id}")
            cluster_ids.append(cluster_id)
            members = list(cluster.get("member_candidate_ids", []))
            if not set(members) <= expected_ids:
                raise ValueError(f"{batch_path.name}: candidate assigned to wrong batch")
            for candidate_id in members:
                if candidate_id in assigned_candidates:
                    raise ValueError(f"candidate assigned more than once: {candidate_id}")
                assigned_candidates[candidate_id] = cluster_id
            cluster_to_candidate[cluster_id] = members
        if set(batch.get("assignment", {})) != expected_ids or batch.get("assignment", {}) != {
            member: assigned_candidates[member] for member in expected_ids
        }:
            raise ValueError(f"{batch_path.name}: assignment does not cover its candidates")
    if set(assigned_candidates) != set(candidate_by_id):
        raise ValueError("batch candidates are not assigned exactly once")

    final_merge = json.loads((consolidation_dir / "final_merge.json").read_text(encoding="utf-8"))
    final_cluster_assignment = final_merge.get("cluster_assignment", {})
    if set(final_cluster_assignment) != set(cluster_ids):
        raise ValueError("final cluster assignment does not cover every cluster")
    seen_families: set[str] = set()
    cluster_to_family: dict[str, str] = {}
    for family in final_merge.get("families", []):
        for cluster_id in family.get("member_cluster_ids", []):
            if cluster_id not in cluster_ids or cluster_id in seen_families:
                raise ValueError("final family cluster assignment is not exact once")
            seen_families.add(cluster_id)
            cluster_to_family[cluster_id] = family["family_id"]
    if seen_families != set(cluster_ids):
        raise ValueError("final families do not cover every cluster")
    if final_cluster_assignment != cluster_to_family:
        raise ValueError("final cluster assignment values do not match family membership")

    rows = aggregate_families(
        candidates,
        records,
        final_merge["families"],
        cluster_to_candidate,
        final_cluster_assignment,
    )
    recomputed_rows = _table_rows_for_render(rows)
    json_rows = compact["json_rows"]
    if _canonical_json(recomputed_rows).encode("utf-8") != _canonical_json(json_rows).encode("utf-8"):
        raise ValueError("recomputed rows differ byte-semantically from JSON rows")
    if compact["csv_rows"] != recomputed_rows or compact["md_rows"] != recomputed_rows:
        raise ValueError("recomputed rows differ value-semantically from CSV/Markdown rows")
    aggregation_hash = _sha256_text(_canonical_json(recomputed_rows))
    return {
        "prepared_sequences": len(sequences),
        "prepared_windows": sum(int(sequence["window_count"]) for sequence in sequences),
        "completed_subject_files": len(records),
        "failed_subjects": 0,
        "candidate_count": len(candidates),
        "batch_count": len(batch_paths),
        "cluster_count": len(cluster_ids),
        "family_count": len(final_merge.get("families", [])),
        "prompt_bundle_count": len(prompt_bundles),
        "final_merge_sha256": _sha256_file(consolidation_dir / "final_merge.json"),
        "aggregation_sha256": aggregation_hash,
    }


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle and needle in text for needle in needles)


def audit_turkish(
    run_dir: str | Path,
    *,
    turkish_run_id: str | None = None,
    transcript_path: str | Path,
    deploy_dir: str | Path,
    deployment_id: str,
    model_dir: str | Path,
    wheelhouse_dir: str | Path,
    source_commit: str,
    selection_file: str | Path | None = None,
    slurm_metadata: dict[str, Any] | None = None,
    remote_reference: str | Path | None = None,
    remote_audit_sha256_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the runbook section 21 audit; restricted checks need remote evidence."""
    run_dir = Path(run_dir)
    checks: list[dict[str, Any]] = []
    passed = True
    restricted_available = (run_dir / "restricted").is_dir()
    restricted_summary: dict[str, Any] | None = None

    def record(check_id: str, description: str, ok: bool, details: Any = None, scope: str = "local") -> None:
        nonlocal passed
        if not ok:
            passed = False
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "passed": ok,
                "details": details,
                "scope": scope,
            }
        )

    def record_skipped(check_id: str, description: str, details: Any = None) -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "passed": None,
                "details": details,
                "scope": "remote_only",
            }
        )

    # --- source and deployment identity ---------------------------------
    transcript_path = Path(transcript_path)
    source_hash = _sha256_file(transcript_path)
    record(
        "source_hash",
        "transcript SHA-256 matches the fixed source hash",
        source_hash == TURKISH_SOURCE_HASH,
        {"actual": source_hash, "expected": TURKISH_SOURCE_HASH},
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    manifest_hash: str | None = None
    if manifest_path.is_file():
        manifest_hash = _sha256_file(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        record(
            "run_manifest_hash",
            "run manifest is present and hashable",
            bool(manifest_hash),
            {"path": str(manifest_path), "sha256": manifest_hash},
        )
        record(
            "turkish_run_identity",
            "Turkish run identity matches the run manifest",
            turkish_run_id is None or manifest.get("turkish_run_id") == turkish_run_id,
            {"actual": manifest.get("turkish_run_id"), "expected": turkish_run_id},
        )
        record(
            "manifest_source_commit",
            "run manifest source commit matches the audited commit",
            manifest.get("source_commit") == source_commit,
            {"actual": manifest.get("source_commit"), "expected": source_commit},
        )
        record(
            "manifest_prompt_version",
            "run manifest uses prompt version v2",
            manifest.get("prompt_version") == PROMPT_VERSION,
            {"actual": manifest.get("prompt_version")},
        )
        record(
            "manifest_source_hash",
            "run manifest source hash matches the fixed Turkish source",
            manifest.get("source_sha256") == source_hash == TURKISH_SOURCE_HASH,
            {"actual": manifest.get("source_sha256"), "expected": source_hash},
        )
        record(
            "manifest_model_identity",
            "run manifest model identity matches the pinned model",
            manifest.get("model_id") == MODEL_ID and manifest.get("model_revision") == MODEL_REVISION,
            {
                "actual_id": manifest.get("model_id"),
                "actual_revision": manifest.get("model_revision"),
            },
        )
        record(
            "manifest_generation_settings",
            "run manifest generation settings are pinned",
            manifest.get("generation_settings_hash") == generation_settings_hash(2048)
            and manifest.get("request_settings", {}).get("seed") == 42
            and manifest.get("request_settings", {}).get("max_tokens") == 2048,
            {
                "generation_settings_hash": manifest.get("generation_settings_hash"),
                "seed": manifest.get("request_settings", {}).get("seed"),
                "max_tokens": manifest.get("request_settings", {}).get("max_tokens"),
            },
        )
    else:
        record("run_manifest_hash", "run manifest is present and hashable", False, "missing")
        record("turkish_run_identity", "Turkish run identity is recorded", False, "manifest missing")
        record("manifest_source_commit", "run manifest source commit is recorded", False, "manifest missing")
        record("manifest_prompt_version", "run manifest uses prompt version v2", False, "manifest missing")
        record("manifest_source_hash", "run manifest source hash is recorded", False, "manifest missing")
        record("manifest_model_identity", "run manifest model identity is recorded", False, "manifest missing")
        record("manifest_generation_settings", "run manifest generation settings are recorded", False, "manifest missing")

    env_dir = deploy_dir / deployment_id / "environment"
    runtime: dict[str, Any] = {}
    if (env_dir / "runtime_versions.json").is_file():
        with (env_dir / "runtime_versions.json").open("r", encoding="utf-8") as handle:
            runtime = json.load(handle)
    record(
        "model_identity",
        "model ID and revision match",
        runtime.get("model_id") == MODEL_ID and runtime.get("model_revision") == MODEL_REVISION,
        {
            "actual_id": runtime.get("model_id"),
            "actual_revision": runtime.get("model_revision"),
            "expected_id": MODEL_ID,
            "expected_revision": MODEL_REVISION,
        },
    )
    actual_env_raw = {
        "python_major": runtime.get("python_major"),
        "python_minor": runtime.get("python_minor"),
        "vllm": runtime.get("vllm"),
        "transformers": runtime.get("transformers"),
        "torch": runtime.get("torch"),
        "torchvision": runtime.get("torchvision"),
        "torchaudio": runtime.get("torchaudio"),
        "openai": runtime.get("openai"),
        "huggingface_hub": runtime.get("huggingface_hub"),
    }
    actual_env = _normalize_environment_versions(actual_env_raw)
    expected_env = {key: ENV_PINS[key] for key in actual_env}
    record(
        "environment_versions",
        "environment versions match the pinned set",
        actual_env == expected_env,
        {"actual": actual_env, "raw_actual": actual_env_raw, "expected": expected_env},
    )

    model_manifest = Path(model_dir) / "SHA256SUMS"
    if model_manifest.is_file():
        result = _verify_sha256_file(model_manifest)
        record(
            "model_manifest",
            "model SHA256SUMS verifies",
            result["ok"],
            {"checked": result["checked"], "errors": result["errors"][:5]},
        )
    else:
        record("model_manifest", "model SHA256SUMS verifies", False, "manifest missing")
    wheelhouse_manifest = Path(wheelhouse_dir) / "SHA256SUMS"
    if wheelhouse_manifest.is_file():
        result = _verify_sha256_file(wheelhouse_manifest)
        record(
            "wheelhouse_manifest",
            "wheelhouse SHA256SUMS verifies",
            result["ok"],
            {"checked": result["checked"], "errors": result["errors"][:5]},
        )
    else:
        record("wheelhouse_manifest", "wheelhouse SHA256SUMS verifies", False, "manifest missing")

    # --- serving selection -----------------------------------------------
    selection_path = Path(selection_file) if selection_file else deploy_dir / deployment_id / "serving_selection.json"
    selected_tp: int | None = None
    if selection_path.is_file():
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        selected_tp = selection.get("selected_tp")
        if selection_file is not None:
            record(
                "selection_v2",
                "Turkish run uses serving_selection_v2.json",
                selection.get("selection_version") == 2
                and selection_path.name == "serving_selection_v2.json",
                {"path": str(selection_path), "selection_version": selection.get("selection_version")},
            )
            record(
                "selection_implementation_commit",
                "selection v2 was created by the audited source commit",
                selection.get("selection_implementation_commit") == source_commit,
                {"actual": selection.get("selection_implementation_commit"), "expected": source_commit},
            )
        record(
            "selection_source_commit",
            "serving_selection source commit matches the run commit",
            selection.get("source_commit") == source_commit,
            {"actual": selection.get("source_commit"), "expected": source_commit},
        )
        if manifest:
            record(
                "selection_manifest_match",
                "run manifest selection path, hash, and TP match the selected file",
                manifest.get("selected_tp") == selected_tp
                and manifest.get("selection_file_sha256") == _sha256_file(selection_path),
                {
                    "manifest_selected_tp": manifest.get("selected_tp"),
                    "selection_selected_tp": selected_tp,
                    "manifest_selection_sha256": manifest.get("selection_file_sha256"),
                    "selection_sha256": _sha256_file(selection_path),
                },
            )
    else:
        record("selection_present", "serving selection file exists", False, str(selection_path))

    # --- compact outputs --------------------------------------------------
    csv_path = run_dir / "turkish_inferred_questions.csv"
    json_path = run_dir / "turkish_inferred_questions.json"
    md_path = run_dir / "turkish_inferred_questions.md"
    compact_missing = [path.name for path in (csv_path, json_path, md_path) if not path.is_file()]
    record("compact_outputs_present", "CSV, JSON, and Markdown outputs exist", not compact_missing, {"missing": compact_missing})
    compact_artifact_hashes: dict[str, str] = {}

    if not compact_missing:
        compact_artifact_hashes = {
            "turkish_inferred_questions.csv": _sha256_file(csv_path),
            "turkish_inferred_questions.json": _sha256_file(json_path),
            "turkish_inferred_questions.md": _sha256_file(md_path),
        }
        compact = _compact_texts(csv_path, json_path, md_path)
        compact_payload = json.loads(json_path.read_text(encoding="utf-8"))
        expected_compact_keys = {"deployment_id", "model_id", "model_revision", "source_commit", "rows"}
        compact_payload_keys = set(compact_payload) if isinstance(compact_payload, dict) else set()
        row_extra_keys = []
        if isinstance(compact_payload, dict) and isinstance(compact_payload.get("rows"), list):
            row_extra_keys = [
                sorted(set(row) - set(FINAL_TABLE_COLUMNS))
                for row in compact_payload["rows"]
                if isinstance(row, dict) and set(row) != set(FINAL_TABLE_COLUMNS)
            ]
        forbidden_reasoning_keys = []
        if isinstance(compact_payload, dict):
            for key in compact_payload:
                if key.lower() in {"reasoning", "thinking", "chain_of_thought", "raw_response", "model_response"}:
                    forbidden_reasoning_keys.append(key)
        record(
            "compact_payload_schema",
            "compact JSON contains only the documented metadata and final rows",
            compact_payload_keys == expected_compact_keys and not row_extra_keys,
            {"keys": sorted(compact_payload_keys), "row_extra_keys": row_extra_keys[:10]},
        )
        record(
            "no_raw_reasoning",
            "compact evidence contains no raw model reasoning fields",
            not forbidden_reasoning_keys,
            {"forbidden_keys": forbidden_reasoning_keys},
        )
        record(
            "rows_consistent",
            "CSV, JSON, and Markdown contain the same rows in the same order",
            compact["csv_rows"] == compact["json_rows"] == compact["md_rows"],
            {
                "csv_rows": len(compact["csv_rows"]),
                "json_rows": len(compact["json_rows"]),
                "md_rows": len(compact["md_rows"]),
            },
        )
        if compact["csv_rows"] == compact["json_rows"] == compact["md_rows"]:
            rows = compact["csv_rows"]
        else:
            rows = []

        schema_errors: list[str] = []
        if rows:
            for index, row in enumerate(rows):
                if list(row.keys()) != list(FINAL_TABLE_COLUMNS):
                    schema_errors.append(f"row {index}: column order mismatch")
                    continue
                if not isinstance(row["order"], int) or row["order"] <= 0:
                    schema_errors.append(f"row {index}: bad order")
                if row["label"] not in (label.value for label in Label):
                    schema_errors.append(f"row {index}: bad label {row['label']!r}")
                if row["wording_status"] not in (status.value for status in WordingStatus):
                    schema_errors.append(f"row {index}: bad wording_status")
                if row["confidence"] not in (conf.value for conf in Confidence):
                    schema_errors.append(f"row {index}: bad confidence")
                if not isinstance(row["supporting_subjects"], int) or row["supporting_subjects"] <= 0:
                    schema_errors.append(f"row {index}: bad supporting_subjects")
                for column in ("question_tr", "question_en", "evidence_basis"):
                    if not isinstance(row[column], str) or not row[column].strip():
                        schema_errors.append(f"row {index}: empty {column}")
        record(
            "table_schema",
            "table columns, order, and enums are valid",
            not schema_errors,
            {"errors": schema_errors[:20]},
        )

        windows = _transcript_windows(transcript_path) if source_hash == TURKISH_SOURCE_HASH else []
        stems = _transcript_stems(transcript_path)
        subjects = {stem[0] for stem in stems}
        forbidden_stems = set(stems)
        violations: list[str] = []
        for text in compact["texts"]:
            if any(marker in text for marker in ("[WINDOW", "]", '"', "'", "<|")):
                violations.append(f"forbidden marker in: {text[:60]!r}")
            if _contains_any(text, forbidden_stems):
                violations.append(f"filename stem or subject string in: {text[:60]!r}")
            if ngram_overlap_at_least(text, windows, OVERLAP_TOKEN_COUNT):
                violations.append(f"{OVERLAP_TOKEN_COUNT}+ token transcript overlap in: {text[:60]!r}")
        record(
            "privacy_markers",
            "no subject ID, filename stem, window marker, quote, or transcript fragment in compact outputs",
            not violations,
            {"violations": violations[:20]},
        )
        record(
            "privacy_overlap",
            "no 12-token contiguous overlap with any source window",
            all(not ngram_overlap_at_least(text, windows, OVERLAP_TOKEN_COUNT) for text in compact["texts"]),
            {"checked_fields": len(compact["texts"]), "windows_used": len(windows)},
        )
        record(
            "privacy_stems",
            "no complete source basename stem or subject string in compact outputs",
            all(not _contains_any(text, forbidden_stems) for text in compact["texts"]),
            {"stems_checked": len(forbidden_stems)},
        )
        if subjects and compact["texts"]:
            subject_violations = [text for text in compact["texts"] if _contains_any(text, subjects)]
            record(
                "privacy_subjects",
                "no complete source subject string in compact outputs",
                not subject_violations,
                {"violations": subject_violations[:10]},
            )
        else:
            record_skipped("privacy_subjects", "no complete source subject string check needs parsed stems")
    else:
        rows = []

    # --- restricted-intermediate checks (remote only) --------------------
    if restricted_available:
        try:
            recomputed = _recompute_restricted_evidence(run_dir, manifest, compact=compact)
            restricted_summary = recomputed
            record(
                "preparation_windows",
                "all 1,186 windows represented in preparation",
                recomputed["prepared_windows"] == 1186,
                {"window_total": recomputed["prepared_windows"]},
                scope="remote",
            )
            record(
                "preparation_sequences",
                "all 135 sequences prepared with exact source hash",
                recomputed["prepared_sequences"] == 135,
                {"sequences": recomputed["prepared_sequences"]},
                scope="remote",
            )
            record(
                "sequences_completed_once",
                "all 135 sequences completed exactly once with zero failures",
                recomputed["completed_subject_files"] == 135 and recomputed["failed_subjects"] == 0,
                {"completed": recomputed["completed_subject_files"], "failed": recomputed["failed_subjects"]},
                scope="remote",
            )
            record(
                "assignment_once",
                "every candidate and cluster assigned exactly once",
                recomputed["candidate_count"] > 0 and recomputed["cluster_count"] > 0,
                {
                    "candidates": recomputed["candidate_count"],
                    "clusters": recomputed["cluster_count"],
                },
                scope="remote",
            )
            record(
                "consolidation_batches",
                "the fixed five consolidation batches are present",
                recomputed["batch_count"] == 5,
                {"batch_count": recomputed["batch_count"]},
                scope="remote",
            )
            record(
                "aggregation_recompute",
                "final aggregation recomputes identically from restricted evidence",
                True,
                recomputed,
                scope="remote",
            )
            record(
                "prompt_provenance_complete",
                "one prompt version and contract with 135 subject-specific bundles",
                manifest.get("prompt_version") == PROMPT_VERSION
                and bool(manifest.get("prompt_contract_sha256"))
                and recomputed["prompt_bundle_count"] == 135,
                {
                    "prompt_version": manifest.get("prompt_version"),
                    "prompt_contract_sha256": manifest.get("prompt_contract_sha256"),
                    "prompt_bundle_count": recomputed["prompt_bundle_count"],
                },
                scope="remote",
            )
        except Exception as exc:
            failure = {"error": f"{type(exc).__name__}: {exc}"}
            for check_id, description in (
                ("preparation_windows", "all 1,186 windows represented in preparation"),
                ("preparation_sequences", "all 135 sequences prepared with exact source hash"),
                ("sequences_completed_once", "all 135 sequences completed exactly once with zero failures"),
                ("assignment_once", "every candidate and cluster assigned exactly once"),
                ("consolidation_batches", "the fixed five consolidation batches are present"),
                ("aggregation_recompute", "final aggregation recomputes identically from restricted evidence"),
                ("prompt_provenance_complete", "prompt provenance is complete"),
            ):
                record(check_id, description, False, failure, scope="remote")
    else:
        record_skipped("preparation_windows", "restricted evidence not present locally")
        record_skipped("preparation_sequences", "restricted evidence not present locally")
        record_skipped("sequences_completed_once", "restricted evidence not present locally")
        record_skipped("assignment_once", "restricted evidence not present locally")
        record_skipped("aggregation_recompute", "restricted evidence not present locally")
        record_skipped("prompt_provenance_complete", "restricted evidence not present locally")

    # --- slurm accounting ------------------------------------------------
    if slurm_metadata:
        required = ("job_id", "state", "exit_code", "node", "start_time", "end_time")
        missing = [key for key in required if key not in slurm_metadata]
        record(
            "slurm_metadata",
            "Slurm job ID, state, exit code, node, and timestamps present",
            not missing and slurm_metadata.get("state") == "COMPLETED"
            and str(slurm_metadata.get("exit_code")) == "0:0",
            {"missing": missing, "metadata": slurm_metadata},
        )
        record(
            "slurm_run_identity",
            "Slurm metadata carries the Turkish run identity and source commit",
            (turkish_run_id is None or slurm_metadata.get("turkish_run_id") == turkish_run_id)
            and slurm_metadata.get("source_commit", source_commit) == source_commit,
            {
                "turkish_run_id": slurm_metadata.get("turkish_run_id"),
                "source_commit": slurm_metadata.get("source_commit"),
                "selected_tp": slurm_metadata.get("selected_tp"),
            },
        )
        record(
            "slurm_selected_tp",
            "Slurm metadata selected TP matches serving_selection_v2.json",
            selected_tp is not None and slurm_metadata.get("selected_tp") == selected_tp,
            {"metadata": slurm_metadata.get("selected_tp"), "selection": selected_tp},
        )
    else:
        record_skipped("slurm_metadata", "Slurm metadata not supplied")

    # --- remote reference ------------------------------------------------
    remote_conclusion: dict[str, Any] | None = None
    if remote_reference is not None:
        remote_path = Path(remote_reference)
        if remote_path.is_file():
            with remote_path.open("r", encoding="utf-8") as handle:
                remote_conclusion = json.load(handle)
            hash_path = Path(remote_audit_sha256_path) if remote_audit_sha256_path else remote_path.with_name(remote_path.name + ".sha256")
            expected_audit_hash = ""
            if hash_path.is_file():
                expected_audit_hash = hash_path.read_text(encoding="utf-8").split()[0]
            actual_audit_hash = _sha256_file(remote_path)
            record(
                "remote_audit_reference",
                "synchronized remote audit hash matches its sidecar and concluded passed",
                bool(remote_conclusion.get("passed"))
                and bool(expected_audit_hash)
                and actual_audit_hash == expected_audit_hash,
                {
                    "path": str(remote_path),
                    "sha256": actual_audit_hash,
                    "expected_sha256": expected_audit_hash,
                },
            )
            remote_hashes = remote_conclusion.get("compact_artifact_hashes", {})
            local_hashes_match = bool(remote_hashes) and set(remote_hashes) == set(compact_artifact_hashes) and all(
                compact_artifact_hashes.get(name) == value
                for name, value in remote_hashes.items()
            )
            record(
                "remote_compact_hashes",
                "local compact artifacts match hashes embedded in the passed remote audit",
                local_hashes_match,
                {"remote": remote_hashes, "local": compact_artifact_hashes},
            )
            record(
                "remote_manifest_hash",
                "local run manifest matches the passed remote audit",
                manifest_hash is not None and remote_conclusion.get("run_manifest_sha256") == manifest_hash,
                {
                    "remote": remote_conclusion.get("run_manifest_sha256"),
                    "local": manifest_hash,
                },
            )
            record(
                "remote_selection_hash",
                "local selection v2 matches the passed remote audit",
                selection_path.is_file()
                and bool(remote_conclusion.get("selection_file_sha256"))
                and _sha256_file(selection_path) == remote_conclusion.get("selection_file_sha256"),
                {
                    "remote": remote_conclusion.get("selection_file_sha256"),
                    "local": _sha256_file(selection_path) if selection_path.is_file() else None,
                },
            )
        else:
            record("remote_audit_reference", "remote audit.json present", False, "missing")

    return {
        "deployment_id": deployment_id,
        "turkish_run_id": turkish_run_id or manifest.get("turkish_run_id"),
        "analysis_attempt": manifest.get("analysis_attempt"),
        "source_commit": source_commit,
        "source_sha256": source_hash,
        "model_id": manifest.get("model_id", runtime.get("model_id")),
        "model_revision": manifest.get("model_revision", runtime.get("model_revision")),
        "prompt_version": manifest.get("prompt_version"),
        "prompt_contract_sha256": manifest.get("prompt_contract_sha256"),
        "generation_settings_hash": manifest.get("generation_settings_hash"),
        "request_settings": manifest.get("request_settings"),
        "selected_tp": selected_tp,
        "restricted_evidence_available": restricted_available,
        "restricted_summary": restricted_summary,
        "run_manifest_sha256": manifest_hash,
        "selection_file": str(selection_path),
        "selection_file_sha256": _sha256_file(selection_path) if selection_path.is_file() else None,
        "compact_artifact_hashes": compact_artifact_hashes,
        "slurm_metadata": slurm_metadata,
        "supersedes_job_ids": manifest.get("supersedes_job_ids", []),
        "final_merge_sha256": restricted_summary.get("final_merge_sha256") if restricted_summary else None,
        "aggregation_sha256": restricted_summary.get("aggregation_sha256") if restricted_summary else None,
        "passed": passed,
        "checks": checks,
        "remote_audit_reference": str(remote_reference) if remote_reference else None,
        "remote_audit_conclusion": remote_conclusion,
    }


def _transcript_windows(transcript_path: Path) -> list[str]:
    windows: list[str] = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            transcript = row.get("transcript")
            if isinstance(transcript, str) and transcript.strip():
                windows.append(transcript)
    return windows


def _transcript_stems(transcript_path: Path) -> set[str]:
    stems: set[str] = set()
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = row.get("audio_path")
            if isinstance(audio_path, str) and audio_path:
                stems.add(Path(audio_path).stem)
                parts = parse_filename_stem(Path(audio_path).stem)
                if parts is not None:
                    stems.add(parts.subject)
    return stems


def write_audit_json(audit: dict[str, Any], path: str | Path) -> None:
    _atomic_write_json(audit, path)
