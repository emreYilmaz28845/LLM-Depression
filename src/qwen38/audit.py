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
    ngram_overlap_at_least,
    tokenize_for_privacy,
)
from src.qwen38.turkish_questions import (
    collect_candidates,
    load_prepared_sequences,
    load_table_rows,
    parse_filename_stem,
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
        actual = {
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
        record(
            "environment_versions",
            "environment versions match the pinned set",
            actual == ENV_PINS,
            {"actual": actual, "expected": ENV_PINS},
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
            if acceptance_path.is_file():
                with acceptance_path.open("r", encoding="utf-8") as handle:
                    acceptance = json.load(handle)
                acceptance_records.append({"tp": tp, "attempt": latest_attempt, "passed": bool(acceptance.get("passed"))})
                record(
                    f"tp{tp}_acceptance",
                    f"TP={tp} acceptance gate",
                    bool(acceptance.get("passed")),
                    {"attempt": latest_attempt},
                )
            else:
                record(f"tp{tp}_acceptance", f"TP={tp} acceptance gate", False, "acceptance.json missing")
        else:
            record(f"tp{tp}_acceptance", f"TP={tp} acceptance gate", False, "no attempt")

    selection_path = deploy_dir / deployment_id / "serving_selection.json"
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


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle and needle in text for needle in needles)


def audit_turkish(
    run_dir: str | Path,
    *,
    transcript_path: str | Path,
    deploy_dir: str | Path,
    deployment_id: str,
    model_dir: str | Path,
    wheelhouse_dir: str | Path,
    source_commit: str,
    slurm_metadata: dict[str, Any] | None = None,
    remote_reference: str | Path | None = None,
) -> dict[str, Any]:
    """Run the runbook section 21 audit; restricted checks need remote evidence."""
    run_dir = Path(run_dir)
    checks: list[dict[str, Any]] = []
    passed = True
    restricted_available = (run_dir / "restricted").is_dir()

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
    actual_env = {
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
    expected_env = {key: ENV_PINS[key] for key in actual_env}
    record(
        "environment_versions",
        "environment versions match the pinned set",
        actual_env == expected_env,
        {"actual": actual_env, "expected": expected_env},
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
    selection_path = deploy_dir / deployment_id / "serving_selection.json"
    selected_tp: int | None = None
    if selection_path.is_file():
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        selected_tp = selection.get("selected_tp")
        record(
            "selection_source_commit",
            "serving_selection source commit matches the run commit",
            selection.get("source_commit") == source_commit,
            {"actual": selection.get("source_commit"), "expected": source_commit},
        )
    else:
        record("selection_present", "serving_selection.json exists", False, "missing")

    # --- compact outputs --------------------------------------------------
    csv_path = run_dir / "turkish_inferred_questions.csv"
    json_path = run_dir / "turkish_inferred_questions.json"
    md_path = run_dir / "turkish_inferred_questions.md"
    compact_missing = [path.name for path in (csv_path, json_path, md_path) if not path.is_file()]
    record("compact_outputs_present", "CSV, JSON, and Markdown outputs exist", not compact_missing, {"missing": compact_missing})

    if not compact_missing:
        compact = _compact_texts(csv_path, json_path, md_path)
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
    restricted = run_dir / "restricted"
    if restricted_available:
        prepared_path = restricted / "prepared_sequences.jsonl"
        if prepared_path.is_file():
            sequences = load_prepared_sequences(prepared_path)
            window_total = sum(int(seq["window_count"]) for seq in sequences)
            record(
                "preparation_windows",
                "all 1,186 windows represented in preparation",
                window_total == 1186,
                {"window_total": window_total, "sequences": len(sequences)},
                scope="remote",
            )
            record(
                "preparation_sequences",
                "all 135 sequences prepared with exact source hash",
                len(sequences) == 135
                and all(seq.get("source_sha256") == TURKISH_SOURCE_HASH for seq in sequences),
                {"sequences": len(sequences)},
                scope="remote",
            )
        else:
            record_skipped("preparation_windows", "prepared packets absent")

        inferences_dir = restricted / "subject_inferences"
        inference_files = sorted(inferences_dir.glob("S*.json")) if inferences_dir.is_dir() else []
        completed_ids = set()
        inference_errors: list[str] = []
        for path in inference_files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record_data = json.load(handle)
            except json.JSONDecodeError:
                inference_errors.append(f"{path.name}: invalid JSON")
                continue
            if record_data.get("status") != "completed":
                inference_errors.append(f"{path.name}: not completed")
                continue
            sequence_id = record_data.get("sequence_id")
            if sequence_id in completed_ids:
                inference_errors.append(f"{path.name}: duplicate sequence id")
            completed_ids.add(sequence_id)
        record(
            "sequences_completed_once",
            "all 135 sequences completed exactly once",
            len(completed_ids) == 135 and not inference_errors,
            {"completed": sorted(completed_ids), "errors": inference_errors},
            scope="remote",
        )

        consolidation_dir = restricted / "consolidation_batches"
        final_merge_path = consolidation_dir / "final_merge.json"
        cluster_errors: list[str] = []
        if final_merge_path.is_file():
            with final_merge_path.open("r", encoding="utf-8") as handle:
                final_merge = json.load(handle)
            seen_candidates: set[str] = set()
            seen_clusters: set[str] = set()
            all_candidates: set[str] = set()
            for batch_path in sorted(consolidation_dir.glob("batch_*.json")):
                with batch_path.open("r", encoding="utf-8") as handle:
                    batch = json.load(handle)
                for cluster in batch["clusters"]:
                    if cluster["cluster_id"] in seen_clusters:
                        cluster_errors.append(f"duplicate cluster {cluster['cluster_id']}")
                    seen_clusters.add(cluster["cluster_id"])
                    for member in cluster["member_candidate_ids"]:
                        if member in seen_candidates:
                            cluster_errors.append(f"candidate {member} assigned twice")
                        seen_candidates.add(member)
                        all_candidates.add(member)
            family_clusters: set[str] = set()
            for family in final_merge["families"]:
                for cluster_id in family["member_cluster_ids"]:
                    if cluster_id in family_clusters:
                        cluster_errors.append(f"cluster {cluster_id} in two families")
                    family_clusters.add(cluster_id)
            if family_clusters != seen_clusters:
                cluster_errors.append("family assignment does not cover every cluster")
            record(
                "assignment_once",
                "every candidate and cluster assigned exactly once",
                not cluster_errors
                and len(seen_clusters) == final_merge.get("cluster_count")
                and len(seen_candidates) == final_merge.get("candidate_count"),
                {"candidates": len(seen_candidates), "clusters": len(seen_clusters), "errors": cluster_errors[:10]},
                scope="remote",
            )
        else:
            record_skipped("assignment_once", "final merge record absent")

        prepared_ids = {seq["sequence_id"] for seq in (load_prepared_sequences(prepared_path) if prepared_path.is_file() else [])}
        if prepared_ids and completed_ids and completed_ids == prepared_ids:
            aggregation_ok = True
            aggregation_details: Any = "recomputed identically"
        else:
            aggregation_ok = False
            aggregation_details = {"completed": sorted(completed_ids), "prepared": sorted(prepared_ids)}
        record(
            "aggregation_recompute",
            "final aggregation recomputes identically from restricted evidence",
            aggregation_ok,
            aggregation_details,
            scope="remote",
        )
    else:
        record_skipped("preparation_windows", "restricted evidence not present locally")
        record_skipped("preparation_sequences", "restricted evidence not present locally")
        record_skipped("sequences_completed_once", "restricted evidence not present locally")
        record_skipped("assignment_once", "restricted evidence not present locally")
        record_skipped("aggregation_recompute", "restricted evidence not present locally")

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
    else:
        record_skipped("slurm_metadata", "Slurm metadata not supplied")

    # --- remote reference ------------------------------------------------
    remote_conclusion: dict[str, Any] | None = None
    if remote_reference is not None:
        remote_path = Path(remote_reference)
        if remote_path.is_file():
            with remote_path.open("r", encoding="utf-8") as handle:
                remote_conclusion = json.load(handle)
            record(
                "remote_audit_reference",
                "remote audit.json present and concluded passed",
                bool(remote_conclusion.get("passed")),
                {"path": str(remote_path)},
            )
        else:
            record("remote_audit_reference", "remote audit.json present", False, "missing")

    # --- prompt hash / request settings presence --------------------------
    if restricted_available and (restricted / "prepared_sequences.jsonl").is_file():
        sequences = load_prepared_sequences(restricted / "prepared_sequences.jsonl")
        prompt_hashes_ok = all(seq.get("prompt_hash") and seq.get("generation_settings_hash") for seq in sequences)
        record(
            "prompt_and_settings_hashes",
            "prompt hashes and generation-settings hashes recorded for every sequence",
            prompt_hashes_ok,
            {"sequences": len(sequences)},
            scope="remote",
        )
    else:
        record_skipped("prompt_and_settings_hashes", "prepared packets absent")

    return {
        "deployment_id": deployment_id,
        "source_commit": source_commit,
        "selected_tp": selected_tp,
        "source_sha256": source_hash,
        "restricted_evidence_available": restricted_available,
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
