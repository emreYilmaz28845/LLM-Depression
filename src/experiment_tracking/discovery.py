from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_sha256, read_json, read_jsonl, sha256_file

CHECKPOINT_LOCATIONS = ("best_model", "last_model")

EVAL_LOCATIONS = (
    "best_model/standalone_eval",
    "last_model/standalone_eval",
    "eval/best_checkpoint",
    "eval/best_validation",
)

BEST_EVAL_LOCATIONS = (
    "best_model/standalone_eval",
    "eval/best_checkpoint",
    "eval/best_validation",
)

LAST_EVAL_LOCATION = "last_model/standalone_eval"

RUN_ROOT = "run_root"
LOGS = "logs"

_FOLD_DIR_PATTERN = re.compile(r"^fold_[0-9]+$")

# Non-canonical trees inside a scan root that carry run_config.yaml files at
# misleading depths (audit snapshots, experiments layouts). Their modality and
# dataset cannot be derived by path, so discovery skips them; the legacy
# adapters remain responsible for documented non-canonical layouts.
_NON_CANONICAL_ROOTS = ("experiments", "audits")

# Run layouts: output_model/<modality>/<dataset>/<run>/fold_* and the newer
# output_model/<campaign>/<modality>/<dataset>/<run>/fold_*. Both parse with
# modality/dataset/run taken from the three parts before fold_*. Post-hoc
# head attempts live one level deeper: fold_* / <experiment_id> / with the
# run_config.yaml and sidecars inside the attempt directory.
_RUN_GLOBS = (
    "*/*/*/fold_*/run_config.yaml",
    "*/*/*/*/fold_*/run_config.yaml",
    "*/*/*/*/fold_*/*/run_config.yaml",
)


@dataclasses.dataclass(frozen=True)
class DiscoveredArtifact:
    relative_path: str
    artifact_type: str
    kind: str
    location: str
    sha256: str | None
    size_bytes: int | None
    parse_ok: bool | None
    json_content: Any = None


@dataclasses.dataclass(frozen=True)
class DiscoveredRun:
    scan_root: str
    modality: str
    dataset: str
    run_name: str
    fold: int
    fold_dir: str
    run_config_path: str
    run_config_file_sha256: str | None
    run_config_parse_ok: bool
    resolved_config: dict[str, Any] | None
    protocol: dict[str, Any] | None
    resolved_config_sha256: str | None
    artifacts: tuple[DiscoveredArtifact, ...]
    warnings: tuple[str, ...]


def _classify_logs_file(name: str) -> tuple[str, str] | None:
    if name == "training_history.json":
        return "training_history", "training_history"
    if name == "selected_checkpoint_selection_metrics.json":
        return "metrics", "selection_metrics"
    if name == "split_used.json":
        return "split", "split_used"
    if name == "audio_budget_audit_train.json":
        return "audit", "audio_budget_audit"
    if name == "sample_partition_counts.json":
        return "audit", "partition_counts"
    if name == "peak_gpu_memory.json":
        return "audit", "peak_gpu_memory"
    if name.endswith("_truncation.jsonl") or name.endswith("_truncation.json"):
        return "audit", "truncation"
    return None


def _classify_eval_file(name: str) -> tuple[str, str] | None:
    if name == "eval_config.yaml":
        return "run_config", "eval_config"
    if name == "confusion_matrix.json":
        return "metrics", "confusion_matrix"
    if name == "final_and_best_validation_metrics.json":
        return "metrics", "selection_metrics"
    if name.startswith("metrics_") and name.endswith(".json"):
        return "metrics", "metrics"
    if name == "metrics.json":
        return "metrics", "metrics"
    if name.startswith("predictions") and (name.endswith(".csv") or name.endswith(".jsonl")):
        lowered = name.lower()
        if "subject" in lowered:
            kind = "subject_predictions"
        elif "sample" in lowered:
            kind = "sample_predictions"
        elif "headline" in lowered:
            kind = "headline_predictions"
        else:
            kind = "predictions"
        return "predictions", kind
    if name == "subject_predictions.csv":
        return "predictions", "subject_predictions"
    return None


def _parse_content(path: Path) -> tuple[bool, Any]:
    try:
        if path.suffix == ".json":
            return True, read_json(path)
        if path.suffix == ".jsonl":
            return True, read_jsonl(path)
        return True, None
    except (ValueError, OSError):
        return False, None


def _discover_artifacts(fold_dir: Path) -> tuple[tuple[DiscoveredArtifact, ...], tuple[str, ...]]:
    artifacts: list[DiscoveredArtifact] = []
    warnings: list[str] = []

    def add(rel_path: Path, artifact_type: str, kind: str, location: str, parse: bool) -> None:
        full = fold_dir / rel_path
        try:
            sha256 = sha256_file(full)
            size_bytes = full.stat().st_size
        except OSError:
            warnings.append(f"unreadable artifact: {rel_path.as_posix()}")
            return
        parse_ok: bool | None = None
        content: Any = None
        if parse:
            parse_ok, content = _parse_content(full)
            if not parse_ok:
                warnings.append(f"unreadable JSON: {rel_path.as_posix()}")
        artifacts.append(
            DiscoveredArtifact(
                relative_path=rel_path.as_posix(),
                artifact_type=artifact_type,
                kind=kind,
                location=location,
                sha256=sha256,
                size_bytes=size_bytes,
                parse_ok=parse_ok,
                json_content=content,
            )
        )

    for checkpoint in CHECKPOINT_LOCATIONS:
        if (fold_dir / checkpoint).is_dir():
            artifacts.append(
                DiscoveredArtifact(
                    relative_path=checkpoint,
                    artifact_type="checkpoint",
                    kind="checkpoint_dir",
                    location=checkpoint,
                    sha256=None,
                    size_bytes=None,
                    parse_ok=None,
                )
            )

    for name in ("final_summary.json", "final_summary.csv", "final_summary_active.csv"):
        path = fold_dir / name
        if path.is_file():
            kind = "final_summary_active" if "active" in name else "final_summary"
            add(path.relative_to(fold_dir), "summary", kind, RUN_ROOT, parse=name.endswith(".json"))

    run_config_yaml = fold_dir / "run_config.yaml"
    if run_config_yaml.is_file():
        add(run_config_yaml.relative_to(fold_dir), "run_config", "run_config", RUN_ROOT, parse=False)

    best_vs_last = fold_dir / "best_vs_last_checkpoint_metrics.json"
    if best_vs_last.is_file():
        add(best_vs_last.relative_to(fold_dir), "metrics", "best_vs_last_metrics", RUN_ROOT, parse=True)

    logs_dir = fold_dir / "logs"
    if logs_dir.is_dir():
        for entry in sorted(logs_dir.iterdir()):
            if not entry.is_file():
                continue
            classified = _classify_logs_file(entry.name)
            if classified is None:
                continue
            artifact_type, kind = classified
            add(entry.relative_to(fold_dir), artifact_type, kind, LOGS, parse=entry.suffix in (".json", ".jsonl"))

    for location in EVAL_LOCATIONS:
        location_dir = fold_dir / location
        if not location_dir.is_dir():
            continue
        for entry in sorted(location_dir.rglob("*")):
            if not entry.is_file():
                continue
            classified = _classify_eval_file(entry.name)
            if classified is None:
                continue
            artifact_type, kind = classified
            add(entry.relative_to(fold_dir), artifact_type, kind, location, parse=entry.suffix in (".json", ".jsonl"))

    return tuple(artifacts), tuple(warnings)


def _discover_run(scan_root: Path, fold_dir: Path, *, parts_offset: int = 0) -> DiscoveredRun:
    warnings: list[str] = []
    relative = fold_dir.relative_to(scan_root)
    parts = relative.parts
    # Post-hoc head attempts add one level (fold_*/<experiment_id>); the
    # identity still comes from the three parts before the fold directory.
    modality, dataset, run_name = parts[-4 - parts_offset], parts[-3 - parts_offset], parts[-2 - parts_offset]
    fold = int(parts[-1 - parts_offset].split("_", 1)[1])
    run_config_path = fold_dir / "run_config.yaml"

    run_config_file_sha256: str | None = None
    run_config_parse_ok = False
    resolved_config: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    resolved_config_sha256: str | None = None
    if run_config_path.is_file():
        try:
            run_config_file_sha256 = sha256_file(run_config_path)
            data = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            warnings.append(f"run_config unreadable: {error}")
            data = None
        if isinstance(data, dict):
            run_config_parse_ok = True
            resolved_config = data.get("config") if isinstance(data.get("config"), dict) else None
            protocol = {key: value for key, value in data.items() if key != "config"}
            if resolved_config is not None:
                resolved_config_sha256 = canonical_sha256(resolved_config)
            else:
                warnings.append("run_config has no resolved config: section")
        else:
            warnings.append("run_config does not parse to an object")

    artifacts, artifact_warnings = _discover_artifacts(fold_dir)
    warnings.extend(artifact_warnings)

    return DiscoveredRun(
        scan_root=str(scan_root),
        modality=modality,
        dataset=dataset,
        run_name=run_name,
        fold=fold,
        fold_dir=str(fold_dir),
        run_config_path=str(run_config_path),
        run_config_file_sha256=run_config_file_sha256,
        run_config_parse_ok=run_config_parse_ok,
        resolved_config=resolved_config,
        protocol=protocol,
        resolved_config_sha256=resolved_config_sha256,
        artifacts=tuple(artifacts),
        warnings=tuple(warnings),
    )


def discover_runs(scan_root: str | Path) -> list[DiscoveredRun]:
    root = Path(scan_root)
    if not root.is_dir():
        raise ValueError(f"scan root is not a directory: {root}")
    runs: list[DiscoveredRun] = []
    seen: set[str] = set()
    for pattern in _RUN_GLOBS:
        for run_config_path in sorted(root.glob(pattern)):
            fold_dir = run_config_path.parent
            parts_offset = 0
            if _FOLD_DIR_PATTERN.fullmatch(fold_dir.name) is None:
                # Post-hoc head attempt: the run_config sits inside
                # fold_<n>/<experiment_id>/; the parent must be the fold dir.
                if _FOLD_DIR_PATTERN.fullmatch(fold_dir.parent.name) is None:
                    continue
                parts_offset = 1
            relative = fold_dir.relative_to(root)
            if relative.parts and relative.parts[0] in _NON_CANONICAL_ROOTS:
                continue
            if str(fold_dir) in seen:
                continue
            seen.add(str(fold_dir))
            runs.append(_discover_run(root, fold_dir, parts_offset=parts_offset))
    return runs


def discover_run_at(fold_dir: str | Path, scan_root: str | Path | None = None) -> DiscoveredRun:
    target = Path(fold_dir)
    if scan_root is None:
        scan_root = target.parent.parent.parent.parent
    return _discover_run(Path(scan_root), target)
