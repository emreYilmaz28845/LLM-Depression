from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from .canonical import read_json, sha256_file
from .discovery import DiscoveredRun, discover_run_at, discover_runs
from .qualification import STATUS_QUARANTINED_AMBIGUOUS, STATUS_REJECTED, qualify_run

_TRANSLATED_NAME_PATTERN = re.compile(r"(^|_)en(_|-)")

ADAPTER_ORDINARY = "ordinary"
ADAPTER_TRANSLATED = "translated"
ADAPTER_MERGED = "merged"
ADAPTER_HIDDEN_CLASSIFIER = "hidden_classifier"


@dataclasses.dataclass(frozen=True)
class AdapterCandidate:
    adapter_type: str
    dataset: str | None
    modality: str | None
    run_name: str
    fold: int | None
    root: str
    evidence_paths: tuple[str, ...]
    metadata: dict[str, Any]
    quarantine_reasons: tuple[str, ...] = ()


def _candidate(
    adapter_type: str,
    run_name: str,
    fold: int | None,
    root: Path,
    evidence_paths: list[Path],
    metadata: dict[str, Any],
    quarantine_reasons: list[str] | None = None,
) -> AdapterCandidate:
    return AdapterCandidate(
        adapter_type=adapter_type,
        dataset=metadata.get("dataset"),
        modality=metadata.get("modality"),
        run_name=run_name,
        fold=fold,
        root=str(root),
        evidence_paths=tuple(str(path) for path in sorted(evidence_paths)),
        metadata=metadata,
        quarantine_reasons=tuple(quarantine_reasons or []),
    )


def is_translated_run_name(run_name: str) -> bool:
    return _TRANSLATED_NAME_PATTERN.search(run_name) is not None


def discover_translated_runs(scan_root: str | Path) -> list[AdapterCandidate]:
    candidates: list[AdapterCandidate] = []
    for run in discover_runs(scan_root):
        if not is_translated_run_name(run.run_name):
            continue
        candidates.append(
            _candidate(
                ADAPTER_TRANSLATED,
                run.run_name,
                run.fold,
                Path(run.fold_dir),
                [Path(run.run_config_path)],
                {
                    "dataset": (run.resolved_config or {}).get("dataset"),
                    "modality": run.modality,
                    "fold": run.fold,
                    "run_config_file_sha256": run.run_config_file_sha256,
                },
            )
        )
    return candidates


def discover_merged_runs(outputs_root: str | Path) -> list[AdapterCandidate]:
    root = Path(outputs_root)
    candidates: list[AdapterCandidate] = []
    for config_path in sorted(root.glob("symmetric_merged/*/*/cv/fold_*/resolved_merged_config.json")):
        fold_dir = config_path.parent
        try:
            content = read_json(config_path)
        except (ValueError, OSError) as error:
            candidates.append(
                _candidate(
                    ADAPTER_MERGED,
                    fold_dir.parent.name,
                    _fold_number(fold_dir.name),
                    fold_dir,
                    [config_path],
                    {},
                    [f"merged config unreadable: {error}"],
                )
            )
            continue
        if not isinstance(content, dict):
            candidates.append(
                _candidate(
                    ADAPTER_MERGED,
                    fold_dir.parent.name,
                    _fold_number(fold_dir.name),
                    fold_dir,
                    [config_path],
                    {},
                    ["merged config is not an object"],
                )
            )
            continue
        modality = content.get("modality")
        if not isinstance(modality, str):
            modality = fold_dir.parent.parent.name
        components = content.get("components")
        if isinstance(components, list) and components and isinstance(components[0], dict):
            dataset = components[0].get("dataset")
        else:
            dataset = None
        heads = content.get("heads") if isinstance(content.get("heads"), dict) else {}
        evidence = [config_path]
        for head_name in ("logreg", "xgb_fixed", "xgb_optuna"):
            head_dir = fold_dir / "heads" / head_name
            for path in sorted(head_dir.rglob("*.json")):
                evidence.append(path)
        candidates.append(
            _candidate(
                ADAPTER_MERGED,
                content.get("name") or fold_dir.parent.name,
                _fold_number(fold_dir.name),
                fold_dir,
                evidence,
                {
                    "dataset": dataset,
                    "modality": modality,
                    "protocol": content.get("protocol"),
                    "seed": content.get("seed"),
                    "heads": sorted(heads.keys()),
                },
            )
        )
    return candidates


def _fold_number(fold_dir_name: str) -> int | None:
    match = re.fullmatch(r"fold_([0-9]+)", fold_dir_name)
    return int(match.group(1)) if match else None


def discover_hidden_classifiers(outputs_root: str | Path) -> list[AdapterCandidate]:
    root = Path(outputs_root)
    candidates: list[AdapterCandidate] = []
    for metadata_path in sorted(root.glob("hidden_classifiers/*/*/*/fold_*/*/classifier_metadata.json")):
        variant_dir = metadata_path.parent
        fold_dir = variant_dir.parent
        run_dir = fold_dir.parent
        fold_number = _fold_number(fold_dir.name)
        try:
            content = read_json(metadata_path)
        except (ValueError, OSError) as error:
            candidates.append(
                _candidate(
                    ADAPTER_HIDDEN_CLASSIFIER,
                    run_dir.name,
                    fold_number,
                    variant_dir,
                    [metadata_path],
                    {},
                    [f"classifier metadata unreadable: {error}"],
                )
            )
            continue
        if not isinstance(content, dict):
            candidates.append(
                _candidate(
                    ADAPTER_HIDDEN_CLASSIFIER,
                    run_dir.name,
                    fold_number,
                    variant_dir,
                    [metadata_path],
                    {},
                    ["classifier metadata is not an object"],
                )
            )
            continue
        quarantine: list[str] = []
        target_trials = content.get("target_trials")
        completed_trials = content.get("completed_trials")
        if content.get("objective") and isinstance(target_trials, int) and isinstance(completed_trials, int):
            if completed_trials < target_trials:
                quarantine.append(f"incomplete trials: {completed_trials}/{target_trials}")
        evidence = [metadata_path]
        for name in ("metrics.json", "inner_oof_metrics.json", "inner_fold_metrics.json"):
            path = variant_dir / name
            if path.is_file():
                evidence.append(path)
        for name in ("predictions_subject_level.csv", "predictions_subject_level.jsonl"):
            path = variant_dir / name
            if path.is_file():
                evidence.append(path)
        not_run = completed_trials == 0 and content.get("best_value") is None
        candidates.append(
            _candidate(
                ADAPTER_HIDDEN_CLASSIFIER,
                content.get("run_name") or run_dir.name,
                fold_number,
                variant_dir,
                evidence,
                {
                    "dataset": content.get("dataset"),
                    "modality": content.get("modality"),
                    "condition": content.get("condition"),
                    "classifier_variant": content.get("classifier_variant"),
                    "classifier_family": content.get("classifier_family"),
                    "seed": content.get("seed"),
                    "objective": content.get("objective"),
                    "best_value": content.get("best_value"),
                    "completed_trials": completed_trials,
                    "target_trials": target_trials,
                    "optuna_not_run": not_run,
                    "search_config_sha256": content.get("search_config_sha256"),
                },
                quarantine,
            )
        )
    return candidates


def inventory_evidence(
    ordinary_scan_root: str | Path, outputs_root: str | Path
) -> dict[str, Any]:
    ordinary_runs = discover_runs(ordinary_scan_root)
    ordinary_qualified = 0
    ordinary_quarantined = 0
    ordinary_rejected = 0
    quarantined_runs: list[dict[str, Any]] = []
    for run in ordinary_runs:
        result = qualify_run(run)
        if result.status == STATUS_QUARANTINED_AMBIGUOUS or result.status == STATUS_REJECTED:
            ordinary_quarantined += 1
            quarantined_runs.append(
                {
                    "run_dir": run.fold_dir,
                    "status": result.status,
                    "reasons": list(result.reasons),
                }
            )
        else:
            ordinary_qualified += 1
    translated = discover_translated_runs(ordinary_scan_root)
    merged = discover_merged_runs(outputs_root)
    hidden = discover_hidden_classifiers(outputs_root)
    incomplete_hidden = [
        {
            "root": candidate.root,
            "reasons": list(candidate.quarantine_reasons),
        }
        for candidate in hidden
        if candidate.quarantine_reasons
    ]
    unrecoverable = [
        {
            "root": candidate.root,
            "reasons": list(candidate.quarantine_reasons),
        }
        for candidate in (*merged, *hidden)
        if candidate.quarantine_reasons and not candidate.metadata
    ]
    return {
        "ordinary": {
            "fold_runs": len(ordinary_runs),
            "qualified": ordinary_qualified,
            "quarantined": ordinary_quarantined,
        },
        "translated": {"fold_runs": len(translated)},
        "merged": {"fold_candidates": len(merged)},
        "hidden_classifiers": {
            "fold_variant_candidates": len(hidden),
            "incomplete": incomplete_hidden,
        },
        "quarantined_runs": quarantined_runs,
        "unrecoverable": unrecoverable,
        "backfill_report_path": None,
    }


def write_inventory_report(inventory: dict[str, Any], output_path: str | Path) -> None:
    from .canonical import write_json_atomic

    report = dict(inventory)
    report["backfill_report_path"] = str(output_path)
    write_json_atomic(output_path, report)
