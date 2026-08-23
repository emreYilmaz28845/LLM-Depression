#!/usr/bin/env python3
"""Generate the Optuna-100 selection definitions and W&B selection entries.

Creates/updates:
- experiments/definitions/workbook_optuna100_selection.yaml: explicit
  workbook-cell selections for the 248 standardized Optuna-100 studies
  (native 126, english 80, merged 36, officialdev 6), with no metric values
  and no attempt IDs until qualified evidence exists.
- appends 248 skip_not_run entries to
  experiments/definitions/workbook_wandb_selection.yaml so the W&B resolver
  can never export a not-yet-run study. Idempotent: entries that already
  exist are left untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = PROJECT_ROOT / "experiments/definitions/workbook_optuna100_selection.yaml"
WANDB_SELECTION_PATH = PROJECT_ROOT / "experiments/definitions/workbook_wandb_selection.yaml"

PROTOCOL = "harmonized_optuna100_v1"
VIEW = "harmonized_all_windows_full_coverage"
NAMESPACE = "headline/binary_strict"
AGGREGATION = "subject_level"

QWEN_BACKEND = "qwen_hidden_xgb_optuna100"
GEMMA_BACKEND = "gemma4_hidden_xgb_optuna100"

# (family, dataset, modality, folds) for the standalone families.
NATIVE_CELLS = [
    ("Native", "d3tec", "audio_text", list(range(5))),
    ("Native", "d3tec", "audio_only", list(range(5))),
    ("Native", "d3tec", "text_only", list(range(5))),
    ("Native", "androids_interview", "audio_text", list(range(5))),
    ("Native", "androids_interview", "audio_only", list(range(5))),
    ("Native", "androids_interview", "text_only", list(range(5))),
    ("Native", "cmdc", "audio_text", list(range(5))),
    ("Native", "cmdc", "audio_only", list(range(5))),
    ("Native", "cmdc", "text_only", list(range(5))),
    ("Native", "turkish", "audio_text", list(range(5))),
    ("Native", "turkish", "audio_only", list(range(5))),
    ("Native", "turkish", "text_only", list(range(5))),
    ("Native", "daic", "audio_text", [0]),
    ("Native", "daic", "audio_only", [0]),
    ("Native", "daic", "text_only", [0]),
]
ENGLISH_CELLS = [
    ("English", "d3tec", "audio_text", list(range(5))),
    ("English", "d3tec", "text_only", list(range(5))),
    ("English", "androids_interview", "audio_text", list(range(5))),
    ("English", "androids_interview", "text_only", list(range(5))),
    ("English", "cmdc", "audio_text", list(range(5))),
    ("English", "cmdc", "text_only", list(range(5))),
    ("English", "turkish", "audio_text", list(range(5))),
    ("English", "turkish", "text_only", list(range(5))),
]
MERGED_CELLS = [
    ("Symmetric merged", "merged", "audio_text", "cv", list(range(5))),
    ("Symmetric merged", "merged", "audio_only", "cv", list(range(5))),
    ("Symmetric merged", "merged", "text_only", "cv", list(range(5))),
    ("Symmetric merged", "merged", "audio_text", "final", [0]),
    ("Symmetric merged", "merged", "audio_only", "final", [0]),
    ("Symmetric merged", "merged", "text_only", "final", [0]),
]
OFFICIALDEV_CELLS = [
    ("DAIC official development", "daic", "audio_text", [0]),
    ("DAIC official development", "daic", "audio_only", [0]),
    ("DAIC official development", "daic", "text_only", [0]),
]


def standalone_studies(cells) -> list[dict]:
    studies = []
    for family, dataset, modality, folds in cells:
        for backend, prediction_backend in (("qwen", QWEN_BACKEND), ("gemma4", GEMMA_BACKEND)):
            for fold in folds:
                studies.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "modality": modality,
                        "fold": fold,
                        "backend": backend,
                        "prediction_backend": prediction_backend,
                    }
                )
    return studies


def merged_studies() -> list[dict]:
    studies = []
    for family, dataset, modality, stage, folds in MERGED_CELLS:
        for backend, prediction_backend in (("qwen", QWEN_BACKEND), ("gemma4", GEMMA_BACKEND)):
            for fold in folds:
                studies.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "modality": modality,
                        "stage": stage,
                        "fold": fold,
                        "backend": backend,
                        "prediction_backend": f"{prediction_backend}_symmetric_merged",
                    }
                )
    return studies


def all_studies() -> list[dict]:
    return standalone_studies(NATIVE_CELLS) + standalone_studies(ENGLISH_CELLS) + merged_studies() + standalone_studies(OFFICIALDEV_CELLS)


def cell_of(study: dict) -> str:
    dataset = study["dataset"]
    dataset_label = {
        "d3tec": "D3TEC", "androids_interview": "Androids Interview", "cmdc": "CMDC",
        "turkish": "Turkish t17", "daic": "DAIC-WOZ", "merged": "Merged",
    }[dataset]
    modality_label = {"audio_text": "Audio + Text", "audio_only": "Audio only", "text_only": "Text only"}[study["modality"]]
    stage = f" — {study['stage'].upper()}" if study.get("stage") else ""
    backend_label = "Qwen" if study["backend"] == "qwen" else "Gemma 4"
    return f"{study['family']} — {dataset_label}{stage}|{modality_label} — {backend_label}"


def selection_id_of(study: dict) -> str:
    dataset = study["dataset"]
    stage = f"_{study['stage']}" if study.get("stage") else ""
    return f"Optuna100|{study['family']}|{dataset}{stage}|{study['modality']}|{study['backend']}|fold{study['fold']}"


def build_selection_file() -> None:
    studies = all_studies()
    selections = []
    for study in studies:
        selections.append(
            {
                "cell": cell_of(study),
                "dataset": study["dataset"],
                "modality": study["modality"],
                "metric": "macro_f1",
                "namespace": NAMESPACE,
                "backend": study["prediction_backend"],
                "view": VIEW,
                "aggregation": AGGREGATION,
                "attempt_id": None,
                "fold": study["fold"],
                "family": study["family"],
                "stage": study.get("stage"),
                "protocol_profile": PROTOCOL,
            }
        )
    payload = {
        "schema_version": "audiollm.selected_results.v1",
        "note": "Generated by tools/build_optuna100_wandb_selection.py. "
                "Attempt IDs and values populate after qualification (Tasks 9-12); "
                "blank cells stay blank until then.",
        "selections": selections,
    }
    rendered = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    if SELECTION_PATH.is_file() and SELECTION_PATH.read_text(encoding="utf-8") != rendered:
        raise SystemExit(f"refusing to overwrite a different {SELECTION_PATH.name}; review first")
    SELECTION_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {SELECTION_PATH} ({len(selections)} selections)")


def _refresh_workbook_hashes(payload: dict) -> None:
    """Re-hash the generated workbook and its builder into the selection
    workbook block after the builder changes."""
    import hashlib

    workbook_block = payload.get("workbook") or {}
    workbook_path = PROJECT_ROOT / workbook_block["path"]
    builder_path = PROJECT_ROOT / workbook_block["builder_path"]
    if not workbook_path.is_file():
        raise SystemExit(
            f"workbook {workbook_path} is missing; regenerate it with "
            "python scripts/build_clean_workbook.py first"
        )
    if not builder_path.is_file():
        raise SystemExit(f"workbook builder missing: {builder_path}")

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    workbook_block["sha256"] = digest(workbook_path)
    workbook_block["builder_sha256"] = digest(builder_path)


def build_wandb_entries() -> None:
    original_text = WANDB_SELECTION_PATH.read_text(encoding="utf-8")
    payload = yaml.safe_load(original_text)
    entries = payload.setdefault("entries", [])
    existing_ids = {entry["selection_id"] for entry in entries}
    studies = all_studies()
    added = 0
    for study in studies:
        selection_id = selection_id_of(study)
        if selection_id in existing_ids:
            continue
        modality_label = {"audio_text": "Audio + Text", "audio_only": "Audio only", "text_only": "Text only"}[study["modality"]]
        dataset_label = {
            "d3tec": "D3TEC", "androids_interview": "Androids Interview", "cmdc": "CMDC",
            "turkish": "Turkish t17", "daic": "DAIC-WOZ", "merged": "Merged",
        }[study["dataset"]]
        entries.append(
            {
                "selection_id": selection_id,
                "provenance_key": {
                    "experiment": study["family"],
                    "dataset": dataset_label,
                    "modality": modality_label,
                    "method": f"Optuna-100 XGB ({'Qwen' if study['backend'] == 'qwen' else 'Gemma 4'})",
                },
                "source_type": "hidden_classifier",
                "reason": "optuna100_harmonized_v1",
                "dependency_policy": "all_contributing_folds",
                "expected_folds": [study["fold"]],
                "required_evaluations": [
                    {
                        "dataset": "daic" if study["dataset"] in ("daic",) else study["dataset"],
                        "namespace": NAMESPACE,
                        "backend": study["prediction_backend"],
                        "aggregation": AGGREGATION,
                        "checkpoint_role": "best_model",
                        "view": VIEW,
                    }
                ],
                "group": study["family"],
                "wandb_policy": "skip_not_run",
                "blocking_reasons": ["study not run yet"],
            }
        )
        existing_ids.add(selection_id)
        added += 1
    workbook_block = payload["workbook"]
    old_hashes = (workbook_block.get("sha256"), workbook_block.get("builder_sha256"))
    _refresh_workbook_hashes(payload)
    new_hashes = (workbook_block.get("sha256"), workbook_block.get("builder_sha256"))
    if added == 0:
        if old_hashes == new_hashes:
            print(f"updated {WANDB_SELECTION_PATH}: 0 new entries, total {len(entries)}")
            return
        rendered = re.sub(
            r"(?m)^(  sha256: )[0-9a-f]{64}$",
            rf"\g<1>{new_hashes[0]}",
            original_text,
            count=1,
        )
        rendered = re.sub(
            r"(?m)^(  builder_sha256: )[0-9a-f]{64}$",
            rf"\g<1>{new_hashes[1]}",
            rendered,
            count=1,
        )
        WANDB_SELECTION_PATH.write_text(rendered, encoding="utf-8")
        print(f"updated {WANDB_SELECTION_PATH}: 0 new entries, total {len(entries)}")
        return
    rendered = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    WANDB_SELECTION_PATH.write_text(rendered, encoding="utf-8")
    print(f"updated {WANDB_SELECTION_PATH}: {added} new entries, total {len(entries)}")


def main() -> int:
    build_selection_file()
    build_wandb_entries()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
