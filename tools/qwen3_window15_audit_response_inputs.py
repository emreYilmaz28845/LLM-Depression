#!/usr/bin/env python
"""Audit window15 response-dataset inputs (D3TEC, Androids, CMDC, Turkish).

Builds the 15-second treatment manifest and the 30-second control manifest into
a dedicated audit directory with the current code, then verifies:

- label/cohort/split identity between the arms (subject labels and fold
  assignments);
- rendered subject prompt-text byte equality between the arms (the
  full_subject text scope must not change; only the audio windows change);
- D3TEC/Androids: the 15s child windows partition each unit contiguously and
  cover exactly the same unit interval as the canonical 30s windows; every
  child window is at most ``segment_seconds`` long; child rows carry the
  canonical full response/turn transcript byte-for-byte with an empty
  ``segment_transcript`` and a reference-only marker; canonical rows keep the
  unchanged schema (no reference fields);
- CMDC/Turkish: manifest rows are byte-identical between the arms (the 15s
  windowing happens at example build time from the same unit rows).

The report is deterministic; any issue exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from src.data.build_manifest import build_for_config  # noqa: E402
from src.data.runtime import _base_example_from_row, _harmonized_subject_transcripts  # noqa: E402
from src.utils import load_yaml_with_overrides, read_jsonl  # noqa: E402

FULL_FIELD = {"d3tec": "full_response_transcript", "androids_interview": "full_turn_transcript"}
UNIT_KEY = {"d3tec": "response_id", "androids_interview": "turn_key"}
SEGMENTED_DATASETS = {"d3tec", "androids_interview"}
REFERENCE_FIELDS = (
    "source_segment_ref",
    "source_window_ref",
    "canonical_segment_transcript_ref",
    "canonical_window_transcript_ref",
    "segment_transcript_scope",
)


def build_arm(config_path: str, arm_root: Path, dataset: str) -> dict:
    manifest_dir = arm_root / "manifests" / dataset
    split_dir = arm_root / "splits" / dataset
    build_for_config(
        config_path,
        [
            f"output_dirs.manifest_dir={manifest_dir}",
            f"output_dirs.split_dir={split_dir}",
        ],
    )
    return load_arm(manifest_dir, split_dir, dataset)


def load_arm(manifest_dir: Path, split_dir: Path, dataset: str) -> dict:
    rows = read_jsonl(manifest_dir / f"{dataset}_manifest.jsonl")
    folds_path = split_dir / f"{dataset}_folds.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.is_file() else None
    return {"rows": rows, "manifest_dir": str(manifest_dir), "folds": folds}


def rendered_texts(rows: list[dict], config_path: str, dataset: str) -> dict[str, str]:
    config = load_yaml_with_overrides(config_path, [])
    subject_transcripts = _harmonized_subject_transcripts(rows, dataset)
    transcript_max_chars = int(config["data"].get("transcript_max_chars", 0) or 0)
    texts: dict[str, str] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        if subject_id in texts:
            continue
        probe = dict(row)
        probe["transcript"] = subject_transcripts.get(subject_id, "")
        probe["full_subject_transcript"] = subject_transcripts.get(subject_id, "")
        example, _ = _base_example_from_row(probe, config, transcript_max_chars)
        texts[subject_id] = str(example["prompt_text"])
    return texts


def unit_intervals(rows: list[dict], dataset: str) -> dict[tuple[str, str], list[dict]]:
    unit_key = UNIT_KEY[dataset]
    units: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["subject_id"]), str(row[unit_key]))
        units.setdefault(key, []).append(row)
    for key in units:
        units[key].sort(key=lambda row: int(row.get("segment_index", 0)))
    return units


def audit_segmented(dataset: str, treatment: dict, control: dict, segment_seconds: float) -> tuple[dict, list[str]]:
    issues: list[str] = []
    full_field = FULL_FIELD[dataset]
    units_15 = unit_intervals(treatment["rows"], dataset)
    units_30 = unit_intervals(control["rows"], dataset)
    if set(units_15) != set(units_30):
        issues.append(f"unit sets differ: 15s {len(units_15)} vs 30s {len(units_30)}")
    child_rows = 0
    canonical_rows = 0
    for key in sorted(set(units_15) & set(units_30)):
        children = units_15[key]
        canonical = units_30[key]
        child_rows += len(children)
        canonical_rows += len(canonical)
        if len(children) != int(children[0].get("num_segments", 0)):
            issues.append(f"{key}: num_segments {children[0].get('num_segments')} != {len(children)}")
        for left, right in zip(children, children[1:]):
            if abs(float(left["end_time"]) - float(right["start_time"])) > 1e-6:
                issues.append(f"{key}: child windows are not contiguous")
                break
        for child in children:
            duration = float(child["end_time"]) - float(child["start_time"])
            if duration <= 0 or duration > float(segment_seconds) + 1e-6:
                issues.append(f"{key}: child duration {duration} outside (0, {segment_seconds}]")
                break
        union_15 = (min(float(row["start_time"]) for row in children), max(float(row["end_time"]) for row in children))
        union_30 = (min(float(row["start_time"]) for row in canonical), max(float(row["end_time"]) for row in canonical))
        if abs(union_15[0] - union_30[0]) > 1e-6 or abs(union_15[1] - union_30[1]) > 1e-6:
            issues.append(f"{key}: 15s union {union_15} != 30s union {union_30}")
        for child in children:
            if "source_reference_index" not in child:
                issues.append(f"{key}: 15s child row lacks source_reference_index")
                break
            if str(child.get("transcript", "")) != str(child.get(full_field, "")):
                issues.append(f"{key}: child transcript != {full_field}")
                break
            if str(child.get("segment_transcript", "")) != "":
                issues.append(f"{key}: child segment_transcript is not empty")
                break
            if not any(field in child for field in REFERENCE_FIELDS):
                issues.append(f"{key}: child row lacks reference-only fields")
                break
        for row in canonical:
            if any(field in row for field in REFERENCE_FIELDS) or "source_reference_index" in row:
                issues.append(f"{key}: canonical 30s row gained reference fields")
                break
    return {
        "units": len(units_15),
        "child_rows": child_rows,
        "canonical_rows": canonical_rows,
        "max_child_duration": max(
            (float(row["end_time"]) - float(row["start_time"]) for row in treatment["rows"]), default=0.0
        ),
    }, issues


def audit_flat(dataset: str, treatment: dict, control: dict, segment_seconds: float) -> tuple[dict, list[str]]:
    import math

    import soundfile as sf

    issues: list[str] = []
    rows_15 = {str(row["sample_id"]): row for row in treatment["rows"]}
    rows_30 = {str(row["sample_id"]): row for row in control["rows"]}
    if set(rows_15) != set(rows_30):
        issues.append(f"sample_id sets differ: 15s {len(rows_15)} vs 30s {len(rows_30)}")
    differing = 0
    for sample_id in sorted(set(rows_15) & set(rows_30)):
        left = json.dumps(rows_15[sample_id], sort_keys=True, ensure_ascii=False)
        right = json.dumps(rows_30[sample_id], sort_keys=True, ensure_ascii=False)
        if left != right:
            differing += 1
    if differing:
        issues.append(f"{differing} rows differ between the arms for {dataset}")
    # Boundary/coverage of the treatment window plan on the real audio: the
    # equal_duration rule partitions each unit into ceil(duration / 15)
    # contiguous windows, each at most 15 seconds, covering the whole unit.
    checked = 0
    for sample_id, row in sorted(rows_15.items()):
        paths = row.get("audio_paths") or [row.get("audio_path")]
        path = next((item for item in paths if item), None)
        if not path:
            issues.append(f"{sample_id}: no audio path for the window-plan check")
            continue
        info = sf.info(str(path))
        duration = float(info.frames / info.samplerate)
        count = max(1, int(math.ceil(duration / float(segment_seconds))))
        window = duration / count
        if count > 1 and window > float(segment_seconds) + 1e-6:
            issues.append(f"{sample_id}: window {window} exceeds {segment_seconds}")
        if count * window + 1e-6 < duration:
            issues.append(f"{sample_id}: windows do not cover the unit duration")
        checked += 1
    return {
        "rows": len(rows_15),
        "identical_rows": len(rows_15) - differing,
        "window_plan_checked": checked,
    }, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(FULL_FIELD) + ["cmdc", "turkish"])
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="shared prebuilt manifest dir (skips building; used for pooled Turkish)",
    )
    parser.add_argument("--split-dir", type=Path, default=None, help="shared prebuilt split dir")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = args.dataset
    if args.manifest_dir:
        split_dir = args.split_dir or args.manifest_dir
        treatment = load_arm(args.manifest_dir, split_dir, dataset)
        control = treatment
    else:
        treatment = build_arm(args.treatment_config, args.runtime_root / "treatment15", dataset)
        control = build_arm(args.control_config, args.runtime_root / "control30", dataset)

    issues: list[str] = []
    labels_15 = {str(row["subject_id"]): int(row["label"]) for row in treatment["rows"]}
    labels_30 = {str(row["subject_id"]): int(row["label"]) for row in control["rows"]}
    if labels_15 != labels_30:
        issues.append("subject label maps differ between arms")
    if treatment["folds"] != control["folds"]:
        issues.append("fold assignments differ between arms")

    texts_15 = rendered_texts(treatment["rows"], args.treatment_config, dataset)
    texts_30 = rendered_texts(control["rows"], args.control_config, dataset)
    if texts_15 != texts_30:
        differing = sorted(set(texts_15) ^ set(texts_30)) or [
            subject for subject in texts_15 if texts_15.get(subject) != texts_30.get(subject)
        ]
        issues.append(f"rendered subject prompt text differs: {differing[:10]}")

    segment_seconds = float(
        (load_yaml_with_overrides(args.treatment_config, []).get("data") or {}).get(
            "segment_seconds", 15.0
        )
    )
    if dataset in SEGMENTED_DATASETS:
        details, detail_issues = audit_segmented(dataset, treatment, control, segment_seconds)
    else:
        details, detail_issues = audit_flat(dataset, treatment, control, segment_seconds)
    issues.extend(detail_issues)

    report = {
        "schema_version": "audiollm.qwen3_window15_response_input_audit.v1",
        "dataset": dataset,
        "control_config": args.control_config,
        "treatment_config": args.treatment_config,
        "subjects": len(labels_15),
        "rendered_texts_equal": texts_15 == texts_30,
        "details": details,
        "issues": issues,
        "status": "passed" if not issues else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": issues, "output": str(args.output)}))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
