"""Tests for tools/paired_significance.py and the pre-declared family file."""

from __future__ import annotations

import csv
import importlib.util as _ilu
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = _ilu.spec_from_file_location("paired_significance", ROOT / "tools/paired_significance.py")
paired_significance = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(paired_significance)

FAMILY_PATH = ROOT / "experiments/definitions/significance_family.yaml"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_subjects_strips_dataset_namespace():
    rows = [
        {"subject_id": "androids_interview::01_C", "label": 0, "prediction": 0},
        {"subject_id": "01_P", "label": 1, "prediction": 1},
    ]
    out = paired_significance.normalize_subjects(rows, "androids_interview")
    assert [row["subject_id"] for row in out] == ["01_C", "01_P"]
    # another dataset's namespace is not stripped
    assert paired_significance.normalize_subjects(rows[:1], "d3tec")[0]["subject_id"] == "androids_interview::01_C"
    with pytest.raises(paired_significance.SignificanceError, match="duplicate subject id"):
        paired_significance.normalize_subjects(
            [{"subject_id": "01_C", "label": 0, "prediction": 0},
             {"subject_id": "d::01_C", "label": 0, "prediction": 0}], "d")


def test_read_rows_and_pool_filter_and_deduplicate(tmp_path: Path):
    csv_path = tmp_path / "fold_0.csv"
    _write_csv(csv_path, [
        {"dataset": "d3tec", "subject_id": "s1", "label": 1, "prediction": 1},
        {"dataset": "cmdc", "subject_id": "s9", "label": 1, "prediction": 0},
    ])
    jsonl_path = tmp_path / "fold_1.jsonl"
    jsonl_path.write_text(
        '\n'.join(json.dumps(row) for row in [
            {"dataset": "d3tec", "subject_id": "s2", "label": 0, "prediction": 1},
            {"dataset": "cmdc", "subject_id": "s8", "label": 0, "prediction": 0},
        ]) + "\n",
        encoding="utf-8",
    )
    assert [row["subject_id"] for row in paired_significance.read_rows(csv_path, "d3tec")] == ["s1"]
    assert [row["subject_id"] for row in paired_significance.read_rows(jsonl_path)] == ["s2", "s8"]
    pooled = paired_significance.pool([(csv_path, "d3tec"), (jsonl_path, "d3tec")])
    assert sorted(row["subject_id"] for row in pooled) == ["s1", "s2"]
    with pytest.raises(paired_significance.SignificanceError, match="more than one fold file"):
        paired_significance.pool([(csv_path, "d3tec"), (csv_path, "d3tec")])
    with pytest.raises(paired_significance.SignificanceError, match="no subject rows"):
        paired_significance.pool([(csv_path, "missing_dataset")])


def test_resolve_side_path_kind_is_fail_closed(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(paired_significance.SignificanceError, match="missing predictions file"):
        paired_significance.resolve_side({"kind": "path", "path": str(missing)}, {}, None)
    with pytest.raises(paired_significance.SignificanceError, match="unknown side kind"):
        paired_significance.resolve_side({"kind": "mystery"}, {}, None)


def test_declared_family_is_frozen_and_structured():
    payload = yaml.safe_load(FAMILY_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "audiollm.significance_family.v1"
    assert payload["alpha"] == 0.05
    assert payload["iterations"] == 10000 and payload["seed"] == 1337
    assert payload["metrics"] == ["macro_f1", "macro_recall"]
    blocks = {block["id"]: block for block in payload["families"]}
    assert list(blocks) == [
        "model_qwen_vs_gemma4_teacher_forced",
        "route_pairs_native",
        "native_vs_english_transcript",
        "standalone_vs_merged_audio_text",
        "joint_k4_v1_vs_runtime",
    ]
    assert [len(blocks[key]["comparisons"]) for key in blocks] == [15, 90, 36, 16, 6]
    known_kinds = {"record", "merged_cv", "merged_final", "joint_k4", "path"}
    ids = []
    for block in payload["families"]:
        for comparison in block["comparisons"]:
            ids.append(comparison["id"])
            for side in ("baseline", "comparison"):
                assert comparison[side]["kind"] in known_kinds
    assert len(ids) == len(set(ids)) == 163
    # Turkish standalone-vs-merged and merged XGBoost stay excluded on purpose.
    assert any("Turkish standalone-versus-merged" in note for note in payload["excluded"])
    assert any("Merged XGBoost" in note for note in payload["excluded"])
