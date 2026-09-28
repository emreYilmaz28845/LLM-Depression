"""Tests for the retrospective paired-significance analysis."""

from __future__ import annotations

import csv
import importlib.util as _ilu
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = _ilu.spec_from_file_location("paired_significance", ROOT / "tools/paired_significance.py")
paired_significance = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(paired_significance)

FAMILY_PATH = ROOT / "experiments/definitions/significance_family.yaml"

from src.metrics import classification_metrics  # noqa: E402  (loaded after the tool inserts the repo root)


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
    with pytest.raises(paired_significance.SignificanceError, match="duplicate subject/seed"):
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
    assert payload["analysis_status"] == "retrospective_exploratory"
    assert payload["iterations"] == 1000000 and payload["seed"] == 1337
    assert payload["bootstrap_iterations"] == 100000
    assert payload["metrics"] == ["macro_f1", "positive_f1", "macro_recall"]
    blocks = {block["id"]: block for block in payload["families"]}
    assert list(blocks) == [
        "model_qwen_vs_gemma4_teacher_forced",
        "route_pairs_native",
        "native_vs_english_transcript",
        "standalone_vs_merged_audio_text",
        "joint_k4_v1_vs_runtime",
        "daic_official_development_backbone",
    ]
    assert [len(blocks[key]["comparisons"]) for key in blocks] == [15, 90, 36, 16, 6, 1]
    assert [item["kind"] for item in payload["generated_families"]] == [
        "backbone_same_route", "native_en_hidden_heads"
    ]
    known_kinds = {"record", "merged_cv", "merged_final", "joint_k4", "path", "native_en_head"}
    ids = []
    for block in payload["families"]:
        for comparison in block["comparisons"]:
            ids.append(comparison["id"])
            for side in ("baseline", "comparison"):
                assert comparison[side]["kind"] in known_kinds
    assert len(ids) == len(set(ids)) == 164
    # Turkish standalone-vs-merged and merged XGBoost stay excluded on purpose.
    assert any("Turkish standalone-versus-merged" in note for note in payload["excluded"])
    assert any("Merged XGBoost" in note for note in payload["excluded"])


@pytest.mark.parametrize(("block", "comparison", "expected"), [
    ("model_qwen_vs_gemma4_teacher_forced", "CMDC|A+T|Qwen vs Gemma 4",
     "F1|backbone|dataset=CMDC|condition=native|route=teacher-forced"),
    ("native_vs_english_transcript", "CMDC|T|Qwen|teacher-forced|native vs English",
     "F2|translation|dataset=CMDC|backbone=Qwen|route=teacher-forced"),
    ("route_pairs_native", "Turkish|T|Qwen|teacher-forced vs XGB-100",
     "F3|route|dataset=Turkish|modality=T|backbone=Qwen|condition=native"),
    ("standalone_vs_merged_audio_text", "CMDC|Gemma 4|LogReg|standalone vs merged",
     "F4|training-regime|dataset=CMDC|modality=A+T|backbone=Gemma 4|route=LogReg"),
    ("joint_k4_v1_vs_runtime", "DAIC|A|LogReg|packed30 v1 vs joint-K4",
     "F5|recipe|dataset=DAIC|modality=A|backbone=Qwen|method=LogReg"),
])
def test_correction_family_contract(block, comparison, expected):
    assert paired_significance.correction_family_id(block, comparison) == expected


def _single_seed_sides(tmp_path: Path) -> tuple[Path, Path]:
    """Baseline gets one subject right, the comparison gets seven of eight."""
    baseline = tmp_path / "baseline.csv"
    comparison = tmp_path / "comparison.csv"
    _write_csv(baseline, [
        {"subject_id": "s1", "label": 1, "prediction": 1},
        {"subject_id": "s2", "label": 1, "prediction": 0},
        {"subject_id": "s3", "label": 1, "prediction": 0},
        {"subject_id": "s4", "label": 1, "prediction": 0},
        {"subject_id": "s5", "label": 0, "prediction": 1},
        {"subject_id": "s6", "label": 0, "prediction": 1},
        {"subject_id": "s7", "label": 0, "prediction": 1},
        {"subject_id": "s8", "label": 0, "prediction": 1},
    ])
    _write_csv(comparison, [
        {"subject_id": "s1", "label": 1, "prediction": 1},
        {"subject_id": "s2", "label": 1, "prediction": 1},
        {"subject_id": "s3", "label": 1, "prediction": 1},
        {"subject_id": "s4", "label": 1, "prediction": 1},
        {"subject_id": "s5", "label": 0, "prediction": 0},
        {"subject_id": "s6", "label": 0, "prediction": 0},
        {"subject_id": "s7", "label": 0, "prediction": 0},
        {"subject_id": "s8", "label": 0, "prediction": 1},
    ])
    return baseline, comparison


def _synthetic_family(tmp_path: Path, baseline: Path, comparison: Path) -> Path:
    family_path = tmp_path / "family.yaml"
    family_path.write_text(yaml.safe_dump({
        "schema_version": "audiollm.significance_family.v1",
        "alpha": 0.05,
        "families": [{
            "id": "synthetic_block",
            "comparisons": [{
                "id": "synthetic|A vs B",
                "dataset": None,
                "baseline": {"kind": "path", "path": str(baseline)},
                "comparison": {"kind": "path", "path": str(comparison)},
            }],
        }],
    }), encoding="utf-8")
    return family_path


def test_mcnemar_rows_single_seed_exact_binomial(tmp_path: Path):
    baseline, comparison = _single_seed_sides(tmp_path)
    rows = paired_significance.mcnemar_rows(
        yaml.safe_load(_synthetic_family(tmp_path, baseline, comparison).read_text(encoding="utf-8")),
        {}, None, alpha=0.05,
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row["baseline_only_correct"], row["comparison_only_correct"]) == (0, 6)
    assert row["discordant_subjects"] == 6 and row["subjects"] == 8
    # two-sided exact binomial: 2 * C(6,0) / 2**6
    assert row["p_value"] == pytest.approx(0.03125)
    assert row["uncorrected_significant"] is True
    assert (row["seed"], row["seeds_in_comparison"]) == (0, 1)
    assert row["block"] == "synthetic_block" and row["comparison_id"] == "synthetic|A vs B"


def test_mcnemar_rows_keep_seeds_separate(tmp_path: Path):
    baseline = tmp_path / "baseline.csv"
    comparison = tmp_path / "comparison.csv"
    baseline_rows, comparison_rows = [], []
    for seed in (0, 1, 2):
        for subject, label in [("s1", 1), ("s2", 0), ("s3", 1), ("s4", 0)]:
            baseline_rows.append({"subject_id": subject, "label": label, "prediction": label, "seed": seed})
            comparison_rows.append({
                "subject_id": subject, "label": label, "seed": seed,
                "prediction": 1 - label if subject == "s3" else label,
            })
    _write_csv(baseline, baseline_rows)
    _write_csv(comparison, comparison_rows)
    rows = paired_significance.mcnemar_rows(
        yaml.safe_load(_synthetic_family(tmp_path, baseline, comparison).read_text(encoding="utf-8")),
        {}, None, alpha=0.05,
    )
    assert [row["seed"] for row in rows] == [0, 1, 2]
    assert all(row["seeds_in_comparison"] == 3 and row["subjects"] == 4 for row in rows)
    assert all(row["baseline_accuracy"] == 1.0 for row in rows)


def test_mcnemar_rows_refuse_mismatched_seed_sets(tmp_path: Path):
    baseline = tmp_path / "baseline.csv"
    comparison = tmp_path / "comparison.csv"
    _write_csv(baseline, [
        {"subject_id": "s1", "label": 1, "prediction": 1, "seed": seed} for seed in (0, 1)
    ])
    _write_csv(comparison, [{"subject_id": "s1", "label": 1, "prediction": 0, "seed": 0}])
    with pytest.raises(paired_significance.SignificanceError, match="seed sets differ"):
        paired_significance.mcnemar_rows(
            yaml.safe_load(_synthetic_family(tmp_path, baseline, comparison).read_text(encoding="utf-8")),
            {}, None, alpha=0.05,
        )


def test_mcnemar_table_cli_writes_uncorrected_sidecar(tmp_path: Path, monkeypatch):
    baseline, comparison = _single_seed_sides(tmp_path)
    family_path = _synthetic_family(tmp_path, baseline, comparison)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    table_path = tmp_path / "mcnemar_table.csv"
    monkeypatch.setattr(sys, "argv", [
        "paired_significance.py", "--family", str(family_path), "--evidence", str(evidence_path),
        "--mcnemar-table", str(table_path),
    ])
    assert paired_significance.main() == 0
    metadata = json.loads(table_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "audiollm.mcnemar_table.v1"
    assert metadata["correction"] == "none"
    assert (metadata["comparisons"], metadata["p_values"]) == (1, 1)
    assert metadata["expected_false_positives_at_alpha"] == 0.05
    written = list(csv.DictReader(table_path.open(encoding="utf-8")))
    assert len(written) == 1
    assert written[0]["uncorrected_significant"] == "True"
    assert float(written[0]["p_value"]) == pytest.approx(0.03125)


def _synthetic_report(family_sha256: str) -> dict:
    return {
        "schema_version": "audiollm.significance_report.v1",
        "family_sha256": family_sha256,
        "alpha": 0.05,
        "metrics": ["macro_f1", "macro_recall"],
        "results": {"blocks": [{
            "id": "synthetic_block",
            "comparisons": [{
                "id": "synthetic|A vs B",
                "dataset": "d3tec",
                "n_subjects": 62,
                "n_seeds": 1,
                "correction_family": "F1|backbone|dataset=d3tec",
                "metrics": {
                    "macro_f1": {
                        "permutation": {"observed_delta": -0.13, "p_value": 0.0068,
                                        "method": "exact_subject_paired", "p_value_holm_global": 1.0},
                        "bootstrap": {"ci_low": -0.2268, "ci_high": -0.0463},
                    },
                    "macro_recall": {
                        "permutation": {"observed_delta": -0.1346, "p_value": 0.42,
                                        "method": "exact_subject_paired", "p_value_holm_global": 1.0},
                        "bootstrap": {"ci_low": -0.2308, "ci_high": -0.0481},
                    },
                },
                "mcnemar": {"status": "tested", "p_value": 0.0117},
            }],
        }]},
    }


def test_metric_rows_drop_correction_columns():
    rows = paired_significance.metric_rows(_synthetic_report("deadbeef"))
    assert [row["metric"] for row in rows] == ["macro_f1", "macro_recall"]
    assert not any("holm" in key for key in rows[0])
    assert rows[0]["p_value"] == pytest.approx(0.0068)
    assert rows[0]["uncorrected_significant"] is True
    assert rows[1]["uncorrected_significant"] is False
    assert rows[0]["mcnemar_p_value"] == pytest.approx(0.0117)


def test_metric_table_cli_refuses_a_stale_report(tmp_path: Path, monkeypatch):
    family_path = tmp_path / "family.yaml"
    family_path.write_text(yaml.safe_dump({
        "schema_version": "audiollm.significance_family.v1", "alpha": 0.05, "families": [],
    }), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_synthetic_report("not-the-family-hash")), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "paired_significance.py", "--family", str(family_path), "--evidence", str(evidence_path),
        "--metric-table", str(tmp_path / "metric_table.csv"), "--from-report", str(report_path),
    ])
    with pytest.raises(paired_significance.SignificanceError, match="different family file"):
        paired_significance.main()


def test_metric_table_cli_writes_uncorrected_sidecar(tmp_path: Path, monkeypatch):
    family_path = tmp_path / "family.yaml"
    family_path.write_text(yaml.safe_dump({
        "schema_version": "audiollm.significance_family.v1", "alpha": 0.05, "families": [],
    }), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(_synthetic_report(paired_significance.sha256_file(family_path))), encoding="utf-8"
    )
    table_path = tmp_path / "metric_table.csv"
    monkeypatch.setattr(sys, "argv", [
        "paired_significance.py", "--family", str(family_path), "--evidence", str(evidence_path),
        "--metric-table", str(table_path), "--from-report", str(report_path),
    ])
    assert paired_significance.main() == 0
    metadata = json.loads(table_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["correction"] == "none"
    assert metadata["schema_version"] == "audiollm.metric_table.v1"
    assert (metadata["comparisons"], metadata["tests"]) == (1, 2)
    assert metadata["expected_false_positives_at_alpha"] == 0.1
    written = list(csv.DictReader(table_path.open(encoding="utf-8")))
    assert [row["metric"] for row in written] == ["macro_f1", "macro_recall"]
    assert "p_value_holm_global" not in written[0]


def _report_fixture(
    tmp_path: Path, *, baseline_prediction: int = 0, delta_offset: float = 0.0, invalid: bool = False
) -> tuple[dict, Path, Path]:
    """A minimal report payload plus the family and prediction files it points at."""
    baseline = tmp_path / "baseline.csv"
    comparison = tmp_path / "comparison.csv"
    rows_baseline = [
        {"subject_id": "s1", "label": 1, "prediction": 1},
        {"subject_id": "s2", "label": 1, "prediction": baseline_prediction},
        {"subject_id": "s3", "label": 0, "prediction": 0},
        {"subject_id": "s4", "label": 0, "prediction": 0},
    ]
    if invalid:
        rows_baseline[1] = {"subject_id": "s2", "label": 1, "prediction": -1}
    rows_comparison = [
        {"subject_id": "s1", "label": 1, "prediction": 1},
        {"subject_id": "s2", "label": 1, "prediction": 1},
        {"subject_id": "s3", "label": 0, "prediction": 0},
        {"subject_id": "s4", "label": 0, "prediction": 0},
    ]
    _write_csv(baseline, rows_baseline)
    _write_csv(comparison, rows_comparison)

    def strict_score(rows: list[dict]) -> float:
        return classification_metrics(
            [int(row["label"]) for row in rows],
            [int(row["prediction"]) if int(row["prediction"]) in (0, 1) else 1 - int(row["label"]) for row in rows],
        )["macro_f1"]

    scores = {"baseline": strict_score(rows_baseline), "comparison": strict_score(rows_comparison)}
    permutation = {
        "observed_delta": scores["comparison"] - scores["baseline"] + delta_offset,
        "p_value": 0.03125,
        "method": "exact_subject_paired",
        "p_value_holm_family": 0.03125,
        "p_value_holm_primary_family": 0.03125,
        "p_value_holm_joint_block": 0.0625,
        "p_value_holm_metric_block": 0.0625,
        "p_value_holm_global": 1.0,
        "primary_significant": True,
    }
    mcnemar = {
        "status": "tested", "p_value": 1.0, "baseline_only_correct": 0, "comparison_only_correct": 1,
        "p_value_holm_block": 1.0, "p_value_holm_primary_family": 1.0, "p_value_holm_global": 1.0,
        "primary_significant": False,
    }
    payload = {
        "schema_version": "audiollm.significance_report.v1",
        "family_sha256": None,
        "alpha": 0.05,
        "metrics": ["macro_f1"],
        "primary_metric": "macro_f1",
        "results": {"blocks": [{
            "id": "synthetic_block",
            "comparisons": [{
                "id": "synthetic|A vs B",
                "dataset": "d3tec",
                "n_subjects": 4,
                "n_seeds": 1,
                "correction_family": "F1|backbone|dataset=d3tec",
                "baseline_files": [str(baseline)],
                "baseline_file_sha256": {str(baseline): paired_significance.sha256_file(baseline)},
                "comparison_files": [str(comparison)],
                "comparison_file_sha256": {str(comparison): paired_significance.sha256_file(comparison)},
                "metrics": {"macro_f1": {"permutation": permutation, "bootstrap": {"ci_low": -0.1, "ci_high": 0.5}}},
                "mcnemar": mcnemar,
            }],
        }]},
    }
    return payload, baseline, comparison


def _write_report_fixture(tmp_path: Path, payload: dict) -> tuple[Path, Path, Path]:
    family_path = tmp_path / "family.yaml"
    family_path.write_text(yaml.safe_dump({
        "schema_version": "audiollm.significance_family.v1", "alpha": 0.05, "families": [],
        "excluded": ["literature rows are not pairable"],
    }), encoding="utf-8")
    payload["family_sha256"] = paired_significance.sha256_file(family_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return family_path, evidence_path, report_path


def _run_report_export(tmp_path: Path, monkeypatch, family: Path, evidence: Path, report: Path) -> int:
    monkeypatch.setattr(sys, "argv", [
        "paired_significance.py", "--family", str(family), "--evidence", str(evidence),
        "--from-report", str(report), "--report-export", str(tmp_path / "export"),
    ])
    return paired_significance.main()


def test_report_export_writes_tables_and_checks_delta(tmp_path: Path, monkeypatch):
    payload, _, _ = _report_fixture(tmp_path)
    family, evidence, report = _write_report_fixture(tmp_path, payload)
    assert _run_report_export(tmp_path, monkeypatch, family, evidence, report) == 0
    out = tmp_path / "export"
    for name in ("significance_full.csv", "family_audit.csv", "coverage_report.csv",
                 "results_table.csv", "paired_subjects.csv", "report_export.json"):
        assert (out / name).is_file(), name
    metadata = json.loads((out / "report_export.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "audiollm.report_export.v1"
    assert metadata["delta_checks"] == {"matched": 1, "mismatch": 0}
    assert metadata["comparisons"] == 1 and metadata["paired_rows"] == 4

    results = list(csv.DictReader((out / "results_table.csv").open(encoding="utf-8")))
    assert results[0]["delta_check"] == "matched"
    assert float(results[0]["score_baseline"]) == pytest.approx(0.7333333333)
    assert float(results[0]["score_comparison"]) == pytest.approx(1.0)
    assert float(results[0]["delta"]) == pytest.approx(0.2666666667)
    assert results[0]["adjusted_significant"] == "True"

    paired = list(csv.DictReader((out / "paired_subjects.csv").open(encoding="utf-8")))
    assert [row["subject_id"] for row in paired] == ["s1", "s2", "s3", "s4"]
    assert paired[1]["baseline_prediction"] == "0" and paired[1]["baseline_correct"] == "0"
    assert paired[1]["comparison_correct"] == "1"

    coverage = list(csv.DictReader((out / "coverage_report.csv").open(encoding="utf-8")))
    assert [row["status"] for row in coverage] == ["tested", "not_testable"]


def test_report_export_flags_a_delta_mismatch(tmp_path: Path, monkeypatch):
    payload, _, _ = _report_fixture(tmp_path, delta_offset=0.2)
    family, evidence, report = _write_report_fixture(tmp_path, payload)
    assert _run_report_export(tmp_path, monkeypatch, family, evidence, report) == 1
    metadata = json.loads((tmp_path / "export" / "report_export.json").read_text(encoding="utf-8"))
    assert metadata["delta_checks"]["mismatch"] == 1
    results = list(csv.DictReader((tmp_path / "export" / "results_table.csv").open(encoding="utf-8")))
    assert results[0]["delta_check"] == "mismatch"


def test_load_verified_side_maps_invalid_to_the_wrong_class(tmp_path: Path):
    payload, baseline, _ = _report_fixture(tmp_path, invalid=True)
    comparison = payload["results"]["blocks"][0]["comparisons"][0]
    rows = paired_significance.load_verified_side(
        [str(baseline)], comparison["baseline_file_sha256"], "d3tec", "test",
    )
    by_subject = {row["subject_id"]: row for row in rows}
    assert by_subject["s2"]["invalid_output"] is True
    assert by_subject["s2"]["prediction"] == 0  # 1 - label, i.e. wrong for a positive subject
    assert by_subject["s1"]["invalid_output"] is False


def test_load_verified_side_uses_recorded_seeds(tmp_path: Path):
    first = tmp_path / "seed_7.csv"
    second = tmp_path / "seed_1337.csv"
    rows = [{"subject_id": "s1", "label": 1, "prediction": 1}]
    _write_csv(first, rows)
    _write_csv(second, rows)
    loaded = paired_significance.load_verified_side(
        [str(first), str(second)], {}, None, "test", {str(first): 7, str(second): 1337},
    )
    assert sorted(row["seed"] for row in loaded) == [7, 1337]
    with pytest.raises(paired_significance.SignificanceError, match="more than one fold file"):
        paired_significance.load_verified_side([str(first), str(second)], {}, None, "test")
