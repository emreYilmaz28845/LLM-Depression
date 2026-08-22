"""Tests for the deterministic native-versus-English study summarizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import tools.summarize_native_en_text_heads as summ


def _attempt_dir(base: Path, name: str, *, macro: float, pos: float, state: str = "REPORTABLE", datasets: list[str] | None = None, predictions: list[dict] | None = None) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state}))
    if datasets is None:
        metrics = [
            {"name": "macro_f1", "value": macro},
            {"name": "positive_f1", "value": pos},
        ]
        evaluations = {
            "schema_version": "audiollm.evaluations.v1",
            "evaluations": [
                {
                    "dataset": "cmdc",
                    "metrics": metrics,
                    "backend": "qwen_hidden_logreg_raw",
                    "evaluation_view": "harmonized_all_windows_full_coverage",
                    "aggregation": "subject_level",
                }
            ],
        }
        (d / "evaluations.json").write_text(json.dumps(evaluations))
    else:
        evaluations = {"evaluations": []}
        rows = []
        for ds in datasets:
            evaluations["evaluations"].append(
                {
                    "dataset": ds,
                    "metrics": [
                        {"name": "macro_f1", "value": macro},
                        {"name": "positive_f1", "value": pos},
                    ],
                }
            )
            for i in range(4):
                rows.append({"dataset": ds, "subject_id": f"{ds}-{i}", "label": i % 2, "prediction": i % 2})
        (d / "evaluations.json").write_text(json.dumps(evaluations))
        (d / "predictions_subject_level.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return d


def _manifest(tmp_path: Path, cells: list[dict]) -> Path:
    doc = {
        "schema_version": summ.MANIFEST_SCHEMA,
        "group_id": "native-en-text-heads-20260822",
        "cells": cells,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _standalone_cell(tmp_path: Path, dataset: str, condition: str, head_seed_macro: dict[int, tuple[float, float]]) -> dict:
    seeds = []
    for seed, (macro, pos) in head_seed_macro.items():
        folds = []
        for fold in range(5):
            d = _attempt_dir(
                tmp_path,
                f"{dataset}-{condition}-{seed}-f{fold}",
                macro=macro + fold * 0.01,
                pos=pos + fold * 0.01,
            )
            folds.append({"fold": fold, "attempt_dir": str(d)})
        seeds.append({"seed": seed, "folds": folds})
    aggregation = (
        "pooled_subject_level" if dataset in summ.POOL_DATASETS else "fold_mean_unweighted"
    )
    return {
        "endpoint": "standalone",
        "dataset": dataset,
        "condition": condition,
        "backbone": "qwen",
        "head": "logreg",
        "aggregation": aggregation,
        "seeds": seeds,
    }


class TestValidation:
    def test_rejects_wrong_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"schema_version": "other", "group_id": "g", "cells": [{}]}))
        with pytest.raises(ValueError, match="schema_version"):
            summ.summarize(path)

    def test_rejects_duplicate_cells(self, tmp_path: Path) -> None:
        cell = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)})
        # english variant missing -> paired check would fail; duplicate triggers first
        cells = [
            _standalone_cell(tmp_path, "d3tec", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)}),
        ]
        manifest = _manifest(tmp_path, [cell])
        with pytest.raises(ValueError, match="missing a condition"):
            summ.summarize(manifest)

    def test_rejects_wrong_aggregation_declaration(self, tmp_path: Path) -> None:
        cell = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)})
        cell["aggregation"] = "pooled_subject_level"
        cells = [
            cell,
            _standalone_cell(tmp_path, "cmdc", "english", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)}),
        ]
        manifest = _manifest(tmp_path, cells)
        with pytest.raises(ValueError, match="aggregation must be"):
            summ.summarize(manifest)

    def test_requires_all_three_seeds(self, tmp_path: Path) -> None:
        cell = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6)})
        cells = [
            cell,
            _standalone_cell(tmp_path, "cmdc", "english", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)}),
        ]
        manifest = _manifest(tmp_path, cells)
        with pytest.raises(ValueError, match="seeds"):
            summ.summarize(manifest)

    def test_non_reportable_refused(self, tmp_path: Path) -> None:
        cell = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)})
        bad = tmp_path / "nonreportable-attempt"
        bad.mkdir()
        (bad / "status.json").write_text(json.dumps({"state": "RUNNING"}))
        cell["seeds"][1]["folds"][0]["attempt_dir"] = str(bad)
        cells = [
            cell,
            _standalone_cell(tmp_path, "cmdc", "english", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)}),
        ]
        manifest = _manifest(tmp_path, cells)
        with pytest.raises(ValueError, match="RUNNING"):
            summ.summarize(manifest)


class TestAggregationHierarchy:
    def test_fold_mean_and_paired_deltas_cmdc(self, tmp_path: Path) -> None:
        # seed-level scores: native 0.50/0.51/0.52; english +0.10 each
        native = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.50, 0.60), 1337: (0.51, 0.61), 2024: (0.52, 0.62)})
        english = _standalone_cell(tmp_path, "cmdc", "english", {7: (0.60, 0.70), 1337: (0.61, 0.71), 2024: (0.62, 0.72)})
        report = summ.summarize(_manifest(tmp_path, [native, english]))
        comp = next(c for c in report["comparisons"] if c["dataset"] == "cmdc")
        n = comp["native"]["macro_f1"]
        assert n["per_seed"] == {"7": pytest.approx(0.52), "1337": pytest.approx(0.53), "2024": pytest.approx(0.54)}
        assert n["mean"] == pytest.approx(0.53)
        assert n["sample_sd"] == pytest.approx(0.01, abs=1e-9)
        d = comp["paired_delta"]["macro_f1"]
        assert d["per_seed"] == {"7": pytest.approx(0.10), "1337": pytest.approx(0.10), "2024": pytest.approx(0.10)}
        assert d["mean"] == pytest.approx(0.10)

    def test_pooled_d3tec_uses_predictions_across_folds(self, tmp_path: Path) -> None:
        seeds = {}
        for seed, macro in ((7, 0.7), (1337, 0.7), (2024, 0.7)):
            folds = []
            for fold in range(5):
                d = tmp_path / f"d3tec-native-{seed}-f{fold}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "status.json").write_text(json.dumps({"state": "REPORTABLE"}))
                rows = [
                    {"dataset": "d3tec", "subject_id": f"s{seed}-f{fold}-{i}", "label": i % 2, "prediction": i % 2}
                    for i in range(10)
                ]
                (d / "predictions_subject_level.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                folds.append({"fold": fold, "attempt_dir": str(d)})
            seeds[seed] = folds
        native = {
            "endpoint": "standalone",
            "dataset": "d3tec",
            "condition": "native",
            "backbone": "qwen",
            "head": "logreg",
            "aggregation": "pooled_subject_level",
            "seeds": [{"seed": s, "folds": f} for s, f in seeds.items()],
        }
        english = _standalone_cell(tmp_path, "d3tec", "english", {7: (0.7, 0.8), 1337: (0.7, 0.8), 2024: (0.7, 0.8)})
        report = summ.summarize(_manifest(tmp_path, [native, english]))
        comp = next(c for c in report["comparisons"] if c["dataset"] == "d3tec")
        # Perfect predictions pool to exactly these values regardless of stored evals.
        assert comp["native"]["macro_f1"]["per_seed"]["7"] == pytest.approx(1.0)

    def test_pooled_duplicate_subject_across_folds_refused(self, tmp_path: Path) -> None:
        folds = []
        for fold in range(5):
            d = tmp_path / f"dup-f{fold}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "status.json").write_text(json.dumps({"state": "REPORTABLE"}))
            rows = [{"dataset": "d3tec", "subject_id": "same", "label": 0, "prediction": 0}]
            (d / "predictions_subject_level.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            folds.append({"fold": fold, "attempt_dir": str(d)})
        native = {
            "endpoint": "standalone",
            "dataset": "d3tec",
            "condition": "native",
            "backbone": "qwen",
            "head": "logreg",
            "aggregation": "pooled_subject_level",
            "seeds": [{"seed": 7, "folds": folds}, {"seed": 1337, "folds": folds}, {"seed": 2024, "folds": folds}],
        }
        english = _standalone_cell(tmp_path, "d3tec", "english", {7: (0.5, 0.5), 1337: (0.5, 0.5), 2024: (0.5, 0.5)})
        with pytest.raises(ValueError, match="more than one held-out fold"):
            summ.summarize(_manifest(tmp_path, [native, english]))

    def test_merged_cv_mean_of_five_datasets(self, tmp_path: Path) -> None:
        def merged_attempt(name: str, macro: float) -> str:
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "status.json").write_text(json.dumps({"state": "REPORTABLE"}))
            evaluations = {
                "evaluations": [
                    {
                        "dataset": ds,
                        "metrics": [
                            {"name": "macro_f1", "value": macro},
                            {"name": "positive_f1", "value": macro},
                        ],
                    }
                    for ds in ["daic", "cmdc", "turkish", "d3tec", "androids_interview"]
                ]
            }
            (d / "evaluations.json").write_text(json.dumps(evaluations))
            return str(d)

        def cell(condition: str, macro: float):
            seeds = []
            for seed in summ.STUDY_SEEDS:
                folds = [
                    {"fold": f, "attempt_dir": merged_attempt(f"mg-{condition}-{seed}-{f}", macro)}
                    for f in range(5)
                ]
                seeds.append({"seed": seed, "folds": folds})
            return {
                "endpoint": "merged_cv",
                "dataset": "",
                "condition": condition,
                "backbone": "gemma4",
                "head": "xgb_optuna100",
                "aggregation": "mean_dataset_fold_mean_unweighted",
                "seeds": seeds,
            }

        report = summ.summarize(_manifest(tmp_path, [cell("native", 0.4), cell("english", 0.6)]))
        comp = report["comparisons"][0]
        assert comp["native"]["macro_f1"]["mean"] == pytest.approx(0.4)
        assert comp["paired_delta"]["positive_f1"]["mean"] == pytest.approx(0.2)


def test_markdown_render_contains_all_comparisons(tmp_path: Path) -> None:
    native = _standalone_cell(tmp_path, "cmdc", "native", {7: (0.5, 0.6), 1337: (0.5, 0.6), 2024: (0.5, 0.6)})
    english = _standalone_cell(tmp_path, "cmdc", "english", {7: (0.55, 0.65), 1337: (0.55, 0.65), 2024: (0.55, 0.65)})
    report = summ.summarize(_manifest(tmp_path, [native, english]))
    md = summ.render_markdown(report)
    assert "| standalone | cmdc | qwen | logreg | macro_f1 |" in md
