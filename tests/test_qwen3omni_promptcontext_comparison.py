from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools import build_qwen3omni_promptcontext_comparison as comparison


def _write_reference(
    root: Path,
    *,
    dataset: str,
    modality: str,
    macro_f1: float,
    transcript_condition: str | None = None,
) -> dict:
    reference_dir = root / "outputs/experiment_reports/likelihood_canonical_values"
    reference_dir.mkdir(parents=True, exist_ok=True)
    prediction = root / f"predictions_{dataset}_{modality}.csv".replace(" ", "_")
    prediction.write_text("subject_id,prediction\n300,1\n", encoding="utf-8")
    cell = {
        "dataset": dataset,
        "modality": modality,
        "aggregation": "five_fold_mean",
        "likelihood": {"macro_f1": macro_f1, "positive_f1": 0.5, "uar": 0.6},
        "files": [
            {
                "path": str(prediction),
                "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            }
        ],
    }
    if transcript_condition is not None:
        cell["transcript_condition"] = transcript_condition
    payload = {"schema_version": "audiollm.likelihood_canonical_values.v1", "cells": [cell]}
    (reference_dir / "derived_values.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_fold(
    root: Path,
    *,
    dataset: str,
    modality: str,
    run_name: str,
    fold: int,
    macro_f1: float,
    smoke: bool = False,
) -> Path:
    name = f"smoke_{run_name}" if smoke else run_name
    run_dir = (
        root
        / f"output_model/promptcontext_v1_qwen3omni_likelihood/{modality}/{dataset}/{name}/fold_{fold}"
    )
    (run_dir / "best_model/standalone_eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "fold": fold,
                "evaluation": {"aggregation_level": "response_subject"},
                "selection_protocol": {"metric_name": "inner_val_macro_f1", "metric_mode": "max"},
                "training_strategy": {
                    "strategy": "fsdp",
                    "world_size": 8,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "effective_global_batch_size": 128,
                    "activation_offload": "cpu",
                },
                "model_load_audit": {
                    "total_parameters": 31_732_574_832,
                    "lora_trainable_params": 13_369_344,
                    "num_hidden_layers": 48,
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 768,
                    "hidden_size": 2048,
                },
                "tracking": {"attempt_id": f"20260923T000000Z-{name}-deadbeef-cafebabe"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "evaluations.json").write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "backend": "likelihood",
                        "evaluation_view": "harmonized_all_windows_full_coverage",
                        "aggregation": "subject_level",
                        "checkpoint_role": "best_model",
                        "locally_verified": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "jobs.jsonl").write_text(
        json.dumps({"slurm_job_id": str(100 + fold), "event_type": "SUBMITTED"})
        + "\n"
        + json.dumps({"slurm_job_id": str(200 + fold), "event_type": "TERMINAL"})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "best_model/standalone_eval/metrics_likelihood.json").write_text(
        json.dumps(
            {
                "headline_metrics": {
                    "binary_strict_macro_f1": macro_f1,
                    "binary_strict_positive_f1": 0.61,
                    # UAR of [[14, 5], [5, 23]]: ((23/28) + (14/19)) / 2.
                    "binary_strict_uar": 0.779135,
                    "binary_strict_confusion_matrix": [[14, 5], [5, 23]],
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _build(monkeypatch, rows: list[dict], root: Path, suffix: str = "prod_test"):
    monkeypatch.setattr(comparison, "_rows_for", lambda _suffix, _daic: rows)
    return comparison.build(root, suffix, "fold0_prod_test")


def _reference_row(
    root: Path,
    dataset: str,
    modality: str,
    reference_dataset: str,
    reference_modality: str,
    transcript_condition: str | None = None,
) -> dict:
    return {
        "kind": "derived_reference",
        "model": comparison.QWEN2_MODEL,
        "dataset": dataset,
        "modality": modality,
        "reference_dataset": reference_dataset,
        "reference_modality": reference_modality,
        "transcript_condition": transcript_condition,
        "endpoint": "selected_validation_view",
        "project_root": str(root),
    }


def _run_row(
    root: Path,
    dataset: str,
    modality: str,
    *,
    mode: str = "fold_mean",
    run_name: str | None = None,
) -> dict:
    name = run_name or f"qwen3omni_{dataset}_{modality}_f{{fold}}_prod_test"
    return {
        "kind": "run",
        "model": comparison.OMNI_MODEL,
        "dataset": dataset,
        "modality": modality,
        "endpoint": "selected_validation_view",
        "mode": mode,
        "run_dir_template": (
            "output_model/promptcontext_v1_qwen3omni_likelihood/"
            f"{modality}/{dataset}/{name}/fold_{{fold}}"
        ),
        "backend": "likelihood",
        "project_root": str(root),
    }


def test_reference_and_single_fold_run_rows_are_read_from_local_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    _write_reference(tmp_path, dataset="DAIC", modality="audio_only", macro_f1=0.539216)
    _write_fold(
        tmp_path,
        dataset="daic",
        modality="audio_only",
        run_name="qwen3omni_daic_audio_only_fold0_prod_test",
        fold=0,
        macro_f1=0.71,
    )
    rows = [
        _reference_row(tmp_path, "daic", "audio_only", "DAIC", "audio_only"),
        _run_row(tmp_path, "daic", "audio_only", mode="single_fold", run_name="qwen3omni_daic_audio_only_fold0_prod_test"),
    ]
    built, by_fold, provenance = _build(monkeypatch, rows, tmp_path)
    reference, run = built

    assert reference["macro_f1"] == "0.539216"
    assert "derived canonical likelihood evidence" in reference["qualification"]
    assert reference["evidence_path"].endswith("predictions_DAIC_audio_only.csv")
    assert "older inline prompt" in reference["qualification"]
    assert reference["endpoint"] == "selected_validation_view"

    assert run["macro_f1"] == "0.710000"
    assert run["positive_f1"] == "0.610000"
    assert run["uar"] == "0.779135"
    assert run["gpu_shape"] == "8 ranks, per-rank batch 1"
    assert run["activation_offload"] == "cpu"
    assert run["effective_batch"] == "128"
    assert run["trainable_parameters"] == "13369344"
    assert run["folds_verified"] == "1/1"
    assert run["attempt_id"].startswith("20260923T000000Z-")
    assert run["jobs"] == "100,200"
    assert "active = total" in run["notes"]
    assert "promptcontext_v1 prompt" in run["qualification"]
    assert provenance["reference"]["path"] == comparison.DERIVED_REFERENCE
    assert len(by_fold) == 1 and by_fold[0]["macro_f1"] == "0.710000"


def test_fold_mean_requires_all_five_verified_folds(monkeypatch, tmp_path: Path) -> None:
    for fold, value in enumerate((0.60, 0.62, 0.64, 0.66, 0.68)):
        _write_fold(
            tmp_path,
            dataset="d3tec",
            modality="audio_text",
            run_name=f"qwen3omni_d3tec_audio_text_f{fold}_prod_test",
            fold=fold,
            macro_f1=value,
        )
    rows = [_run_row(tmp_path, "d3tec", "audio_text")]
    built, by_fold, _ = _build(monkeypatch, rows, tmp_path)
    row = built[0]
    assert row["folds_verified"] == "5/5"
    assert row["macro_f1"] == "0.640000"
    assert row["positive_f1"] == "0.610000"
    assert len(by_fold) == 5
    assert [entry["fold"] for entry in by_fold] == ["0", "1", "2", "3", "4"]


def test_partial_fold_coverage_leaves_the_headline_blank(monkeypatch, tmp_path: Path) -> None:
    for fold in (0, 1, 2):
        _write_fold(
            tmp_path,
            dataset="cmdc",
            modality="audio_only",
            run_name=f"qwen3omni_cmdc_audio_only_f{fold}_prod_test",
            fold=fold,
            macro_f1=0.5,
        )
    rows = [_run_row(tmp_path, "cmdc", "audio_only")]
    built, _, _ = _build(monkeypatch, rows, tmp_path)
    row = built[0]
    assert row["macro_f1"] == ""
    assert row["folds_verified"] == "3/5"
    assert "folds without verified metrics ['3', '4']" in row["notes"]


def test_reference_hash_mismatch_fails_closed(monkeypatch, tmp_path: Path) -> None:
    payload = _write_reference(tmp_path, dataset="Turkish", modality="Audio only", macro_f1=0.408934,
                               transcript_condition="native")
    recorded = tmp_path / "predictions_Turkish_Audio_only.csv"
    recorded.write_text("tampered\n", encoding="utf-8")
    assert hashlib.sha256(recorded.read_bytes()).hexdigest() != payload["cells"][0]["files"][0]["sha256"]
    rows = [_reference_row(tmp_path, "turkish", "audio_only", "Turkish", "Audio only", "native")]
    with pytest.raises(ValueError, match="derived reference hash mismatch"):
        _build(monkeypatch, rows, tmp_path)


def test_missing_run_directory_leaves_the_row_blank(monkeypatch, tmp_path: Path) -> None:
    rows = [_run_row(tmp_path, "androids_interview", "audio_text", run_name="qwen3omni_absent_f{fold}_prod_test")]
    built, _, _ = _build(monkeypatch, rows, tmp_path)
    row = built[0]
    assert row["macro_f1"] == ""
    assert row["folds_verified"] == "0/5"
    assert "run directory not present" in row["notes"]


def test_smoke_runs_are_refused(monkeypatch, tmp_path: Path) -> None:
    _write_fold(
        tmp_path,
        dataset="d3tec",
        modality="audio_only",
        run_name="qwen3omni_d3tec_audio_only_smoke_20260923",
        fold=0,
        macro_f1=0.5,
        smoke=True,
    )
    rows = [
        _run_row(
            tmp_path,
            "d3tec",
            "audio_only",
            run_name="smoke_qwen3omni_d3tec_audio_only_smoke_20260923",
        )
    ]
    with pytest.raises(ValueError, match="Refusing to include smoke run"):
        _build(monkeypatch, rows, tmp_path)


def test_declared_rows_cover_the_eight_cells_the_two_daic_rows_and_every_reference() -> None:
    rows = comparison._rows_for("prod_test", "fold0_prod_test")
    runs = [row for row in rows if row["kind"] == "run"]
    references = [row for row in rows if row["kind"] == "derived_reference"]
    assert len(runs) == 10 and len(references) == 10
    cells = {(row["dataset"], row["modality"]) for row in runs}
    assert cells == {
        ("d3tec", "audio_only"),
        ("d3tec", "audio_text"),
        ("androids_interview", "audio_only"),
        ("androids_interview", "audio_text"),
        ("cmdc", "audio_only"),
        ("cmdc", "audio_text"),
        ("turkish", "audio_only"),
        ("turkish", "audio_text"),
        ("daic", "audio_only"),
        ("daic", "audio_text"),
    }
    assert all(
        "promptcontext_v1_qwen3omni_likelihood" in row["run_dir_template"] for row in runs
    )
    assert comparison.REFERENCE_MODALITY_LABELS["turkish"] == {
        "audio_only": "Audio only",
        "audio_text": "Audio + Text",
    }
    turkish_reference = next(
        row for row in references if row["dataset"] == "turkish" and row["modality"] == "audio_only"
    )
    assert turkish_reference["transcript_condition"] == "native"
