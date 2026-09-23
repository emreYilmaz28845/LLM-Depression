from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools import build_qwen3omni_daic_comparison as comparison


def _write_reference(root: Path, *, modality: str, macro_f1: float) -> dict:
    reference_dir = root / "outputs/experiment_reports/likelihood_canonical_values"
    reference_dir.mkdir(parents=True, exist_ok=True)
    prediction = root / f"predictions_{modality}.csv"
    prediction.write_text("subject_id,prediction\n300,1\n", encoding="utf-8")
    payload = {
        "schema_version": "audiollm.likelihood_canonical_values.v1",
        "cells": [
            {
                "dataset": "DAIC",
                "modality": modality,
                "aggregation": "single_test_fold",
                "likelihood": {"macro_f1": macro_f1, "positive_f1": 0.5, "uar": 0.6},
                "files": [
                    {
                        "path": str(prediction),
                        "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
    }
    (reference_dir / "derived_values.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_run(root: Path, *, modality: str, run_name: str, smoke: bool = False) -> Path:
    name = f"smoke_{run_name}" if smoke else run_name
    run_dir = (
        root
        / f"output_model/promptcontext_v1_qwen3omni_likelihood/{modality}/daic/{name}/fold_0"
    )
    (run_dir / "best_model/standalone_eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "fold": 0,
                "evaluation": {"aggregation_level": "subject"},
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
        json.dumps({"slurm_job_id": "123", "event_type": "SUBMITTED"})
        + "\n"
        + json.dumps({"slurm_job_id": "124", "event_type": "TERMINAL"})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "best_model/standalone_eval/metrics_likelihood.json").write_text(
        json.dumps(
            {
                "headline_metrics": {
                    "binary_strict_macro_f1": 0.71,
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


def _build_with_rows(monkeypatch, rows: list[dict], project_root: Path):
    monkeypatch.setattr(comparison, "ROWS", rows)
    return comparison.build(project_root)


def test_reference_and_run_rows_are_read_from_local_evidence(monkeypatch, tmp_path: Path) -> None:
    _write_reference(tmp_path, modality="audio_only", macro_f1=0.539216)
    _write_run(tmp_path, modality="audio_only", run_name="prod_audio_only")
    rows = [
        {
            "kind": "derived_reference",
            "model": "Qwen2-Audio-7B-Instruct",
            "modality": "audio_only",
            "project_root": str(tmp_path),
        },
        {
            "kind": "run",
            "model": "Qwen3-Omni-30B-A3B Thinker",
            "modality": "audio_only",
            "run_dir": "output_model/promptcontext_v1_qwen3omni_likelihood/audio_only/daic/prod_audio_only/fold_0",
            "backend": "likelihood",
            "project_root": str(tmp_path),
        },
    ]
    built, provenance = _build_with_rows(monkeypatch, rows, tmp_path)

    reference, run = built
    assert reference["macro_f1"] == "0.539216"
    assert "derived canonical likelihood evidence" in reference["qualification"]
    assert reference["evidence_path"].endswith("predictions_audio_only.csv")
    assert "older inline prompt" in reference["qualification"]

    assert run["macro_f1"] == "0.710000"
    assert run["positive_f1"] == "0.610000"
    # UAR is recomputed from the confusion matrix and must match the recorded value.
    assert run["uar"] == "0.779135"
    assert run["gpu_shape"] == "8 ranks, per-rank batch 1"
    assert run["activation_offload"] == "cpu"
    assert run["effective_batch"] == "128"
    assert run["trainable_parameters"] == "13369344"
    assert run["attempt_id"].startswith("20260923T000000Z-prod_audio_only")
    assert run["jobs"] == "123,124"
    assert "active = total" in run["notes"]
    assert run["qualification"] == "promptcontext_v1 prompt; locally validated run"
    assert provenance["reference"]["path"] == comparison.DERIVED_REFERENCE


def test_reference_hash_mismatch_fails_closed(monkeypatch, tmp_path: Path) -> None:
    payload = _write_reference(tmp_path, modality="audio_text", macro_f1=0.735279)
    recorded = tmp_path / "predictions_audio_text.csv"
    recorded.write_text("tampered\n", encoding="utf-8")
    assert hashlib.sha256(recorded.read_bytes()).hexdigest() != payload["cells"][0]["files"][0]["sha256"]
    rows = [
        {
            "kind": "derived_reference",
            "model": "Qwen2-Audio-7B-Instruct",
            "modality": "audio_text",
            "project_root": str(tmp_path),
        }
    ]
    with pytest.raises(ValueError, match="derived reference hash mismatch"):
        _build_with_rows(monkeypatch, rows, tmp_path)


def test_missing_run_directory_leaves_the_row_blank(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "kind": "run",
            "model": "Qwen3-Omni-30B-A3B Thinker",
            "modality": "audio_text",
            "run_dir": "output_model/promptcontext_v1_qwen3omni_likelihood/audio_text/daic/absent/fold_0",
            "backend": "likelihood",
            "project_root": str(tmp_path),
        }
    ]
    built, _ = _build_with_rows(monkeypatch, rows, tmp_path)
    row = built[0]
    assert row["macro_f1"] == ""
    assert "run directory not present" in row["notes"]


def test_smoke_runs_are_refused(monkeypatch, tmp_path: Path) -> None:
    _write_run(tmp_path, modality="audio_only", run_name="prod_audio_only", smoke=True)
    rows = [
        {
            "kind": "run",
            "model": "Qwen3-Omni-30B-A3B Thinker",
            "modality": "audio_only",
            "run_dir": "output_model/promptcontext_v1_qwen3omni_likelihood/audio_only/daic/smoke_prod_audio_only/fold_0",
            "backend": "likelihood",
            "project_root": str(tmp_path),
        }
    ]
    with pytest.raises(ValueError, match="Refusing to include smoke run"):
        _build_with_rows(monkeypatch, rows, tmp_path)


def test_declared_rows_cover_both_modalities_and_the_reference() -> None:
    kinds = [(row["kind"], row["modality"]) for row in comparison.ROWS]
    assert kinds == [
        ("derived_reference", "audio_only"),
        ("derived_reference", "audio_text"),
        ("run", "audio_only"),
        ("run", "audio_text"),
    ]
    assert all("promptcontext_v1_qwen3omni_likelihood" in row["run_dir"] for row in comparison.ROWS if row["kind"] == "run")
