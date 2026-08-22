"""Tests for the LogReg head-attempt campaign (standalone + merged)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.features import logreg_head_campaign as campaign
from src.features.posthoc_head_campaign import PosthocError


def _git_init(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "t@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name", "t"], check=True
    )
    (repo_root / ".provenance").mkdir(exist_ok=True)
    (repo_root / ".provenance" / "git_commit.txt").write_text("a" * 40)
    (repo_root / ".provenance" / "source_manifest.json").write_text(json.dumps({"files": []}))


def _standalone_cache(cache_dir: Path, *, rows: int = 20) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(cache_dir / "outer_train.npz", vectors=rng.normal(size=(rows, 8)).astype(np.float32))
    np.savez(cache_dir / "final_eval.npz", vectors=rng.normal(size=(rows, 8)).astype(np.float32))
    for name in ("outer_train", "final_eval"):
        with open(cache_dir / f"{name}_rows.jsonl", "w") as handle:
            for index in range(rows):
                handle.write(
                    json.dumps(
                        {
                            "sample_id": f"{name}-{index}",
                            "subject_id": f"s{index}",
                            "label": index % 2,
                        }
                    )
                    + "\n"
                )
    (cache_dir / "extraction_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "cmdc",
                "input_modality": "text_only",
                "model_backend": "",
                "hidden_dimension": 8,
                "manifest_sha256": "d3" * 32,
                "split_metadata_sha256": "5e" * 32,
                "cache_config": {
                    "schema_version": "qwen_hidden_features_v2",
                    "model_name_or_path": "Qwen2-7B-Instruct",
                    "checkpoint_dir": "/gpfs/x/fold_0/best_model",
                    "adapter_config_sha256": "ac" * 32,
                    "adapter_sha256": "ad" * 32,
                },
                "evaluation_provenance": {"evaluation_protocol": ""},
            }
        )
    )


def _spec(tmp_path: Path, *, family: str = "standalone", backend: str = "") -> dict:
    return {
        "schema_version": campaign.TASK_SCHEMA_VERSION,
        "dataset": "merged" if family == "merged" else "cmdc",
        "modality": "text_only",
        "condition": "native_qwen" if not backend else f"native_{backend}",
        "fold": 0,
        "seed": 1337,
        "head_seed": 1337,
        "family": family,
        "backend": backend or "qwen",
        "stage": "cv" if family == "merged" else None,
        "cache_dir": str(tmp_path / "cache"),
        "experiment_id": campaign.LOGREG_RAW_EXPERIMENT_ID,
        "group_id": "native-en-text-heads-20260822",
        "run_name": "tnh-native-qwen-s1337",
        "branch": "agent/exp-native-en-text-heads",
        "merged_sha": "a" * 40,
        "parent": {
            "parent_attempt_id": None,
            "parent_fold_dir": "/gpfs/x/run/fold_0",
            "parent_checkpoint_path": "/gpfs/x/run/fold_0/best_model",
        },
        "evaluation_qualifiers": {
            "split_name": "final_eval" if family == "standalone" else "outer_holdout",
            "split_protocol": (
                "symmetric_merged_cv_outer_holdout"
                if family == "merged"
                else "harmonized_inner_holdout"
            ),
            "evaluation_view": "harmonized_all_windows_full_coverage",
            "aggregation": "subject_level",
            "metric_namespace": "headline/binary_strict",
        },
    }


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "task_spec.json"
    path.write_text(json.dumps(spec))
    return path


def _attempt_dir(tmp_path: Path) -> Path:
    return tmp_path / "attempt" / "fold_0" / campaign.LOGREG_RAW_EXPERIMENT_ID


def _fake_variant_outputs(attempt: Path, *, merged: bool) -> None:
    """Write the artifacts a classifier run would produce."""
    rows = []
    for dataset in (("daic", "d3tec") if merged else (None,)):
        for index in range(10):
            row = {
                "subject_id": f"{dataset or 'x'}-{index}",
                "label": index % 2,
                "prediction": index % 2,
            }
            if dataset:
                row["dataset"] = dataset
            rows.append(row)
    fields = sorted(rows[0])
    with open(attempt / "predictions_subject_level.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    import csv

    with open(attempt / "predictions_subject_level.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    from src.metrics import classification_metrics

    def metrics_for(selected):
        m = classification_metrics([r["label"] for r in selected], [r["prediction"] for r in selected])
        tn, fp = m["confusion_matrix"][0]
        fn, tp = m["confusion_matrix"][1]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        m["positive_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return m

    if merged:
        per_dataset = {}
        for dataset in ("daic", "d3tec"):
            per_dataset[dataset] = metrics_for([r for r in rows if r["dataset"] == dataset])
        (attempt / "metrics_by_dataset.json").write_text(json.dumps(per_dataset))
    else:
        (attempt / "metrics.json").write_text(json.dumps(metrics_for(rows)))
    (attempt / "pipeline.joblib").write_bytes(b"pipeline")
    (attempt / "classifier_metadata.json").write_text(
        json.dumps({"prediction_backend": "backend", "seed": 1337})
    )
    if not merged:
        (attempt / "sampling_audit.json").write_text(json.dumps({"policy": "none"}))
        (attempt / "result_config.json").write_text(json.dumps({"variant": "logreg_raw"}))
        for name in ("predictions_sample_level.jsonl", "predictions_sample_level.csv"):
            (attempt / name).write_text("" if name.endswith(".jsonl") else "")


class TestTaskSpecValidation:
    def test_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path)
        spec["schema_version"] = "audiollm.other.v1"
        with pytest.raises(PosthocError, match="unsupported schema_version"):
            campaign.load_task_spec(_write_spec(tmp_path, spec))

    def test_rejects_wrong_experiment_id(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path)
        spec["experiment_id"] = "xgb_optuna100_harmonized_v1"
        with pytest.raises(PosthocError, match="experiment_id"):
            campaign.load_task_spec(_write_spec(tmp_path, spec))

    def test_rejects_non_fixed_head_seed(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path)
        spec["head_seed"] = 7
        with pytest.raises(PosthocError, match="fixed to 1337"):
            campaign.load_task_spec(_write_spec(tmp_path, spec))

    def test_merged_requires_stage(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path, family="merged")
        spec.pop("stage")
        with pytest.raises(PosthocError, match="stage"):
            campaign.load_task_spec(_write_spec(tmp_path, spec))


class TestBackendStrings:
    def test_prediction_backend_matches_locked_names(self) -> None:
        assert campaign.prediction_backend("", merged=False) == "qwen_hidden_logreg_raw"
        assert campaign.prediction_backend("qwen2audio", merged=False) == "qwen_hidden_logreg_raw"
        assert campaign.prediction_backend("gemma4", merged=False) == "gemma4_hidden_logreg_raw"
        assert campaign.prediction_backend("", merged=True) == (
            "qwen_hidden_logreg_raw_symmetric_merged"
        )
        assert campaign.prediction_backend("gemma4", merged=True) == (
            "gemma4_hidden_logreg_raw_symmetric_merged"
        )

    def test_heads_module_constants_agree(self) -> None:
        from src.merged.heads import (
            GEMMA4_LOGREG_RAW_MERGED_BACKEND,
            QWEN_LOGREG_RAW_MERGED_BACKEND,
        )

        assert QWEN_LOGREG_RAW_MERGED_BACKEND == campaign.prediction_backend("", merged=True)
        assert GEMMA4_LOGREG_RAW_MERGED_BACKEND == campaign.prediction_backend(
            "gemma4", merged=True
        )


class TestCreateAttempt:
    def test_standalone_lifecycle_and_run_config(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        _standalone_cache(tmp_path / "cache")
        spec = _spec(tmp_path)
        result = campaign.create_attempt(
            repo_root=repo,
            attempt_dir=_attempt_dir(tmp_path),
            task_spec=_write_spec(tmp_path, spec),
        )
        attempt = Path(result["attempt_dir"])
        assert result["state"] == "PLANNED"
        run_config = yaml.safe_load((attempt / "run_config.yaml").read_text())
        config = run_config["config"]
        assert config["classifier"]["prediction_backend"] == "qwen_hidden_logreg_raw"
        assert config["classifier"]["head_seed"] == 1337
        assert config["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage"
        metadata = json.loads((attempt / "metadata.json").read_text())
        assert metadata["source"]["deployed_source_sha256"]
        # Attempt dir reuse is refused.
        with pytest.raises(Exception):
            campaign.create_attempt(
                repo_root=repo,
                attempt_dir=attempt,
                task_spec=_write_spec(tmp_path, spec),
            )

    def test_head_mismatch_refused(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        _standalone_cache(tmp_path / "cache")
        spec = _spec(tmp_path)
        spec["merged_sha"] = "b" * 40
        with pytest.raises(PosthocError, match="not the deployed SHA"):
            campaign.create_attempt(
                repo_root=repo,
                attempt_dir=_attempt_dir(tmp_path),
                task_spec=_write_spec(tmp_path, spec),
            )


class TestMaterializeAndVerify:
    def test_standalone_materialize_verify_reportable(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        _standalone_cache(tmp_path / "cache")
        spec = _spec(tmp_path)
        created = campaign.create_attempt(
            repo_root=repo,
            attempt_dir=_attempt_dir(tmp_path),
            task_spec=_write_spec(tmp_path, spec),
        )
        attempt = Path(created["attempt_dir"])
        campaign.mark_deployed(attempt)
        campaign.record_job(
            attempt,
            job_key="logreg",
            job_type="hidden_classifier",
            event_type="SUBMITTED",
            slurm_job_id="1001",
            status="PENDING",
        )
        campaign.transition(attempt, "SUBMITTED")
        campaign.transition(attempt, "RUNNING")
        _fake_variant_outputs(attempt, merged=False)
        payload = campaign.materialize_mn5_evidence(attempt)
        assert payload["evaluations"] == 1
        evaluations = json.loads((attempt / "evaluations.json").read_text())["evaluations"]
        assert evaluations[0]["backend"] == "qwen_hidden_logreg_raw"
        verified = campaign.verify_local(attempt)
        assert verified["state"] == "REPORTABLE"

    def test_merged_per_dataset_evaluations(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        rng = np.random.default_rng(1)
        np.savez(cache / "outer_train.npz", vectors=rng.normal(size=(10, 8)).astype(np.float32))
        np.savez(cache / "outer_holdout.npz", vectors=rng.normal(size=(10, 8)).astype(np.float32))
        (cache / "feature_metadata.json").write_text(
            json.dumps(
                {
                    "modality": "text_only",
                    "model_backend": "",
                    "manifest_hash": "d4" * 32,
                    "split_hash": "6a" * 32,
                    "cache_config": {},
                }
            )
        )
        for name in ("outer_train", "outer_holdout"):
            with open(cache / f"{name}_rows.jsonl", "w") as handle:
                for index in range(5):
                    handle.write(
                        json.dumps({"sample_id": f"{name}-{index}", "subject_id": f"s{index}", "label": index % 2})
                        + "\n"
                    )
        spec = _spec(tmp_path, family="merged", backend="gemma4")
        created = campaign.create_attempt(
            repo_root=repo,
            attempt_dir=_attempt_dir(tmp_path),
            task_spec=_write_spec(tmp_path, spec),
        )
        attempt = Path(created["attempt_dir"])
        run_config = yaml.safe_load((attempt / "run_config.yaml").read_text())
        assert (
            run_config["config"]["classifier"]["prediction_backend"]
            == "gemma4_hidden_logreg_raw_symmetric_merged"
        )
        campaign.mark_deployed(attempt)
        campaign.record_job(
            attempt,
            job_key="logreg",
            job_type="hidden_classifier",
            event_type="SUBMITTED",
            slurm_job_id="2001",
            status="PENDING",
        )
        campaign.transition(attempt, "SUBMITTED")
        campaign.transition(attempt, "RUNNING")
        _fake_variant_outputs(attempt, merged=True)
        payload = campaign.materialize_mn5_evidence(attempt)
        assert payload["evaluations"] == 2
        verified = campaign.verify_local(attempt)
        assert verified["state"] == "REPORTABLE"

    def test_tampered_predictions_fail_verification(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        _standalone_cache(tmp_path / "cache")
        spec = _spec(tmp_path)
        created = campaign.create_attempt(
            repo_root=repo,
            attempt_dir=_attempt_dir(tmp_path),
            task_spec=_write_spec(tmp_path, spec),
        )
        attempt = Path(created["attempt_dir"])
        campaign.mark_deployed(attempt)
        campaign.record_job(
            attempt,
            job_key="logreg",
            job_type="hidden_classifier",
            event_type="SUBMITTED",
            slurm_job_id="3001",
            status="PENDING",
        )
        campaign.transition(attempt, "SUBMITTED")
        campaign.transition(attempt, "RUNNING")
        _fake_variant_outputs(attempt, merged=False)
        campaign.materialize_mn5_evidence(attempt)
        # Flip one stored prediction without touching metrics.json; either the
        # artifact hash gate or the metric recomputation must fail closed.
        rows = [json.loads(line) for line in (attempt / "predictions_subject_level.jsonl").read_text().splitlines()]
        rows[0]["prediction"] = int(not rows[0]["prediction"])
        (attempt / "predictions_subject_level.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        with pytest.raises(PosthocError, match="hash mismatch|does not match"):
            campaign.verify_local(attempt)
