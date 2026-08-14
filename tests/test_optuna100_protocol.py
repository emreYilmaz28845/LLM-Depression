"""Tests for the harmonized Optuna-100 protocol and the general post-hoc
head-attempt workflow (runbook Task 3)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.features import optuna100_policy as policy
from src.features import posthoc_head_campaign as campaign
from baselines import qwen_hidden_xgb_optuna as optuna_xgb

ROOT = Path(__file__).resolve().parents[1]


class _FakeClassifier:
    def __init__(self, params=None, fixed_params=None, **kwargs):
        self.params = dict(params or {})
        self.fixed_params = dict(fixed_params or {})
        self.params.update(kwargs)

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        rng = np.random.default_rng(7)
        positive = rng.random(len(x))
        return np.stack([1.0 - positive, positive], axis=1)


def make_cache(tmp_path: Path, *, backend: str = "qwen", dataset: str = "cmdc", modality: str = "text_only", fold: int = 0, n_train: int = 24, n_test: int = 6, dim: int = 8) -> tuple[Path, dict]:
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    train_rows = []
    for index in range(n_train):
        train_rows.append(
            {
                "sample_id": f"t{index}",
                "subject_id": f"s{index}",
                "label": index % 2,
                "response_id": f"s{index}::r0" if dataset == "cmdc" else None,
            }
        )
    test_rows = []
    for index in range(n_test):
        test_rows.append(
            {
                "sample_id": f"e{index}",
                "subject_id": f"e{index}",
                "label": index % 2,
            }
        )
    np.savez(cache / "outer_train.npz", vectors=rng.normal(size=(n_train, dim)).astype(np.float32))
    np.savez(cache / "final_eval.npz", vectors=rng.normal(size=(n_test, dim)).astype(np.float32))
    with open(cache / "outer_train_rows.jsonl", "w") as handle:
        for row in train_rows:
            handle.write(json.dumps(row) + "\n")
    with open(cache / "final_eval_rows.jsonl", "w") as handle:
        for row in test_rows:
            handle.write(json.dumps(row) + "\n")
    parent_attempt = "20260814T000000Z-parent_attempt-aaaaaaaa-11111111"
    checkpoint_dir = str(tmp_path / "parent_fold" / "best_model")
    metadata = {
        "dataset": dataset,
        "input_modality": modality,
        "condition": modality,
        "fold": fold,
        "model_backend": "gemma4" if backend == "gemma4" else None,
        "evaluation_provenance": {
            "evaluation_protocol": "table_aligned_outer_validation" if dataset == "cmdc" else "saved_final_evaluation",
            "split_name": "val" if dataset == "cmdc" else "test",
        },
        "cache_config": {
            "schema_version": "gemma4_hidden_cache.v1" if backend == "gemma4" else "qwen_hidden_cache.v2",
            "checkpoint_dir": checkpoint_dir,
            "adapter_config_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "saved_run_config_sha256": "c" * 64,
            "saved_split_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
            "split_metadata_sha256": "f" * 64,
            "parent_attempt_id": parent_attempt,
            "subject_selection_sha256": None,
        },
    }
    with open(cache / "extraction_metadata.json", "w") as handle:
        json.dump(metadata, handle)
    return cache, metadata


def run_study(cache: Path, output: Path, *, target: int = 100) -> dict:
    return optuna_xgb.run_optuna_raw_xgb(
        cache_dir=cache,
        output_dir=output,
        objective_name="macro_f1",
        target_trials=target,
        inner_folds=3,
        seed=1337,
        inner_seed=1337,
        xgb_threads=20,
        experiment_id=policy.EXPERIMENT_ID,
        sampling_mode="none",
        protocol_profile=policy.PROTOCOL_PROFILE,
    )


class TestPolicy:
    def test_search_space_matches_runbook_table(self) -> None:
        space = policy.resolved_search_space()
        assert space["n_estimators"] == {"kind": "int", "low": 100, "high": 1000, "step": 50}
        assert space["learning_rate"] == {"kind": "float", "low": 0.005, "high": 0.2, "log": True}
        assert space["max_depth"] == {"kind": "int", "low": 1, "high": 6}
        assert space["min_child_weight"] == {"kind": "float", "low": 0.5, "high": 20.0, "log": True}
        assert space["subsample"] == {"kind": "float", "low": 0.5, "high": 1.0}
        assert space["colsample_bytree"] == {"kind": "float", "low": 0.1, "high": 1.0}
        assert space["gamma"] == {"kind": "float", "low": 1e-8, "high": 5.0, "log": True}
        assert space["reg_alpha"] == {"kind": "float", "low": 1e-8, "high": 20.0, "log": True}
        assert space["reg_lambda"] == {"kind": "float", "low": 1e-3, "high": 50.0, "log": True}
        assert space["scale_pos_weight"] == {"kind": "float", "low": 0.25, "high": 4.0, "log": True}
        assert set(space) == set(policy.SEARCH_SPACE)

    def test_production_target_enforcement(self) -> None:
        policy.assert_production_target(100)
        with pytest.raises(ValueError, match="exactly 100"):
            policy.assert_production_target(50)
        with pytest.raises(ValueError, match="exactly 100"):
            policy.assert_production_target(150)

    def test_protocol_settings_enforcement(self) -> None:
        policy.assert_protocol_settings(
            inner_folds=3, seed=1337, inner_seed=1337, sampling_mode="none", objective_name="macro_f1"
        )
        with pytest.raises(ValueError, match="inner folds"):
            policy.assert_protocol_settings(
                inner_folds=5, seed=1337, inner_seed=1337, sampling_mode="none", objective_name="macro_f1"
            )
        with pytest.raises(ValueError, match="seed"):
            policy.assert_protocol_settings(
                inner_folds=3, seed=7, inner_seed=1337, sampling_mode="none", objective_name="macro_f1"
            )
        with pytest.raises(ValueError, match="sampling"):
            policy.assert_protocol_settings(
                inner_folds=3, seed=1337, inner_seed=1337, sampling_mode="legacy", objective_name="macro_f1"
            )
        with pytest.raises(ValueError, match="objective"):
            policy.assert_protocol_settings(
                inner_folds=3, seed=1337, inner_seed=1337, sampling_mode="none", objective_name="positive_f1"
            )

    def test_distinct_backend_identities(self) -> None:
        assert policy.prediction_backend(None) == "qwen_hidden_xgb_optuna100"
        assert policy.prediction_backend("qwen2audio") == "qwen_hidden_xgb_optuna100"
        assert policy.prediction_backend("gemma4") == "gemma4_hidden_xgb_optuna100"
        assert (
            policy.prediction_backend("gemma4", merged=True)
            == "gemma4_hidden_xgb_optuna100_symmetric_merged"
        )
        assert (
            policy.prediction_backend(None, merged=True)
            == "qwen_hidden_xgb_optuna100_symmetric_merged"
        )

    def test_protocol_block_records_exact_contract(self) -> None:
        block = policy.protocol_block(
            dataset="cmdc", condition="text_only", modality="text_only",
            fold=2, seed=1337, objective="macro_f1", model_backend="gemma4",
        )
        assert block["protocol_profile"] == "harmonized_optuna100_v1"
        assert block["target_completed_trials"] == 100
        assert block["sampler"] == "TPESampler"
        assert block["sampler_seed"] == block["model_seed"] == block["inner_split_seed"] == 1337
        assert block["inner_folds"] == 3
        assert block["threshold"] == 0.5
        assert block["sampling_mode"] == "none"
        assert block["packages"] == {"optuna": "4.4.0", "xgboost": "2.1.4"}
        assert block["prediction_backend"] == "gemma4_hidden_xgb_optuna100"


class TestOptunaProfileRun:
    def test_rejects_non_100_target(self, tmp_path: Path) -> None:
        cache, _ = make_cache(tmp_path)
        with pytest.raises(ValueError, match="exactly 100"):
            run_study(cache, tmp_path / "out", target=50)

    def test_rejects_wrong_sampling_mode(self, tmp_path: Path) -> None:
        cache, _ = make_cache(tmp_path)
        with pytest.raises(ValueError, match="sampling"):
            optuna_xgb.run_optuna_raw_xgb(
                cache_dir=cache,
                output_dir=tmp_path / "out",
                objective_name="macro_f1",
                target_trials=100,
                inner_folds=3,
                seed=1337,
                xgb_threads=20,
                experiment_id=policy.EXPERIMENT_ID,
                sampling_mode="legacy",
                protocol_profile=policy.PROTOCOL_PROFILE,
            )

    @pytest.mark.parametrize("backend,expected_backend", [
        ("qwen", "qwen_hidden_xgb_optuna100"),
        ("gemma4", "gemma4_hidden_xgb_optuna100"),
    ])
    def test_full_100_trial_study_writes_qualified_evidence(
        self, tmp_path: Path, backend: str, expected_backend: str
    ) -> None:
        cache, _ = make_cache(tmp_path, backend=backend)
        output = tmp_path / "fold_0" / policy.EXPERIMENT_ID
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(optuna_xgb, "_classifier", _FakeClassifier)
            summary = run_study(cache, output)
        assert summary["variant"] == policy.EXPERIMENT_ID
        assert summary["macro_f1"] is not None
        study_config = json.loads((output / "study_config.json").read_text())
        assert study_config["canonical_config"]["protocol_profile"] == "harmonized_optuna100_v1"
        assert study_config["canonical_config"]["prediction_backend"] == expected_backend
        assert study_config["canonical_config"]["search_profile"] == "harmonized_optuna100_v1"
        assert study_config["canonical_config"]["target_trials"] == 100
        assert study_config["canonical_config"]["search_space"]["scale_pos_weight"] == {
            "kind": "float", "low": 0.25, "high": 4.0, "log": True
        }
        metadata = json.loads((output / "classifier_metadata.json").read_text())
        assert metadata["prediction_backend"] == expected_backend
        assert metadata["protocol_profile"] == "harmonized_optuna100_v1"
        assert metadata["completed_trials"] == 100
        assert metadata["model_backend"] == ("gemma4" if backend == "gemma4" else None)
        sample_rows = [
            json.loads(line)
            for line in (output / "predictions_sample_level.jsonl").read_text().splitlines()
        ]
        assert all(row["prediction_backend"] == expected_backend for row in sample_rows)
        trials = (output / "trials.csv").read_text().splitlines()
        assert len(trials) == 101  # header + 100 trials
        assignments = json.loads((output / "inner_subject_assignments.json").read_text())
        subjects = [item["subject_id"] for item in assignments["subjects"]]
        for fold in assignments["folds"]:
            assert not set(fold["train_subject_ids"]) & set(fold["validation_subject_ids"])
            train_labels = {
                item["label"]
                for item in assignments["subjects"]
                if item["subject_id"] in fold["train_subject_ids"]
            }
            val_labels = {
                item["label"]
                for item in assignments["subjects"]
                if item["subject_id"] in fold["validation_subject_ids"]
            }
            assert train_labels == {0, 1}
            assert val_labels == {0, 1}
        validation_coverage = [
            subject
            for fold in assignments["folds"]
            for subject in fold["validation_subject_ids"]
        ]
        assert Counter(validation_coverage) == Counter(subjects)

    def test_resume_identical_partial_study_and_refuse_changed_identity(
        self, tmp_path: Path
    ) -> None:
        cache, _ = make_cache(tmp_path)
        output = tmp_path / "fold_0" / policy.EXPERIMENT_ID
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(optuna_xgb, "_classifier", _FakeClassifier)
            run_study(cache, output)
            completed_before = len(
                [line for line in (output / "trials.csv").read_text().splitlines()[1:] if line]
            )
            # Identical rerun: completes nothing new, same config hash.
            summary = run_study(cache, output)
        completed_after = len(
            [line for line in (output / "trials.csv").read_text().splitlines()[1:] if line]
        )
        assert completed_before == completed_after == 100
        # Changed identity under the same output dir is refused on resume.
        with pytest.raises(ValueError, match="study_config"):
            optuna_xgb.run_optuna_raw_xgb(
                cache_dir=cache,
                output_dir=output,
                objective_name="macro_f1",
                target_trials=100,
                inner_folds=3,
                seed=1337,
                xgb_threads=10,
                experiment_id=policy.EXPERIMENT_ID,
                sampling_mode="none",
                protocol_profile=policy.PROTOCOL_PROFILE,
            )


def _task_spec(tmp_path: Path, cache: Path, parent_attempt: str, parent_fold: Path) -> Path:
    spec = {
        "schema_version": "audiollm.posthoc_head_task.v1",
        "dataset": "cmdc",
        "modality": "text_only",
        "condition": "text_only",
        "fold": 0,
        "seed": 1337,
        "family": "native",
        "backend": "qwen",
        "cache_dir": str(cache),
        "experiment_id": policy.EXPERIMENT_ID,
        "objective": "macro_f1",
        "target_trials": 100,
        "group_id": "native-optuna100-test",
        "logical_run_name": "harmonized_v1_optuna100_cmdc_text_only_seed1337",
        "run_name": "harmonized_v1_optuna100_test_cmdc_text_only",
        "branch": "main",
        "merged_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "github_issue": None,
        "pr": None,
        "parent": {
            "parent_attempt_id": parent_attempt,
            "parent_fold_dir": str(parent_fold),
            "parent_checkpoint_path": str(parent_fold / "best_model"),
            "adapter_config_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
        },
        "evaluation_qualifiers": {
            "dataset": "cmdc",
            "split_name": "val",
            "split_protocol": "table_aligned_outer_validation",
            "evaluation_view": "harmonized_all_windows_full_coverage",
            "aggregation": "subject_level",
            "metric_namespace": "headline/binary_strict",
            "support": None,
        },
    }
    path = tmp_path / "task_spec.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


def _make_completed_study_attempt(tmp_path: Path, cache: Path, parent_attempt: str, parent_fold: Path) -> Path:
    """Create an attempt and fill it with a finished study's artifacts."""
    spec_path = _task_spec(tmp_path, cache, parent_attempt, parent_fold)
    attempt_dir = (
        tmp_path
        / "output_model"
        / "harmonized_v1_optuna100"
        / "text_only"
        / "cmdc"
        / "harmonized_v1_optuna100_test_cmdc_text_only"
        / "fold_0"
        / policy.EXPERIMENT_ID
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(campaign, "_require_clean_production_source", lambda repo_root: None)
    monkeypatch.setattr(
        campaign, "_git_commit",
        lambda repo_root: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    try:
        created = campaign.create_attempt(
            repo_root=ROOT, attempt_dir=attempt_dir, task_spec=spec_path
        )
    finally:
        monkeypatch.undo()
    study_dir = tmp_path / "study_root" / "fold_0" / policy.EXPERIMENT_ID
    with pytest.MonkeyPatch.context() as m:
        m.setattr(optuna_xgb, "_classifier", _FakeClassifier)
        run_study(cache, study_dir)
    for name in campaign.STUDY_ARTIFACT_FILES:
        (study_dir / name).rename(attempt_dir / name)
    return attempt_dir, created["attempt_id"]



def _run_job_events(attempt_dir: Path) -> None:
    campaign.mark_deployed(attempt_dir)
    campaign.record_job(
        attempt_dir,
        job_key="optuna", job_type="hidden_classifier", event_type="SUBMITTED",
        slurm_job_id="1001", status="PENDING",
    )
    campaign.transition(attempt_dir, "SUBMITTED", reason="submitted")
    campaign.transition(attempt_dir, "RUNNING", reason="started")


class TestPosthocAttempt:
    def test_create_attempt_writes_sidecars_and_parent_identity(self, tmp_path: Path) -> None:
        cache, metadata = make_cache(tmp_path)
        parent_attempt = metadata["cache_config"]["parent_attempt_id"]
        parent_fold = tmp_path / "parent_fold"
        parent_fold.mkdir()
        (parent_fold / "metadata.json").write_text(
            json.dumps({"attempt_id": parent_attempt})
        )
        spec_path = _task_spec(tmp_path, cache, parent_attempt, parent_fold)
        attempt_dir = (
            tmp_path
            / "output_model"
            / "harmonized_v1_optuna100"
            / "text_only"
            / "cmdc"
            / "run"
            / "fold_0"
            / policy.EXPERIMENT_ID
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(campaign, "_require_clean_production_source", lambda repo_root: None)
            monkeypatch.setattr(
                campaign, "_git_commit",
                lambda repo_root: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            )
            created = campaign.create_attempt(
                repo_root=ROOT, attempt_dir=attempt_dir, task_spec=spec_path
            )
        assert created["state"] == "PLANNED"
        for name in (
            "run_config.yaml", "metadata.json", "status.json", "jobs.jsonl",
            "artifacts.json", "evaluations.json", "source_manifest.json",
        ):
            assert (attempt_dir / name).is_file()
        metadata_doc = json.loads((attempt_dir / "metadata.json").read_text())
        assert metadata_doc["parent"]["parent_attempt_id"] == parent_attempt
        run_config = json.loads((attempt_dir / "run_config.yaml").read_text())
        assert run_config["config"]["classifier"]["prediction_backend"] == "qwen_hidden_xgb_optuna100"
        assert run_config["config"]["classifier"]["protocol"]["protocol_profile"] == "harmonized_optuna100_v1"
        assert run_config["config"]["cache"]["cache_dir"] == str(cache)
        # Existing destination refused.
        with pytest.raises(FileExistsError):
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(campaign, "_require_clean_production_source", lambda repo_root: None)
                monkeypatch.setattr(
                    campaign, "_git_commit",
                    lambda repo_root: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                )
                campaign.create_attempt(repo_root=ROOT, attempt_dir=attempt_dir, task_spec=spec_path)

    def test_materialize_idempotent_and_refuses_changed_content(self, tmp_path: Path) -> None:
        cache, metadata = make_cache(tmp_path, n_train=24, n_test=6)
        parent_attempt = metadata["cache_config"]["parent_attempt_id"]
        parent_fold = tmp_path / "parent_fold"
        parent_fold.mkdir()
        (parent_fold / "metadata.json").write_text(json.dumps({"attempt_id": parent_attempt}))
        attempt_dir, attempt_id = _make_completed_study_attempt(
            tmp_path, cache, parent_attempt, parent_fold
        )
        _run_job_events(attempt_dir)
        first = campaign.materialize_mn5_evidence(attempt_dir)
        assert first["evaluations"] == 1
        evaluations = json.loads((attempt_dir / "evaluations.json").read_text())
        entry = evaluations["evaluations"][0]
        assert entry["backend"] == "qwen_hidden_xgb_optuna100"
        assert entry["split_protocol"] == "table_aligned_outer_validation"
        assert entry["metrics"][0]["support"] == 6
        assert entry["metrics"][5]["name"] == "macro_f1"
        assert entry["metrics"][5]["value"] is not None
        second = campaign.materialize_mn5_evidence(attempt_dir)
        assert second["evaluations"] == 1
        evaluations_after = json.loads((attempt_dir / "evaluations.json").read_text())
        assert evaluations_after == evaluations
        # Changed content under the same evaluation ID is refused.
        evaluations_after["evaluations"][0]["metrics"][5]["value"] = 0.999
        (attempt_dir / "evaluations.json").write_text(json.dumps(evaluations_after))
        with pytest.raises(campaign.PosthocError, match="refusing to change"):
            campaign.materialize_mn5_evidence(attempt_dir)

    def test_verify_local_recomputes_and_reaches_reportable(self, tmp_path: Path) -> None:
        cache, metadata = make_cache(tmp_path)
        parent_attempt = metadata["cache_config"]["parent_attempt_id"]
        parent_fold = tmp_path / "parent_fold"
        parent_fold.mkdir()
        (parent_fold / "metadata.json").write_text(json.dumps({"attempt_id": parent_attempt}))
        attempt_dir, attempt_id = _make_completed_study_attempt(
            tmp_path, cache, parent_attempt, parent_fold
        )
        _run_job_events(attempt_dir)
        campaign.materialize_mn5_evidence(attempt_dir)
        result = campaign.verify_local(attempt_dir)
        assert result["state"] == "REPORTABLE"
        status = json.loads((attempt_dir / "status.json").read_text())
        assert status["state"] == "REPORTABLE"
        assert [entry["to"] for entry in status["history"]] == [
            "DEPLOYED", "SUBMITTED", "RUNNING", "COMPLETED_ON_MN5",
            "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE",
        ]
        evaluations = json.loads((attempt_dir / "evaluations.json").read_text())
        assert all(entry["reportable"] and entry["locally_verified"] for entry in evaluations["evaluations"])

    def test_failed_attempt_cannot_become_reportable(self, tmp_path: Path) -> None:
        cache, metadata = make_cache(tmp_path)
        parent_attempt = metadata["cache_config"]["parent_attempt_id"]
        parent_fold = tmp_path / "parent_fold"
        parent_fold.mkdir()
        (parent_fold / "metadata.json").write_text(json.dumps({"attempt_id": parent_attempt}))
        attempt_dir, attempt_id = _make_completed_study_attempt(
            tmp_path, cache, parent_attempt, parent_fold
        )
        _run_job_events(attempt_dir)
        campaign.transition(attempt_dir, "FAILED", reason="node failure")
        with pytest.raises((campaign.PosthocError, ValueError)):
            campaign.verify_local(attempt_dir)
        status = json.loads((attempt_dir / "status.json").read_text())
        assert status["state"] == "FAILED"

    def test_posthoc_attempt_discovered_and_imported_as_modern(self, tmp_path: Path) -> None:
        from src.experiment_tracking.discovery import discover_runs
        from src.experiment_tracking.qualification import qualify_run
        from src.experiment_tracking.sidecars import read_modern_sidecars

        cache, metadata = make_cache(tmp_path, n_train=24, n_test=6)
        parent_attempt = metadata["cache_config"]["parent_attempt_id"]
        parent_fold = tmp_path / "parent_fold"
        parent_fold.mkdir()
        (parent_fold / "metadata.json").write_text(json.dumps({"attempt_id": parent_attempt}))
        attempt_dir, attempt_id = _make_completed_study_attempt(
            tmp_path, cache, parent_attempt, parent_fold
        )
        _run_job_events(attempt_dir)
        campaign.materialize_mn5_evidence(attempt_dir)
        campaign.verify_local(attempt_dir)

        scan_root = tmp_path / "output_model"
        runs = discover_runs(scan_root)
        assert len(runs) == 1
        discovered = runs[0]
        assert discovered.fold == 0
        assert discovered.run_name == "harmonized_v1_optuna100_test_cmdc_text_only"
        assert discovered.modality == "text_only"
        assert discovered.dataset == "cmdc"
        assert Path(discovered.fold_dir) == attempt_dir
        sidecars = read_modern_sidecars(discovered.fold_dir)
        assert sidecars is not None
        assert sidecars.state == "REPORTABLE"
        result = qualify_run(discovered)
        assert result.status == "REPORTABLE" or result.status == "QUALIFIED"
        # The parent object is part of the modern metadata.
        assert sidecars.metadata["parent"]["parent_attempt_id"] == parent_attempt


class TestResolver:
    def test_resolver_counts_and_missing_caches(self) -> None:
        import tools.resolve_optuna100_manifest as resolver

        for family, expected in (("native", 126), ("english", 80), ("officialdev", 6)):
            manifest = resolver.resolve(
                family=family,
                run_id="t3_test",
                merged_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                branch="main",
                github_issue=None,
                pr=None,
                require_caches=False,
            )
            assert manifest["study_count"] == expected
            assert manifest["per_backend"] == {"qwen": expected // 2, "gemma4": expected // 2}
            assert manifest["protocol_profile"] == "harmonized_optuna100_v1"

    def test_resolver_require_caches_fails_on_missing_gemma_caches(self) -> None:
        import tools.resolve_optuna100_manifest as resolver

        with pytest.raises(FileNotFoundError, match="missing"):
            resolver.resolve(
                family="native",
                run_id="t3_test",
                merged_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                branch="main",
                github_issue=None,
                pr=None,
                require_caches=True,
            )
