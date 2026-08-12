from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.experiment_tracking.canonical import read_json, sha256_file
from src.experiment_tracking.identity import evaluation_id
from src.experiment_tracking.lifecycle import read_job_events, read_status
from src.features import gemma4_hidden_campaign as _campaign_reader
from src.features import gemma4_hidden_campaign as campaign

PARENT_ATTEMPT = "20260812T031624Z-gemma4_daic_audio_text_seed1337-a6749b05-146c8805"
MERGE_SHA = "a" * 40

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def _make_parent(tmp_path: Path, modality: str = "audio_text") -> Path:
    fold_dir = tmp_path / "parent" / "fold_0"
    (fold_dir / "best_model").mkdir(parents=True)
    (fold_dir / "logs").mkdir()
    (fold_dir / "run_config.yaml").write_text(
        "manifest_hash: " + campaign.MANIFEST_SHA256 + "\n"
        "split_metadata_hash: " + campaign.SPLIT_SHA256 + "\n"
        "resolved_model_name_or_path: /gpfs/models/gemma-4-12B-it/rev\n",
        encoding="utf-8",
    )
    (fold_dir / "metadata.json").write_text(
        json.dumps({"attempt_id": PARENT_ATTEMPT}), encoding="utf-8"
    )
    (fold_dir / "best_model" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (fold_dir / "best_model" / "adapter_model.safetensors").write_text(
        "adapter", encoding="utf-8"
    )
    return fold_dir


def _expected_adapter_hashes(modality: str) -> dict[str, str]:
    return campaign.EXPECTED_ADAPTER_HASHES[modality]


@pytest.fixture()
def _patch_identity(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "_git_commit", lambda root: MERGE_SHA)
    monkeypatch.setattr(
        campaign,
        "_source_manifest_records",
        lambda root: [{"path": "src/x.py", "sha256": "b" * 64, "size_bytes": 1}],
    )
    monkeypatch.setattr(campaign, "_require_clean_production_source", lambda root: None)
    monkeypatch.setattr(
        campaign,
        "_verify_parent_identity",
        lambda modality, parent_fold_dir: dict(_expected_adapter_hashes(modality)),
    )


def test_create_attempt_writes_full_sidecar_set(tmp_path: Path, _patch_identity) -> None:
    parent = _make_parent(tmp_path)
    attempt_dir = tmp_path / "attempt" / "fold_0"
    result = campaign.create_attempt(
        repo_root=tmp_path,
        attempt_dir=attempt_dir,
        modality="audio_text",
        run_name="gemma4_daic_audio_text_fixed_heads_seed1337",
        group_id="gemma4-daic-fixed-heads-v1-aaaaaaaa",
        parent_fold_dir=parent,
        parent_attempt_id=PARENT_ATTEMPT,
        merged_sha=MERGE_SHA,
        branch="main",
        pr_number=42,
    )
    assert result["attempt_id"].startswith("2026")
    for name in (
        "run_config.yaml",
        "metadata.json",
        "status.json",
        "jobs.jsonl",
        "artifacts.json",
        "evaluations.json",
        "source_manifest.json",
    ):
        assert (attempt_dir / name).is_file(), name
    metadata = read_json(attempt_dir / "metadata.json")
    assert metadata["parent"]["parent_attempt_id"] == PARENT_ATTEMPT
    assert metadata["parent"]["parent_checkpoint_role"] == "best_model"
    assert metadata["parent"]["adapter_sha256"] == campaign.EXPECTED_ADAPTER_HASHES[
        "audio_text"
    ]["adapter_sha256"]
    assert metadata.get("supersedes_attempt_id") is None
    run_config = read_json(attempt_dir / "run_config.yaml")
    assert run_config["config"]["method"] == "gemma4_hidden_fixed_heads"
    assert run_config["config"]["hidden_state"]["dimension"] == 3840
    assert run_config["config"]["hidden_state"]["cache_schema"] == "gemma4_hidden_cache.v1"
    assert run_config["config"]["classifiers"]["variants"] == ["logreg_raw", "xgb_raw"]
    assert run_config["config"]["implementation"]["merged_sha"] == MERGE_SHA
    assert run_config["tracking"]["attempt_id"] == result["attempt_id"]
    assert run_config["config"]["evaluation"]["support"] == 47
    assert run_config["manifest_sha256"] == campaign.MANIFEST_SHA256
    assert run_config["split_metadata_hash"] == campaign.SPLIT_SHA256
    status = read_status(attempt_dir / "status.json")
    assert status["state"] == "PLANNED"
    sidecars = campaign._read_sidecars(attempt_dir)
    assert sidecars is not None
    assert sidecars.state == "PLANNED"


def test_create_attempt_refuses_dirty_source(tmp_path: Path, _patch_identity) -> None:
    parent = _make_parent(tmp_path)

    def _dirty(root):
        raise campaign.CampaignError("production source is dirty")

    campaign._require_clean_production_source = _dirty
    with pytest.raises(campaign.CampaignError, match="dirty"):
        campaign.create_attempt(
            repo_root=tmp_path,
            attempt_dir=tmp_path / "attempt" / "fold_0",
            modality="audio_text",
            run_name="gemma4_daic_audio_text_fixed_heads_seed1337",
            group_id="g",
            parent_fold_dir=parent,
            parent_attempt_id=PARENT_ATTEMPT,
            merged_sha=MERGE_SHA,
            branch="main",
            pr_number=None,
        )


def test_create_attempt_refuses_parent_hash_mismatch(tmp_path: Path, _patch_identity) -> None:
    parent = _make_parent(tmp_path)

    def _bad_identity(modality, parent_fold_dir):
        raise campaign.CampaignError(
            f"parent adapter_model.safetensors hash mismatch for {modality}"
        )

    campaign._verify_parent_identity = _bad_identity
    with pytest.raises(campaign.CampaignError, match="adapter_model.safetensors hash mismatch"):
        campaign.create_attempt(
            repo_root=tmp_path,
            attempt_dir=tmp_path / "attempt" / "fold_0",
            modality="audio_text",
            run_name="gemma4_daic_audio_text_fixed_heads_seed1337",
            group_id="g",
            parent_fold_dir=parent,
            parent_attempt_id=PARENT_ATTEMPT,
            merged_sha=MERGE_SHA,
            branch="main",
            pr_number=None,
        )


def test_create_attempt_refuses_parent_config_hash_mismatch(
    tmp_path: Path, _patch_identity
) -> None:
    parent = _make_parent(tmp_path)
    (parent / "run_config.yaml").write_text(
        "manifest_hash: " + "1" * 64 + "\n", encoding="utf-8"
    )
    with pytest.raises(campaign.CampaignError, match="hashes do not match"):
        campaign.create_attempt(
            repo_root=tmp_path,
            attempt_dir=tmp_path / "attempt" / "fold_0",
            modality="audio_text",
            run_name="gemma4_daic_audio_text_fixed_heads_seed1337",
            group_id="g",
            parent_fold_dir=parent,
            parent_attempt_id=PARENT_ATTEMPT,
            merged_sha=MERGE_SHA,
            branch="main",
            pr_number=None,
        )


def _create_attempt(tmp_path: Path, _patch_identity, modality: str = "audio_text") -> Path:
    parent = _make_parent(tmp_path, modality)
    attempt_dir = tmp_path / f"attempt_{modality}" / "fold_0"
    campaign.create_attempt(
        repo_root=tmp_path,
        attempt_dir=attempt_dir,
        modality=modality,
        run_name=f"gemma4_daic_{modality}_fixed_heads_seed1337",
        group_id="gemma4-daic-fixed-heads-v1-aaaaaaaa",
        parent_fold_dir=parent,
        parent_attempt_id=PARENT_ATTEMPT,
        merged_sha=MERGE_SHA,
        branch="main",
        pr_number=42,
    )
    return attempt_dir


def test_sidecars_validate_through_completed_on_mn5(tmp_path: Path, _patch_identity) -> None:
    attempt_dir = _create_attempt(tmp_path, _patch_identity)
    result = campaign.mark_deployed(attempt_dir, reason="deployed")
    assert result["state"] == "DEPLOYED"
    campaign.transition(attempt_dir, "SUBMITTED", reason="submitted")
    campaign.transition(attempt_dir, "RUNNING", reason="started")
    # Simulate MN5 outputs.
    features = attempt_dir / "hidden_features"
    features.mkdir()
    for name in (
        "outer_train.npz",
        "outer_train_rows.jsonl",
        "final_eval.npz",
        "final_eval_rows.jsonl",
        "extraction_metadata.json",
    ):
        (features / name).write_text("x" if name.endswith(".jsonl") else "x", encoding="utf-8")
    classifiers = attempt_dir / "hidden_classifiers"
    classifiers.mkdir()
    for variant in ("logreg_raw", "xgb_raw"):
        vdir = classifiers / variant
        vdir.mkdir(parents=True)
        metrics = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.6,
            "positive_f1": 0.666,
            "negative_f1": 0.8,
            "macro_f1": 0.733,
            "confusion_matrix": [[30, 5], [6, 6]],
        }
        (vdir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (vdir / "result_config.json").write_text("{}", encoding="utf-8")
        (vdir / "classifier_metadata.json").write_text("{}", encoding="utf-8")
        (vdir / "sampling_audit.json").write_text("{}", encoding="utf-8")
        (vdir / "pipeline.joblib").write_text("p", encoding="utf-8")
        (vdir / "predictions_sample_level.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "subject_id": str(300 + i),
                        "label": i % 2,
                        "probability": 0.9 if i % 2 else 0.1,
                        "predicted_class": i % 2,
                        "classifier_aggregation": "mean_depressed_probability_threshold_0_5",
                        "protocol_id": "daic_participant_speech_packed30_v1",
                    }
                )
                for i in range(8)
            )
            + "\n",
            encoding="utf-8",
        )
        (vdir / "predictions_sample_level.csv").write_text("x\n", encoding="utf-8")
        (vdir / "predictions_subject_level.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "subject_id": str(300 + i),
                        "label": i % 2,
                        "prediction": i % 2,
                        "probability": 0.9 if i % 2 else 0.1,
                        "prediction_backend": "gemma4_hidden_logreg_raw"
                        if variant == "logreg_raw"
                        else "gemma4_hidden_xgb_raw",
                    }
                )
                for i in range(8)
            )
            + "\n",
            encoding="utf-8",
        )
        (vdir / "predictions_subject_level.csv").write_text("x\n", encoding="utf-8")
    (classifiers / "variant_summary.json").write_text("[]", encoding="utf-8")
    (classifiers / "variant_summary.csv").write_text("x\n", encoding="utf-8")

    result = campaign.materialize_mn5_evidence(
        attempt_dir, tmp_path / "parent" / "fold_0"
    )
    assert result["state"] == "COMPLETED_ON_MN5"
    artifacts = read_json(attempt_dir / "artifacts.json")
    assert len(artifacts["artifacts"]) == 2 + 5 + 2 * 9 + 2
    assert {a["role"] for a in artifacts["artifacts"]} >= {"run_config", "source_manifest"}
    assert all(
        artifact["exists_on_mn5"] is True
        and artifact["exists_locally"] is False
        and artifact["locally_verified"] is False
        for artifact in artifacts["artifacts"]
    )
    evaluations = read_json(attempt_dir / "evaluations.json")
    assert len(evaluations["evaluations"]) == 2
    backends = {entry["backend"] for entry in evaluations["evaluations"]}
    assert backends == {"gemma4_hidden_logreg_raw", "gemma4_hidden_xgb_raw"}
    for entry in evaluations["evaluations"]:
        assert entry["dataset"] == "daic"
        assert entry["split_name"] == "test"
        assert entry["split_protocol"] == "daic_official_train_fit_locked_test_evaluation"
        assert entry["checkpoint_role"] == "best_model"
        assert entry["evaluation_view"] == "harmonized_all_windows_full_coverage"
        assert entry["aggregation"] == "subject_level"
        assert entry["metric_namespace"] == "headline/binary_strict"
        assert [metric["name"] for metric in entry["metrics"]] == [
            "accuracy",
            "precision",
            "recall",
            "positive_f1",
            "negative_f1",
            "macro_f1",
        ]
        assert all(metric["support"] == 47 for metric in entry["metrics"])
        assert entry["locally_verified"] is False
        assert entry["reportable"] is False

    # Materialization is idempotent; changed content is refused.
    first = read_json(attempt_dir / "evaluations.json")
    second = campaign.materialize_mn5_evidence(attempt_dir, tmp_path / "parent" / "fold_0")
    assert read_json(attempt_dir / "evaluations.json") == first
    tampered = read_json(attempt_dir / "hidden_classifiers" / "logreg_raw" / "metrics.json")
    tampered["macro_f1"] = 0.1
    (attempt_dir / "hidden_classifiers" / "logreg_raw" / "metrics.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(campaign.CampaignError, match="existing backend|refusing to change evaluation"):
        campaign.materialize_mn5_evidence(attempt_dir, tmp_path / "parent" / "fold_0")
    sidecars = campaign._read_sidecars(attempt_dir)
    assert sidecars.state == "COMPLETED_ON_MN5"


def test_two_evaluation_ids_are_stable_and_distinct(tmp_path: Path, _patch_identity) -> None:
    attempt_dir = _create_attempt(tmp_path, _patch_identity)
    parent = tmp_path / "parent" / "fold_0"
    features = attempt_dir / "hidden_features"
    features.mkdir()
    for name in (
        "outer_train.npz",
        "outer_train_rows.jsonl",
        "final_eval.npz",
        "final_eval_rows.jsonl",
        "extraction_metadata.json",
    ):
        (features / name).write_text("x", encoding="utf-8")
    classifiers = attempt_dir / "hidden_classifiers"
    classifiers.mkdir()
    for variant in ("logreg_raw", "xgb_raw"):
        vdir = classifiers / variant
        vdir.mkdir(parents=True)
        (vdir / "metrics.json").write_text(
            json.dumps(
                {
                    "accuracy": 0.8,
                    "precision": 0.75,
                    "recall": 0.6,
                    "positive_f1": 0.666,
                    "negative_f1": 0.8,
                    "macro_f1": 0.733,
                    "confusion_matrix": [[30, 5], [6, 6]],
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "result_config.json",
            "classifier_metadata.json",
            "sampling_audit.json",
            "predictions_sample_level.jsonl",
            "predictions_sample_level.csv",
            "predictions_subject_level.jsonl",
            "predictions_subject_level.csv",
        ):
            (vdir / name).write_text("x", encoding="utf-8")
        (vdir / "pipeline.joblib").write_text("p", encoding="utf-8")
    (classifiers / "variant_summary.json").write_text("[]", encoding="utf-8")
    (classifiers / "variant_summary.csv").write_text("x\n", encoding="utf-8")
    campaign.materialize_mn5_evidence(attempt_dir, parent)
    evaluations = read_json(attempt_dir / "evaluations.json")["evaluations"]
    assert len({entry["evaluation_id"] for entry in evaluations}) == 2
    metadata = read_json(attempt_dir / "metadata.json")
    for entry in evaluations:
        metrics_sha = sha256_file(
            attempt_dir / "hidden_classifiers" / (
                "logreg_raw" if entry["backend"] == "gemma4_hidden_logreg_raw" else "xgb_raw"
            ) / "metrics.json"
        )
        expected = evaluation_id(
            attempt_id=metadata["attempt_id"],
            fold=0,
            dataset="daic",
            split_name="test",
            split_protocol="daic_official_train_fit_locked_test_evaluation",
            checkpoint_role="best_model",
            checkpoint_path=str(parent / "best_model"),
            backend=entry["backend"],
            evaluation_view="harmonized_all_windows_full_coverage",
            aggregation="subject_level",
            metric_namespace="headline/binary_strict",
            metrics_artifact_sha256=metrics_sha,
        )
        assert entry["evaluation_id"] == expected


def test_job_events_append_without_rewriting(tmp_path: Path, _patch_identity) -> None:
    attempt_dir = _create_attempt(tmp_path, _patch_identity)
    campaign.record_job(
        attempt_dir,
        job_key="extract",
        job_type="hidden_extraction",
        event_type="SUBMITTED",
        slurm_job_id="111",
        status="PENDING",
    )
    campaign.record_job(
        attempt_dir,
        job_key="heads",
        job_type="hidden_classifier",
        event_type="SUBMITTED",
        slurm_job_id="222",
        dependency_job_ids=["111"],
        status="PENDING",
    )
    events = read_job_events(attempt_dir / "jobs.jsonl")
    assert len(events) == 2
    assert events[1]["dependency_job_ids"] == ["111"]
    campaign.record_job(
        attempt_dir,
        job_key="extract",
        job_type="hidden_extraction",
        event_type="COMPLETED",
        slurm_job_id="111",
        status="COMPLETED",
    )
    events = read_job_events(attempt_dir / "jobs.jsonl")
    assert len(events) == 3
    assert events[0]["event_id"] != events[2]["event_id"]


def test_failure_and_cancellation_paths_never_complete(tmp_path: Path, _patch_identity) -> None:
    attempt_dir = _create_attempt(tmp_path, _patch_identity)
    campaign.mark_deployed(attempt_dir)
    campaign.transition(attempt_dir, "SUBMITTED", reason="submitted")
    campaign.transition(attempt_dir, "RUNNING", reason="started")
    campaign.transition(attempt_dir, "FAILED", reason="node failure")
    with pytest.raises(ValueError, match="transition"):
        campaign.transition(attempt_dir, "COMPLETED_ON_MN5", reason="ignored")
    sidecars = campaign._read_sidecars(attempt_dir)
    assert sidecars.state == "FAILED"
    campaign.transition(attempt_dir, "SUPERSEDED", reason="retry")
    sidecars = campaign._read_sidecars(attempt_dir)
    assert sidecars.state == "SUPERSEDED"


def test_shell_syntax_and_no_gpu_for_heads(tmp_path: Path) -> None:
    for script in (
        "scripts/run_gemma4_hidden_extract_slurm.sh",
        "scripts/run_gemma4_hidden_heads_slurm.sh",
        "scripts/submit_gemma4_daic_hidden_heads.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", script], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
    heads = (PROJECT_ROOT / "scripts/run_gemma4_hidden_heads_slurm.sh").read_text(
        encoding="utf-8"
    )
    assert "--gres=gpu" not in heads
    extract = (
        PROJECT_ROOT / "scripts/run_gemma4_hidden_extract_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "--gres=gpu:1" in extract
    for script in (
        "scripts/run_gemma4_hidden_extract_slurm.sh",
        "scripts/run_gemma4_hidden_heads_slurm.sh",
    ):
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        assert "HF_HUB_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
        assert "HF_DATASETS_OFFLINE=1" in text
        assert "TOKENIZERS_PARALLELISM=false" in text
        for forbidden in ("huggingface-cli", "pip ", "git clone", "wget ", "curl "):
            assert forbidden not in text, f"{script} contains {forbidden!r}"
    extract_env = (
        PROJECT_ROOT / "scripts/run_gemma4_hidden_extract_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "gemma4_12b_tf5_14_1" in extract_env
    heads_env = (
        PROJECT_ROOT / "scripts/run_gemma4_hidden_heads_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "qwen_mn5_rebuilt" in heads_env
    assert ".deps/qwen_hidden" in heads_env
    submit = (
        PROJECT_ROOT / "scripts/submit_gemma4_daic_hidden_heads.sh"
    ).read_text(encoding="utf-8")
    assert "--dependency=\"afterok:${EXTRACT_JOB_ID}\"" in submit.replace(" ", "")
    assert "sbatch --parsable" in submit


def test_campaign_cli_help_has_all_subcommands(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools/gemma4_hidden_campaign.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for command in (
        "create-attempt",
        "mark-deployed",
        "record-job",
        "transition",
        "materialize-mn5-evidence",
        "verify-local",
    ):
        assert command in result.stdout
