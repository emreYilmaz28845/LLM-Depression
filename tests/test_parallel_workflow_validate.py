from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.validate import (
    ValidationError,
    advance_lifecycle,
    finish_gates,
    read_state,
    recompute_strict_headline,
    validate_attempt,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


PRED_ROWS = (
    "subject_id,label,label_text,prediction_backend,prediction,prediction_text\n"
    "301,0,Non-depressed,original_teacher_forced,0,Non-depressed\n"
    "384,1,Depressed,original_teacher_forced,1,Depressed\n"
    "320,0,Non-depressed,original_teacher_forced,1,Depressed\n"
    "365,1,Depressed,original_teacher_forced,-1,INVALID\n"
)


def _build_attempt(tmp_path: Path) -> Path:
    fold = tmp_path / "output_model" / "camp" / "audio_text" / "daic" / "run1" / "fold_0"
    standalone = fold / "best_model" / "standalone_eval"
    metrics_path = _write(standalone / "metrics_original_teacher_forced.json", "")
    preds_path = _write(standalone / "predictions_subject_level.csv", PRED_ROWS)
    recomputed = recompute_strict_headline(preds_path)
    metrics = {
        "binary_strict_macro_f1": recomputed["binary_strict_macro_f1"],
        "binary_strict_positive_f1": recomputed["binary_strict_positive_f1"],
        "binary_strict_accuracy": recomputed["binary_strict_accuracy"],
        "accuracy": recomputed["binary_strict_accuracy"],
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    _write(fold / "run_config.yaml",
           "dataset: daic\nevaluation:\n  sample_prediction_mode: original_teacher_forced\n"
           "  evaluation_view: harmonized_all_windows_full_coverage\n  aggregation_level: subject\n")
    _write(fold / "metadata.json", json.dumps({
        "schema_version": "audiollm.metadata.v1",
        "attempt_id": "20260821T000000Z-run1-abcdef01-12345678",
        "logical_run_name": "run1",
        "fold": 0,
        "seed": 1337,
        "created_at_utc": "2026-08-21T00:00:00.000000Z",
        "source": {"git_commit": "a" * 40, "git_branch": "agent/x", "git_dirty": False,
                   "deployed_source_sha256": "b" * 64},
        "research": {"github_issue": None, "github_pr": None},
        "hashes": {"resolved_config": None, "manifest": None, "split": None},
        "paths": {"run_config": "run_config.yaml", "best_model": "best_model"},
        "wandb": {"sync_status": "NOT_EXPORTED"},
    }))
    _TS = "2026-08-21T00:00:00.000000Z"
    _write(fold / "status.json", json.dumps({
        "schema_version": "audiollm.status.v1",
        "attempt_id": "20260821T000000Z-run1-abcdef01-12345678", "fold": 0,
        "state": "COMPLETED_ON_MN5",
        "updated_at_utc": _TS,
        "history": [
            {"from": "PLANNED", "to": "DEPLOYED", "at_utc": _TS},
            {"from": "DEPLOYED", "to": "SUBMITTED", "at_utc": _TS},
            {"from": "SUBMITTED", "to": "RUNNING", "at_utc": _TS},
            {"from": "RUNNING", "to": "COMPLETED_ON_MN5", "at_utc": _TS},
        ],
    }))
    from src.experiment_tracking.lifecycle import new_job_event
    attempt_id = "20260821T000000Z-run1-abcdef01-12345678"
    train_ev = new_job_event(job_key="train", job_type="train", event_type="COMPLETED",
                             attempt_id=attempt_id, fold=0, slurm_job_id="101", status="COMPLETED")
    train_ev["exit_code"] = "0:0"
    eval_ev = new_job_event(job_key="best_eval", job_type="evaluation", event_type="COMPLETED",
                            attempt_id=attempt_id, fold=0, slurm_job_id="102", status="COMPLETED")
    eval_ev["exit_code"] = "0:0"
    _write(fold / "jobs.jsonl", json.dumps(train_ev) + "\n" + json.dumps(eval_ev) + "\n")
    eval_artifact_rel = "best_model/standalone_eval/metrics_original_teacher_forced.json"
    preds_rel = "best_model/standalone_eval/predictions_subject_level.csv"
    _write(fold / "artifacts.json", json.dumps({
        "schema_version": "audiollm.artifacts.v1",
        "attempt_id": "20260821T000000Z-run1-abcdef01-12345678",
        "fold": 0,
        "artifacts": [
            {"artifact_id": "art-" + "a" * 24, "artifact_type": "metrics", "role": "headline",
             "path": eval_artifact_rel, "sha256": _sha(metrics_path), "locally_verified": True},
            {"artifact_id": "art-" + "b" * 24, "artifact_type": "predictions", "role": "subject_level",
             "path": preds_rel, "sha256": _sha(preds_path), "locally_verified": True},
        ]}))
    _write(fold / "evaluations.json", json.dumps({
        "schema_version": "audiollm.evaluations.v1",
        "attempt_id": "20260821T000000Z-run1-abcdef01-12345678",
        "fold": 0,
        "evaluations": [
            {"evaluation_id": "eval-" + "c" * 24,
             "dataset": "daic", "split_name": "test", "split_protocol": "cv_protocol_train_val",
             "checkpoint_role": "best_model", "checkpoint_path": "best_model",
             "backend": "original_teacher_forced",
             "evaluation_view": "harmonized_all_windows_full_coverage",
             "aggregation": "subject", "metric_namespace": "headline/binary_strict",
             "metrics_artifact_path": eval_artifact_rel,
             "predictions_artifact_path": preds_rel,
             "locally_verified": True, "reportable": True, "warnings": [],
             "metrics": [
                 {"name": "macro_f1", "value": recomputed["binary_strict_macro_f1"]},
                 {"name": "positive_f1", "value": recomputed["binary_strict_positive_f1"]},
             ]},
        ]}))
    return fold


KW = dict(
    expected_attempt_id="20260821T000000Z-run1-abcdef01-12345678",
    expected_dataset="daic",
    expected_evaluation_view="harmonized_all_windows_full_coverage",
    expected_backend="original_teacher_forced",
    expected_aggregation="subject",
)


def test_recompute_strict_counts_invalid_as_wrong(tmp_path):
    preds = tmp_path / "p.csv"
    preds.write_text(PRED_ROWS, encoding="utf-8")
    r = recompute_strict_headline(preds)
    # labels 0,1,0,1; preds 0,1,1,INVALID -> tp=1 fp=1 fn=1 tn=1
    assert r["support"] == 4
    assert r["binary_strict_accuracy"] == 0.5
    assert r["binary_strict_positive_f1"] == 0.5


def test_recompute_strict_counts_invalid_negative_as_false_positive(tmp_path):
    preds = tmp_path / "p.csv"
    preds.write_text(
        "subject_id,label,label_text,prediction_backend,prediction,prediction_text\n"
        "301,0,Non-depressed,original_teacher_forced,-1,INVALID\n"
        "384,1,Depressed,original_teacher_forced,1,Depressed\n",
        encoding="utf-8",
    )
    r = recompute_strict_headline(preds)
    assert r["binary_strict_accuracy"] == 0.5
    assert r["binary_strict_positive_f1"] == pytest.approx(2 / 3)


def test_happy_path_reaches_reportable_stepwise(tmp_path):
    fold = _build_attempt(tmp_path)
    result = validate_attempt(fold, **KW)
    assert result["ok"] is True, result["issues"]
    assert read_state(fold)[0] == "COMPLETED_ON_MN5"
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is True and finish["state"] == "REPORTABLE"
    # history preserved every step
    state, history = read_state(fold)
    transitions = [(h.get("from"), h.get("to")) for h in history]
    assert ("COMPLETED_ON_MN5", "SYNCED_LOCALLY") in transitions
    assert ("SYNCED_LOCALLY", "LOCALLY_VALIDATED") in transitions
    assert ("LOCALLY_VALIDATED", "REPORTABLE") in transitions
    assert state == "REPORTABLE"
    assert "SUPERSEDED" not in [t[1] for t in transitions]


def test_tampered_hash_fails_validation(tmp_path):
    fold = _build_attempt(tmp_path)
    (fold / "best_model/standalone_eval/metrics_original_teacher_forced.json").write_text(
        '{"binary_strict_macro_f1": 0.99}', encoding="utf-8")
    result = validate_attempt(fold, **KW)
    assert result["ok"] is False
    assert any("differs from recorded SHA-256" in i for i in result["issues"])


def test_recompute_mismatch_fails(tmp_path):
    fold = _build_attempt(tmp_path)
    # keep artifact hash consistent but change predictions so recomputation differs
    preds = fold / "best_model/standalone_eval/predictions_subject_level.csv"
    preds.write_text(PRED_ROWS.replace("365,1,Depressed,original_teacher_forced,-1,INVALID",
                                       "365,1,Depressed,original_teacher_forced,1,Depressed"), encoding="utf-8")
    arts = json.loads((fold / "artifacts.json").read_text())
    for a in arts["artifacts"]:
        if a["path"].endswith("predictions_subject_level.csv"):
            a["sha256"] = _sha(preds)
    (fold / "artifacts.json").write_text(json.dumps(arts))
    result = validate_attempt(fold, **KW)
    assert result["ok"] is False
    assert any("recomputed" in i and "differs from recorded" in i for i in result["issues"])


def test_train_time_only_evaluation_cannot_become_reportable(tmp_path):
    fold = _build_attempt(tmp_path)
    import shutil
    shutil.rmtree(fold / "best_model" / "standalone_eval")
    result = validate_attempt(fold, **KW)
    assert result["ok"] is False
    assert any("not an allowed substitute" in i for i in result["issues"])
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is False


def test_missing_qualifier_fails(tmp_path):
    fold = _build_attempt(tmp_path)
    bad = dict(KW, expected_evaluation_view="some_other_view")
    result = validate_attempt(fold, **bad)
    assert result["ok"] is False
    assert any("evaluation_view" in i for i in result["issues"])


def test_cancelled_required_job_blocks_finish(tmp_path):
    fold = _build_attempt(tmp_path)
    from src.experiment_tracking.lifecycle import new_job_event
    attempt_id = "20260821T000000Z-run1-abcdef01-12345678"
    train_ev = new_job_event(job_key="train", job_type="train", event_type="COMPLETED",
                             attempt_id=attempt_id, fold=0, slurm_job_id="101", status="COMPLETED")
    train_ev["exit_code"] = "0:0"
    eval_ev = new_job_event(job_key="best_eval", job_type="evaluation", event_type="CANCELLED",
                            attempt_id=attempt_id, fold=0, slurm_job_id="102", status="CANCELLED")
    eval_ev["exit_code"] = "0:15"
    (fold / "jobs.jsonl").write_text(json.dumps(train_ev) + "\n" + json.dumps(eval_ev) + "\n", encoding="utf-8")
    finish = finish_gates(fold, **KW)
    assert finish["ok"] is False
    assert "best_eval" in finish["next_action"]


def test_lifecycle_jump_refused(tmp_path):
    fold = _build_attempt(tmp_path)
    with pytest.raises(ValidationError, match="invalid lifecycle advancement"):
        advance_lifecycle(fold, "REPORTABLE")


def test_duplicate_evaluation_id_with_changed_content_fails(tmp_path):
    fold = _build_attempt(tmp_path)
    evaluations = json.loads((fold / "evaluations.json").read_text())
    dup = dict(evaluations["evaluations"][0])
    dup["metrics"] = [{"name": "macro_f1", "value": 0.999}]
    evaluations["evaluations"].append(dup)
    (fold / "evaluations.json").write_text(json.dumps(evaluations))
    result = validate_attempt(fold, **KW)
    assert result["ok"] is False
    assert any("idempotency violated" in i for i in result["issues"])


def test_last_model_standalone_refused(tmp_path):
    fold = _build_attempt(tmp_path)
    lm = fold / "last_model" / "standalone_eval"
    lm.mkdir(parents=True)
    (lm / "metrics_original_teacher_forced.json").write_text("{}")
    result = validate_attempt(fold, **KW)
    assert result["ok"] is False
    assert any("last_model" in i for i in result["issues"])
