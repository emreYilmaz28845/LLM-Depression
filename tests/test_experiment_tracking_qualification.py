from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.experiment_tracking import qualification, schemas
from src.experiment_tracking.discovery import discover_runs
from src.experiment_tracking.qualification import (
    STATUS_QUALIFIED,
    STATUS_QUARANTINED_AMBIGUOUS,
    STATUS_REJECTED,
    build_evaluations_record,
    is_reportable,
    qualify_run,
)

from test_experiment_tracking_discovery import (
    MANIFEST_HASH,
    SPLIT_HASH,
    build_standard_run,
    metrics_content,
    write_run_config,
    write_standalone_eval,
)


def _qualify(tmp_path: Path) -> qualification.QualificationResult:
    return qualify_run(discover_runs(tmp_path)[0])


def test_complete_teacher_forced_run_is_qualified_and_reportable(tmp_path: Path) -> None:
    build_standard_run(tmp_path)
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.reasons == ()
    assert len(result.evaluations) == 1
    evaluation = result.evaluations[0]
    assert evaluation.dataset == "daic"
    assert evaluation.split_name == "test"
    assert evaluation.split_protocol == "fixed_train_val_test"
    assert evaluation.checkpoint_role == "best_model"
    assert evaluation.checkpoint_path == "best_model"
    assert evaluation.backend == "original_teacher_forced"
    assert evaluation.evaluation_view == "full_coverage_k4"
    assert evaluation.aggregation == "subject_level"
    assert evaluation.metric_namespace == "headline/binary_strict"
    assert evaluation.metrics_artifact_path == "best_model/standalone_eval/metrics_original_teacher_forced_full_coverage_k4.json"
    assert evaluation.predictions_artifact_path == "best_model/standalone_eval/predictions_subject_level.csv"
    names = {metric.name: metric for metric in evaluation.metrics}
    assert names["positive_f1"].value == 0.718
    assert names["positive_f1"].support == 47
    assert evaluation.reportable
    assert evaluation.reportability_issues == ()
    assert evaluation.evaluation_id.startswith("eval-")
    assert evaluation.attempt_id.startswith("legacy-")
    ok, errors = schemas.validate_evaluations(build_evaluations_record(result))
    assert ok, errors


def test_complete_run_evaluation_ids_are_deterministic(tmp_path: Path) -> None:
    build_standard_run(tmp_path)
    first = _qualify(tmp_path).evaluations[0].evaluation_id
    second = _qualify(tmp_path).evaluations[0].evaluation_id
    assert first == second


def test_missing_predictions_is_qualified_but_not_reportable(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    (fold_dir / "best_model" / "standalone_eval" / "predictions_subject_level.csv").unlink()
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    evaluation = result.evaluations[0]
    assert evaluation.predictions_artifact_path is None
    assert not evaluation.reportable
    assert "missing predictions artifact" in evaluation.reportability_issues
    assert "missing subject-level predictions artifact" in evaluation.warnings


def test_best_model_preferred_and_last_model_never_substituted(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics_original_teacher_forced_fixed_k4.json"],
        contents=[metrics_content(view="fixed_k4")],
        location="last_model/standalone_eval",
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.evaluations[0].evaluation_view == "full_coverage_k4"
    assert any("last_model evaluation evidence ignored" in warning for warning in result.warnings)


def test_only_last_model_evidence_is_not_qualified_as_best(tmp_path: Path) -> None:
    import shutil

    fold_dir = build_standard_run(tmp_path)
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics_original_teacher_forced_fixed_k4.json"],
        contents=[metrics_content(view="fixed_k4")],
        location="last_model/standalone_eval",
    )
    shutil.rmtree(fold_dir / "best_model")
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.evaluations == ()
    assert any("last_model evidence never substituted" in warning for warning in result.warnings)


def test_two_daic_views_produce_separate_evaluations(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    target = fold_dir / "best_model" / "standalone_eval"
    for name in ("metrics_original_teacher_forced_full_coverage_k4.json",):
        (target / name).unlink()
    write_standalone_eval(
        fold_dir,
        metrics_files=[
            "metrics_original_teacher_forced_full_coverage_k4.json",
            "metrics_original_teacher_forced_fixed_k4.json",
        ],
        contents=[metrics_content(view="full_coverage_k4"), metrics_content(view="fixed_k4")],
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    views = sorted(evaluation.evaluation_view for evaluation in result.evaluations)
    assert views == ["fixed_k4", "full_coverage_k4"]
    assert result.evaluations[0].evaluation_id != result.evaluations[1].evaluation_id


def test_pooled_and_fold_mean_summaries_are_separate_evaluations(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    summary = {
        "active_backend": "original_teacher_forced",
        "active_backend_pooled_metrics": {
            "accuracy": 0.76,
            "precision": 0.56,
            "recall": 1.0,
            "positive_f1": 0.718,
            "macro_f1": 0.759,
            "weighted_f1": 0.776,
        },
        "active_backend_metric_summary": {
            "accuracy": {"mean": 0.72, "std": 0.04},
            "positive_f1": {"mean": 0.70, "std": 0.05},
        },
        "active_backend_summary_row": {
            "folds": 1,
            "active_backend": "original_teacher_forced",
            "pooled_support_negative": 33,
            "pooled_support_positive": 14,
            "pooled_positive_f1": 0.718,
        },
    }
    (fold_dir / "final_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    aggregations = [evaluation.aggregation for evaluation in result.evaluations]
    assert aggregations.count("subject_level") == 1
    assert aggregations.count("pooled_subject_level") == 1
    assert aggregations.count("fold_mean") == 1
    pooled = next(e for e in result.evaluations if e.aggregation == "pooled_subject_level")
    assert pooled.metrics[0].support == 47
    fold_mean = next(e for e in result.evaluations if e.aggregation == "fold_mean")
    std_names = {metric.name for metric in fold_mean.metrics}
    assert "positive_f1_std" in std_names
    assert not pooled.reportable
    assert "aggregate summary has no single predictions artifact" in pooled.warnings


def test_ambiguous_duplicate_eval_directories_are_quarantined(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics.json"],
        contents=[metrics_content(view="full_coverage_k4")],
        location="eval/best_checkpoint",
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUARANTINED_AMBIGUOUS
    assert any("multiple_eval_locations" in reason for reason in result.reasons)
    assert result.evaluations == ()


def test_ambiguous_duplicate_metrics_with_same_identity_are_quarantined(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    target = fold_dir / "best_model" / "standalone_eval"
    for name in ("metrics_original_teacher_forced_full_coverage_k4.json",):
        (target / name).unlink()
    (target / "metrics.json").write_text(
        json.dumps(metrics_content(view="full_coverage_k4")), encoding="utf-8"
    )
    (target / "metrics_original_teacher_forced.json").write_text(
        json.dumps(metrics_content(view="full_coverage_k4")), encoding="utf-8"
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUARANTINED_AMBIGUOUS
    assert any("duplicate_metrics_same_identity" in reason for reason in result.reasons)


def test_train_val_protocol_without_held_out_test(tmp_path: Path) -> None:
    fold_dir = tmp_path / "audio_text" / "turkish" / "train_val_run" / "fold_0"
    fold_dir.mkdir(parents=True)
    write_run_config(
        fold_dir,
        dataset="turkish",
        cv_protocol="train_val",
        final_eval_split=None,
        final_eval_partition=None,
    )
    write_standalone_eval(fold_dir)
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    evaluation = result.evaluations[0]
    assert evaluation.split_name == "val"
    assert evaluation.split_protocol == "train_val"
    assert any("no held-out test split" in warning for warning in evaluation.warnings)
    assert not evaluation.reportable
    assert any(
        "evaluation warnings present" in issue for issue in evaluation.reportability_issues
    )


def test_corrupt_metrics_alone_is_rejected(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    target = fold_dir / "best_model" / "standalone_eval"
    for name in list(target.glob("metrics_*.json")):
        name.unlink()
    write_standalone_eval(fold_dir, corrupt_metrics=True)
    result = _qualify(tmp_path)
    assert result.status == STATUS_REJECTED
    assert "metrics_unreadable" in result.reasons


def test_corrupt_metrics_alongside_valid_are_ignored_with_warning(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    target = fold_dir / "best_model" / "standalone_eval"
    (target / "metrics_original_teacher_forced_full_coverage_k4.json").unlink()
    write_standalone_eval(fold_dir, metrics_files=["metrics.json"], corrupt_metrics=True)
    (target / "metrics_original_teacher_forced_full_coverage_k4.json").write_text(
        json.dumps(metrics_content(view="full_coverage_k4")), encoding="utf-8"
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert len(result.evaluations) == 1
    assert any("unreadable metrics artifact ignored" in warning for warning in result.warnings)


def test_missing_hashes_block_reportability(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    write_run_config(fold_dir, manifest_hash=None, split_hash=None)
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert any("missing manifest hash" in warning for warning in result.warnings)
    assert any("missing split hash" in warning for warning in result.warnings)
    evaluation = result.evaluations[0]
    assert not evaluation.reportable
    assert "missing manifest hash" in evaluation.reportability_issues
    assert "missing split hash" in evaluation.reportability_issues


def test_valid_only_namespace_is_rejected_as_headline(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    target = fold_dir / "best_model" / "standalone_eval"
    for name in list(target.glob("metrics_*.json")):
        name.unlink()
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics_original_teacher_forced.json"],
        contents=[metrics_content(binary_strict=False)],
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_REJECTED
    assert "no_headline_namespace" in result.reasons
    assert any("valid_only" in warning for warning in result.warnings)


def test_is_reportable_applies_job_history_check_only_when_requested() -> None:
    evaluation = qualification.QualifiedEvaluation(
        attempt_id="legacy-legacy-attempt-v1-" + "a" * 24,
        evaluation_id="eval-" + "b" * 24,
        dataset="daic",
        split_name="test",
        split_protocol="fixed_train_val_test",
        checkpoint_role="best_model",
        checkpoint_path="best_model",
        backend="original_teacher_forced",
        evaluation_view="full_coverage_k4",
        aggregation="subject_level",
        metric_namespace="headline/binary_strict",
        metrics_artifact_path="m.json",
        predictions_artifact_path="p.csv",
        metrics=(qualification.QualifiedMetric(name="positive_f1", value=0.7, support=47),),
        locally_verified=False,
        reportable=False,
        reportability_issues=(),
        warnings=(),
    )
    hashes = {"resolved_config": True, "manifest": True, "split": True}
    ok, issues = is_reportable(
        evaluation, hashes_present=hashes, include_job_history_check=True
    )
    assert not ok
    assert "job/resubmit history not recorded" in issues
    ok, issues = is_reportable(
        evaluation, hashes_present=hashes, include_job_history_check=True, legacy_exception=True
    )
    assert ok, issues
    ok, issues = is_reportable(evaluation, hashes_present=hashes)
    assert ok, issues


HARMONIZED_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1"


def _harmonized_run(tmp_path: Path, *, dataset: str = "d3tec") -> Path:
    fold_dir = build_standard_run(tmp_path, run_name="harmonized_run")
    for stale in (fold_dir / "best_model" / "standalone_eval").glob("metrics_*.json"):
        stale.unlink()
    write_run_config(
        fold_dir,
        dataset=dataset,
        split_mode="cv",
        cv_protocol="train_val_test",
        final_eval_split=None,
        final_eval_partition=None,
        recipe_id=HARMONIZED_RECIPE,
    )
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics_original_teacher_forced.json"],
        contents=[metrics_content(view=None)],
    )
    return fold_dir


def test_harmonized_recipe_view_falls_back_to_recipe_identity(tmp_path: Path) -> None:
    _harmonized_run(tmp_path)
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    evaluation = result.evaluations[0]
    assert evaluation.evaluation_view == "harmonized_all_windows_full_coverage"
    assert evaluation.backend == "original_teacher_forced"


def test_harmonized_en_recipe_uses_same_qualifiers_as_native(tmp_path: Path) -> None:
    EN_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1"
    fold_dir = _harmonized_run(tmp_path, dataset="d3tec")
    run_config = yaml.safe_load((fold_dir / "run_config.yaml").read_text(encoding="utf-8"))
    run_config["config"]["recipe_id"] = EN_RECIPE
    (fold_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.reasons == ()
    evaluation = result.evaluations[0]
    assert evaluation.evaluation_view == "harmonized_all_windows_full_coverage"
    assert evaluation.backend == "original_teacher_forced"
    assert evaluation.metrics_artifact_path.startswith("best_model/standalone_eval/")


def test_harmonized_recipe_prefers_standalone_eval_location(tmp_path: Path) -> None:
    fold_dir = _harmonized_run(tmp_path)
    in_train = fold_dir / "eval" / "best_checkpoint"
    in_train.mkdir(parents=True)
    (in_train / "metrics_original_teacher_forced.json").write_text(
        json.dumps(metrics_content(view=None)), encoding="utf-8"
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.reasons == ()
    evaluation = result.evaluations[0]
    assert evaluation.metrics_artifact_path.startswith("best_model/standalone_eval/")
    assert any("duplicate evaluation evidence" in warning for warning in result.warnings)


def test_level_suffixed_metrics_copy_is_not_ambiguous(tmp_path: Path) -> None:
    fold_dir = _harmonized_run(tmp_path)
    (fold_dir / "best_model" / "standalone_eval" / "metrics_subject_level_original_teacher_forced.json").write_text(
        json.dumps(metrics_content(view=None)), encoding="utf-8"
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    assert result.reasons == ()
    assert any("level-suffixed metrics copy ignored" in warning for warning in result.warnings)


def test_harmonized_train_val_reports_without_held_out_test_warning(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path, run_name="harmonized_train_val_run")
    write_run_config(
        fold_dir,
        dataset="turkish",
        split_mode="cv",
        cv_protocol="train_val",
        final_eval_split=None,
        final_eval_partition=None,
        recipe_id=HARMONIZED_RECIPE,
    )
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics_original_teacher_forced.json"],
        contents=[metrics_content(view=None)],
    )
    result = _qualify(tmp_path)
    assert result.status == STATUS_QUALIFIED
    evaluation = result.evaluations[0]
    assert evaluation.split_protocol == "train_val"
    assert not any("no held-out test split" in warning for warning in evaluation.warnings)
    assert evaluation.reportable
