from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.experiment_tracking import discovery
from src.experiment_tracking.canonical import canonical_sha256, sha256_file

MANIFEST_HASH = "m" * 64
SPLIT_HASH = "s" * 64


def write_run_config(
    fold_dir: Path,
    *,
    dataset: str = "daic",
    cv_protocol: str | None = None,
    final_eval_split: str = "test",
    manifest_hash: str | None = MANIFEST_HASH,
    split_hash: str | None = SPLIT_HASH,
    final_eval_partition: str = "test",
    split_mode: str = "fixed",
    recipe_id: str | None = None,
) -> None:
    split = {
        "mode": split_mode,
        "train_partition": "train",
        "selection_partition": "val",
        "final_eval_partition": final_eval_partition,
    }
    if cv_protocol is not None:
        split["cv_protocol"] = cv_protocol
    config = {
        "dataset": dataset,
        "seed": 1337,
        "split": split,
        "evaluation": {
            "sample_prediction_mode": "original_teacher_forced",
            "aggregation_level": "subject",
            "headline_mode": "original_teacher_forced",
        },
    }
    if recipe_id is not None:
        config["recipe_id"] = recipe_id
    top = {
        "fold": 0,
        "base_config_path": "configs/main/daic_audio_text_selposf1_tf.yaml",
        "config": config,
    }
    if final_eval_split:
        top["final_eval_protocol"] = {"final_eval_split_name": final_eval_split}
    if cv_protocol is not None:
        top["cv_protocol"] = cv_protocol
    if manifest_hash is not None:
        top["manifest_hash"] = manifest_hash
    if split_hash is not None:
        top["split_metadata_hash"] = split_hash
    (fold_dir / "run_config.yaml").write_text(yaml.safe_dump(top, sort_keys=False), encoding="utf-8")


def metrics_content(
    *,
    backend: str = "original_teacher_forced",
    aggregation: str = "subject",
    view: str | None = None,
    binary_strict: bool = True,
) -> dict:
    content = {
        "prediction_backend": backend,
        "evaluation_protocol_name": "teacher_forced_label_span",
        "aggregation_level": aggregation,
        "num_units": 47,
        "num_subjects": 47,
        "accuracy": 0.76,
        "precision": 0.56,
        "recall": 1.0,
        "positive_f1": 0.718,
        "macro_f1": 0.759,
        "weighted_f1": 0.776,
    }
    prefix = "binary_strict" if binary_strict else "valid_only"
    content.update(
        {
            f"{prefix}_accuracy": 0.76,
            f"{prefix}_precision": 0.56,
            f"{prefix}_recall": 1.0,
            f"{prefix}_positive_f1": 0.718,
            f"{prefix}_macro_f1": 0.759,
            f"{prefix}_weighted_f1": 0.776,
        }
    )
    if view is not None:
        content["evaluation_view"] = view
    return content


def write_standalone_eval(
    fold_dir: Path,
    *,
    metrics_files: list[str] | None = None,
    contents: list[dict] | None = None,
    predictions: bool = True,
    location: str = "best_model/standalone_eval",
    corrupt_metrics: bool = False,
) -> None:
    target = fold_dir / location
    target.mkdir(parents=True, exist_ok=True)
    if metrics_files is None:
        metrics_files = ["metrics_original_teacher_forced_full_coverage_k4.json"]
    if contents is None:
        contents = [metrics_content(view="full_coverage_k4")]
    for index, name in enumerate(metrics_files):
        if corrupt_metrics and index == 0:
            (target / name).write_text("{ not valid json", encoding="utf-8")
        else:
            content = contents[index] if index < len(contents) else contents[-1]
            (target / name).write_text(json.dumps(content), encoding="utf-8")
    if predictions:
        (target / "predictions_subject_level.csv").write_text("subject_id,prediction\n", encoding="utf-8")
    (target / "eval_config.yaml").write_text("sample_prediction_mode: original_teacher_forced\n", encoding="utf-8")


def build_standard_run(tmp_path: Path, run_name: str = "daic_run") -> Path:
    fold_dir = tmp_path / "audio_text" / "daic" / run_name / "fold_0"
    fold_dir.mkdir(parents=True)
    write_run_config(fold_dir)
    write_standalone_eval(fold_dir)
    (fold_dir / "last_model").mkdir()
    (fold_dir / "logs").mkdir()
    (fold_dir / "logs" / "training_history.json").write_text(
        json.dumps([{"epoch": 0, "train_loss": 0.5}]), encoding="utf-8"
    )
    (fold_dir / "logs" / "selected_checkpoint_selection_metrics.json").write_text(
        json.dumps({"selected_epoch": 4, "selection_metric": "inner_val_positive_f1"}), encoding="utf-8"
    )
    return fold_dir


def test_discover_runs_finds_ordinary_runs_by_structure(tmp_path: Path) -> None:
    build_standard_run(tmp_path, run_name="daic_rotary_k4_seed1337")
    build_standard_run(tmp_path, run_name="cmdc_text_only_run")
    runs = discovery.discover_runs(tmp_path)
    assert len(runs) == 2
    by_name = {run.run_name: run for run in runs}
    run = by_name["daic_rotary_k4_seed1337"]
    assert run.modality == "audio_text"
    assert run.dataset == "daic"
    assert run.fold == 0
    assert run.fold_dir == str(fold_dir := tmp_path / "audio_text" / "daic" / "daic_rotary_k4_seed1337" / "fold_0")
    assert run.run_config_path == f"{run.fold_dir}/run_config.yaml"
    assert run.run_config_parse_ok
    assert run.resolved_config["dataset"] == "daic"
    assert run.protocol["manifest_hash"] == MANIFEST_HASH
    assert run.resolved_config_sha256 == canonical_sha256(run.resolved_config)
    assert run.run_config_file_sha256 == sha256_file(run.run_config_path)


def test_discovery_enumerates_typed_artifacts(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    run = discovery.discover_runs(tmp_path)[0]
    kinds = [(artifact.kind, artifact.location) for artifact in run.artifacts]
    assert ("metrics", "best_model/standalone_eval") in kinds
    assert ("subject_predictions", "best_model/standalone_eval") in kinds
    assert ("eval_config", "best_model/standalone_eval") in kinds
    assert ("checkpoint_dir", "best_model") in kinds
    assert ("checkpoint_dir", "last_model") in kinds
    assert ("training_history", "logs") in kinds
    assert ("selection_metrics", "logs") in kinds
    metrics = [artifact for artifact in run.artifacts if artifact.kind == "metrics"]
    assert all(artifact.sha256 and artifact.size_bytes and artifact.parse_ok for artifact in metrics)
    assert metrics[0].json_content["prediction_backend"] == "original_teacher_forced"


def test_discovery_is_read_only(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    discovery.discover_runs(tmp_path)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert before == after
    assert (fold_dir / "metadata.json").exists() is False
    assert (fold_dir / "status.json").exists() is False


def test_discovery_flags_unparseable_run_config(tmp_path: Path) -> None:
    fold_dir = tmp_path / "text_only" / "cmdc" / "broken_run" / "fold_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "run_config.yaml").write_text(": : : not yaml\n", encoding="utf-8")
    run = discovery.discover_runs(tmp_path)[0]
    assert not run.run_config_parse_ok
    assert run.resolved_config is None
    assert any("run_config" in warning for warning in run.warnings)


def test_discovery_skips_non_numeric_fold_dirs(tmp_path: Path) -> None:
    for name in ("fold_0", "fold_x"):
        fold_dir = tmp_path / "audio_only" / "daic" / "weird_run" / name
        fold_dir.mkdir(parents=True)
        write_run_config(fold_dir)
    runs = discovery.discover_runs(tmp_path)
    assert [run.fold for run in runs] == [0]


def test_discovery_reports_run_config_without_resolved_config_section(tmp_path: Path) -> None:
    fold_dir = tmp_path / "audio_text" / "daic" / "no_config_key" / "fold_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "run_config.yaml").write_text(
        yaml.safe_dump({"fold": 0, "manifest_hash": MANIFEST_HASH}), encoding="utf-8"
    )
    run = discovery.discover_runs(tmp_path)[0]
    assert run.run_config_parse_ok
    assert run.resolved_config is None
    assert run.resolved_config_sha256 is None
    assert any("no resolved config" in warning for warning in run.warnings)
