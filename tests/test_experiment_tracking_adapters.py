from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_clean_workbook import validate_selected_results
from src.experiment_tracking import adapters, registry
from src.experiment_tracking.discovery import discover_runs
from src.experiment_tracking.qualification import qualify_run

from test_experiment_registry import _import_tree
from test_experiment_tracking_discovery import build_standard_run, write_standalone_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_substep1_translated_output_roots_adapter(tmp_path: Path) -> None:
    native = build_standard_run(tmp_path, run_name="daic_posf1_tf_daic_audio_text_selmacrof1_tf")
    translated_fold = tmp_path / "audio_text" / "daic" / "daic_en_teacher_forced_adapter_no_projector" / "fold_0"
    translated_fold.mkdir(parents=True)
    from test_experiment_tracking_discovery import write_run_config

    write_run_config(translated_fold)
    write_standalone_eval(translated_fold)
    candidates = adapters.discover_translated_runs(tmp_path)
    assert [candidate.run_name for candidate in candidates] == ["daic_en_teacher_forced_adapter_no_projector"]
    assert candidates[0].adapter_type == adapters.ADAPTER_TRANSLATED
    assert candidates[0].dataset == "daic"


def test_substep2_merged_runs_adapter(tmp_path: Path) -> None:
    fold_dir = (
        tmp_path
        / "symmetric_merged"
        / "audio_text"
        / "merged_retrain_randomk_20260805_260064c"
        / "cv"
        / "fold_0"
    )
    heads = fold_dir / "heads"
    (heads / "logreg").mkdir(parents=True)
    (heads / "xgb_fixed").mkdir(parents=True)
    config = {
        "name": "symmetric_merged_audio_text",
        "protocol": "symmetric_merged",
        "modality": "audio_text",
        "seed": 1337,
        "components": [{"dataset": "daic"}, {"dataset": "cmdc"}],
        "heads": {"fixed_xgb": {}, "optuna": {}},
    }
    (fold_dir / "resolved_merged_config.json").write_text(json.dumps(config), encoding="utf-8")
    (heads / "logreg" / "metrics.json").write_text("{}", encoding="utf-8")
    (heads / "xgb_fixed" / "metrics.json").write_text("{}", encoding="utf-8")
    candidates = adapters.discover_merged_runs(tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.adapter_type == adapters.ADAPTER_MERGED
    assert candidate.fold == 0
    assert candidate.metadata["protocol"] == "symmetric_merged"
    assert candidate.metadata["dataset"] == "daic"
    assert candidate.metadata["modality"] == "audio_text"
    assert "fixed_xgb" in candidate.metadata["heads"]
    assert len(candidate.evidence_paths) >= 3


def test_substep3_hidden_classifier_adapter(tmp_path: Path) -> None:
    variant_dir = (
        tmp_path
        / "hidden_classifiers"
        / "daic"
        / "audio_text"
        / "daic_posf1_tf_daic_audio_text_selmacrof1_tf"
        / "fold_0"
        / "xgb_optuna_raw"
    )
    variant_dir.mkdir(parents=True)
    metadata = {
        "dataset": "daic",
        "modality": "audio_text",
        "condition": "audio_text",
        "fold": 0,
        "run_name": "daic_posf1_tf_daic_audio_text_selmacrof1_tf",
        "classifier_variant": "xgb_optuna_raw",
        "classifier_family": "xgb",
        "seed": 1337,
        "objective": "positive_f1",
        "best_value": 0.825,
        "completed_trials": 50,
        "target_trials": 50,
        "search_config_sha256": "c" * 64,
    }
    (variant_dir / "classifier_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (variant_dir / "metrics.json").write_text(json.dumps({"positive_f1": 0.8}), encoding="utf-8")
    (variant_dir / "predictions_subject_level.csv").write_text("a\n", encoding="utf-8")
    candidates = adapters.discover_hidden_classifiers(tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.adapter_type == adapters.ADAPTER_HIDDEN_CLASSIFIER
    assert candidate.fold == 0
    assert candidate.metadata["classifier_variant"] == "xgb_optuna_raw"
    assert candidate.metadata["optuna_not_run"] is False
    assert candidate.quarantine_reasons == ()
    assert len(candidate.evidence_paths) == 3


def test_substep3_optuna_not_run_is_flagged(tmp_path: Path) -> None:
    variant_dir = (
        tmp_path
        / "hidden_classifiers"
        / "daic"
        / "audio_text"
        / "run"
        / "fold_0"
        / "xgb_optuna_raw"
    )
    variant_dir.mkdir(parents=True)
    metadata = {
        "dataset": "daic",
        "modality": "audio_text",
        "run_name": "run",
        "classifier_variant": "xgb_optuna_raw",
        "objective": "positive_f1",
        "best_value": None,
        "completed_trials": 0,
        "target_trials": 50,
    }
    (variant_dir / "classifier_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    candidate = adapters.discover_hidden_classifiers(tmp_path)[0]
    assert candidate.metadata["optuna_not_run"] is True
    assert candidate.quarantine_reasons


def test_substep4_backfill_inventory_quarantines_ambiguous(tmp_path: Path) -> None:
    fold_dir = build_standard_run(tmp_path, run_name="daic_run")
    write_standalone_eval(
        fold_dir,
        metrics_files=["metrics.json"],
        contents=[{"prediction_backend": "original_teacher_forced", "aggregation_level": "subject", "binary_strict_positive_f1": 0.7, "num_units": 47}],
        location="eval/best_checkpoint",
    )
    inventory = adapters.inventory_evidence(tmp_path, tmp_path)
    assert inventory["ordinary"]["fold_runs"] == 1
    assert inventory["ordinary"]["qualified"] == 0
    assert inventory["ordinary"]["quarantined"] == 1
    assert inventory["quarantined_runs"][0]["status"] == "QUARANTINED_AMBIGUOUS"
    report_path = tmp_path / "backfill_report.json"
    adapters.write_inventory_report(inventory, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["backfill_report_path"] == str(report_path)


def test_substep5_selected_export_resolves_explicit_selection(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    connection.close()
    selection_yaml = tmp_path / "selection.yaml"
    selection_yaml.write_text(
        (
            "selections:\n"
            "  - cell: DAIC|Audio + Text\n"
            "    dataset: daic\n"
            "    modality: audio_text\n"
            "    metric: positive_f1\n"
            "    namespace: headline/binary_strict\n"
            "    backend: original_teacher_forced\n"
            "    view: full_coverage_k4\n"
            "    aggregation: subject_level\n"
            "  - cell: CMDC|Audio only\n"
            "    dataset: cmdc\n"
            "    modality: audio_only\n"
            "    metric: positive_f1\n"
            "    namespace: headline/binary_strict\n"
            "    backend: original_teacher_forced\n"
            "    view: full_coverage_k4\n"
            "    aggregation: subject_level\n"
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected_results.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "export_selected_results.py"),
            "--db",
            str(db_path),
            "--selection",
            str(selection_yaml),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    by_cell = {selection["cell"]: selection for selection in payload["selections"]}
    assert by_cell["DAIC|Audio + Text"]["status"] == "selected"
    assert by_cell["DAIC|Audio + Text"]["value"] == 0.718
    assert by_cell["DAIC|Audio + Text"]["provenance"]["evaluation_id"].startswith("eval-")
    assert by_cell["CMDC|Audio only"]["status"] == "legacy_unmigrated"
    assert by_cell["CMDC|Audio only"]["value"] is None
    assert payload["status_counts"]["legacy_unmigrated"] == 1


def test_substep5_workbook_validation_mismatch_is_rejected(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    connection.close()
    selection_yaml = tmp_path / "selection.yaml"
    selection_yaml.write_text(
        (
            "selections:\n"
            "  - cell: DAIC|Audio + Text\n"
            "    dataset: daic\n"
            "    modality: audio_text\n"
            "    metric: positive_f1\n"
            "    namespace: headline/binary_strict\n"
            "    backend: original_teacher_forced\n"
            "    view: full_coverage_k4\n"
            "    aggregation: subject_level\n"
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected_results.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "export_selected_results.py"),
            "--db",
            str(db_path),
            "--selection",
            str(selection_yaml),
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    matching = {("DAIC", "Audio + Text"): 0.718}
    ok, report = validate_selected_results(output, matching)
    assert ok
    assert not any("differs from" in line for line in report)
    mismatched = {("DAIC", "Audio + Text"): 0.5}
    ok, report = validate_selected_results(output, mismatched)
    assert not ok
    assert any("differs from workbook" in line for line in report)
    blank = {("CMDC", "Audio only"): None}
    ok, report = validate_selected_results(output, blank)
    assert ok
    assert any("legacy-unmigrated" in line for line in report)
    assert not any("= 0.0" in line or "= 0 " in line for line in report)


def test_workbook_validation_cli_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    connection, db_path = _import_tree(tmp_path)
    connection.close()
    selection_yaml = tmp_path / "selection.yaml"
    selection_yaml.write_text(
        (
            "selections:\n"
            "  - cell: DAIC|Audio + Text\n"
            "    dataset: daic\n"
            "    modality: audio_text\n"
            "    metric: positive_f1\n"
            "    namespace: headline/binary_strict\n"
            "    backend: original_teacher_forced\n"
            "    view: full_coverage_k4\n"
            "    aggregation: subject_level\n"
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected_results.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "export_selected_results.py"),
            "--db",
            str(db_path),
            "--selection",
            str(selection_yaml),
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_clean_workbook.py"),
            "--validate-selected",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1
    assert "differs from workbook" in result.stdout
