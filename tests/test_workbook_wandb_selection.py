from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.experiment_tracking import registry, wandb_export
from src.experiment_tracking.canonical import sha256_file
from src.experiment_tracking.workbook_selection import (
    DEPENDENCY_INVENTORY_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
    WorkbookSelectionError,
    build_dependency_inventory,
    load_selection,
    provenance_key_of,
    resolve_manifest,
    verify_selection_hashes,
)

from test_experiment_tracking_discovery import build_standard_run, write_run_config, write_standalone_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEADERS = [
    "Experiment",
    "Dataset",
    "Modality",
    "Method",
    "Macro-F1",
    "Source run / checkpoint",
    "Aggregation / eval view",
    "Local artifact",
    "Verification (2026-08-06)",
]


def _write_workbook(path: Path, rows: list[list]) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Provenance"
    ws.append(["Provenance — every headline number maps to a run"])
    ws.append(HEADERS)
    for row in rows:
        ws.append(row)
    wb.save(path)


def _write_builder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def build_workbook(): ...\n", encoding="utf-8")


def _write_selection(
    path: Path,
    entries: list[dict],
    workbook_path: Path,
    builder_path: Path,
) -> dict:
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "workbook": {
            "path": workbook_path.name,
            "sha256": sha256_file(workbook_path),
            "builder_path": str(builder_path),
            "builder_sha256": sha256_file(builder_path),
            "provenance_sheet": "Provenance",
        },
        "entries": entries,
    }
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    return selection


def _sync_entry(run_name: str = "daic_run", expected_folds: list[int] | None = None) -> dict:
    return {
        "selection_id": f"standalone|DAIC|Audio + Text|Fine-tuned Qwen-{run_name}",
        "provenance_key": {
            "experiment": "Standalone",
            "dataset": "DAIC",
            "modality": "Audio + Text",
            "method": "Fine-tuned Qwen",
        },
        "source_type": "ordinary_qwen",
        "reason": "canonical_workbook",
        "dependency_policy": "all_contributing_folds",
        "logical_run_name": run_name,
        "attempt_ids": [],
        "expected_folds": expected_folds if expected_folds is not None else [],
        "required_evaluations": [
            {
                "dataset": "daic",
                "namespace": "headline/binary_strict",
                "backend": "original_teacher_forced",
                "view": "full_coverage_k4",
                "aggregation": "subject_level",
                "checkpoint_role": "best_model",
            }
        ],
        "group": "Standalone Qwen",
        "wandb_policy": "sync",
    }


def _base_row(run_name: str = "daic_run") -> list:
    return [
        "Standalone",
        "DAIC",
        "Audio + Text",
        "Fine-tuned Qwen",
        0.840678,
        f"{run_name}/fold_0/best_model",
        "full-coverage K4-bundle view",
        "outputs/daic_k4_coverage_audit/x/coverage_audit.json",
        "recomputed from local artifact",
    ]


def _resolved_units(tmp_path: Path, selection: dict, rows: list[list]) -> dict:
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, rows)
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, selection["entries"], workbook, builder)
    payload = load_selection(selection_path)
    assert verify_selection_hashes(payload, workbook_path=workbook, builder_path=builder) == []
    inventory = build_dependency_inventory(workbook, builder_path=builder)
    return resolve_manifest(payload, inventory, db_path=selection["db"], selection_path=selection_path)


def _import_multi_fold(tmp_path: Path, run_name: str = "daic_run", folds: int = 5) -> Path:
    for fold in range(folds):
        fold_dir = tmp_path / "audio_text" / "daic" / run_name / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        write_run_config(fold_dir)
        write_standalone_eval(fold_dir)
        (fold_dir / "logs").mkdir(exist_ok=True)
        (fold_dir / "logs" / "training_history.json").write_text(
            json.dumps([{"epoch": 0, "train_loss": 0.5}]), encoding="utf-8"
        )
    db_path = tmp_path / "registry.sqlite"
    summary = registry.rebuild_registry(tmp_path, db_path)
    assert summary["imported_runs"] == folds
    return db_path


def _run_tool(tool: str, *args: str, expected_code: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / tool), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={"WANDB_MODE": "offline", **__import__("os").environ},
    )
    assert result.returncode == expected_code, result.stderr + result.stdout
    return result


# --------------------------------------------------------------------------- inventory
def test_provenance_keys_are_stable_and_duplicates_reported(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row(), _base_row()])
    inventory = build_dependency_inventory(workbook, builder_path=tmp_path / "builder.py")
    assert inventory["schema_version"] == DEPENDENCY_INVENTORY_SCHEMA_VERSION
    assert inventory["summary"]["data_rows"] == 2
    assert len(inventory["duplicates"]) == 1
    assert inventory["duplicates"][0]["provenance_key"] == "Standalone|DAIC|Audio + Text|Fine-tuned Qwen"
    assert provenance_key_of(
        inventory["rows"][0]["raw"]
    ) == "Standalone|DAIC|Audio + Text|Fine-tuned Qwen"


def test_malformed_rows_are_reported_not_guessed(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    bad = _base_row()
    bad[2] = None
    _write_workbook(workbook, [bad])
    inventory = build_dependency_inventory(workbook, builder_path=tmp_path / "builder.py")
    assert inventory["summary"]["malformed_rows"] == 1
    assert inventory["malformed"][0]["reason"].startswith("missing key column")
    assert inventory["rows"][0]["provenance_key"] is None


def test_inventory_never_infers_attempt_ids_from_free_text(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    row = _base_row()
    row[5] = "daic_main_k4_control_20260804_f26dd45/fold_0/best_model (A+T); attempt 44394029"
    _write_workbook(workbook, [row])
    inventory = build_dependency_inventory(workbook, builder_path=tmp_path / "builder.py")
    serialized = json.dumps(inventory)
    assert "attempt_ids" not in serialized
    assert "attempt_id" not in serialized
    assert inventory["rows"][0]["raw"]["source_run"] == row[5]
    assert inventory["rows"][0]["source_type"] == "ordinary_qwen"


def test_blank_not_run_rows_are_marked_blank(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    row = _base_row()
    row[4] = None
    row[8] = "optuna not run on retrain checkpoints"
    _write_workbook(workbook, [row])
    inventory = build_dependency_inventory(workbook, builder_path=tmp_path / "builder.py")
    assert inventory["rows"][0]["blank"] is True
    assert "not run" in inventory["rows"][0]["blank_reason"]


# --------------------------------------------------------------------------- selection validation
def test_workbook_hash_mismatch_refuses_resolution(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    selection = _write_selection(selection_path, [_sync_entry()], workbook, builder)
    selection["workbook"]["sha256"] = "f" * 64
    selection_path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    payload = load_selection(selection_path)
    failures = verify_selection_hashes(payload, workbook_path=workbook, builder_path=builder)
    assert failures and "workbook sha256 mismatch" in failures[0]


def test_builder_hash_mismatch_refuses_resolution(tmp_path: Path) -> None:
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    builder.write_text("changed\n", encoding="utf-8")
    payload = load_selection(selection_path)
    failures = verify_selection_hashes(payload, workbook_path=workbook, builder_path=builder)
    assert failures and "builder sha256 mismatch" in failures[0]


def test_selection_validation_rejects_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: wrong\nentries: []\n", encoding="utf-8")
    with pytest.raises(WorkbookSelectionError, match="schema_version"):
        load_selection(path)


# --------------------------------------------------------------------------- manifest resolution
def test_single_row_resolves_to_all_contributing_folds(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=5)
    selection = {"db": str(db_path), "entries": [_sync_entry(expected_folds=[0, 1, 2, 3, 4])]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["summary"]["sync_units"] == 5
    assert sorted(unit["fold"] for unit in manifest["export_units"]) == [0, 1, 2, 3, 4]
    for unit in manifest["export_units"]:
        assert unit["export_decision"] == "sync"
        assert unit["reportable"] is True
        assert unit["tags"][0] == "workbook-selected"
        assert "dataset:daic" in unit["tags"]
        assert "source-type:ordinary_qwen" in unit["tags"]
        assert unit["lifecycle_state"] == "IMPORTED_LEGACY"
    assert manifest["summary"]["unresolved_rows"] == 0


def test_multiple_rows_deduplicate_to_one_wandb_run(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=5)
    entry = _sync_entry(expected_folds=[0, 1, 2, 3, 4])
    second = dict(entry)
    second["selection_id"] = "en-native|DAIC|Audio + Text|Native (fold-mean)"
    second["provenance_key"] = {
        "experiment": "EN translation",
        "dataset": "DAIC",
        "modality": "Audio + Text",
        "method": "Native (fold-mean)",
    }
    selection = {"db": str(db_path), "entries": [entry, second]}
    rows = [_base_row(), ["EN translation", "DAIC", "Audio + Text", "Native (fold-mean)", 0.84,
                          "daic_run", "5-fold CV mean", "output_model/x/final_summary.json", "recomputed"]]
    manifest = _resolved_units(tmp_path, selection, rows)
    assert manifest["summary"]["sync_units"] == 5
    assert manifest["summary"]["deduplicated_units"] == 5
    for unit in manifest["export_units"]:
        assert len(unit["selection_ids"]) == 2
        assert len(unit["provenance_keys"]) == 2


def test_pooled_fold_mean_rows_never_create_synthetic_runs(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=5)
    entry = _sync_entry(expected_folds=[0, 1, 2, 3, 4])
    entry["wandb_policy"] = "skip_derived_only"
    entry["blocking_reasons"] = ["pooled aggregate; no synthetic training run"]
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["export_units"] == []
    assert manifest["entries"][0]["status"] == "skip_derived_only"


def test_hidden_and_merged_rows_stay_pending_without_importer(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["source_type"] = "hidden_classifier"
    entry["wandb_policy"] = "pending_importer_support"
    entry["blocking_reasons"] = ["hidden-classifier adapter inventories only"]
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["entries"][0]["status"] == "pending_importer_support"


def test_pending_wandb_reconciliation_rows_produce_no_export_units(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["wandb_policy"] = "pending_wandb_reconciliation"
    entry["blocking_reasons"] = [
        "existing cloud run is keyed to a synthetic legacy attempt; "
        "exporting the real attempt would create a duplicate until the "
        "researcher selects a reconciliation strategy"
    ]
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["export_units"] == []
    assert manifest["summary"]["blocked_units"] == 0
    assert manifest["entries"][0]["status"] == "pending_wandb_reconciliation"
    assert manifest["entries"][0]["reasons"] == entry["blocking_reasons"]
    assert manifest["entries"][0]["unit_run_ids"] == []
    assert manifest["summary"]["unresolved_rows"] == 0


def test_mn5_only_rows_stay_pending(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["wandb_policy"] = "pending_local_evidence"
    entry["blocking_reasons"] = ["MN5-only evidence; not synced"]
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["entries"][0]["status"] == "pending_local_evidence"


def test_blank_not_run_rows_stay_skipped(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["wandb_policy"] = "skip_not_run"
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["entries"][0]["status"] == "skip_not_run"


def test_missing_local_evidence_blocks_sync_unit(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    metrics = (
        tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
        / "best_model" / "standalone_eval" / "metrics_original_teacher_forced_full_coverage_k4.json"
    )
    metrics.unlink()
    selection = {"db": str(db_path), "entries": [_sync_entry()]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    unit = manifest["blocked_units"][0]
    assert unit["export_decision"] == "blocked"
    assert any("local evidence" in reason and "missing on disk" in reason for reason in unit["blocking_reasons"])


def test_local_evidence_sha_mismatch_blocks_sync_unit(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    predictions = (
        tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
        / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
    )
    predictions.write_text("subject_id,prediction\nchanged\n", encoding="utf-8")
    selection = {"db": str(db_path), "entries": [_sync_entry()]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert any("sha256 mismatch" in reason for reason in manifest["blocked_units"][0]["blocking_reasons"])


def test_missing_evaluation_qualifier_blocks_sync_unit(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["required_evaluations"][0]["view"] = "fixed_k4"
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert any("no evaluation matching required qualifiers" in reason for reason in manifest["blocked_units"][0]["blocking_reasons"])


def test_ambiguous_logical_run_requires_explicit_attempt_ids(tmp_path: Path) -> None:
    _import_multi_fold(tmp_path, run_name="daic_run", folds=1)
    other = tmp_path / "audio_only" / "daic" / "daic_run" / "fold_0"
    other.mkdir(parents=True)
    write_run_config(other)
    write_standalone_eval(other)
    registry.rebuild_registry(tmp_path, tmp_path / "registry.sqlite")
    db_path = tmp_path / "registry.sqlite"
    connection = registry.connect(db_path)
    try:
        attempts = [row["attempt_id"] for row in connection.execute("SELECT attempt_id FROM run_attempts").fetchall()]
        assert len(attempts) == 2
    finally:
        connection.close()
    entry = _sync_entry()
    selection = {"db": str(db_path), "entries": [entry]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 0
    assert manifest["entries"][0]["status"] == "blocked"
    assert any("multiple logical runs match" in reason for reason in manifest["entries"][0]["reasons"])
    explicit = dict(entry)
    explicit["attempt_ids"] = [attempts[0]]
    selection = {"db": str(db_path), "entries": [explicit]}
    manifest = _resolved_units(tmp_path, selection, [_base_row()])
    assert manifest["summary"]["sync_units"] == 1
    assert manifest["entries"][0]["status"] == "resolved"


def test_resolver_lists_unresolved_rows(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["wandb_policy"] = "skip_not_run"
    selection = {"db": str(db_path), "entries": [entry]}
    workbook = tmp_path / "wb.xlsx"
    uncovered = _base_row()
    uncovered[2] = "Audio only"
    _write_workbook(workbook, [_base_row(), uncovered])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [entry], workbook, builder)
    payload = load_selection(selection_path)
    inventory = build_dependency_inventory(workbook, builder_path=builder)
    manifest = resolve_manifest(payload, inventory, db_path=db_path, selection_path=selection_path)
    assert manifest["summary"]["unresolved_rows"] == 1
    assert manifest["unresolved_entries"][0]["provenance_key"] == "Standalone|DAIC|Audio only|Fine-tuned Qwen"


def test_stale_selection_entry_recorded_without_deletion(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    entry = _sync_entry()
    entry["wandb_policy"] = "skip_not_run"
    selection = {"db": str(db_path), "entries": [entry]}
    changed = _base_row()
    changed[3] = "Fine-tuned Qwen v2"
    manifest = _resolved_units(tmp_path, selection, [changed])
    assert manifest["summary"]["stale_entries"] == 1
    assert manifest["stale_entries"][0]["status"] == "stale_not_in_workbook"
    assert manifest["summary"]["sync_units"] == 0
    exporter_source = (PROJECT_ROOT / "tools" / "export_run_to_wandb.py").read_text(encoding="utf-8")
    assert "def _delete" not in exporter_source
    assert "api.delete" not in exporter_source


# --------------------------------------------------------------------------- CLI + exporter
def test_cli_inventory_then_resolve_then_dry_run_export(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=5)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry(expected_folds=[0, 1, 2, 3, 4])], workbook, builder)
    inventory_out = tmp_path / "inventory.json"
    _run_tool("build_workbook_dependency_inventory.py", "--workbook", str(workbook), "--output", str(inventory_out))
    inventory = json.loads(inventory_out.read_text(encoding="utf-8"))
    assert inventory["summary"]["data_rows"] == 1
    manifest_out = tmp_path / "manifest.json"
    _run_tool(
        "resolve_workbook_wandb_selection.py",
        "--selection", str(selection_path),
        "--db", str(db_path),
        "--workbook", str(workbook),
        "--builder", str(builder),
        "--output", str(manifest_out),
    )
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["summary"]["sync_units"] == 5
    dry_run_out = tmp_path / "dry_run.json"
    result = _run_tool(
        "export_run_to_wandb.py",
        "--manifest", str(manifest_out),
        "--db", str(db_path),
        "--mode", "dry_run",
        "--output", str(dry_run_out),
    )
    assert "wrote" in result.stdout
    audit = json.loads(dry_run_out.read_text(encoding="utf-8"))
    assert audit["summary"]["unit_count"] == 5
    assert audit["summary"]["complete_units"] == 5
    assert len({unit["payload_sha256"] for unit in audit["units"]}) == 5
    for unit in audit["units"]:
        assert "workbook-selected" in unit["tags"]


def test_dry_run_audit_is_deterministic(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=5)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry(expected_folds=[0, 1, 2, 3, 4])], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(first))
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_dry_run_payloads_contain_no_sensitive_material(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    audit_out = tmp_path / "dry.json"
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(audit_out))
    serialized = audit_out.read_text(encoding="utf-8")
    for forbidden in ("/gpfs", "transcript", "subject_id", "prompt"):
        assert forbidden not in serialized


def test_cloud_export_refused_when_payload_changed(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    approved = tmp_path / "approved.json"
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(approved))
    metrics = (
        tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0"
        / "best_model" / "standalone_eval" / "metrics_original_teacher_forced_full_coverage_k4.json"
    )
    content = json.loads(metrics.read_text(encoding="utf-8"))
    content["binary_strict_positive_f1"] = 0.999
    metrics.write_text(json.dumps(content), encoding="utf-8")
    result = _run_tool(
        "export_run_to_wandb.py",
        "--manifest", str(manifest_out),
        "--db", str(db_path),
        "--mode", "cloud",
        "--approved-dry-run", str(approved),
        expected_code=2,
    )
    assert "refusing cloud export" in result.stderr
    assert "sha256 mismatch" in result.stderr


def test_cloud_export_refused_when_registry_evidence_changed(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    approved = tmp_path / "approved.json"
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(approved))
    connection = registry.connect(db_path)
    try:
        connection.execute("UPDATE run_attempts SET current_state = 'LOCALLY_VALIDATED'")
        connection.commit()
    finally:
        connection.close()
    result = _run_tool(
        "export_run_to_wandb.py",
        "--manifest", str(manifest_out),
        "--db", str(db_path),
        "--mode", "cloud",
        "--approved-dry-run", str(approved),
        expected_code=2,
    )
    assert "registry evidence" in result.stderr
    assert "refusing cloud export" in result.stderr


def test_cloud_export_refused_when_payload_hash_changed(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    approved = tmp_path / "approved.json"
    _run_tool("export_run_to_wandb.py", "--manifest", str(manifest_out), "--db", str(db_path),
              "--mode", "dry_run", "--output", str(approved))
    run_config = (
        tmp_path / "audio_text" / "daic" / "daic_run" / "fold_0" / "run_config.yaml"
    )
    content = yaml.safe_load(run_config.read_text(encoding="utf-8"))
    content["config"]["seed"] = 999
    run_config.write_text(yaml.safe_dump(content), encoding="utf-8")
    result = _run_tool(
        "export_run_to_wandb.py",
        "--manifest", str(manifest_out),
        "--db", str(db_path),
        "--mode", "cloud",
        "--approved-dry-run", str(approved),
        expected_code=2,
    )
    assert "refusing cloud export" in result.stderr
    assert "payload changed since approved dry run" in result.stderr


def test_cloud_export_requires_approved_dry_run(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    _run_tool(
        "export_run_to_wandb.py",
        "--manifest", str(manifest_out),
        "--db", str(db_path),
        "--mode", "cloud",
        expected_code=2,
    )


def test_exporter_run_id_matches_manifest_run_id(tmp_path: Path) -> None:
    db_path = _import_multi_fold(tmp_path, folds=1)
    workbook = tmp_path / "wb.xlsx"
    _write_workbook(workbook, [_base_row()])
    builder = tmp_path / "builder.py"
    _write_builder(builder)
    selection_path = tmp_path / "selection.yaml"
    _write_selection(selection_path, [_sync_entry()], workbook, builder)
    manifest_out = tmp_path / "manifest.json"
    _run_tool("resolve_workbook_wandb_selection.py", "--selection", str(selection_path),
              "--db", str(db_path), "--workbook", str(workbook), "--builder", str(builder),
              "--output", str(manifest_out))
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    connection = registry.connect(db_path)
    try:
        attempt_id = manifest["export_units"][0]["attempt_id"]
        plan = wandb_export.build_export_plan(connection, attempt_id)
    finally:
        connection.close()
    assert plan["run_id"] == manifest["export_units"][0]["wandb_run_id"]


def test_real_selection_yaml_covers_all_real_workbook_rows() -> None:
    workbook = PROJECT_ROOT / "depression_results_clean.xlsx"
    selection_path = PROJECT_ROOT / "experiments/definitions/workbook_wandb_selection.yaml"
    if not workbook.is_file() or not selection_path.is_file():
        pytest.skip("real workbook or selection not present")
    inventory = build_dependency_inventory(workbook)
    selection = load_selection(selection_path)
    assert inventory["summary"]["duplicate_keys"] == 0
    assert inventory["summary"]["malformed_rows"] == 0
    keys = {row["provenance_key"] for row in inventory["rows"]}
    entry_keys = {
        "|".join(
            [
                entry["provenance_key"]["experiment"],
                entry["provenance_key"]["dataset"],
                entry["provenance_key"]["modality"],
                entry["provenance_key"]["method"],
            ]
        )
        for entry in selection["entries"]
        if not entry["selection_id"].startswith("Optuna100|")
    }
    # The Optuna-100 entries are deliberate future provenance rows (blank
    # until evidence exists); every real workbook row must still be covered.
    assert keys <= entry_keys
    from collections import Counter

    policies = Counter(entry["wandb_policy"] for entry in selection["entries"])
    assert policies["sync"] == 145
    assert policies["pending_wandb_reconciliation"] == 15
    assert policies["quarantine_ambiguous"] == 4
    assert policies["pending_importer_support"] == 108
    assert policies["pending_local_evidence"] == 0
    assert policies["skip_derived_only"] == 40
