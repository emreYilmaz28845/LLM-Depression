"""The production head dispatch: plan filtering, submit script, coverage.

These tests never touch the network. They pin the invariants the campaign
depends on: exactly two fixed variants, explicit classifier seed 1337, an
afterok dependency, extraction with SKIP_CLASSIFIERS=1, no smoke subject
selection, and coverage reconciliation over the planned keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import qwen3_heads_dispatch as dispatch

ROUTE = {
    "route_id": "d3tec_text_only_native",
    "config": "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "dataset": "d3tec",
    "modality": "text_only",
    "language": "native",
    "backend": "qwen38",
}


def _parent() -> dict:
    return {
        "attempt_id": "attempt-parent",
        "run_name": "run_parent",
        "fold_dir": "/gpfs/parent/fold_0",
        "checkpoint_dir": "/gpfs/parent/fold_0/best_model",
        "checkpoint_adapter_config_sha256": "a" * 64,
        "checkpoint_adapter_model_sha256": "b" * 64,
        "split_fingerprint": {"sha256": "c" * 64},
        "manifest_hash_recorded": "d" * 64,
        "selection": "explicit_parent_map",
        "selection_reason": "canonical production run",
        "excluded_attempts": [],
        "state": "REPORTABLE",
    }


def _planner_matrix() -> dict:
    return {
        "schema_version": "audiollm.qwen3_heads_matrix.v1",
        "planned_seeds": [7, 1337, 2024],
        "head_seed": 1337,
        "head_variants": ["logreg_raw", "xgb_raw"],
        "routes": [
            {
                **ROUTE,
                "jobs": [
                    {
                        "route_id": ROUTE["route_id"],
                        "config": ROUTE["config"],
                        "seed": 1337,
                        "fold": 0,
                        "parent_status": "resolved",
                        "parent": _parent(),
                        "extract": {"gpus": 1, "cache_dir": "/gpfs/cache/d3tec_text_only_native/run_parent_fold_0_pseed1337_hseed1337"},
                        "heads": {"seed": 1337, "variants": ["logreg_raw", "xgb_raw"], "depends_on": "extract"},
                    },
                    {
                        "route_id": ROUTE["route_id"],
                        "config": ROUTE["config"],
                        "seed": 2024,
                        "fold": 0,
                        "parent_status": "waiting_for_checkpoint",
                        "reason": "no eligible 2024-seed run",
                    },
                ],
            },
            {
                "route_id": "d3tec_text_only_english",
                "config": "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_en_qwen38_27b.yaml",
                "dataset": "d3tec",
                "modality": "text_only",
                "language": "english",
                "backend": "qwen38",
                "jobs": [],
            },
        ],
    }


def test_plan_filters_language_seeds_and_records_counts(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_planner_matrix()), encoding="utf-8")
    output = tmp_path / "plan.json"
    rc = dispatch.main(
        ["plan", "--matrix", str(matrix_path), "--language", "native", "--output", str(output)]
    )
    assert rc == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["summary"] == {
        "resolved": 1,
        "waiting_for_checkpoint": 1,
        "blocked_failed_parent": 0,
        "jobs": 2,
    }
    assert [route["route_id"] for route in plan["routes"]] == [ROUTE["route_id"]]
    resolved = plan["routes"][0]["jobs"][0]
    assert resolved["head_seed"] == 1337
    assert resolved["variants"] == ["logreg_raw", "xgb_raw"]
    assert resolved["parent"]["selection"] == "explicit_parent_map"
    waiting = plan["routes"][0]["jobs"][1]
    assert waiting["parent_status"] == "waiting_for_checkpoint"


def test_init_payload_pins_seeds_schema_and_tracking_kind(tmp_path: Path) -> None:
    job = {
        "logical_run_name": "q3ms_head_d3tec_text_only_native_s1337_f0",
        "seed": 1337,
        "fold": 0,
        "parent": {
            "attempt_id": "attempt-parent",
            "run_name": "run_parent",
            "fold_dir": "/gpfs/parent/fold_0",
            "checkpoint_dir": "/gpfs/parent/fold_0/best_model",
            "adapter_config_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "split_fingerprint": {"sha256": "c" * 64},
            "manifest_hash": "d" * 64,
        },
    }
    deployment = {
        "deployment_id": "dep-1",
        "git_commit": "e" * 40,
        "git_branch_at_deploy": "agent/feat-qwen3-multiseed-native-20261002",
        "git_dirty": False,
        "source_manifest_sha256": "f" * 64,
    }
    payload = dispatch._init_payload(
        job=job,
        route={**ROUTE, "dataset_variant": None, "aggregation": "subject"},
        attempt_id="20261002T000000Z-q3ms_head_d3tec_text_only_native_s1337_f0-eeeeeeee-deadbeef",
        remote_attempt_dir="/gpfs/runtime/heads/d3tec_text_only_native/attempt",
        deployment=deployment,
        group_id="qwen3-multiseed-native-20261002",
    )
    context = payload["context"]
    assert context["tracking_kind"] == dispatch.TRACKING_KIND
    assert context["run_schema_version"] == dispatch.RUN_SCHEMA
    assert context["seed"] == 1337
    assert context["required_jobs"] == ["extract", "classifier"]
    scientific = payload["config"]
    assert scientific["parent_training_seed"] == 1337
    assert scientific["head_seed"] == 1337
    assert scientific["split_seed"] == 1337
    assert scientific["classifier"]["variants"] == ["logreg_raw", "xgb_raw"]
    assert scientific["classifier"]["sampling_mode"] == "legacy"
    assert payload["parent"]["parent_attempt_id"] == "attempt-parent"


def test_submit_script_uses_afterok_fixed_seed_and_no_smoke_limits(tmp_path: Path) -> None:
    job = {
        "registry_key": "d3tec_text_only_native|1337|0",
        "attempt_id": "attempt-1",
        "logical_run_name": "q3ms_head_d3tec_text_only_native_s1337_f0",
        "remote_attempt_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1",
        "classifier_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1/classifier",
        "log_root": "/gpfs/runtime/logs/heads/d3tec_text_only_native/attempt-1",
        "cache_dir": "/gpfs/runtime/heads_cache/d3tec/text_only/run_parent_fold_0_pseed1337_hseed1337",
        "parent": _parent(),
        "condition": "text_only",
        "extract_gpus": 1,
        "payload": {"attempt_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1"},
    }
    script = dispatch._build_submit_script(
        code_root="/gpfs/deploy/code",
        jobs=[job],
        scheduler_env={"attempt-1": {"ENV_ACTIVATE": "/gpfs/venvs/qwen38/bin/activate"}},
        qwen_hidden_deps="/gpfs/deps/qwen_hidden",
        log_root="/gpfs/runtime/logs/heads",
    )
    assert "SKIP_CLASSIFIERS=1" in script
    assert "--dependency=afterok:$Q3MS_EXTRACT_ID" in script
    assert "SEED=1337" in script
    assert "CLASSIFIER_VARIANTS=logreg_raw:xgb_raw" in script
    assert "MATERIALIZE_VARIANTS=logreg_raw:xgb_raw" in script
    assert "ATTEMPT_DIR=/gpfs/runtime/heads/d3tec_text_only_native/attempt-1" in script
    assert "SUBJECT_SELECTION" not in script
    assert "Optuna" not in script
    assert "pca" not in script.lower()


def test_parse_submit_output_reads_ids_and_errors() -> None:
    output = "\n".join(
        [
            "=== JOB d3tec_text_only_native|1337|0 ===",
            "EXTRACT_ID=111",
            "CLASSIFIER_ID=222",
            "=== JOB d3tec_text_only_native|7|0 ===",
            "ERROR=extract sbatch returned no id",
        ]
    )
    parsed = dispatch._parse_submit_output(output)
    assert parsed["d3tec_text_only_native|1337|0"] == {
        "extract_job_id": "111",
        "classifier_job_id": "222",
    }
    assert parsed["d3tec_text_only_native|7|0"]["error"] == "extract sbatch returned no id"


def test_coverage_reconciles_planned_keys(tmp_path: Path) -> None:
    plan = {
        "schema_version": dispatch.SCHEMA_VERSION,
        "routes": [
            {
                **ROUTE,
                "jobs": [
                    {
                        "route_id": ROUTE["route_id"],
                        "seed": 1337,
                        "fold": 0,
                        "parent_status": "resolved",
                        "head_seed": 1337,
                        "parent": {"attempt_id": "attempt-parent"},
                    },
                    {
                        "route_id": ROUTE["route_id"],
                        "seed": 2024,
                        "fold": 0,
                        "parent_status": "waiting_for_checkpoint",
                        "head_seed": 1337,
                    },
                ],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "status.json").write_text(json.dumps({"state": "REPORTABLE"}), encoding="utf-8")
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps(
            {
                "registry_key": "d3tec_text_only_native|1337|0",
                "attempt_id": "attempt-head",
                "local_mirror": str(mirror),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "coverage.json"
    rc = dispatch.main(
        [
            "coverage",
            "--plan",
            str(plan_path),
            "--registry",
            str(registry),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    coverage = json.loads(output.read_text(encoding="utf-8"))
    assert coverage["counts"] == {"reportable": 1, "waiting_for_checkpoint": 1}
    reportable = next(row for row in coverage["rows"] if row["coverage_status"] == "reportable")
    assert reportable["parent_training_seed"] == 1337
    assert reportable["head_seed"] == 1337


def test_tracking_schema_override_and_uar_metric(tmp_path: Path) -> None:
    from src.native_en_text_heads_tracking import initialize_head_attempt, _metric_payload

    attempt_dir = tmp_path / "attempt"
    initialize_head_attempt(
        attempt_dir,
        context={
            "attempt_id": "attempt-x",
            "logical_run_name": "run-x",
            "fold": 0,
            "seed": 1337,
            "tracking_kind": dispatch.TRACKING_KIND,
            "run_schema_version": dispatch.RUN_SCHEMA,
            "group_id": "group-x",
        },
        config={"classifier": {"prediction_backend": "likelihood"}},
        parent={"parent_attempt_id": "p", "parent_checkpoint_path": "/gpfs/p"},
    )
    import yaml

    run_config = yaml.safe_load((attempt_dir / "run_config.yaml").read_text(encoding="utf-8"))
    assert run_config["schema_version"] == dispatch.RUN_SCHEMA
    metrics = _metric_payload(
        [
            {"label": 1, "predicted_class": 1},
            {"label": 0, "predicted_class": 0},
            {"label": 1, "predicted_class": 0},
            {"label": 0, "predicted_class": 1},
        ]
    )
    assert metrics["uar"] == metrics["macro_recall"]
