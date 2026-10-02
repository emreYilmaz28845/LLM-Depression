"""The production head dispatch: lane identity, plan filtering, isolation.

These tests never touch the network. They pin the invariants the campaign
depends on: exactly two fixed variants, explicit classifier seed 1337, an
afterok dependency, extraction with SKIP_CLASSIFIERS=1, no smoke subject
selection, lane-derived campaign/tracking identity for Native and English, and
fail-closed refusal to write into another lane's outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools import qwen3_heads_dispatch as dispatch

ROUTE = {
    "route_id": "d3tec_text_only_native",
    "config": "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "dataset": "d3tec",
    "modality": "text_only",
    "language": "native",
    "backend": "qwen38",
}


def _write_synthetic_lane(
    root: Path,
    *,
    experiment_id: str,
    campaign: str,
    language: str,
    tracking_kind: str | None = None,
    run_schema: str | None = None,
    evidence_dir: str | None = None,
) -> Path:
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    group_scope = {"campaign": campaign, "language": language}
    if tracking_kind:
        group_scope["head_tracking_kind"] = tracking_kind
    if run_schema:
        group_scope["head_run_schema"] = run_schema
    if evidence_dir:
        group_scope["evidence_dir"] = evidence_dir
    group_rel = f"experiments/definitions/{experiment_id}-group.yaml"
    group_path = root / group_rel
    group_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "audiollm.experiment_group.v1",
                "group_id": f"{experiment_id}-group",
                "title": "t",
                "research_question": "q",
                "dataset": "d",
                "baseline": "b",
                "treatment": "t",
                "expected_seeds": [7, 1337, 2024],
                "expected_folds": [0],
                "primary_metric": {
                    "namespace": "headline/binary_strict",
                    "name": "macro_f1",
                    "backend": "likelihood",
                    "aggregation": "subject_level",
                    "evaluation_view": "harmonized_all_windows_full_coverage",
                },
                "scope": group_scope,
            }
        ),
        encoding="utf-8",
    )
    definition_rel = f"experiments/definitions/lanes/{experiment_id}.yaml"
    definition_path = root / definition_rel
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "audiollm.experiment_lane.v1",
                "experiment_id": experiment_id,
                "slug": experiment_id,
                "tier": 2,
                "branch": f"agent/{experiment_id}",
                "worktree": str(root),
                "parent_branch": None,
                "parent_sha": "c" * 40,
                "created_at_utc": "2026-10-02T00:00:00Z",
                "experiment_group_path": group_rel,
                "type": "complementary",
            }
        ),
        encoding="utf-8",
    )
    pin = {
        "schema_version": "audiollm.agent_pin.v1",
        "experiment_id": experiment_id,
        "tier": 2,
        "worktree": str(root),
        "branch": f"agent/{experiment_id}",
        "definition_path": definition_rel,
    }
    pin_path = root / ".agent-pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    return pin_path


@pytest.fixture()
def native_identity(tmp_path: Path) -> dispatch.LaneIdentity:
    root = tmp_path / "native"
    pin = _write_synthetic_lane(
        root,
        experiment_id="feat-qwen3-multiseed-native-20261002",
        campaign="qwen3_multiseed_native_20261002",
        language="native",
        tracking_kind="qwen3_multiseed_native_head",
        run_schema="audiollm.qwen3_multiseed_head_run.v1",
        evidence_dir="outputs/qwen3_multiseed_native_20261002",
    )
    return dispatch.resolve_lane_identity(project_root=root, pin_path=pin)


@pytest.fixture()
def english_identity(tmp_path: Path) -> dispatch.LaneIdentity:
    root = tmp_path / "english"
    pin = _write_synthetic_lane(
        root,
        experiment_id="feat-qwen3-multiseed-english-20261002",
        campaign="qwen3_multiseed_english_20261002",
        language="english",
        evidence_dir="outputs/qwen3_multiseed_english_20261002",
    )
    return dispatch.resolve_lane_identity(project_root=root, pin_path=pin)


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


def _planner_matrix(cache_root: str, language: str = "native") -> dict:
    return {
        "schema_version": "audiollm.qwen3_heads_matrix.v1",
        "planned_seeds": [7, 1337, 2024],
        "head_seed": 1337,
        "head_variants": ["logreg_raw", "xgb_raw"],
        "cache_root": cache_root,
        "routes": [
            {
                **ROUTE,
                "language": language,
                "jobs": [
                    {
                        "route_id": ROUTE["route_id"],
                        "config": ROUTE["config"],
                        "seed": 1337,
                        "fold": 0,
                        "parent_status": "resolved",
                        "parent": _parent(),
                        "extract": {
                            "gpus": 1,
                            "cache_dir": f"{cache_root}/d3tec/text_only/run_parent_fold_0_pseed1337_hseed1337",
                        },
                        "heads": {
                            "seed": 1337,
                            "variants": ["logreg_raw", "xgb_raw"],
                            "depends_on": "extract",
                        },
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


def test_identity_resolves_campaign_tracking_and_evidence(native_identity, english_identity) -> None:
    assert native_identity.campaign == "qwen3_multiseed_native_20261002"
    assert native_identity.tracking_kind == "qwen3_multiseed_native_head"
    assert native_identity.run_schema == "audiollm.qwen3_multiseed_head_run.v1"
    assert native_identity.evidence_dir.name == "qwen3_multiseed_native_20261002"
    assert native_identity.language == "native"
    assert native_identity.runtime_root.name == "feat-qwen3-multiseed-native-20261002"
    assert english_identity.campaign == "qwen3_multiseed_english_20261002"
    assert english_identity.tracking_kind == "qwen3_multiseed_english_20261002_head"
    assert english_identity.run_schema == "audiollm.qwen3_multiseed_english_20261002_head_run.v1"
    assert english_identity.evidence_dir.name == "qwen3_multiseed_english_20261002"
    assert english_identity.language == "english"
    assert english_identity.runtime_root.name == "feat-qwen3-multiseed-english-20261002"
    assert native_identity.campaign not in str(english_identity.evidence_dir)
    assert native_identity.tracking_kind != english_identity.tracking_kind


def test_identity_campaign_override_must_match(native_identity, tmp_path: Path) -> None:
    with pytest.raises(dispatch.DispatchError, match="does not match the linked group campaign"):
        dispatch.resolve_lane_identity(
            project_root=native_identity.project_root,
            pin_path=native_identity.project_root / ".agent-pin.json",
            campaign_override="qwen3_multiseed_english_20261002",
        )


def test_identity_requires_campaign_and_language(tmp_path: Path) -> None:
    root = tmp_path / "nolabel"
    pin = _write_synthetic_lane(
        root,
        experiment_id="feat-nolabel-20261002",
        campaign="placeholder",
        language="placeholder",
    )
    group_path = root / "experiments/definitions/feat-nolabel-20261002-group.yaml"
    group = yaml.safe_load(group_path.read_text(encoding="utf-8"))
    del group["scope"]["campaign"]
    del group["scope"]["language"]
    group_path.write_text(yaml.safe_dump(group), encoding="utf-8")
    with pytest.raises(dispatch.DispatchError, match="scope.campaign"):
        dispatch.resolve_lane_identity(project_root=root, pin_path=pin)
    with pytest.raises(dispatch.DispatchError, match="scope.language"):
        dispatch.resolve_lane_identity(project_root=root, pin_path=pin, campaign_override="x")


def test_identity_evidence_dir_must_stay_under_outputs(tmp_path: Path) -> None:
    root = tmp_path / "escape"
    pin = _write_synthetic_lane(
        root,
        experiment_id="feat-escape-20261002",
        campaign="escape_campaign",
        language="native",
        evidence_dir="../../../etc",
    )
    with pytest.raises(dispatch.DispatchError, match="escapes the lane root"):
        dispatch.resolve_lane_identity(project_root=root, pin_path=pin)


def test_require_under_rejects_escape() -> None:
    dispatch.require_under("/gpfs/root/a/b", "/gpfs/root", "x")
    with pytest.raises(dispatch.DispatchError):
        dispatch.require_under("/gpfs/root/../other", "/gpfs/root", "x")
    with pytest.raises(dispatch.DispatchError):
        dispatch.require_under("/gpfs/other", "/gpfs/root", "x")


def test_validate_plan_identity_rejects_foreign_campaign(native_identity) -> None:
    plan = {
        "schema_version": dispatch.SCHEMA_VERSION,
        "campaign": "qwen3_multiseed_english_20261002",
        "group_id": native_identity.group_id,
        "experiment_id": native_identity.experiment_id,
        "language": "native",
        "cache_root": str(native_identity.runtime_root / "heads_cache"),
    }
    with pytest.raises(dispatch.DispatchError, match="does not match this lane"):
        dispatch.validate_plan_identity(plan, native_identity)


def test_validate_plan_identity_requires_lane_cache_root(native_identity) -> None:
    base = {
        "schema_version": dispatch.SCHEMA_VERSION,
        "campaign": native_identity.campaign,
        "group_id": native_identity.group_id,
        "experiment_id": native_identity.experiment_id,
        "language": native_identity.language,
    }
    with pytest.raises(dispatch.DispatchError, match="no lane-owned cache_root"):
        dispatch.validate_plan_identity(dict(base), native_identity)
    with pytest.raises(dispatch.DispatchError, match="escapes the lane root"):
        dispatch.validate_plan_identity(
            {**base, "cache_root": "/gpfs/projects/other/cache"}, native_identity
        )
    assert dispatch.validate_plan_identity(
        {**base, "cache_root": str(native_identity.runtime_root / "heads_cache")},
        native_identity,
    ) == native_identity.runtime_root / "heads_cache"


def test_plan_filters_language_seeds_and_records_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_identity
) -> None:
    monkeypatch.setattr(dispatch, "resolve_lane_identity", lambda **kwargs: native_identity)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(_planner_matrix(str(native_identity.runtime_root / "heads_cache"))),
        encoding="utf-8",
    )
    output = native_identity.evidence_dir / "head_dispatch_plan_v1.json"
    rc = dispatch.main(["plan", "--matrix", str(matrix_path), "--output", str(output)])
    assert rc == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["campaign"] == native_identity.campaign
    assert plan["tracking_kind"] == native_identity.tracking_kind
    assert plan["run_schema"] == native_identity.run_schema
    assert plan["language"] == "native"
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


def test_plan_refuses_foreign_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_identity
) -> None:
    monkeypatch.setattr(dispatch, "resolve_lane_identity", lambda **kwargs: native_identity)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(_planner_matrix("/gpfs/projects/etur92/ozu647717/AudioLLM/other_cache")),
        encoding="utf-8",
    )
    rc = dispatch.main(
        [
            "plan",
            "--matrix",
            str(matrix_path),
            "--output",
            str(native_identity.evidence_dir / "plan.json"),
        ]
    )
    assert rc == 1


def test_init_payload_carries_lane_identity(native_identity, english_identity) -> None:
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
        "git_branch_at_deploy": "agent/x",
        "git_dirty": False,
        "source_manifest_sha256": "f" * 64,
    }
    native_payload = dispatch._init_payload(
        job=job,
        route={**ROUTE, "dataset_variant": None, "aggregation": "subject"},
        attempt_id="attempt-native",
        remote_attempt_dir=str(native_identity.runtime_root / "heads" / "x" / "attempt-native"),
        deployment=deployment,
        identity=native_identity,
    )
    assert native_payload["context"]["group_id"] == native_identity.group_id
    assert native_payload["context"]["tracking_kind"] == native_identity.tracking_kind
    assert native_payload["context"]["run_schema_version"] == native_identity.run_schema
    assert native_payload["config"]["campaign"] == native_identity.campaign
    assert native_payload["config"]["parent_training_seed"] == 1337
    assert native_payload["config"]["head_seed"] == 1337
    assert native_payload["config"]["split_seed"] == 1337
    assert native_payload["config"]["classifier"]["variants"] == ["logreg_raw", "xgb_raw"]
    assert native_payload["config"]["classifier"]["sampling_mode"] == "legacy"

    english_route = {
        **ROUTE,
        "route_id": "d3tec_text_only_english",
        "language": "english",
        "dataset_variant": None,
        "aggregation": "subject",
    }
    english_job = {
        **job,
        "logical_run_name": "q3ms_head_d3tec_text_only_english_s1337_f0",
    }
    english_payload = dispatch._init_payload(
        job=english_job,
        route=english_route,
        attempt_id="attempt-english",
        remote_attempt_dir=str(english_identity.runtime_root / "heads" / "x" / "attempt-english"),
        deployment=deployment,
        identity=english_identity,
    )
    assert english_payload["config"]["campaign"] == english_identity.campaign
    assert english_payload["context"]["tracking_kind"] == english_identity.tracking_kind
    assert english_payload["context"]["run_schema_version"] == english_identity.run_schema
    assert native_identity.campaign not in json.dumps(english_payload)


def test_submit_script_uses_afterok_fixed_seed_and_no_smoke_limits() -> None:
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


def test_repair_script_reuses_existing_extract_and_records_classifier_only() -> None:
    job = {
        "registry_key": "d3tec_text_only_native|1337|0",
        "attempt_id": "attempt-1",
        "remote_attempt_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1",
        "classifier_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1/classifier",
        "log_root": "/gpfs/runtime/logs/heads/d3tec_text_only_native/attempt-1",
        "cache_dir": "/gpfs/runtime/heads_cache/d3tec/text_only/run_parent_fold_0_pseed1337_hseed1337",
        "parent": {"checkpoint_dir": "/gpfs/parent/best_model"},
        "condition": "text_only",
        "extract_gpus": 1,
        "submit_extract": False,
        "existing_extract_id": "111",
    }
    script = dispatch._build_repair_script(
        code_root="/gpfs/deploy/code",
        jobs=[job],
        scheduler_env={"attempt-1": {"ENV_ACTIVATE": "/gpfs/venvs/qwen38/bin/activate"}},
        qwen_hidden_deps="/gpfs/deps/qwen_hidden",
    )
    assert "export Q3MS_EXTRACT_ID=111" in script
    assert "Q3MS_RECORD_EXTRACT=0" in script
    assert "--dependency=afterok:$Q3MS_EXTRACT_ID" in script
    assert "SEED=1337" in script
    assert "CLASSIFIER_VARIANTS=logreg_raw:xgb_raw" in script
    assert "sbatch --parsable" in script


def test_repair_script_submits_both_legs_when_extract_is_missing() -> None:
    job = {
        "registry_key": "d3tec_text_only_native|1337|0",
        "attempt_id": "attempt-1",
        "remote_attempt_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1",
        "classifier_dir": "/gpfs/runtime/heads/d3tec_text_only_native/attempt-1/classifier",
        "log_root": "/gpfs/runtime/logs/heads/d3tec_text_only_native/attempt-1",
        "cache_dir": "/gpfs/runtime/heads_cache/d3tec/text_only/run_parent_fold_0_pseed1337_hseed1337",
        "parent": {"checkpoint_dir": "/gpfs/parent/best_model"},
        "condition": "text_only",
        "extract_gpus": 1,
        "submit_extract": True,
        "existing_extract_id": "",
    }
    script = dispatch._build_repair_script(
        code_root="/gpfs/deploy/code",
        jobs=[job],
        scheduler_env={"attempt-1": {"ENV_ACTIVATE": "/gpfs/venvs/qwen38/bin/activate"}},
        qwen_hidden_deps="/gpfs/deps/qwen_hidden",
    )
    assert "Q3MS_RECORD_EXTRACT=1" in script
    assert "SKIP_CLASSIFIERS=1" in script


def test_registry_read_keeps_latest_per_key(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({"registry_key": "k", "attempt_id": "a", "extract_job_id": "1"})
        + "\n"
        + json.dumps({"registry_key": "k", "attempt_id": "a", "extract_job_id": "2"})
        + "\n",
        encoding="utf-8",
    )
    entries = dispatch._read_registry(registry)
    assert len(entries) == 1
    assert entries[0]["extract_job_id"] == "2"


def test_coverage_reconciles_planned_keys(
    monkeypatch: pytest.MonkeyPatch, native_identity
) -> None:
    monkeypatch.setattr(dispatch, "resolve_lane_identity", lambda **kwargs: native_identity)
    plan = {
        "schema_version": dispatch.SCHEMA_VERSION,
        "campaign": native_identity.campaign,
        "group_id": native_identity.group_id,
        "experiment_id": native_identity.experiment_id,
        "language": native_identity.language,
        "cache_root": str(native_identity.runtime_root / "heads_cache"),
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
    plan_path = native_identity.evidence_dir / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    mirror = native_identity.evidence_dir / "head_attempts" / "attempt-head"
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "status.json").write_text(json.dumps({"state": "REPORTABLE"}), encoding="utf-8")
    registry = native_identity.evidence_dir / "head_submissions.jsonl"
    registry.write_text(
        json.dumps(
            {
                "registry_key": "d3tec_text_only_native|1337|0",
                "attempt_id": "attempt-head",
                "local_mirror": str(mirror),
                "remote_attempt_dir": str(
                    native_identity.runtime_root / "heads" / "d3tec_text_only_native" / "attempt-head"
                ),
                "campaign": native_identity.campaign,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = native_identity.evidence_dir / "coverage.json"
    rc = dispatch.main(["coverage", "--plan", str(plan_path), "--output", str(output)])
    assert rc == 0
    coverage = json.loads(output.read_text(encoding="utf-8"))
    assert coverage["campaign"] == native_identity.campaign
    assert coverage["counts"] == {"reportable": 1, "waiting_for_checkpoint": 1}


def test_coverage_refuses_foreign_plan(
    monkeypatch: pytest.MonkeyPatch, native_identity
) -> None:
    monkeypatch.setattr(dispatch, "resolve_lane_identity", lambda **kwargs: native_identity)
    plan = {
        "schema_version": dispatch.SCHEMA_VERSION,
        "campaign": "qwen3_multiseed_english_20261002",
        "group_id": native_identity.group_id,
        "experiment_id": native_identity.experiment_id,
        "language": "native",
        "cache_root": str(native_identity.runtime_root / "heads_cache"),
        "routes": [],
    }
    plan_path = native_identity.evidence_dir / "foreign_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    rc = dispatch.main(["coverage", "--plan", str(plan_path)])
    assert rc == 1


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
            "tracking_kind": "qwen3_multiseed_native_head",
            "run_schema_version": "audiollm.qwen3_multiseed_head_run.v1",
            "group_id": "group-x",
        },
        config={"classifier": {"prediction_backend": "likelihood"}},
        parent={"parent_attempt_id": "p", "parent_checkpoint_path": "/gpfs/p"},
    )
    run_config = yaml.safe_load((attempt_dir / "run_config.yaml").read_text(encoding="utf-8"))
    assert run_config["schema_version"] == "audiollm.qwen3_multiseed_head_run.v1"
    metrics = _metric_payload(
        [
            {"label": 1, "predicted_class": 1},
            {"label": 0, "predicted_class": 0},
            {"label": 1, "predicted_class": 0},
            {"label": 0, "predicted_class": 1},
        ]
    )
    assert metrics["uar"] == metrics["macro_recall"]
