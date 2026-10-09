"""Acceptance tests for the lane-owned treatment head planner.

The planner emits the compact ``audiollm.qwen3_heads_matrix.v1`` payload for
exactly the approved Native treatment keys; only cells whose parent chain is
proven in place on MN5 GPFS may resolve. These tests are hermetic: the remote
reader is injected, so no SSH, no adapter bytes, and no cluster state is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools import qwen3_legacy_prompt_head_plan as planner

PASS_SYSTEM = "Synthetic system prompt for head planner tests."
CACHE_ROOT = str(planner.RUNTIME_ROOT / "heads_cache")
FOLDS_SHA = "folds-sha-aaaaaaaa"
MANIFEST_RECORDED = "manifest-recorded-bbbbbbbb"
MANIFEST_FILE = "manifest-file-cccccccc"
ATTEMPT = "att-fixture-1"
KEY = "d3tec_text_only|s7|f1"
SPLIT_USED_SHA = "split-used-sha"


def _suite(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"prompt": {"system": PASS_SYSTEM}}), encoding="utf-8")
    prompt_sha = planner.sha256_text(PASS_SYSTEM)
    contract = {
        "datasets": {
            "d3tec": {
                "manifest_jsonl_sha256": MANIFEST_FILE,
                "folds_json_sha256": FOLDS_SHA,
                "manifest_metadata_sha256": "meta-dddddddd",
                "recorded_manifest_hash": MANIFEST_RECORDED,
            }
        }
    }
    matrix = {
        "schema_version": "audiollm.qwen3_legacy_prompt_matrix.v1",
        "campaign": "qwen3_legacy_prompt_20261008",
        "training_seeds": [7],
        "head_seed": 1337,
        "routes": [
            {
                "route_id": "d3tec_text_only",
                "config": str(config),
                "dataset": "d3tec",
                "dataset_dir": "d3tec",
                "modality": "text_only",
                "backend": "qwen38",
                "prompt_system_sha256": prompt_sha,
            }
        ],
        "fits": [
            {
                "key": KEY,
                "route_id": "d3tec_text_only",
                "dataset": "d3tec",
                "dataset_dir": "d3tec",
                "modality": "text_only",
                "seed": 7,
                "fold": 1,
                "run_name": "fixture_run_s7_f1",
                "config": str(config),
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text("", encoding="utf-8")
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text("", encoding="utf-8")
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    return {
        "matrix_path": matrix_path,
        "contract_path": contract_path,
        "ledger": ledger,
        "receipts": receipts,
        "run_root": run_root,
        "prompt_sha": prompt_sha,
        "config": config,
    }


def _validate(suite: dict, attempt: str = ATTEMPT) -> None:
    suite["ledger"].write_text(
        json.dumps(
            {
                "key": KEY,
                "run_name": "fixture_run_s7_f1",
                "status": "submitted",
                "attempt_id": attempt,
                "job_ids": {"train": "1", "best_eval": "2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite["receipts"].write_text(
        json.dumps({"key": KEY, "attempt_id": attempt, "stage": "validate", "ok": True}) + "\n",
        encoding="utf-8",
    )


def _run_config(attempt: str, prompt_sha: str, **overrides) -> dict:
    payload = {
        "fold": 1,
        "input_modality": "text_only",
        "tracking": {"attempt_id": attempt},
        "config": {"dataset": "d3tec", "seed": 7, "model_backend": "qwen38"},
        "prompt_context": {
            "version": None,
            "question_context_version": "legacy_v1",
            "system_prompt_sha256": prompt_sha,
        },
        "base_config_path": "/gpfs/deployments/x/code/configs/main/cfg.yaml",
        "split_metadata_path": (
            "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
            "feat-qwen3-legacy-prompt-20261008/splits/d3tec/d3tec_folds.json"
        ),
        "split_metadata_hash": FOLDS_SHA,
        "manifest_hash": MANIFEST_RECORDED,
        "evaluation_resource_shape": {"nodes": 1, "gpus_per_node": 1, "sharded": False},
    }
    payload.update(overrides)
    return payload


def _remote_block(
    key: str,
    *,
    run_config: dict | None,
    metadata: dict | None = None,
    status: dict | None = None,
    omit_metadata: bool = False,
    omit_status: bool = False,
    adapter_config: str = "ac-sha",
    adapter_model: str = "am-sha",
    split_metadata: str = FOLDS_SHA,
    split_used: str = SPLIT_USED_SHA,
    manifest: str = MANIFEST_FILE,
) -> str:
    if metadata is None and not omit_metadata:
        metadata = {"attempt_id": ATTEMPT, "fold": 1, "seed": 7}
    if status is None and not omit_status:
        status = {"attempt_id": ATTEMPT, "fold": 1, "state": "COMPLETED_ON_MN5"}
    lines = [f"===CELL {key}===", "DIR=1"]
    if run_config is not None:
        lines += ["RUNCONFIG_BEGIN", yaml.safe_dump(run_config, sort_keys=False), "RUNCONFIG_END"]
    if metadata is not None:
        lines += ["METADATA_BEGIN", json.dumps(metadata), "METADATA_END"]
    if status is not None:
        lines += ["STATUS_BEGIN", json.dumps(status), "STATUS_END"]
    for label, value in (
        ("ADAPTER_CONFIG_SHA", adapter_config),
        ("ADAPTER_MODEL_SHA", adapter_model),
        ("SPLIT_METADATA_SHA", split_metadata),
        ("SPLIT_USED_SHA", split_used),
        ("MANIFEST_SHA", manifest),
    ):
        if value:
            lines.append(f"{label}={value}")
    return "\n".join(lines) + "\n"


def _plan(suite: dict, reader):
    return planner.build_matrix(
        matrix_path=suite["matrix_path"],
        contract_path=suite["contract_path"],
        ledger_path=suite["ledger"],
        receipts_path=suite["receipts"],
        run_root=suite["run_root"],
        cache_root=CACHE_ROOT,
        remote_reader=reader,
    )


def _resolved(suite: dict, block_kwargs: dict | None = None):
    _validate(suite)
    kwargs: dict = {"run_config": _run_config(ATTEMPT, suite["prompt_sha"])}
    kwargs.update(block_kwargs or {})
    return _plan(suite, lambda script: _remote_block(KEY, **kwargs))


def _job(payload: dict) -> dict:
    return payload["routes"][0]["jobs"][0]


def test_emits_exactly_approved_keys_and_waits_without_validation(tmp_path: Path) -> None:
    suite = _suite(tmp_path)

    def must_not_run(script: str) -> str:
        raise AssertionError("no remote read may happen without validated parents")

    payload = _plan(suite, must_not_run)
    approved = planner.guard.approved_head_keys(suite["matrix_path"])
    assert not planner.validate_matrix(payload, approved, str(planner.RUNTIME_ROOT))
    assert payload["summary"]["resolved"] == 0
    assert payload["summary"]["waiting_for_checkpoint"] == 1
    assert _job(payload)["parent_status"] == "waiting_for_checkpoint"
    assert _job(payload)["reason"] == "no validated training parent for this cell yet"


def test_resolved_cell_proves_full_chain_in_place(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    scripts: list[str] = []
    _validate(suite)

    def reader(script: str) -> str:
        scripts.append(script)
        return _remote_block(
            KEY,
            run_config=_run_config(ATTEMPT, suite["prompt_sha"]),
            metadata={"attempt_id": ATTEMPT, "fold": 1, "seed": 7},
            status={"attempt_id": ATTEMPT, "fold": 1, "state": "COMPLETED_ON_MN5"},
        )

    payload = _plan(suite, reader)
    approved = planner.guard.approved_head_keys(suite["matrix_path"])
    assert not planner.validate_matrix(payload, approved, str(planner.RUNTIME_ROOT))
    assert payload["summary"]["resolved"] == 1
    job = _job(payload)
    parent = job["parent"]
    assert parent["attempt_id"] == ATTEMPT
    assert parent["checkpoint_adapter_config_sha256"] == "ac-sha"
    assert parent["checkpoint_adapter_model_sha256"] == "am-sha"
    assert parent["split_fingerprint"]["sha256"] == SPLIT_USED_SHA
    assert parent["split_fingerprint"]["source"].startswith("remote logs/split_used.json")
    assert parent["manifest_hash_recorded"] == MANIFEST_RECORDED
    assert parent["verification"]["split_metadata_sha256"] == FOLDS_SHA
    assert parent["verification"]["manifest_jsonl_sha256"] == MANIFEST_FILE
    # The released d3tec s7 f1 parent keeps its zero-delta epoch-1 qualifier
    # without any scalar dependence on a specific fold's score.
    assert parent["selection"] == "validated_receipt_epoch1_zero_delta"
    assert "zero LoRA delta" in parent["selection_reason"]
    assert "0.523" not in parent["selection_reason"]
    assert job["extract"]["gpus"] == 1
    assert len(scripts) == 1
    assert "SPLIT_METADATA_SHA=$(sha256sum" in scripts[0]
    assert "fixture_run_s7_f1/fold_1" in scripts[0]


def test_prompt_sha_and_version_mismatches_stay_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(
        suite, {"run_config": _run_config(ATTEMPT, "wrong-prompt-sha")}
    )
    assert "prompt sha mismatch" in _job(payload)["reason"]

    suite2 = _suite(tmp_path / "second")
    payload2 = _resolved(
        suite2,
        {
            "run_config": _run_config(
                ATTEMPT, suite2["prompt_sha"], prompt_context={
                    "version": "promptcontext_v1",
                    "system_prompt_sha256": suite2["prompt_sha"],
                }
            )
        },
    )
    assert "legacy inline prompt" in _job(payload2)["reason"]


def test_split_and_manifest_drift_stays_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(
        suite, {"run_config": _run_config(ATTEMPT, suite["prompt_sha"], split_metadata_hash="drifted")}
    )
    assert "recorded split_metadata_hash" in _job(payload)["reason"]

    suite2 = _suite(tmp_path / "second")
    payload2 = _resolved(
        suite2,
        {
            "run_config": _run_config(ATTEMPT, suite2["prompt_sha"], split_metadata_hash="wrong-bytes"),
            "split_metadata": "wrong-bytes",
        },
    )
    assert "pinned contract" in _job(payload2)["reason"]

    suite3 = _suite(tmp_path / "third")
    payload3 = _resolved(suite3, {"split_metadata": ""})
    assert "not hashed in place" in _job(payload3)["reason"]

    suite4 = _suite(tmp_path / "fourth")
    payload4 = _resolved(suite4, {"manifest": "drifted-file-hash"})
    assert "manifest file hash" in _job(payload4)["reason"]

    suite5 = _suite(tmp_path / "fifth")
    payload5 = _resolved(
        suite5, {"run_config": _run_config(ATTEMPT, suite5["prompt_sha"], manifest_hash="drifted")}
    )
    assert "manifest hash" in _job(payload5)["reason"]


def test_missing_manifest_and_split_used_stay_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(suite, {"manifest": ""})
    assert "not hashed in place" in _job(payload)["reason"]

    suite2 = _suite(tmp_path / "second")
    payload2 = _resolved(suite2, {"split_used": ""})
    assert "split fingerprint required" in _job(payload2)["reason"]


def test_missing_metadata_and_status_stay_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(suite, {"omit_metadata": True})
    assert "metadata.json missing" in _job(payload)["reason"]

    suite2 = _suite(tmp_path / "second")
    payload2 = _resolved(suite2, {"omit_status": True})
    assert "status.json missing" in _job(payload2)["reason"]


def test_metadata_and_status_identity_mismatches_stay_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(suite, {"metadata": {"attempt_id": ATTEMPT, "fold": 2, "seed": 7}})
    assert "metadata fold mismatch" in _job(payload)["reason"]

    suite2 = _suite(tmp_path / "second")
    payload2 = _resolved(suite2, {"metadata": {"attempt_id": ATTEMPT, "fold": 1, "seed": 8}})
    assert "metadata seed mismatch" in _job(payload2)["reason"]

    suite3 = _suite(tmp_path / "third")
    payload3 = _resolved(
        suite3, {"status": {"attempt_id": ATTEMPT, "fold": 1, "state": "FAILED"}}
    )
    assert "contradicts the validated attempt" in _job(payload3)["reason"]


def test_stale_running_remote_status_still_resolves(tmp_path: Path) -> None:
    """GPFS status sidecars are known to lag at RUNNING; identity is what counts."""
    suite = _suite(tmp_path)
    payload = _resolved(
        suite, {"status": {"attempt_id": ATTEMPT, "fold": 1, "state": "RUNNING"}}
    )
    assert _job(payload)["parent_status"] == "resolved"
    assert _job(payload)["parent"]["state"] == "RUNNING"


def test_run_config_identity_mismatches_stay_waiting(tmp_path: Path) -> None:
    cases = [
        ({"fold": 2}, "run_config fold mismatch"),
        ({"input_modality": "audio_only"}, "run_config modality mismatch"),
        ({"config": {"dataset": "cmdc", "seed": 7, "model_backend": "qwen38"}}, "dataset mismatch"),
        ({"config": {"dataset": "d3tec", "seed": 8, "model_backend": "qwen38"}}, "seed mismatch"),
        ({"config": {"dataset": "d3tec", "seed": 7, "model_backend": "qwen3omni"}}, "backend mismatch"),
        ({"base_config_path": "/gpfs/deployments/x/code/configs/main/other.yaml"}, "different config"),
    ]
    for index, (overrides, reason) in enumerate(cases):
        suite = _suite(tmp_path / f"case{index}")
        payload = _resolved(
            suite, {"run_config": _run_config(ATTEMPT, suite["prompt_sha"], **overrides)}
        )
        assert reason in _job(payload)["reason"], (overrides, _job(payload)["reason"])


def test_attempt_mismatch_stays_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(
        suite,
        {
            "run_config": _run_config("other-attempt", suite["prompt_sha"]),
            "metadata": {"attempt_id": "other-attempt", "fold": 1, "seed": 7},
            "status": {"attempt_id": "other-attempt", "fold": 1, "state": "COMPLETED_ON_MN5"},
        },
    )
    assert "remote attempt" in _job(payload)["reason"]


def test_missing_adapter_files_stay_waiting(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _resolved(suite, {"adapter_model": ""})
    assert "adapter files missing" in _job(payload)["reason"]


def test_route_prompt_pin_drift_refuses(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    matrix = json.loads(suite["matrix_path"].read_text(encoding="utf-8"))
    matrix["routes"][0]["prompt_system_sha256"] = "pinned-but-wrong"
    suite["matrix_path"].write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(planner.PlannerError):
        _plan(suite, lambda script: "")


def test_duplicate_source_fit_keys_refuse(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    matrix = json.loads(suite["matrix_path"].read_text(encoding="utf-8"))
    matrix["fits"].append(dict(matrix["fits"][0]))
    suite["matrix_path"].write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(planner.PlannerError):
        _plan(suite, lambda script: "")


def test_daic_referenced_partitions_file_uses_the_lane_pin() -> None:
    contract = {"folds_json_sha256": "folds-pin"}
    assert (
        planner.expected_split_pin(
            "daic", "/gpfs/.../splits/daic/daic_subject_partitions.json", contract
        )
        == planner.SPLIT_FILE_PINS[("daic", "daic_subject_partitions.json")]
    )
    assert (
        planner.expected_split_pin("d3tec", "/gpfs/.../splits/d3tec/d3tec_folds.json", contract)
        == "folds-pin"
    )


def test_validate_matrix_rejects_duplicate_keys(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    payload = _plan(suite, lambda script: "")
    approved = planner.guard.approved_head_keys(suite["matrix_path"])
    payload["routes"][0]["jobs"].append(dict(payload["routes"][0]["jobs"][0]))
    failures = planner.validate_matrix(payload, approved, str(planner.RUNTIME_ROOT))
    assert any("duplicate job keys" in failure for failure in failures)
    assert any("key set drift" in failure for failure in failures)


def test_ledger_run_name_overrides_matrix_for_a_retried_attempt(tmp_path: Path) -> None:
    """A bounded retry uses a new run name; the ledger is authoritative for it."""
    suite = _suite(tmp_path)
    _validate(suite)  # ledger record with run_name "fixture_run_s7_f1"
    ledger = suite["ledger"]
    lines = ledger.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["run_name"] = "fixture_run_s7_f1_r1"  # retry run name
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    scripts: list[str] = []

    def reader(script: str) -> str:
        scripts.append(script)
        return _remote_block(
            KEY,
            run_config=_run_config(ATTEMPT, suite["prompt_sha"]),
        )

    payload = _plan(suite, reader)
    job = _job(payload)
    assert job["parent_status"] == "resolved"
    assert job["parent"]["run_name"] == "fixture_run_s7_f1_r1"
    assert job["parent"]["fold_dir"].endswith("fixture_run_s7_f1_r1/fold_1")
    assert "fixture_run_s7_f1_r1/fold_1" in scripts[0]
    assert "fixture_run_s7_f1/fold_1'" not in scripts[0]
