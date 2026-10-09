"""Focused tests for the window15 head planner and its guarded dispatch wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import qwen3_window15_heads_guard as guard  # noqa: E402
from tools.qwen3_window15_heads_plan import (  # noqa: E402
    PlanError,
    build_plan,
)


def _fixture_rows(count: int = 126) -> list[dict]:
    rows = []
    routes = [
        "daic_audio_only",
        "d3tec_audio_only",
        "androids_interview_audio_text",
        "cmdc_audio_text",
        "turkish_audio_only",
    ]
    for route in routes:
        for seed in (7, 1337, 2024):
            for fold in range(10):
                rows.append(
                    {
                        "registry_key": f"{route}|{seed}|{fold}",
                        "route_id": route,
                        "seed": seed,
                        "fold": fold,
                        "planned_run_name": f"q3w15_{route}_s{seed}_f{fold}",
                        "treatment_config": f"configs/experiments/window15/{route}_harmonized_selmacrof1_likelihood_v1_window15.yaml",
                    }
                )
    return rows[:count]


def _write_config(config_dir: Path, relative: str, dataset: str, use_text: bool) -> None:
    target = config_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "dataset": dataset,
                "data": {
                    "use_audio": True,
                    "use_text": use_text,
                    "segment_seconds": 15.0,
                    "processor_min_audio_samples": 201,
                },
            }
        ),
        encoding="utf-8",
    )


def test_planner_binds_exact_treatment_attempts_and_rejects_control_configs(tmp_path, monkeypatch) -> None:
    work = tmp_path

    def fake_path(*parts: str) -> Path:
        return work.joinpath(*parts)

    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(
            work,
            row["treatment_config"],
            str(row["route_id"]).split("_audio")[0],
            use_text="audio_text" in row["route_id"],
        )
    contract = work / "contract.json"
    contract.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger = work / "submissions.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": row["registry_key"],
                    "status": "submitted",
                    "attempt_id": f"attempt-{index}",
                    "job_ids": {"train": str(1000 + index), "best_eval": str(2000 + index)},
                }
            )
            for index, row in enumerate(rows[:2])
        )
        + "\n",
        encoding="utf-8",
    )
    # submission contracts with remote fold dirs for the two submitted keys
    for index, row in enumerate(rows[:2]):
        attempt_dir = work / "outputs" / "exp_submit" / f"attempt-{index}"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "contract.json").write_text(
            json.dumps(
                {
                    "dataset": str(row["route_id"]).split("_audio")[0],
                    "config_path_remote": f"/remote/code/{row['treatment_config']}",
                    "deployment_id": "feat-qwen3-window15-20261008-xyz",
                    "fold_dir": f"/remote/output_model/qwen3_window15_20261008/{row['route_id']}/{row['planned_run_name']}/fold_{row['fold']}",
                }
            ),
            encoding="utf-8",
        )
    # one locally validated fold for the first key
    fold_dir = (
        work
        / "output_model"
        / ("audio_text" if "audio_text" in rows[0]["route_id"] else "audio_only")
        / str(rows[0]["route_id"]).split("_audio")[0]
        / rows[0]["planned_run_name"]
        / f"fold_{rows[0]['fold']}"
    )
    fold_dir.mkdir(parents=True)
    (fold_dir / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED"}), encoding="utf-8")

    parent_map, audit = build_plan(contract, ledger, exp_submit_dir=work / "outputs" / "exp_submit")
    assert audit["keys_total"] == 126
    assert len(parent_map["entries"]) == 2
    assert {entry["parent_attempt_id"] for entry in parent_map["entries"]} == {"attempt-0", "attempt-1"}
    assert all(entry["config"].startswith("configs/experiments/window15/") for entry in parent_map["entries"])
    assert all("configs/main/" not in entry["config"] for entry in parent_map["entries"])
    assert all(entry["fold_dir"].startswith("/remote/output_model/") for entry in parent_map["entries"])
    counts = audit["status_counts"]
    assert counts["eligible"] == 1
    assert counts["waiting_training"] == 125
    assert counts["eligible"] + counts["waiting_training"] + counts["waiting_validation"] + counts["blocked_failed"] == 126

    # a control config is refused outright
    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["treatment_config"] = "configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
    bad_contract = work / "bad_contract.json"
    bad_contract.write_text(json.dumps({"rows": bad_rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="not a window15 treatment config"):
        build_plan(bad_contract, ledger)
    # duplicate keys are refused
    dup_rows = [dict(rows[0]), dict(rows[0])] + [dict(row) for row in rows[2:]]
    dup_contract = work / "dup_contract.json"
    dup_contract.write_text(json.dumps({"rows": dup_rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="duplicate treatment key"):
        build_plan(dup_contract, ledger)


def test_planner_requires_processor_minimum(tmp_path, monkeypatch) -> None:
    work = tmp_path
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(
            work,
            row["treatment_config"],
            str(row["route_id"]).split("_audio")[0],
            use_text="audio_text" in row["route_id"],
        )
    # strip the key from one config
    stripped = work / rows[0]["treatment_config"]
    payload = yaml.safe_load(stripped.read_text())
    del payload["data"]["processor_min_audio_samples"]
    stripped.write_text(yaml.safe_dump(payload), encoding="utf-8")
    contract = work / "contract.json"
    contract.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="processor_min_audio_samples"):
        build_plan(contract, tmp_path / "missing.jsonl")


def test_guard_eligibility_and_delivery_parsing(tmp_path) -> None:
    audit = {
        "keys_total": 126,
        "keys": [
            {"key": "a|7|0", "status": "eligible", "route_id": "a", "seed": 7, "fold": 0, "attempt_id": "x"},
            {"key": "b|7|0", "status": "waiting_training", "route_id": "b", "seed": 7, "fold": 0},
            {"key": "c|7|0", "status": "blocked_failed", "route_id": "c", "seed": 7, "fold": 0},
        ],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    eligible = guard.eligible_keys(audit_path)
    assert [item["key"] for item in eligible] == ["a|7|0"]

    bad_audit = tmp_path / "bad_audit.json"
    bad_audit.write_text(json.dumps({"keys_total": 3, "keys": []}), encoding="utf-8")
    with pytest.raises(guard.GuardError, match="126 keys"):
        guard.eligible_keys(bad_audit)

    good = "=== JOB a|7|0 ===\nEXTRACT_ID=47110001\nCLASSIFIER_ID=47110002\n"
    parsed = guard.parse_submit_ids(good)
    assert guard._numeric_ids(parsed["a|7|0"])
    missing = guard.parse_submit_ids("=== JOB a|7|0 ===\nEXTRACT_ID=47110001\n")
    assert not guard._numeric_ids(missing["a|7|0"])
    errored = guard.parse_submit_ids("=== JOB a|7|0 ===\nERROR=boom\n")
    assert not guard._numeric_ids(errored["a|7|0"])


def test_guard_registry_keys_are_read_as_settled(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "\n".join(
            [
                json.dumps({"key": "a|7|0", "attempt_id": "x"}),
                json.dumps({"route_id": "b", "parent_training_seed": 1337, "fold": 2, "attempt_id": "y"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    settled = guard._registry_keys(registry)
    assert ("a", "7", "0") in settled
    assert ("b", "1337", "2") in settled


def test_guard_submit_lock_refuses_second_submitter(tmp_path, monkeypatch, capsys) -> None:
    lock_path = tmp_path / "submit.lock"
    monkeypatch.setattr(guard, "LOCK", lock_path)
    hold = lock_path.open("w")
    import fcntl

    fcntl.flock(hold, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        class Args:
            command = "submit"

        rc = guard.command_submit(Args())
        captured = capsys.readouterr()
        assert rc == 2
        assert "another head submitter" in captured.out
    finally:
        fcntl.flock(hold, fcntl.LOCK_UN)
        hold.close()
