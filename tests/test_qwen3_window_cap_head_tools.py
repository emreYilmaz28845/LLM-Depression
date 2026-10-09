"""Tests for the tracked window-cap production head planner/wave/refill tools.

Covers the meaningful regressions found during independent review:

* arm-explicit identities: three arms sharing one base route/seed/fold get
  distinct head keys, logical names and cache dirs; the real-size matrix must
  yield 378 distinct keys and an arm-collapsed matrix fails closed;
* scheduler readiness: only exact top-level sacct records count (``.batch``
  steps are ignored; step-only output leaves UNKNOWN; contradictory duplicate
  top-level records are refused); missing exit is UNKNOWN, never PASS;
* strict same-attempt identity, mask agreement, and non-null/non-empty split
  fingerprints;
* readiness withdrawal: a rebuild always rewrites the plan (fresh token,
  empty routes) and the wave driver refuses stale or untokened plans;
* exclusive lane submission lock shared by wave and refill drivers;
* no duplicate wave keys and full 2*N reservation flags;
* the refill driver builds guarded submissions with the lane admission paths.

All fixtures are synthetic temp directories; no ignored private files are read.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qwen3_window_cap_head_plan as planner  # noqa: E402
from tools import qwen3_window_cap_heads_wave as driver  # noqa: E402
from tools import qwen3_window_cap_refill as refill  # noqa: E402

FRACTION = {"cap25": 0.25, "cap50": 0.5, "cap75": 0.75}


def make_row(
    arm: str,
    *,
    seed: int = 7,
    fold: int = 0,
    route: str = "androids_interview_audio_only_native",
    attempt: str = "A",
) -> dict:
    return {
        "route_id": route,
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "dataset": "androids_interview",
        "modality": "audio_only",
        "run_name": f"windowcap_{route}_s{seed}_f{fold}_{arm}",
        "fraction": FRACTION[arm],
        "attempt_id": attempt,
        "config": f"configs/experiments/window_cap/{route}_{arm}.yaml",
    }


def synthetic_matrix_rows() -> list[dict]:
    return [
        make_row(arm, seed=seed, fold=0, route=f"route{r}")
        for r in range(42)
        for seed in (7, 1337, 2024)
        for arm in ("cap25", "cap50", "cap75")
    ]


def make_fold(
    root: Path,
    row: dict,
    *,
    train_id: str = "1001",
    eval_id: str = "1002",
    local_train_event: str = "COMPLETED",
    exit_code=None,
) -> Path:
    fold = root / row["modality"] / row["dataset"] / row["run_name"] / f"fold_{row['fold']}"
    fold.mkdir(parents=True, exist_ok=True)
    mask = {
        "selection_sha256": "a" * 64,
        "baseline_input_sha256": "b" * 64,
        "fraction": row["fraction"],
        "sampling_seed": 1337,
    }
    (fold / "window_cap_mask.json").write_text(json.dumps(mask), encoding="utf-8")
    run_config = {
        "tracking": {"attempt_id": row["attempt_id"]},
        "config": {
            "training": {
                "window_cap": {
                    "enabled": True,
                    "fraction": row["fraction"],
                    "sampling_seed": 1337,
                    "algorithm_version": "sha256-subject-permutation-v1",
                    "selection_sha256": "a" * 64,
                    "baseline_input_sha256": "b" * 64,
                }
            },
            "evaluation": {},
        },
    }
    (fold / "run_config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    (fold / "metadata.json").write_text(
        json.dumps({"attempt_id": row["attempt_id"]}), encoding="utf-8"
    )
    events = [
        {"event_type": "SUBMITTED", "job_key": "train", "slurm_job_id": train_id, "status": "PENDING"},
        {"event_type": "SUBMITTED", "job_key": "best_eval", "slurm_job_id": eval_id, "status": "PENDING"},
        {"event_type": local_train_event, "job_key": "train", "exit_code": exit_code, "status": "COMPLETED"},
        {"event_type": "COMPLETED", "job_key": "best_eval", "exit_code": exit_code, "status": "COMPLETED"},
    ]
    (fold / "jobs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    logs = fold / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "split_used.json").write_text(
        json.dumps(
            {
                "train_subject_ids": ["s1", "s2"],
                "selection_subject_ids": ["s3"],
                "final_eval_subject_ids": ["s4"],
            }
        ),
        encoding="utf-8",
    )
    return fold


def make_route(arm: str, *, seed: int = 7, fold: int = 0) -> dict:
    row = make_row(arm, seed=seed, fold=fold)
    return {
        "route_id": planner.head_route_id(row),
        "arm": arm,
        "jobs": [{"seed": seed, "fold": fold}],
    }


# --- arm-explicit identities -------------------------------------------------


def test_three_arms_same_seed_fold_distinct():
    rows = [make_row(arm) for arm in ("cap25", "cap50", "cap75")]
    assert len({planner.head_key(r) for r in rows}) == 3
    assert len({planner.logical_run_name(r) for r in rows}) == 3
    assert len({planner.cache_dir(r) for r in rows}) == 3
    assert len({planner.head_route_id(r) for r in rows}) == 3


def test_378_distinct_expected_keys():
    rows = synthetic_matrix_rows()
    matrix = {"treatments": rows}
    keys = planner.assert_matrix_keys(matrix)
    assert len(rows) == 378
    assert len(keys) == 378
    collapsed = {"treatments": rows[::3]}
    assert len(collapsed["treatments"]) == 126
    with pytest.raises(SystemExit):
        planner.assert_matrix_keys(collapsed)


# --- scheduler-confirmed readiness ------------------------------------------


def test_missing_exit_is_unknown_not_pass(tmp_path):
    row = make_row("cap25")
    make_fold(tmp_path, row)
    ok, reason = planner.readiness_local(row, raw_root=tmp_path)
    assert ok, reason
    ok, reason = planner.readiness_scheduler(row, {}, raw_root=tmp_path)
    assert not ok and "UNKNOWN" in reason
    good = {
        "1001": {"state": "COMPLETED", "exit": "0:0"},
        "1002": {"state": "COMPLETED", "exit": "0:0"},
    }
    assert planner.readiness_scheduler(row, good, raw_root=tmp_path)[0]
    bad_exit = {
        "1001": {"state": "COMPLETED", "exit": "0:1"},
        "1002": {"state": "COMPLETED", "exit": "0:0"},
    }
    assert not planner.readiness_scheduler(row, bad_exit, raw_root=tmp_path)[0]
    running = {
        "1001": {"state": "RUNNING", "exit": "0:0"},
        "1002": {"state": "PENDING", "exit": "0:0"},
    }
    assert not planner.readiness_scheduler(row, running, raw_root=tmp_path)[0]
    contradiction = make_row("cap50")
    make_fold(tmp_path, contradiction, local_train_event="FAILED")
    ok, reason = planner.readiness_local(contradiction, raw_root=tmp_path)
    assert not ok and "terminal event FAILED" in reason


def test_parse_sacct_exact_top_level_only():
    assert planner.parse_sacct(
        "1001.batch|COMPLETED|0:0\n1001.extern|COMPLETED|0:0\n", {"1001"}
    ) == {}
    parsed = planner.parse_sacct("1001|FAILED|1:0\n1001.batch|COMPLETED|0:0\n", {"1001"})
    assert parsed["1001"] == {"state": "FAILED", "exit": "1:0"}
    parsed = planner.parse_sacct("1001|COMPLETED|0:0\n", {"1001"})
    assert parsed["1001"] == {"state": "COMPLETED", "exit": "0:0"}


def test_parse_sacct_contradictory_duplicates_refused():
    parsed = planner.parse_sacct("1001|COMPLETED|0:0\n1001|FAILED|1:0\n", {"1001"})
    assert parsed["1001"]["state"] == "CONTRADICTION"
    same = planner.parse_sacct("1001|COMPLETED|0:0\n1001|COMPLETED|0:0\n", {"1001"})
    assert same["1001"] == {"state": "COMPLETED", "exit": "0:0"}


def test_step_only_scheduler_leaves_unknown(tmp_path):
    row = make_row("cap25")
    make_fold(tmp_path, row)
    step_only = planner.parse_sacct(
        "1001.batch|COMPLETED|0:0\n1002.batch|COMPLETED|0:0\n", {"1001", "1002"}
    )
    ok, reason = planner.readiness_scheduler(row, step_only, raw_root=tmp_path)
    assert not ok and "UNKNOWN" in reason
    contradiction = planner.parse_sacct(
        "1001|COMPLETED|0:0\n1001|FAILED|1:0\n1002|COMPLETED|0:0\n", {"1001", "1002"}
    )
    ok, reason = planner.readiness_scheduler(row, contradiction, raw_root=tmp_path)
    assert not ok and "CONTRADICTION" in reason


def test_missing_or_empty_split_fingerprint_rejected(tmp_path):
    row = make_row("cap25")
    fold = make_fold(tmp_path, row)
    (fold / "logs" / "split_used.json").unlink()
    ok, reason = planner.readiness_local(row, raw_root=tmp_path)
    assert not ok and "split_used" in reason
    (fold / "logs" / "split_used.json").write_text("{}", encoding="utf-8")
    ok, reason = planner.readiness_local(row, raw_root=tmp_path)
    assert not ok and "empty split fingerprint" in reason


def test_attempt_identity_mismatch_rejected(tmp_path):
    row = make_row("cap25")
    make_fold(tmp_path, row)
    bad = dict(row)
    bad["attempt_id"] = "OTHER"
    ok, reason = planner.readiness_local(bad, raw_root=tmp_path)
    assert not ok and "attempt id mismatch" in reason


def test_mask_disagreement_rejected(tmp_path):
    row = make_row("cap25")
    fold = make_fold(tmp_path, row)
    mask = json.loads((fold / "window_cap_mask.json").read_text(encoding="utf-8"))
    mask["fraction"] = 0.5
    (fold / "window_cap_mask.json").write_text(json.dumps(mask), encoding="utf-8")
    ok, reason = planner.readiness_local(row, raw_root=tmp_path)
    assert not ok and "disagrees with run_config" in reason


# --- readiness withdrawal / stale plan --------------------------------------


def test_readiness_withdrawal_no_stale_dispatch(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "production_matrix.json").write_text(
        json.dumps({"treatments": synthetic_matrix_rows()}), encoding="utf-8"
    )
    plan_path = campaign / "heads_production_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "routes": [
                    {"route_id": "route0_cap25", "arm": "cap25", "jobs": [{"seed": 7, "fold": 0}]}
                ],
                "build_token": "old-token",
            }
        ),
        encoding="utf-8",
    )
    assert planner.main(["--campaign-dir", str(campaign), "--no-fetch"]) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["routes"] == []
    assert plan["build_token"] != "old-token"
    wave = driver.build_wave(plan, 2, set())
    assert wave["summary"]["resolved"] == 0


def test_plan_age_and_token_freshness():
    fresh = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert driver.plan_age_seconds(fresh) < 30
    assert (
        driver.plan_age_seconds({"created_at_utc": "2020-01-01T00:00:00Z"})
        > driver.PLAN_MAX_AGE_SECONDS
    )
    assert driver.plan_age_seconds({}) is None


def test_wave_paused_without_marker(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    assert driver.main(["--campaign-dir", str(campaign)]) == 0


def test_stale_plan_refused(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "heads_dispatch_enabled").write_text("", encoding="utf-8")
    (campaign / "heads_production_plan.json").write_text(
        json.dumps({"routes": [], "build_token": "old", "created_at_utc": "2020-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(driver, "rebuild_plan", lambda: 0)
    assert driver.main(["--campaign-dir", str(campaign)]) == 1


# --- submission lock ---------------------------------------------------------


def test_submission_lock_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SUBMISSION_LOCK", tmp_path / "submission.lock")
    first = driver.acquire_submission_lock()
    assert first is not None
    assert driver.acquire_submission_lock() is None
    first.close()
    second = driver.acquire_submission_lock()
    assert second is not None
    second.close()


def test_refill_submission_lock_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(refill, "SUBMISSION_LOCK", tmp_path / "submission.lock")
    first = refill.acquire_submission_lock()
    assert first is not None
    assert refill.acquire_submission_lock() is None
    first.close()


# --- wave keys and reservation flags ----------------------------------------


def test_wave_no_duplicates_and_full_reservation(tmp_path):
    driver.configure(campaign_dir=tmp_path)
    plan = {
        "routes": [make_route("cap25"), make_route("cap50"), make_route("cap75")],
        "build_token": "token",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {"expected_keys": 378},
    }
    wave = driver.build_wave(plan, 3, set())
    keys = driver.wave_keys(wave)
    assert len(keys) == 3 and len(set(keys)) == 3
    filtered = driver.build_wave(
        plan, 3, {f"{planner.head_route_id(make_row('cap50'))}|7|0"}
    )
    assert filtered["summary"]["resolved"] == 2
    command = driver.guard_command(tmp_path / "wave.json", 3)
    assert command[command.index("--jobs-this-submit") + 1] == "2"
    assert command[command.index("--wave-fits") + 1] == "3"
    assert command[command.index("--delivery-file") + 1] == str(driver.SUBMIT_OUTPUT)


# --- refill command shape ----------------------------------------------------


def test_refill_build_command_guarded(tmp_path):
    refill.configure(
        campaign_dir=tmp_path,
        ledger=tmp_path / "ledger.json",
        local_run_root=tmp_path / "out",
    )
    row = make_row("cap25")
    row["manifest_policy"] = "prebuilt"
    command = refill.build_command(row)
    assert command[0] == sys.executable
    assert str(refill.GUARD) in command
    assert "--ledger" in command
    assert command[command.index("--ledger") + 1] == str(tmp_path / "ledger.json")
    assert command[command.index("run-submit") + 1 : command.index("run-submit") + 7] == [
        "--jobs-this-submit",
        "2",
        "--wave-fits",
        "1",
        "--kind",
        "fit",
    ]
    assert "tools/exp.py" in command
    assert command[-1] == "prebuilt"


def test_plan_subset_contains_only_intended_keys(tmp_path):
    """--resubmit-key does not restrict dispatcher selection; the executable
    plan must contain exactly the admitted keys so the reservation covers
    every emitted chain."""
    driver.configure(campaign_dir=tmp_path)
    plan = {
        "routes": [
            {"route_id": "r1_cap25", "arm": "cap25", "jobs": [{"seed": 7, "fold": 0}]},
            {"route_id": "r2_cap25", "arm": "cap25", "jobs": [{"seed": 7, "fold": 0}]},
            {"route_id": "r3_cap50", "arm": "cap50", "jobs": [{"seed": 7, "fold": 1}]},
        ],
        "build_token": "old-token",
        "summary": {"expected_keys": 378},
    }
    subset = driver.plan_subset_for_keys(plan, {"r1_cap25|7|0", "r3_cap50|7|1"})
    assert driver.wave_keys(subset) == ["r1_cap25|7|0", "r3_cap50|7|1"]
    assert subset["build_token"] != "old-token"
    assert subset["summary"]["resolved"] == 2
    assert subset["summary"]["intended_keys"] == ["r1_cap25|7|0", "r3_cap50|7|1"]
    with pytest.raises(ValueError, match="missing from the plan"):
        driver.plan_subset_for_keys(plan, {"r9_cap75|7|0"})
