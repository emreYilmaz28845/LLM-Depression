"""Tests for the guarded head-chain integration (lane-owned wrapper).

Pinned semantics: head deliveries need an exact two-job numeric proof, head
chains only bind to the exact validated parent attempt, head jobs count in the
same 80-slot own accounting as fits, eligible head chains are submitted before
new fits within the shared live budget, uncertain head records preserve known
IDs and exclude the key from automatic reissue, and stale plans are rebuilt
when newly validated parents land.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import qwen3_legacy_prompt_dispatch as dispatch
from tools import qwen3_legacy_prompt_heads_guard as guard


def test_valid_head_delivery_requires_exact_unique_numeric_pair() -> None:
    assert guard.valid_head_delivery({"extract_job_id": "111", "classifier_job_id": "222"})
    assert not guard.valid_head_delivery({"extract_job_id": "111"})
    assert not guard.valid_head_delivery({"extract_job_id": "111", "classifier_job_id": "111"})
    assert not guard.valid_head_delivery({"extract_job_id": "abc", "classifier_job_id": "222"})
    assert not guard.valid_head_delivery({"extract_job_id": "111", "classifier_job_id": "222", "error": "boom"})
    assert not guard.valid_head_delivery(
        {"extract_job_id": "111", "classifier_job_id": "222", "extra": "333"}
    )


def test_parse_head_submit_output_blocks() -> None:
    parsed = guard.parse_head_submit_output(
        "=== JOB route|7|0 ===\nEXTRACT_ID=4701\nCLASSIFIER_ID=4702\n"
        "=== JOB route|7|1 ===\nERROR=sbatch failed\n"
    )
    assert parsed["route|7|0"] == {"extract_job_id": "4701", "classifier_job_id": "4702"}
    assert parsed["route|7|1"] == {"error": "sbatch failed"}


def make_plan(entries: list[dict]) -> dict:
    return {"routes": [{"route_id": "daic_text_only", "jobs": entries}]}


def test_eligibility_requires_the_exact_validated_parent_attempt() -> None:
    plan = make_plan(
        [
            {
                "registry_key": "daic_text_only|7|0",
                "seed": 7,
                "fold": 0,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att-good"},
            },
            {
                "registry_key": "daic_text_only|7|1",
                "seed": 7,
                "fold": 1,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att-stale"},
            },
        ]
    )
    validated = {("daic_text_only", 7, 0): "att-good", ("daic_text_only", 7, 1): "att-new"}
    eligible = guard.eligible_head_jobs(plan, validated, submitted_keys=set())
    assert [job["key"] for job in eligible] == ["daic_text_only|7|0"]
    # Already submitted keys are skipped.
    assert guard.eligible_head_jobs(plan, validated, {"daic_text_only|7|0"}) == []


def test_eligibility_is_restricted_to_approved_native_keys() -> None:
    plan = make_plan(
        [
            {
                "registry_key": "daic_text_only|7|0",
                "seed": 7,
                "fold": 0,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att-good"},
            },
            {
                "registry_key": "d3tec_audio_text_english|7|0",
                "seed": 7,
                "fold": 0,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att-other"},
            },
        ]
    )
    validated = {
        ("daic_text_only", 7, 0): "att-good",
        ("d3tec_audio_text_english", 7, 0): "att-other",
    }
    eligible = guard.eligible_head_jobs(
        plan,
        validated,
        submitted_keys=set(),
        approved_keys={"daic_text_only|7|0"},
    )
    assert [job["key"] for job in eligible] == ["daic_text_only|7|0"]


def test_validated_cells_requires_exact_attempt(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "key": "daic_text_only|s7|f0",
                "run_name": "run_a",
                "status": "submitted",
                "attempt_id": "att-new",
                "job_ids": {"train": "1", "best_eval": "2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        json.dumps(
            {
                "key": "daic_text_only|s7|f0",
                "attempt_id": "att-old",
                "stage": "validate",
                "ok": True,
                "rc": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert guard.validated_cells(receipts, set(), ledger) == {}
    receipts.write_text(
        receipts.read_text(encoding="utf-8")
        + json.dumps(
            {
                "key": "daic_text_only|s7|f0",
                "attempt_id": "att-new",
                "stage": "validate",
                "ok": True,
                "rc": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert guard.validated_cells(receipts, set(), ledger) == {("daic_text_only", 7, 0): "att-new"}


def test_accounting_includes_head_deliveries(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "head_attempts/att-1").mkdir(parents=True)
    (evidence / "head_submissions.jsonl").write_text(
        json.dumps({"registry_key": "route|7|0", "extract_job_id": "4701", "classifier_job_id": "4702"})
        + "\n",
        encoding="utf-8",
    )
    (evidence / "head_attempts/att-1/jobs.jsonl").write_text(
        json.dumps({"job_key": "extract", "event_type": "SUBMITTED", "slurm_job_id": "4703"}) + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text("", encoding="utf-8")
    ids, uncertain, unknown = dispatch.own_job_ids(ledger, tmp_path / "empty", evidence)
    assert set(ids) == {"4701", "4702", "4703"} and unknown == [] and uncertain == 0


def test_pass_prioritizes_heads_within_the_shared_budget() -> None:
    state = {"own": 76}
    order: list[str] = []

    def reconcile():
        return state["own"], 0

    def submit_head(job):
        state["own"] += 2
        order.append("head")
        return {"status": "submitted", "job_ids": {"extract": "1", "classifier": "2"}}

    def submit_fit(fit):
        state["own"] += 2
        order.append("fit")
        return {"status": "submitted", "job_ids": {"train": "3", "best_eval": "4"}}

    summary = guard.run_campaign_pass(
        head_jobs=[{"key": "r|7|0", "parent_attempt_id": "att"}],
        fit_jobs=[{"key": "r|s7|f0"}, {"key": "r|s7|f1"}],
        reconcile=reconcile,
        submit_head=submit_head,
        submit_fit=submit_fit,
        on_record=lambda record: None,
        max_fits=40,
    )
    assert order == ["head", "fit"]  # heads first, then one fit fills the budget
    assert summary["heads_submitted"] == 1 and summary["fits_submitted"] == 1
    assert summary["refused"] == 1 and "no lane headroom" in summary["reason"]


def test_pass_refuses_without_headroom_and_submits_nothing() -> None:
    def reconcile():
        return 79, 0

    summary = guard.run_campaign_pass(
        head_jobs=[{"key": "r|7|0"}],
        fit_jobs=[{"key": "r|s7|f0"}],
        reconcile=reconcile,
        submit_head=lambda job: (_ for _ in ()).throw(AssertionError("must not submit")),
        submit_fit=lambda fit: (_ for _ in ()).throw(AssertionError("must not submit")),
        on_record=lambda record: None,
        max_fits=40,
    )
    assert summary["heads_submitted"] == 0 and summary["fits_submitted"] == 0
    assert summary["refused"] == 1


def test_pass_stops_on_unproven_head_delivery() -> None:
    def reconcile():
        return 70, 0

    calls = {"fits": 0}

    summary = guard.run_campaign_pass(
        head_jobs=[{"key": "r|7|0"}],
        fit_jobs=[{"key": "r|s7|f0"}],
        reconcile=reconcile,
        submit_head=lambda job: {"status": "uncertain", "job_ids": {}},
        submit_fit=lambda fit: calls.__setitem__("fits", calls["fits"] + 1)
        or {"status": "submitted"},
        on_record=lambda record: None,
        max_fits=40,
    )
    assert summary["heads_submitted"] == 0 and summary["refused"] == 1
    assert calls["fits"] == 0  # the pass stops; no silent fallback to fits


def test_uncertain_head_record_preserves_known_ids() -> None:
    record = guard.head_ledger_record(
        {"key": "r|7|0", "parent_attempt_id": "att"},
        {"extract_job_id": "4701"},
        "att-head",
        "uncertain",
        tail="partial",
    )
    assert record["status"] == "uncertain"
    assert record["job_ids"] == {"extract": "4701"}
    assert record["attempt_id"] == "att-head"


def test_timeout_preserves_partial_stdout_and_ids() -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            1800,
            output="=== JOB r|7|0 ===\nEXTRACT_ID=4701\n",
            stderr="ssh stalled",
        )

    job = {"key": "r|7|0", "parent_attempt_id": "att"}
    record = guard.submit_head_via_shared(job, "dep", runner)
    assert record["status"] == "uncertain"
    assert record["job_ids"] == {"extract": "4701"}
    assert "EXTRACT_ID=4701" in record["tail"] and "ssh stalled" in record["tail"]


def test_timeout_then_record_then_next_pass_excludes(tmp_path: Path) -> None:
    plan = make_plan(
        [
            {
                "registry_key": "daic_text_only|7|0",
                "seed": 7,
                "fold": 0,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att-good"},
            }
        ]
    )
    validated = {("daic_text_only", 7, 0): "att-good"}
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text("", encoding="utf-8")

    def on_record(record: dict) -> None:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    summary = guard.run_campaign_pass(
        head_jobs=[{"key": "daic_text_only|7|0"}],
        fit_jobs=[],
        reconcile=lambda: (70, 0),
        submit_head=lambda job: guard.head_ledger_record(
            job, {"extract_job_id": "4701"}, "att-head", "uncertain", tail="timeout"
        ),
        submit_fit=lambda fit: {"status": "submitted"},
        on_record=on_record,
        max_fits=1,
    )
    assert summary["heads_submitted"] == 0 and summary["refused"] == 1
    recorded = guard.head_recorded_keys(ledger)
    assert recorded == {"daic_text_only|7|0"}
    # No automatic unchanged retry: the recorded key is excluded even though no
    # registry entry exists.
    assert guard.eligible_head_jobs(plan, validated, set(), recorded) == []


def test_plan_refresh_detection() -> None:
    cell = ("daic_text_only", 7, 0)
    assert guard.plan_needs_refresh({cell: "att"}, None)
    resolved_plan = make_plan(
        [
            {
                "registry_key": "daic_text_only|7|0",
                "seed": 7,
                "fold": 0,
                "parent_status": "resolved",
                "parent": {"attempt_id": "att"},
            }
        ]
    )
    assert not guard.plan_needs_refresh({cell: "att"}, resolved_plan)
    assert guard.plan_needs_refresh({("daic_text_only", 7, 1): "att"}, resolved_plan)


def test_maybe_refresh_plan_is_throttled(tmp_path: Path) -> None:
    stamp = tmp_path / ".head_plan_refresh_stamp"
    stamp.write_text("{}", encoding="utf-8")
    attempted = guard.maybe_refresh_plan(
        {("daic_text_only", 7, 0): "att"},
        plan_path=tmp_path / "missing_plan.json",
        matrix_path=tmp_path / "matrix.json",
        evidence_dir=tmp_path,
        run_root=tmp_path,
        runtime_cache_root="/gpfs/example/heads_cache",
        refresh_interval=3600,
    )
    assert attempted is False  # fresh stamp: no rebuild subprocess
