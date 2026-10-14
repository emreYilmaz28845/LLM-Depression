"""Fake-run batch proofs for the terminal-cell collector.

These tests pin the collection-correctness rules: already validated cells are
skipped on durable receipts or managed lifecycle evidence that proves the exact
current attempt (never a stale prior attempt), the per-pass limit applies only
to attempted cells so later cells cannot starve, collect success and validation
success are counted separately, and a repeatedly failing validator moves the
cell to explicit diagnosis instead of looping forever.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import collect_legacy_prompt_cells as cells


def make_records(count: int) -> list[dict]:
    return [
        {
            "key": f"route|s7|f{index}",
            "run_name": f"run_{index}",
            "attempt_id": f"att-{index}",
            "status": "submitted",
            "job_ids": {"train": str(1000 + index), "best_eval": str(2000 + index)},
        }
        for index in range(count)
    ]


def completed_states(records: list[dict]) -> dict[str, str]:
    states: dict[str, str] = {}
    for record in records:
        states[record["job_ids"]["train"]] = "COMPLETED"
        states[record["job_ids"]["best_eval"]] = "COMPLETED"
    return states


def test_two_passes_advance_through_twelve_distinct_cells(tmp_path: Path) -> None:
    records = make_records(12)
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"
    validated: list[str] = []

    def collect(record: dict) -> int:
        return 0

    def validate(record: dict) -> int:
        validated.append(record["key"])
        return 0

    first = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=set(),
        collect=collect,
        validate=validate,
        limit=6,
    )
    second = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=set(),
        collect=collect,
        validate=validate,
        limit=6,
    )
    assert first == {
        "attempted": 6,
        "collected_ok": 6,
        "validated_ok": 6,
        "failed_collect": 0,
        "failed_validate": 0,
        "skipped_receipt": 0,
        "skipped_lifecycle": 0,
        "skipped_diagnosis": 0,
    }
    assert second["validated_ok"] == 6 and second["skipped_receipt"] == 6 and second["attempted"] == 6
    assert validated == [f"route|s7|f{index}" for index in range(12)]
    assert cells.successful_receipts(receipts) == {
        (f"route|s7|f{index}", f"att-{index}") for index in range(12)
    }


def test_failed_validate_is_not_counted_and_moves_to_diagnosis(tmp_path: Path) -> None:
    records = make_records(1)
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"

    def collect(record: dict) -> int:
        return 0

    def validate(record: dict) -> int:
        return 1

    for _ in range(2):
        summary = cells.run_pass(
            records,
            states,
            receipts_path=receipts,
            failures_path=failures,
            lifecycle=set(),
            collect=collect,
            validate=validate,
            limit=6,
        )
        assert summary["validated_ok"] == 0
        assert summary["collected_ok"] == 1
        assert summary["failed_validate"] == 1
    assert cells.successful_receipts(receipts) == set()
    third = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=set(),
        collect=collect,
        validate=validate,
        limit=6,
    )
    assert third["attempted"] == 0 and third["skipped_diagnosis"] == 1


def test_failed_collect_is_separate_and_does_not_call_validate(tmp_path: Path) -> None:
    records = make_records(1)
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"

    def collect(record: dict) -> int:
        return 1

    def validate(record: dict) -> int:
        raise AssertionError("validate must not run after a failed collect")

    summary = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=set(),
        collect=collect,
        validate=validate,
        limit=6,
    )
    assert summary["attempted"] == 1 and summary["failed_collect"] == 1
    assert summary["collected_ok"] == 0 and summary["validated_ok"] == 0
    failure_records = cells.load_jsonl(failures)
    assert failure_records[0]["stage"] == "collect" and failure_records[0]["ok"] is False


def test_lifecycle_validated_cells_skip_without_receipts(tmp_path: Path) -> None:
    records = make_records(2)
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"
    validated: list[str] = []

    summary = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle={(records[0]["run_name"], records[0]["attempt_id"], 0)},
        collect=lambda record: 0,
        validate=lambda record: validated.append(record["key"]) or 0,
        limit=6,
    )
    assert summary["skipped_lifecycle"] == 1 and summary["validated_ok"] == 1
    assert validated == [records[1]["key"]]


def test_stale_receipt_for_prior_attempt_does_not_skip_new_attempt(tmp_path: Path) -> None:
    records = make_records(1)
    record = records[0]
    record["attempt_id"] = "att-new"
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"
    cells.append_jsonl(
        receipts,
        {"key": record["key"], "attempt_id": "att-old", "stage": "validate", "ok": True, "rc": 0},
    )
    validated: list[str] = []
    summary = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=set(),
        collect=lambda item: 0,
        validate=lambda item: validated.append(item["attempt_id"]) or 0,
        limit=6,
    )
    assert summary["attempted"] == 1 and summary["skipped_receipt"] == 0
    assert summary["validated_ok"] == 1 and validated == ["att-new"]
    # Append-only: the prior attempt's receipt is preserved alongside the new one.
    assert cells.successful_receipts(receipts) == {
        (record["key"], "att-old"),
        (record["key"], "att-new"),
    }


def test_stale_lifecycle_proof_does_not_skip_different_attempt(tmp_path: Path) -> None:
    records = make_records(1)
    record = records[0]
    record["attempt_id"] = "att-new"
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    failures = tmp_path / "failures.jsonl"
    stale = {(record["run_name"], "att-old", 0)}
    summary = cells.run_pass(
        records,
        states,
        receipts_path=receipts,
        failures_path=failures,
        lifecycle=stale,
        collect=lambda item: 0,
        validate=lambda item: 0,
        limit=6,
    )
    assert summary["attempted"] == 1 and summary["skipped_lifecycle"] == 0
    # Isolate the lifecycle shortcut with a fresh receipt ledger for this run.
    current = {(record["run_name"], "att-new", 0)}
    fresh_receipts = tmp_path / "receipts_lifecycle.jsonl"
    summary = cells.run_pass(
        records,
        states,
        receipts_path=fresh_receipts,
        failures_path=failures,
        lifecycle=current,
        collect=lambda item: 0,
        validate=lambda item: 0,
        limit=6,
    )
    assert summary["attempted"] == 0 and summary["skipped_lifecycle"] == 1


def test_lifecycle_scan_reads_exact_attempt_and_fold(tmp_path: Path) -> None:
    fold = tmp_path / "text_only/daic/run_x/fold_2"
    fold.mkdir(parents=True)
    (fold / "status.json").write_text(
        json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": "att-9", "fold": 2}),
        encoding="utf-8",
    )
    other = tmp_path / "text_only/daic/run_y/fold_0"
    other.mkdir(parents=True)
    (other / "status.json").write_text(
        json.dumps({"state": "LOCALLY_VALIDATED"}),
        encoding="utf-8",
    )
    proofs = cells.lifecycle_validated(tmp_path)
    assert proofs == {("run_x", "att-9", 2)}


def test_prior_attempt_failures_do_not_block_new_attempt(tmp_path: Path) -> None:
    records = make_records(1)
    record = records[0]
    record["attempt_id"] = "att-new"
    states = completed_states(records)
    failures = tmp_path / "failures.jsonl"
    for _ in range(2):
        cells.append_jsonl(
            failures,
            {
                "key": record["key"],
                "attempt_id": "att-old",
                "stage": "validate",
                "ok": False,
                "rc": 1,
            },
        )
    summary = cells.run_pass(
        records,
        states,
        receipts_path=tmp_path / "receipts.jsonl",
        failures_path=failures,
        lifecycle=set(),
        collect=lambda item: 0,
        validate=lambda item: 0,
        limit=6,
    )
    assert summary["attempted"] == 1 and summary["skipped_diagnosis"] == 0
    assert summary["validated_ok"] == 1
    assert len(cells.load_jsonl(failures)) == 2  # append-only history preserved

    # The current attempt's own two failures still move it to diagnosis.
    for _ in range(2):
        cells.append_jsonl(
            failures,
            {
                "key": record["key"],
                "attempt_id": "att-new",
                "stage": "validate",
                "ok": False,
                "rc": 1,
            },
        )
    blocked = cells.run_pass(
        records,
        states,
        receipts_path=tmp_path / "fresh_receipts.jsonl",
        failures_path=failures,
        lifecycle=set(),
        collect=lambda item: 0,
        validate=lambda item: 0,
        limit=6,
    )
    assert blocked["attempted"] == 0 and blocked["skipped_diagnosis"] == 1


def test_limit_applies_only_to_attempted_cells(tmp_path: Path) -> None:
    records = make_records(10)
    states = completed_states(records)
    receipts = tmp_path / "receipts.jsonl"
    for record in records[:6]:
        cells.append_jsonl(
            receipts,
            {
                "key": record["key"],
                "attempt_id": record["attempt_id"],
                "stage": "validate",
                "ok": True,
                "rc": 0,
            },
        )
    plan, counters = cells.plan_pass(
        records,
        states,
        receipts=cells.successful_receipts(receipts),
        lifecycle=set(),
        failures={},
        limit=3,
    )
    assert counters["skipped_receipt"] == 6
    assert [record["key"] for record in plan] == [f"route|s7|f{index}" for index in (6, 7, 8)]
