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


def test_plan_refresh_normalizes_keyless_shared_plan_and_replaced_attempts() -> None:
    """Real shared plans omit registry_key and refreshes must honor replacements."""
    cell = ("daic_text_only", 7, 0)
    plan = {
        "routes": [
            {
                "route_id": "daic_text_only",
                "jobs": [
                    {
                        "seed": 7,
                        "fold": 0,
                        "parent_status": "resolved",
                        "parent": {"attempt_id": "att-1"},
                    }
                ],
            }
        ]
    }
    assert guard.plan_resolved_cells(plan) == {cell}
    assert not guard.plan_needs_refresh({cell: "att-1"}, plan)
    assert guard.plan_needs_refresh({cell: "att-2"}, plan)


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


def test_maybe_refresh_plan_skips_without_lane_planner(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("no subprocess may run without the lane planner")

    monkeypatch.setattr(guard, "TREATMENT_PLANNER", tmp_path / "missing_planner.py")
    monkeypatch.setattr(guard.subprocess, "run", boom)
    attempted = guard.maybe_refresh_plan(
        {("daic_text_only", 7, 0): "att"},
        plan_path=tmp_path / "missing_plan.json",
        matrix_path=tmp_path / "matrix.json",
        evidence_dir=tmp_path,
        run_root=tmp_path,
        runtime_cache_root="/gpfs/example/heads_cache",
        refresh_interval=3600,
    )
    assert attempted is False


def test_maybe_refresh_plan_uses_only_the_lane_planner(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    planner = tmp_path / "planner.py"
    planner.write_text("", encoding="utf-8")
    monkeypatch.setattr(guard, "TREATMENT_PLANNER", planner)
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    attempted = guard.maybe_refresh_plan(
        {("daic_text_only", 7, 0): "att"},
        plan_path=tmp_path / "missing_plan.json",
        matrix_path=tmp_path / "matrix.json",
        evidence_dir=tmp_path,
        run_root=tmp_path,
        runtime_cache_root="/gpfs/example/heads_cache",
        refresh_interval=3600,
    )
    assert attempted is True and len(commands) == 2
    assert str(planner) in commands[0]
    assert any("qwen3_heads_dispatch.py" in str(token) for token in commands[1])
    assert "plan" in commands[1]
    assert not any("qwen3_heads_matrix.py" in str(token) for token in commands[0] + commands[1])


def test_shared_plan_without_registry_key_is_eligible() -> None:
    """The shared dispatch plan does not emit registry_key; the guard derives it."""
    plan = {
        "routes": [
            {
                "route_id": "d3tec_text_only",
                "jobs": [
                    {
                        "seed": 7,
                        "fold": 1,
                        "parent_status": "resolved",
                        "parent": {"attempt_id": "att-f1", "adapter_sha256": "abc"},
                    }
                ],
            }
        ]
    }
    jobs = guard.eligible_head_jobs(
        plan,
        {("d3tec_text_only", 7, 1): "att-f1"},
        set(),
        set(),
        {"d3tec_text_only|7|1"},
    )
    assert [job["key"] for job in jobs] == ["d3tec_text_only|7|1"]


def _write_registry(path: Path, entries: list[dict]) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _registry_entry(**overrides) -> dict:
    entry = {
        "registry_key": "daic_text_only|7|0",
        "attempt_id": "head-att-1",
        "deployment_id": "dep-1",
        "parent_attempt_id": "parent-att-1",
        "extract_job_id": "47080284",
        "classifier_job_id": "47080285",
        "error": None,
    }
    entry.update(overrides)
    return entry


def test_submit_head_consumes_fresh_registry_proof_not_stdout(tmp_path: Path) -> None:
    """The real shared CLI prints a summary; the registry entry is the proof."""
    registry = tmp_path / "head_submissions.jsonl"
    registry.write_text("", encoding="utf-8")
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}

    def runner(command, **kwargs):
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_registry_entry()) + "\n")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="  daic_text_only|7|0: submitted\nregistry updated: 1 entries\n",
            stderr="",
        )

    record = guard.submit_head_via_shared(job, "dep-1", runner, registry_path=registry)
    assert record["status"] == "submitted"
    assert record["attempt_id"] == "head-att-1"
    assert record["job_ids"] == {"extract": "47080284", "classifier": "47080285"}


def test_submit_head_stale_registry_entry_is_not_consumed(tmp_path: Path) -> None:
    registry = tmp_path / "head_submissions.jsonl"
    _write_registry(registry, [_registry_entry(attempt_id="stale-att")])
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="  daic_text_only|7|0: submitted\n", stderr="")

    record = guard.submit_head_via_shared(job, "dep-1", runner, registry_path=registry)
    assert record["status"] == "uncertain"
    assert record["attempt_id"] == "stale-att"  # preserved for ownership


def test_submit_head_timeout_with_fresh_registry_proof_is_submitted(tmp_path: Path) -> None:
    registry = tmp_path / "head_submissions.jsonl"
    registry.write_text("", encoding="utf-8")
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}

    def runner(command, **kwargs):
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_registry_entry(attempt_id="delivered-att")) + "\n")
        raise subprocess.TimeoutExpired(command, 1800, output="", stderr="ssh stalled")

    record = guard.submit_head_via_shared(job, "dep-1", runner, registry_path=registry)
    assert record["status"] == "submitted"
    assert record["attempt_id"] == "delivered-att"


def test_fresh_registry_proof_rejects_mismatches(tmp_path: Path) -> None:
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}
    registry = tmp_path / "head_submissions.jsonl"

    def fresh(entries, job_=job, deployment="dep-1"):
        _write_registry(registry, entries)
        return guard._fresh_registry_delivery(job_["key"], set(), job_, deployment, registry)

    assert fresh([_registry_entry()]) is not None
    assert fresh([_registry_entry(deployment_id="other")]) is None
    assert fresh([_registry_entry(parent_attempt_id="other")]) is None
    assert fresh([_registry_entry(classifier_job_id="47080284")]) is None
    assert fresh([_registry_entry(extract_job_id="abc")]) is None
    assert fresh([_registry_entry(error="boom")]) is None
    assert fresh([_registry_entry(), _registry_entry(attempt_id="second")]) is None


def test_reconcile_delivered_heads_appends_exact_proof_and_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "key": "head::daic_text_only|7|0",
                "stage": "head",
                "status": "uncertain",
                "attempt_id": "head-att-1",
                "parent_attempt_id": "parent-att-1",
                "job_ids": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "head_submissions.jsonl"
    _write_registry(registry, [_registry_entry()])
    summary = guard.reconcile_delivered_heads(ledger, registry)
    assert summary == {"reconciled": 1, "already_submitted": 0, "unresolved": 0}
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["status"] == "submitted"
    assert lines[-1]["attempt_id"] == "head-att-1"
    assert lines[-1]["job_ids"] == {"extract": "47080284", "classifier": "47080285"}
    assert lines[-1]["reconciled"] is True
    assert guard.head_recorded_keys(ledger) == {"daic_text_only|7|0"}
    again = guard.reconcile_delivered_heads(ledger, registry)
    assert again == {"reconciled": 0, "already_submitted": 1, "unresolved": 0}
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


def test_reconcile_leaves_unproven_head_reserved(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "key": "head::daic_text_only|7|0",
                "stage": "head",
                "status": "uncertain",
                "attempt_id": "head-att-1",
                "parent_attempt_id": "parent-att-1",
                "job_ids": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "head_submissions.jsonl"
    _write_registry(registry, [_registry_entry(attempt_id="different-attempt")])
    summary = guard.reconcile_delivered_heads(ledger, registry)
    assert summary == {"reconciled": 0, "already_submitted": 0, "unresolved": 1}
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert guard.head_recorded_keys(ledger) == {"daic_text_only|7|0"}  # reservation kept


def test_submit_head_stdout_blocks_cannot_bypass_registry_rejection(tmp_path: Path) -> None:
    """Structured stdout without an exact fresh registry proof stays uncertain."""
    registry = tmp_path / "head_submissions.jsonl"
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}
    stdout = "=== JOB daic_text_only|7|0 ===\nEXTRACT_ID=4701\nCLASSIFIER_ID=4702\n"

    def runner(command, **kwargs):
        _write_registry(registry, [_registry_entry(parent_attempt_id="other-parent")])
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    record = guard.submit_head_via_shared(job, "dep-1", runner, registry_path=registry)
    assert record["status"] == "uncertain"  # wrong parent: no identity bypass
    assert record["job_ids"] == {"extract": "4701", "classifier": "4702"}  # IDs kept

    def runner_wrong_deployment(command, **kwargs):
        _write_registry(registry, [_registry_entry(deployment_id="other-dep")])
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    record2 = guard.submit_head_via_shared(job, "dep-1", runner_wrong_deployment, registry_path=registry)
    assert record2["status"] == "uncertain"

    def runner_conflicting(command, **kwargs):
        _write_registry(
            registry,
            [_registry_entry(), _registry_entry(extract_job_id="999", classifier_job_id="998")],
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    record3 = guard.submit_head_via_shared(job, "dep-1", runner_conflicting, registry_path=registry)
    assert record3["status"] == "uncertain"  # conflicting registry fail-closed


def _head_uncertain_record(**overrides) -> dict:
    record = {
        "key": "head::daic_text_only|7|0",
        "stage": "head",
        "status": "uncertain",
        "attempt_id": "head-att-1",
        "parent_attempt_id": "parent-att-1",
        "job_ids": {},
    }
    record.update(overrides)
    return record


def test_reconcile_requires_unique_consistent_registry_proof(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    registry = tmp_path / "head_submissions.jsonl"
    ledger.write_text(json.dumps(_head_uncertain_record()) + "\n", encoding="utf-8")

    # Wrong parent attempt: unresolved, nothing appended.
    _write_registry(registry, [_registry_entry(parent_attempt_id="other-parent")])
    assert guard.reconcile_delivered_heads(ledger, registry) == {
        "reconciled": 0,
        "already_submitted": 0,
        "unresolved": 1,
    }
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1

    # Two candidates for the same key+attempt: ambiguous, unresolved.
    _write_registry(registry, [_registry_entry(), _registry_entry(attempt_id="head-att-1")])
    assert guard.reconcile_delivered_heads(ledger, registry)["unresolved"] == 1

    # Recorded requested deployment must match the registry entry.
    ledger.write_text(
        json.dumps(_head_uncertain_record(requested_deployment_id="dep-1")) + "\n",
        encoding="utf-8",
    )
    _write_registry(registry, [_registry_entry(deployment_id="dep-2")])
    assert guard.reconcile_delivered_heads(ledger, registry)["unresolved"] == 1
    _write_registry(registry, [_registry_entry(deployment_id="dep-1")])
    assert guard.reconcile_delivered_heads(ledger, registry)["reconciled"] == 1


def test_fresh_registry_proof_requires_nonblank_attempt_and_parent(tmp_path: Path) -> None:
    job = {"key": "daic_text_only|7|0", "parent_attempt_id": "parent-att-1"}
    registry = tmp_path / "head_submissions.jsonl"

    def fresh(entries, job_=job):
        _write_registry(registry, entries)
        return guard._fresh_registry_delivery(job_["key"], set(), job_, "dep-1", registry)

    assert fresh([_registry_entry(attempt_id="")]) is None
    assert fresh([_registry_entry(attempt_id=None)]) is None
    assert fresh([_registry_entry(parent_attempt_id="")]) is None
    assert fresh([_registry_entry(parent_attempt_id=None)]) is None
    # A blank expected parent can never match a blank entry parent either.
    blank_job = {"key": "daic_text_only|7|0", "parent_attempt_id": ""}
    assert fresh([_registry_entry(parent_attempt_id="")], job_=blank_job) is None


def test_reconcile_refuses_missing_identity_instead_of_matching_empty(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    registry = tmp_path / "head_submissions.jsonl"
    # Record and registry entry both lack attempt/parent: empty strings must not
    # reconcile a delivery.
    ledger.write_text(
        json.dumps(_head_uncertain_record(attempt_id="", parent_attempt_id="")) + "\n",
        encoding="utf-8",
    )
    _write_registry(registry, [_registry_entry(attempt_id="", parent_attempt_id="")])
    assert guard.reconcile_delivered_heads(ledger, registry) == {
        "reconciled": 0,
        "already_submitted": 0,
        "unresolved": 1,
    }
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    # Missing parent on the record: refused even when the entry also lacks it.
    ledger.write_text(
        json.dumps(_head_uncertain_record(parent_attempt_id="")) + "\n", encoding="utf-8"
    )
    _write_registry(registry, [_registry_entry(parent_attempt_id="")])
    assert guard.reconcile_delivered_heads(ledger, registry)["unresolved"] == 1
