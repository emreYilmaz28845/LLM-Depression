"""Tests for the incremental window-cap collect/validate driver.

Covers: exact-owned 0:0 fit/head selection (unknowns and superseded preserved),
production-only head keys (smokes excluded), ledger idempotency, both-variant
mask-membership checks, official-tool delegation (no --delete, no adapters),
and the watcher exit gate (never exit before 378 validated chains).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qwen3_window_cap_collect_validate as cv  # noqa: E402
from tools import qwen3_window_cap_head_plan as planner  # noqa: E402


def make_row(arm: str = "cap25", *, seed: int = 7, fold: int = 0, attempt: str = "ATT") -> dict:
    return {
        "route_id": "androids_interview_audio_only_native",
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "dataset": "androids_interview",
        "modality": "audio_only",
        "run_name": f"windowcap_androids_interview_audio_only_s{seed}_f{fold}_{arm}",
        "fraction": {"cap25": 0.25, "cap50": 0.5, "cap75": 0.75}[arm],
        "attempt_id": attempt,
        "config": f"configs/experiments/window_cap/androids_audio_only_{arm}.yaml",
    }


def make_fold(root: Path, row: dict) -> Path:
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
    (fold / "metadata.json").write_text(json.dumps({"attempt_id": row["attempt_id"]}), encoding="utf-8")
    events = [
        {"event_type": "SUBMITTED", "job_key": "train", "slurm_job_id": "1001", "status": "PENDING"},
        {"event_type": "SUBMITTED", "job_key": "best_eval", "slurm_job_id": "1002", "status": "PENDING"},
        {"event_type": "COMPLETED", "job_key": "train", "exit_code": None, "status": "COMPLETED"},
        {"event_type": "COMPLETED", "job_key": "best_eval", "exit_code": None, "status": "COMPLETED"},
    ]
    (fold / "jobs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    logs = fold / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "split_used.json").write_text(
        json.dumps({"train_subject_ids": ["s1"], "selection_subject_ids": ["s2"], "final_eval_subject_ids": ["s3"]}),
        encoding="utf-8",
    )
    return fold


GOOD_SCHED = {"1001": {"state": "COMPLETED", "exit": "0:0"}, "1002": {"state": "COMPLETED", "exit": "0:0"}}


# --- fit selection -----------------------------------------------------------


def test_select_ready_fits_requires_confirmed_0_0(tmp_path):
    row = make_row()
    make_fold(tmp_path, row)
    ready, reasons = cv.select_ready_fits([row], GOOD_SCHED, {}, raw_root=tmp_path)
    assert len(ready) == 1
    running = {"1001": {"state": "RUNNING", "exit": "0:0"}, "1002": {"state": "PENDING", "exit": "0:0"}}
    ready, reasons = cv.select_ready_fits([row], running, {}, raw_root=tmp_path)
    assert not ready and any("RUNNING" in r for r in reasons)
    ready, reasons = cv.select_ready_fits([row], {}, {}, raw_root=tmp_path)
    assert not ready and any("UNKNOWN" in r for r in reasons)


def test_select_ready_fits_skips_validated(tmp_path):
    row = make_row()
    make_fold(tmp_path, row)
    ledger = {cv.fit_key(row): {"validated": True, "attempt_id": row["attempt_id"]}}
    ready, reasons = cv.select_ready_fits([row], GOOD_SCHED, ledger, raw_root=tmp_path)
    assert not ready and reasons.get("already validated") == 1
    old_ledger = {cv.fit_key(row): {"validated": True, "attempt_id": "OLD-ATTEMPT"}}
    ready, reasons = cv.select_ready_fits([row], GOOD_SCHED, old_ledger, raw_root=tmp_path)
    assert ready and not reasons.get("already validated")


# --- head selection ----------------------------------------------------------


def make_entry(
    key: str,
    *,
    extract: str = "2001",
    classifier: str = "2002",
    attempt: str = "HEADATT",
    local_mirror=None,
) -> dict:
    entry = {
        "registry_key": key,
        "attempt_id": attempt,
        "extract_job_id": extract,
        "classifier_job_id": classifier,
    }
    if local_mirror is not None:
        entry["local_mirror"] = str(local_mirror)
    return entry


def test_select_ready_heads_production_only_and_confirmed(tmp_path):
    expected = {planner.head_key(make_row("cap25"))}
    prod = make_entry(f"{planner.head_route_id(make_row('cap25'))}|7|0")
    smoke = make_entry("daic_audio_only_native|7|0")
    sched = {"2001": {"state": "COMPLETED", "exit": "0:0"}, "2002": {"state": "COMPLETED", "exit": "0:0"}}
    ready, reasons = cv.select_ready_heads([prod, smoke], expected, sched, {})
    assert [e["registry_key"] for e in ready] == [prod["registry_key"]]
    assert reasons.get("not a production treatment chain") == 1
    partial = {"2001": {"state": "COMPLETED", "exit": "0:0"}, "2002": {"state": "PENDING", "exit": "0:0"}}
    ready, reasons = cv.select_ready_heads([prod], expected, partial, {})
    assert not ready and any("PENDING" in r for r in reasons)
    ready, reasons = cv.select_ready_heads([prod], expected, {}, {})
    assert not ready and any("UNKNOWN" in r for r in reasons)


def test_select_ready_heads_skips_validated_and_missing_ids():
    expected = {planner.head_key(make_row("cap25"))}
    prod = make_entry(f"{planner.head_route_id(make_row('cap25'))}|7|0")
    ledger = {prod["registry_key"]: {"validated": True, "attempt_id": prod["attempt_id"]}}
    ready, reasons = cv.select_ready_heads([prod], expected, {}, ledger)
    assert not ready and reasons.get("already validated") == 1
    old_ledger = {prod["registry_key"]: {"validated": True, "attempt_id": "OLD-ATTEMPT"}}
    ready, reasons = cv.select_ready_heads([prod], expected, {}, old_ledger)
    assert not ready and reasons.get("already validated") is None
    broken = make_entry(prod["registry_key"], extract="", classifier="")
    ready, reasons = cv.select_ready_heads([broken], expected, {}, {})
    assert not ready and reasons.get("missing numeric job ids") == 1


# --- membership --------------------------------------------------------------


def membership_fixture():
    from src.data.window_cap import compute_selection_sha256

    mask = {
        "algorithm_version": "sha256-subject-permutation-v1",
        "sampling_seed": 1337,
        "fraction": 0.25,
        "subjects": {"s1": ["a", "b"], "s2": ["c"]},
        "total_selected": 3,
    }
    mask["selection_sha256"] = compute_selection_sha256(
        mask["algorithm_version"], mask["sampling_seed"], mask["fraction"], mask["subjects"]
    )
    window_cap = {
        "selection_sha256": mask["selection_sha256"],
        "baseline_input_sha256": "b" * 64,
        "fraction": 0.25,
        "sampling_seed": 1337,
        "algorithm_version": mask["algorithm_version"],
    }
    extraction = {
        "parent_attempt_id": "PARENT",
        "fold": 0,
        "adapter_config_sha256": "c" * 64,
        "adapter_sha256": "d" * 64,
    }
    head_metadata = {
        "attempt_id": "HEAD",
        "seed": 7,
        "fold": 0,
        "parent": {"parent_attempt_id": "PARENT"},
        "source": {"deployment_id": "DEPLOY"},
    }
    variant = {
        "train_mask": {
            **window_cap,
            "capped_train_row_ids": ["a", "b", "c"],
            "retained_val_row_ids": [],
            "retained_val_subject_count": 0,
        },
        "training_row_ids": ["a", "b", "c"],
        "parent_attempt_id": "PARENT",
        "fold": 0,
        "checkpoint_hashes": {"adapter_config_sha256": "c" * 64, "adapter_sha256": "d" * 64},
        "cache_identity": {
            "extraction_metadata.json": {"sha256": "E"},
            "outer_train_rows.jsonl": {"sha256": "P"},
        },
        "fit_weight_audit": {
            "schema_version": "hidden_classifier_weight_audit.v1",
            "policy": "uniform_rows",
            "row_count": 3,
            "mean_weight": 1.0,
            "subject_count": 2,
        },
    }
    return mask, window_cap, extraction, head_metadata, variant


def run_membership(mask, window_cap, extraction, head_metadata, variants):
    return cv.membership_issues(
        parent_attempt_id="PARENT",
        head_attempt_id="HEAD",
        fold=0,
        parent_training_seed=7,
        mask=mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants=variants,
        extraction_sha256="E",
        expected_head_deployment="DEPLOY",
        train_subjects={"s1", "s2"},
        val_subjects=set(),
        pool_rows=[
            {"subject_id": "s1", "sample_id": "a"},
            {"subject_id": "s1", "sample_id": "b"},
            {"subject_id": "s2", "sample_id": "c"},
        ],
        pool_rows_sha256="P",
    )


def test_membership_passes_for_both_variants():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    assert run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": variant, "xgb_raw": dict(variant)}) == []


def test_membership_flags_wrong_rows_and_missing_variant():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    bad = {**variant, "train_mask": dict(variant["train_mask"])}
    bad["train_mask"]["capped_train_row_ids"] = ["a", "b"]
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": bad, "xgb_raw": variant})
    assert any("capped train rows do not equal mask membership" in i for i in issues)
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": variant})
    assert any("xgb_raw: classifier metadata missing" in i for i in issues)
    bad_hash = dict(variant)
    bad_hash["train_mask"] = {**variant["train_mask"], "selection_sha256": "f" * 64}
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": bad_hash, "xgb_raw": variant})
    assert any("train_mask selection_sha256 mismatch" in i for i in issues)


def test_membership_binds_parent_not_head_attempt():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    # extraction/head parent fields must bind the PARENT attempt; a head-attempt
    # value there is a failure (this was the pre-fix confusion).
    bad_extraction = dict(extraction)
    bad_extraction["parent_attempt_id"] = "HEAD"
    issues = run_membership(mask, window_cap, bad_extraction, head_metadata, {"logreg_raw": variant, "xgb_raw": variant})
    assert any("extraction parent attempt mismatch" in i for i in issues)
    # head identity is checked separately against head_metadata.attempt_id
    bad_head = dict(head_metadata)
    bad_head["attempt_id"] = "OTHER"
    issues = run_membership(mask, window_cap, extraction, bad_head, {"logreg_raw": variant, "xgb_raw": variant})
    assert any("head attempt identity mismatch" in i for i in issues)


# --- official runners --------------------------------------------------------


def test_process_fits_batch_collect_status_validate_and_lifecycle(tmp_path, monkeypatch):
    row = make_row()
    calls = []
    fold = tmp_path / "fold_0"

    def runner(argv):
        calls.append(argv)
        if len(argv) > 2 and argv[2] == "validate":
            fold.mkdir(exist_ok=True)
            (fold / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED"}), encoding="utf-8")
        return 0, "ok", ""

    monkeypatch.setattr(cv, "append_ledger", lambda record, path=None: None)
    records = cv.process_fits_batch([row], runner=runner, fold_resolver=lambda attempt: fold)
    assert [c[2] for c in calls] == ["collect", "status", "validate"]
    assert "--execute" in calls[0] and "--attempt-id" in calls[0]
    assert records[0]["validated"] is True and records[0]["final_state"] == "LOCALLY_VALIDATED"
    assert all("--delete" not in " ".join(call) for call in calls)


def test_process_fits_batch_fails_closed_without_lifecycle_advance(tmp_path, monkeypatch):
    row = make_row()
    fold = tmp_path / "fold_0"

    def runner(argv):
        if len(argv) > 2 and argv[2] == "validate":
            fold.mkdir(exist_ok=True)
            (fold / "status.json").write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
        return 0, "ok", ""

    monkeypatch.setattr(cv, "append_ledger", lambda record, path=None: None)
    records = cv.process_fits_batch([row], runner=runner, fold_resolver=lambda attempt: fold)
    assert records[0]["validated"] is False
    assert "not advanced" in records[0]["note"]


def test_collect_fit_skips_when_already_collected(tmp_path):
    row = make_row()
    fold = tmp_path / "fold_0"
    fold.mkdir()
    (fold / "status.json").write_text("{}", encoding="utf-8")
    (fold / "run_config.yaml").write_text("", encoding="utf-8")
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0, "ok", ""

    record = cv.collect_fit(row, runner=runner, fold_resolver=lambda attempt: fold)
    assert record["collect_ok"] is True and calls == []
    assert "already collected" in record["note"]


def test_process_fits_batch_collect_failure_skips_validate(tmp_path, monkeypatch):
    row = make_row()
    calls = []

    def runner(argv):
        calls.append(argv)
        return 1, "", "boom"

    monkeypatch.setattr(cv, "append_ledger", lambda record, path=None: None)
    records = cv.process_fits_batch([row], runner=runner, fold_resolver=lambda attempt: None)
    assert records[0]["validated"] is False and records[0]["collect_ok"] is False
    assert len(calls) == 1  # no status, no validate


def test_process_head_passes_registry_and_gates_on_membership(tmp_path):
    cv.configure(campaign_dir=tmp_path)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    entry = make_entry("androids_interview_audio_only_native_cap25|7|0", local_mirror=mirror)
    calls = []

    def runner(argv):
        calls.append(argv)
        if len(argv) > 2 and argv[2] == "validate":
            (mirror / "status.json").write_text(
                json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": entry["attempt_id"]}),
                encoding="utf-8",
            )
        return 0, "ok", ""

    ok = cv.process_head(entry, runner=runner, membership_checker=lambda e: [])
    assert ok["validated"] is True and len(calls) == 2
    assert calls[0][2] == "collect" and "--registry" in calls[0]
    assert calls[1][2] == "validate" and "--registry" in calls[1]
    calls.clear()
    bad = cv.process_head(entry, runner=runner, membership_checker=lambda e: ["rows mismatch"])
    assert bad["validated"] is False and bad["membership_ok"] is False
    assert len(calls) == 1  # validate never ran


def test_process_head_no_fake_rc0_acceptance(tmp_path):
    cv.configure(campaign_dir=tmp_path)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    entry = make_entry("androids_interview_audio_only_native_cap25|7|0", local_mirror=mirror)

    def runner(argv):
        return 0, "ok", ""

    record = cv.process_head(entry, runner=runner, membership_checker=lambda e: [])
    assert record["validated"] is False and "status.json missing" in record["note"]
    (mirror / "status.json").write_text(
        json.dumps({"state": "SYNCED_LOCALLY", "attempt_id": entry["attempt_id"]}), encoding="utf-8"
    )
    record = cv.process_head(entry, runner=runner, membership_checker=lambda e: [])
    assert record["validated"] is False and "not advanced" in record["note"]
    (mirror / "status.json").write_text(
        json.dumps({"state": "LOCALLY_VALIDATED", "attempt_id": "OTHER"}), encoding="utf-8"
    )
    record = cv.process_head(entry, runner=runner, membership_checker=lambda e: [])
    assert record["validated"] is False and "attempt mismatch" in record["note"]


# --- ledger and watcher gate -------------------------------------------------


def test_ledger_last_wins(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    cv.append_ledger({"key": "k", "validated": False}, ledger)
    cv.append_ledger({"key": "k", "validated": True}, ledger)
    records = cv.load_ledger(ledger)
    assert records["k"]["validated"] is True


def test_watcher_done_requires_all_chains_validated():
    assert (
        cv.watcher_done(
            {"fits_submitted": 378, "fits_validated": 377, "heads_dispatched": 378, "heads_validated": 378}
        )
        is False
    )
    assert (
        cv.watcher_done(
            {"fits_submitted": 378, "fits_validated": 378, "heads_dispatched": 378, "heads_validated": 377}
        )
        is False
    )
    assert (
        cv.watcher_done(
            {"fits_submitted": 378, "fits_validated": 378, "heads_dispatched": 378, "heads_validated": 378}
        )
        is True
    )
    assert (
        cv.watcher_done(
            {"fits_submitted": 40, "fits_validated": 4, "heads_dispatched": 0, "heads_validated": 0}
        )
        is False
    )


def test_heads_old_attempt_validation_ignored(tmp_path):
    cv.configure(campaign_dir=tmp_path)
    key = "androids_interview_audio_only_native_cap25|7|0"
    entry_old = make_entry(key, attempt="OLD")
    entry_new = make_entry(key, attempt="NEW")
    expected = {key}
    ledger = {key: {"validated": True, "attempt_id": "OLD"}}
    doc = cv.progress([], [entry_old, entry_new], expected, ledger)
    assert doc["heads_dispatched"] == 1 and doc["heads_validated"] == 0
    ledger[key] = {"validated": True, "attempt_id": "NEW"}
    doc = cv.progress([], [entry_old, entry_new], expected, ledger)
    assert doc["heads_validated"] == 1


def test_fetch_remote_file_publishes_only_on_hash_match(tmp_path):
    import hashlib

    def downloader(remote, dest):
        dest.write_bytes(b"hello")
        return True

    ok, reason = cv.fetch_remote_file(
        "remote/f.json",
        tmp_path / "f.json",
        remote_hasher=lambda remote: "a" * 64,
        downloader=downloader,
    )
    assert not ok and "mismatch" in reason
    assert not (tmp_path / "f.json").exists()
    assert not (tmp_path / "f.json.part").exists()
    expected = hashlib.sha256(b"hello").hexdigest()
    ok, reason = cv.fetch_remote_file(
        "remote/f.json",
        tmp_path / "f.json",
        remote_hasher=lambda remote: expected,
        downloader=downloader,
    )
    assert ok and (tmp_path / "f.json").read_bytes() == b"hello"
    assert not (tmp_path / "f.json.part").exists()


def test_fetch_remote_file_refuses_existing_mismatch(tmp_path):
    import hashlib

    local = tmp_path / "f.json"
    local.write_bytes(b"old")
    expected = hashlib.sha256(b"new").hexdigest()
    ok, reason = cv.fetch_remote_file(
        "remote/f.json",
        local,
        remote_hasher=lambda remote: expected,
        downloader=lambda remote, dest: True,
    )
    assert not ok and "existing" in reason
    assert local.read_bytes() == b"old"  # preserved, never overwritten


def test_membership_flags_head_source_deployment_mismatch():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    bad_head = dict(head_metadata)
    bad_head["source"] = {"deployment_id": "OTHER-DEPLOYMENT"}
    issues = run_membership(mask, window_cap, extraction, bad_head, {"logreg_raw": variant, "xgb_raw": variant})
    assert any("head source deployment mismatch" in i for i in issues)


def test_membership_flags_missing_or_bad_weight_audit():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    no_audit = dict(variant)
    no_audit.pop("fit_weight_audit")
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": no_audit, "xgb_raw": variant})
    assert any("logreg_raw: fit weight audit missing" in i for i in issues)
    bad_mean = dict(variant)
    bad_mean["fit_weight_audit"] = {**variant["fit_weight_audit"], "mean_weight": 1.5}
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": bad_mean, "xgb_raw": variant})
    assert any("mean_weight is not finite 1.0" in i for i in issues)
    bad_rows = dict(variant)
    bad_rows["fit_weight_audit"] = {**variant["fit_weight_audit"], "row_count": 2}
    issues = run_membership(mask, window_cap, extraction, head_metadata, {"logreg_raw": bad_rows, "xgb_raw": variant})
    assert any("row_count is not the exact selected row count" in i for i in issues)


def test_weight_policy_canonical_mapping():
    assert cv.expected_weight_policy({"dataset": "daic"}) == "inverse_chunks_per_subject_rescaled_to_mean_one"
    assert (
        cv.expected_weight_policy({"dataset": "d3tec", "input_modality": "audio_text"})
        == "inverse_segments_per_response_rescaled_to_mean_one"
    )
    assert (
        cv.expected_weight_policy({"dataset": "d3tec", "input_modality": "text_only"})
        == "one_vector_per_subject_unweighted"
    )
    assert (
        cv.expected_weight_policy(
            {"dataset": "turkish", "dataset_variant": "pooled_t17", "input_modality": "text_only"}
        )
        == "one_vector_per_subject_unweighted"
    )
    assert (
        cv.expected_weight_policy({"dataset": "androids_interview", "input_modality": "audio_only"})
        == "uniform_rows"
    )


def _audit_variant(base: dict, **overrides):
    variant = {**base, "train_mask": dict(base["train_mask"])}
    variant["fit_weight_audit"] = {**base["fit_weight_audit"], **overrides}
    return variant


def test_weight_audit_invalid_cases_are_refused():
    import math as _math

    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    cases = [
        {"mean_weight": float("nan")},
        {"mean_weight": True},
        {"mean_weight": None},
        {"row_count": None},
        {"row_count": True},
        {"row_count": -1},
        {"row_count": 3.0},
        {"subject_count": None},
        {"subject_count": -1},
        {"policy": "legacy_uniform_rows"},
        {"policy": None},
    ]
    for overrides in cases:
        issues = run_membership(
            mask,
            window_cap,
            extraction,
            head_metadata,
            {"logreg_raw": _audit_variant(variant, **overrides), "xgb_raw": variant},
        )
        assert issues, f"expected an issue for {overrides}"


def _val_fixture():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    variant = {**variant, "train_mask": dict(variant["train_mask"])}
    variant["train_mask"]["retained_val_row_ids"] = ["v1"]
    variant["train_mask"]["retained_val_subject_count"] = 1
    variant["training_row_ids"] = ["a", "b", "c", "v1"]
    variant["fit_weight_audit"] = {**variant["fit_weight_audit"], "row_count": 4, "subject_count": 3}
    pool_rows = [
        {"subject_id": "s1", "sample_id": "a"},
        {"subject_id": "s1", "sample_id": "b"},
        {"subject_id": "s2", "sample_id": "c"},
        {"subject_id": "s9", "sample_id": "v1"},
    ]
    return mask, window_cap, extraction, head_metadata, variant, pool_rows


def test_membership_retains_full_inner_val_rows():
    mask, window_cap, extraction, head_metadata, variant, pool_rows = _val_fixture()
    issues = cv.membership_issues(
        parent_attempt_id="PARENT",
        head_attempt_id="HEAD",
        fold=0,
        parent_training_seed=7,
        mask=mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants={"logreg_raw": variant, "xgb_raw": variant},
        extraction_sha256="E",
        expected_head_deployment="DEPLOY",
        train_subjects={"s1", "s2"},
        val_subjects={"s9"},
        pool_rows=pool_rows,
        pool_rows_sha256="P",
    )
    assert issues == [], issues
    dropped = {**variant, "train_mask": dict(variant["train_mask"]), "training_row_ids": ["a", "b", "c"]}
    issues = cv.membership_issues(
        parent_attempt_id="PARENT",
        head_attempt_id="HEAD",
        fold=0,
        parent_training_seed=7,
        mask=mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants={"logreg_raw": dropped, "xgb_raw": variant},
        extraction_sha256="E",
        expected_head_deployment="DEPLOY",
        train_subjects={"s1", "s2"},
        val_subjects={"s9"},
        pool_rows=pool_rows,
        pool_rows_sha256="P",
    )
    assert any("capped+retained union" in i for i in issues)


def test_membership_enforces_split_and_pool_subject_sets():
    mask, window_cap, extraction, head_metadata, variant = membership_fixture()
    # missing training subject: mask covers only s1 while the split train is s1+s2
    short_mask = {**mask, "subjects": {"s1": list(mask["subjects"]["s1"])}}
    issues = cv.membership_issues(
        parent_attempt_id="PARENT",
        head_attempt_id="HEAD",
        fold=0,
        parent_training_seed=7,
        mask=short_mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants={"logreg_raw": variant, "xgb_raw": variant},
        extraction_sha256="E",
        expected_head_deployment="DEPLOY",
        train_subjects={"s1", "s2"},
        val_subjects=set(),
        pool_rows=[
            {"subject_id": "s1", "sample_id": "a"},
            {"subject_id": "s1", "sample_id": "b"},
            {"subject_id": "s2", "sample_id": "c"},
        ],
        pool_rows_sha256="P",
    )
    assert any("mask subjects do not equal the authoritative train subjects" in i for i in issues)
    # extra cache subject: the pool contains s9 which is not in train or val
    issues = cv.membership_issues(
        parent_attempt_id="PARENT",
        head_attempt_id="HEAD",
        fold=0,
        parent_training_seed=7,
        mask=mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants={"logreg_raw": variant, "xgb_raw": variant},
        extraction_sha256="E",
        expected_head_deployment="DEPLOY",
        train_subjects={"s1", "s2"},
        val_subjects=set(),
        pool_rows=[
            {"subject_id": "s1", "sample_id": "a"},
            {"subject_id": "s1", "sample_id": "b"},
            {"subject_id": "s2", "sample_id": "c"},
            {"subject_id": "s9", "sample_id": "v1"},
        ],
        pool_rows_sha256="P",
    )
    assert any("cached outer_train subjects do not equal the expected canonical pool" in i for i in issues)


def test_confounded_attempt_excluded_from_selection_and_progress(tmp_path):
    cv.configure(campaign_dir=tmp_path)
    key = "androids_interview_audio_only_native_cap25|7|0"
    entry = make_entry(key, attempt="CONF-ATTEMPT")
    expected = {key}
    (tmp_path / "head_confounded_attempts.jsonl").write_text(
        json.dumps(
            {
                "attempt_id": "CONF-ATTEMPT",
                "registry_key": key,
                "deployment_id": "old-deployment-full-id",
                "confounded": True,
                "mismatch_evidence": {"pool_policy": "train+inner-val", "old_semantics": "mask-only"},
            }
        )
        + "\n"
        + json.dumps({"attempt_id": "NO-EVIDENCE", "confounded": True})
        + "\n",
        encoding="utf-8",
    )
    ledger = {key: {"validated": True, "attempt_id": "CONF-ATTEMPT"}}
    ready, reasons = cv.select_ready_heads([entry], expected, {}, ledger)
    assert not ready and reasons.get("confounded old attempt (superseded)") == 1
    doc = cv.progress([], [entry], expected, ledger)
    assert doc["heads_validated"] == 0
    assert doc["heads_confounded_old_technical"] == 1
