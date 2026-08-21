from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.compare import (
    ComparisonError,
    compare_group,
    load_attempt_evidence,
)


METRIC_VALUE = 0.625


def _make_attempt(root: Path, attempt_id: str, group_id: str, state="REPORTABLE",
                  view="harmonized_all_windows_full_coverage", namespace="headline/binary_strict",
                  metric_value=METRIC_VALUE, seed=1337):
    campaign, modality, dataset, run_name = "camp", "audio_text", "daic", f"run_{attempt_id[-16:]}"
    rel = f"output_model/{campaign}/{modality}/{dataset}/{run_name}/fold_0"
    fold = root / rel
    fold.mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "exp_submit" / attempt_id).mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "exp_submit" / attempt_id / "contract.json").write_text(json.dumps({
        "attempt_id": attempt_id,
        "group_id": group_id,
        "experiment_id": group_id,
        "fold": 0,
        "seed": seed,
        "run_name": run_name,
        "local_fold_rel": rel,
    }), encoding="utf-8")
    (fold / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (fold / "jobs.jsonl").write_text("{}\n", encoding="utf-8")
    (fold / "evaluations.json").write_text(json.dumps({"evaluations": [
        {"evaluation_id": "eval-" + attempt_id[-24:].rjust(24, "0")[:24].replace("-", "0"),
         "dataset": dataset, "split_name": "test", "split_protocol": "train_val",
         "checkpoint_role": "best_model", "checkpoint_path": "best_model",
         "backend": "original_teacher_forced", "evaluation_view": view,
         "aggregation": "subject_level", "metric_namespace": namespace,
         "metrics": [{"name": "macro_f1", "value": metric_value}]},
    ]}), encoding="utf-8")


KW = dict(
    dataset="daic", metric="macro_f1", namespace="headline/binary_strict",
    backend="original_teacher_forced", view="harmonized_all_windows_full_coverage",
    aggregation="subject_level",
)


def test_compare_selects_unambiguous_winner(tmp_path):
    _make_attempt(tmp_path, "20260821T000000Z-aaa-11111111-aaaaaaaa", "grp", metric_value=0.70)
    _make_attempt(tmp_path, "20260821T000000Z-bbb-22222222-bbbbbbbb", "grp", metric_value=0.55)
    audit = compare_group(tmp_path, group_id="grp",
                          attempt_ids=["20260821T000000Z-aaa-11111111-aaaaaaaa",
                                       "20260821T000000Z-bbb-22222222-bbbbbbbb"],
                          tie_rule="max", **KW)
    assert audit["unambiguous_winner"] == "20260821T000000Z-aaa-11111111-aaaaaaaa"
    assert audit["automatic_selection_authorized"] is True
    # deterministic: identical inputs produce byte-identical audit
    audit2 = compare_group(tmp_path, group_id="grp",
                           attempt_ids=["20260821T000000Z-bbb-22222222-bbbbbbbb",
                                        "20260821T000000Z-aaa-11111111-aaaaaaaa"],
                           tie_rule="max", **KW)
    assert json.dumps(audit, sort_keys=True) == json.dumps(audit2, sort_keys=True)


def test_compare_requires_reportable_attempts(tmp_path):
    _make_attempt(tmp_path, "20260821T000000Z-aaa-11111111-aaaaaaaa", "grp")
    _make_attempt(tmp_path, "20260821T000000Z-bbb-22222222-bbbbbbbb", "grp", state="LOCALLY_VALIDATED")
    with pytest.raises(ComparisonError, match="not REPORTABLE"):
        compare_group(tmp_path, group_id="grp",
                      attempt_ids=["20260821T000000Z-aaa-11111111-aaaaaaaa",
                                   "20260821T000000Z-bbb-22222222-bbbbbbbb"],
                      tie_rule="max", **KW)


def test_compare_refuses_outside_group(tmp_path):
    _make_attempt(tmp_path, "20260821T000000Z-aaa-11111111-aaaaaaaa", "grpA")
    _make_attempt(tmp_path, "20260821T000000Z-bbb-22222222-bbbbbbbb", "grpB")
    with pytest.raises(ComparisonError, match="belongs to group"):
        compare_group(tmp_path, group_id="grpA",
                      attempt_ids=["20260821T000000Z-aaa-11111111-aaaaaaaa",
                                   "20260821T000000Z-bbb-22222222-bbbbbbbb"],
                      tie_rule="max", **KW)


def test_compare_refuses_mixed_views_and_missing_tie_rule(tmp_path):
    _make_attempt(tmp_path, "20260821T000000Z-aaa-11111111-aaaaaaaa", "grp")
    _make_attempt(tmp_path, "20260821T000000Z-bbb-22222222-bbbbbbbb", "grp",
                  view="fixed_k4_full_coverage")
    with pytest.raises(ComparisonError, match="qualifiers"):
        compare_group(tmp_path, group_id="grp",
                      attempt_ids=["20260821T000000Z-aaa-11111111-aaaaaaaa",
                                   "20260821T000000Z-bbb-22222222-bbbbbbbb"],
                      tie_rule="max", **KW)
    _make_attempt(tmp_path, "20260821T000000Z-ccc-33333333-cccccccc", "grp2")
    _make_attempt(tmp_path, "20260821T000000Z-ddd-44444444-dddddddd", "grp2")
    with pytest.raises(ComparisonError, match="tie_rule"):
        compare_group(tmp_path, group_id="grp2",
                      attempt_ids=["20260821T000000Z-ccc-33333333-cccccccc",
                                   "20260821T000000Z-ddd-44444444-dddddddd"],
                      tie_rule="winner_highest", **KW)


def test_compare_tie_is_ambiguous_without_selection(tmp_path):
    _make_attempt(tmp_path, "20260821T000000Z-aaa-11111111-aaaaaaaa", "grp", metric_value=0.7)
    _make_attempt(tmp_path, "20260821T000000Z-bbb-22222222-bbbbbbbb", "grp", metric_value=0.7)
    audit = compare_group(tmp_path, group_id="grp",
                          attempt_ids=["20260821T000000Z-aaa-11111111-aaaaaaaa",
                                       "20260821T000000Z-bbb-22222222-bbbbbbbb"],
                          tie_rule="max", **KW)
    assert audit["unambiguous_winner"] is None
    assert audit["automatic_selection_authorized"] is False
    assert len(audit["tied_winners"]) == 2


def test_load_evidence_requires_contract(tmp_path):
    with pytest.raises(ComparisonError, match="no recorded submission contract"):
        load_attempt_evidence(tmp_path, "missing-attempt")
