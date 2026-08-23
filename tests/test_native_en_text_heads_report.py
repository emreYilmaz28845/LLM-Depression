import json

import pytest

from tools import native_en_text_heads_report as report


def _cell(condition: str, values: list[tuple[float, float]]) -> dict:
    return {
        "endpoint": "standalone",
        "dataset": "d3tec",
        "backbone": "qwen",
        "head": "logreg",
        "condition": condition,
        "aggregation": "pooled subject-level across five outer folds",
        "seed_count": 3,
        "provenance_key": f"key|{condition}",
        "provenance_status": "reportable_local_evidence",
        "seed_rows": [
            {
                "seed": seed,
                "native_or_english": condition,
                "macro_f1": macro,
                "positive_f1": positive,
                "provenance": [],
            }
            for seed, (macro, positive) in zip((7, 1337, 2024), values)
        ],
    }


def test_summary_pair_uses_three_seed_sample_sd_and_paired_deltas() -> None:
    native = _cell("native", [(0.4, 0.3), (0.5, 0.4), (0.6, 0.5)])
    english = _cell("english", [(0.5, 0.35), (0.55, 0.5), (0.8, 0.65)])

    summary = report._summary_pair(native, english)

    assert summary["seed_count"] == 3
    assert summary["native_macro_mean"] == pytest.approx(0.5)
    assert summary["english_macro_mean"] == pytest.approx(0.6166666667)
    assert summary["delta_macro_mean"] == pytest.approx(0.1166666667)
    assert summary["native_macro_sd"] == pytest.approx(0.1)
    assert len(summary["seed_details"]) == 3
    assert summary["seed_details"][0]["delta_positive_f1"] == pytest.approx(0.05)


def test_matrix_key_keeps_standalone_datasets_distinct() -> None:
    job = {
        "endpoint": "standalone",
        "condition": "native",
        "backbone": "qwen",
        "method": "logreg",
        "seed": 7,
        "fold": 0,
    }
    d3tec = {"job": job, "evaluations": [{"dataset": "d3tec"}]}
    androids = {"job": job, "evaluations": [{"dataset": "androids_interview"}]}

    assert report._matrix_key(d3tec) != report._matrix_key(androids)
    assert report._record_has_dataset(d3tec, "standalone", "d3tec")
    assert not report._record_has_dataset(d3tec, "standalone", "cmdc")


def test_report_rejects_incomplete_plan(tmp_path) -> None:
    plan = {
        "schema_version": "native_en_text_heads_v2_submission_plan.v1",
        "group_id": report.GROUP_ID,
        "source_commit": "a" * 40,
        "deployment_id": "dep-test",
        "jobs": [
            {
                "attempt_id": "attempt-one",
                "method": "logreg",
                "endpoint": "standalone",
                "condition": "native",
                "backbone": "qwen",
                "dataset": "d3tec",
                "seed": 7,
                "fold": 0,
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(report.ReportError, match="submission contract is missing"):
        report.build_report(path)


def test_source_provenance_accepts_explicit_retry_source_policy() -> None:
    plan = {
        "source_commit": "new" * 10,
        "deployment_id": "dep-final",
        "evidence_default_source": {
            "git_commit": "old" * 10,
            "deployment_id": "dep-old",
            "source_manifest_sha256": "old-manifest",
        },
        "evidence_source_overrides": {
            "retry-attempt": {
                "git_commit": "retry" * 8,
                "deployment_id": "dep-retry",
                "source_manifest_sha256": "retry-manifest",
                "reason": "bounded replacement",
            }
        },
    }
    metadata = {
        "attempt_id": "retry-attempt",
        "source": {
            "git_commit": "retry" * 8,
            "deployment_id": "dep-retry",
            "deployed_source_sha256": "retry-manifest",
            "git_branch": "agent/test",
        },
    }
    result = report._source_provenance(
        metadata,
        plan,
        {"attempt_id": "retry-attempt"},
    )

    assert result["git_commit"] == "retry" * 8
    assert result["deployment_id"] == "dep-retry"
    assert result["evidence_source_policy"]["reason"] == "bounded replacement"


def test_source_provenance_rejects_unlisted_source_mismatch() -> None:
    plan = {
        "source_commit": "new" * 10,
        "deployment_id": "dep-final",
    }
    metadata = {
        "attempt_id": "unlisted-attempt",
        "source": {"git_commit": "old" * 10},
    }

    with pytest.raises(report.ReportError, match="source git_commit mismatch"):
        report._source_provenance(
            metadata,
            plan,
            {"attempt_id": "unlisted-attempt"},
        )
