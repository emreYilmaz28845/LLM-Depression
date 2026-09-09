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


def _merged_record(attempt_id: str) -> dict:
    return {
        "attempt_id": attempt_id,
        "logical_run_name": "run",
        "fold": 0,
        "seed": 7,
        "config_path": f"{attempt_id}/run_config.yaml",
        "config_sha256": "c",
        "manifest_sha256": "m",
        "split_sha256": "s",
        "checkpoint_path": "ckpt",
        "jobs": {"slurm_job_ids": [1], "failures": []},
        "source": {"git_commit": "x"},
        "evaluations": [
            {
                "evaluation_id": f"eval-{dataset}-{attempt_id}",
                "dataset": dataset,
                "metrics_path": f"{attempt_id}/metrics.json",
                "metrics_sha256": f"h-{dataset}",
                "prediction_path": f"{attempt_id}/predictions.jsonl",
                "prediction_sha256": f"p-{dataset}",
            }
            for dataset in ("androids_interview", "cmdc", "d3tec", "daic", "turkish")
        ],
        "backend": "tf",
        "split_seed": 1337,
        "head_seed": 1337,
    }


def test_record_provenance_can_filter_to_one_merged_dataset() -> None:
    provenance = report._record_provenance(_merged_record("att-1"))

    assert len(provenance["evaluation_ids"]) == 5
    assert len(provenance["metrics_artifacts"]) == 5

    daic = report._record_provenance(_merged_record("att-1"), "daic")

    assert daic["evaluation_ids"] == ["eval-daic-att-1"]
    assert len(daic["metrics_artifacts"]) == 1
    assert daic["metrics_artifacts"][0]["sha256"] == "h-daic"

    with pytest.raises(report.ReportError, match="exactly one nosuch evaluation"):
        report._record_provenance(_merged_record("att-1"), "nosuch")


def test_merged_cv_per_dataset_cell_means_that_datasets_fold_scores(monkeypatch) -> None:
    records = [{"seed": seed, "fold": fold} for seed in report.TRAINING_SEEDS for fold in range(5)]

    def fake_fold_metrics(record, dataset=None):
        base = record["fold"] / 4
        extra = 0.1 if dataset == "daic" else 0.0
        return {"macro_f1": base + extra, "positive_f1": base / 2}

    seen: list[str | None] = []

    def fake_provenance(record, dataset=None):
        seen.append(dataset)
        return {"dataset": dataset}

    monkeypatch.setattr(report, "_fold_metrics", fake_fold_metrics)
    monkeypatch.setattr(report, "_record_provenance", fake_provenance)

    cell = report._aggregate_cell(
        records, endpoint="merged_cv", condition="native",
        backbone="qwen", method="logreg", dataset="daic", per_dataset=True,
    )

    assert cell["aggregation"] == report.MERGED_CV_PER_DATASET_AGGREGATION
    assert cell["dataset"] == "daic"
    for row in cell["seed_rows"]:
        assert row["macro_f1"] == pytest.approx(0.6)
        assert row["positive_f1"] == pytest.approx(0.25)
    # One provenance entry per fold, each restricted to the daic evaluation.
    assert seen == ["daic"] * 15
    assert all(entry["dataset"] == "daic" for row in cell["seed_rows"] for entry in row["provenance"])

    rollup = report._aggregate_cell(
        records, endpoint="merged_cv", condition="native",
        backbone="qwen", method="logreg", dataset="merged",
    )

    assert rollup["aggregation"] == report.MERGED_CV_AGGREGATION
    for row in rollup["seed_rows"]:
        assert row["macro_f1"] == pytest.approx(0.5)
    assert seen == ["daic"] * 15 + [None] * 15
