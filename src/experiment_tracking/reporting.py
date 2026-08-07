from __future__ import annotations

import json
import math
import statistics
from typing import Any, Iterable, Sequence

from .constants import SCHEMA_VERSION_REPORT

MN5_ONLY_MARKER = "MN5-only, not locally verifiable"

_FAILED_JOB_STATUSES = (
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
)


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None})


def _evaluation_mn5_only(artifact_flags: dict[str, dict[str, Any]], evaluation: dict[str, Any]) -> bool:
    for key in ("metrics_artifact_id", "predictions_artifact_id"):
        artifact_id = evaluation.get(key)
        if artifact_id and artifact_flags.get(artifact_id, {}).get("exists_locally") != 1:
            return True
    return False


def build_run_report(
    connection: Any,
    attempt_id: str,
    fold: int | None = None,
    *,
    generated_at_utc: str | None = None,
    conclusion: str | None = None,
) -> dict[str, Any]:
    payload = registry_show_attempt(connection, attempt_id, fold)
    attempt = payload["attempt"]
    logical = payload["logical_run"] or {}
    evaluations: list[dict[str, Any]] = []
    mn5_only: list[str] = []
    warnings: list[str] = []
    locally_verified = True
    artifact_flags: dict[str, dict[str, Any]] = {}
    for artifact in payload["artifacts"]:
        artifact_flags[artifact["artifact_id"]] = artifact
    for entry in sorted(payload["evaluations"], key=lambda item: item["evaluation"]["evaluation_id"]):
        evaluation = entry["evaluation"]
        evidence = artifact_flags.get(evaluation.get("metrics_artifact_id")) or {}
        mn5 = _evaluation_mn5_only(artifact_flags, evaluation)
        if mn5:
            mn5_only.append(evaluation["evaluation_id"])
        locally_verified = locally_verified and (evaluation.get("locally_verified") == 1)
        if evaluation.get("warnings_json"):
            try:
                warnings.extend(json.loads(evaluation["warnings_json"]))
            except (ValueError, TypeError):
                warnings.append(evaluation["warnings_json"])
        evaluations.append(
            {
                "evaluation_id": evaluation["evaluation_id"],
                "dataset": evaluation["dataset"],
                "split_name": evaluation["split_name"],
                "split_protocol": evaluation["split_protocol"],
                "checkpoint_role": evaluation["checkpoint_role"],
                "checkpoint_path": evaluation["checkpoint_path"],
                "backend": evaluation["backend"],
                "evaluation_view": evaluation["evaluation_view"],
                "aggregation": evaluation["aggregation"],
                "metric_namespace": evaluation["metric_namespace"],
                "metrics_artifact_path": evidence.get("path"),
                "predictions_artifact_path": (
                    artifact_flags.get(evaluation.get("predictions_artifact_id")) or {}
                ).get("path"),
                "locally_verified": evaluation.get("locally_verified") == 1,
                "reportable": evaluation.get("reportable") == 1,
                "metrics": sorted(entry["metrics"], key=lambda metric: metric["metric_name"]),
            }
        )
    jobs = sorted(payload["jobs"], key=lambda job: (job["at_utc"], job["event_id"]))
    failed_jobs = [
        job for job in jobs if job.get("status") in _FAILED_JOB_STATUSES
    ]
    resubmitted_jobs = [job for job in jobs if job.get("resubmission_of_job_id")]
    checkpoints = _sorted_unique(
        evaluation["checkpoint_role"] + ":" + (evaluation["checkpoint_path"] or "")
        for evaluation in evaluations
    )
    return {
        "schema_version": SCHEMA_VERSION_REPORT,
        "group_id": None,
        "logical_run_name": logical.get("logical_run_name"),
        "attempt_id": attempt_id,
        "fold": fold,
        "seed": logical.get("seed"),
        "dataset": logical.get("dataset"),
        "modality": logical.get("modality"),
        "status": attempt.get("current_state"),
        "locally_verified": locally_verified,
        "git": {
            "git_commit": attempt.get("git_commit"),
            "git_branch": attempt.get("git_branch"),
            "git_dirty": attempt.get("git_dirty"),
            "github_issue": attempt.get("github_issue"),
            "github_pr": attempt.get("github_pr"),
        },
        "hashes": {
            "resolved_config_sha256": attempt.get("resolved_config_sha256"),
            "manifest_sha256": attempt.get("manifest_sha256"),
            "split_sha256": attempt.get("split_sha256"),
            "deployed_source_sha256": attempt.get("deployed_source_sha256"),
        },
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "jobs": jobs,
        "failed_jobs": failed_jobs,
        "resubmitted_jobs": resubmitted_jobs,
        "wandb": {"run_id": None, "url": None, "sync_status": None},
        "warnings": _sorted_unique(warnings),
        "mn5_only": mn5_only,
        "conclusion": conclusion,
        "generated_at_utc": generated_at_utc,
    }


def registry_show_attempt(connection: Any, attempt_id: str, fold: int | None) -> dict[str, Any]:
    from .registry import show_attempt

    return show_attempt(connection, attempt_id, fold)


def render_run_report_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Run report: {payload['logical_run_name']}")
    lines.append("")
    lines.append(f"- attempt_id: `{payload['attempt_id']}`")
    lines.append(f"- fold: {payload['fold']}")
    lines.append(f"- dataset: {payload['dataset']}")
    lines.append(f"- modality: {payload['modality']}")
    lines.append(f"- seed: {payload['seed']}")
    lines.append(f"- status: {payload['status']}")
    lines.append(f"- locally_verified: {payload['locally_verified']}")
    lines.append("")
    lines.append("## Git")
    for key, value in sorted(payload["git"].items()):
        lines.append(f"- {key}: {_render_value(value)}")
    lines.append("")
    lines.append("## Hashes")
    for key, value in sorted(payload["hashes"].items()):
        lines.append(f"- {key}: {_render_value(value)}")
    lines.append("")
    lines.append("## Checkpoints")
    for checkpoint in payload["checkpoints"]:
        lines.append(f"- {checkpoint}")
    lines.append("")
    lines.append("## Evaluations")
    for evaluation in payload["evaluations"]:
        lines.append(f"### {evaluation['evaluation_id']}")
        lines.append(f"- dataset: {_render_value(evaluation['dataset'])}")
        lines.append(f"- split: {_render_value(evaluation['split_name'])} ({_render_value(evaluation['split_protocol'])})")
        lines.append(f"- checkpoint: {_render_value(evaluation['checkpoint_role'])} at `{_render_value(evaluation['checkpoint_path'])}`")
        lines.append(f"- backend: {_render_value(evaluation['backend'])}")
        lines.append(f"- evaluation_view: {_render_value(evaluation['evaluation_view'])}")
        lines.append(f"- aggregation: {_render_value(evaluation['aggregation'])}")
        lines.append(f"- metric_namespace: {_render_value(evaluation['metric_namespace'])}")
        lines.append(f"- metrics_artifact_path: `{_render_value(evaluation['metrics_artifact_path'])}`")
        lines.append(f"- predictions_artifact_path: `{_render_value(evaluation['predictions_artifact_path'])}`")
        lines.append(f"- locally_verified: {evaluation['locally_verified']}")
        lines.append(f"- reportable: {evaluation['reportable']}")
        if evaluation["evaluation_id"] in payload["mn5_only"]:
            lines.append(f"- evidence: **{MN5_ONLY_MARKER}**")
        for metric in evaluation["metrics"]:
            lines.append(
                f"- metric `{metric['metric_name']}` = {_render_value(metric['metric_value'])}"
                f" (support {_render_value(metric['support'])})"
            )
        lines.append("")
    if payload["failed_jobs"]:
        lines.append("## Failed or cancelled jobs")
        for job in payload["failed_jobs"]:
            lines.append(
                f"- {job['job_key']} job {_render_value(job['slurm_job_id'])}: "
                f"{job['status']} ({_render_value(job['reason'])}) at {job['at_utc']}"
            )
        lines.append("")
    if payload["resubmitted_jobs"]:
        lines.append("## Resubmitted jobs")
        for job in payload["resubmitted_jobs"]:
            lines.append(
                f"- {job['job_key']} job {_render_value(job['slurm_job_id'])} "
                f"resubmission of {_render_value(job['resubmission_of_job_id'])}"
            )
        lines.append("")
    if payload["warnings"]:
        lines.append("## Warnings")
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    if payload["mn5_only"]:
        lines.append(f"## {MN5_ONLY_MARKER}")
        for evaluation_id in payload["mn5_only"]:
            lines.append(f"- {evaluation_id}")
        lines.append("")
    lines.append("## Researcher interpretation")
    lines.append(_render_value(payload["conclusion"]))
    if payload["generated_at_utc"] is not None:
        lines.append("")
        lines.append(f"_generated_at: {payload['generated_at_utc']}_")
    return "\n".join(lines) + "\n"


def _render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def _candidate_rows(
    connection: Any,
    attempt_ids: Sequence[str],
    qualifiers: dict[str, str],
    metric_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for attempt_id in attempt_ids:
        payload = registry_show_attempt(connection, attempt_id, None)
        for entry in payload["evaluations"]:
            evaluation = entry["evaluation"]
            if any(evaluation.get(key) != value for key, value in qualifiers.items()):
                continue
            evidence = {}
            for artifact in payload["artifacts"]:
                evidence[artifact["artifact_id"]] = artifact
            for metric in entry["metrics"]:
                if metric_name is not None and metric["metric_name"] != metric_name:
                    continue
                rows.append(
                    {
                        "attempt_id": attempt_id,
                        "logical_run_name": payload["logical_run"]["logical_run_name"],
                        "seed": payload["logical_run"].get("seed"),
                        "fold": payload["folds"][0]["fold"] if payload["folds"] else None,
                        "evaluation_id": evaluation["evaluation_id"],
                        "metric_name": metric["metric_name"],
                        "metric_value": metric["metric_value"],
                        "support": metric["support"],
                        "dataset": evaluation["dataset"],
                        "split_name": evaluation["split_name"],
                        "split_protocol": evaluation["split_protocol"],
                        "backend": evaluation["backend"],
                        "evaluation_view": evaluation["evaluation_view"],
                        "aggregation": evaluation["aggregation"],
                        "metric_namespace": evaluation["metric_namespace"],
                        "checkpoint_role": evaluation["checkpoint_role"],
                        "evidence_path": (evidence.get(evaluation.get("metrics_artifact_id")) or {}).get("path"),
                    }
                )
    return rows


def compatibility_issues(candidates: Sequence[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    dimensions = {
        "dataset": "dataset",
        "split_protocol": "split protocol",
        "backend": "prediction backend",
        "evaluation_view": "evaluation view",
        "aggregation": "aggregation unit",
        "metric_namespace": "metric namespace",
        "checkpoint_role": "checkpoint role",
        "split_name": "split name",
    }
    for key, label in dimensions.items():
        values = _sorted_unique(candidate[key] for candidate in candidates)
        if len(values) > 1:
            issues.append(f"incompatible {label}: {values}")
    return issues


def build_group_report(
    connection: Any,
    attempt_ids: Sequence[str],
    *,
    metric_name: str,
    namespace: str,
    backend: str,
    view: str,
    aggregation: str,
    compare_a: Sequence[str] | None = None,
    compare_b: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_folds: Sequence[int] | None = None,
    research_question: str | None = None,
    hypothesis: str | None = None,
    baseline: str | None = None,
    treatment: str | None = None,
    generated_at_utc: str | None = None,
    conclusion: str | None = None,
) -> dict[str, Any]:
    qualifiers = {
        "metric_namespace": namespace,
        "backend": backend,
        "evaluation_view": view,
        "aggregation": aggregation,
    }
    candidates = _candidate_rows(connection, attempt_ids, qualifiers, metric_name=metric_name)
    base_qualifiers = {
        "metric_namespace": namespace,
        "backend": backend,
    }
    pooled_rows = _candidate_rows(
        connection,
        attempt_ids,
        {**base_qualifiers, "aggregation": "pooled_subject_level"},
        metric_name=metric_name,
    )
    fold_mean_rows = _candidate_rows(
        connection,
        attempt_ids,
        {**base_qualifiers, "aggregation": "fold_mean"},
        metric_name=metric_name,
    )
    issues = compatibility_issues(candidates)
    jobs_by_attempt: dict[str, list[dict[str, Any]]] = {}
    for attempt_id in attempt_ids:
        payload = registry_show_attempt(connection, attempt_id, None)
        jobs_by_attempt[attempt_id] = sorted(
            payload["jobs"], key=lambda job: (job["at_utc"], job["event_id"])
        )
    failed_jobs = [
        f"{job['slurm_job_id']}:{job['status']}"
        for attempt_id in attempt_ids
        for job in jobs_by_attempt[attempt_id]
        if job.get("status") in _FAILED_JOB_STATUSES
    ]
    resubmitted_jobs = [
        f"{job['slurm_job_id']}->{job['resubmission_of_job_id']}"
        for attempt_id in attempt_ids
        for job in jobs_by_attempt[attempt_id]
        if job.get("resubmission_of_job_id")
    ]
    per_fold: dict[Any, list[dict[str, Any]]] = {}
    per_seed: dict[Any, list[dict[str, Any]]] = {}
    for candidate in candidates:
        per_fold.setdefault(candidate["fold"], []).append(candidate)
        per_seed.setdefault(candidate["seed"], []).append(candidate)
    values = [candidate["metric_value"] for candidate in candidates if isinstance(candidate["metric_value"], (int, float))]
    completed_pairs = sorted({(candidate["seed"], candidate["fold"]) for candidate in candidates})
    expected_pairs = sorted(
        (seed, fold) for seed in (expected_seeds or []) for fold in (expected_folds or [])
    )
    paired_deltas: list[dict[str, Any]] | None = None
    if compare_a and compare_b:
        a_by_pair = {
            (c["seed"], c["fold"]): c["metric_value"]
            for c in _candidate_rows(connection, compare_a, qualifiers, metric_name=metric_name)
        }
        b_by_pair = {
            (c["seed"], c["fold"]): c["metric_value"]
            for c in _candidate_rows(connection, compare_b, qualifiers, metric_name=metric_name)
        }
        common = sorted(set(a_by_pair) & set(b_by_pair))
        paired_deltas = [
            {
                "seed": pair[0],
                "fold": pair[1],
                "baseline_value": a_by_pair[pair],
                "treatment_value": b_by_pair[pair],
                "delta": b_by_pair[pair] - a_by_pair[pair],
            }
            for pair in common
        ]
    return {
        "schema_version": SCHEMA_VERSION_REPORT,
        "group_id": None,
        "research_question": research_question,
        "hypothesis": hypothesis,
        "baseline": baseline,
        "treatment": treatment,
        "metric_qualifiers": qualifiers,
        "expected_matrix": {"seeds": sorted(expected_seeds or []), "folds": sorted(expected_folds or [])},
        "completed_matrix": completed_pairs,
        "missing_matrix": sorted(set(expected_pairs) - set(completed_pairs)),
        "included_attempt_ids": sorted(attempt_ids),
        "excluded_attempt_ids": [],
        "failed_jobs": sorted(set(failed_jobs)),
        "resubmitted_jobs": sorted(set(resubmitted_jobs)),
        "per_fold": {str(key): sorted(values, key=lambda c: c["attempt_id"]) for key, values in sorted(per_fold.items())},
        "per_seed": {str(key): sorted(values, key=lambda c: c["attempt_id"]) for key, values in sorted(per_seed.items())},
        "aggregate": {
            **_mean_std(values),
            "support_total": sum(c["support"] for c in candidates if c["support"] is not None),
            "pooled": pooled_rows,
            "fold_mean": fold_mean_rows,
        },
        "paired_deltas": paired_deltas,
        "compatibility": {"ok": not issues, "issues": issues},
        "provenance": sorted(
            (
                {
                    "attempt_id": candidate["attempt_id"],
                    "evaluation_id": candidate["evaluation_id"],
                    "metric_name": candidate["metric_name"],
                    "evidence_path": candidate["evidence_path"],
                }
                for candidate in candidates
            ),
            key=lambda item: (item["attempt_id"], item["evaluation_id"], item["metric_name"]),
        ),
        "conclusion": conclusion,
        "generated_at_utc": generated_at_utc,
    }


def render_group_report_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Experiment group report")
    lines.append("")
    lines.append(f"- research_question: {_render_value(payload['research_question'])}")
    lines.append(f"- hypothesis: {_render_value(payload['hypothesis'])}")
    lines.append(f"- baseline: {_render_value(payload['baseline'])}")
    lines.append(f"- treatment: {_render_value(payload['treatment'])}")
    lines.append("")
    lines.append("## Metric qualifiers")
    for key, value in sorted(payload["metric_qualifiers"].items()):
        lines.append(f"- {key}: {_render_value(value)}")
    lines.append("")
    lines.append("## Matrix")
    lines.append(f"- expected: {payload['expected_matrix']}")
    lines.append(f"- completed: {payload['completed_matrix']}")
    lines.append(f"- missing: {payload['missing_matrix']}")
    lines.append(f"- included attempts: {len(payload['included_attempt_ids'])}")
    lines.append(f"- excluded attempts: {len(payload['excluded_attempt_ids'])}")
    lines.append("")
    if payload["failed_jobs"]:
        lines.append("## Failed jobs")
        for job in payload["failed_jobs"]:
            lines.append(f"- {job}")
        lines.append("")
    if payload["resubmitted_jobs"]:
        lines.append("## Resubmitted jobs")
        for job in payload["resubmitted_jobs"]:
            lines.append(f"- {job}")
        lines.append("")
    lines.append("## Aggregation")
    aggregate = payload["aggregate"]
    lines.append(f"- mean: {_render_value(aggregate['mean'])}")
    lines.append(f"- std: {_render_value(aggregate['std'])}")
    lines.append(f"- n: {aggregate['n']}")
    lines.append(f"- support_total: {aggregate['support_total']}")
    for candidate in aggregate["pooled"]:
        lines.append(
            f"- pooled ({candidate['evaluation_id']}): {_render_value(candidate['metric_value'])} "
            f"from `{_render_value(candidate['evidence_path'])}`"
        )
    for candidate in aggregate["fold_mean"]:
        lines.append(
            f"- fold_mean ({candidate['evaluation_id']}): {_render_value(candidate['metric_value'])} "
            f"from `{_render_value(candidate['evidence_path'])}`"
        )
    lines.append("")
    if payload["paired_deltas"] is not None:
        lines.append("## Paired deltas (treatment - baseline)")
        for delta in payload["paired_deltas"]:
            lines.append(
                f"- seed {delta['seed']} fold {delta['fold']}: "
                f"{_render_value(delta['baseline_value'])} -> {_render_value(delta['treatment_value'])} "
                f"(delta {_render_value(delta['delta'])})"
            )
        lines.append("")
    lines.append("## Compatibility gate")
    if payload["compatibility"]["ok"]:
        lines.append("- compatible")
    else:
        for issue in payload["compatibility"]["issues"]:
            lines.append(f"- **{issue}**")
    lines.append("")
    lines.append("## Provenance")
    for item in payload["provenance"]:
        lines.append(
            f"- `{item['attempt_id']}` / `{item['evaluation_id']}` / {item['metric_name']} "
            f"-> `{_render_value(item['evidence_path'])}`"
        )
    lines.append("")
    lines.append("## Researcher interpretation")
    lines.append(_render_value(payload["conclusion"]))
    if payload["generated_at_utc"] is not None:
        lines.append("")
        lines.append(f"_generated_at: {payload['generated_at_utc']}_")
    return "\n".join(lines) + "\n"
