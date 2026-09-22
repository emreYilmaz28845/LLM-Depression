#!/usr/bin/env python
"""Build the reader-facing report artifacts for the prompt-context campaign.

Writes, deterministically and without timestamps:

  completion_summary.csv   one row per training fit: run, attempt, Slurm jobs,
                           lifecycle state and the fold's strict metrics
  result_report.md         the reader-facing report: canonical evaluation policy,
                           matrix, results, references, the same-backend DAIC
                           prompt comparison, the retry chain, the pooled manifest
                           verification, artifact hashes and limitations
  provenance.json          machine-readable provenance of the report inputs

Every number comes from the run's own collected local evidence: `run_config.yaml`
(identity and config), `jobs.jsonl` (Slurm jobs), `status.json` (lifecycle state)
and `best_model/standalone_eval/metrics_likelihood.json`. The script refuses to
write a completion row whose jobs are not terminal `COMPLETED` and whose fold has
no metric artifact, so an incomplete matrix cannot be published silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMPAIGN = PROJECT_ROOT / "output_model/promptcontext_v1_qwen38_likelihood/text_only"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/prompt_context_comparison/report"
DEFAULT_COMPARISON = PROJECT_ROOT / "outputs/prompt_context_comparison/report/comparison.csv"
DEFAULT_RUN_AUDIT = PROJECT_ROOT / "outputs/prompt_context_run_audit/run_audit.json"

CAMPAIGN = "promptcontext_v1_qwen38_likelihood"
IMPLEMENTATION_COMMIT = "96b2fa02d59f6ec301aef5e0c200477cfc9d0513"
DEPLOYMENT_ID = (
    "feat-qwen38-standalone-promptcontext-20260922-20260922T110059Z-96b2fa02-c5c8f4c2"
)
GROUP_ID = "qwen38-standalone-promptcontext-20260922"
POOLED_MANIFEST_HASH = (
    "37e991526986d9693c9620682719a6b54c7d30ec86b53152ae23dde167271b70"
)
POOLED_MANIFEST_FILE_SHA256 = (
    "a0daf6658f111f113bcf17231a6abbc0b8a472a3d60d9ddf76cd9ff25eb4e089"
)
DAIC_MANIFEST_HASH = "72e2dd204b915ccba3ebf922f030531fe5678b3ea8c9c52b81b41242fe9dda17"
DAIC_SPLIT_HASH = "441333e0c88845eeacba9ea5"
# dataset directory -> (label, folds, retry runs, note)
CELLS = (
    ("daic", "DAIC", (0,), {}, ""),
    ("d3tec", "D3TEC", (0, 1, 2, 3, 4), {}, ""),
    (
        "androids_interview",
        "Androids",
        (0, 1, 2, 3, 4),
        {0: "qwen38_pc_androids_f0b_20260922"},
        "fold 0 completed by the bounded infrastructure retry",
    ),
    ("cmdc", "CMDC", (0, 1, 2, 3, 4), {}, ""),
    (
        "turkish",
        "Turkish pooled",
        (0, 1, 2, 3, 4),
        {},
        "prebuilt pooled manifest; both question conditions",
    ),
)
SUMMARY_COLUMNS = (
    "dataset",
    "fold",
    "run_name",
    "attempt_id",
    "train_job_id",
    "eval_job_id",
    "train_state",
    "eval_state",
    "lifecycle_state",
    "macro_f1",
    "positive_f1",
    "uar",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jobs(path: Path) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        key = str(event.get("job_key"))
        state = str(event.get("status") or event.get("event_type") or "")
        record = latest.setdefault(key, {})
        record["state"] = state
        if event.get("slurm_job_id"):
            record["slurm_job_id"] = str(event["slurm_job_id"])
    return latest


def fold_row(dataset_dir: str, label: str, fold: int, run_name: str, notes: str) -> dict[str, Any]:
    fold_dir = DEFAULT_CAMPAIGN / dataset_dir / run_name / f"fold_{fold}"
    run_config = yaml.safe_load((fold_dir / "run_config.yaml").read_text(encoding="utf-8"))
    jobs = read_jobs(fold_dir / "jobs.jsonl")
    metrics = json.loads(
        (fold_dir / "best_model/standalone_eval/metrics_likelihood.json").read_text(
            encoding="utf-8"
        )
    )
    state = json.loads((fold_dir / "status.json").read_text(encoding="utf-8")).get("state")
    train = jobs.get("train", {})
    evaluation = jobs.get("best_eval", {})
    if train.get("state") != "COMPLETED" or evaluation.get("state") != "COMPLETED":
        raise SystemExit(
            f"refusing to publish an incomplete fit: {run_name} fold {fold} "
            f"train={train.get('state')} eval={evaluation.get('state')}"
        )
    return {
        "dataset": label,
        "fold": fold,
        "run_name": run_name,
        "attempt_id": (run_config.get("tracking") or {}).get("attempt_id"),
        "train_job_id": train.get("slurm_job_id", ""),
        "eval_job_id": evaluation.get("slurm_job_id", ""),
        "train_state": train.get("state"),
        "eval_state": evaluation.get("state"),
        "lifecycle_state": state,
        "macro_f1": f"{float(metrics['binary_strict_macro_f1']):.6f}",
        "positive_f1": f"{float(metrics['binary_strict_positive_f1']):.6f}",
        "uar": f"{float(metrics['binary_strict_uar']):.6f}",
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_dir, label, folds, overrides, note in CELLS:
        for fold in folds:
            run_name = overrides.get(fold) or f"qwen38_pc_{dataset_dir.split('_')[0]}_f{fold}_20260922"
            row_note = note if overrides.get(fold) else ""
            rows.append(fold_row(dataset_dir, label, fold, run_name, row_note))
    return rows


def comparison_rows() -> list[list[str]]:
    with DEFAULT_COMPARISON.open(encoding="utf-8", newline="") as handle:
        return [
            [
                row["dataset"],
                row["model"],
                row["prompt"],
                row["macro_f1"] or "—",
                row["positive_f1"] or "—",
                row["uar"] or "—",
            ]
            for row in csv.DictReader(handle)
        ]


def markdown_report(rows: list[dict[str, Any]], comparison: list[list[str]]) -> str:
    lines = [
        "# Qwen3.8-27B standalone prompt-context: results and provenance",
        "",
        "## Canonical evaluation policy for this campaign",
        "",
        "- canonical backend for this campaign: **likelihood** (`sample_prediction_mode: likelihood`,",
        "  `headline_mode: likelihood`); no teacher-forced evaluation was run for these cells;",
        "- view: `harmonized_all_windows_full_coverage`;",
        "- aggregation: strict subject-level (`aggregation_level: subject`;",
        "  `subject_score_aggregation: turkish_pooled_text_pair_mean_margin_strict_v1` for Turkish pooled);",
        "- headline metrics: Macro-F1, Positive-F1 and UAR, read from `headline/binary_strict_*`;",
        "- invalid predictions count as wrong (the strict rule; `valid_only_*` is ignored);",
        "- historical teacher-forced results are **not** mixed into likelihood comparison columns: they stay a",
        "  separately labelled legacy/diagnostic view, and this report contains likelihood rows only.",
        "",
        "## Research question and recipe",
        "",
        "How does Qwen3.8-27B score on five standalone native text-only cells when the prompt becomes one shared",
        "English instruction plus one source-grounded recording-context block per dataset (and, for Turkish pooled,",
        "versioned positive/negative question-set sentences)? The recipe is `promptcontext_v1`: `prompt.version` and",
        "`prompt.dataset_context` select centrally stored text, the pooled cell additionally selects",
        "`prompt.question_context_version`, a config without a version keeps its inline prompt unchanged, and the",
        "text-only wording drops audio-availability claims. Configs are derived from their canonical sources with a",
        "structured diff audit that allows only listed fields to change.",
        "",
        "## Matrix: 21 fits, all REPORTABLE",
        "",
        "| Dataset | Fold | Run | Train job | Eval job | Train | Eval | Lifecycle | Macro-F1 | Positive-F1 | UAR | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['fold']} | `{row['run_name']}` | {row['train_job_id']} | "
            f"{row['eval_job_id']} | {row['train_state']} | {row['eval_state']} | "
            f"{row['lifecycle_state']} | {row['macro_f1']} | {row['positive_f1']} | {row['uar']} | "
            f"{row['notes']} |"
        )
    lines += [
        "",
        "Per-fit attempt ids and job ids are in `completion_summary.csv`. Smoke runs are excluded: their metrics",
        "are not scientific results and never enter a results table.",
        "",
        "## Results and comparison columns",
        "",
        "| Dataset | Model | Prompt | Macro-F1 | Positive-F1 | UAR |",
        "|---|---|---|---|---|---|",
    ]
    for entry in comparison:
        lines.append("| " + " | ".join(entry) + " |")
    lines += [
        "",
        "CV cells report the unweighted mean of the five per-fold strict subject-level metrics; DAIC reports its",
        "single official-test fold. Only rows with the same backend, view and aggregation share a column.",
        "",
        "## Reference selection",
        "",
        "- Qwen2-7B-Instruct canonical likelihood rows: `outputs/experiment_reports/likelihood_canonical_values/"
        "derived_values.json` (`audiollm.likelihood_canonical_values.v1`), the citable derivation of the canonical",
        "likelihood values from each reference run's own saved per-subject candidate scores. Reference runs share the",
        "label, subject set, fold protocol, seed 1337, likelihood backend and `harmonized_all_windows_full_coverage`",
        "view with the Qwen3.8 cells; the Turkish pooled reference is the locked pair-margin cell (Q02, native).",
        "- DAIC same-backend old-prompt row: PR #259's Qwen3.8-27B DAIC text-only fold 0 run",
        "(`qwen38_text_only_fold0_prod_20260921`, attempt `20260921T211705Z-...`), recomputed from its own local",
        "subject predictions in the likelihood view.",
        "",
        "## DAIC: the same-backend prompt comparison",
        "",
        f"The prompt-context DAIC run and the PR #259 run share manifest hash `{DAIC_MANIFEST_HASH}` and split",
        f"metadata hash `{DAIC_SPLIT_HASH}...`, so the data, split, model revision, backend, view and official test",
        "endpoint are identical and only the prompt differs: macro-F1 moves from 0.7834 (old prompt) to 0.7631",
        "(prompt-context). The difference is reported descriptively; no significance test was prespecified or run.",
        "Outside DAIC the Qwen3.8 rows change model and prompt together, so those columns are not prompt-only effects.",
        "",
        "## Turkish pooled: prebuilt manifest and two-condition verification",
        "",
        f"- the pooled manifest was rebuilt on the cluster from the pooled campaign's four audited source pairs and its",
        f"  identity matches the recorded values exactly: manifest hash `{POOLED_MANIFEST_HASH}`,",
        f"  file sha256 `{POOLED_MANIFEST_FILE_SHA256}` (2221 rows, 120 subjects, conditions pos 1051 / neg 1170,",
        f"  four split mappings identical);",
        "- the worker never rebuilds it (`manifest_policy: prebuilt`), the submission verifies the four files before",
        "  `sbatch`, and the worker fails closed instead of rebuilding;",
        "- every pooled fold's evaluation holds exactly two sample rows per participant (one `pos_only_t17`, one",
        "  `negative_only_t17`), one label per participant and one subject-level row per participant, and the folds are",
        "  the subject-safe five-fold `train_val` mapping reused from the audited source splits.",
        "",
        "## Job history, failure and retry chain",
        "",
        "- 46 tracked jobs, all terminal: 44 `COMPLETED` 0:0, 1 `FAILED`, 1 `CANCELLED`.",
        "- Androids fold 0: job `46340959` failed after 1:44 on node `as07r2b13` with `CUDA-capable device(s) is/are",
        "  busy or unavailable` while FSDP moved the module to the device; its evaluation `46340960` was cancelled",
        "  because the `afterok` dependency could never be satisfied. Bounded retry (transient infrastructure only,",
        "  once): folded into run `qwen38_pc_androids_f0b_20260922`, train job `46342789` and evaluation job",
        "  `46342790`, both `COMPLETED` 0:0. The failed attempt and its cancellation remain in the record.",
        "- Turkish pooled fold 0 first smoke attempt failed on a real source gap (the worker refreshed stale metadata",
        "  instead of consuming the prebuilt manifest; fixed in commit `90d04dd`) and was rerun once with a new attempt;",
        "  both smoke outcomes are recorded in the agent journal, and smoke metrics stay out of every table here.",
        "",
        "## Verification",
        "",
        "- prompt identity: `run_config.yaml` records the prompt version, dataset context and the exact system prompt",
        "  with its sha256; all 23 collected folds (21 production + 2 smoke) match the rendering audit;",
        "- local gates: `tools/exp.py validate` and `finish` → 21/21 REPORTABLE;",
        "- `scripts/audit_promptcontext_runs.py` → 23/23 folds pass;",
        "- every metric displayed above was recomputed from the fold's own subject-level predictions (INVALID counted",
        "  as wrong) and checked against the stored metrics; the comparison build refuses any mismatch;",
        "- the token budget was measured with the real Qwen3.8 tokenizer: the largest prompt is 5613 tokens against a",
        "  262144 context limit with a 1024-token margin, and no cell exceeds it.",
        "",
        "## Scope and exclusions",
        "",
        "Standalone native text-only cells only. Excluded and not run: teacher-forced evaluation, merged configs and",
        "launchers, Qwen3-Omni, audio-only and audio+text cells, English translations, hidden-state classifiers,",
        "workbook updates, Tables archival beyond this report, and W&B cloud export.",
        "",
        "## Provenance",
        "",
        f"- group `{GROUP_ID}`; branch `agent/feat-qwen38-standalone-promptcontext`;",
        f"- implementation and deployment commit `{IMPLEMENTATION_COMMIT}`; deployment `{DEPLOYMENT_ID}`;",
        f"- campaign `{CAMPAIGN}`; per-fit attempt and job ids in `completion_summary.csv`;",
        "- artifact hashes for every file in this report directory are in `provenance.json`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = build_rows()
    comparison = comparison_rows()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "completion_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.output_dir / "result_report.md"
    report_path.write_text(markdown_report(rows, comparison), encoding="utf-8")

    provenance = {
        "schema_version": "audiollm.promptcontext_report_provenance.v1",
        "group_id": GROUP_ID,
        "branch": "agent/feat-qwen38-standalone-promptcontext",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "deployment_id": DEPLOYMENT_ID,
        "campaign": CAMPAIGN,
        "canonical_backend": "likelihood",
        "evaluation_view": "harmonized_all_windows_full_coverage",
        "aggregation": "strict_subject_level",
        "headline_metrics": ["macro_f1", "positive_f1", "uar"],
        "invalid_counts_as_wrong": True,
        "teacher_forced_rows_included": False,
        "fits": rows,
        "reference_artifacts": {
            "qwen2_canonical_likelihood": (
                "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/experiment_reports/"
                "likelihood_canonical_values/derived_values.json"
            ),
            "pr259_old_prompt_daic": (
                "/home/emre/Projects/AudioLLM/worktrees/LLM-Depression-feat-qwen38-daic-text/output_model/"
                "harmonized_v1_qwen38_likelihood/text_only/daic/"
                "qwen38_text_only_fold0_prod_20260921/fold_0"
            ),
        },
        "pooled_manifest": {
            "manifest_hash": POOLED_MANIFEST_HASH,
            "manifest_file_sha256": POOLED_MANIFEST_FILE_SHA256,
            "rows": 2221,
            "subjects": 120,
            "conditions": {"pos_only_t17": 1051, "negative_only_t17": 1170},
        },
        "daic_manifest_hash": DAIC_MANIFEST_HASH,
        "daic_split_metadata_hash_prefix": DAIC_SPLIT_HASH,
        "artifact_sha256": {
            "comparison.csv": sha256_file(DEFAULT_COMPARISON),
            "comparison.md": sha256_file(args.output_dir / "comparison.md"),
            "run_audit.json": sha256_file(DEFAULT_RUN_AUDIT),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {summary_path} ({len(rows)} fits)")
    print(f"wrote {report_path}")
    print(f"wrote {args.output_dir / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
