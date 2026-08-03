# Symmetric Merged Current Situation

Status captured: **2026-08-03 15:01:37 +03:00** (Europe/Istanbul).

## Canonical documentation paths

- Protocol plan: [`docs/SYMMETRIC_MERGED_PROTOCOL_PLAN.md`](SYMMETRIC_MERGED_PROTOCOL_PLAN.md)
- MN5 execution runbook: [`docs/MN5_AGENT_EXECUTION_RUNBOOK.md`](MN5_AGENT_EXECUTION_RUNBOOK.md)
- This status note: [`docs/SYMMETRIC_MERGED_CURRENT_STATUS.md`](SYMMETRIC_MERGED_CURRENT_STATUS.md)

## Current execution

- Run ID: `symmetric_merged_smoke_6fba6e632653`
- Source commit submitted for the corrected CV retries: `9888e81`
- Repository implementation commit currently on `main`: `6936aff` (`Preserve retry registry dependencies`)
- Smoke chain: 3 jobs completed and the strict smoke audit passed.
- CV registry: `active`; `terminal=false`; observed failures: none.
- No final-stage jobs have been submitted.

The authoritative MN5 registry is:

`/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/outputs/symmetric_merged_jobs/symmetric_merged_smoke_6fba6e632653.json`

The local checkout did not contain a copy of this run-specific registry at the
time of this check. It must be retrieved during the planned result
synchronization before local reporting is finalized.

## CV job counts

The registry contains 48 jobs total: 3 smoke jobs and 45 CV jobs.

| Stage | Completed | Running | Pending | Failed | Terminal |
|---|---:|---:|---:|---:|---|
| Smoke | 3 | 0 | 0 | 0 | yes |
| CV | 15 | 10 | 20 | 0 | no |

The 10 training jobs are running. Their 20 dependent postprocessing and head
jobs are pending with the normal Slurm `Dependency` reason. This is expected:
the descendants will become eligible as their training jobs complete. There
are currently no `DependencyNeverSatisfied` jobs and no reason to cancel or
resubmit healthy work.

## Remaining gates

1. Let all 45 CV logical jobs reach successful terminal state (`0:0`).
2. Run strict CV acceptance audits for `audio_text`, `audio_only`, and
   `text_only`.
3. Submit the 9 final DAIC-official-test jobs only after all CV audits pass.
4. Audit the final stage, then selectively synchronize results through
   `transfer1` without `--delete` or heavy artifacts.
5. Run the local omitted-heavy-artifact audit, generate the CSVs, workbook,
   and Markdown report, validate their consistency, then commit and push the
   final artifacts.

Until the CV gate clears, no additional job submission is required.
