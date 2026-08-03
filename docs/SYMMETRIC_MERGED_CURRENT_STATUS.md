# Symmetric Merged Current Situation

Status captured: **2026-08-03 16:15:00 +03:00** (Europe/Istanbul).

## Canonical documentation paths

- Protocol plan: [`docs/SYMMETRIC_MERGED_PROTOCOL_PLAN.md`](SYMMETRIC_MERGED_PROTOCOL_PLAN.md)
- MN5 execution runbook: [`docs/MN5_AGENT_EXECUTION_RUNBOOK.md`](MN5_AGENT_EXECUTION_RUNBOOK.md)
- This status note: [`docs/SYMMETRIC_MERGED_CURRENT_STATUS.md`](SYMMETRIC_MERGED_CURRENT_STATUS.md)

## Current execution

- Run ID: `symmetric_merged_smoke_6fba6e632653`
- Source commit submitted for the corrected CV retries: `9888e81`
- Repository implementation commit currently on `main`: `50623d0` (`Accept abbreviated provenance commits`)
- Smoke chain: 3 jobs completed and the strict smoke audit passed.
- CV registry: `active`; `terminal=false`; observed failures: none after the
  scoped retry described below.
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

## Scoped retry on 2026-08-03

- The older `audio_only` fold-1 training job `44121800` failed after rank 0
  finished while ranks 1-3 timed out in the early-stop broadcast. Its
  provenance was `f125d79`, predating the distributed early-stop correction
  in `d82a151`.
- Only its impossible descendants (`44121801`, `44121802`) were cancelled.
  The replacement chain is `44136283 -> 44136285 -> 44136286`; its training
  job is running with corrected source `9888e81`.
- A full-vs-abbreviated Git hash comparison briefly caused 15 redundant
  text-only jobs to be submitted. They were cancelled before starting, the
  registry was restored to the original successful text-only job IDs, and
  commit `50623d0` now compares valid Git abbreviations safely. The local and
  MN5 checksums for the four changed files match.
- The registry once again contains exactly the intended 3 smoke and 45 CV
  logical jobs. No healthy production job was cancelled or resubmitted.

## Live follow-up at 16:04

- CV remains active with 7 training jobs completed, 8 training jobs running,
  6 postprocess jobs completed, 1 postprocess job running, 8 postprocess jobs
  pending, 5 heads completed, 1 head running, and 9 heads pending; observed
  failures remain zero.
- `audio_only` CV fold 0 completed its replacement-safe training,
  postprocessing (`44130958`, `ExitCode=0:0`), and released head job
  `44130959`. Its CPU Optuna study has completed 19 of 150 trials with no
  failed trial state; the current best objective is `0.9474053406158222`.
- The remaining pending jobs show only expected dependency waits. No final
  stage has been submitted and the CV acceptance gate remains closed.

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
