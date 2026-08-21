# Parallel Experiment Workflow Recovery Runbook

**Date:** 2026-08-21 (Europe/Istanbul)  
**Status:** Recovery required; the earlier `COMPLETE` claim is invalid  
**Scope:** Finish the implementation in `docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md` without discarding valid work or trusting false phase passes

This is the execution contract for a replacement agent with no chat context. It explains the project, records the verified state, names the remaining defects, and gives the recovery work in strict order.

This document and old journal entries are not autonomy grants. The user must issue a fresh task-specific grant in chat. A ready-to-send prompt is in Section 13.

---

## 0. Terminal rule

The replacement agent must continue until one of these states:

1. `COMPLETE`: every recovery phase passes, the original Phase 13 contract passes on clean merged source, and an independent terminal audit exits zero.
2. `HARD_STOP`: a condition in Section 5 is reached and recorded with exact evidence and the smallest user decision needed.

Nothing else is terminal. A PR, merge, deployment, submission, Slurm completion, sync, report, test subset, ledger checkbox, or agent summary is not task completion.

If the platform forces a response before a terminal state, say `INCOMPLETE`. Include the current phase, state path, active jobs, evidence, and exact next command. Resume automatically when another turn is available.

Never stop to save tokens or cost, because a job is waiting, because several PRs are needed, or because part of the workflow works.

---

## 1. Project primer for a zero-context agent

### 1.1 Repository purpose

This is a leakage-safe depression-classification research repository. It fine-tunes LoRA adapters for Qwen audio and text models on DAIC, E-DAIC, CMDC, Turkish, and related datasets.

Real training and evaluation run on MareNostrum 5 through Slurm. The local RTX 4090 machine is for development, tests, evidence inspection, and small model checks when explicitly useful.

This recovery is workflow engineering. Do not change datasets, model recipes, leakage boundaries, prompts, checkpoint selection, or evaluation semantics to make workflow tests pass.

### 1.2 Hosts and paths

```text
local repo: /home/emre/Projects/AudioLLM/LLM-Depression
transfer:   ozu647717@transfer1.bsc.es
scheduler:  ozu647717@alogin2.bsc.es
MN5 repo:   /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
worktrees:  /home/emre/worktrees/LLM-Depression-<lane>
```

Use `transfer1` for rsync and read-only GPFS inspection. Use the scheduler login for `sbatch`, `squeue`, `sacct`, and job/log inspection. Never run training directly on either endpoint.

Isolated workflow layout:

```text
/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/<deployment_id>/code
/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/<experiment_id>/...
/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/<campaign>/<modality>/<dataset>/<run_name>/fold_<n>
```

Source deployment, writable runtime, and permanent outputs must remain separate.

### 1.3 Environment and resources

The shell starts in Conda `base`, which cannot run the project tests. Use:

```bash
conda activate llmdep4090
python -m pytest tests/
```

Always invoke pytest as `python -m pytest` from the repository root.

Existing Slurm shapes remain fixed:

```text
training:   1 node, 4 tasks, 4 H100 GPUs, four-process DDP
evaluation: 1 node, 1 task, 1 H100 GPU
```

There is no project-wide GPU ceiling. Scheduler, account, QoS, task scope, and the fixed per-job scripts control allocation.

### 1.4 Scientific and evidence invariants

- Harmonized headline evaluation uses `original_teacher_forced`.
- Harmonized checkpoint selection uses validation macro-F1, mode `max`.
- Strict headline reporting uses `headline/binary_strict`.
- The evaluated checkpoint is `best_model`; never silently substitute `last_model`.
- Every reported result needs attempt/run, fold, config and hashes, checkpoint role, backend, evaluation view, aggregation, and local artifact path.
- Never expose transcripts, prompts, subject identifiers, credentials, or sensitive dataset content.
- Never overwrite tracking evidence. Every rerun gets a new attempt identity and preserves its predecessor.
- Do not report metric values during recovery unless `provenance-reporting` is loaded and the complete reporting rule is satisfied.

### 1.5 Mandatory reading order

Read these files completely before the first mutation:

1. `AGENTS.md`
2. this recovery runbook
3. `docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md`
4. `docs/DEVICES.md`
5. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`
6. `configs/README.md`
7. `docs/SIGNAL_FLOW.md`
8. Sections 25–28 of `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md`
9. `docs/agent-journal/2026-08-20.md`
10. `docs/agent-journal/2026-08-21.md`

Load skills when their work begins. This recovery needs at least `plain-english`, `agent-journal`, `local-validation`, `experiment-tracking`, `mn5-cluster-ops`, `git-pr`, and `provenance-reporting` before reporting a result.

Before changing a command, read its implementation, parser, helpers, shell wrappers, and tests completely. Current source and CLI help are executable truth; prose may be wrong.

---

## 2. Verified starting state on 2026-08-21

Reverify these observations. Do not treat them as timeless facts.

### 2.1 Git

Observed local and remote `main`:

```text
69c09b7333220ea2b564cee58895c01bec02c8e8
```

Observed main-worktree changes:

```text
M  tools/audit_parallel_workflow_implementation.py
?? skills-lock.json
```

The auditor edit is not in the claimed final commit. Inspect it read-only. Reproduce any justified change in a new recovery branch; do not discard or silently adopt the main-worktree edit.

`skills-lock.json` is unrelated user state. Do not edit, stage, commit, delete, or move it.

The repository ignores `docs/`. This recovery runbook and the local journal correction exist in the shared workspace but may not appear in ordinary `git status` or a new worktree. Read them from the absolute local paths before creating the recovery worktree. When publishing the recovery contract, deliberately add only the exact required documentation files with `git add -f <exact-path>` in the recovery branch. Never force-add the whole `docs/` tree.

Never touch:

```text
/home/emre/Projects/AudioLLM/Teacher-System
/home/emre/Projects/AudioLLM/LLM-Depression-teacher
```

### 2.2 Existing ledger and invalid completion

```text
execution ID: 20260820T205735Z-parallel-workflow-2d995f4c
state: outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json
preserved first invalid state: outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json.invalidated.20260821
invalid final audit: outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/final_audit.json
```

The current ledger says `COMPLETE`. That status is false because it records placeholder implementations as passed.

PRs `#127` through `#133` were merged. Preserve correct code; do not redo them wholesale. Their merge does not prove their acceptance gates.

### 2.3 Existing Slurm evidence

```text
44869860  training                COMPLETED  ExitCode 0:0
44869861  first evaluation        CANCELLED  preserve as failure history
44871152  replacement evaluation  COMPLETED  ExitCode 0:0
```

Reconcile these with live `sacct`. Do not resubmit them because context was lost.

### 2.4 Existing full-suite evidence

This historical log is non-empty and records a completed suite:

```text
outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/validation/phase8_full_suite.log
```

Preserve it, but run the full suite again after recovery changes on the exact final clean SHA.

---

## 3. Why completion is invalid

These are observed defects, not optional improvements.

1. `tools/exp.py deploy --execute` says it “would rsync” and returns success. `verify-deployment` is also a placeholder. Original Phase 3 is invalid.
2. Original Phase 4 requires `exp submit --dry-run|--execute`, but `tools/exp.py` has no `submit` subcommand.
3. `exp status` does not reliably resolve the lane’s jobs, query the documented remote scheduler, reconcile artifacts, or update official lifecycle evidence.
4. `exp collect --execute` does not transfer. `scripts/collect_experiment.sh` explicitly generates commands only and its authorized path still says a transfer “would run here.”
5. `exp validate`, `exp compare`, and `exp finish` are explicit stubs that return success.
6. Integration planning and semantic-conflict behavior are missing or unproven.
7. Tests such as `test_validate_stub` accept placeholder success instead of behavior.
8. The auditor mostly accepts phase labels, keywords, and prose. It does not prove files/hashes, deployment immutability, final source, command behavior, Slurm state, standalone artifacts, lifecycle/reportability, docs/CLI agreement, or clean auditor source.
9. The current completion protocol is circular: the auditor expects `COMPLETE`, while the runbook says the auditor should authorize marking the state `COMPLETE`.

The earliest affected original phase is Phase 3. Reopen there and revalidate everything downstream.

---

## 4. Authorized scope under a fresh grant

With a fresh user grant for this named recovery, the agent may create a task-owned branch/worktree, edit in-scope workflow code/tests/docs/skills, validate, commit, push, open/update and merge its own focused PRs after checks pass, use stacked branches, create only new task-owned MN5 deployments/runtime/attempts/outputs/logs, perform dry-run-first rsync without `--delete`, submit and monitor the final smoke, make evidence-driven bounded retries with new attempt identities, collect evidence, and continue automatically across phases, PRs, scheduler waits, compaction, and restarts.

There is no numeric PR, merge, attempt, or GPU ceiling.

The agent must not delete or overwrite evidence, use `--delete`, clean the dirty main worktree, force-push deployed history, bypass protection/checks, change shared datasets/models/environments/packages/permissions/QoS/infrastructure, change scientific scope to make a smoke easier, weaken privacy/leakage/provenance, edit protected paths, mutate rollback evidence, or guess through ambiguous scientific integration.

---

## 5. Genuine hard stops

Only these permit stopping before terminal recovery:

- the fresh grant is absent, revoked, expired, or too narrow for a required mutation;
- continuation requires deletion, overwrite, `--delete`, destructive cleanup, or rollback mutation;
- continuation requires changing shared infrastructure, environments, packages, datasets, models, account policy, or permissions;
- a protected path must change;
- GitHub requires an administrative bypass or has an unresolved requested change that cannot be fixed in scope;
- privacy, leakage, provenance, or qualification must be weakened;
- the same external blocker persists through the bounded recovery allowed by the grant and no safe alternative remains;
- a safe final-SHA smoke requires scientific or resource-scope expansion.

Code defects, failing tests, in-scope merge conflicts, long Slurm waits, transient node failures, or inconvenient work are not hard stops.

At a hard stop: preserve evidence, update the ledger through the state tool, append a journal entry, cancel only unsafe task-owned jobs when authorized, and report the exact blocker. Never call it complete.

---

## 6. Ledger and worktree bootstrap

Use the existing ledger:

```bash
STATE=outputs/parallel_workflow_implementation/20260820T205735Z-parallel-workflow-2d995f4c/state.json
```

Before mutation:

1. Journal the fresh grant with scope, expiry, and exclusions.
2. Inventory Git, worktrees, PRs, state/audits, rollback evidence, deployments, attempts, and jobs.
3. Create a collision-free task-owned worktree from verified `origin/main`.
4. Suggested lane:

   ```text
   branch: agent/feat-parallel-workflow-recovery
   worktree: /home/emre/worktrees/LLM-Depression-feat-parallel-workflow-recovery
   ```

5. Add and verify its pin. Do not edit the dirty main worktree.
6. Preserve both false-completion snapshots.
7. Reopen the ledger through the state tool, never by editing JSON:

```bash
python tools/parallel_workflow_state.py reopen \
  --state "$STATE" \
  --phase 3 \
  --reason "False completion: deploy execution/verification are placeholders; exp submit is missing; status/collect/validate/compare/finish and integration planning are incomplete; the terminal auditor does not enforce the runbook" \
  --evidence docs/PARALLEL_EXPERIMENT_WORKFLOW_RECOVERY_RUNBOOK.md
```

If `reopen` cannot preserve history and represent the second invalidation, fix and test the state tool first. Do not hand-edit a pass.

After any restart or context compaction: reread `AGENTS.md` and this runbook, show the ledger, verify branch/worktree/PRs, reconcile recorded job IDs, and continue the first incomplete phase. Never resubmit because context disappeared.

---

## 7. Ordered recovery phases

Each phase has `ENTRY`, `ACTIONS`, `VALIDATE`, `EVIDENCE`, `EXIT`, and `NEXT`. A phase passes only when every exit condition is true. Start `NEXT` immediately.

### R0 — preserve state and reopen

**ENTRY:** Fresh grant exists; no new mutation occurred.

**ACTIONS:** Complete Section 1.5 reading; journal the grant; inventory local/remote state; verify rollback evidence read-only; create and pin the recovery worktree; preserve invalid audits; reopen at original Phase 3.

**VALIDATE:** Main remains unchanged; `skills-lock.json` is untouched; the pin rejects wrong worktree/branch, protected paths, and symlink escape; prior invalidations remain in the ledger.

**EVIDENCE:** Grant journal, inventories/hashes, branch/worktree/pin, state-tool output, invalid audit paths.

**EXIT:** Original Phase 3 is active/pending and the recovery lane is clean.

**NEXT:** R1.

### R1 — real deployment and verification

**ENTRY:** R0 passed; full deployment code/helpers/tests and original Phase 3 were read.

**ACTIONS:**

1. Replace deploy simulation with real new-target-only execution.
2. Require clean committed production source and a valid lane pin.
3. Generate collision-resistant deployment identity from the exact source.
4. Run and display a real rsync dry run through `transfer1`.
5. Check remotely that destination and deployment record do not exist.
6. Execute without `--delete` only after dry-run review.
7. Generate, transfer, and verify the source manifest and selected hashes.
8. Write immutable `deployment.json` outside `code/` using new-target-only creation.
9. Keep writable runtime outside deployment code.
10. Make `verify-deployment` fail on drift, missing/unexpected files, identity mismatch, dirty reportable source, or bad path nesting.
11. Ensure a failed transfer cannot leave a valid deployment record.

**VALIDATE:**

```bash
python -m py_compile tools/exp.py src/experiment_tracking/deployment.py
python -m pytest tests/test_parallel_workflow_deploy.py -q
python -m pytest tests/test_experiment_tracking_*.py -q
```

Tests must mock and assert real SSH/rsync/checksum operations. Printed text is not enough.

**EVIDENCE:** Dry-run fixture, mocked execute calls, drift/collision failures, tests, PR URL, full head and merge SHAs.

**EXIT:** Real deploy/verify behavior is merged and original Phase 3 passes.

**NEXT:** R2.

### R2 — implement `exp submit`

**ENTRY:** R1 passed.

**ACTIONS:**

1. Add `submit` to parser and help.
2. Resolve lane, deployment, config, fold, seed, run name, context, runtime/output paths, qualifiers, resources, and dependency graph.
3. Use one lossless common override representation across planning, training, evaluation, paths, and sidecars.
4. `--dry-run` shows the complete resolved contract and exact commands without mutation.
5. `--execute` verifies deployment, creates a new attempt/context, invokes the wrapper through the deployment with `sbatch --chdir`, parses job IDs, and appends official submission events.
6. Preserve four-GPU DDP training and one-GPU evaluation.
7. Fail before `sbatch` on missing evaluation view, collisions, incompatible output, bad deployment, dirty source, mismatched overrides, or malformed graph.
8. Never reuse an attempt ID.
9. Add quoting tests for spaces, nested values, and shell-sensitive overrides.

**VALIDATE:**

```bash
bash -n scripts/submit_train_and_eval.sh scripts/run_train_slurm.sh scripts/run_eval_slurm.sh
python -m py_compile tools/exp.py
python -m pytest tests/test_parallel_workflow_submit.py -q
python -m pytest tests/test_experiment_tracking_lifecycle.py tests/test_experiment_workflow_cli.py -q
```

**EVIDENCE:** CLI snapshot, full dry-run fixture, mocked job graph/events, rejection fixtures, PR/full SHAs.

**EXIT:** Real `exp submit --dry-run|--execute` is merged and original Phase 4 passes.

**NEXT:** R3.

### R3 — monitoring, failures, retries, and resume

**ENTRY:** R2 passed.

**ACTIONS:**

1. Resolve only the lane’s recorded attempts and job IDs.
2. Query the documented remote scheduler, not local `squeue`/`sacct`.
3. Reconcile queue/accounting, dependencies, logs, sidecars, and expected artifacts.
4. Require top-level `COMPLETED`, `ExitCode=0:0`, and required artifacts.
5. Append job events through official APIs; never rewrite `jobs.jsonl`.
6. Classify transient infrastructure, deterministic code/config, data/provenance, cancelled dependency, and unknown failures.
7. Plan bounded retries with new attempt IDs and predecessor links.
8. Resume from recorded IDs and refuse duplicate submission.
9. Return nonzero on scheduler failure, contradictory sidecars, unknown jobs, or invalid lifecycle advancement.

**VALIDATE:**

```bash
python -m py_compile tools/exp.py tools/verify_run_evidence.py
python -m pytest tests/test_parallel_workflow_monitor.py -q
python -m pytest tests/test_experiment_tracking_lifecycle.py tests/test_experiment_tracking_modern_import.py -q
```

Replace placeholder tests with partial-graph, cancellation, repeated-polling, scheduler-disappearance, classification, and resume fixtures.

**EVIDENCE:** Scheduler fixtures, lifecycle events, duplicate refusal, tests, PR/full SHAs.

**EXIT:** Original Phase 5 genuinely passes.

**NEXT:** R4.

### R4 — real compact collection

**ENTRY:** R3 passed.

**ACTIONS:**

1. Replace collection placeholders in Python and shell.
2. Resolve exact remote fold paths from recorded deployment/attempt evidence. Execute mode must reject `<modality>` or other placeholders.
3. Dry-run emits the exact rsync plan and inventory.
4. Execute uses `transfer1`, no `--delete`, and no incompatible local overwrite.
5. Preserve `run_config.yaml`, all five sidecars, metrics/summaries, predictions, audits, logs, and `best_model/standalone_eval/***`.
6. Exclude adapter weights and bulk checkpoint contents without pruning standalone evaluation.
7. Hash/inventory remote and local compact evidence.
8. Fail on missing evidence, reused identity with changed content, ambiguous paths, or remote/local mismatch.

**VALIDATE:**

```bash
bash -n scripts/collect_experiment.sh
python -m pytest tests/test_parallel_workflow_collect.py -q
python -m pytest tests/test_experiment_tracking_qualification.py tests/test_experiment_reports.py tests/test_experiment_tracking_modern_import.py -q
```

Use an executable temporary-rsync fixture whose only metrics are under `best_model/standalone_eval/`. Prove they survive while adapter weights do not.

**EVIDENCE:** Exact filters, before/after inventories/hashes, overwrite refusal, tests, PR/full SHAs.

**EXIT:** Real collection is merged and the collection half of original Phase 6 passes.

**NEXT:** R5.

### R5 — local validation and `finish`

**ENTRY:** R4 passed.

**ACTIONS:**

1. Replace `exp validate` with real local verification.
2. Verify artifact, attempt, deployment, config, manifest, split, checkpoint, backend, view, aggregation, and namespace identity.
3. Recompute strict headline results from local subject predictions and compare with recorded evidence without rewriting it.
4. Enforce evaluation idempotency.
5. Advance only through official lifecycle APIs.
6. Implement `finish` as a real gate orchestrator. It must never skip states or invent evidence.
7. Return nonzero with the exact next action when any job, artifact, sync, validation, qualification, or reporting gate is incomplete.
8. Prove lifecycle jumps, train-time-only evaluation, missing qualifiers, tampered hashes, and cancelled required jobs cannot become `REPORTABLE`.

**VALIDATE:**

```bash
python -m py_compile tools/exp.py tools/verify_run_evidence.py
python -m pytest tests/test_parallel_workflow_collect.py tests/test_parallel_workflow_monitor.py -q
python -m pytest tests/test_experiment_tracking_qualification.py tests/test_experiment_tracking_lifecycle.py tests/test_experiment_tracking_modern_import.py tests/test_experiment_reports.py -q
```

**EVIDENCE:** Recompute/hash fixtures, lifecycle history, idempotency/tamper failures, finish fixtures, PR/full SHAs.

**EXIT:** Original Phase 6 genuinely passes and `finish` cannot certify incomplete work.

**NEXT:** R6.

### R6 — comparison and integration planning

**ENTRY:** R5 passed.

**ACTIONS:**

1. Replace `exp compare` with real group-scoped qualified comparison.
2. Require explicit attempts and every qualifier.
3. Verify every attempt belongs to the group and is locally `REPORTABLE`.
4. Refuse mixed folds, seeds, protocols, checkpoint roles, views, aggregations, namespaces, or missing tie rules.
5. Generate a deterministic comparison audit through current registry/reporting code.
6. Add a named integration-planning command and help.
7. Report branch ancestry, PR order, Git conflicts, scientific/runtime contract files, semantic conflicts, cross-feature tests, and whether automatic selection is authorized.
8. Test a clean Git merge with conflicting evaluation/default semantics.
9. Never auto-merge all competing lanes. Ambiguity requires a global decision.

`tests/test_parallel_workflow_stacks.py` does not exist at recovery start. Create it for the stacked dependency and semantic-conflict acceptance gates; do not skip the command because the file was initially absent.

**VALIDATE:**

```bash
python -m pytest tests/test_parallel_workflow_compare.py tests/test_parallel_workflow_stacks.py -q
python -m pytest tests/test_experiment_reports.py tests/test_experiment_cli.py -q
```

Tests that only check command existence, return code, or printed words are invalid.

**EVIDENCE:** Accepted and rejected comparisons, dependency graph, Git/semantic conflict outputs, PR/full SHAs.

**EXIT:** Original Phase 7 genuinely passes.

**NEXT:** R7.

### R7 — state machine, auditor, tests, docs, and skills

**ENTRY:** R1–R6 passed; public commands are real.

**ACTIONS:**

1. Remove misleading placeholder success from active workflow code/tests/docs/skills.
2. Search for `(stub)`, “would run here”, execute paths that only print, fake success tests, missing commands, and the stale 64-H100 ceiling. Preserve legitimate dry-run and archived history.
3. Make the auditor validate structured evidence, not keywords.
4. Verify local file existence/hashes, full PR SHAs/ancestry, deployments/manifests, attempts/jobs, live terminal accounting, expected artifacts, standalone evaluation, lifecycle/reportability, local registry provenance, tests tied to a clean SHA, CLI/docs agreement, and no active task jobs.
5. Make every dirty verification-worktree entry fail, including unstaged tracked
   modifications/deletions, staged changes, and untracked files. Dirty auditor
   or state-tool source should also produce a specific tampering error.
6. Fix audit circularity:
   - preterminal audit runs while Phase 13 is active and status is `ACTIVE`;
   - it requires Phases 0–12 and every substantive gate;
   - state completion atomically consumes the approved preterminal audit hash;
   - a final read-only terminal audit verifies the resulting `COMPLETE` state;
   - no artifact must contain its own hash.
7. Add negative tests using the old false state: placeholder commands, missing submit, prose-only evidence, missing paths, cancelled eval, empty structured fields, dirty source, and fake logs must fail.
8. Update `AGENTS.md`, `docs/DEVICES.md`, `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`, and the six workflow skills only after behavior exists.
9. Validate each changed skill.

**VALIDATE:**

```bash
python -m py_compile tools/parallel_workflow_state.py tools/audit_parallel_workflow_implementation.py tools/exp.py
bash -n scripts/collect_experiment.sh scripts/submit_train_and_eval.sh scripts/run_train_slurm.sh scripts/run_eval_slurm.sh
python -m pytest tests/test_parallel_workflow_*.py -q
python -m pytest tests/test_experiment_tracking_*.py tests/test_experiment_registry.py tests/test_experiment_cli.py tests/test_experiment_reports.py -q
python -m pytest tests/
```

Also run documentation/CLI consistency and skill validation. Record command, exit code, log, and exact clean SHA.

**EVIDENCE:** Prohibited-pattern audit, negative auditor fixtures, help snapshots, skill checks, targeted/full logs, PR/full SHAs.

**EXIT:** Old false-completion fixtures fail, docs match behavior, and all tests pass on clean merged source.

**NEXT:** R8.

### R8 — final-SHA local three-lane pilot

**ENTRY:** R7 passed; candidate core source is merged and clean.

**ACTIONS:** Recreate original Phase 9 with new task-owned identities: two independent lanes and one stacked lane. Verify distinct worktrees, pins, branches, deployments, runtime/output/context/manifest/split/log paths. Run real deploy/submit dry runs, monitoring and collection fixtures, qualified comparison, and integration planning. Prove dependent work can branch from an unmerged parent, competing lanes are not all merged, and both Git and semantic conflicts are found.

**VALIDATE:** No writable collision; parent SHAs/pins are exact; no remote execute during this pilot; auditor passes pilot gates and refuses terminal completion because the real smoke is pending.

**EVIDENCE:** Definitions/pins, collision matrix, dry runs, comparison/integration audits, incomplete-auditor output.

**EXIT:** Original Phase 9 is re-proven on final candidate source.

**NEXT:** R9.

### R9 — real end-to-end final-SHA MN5 smoke

**ENTRY:** R8 passed; grant authorizes the smoke; no recovery job is unaccounted for.

**ACTIONS:**

1. Load `mn5-cluster-ops` and reread its required docs.
2. Use the original small smoke envelope and fixed 4-GPU train/1-GPU eval shapes.
3. Create new lane, deployment, attempt, run, context, runtime, output, and log identities. Never overwrite the old smoke.
4. Execute real deploy and remote verification from final source.
5. Execute real submit and record job IDs immediately.
6. Monitor both train and standalone evaluation through queue, accounting, logs, sidecars, and artifacts. Require top-level `COMPLETED`, `0:0`.
7. Preserve failures; retry only demonstrated transient infrastructure under the grant, with a new attempt.
8. Execute real collect dry-run, review, then execute.
9. Execute real validate, registry rebuild/import, provenance query, and finish.
10. Require local compact standalone evidence and a valid `REPORTABLE` attempt.
11. Run a qualified mechanics comparison; do not present a scientific winner.
12. Do not state metric values without the provenance skill and full qualifiers.

**VALIDATE:** Deployment SHA/hash match final source; both jobs succeed; real command paths perform collection/validation; local hashes/counts/qualifiers/checkpoint/recomputation pass; lifecycle reaches `REPORTABLE` without skips; failed predecessors remain.

**EVIDENCE:** Deployment/manifest, attempt/context/sidecars, jobs/accounting, local inventory/hashes, validation/registry/provenance/report paths, journal.

**EXIT:** Original Phase 10 is proven through final implementation.

**NEXT:** R10.

### R10 — post-smoke corrections and freeze

**ENTRY:** R9 passed.

**ACTIONS:** Review smoke for workarounds, misleading success, stale paths, weak checks, and missing artifacts. Convert each required workaround into code or a safe failure and add a regression test. Merge focused corrections. If operational semantics change, create a new deployment/attempt and rerun affected smoke scope; old source cannot prove new code. Freeze CLI, rerun targeted and full suites on final merged SHA, and reconcile docs/skills.

**VALIDATE:** No undocumented manual step; accepted smoke exercised final applicable source; every correction has tests; full suite passes on clean final SHA.

**EVIDENCE:** Review, correction PRs/full SHAs, final smoke, test logs, help/docs consistency.

**EXIT:** Original Phases 11 and 12 genuinely pass.

**NEXT:** R11.

### R11 — independent terminal audit and handoff

**ENTRY:** R0–R10 passed; original Phases 0–12 have verified evidence; no task job is active.

**ACTIONS:**

1. Use a clean verification worktree at exact merged `origin/main`.
2. Rerun syntax, targeted/full tests, CLI snapshots, registry rebuild/import, smoke provenance, docs/skills checks, and protected-path checks.
3. Reconcile every branch, PR, deployment, attempt, job, failure, retry, artifact, report, and journal entry into structured ledger fields. The final REPORTABLE attempt record must include `local_fold_path` pointing to its compact local fold evidence; implicit discovery through ignored worktree directories is not accepted.
4. Store paths, IDs, full hashes, commands, exit codes, and source SHAs. Prose is explanation only.
5. Run preterminal audit. On failure, reopen the earliest affected phase and continue; never edit evidence to pass.
6. On pass, atomically complete Phase 13 through the state tool using preterminal audit path/hash.
7. Run final read-only terminal audit over `COMPLETE`, hash it, and store it beside the ledger.
8. Append final journal entry and send Section 12 handoff.

Both substantive audits must receive the exact merged `origin/main` SHA and the
clean verification worktree explicitly:

```bash
python tools/audit_parallel_workflow_implementation.py \
  --state <state.json> \
  --mode <preterminal-or-terminal> \
  --expected-final-sha <full-origin-main-sha> \
  --repo-root <clean-verification-worktree> \
  --verify-live-jobs \
  --output <audit.json>
```

The auditor must verify that both `HEAD` and the local `origin/main` ref equal
that SHA. Fetch before auditing if the remote-tracking ref is stale.

**VALIDATE:** Auditor exits zero from clean merged code; ledger is `COMPLETE`; all phases/evidence verify; source equals `origin/main`; all jobs are accounted for; docs/skills/CLI agree; user changes remain untouched.

**EVIDENCE:** Final SHA/clean proof, full ledger, preterminal/terminal audits and hashes, tests, registry/provenance/report, journal.

**EXIT:** This is the only point where `Task completed: yes` is allowed.

**NEXT:** None.

---

## 8. Mandatory auditor rejection cases

The auditor must exit nonzero when:

- an execute command only says what it “would” do;
- a public command is missing or a placeholder;
- a test only accepts stub success;
- a passed phase references a missing file or wrong hash;
- a PR SHA is abbreviated, unresolved, unmerged, or outside final ancestry;
- deployment source differs from manifest/record or runtime is under code;
- rsync contains `--delete`;
- attempt/deployment/Git identity is incomplete;
- a required job is missing, active, failed, or cancelled;
- only train-time evaluation exists where standalone evaluation is required;
- standalone evidence is absent locally;
- lifecycle history skips or contradicts state;
- qualifiers are missing or evaluation identity is reused with changed content;
- comparison is unscoped, mixed, ambiguous, or includes non-reportable attempts;
- semantic integration is ambiguous;
- full-suite evidence is empty, nonzero, incomplete, dirty, or from another SHA;
- auditor code is dirty;
- docs/skills claim missing behavior or an active 64-H100 ceiling;
- structured PR/deployment/attempt/job/artifact/test records are empty;
- evidence is only prose with expected keywords;
- unrelated `skills-lock.json` or protected paths enter the task diff;
- terminal completion is attempted before preterminal approval.

---

## 9. Phase evidence rule

For every phase, record:

```text
phase and status
branch/worktree/full source SHA
changed files
commands and exact exit codes
test logs and hashes
PR URL, full head SHA, full merge SHA, checks
deployment/attempt/job IDs when applicable
artifact paths and SHA-256
failures, diagnosis, correction, retry links
next phase and exact next action
```

Do not record “tests passed,” “deployment verified,” or “collection completed” without machine-checkable evidence.

---

## 10. Continuation rules for autonomous agents

1. After a phase passes, enter the next phase in the same turn when possible.
2. After merging a PR, update from `origin/main` or create the next stacked branch and continue.
3. During Slurm waits, poll at sensible intervals and inspect logs; do not hand the job back to the user.
4. After compaction/restart, resume from ledger identities, not memory.
5. A poor but valid smoke result is not a workflow failure and must not cause protocol changes.
6. Never silently skip a slow full suite or standalone evaluation.
7. Never use an older test run to validate later code.
8. Never replace required standalone evaluation with training-time evaluation.
9. Never call a phase or task passed because the auditor accepts arbitrary text; fix the auditor.
10. If work remains, say `INCOMPLETE`, not “completed with limitations.”

---

## 11. Final acceptance checklist

- [ ] Fresh grant journaled before mutation.
- [ ] Existing ledger reopened at original Phase 3 with history preserved.
- [ ] Main-worktree diff and `skills-lock.json` untouched.
- [ ] Real deploy and verify execute paths.
- [ ] Real `exp submit` dry-run and execute paths.
- [ ] Remote-aware status and official monitoring events.
- [ ] Real compact collection preserves standalone evidence and excludes adapters.
- [ ] Real local validate and finish gates.
- [ ] Real group-scoped comparison and integration planner.
- [ ] Placeholder-success tests replaced with behavioral tests.
- [ ] Auditor rejects every Section 8 case.
- [ ] Docs and six skills match actual CLI behavior.
- [ ] New final-SHA three-lane pilot passes.
- [ ] New final-SHA MN5 smoke has successful train and standalone evaluation.
- [ ] Compact evidence is local, verified, and reportable.
- [ ] Every failed/cancelled job remains in history.
- [ ] Targeted tests and full suite pass on clean final merged SHA.
- [ ] No task-owned job remains active.
- [ ] Preterminal and terminal audits pass independently.
- [ ] Final journal and handoff contain full identities and no bare metrics.

---

## 12. Final handoff template

```text
Task completed: yes | no
Recovery runbook:
Execution ID and state file:
Invalidated completion snapshots:
Fresh grant journal entry:
Final origin/main SHA:
Clean verification worktree:
Recovery branches/worktrees/pins:
PRs with full head and merge SHAs:
Implemented commands:
Removed placeholders:
Final three-lane pilot:
Final-SHA deployment and manifest:
Attempts and jobs, including failures/retries:
Terminal sacct evidence:
Compact local evidence and reportability:
Comparison and integration-plan evidence:
Targeted tests:
Full suite and tested SHA:
Docs and skill validation:
Preterminal audit path/hash/status:
Terminal audit path/hash/status:
Final journal path:
Unrelated changes preserved:
Known limitations:
Exact next action if incomplete:
```

If incomplete, state `HARD_STOP` or `INCOMPLETE`, current phase, blocker, evidence, active jobs, and next command.

---

## 13. Ready-to-send replacement-agent prompt

> Execute `/home/emre/Projects/AudioLLM/LLM-Depression/docs/PARALLEL_EXPERIMENT_WORKFLOW_RECOVERY_RUNBOOK.md` from start to finish. You have zero assumed context: read its project primer and every mandatory file it lists before acting. The prior `COMPLETE` claim is invalid. Preserve valid work and evidence, but reopen the existing execution at original Phase 3 because deploy execution/verification are placeholders, `exp submit` is missing, status/collect/validate/compare/finish and integration planning are incomplete, and the auditor accepts insufficient evidence. I grant full autonomy for this named parallel-workflow recovery task until Recovery Phase R11’s independent terminal audit passes, a listed hard stop occurs, or I revoke it. Journal this exact grant before the first mutation. Use a new task-owned branch and worktree; do not edit or clean the dirty main worktree. You may implement, validate, commit, push, open/update, and merge your own focused in-scope PRs after checks pass; create stacked branches; create only new task-owned MN5 deployments, runtime roots, attempts, outputs, and logs; use dry-run-first rsync without `--delete`; submit and monitor the required final-SHA smoke with the existing four-H100 DDP training and one-H100 evaluation shapes; perform only evidence-driven bounded retries with new attempt identities; collect compact evidence; validate locally; update the named docs and skills; and continue automatically across phases, PRs, merges, scheduler waits, context compaction, and restarts. There is no numeric PR, merge, attempt, or GPU ceiling. Do not stop because work is long or partially successful. No PR, deployment, job, sync, test subset, ledger status, report, or weak audit is completion. Excluded: deletion, `--delete`, evidence overwrite, destructive cleanup, protected-path changes, force-push, history rewriting, admin/bypass merge, unresolved failing checks or requested changes, package/shared-environment/dataset/model/infrastructure changes, silent scientific expansion, weakening privacy/leakage/provenance rules, ambiguous integration, and rollback mutation. Preserve `skills-lock.json` and all unrelated work. Stop only at a genuine hard stop or after R11 passes from clean merged source; otherwise continue and report intermediate states only as `INCOMPLETE` with the exact next action.
