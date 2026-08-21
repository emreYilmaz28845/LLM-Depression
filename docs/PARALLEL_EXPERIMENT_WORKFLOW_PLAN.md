# Parallel Experiment Workflow — End-to-End Implementation Runbook

> **Recovery notice (2026-08-21):** The first implementation was incorrectly marked complete while required commands remained missing or placeholders. A replacement agent must use `docs/PARALLEL_EXPERIMENT_WORKFLOW_RECOVERY_RUNBOOK.md`, reopen the existing execution at Phase 3, and complete Recovery Phases R0–R11. This original runbook remains the architecture and acceptance contract.

**Date:** 2026-08-20 (Europe/Istanbul)  
**Status:** Execution contract; implementation pending  
**Scope:** Run 2–5 concurrent code-changing experiments across local Git worktrees and isolated MN5 deployments. Preserve high agent autonomy inside each experiment lane while keeping shared state, evidence, and rollback paths safe.

This document is both the settled design and the end-to-end implementation runbook. Commands marked **proposed** do not exist yet. Current executable truth remains the source, current YAML files, CLI `--help`, `docs/DEVICES.md`, and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` until Phase 12 switches the repository contract to the new workflow.

---

## 0. Execution contract — read this before doing anything

### 0.1 One task, not a menu

The task is to implement, validate, pilot, activate, and hand off the complete parallel experiment workflow in this document. The numbered phases in Section 15 are one continuous task. They are not independent suggestions.

The executor must start at Phase 0, or at the first incomplete phase recorded in the execution ledger, and continue in order until one of two terminal states:

```text
COMPLETE   every required phase passed and the final acceptance auditor passed
HARD_STOP a listed hard-stop condition prevents safe continuation
```

No other terminal state is allowed.

The following are intermediate milestones, not completion:

- analysis finished;
- a plan or design was written;
- one phase passed;
- tests passed for one PR;
- a PR was opened or merged;
- code was deployed;
- a smoke or production job was submitted;
- `squeue` became empty;
- a Slurm job completed;
- evidence was synchronized;
- a registry was rebuilt;
- a report was generated;
- the agent reached a convenient context, time, token, or cost boundary.

After an intermediate milestone, record it, send a short progress update if useful, and enter the next required phase. Do not send a final handoff.

### 0.2 Definition of done

The implementation is complete only when all of these are true:

1. Phases 0–13 in Section 15 have status `PASSED` in the execution ledger.
2. All tracked implementation PRs required by the runbook are merged through normal repository rules when the active grant permits merging. If the grant does not permit merging, the task cannot be declared complete; it must stop at the explicit authority hard stop and request the missing authority.
3. The isolated-lane local dry-run pilot passes for three lanes, including one stacked lane.
4. One real tracked MN5 smoke reaches terminal Slurm success, compact evidence is local, hashes and headline metrics are verified locally, and the attempt reaches `REPORTABLE`.
5. `AGENTS.md`, the affected `.agents/skills/*/SKILL.md` files, `docs/DEVICES.md`, and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` describe the implemented workflow rather than the old mutable-checkout workflow.
6. Targeted tests and the full local suite pass in `llmdep4090`.
7. The final implementation auditor exits zero and writes a deterministic audit artifact.
8. The final journal entry and handoff contain real branches, full SHAs, PRs, deployment IDs, attempt IDs, job IDs, validation commands, artifacts, remaining limitations, and the exact terminal audit path.

An agent must not write “done,” “completed,” “ready,” or equivalent unless all eight conditions pass.

### 0.3 Autonomy is explicit, not assumed

This runbook is not itself an autonomy grant. The user must send the grant in Section 19 in chat. Before the first mutation, append the exact grant, scope, budgets, expiry, and exclusions to the current Europe/Istanbul agent journal. That journal entry is the first mutation.

Under an active grant matching Section 19:

- continue through normal phases without waiting for user review;
- commit, push, open, update, and normally merge the executor's own in-scope PRs after required checks pass;
- use stacked branches rather than waiting for unrelated `main` integration;
- perform reviewed non-destructive lane-owned MN5 transfers;
- submit, monitor, cancel, diagnose, and perform bounded retries for the runbook's smoke jobs;
- synchronize compact evidence and complete local reporting;
- make the smallest evidence-driven in-scope fix when a phase exposes a defect;
- create a new PR, deployment, and attempt when source identity changes;
- resume after each merge, job, tool wait, context compaction, or process restart.

Do not ask for approval at routine phase boundaries. Ask only when a hard stop requires authority or a scientific/global decision outside the grant.

### 0.4 Persistent execution ledger

Create this ignored state root during Phase 0:

```text
outputs/parallel_workflow_implementation/<execution_id>/
  state.json
  final_audit.json
  inventories/
  dry_runs/
  validation/
```

`state.json` is mutable operational state, not scientific evidence. It must contain:

```json
{
  "schema_version": "audiollm.parallel_workflow_execution.v1",
  "execution_id": "<UTC>-parallel-workflow-<git8>",
  "runbook_path": "docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md",
  "runbook_sha256_at_start": "<sha256>",
  "grant_journal_path": "docs/agent-journal/YYYY-MM-DD.md",
  "status": "ACTIVE",
  "current_phase": 0,
  "phases": {
    "0": {"status": "IN_PROGRESS", "evidence": [], "next_action": "..."},
    "1": {"status": "PENDING", "evidence": [], "next_action": "..."}
  },
  "branches": [],
  "prs": [],
  "deployments": [],
  "attempts": [],
  "jobs": [],
  "hard_stop": null,
  "updated_at_utc": "<UTC>"
}
```

Phase 0 implements `tools/parallel_workflow_state.py` with these commands:

```text
init --runbook <path> --execution-id <id> --output <state.json>
show --state <state.json>
enter --state <state.json> --phase <n> --next-action <text>
record --state <state.json> --phase <n> --evidence <path-or-id>
pass --state <state.json> --phase <n> --next-phase <n>
hard-stop --state <state.json> --phase <n> --reason <text> --evidence <path>
complete --state <state.json> --audit <final_audit.json>
```

Updates must be atomic. The tool must reject:

- skipping a phase;
- marking a phase passed without its required evidence keys;
- entering a later phase while the current phase is not passed;
- marking the execution complete while any phase is not passed;
- changing `execution_id` or the starting runbook hash;
- clearing a hard stop without an explicit resume record tied to new user authority or changed external evidence.

Until the state tool is implemented and tested in Phase 0, bootstrap the initial JSON with `apply_patch`. After that, use only the state tool.

### 0.5 Resume protocol

At the beginning of every agent turn, after context compaction, or after any interruption:

1. Read `AGENTS.md` and this runbook.
2. Locate the active state file under `outputs/parallel_workflow_implementation/`.
3. Run `python tools/parallel_workflow_state.py show --state <state.json>` once the tool exists.
4. Verify the recorded branch, worktree, open PR, deployment, attempts, and jobs against real state.
5. Continue the recorded `next_action` for the first non-passed phase.
6. Never restart a passed phase unless its evidence was invalidated. If invalidated, record the reason and return to the earliest affected phase through the state tool.
7. Never resubmit a job merely because conversational context was lost. Resume from recorded job IDs.

If the hosting system forces the agent to end a turn before completion, the response must say `INCOMPLETE`, name the current phase, state file, evidence, and exact next action. It must not claim task completion. On automatic continuation, resume immediately.

### 0.6 Phase execution rule

Every phase in Section 15 has six fields:

```text
ENTRY       facts that must already be true
ACTIONS     exact in-scope work
VALIDATE    commands and behavioral checks
EVIDENCE    items that must be recorded in state.json
EXIT        conditions required to mark the phase PASSED
NEXT        the only normal next phase
```

For every phase:

1. Verify `ENTRY`.
2. Mark the phase `IN_PROGRESS` with the first `next_action`.
3. Perform only the listed work and the smallest fixes required by observed failures.
4. Run every required validation command.
5. Record real evidence paths and IDs.
6. Mark the phase `PASSED` only when every `EXIT` condition is true.
7. Immediately enter `NEXT`.

Do not reopen fixed scientific or architectural decisions. Do not add “nice to have” work. Do not silently skip a command because it looks expensive. If a required check cannot run, the phase is not passed; follow its failure or hard-stop rule.

### 0.7 Hard stops

Only these conditions allow the executor to stop before Phase 13:

- the active grant is missing, expired, revoked, or does not authorize a required mutation;
- continuation requires deletion, `--delete`, evidence overwrite, destructive Git cleanup, or rollback mutation not explicitly granted;
- continuation requires changing shared datasets, models, environments, packages, permissions, QoS, account policy, or other shared infrastructure;
- a protected path would need to be changed;
- the implementation would require expanding the fixed scientific or resource scope;
- GitHub requires admin/bypass merge or has unresolved requested changes that cannot be fixed in scope;
- rollback inventory or archive verification fails and the next action could risk the captured state;
- the real MN5 smoke cannot be made safe without an excluded action or after the same external blocker persists through the bounded retry allowed for that failure;
- privacy, leakage, provenance, or evaluation qualification would need to be weakened;
- the same external blocker persists after the bounded recovery allowed by the grant and no safe in-scope alternative remains.

Tests failing, code defects, merge conflicts inside the task, transient MN5 failures, a PR needing another in-scope fix, a long scheduler wait, or a poor-but-valid smoke metric are not hard stops. Diagnose and continue.

When a hard stop occurs:

1. preserve all evidence;
2. update `state.json` to `HARD_STOP` with exact evidence and the smallest needed user decision;
3. append a journal entry;
4. cancel only unsafe or no-longer-valid task-owned jobs when authorized;
5. send a final blocker handoff. Do not call the implementation complete.

### 0.8 Communication rule

During execution:

- send concise progress updates at phase boundaries and during long monitoring;
- call intermediate outcomes “Phase N passed,” never “task completed”;
- keep monitoring active jobs; do not hand them back merely because they take time;
- never use cost saving, brevity, or token conservation as a reason to omit a required phase;
- reserve the final handoff for `COMPLETE` or `HARD_STOP` only.

---

## 1. Goals and non-goals

The workflow must provide:

- one isolated local branch and worktree per code-changing experiment;
- one immutable, content-addressed MN5 source deployment per deployed commit;
- separate writable runtime and output paths per experiment lane;
- complete attempt, job, artifact, evaluation, and resubmission provenance;
- enough task-scoped authority for an agent to work overnight without waiting at routine steps;
- stacked branches when follow-up work depends on an unmerged experiment;
- result-based handling of competing experiments;
- controlled integration of complementary experiments;
- explicit stops for destructive, global, ambiguous, or out-of-scope actions.

The workflow must not:

- make `main` a prerequisite for continued experiment work;
- let one lane overwrite another lane's source, contexts, logs, manifests, splits, or checkpoints;
- treat a PR as a mandatory pause;
- merge every successful experiment automatically;
- duplicate datasets, base models, or shared environments;
- invent a second mutable source of truth for tracking data;
- call submission or synchronization “completion.”

---

## 2. Current state and main risks

### Local

Observed on 2026-08-20:

- Main worktree: `/home/emre/Projects/AudioLLM/LLM-Depression`, branch `main`, commit `2d995f4cb2b17dad6921f286a0f7b10fbb257c8f`.
- The main worktree is clean except for the unrelated untracked `skills-lock.json`.
- Thirteen additional worktrees exist as siblings or under `/tmp`; their naming is ad hoc.
- New experiment worktrees do not have a single managed root.
- No enforced mapping exists between an agent, branch, worktree, experiment, and deployment.

Protected paths:

```text
/home/emre/Projects/AudioLLM/Teacher-System
/home/emre/Projects/AudioLLM/LLM-Depression-teacher
```

New tooling must never edit, clean, move, or remove these paths. Worktree inventory may show the second path because Git already knows about it, but helpers must exclude it from managed-lane discovery and cleanup.

### MN5

Observed by read-only inspection on 2026-08-20:

- The permanent checkout is `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression`.
- Its Git HEAD and working tree do not match the local source snapshot. It is a long-lived mutable rsync target.
- Current Slurm scripts contain a fixed `#SBATCH --chdir` pointing to the permanent checkout, then `cd "$PROJECT_ROOT"` at runtime.
- Generated manifests, splits, Slurm logs, contexts, checkpoints, metrics, and predictions are writable runtime data.
- Current jobs can read source from a directory that a later rsync changes.

### Risks this plan must remove

| Risk | Required control |
|---|---|
| Source changes while a job is running | Unique deployment path; never redeploy to an existing deployment ID |
| Checkpoint collision | Unique campaign/run root plus run-name collision check |
| Agent edits the wrong worktree | `.agent-pin.json` plus path/branch verification before mutation |
| Dirty production source | Fail closed before production deployment |
| Evaluation points at a different run root | Resolve one common override set and use it for train, evaluation, sidecars, and checkpoint paths |
| Read-only source blocks runtime writes | Separate immutable code from writable runtime and output roots |
| Compact evidence lost during sync | Explicit include rules for evaluation evidence before checkpoint exclusions |
| Registry misclassifies modality | Preserve the supported campaign/modality/dataset/run/fold layout |
| Overnight agent pauses at every routine step | Task-scoped lane autonomy grant with a fixed experiment envelope |
| Autonomous agent changes shared state | Separate lane authority from integration/global authority |

---

## 3. Target layout

### Local

```text
/home/emre/Projects/AudioLLM/LLM-Depression/     main worktree; integration boundary

~/worktrees/
  LLM-Depression-exp-rotary/                     agent/exp-rotary
  LLM-Depression-exp-balanced/                   agent/exp-balanced
  LLM-Depression-feat-gemma/                     agent/feat-gemma

Each managed worktree:
  .agent-pin.json                                 ignored local control file
  outputs/                                        local lane evidence/registry input
  output_model/                                   locally synced compact evidence
  logs/                                           locally synced lane logs
```

Existing sibling worktrees stay where they are and age out. Do not move or delete them as part of this migration.

The implementation must add `.agent-pin.json` to `.gitignore` before creating managed worktrees. This keeps the pin local without making every worktree dirty.

### MN5

```text
/gpfs/projects/etur92/ozu647717/AudioLLM/
  Datasets/                                       shared, read-only to lanes
  models/                                         shared, read-only to lanes
  venvs/                                          shared; no lane package changes

  deployments/
    <deployment_id>/
      code/                                       source snapshot; never overwritten
      deployment.json                             immutable source/deployment identity

  experiment_runtime/
    <experiment_id>/
      contexts/                                   per-attempt/fold context JSON
      manifests/                                  generated or explicitly copied inputs
      splits/                                     generated split evidence
      logs/slurm_train/
      logs/slurm_eval/
      wandb/                                      optional offline files

  LLM-Depression/
    output_model/
      <campaign>/<modality>/<dataset>/<run_name>/fold_<n>/
```

The permanent checkout remains the result root and administrative fallback. New jobs do not execute its source.

The output layout deliberately keeps the supported discovery shape:

```text
output_model/<campaign>/<modality>/<dataset>/<run_name>/fold_<n>
```

Do not shorten this to `output_model/<experiment>/<dataset>/<run>`. The current importer derives modality and dataset from path position.

### Writable-path contract

Every submission resolves and passes the same absolute overrides to training and evaluation:

```text
output_dirs.manifest_dir = <runtime_root>/manifests/<dataset>
output_dirs.split_dir    = <runtime_root>/splits/<dataset>
output_dirs.run_root     = <permanent_project>/output_model/<campaign>/<modality>/<dataset>
LOG_ROOT                 = <runtime_root>/logs/<job_type>/<dataset>
EXPERIMENT_CONTEXT       = <runtime_root>/contexts/<attempt_id>/fold_<n>/context.json
```

The source deployment itself must not receive runtime writes. Deployment immutability is enforced by refusing to reuse an existing deployment path and by verifying its manifest hash before submission. Changing remote permissions is not required. If a later implementation adds `chmod`, the lane grant must explicitly authorize permission changes within that isolated deployment.

---

## 4. Experiment identity and tracking

Use the existing hierarchy:

```text
experiment group
  -> logical run
    -> attempt
      -> fold
        -> jobs
          -> evaluations, metrics, and artifacts
```

Use existing sidecars beside `run_config.yaml`:

- `metadata.json`: attempt, source, config, seed, manifest, and split identity;
- `status.json`: lifecycle state and transition history;
- `jobs.jsonl`: append-only job events and resubmission chain;
- `artifacts.json`: hashed artifacts and location flags;
- `evaluations.json`: idempotent, qualified evaluation records.

Do not create another mutable “canonical experiment manifest” containing job IDs, checkpoint paths, metrics, and lifecycle state. Those fields already have authoritative homes.

Add one immutable deployment record at `deployments/<deployment_id>/deployment.json`:

```yaml
schema_version: audiollm.deployment.v1
deployment_id: exp-rotary-20260820-a1b2c3d4-9f8e7d6c
experiment_id: exp-rotary-20260820
git_commit: <40-char SHA>
git_branch_at_deploy: agent/exp-rotary
git_dirty: false
source_manifest_sha256: <sha256>
uncommitted_patch_sha256: <empty-patch sha256 for clean production source>
deployed_code_path: /gpfs/.../deployments/<deployment_id>/code
created_at_utc: <UTC timestamp>
```

`metadata.json.source.deployed_source_sha256` must be copied from this verified deployment record when the attempt is created. A registry rebuild cannot invent or backfill a missing deployment hash. Evidence repair must use verified source evidence and the official evidence tools.

### Evaluation qualification

Every production evaluation must record all qualifiers:

```text
dataset
split and split protocol
checkpoint role
backend = original_teacher_forced for harmonized headline evaluation
evaluation_view = exact named view for that recipe
aggregation = subject_level
namespace = headline/binary_strict
```

The base Qwen DAIC harmonized config currently does not record `evaluation.evaluation_view`. The implementation must add or pass the exact view to both train and evaluation. Submission must fail if a production context would create a null evaluation view.

---

## 5. Local branch and worktree strategy

### Tier 0 — CLI-only experiment

- No tracked source or config change.
- Use a committed SHA plus explicit `--set` overrides.
- Stay on `main` only if the main worktree is not being used as an agent editing workspace.
- Record the full override list in the experiment context.

Creating or editing a tracked YAML is not Tier 0. It requires a branch and PR like any other reproducibility change.

### Tier 1 — competing or short code experiment

```text
branch:   agent/exp-<slug>
worktree: ~/worktrees/LLM-Depression-exp-<slug>
merge:    squash if selected for integration
```

Keep losing branches and evidence until the comparison is complete and reporting references are stable. Do not auto-merge all successful lanes.

### Tier 2 — complementary or long-lived feature

```text
branch:   agent/feat-<slug>
worktree: ~/worktrees/LLM-Depression-feat-<slug>
merge:    merge commit when history materially helps review or bisecting
```

### Stacked/dependent branches

A follow-up experiment does not wait for its parent PR to merge:

```text
agent/exp-base
  -> agent/exp-base-pooling
    -> agent/exp-base-pooling-calibration
```

Each child records:

- parent branch;
- parent commit SHA used as its base;
- its own full commit SHA;
- dependency PR when one exists;
- deployment ID and attempt IDs produced from that exact source.

Do not rewrite a deployed commit's history. If the parent changes, create a new commit or new dependent lane, deploy a new snapshot, and create a new attempt. Never reuse the old attempt identity.

---

## 6. Agent pinning

`tools/exp.py create` will write an ignored `.agent-pin.json`:

```json
{
  "schema_version": "audiollm.agent_pin.v1",
  "experiment_id": "exp-rotary-20260820",
  "worktree": "/home/emre/worktrees/LLM-Depression-exp-rotary",
  "branch": "agent/exp-rotary",
  "allowed_paths": ["/home/emre/worktrees/LLM-Depression-exp-rotary"],
  "protected_paths": [
    "/home/emre/Projects/AudioLLM/Teacher-System",
    "/home/emre/Projects/AudioLLM/LLM-Depression-teacher"
  ]
}
```

Before any edit, commit, deploy, or submission, `tools/check_worktree_pin.py` must verify:

- real current directory is inside the pinned worktree;
- Git top-level equals the pinned worktree;
- checked-out branch equals the pin;
- the experiment definition matches the pin;
- the target path is inside `allowed_paths` and outside every protected path.

The agent journal entry must record the worktree, branch, full commit SHA, experiment ID, deployment ID, and attempt IDs when available.

---

## 7. High-autonomy experiment lanes

### Principle

Maximize autonomy inside isolated lanes. Control changes to shared state.

An agent may continue overnight without waiting at every routine step only when the user has issued a task-specific autonomy grant for that named lane. Silence, absence, this plan, or `AGENTS.md` alone is not a grant.

The grant must be journaled before the first mutation and must define:

- named experiment lane and scientific question;
- allowed branch/worktree/deployment/runtime/output prefixes;
- datasets, modalities, folds, seeds, configs, and allowed overrides;
- expected job shapes, concurrency, completion conditions, and any user-chosen time or storage boundary; do not invent a PR or GPU ceiling;
- completion condition and hard stops;
- whether isolated deployment creation and deployment-local immutability controls are authorized;
- expiry on completion, source-identity change outside the lane, hard stop, revocation, or budget exhaustion;
- exclusions.

### Lane actions allowed under a suitable grant

Within the fixed experiment envelope, the lane agent may:

- edit its own worktree;
- run local validation;
- commit and push its own branch;
- open and update its own PR without pausing;
- create a new isolated MN5 deployment path after a reviewed dry run;
- submit and cancel its own Slurm jobs;
- monitor through terminal `sacct` evidence;
- inspect its own logs and artifacts;
- diagnose failures and make an evidence-driven code/config fix;
- create a new deployment and new attempt for changed source;
- perform one bounded retry per transient-infrastructure failure, using a new attempt identity;
- retrieve compact evidence and logs without `--delete`;
- validate locally, generate qualified reports, and continue to the next pre-authorized iteration;
- create stacked branches for follow-up experiments inside the granted envelope.

These actions do not require `main` to change first.

### Actions that remain hard stops unless separately and explicitly granted

- destructive cleanup or `--delete`;
- overwriting an existing deployment, run, checkpoint, context, or evidence file;
- changing shared datasets, base models, environments, packages, QoS, permissions, or infrastructure;
- expanding to an unlisted dataset, model, modality, scientific question, or resource envelope;
- weakening leakage, privacy, provenance, audit, or reporting controls;
- force-pushing deployed history;
- bypassing branch protection or checks;
- resolving ambiguous scientific or semantic integration by guesswork;
- modifying the rollback capture or deleting its artifacts.

### Example grant template

```text
I grant full autonomy for experiment lane <experiment_id>.
Scope: branch <branch>, worktree <path>, deployments/<prefix>,
experiment_runtime/<experiment_id>, output_model/<campaign>/<modality>/<dataset>,
configs <list>, overrides <list>, folds/seeds <list>.
Budget: no artificial PR or GPU ceiling; use the script-defined job shapes and
scheduler/account/QoS availability. Retries remain evidence-driven and bounded
as specified below. Record any user-chosen time or storage boundary.
The agent may edit, validate, commit, push, open/update PRs, create new isolated
deployments, submit/cancel/monitor Slurm jobs, make evidence-driven fixes,
perform bounded transient retries with new attempt IDs, retrieve compact
evidence, validate locally, report, and create stacked dependent branches.
The grant expires on the first passing completion condition, budget exhaustion,
a hard stop, source-scope change, my revocation, or <date/time>.
Excluded: deletion, --delete, evidence overwrite, shared infrastructure or
environment changes, package installs, silent scientific expansion, force-push,
protection bypass, ambiguous integration, and rollback-state changes.
```

The concrete grant may be narrower. Maximum counts are ceilings, not targets. Stop at the first valid completion condition.

---

## 8. Experiment agents and integration/orchestrator agent

### Experiment agent

Owns one lane:

- branch and worktree;
- code and config changes inside the lane;
- local validation;
- deployment and attempt creation;
- Slurm submission and monitoring;
- failure diagnosis and bounded iteration;
- compact-evidence retrieval and local validation;
- lane report and handoff.

It does not decide ambiguous changes to `main` or another lane.

### Integration/orchestrator agent

Coordinates shared state without editing active experiment worktrees. Its responsibilities may include:

- allocate unique lane, campaign, deployment, runtime, and run-name prefixes;
- maintain the dependency graph for stacked branches;
- check total active scheduler demand and output collisions;
- compare only qualified, locally verified attempts from the declared experiment group;
- identify Git conflicts and semantic conflicts between complementary lanes;
- build a dedicated integration branch/worktree;
- run integration validation;
- open or update integration PRs;
- merge only when a task-specific integration grant permits it and repository checks pass.

The orchestrator must not auto-merge competing experiments merely because they completed. It may select a winner automatically only when the plan fixed the comparison set, metric, view, aggregation, seeds/folds, tie rule, minimum evidence, and stop conditions before execution. Otherwise it reports the qualified comparison and stops for a decision.

### PRs are coordination objects, not experiment gates

- A lane opens a draft or normal PR for reviewability and provenance.
- The lane continues to deploy and iterate from its exact branch commit.
- A dependent experiment branches from the required unmerged commit and records the dependency.
- Merging to `main` is an integration event, not permission to continue experimenting.

---

## 9. MN5 deployment and submission

### Deploy a clean source snapshot

Proposed `exp deploy` flow:

1. Verify the worktree pin.
2. Require clean committed source for production.
3. Capture `.provenance` locally.
4. Compute the deployment ID from the lane, full source identity, timestamp, and collision-resistant suffix.
5. Dry-run rsync to a path that must not already exist.
6. Transfer source through `transfer1` without `--delete`.
7. Transfer required contexts or generated inputs explicitly; `.gitignore` filtering will not copy them automatically.
8. Verify source manifest and selected file hashes remotely.
9. Write `deployment.json` once. Refuse later mutation of the deployment path.

Do not use the permanent MN5 checkout as the deployment source for new jobs.

### Resolve one common resolved override set

The current `submit_train_and_eval.sh` calculates checkpoint paths from the YAML before applying `EXTRA_TRAIN_ARGS`. Therefore the implementation must change the wrapper before run-root isolation is used.

Required wrapper behavior:

- accept one common resolved override array for train and evaluation;
- load the YAML with those overrides before calculating `FOLD_DIR`;
- pass identical path, split-protocol, prediction-mode, and evaluation-view overrides to training and evaluation;
- derive `CHECKPOINT_DIR`, evaluation output, submission sidecars, and displayed paths from the resolved config;
- fail if train/evaluation resolved run roots differ;
- pass `sbatch --chdir="$PROJECT_ROOT"` so the command-line value overrides the static script directive;
- export the isolated `LOG_ROOT` and `EXPERIMENT_CONTEXT`;
- refuse an existing incompatible run directory.

### Submission preflight

Before submission, record and verify:

- deployment ID and remote source hash;
- experiment group, logical run, attempt, fold, and seed;
- resolved config hash and full overrides;
- manifest and split paths/hashes;
- exact evaluation view, backend, aggregation, and namespace;
- unique run root and run name;
- expected jobs, dependencies, GPU shape, wall time, and storage;
- available scheduler endpoint and environment;
- active jobs that share the account or experiment campaign.

There is no repository-wide 64-H100 ceiling. Submit as many isolated lanes as the scheduler, account, and QoS grant, while preserving each job's configured GPU shape and any narrower lane budget.

Training and evaluation must run only through Slurm compute allocations, never directly on `transfer1` or a scheduler login.

---

## 10. Monitoring, failure handling, and lifecycle

The lane agent remains responsible until every job is terminal and compact evidence is locally validated.

For every job:

1. Record the returned Slurm ID in `jobs.jsonl`.
2. Poll `squeue` while active.
3. Reconcile terminal state with `sacct`; an empty queue is not success.
4. Inspect logs and expected artifacts.
5. Append terminal job evidence.
6. Move lifecycle state only through valid official transitions.

Lifecycle:

```text
PLANNED -> DEPLOYED -> SUBMITTED -> RUNNING -> COMPLETED_ON_MN5
        -> SYNCED_LOCALLY -> LOCALLY_VALIDATED -> REPORTABLE
```

`exp finish` must not jump from `RUNNING` to `REPORTABLE`. It may advance each state only after its gate passes.

Failure rules:

- record exact job state, exit code, node, elapsed time, logs, and partial artifacts;
- preserve failed and cancelled attempts;
- use a new attempt identity for every rerun;
- link retries with `supersedes_attempt_id` where appropriate;
- retry a transient infrastructure failure at most once per failed job under a lane grant;
- for code/config fixes, validate, commit, deploy a new source snapshot, and create a new attempt;
- never overwrite the failed attempt or reuse its output directory.

---

## 11. Compact-evidence collection and local validation

Checkpoint exclusions must not hide evaluation evidence. Collection must explicitly include compact evidence before excluding adapter contents.

Illustrative rsync filter order:

```bash
rsync -avhn --itemize-changes --prune-empty-dirs \
  --include='*/' \
  --include='run_config.yaml' \
  --include='metadata.json' \
  --include='status.json' \
  --include='jobs.jsonl' \
  --include='artifacts.json' \
  --include='evaluations.json' \
  --include='logs/*.json' \
  --include='logs/*.jsonl' \
  --include='best_model/standalone_eval/***' \
  --include='eval/***' \
  --include='final_summary.json' \
  --exclude='best_model/***' \
  --exclude='last_model/***' \
  --exclude='*' \
  <remote-fold>/ <local-fold>/
```

The implemented collector must have tests proving that it transfers:

- strict headline metrics;
- subject-level predictions;
- evaluation configuration/view;
- all tracking sidecars;
- `run_config.yaml`;
- audit JSONs;
- deployment provenance reference;
- relevant Slurm logs from the separate runtime root.

After sync:

1. transition to `SYNCED_LOCALLY` through the official API;
2. verify artifact hashes;
3. recompute or verify headline metrics locally;
4. validate evaluation qualifiers and checkpoint role;
5. transition to `LOCALLY_VALIDATED`;
6. transition to `REPORTABLE` only when all qualification gates pass;
7. rebuild the local registry from evidence;
8. generate deterministic reports.

Use `provenance-reporting` before presenting any metric.

---

## 12. Comparing experiments and integrating winners

### Competing experiments

Define the comparison contract before submission:

- exact experiment group and allowed attempt IDs;
- same dataset, split, seed/fold set, scientific recipe, and resource-relevant settings;
- metric name and namespace;
- backend, evaluation view, and aggregation;
- fold-mean or pooled convention;
- minimum completion/qualification requirements;
- tie rule and uncertainty rule.

Do not use the current global `tools/exp.py best` query by itself because it is not group-scoped and may return an unrelated historical run. Generate a group report from an explicit attempt list or implement a group-scoped comparison command.

Only the selected winner becomes an integration candidate. Losing branches and evidence remain available until reports and provenance references are stable.

### Complementary experiments

Use a dedicated integration branch/worktree. Check two conflict classes:

1. **Git conflicts:** overlapping files and lines.
2. **Semantic conflicts:** incompatible defaults, duplicated strategy registries, changed data assumptions, different evaluation semantics, or interacting resource behavior even when Git merges cleanly.

Prefer configurable strategies when alternatives should coexist. Run targeted tests for each component, cross-feature tests, and the full suite when shared runtime behavior changes.

If the intended combined behavior is ambiguous, stop for a global decision. An integration agent must not guess which scientific semantics should win.

---

## 13. Proposed helper tooling

These commands are proposed and are not currently implemented:

```text
tools/exp.py
  create <slug> --tier {0,1,2} [--from <branch-or-sha>]
  deploy <slug> --dry-run|--execute
  submit <slug> --config <yaml> --fold <n>
  status <slug>
  collect <slug> --dry-run|--execute
  validate <slug>
  compare --group <group-id> --attempts <csv> <full qualifiers>
  finish <slug>
  cleanup <slug> --plan
```

Implementation rules:

- one task card and focused PR per subcommand or coherent prerequisite;
- `create` writes the ignored pin and tracked experiment definition;
- `deploy` refuses dirty production source and existing deployment targets;
- `submit` uses one resolved common override set;
- `status` combines sidecars, `squeue`, and `sacct` without inventing transitions;
- `collect` dry-runs first and preserves compact evaluation evidence;
- `validate` performs local hash and metric checks;
- `compare` is group/attempt scoped and requires full qualifiers;
- `finish` enforces every lifecycle gate;
- `cleanup` only produces an exact deletion plan; execution remains separately confirmed and destructive.

Do not describe these commands as executable until their CLI help and tests exist.

---

## 14. Safe rollback package

“Rollback” has two distinct meanings and must not be described as one exact backdoor.

### A. Local tracked-source rollback

An annotated tag such as `pre-parallel-workflow-20260820` can restore tracked source on `main` after explicit review and approval. It does not restore:

- ignored files;
- untracked files;
- other branches;
- worktree registrations;
- local outputs or environments;
- the dirty MN5 checkout.

Never use `git clean -fdx` as part of this rollback. It would remove ignored repository assets including `docs/`, `.agents/`, `.provenance/`, `.deps/`, and other local state.

Before any tracked-source reset:

1. inspect and save `git status`, diff, branch, and worktree inventory;
2. confirm the exact target tag/SHA;
3. preserve unrelated user changes;
4. obtain explicit approval for the destructive reset;
5. reset tracked source only;
6. handle workflow-created worktrees and branches from an explicit inventory, one path at a time.

### B. MN5 pre-workflow-state recovery

Before changing the current dirty MN5 checkout:

1. create a reviewed archive of the complete source checkout while explicitly excluding large runtime trees such as `output_model/`, `outputs/`, and `logs/`;
2. include top-level tracked and untracked source files, `.git`, `.provenance`, configs, scripts, and source code;
3. record archive SHA-256, file inventory, permissions, MN5 HEAD, and dirty diff;
4. copy the compact inventory and archive hash locally;
5. inspect the archive before any reset or cleanup.

Restoring the original dirty MN5 state means extracting the archive to a staging directory, comparing it, and selectively restoring the captured paths with explicit approval. Resetting MN5 to the local tag restores a different clean baseline and is not equivalent.

No rollback step may automatically delete `deployments/`, `experiment_runtime/`, outputs, branches, or worktrees. Cleanup always uses an explicit allowlist of workflow-created paths and separate approval.

---

## 15. Detailed implementation phases

Use one focused PR per phase unless a phase explicitly says it is evidence-only. If a phase exposes a prerequisite defect, fix the smallest coherent cause in an additional in-scope PR, validate it, merge it under the active grant, record it, and return to the same phase. Do not consume a later phase to hide unfinished work.

### Phase 0 — authority, baseline, state machine, and final-auditor skeleton

**ENTRY**

- The user has sent the Section 19 grant in chat.
- The exact grant has been appended to the current Istanbul-date journal.
- The executor has read every mandatory file and skill listed below.
- No task mutation other than the grant journal entry has occurred.

Mandatory reading, in order:

1. `AGENTS.md` supplied in the task context;
2. `docs/DEVICES.md`;
3. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`;
4. `configs/README.md`;
5. `docs/SIGNAL_FLOW.md`;
6. Sections 25–28 of `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md`;
7. this runbook completely;
8. current `tools/exp.py`, experiment-tracking lifecycle/sidecar code, submission wrappers, Slurm workers, collection script, tests, and `.gitignore`.

Required skills when their work begins:

- `plain-english`;
- `agent-journal`;
- `experiment-tracking`;
- `local-validation`;
- `git-pr`;
- `mn5-cluster-ops`;
- `provenance-reporting` before any metric is shown.

**ACTIONS**

1. Record local baseline:

   ```bash
   git status --short
   git branch --show-current
   git rev-parse HEAD
   git rev-parse origin/main
   git worktree list --porcelain
   python tools/exp.py --help
   ```

2. Preserve `skills-lock.json` and every unrelated change. Do not stage, move, or edit them.
3. Create a clean implementation worktree at:

   ```text
   ~/worktrees/LLM-Depression-feat-parallel-workflow
   ```

   with branch:

   ```text
   agent/feat-parallel-experiment-workflow
   ```

4. Bootstrap `state.json` under `outputs/parallel_workflow_implementation/<execution_id>/` using `apply_patch`.
5. Implement `tools/parallel_workflow_state.py` and `tests/test_parallel_workflow_state.py`.
6. Implement the skeleton `tools/audit_parallel_workflow_implementation.py`. At this phase it must validate the ledger schema, phase ordering, referenced evidence existence, and terminal-completion prohibition. Later phases extend its behavioral checks.
7. Add CLI help, atomic-write behavior, explicit error messages, and tests for interruption/resume behavior.
8. Run validation, commit, push, open a normal PR, and merge it under the active grant after checks pass. Do not delete the branch.
9. Update the bootstrapped ledger with the merged PR/head/merge SHAs and switch to using the state tool.

**VALIDATE**

```bash
conda activate llmdep4090
python -m py_compile tools/parallel_workflow_state.py tools/audit_parallel_workflow_implementation.py
python -m pytest tests/test_parallel_workflow_state.py -q
python -m pytest tests/test_experiment_tracking_lifecycle.py -q
python tools/parallel_workflow_state.py show --state <state.json>
python tools/audit_parallel_workflow_implementation.py --state <state.json> --allow-incomplete
```

Negative tests must prove that the tool rejects skipped phases, missing evidence, completion with pending phases, execution-ID changes, and silent hard-stop clearing.

**EVIDENCE**

- grant journal path;
- baseline Git SHA/branch/status inventory;
- implementation worktree and branch;
- state file path and SHA-256;
- test output path;
- PR URL, validated head SHA, merge SHA;
- incomplete-auditor output showing Phase 0 only.

**EXIT**

- State tooling and auditor skeleton are merged and executable.
- Phase 0 evidence is recorded.
- The auditor refuses terminal completion.
- Unrelated work is unchanged.

**NEXT:** Immediately enter Phase 1.

### Phase 1 — non-destructive rollback capture

**ENTRY**

- Phase 0 is `PASSED`.
- The active grant authorizes creating the rollback tag/archive and read-only MN5 inspection.
- No reset, clean, removal, or overwrite is planned.

**ACTIONS**

1. Re-read the rollback rules in Section 14 and the current device/runbook safety boundary.
2. Create local inventories under `<state-root>/inventories/`:

   - `local_status.txt`;
   - `local_diff_stat.txt`;
   - `local_worktrees.txt`;
   - `local_branches.txt`;
   - `local_head.txt`;
   - `source_manifest_sha256.txt` when `.provenance/source_manifest.json` exists.

3. Verify the intended annotated tag target. Create and push `pre-parallel-workflow-20260820` only if it does not already exist and the target is the reviewed baseline SHA. If it exists, verify it; never move it.
4. Through `transfer1`, collect a read-only MN5 inventory of HEAD, status, diff stat, top-level source paths, `.provenance`, and disk space.
5. Create a source backup archive at a new unique path under:

   ```text
   /gpfs/projects/etur92/ozu647717/AudioLLM/.backdoor/
   ```

   Include the source checkout and Git/provenance state. Explicitly exclude `output_model/`, `outputs/`, `logs/`, caches, datasets, models, environments, and any unrelated large runtime tree.
6. Generate the archive file list before creation and review it for secrets, subject data, and unexpected large files. The archive remains on GPFS; synchronize only its compact inventory and SHA-256 locally.
7. Verify the archive with a listing and checksum. Do not extract it over the project.
8. Write `docs/BACKDOOR_SNAPSHOT_20260820.md` with facts, limitations, exact archive path/hash, and targeted recovery instructions. It must explicitly say the local tag and dirty MN5 archive restore different states.
9. Do not reset or clean the MN5 checkout. The new deployment workflow does not require cleaning it.

**VALIDATE**

- Local tag resolves to the recorded full SHA and matches the remote tag when pushed.
- Every inventory file is non-empty where expected.
- Remote archive exists, lists successfully, and matches the recorded SHA-256.
- The compact local record contains no credentials, subject data, or raw transcripts.
- `rg` finds no executable `git clean -fdx`, broad `rm -rf`, wildcard deployment deletion, or automatic restore command in the rollback docs.

**EVIDENCE**

- local tag and full SHA;
- local inventories;
- MN5 inventory path;
- remote archive path, size, SHA-256, and file-list path;
- local compact backup record;
- validation output.

**EXIT**

- Rollback evidence is verified without changing current MN5 source.
- The permanent checkout remains untouched.
- Recovery limits are documented accurately.

**NEXT:** Immediately enter Phase 2.

### Phase 2 — worktree pins and lane creation

**ENTRY**

- Phase 1 is `PASSED`.
- Rollback evidence is recorded and intact.

**ACTIONS**

1. Add `.agent-pin.json` to `.gitignore`.
2. Implement `tools/check_worktree_pin.py` with the contract in Section 6.
3. Extend `tools/exp.py` with proposed `create` behavior without breaking the existing read-only query commands.
4. `create` must support:

   - Tier 1 and Tier 2 naming;
   - `--from <branch-or-sha>` for stacked/dependent branches;
   - unique worktrees only under `~/worktrees/`;
   - tracked experiment definitions under `experiments/definitions/`;
   - ignored pins with canonical absolute paths;
   - parent branch/SHA and dependency metadata;
   - dry-run output before mutation;
   - collision refusal;
   - exclusion of the two protected paths.

5. Do not implement deletion or cleanup execution. A future cleanup command may only plan an allowlisted operation.
6. Add tests using temporary Git repositories and worktrees. Never point tests at real protected paths for mutation.
7. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
python -m py_compile tools/check_worktree_pin.py tools/exp.py
python -m pytest tests/test_experiment_cli.py tests/test_parallel_workflow_worktrees.py -q
python -m pytest tests/test_experiment_tracking_*.py -q
```

Tests must prove:

- the pin does not dirty the worktree;
- wrong CWD, wrong branch, stale experiment ID, symlink escape, `..` escape, and protected paths fail closed;
- stacked child creation records the exact parent SHA;
- an existing worktree/branch/slug collision is refused;
- existing `exp.py list|show|provenance|jobs|best` behavior remains intact.

**EVIDENCE**

- changed files;
- test outputs;
- dry-run examples for Tier 1, Tier 2, and stacked creation;
- PR/head/merge SHAs.

**EXIT**

- Managed lane creation and pin enforcement are merged.
- No real experiment lane or protected path was removed.

**NEXT:** Immediately enter Phase 3.

### Phase 3 — immutable deployment and runtime-root creation

**ENTRY**

- Phase 2 is `PASSED`.
- Pin enforcement is active for managed worktrees.

**ACTIONS**

1. Define and validate `audiollm.deployment.v1` without duplicating jobs, metrics, evaluations, or lifecycle state.
2. Implement proposed `exp deploy` with:

   - required pin check;
   - clean committed production source check;
   - explicit dirty smoke/debug mode that is never reportable;
   - `.provenance` capture;
   - collision-resistant deployment ID;
   - new-target-only rsync through `transfer1`;
   - automatic dry run and itemized review artifact;
   - no `--delete`;
   - remote source-manifest and selected-file hash verification;
   - immutable `deployment.json` written once outside `code/`;
   - new writable `experiment_runtime/<experiment_id>/` directories;
   - refusal to modify an existing deployment target;
   - no remote permission change unless the active grant explicitly names it.

3. Separate transfer planning from execution so tests can validate commands without SSH.
4. Add path validators that restrict mutations to the lane's deployment and runtime prefixes.
5. Add disk-space estimation and require the estimate in the dry-run record.
6. Add a read-only `verify-deployment` action that recomputes hashes and detects post-deploy drift.
7. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
python -m py_compile tools/exp.py src/experiment_tracking/*.py
python -m pytest tests/test_parallel_workflow_deploy.py -q
python -m pytest tests/test_experiment_tracking_*.py -q
```

Tests must prove:

- dirty production source fails;
- dirty smoke is labeled non-reportable;
- deployment IDs change when source changes;
- existing deployment targets fail before rsync;
- no generated command contains `--delete`;
- runtime roots are outside deployment code;
- deployment records contain the full SHA and source-manifest hash;
- no tracking sidecar responsibility is duplicated.

**EVIDENCE**

- deployment schema and example;
- offline dry-run command artifacts;
- negative-test output;
- PR/head/merge SHAs.

**EXIT**

- Deployment planning/execution and verification are merged.
- No real MN5 deployment has been made yet.

**NEXT:** Immediately enter Phase 4.

### Phase 4 — common path resolution and safe submission

**ENTRY**

- Phase 3 is `PASSED`.
- Deployment and runtime path contracts are implemented.

**ACTIONS**

1. Add one common resolved override representation used by planning, training, manifest building, evaluation, checkpoint discovery, and submission sidecars.
2. Update `scripts/submit_train_and_eval.sh` so it loads the config with common overrides before calculating `RUN_ROOT`, `FOLD_DIR`, checkpoint paths, or evaluation output.
3. Pass identical relevant overrides to `run_train_slurm.sh` and `run_eval_slurm.sh`.
4. Export isolated `LOG_ROOT`, `EXPERIMENT_CONTEXT`, `PROJECT_ROOT`, and resolved paths.
5. Pass `sbatch --chdir="$PROJECT_ROOT"` so the deployment path overrides the fixed worker directive. Keep the workers' explicit `cd "$PROJECT_ROOT"` check.
6. Add collision checks for run, fold, context, and output paths. Existing compatible continuation is not assumed; fail unless an explicit tested continuation mode exists.
7. Require production evaluation qualifiers. For the harmonized DAIC smoke, explicitly resolve:

   ```text
   evaluation.sample_prediction_mode=original_teacher_forced
   evaluation.evaluation_view=harmonized_all_windows_full_coverage
   evaluation.aggregation_level=subject
   ```

8. Preserve the existing DDP and evaluation resource shapes:

   ```text
   train: 1 node, 4 tasks, 4 H100s, NPROC_PER_NODE=4
   eval:  1 node, 1 task, 1 H100
   ```

   Do not add a one-GPU training mode or change the DDP shape for this implementation. The smoke is small because of epoch/subject overrides, not because the launcher changes GPU shape. There is no runbook-wide or project-wide GPU ceiling; scheduler, account, QoS, and the script-defined per-job shape govern allocation.
9. Implement proposed `exp submit --dry-run` and `--execute`. It must display the full resolved contract and expected job graph before submission.
10. Preserve shell quoting by using files/JSON or arrays rather than lossy whitespace splitting for complex overrides.
11. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
bash -n scripts/submit_train_and_eval.sh scripts/run_train_slurm.sh scripts/run_eval_slurm.sh
python -m pytest tests/test_parallel_workflow_submit.py -q
python -m pytest tests/test_experiment_tracking_lifecycle.py tests/test_experiment_workflow_cli.py -q
```

Tests must prove:

- train, eval, checkpoint, fold, context, and submission-event paths match;
- manifest/split/log writes are outside deployment code;
- common overrides with spaces and nested values survive quoting;
- missing evaluation view fails in production mode;
- dynamic `--chdir` is present;
- a run collision fails before `sbatch`;
- smoke shape preserves the existing 4-GPU DDP train plus 1-GPU eval scripts;
- default production resource shapes are unchanged.

**EVIDENCE**

- resolved-contract fixture;
- exact dry-run output for the future real smoke;
- shell and pytest outputs;
- PR/head/merge SHAs.

**EXIT**

- Common-path submission is merged and all offline submission tests pass.
- No Slurm job has been submitted yet.

**NEXT:** Immediately enter Phase 5.

### Phase 5 — lifecycle monitoring, failures, retries, and resumption

**ENTRY**

- Phase 4 is `PASSED`.
- Submission creates correct attempt and job identities.

**ACTIONS**

1. Implement proposed `exp status` and monitoring support by reusing current lifecycle APIs and relevant monitor code.
2. Reconcile `squeue`, `sacct`, job logs, dependencies, and expected artifacts. `sacct` terminal evidence is authoritative for job completion.
3. Record job events append-only. Never rewrite earlier events.
4. Implement valid lifecycle gates only:

   ```text
   PLANNED -> DEPLOYED -> SUBMITTED -> RUNNING -> COMPLETED_ON_MN5
   ```

5. A top-level job is successful only with terminal `COMPLETED`, `ExitCode=0:0`, and required artifacts.
6. Implement failure classification:

   - transient infrastructure;
   - deterministic code/config;
   - data/provenance/qualification;
   - cancelled dependency;
   - unknown, requiring diagnosis.

7. Implement bounded retry planning. A retry always gets a new attempt ID and records its predecessor. Unchanged retry is allowed only once for a demonstrated transient failure.
8. Add resume behavior that reads recorded job IDs and refuses duplicate submission after context loss.
9. Add tests for dependency cancellation, partial job graphs, repeated polling, terminal reconciliation, and source-changing fixes.
10. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
python -m py_compile scripts/monitor_experiment.py tools/exp.py tools/verify_run_evidence.py
python -m pytest tests/test_parallel_workflow_monitor.py -q
python -m pytest tests/test_experiment_tracking_lifecycle.py tests/test_experiment_tracking_modern_import.py -q
```

**EVIDENCE**

- simulated `squeue`/`sacct` fixtures and outcomes;
- retry-chain fixtures;
- resume-without-resubmit test;
- PR/head/merge SHAs.

**EXIT**

- Monitoring, failure classification, and retry identity are merged.
- The state ledger can preserve monitoring progress across agent restarts.

**NEXT:** Immediately enter Phase 6.

### Phase 6 — compact collection, local verification, and reportability

**ENTRY**

- Phase 5 is `PASSED`.
- Terminal MN5 evidence can be represented correctly.

**ACTIONS**

1. Implement proposed `exp collect` with dry-run first and the filter semantics in Section 11.
2. Collect lane runtime logs separately from permanent fold evidence.
3. Preserve:

   - `run_config.yaml`;
   - all five sidecar types;
   - metrics JSON;
   - strict headline metrics;
   - sample and subject predictions;
   - evaluation view/backend/aggregation/namespace;
   - audit JSONs;
   - deployment record/reference;
   - failure and resubmission logs.

4. Exclude adapter contents under `best_model/` and `last_model/` by default without excluding `best_model/standalone_eval/`.
5. Implement proposed `exp validate`:

   - verify hashes;
   - recompute headline metrics from local subject predictions;
   - compare remote/local counts;
   - enforce checkpoint role and evaluation qualifiers;
   - transition through `SYNCED_LOCALLY`, `LOCALLY_VALIDATED`, and `REPORTABLE` only through official APIs.

6. Make validation idempotent. Changed content under the same evaluation identity must fail.
7. Extend the implementation auditor with collection and reportability checks.
8. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
bash -n scripts/collect_experiment.sh
python -m pytest tests/test_parallel_workflow_collect.py -q
python -m pytest tests/test_experiment_tracking_qualification.py tests/test_experiment_reports.py tests/test_experiment_tracking_modern_import.py -q
```

Tests must use a fixture where metrics exist only under `best_model/standalone_eval/`; collection must retain them while adapter weights remain excluded.

**EVIDENCE**

- dry-run filter output;
- fixture collection inventory;
- local recomputation test;
- lifecycle transition test;
- PR/head/merge SHAs.

**EXIT**

- Compact collection and local validation are merged.
- A fixture attempt can reach `REPORTABLE`; ambiguous or tampered evidence cannot.

**NEXT:** Immediately enter Phase 7.

### Phase 7 — group comparison, stacked branches, and integration orchestration

**ENTRY**

- Phase 6 is `PASSED`.
- Qualified reportable attempts can be queried locally.

**ACTIONS**

1. Implement proposed group-scoped `exp compare` with an explicit attempt list and all required qualifiers.
2. Refuse attempts outside the declared group, non-reportable attempts, mixed folds/seeds/protocols, missing evaluation views, mixed aggregations, or ambiguous tie rules.
3. Record the predeclared comparison contract and deterministic selection audit.
4. Extend lane definitions with parent branch/SHA and dependent PR metadata.
5. Implement read-only dependency-graph inspection for stacked branches.
6. Implement an integration planning command that reports:

   - branch dependency order;
   - Git conflict candidates;
   - files touching shared scientific/runtime contracts;
   - required cross-feature tests;
   - whether automatic winner selection is permitted by the comparison contract.

7. Do not implement automatic merge of every successful lane.
8. Automatic winner selection is allowed only under the complete fixed contract in Section 12. Otherwise produce an ambiguity result and require a global decision.
9. Add semantic-conflict fixtures where Git merges cleanly but defaults/evaluation behavior conflict.
10. Commit, validate, push, open, and normally merge the focused PR.

**VALIDATE**

```bash
python -m pytest tests/test_parallel_workflow_compare.py tests/test_parallel_workflow_stacks.py -q
python -m pytest tests/test_experiment_reports.py tests/test_experiment_cli.py -q
```

**EVIDENCE**

- qualified group comparison fixture;
- rejected unrelated historical best fixture;
- stacked dependency graph fixture;
- semantic-conflict fixture;
- PR/head/merge SHAs.

**EXIT**

- Group comparison and integration planning are merged.
- The global unscoped `best` query is not used to choose experiment winners.

**NEXT:** Immediately enter Phase 8.

### Phase 8 — implementation-wide local validation and security audit

**ENTRY**

- Phases 0–7 are `PASSED`.
- All core workflow components are merged.

**ACTIONS**

1. Update the implementation worktree to the exact merged `origin/main` state without discarding unrelated work.
2. Run static searches for:

   - broad destructive commands;
   - `--delete`;
   - hard-coded protected paths used as mutation targets;
   - fixed permanent-checkout `PROJECT_ROOT` assumptions in new paths;
   - stale 64-H100 cap claims;
   - logging of credentials or sensitive data;
   - unqualified metric comparison;
   - direct SQLite writes from Slurm workers.

3. Run syntax checks for every changed Python and shell file.
4. Run all targeted workflow/tracking/submission tests.
5. Run the full suite from `llmdep4090`:

   ```bash
   python -m pytest tests/
   ```

6. Extend the final auditor so it checks merged source identity, required tests, prohibited patterns, CLI help, and expected schemas.
7. Fix every in-scope failure through focused PRs. Do not waive or delete a failing test.

**VALIDATE**

- All syntax checks pass.
- All targeted tests pass.
- Full suite passes, with any pre-existing/data-root skip listed exactly.
- `git status` contains no unintended tracked or staged files.
- The incomplete final auditor exits nonzero for missing pilot and smoke evidence.

**EVIDENCE**

- environment and Python version;
- changed-file inventory;
- static audit output;
- targeted and full-suite outputs;
- any correction PRs/head/merge SHAs;
- incomplete-auditor output.

**EXIT**

- Core implementation is locally clean and fully tested.
- The auditor still refuses completion because Phases 9–13 are pending.

**NEXT:** Immediately enter Phase 9.

### Phase 9 — three-lane local dry-run pilot

**ENTRY**

- Phase 8 is `PASSED`.
- No real MN5 mutation is needed for this phase.

**ACTIONS**

1. From a reviewed base SHA, create three disposable-but-preserved pilot lanes:

   ```text
   pilot-a: Tier 1 branch from main
   pilot-b: Tier 1 branch from main
   pilot-c: stacked child branch from pilot-a's exact SHA
   ```

2. Use harmless tracked fixture changes or purpose-built test fixtures. Do not modify scientific production behavior merely to create a pilot diff.
3. Verify pins prevent cross-lane edits.
4. Generate deployment dry runs for all three lanes with distinct deployment/runtime/output prefixes.
5. Generate submission dry runs with distinct attempt, run, fold, log, context, and output paths.
6. Simulate job graphs, one transient retry, one deterministic source-fix redeployment, compact collection, local validation, and group comparison.
7. Confirm pilot-c records pilot-a as its parent and does not require pilot-a to merge into `main`.
8. Run the integration planner for pilot-a and pilot-b with both a Git-conflict and semantic-conflict fixture.
9. Leave pilot worktrees/branches intact and inventory them. Cleanup is not part of this task.
10. Extend the auditor to verify all pilot evidence.

**VALIDATE**

```bash
python -m pytest tests/test_parallel_workflow_pilot.py -q
python tools/audit_parallel_workflow_implementation.py --state <state.json> --allow-incomplete
```

The auditor must pass the local-pilot checks and still refuse terminal completion because the real smoke and contract switch are pending.

**EVIDENCE**

- three lane IDs, branches, worktrees, pins, and parent SHAs;
- deployment and submission dry-run artifacts;
- collision matrix proving no shared writable path;
- retry/redeployment simulation;
- collection and comparison outputs;
- integration conflict reports;
- pilot audit output.

**EXIT**

- All three lanes pass the dry-run pilot.
- Stacked continuation works without `main` integration.
- No MN5 source or job state changed.

**NEXT:** Immediately enter Phase 10.

### Phase 10 — one real end-to-end tracked MN5 smoke

**ENTRY**

- Phase 9 is `PASSED`.
- `docs/DEVICES.md`, `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`, the `mn5-cluster-ops` skill, selected config, resolved dry run, submission wrapper, and workers have been reread completely.
- The active grant covers a new isolated deployment, runtime, smoke jobs, monitoring, cancellation, compact sync, and evidence-driven in-scope correction attempts.
- No destructive or shared-environment action is needed.

**Fixed smoke contract**

```text
config: configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml
fold: 0
campaign: parallel_workflow_smoke_v1
modality: audio_text
dataset: daic
train GPUs: 4 (existing DDP launcher; do not change)
eval GPUs: 1
training.num_train_epochs: 1
split.smoke_subject_limit: 6
evaluation.sample_prediction_mode: original_teacher_forced
evaluation.evaluation_view: harmonized_all_windows_full_coverage
evaluation.aggregation_level: subject
namespace: headline/binary_strict
checkpoint sync: compact evidence only unless adapter retrieval becomes necessary for local verification
```

The smoke tests workflow mechanics, not scientific quality. A poor but valid metric is not a failure and must not trigger tuning.

**ACTIONS**

1. Create a dedicated smoke lane from the exact merged implementation SHA. Record pin, branch, source SHA, deployment ID, runtime root, campaign/run path, group/logical run, attempt, fold, and seed.
2. Run full local preflight and the exact `exp deploy --dry-run`.
3. Verify `transfer1`, scheduler-login commands, GPFS paths, environment, source target absence, disk space, datasets, model snapshot, config, and resolved contract.
4. Deploy to a new isolated path. Verify remote hashes and `deployment.json`.
5. Explicitly transfer the experiment context and any required generated manifest/split inputs that `.gitignore` excludes.
6. Run `exp submit --dry-run`; reconcile exact job count and 1-GPU resource shape.
7. Submit through Slurm. Record train and dependent eval job IDs immediately.
8. Monitor continuously with `squeue`, `sacct`, logs, and artifacts. A pending job is normal. An empty queue is not success.
9. On failure:

   - record and classify it;
   - preserve the failed attempt;
   - transient infrastructure: one unchanged retry maximum with a new attempt;
   - deterministic defect: make the smallest fix, validate, PR/merge, new deployment, new attempt;
   - one unchanged retry is allowed for each demonstrated transient-infrastructure failure;
   - deterministic code/config failures require a validated fix, new deployment, and new attempt; no arbitrary total-attempt ceiling applies;
   - never overwrite an attempt or deployment.

10. After terminal remote success, run the remote acceptance audit.
11. Dry-run compact collection, inspect it, then synchronize compact evidence and runtime logs through `transfer1`.
12. Locally verify hashes, recompute headline metrics, rebuild the registry, generate a deterministic run report, and advance lifecycle to `REPORTABLE` through official APIs.
13. Use `provenance-reporting` for the smoke handoff. Do not put the metric into the implementation runbook; store it only in provenance-complete evidence/journal/report.
14. Extend the final auditor with the real deployment, attempt, job, collection, local-verification, and reportability evidence.

**VALIDATE**

- Remote deployment hash equals the local captured source hash.
- Runtime writes are outside deployment code.
- Top-level train and eval jobs are `COMPLETED` with `ExitCode=0:0`.
- Required artifacts exist.
- Compact local collection contains standalone evaluation evidence and excludes bulk adapters by default.
- Local metric recomputation exactly matches the synced artifact.
- Evaluation qualifiers match the fixed contract.
- Registry imports the real attempt ID with correct modality/dataset/campaign.
- Attempt state is `REPORTABLE`.
- Deterministic run report exists.

**EVIDENCE**

- smoke lane/worktree/branch/full SHA;
- deployment ID/path/hash and runtime root;
- resolved config and override hash;
- attempt and job IDs, terminal `sacct` output, logs;
- failed/retry chain if any;
- remote audit;
- collection dry run and local inventory;
- local verification output;
- registry/show/provenance output;
- report path;
- smoke audit path.

**EXIT**

- One attempt satisfies the full real-smoke contract and is `REPORTABLE`.
- Every failed attempt/job is preserved and reconciled.
- No production experiment was started.

**NEXT:** Immediately enter Phase 11.

### Phase 11 — post-smoke correction and implementation freeze

**ENTRY**

- Phase 10 is `PASSED`.
- Real smoke evidence is local and reportable.

**ACTIONS**

1. Review the smoke for any workaround, manual step, stale path, misleading message, weak collision check, missing artifact, or undocumented recovery.
2. Convert every required manual workaround into implementation or explicit runbook behavior.
3. Add regression tests for each smoke-discovered defect.
4. Make focused correction PRs, validate, and normally merge under the grant.
5. If source changes affect deployment/submission/collection semantics, re-run the affected smoke scope with a new deployment and attempt. The final accepted smoke must exercise the final implementation SHA. Preserve every failed attempt and keep unchanged transient retries bounded to one per demonstrated failure.
6. Freeze the implementation interface and record CLI help for every new command.
7. Run targeted tests and the full suite again on the final implementation SHA.
8. Extend the final auditor with final-SHA matching and no-manual-workaround checks.

**VALIDATE**

- Final smoke deployment source SHA equals the final core implementation SHA, or a documented non-runtime-only later change is proven not to affect the smoke contract.
- Every discovered defect has a test.
- Full suite passes.
- CLI help matches the runbook.
- Auditor refuses completion only because Phase 12/13 are pending.

**EVIDENCE**

- post-smoke review;
- correction PRs/head/merge SHAs;
- final implementation SHA;
- final targeted/full test outputs;
- CLI help snapshots;
- incomplete-auditor output.

**EXIT**

- Core implementation is frozen, tested, and proven by the real smoke.

**NEXT:** Immediately enter Phase 12.

### Phase 12 — switch repository instructions and skills to the new workflow

**ENTRY**

- Phase 11 is `PASSED`.
- The new commands and behavior exist and match their CLI help.
- The real smoke passed on the final applicable implementation.

This phase happens late on purpose. Before it passes, old instructions remain the executable default. Do not make unavailable proposed commands authoritative early.

**ACTIONS**

1. Update `AGENTS.md`:

   - make managed worktrees and isolated deployments the default for code-changing experiments;
   - document source/runtime/output separation and supported output layout;
   - require pin verification;
   - replace mutable-checkout sync/submission examples with implemented lane commands;
   - retain a clearly labeled legacy fallback only where still necessary;
   - document common resolved overrides and explicit evaluation view;
   - document lane autonomy grants, stacked branches, PR non-blocking behavior, orchestrator boundaries, and hard stops;
   - keep “submission is not completion” and provenance rules;
   - retain the current no-project-wide-GPU-cap rule.

2. Update `.agents/skills/mn5-cluster-ops/SKILL.md`:

   - use isolated deployment/runtime/output paths;
   - use implemented deploy/submit/status/collect/validate commands;
   - require common path resolution and compact evidence;
   - add lane-grant behavior and resume rules;
   - remove the stale 64-H100 ceiling;
   - preserve endpoint, Slurm-only, no-delete, and shared-environment safety rules.

3. Update `.agents/skills/git-pr/SKILL.md`:

   - allow dependent PRs to target recorded parent branches;
   - distinguish Tier 1 squash from Tier 2/integration merge commits;
   - state that a PR is not an experiment pause;
   - preserve no-force-push, checks, review, and grant-based merge safeguards;
   - assign ambiguous cross-lane integration to the orchestrator/human decision boundary.

4. Update `.agents/skills/experiment-tracking/SKILL.md`:

   - add deployment identity/reference;
   - enforce campaign/modality/dataset/run/fold layout;
   - require evaluation view for production reportability;
   - require a new attempt for every rerun/redeployment;
   - use group-scoped comparison rather than global `best` for winner selection.

5. Update `.agents/skills/local-validation/SKILL.md`:

   - add pin, deployment immutability, common-path, output-layout, collection-filter, stacked-branch, and final-auditor tests;
   - require the full suite for workflow contract changes.

6. Update `.agents/skills/agent-journal/SKILL.md`:

   - require lane grant, worktree, branch ancestry, deployment ID/hash, attempt/job IDs, phase state, and integration decision when available;
   - distinguish progress entries from terminal completion.

7. Update `.agents/skills/provenance-reporting/SKILL.md`:

   - include deployment ID/hash and stacked source ancestry;
   - require explicit comparison group/attempt list for winner claims;
   - preserve existing metric provenance rules.

8. Do not change `.agents/skills/plain-english` or `.agents/skills/grilling` unless validation finds a direct contradiction.
9. Update `docs/DEVICES.md` and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` with exact implemented commands, writable roots, resumption, monitoring, collection, and authority rules.
10. Update `configs/README.md` only if implemented evaluation-view or override conventions affect config authors.
11. Update this runbook's command status from proposed to implemented where true. Preserve historical design context without leaving contradictory procedures.
12. Run skill validation, document consistency audits, targeted tests, and the full suite.
13. Package these contract changes in one focused activation PR or a small documented stack. Normally merge under the active grant after checks pass.

**VALIDATE**

```bash
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/mn5-cluster-ops
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/git-pr
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/experiment-tracking
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/local-validation
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/agent-journal
python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/provenance-reporting
python -m pytest tests/test_parallel_workflow_docs.py -q
python -m pytest tests/
```

Document tests must fail on:

- stale 64-H100 ceilings;
- mutable-checkout execution presented as the default;
- missing lane-grant boundaries;
- `best_model/` exclusions that also hide compact evaluation evidence;
- instructions that always target PRs to `main`;
- instructions that call submission or a phase boundary completion;
- references to proposed commands that do not exist in CLI help.

**EVIDENCE**

- changed instruction/skill/doc inventory;
- skill validation outputs;
- consistency test output;
- full-suite output;
- activation PR stack, head SHAs, merge SHAs;
- post-merge CLI help and instruction audit.

**EXIT**

- Repository instructions and skills describe the implemented workflow consistently.
- No stale operational contradiction remains in the named files.
- New agents will be routed to this workflow.

**NEXT:** Immediately enter Phase 13.

### Phase 13 — terminal audit, final journal, and handoff

**ENTRY**

- Phases 0–12 are `PASSED`.
- All required PRs are merged under the active grant.
- No task-owned job is pending or running.
- Compact smoke evidence is local and reportable.

**ACTIONS**

1. Update the implementation worktree to the final merged source and record the full SHA.
2. Re-run:

   - syntax checks for all changed Python/shell files;
   - every targeted parallel-workflow, tracking, submission, collection, docs, and skill test;
   - full `python -m pytest tests/`;
   - implementation CLI `--help` snapshots;
   - local registry rebuild and smoke provenance query;
   - final instruction/skill consistency audit.

3. Reconcile every branch, PR, deployment, attempt, Slurm job, failure, retry, artifact, report, and journal entry in `state.json`.
4. Run the final auditor without `--allow-incomplete`:

   ```bash
   python tools/audit_parallel_workflow_implementation.py \
     --state <state.json> \
     --output <state-root>/final_audit.json
   ```

5. The auditor must fail closed on any missing phase, evidence path, hash, job terminal state, reportability gate, merged source identity, instruction update, protected-path rule, or test result.
6. If the auditor fails, return to the earliest affected phase, fix it, and rerun from there. Do not edit the audit artifact to pass.
7. When the auditor exits zero, mark the execution `COMPLETE` through the state tool.
8. Append the final journal entry. Do not copy secrets, raw transcripts, subject identifiers, or bare metrics.
9. Send the final handoff using the exact template below.

**VALIDATE**

- Final auditor exits zero.
- `state.json.status == COMPLETE` and every phase is `PASSED`.
- Final source equals the recorded merged `origin/main` SHA.
- No task-owned jobs remain active.
- Unrelated `skills-lock.json` and other user work remain untouched.

**EVIDENCE**

- final merged SHA;
- complete state ledger;
- final audit JSON and SHA-256;
- final targeted/full validation outputs;
- registry/provenance/report paths;
- final journal path;
- final handoff.

**EXIT**

- The Definition of Done in Section 0.2 is fully satisfied.
- This is the only normal point where the agent may say the task is complete.

**NEXT:** None. Send the final handoff.

Final handoff template:

```text
Task completed: yes
Execution ID and state file:
Final audit path/hash/status:
Final main SHA:
Implementation worktree/branches:
PRs and merge SHAs by phase:
Rollback tag and MN5 archive evidence:
Implemented commands:
Three-lane dry-run pilot evidence:
Real smoke deployment/attempt/job IDs:
Failed attempts/retries and causes:
Compact local evidence and reportability:
Targeted tests:
Full test suite:
AGENTS/docs/skills activation PR:
Unrelated changes preserved:
Known limitations and legacy fallback:
```

---

## 16. Final acceptance matrix

The final auditor must enforce this matrix. A checkbox in prose is not evidence by itself.

### Control and continuation

- [ ] Grant journaled before mutation.
- [ ] Persistent state tool uses atomic updates.
- [ ] Phase skipping and premature completion are rejected.
- [ ] Resume after context loss uses recorded branches/jobs rather than duplicating work.
- [ ] Final audit is required to mark the task complete.

### Local isolation

- [ ] Managed lanes live under `~/worktrees/`.
- [ ] `.agent-pin.json` does not dirty a worktree.
- [ ] Wrong worktree, branch, symlink escape, and protected paths fail closed.
- [ ] Stacked child records exact parent branch/SHA.
- [ ] Existing user worktrees and `skills-lock.json` are preserved.

### MN5 isolation

- [ ] Deployment targets are new and cannot be overwritten.
- [ ] Source deployment, writable runtime, and permanent output roots are separate.
- [ ] Remote hashes match the deployment record.
- [ ] Runtime writes never land under deployment code.
- [ ] No generated sync uses `--delete`.

### Submission and tracking

- [ ] Train, evaluation, checkpoints, sidecars, contexts, and logs use one resolved contract.
- [ ] Dynamic Slurm chdir selects the deployment.
- [ ] Missing evaluation view fails closed for production.
- [ ] Output layout imports as campaign/modality/dataset/run/fold.
- [ ] Reruns use new attempts and preserve failure links.
- [ ] Monitoring requires terminal `sacct` evidence and artifacts.

### Collection and reporting

- [ ] Compact evaluation evidence survives checkpoint exclusions.
- [ ] Local hash and metric verification pass.
- [ ] Lifecycle transitions cannot be skipped.
- [ ] Smoke reaches `REPORTABLE` under its real attempt ID.
- [ ] Winner comparisons are group/attempt scoped with full qualifiers.

### Autonomy and integration

- [ ] PRs do not block in-lane work.
- [ ] Stacked branches work before parent integration to `main`.
- [ ] Competing experiments are not all auto-merged.
- [ ] Complementary integration checks Git and semantic conflicts.
- [ ] Lane agents cannot mutate shared infrastructure or perform cleanup.
- [ ] Only a complete predeclared comparison contract permits automatic winner selection.

### Repository contract

- [ ] `AGENTS.md` names the new default workflow.
- [ ] `mn5-cluster-ops`, `git-pr`, `experiment-tracking`, `local-validation`, `agent-journal`, and `provenance-reporting` skills are updated and validated.
- [ ] `docs/DEVICES.md` and the MN5 runbook match implemented CLI help.
- [ ] No stale 64-H100 ceiling remains.
- [ ] No broad destructive rollback command remains.
- [ ] Targeted tests and the full suite pass.

---

## 17. Fixed decisions

1. New managed worktrees live under `~/worktrees/`; existing worktrees stay in place.
2. `Teacher-System` and `LLM-Depression-teacher` are excluded from editing and cleanup tooling.
3. Production deployments require clean committed source.
4. Source deployments are unique and never overwritten.
5. Source, runtime, and permanent outputs are separate.
6. Run roots keep the campaign/modality/dataset/run/fold layout.
7. CLI-only overrides may stay on a committed SHA; tracked YAML changes use a branch.
8. Tier 1 winner integrations normally squash; Tier 2/integration changes may use merge commits.
9. Stacked branches allow dependent work without waiting for `main`.
10. PRs provide review and provenance but do not block lane execution.
11. Competing experiments do not all auto-merge.
12. Complementary integration checks both Git and semantic conflicts.
13. An integration/orchestrator agent owns cross-lane coordination.
14. High autonomy requires a named, journaled, task-scoped lane grant with budgets and hard stops.
15. Routine in-lane iteration continues overnight under that grant.
16. Destructive cleanup, shared infrastructure changes, evidence overwrite, silent expansion, ambiguous integration, and rollback mutation are hard stops.
17. No automatic deletion or TTL is enabled initially.
18. Rollback uses reviewed inventories and targeted restoration; no `git clean -fdx`, wildcard removal, or one-command destructive restore script.
19. No project-wide 64-H100 ceiling exists; scheduler/account/QoS and narrower task grants control concurrency.
20. A phase, PR, deployment, job, sync, validation, or report is not task completion.
21. Only the Phase 13 terminal auditor may authorize a normal completion claim.

---

## 18. Provenance and maintenance of this runbook

- Initial investigation and decisions: 2026-08-20 chat and repository inspection.
- Safety and tracking review: 2026-08-20 against current scripts, configs, lifecycle code, and CLI help.
- Detailed autonomy and continuation contract: 2026-08-20, modeled on the successful end-to-end Gemma 4 DAIC fixed-head execution recorded in `docs/agent-journal/2026-08-12.md` and its implementation runbook.
- Local source reviewed at `2d995f4cb2b17dad6921f286a0f7b10fbb257c8f` on `main`.
- MN5 observations in the original investigation were read-only; authoring this runbook performed no MN5 mutation.
- Key current files: `AGENTS.md`, `docs/DEVICES.md`, `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`, `configs/README.md`, `docs/SIGNAL_FLOW.md`, `scripts/submit_train_and_eval.sh`, `scripts/run_train_slurm.sh`, `scripts/run_eval_slurm.sh`, `scripts/collect_experiment.sh`, `tools/exp.py`, `src/train.py`, `src/evaluate.py`, and `src/experiment_tracking/`.

When implementation changes the commands or contracts, update this runbook in the same PR as the behavior. Do not leave an obsolete command as executable guidance.

---

## 19. Short prompt to give the implementation agent

Send this text in chat together with the runbook path:

> Implement the complete parallel experiment workflow in `/home/emre/Projects/AudioLLM/LLM-Depression/docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md`. Read the entire runbook and every mandatory file it names before acting. Inspect the journal, existing branches/worktrees, and any active execution ledger first; resume the first incomplete phase without repeating passed work or resubmitting recorded jobs. I grant full autonomy for this named implementation task until the Phase 13 final auditor passes, a listed hard stop occurs, or I revoke it. Journal this exact grant before the first mutation unless the same active grant is already recorded. You may edit, validate, commit, push, open/update, and normally merge as many focused in-scope PRs as the implementation requires; create the reviewed rollback tag/archive; create only new task-owned worktrees, MN5 deployments, runtime roots, contexts, outputs, and logs; perform non-destructive dry-run-first rsync; submit the task-owned smoke and correction jobs required by the runbook using the existing script-defined four-H100 DDP training shape and one-H100 evaluation shape; monitor/cancel/retry as the runbook permits; sync compact evidence; validate locally; update `AGENTS.md`, the named docs and `.agents` skills; and continue automatically between phases, PRs, jobs, context compaction, and agent restarts. There is no numeric PR, merge, attempt, or GPU ceiling imposed by this runbook; scheduler, account, QoS, fixed per-job script shape, task scope, and evidence-driven retry rules govern execution. A phase, PR, deployment, submission, job completion, sync, validation, or report is not task completion. Do not stop before the final auditor passes. Excluded: deletion, `--delete`, evidence overwrite, protected-path changes, force-push, history rewriting, admin/bypass merge, unresolved failing checks or requested changes, package/shared-environment/dataset/model/infrastructure changes, silent scope expansion, weakening privacy/leakage/provenance rules, ambiguous integration, and rollback mutation. If a genuine listed hard stop occurs, preserve evidence, update the ledger and journal, and report the exact blocker; otherwise keep going through every phase until the final auditor passes.
