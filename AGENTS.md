# AGENTS.md

Leakage-safe depression-classification research repo: LoRA fine-tuning of Qwen2-Audio-7B / Qwen2-7B on DAIC, EDAIC, CMDC, and Turkish datasets. Two runtime homes: this local machine (RTX 4090, dev/tests) and MareNostrum 5 (H100s, all real training). `README.md` at the repo root is the de-facto overview. Trust `configs/README.md` and the current YAML files for exact experiment settings.

## Read first, in order

1. `docs/DEVICES.md` — host topology, envs, sync directions, safety boundary. Mandatory before any cluster or rsync action.
2. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` — full experiment lifecycle (submit → monitor → sync back → validate). "Submission is not completion."
3. `configs/README.md` — canonical config recipe and naming.
4. `docs/SIGNAL_FLOW.md` — pipeline architecture (manifest → examples → collator → model → metrics).

Note: `docs/`, `outputs/`, `logs/`, `output_model/` are **gitignored** — they are not committed and are not copied by `scripts/sync_to_cluster.sh`. If a cluster job needs a generated manifest, rsync that file explicitly.

## Skills

This repo ships agent skills under `.agents/skills/`, tracked in git. Load a skill with the `skill` tool when its job comes up; do not skim the files ahead of time. They are:

- `plain-english` — writing and revising human-facing text. See the "Writing style" section below.
- `agent-journal` — writing journal entries after meaningful work. See the "Agent journal" section below.
- `git-pr` — committing, pushing, and opening PRs for agent changes.
- `local-validation` — selecting and running proportionate local validation before a PR or cluster sync.
- `mn5-cluster-ops` — any MareNostrum 5 transfer, submission, monitoring, or result-sync task.
- `experiment-tracking` — planning or registering experiments, querying the registry, generating reports, exporting to W&B.
- `provenance-reporting` — reporting any result with complete provenance. See the "Reporting rule" section below.

## Writing style — plain English (MANDATORY)

- Use the `plain-english` skill whenever writing human-facing text: chat replies, documentation, reports, PR descriptions, comments, summaries, handoffs, warnings, and error messages.
- Write in simple, direct, natural English. Put the main point first.
- Avoid corporate, academic, inflated, canned, or AI-sounding language. Cut filler and vague praise.
- Keep exact technical and scientific terms when they matter. Plain English must not weaken accuracy, safety rules, or provenance.
- Say clearly what was done, what failed or was skipped, and what the user needs to do next.

## Agent journal (MANDATORY for meaningful work)

- Use the `agent-journal` skill after meaningful PR work, experiments or training runs, experiment results, important debugging findings, architecture/design decisions, methodology changes, reproducibility changes, or a significant blocker.
- Do not journal trivial edits, routine inspection, or status checks that produced no new finding.
- Store entries in `docs/agent-journal/YYYY-MM-DD.md`, using the `Europe/Istanbul` calendar date. Append multiple entries to the same daily file; never overwrite earlier entries.
- Explain the context, why the work was needed, the decision and its reason, what changed or ran, the hypothesis or expected outcome when relevant, the actual result or current state, and what should happen next.
- Include real references when available: experiment/run/attempt/job IDs, branch, full commit SHA, PR, checkpoint, config, dataset version, and evidence paths. Never invent identifiers.
- Write entries in plain English. Do not copy secrets, credentials, raw transcripts, subject identifiers, or sensitive dataset content.
- The journal is a narrative index, not experiment evidence. Keep `run_config.yaml`, tracking sidecars, local artifacts, generated reports, and PRs authoritative. Apply the provenance rule below to every result written in the journal.
- Write the entry after the meaningful milestone is known. If a PR or experiment later gets a new ID or outcome, append a new entry instead of rewriting history.

## Local environment

- Shell starts in conda `base` (Python 3.13, **no torch**) — tests error there. Activate the project env first:
  ```bash
  conda activate llmdep4090
  ```
- Run tests from the repo root with `python -m pytest tests/` (tests import both `src` and `scripts` packages; bare `pytest` fails to resolve them). Verified state: 394 passed in ~47 s. Run a single file with `python -m pytest tests/test_x.py -q`.
- **Before reporting any result, read the "Reporting rule — no result without provenance (MANDATORY)" section below.**
- No-model sanity suite: `./scripts/sanity_tests_no_model.sh` (builds manifests, audits splits; needs dataset roots).
- Config defaults are BSC/GPFS absolute paths. For local runs, override via env vars: `DAIC_DATASET_ROOT`, `EDAIC_DATASET_ROOT`, `CMDC_DATASET_ROOT`, `TURKISH_DATASET_ROOT`, `MODEL_PATH` (audio model), `TEXT_MODEL_PATH` (text model). Local model copies live under `/media/emre/Backup/AudioLLM/models/`.

## Config conventions

- Run only `configs/main/*.yaml`. The harmonized family is named `<dataset>[_t<threshold>]_<modality>_harmonized_selmacrof1_tf[_variant].yaml`. `configs/experiments/` is active non-headline work; `configs/archive/` is history — do not "fix" or resurrect archived configs. E-DAIC remains an explicitly documented out-of-family legacy exception.
- `configs/quarantines.yaml` is referenced by every config via `${PROJECT_ROOT}/configs/quarantines.yaml` — never move it.
- YAML values support `${VAR}` / `${VAR:-default}` interpolation; `${PROJECT_ROOT}` auto-resolves to the repo root.
- Any key can be overridden from CLI with `--set path.to.key=value` (e.g. `--set lora.last_n_layers=2`).

## Recipe invariants — do not regress these

- Headline evaluation is **teacher-forced** (`evaluation.sample_prediction_mode: original_teacher_forced`), and harmonized checkpoint selection is **macro-F1** (`inner_val_macro_f1`, mode max). Report positive-F1 alongside macro-F1. AUROC is intentionally not a headline metric under this recipe.
- Read `headline/binary_strict_*` metrics (INVALID counts as wrong); ignore `valid_only_*`.
- Audio encoder is **frozen by default** (guarded by `enforce_audio_encoder_freeze`); only archived `*_nofreeze` configs train it. `DepAdapter` and projector training are opt-in via `audio_adapter.*`.
- Harmonized audio input contains exactly one chunk per prompt. D3TEC, Turkish, Androids, and CMDC use every natural-unit window with hierarchical subject/unit/window weights. DAIC uses every participant-only packed30 chunk with subject-normalized weights. Joint K4 bundles are archived behavior, not the harmonized recipe.
- External labels stay `Depressed` / `Non-depressed`; prompts in English, transcripts in original language.

## Training / evaluation entrypoints

```bash
torchrun --nproc_per_node=4 src/train.py --config configs/main/<cfg>.yaml --fold 0 --run_name <name>
python src/evaluate.py --config configs/main/<cfg>.yaml --fold 0 \
  --checkpoint_dir output_model/<modality>/<dataset>/<run_name>/fold_0/best_model
python src/data/build_manifest.py --config configs/main/<cfg>.yaml   # shared across modalities
```

- Local smoke run (single GPU, tiny): `--set training.num_train_epochs=1 --set split.smoke_subject_limit=6` with `--nproc_per_node=1`.
- Manifests/splits are shared across modalities; build inputs must include transcripts even when `use_text=false`.
- Optuna HPO: `scripts/run_optuna_slurm.sh` (see `README.md` for the search-space defaults).

## Experiment tracking workflow (implemented, Tasks 0–9)

Design doc: `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md` — Sections 25–28 are the agent execution contract: implement exactly one task card per change set, run its acceptance gate, report per Section 28.

- Core package: `src/experiment_tracking/` (`constants`, `canonical`, `identity`, `schemas`, `lifecycle`, `discovery`, `qualification`, `registry`, `reporting`, `wandb_export`, `adapters`); schemas under `experiments/schemas/`; SQLite migration `src/experiment_tracking/migrations/001_initial.sql`. Stdlib only; no W&B import outside the exporter.
- Per-fold sidecars beside `run_config.yaml`: `metadata.json`, `status.json`, `jobs.jsonl` (append-only), `artifacts.json`, `evaluations.json`. Lifecycle states: `PLANNED → DEPLOYED → SUBMITTED → RUNNING → COMPLETED_ON_MN5 → SYNCED_LOCALLY → LOCALLY_VALIDATED → REPORTABLE`; `IMPORTED_LEGACY/FAILED/CANCELLED/… → SUPERSEDED`. FAILED/CANCELLED never complete; reruns are new attempts.
- New runs: pass `--experiment-context <json>` to `src/train.py`/`src/evaluate.py` (rank 0 writes sidecars; `run_config.yaml` gains a `tracking:` block; evaluation records are idempotent — same id same content skips, different content refuses). On the cluster, set `EXPERIMENT_CONTEXT=/gpfs/.../context.json` — `scripts/run_train_slurm.sh` and `scripts/run_eval_slurm.sh` append the flag, and `scripts/submit_train_and_eval.sh` records SUBMITTED job events into the fold's `jobs.jsonl`. Context must be rsynced explicitly (it lives under gitignored paths).
- SQLite registry: `outputs/experiment_registry/experiments.sqlite` (rebuildable from per-run evidence; rebuild writes a temp DB and atomically replaces). Entrypoints: `tools/rebuild_experiment_registry.py --scan-root output_model`, `tools/import_experiment.py --run-dir <run>`, `tools/exp.py list|show <attempt-id>|provenance <metric-id>|jobs [--failed]|best` — `best` requires full qualifiers (`--dataset --metric --namespace --backend --view --aggregation`); underqualified queries are refused.
- Modern tracked runs (fold dirs with valid `metadata.json`/`status.json`/`jobs.jsonl`/`artifacts.json`/`evaluations.json`) import under their real attempt IDs with Git/lifecycle/job/artifact/evaluation provenance; they are never duplicated as synthetic legacy attempts. Malformed or contradictory sidecars fail closed (import `REJECTED`, recorded in `registry_imports`) instead of falling back to the legacy path. `metadata.supersedes_attempt_id` records retry links and is consumed by the importer (resolved on rebuild when both attempts are in the tree). Evidence corrections go through `tools/verify_run_evidence.py` (verify-artifacts / verify-evaluations / append-events / transition / set-supersedes) — lifecycle transitions and job events use the official APIs.
- Reports: `tools/generate_run_report.py --attempt-id <id> [--fold n]` and `tools/generate_group_report.py --attempts <csv> --metric-name … --namespace … --backend … --view … --aggregation …` → `report.json`/`report.md`. Deterministic (no timestamps unless `--with-timestamp`); incompatible group aggregation is refused with explicit reasons; `MN5-only, not locally verifiable` is marked explicitly; researcher conclusion stays blank unless supplied.
- W&B (evidence-first, post-run export only — never in the training loop): `tools/export_run_to_wandb.py --run-dir <run>|--attempt-id <id>|--group <gid> --mode dry_run|offline|cloud`. Cloud mode maps to wandb `online`; requires the locally logged-in account (`~/.netrc`), entity `emre9766-audio-llm`, project `audiollm-depression`. Dry-run never touches the network. Payloads are recursively safe-filtered (transcripts, prompts, subject ids, absolute dataset paths, credentials are dropped and listed in `exclusions`); incomplete legacy evidence is tagged `incomplete` and never promoted to REPORTABLE. Deterministic run ids; reruns resume, never duplicate.
- Historical evidence adapters (inventory only): `src/experiment_tracking/adapters.py` — translated runs (`*_en*` names), merged runs (`resolved_merged_config.json`), hidden classifiers (`classifier_metadata.json`), plus `inventory_evidence()` with quarantine lists.
- Workbook: `scripts/build_clean_workbook.py` remains script-only; `--validate-selected <selected_results.json>` cross-checks explicit selections (from `tools/export_selected_results.py --selection <yaml>`) against headline cells — mismatches exit non-zero, missing records are listed as `legacy-unmigrated`, never zeroed, and Optuna-not-run fields stay blank.
- Task 9 smoke (2026-08-07): run `smoke_t9_v2_2d6a466` (train 44394029, eval 44394346 COMPLETED; failed 44393943/44394030 recorded with resubmission chain); W&B runs `wandb-d424349b1304ae26a5f215df` and `wandb-1bdab9a238649a5b3d943706`; audit: `outputs/experiment_reports/smoke_audit_task9.json`.
- W&B workbook-selected backfill (`docs/WANDB_WORKBOOK_SELECTION_PLAN.md`, implemented 2026-08-07): only execution runs supporting `depression_results_clean.xlsx` Provenance values are exported. Tracked input `experiments/definitions/workbook_wandb_selection.yaml` (188 entries, one policy each; no metric values inside); generated evidence under `outputs/experiment_registry/`: `workbook_dependency_inventory.json`, `workbook_wandb_manifest.json`, `workbook_wandb_dry_run.json`, `workbook_wandb_export_audit.json`. Flow: `tools/build_workbook_dependency_inventory.py` (read-only) → `tools/resolve_workbook_wandb_selection.py` (hash-verified, dedupes by W&B run id, zero network) → `tools/export_run_to_wandb.py --manifest <m> --mode dry_run` then `--mode cloud --approved-dry-run <audit>` (cloud refuses any workbook/registry-evidence/payload hash change since the approved dry run; requires explicit authorization). Policies: `sync` / `pending_local_evidence` / `pending_importer_support` / `quarantine_ambiguous` / `skip_not_run` / `skip_derived_only`. Current state (2026-08-07 snapshot): 0 sync units — 95 pending importer support (hidden/merged adapters are inventory-only), 22 quarantine (15 ordinary runs lack the `evaluation_view` qualifier in evidence; 7 runs under non-canonical `output_model/experiments/` layout), 12 pending local evidence (EN-translated on MN5), 59 skip derived. Key follow-ups recorded in the audit `next_steps`: record `evaluation_view` in evaluation evidence; non-canonical layout support; qualified hidden/merged/EN importers; sync `output_model_en`.
- Known limitations: legacy runs are not REPORTABLE until an evaluation view is recorded (fail-closed by design); modality for runs imported from non-canonical layouts is path-derived; train-side job events carry null `slurm_job_id` (context minted pre-submission); runs whose sidecars mix two attempts (e.g. harmonized_v1 dirs reused for a retry) are REJECTED by the modern importer until the evidence is split per attempt.

## Cluster (MN5) operations

- Two endpoints, different jobs: `ozu647717@transfer1.bsc.es` = rsync/file inspection only; `ozu647717@alogin2.bsc.es` (fallback `alogin1`) = `sbatch`/`squeue`/`sacct`. **Never run training on either** — only via Slurm. Project path on GPFS: `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression`.
- GPU parallelism: there is no project-wide GPU cap (the professor explicitly waived the old 64-H100 ceiling on 2026-08-18). Run as many jobs in parallel as the scheduler, account, and QoS will grant. Launcher defaults size one lane per task — one four-GPU training lane per training job and one one-GPU lane per auxiliary job — so the whole matrix can run at once; tune `MAX_CONCURRENT_TRAINS` / `MAX_CONCURRENT_AUX` (or `MAX_CONCURRENT_POSTPROCESS`) if a plan needs fewer or more lanes. The scheduler may still grant fewer GPUs than requested. Keep each job's configured GPU shape unless the experiment plan explicitly changes it.
- Local→cluster: `bash scripts/sync_to_cluster.sh` (captures `.provenance`, respects `.gitignore`). Never add `--delete` unless explicitly requested and reviewed.
- Canonical single-fold submission on the login node:
  ```bash
  CONFIG="$PWD/configs/main/<cfg>.yaml" RUN_NAME=<unique> FOLD=0 bash scripts/submit_train_and_eval.sh
  ```
  Do not reuse run names without checking continuation/overwrite behavior.
- Cluster env init: `module purge; module load bsc/1.0 miniforge/24.3.0-0; source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate`. Hidden-state XGBoost/Optuna workflows additionally need `export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"`.
- Results→local: rsync `output_model/` **excluding `best_model/` and `last_model/`** (user's storage-saving convention); pull `logs/` separately; fetch specific `best_model/` dirs only when required.
- Authorization: the full experiment loop is pre-authorized — submitting and cancelling MN5 jobs (training/evaluation only ever via Slurm, never on a login/transfer node), syncing cluster results back locally, and W&B cloud export (evidence-first: approved unchanged dry-run audit, then cloud). Explicit user approval is still required before overwriting remote files, deleting anything, or changing shared environments/resources.
- Per-task autonomy grants: for one named task the user may grant full autonomy in chat (e.g. "full autonomy for this task until done"). Before the first mutation the agent must journal the grant — exact wording, scope, expiry condition (task completion, source-SHA change, hard stop, or user revocation), and exclusions. An active recorded full-autonomy grant authorizes the agent to merge its own in-scope PRs, including prerequisite or stacked PRs in dependency order, after the required validation and repository checks pass. The agent may continue the named task after each merge without waiting for user review. It must not merge unrelated PRs, bypass branch protection, use an administrative/forced merge, or merge with unresolved failing checks or requested changes. Actions stay excluded unless the grant names them: `--delete`/destructive sync, deletions, evidence overwrite, silent experiment expansion, package installs, shared-environment changes. Retries/cancellations under a grant stay bounded: transient-infrastructure failures only, at most once per failed job, reason recorded, new attempt identity. A grant covers only the named task — never later tasks or sessions. Silence or user absence is not a grant; AGENTS.md is not a grant source.
- Git: agents are **pre-authorized** to commit, push, and open PRs for changes they made (use the `git-pr` skill) — no per-action approval needed. This covers branches `agent/<topic>` → PR to `main`; cluster-side mutations follow the authorization bullet above. Without an active recorded full-autonomy grant, open the PR for the user and never merge it. With such a grant, merge only the agent's own in-scope task PRs under the safeguards above.

## Reporting rule — no result without provenance (MANDATORY)

Every training/eval result that is synced back from the cluster, written to a report, or shown to anyone
must come with its provenance. A bare number is a bug. Violating this rule is how the old workbook became
untrustworthy (see `docs/archive/results_20260806/README.md`).

- **A reported number must be traceable to:** run name / run ID + fold, the `run_config.yaml` path in its
  run dir (config hash, manifest hash, split hash, seed), the evaluation view/backend
  (`original_teacher_forced`, likelihood, ...), the **aggregation** (fold-mean vs pooled subject-level —
  they differ, e.g. Androids 0.895 pooled vs 0.887 fold-mean), and a local artifact path
  (`final_summary.json`, `metrics_*.json`, `predictions_subject_level.csv`, audit JSON).
- **The canonical report is `depression_results_clean.xlsx`, generated by `scripts/build_clean_workbook.py`
  (detailed variant via `--detailed`). Never hand-edit the workbook cells.** New numbers are added to the
  script's data tables (with their provenance strings), then regenerated. The `Provenance` sheet is the
  lookup table for every headline value.
- **Syncing results is not reporting.** After a cluster→local rsync, recompute or verify the headline
  numbers locally from the synced artifacts before writing them anywhere; flag any value that could only
  be read on MN5 (e.g. `output_model_en/.../final_summary.json`) as "MN5-only, not locally verifiable".
- **Cluster→local syncs must include the compact evidence**, not just checkpoints: metrics JSONs,
  `predictions_subject_level.*`, `final_summary.json`, `run_config.yaml`, audit JSONs. Excluding
  `best_model/`/`last_model/` is fine; excluding the metrics/predictions is not.
- **Ambiguity must be resolved, not inherited:** state the DAIC eval view (full-coverage K4 bundles vs
  fixed-K4 — same checkpoint scores 0.841 vs 0.755), the D3TEC recipe (normalized vs rotary), the merged
  run ID per modality (retrain runs vs the smoke run), and the Optuna/Subject-OS status (not run → blank,
  not silently omitted or invented).
- **Head / hidden-classifier results:** matrix configs live in `configs/features/*.yaml` (e.g.
  `translation_en_matrix.yaml`); outputs under `outputs/hidden_classifiers/<dataset>/<condition>/<run>/fold_<n>/`;
  report the aggregation convention used (workbook convention = pooled 5-fold subject-level) and the job IDs.
- **Resubmits must be recorded:** if jobs fail and are rerun (e.g. degraded node), the report notes the
  failed job IDs, the failure, and the rerun IDs — provenance includes how the number was obtained.

## Checkpoints & provenance

- Layout: `output_model/<modality>/<dataset>/<run_name>/fold_<n>/{best_model,last_model,logs,run_config.yaml}`.
- `best_model` is the evaluated checkpoint (validation macro-F1 selection for harmonized runs); never substitute `last_model` silently.
- Checkpoints are ~160 MB LoRA adapters, not full models — evaluation needs the matching base model (audio/audio+text → Qwen2-Audio-7B-Instruct, text-only → Qwen2-7B-Instruct).
- `run_config.yaml` in each run dir is the authoritative record of resolved config/overrides/hashes — a similar-looking run name or log filename is not proof of provenance.

## Gotchas

- `src/evaluate.py` bypasses `AudioTextDataset` (deterministic, no augmentation) — don't route eval through the training dataset.
- Turkish pipeline: leakage unit is `patient_id`, 5-fold CV, default `cv_protocol: train_val` (outer fold is both selection and reported score — not a held-out test).
- `.provenance/` is refreshed by `scripts/capture_provenance.sh` on every cluster sync; don't hand-edit it.

## Parallel Experiment Workflow (Implemented Phases 0-13)

**Default for code-changing experiments:** Use managed worktrees and isolated deployments (Phases 0-13). The legacy mutable-checkout workflow (`bash scripts/sync_to_cluster.sh` → `CONFIG=... bash scripts/submit_train_and_eval.sh` on the permanent checkout) remains as a clearly labeled fallback only where the new lane tooling is not yet used.

### Managed worktrees and pins
- Create lane worktrees only under `~/worktrees/` via `python tools/exp.py create <slug> --tier {1,2} [--from <branch-or-sha>] [--dry-run]`
- Tier 1 (`agent/exp-*`, squash on merge) for competing/short experiments; Tier 2 (`agent/feat-*`, merge commit) for complementary/long-lived features
- Stacked branches: `agent/exp-base -> agent/exp-base-pooling -> ...` each records parent branch/SHA and dependency PR; no history rewrite of deployed commits
- Each managed worktree has ignored `.agent-pin.json` (schema `audiollm.agent_pin.v1`, canonical absolute paths, `allowed_paths`, `protected_paths` including `Teacher-System` and `LLM-Depression-teacher`); `.gitignore` already contains `.agent-pin.json` so the pin does not dirty the worktree
- Before any edit, commit, deploy, or submission, run `python tools/check_worktree_pin.py` (verifies CWD inside pinned worktree, git top-level equals pin, branch equals pin, experiment definition matches `experiments/definitions/`, target inside `allowed_paths` and outside protected paths; fails closed on wrong CWD/branch, stale experiment ID, symlink/`..` escape, protected path)

### Isolated deployments, runtime, and outputs
- Source deployments are immutable and content-addressed: `tools/exp.py deploy <slug> --dry-run|--execute` (requires clean committed source for production, `--allow-dirty` for smoke/debug non-reportable, `.provenance` capture, collision-resistant ID `exp-...-<timestamp>-<commit8>-<rand>`, new-target-only rsync via `transfer1` without `--delete`, remote manifest hash verification, immutable `deployments/<deployment_id>/deployment.json` outside `code/`, writable `experiment_runtime/<experiment_id>/` for contexts/manifests/splits/logs, refusal to overwrite existing deployment)
- Writable-path contract (one common resolved override set for train, evaluation, checkpoint, sidecar, and submission): `output_dirs.manifest_dir = <runtime>/manifests/<dataset>`, `output_dirs.split_dir = <runtime>/splits/<dataset>`, `output_dirs.run_root = <permanent>/output_model/<campaign>/<modality>/<dataset>`, `LOG_ROOT = <runtime>/logs/<job_type>/<dataset>`, `EXPERIMENT_CONTEXT = <runtime>/contexts/<attempt_id>/fold_<n>/context.json`
- Source deployment (`.../deployments/<id>/code`) never receives runtime writes; changing remote permissions is not required
- Output layout keeps supported `output_model/<campaign>/<modality>/<dataset>/<run_name>/fold_<n>` for importer modality/dataset derivation

### Common path resolution and submission
- One common resolved override array is used for train, evaluation, manifest, checkpoint, and sidecar paths; it travels losslessly as base64 JSON (`OVERRIDES_JSON_B64`) from `exp submit` through `sbatch --export` into both workers (whitespace-split `EXTRA_*_ARGS` remains a legacy fallback); `scripts/submit_train_and_eval.sh` loads YAML with those overrides before calculating `FOLD_DIR`, `CHECKPOINT_DIR`, etc., and passes identical overrides to `run_train_slurm.sh`/`run_eval_slurm.sh`
- Submit with `python tools/exp.py submit <slug> --config <yaml> --fold <n> --run-name <name> --campaign <c> --modality <m> --dataset <d> --set k=v ... --dry-run|--execute`: dry-run prints the full resolved contract and exact remote script; execute verifies the deployment remotely, refuses collisions/missing `evaluation.evaluation_view`/attempt reuse before any sbatch, transfers the attempt context new-target-only, runs the wrapper on the scheduler login, parses job IDs, and appends SUBMITTED events via the state tool
- Pass `sbatch --chdir="$PROJECT_ROOT"` so the deployment path overrides the fixed `#SBATCH --chdir` in workers; export isolated `LOG_ROOT` and `EXPERIMENT_CONTEXT`; fail closed on run/fold/context collision and on missing `evaluation.evaluation_view` for production (e.g., `harmonized_all_windows_full_coverage` for harmonized DAIC smoke with `original_teacher_forced` and `subject` aggregation)
- Preserve per-job shapes: train 1 node, 4 tasks, 4 H100s, `NPROC_PER_NODE=4` (DDP); eval 1 node, 1 task, 1 H100; do not add 1-GPU training mode

### Monitoring, collection, validation, comparison
- Monitor: `python tools/exp.py status [<slug>]` reconciles only the lane's recorded jobs against the remote scheduler (`squeue`/`sacct` over SSH to the MN5 scheduler login), appends newly-terminal TERMINAL events append-only, classifies failures (transient infrastructure / deterministic code-config / cancelled dependency / unknown), plans bounded retries (one unchanged retry per demonstrated transient failure; deterministic failures need fix + new deployment + new attempt), and exits nonzero on contradictions or unknown jobs
- Collect: `python tools/exp.py collect <slug> --attempt-id <id> --dry-run|--execute` resolves the exact remote fold dir from recorded contracts, transfers compact evidence through transfer1 without `--delete`, preserves `best_model/standalone_eval/**` while excluding adapter weights, refuses incompatible local overwrites, and verifies remote/local hash agreement; `scripts/collect_experiment.sh` delegates to the same implementation
- Validate: `python tools/exp.py validate --attempt-id <id>` verifies sidecar/artifact hashes, recomputes binary_strict headline metrics from local subject predictions (INVALID counts as wrong), enforces evaluation idempotency and qualifiers, then advances COMPLETED_ON_MN5 → SYNCED_LOCALLY → LOCALLY_VALIDATED through official single-step transitions
- Finish: `python tools/exp.py finish --attempt-id <id>` is the gate orchestrator to REPORTABLE — requires COMPLETED 0:0 job events for train and best_eval plus all validation gates; never skips states; exits nonzero with the exact blocking gate
- Verify deployment: `python tools/exp.py verify-deployment <deployment-id>` read-only identity/manifest/drift check
- Compare: `python tools/exp.py compare --group <id> --attempts <csv> --dataset --metric --namespace --backend --view --aggregation --tie-rule {max,min}` loads only REPORTABLE local evidence, refuses mixed qualifiers/folds/seeds, writes a deterministic audit; ties authorize nothing. Plan integration: `python tools/exp.py plan-integration --branch-a A --branch-b B [--base origin/main]` reports ancestry, Git conflicts (read-only merge-tree), semantic-conflict candidates in contract files, required cross-feature tests; automatic merge stays unauthorized

### Lane autonomy, PRs, and integration
- A *task-scoped lane grant* (journaled: lane, question, branch/worktree/deployment/runtime/output prefixes, datasets/modalities/folds/seeds/configs/overrides, job shapes, completion/hard-stop conditions, expiry) allows a lane agent to edit its worktree, validate, commit/push, open/update its PR, create new isolated deployments, submit/cancel/monitor Slurm jobs, fix and redeploy with new attempt IDs, retrieve compact evidence, and create stacked branches — all without waiting for `main`
- PRs are coordination objects, not gates: a lane opens a (draft) PR for review/provenance but continues to iterate from its exact branch commit; dependent lanes branch from the unmerged parent commit and record the dependency
- Competing lanes do not all auto-merge; only the winner under a predeclared group-scoped comparison contract (group, attempt list, dataset, metric, namespace, backend, view, aggregation, fold-mean vs pooled, seeds/folds, tie rule) becomes an integration candidate
- Complementary integration uses a dedicated integration branch/worktree and checks both Git conflicts and semantic conflicts (defaults, registries, data assumptions, evaluation semantics); ambiguous integration is a hard stop for human decision
- Destructive cleanup/`--delete`, overwriting deployments/runs/evidence, shared dataset/model/env/package/QoS/permission changes, silent scientific expansion, provenance weakening, force-push, history rewrite, admin/bypass merge, and rollback mutation remain hard stops unless separately and explicitly granted

### Provenance and reporting
- Per-fold sidecars beside `run_config.yaml` (`metadata.json`, `status.json`, `jobs.jsonl`, `artifacts.json`, `evaluations.json`) plus `deployments/<id>/deployment.json` (with `source_manifest_sha256` copied to `metadata.json.source.deployed_source_sha256`) are authoritative; no second mutable experiment manifest
- Every production evaluation must record `dataset`, `split`/`split_protocol`, `checkpoint_role`, `backend`, `evaluation_view`, `aggregation`, `namespace`; submission fails on null view
- Winner selection uses group-scoped `exp compare --group <id> --attempts <csv> --dataset ... --metric ... --namespace ... --backend ... --view ... --aggregation ...` or deterministic `group_report`; global `exp best` alone is not used for winners

