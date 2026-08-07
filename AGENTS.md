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

- Run only `configs/main/*.yaml` (canonical, one per dataset × modality, named `<dataset>[_t<threshold>]_<modality>_selposf1_tf.yaml`). `configs/experiments/` is active non-headline work; `configs/archive/` is history — do not "fix" or resurrect archived configs.
- `configs/quarantines.yaml` is referenced by every config via `${PROJECT_ROOT}/configs/quarantines.yaml` — never move it.
- YAML values support `${VAR}` / `${VAR:-default}` interpolation; `${PROJECT_ROOT}` auto-resolves to the repo root.
- Any key can be overridden from CLI with `--set path.to.key=value` (e.g. `--set lora.last_n_layers=2`).

## Recipe invariants — do not regress these

- Headline metric is **teacher-forced** generation (`evaluation.sample_prediction_mode: original_teacher_forced`), checkpoint selection is **positive-F1** (`inner_val_positive_f1`, mode max). AUROC is intentionally not reported under this recipe.
- Read `headline/binary_strict_*` metrics (INVALID counts as wrong); ignore `valid_only_*`.
- Audio encoder is **frozen by default** (guarded by `enforce_audio_encoder_freeze`); only archived `*_nofreeze` configs train it. `DepAdapter` and projector training are opt-in via `audio_adapter.*`.
- DAIC leakage constraint: `sample_mode: subject_audio` with a **fixed K=4** chunks per example — chunk count perfectly encodes the DAIC label, so variable chunk counts leak. Training resamples K per epoch; canonical evaluation uses deterministic balanced K4 bundles for full subject coverage.
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
- Reports: `tools/generate_run_report.py --attempt-id <id> [--fold n]` and `tools/generate_group_report.py --attempts <csv> --metric-name … --namespace … --backend … --view … --aggregation …` → `report.json`/`report.md`. Deterministic (no timestamps unless `--with-timestamp`); incompatible group aggregation is refused with explicit reasons; `MN5-only, not locally verifiable` is marked explicitly; researcher conclusion stays blank unless supplied.
- W&B (evidence-first, post-run export only — never in the training loop): `tools/export_run_to_wandb.py --run-dir <run>|--attempt-id <id>|--group <gid> --mode dry_run|offline|cloud`. Cloud mode maps to wandb `online`; requires the locally logged-in account (`~/.netrc`), entity `emre9766-audio-llm`, project `audiollm-depression`. Dry-run never touches the network. Payloads are recursively safe-filtered (transcripts, prompts, subject ids, absolute dataset paths, credentials are dropped and listed in `exclusions`); incomplete legacy evidence is tagged `incomplete` and never promoted to REPORTABLE. Deterministic run ids; reruns resume, never duplicate.
- Historical evidence adapters (inventory only): `src/experiment_tracking/adapters.py` — translated runs (`*_en*` names), merged runs (`resolved_merged_config.json`), hidden classifiers (`classifier_metadata.json`), plus `inventory_evidence()` with quarantine lists.
- Workbook: `scripts/build_clean_workbook.py` remains script-only; `--validate-selected <selected_results.json>` cross-checks explicit selections (from `tools/export_selected_results.py --selection <yaml>`) against headline cells — mismatches exit non-zero, missing records are listed as `legacy-unmigrated`, never zeroed, and Optuna-not-run fields stay blank.
- Task 9 smoke (2026-08-07): run `smoke_t9_v2_2d6a466` (train 44394029, eval 44394346 COMPLETED; failed 44393943/44394030 recorded with resubmission chain); W&B runs `wandb-d424349b1304ae26a5f215df` and `wandb-1bdab9a238649a5b3d943706`; audit: `outputs/experiment_reports/smoke_audit_task9.json`.
- W&B workbook-selected backfill (`docs/WANDB_WORKBOOK_SELECTION_PLAN.md`, implemented 2026-08-07): only execution runs supporting `depression_results_clean.xlsx` Provenance values are exported. Tracked input `experiments/definitions/workbook_wandb_selection.yaml` (188 entries, one policy each; no metric values inside); generated evidence under `outputs/experiment_registry/`: `workbook_dependency_inventory.json`, `workbook_wandb_manifest.json`, `workbook_wandb_dry_run.json`, `workbook_wandb_export_audit.json`. Flow: `tools/build_workbook_dependency_inventory.py` (read-only) → `tools/resolve_workbook_wandb_selection.py` (hash-verified, dedupes by W&B run id, zero network) → `tools/export_run_to_wandb.py --manifest <m> --mode dry_run` then `--mode cloud --approved-dry-run <audit>` (cloud refuses any workbook/registry-evidence/payload hash change since the approved dry run; requires explicit authorization). Policies: `sync` / `pending_local_evidence` / `pending_importer_support` / `quarantine_ambiguous` / `skip_not_run` / `skip_derived_only`. Current state (2026-08-07 snapshot): 0 sync units — 95 pending importer support (hidden/merged adapters are inventory-only), 22 quarantine (15 ordinary runs lack the `evaluation_view` qualifier in evidence; 7 runs under non-canonical `output_model/experiments/` layout), 12 pending local evidence (EN-translated on MN5), 59 skip derived. Key follow-ups recorded in the audit `next_steps`: record `evaluation_view` in evaluation evidence; non-canonical layout support; qualified hidden/merged/EN importers; sync `output_model_en`.
- Known limitations: legacy runs are not REPORTABLE until an evaluation view is recorded (fail-closed by design); the legacy importer does not yet consume `metadata.json` sidecars (git fields null); modality for runs imported from non-canonical layouts is path-derived; train-side job events carry null `slurm_job_id` (context minted pre-submission).

## Cluster (MN5) operations

- Two endpoints, different jobs: `ozu647717@transfer1.bsc.es` = rsync/file inspection only; `ozu647717@alogin2.bsc.es` (fallback `alogin1`) = `sbatch`/`squeue`/`sacct`. **Never run training on either** — only via Slurm. Project path on GPFS: `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression`.
- Local→cluster: `bash scripts/sync_to_cluster.sh` (captures `.provenance`, respects `.gitignore`). Never add `--delete` unless explicitly requested and reviewed.
- Canonical single-fold submission on the login node:
  ```bash
  CONFIG="$PWD/configs/main/<cfg>.yaml" RUN_NAME=<unique> FOLD=0 bash scripts/submit_train_and_eval.sh
  ```
  Do not reuse run names without checking continuation/overwrite behavior.
- Cluster env init: `module purge; module load bsc/1.0 miniforge/24.3.0-0; source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate`. Hidden-state XGBoost/Optuna workflows additionally need `export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"`.
- Results→local: rsync `output_model/` **excluding `best_model/` and `last_model/`** (user's storage-saving convention); pull `logs/` separately; fetch specific `best_model/` dirs only when required.
- Authorization: SSH access ≠ permission to mutate. Explicit user approval required before submitting/cancelling jobs, overwriting remote files, or deleting anything.
- Git: agents are **pre-authorized** to commit, push, and open PRs for changes they made (use the `git-pr` skill) — no per-action approval needed. This covers branches `agent/<topic>` → PR to `main`; cluster-side mutations still require explicit approval.

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
- `best_model` is the evaluated checkpoint (val positive-F1 selection); never substitute `last_model` silently.
- Checkpoints are ~160 MB LoRA adapters, not full models — evaluation needs the matching base model (audio/audio+text → Qwen2-Audio-7B-Instruct, text-only → Qwen2-7B-Instruct).
- `run_config.yaml` in each run dir is the authoritative record of resolved config/overrides/hashes — a similar-looking run name or log filename is not proof of provenance.

## Gotchas

- `src/evaluate.py` bypasses `AudioTextDataset` (deterministic, no augmentation) — don't route eval through the training dataset.
- Turkish pipeline: leakage unit is `patient_id`, 5-fold CV, default `cv_protocol: train_val` (outer fold is both selection and reported score — not a held-out test).
- `.provenance/` is refreshed by `scripts/capture_provenance.sh` on every cluster sync; don't hand-edit it.
