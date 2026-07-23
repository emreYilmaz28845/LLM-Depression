# Optuna-Tuned Raw XGBoost Hidden-State Experiment

Status: implementation handoff; not yet implemented or submitted.

Last updated: 2026-07-23 (Europe/Istanbul).

## Purpose

Add leakage-safe Optuna hyperparameter tuning for the existing raw Qwen
hidden-state XGBoost classifiers. The experiment covers the no-emotion
audio+text, audio-only, and text-only conditions for DAIC, CMDC, and Turkish.
It reuses the hidden-vector caches already stored on GPFS and must not extract
Qwen features, retrain Qwen, request a GPU, add PCA, or include an emotion
condition.

Read [`docs/DEVICES.md`](DEVICES.md) before running anything locally or on
MareNostrum 5 (MN5). It contains the authoritative host roles, repository
paths, environment activation, transfer commands, scheduler cautions, and
authorization boundaries. Also read
[`docs/ENVIRONMENT_NOTES.md`](ENVIRONMENT_NOTES.md) before changing Python
dependencies.

## Experiment contract

Run 33 independent outer evaluations:

| Dataset | Modalities | Outer folds per modality | Evaluations | Tuning objective |
|---|---:|---:|---:|---|
| DAIC | 3 | 1 | 3 | positive F1 |
| CMDC | 3 | 5 | 15 | positive F1 |
| Turkish | 3 | 5 | 15 | macro-F1 |
| **Total** |  |  | **33** |  |

Each outer evaluation uses 50 sequential Optuna trials. Each trial performs
three inner, subject-level fits, so the production run contains:

- 1,650 completed trials;
- 4,950 inner XGBoost fits;
- 33 final XGBoost fits; and
- no Qwen inference or training.

DAIC and CMDC use positive F1 to match the repository's headline convention.
Turkish uses macro-F1 because its subject distribution is 83 positive and 37
negative. The Turkish results remain table-aligned outer-validation estimates,
not unseen-test estimates: the underlying Qwen checkpoints were selected using
those validation folds. Preserve that warning in every report.

## Repository facts established before implementation

The existing relevant code is:

- `baselines/qwen_hidden_classifier.py`: loads `outer_train` and `final_eval`,
  fits fixed classifiers, writes per-fold artifacts, and defines the current
  XGBoost defaults.
- `src.aggregate.aggregate_binary_classifier_predictions`: the required
  response-to-subject aggregation. It uses majority vote over thresholded
  response predictions and a summed probability-margin tie break. Do not
  replace this with mean-probability thresholding or response-level scoring.
- `baselines/summarize_qwen_hidden.py`: discovers
  `.../fold_*/*/metrics.json`, pools subject predictions, checks held-out
  subject uniqueness, and emits fold mean/SD plus pooled metrics.
- `tests/test_qwen_hidden_pipeline.py`: the current hidden-state regression
  tests.
- `configs/features/primary_matrix.yaml`: 18 no-emotion DAIC/CMDC cache
  entries.
- `configs/features/turkish_matrix.yaml`: 15 no-emotion Turkish cache entries.
- `scripts/submit_qwen_hidden_matrix.sh` and
  `scripts/run_qwen_hidden_extract_slurm.sh`: extraction-oriented GPU paths.
  They must not be reused for this experiment because they validate
  checkpoints, run Qwen extraction, and request a GPU.
- `scripts/run_qwen_hidden_classifiers.sh`: a useful cache-to-classifier
  wrapper pattern, but it does not tune.
- `requirements_hidden_features.txt`: currently pins only
  `xgboost-cpu==2.1.4`.

The local checkout contains synced classifier results but no
`outputs/hidden_features` directory. The existing hidden matrices on GPFS are
the authoritative inputs, and Turkish matrices are cluster-side. Therefore,
develop and unit-test locally with synthetic caches, but run the real smoke and
production studies on MN5 CPU nodes.

The current local branch was `main` at `bb4504e`, aligned with `origin/main`
when this handoff was written. The worktree already had unrelated user changes:

```text
 D configs/main/turkish_t17_audio_only_selposf1_tf.yaml
 D configs/main/turkish_t17_audio_text_selposf1_tf.yaml
 D configs/main/turkish_t17_text_only_selposf1_tf.yaml
 D configs/main/turkish_t21_audio_only_selposf1_tf_qwen3asr.yaml
 D configs/main/turkish_t21_audio_text_selposf1_tf_qwen3asr.yaml
 D configs/main/turkish_t21_text_only_selposf1_tf_qwen3asr.yaml
 M scripts/run_daic_reg_sweep_sequential.sh
?? configs/archive/turkish/turkish_t17_audio_only_selposf1_tf.yaml
?? configs/archive/turkish/turkish_t17_audio_text_selposf1_tf.yaml
?? configs/archive/turkish/turkish_t17_text_only_selposf1_tf.yaml
?? configs/archive/turkish/turkish_t21_audio_only_selposf1_tf_qwen3asr.yaml
?? configs/archive/turkish/turkish_t21_audio_text_selposf1_tf_qwen3asr.yaml
?? configs/archive/turkish/turkish_t21_text_only_selposf1_tf_qwen3asr.yaml
```

Re-check `git status --short --branch` because this list can become stale.
Preserve these changes and exclude them from experiment commits unless the user
explicitly says otherwise. The repository `.gitignore` ignores `docs/`, so
this handoff itself may require `git add -f
docs/OPTUNA_TUNED_RAW_XGBOOST_PLAN.md` if it is to be committed.

## Proposed files and interfaces

Use names consistent with the repository, unless inspection during
implementation reveals a better established convention:

```text
baselines/qwen_hidden_xgb_optuna.py
configs/features/optuna_raw_matrix.yaml
scripts/run_qwen_hidden_optuna_slurm.sh
scripts/submit_qwen_hidden_optuna_matrix.sh
tests/test_qwen_hidden_optuna.py
```

Extend `baselines/summarize_qwen_hidden.py` rather than creating a disconnected
report format. Add Optuna to a dedicated dependency file or to
`requirements_hidden_features.txt` only after checking how the MN5 environment
is assembled. The intended cluster versions are Optuna 4.4.0 and XGBoost
2.1.4.

The tuning CLI should accept:

```text
--cache-dir PATH
--output-dir PATH
--objective {positive_f1,macro_f1}
--target-trials INT        # default 50
--inner-folds INT          # default 3
--seed INT                 # default 1337
--xgb-threads INT          # default should match the Slurm allocation
```

The classifier variant and output directory name must be
`xgb_optuna_raw`.

## Leakage-safe implementation design

### 1. Validate and split only the outer-training partition

The objective must load only:

```text
outer_train.npz
outer_train_rows.jsonl
extraction_metadata.json
```

Do not load, memory-map, hash, inspect, or otherwise touch
`final_eval.npz`/`final_eval_rows.jsonl` while creating the study, constructing
inner folds, or executing trials.

Build a single label per subject from `outer_train_rows.jsonl`. Reject
inconsistent labels within a subject. Sort subject IDs before splitting so
input row order cannot change assignments. Use
`sklearn.model_selection.StratifiedKFold` with:

```python
n_splits=3
shuffle=True
random_state=1337
```

Validate that each class has at least three subjects. Map each subject fold
back to all feature-row indices for that subject. Every response belonging to a
subject must remain on the same side of an inner train/validation boundary.

Persist the assignments once, before tuning. They must demonstrate:

- pairwise-disjoint validation subject sets;
- no subject overlap between inner train and validation;
- every outer-training subject appears in validation exactly once; and
- every response row maps to its subject's fold.

### 2. Execute one pooled subject-level OOF objective per trial

Use `optuna.samplers.TPESampler(seed=1337)` and run trials sequentially within
each SQLite study (`n_jobs=1` at the Optuna level). The 33 Slurm jobs provide
the experiment-level parallelism.

For each trial:

1. Suggest one parameter set from the declared search space.
2. For each of the three inner folds, fit on the rows belonging to the other
   two subject folds.
3. Predict validation response probabilities.
4. Convert response probabilities to classes at the fixed threshold `0.5`.
5. Form rows compatible with
   `aggregate_binary_classifier_predictions`.
6. Aggregate each validation fold from responses to subjects using that
   existing function.
7. Append the fold's subject predictions to a trial-wide OOF collection.
8. After all three folds, assert that each outer-training subject appears
   exactly once in the subject OOF collection.
9. Compute the objective from the pooled subject predictions, not by averaging
   three fold scores and not from response-level predictions.

Use `classification_metrics` for the pooled positive F1 or macro-F1. Record
per-inner-fold metrics and pooled OOF metrics in trial user attributes as
JSON-serializable values. This allows the winning trial's diagnostics to be
saved without refitting its three inner models. Do not recompute the best
trial after tuning: doing so would exceed the accepted count of 4,950 inner
fits.

### 3. XGBoost configuration

Search exactly this space:

| Parameter | Optuna suggestion |
|---|---|
| `n_estimators` | integer 100 to 1,000, step 50 |
| `learning_rate` | float 0.005 to 0.2, log |
| `max_depth` | integer 1 to 6 |
| `min_child_weight` | float 0.5 to 20, log |
| `subsample` | float 0.5 to 1.0 |
| `colsample_bytree` | float 0.1 to 1.0 |
| `gamma` | float `1e-8` to 5, log |
| `reg_alpha` | float `1e-8` to 20, log |
| `reg_lambda` | float `1e-3` to 50, log |
| `scale_pos_weight` | float 0.25 to 4.0, log |

Fixed arguments:

```python
objective="binary:logistic"
tree_method="hist"
eval_metric="logloss"
random_state=1337
n_jobs=<CLI xgb-threads>
```

Do not add early stopping, threshold selection, PCA, standardization, automatic
class weighting, GPU tree methods, or any parameter not declared above. Keep
the prediction threshold at 0.5.

### 4. Resume safely

Create one SQLite database per outer fold, inside that fold's
`xgb_optuna_raw` directory. Use a deterministic study name and
`load_if_exists=True`.

Create a canonical JSON configuration containing at least:

- a schema/version identifier;
- dataset, modality/condition, and outer fold;
- cache identity and extraction metadata identity;
- objective name;
- target trial count;
- inner-fold count and seed;
- aggregation method name and threshold;
- all fixed XGBoost parameters;
- complete search-space names, types, bounds, steps, and log flags; and
- relevant package versions.

Hash the canonical JSON with SHA-256. Store both the canonical configuration
and hash in a sidecar JSON file and in Optuna study user attributes. Validate
them before resuming. Refuse to resume if either differs; do not silently reuse
a database created under another objective or search space.

`target_trials=50` means 50 trials in Optuna's `COMPLETE` state total:

```python
remaining = target_trials - completed_trial_count
```

Failed, pruned, waiting, and interrupted/running trials do not count as
completed. Run only `remaining` new trials. Refuse a target lower than the
already-completed count, since it cannot satisfy exact-total semantics. At
normal completion, assert that the completed count is exactly the target.

One job owns one SQLite file. Never point multiple jobs at the same database.

### 5. Final fit and outer evaluation

Only after tuning has reached its target:

1. Obtain the best completed trial.
2. Fit one XGBoost classifier with its parameters on every
   `outer_train` response and subject.
3. Only now load `final_eval.npz` and `final_eval_rows.jsonl`.
4. Verify outer-training and final-evaluation subject sets are disjoint.
5. Predict final-evaluation responses once at threshold 0.5.
6. Apply `aggregate_binary_classifier_predictions`.
7. Write artifacts in the same field conventions used by
   `qwen_hidden_classifier.py`.

Keep tuning and final-evaluation functions separate enough that a unit test can
call or monkeypatch the objective and prove `_load_partition(...,
"final_eval")` is never invoked. A crash after tuning but before final
evaluation should be safely restartable: validate the study, skip additional
trials when 50 are complete, then rebuild the final model and artifacts.

## Artifact contract

For every path of the form:

```text
outputs/hidden_classifiers/<dataset>/<condition>/<run_name>/fold_<n>/xgb_optuna_raw/
```

write:

```text
study.sqlite3
study_config.json
trials.csv
best_params.json
inner_subject_assignments.json
inner_fold_metrics.json
inner_oof_metrics.json
pipeline.joblib
predictions_sample_level.jsonl
predictions_sample_level.csv
predictions_subject_level.jsonl
predictions_subject_level.csv
metrics.json
classifier_metadata.json
```

`trials.csv` should include all Optuna trial states and parameters, not only
completed trials. `best_params.json` should include the best trial number,
objective name/value, suggested parameters, fixed parameters, and completed
trial count. The inner metrics files come from the winning trial's recorded
attributes.

Metadata must include enough provenance to audit:

- dataset, modality, condition, outer fold, and run name;
- `classifier_variant: xgb_optuna_raw`;
- input dimension and `post_pca_dimension` equal to the input dimension;
- threshold, seed, inner-fold count, and target/completed trials;
- objective and best value;
- search/configuration hash;
- Optuna and XGBoost versions;
- all outer-training row and subject IDs;
- all held-out row and subject IDs;
- zero outer subject overlap;
- inner subject coverage and assignments;
- cache and extraction-metadata paths; and
- the original extraction metadata/evaluation protocol, especially the
  Turkish table-aligned warning.

Write artifacts atomically where practical so a preempted job does not leave a
valid-looking partial JSON/CSV.

## Matrix and Slurm design

Create a new no-emotion matrix by combining the experiments in
`primary_matrix.yaml` and `turkish_matrix.yaml`. Make the objective explicit on
every matrix item:

- `positive_f1` for all DAIC and CMDC entries;
- `macro_f1` for all Turkish entries.

Do not include the emotion matrices. Derive cache and classifier paths exactly
as the current submission script does:

```text
cache =
  outputs/hidden_features/<dataset>/<condition>/<basename(run_dir)>/fold_<fold>

output =
  outputs/hidden_classifiers/<dataset>/<condition>/<basename(run_dir)>/fold_<fold>/xgb_optuna_raw
```

The matrix expands to exactly:

- DAIC: 3 modalities × fold 0 = 3 jobs;
- CMDC: 3 modalities × folds 0–4 = 15 jobs;
- Turkish: 3 modalities × folds 0–4 = 15 jobs.

The submitter should validate `outer_train.npz`,
`outer_train_rows.jsonl`, `final_eval.npz`,
`final_eval_rows.jsonl`, and `extraction_metadata.json` before submission. It
must not require a Qwen checkpoint or invoke the extraction launcher. A
`DRY_RUN=1` execution must print exactly 33 `sbatch` commands and submit
nothing. Add a machine-readable count or final assertion so an accidental
matrix change cannot silently alter the job count.

The Slurm worker is CPU-only:

```text
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
```

Do not include `--gres=gpu`, `--gpus`, or any GPU partition request. Initialize
the runtime according to `docs/DEVICES.md`, set `xgb-threads=20`, print package
versions and the exact command, and log stdout/stderr under a dedicated
directory such as `logs/slurm_qwen_hidden_optuna/`. Avoid `nvidia-smi` in this
CPU worker.

## Environments and how to run

These examples are a handoff template. Recheck paths and package versions
before using them.

### Local development and tests

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
hostname
git status --short --branch
source /home/emre/miniconda3/etc/profile.d/conda.sh
conda activate llmdep4090
python -V
python -c "import numpy, sklearn, optuna, xgboost; print(numpy.__version__, sklearn.__version__, optuna.__version__, xgboost.__version__)"
```

If Optuna/XGBoost are absent or incompatible, do not mutate a shared
environment casually. Prefer a task-specific virtual/Conda environment or the
repository's target-directory dependency pattern, document it, and pin
Optuna 4.4.0 plus XGBoost 2.1.4.

Run focused and regression tests:

```bash
python -m unittest tests.test_qwen_hidden_optuna -v
python -m unittest tests.test_qwen_hidden_pipeline -v
```

A synthetic smoke cache should contain repeated responses per subject in both
classes, separate outer-training/final-evaluation subjects, and the same
`*.npz`, `*_rows.jsonl`, and `extraction_metadata.json` layout as a real cache.
Run two trials and then rerun the same command to prove it remains at two
completed trials:

```bash
python baselines/qwen_hidden_xgb_optuna.py \
  --cache-dir /tmp/<synthetic-cache> \
  --output-dir /tmp/<synthetic-result>/xgb_optuna_raw \
  --objective positive_f1 \
  --target-trials 2 \
  --inner-folds 3 \
  --seed 1337 \
  --xgb-threads 2
```

Also rerun with one deliberately incompatible option (for example,
`--objective macro_f1`) and assert that the configuration-hash check rejects
the database.

### Transfer endpoint and MN5

The known transfer endpoint and GPFS project path are:

```text
ozu647717@transfer1.bsc.es
/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
```

`transfer1` is for inspection and file transfer. Do not run Python training
there. It may expose `sbatch`, but that does not prove it is connected to the
intended compute scheduler. Follow `docs/DEVICES.md`: verify the scheduler on
the user's designated MN5 login host and do not guess a login hostname.

The expected compute-node initialization is:

```bash
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate
cd /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import optuna, xgboost, sklearn; print(optuna.__version__, xgboost.__version__, sklearn.__version__)"
```

The previous experiment assumptions say the existing environment has Optuna
4.4.0 and XGBoost 2.1.4. Verify this on MN5; do not trust the copied local
environment. Note that `.deps/qwen_hidden` can shadow packages from the
activated environment.

### Selective code synchronization

Do not use a broad sync while the worktree contains unrelated changes. After
committing the implementation, first inspect both sides and dry-run a
selective transfer of only the committed files. A typical shape is:

```bash
rsync -avzn --relative \
  baselines/qwen_hidden_xgb_optuna.py \
  baselines/summarize_qwen_hidden.py \
  configs/features/optuna_raw_matrix.yaml \
  scripts/run_qwen_hidden_optuna_slurm.sh \
  scripts/submit_qwen_hidden_optuna_matrix.sh \
  tests/test_qwen_hidden_optuna.py \
  requirements_hidden_features.txt \
  ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/
```

Review the dry run, remove files that were not actually changed, then repeat
without `-n`. If another route or login host is required, use the instructions
in `docs/DEVICES.md`. Never add `--delete`.

### Cluster smoke

Before production, run a disposable two-trial study against one real cache
known to contain repeated responses per subject. Use a separate smoke output
directory/database, not a future production directory. Verify:

- the worker has no GPU request;
- two trials complete;
- each trial has three inner fits;
- the rerun adds zero trials;
- subject assignments and pooled OOF coverage are complete;
- final evaluation happens only after both trials;
- all artifacts load; and
- logs contain no traceback.

The concrete command should use the same CLI shown above, with the real GPFS
cache and `--xgb-threads 20`.

### Production dry run and submission

On the correct MN5 submission host:

```bash
cd /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
DRY_RUN=1 bash scripts/submit_qwen_hidden_optuna_matrix.sh
```

Count the emitted `sbatch` commands and require exactly 33. Inspect representative
DAIC, CMDC, and Turkish commands for objective, fold, cache, and output paths.
Then, only with the user's authorization for the compute spend:

```bash
bash scripts/submit_qwen_hidden_optuna_matrix.sh | tee optuna_submission_job_ids.tsv
```

Record all 33 job IDs. Monitor with `squeue` and `sacct`; inspect the dedicated
logs rather than assuming disappearance from `squeue` means success. Do not
resubmit a failed job blindly: inspect its SQLite state and configuration hash,
then resume the same fold if safe.

## Test plan

Add focused tests that prove:

1. Inner folds are deterministic for the same seed and stable under input-row
   reordering.
2. Validation subject sets are disjoint and cover every outer-training subject
   exactly once.
3. Subject-label stratification is preserved as closely as
   `StratifiedKFold` permits.
4. No response from a subject crosses an inner train/validation boundary.
5. Inconsistent labels within one subject are rejected.
6. The objective never loads `final_eval` vectors or labels.
7. A constructed repeated-response example gives a different response-level
   score from the repository aggregation, and the returned objective matches
   the subject-level value.
8. DAIC/CMDC map to `positive_f1`; Turkish maps to `macro_f1`, with explicit
   matrix values also validated.
9. Every sampled search parameter is within its declared range and integer
   steps are respected.
10. Restarting a two-trial study does not add two more trials; the same logic
    reaches exactly 50 completed trials in a cheap/mocked test.
11. Failed/incomplete trials do not count toward the target.
12. An incompatible configuration hash is rejected before optimization.
13. Final fitting uses every outer-training subject.
14. Final-evaluation loading and scoring occur only after tuning completes.
15. Outer train/evaluation subject overlap is rejected.
16. The summarizer discovers `xgb_optuna_raw` beside `xgb_raw` and
    `logreg_raw`.
17. Existing `tests.test_qwen_hidden_pipeline` continues to pass.
18. The matrix dry run emits exactly 33 CPU-only jobs, 18
    positive-F1 and 15 macro-F1, with no emotion condition or GPU flag.

Use small synthetic matrices and monkeypatched/fake estimators where possible
so unit tests do not execute hundreds of real XGBoost fits.

## Result synchronization and reporting

After every production job succeeds, sync only the new tuned result directories
and Optuna logs back to local storage. Because `outputs/` and `logs/` are
ignored, rsync is appropriate; do not try to commit raw study artifacts.
Follow the BSC-to-local patterns in `docs/DEVICES.md`, narrowing include rules
to `xgb_optuna_raw/` and `slurm_qwen_hidden_optuna/` where practical.

Audit before summarizing:

- 33 SQLite studies are present;
- each has exactly 50 completed trials;
- total completed trials are 1,650;
- every fold has the complete artifact set;
- every metadata file reports zero outer subject overlap;
- every best-trial OOF artifact covers all outer-training subjects once;
- DAIC/CMDC objectives are positive F1 and Turkish objectives are macro-F1;
- all final predictions and metrics parse;
- there are no failed jobs, tracebacks, missing folds, or OOM events.

Then run the extended hidden-classifier summarizer. Update:

- `qwen_hidden_best_results_no_emotion.csv`, the compact no-emotion table; and
- the full no-emotion report/table used by the current repository, preserving
  its existing formatting and provenance language.

Report both ranking and supporting metrics:

- Turkish primary ranking: macro-F1;
- DAIC/CMDC primary ranking: positive F1;
- accuracy, positive F1, macro-F1, precision, recall, negative F1, and AUROC;
- fold mean ± sample SD;
- pooled subject-level confusion matrices;
- deltas against fixed `xgb_raw`, `logreg_raw`, Qwen, and majority controls.

Do not compare response-level metrics against subject-level metrics. Keep model
family/variant labels unambiguous so tuned XGBoost appears beside, rather than
overwriting, fixed XGBoost.

## Git workflow

Implementation and report updates are two reviewable stages:

1. Implement code, tests, matrix, and scripts; run local validation.
2. Stage only intended files, inspect the staged diff, commit, and push to
   `main` without force-pushing.
3. Selectively sync that exact implementation to MN5.
4. Smoke, dry-run, submit, monitor, and sync results.
5. Update the compact CSV and full report locally.
6. Stage only report-related changes, inspect them, commit, and push to
   `main`.

Before either push:

```bash
git status --short --branch
git diff --check
git diff --cached --stat
git diff --cached
git fetch origin
```

Stop for review if `origin/main` moved incompatibly. Do not force-push, reset
the worktree, or include the unrelated Turkish archival/config changes listed
above.

## Completion checklist

- [ ] Tuner uses only `outer_train` during all Optuna trials.
- [ ] Three deterministic, stratified, subject-disjoint inner folds.
- [ ] Pooled subject-level OOF objective uses existing aggregation.
- [ ] Exact search space, fixed threshold, and fixed XGBoost settings.
- [ ] One resumable, hash-validated SQLite database per outer evaluation.
- [ ] `target_trials` counts total completed trials.
- [ ] Final model trains on all outer-training data.
- [ ] `final_eval` is loaded and evaluated only after tuning.
- [ ] Full artifact/provenance contract is present.
- [ ] Summarizer includes `xgb_optuna_raw`.
- [ ] Focused and existing hidden-classifier tests pass.
- [ ] CPU-only Slurm worker and exactly 33 dry-run jobs.
- [ ] Two-trial repeated-response smoke passes and resumes without new trials.
- [ ] 33 production jobs complete: 1,650 trials and 4,983 total XGBoost fits.
- [ ] Results/logs are synced and audited locally.
- [ ] Compact CSV and full report are updated with dataset-appropriate ranking.
- [ ] Implementation and report commits are pushed to `main` without including
      unrelated worktree changes.
