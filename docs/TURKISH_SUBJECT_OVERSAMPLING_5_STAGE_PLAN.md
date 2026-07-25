# Turkish BDI≥17 Subject-Level Oversampling: Five-Stage Plan

Status: decision-complete implementation and MN5 execution handoff.

This plan is for an agent that must implement, execute, monitor, retrieve, and
report the experiment. Finishing the code or submitting Slurm jobs is not the
terminal condition.

Before taking action, read these files completely:

1. `docs/DEVICES.md`
2. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`
3. This plan

The operational instructions in the first two files are mandatory. In
particular:

- use `ozu647717@transfer1.bsc.es` for rsync and GPFS file transfer;
- use `ozu647717@alogin1.bsc.es` for `sbatch`, `squeue`, and `sacct`;
- never run training or classifier experiments directly on a login or
  transfer node;
- run a smoke study before production;
- remain responsible for every submitted job until terminal Slurm accounting
  and artifact audits prove success;
- dry-run and then sync the exact results, manifests, audits, and logs back
  locally;
- update reports and tracked result files, then commit and push when
  authorized.

## 1. Motivation and current state

Turkish BDI≥17 contains 120 subjects:

```text
83 depressed / positive
37 non-depressed / negative
```

The minority class is therefore label `0`, not label `1`. Oversampling must
target non-depressed subjects.

The present code already has imbalance controls, but they are not equivalent
to pure subject oversampling:

- Qwen training uses `training.class_balance: weighted_sampler`.
- That sampler gives inverse-frequency row weights but keeps
  `num_samples=len(train_examples)`, so it combines minority oversampling with
  majority undersampling.
- Logistic regression uses `class_weight="balanced"`.
- Optuna XGBoost searches `scale_pos_weight`.

The hidden caches are response-level. One inspected Turkish outer-training
cache contains 95 subjects and 831 response rows:

```text
29 negative subjects / 252 negative rows
66 positive subjects / 579 positive rows
```

Individual subjects have between 5 and 19 responses. Independently
oversampling response rows would therefore distort subject contribution.
Complete subject groups must be duplicated together.

The best current Turkish hidden-state Macro-F1 is fixed raw text-only XGBoost
at `0.623`. Turkish audio-only has shown substantial tuning-seed sensitivity.
Macro-F1, negative F1, and negative recall—not positive F1 alone—must drive
this experiment.

Only Turkish BDI≥17 is in scope. Do not add or report BDI≥21.

## 2. Required sampling behavior

Add a shared subject-group oversampling implementation usable by Qwen training
and hidden-state classifiers.

For a training partition:

1. Validate binary labels and one consistent label per subject.
2. Count unique subjects by label.
3. Detect the minority label; assert that it is label `0` for this experiment.
4. Retain every original majority and minority subject once.
5. Compute:

   ```text
   target_minority_occurrences = ceil(ratio × majority_subject_count)
   ```

6. Sample only the additional required minority subject IDs with replacement.
7. For every selected subject occurrence, include all response/example rows
   belonging to that subject.
8. Shuffle the fixed expanded index multiset deterministically during
   training.

Supported ratios:

```text
0.75
1.00
```

Supported sampling seeds:

```text
7
1337
2024
```

Do not generate synthetic hidden vectors or use SMOTE. Do not oversample
inner-validation or outer-evaluation partitions.

Every fit must save a sampling audit containing:

- strategy and requested ratio;
- sampling seed;
- detected minority label;
- original/final subject-occurrence counts by class;
- original/final row counts by class;
- duplicate multiplicity for each subject;
- hashes of the source row/subject assignments;
- explicit confirmation that validation/evaluation indices were untouched.

## 3. Interfaces and compatibility

### Qwen training

Extend the existing class-balance interface:

```yaml
training:
  class_balance: minority_subject_oversample
  oversampling_ratio: 0.75
  oversampling_seed: 1337
```

Requirements:

- preserve existing `none` and `weighted_sampler` behavior;
- reject missing/out-of-range ratios for the new mode;
- reject inconsistent subject labels or a one-class training partition;
- keep the oversampled epoch length stable;
- preserve deterministic behavior for the same seed;
- write the sampling audit beside the resolved run configuration.

The matched Qwen control remains:

```yaml
training:
  class_balance: weighted_sampler
```

### Hidden-state classifiers

Add equivalent CLI/configuration arguments:

```text
--sampling-mode none|minority_subject_oversample
--oversampling-ratio 0.75|1.0
--oversampling-seed INT
```

For experiments intended to isolate oversampling:

- LogReg uses `class_weight=None`.
- XGBoost uses `scale_pos_weight=1`.
- No PCA is used.
- Threshold remains `0.5`.

Optuna continues using the standard d6 search ranges, but
`scale_pos_weight` is removed from the search and fixed to `1`.

### Identity and resumability

Sampling mode, ratio, and seed must be included in:

- output/experiment ID;
- configuration hash;
- SQLite study name;
- metadata;
- prediction rows;
- Slurm log directory.

Existing outputs must never be overwritten or resumed under an incompatible
sampling configuration.

Suggested identity patterns:

```text
hidden_os_screen_none
hidden_os_screen_ros075_os7
hidden_os_screen_ros100_os1337
xgb_optuna_raw_oscontrol_t100_d6_seed1337
xgb_optuna_raw_ros075_t100_d6_seed1337_os7
t17_selmacro_qwen3asr_weighted_<modality>
t17_selmacro_qwen3asr_ros075_os1337_<modality>
```

Use `ros100` instead of `ros075` if the full-balance profile wins.

## 4. Five experimental stages

Later GPU stages are evidence-gated. A skipped stage is a valid scientific
outcome when the preceding gate fails.

### Stage 1 — Implementation validation and MN5 smoke

Implement the shared sampling logic, Qwen integration, hidden-head
integration, manifests, Slurm wrappers, audits, and summarizers.

Local tests must cover:

- subject label consistency;
- exact target subject-occurrence ratio;
- every original subject retained once;
- only minority subjects duplicated;
- all responses from a duplicated subject share its multiplicity;
- deterministic equality for the same seed;
- controlled variation across seeds;
- no validation/evaluation modification;
- unchanged existing weighted-sampler behavior;
- invalid ratio/configuration rejection;
- output/configuration collision rejection.

Run MN5 smokes in unique directories:

- hidden no-sampling, `0.75`, and `1.00` profiles;
- one-epoch subject-limited Qwen weighted control;
- one-epoch subject-limited Qwen full-oversampling profile.

Monitor with `squeue`, `sacct`, and logs. Require `COMPLETED` and
`ExitCode=0:0`, then inspect sampling audits and normal artifacts. Test
idempotent restart where supported.

### Stage 2 — Hidden-head strategy screening

Use only each cache's `outer_train` partition. The stage must not load
`final_eval`.

Conditions:

- audio+text;
- audio-only;
- text-only;
- five outer caches per condition.

Heads:

- fixed raw LogReg with `class_weight=None`;
- fixed raw XGBoost with `scale_pos_weight=1`.

Profiles:

- no sampling once;
- ratio `0.75` with seeds `7`, `1337`, and `2024`;
- ratio `1.00` with seeds `7`, `1337`, and `2024`.

Use the same deterministic three subject-level inner folds for all profiles.
Score pooled subject-level inner-OOF Macro-F1.

Bundle the seven sampling profiles and two heads within each outer-evaluation
job:

```text
15 Slurm jobs
630 fixed-head inner fits
```

Choose one global ratio for Stage 3 using mean inner-OOF Macro-F1 across
modalities, outer caches, heads, and seeds. If the ratios differ by at most
`0.005`, select `0.75`.

Do not inspect outer-evaluation predictions when selecting the ratio.

### Stage 3 — Nested Optuna confirmation

Compare the selected ratio against a matched no-sampling XGBoost control.

Fixed settings:

```text
objective=binary:logistic
tree_method=hist
eval_metric=logloss
scale_pos_weight=1
threshold=0.5
TPE/XGBoost seed=1337
inner folds=3
trials=100
search profile=standard_d6
```

Studies:

- 15 no-sampling controls;
- 15 outer evaluations × three oversampling seeds = 45 oversampling studies.

Expected work:

```text
60 studies
6,000 Optuna trials
18,000 inner XGBoost fits
60 final fits
```

For every trial, oversample only the inner-training subjects, aggregate
validation responses to subjects, and optimize pooled inner-OOF Macro-F1.
After tuning, oversample the complete outer-training partition, refit once,
and evaluate the untouched outer fold once.

Proceed to Qwen only when at least one modality satisfies all of:

- mean pooled outer Macro-F1 gain across sampling seeds ≥ `0.02`;
- at least two of three sampling seeds beat the matched control;
- mean negative recall gain ≥ `0.05`;
- positive recall loss ≤ `0.10`;
- mean Macro-F1 across all three modalities is not more than `0.01` below the
  matched control.

If this gate fails, stop before GPU expansion and report that explicit
oversampling did not justify Qwen retraining.

### Stage 4 — Matched Qwen pilot

Select the qualifying Stage-3 modality with the largest mean Macro-F1 gain.
Resolve ties in this fixed order:

```text
audio-only
text-only
audio+text
```

Run folds `0` and `1` for:

- the existing weighted sampler;
- the selected pure subject-oversampling ratio with sampling seed `1337`.

This is four Qwen training jobs.

Both profiles must use:

- Qwen3ASR Turkish transcripts;
- identical BDI≥17 subject splits;
- model/training seed `1337`;
- `inner_val_macro_f1` checkpoint selection and early stopping;
- eight-epoch maximum;
- identical prompts, LoRA settings, evaluation, and threshold.

Do not use the existing Whisper-based Macro-F1 configs as the matched control.
Create Qwen3ASR Macro-F1 configurations based on the current reported
Qwen3ASR runs, changing only checkpoint-selection and sampling fields.

Proceed to Stage 5 only if:

- mean selected inner-validation Macro-F1 gain ≥ `0.015`;
- neither oversampled fold loses more than `0.03` inner-validation Macro-F1;
- pooled pilot outer Macro-F1 is not below control;
- pooled pilot negative recall is not below control.

### Stage 5 — Full Qwen confirmation and reporting

Complete matched five-fold experiments for all three modalities:

```text
weighted-sampler control: 15 runs
pure subject oversampling: 15 runs
total: 30 runs
```

Reuse the four Stage-4 jobs when their configuration hashes match exactly;
submit the remaining 26 runs.

Chain folds within each modality and allow independent modality chains to run
concurrently, subject to MN5 allocation policy. Estimate 12–20 GB of new GPFS
storage before submission.

After all jobs complete:

- verify every top-level and dependent job with `sacct`;
- scan logs for tracebacks, OOMs, timeouts, and failed dependencies;
- run remote completeness/leakage/provenance audits;
- dry-run and then selectively rsync results, audits, summaries, and logs
  through `transfer1`;
- do not download all best/last checkpoints unless explicitly requested;
- rerun audits and summaries locally.

## 5. Metrics, reporting, and acceptance

Primary metric:

```text
subject-level Macro-F1
```

Also report:

- negative F1;
- negative recall/specificity;
- positive F1 and recall;
- balanced accuracy;
- accuracy;
- AUROC;
- fold mean±SD;
- pooled confusion matrix;
- oversampling-seed range/SD;
- delta against the matched control and current published baselines.

Update:

- `qwen_hidden_best_results_no_emotion.csv`;
- the full Turkish oversampling report;
- `depression_results_combined_with_posf1_graphs.xlsx`, including its
  Macro-F1 Summary page;
- any compact no-emotion table used by the thesis.

Do not add BDI≥21 rows. Retain the warning that Turkish uses table-aligned
outer validation because its checkpoints and experiment development use those
validation folds.

Execution acceptance requires:

- exact expected study/job counts or a documented gate-based stop;
- zero subject leakage;
- complete inner OOF coverage;
- untouched validation/evaluation partitions;
- compatible configuration hashes;
- all required artifacts;
- no failed/missing studies or unaccounted Slurm jobs;
- results and logs synchronized locally;
- final report and tracked tables committed and pushed when authorized.

A Macro-F1 improvement is not required for technical completion. If
oversampling does not help, report the negative result without selecting a
favorable seed or hiding a failed modality.

## 6. Mandatory agent execution instruction

Give the implementing agent this instruction together with the plan:

```text
Read docs/DEVICES.md, docs/MN5_AGENT_EXECUTION_RUNBOOK.md, and
docs/TURKISH_SUBJECT_OVERSAMPLING_5_STAGE_PLAN.md completely.

Your task is operational, not advisory. Implement and locally validate the
five-stage Turkish BDI≥17 subject-oversampling experiment. Use transfer1 for
selective rsync and checksum verification, and alogin1 for Slurm submission
and monitoring. Do not run experiments on login or transfer nodes.

Run unique MN5 smoke jobs before production. Record every job ID and remain
responsible until squeue, sacct, exit codes, logs, and artifact audits prove a
terminal outcome. Apply every predeclared stage gate exactly; do not submit a
gated GPU stage when its gate fails.

After successful remote audits, dry-run and selectively rsync the exact
results, manifests, audits, summaries, and logs back locally without
--delete. Re-run audits and summarizers locally, update the CSV, report, and
professor-facing workbook, then commit and push the intended tracked
deliverables when authorized.

Do not stop after implementation, rsync, smoke submission, production
submission, or initial monitoring. Completion means every authorized job is
accounted for, artifacts and logs are local, audits pass, reporting is
updated, and the final Git state is reported.
```
