# Symmetric Five-Dataset Qwen and Hidden-Head Experiment Plan

## Summary

Implement and execute an end-to-end symmetric merged protocol for DAIC, CMDC,
Turkish BDI≥17, D3TEC, and Androids Interview across audio+text, audio-only,
and text-only modalities.

Each fold trains one Qwen model and its Logistic Regression, fixed XGBoost,
and 150-trial Optuna XGBoost heads on the same merged training pool. Every
dataset has equal training and validation influence. The executing agent must
continue through MN5 synchronization, smoke testing, production submission,
monitoring, result retrieval, auditing, workbook updates, and Git publication.

## Protocol and implementation

- Add a neutral `symmetric_merged` experiment configuration rather than
  exposing any dataset as primary.
- Run five outer folds. For every dataset, fold `k` is untouched outer
  evaluation data and the other four folds form the outer training pool.
- For DAIC, generate folds only from the 142 official train+validation
  subjects. Keep all 47 official-test subjects outside CV.
- Inside every outer-training pool, create deterministic, subject-stratified
  inner-validation assignments. Select Qwen checkpoints using the unweighted
  mean of the five datasets' inner-validation Macro-F1 values.
- Preserve each component's prompts, transcript limits, audio construction,
  label threshold, and evaluation hierarchy.
- Train with exhaustive dataset-scaled weighting. Every eligible training
  example is consumed exactly once per epoch without oversampling,
  undersampling, duplication, or class rebalancing. Preserve each dataset's
  natural class prevalence.
- Define the training objective as the unweighted mean of the five dataset
  losses. Within each dataset, give every subject equal total weight, divide a
  subject's weight equally across its responses, and divide each response's
  weight equally across its windows or segments. Normalize the resulting
  example weights to a global mean of one for stable optimization.
- Build deterministic, globally shuffled, dataset-aware gradient-accumulation
  blocks so optimizer steps receive approximately balanced dataset
  contributions despite batch-size-one microbatches. Interleave dataset queues
  while multiple queues remain, drain each tail without duplication, normalize
  accumulated loss by the sum of its example weights, and log the realized
  contribution by dataset for every epoch.
- Namespace identities as `dataset::subject_id` in merged artifacts and
  leakage checks.
- Use at most 20 Qwen epochs with early stopping on mean Macro-F1; retain
  per-dataset metrics at every epoch.
- For each selected fold checkpoint:
  - Produce direct teacher-forced Qwen predictions on all five outer
    holdouts.
  - Freeze Qwen and extract hidden vectors for the complete outer-training
    pool and all outer holdouts.
  - Fit one merged standardized Logistic Regression head and one merged fixed
    XGBoost head using the same exhaustive hierarchical sample weights.
  - Run one 150-trial XGBoost Optuna study using three subject-grouped inner
    folds and unweighted mean per-dataset Macro-F1 as the objective.
  - Refit the selected head on the full outer-training feature pool and
    evaluate each dataset separately.
- Use the existing fold-specific fine-tuned LoRA representation and current
  prompt-final-hidden-state pooling so the head results match the workbook
  methodology. A frozen-base-model control is out of scope.
- Keep the decision threshold at 0.5. Do not tune thresholds using outer-fold
  outcomes.
- Pool outer-fold predictions per dataset for headline CV metrics; also
  retain fold mean and standard deviation. Never pool all datasets into one
  confusion matrix.
- Report the unweighted five-dataset mean and worst-dataset Macro-F1 as
  cross-dataset summaries.

## DAIC official-test stage

- After all CV runs and audits pass, select the fixed final Qwen epoch count
  per modality as the rounded median selected epoch across its five CV folds.
- Train one final merged Qwen per modality without early stopping using:
  - all 142 DAIC train+validation subjects;
  - all available subjects from CMDC, Turkish, D3TEC, and Androids;
  - the same exhaustive dataset-scaled weighting and accumulation schedule.
- Use this final model only for the DAIC official-test result; do not report
  its predictions on datasets whose subjects entered final training.
- Extract final training and DAIC-test features, train final Logistic
  Regression and fixed XGBoost heads, and run a fresh 150-trial grouped Optuna
  search entirely within the merged non-test feature pool.
- Evaluate Qwen and all heads exactly once on the untouched DAIC official
  test.

## Interfaces and artifacts

- Add three merged configs for `audio_text`, `audio_only`, and `text_only`,
  sharing a component list and protocol settings.
- Add a staged submission interface supporting `--stage smoke|cv|final` and
  `--dry-run`, deterministic experiment IDs, collision checks, restart-safe
  skipping, and a machine-readable job registry.
- Add a merged postprocessing worker that loads each checkpoint once,
  evaluates Qwen, extracts all component features, and records
  checkpoint/config/manifest/fold hashes.
- Add a merged head worker that runs Logistic Regression, fixed XGBoost, and
  resumable 150-trial Optuna XGBoost.
- Emit per-run composition, weighting/schedule, split/leakage, selection-history,
  predictions, metrics, feature-provenance, classifier, and Slurm provenance
  artifacts.
- Add an acceptance audit that requires complete
  dataset/fold/modality/method coverage, disjoint train/inner/outer subjects,
  untouched DAIC test data, valid hashes, 150 completed Optuna trials, and no
  missing or failed jobs.

## MN5 execution

- Follow `docs/DEVICES.md` and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`.
- Validate locally with unit tests, syntax checks, config parsing, dry-run job
  counts, and a tiny synthetic end-to-end merged pipeline.
- Commit and push the tested implementation to `main`, capture provenance,
  and verify a clean intended source scope.
- Recheck `transfer1` and `alogin1` connectivity.
- Dry-run a selective rsync of only changed source, configs, scripts, tests,
  and provenance to
  `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression`.
- Perform the real transfer through `ozu647717@transfer1.bsc.es`, never with
  `--delete`, then compare local and remote SHA-256 checksums.
- On `ozu647717@alogin1.bsc.es`, verify modules, environment, models, all five
  dataset roots, dependencies, Slurm directives, free space, and dry-run
  output.
- Submit one uniquely named audio+text fold-0 smoke chain:
  - one epoch;
  - two subjects per class per dataset;
  - two Optuna trials;
  - real train, postprocess, and head workers.
- Monitor the three smoke jobs through `squeue`, `sacct`, logs, exit codes,
  and artifact audits. Reinvoke the smoke to verify completed outputs are
  skipped or resumed rather than duplicated.
- Submit CV only after smoke acceptance:
  - 15 four-GPU Qwen training jobs: 3 modalities × 5 folds;
  - 15 dependent one-GPU evaluation/extraction jobs;
  - 15 dependent 20-CPU head jobs;
  - 45 CV production jobs total.
- Monitor every known job ID until terminal. Require top-level `COMPLETED` and
  `ExitCode=0:0`; inspect warnings and errors and validate artifacts. Retry
  only demonstrated failed scope without resubmitting healthy work.
- Run the complete remote CV audit before starting the result-dependent final
  stage.
- Submit the final DAIC stage:
  - 3 four-GPU final Qwen jobs;
  - 3 dependent one-GPU DAIC evaluation/extraction jobs;
  - 3 dependent 20-CPU head jobs;
  - 9 final production jobs.
- Total expected production scope is 54 jobs, plus the three-job smoke chain.
  Exact resources and wall times must be confirmed after smoke runtime and
  memory observations; initial caps are 72 hours for Qwen training, 48 hours
  for GPU postprocessing, and 12 hours for the 150-trial CPU head worker.
- Maintain a job registry with job IDs, dependencies, modality, fold/stage,
  paths, submission times, states, and retries. Continue monitoring until all
  authorized work and audits are complete.

## Result synchronization and reporting

- After remote audits pass, dry-run and then selectively rsync results through
  `transfer1`.
- Retrieve metrics, predictions, selection histories, audits, resolved
  configs, manifests, provenance, compact summaries, and Slurm logs.
- Do not retrieve `best_model`, `last_model`, Safetensors, PyTorch weights,
  hidden-feature arrays, classifier joblibs, or Optuna SQLite databases. Leave
  heavy artifacts authoritative on GPFS.
- Never use `--delete`; verify transferred hashes and rerun the acceptance
  audit locally. Because the approved sync excludes checkpoints, dense hidden
  feature arrays, and classifier joblibs, invoke the local audit with
  `--allow-omitted-heavy-artifacts`; remote audits remain strict by default.
- Create tracked CSVs for:
  - pooled symmetric CV results by dataset, modality, and method;
  - fold-level mean and standard deviation;
  - five-dataset aggregate and worst-dataset results;
  - DAIC official-test results.
- Update `depression_results_combined_with_posf1_graphs.xlsx` with dedicated
  `Merged Symmetric CV` and `Merged DAIC Official` sheets while preserving
  existing sheets, tables, formulas, and charts.
- Include Accuracy, Positive F1, Precision, Recall, Macro-F1, Negative F1,
  AUROC where available, class supports, confusion matrix, invalid Qwen
  outputs, fold coverage, and protocol labels.
- Add a Markdown execution/results report with experiment IDs, Git commit,
  job IDs, runtime/storage, audit outcome, results, and limitations.
- Validate the workbook with `openpyxl`, compare workbook values with CSVs,
  and test table and filter ranges.
- Commit and push the returned tables, workbook, report, audit summaries, and
  execution metadata. Do not force-add ignored raw outputs.

## Tests and acceptance

- Verify deterministic five-fold coverage and exact one-time outer-holdout
  membership for every development subject.
- Verify DAIC official-test subjects never enter CV training, validation,
  feature fitting, or Optuna.
- Verify every eligible example appears exactly once per epoch and no example
  is duplicated or omitted.
- Verify the normalized loss weights sum equally by dataset, sum equally by
  subject within each dataset, and correctly divide subject weight across
  responses and windows while preserving natural class ratios.
- Verify dataset-aware accumulation is deterministic and its realized
  per-epoch dataset contributions match the configured tolerance.
- Verify checkpoint selection is the arithmetic mean of five per-dataset
  Macro-F1 values, independent of dataset size.
- Verify hidden caches and heads reject mismatched checkpoint, fold, modality,
  manifest, or feature dimensions.
- Verify head inner folds are subject-disjoint and outer labels are
  inaccessible during tuning.
- Verify CV pooled metrics are reconstructed from five non-overlapping
  prediction files.
- Verify final retraining uses the frozen median epoch and evaluates only DAIC
  official test.
- Verify dry-run counts are 45 CV jobs and 9 final jobs and that reruns skip
  compatible completed artifacts.
- Completion requires all jobs accounted for, remote and local audits passing,
  results and logs present locally, workbook and CSVs updated, no weights
  synchronized locally, and final commits pushed.

## Assumptions

- All three modalities and all four method columns are required.
- XGBoost Optuna uses 150 trials and three grouped inner folds.
- Mean Macro-F1 is the sole shared selection and tuning objective.
- Exhaustive dataset-scaled training is the only production weighting policy;
  class-balanced sampling and pilot challengers are out of scope.
- The final DAIC model uses all available non-test data from all five
  datasets.
- Existing MN5 account `etur92`, QoS `acc_ehpc`, environments, dataset roots,
  and base-model paths remain valid; preflight must verify them before
  submission.
- Remote checkpoints and feature caches are retained on GPFS but excluded
  from return synchronization.
