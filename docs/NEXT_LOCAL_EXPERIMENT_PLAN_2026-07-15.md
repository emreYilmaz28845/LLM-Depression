# Conversation Summary and Next Local Experiment Plan

Date: 2026-07-15

Repository: `/home/emre/Projects/AudioLLM/LLM-Depression`

Prior detailed results: [`LOCAL_REPAIR_RUN_FINDINGS_2026-07-14.md`](LOCAL_REPAIR_RUN_FINDINGS_2026-07-14.md)

## 1. Conversation summary

The work started from the forensic plan in [`investigation_plan.md`](investigation_plan.md).
The repository, saved results, data construction, evaluation code, disk usage, and locally
available compute were investigated. The user then authorized conservative disk cleanup,
creation of an isolated environment, model/tool downloads, implementation of the suggested
measurement repairs, and execution of the locally feasible experiments.

Completed work:

1. Created the isolated environment `/home/emre/miniconda3/envs/llmdep4090` without modifying
   the existing environments.
2. Repaired deterministic evaluation behavior and explicit CUDA BF16 loading.
3. Ran the legacy E0 real/silence/audio-shuffle/transcript-shuffle controls.
4. Replaced the fragile single audio view with eight distinct numeric-balanced K=4 views and
   ran all eight conditions for all eight views.
5. Aggregated subject margins across views and computed 10,000-repetition paired subject
   bootstrap confidence intervals.
6. Ran fixed-split subject-level eGeMAPSv02 and frozen WavLM Base+ acoustic ceilings.
7. Attempted one exact full K=4 Qwen2-Audio forward/backward audit. It exceeded the RTX 4090's
   24 GiB VRAM and wrote a no-fallback OOM artifact.
8. Verified the implementation with 54 unit tests, WavLM no-model tests, compilation, shell
   syntax checks, provenance checks, and an independent audit of all 3,008 eight-view
   prediction rows.

Current storage/compute position:

- GPU: RTX 4090, 24 GiB.
- Free root-filesystem space after cleanup and completed runs: approximately 77 GiB.
- Incremental environment, WavLM model, openSMILE, and canonical outputs: approximately
  6.40 GiB.
- Qwen2-Audio inference fits locally; the exact existing rank-16 checkpoint backward does not.

## 2. Key experimental outcome

The checkpoint is primarily text-driven.

- Transcript shuffling produced a large, statistically robust loss: eight-view first-token
  AUROC delta `0.3377`, 95% CI `[0.1286, 0.5395]`; balanced-accuracy delta `0.2338`,
  `[0.0833, 0.3941]`.
- Real audio changed predictions, but its advantage over silence and across-subject shuffled
  audio was not stable. Their correct-class margin, AUROC, and balanced-accuracy confidence
  intervals crossed zero.
- Same-class shuffled audio performed almost like real audio. This is compatible with a
  class-correlated acoustic, source, or preprocessing shortcut, but does not identify which
  shortcut is responsible.
- The fixed-split acoustic ceilings were effectively null: eGeMAPS AUROC `0.5000` and WavLM
  AUROC `0.5195`, with wide confidence intervals and non-significant shuffled controls.

Under the preregistered E0 rule, audio is **weakly used**: it is not literally ignored, but a
reliable benefit from correctly matched audio has not been demonstrated. These results do not
prove that DAIC audio contains no depression-related information; they only show that the
current checkpoint and the two tested fixed-split representations have not established it.

## 3. What can be done locally

| Recommendation | Local feasibility | Decision |
|---|---|---|
| Cross-fold eGeMAPS and WavLM ceilings | Fully feasible | Do next |
| Small subject-level attention/gated-MIL pooling | Fully feasible | Do after linear cross-fold baselines |
| Statistical gate using OOF predictions and shuffled controls | Fully feasible | Mandatory |
| Audit current audio quality/provenance | Partly feasible | Inspect what exists; rebuilding remains blocked |
| Rebuild participant-only timestamped audio | Blocked | Requires raw timestamps, diarization, or participant-channel metadata |
| Smaller rank-4/8 Qwen2-Audio training | Possibly feasible but unbenchmarked | Do not start unless the acoustic gate passes |
| Exact existing rank-16 backward | Not feasible on 24 GiB | Requires a larger GPU or separately registered offload protocol |

## 4. Objective of the next experiment

Determine whether fixed-K DAIC audio contains a reproducible subject-level signal that survives
subject-disjoint folds, seed variation, and shuffled-audio controls.

This experiment must answer one question before any more AudioLLM training:

> Can a small audio-only subject model consistently discriminate depression better than
> majority and shuffled-audio controls without tuning on the official test set?

## 5. Locked data protocol

### 5.1 Data partitions

- Keep the existing official 47-subject test partition locked until all model classes,
  features, hyperparameter grids, seeds, thresholds, and pass/fail rules are frozen.
- Use the existing 107 training plus 35 validation subjects as a 142-subject development set.
- Build five deterministic, stratified, subject-disjoint outer development folds.
- Use only the outer-fold training portion for inner model selection. Do not inspect the
  outer holdout or official test outcomes while selecting hyperparameters.
- After producing complete development out-of-fold predictions and freezing the selected
  protocol, refit once on all 142 development subjects and evaluate the 47 official test
  subjects once.

No sample or chunk may cross a subject boundary between training and evaluation.

### 5.2 Audio sampling

- Use exactly four chunks per subject for all primary comparisons.
- Parse trailing chunk numbers numerically, never lexically.
- Select evenly spaced K=4 chunks using the already tested rule:
  - 10-chunk subjects: numeric positions `[0, 3, 6, 9]`;
  - 15-chunk subjects: numeric positions `[0, 5, 9, 14]`.
- Preserve the selected sample IDs and their hashes in every run.
- Treat numeric suffix order as an ordinal only; do not describe it as verified chronology.

Fixed K removes the direct 10-versus-15 chunk-count leak, but it does not remove the perfect
association between `random_segment`/`segment` preprocessing kind and label. That limitation
must remain visible in every interpretation.

## 6. Phase N0: Freeze and validate the protocol

Before fitting any new model:

1. Write a machine-readable experiment specification containing:
   - manifest and partition hashes;
   - the 142 development and 47 locked-test subject IDs;
   - five outer-fold assignments;
   - inner-fold assignments;
   - selected K=4 sample IDs per subject;
   - model grids, seeds, metrics, threshold rule, bootstrap method, and gate criteria.
2. Verify that cached eGeMAPS and WavLM chunk features match the current manifest, sample IDs,
   model revision, and extraction-code hashes.
3. Fail closed on missing, duplicated, non-finite, or dimensionally inconsistent features.
4. Add tests for subject isolation, fold reproducibility, numeric K=4 selection, shuffled-bundle
   derangement, and OOF coverage.

Expected resources: CPU only, less than 30 minutes, negligible new disk usage.

## 7. Phase N1: Cross-fold linear acoustic ceilings

Run two primary feature families under the identical folds.

### 7.1 eGeMAPSv02

- Per chunk: 88 eGeMAPSv02 functionals.
- Per subject: chunk mean concatenated with population standard deviation, 176 dimensions.
- Model: standardized L2 logistic regression.
- Small locked regularization grid: `C in {0.0001, 0.001, 0.01, 0.1, 1.0}`.
- Select `C` by inner-fold log loss; break ties in favor of smaller `C`.

### 7.2 Frozen WavLM Base+

- Use the already pinned `microsoft/wavlm-base-plus` revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`.
- Per chunk: concatenate time-mean representations from transformer layers 6, 7, and 8.
- Per subject: chunk-vector mean plus population standard deviation, 4,608 dimensions.
- Use the same standardized L2 logistic regression and locked `C` grid.

### 7.3 Controls

For both feature families report:

- development-prevalence constant probability;
- majority label at threshold `0.5`;
- real subject bundle;
- 100 deterministic within-partition derangements of complete K=4 subject bundles;
- labels, folds, and target subject IDs unchanged during shuffling.

### 7.4 Outputs

- One OOF probability per development subject and feature family.
- One locked-test probability per subject after the protocol is frozen.
- Fold-level metrics and selected hyperparameters.
- Pooled OOF metrics.
- Subject-bootstrap confidence intervals.
- Real-minus-shuffled paired differences and empirical permutation p-values.
- Complete provenance, selected sample IDs, and immutable prediction files.

Expected resources using existing caches: primarily CPU, likely minutes rather than hours,
less than 1 GiB additional disk space.

## 8. Phase N2: Small subject-level MIL model

Proceed to N2 after N1 completes, even if the linear models are null, because MIL tests the
specific hypothesis that mean/statistics pooling hides sparse informative chunks. Do not expand
to a large architecture search.

### 8.1 Primary model

Use the four frozen WavLM chunk vectors as instances in one subject bag:

1. optional linear projection of each 2,304-dimensional chunk vector to 128 dimensions;
2. gated-attention pooling with hidden dimension 64 or 128;
3. one small binary classification head;
4. exactly one diagnosis loss per subject;
5. no diagnosis loss attached independently to chunks.

Target substantially fewer than one million trainable parameters. Keep the WavLM encoder
frozen.

### 8.2 Locked training choices

- Model family: mean-pooling MLP control and one gated-attention MIL model.
- Optimizer: AdamW.
- Learning-rate grid: at most `{1e-4, 3e-4}`.
- Weight decay: at most `{0.01, 0.1}`.
- Dropout: fixed at `0.2` unless N0 freezes one alternative.
- Early stopping: inner validation log loss, not positive F1.
- Seeds: five fixed seeds for the MIL model.
- Maximum epochs and patience must be frozen in N0.
- Threshold: `0.5` for primary hard-label metrics; any calibrated threshold is secondary and
  must be fitted using inner training data only.

### 8.3 MIL controls

- Mean-pooling head with comparable capacity.
- Complete K=4 bundle derangement within each fold.
- Attention entropy and per-chunk weights as diagnostics, not proof of clinical relevance.
- Check that attention does not simply identify preprocessing kind, chunk ordinal, silence, or
  energy.

Expected resources: well within RTX 4090 capacity, likely under 2 GiB model/activation memory,
approximately 1-4 hours for five folds and five seeds, and less than 1 GiB new artifacts.
Actual runtime must be measured from a one-fold smoke run before estimating completion time.

## 9. Phase N3: Statistical decision gate

Primary continuous metrics:

- AUROC;
- average precision/AUPRC;
- log loss.

Primary balanced hard-label diagnostics:

- balanced accuracy;
- macro F1;
- positive F1 reported separately, never described as macro F1.

Uncertainty and comparisons:

- pool exactly one OOF prediction per development subject;
- report fold mean, fold standard deviation, and pooled OOF metrics separately;
- use paired subject bootstrap intervals for real-minus-shuffled score/metric differences;
- report results across all five MIL seeds rather than selecting the best seed;
- use the official test set once only after freezing the winner and analysis code.

### 9.1 Pass rule

The audio branch passes only if all of the following hold:

1. pooled development OOF AUROC is above `0.5` and the real-minus-shuffled AUROC confidence
   interval excludes zero;
2. pooled development OOF balanced accuracy exceeds the shuffled control with a paired
   confidence interval excluding zero;
3. the direction is positive in at least four of five outer folds;
4. the MIL result is not driven by one selected seed;
5. the locked official-test result has the same positive direction against shuffled audio;
6. provenance and leakage checks remain clean.

The official test interval may remain wide because it contains only 47 subjects. It is a
directional confirmation, not a new hyperparameter-selection set.

### 9.2 Fail rule

Fail the current audio protocol if no tested representation/pooling model passes the complete
rule. A failure means:

> No reproducible depression signal was established for the tested fixed-K preprocessed DAIC
> audio protocol.

It must not be generalized to all speech representations, raw participant-only audio, or all
depression datasets.

## 10. Phase N4: Conditional follow-up

### If the acoustic gate fails

- Stop new Qwen2-Audio training on the current preprocessed chunks.
- Focus model work on text while clearly labeling it as text-driven depression screening.
- Prioritize obtaining raw participant-only audio, timestamps, diarization, and interviewer
  exclusion before revisiting audio.
- Preserve the null result; do not broaden the hyperparameter search on the same small folds.

### If the acoustic gate passes

1. Audit whether the passing model relies on preprocessing kind, energy, silence, speaker, or
   recorder cues.
2. Run a one-batch memory probe for a deliberately small Qwen2-Audio configuration:
   - LoRA rank 4 or 8;
   - q/v or attention-only targets;
   - last 2 or 4 decoder layers;
   - approximately 0.2-1 million trainable parameters;
   - BF16 and gradient checkpointing;
   - full K=4 audio and unchanged text length.
3. Record peak allocated/reserved memory without optimizer step, then with the intended
   optimizer state if the first probe fits.
4. Do not start full AudioLLM training until the memory probe and a separate training protocol
   review pass.

The existing rank-16/all-layer OOM does not prove that these smaller configurations cannot fit
the 4090; they have not yet been benchmarked.

## 11. Current-provenance audit that is locally possible

Without new source metadata, local checks can still measure:

- file validity, duration, sample rate, channels, clipping, RMS, active-speech proxy, and
  zero/silent audio;
- exact and near-duplicate files;
- class association with duration, energy, silence, chunk ordinal, filename/sample kind, and
  recording format;
- whether MIL attention correlates with these nuisance variables.

Local work cannot establish participant-only speech, interviewer exclusion, correct speaker
identity, timestamp alignment, or true chronology. Those claims remain blocked until raw
alignment or diarization data is supplied.

## 12. Implementation order

```text
N0  Freeze hashes, folds, K4 sample IDs, grids, seeds, metrics and gates
N1  Run nested development CV for eGeMAPS and WavLM linear ceilings
N2  Run mean-pooling and gated-MIL WavLM heads over the same folds
N3  Aggregate OOF predictions, bootstrap, compare shuffled controls, apply gate
N4  Evaluate the locked official test set once
N5  Only after a pass: nuisance audit and small-Qwen memory probe
```

## 13. Required final artifacts

The next experiment is not complete until it produces:

- `experiment_spec.json` with hashes, folds, grids, seeds and gates;
- `fold_assignments.json` and selected K=4 sample IDs;
- per-fold and pooled OOF prediction JSONL files;
- locked-test prediction JSONL files created only after protocol freeze;
- real and shuffled-control metrics with paired intervals;
- MIL seed-level metrics and attention diagnostics;
- provenance containing repository state, environment versions, feature/model hashes, and
  command line;
- a concise Markdown result report stating pass/fail without overstating a null result.

## 14. Immediate next action

Implement Phase N0 and N1 first. Reuse the existing eGeMAPS features and cached WavLM chunk
vectors, keep the official test set locked, and produce development OOF results before writing
the MIL model. No new large model download or new Conda environment is required for these
phases.
