# Turkish BDI>=17 Subject Oversampling Report

Date: 2026-07-25

## Executive result

The five-stage plan reached its predeclared Stage-4 stopping point.

- Stage 2 selected the `0.75` minority-to-majority subject ratio under the
  fixed tie rule.
- Stage 3 completed all 60 Optuna studies and 6,000 trials. Only
  `audio_only` passed the Qwen-entry gate.
- The four-job Qwen pilot completed, but failed two Stage-5 requirements:
  mean selected-fold Macro-F1 gain was `0.013372`, below `0.015`, and fold 0
  lost `0.033107`, beyond the allowed `0.03`.
- Stage 5 was therefore not submitted. This is the required gate-based stop,
  not an incomplete experiment.

Subject oversampling improved the matched hidden-state `audio_only` control
from pooled Macro-F1 `0.490000` to a three-seed mean of `0.524445`. It did not
justify a 30-run Qwen expansion. The two-fold Qwen pilot improved pooled
Macro-F1 from `0.540441` to `0.555867`, but the fold-selection criteria did
not pass.

## Protocol warning

All Turkish results use BDI>=17 and the repository's table-aligned outer
validation folds. Those validation folds are used because the Turkish
checkpoints and experiment development use them. They are not a separate
unseen test set. No BDI>=21 rows were run or added.

Oversampling was applied only to training subject groups. Validation,
selection, and held-out subject indices remained untouched. A duplicated
subject always contributed its complete response group.

## Implementation

The implementation added:

- shared deterministic subject-group oversampling in `src/sampling.py`;
- Qwen training support for `minority_subject_oversample`, with rank-zero
  audit sidecars and stable distributed epoch length;
- fixed-head screening for ratios `0.75` and `1.0`;
- Optuna support for explicit no-sampling and pure-oversampling profiles,
  with XGBoost `scale_pos_weight=1`;
- per-inner-fold and final-fit sampling audits;
- matrix builders, Slurm workers/wrappers, auditors, summarizers, and
  Qwen3ASR Macro-F1 configurations;
- regression coverage for subject integrity, deterministic replay, matrix
  counts, legacy Optuna compatibility, and JSON-safe pilot summaries.

Implementation commits:

- `33bccac` - primary five-stage implementation;
- `cbd4aae` - declared Qwen oversampling override fields;
- `9539888` - handled null Optuna sampling fields;
- `a60c17a` - made Qwen pilot gate metrics JSON serializable.

## Validation and smoke tests

Local:

- `conda run -n llmdep4090 python -m unittest discover -s tests -v`:
  40 passed, 1 dependency-based skip before the final reporting fix.
- The focused reporting regression test passed after `a60c17a`.
- Python compilation, shell syntax checks, CSV parsing, XLSX ZIP validation,
  and `git diff --check` passed.
- `scripts/sanity_tests_no_model.sh` could not run locally because its
  documented DAIC input is GPFS-only and the backup path is absent on this
  host. This is an environment limitation; the MN5 smoke and production
  paths exercised the changed code.

MN5 smoke:

| Purpose | Job | State | Evidence |
|---|---:|---|---|
| Hidden screen smoke | `43809689` | COMPLETED `0:0` | 42/42 fit audits |
| Hidden idempotent restart | `43809710` | COMPLETED `0:0` | identical hashes, no duplicate work |
| Qwen weighted smoke | `43809956` | COMPLETED `0:0` | required artifacts present |
| Qwen OS initial submission | `43809957` | FAILED `1:0` | undeclared config override; fixed by `cbd4aae` |
| Qwen OS corrected retry | `43810107` | COMPLETED `0:0` | 2/4 subjects became 4/4 occurrences; held-out indices untouched |
| Optuna OS smoke | `43810239` | COMPLETED `0:0` | 2 trials, audits pass |
| Optuna idempotent restart | `43810258` | COMPLETED `0:0` | remained exactly 2 trials |

The failed smoke job is retained in the accounting. It was not a scientific
run and was corrected before production.

## Stage 2: fixed-head screen

Jobs `43810170` through `43810184` all completed `0:0`.

- 15 jobs;
- 7 sampling profiles;
- 2 classifier heads;
- 3 inner folds;
- 630 fits;
- 210 summary rows;
- 42 sampling audits per job;
- no outer-evaluation metric inspected for ratio selection.

| Ratio | Mean pooled inner-OOF Macro-F1 |
|---:|---:|
| `0.75` | `0.738365` |
| `1.0` | `0.741749` |

The absolute difference was `0.003384`, within the predeclared `0.005` tie
threshold. The fixed tie rule therefore selected `0.75`.

## Stage 3: matched Optuna panel

Jobs `43810263` through `43810322` all completed `0:0`.

- 60/60 studies;
- 6,000/6,000 complete trials;
- 18,000 inner XGBoost fits;
- 60 final refits;
- complete inner validation coverage;
- zero outer train/held-out subject overlap;
- all held-out indices untouched;
- all configuration and metadata hashes compatible.

### Gate results

| Modality | Control Macro-F1 | OS mean | Gain | Seeds beating | Neg-recall gain | Pos-recall loss | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| audio_text | 0.466667 | 0.472693 | 0.006026 | 3 | 0.027027 | 0.040161 | FAIL |
| audio_only | 0.490000 | 0.524445 | 0.034445 | 3 | 0.072072 | 0.060241 | PASS |
| text_only | 0.583030 | 0.599636 | 0.016606 | 3 | 0.036036 | 0.000000 | FAIL |

The mean Macro-F1 across all modalities improved from `0.513232` to
`0.532258`, so the global non-inferiority condition passed. `audio_only` was
the only qualifying modality and entered Stage 4.

### Complete pooled metrics

`Fold mean +/- SD` is the mean and sample SD of the five fold-level
Macro-F1 values. Other values are recomputed from pooled subject predictions.
AUROC uses pooled subject probabilities.

| Modality | Profile | Acc | Macro-F1 | Fold mean +/- SD | Neg F1 | Neg recall | Pos F1 | Pos recall | Bal acc | AUROC | CM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio_only | control | 0.7167 | 0.4900 | 0.4820 +/- 0.1088 | 0.1500 | 0.0811 | 0.8300 | 1.0000 | 0.5405 | 0.6096 | `[[3,34],[0,83]]` |
| audio_only | OS 1337 | 0.6833 | 0.5200 | 0.5220 +/- 0.1396 | 0.2400 | 0.1622 | 0.8000 | 0.9157 | 0.5389 | 0.6610 | `[[6,31],[7,76]]` |
| audio_only | OS 2024 | 0.6917 | 0.5105 | 0.5030 +/- 0.1418 | 0.2128 | 0.1351 | 0.8083 | 0.9398 | 0.5374 | 0.6441 | `[[5,32],[5,78]]` |
| audio_only | OS 7 | 0.7167 | 0.5428 | 0.5402 +/- 0.1302 | 0.2609 | 0.1622 | 0.8247 | 0.9639 | 0.5630 | 0.6627 | `[[6,31],[3,80]]` |
| audio_text | control | 0.6750 | 0.4667 | 0.4647 +/- 0.1115 | 0.1333 | 0.0811 | 0.8000 | 0.9398 | 0.5104 | 0.6063 | `[[3,34],[5,78]]` |
| audio_text | OS 1337 | 0.6583 | 0.4743 | 0.4731 +/- 0.1230 | 0.1633 | 0.1081 | 0.7853 | 0.9036 | 0.5059 | 0.5874 | `[[4,33],[8,75]]` |
| audio_text | OS 2024 | 0.6500 | 0.4695 | 0.4697 +/- 0.0829 | 0.1600 | 0.1081 | 0.7789 | 0.8916 | 0.4998 | 0.6073 | `[[4,33],[9,74]]` |
| audio_text | OS 7 | 0.6583 | 0.4743 | 0.4731 +/- 0.1230 | 0.1633 | 0.1081 | 0.7853 | 0.9036 | 0.5059 | 0.5799 | `[[4,33],[8,75]]` |
| text_only | control | 0.6417 | 0.5830 | 0.5829 +/- 0.0702 | 0.4267 | 0.4324 | 0.7394 | 0.7349 | 0.5837 | 0.6464 | `[[16,21],[22,61]]` |
| text_only | OS 1337 | 0.6500 | 0.5897 | 0.5889 +/- 0.0634 | 0.4324 | 0.4324 | 0.7470 | 0.7470 | 0.5897 | 0.6171 | `[[16,21],[21,62]]` |
| text_only | OS 2024 | 0.6417 | 0.5943 | 0.5938 +/- 0.0980 | 0.4557 | 0.4865 | 0.7329 | 0.7108 | 0.5987 | 0.6475 | `[[18,19],[24,59]]` |
| text_only | OS 7 | 0.6667 | 0.6149 | 0.6120 +/- 0.0943 | 0.4737 | 0.4865 | 0.7561 | 0.7470 | 0.6167 | 0.6464 | `[[18,19],[21,62]]` |

### Oversampling-seed stability

The SD below is the population SD across the three predeclared pooled seed
results.

| Modality | Mean Macro-F1 | SD | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| audio_text | 0.472693 | 0.002276 | 0.469474 | 0.474303 | 0.004829 |
| audio_only | 0.524445 | 0.013547 | 0.510528 | 0.542806 | 0.032278 |
| text_only | 0.599636 | 0.010949 | 0.589710 | 0.614891 | 0.025181 |

Against the workbook's previous protocol-valid XGBoost Optuna entries, the
three-seed OS means are lower by `0.063441` for audio+text, `0.033008` for
audio-only, and `0.004385` for text-only. The matched Stage-3 controls are
the correct causal comparison; this additional comparison prevents the new
panel from being mistaken for a new overall hidden-state leader.

## Stage 4: matched Qwen pilot

| Profile | Fold 0 selected | Fold 1 selected | Fold mean | Pooled Macro-F1 | Neg F1 | Neg recall | Pos F1 | Pos recall | Bal acc | Accuracy | CM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| weighted control | 0.533107 | 0.532225 | 0.532666 | 0.540441 | 0.375000 | 0.375000 | 0.705882 | 0.705882 | 0.540441 | 0.600000 | `[[6,10],[10,24]]` |
| subject OS 0.75 | 0.500000 | 0.592075 | 0.546037 | 0.555867 | 0.387097 | 0.375000 | 0.724638 | 0.735294 | 0.555147 | 0.620000 | `[[6,10],[9,25]]` |

Teacher-forced Qwen outputs do not expose a continuous subject score, so
AUROC is not available for this pilot.

Jobs:

- weighted fold 0 `43811877`: COMPLETED `0:0`, `00:27:12`;
- weighted fold 1 `43811878`: COMPLETED `0:0`, `00:31:17`;
- oversampled fold 0 `43811879`: COMPLETED `0:0`, `00:17:16`;
- oversampled fold 1 `43811880`: COMPLETED `0:0`, `00:25:45`.

### Stage-5 gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Mean selected-fold Macro-F1 gain | `>=0.015` | `0.013372` | FAIL |
| Minimum fold gain | `>=-0.03` | `-0.033107` | FAIL |
| Pooled Macro-F1 not below control | yes | `0.555867 > 0.540441` | PASS |
| Pooled negative recall not below control | yes | `0.375000 = 0.375000` | PASS |

`proceed_to_full=false`. The remaining 26 Qwen jobs were not submitted.

## Job accounting, audits, and transfer

The consolidated `sacct` record contains 86 jobs:

- 85 `COMPLETED 0:0`;
- 1 expected smoke failure, `43809957 FAILED 1:0`;
- no missing, cancelled, timed-out, out-of-memory, or failed-dependency job.

The classified log scan found 13 raw keyword matches:

- 12 benign messages explaining that in-training held-out evaluation was
  skipped to avoid an NCCL timeout;
- 1 expected signature from failed smoke job `43809957`;
- 0 unexpected signatures.

Remote and local audits agree exactly:

- Stage 2: 15/15 jobs pass;
- Stage 3: 60/60 studies pass;
- Stage 4: 4/4 jobs pass;
- Qwen pilot pooled summary is byte-equivalent as parsed JSON;
- configuration hashes, split disjointness, subject coverage, sampling
  sidecars, and required artifacts pass.

All result directories, matrices, summaries, audits, submission records,
`sacct` records, and experiment-specific logs were transferred with dry-run
first via `transfer1`. Checksum-mode rsync dry-runs reported zero remaining
bytes. Qwen `best_model` and `last_model` checkpoint directories were
intentionally excluded. No transfer used `--delete`.

## Reporting updates

- `qwen_hidden_best_results_no_emotion.csv` now contains the 12 complete
  Stage-3 pooled rows: control plus every OS seed for every modality.
- `depression_results_table_no_emo.csv` contains the three explicitly
  labeled OS seed-mean rows.
- `depression_results_combined_with_posf1_graphs.xlsx` retains its charts,
  adds the OS mean to the Macro-F1 Summary, and includes a dedicated
  `Turkish Oversampling` evidence sheet.
- This report records the full execution and negative Stage-5 decision.

## Conclusion

Pure subject oversampling is useful for probing the Turkish hidden-state
class imbalance: it consistently improves the matched controls and raises
negative recall. Its gain is not large or stable enough to replace the
current hidden-state headline results, and the matched Qwen pilot fails the
predeclared expansion gate. The correct final decision is to retain these
results as a documented sensitivity analysis and not run the full Qwen
confirmation panel.
