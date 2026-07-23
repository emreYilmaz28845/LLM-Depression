# Raw Hidden-State XGBoost Optuna Follow-up Report

Date: 2026-07-23

Related baseline report:
[`QWEN2_HIDDEN_XGBOOST_EXPERIMENT_REPORT_2026-07-22.md`](QWEN2_HIDDEN_XGBOOST_EXPERIMENT_REPORT_2026-07-22.md)

## Summary

The staged Optuna follow-up completed successfully on MareNostrum 5:

- 33 fresh standard-profile studies with 150 trials and inner seed 1337;
- 15 targeted Turkish depth-8 studies;
- a three-inner-seed panel covering all 33 outer evaluations;
- 114 new studies, 17,100 completed trials, 51,300 inner XGBoost fits, and
  114 final fits;
- no failed studies, failed trials, tracebacks, out-of-memory events, subject
  overlap, or missing artifacts.

Increasing the budget from 50 to 150 trials was useful but did not change the
overall model-family conclusion. It improved DAIC audio-only positive F1 from
0.581 to 0.649 and audio+text from 0.643 to 0.667, while CMDC thresholded
results were mostly unchanged. Turkish gains were mixed: audio+text macro-F1
improved from 0.516 to 0.536, but audio-only and text-only declined.

The predeclared inner-seed pilot triggered full expansion because DAIC
text-only positive F1 ranged by 0.045 across seeds, exceeding the 0.03 gate.
Across the complete panel, Turkish audio-only was substantially less stable:
macro-F1 ranged from 0.505 to 0.624. The result shows that one deterministic
three-fold inner split is not sufficient to characterize tuning uncertainty
for every condition.

Depth 8 improved the seed-1337 Turkish audio-only and text-only profiles by
0.023 and 0.031 macro-F1 respectively, but neither became the strongest
Turkish hidden-state result. Fixed raw text-only XGBoost remains best on
Turkish macro-F1 at 0.623. Raw logistic regression remains the more reliable
headline hidden-state probe for DAIC/CMDC.

DAIC and CMDC use their predeclared held-out evaluations. Turkish remains a
**table-aligned outer-validation analysis**, not an unseen-test estimate,
because the underlying Qwen checkpoints were selected using those validation
folds.

## Experiment design

All experiments reused the existing frozen hidden-vector caches. There was no
Qwen extraction, Qwen training, PCA, emotion condition, GPU allocation, early
stopping, or threshold optimization.

The standard profile retained the original ten-parameter ranges. The depth-8
profile changed only the upper bound of `max_depth` from 6 to 8. Every study
used:

- 150 total Optuna trials;
- three stratified subject-level inner folds;
- TPE and XGBoost seed 1337;
- threshold 0.5;
- response-to-subject majority/probability-margin aggregation;
- pooled subject-level inner OOF positive F1 for DAIC/CMDC;
- pooled subject-level inner OOF macro-F1 for Turkish.

The seed panel varied only the inner subject-fold seed: 7, 1337, or 2024.
TPE and XGBoost randomness remained fixed at 1337. Seeds are reported as a
stability analysis; no outer result was used to choose a preferred seed.

Every experiment used a separate directory and SQLite database. The completed
50-trial `xgb_optuna_raw` studies were not resumed, modified, or overwritten.

## Standard 150-trial results

“Primary” is positive F1 for DAIC/CMDC and macro-F1 for Turkish. Parentheses
contain the outer-fold mean and sample SD; DAIC has one held-out split. Deltas
use the same primary metric. Qwen deltas use the rounded no-emotion table, so
they are approximate.

| Dataset | Modality | Primary (mean±SD) | ACC | Positive F1 | Macro-F1 | AUROC | Pooled confusion | Δ 50 trials | Δ fixed XGB | Δ raw LogReg | Δ Qwen | Δ majority |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DAIC | Audio+text | 0.667 (0.667±0.000) | 0.809 | 0.667 | 0.766 | 0.887 | `[[29, 4], [5, 9]]` | +0.024 | +0.046 | -0.133 | -0.051 | +0.667 |
| DAIC | Audio-only | 0.649 (0.649±0.000) | 0.723 | 0.649 | 0.710 | 0.803 | `[[22, 11], [2, 12]]` | +0.068 | +0.056 | +0.020 | +0.054 | +0.649 |
| DAIC | Text-only | 0.741 (0.741±0.000) | 0.851 | 0.741 | 0.818 | 0.918 | `[[30, 3], [4, 10]]` | +0.000 | +0.048 | +0.000 | +0.049 | +0.741 |
| CMDC | Audio+text | 0.980 (0.978±0.050) | 0.987 | 0.980 | 0.985 | 0.994 | `[[52, 0], [1, 25]]` | +0.000 | +0.020 | +0.000 | +0.084 | +0.980 |
| CMDC | Audio-only | 0.980 (0.978±0.050) | 0.987 | 0.980 | 0.985 | 0.988 | `[[52, 0], [1, 25]]` | +0.000 | +0.000 | +0.000 | +0.062 | +0.980 |
| CMDC | Text-only | 0.962 (0.962±0.053) | 0.974 | 0.962 | 0.971 | 0.997 | `[[51, 1], [1, 25]]` | +0.000 | +0.002 | -0.019 | +0.058 | +0.962 |
| Turkish | Audio+text | 0.536 (0.519±0.140) | 0.617 | 0.729 | 0.536 | 0.542 | `[[12, 25], [21, 62]]` | +0.020 | +0.101 | +0.038 | +0.048 | +0.127 |
| Turkish | Audio-only | 0.534 (0.525±0.119) | 0.667 | 0.783 | 0.534 | 0.633 | `[[8, 29], [11, 72]]` | -0.015 | +0.044 | -0.044 | -0.079 | +0.125 |
| Turkish | Text-only | 0.563 (0.561±0.066) | 0.617 | 0.716 | 0.563 | 0.632 | `[[16, 21], [25, 58]]` | -0.041 | -0.060 | +0.008 | +0.051 | +0.154 |

Fifteen of the 33 standard-profile winners occurred after trial 49, and 11
occurred after trial 99. The larger budget therefore found genuinely new
inner-OOF winners. Thresholded outer improvements were more limited, which is
expected because the extra trials optimize inner OOF performance rather than
the untouched outer labels.

The strongest standard-profile observations are:

- DAIC audio-only is the only seed-1337 condition where tuned XGBoost clearly
  exceeds raw logistic regression on positive F1, although it has lower
  accuracy and macro-F1 than the best DAIC probes.
- CMDC audio+text and audio-only make only one error among 78 pooled subjects.
  Additional trials do not improve their already saturated 50-trial results.
- Turkish audio+text improves over fixed raw XGBoost and raw logistic
  regression on macro-F1, but remains far below fixed raw text-only XGBoost.
- The Turkish all-positive control still has positive F1 0.818 but macro-F1
  only 0.409, reinforcing why macro-F1 is the primary Turkish measure.

## Targeted depth-8 sensitivity

Four standard-profile folds selected `max_depth=6`: Turkish audio+text fold 1,
audio-only folds 0 and 4, and text-only fold 4. Under the complete-condition
rule, all five folds of all three Turkish modalities were rerun.

| Modality | ACC | Positive F1 | Macro-F1 | AUROC | Pooled confusion | Δ macro-F1 vs d6 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Audio+text | 0.617 | 0.736 | 0.519 | 0.564 | `[[10, 27], [19, 64]]` | -0.017 |
| Audio-only | 0.683 | 0.793 | 0.557 | 0.665 | `[[9, 28], [10, 73]]` | +0.023 |
| Text-only | 0.642 | 0.733 | 0.594 | 0.638 | `[[18, 19], [24, 59]]` | +0.031 |

Two of 15 depth-profile winners selected depth 8 and one selected depth 7,
showing that the extension was used by the optimizer. It helped audio-only and
text-only, but not audio+text. Even the improved text-only macro-F1 of 0.594
remains below fixed raw text-only XGBoost at 0.623. Depth 8 should therefore
remain a sensitivity result rather than replacing the standard profile.

## Inner-fold seed stability

Metrics below separately pool each seed’s outer predictions before comparing
seeds. Subjects are never concatenated across seeds.

| Dataset | Modality | Primary metric | Seed mean±SD | Minimum | Maximum | Range |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DAIC | Audio+text | Positive F1 | 0.659±0.014 | 0.643 | 0.667 | 0.024 |
| DAIC | Audio-only | Positive F1 | 0.647±0.021 | 0.625 | 0.667 | 0.042 |
| DAIC | Text-only | Positive F1 | 0.756±0.026 | 0.741 | 0.786 | 0.045 |
| CMDC | Audio+text | Positive F1 | 0.974±0.011 | 0.962 | 0.980 | 0.019 |
| CMDC | Audio-only | Positive F1 | 0.974±0.011 | 0.962 | 0.980 | 0.019 |
| CMDC | Text-only | Positive F1 | 0.961±0.001 | 0.960 | 0.962 | 0.002 |
| Turkish | Audio+text | Macro-F1 | 0.543±0.011 | 0.536 | 0.556 | 0.020 |
| Turkish | Audio-only | Macro-F1 | 0.554±0.062 | 0.505 | 0.624 | 0.119 |
| Turkish | Text-only | Macro-F1 | 0.564±0.009 | 0.555 | 0.574 | 0.019 |

The pilot gate was triggered by DAIC text-only: seed 2024 produced positive F1
0.786 while seeds 7 and 1337 produced 0.741. Full expansion then revealed the
larger Turkish audio-only range. In that condition, inner seed 7 reached
macro-F1 0.624, seed 1337 reached 0.534, and seed 2024 reached 0.505.

This does not authorize selecting seed 7 after observing outer results.
Instead, it shows that conclusions based on one inner split are fragile for
Turkish audio-only. The seed mean and dispersion are the appropriate
robustness summary.

## Integrity, execution, and provenance

Automated acceptance audits passed for all four manifests:

| Stage | Studies | Trials | Inner fits | Slurm wall-time range |
| --- | ---: | ---: | ---: | ---: |
| Standard d6, seed 1337 | 33 | 4,950 | 14,850 | 3:40–61:00 |
| Targeted depth 8 | 15 | 2,250 | 6,750 | 3:09–88:38 |
| Seed pilot, seeds 7/2024 | 22 | 3,300 | 9,900 | 2:36–33:49 |
| Conditional expansion | 44 | 6,600 | 19,800 | 5:52–94:21 |
| **Total** | **114** | **17,100** | **51,300** |  |

Every study contains exactly 150 `COMPLETE` trials, a final model, sample and
subject predictions, pooled/fold metrics, inner assignments, configuration
hash, complete trial table, and SQLite study. All inner validation assignments
cover each outer-training subject exactly once. All outer train/evaluation
subject intersections are empty.

The implementation was committed as
`b46d54be0102797c1b9db059e04ed4d5e7dbec0a`; the full-panel stability
summarizer correction is
`70fe4f5c4738c3d6dace1e17a5740b7b9f3f99d4`. Runtime versions were Optuna
4.4.0, XGBoost CPU 2.1.4, and scikit-learn 1.7.0.

MN5 job groups:

- smoke and idempotent rerun: `43730954`, `43730955`, `43730959`;
- standard stage: `43730963`–`43730995`;
- depth stage: `43732172`–`43732187`, with scheduler gaps;
- seed pilot: `43732188`–`43732209`;
- seed expansion: `43733145`–`43733188`.

All jobs completed with exit code 0. No production log contains a traceback,
OOM marker, killed-process marker, or failure signature. The 114 new result
directories occupy 359 MB on GPFS.

## Artifacts

- Compact results: `qwen_hidden_best_results_no_emotion.csv`
- All fold and pooled summaries:
  `outputs/hidden_classifiers/summary.csv` and `summary.json`
- Seed metrics and ranges:
  `outputs/hidden_classifiers/optuna_stability/`
- Resolved manifests, submission records, dry-runs, and audits:
  `outputs/optuna_followup_manifests/`
- Per-study SQLite, configuration, trials, models, and predictions:
  `outputs/hidden_classifiers/<dataset>/<condition>/<run>/fold_<n>/<experiment_id>/`
- Experiment-specific Slurm logs:
  `logs/slurm_qwen_hidden_optuna/<experiment_id>/`

## Conclusion

Expanding from 50 to 150 trials was a reasonable targeted investment: it found
later inner-OOF winners and improved several DAIC results without broadening
all ten ranges. The depth-8 extension provided modest Turkish sensitivity
gains but no new overall leader. The three-seed analysis is the more important
finding: inner-fold choice can materially affect tuned outer performance,
especially for Turkish audio-only.

The recommended reporting remains:

- DAIC/CMDC: emphasize positive F1, with raw logistic regression as the most
  reliable hidden-state probe;
- Turkish: emphasize macro-F1, retain fixed raw text-only XGBoost as the best
  observed hidden-state result, and report seed variability rather than a
  cherry-picked tuned seed;
- preserve the Turkish table-aligned-validation warning in every comparison.
