# Qwen2 Final Hidden-State Classifier Experiment Report

Date: 2026-07-22; Turkish and Optuna extensions: 2026-07-23

Implementation plan: [`QWEN2_HIDDEN_XGBOOST_IMPLEMENTATION_PLAN_2026-07-22.md`](QWEN2_HIDDEN_XGBOOST_IMPLEMENTATION_PLAN_2026-07-22.md)

The completed 150-trial, depth-8, and three-inner-seed follow-up is reported in
[`QWEN2_HIDDEN_XGBOOST_OPTUNA_FOLLOWUP_REPORT_2026-07-23.md`](QWEN2_HIDDEN_XGBOOST_OPTUNA_FOLLOWUP_REPORT_2026-07-23.md).

## Summary

The prompt-only final-hidden-state pipeline was implemented and run for the
aligned DAIC, CMDC, and Turkish audio+text, audio-only, text-only, and
emotion-augmented checkpoints. Forty fold/checkpoint extraction jobs and one
grouped control rerun completed successfully on MareNostrum 5.

The main findings are:

- CMDC hidden-state classifiers were substantially stronger than the current
  Qwen verdict baseline. Raw logistic regression reached pooled ACC 0.987 and
  positive F1 0.980 for all three modalities. Raw XGBoost reached ACC
  0.974-0.987 and positive F1 0.960-0.980.
- DAIC was less uniform. Raw logistic regression was strongest for audio+text
  (ACC 0.851, positive F1 0.800) and improved over the current Qwen verdict
  baseline. Text-only raw XGBoost reproduced the current rounded baseline
  (ACC 0.830, positive F1 0.692). Audio-only raw XGBoost improved accuracy but
  left positive F1 essentially unchanged.
- PCA-32 did not consistently improve XGBoost. It was neutral for CMDC
  audio+text, beneficial for CMDC text-only, and generally harmful on DAIC.
- Logistic regression was at least as competitive as XGBoost and was often
  better. The useful signal is therefore not dependent on a nonlinear tree
  head.
- Adding emotion descriptions did not improve the strongest raw logistic
  result. DAIC no-emotion audio+text remained best at positive F1 0.800; Chinese
  SECap emotion was close at 0.788 and English SECap emotion fell to 0.703.
  CMDC paper-provided Chinese emotion tied the no-emotion logistic result at
  F1 0.980.
- Raw XGBoost reacted differently: emotion increased DAIC F1 from 0.621 to
  0.714 with English captions and increased CMDC F1 from 0.960 to 0.980. These
  gains still did not beat the strongest no-emotion logistic result.
- Turkish results did not reproduce the CMDC gains. Audio+text XGBoost PCA-32
  reached positive F1 0.814 but remained below the all-positive control's
  0.818. Audio-only raw XGBoost reached F1 0.830, below the saved Qwen
  baseline's 0.847. Text-only raw XGBoost had the best Turkish macro-F1, 0.623,
  but positive F1 was only 0.774.
- Leakage-safe Optuna tuning improved fixed raw XGBoost most clearly for CMDC
  audio+text (positive F1 0.960 to 0.980) and DAIC text-only (0.692 to
  0.741). It did not improve the strongest hidden-state result for any dataset:
  fixed logistic regression still leads DAIC/CMDC, while fixed raw text-only
  XGBoost retains the best Turkish macro-F1.
- Majority-class and subject-shuffled-label controls failed on DAIC and CMDC.
  On positive-skewed Turkish data they instead expose why positive F1 alone is
  misleading: predicting every subject positive gives F1 0.818 but macro-F1
  only 0.409.

DAIC and CMDC are the predeclared held-out evaluations described below.
Turkish is a table-aligned five-fold outer-validation analysis; it is not an
unseen-test result.

## Important model-dimension correction

The implementation smoke test found that the plan's universal 3,584-feature
assumption does not match the deployed Qwen2-Audio checkpoint:

| Backbone | Observed final decoder dimension |
| --- | ---: |
| Qwen2-7B-Instruct (text) | 3,584 |
| Qwen2-Audio-7B-Instruct (audio and audio+text) | 4,096 |

The extractor now derives the decoder hidden size from the loaded checkpoint,
accepts only these known dimensions, asserts every extracted row against the
resolved value, and records it in cache and classifier metadata. No projection
was introduced merely to force the audio representation to 3,584 dimensions.

## Protocol

- Representation: `outputs.hidden_states[-1]` at the last valid prompt
  position, before generation.
- Model input: `prompt_text` only. No gold answer, labels tensor, or generated
  answer token enters Qwen extraction.
- Checkpoint: fold-specific `best_model` LoRA adapter and its matching base
  model.
- DAIC: deterministic subject-audio K=4 construction; train+val development
  subjects fit the classical head and official test remains locked evaluation.
- CMDC: one vector and prediction per response, followed by the repository's
  existing majority vote and probability-margin tie behavior.
- Turkish: the classical head fits each saved outer-training fold and evaluates
  the corresponding saved selection fold. Audio modalities are response-level
  before subject aggregation; text-only has one vector per subject.
- PCA: fit independently on each fold's training rows only.
- Class threshold: fixed at 0.5.
- Classifiers: predeclared XGBoost configuration, balanced logistic-regression
  control, majority-class control, and subject-level shuffled-label control.
- Tuned classifier: raw XGBoost only, with 50 sequential Optuna TPE trials per
  outer evaluation and three deterministic stratified subject folds. Each
  trial pools one out-of-fold prediction per training subject after the same
  response-majority/probability-margin aggregation used for final scoring.
  DAIC/CMDC maximize positive F1; Turkish maximizes macro-F1. The final outer
  partition is not loaded until tuning is complete.
- Emotion extension: frozen cache only; no SECap model ran during extraction.
  DAIC used local SECap English and Chinese caption conditions. CMDC used the
  Chinese captions distributed with the DepressInstruct repository.
- Emotion and no-emotion conditions use separately fine-tuned saved adapters,
  so their difference reflects the complete prompt/training condition rather
  than text injection into one identical checkpoint.

The baseline deltas below use the rounded "our" Qwen results in
`depression_results_table_no_emo.csv`; deltas are therefore approximate.

## DAIC official-test results

DAIC has one fixed official test split with 47 subjects (33 non-depressed and
14 depressed), so fold standard deviations are not applicable.

| Modality | Variant | Dim | ACC | Positive F1 | Macro F1 | AUROC | Delta ACC | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio+text | `xgb_raw` | 4,096 | 0.766 | 0.621 | 0.726 | 0.896 | 0.000 | -0.097 |
| Audio+text | `xgb_pca32` | 32 | 0.766 | 0.560 | 0.700 | 0.857 | 0.000 | -0.158 |
| Audio+text | `xgb_pca64` | 64 | 0.787 | 0.583 | 0.720 | 0.868 | +0.021 | -0.135 |
| Audio+text | `logreg_raw` | 4,096 | **0.851** | **0.800** | **0.841** | **0.900** | +0.085 | +0.082 |
| Audio+text | `logreg_pca32` | 32 | 0.787 | 0.667 | 0.755 | 0.848 | +0.021 | -0.051 |
| Audio-only | `xgb_raw` | 4,096 | **0.766** | 0.593 | **0.714** | 0.781 | +0.085 | -0.002 |
| Audio-only | `xgb_pca32` | 32 | 0.745 | 0.538 | 0.681 | 0.753 | +0.064 | -0.057 |
| Audio-only | `xgb_pca64` | 64 | 0.660 | 0.273 | 0.525 | 0.714 | -0.021 | -0.322 |
| Audio-only | `logreg_raw` | 4,096 | 0.723 | **0.629** | 0.704 | 0.779 | +0.042 | +0.034 |
| Audio-only | `logreg_pca32` | 32 | 0.745 | 0.600 | 0.706 | **0.786** | +0.064 | +0.005 |
| Text-only | `xgb_raw` | 3,584 | 0.830 | 0.692 | 0.787 | **0.931** | 0.000 | 0.000 |
| Text-only | `xgb_pca32` | 32 | 0.766 | 0.522 | 0.683 | 0.803 | -0.064 | -0.170 |
| Text-only | `xgb_pca64` | 64 | 0.766 | 0.522 | 0.683 | 0.835 | -0.064 | -0.170 |
| Text-only | `logreg_raw` | 3,584 | **0.851** | 0.741 | 0.818 | 0.926 | +0.021 | +0.049 |
| Text-only | `logreg_pca32` | 32 | **0.851** | **0.759** | **0.825** | 0.861 | +0.021 | +0.067 |

DAIC interpretation:

- The final representation is clearly useful, but the conservative XGBoost
  head is not uniformly the best way to read it.
- Audio+text raw logistic regression is the clearest improvement: it correctly
  identified all 14 depressed subjects, with confusion matrix
  `[[26, 7], [0, 14]]`.
- Audio-only remains difficult. Raw XGBoost improves overall accuracy through
  better negative-class performance, while positive F1 is essentially equal to
  the current Qwen verdict baseline.
- Aggressive PCA can remove useful DAIC signal, particularly for audio-only and
  text-only XGBoost.

## CMDC pooled five-fold results

The pooled out-of-fold set contains all 78 CMDC subjects exactly once (52
non-depressed and 26 depressed). Metrics below are pooled; per-fold means and
standard deviations remain in `outputs/hidden_classifiers/summary.csv`.

| Modality | Variant | Dim | ACC | Positive F1 | Macro F1 | AUROC | Delta ACC | Delta F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio+text | `xgb_raw` | 4,096 | 0.974 | 0.960 | 0.971 | 0.993 | +0.066 | +0.064 |
| Audio+text | `xgb_pca32` | 32 | 0.974 | 0.960 | 0.971 | 0.990 | +0.066 | +0.064 |
| Audio+text | `xgb_pca64` | 64 | 0.974 | 0.960 | 0.971 | 0.991 | +0.066 | +0.064 |
| Audio+text | `logreg_raw` | 4,096 | **0.987** | **0.980** | **0.985** | **0.996** | +0.079 | +0.084 |
| Audio+text | `logreg_pca32` | 32 | **0.987** | **0.980** | **0.985** | 0.994 | +0.079 | +0.084 |
| Audio-only | `xgb_raw` | 4,096 | **0.987** | **0.980** | **0.985** | 0.985 | +0.038 | +0.062 |
| Audio-only | `xgb_pca32` | 32 | 0.949 | 0.917 | 0.940 | 0.989 | 0.000 | -0.001 |
| Audio-only | `xgb_pca64` | 64 | 0.949 | 0.917 | 0.940 | 0.989 | 0.000 | -0.001 |
| Audio-only | `logreg_raw` | 4,096 | **0.987** | **0.980** | **0.985** | 0.986 | +0.038 | +0.062 |
| Audio-only | `logreg_pca32` | 32 | 0.962 | 0.943 | 0.957 | **0.990** | +0.013 | +0.025 |
| Text-only | `xgb_raw` | 3,584 | 0.974 | 0.960 | 0.971 | 0.999 | +0.038 | +0.056 |
| Text-only | `xgb_pca32` | 32 | **0.987** | **0.980** | **0.985** | **1.000** | +0.051 | +0.076 |
| Text-only | `xgb_pca64` | 64 | 0.974 | 0.960 | 0.971 | 0.999 | +0.038 | +0.056 |
| Text-only | `logreg_raw` | 3,584 | **0.987** | **0.980** | **0.985** | 0.999 | +0.051 | +0.076 |
| Text-only | `logreg_pca32` | 32 | **0.987** | **0.980** | **0.985** | 0.999 | +0.051 | +0.076 |

CMDC interpretation:

- Every primary hidden-state variant beats the corresponding audio+text and
  text-only Qwen verdict baseline. Audio-only PCA XGBoost is approximately tied
  with its baseline, while raw heads improve it.
- The best thresholded results make only one error among 78 subjects, with
  confusion matrix `[[52, 0], [1, 25]]`.
- PCA-32 retains nearly all useful CMDC information for text-only and
  audio+text logistic regression. The same reduction is less reliable for
  audio-only XGBoost.
- Near-perfect CMDC scores should be interpreted with the 78-subject sample
  size and repeated-response structure in mind. The split remains strictly
  subject-disjoint, but external validation is still necessary.

## Turkish table-aligned five-fold results

The pooled folds contain 120 unique subjects exactly once: 83 BDI-positive and
37 BDI-negative. The old baseline CSV says `54 dep / 24 non`, which totals only
78 and is stale or inconsistent with the saved folds. Its prevalence value,
0.692, does match the observed 83/120 prevalence.

These are not strict nested-CV estimates. The classical heads use only the
saved outer-training subjects, but each Qwen `best_model` adapter was selected
using the same outer validation fold evaluated here. This reproduces the
baseline table's protocol and should be labeled **table-aligned outer
validation**, not test evaluation.

Values before parentheses are pooled metrics. Parentheses show the five-fold
mean and sample standard deviation.

| Modality | Variant | Dim | ACC | Positive F1 (mean±SD) | Macro F1 (mean±SD) | AUROC | Delta F1 vs Qwen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio+text | `logreg_raw` | 4,096 | 0.600 | 0.724 (0.724±0.092) | 0.498 (0.497±0.157) | 0.480 | -0.085 |
| Audio+text | `logreg_pca32` | 32 | 0.592 | 0.696 (0.693±0.162) | **0.538** (0.551±0.220) | 0.531 | -0.113 |
| Audio+text | `xgb_raw` | 4,096 | 0.650 | 0.784 (0.780±0.078) | 0.435 (0.429±0.066) | **0.600** | -0.025 |
| Audio+text | `xgb_pca32` | 32 | **0.692** | **0.814** (0.815±0.034) | 0.456 (0.452±0.114) | 0.579 | +0.005 |
| Audio+text | `xgb_pca64` | 64 | **0.692** | **0.814** (0.815±0.034) | 0.456 (0.452±0.114) | 0.596 | +0.005 |
| Audio-only | `logreg_raw` | 4,096 | 0.683 | 0.789 (0.787±0.091) | **0.578** (0.583±0.119) | 0.614 | -0.058 |
| Audio-only | `logreg_pca32` | 32 | 0.658 | 0.771 (0.766±0.078) | 0.549 (0.526±0.075) | 0.645 | -0.076 |
| Audio-only | `xgb_raw` | 4,096 | **0.717** | **0.830** (0.831±0.021) | 0.490 (0.482±0.109) | 0.642 | -0.017 |
| Audio-only | `xgb_pca32` | 32 | 0.700 | 0.822 (0.822±0.014) | 0.437 (0.436±0.062) | 0.675 | -0.025 |
| Audio-only | `xgb_pca64` | 64 | 0.700 | 0.822 (0.822±0.014) | 0.437 (0.436±0.062) | **0.684** | -0.025 |
| Text-only | `logreg_raw` | 3,584 | 0.583 | 0.667 (0.665±0.094) | 0.556 (0.556±0.095) | 0.616 | -0.154 |
| Text-only | `logreg_pca32` | 32 | 0.642 | 0.723 (0.723±0.030) | 0.608 (0.606±0.052) | 0.647 | -0.098 |
| Text-only | `xgb_raw` | 3,584 | **0.683** | 0.774 (0.772±0.068) | **0.623** (0.621±0.093) | **0.677** | -0.047 |
| Text-only | `xgb_pca32` | 32 | 0.658 | **0.778** (0.776±0.036) | 0.516 (0.513±0.032) | 0.610 | -0.043 |
| Text-only | `xgb_pca64` | 64 | 0.633 | 0.761 (0.759±0.045) | 0.488 (0.481±0.064) | 0.594 | -0.060 |

Turkish interpretation:

- No primary hidden-state head beats the corresponding saved Qwen baseline on
  both positive F1 and macro-F1. Audio+text PCA XGBoost improves positive F1 by
  only 0.005 while reducing macro-F1 from 0.488 to 0.456.
- Audio-only raw XGBoost predicts every positive subject correctly but also
  labels 34/37 negative subjects positive (`[[3, 34], [0, 83]]`). Its F1 0.830
  is therefore close to the all-positive control and below the Qwen F1 0.847.
- Text-only raw XGBoost is the most balanced hidden probe:
  `[[17, 20], [18, 65]]`, macro-F1 0.623. It improves macro-F1 by 0.111 over
  the saved Qwen text-only baseline, but loses 0.047 positive F1 and 0.026
  accuracy.
- Unlike DAIC and CMDC, logistic regression is not the stronger Turkish head.
  Its best macro-F1 is 0.608 for text-only PCA-32, while raw XGBoost reaches
  0.623.

## Optuna-tuned raw XGBoost extension

This extension tuned only the raw XGBoost head. It reused the existing frozen
hidden matrices and ran no Qwen extraction, Qwen training, PCA, emotion
condition, GPU code, early stopping, or threshold selection. Each of the 33
outer evaluations ran 50 sequential TPE trials with three subject-disjoint
inner folds. The threshold remained 0.5. DAIC and CMDC optimized pooled
subject-level positive F1; Turkish optimized pooled subject-level macro-F1.

Values before parentheses are pooled outer metrics. Parentheses contain the
outer-fold mean and sample standard deviation; DAIC has one fixed outer split,
so its SD is zero.

| Dataset | Modality | ACC | Positive F1 (mean±SD) | Macro F1 (mean±SD) | Neg F1 | Precision | Recall | AUROC | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DAIC | Audio+text | 0.787 | 0.643 (0.643±0.000) | 0.746 (0.746±0.000) | 0.848 | 0.643 | 0.643 | 0.883 | `[[28, 5], [5, 9]]` |
| DAIC | Audio-only | 0.723 | 0.581 (0.581±0.000) | 0.687 (0.687±0.000) | 0.794 | 0.529 | 0.643 | 0.786 | `[[25, 8], [5, 9]]` |
| DAIC | Text-only | 0.851 | 0.741 (0.741±0.000) | 0.818 (0.818±0.000) | 0.896 | 0.769 | 0.714 | 0.911 | `[[30, 3], [4, 10]]` |
| CMDC | Audio+text | 0.987 | 0.980 (0.978±0.050) | 0.985 (0.984±0.035) | 0.990 | 1.000 | 0.962 | 0.994 | `[[52, 0], [1, 25]]` |
| CMDC | Audio-only | 0.987 | 0.980 (0.978±0.050) | 0.985 (0.984±0.035) | 0.990 | 1.000 | 0.962 | 0.987 | `[[52, 0], [1, 25]]` |
| CMDC | Text-only | 0.974 | 0.962 (0.962±0.053) | 0.971 (0.972±0.039) | 0.981 | 0.962 | 0.962 | 0.997 | `[[51, 1], [1, 25]]` |
| Turkish | Audio+text | 0.625 | 0.746 (0.743±0.094) | 0.516 (0.506±0.151) | 0.286 | 0.702 | 0.795 | 0.576 | `[[9, 28], [17, 66]]` |
| Turkish | Audio-only | 0.658 | 0.771 (0.769±0.087) | 0.549 (0.549±0.109) | 0.328 | 0.719 | 0.831 | 0.623 | `[[10, 27], [14, 69]]` |
| Turkish | Text-only | 0.642 | 0.726 (0.727±0.036) | 0.604 (0.603±0.059) | 0.482 | 0.770 | 0.687 | 0.647 | `[[20, 17], [26, 57]]` |

The comparison metric is positive F1 for DAIC/CMDC and macro-F1 for Turkish.
Qwen deltas use the rounded values in
`depression_results_table_no_emo.csv`; the other deltas use unrounded pooled
classifier metrics.

| Dataset | Modality | Tuned metric | Δ vs fixed raw XGB | Δ vs raw LogReg | Δ vs majority | Δ vs Qwen |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DAIC | Audio+text | 0.643 positive F1 | +0.022 | -0.157 | +0.643 | -0.075 |
| DAIC | Audio-only | 0.581 positive F1 | -0.012 | -0.048 | +0.581 | -0.014 |
| DAIC | Text-only | 0.741 positive F1 | +0.048 | +0.000 | +0.741 | +0.049 |
| CMDC | Audio+text | 0.980 positive F1 | +0.020 | +0.000 | +0.980 | +0.084 |
| CMDC | Audio-only | 0.980 positive F1 | +0.000 | +0.000 | +0.980 | +0.062 |
| CMDC | Text-only | 0.962 positive F1 | +0.002 | -0.019 | +0.962 | +0.058 |
| Turkish | Audio+text | 0.516 macro-F1 | +0.081 | +0.017 | +0.107 | +0.028 |
| Turkish | Audio-only | 0.549 macro-F1 | +0.059 | -0.028 | +0.141 | -0.064 |
| Turkish | Text-only | 0.604 macro-F1 | -0.019 | +0.048 | +0.195 | +0.092 |

Interpretation:

- Tuning gives a useful but limited gain for CMDC audio+text, reaching the raw
  logistic result with one error among 78 pooled subjects. CMDC audio-only is
  unchanged, and text-only gains only 0.002 positive F1 while introducing one
  false positive.
- DAIC text-only improves from 0.692 to 0.741 positive F1 and matches raw
  logistic regression, but not the 0.759 PCA-32 logistic result. DAIC
  audio+text improves modestly and audio-only declines.
- Optimizing Turkish macro-F1 makes the audio+text and audio-only heads more
  balanced than fixed raw XGBoost, but their positive F1 falls substantially.
  Tuned text-only reaches macro-F1 0.604, below fixed raw text-only XGBoost
  (0.623) and slightly below PCA-32 logistic regression (0.608). Tuning
  therefore does not change the primary Turkish ranking.
- Turkish remains table-aligned outer validation, not unseen-test evaluation,
  because the saved Qwen adapters were selected using those outer validation
  folds.

## Emotion-extension results

The DAIC probe still trains on all 142 development subjects and evaluates only
the 47 official-test subjects. This extension deliberately does **not** report a
paper-matched 107-train/35-validation result, because the available adapters
were selected using validation performance. CMDC retains five-fold
subject-disjoint out-of-fold evaluation.

### Headline comparison

The Qwen-head values are the corresponding saved verdict-head results. The
hidden classifiers use the final prompt-position representation from that same
condition's adapter.

| Dataset | Condition | Qwen ACC | Qwen F1 | Raw logistic ACC | Raw logistic F1 | Raw XGBoost ACC | Raw XGBoost F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAIC | Audio+text, no emotion | 0.766 | 0.718 | **0.851** | **0.800** | 0.766 | 0.621 |
| DAIC | Audio+text+SECap EN | 0.766 | 0.703 | 0.766 | 0.703 | **0.830** | **0.714** |
| DAIC | Audio+text+SECap ZH | 0.809 | 0.571 | **0.851** | **0.788** | 0.809 | 0.640 |
| CMDC | Audio+text, no emotion | 0.908 | 0.896 | **0.987** | **0.980** | 0.974 | 0.960 |
| CMDC | Audio+text+paper SECap ZH | 0.949 | 0.920 | **0.987** | **0.980** | **0.987** | **0.980** |

For DAIC, no-emotion raw logistic remains the strongest thresholded result. Its
confusion matrix is `[[26, 7], [0, 14]]`, compared with
`[[23, 10], [1, 13]]` for English emotion and `[[27, 6], [1, 13]]` for Chinese
emotion. Chinese captions slightly improve AUROC from 0.900 to 0.911, but at the
fixed 0.5 threshold lose one depressed subject and therefore do not improve F1.

For CMDC, no-emotion and paper-emotion raw logistic have the same confusion
matrix, `[[52, 0], [1, 25]]`. Emotion changes the representation and ranking but
does not change the best thresholded logistic result. Raw XGBoost improves from
two false negatives without emotion to one with emotion.

### Complete primary emotion matrix

| Dataset | Condition | Variant | Dim | ACC | Positive F1 | Macro F1 | AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DAIC | SECap EN | `logreg_raw` | 4,096 | 0.766 | 0.703 | 0.755 | 0.892 |
| DAIC | SECap EN | `logreg_pca32` | 32 | 0.723 | 0.606 | 0.696 | 0.851 |
| DAIC | SECap EN | `xgb_raw` | 4,096 | **0.830** | **0.714** | **0.797** | 0.900 |
| DAIC | SECap EN | `xgb_pca32` | 32 | 0.787 | 0.615 | 0.734 | **0.903** |
| DAIC | SECap EN | `xgb_pca64` | 64 | 0.809 | 0.640 | 0.755 | 0.887 |
| DAIC | SECap ZH | `logreg_raw` | 4,096 | **0.851** | **0.788** | **0.837** | **0.911** |
| DAIC | SECap ZH | `logreg_pca32` | 32 | 0.745 | 0.625 | 0.716 | 0.851 |
| DAIC | SECap ZH | `xgb_raw` | 4,096 | 0.809 | 0.640 | 0.755 | 0.877 |
| DAIC | SECap ZH | `xgb_pca32` | 32 | 0.766 | 0.353 | 0.605 | 0.829 |
| DAIC | SECap ZH | `xgb_pca64` | 64 | 0.723 | 0.235 | 0.533 | 0.859 |
| CMDC | Paper SECap ZH | `logreg_raw` | 4,096 | **0.987** | **0.980** | **0.985** | 0.993 |
| CMDC | Paper SECap ZH | `logreg_pca32` | 32 | 0.974 | 0.960 | 0.971 | **0.997** |
| CMDC | Paper SECap ZH | `xgb_raw` | 4,096 | **0.987** | **0.980** | **0.985** | 0.993 |
| CMDC | Paper SECap ZH | `xgb_pca32` | 32 | 0.962 | 0.939 | 0.955 | 0.993 |
| CMDC | Paper SECap ZH | `xgb_pca64` | 64 | 0.962 | 0.939 | 0.955 | 0.995 |

The full matrix reinforces the original PCA finding: reducing to 32 or 64
components is not reliably beneficial, especially for DAIC Chinese-emotion
XGBoost.

### Emotion-cache provenance

| Condition | Cache SHA-256 | Coverage | Missing behavior |
| --- | --- | ---: | --- |
| DAIC SECap EN | `1a2f4c3ed1a4d25f86b072e2e6d0ee376525725c1c8a5a653f35a533942a2b90` | 2,127/2,170 clips | 43 null translations use the saved neutral fallback: 35 development, 8 test |
| DAIC SECap ZH | `42c3655245e4888ffa9c4236e9a2f66f7ec44a5dea32046b1850f633cbd4f11c` | 2,170/2,170 clips | None |
| CMDC paper SECap ZH | `8383d12987e5376d037a7f678048153fd64ce101aaac862f5a0a3d35f9aa51f0` | 923/923 responses | None |

## Negative controls

The shuffled control permutes labels between subjects and preserves all response
rows within a subject as one group.

| Dataset | Modality | Control | ACC | Positive F1 | AUROC |
| --- | --- | --- | ---: | ---: | ---: |
| DAIC | Audio+text | Majority class | 0.702 | 0.000 | 0.500 |
| DAIC | Audio+text | Subject-shuffled XGBoost | 0.702 | 0.000 | 0.617 |
| DAIC | Audio-only | Majority class | 0.702 | 0.000 | 0.500 |
| DAIC | Audio-only | Subject-shuffled XGBoost | 0.681 | 0.000 | 0.656 |
| DAIC | Text-only | Majority class | 0.702 | 0.000 | 0.500 |
| DAIC | Text-only | Subject-shuffled XGBoost | 0.702 | 0.000 | 0.578 |
| CMDC | Audio+text | Majority class | 0.667 | 0.000 | 0.485 |
| CMDC | Audio+text | Subject-shuffled XGBoost | 0.667 | 0.000 | 0.475 |
| CMDC | Audio-only | Majority class | 0.667 | 0.000 | 0.485 |
| CMDC | Audio-only | Subject-shuffled XGBoost | 0.667 | 0.000 | 0.557 |
| CMDC | Text-only | Majority class | 0.667 | 0.000 | 0.485 |
| CMDC | Text-only | Subject-shuffled XGBoost | 0.667 | 0.000 | 0.324 |
| Turkish | Audio+text | Majority class | 0.692 | 0.818 | 0.487 |
| Turkish | Audio+text | Subject-shuffled XGBoost | 0.692 | 0.818 | 0.558 |
| Turkish | Audio-only | Majority class | 0.692 | 0.818 | 0.487 |
| Turkish | Audio-only | Subject-shuffled XGBoost | 0.692 | 0.818 | 0.546 |
| Turkish | Text-only | Majority class | 0.692 | 0.818 | 0.486 |
| Turkish | Text-only | Subject-shuffled XGBoost | 0.667 | 0.789 | 0.623 |
| DAIC | Audio+text+SECap EN | Majority class | 0.702 | 0.000 | 0.500 |
| DAIC | Audio+text+SECap EN | Subject-shuffled XGBoost | 0.702 | 0.000 | 0.519 |
| DAIC | Audio+text+SECap ZH | Majority class | 0.702 | 0.000 | 0.500 |
| DAIC | Audio+text+SECap ZH | Subject-shuffled XGBoost | 0.702 | 0.000 | 0.394 |
| CMDC | Audio+text+paper SECap ZH | Majority class | 0.667 | 0.000 | 0.485 |
| CMDC | Audio+text+paper SECap ZH | Subject-shuffled XGBoost | 0.667 | 0.000 | 0.539 |

The constant-probability majority control has pooled CMDC AUROC 0.485 rather
than exactly 0.5 because the pooled rank calculation sees tiny fold-to-fold
training-prevalence differences. Its threshold behavior is the meaningful
control result.

## Integrity and reproducibility checks

- The Turkish extension adds 15 extraction metadata files and 11,110 feature
  rows, for 40 extraction metadata files and 30,515 total feature rows overall.
- All repeated-vector determinism checks had maximum absolute difference 0.0.
- All model inputs were built from `prompt_text`; no `labels` key or metadata
  key was passed to Qwen and no generation was performed.
- Every current manifest and split hash matched the corresponding checkpoint's
  saved hash before extraction.
- All training and held-out subject intersections were empty.
- All 280 fixed-classifier metadata files (40 folds/checkpoints x 7 variants)
  passed overlap and PCA-component checks. The 33 tuned metadata files also
  passed the complete outer/inner subject audit.
- CMDC pooled held-out predictions cover 78 unique subjects exactly once for
  every variant.
- Turkish pooled outer-validation predictions cover 120 unique subjects exactly
  once for every variant.
- The tuned matrix contains 33 studies with exactly 1,650 completed trials,
  4,950 inner fits, and 33 final fits. Every trial stays inside its declared
  search bounds and stores pooled subject-level OOF metrics.
- Every tuned inner split is deterministic, stratified, subject-disjoint, and
  covers each outer-training subject exactly once. Every final outer
  train/evaluation intersection is empty and every study/configuration SHA-256
  matches its persisted Optuna attributes.
- Feature extraction records implementation commit
  `14fe9d022ec95d54866e75aa75684921e5a27659`; the grouped subject-shuffle
  correction is commit `8f8584f4e044252ddf04b7d02a0e220a0eb709a7`. Emotion extraction records
  condition-aware implementation commit
  `2242f1f039093d7fe890b3ea882017260e3b1f92`.
- Turkish extraction records protocol-aware implementation commit
  `418f17bbbdc578ba44197d0f3287a5be7b936f95`.
- Optuna tuning uses implementation commit
  `3d41c974b5779e261a72c39e4fa9c720a00d542d`.
- Runtime: Python 3.10.14, Torch 2.3.0+cu121, Transformers 4.55.0, PEFT 0.17.0,
  scikit-learn 1.7.0, project-local Optuna 4.4.0, and project-local
  `xgboost-cpu` 2.1.4.

## Slurm execution

- Text smoke: `43667807` (completed).
- Initial audio smoke: `43667808` (failed safely on the incorrect 3,584
  assertion; no primary result produced).
- Corrected audio smoke: `43668069` (completed, 4,096 dimensions).
- Primary matrix: `43668129` through `43668146` (18/18 completed, exit code 0).
- Corrected grouped control rerun: `43668444` (completed, exit code 0).
- Emotion smoke: `43671537` (completed, exit code 0).
- Emotion matrix: `43671578` through `43671584` (7/7 completed, exit code 0).
- Turkish smoke: `43705279` (completed, exit code 0).
- Turkish matrix: `43705723` through `43705745` (15/15 completed, exit code 0).
- Optuna repeated-response smoke: `43727515`; idempotent resume check:
  `43727544` (both completed, exit code 0).
- Optuna production matrix: `43727794` through `43727828`, with scheduler ID
  gaps (33/33 completed, exit code 0).

Original primary job wall times ranged from 1:28 to 4:47. Optuna production
wall times ranged from 1:16 to 28:21. No production log contained a traceback,
out-of-memory marker, killed-process marker, or error signature.

## Artifacts

- Pooled and fold summary: `outputs/hidden_classifiers/summary.csv` and
  `summary.json`.
- Per-fold metrics, pipelines, sample predictions, subject predictions, and
  classifier provenance: `outputs/hidden_classifiers/<dataset>/<modality>/...`.
- Feature matrices and extraction provenance:
  `outputs/hidden_features/<dataset>/<condition>/...`.
- Slurm logs: `logs/slurm_qwen_hidden/`.
- Optuna studies and complete per-fold tuning provenance:
  `outputs/hidden_classifiers/<dataset>/<condition>/.../xgb_optuna_raw/`.
- Optuna logs and acceptance audit: `logs/slurm_qwen_hidden_optuna/`.
- Experiment matrices: `configs/features/primary_matrix.yaml` and
  `configs/features/emotion_matrix.yaml`, and
  `configs/features/turkish_matrix.yaml`. The tuned matrix is
  `configs/features/optuna_raw_matrix.yaml`.

The earlier 149 MB DAIC/CMDC feature cache remains synchronized locally. The
new Turkish feature matrices remain on GPFS; their compact extraction metadata
and the classifier result tree are available locally. The tuned artifacts add
approximately 53 MB, including all 33 SQLite studies and final fitted
pipelines. No model checkpoints were copied as part of this experiment.

## Conclusion

The final prompt-position hidden state is a materially better CMDC depression
representation than the model's current verdict-token head. On DAIC, it is also
useful, but classifier choice matters: linear heads are more reliable than the
predeclared conservative XGBoost configuration, and low-dimensional PCA can
discard important signal. Frozen emotion descriptions do not improve the
strongest logistic result: they are nearly neutral for DAIC Chinese captions
and CMDC paper captions, and harmful for DAIC English captions. They can help
the weaker raw XGBoost head, but that does not overturn logistic regression as
the preferred probe on DAIC/CMDC. Optuna closes some gaps but does not change
that conclusion. Turkish does not show the same pattern: XGBoost is stronger
than logistic regression there, but most of its positive-F1 score comes from
the 69% positive class prior, and macro-F1 tuning does not beat fixed raw
text-only XGBoost. The most defensible next step is strict nested or external
validation using macro-F1 as a co-primary metric, not selection of a new
variant from these evaluation subjects.
