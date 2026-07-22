# Qwen2 Final Hidden-State Classifier Experiment Report

Date: 2026-07-22

Implementation plan: [`QWEN2_HIDDEN_XGBOOST_IMPLEMENTATION_PLAN_2026-07-22.md`](QWEN2_HIDDEN_XGBOOST_IMPLEMENTATION_PLAN_2026-07-22.md)

## Summary

The prompt-only final-hidden-state pipeline was implemented and run for the
aligned DAIC and CMDC audio+text, audio-only, and text-only checkpoints.
Eighteen fold/checkpoint extraction jobs and one grouped control rerun completed
successfully on MareNostrum 5.

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
- Majority-class and subject-shuffled-label controls failed at the 0.5 decision
  threshold, supporting that the primary results are not class-prior artifacts.

These are predeclared official held-out results, not a basis for selecting a
new variant on the same test subjects.

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
- PCA: fit independently on each fold's training rows only.
- Class threshold: fixed at 0.5.
- Classifiers: predeclared XGBoost configuration, balanced logistic-regression
  control, majority-class control, and subject-level shuffled-label control.

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

The constant-probability majority control has pooled CMDC AUROC 0.485 rather
than exactly 0.5 because the pooled rank calculation sees tiny fold-to-fold
training-prevalence differences. Its threshold behavior is the meaningful
control result.

## Integrity and reproducibility checks

- 18 extraction metadata files were produced.
- 14,412 total feature rows were extracted across training and held-out caches.
- All repeated-vector determinism checks had maximum absolute difference 0.0.
- All model inputs were built from `prompt_text`; no `labels` key or metadata
  key was passed to Qwen and no generation was performed.
- Every current manifest and split hash matched the corresponding checkpoint's
  saved hash before extraction.
- All training and held-out subject intersections were empty.
- All 126 classifier metadata files (18 folds/checkpoints x 7 variants) passed
  overlap and PCA-component checks.
- CMDC pooled held-out predictions cover 78 unique subjects exactly once for
  every variant.
- Feature extraction records implementation commit
  `14fe9d022ec95d54866e75aa75684921e5a27659`; the grouped subject-shuffle
  correction is commit `8f8584f4e044252ddf04b7d02a0e220a0eb709a7`.
- Runtime: Python 3.10.14, Torch 2.3.0+cu121, Transformers 4.55.0, PEFT 0.17.0,
  scikit-learn 1.7.0, and project-local `xgboost-cpu` 2.1.4.

## Slurm execution

- Text smoke: `43667807` (completed).
- Initial audio smoke: `43667808` (failed safely on the incorrect 3,584
  assertion; no primary result produced).
- Corrected audio smoke: `43668069` (completed, 4,096 dimensions).
- Primary matrix: `43668129` through `43668146` (18/18 completed, exit code 0).
- Corrected grouped control rerun: `43668444` (completed, exit code 0).

Primary job wall times ranged from 1:28 to 4:40. No primary job log contained a
traceback, CUDA OOM, killed-process marker, or error signature.

## Artifacts

- Pooled and fold summary: `outputs/hidden_classifiers/summary.csv` and
  `summary.json`.
- Per-fold metrics, pipelines, sample predictions, subject predictions, and
  classifier provenance: `outputs/hidden_classifiers/<dataset>/<modality>/...`.
- Feature matrices and extraction provenance:
  `outputs/hidden_features/<dataset>/<modality>/...`.
- Slurm logs: `logs/slurm_qwen_hidden/`.
- Experiment matrix: `configs/features/primary_matrix.yaml`.

The full 110 MB feature cache remains on GPFS and has also been synchronized
locally. The 79 MB classifier result tree and compact extraction metadata are
available locally. No model checkpoints were copied as part of this experiment.

## Conclusion

The final prompt-position hidden state is a materially better CMDC depression
representation than the model's current verdict-token head. On DAIC, it is also
useful, but classifier choice matters: linear heads are more reliable than the
predeclared conservative XGBoost configuration, and low-dimensional PCA can
discard important signal. The most defensible next step is external or
cross-dataset validation of the fixed raw logistic and raw XGBoost heads, not
selection of a new variant from these official-test results.
