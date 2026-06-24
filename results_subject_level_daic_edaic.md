# Subject-level results — DAIC, EDAIC, CMDC, EATD (held-out test)

> ⚠️ **STALE — audio runs leaked LoRA into the audio encoder.** All `subject_audio*` and
> `audio+text` numbers below were produced before the audio-encoder freeze fix: LoRA `target_modules`
> (`q/k/v_proj`) matched the Whisper `audio_tower`, so the encoder was being fine-tuned (overfit
> liability). Re-run the DAIC/EDAIC reg sweep with `exclude_modules` default-on (see handoff §0) and
> replace these tables. `text_only` rows are unaffected.


Best checkpoint (selected on inner-val), subject-level aggregation. F1 = positive-class F1.
Confusion matrix = `[TN, FP / FN, TP]`. `subject_audio*` = fixed K=4 chunk sampling.
DAIC/EDAIC = fixed train/val/test. CMDC/EATD = CV (metrics pooled over fold holdouts).

## DAIC — test N=47 (33 non-dep / 14 dep)
All-negative baseline: ACC 0.702, F1 0.000.

### Current paper-comparison runs

These rows are the newer frozen-encoder / paper-comparison runs used for the DepressInstruct comparison below. F1 = positive-class F1.

| run | config / source | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- | --- |
| Audio + Text (our) | `configs/daic_subject_audio_text_reg1_selmacrof1_tf.yaml` | 0.787 | 0.737 | 0.583 | 1.000 | — |
| Audio + Text + Emo (EN SECap) | `configs/daic_subject_audio_text_emotion_k4.yaml` | 0.766 | 0.703 | 0.565 | 0.929 | — |
| Audio + Text + Emo (ZH SECap) | `configs/daic_subject_audio_text_emotion_zh_k4.yaml` | 0.809 | 0.571 | 0.857 | 0.429 | — |

| run | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- |
| subject_audio reg1 | 0.574 | 0.545 | 0.400 | 0.857 | 15,18 / 2,12 |
| subject_audio reg2 | 0.298 | 0.459 | 0.298 | 1.000 | 0,33 / 0,14 |
| **subject_audio reg3** | 0.745 | 0.600 | 0.562 | 0.643 | 26,7 / 5,9 |
| subject_audio reg4 | 0.723 | 0.629 | 0.524 | 0.786 | 23,10 / 3,11 |
| text_only reg1 | 0.766 | 0.621 | 0.600 | 0.643 | 27,6 / 5,9 |
| text_only reg2 | 0.745 | 0.667 | 0.545 | 0.857 | 23,10 / 2,12 |
| text_only reg3 | 0.830 | 0.692 | 0.750 | 0.643 | 30,3 / 5,9 |
| text_only reg4 | 0.809 | 0.640 | 0.727 | 0.571 | 30,3 / 6,8 |
| **audio+text reg1** | **0.851** | **0.696** | 0.889 | 0.571 | 32,1 / 6,8 |
| audio+text reg2 | 0.298 | 0.459 | 0.298 | 1.000 | 0,33 / 0,14 |
| audio+text reg3 | 0.830 | 0.636 | 0.875 | 0.500 | 32,1 / 7,7 |
| audio+text reg4 | 0.787 | 0.667 | 0.625 | 0.714 | 27,6 / 4,10 |

## EDAIC — test N=56 (39 non-dep / 17 dep)
All-negative baseline: ACC 0.696, F1 0.000.

| run | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- |
| audio_only reg2 *(chunk baseline)* | 0.679 | 0.437 | 0.467 | 0.412 | 31,8 / 10,7 |
| subject_audio reg2 | 0.536 | 0.500 | 0.371 | 0.765 | 17,22 / 4,13 |
| **subject_audio reg3** | 0.768 | 0.552 | 0.667 | 0.471 | 35,4 / 9,8 |
| subject_audio reg4 | 0.750 | 0.500 | 0.636 | 0.412 | 35,4 / 10,7 |
| **text_only reg1** | 0.732 | 0.667 | 0.536 | 0.882 | 26,13 / 2,15 |
| text_only reg5 | 0.732 | 0.634 | 0.542 | 0.765 | 28,11 / 4,13 |
| text_only reg2 | 0.661 | 0.596 | 0.467 | 0.824 | 23,16 / 3,14 |
| text_only reg3/reg4 | 0.679 | 0.591 | 0.481 | 0.765 | 25,14 / 4,13 |
| text_only reg6 | 0.661 | 0.578 | 0.464 | 0.765 | 24,15 / 4,13 |
| **audio+text reg2** | 0.768 | 0.667 | 0.591 | 0.765 | 30,9 / 4,13 |
| audio+text reg3 | 0.732 | 0.516 | 0.571 | 0.471 | 33,6 / 9,8 |
| audio+text reg4 | 0.714 | 0.652 | 0.517 | 0.882 | 25,14 / 2,15 |

## CMDC — 5-fold CV pooled, N=78 (52 non-dep / 26 dep)
All-negative baseline: ACC 0.667, F1 0.000.

| run | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- |
| audio+text *(default cfg; `configs/cmdc_audio_text.yaml`)* | 0.987 | 0.981 | 0.963 | 1.000 | 51,1 / 0,26 |
| audio+text+emotion_zh *(CV mean; `configs/cmdc_audio_text_emotion_zh.yaml`)* | 0.937 | 0.887 | 1.000 | 0.807 | pooled: 52,0 / 5,21 |
| audio+text+emotion_zh *(pooled metrics; same run)* | 0.936 | 0.894 | 1.000 | 0.808 | 52,0 / 5,21 |
| audio+text+emotion_paper_zh *(CV mean; `configs/cmdc_audio_text_emotion_paper_zh.yaml`)* | 0.948 | 0.919 | 0.967 | 0.887 | pooled: 51,1 / 3,23 |
| audio+text+emotion_paper_zh *(pooled metrics; same run)* | 0.949 | 0.920 | 0.958 | 0.885 | 51,1 / 3,23 |

⚠️ Near-perfect on a 78-subject corpus — **treat as suspect** (likely a corpus-wide artifact / trivially separable, not a clean win). Needs leakage scrutiny before reporting.

## EATD — 3-fold CV pooled, N=162 (132 non-dep / 30 dep)
All-negative baseline: ACC 0.815, F1 0.000.

| run | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- |
| audio+text *(default cfg; `configs/eatd_audio_text.yaml`)* | 0.420 | 0.288 | 0.186 | 0.633 | 49,83 / 11,19 |
| audio+text+emotion_zh *(CV mean; `configs/eatd_audio_text_emotion_zh.yaml`)* | 0.611 | 0.312 | 0.475 | 0.467 | pooled: 85,47 / 16,14 |
| audio+text+emotion_zh *(pooled metrics; same run)* | 0.611 | 0.308 | 0.230 | 0.467 | 85,47 / 16,14 |
| audio+text+emotion_paper_zh *(CV mean; `configs/eatd_audio_text_emotion_paper_zh.yaml`)* | 0.728 | 0.383 | 0.365 | 0.433 | pooled: 105,27 / 17,13 |
| audio+text+emotion_paper_zh *(pooled metrics; same run)* | 0.728 | 0.371 | 0.325 | 0.433 | 105,27 / 17,13 |

⚠️ **Below baseline** — collapsed to over-predicting depression (83 FP). Light reg (rank 16, lr 2e-4) on a heavily imbalanced set; needs the reg3 recipe.

## Best result per dataset

| dataset | best config | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- | --- |
| DAIC | **audio+text reg1** | 0.851 | 0.696 | 0.889 | 0.571 | 32,1 / 6,8 |
| EDAIC | **audio+text reg2** | 0.768 | 0.667 | 0.591 | 0.765 | 30,9 / 4,13 |
| CMDC | audio+text *(default)* ⚠️ | 0.987 | 0.981 | 0.963 | 1.000 | 51,1 / 0,26 |
| EATD | audio+text+emotion_paper_zh *(CV mean)* ⚠️ below baseline | 0.728 | 0.383 | 0.365 | 0.433 | pooled: 105,27 / 17,13 |

## DepressInstruct paper comparison

### DAIC

| Method | Source / config | ACC | F1 | Precision | Recall | Delta F1 vs paper proposed |
| --- | --- | --- | --- | --- | --- | --- |
| Audio + Text (our) | `configs/daic_subject_audio_text_reg1_selmacrof1_tf.yaml` | 0.787 | 0.737 | 0.583 | 1.000 | -0.087 |
| Audio + Text + Emo (our, EN SECap) | `configs/daic_subject_audio_text_emotion_k4.yaml` | 0.766 | 0.703 | 0.565 | 0.929 | -0.121 |
| Audio + Text + Emo (our, ZH SECap) | `configs/daic_subject_audio_text_emotion_zh_k4.yaml` | 0.809 | 0.571 | 0.857 | 0.429 | -0.252 |
| Audio + Text (paper) | DepressInstruct paper | 0.857 | 0.762 | 0.889 | 0.667 | -0.062 |
| Audio + Emotion (paper) | DepressInstruct paper | 0.857 | 0.737 | 1.000 | 0.583 | -0.087 |
| DepressInstruct (paper proposed) | DepressInstruct paper | 0.891 | 0.824 | 0.840 | 0.808 | 0.000 |

### CMDC

CMDC metrics are 5-fold CV. For our emotion_zh row, the main line uses fold means; pooled metrics are listed separately because they are computed from the summed confusion matrix.

| Method | Source / config | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] | Delta F1 vs paper proposed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Audio + Text (our default) | `configs/cmdc_audio_text.yaml` | 0.987 | 0.981 | 0.963 | 1.000 | 51,1 / 0,26 | -0.001 |
| Audio + Text + Emo (our, ZH SECap; CV mean) | `configs/cmdc_audio_text_emotion_zh.yaml` | 0.937 | 0.887 | 1.000 | 0.807 | pooled: 52,0 / 5,21 | -0.094 |
| Audio + Text + Emo (our, ZH SECap; pooled) | same run | 0.936 | 0.894 | 1.000 | 0.808 | 52,0 / 5,21 | -0.088 |
| Audio + Text + Emo (our, paper-prompt ZH SECap; CV mean) | `configs/cmdc_audio_text_emotion_paper_zh.yaml` | 0.948 | 0.919 | 0.967 | 0.887 | pooled: 51,1 / 3,23 | -0.063 |
| Audio + Text + Emo (our, paper-prompt ZH SECap; pooled) | same run | 0.949 | 0.920 | 0.958 | 0.885 | 51,1 / 3,23 | -0.062 |
| Audio + Text (paper) | DepressInstruct paper | 0.989 | 0.982 | 1.000 | 0.967 | — | 0.000 |
| Audio + Emotion (paper) | DepressInstruct paper | 0.989 | 0.982 | 1.000 | 0.967 | — | 0.000 |
| DepressInstruct (paper proposed) | DepressInstruct paper | 0.989 | 0.982 | 1.000 | 0.967 | — | 0.000 |

### EATD

EATD metrics are 3-fold CV. For our emotion_zh row, the main line uses fold means; pooled metrics are listed separately because they are computed from the summed confusion matrix.

| Method | Source / config | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] | Delta F1 vs paper proposed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Audio + Text (our default) | `configs/eatd_audio_text.yaml` | 0.420 | 0.288 | 0.186 | 0.633 | 49,83 / 11,19 | -0.530 |
| Audio + Text + Emo (our, ZH SECap; CV mean) | `configs/eatd_audio_text_emotion_zh.yaml` | 0.611 | 0.312 | 0.475 | 0.467 | pooled: 85,47 / 16,14 | -0.506 |
| Audio + Text + Emo (our, ZH SECap; pooled) | same run | 0.611 | 0.308 | 0.230 | 0.467 | 85,47 / 16,14 | -0.510 |
| Audio + Text + Emo (our, paper-prompt ZH SECap; CV mean) | `configs/eatd_audio_text_emotion_paper_zh.yaml` | 0.728 | 0.383 | 0.365 | 0.433 | pooled: 105,27 / 17,13 | -0.435 |
| Audio + Text + Emo (our, paper-prompt ZH SECap; pooled) | same run | 0.728 | 0.371 | 0.325 | 0.433 | 105,27 / 17,13 | -0.447 |
| Audio + Text (paper) | DepressInstruct paper | 0.937 | 0.800 | 0.714 | 0.909 | — | -0.018 |
| Audio + Emotion (paper) | DepressInstruct paper | 0.937 | 0.737 | 0.875 | 0.636 | — | -0.081 |
| DepressInstruct (paper proposed) | DepressInstruct paper | 0.949 | 0.818 | 0.818 | 0.818 | — | 0.000 |

### Final comparison

| Dataset | Our strongest comparable run | Paper proposed | Result |
| --- | --- | --- | --- |
| DAIC | Audio + Text: ACC 0.787, F1 0.737, P 0.583, R 1.000 | ACC 0.891, F1 0.824, P 0.840, R 0.808 | Our best comparable DAIC run is -0.087 F1 below the proposed model. EN SECap emotion hurts slightly; ZH SECap raises ACC/precision but collapses recall and F1. |
| CMDC | Audio + Text default: ACC 0.987, F1 0.981, P 0.963, R 1.000 | ACC 0.989, F1 0.982, P 1.000, R 0.967 | Our default is effectively tied with the paper on F1. The paper-prompt ZH SECap result (F1 0.919 mean) improves substantially over the earlier ZH extraction (0.887), but remains below audio+text. |
| EATD | Audio + Text + Emo paper-prompt ZH: ACC 0.728, F1 0.383, P 0.365, R 0.433 | ACC 0.949, F1 0.818, P 0.818, R 0.818 | This is our strongest EATD run and improves over both our default (F1 0.288) and earlier ZH extraction (0.312), but remains below the 0.815 all-negative ACC baseline and far below the paper. |

### Compact comparison table

Our CMDC/EATD emotion rows below use arithmetic means across folds, matching the CV summary's `*_mean` fields.

| Dataset | Method | ACC | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| DAIC | Audio + Text (our) | 0.787 | 0.737 | 0.583 | 1.000 |
| DAIC | Audio + Text + Emo (our, ZH SECap) | 0.809 | 0.571 | 0.857 | 0.429 |
| DAIC | Audio + Text + Emo (our, EN SECap) | 0.766 | 0.703 | 0.565 | 0.929 |
| DAIC | Audio + Text (paper) | 0.857 | 0.762 | 0.889 | 0.667 |
| DAIC | DepressInstruct (paper proposed) | 0.891 | 0.824 | 0.840 | 0.808 |
| CMDC | Audio + Text (our) | 0.987 | 0.981 | 0.963 | 1.000 |
| CMDC | Audio + Text + Emo (our, ZH SECap) | 0.937 | 0.887 | 1.000 | 0.807 |
| CMDC | Audio + Text + Emo (our, paper-prompt ZH SECap) | 0.948 | 0.919 | 0.967 | 0.887 |
| CMDC | Audio + Text (paper) | 0.989 | 0.982 | 1.000 | 0.967 |
| CMDC | DepressInstruct (paper proposed) | 0.989 | 0.982 | 1.000 | 0.967 |
| EATD | Audio + Text (our) | 0.420 | 0.288 | 0.186 | 0.633 |
| EATD | Audio + Text + Emo (our, ZH SECap) | 0.611 | 0.312 | 0.475 | 0.467 |
| EATD | Audio + Text + Emo (our, paper-prompt ZH SECap) | 0.728 | 0.383 | 0.365 | 0.433 |
| EATD | Audio + Text (paper) | 0.937 | 0.800 | 0.714 | 0.909 |
| EATD | Audio + Emotion (paper) | 0.937 | 0.737 | 0.875 | 0.636 |
| EATD | DepressInstruct (paper proposed) | 0.949 | 0.818 | 0.818 | 0.818 |

## Takeaways
- **Audio+text matches or slightly beats text-only on DAIC and EDAIC.** DAIC: audio+text reg1 (F1 0.696, ACC 0.851) edges text reg3 (F1 0.692, ACC 0.830) with much higher precision. EDAIC: audio+text reg2 (F1 0.667, ACC 0.768) ties text reg1's F1 with better ACC. So audio adds a little on top of the strong text channel — gain is real but marginal.
- **Light reg stays unstable.** reg2 collapsed to all-positive again on DAIC audio+text (ACC 0.298), while reg2 was the *best* EDAIC audio+text — the lightest configs are dataset-dependent and unreliable. reg3/reg4 are the safe picks for the audio paths.
- **CMDC's 98.7% is a red flag**, not a result — investigate for corpus artifacts before trusting it.
- **EATD failed (below the 0.815 majority baseline)** — the single default config over-predicts on a 4.4:1 imbalanced set. Port the reg3 recipe + class handling before drawing conclusions.
- CMDC and EATD only have the single default audio+text config (no reg sweep / modality ablation yet).
- **Emotion is not helping consistently in our implementation.** DAIC EN SECap reduces F1 from 0.737 to 0.703; DAIC ZH SECap improves ACC/precision but drops recall hard. The paper-prompt extraction improves over the earlier ZH extraction on CMDC (0.919 vs 0.887 F1) and EATD (0.383 vs 0.312), but still does not beat audio+text on CMDC or approach the paper on EATD.
