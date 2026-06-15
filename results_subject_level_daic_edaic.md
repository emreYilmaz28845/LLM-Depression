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
| audio+text *(default cfg)* | 0.987 | 0.981 | 0.963 | 1.000 | 51,1 / 0,26 |

⚠️ Near-perfect on a 78-subject corpus — **treat as suspect** (likely a corpus-wide artifact / trivially separable, not a clean win). Needs leakage scrutiny before reporting.

## EATD — 3-fold CV pooled, N=162 (132 non-dep / 30 dep)
All-negative baseline: ACC 0.815, F1 0.000.

| run | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- |
| audio+text *(default cfg)* | 0.420 | 0.288 | 0.186 | 0.633 | 49,83 / 11,19 |

⚠️ **Below baseline** — collapsed to over-predicting depression (83 FP). Light reg (rank 16, lr 2e-4) on a heavily imbalanced set; needs the reg3 recipe.

## Best result per dataset

| dataset | best config | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --- | --- | --- | --- | --- | --- | --- |
| DAIC | **audio+text reg1** | 0.851 | 0.696 | 0.889 | 0.571 | 32,1 / 6,8 |
| EDAIC | **audio+text reg2** | 0.768 | 0.667 | 0.591 | 0.765 | 30,9 / 4,13 |
| CMDC | audio+text *(default)* ⚠️ | 0.987 | 0.981 | 0.963 | 1.000 | 51,1 / 0,26 |
| EATD | audio+text *(default)* ⚠️ below baseline | 0.420 | 0.288 | 0.186 | 0.633 | 49,83 / 11,19 |

## Takeaways
- **Audio+text matches or slightly beats text-only on DAIC and EDAIC.** DAIC: audio+text reg1 (F1 0.696, ACC 0.851) edges text reg3 (F1 0.692, ACC 0.830) with much higher precision. EDAIC: audio+text reg2 (F1 0.667, ACC 0.768) ties text reg1's F1 with better ACC. So audio adds a little on top of the strong text channel — gain is real but marginal.
- **Light reg stays unstable.** reg2 collapsed to all-positive again on DAIC audio+text (ACC 0.298), while reg2 was the *best* EDAIC audio+text — the lightest configs are dataset-dependent and unreliable. reg3/reg4 are the safe picks for the audio paths.
- **CMDC's 98.7% is a red flag**, not a result — investigate for corpus artifacts before trusting it.
- **EATD failed (below the 0.815 majority baseline)** — the single default config over-predicts on a 4.4:1 imbalanced set. Port the reg3 recipe + class handling before drawing conclusions.
- CMDC and EATD only have the single default audio+text config (no reg sweep / modality ablation yet).
