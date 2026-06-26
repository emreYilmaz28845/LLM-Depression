# Turkish Dataset Statistics

Generated on 2026-06-26 from the local dataset at
`/media/emre/Backup/AudioLLM/Datasets/Turkish`.

Primary source files:

- `metadata_turkish_t25_binary_merged.csv`
- `whisper_transcripts.jsonl`
- `all-files/*.wav`
- cross-checks from repo-generated `outputs/splits_t17`, `outputs/splits_t21`,
  `outputs/splits_t25`, and matching manifests

The binary label convention in this repo is:

```text
depressed = depresyon_skoru >= threshold
non-depressed = depresyon_skoru < threshold
```

## Headline Findings

- The dataset has 120 subjects and 1,051 labeled audio/transcript samples.
- BDI scores range from 3 to 49, with median 21 and mean 21.48.
- Threshold choice strongly changes class balance:
  - t17: 83 / 120 depressed subjects, 69.2%
  - t21: 62 / 120 depressed subjects, 51.7%
  - t25: 46 / 120 depressed subjects, 38.3%
- t21 is the most balanced and aligns with the BDI transition from borderline clinical to moderate depression.
- t17 includes borderline clinical depression as positive, making the task positive-heavy.
- t25 cuts inside the moderate band, making the task negative-heavy and moving 16 moderate-range subjects from positive to negative compared with t21.
- Audio and transcript coverage are complete for labeled metadata rows: 1,051 / 1,051 have matching wav and transcript entries.
- There are 1,186 wav/transcript files total, so 135 files are present locally but not used by the labeled merged metadata.
- File count per subject is not meaningfully correlated with depression score, so chunk/file count is not an obvious label leakage channel here.

## BDI Severity Bands

The score is treated as Beck Depression Inventory style 0-63 scoring:

![Subject-level BDI score distribution with t17, t21, and t25 thresholds](figures/turkish_bdi_score_distribution.png)

| BDI range | Interpretation |
|---:|---|
| 1-10 | Normal ups and downs |
| 11-16 | Mild mood disturbance |
| 17-20 | Borderline clinical depression |
| 21-30 | Moderate depression |
| 31-40 | Severe depression |
| >40 | Extreme depression |

Subject distribution:

| BDI band | Subjects | Percent |
|---|---:|---:|
| 1-10 normal | 14 | 11.7% |
| 11-16 mild mood disturbance | 23 | 19.2% |
| 17-20 borderline clinical | 21 | 17.5% |
| 21-30 moderate | 44 | 36.7% |
| 31-40 severe | 14 | 11.7% |
| >40 extreme | 4 | 3.3% |

Sample distribution:

| BDI band | Samples | Percent |
|---|---:|---:|
| 1-10 normal | 126 | 12.0% |
| 11-16 mild mood disturbance | 202 | 19.2% |
| 17-20 borderline clinical | 182 | 17.3% |
| 21-30 moderate | 400 | 38.1% |
| 31-40 severe | 105 | 10.0% |
| >40 extreme | 36 | 3.4% |

## Overall Subject Statistics

| Variable | N | Mean | SD | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depression score | 120 | 21.48 | 8.88 | 3 | 15.75 | 21 | 27 | 49 |
| anxiety score | 120 | 24.18 | 12.74 | 2 | 14 | 22 | 34 | 54 |
| files per subject | 120 | 8.76 | 2.63 | 5 | 7 | 9 | 10 | 19 |

Score histogram, subject-level:

| Score bin | Subjects |
|---:|---:|
| 0-4 | 2 |
| 5-9 | 7 |
| 10-14 | 15 |
| 15-19 | 27 |
| 20-24 | 23 |
| 25-29 | 27 |
| 30-34 | 10 |
| 35-39 | 4 |
| 40-44 | 4 |
| 45-49 | 1 |

The densest part of the distribution is 15-29. This is exactly where t17, t21,
and t25 operate, so threshold selection has a large effect on class definitions.

## Threshold Balance

![Turkish t17 vs t21 class balance and threshold transition](figures/turkish_t17_vs_t21_balance.png)

Subject-level balance:

| Threshold | Positive definition | Depressed | Non-depressed | Depressed % | Ratio dep:non |
|---:|---|---:|---:|---:|---:|
| 17 | BDI >= 17 | 83 | 37 | 69.2% | 2.24 |
| 21 | BDI >= 21 | 62 | 58 | 51.7% | 1.07 |
| 25 | BDI >= 25 | 46 | 74 | 38.3% | 0.62 |

Sample-level balance:

| Threshold | Depressed samples | Non-depressed samples | Depressed % |
|---:|---:|---:|---:|
| 17 | 723 | 328 | 68.8% |
| 21 | 541 | 510 | 51.5% |
| 25 | 413 | 638 | 39.3% |

Threshold sensitivity:

| Comparison | Subjects that change label | Samples that change label | Interpretation |
|---|---:|---:|---|
| t17 -> t21 | 21 | 182 | Borderline clinical subjects, BDI 17-20, move from positive to negative |
| t21 -> t25 | 16 | 128 | Moderate subjects, BDI 21-24, move from positive to negative |
| t17 -> t25 | 37 | 310 | All BDI 17-24 subjects change label |

Subjects near each threshold:

| Threshold | Subjects within +/-2 points | Subjects exactly at threshold |
|---:|---:|---:|
| 17 | 27 | 6 |
| 21 | 26 | 5 |
| 25 | 24 | 5 |

This confirms that all three thresholds cut through a dense part of the score
distribution. t21 is still preferable if a binary label is required because it is
both clinically legible and balanced.

## Fold Balances

Held-out fold balance, subject-level:

| Threshold | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---:|---|---|---|---|---|
| 17 | 17 dep / 8 non | 17 dep / 8 non | 17 dep / 7 non | 16 dep / 7 non | 16 dep / 7 non |
| 21 | 13 dep / 12 non | 13 dep / 12 non | 12 dep / 12 non | 12 dep / 11 non | 12 dep / 11 non |
| 25 | 10 dep / 15 non | 9 dep / 15 non | 9 dep / 15 non | 9 dep / 15 non | 9 dep / 14 non |

The folds are stratified as expected. t21 gives the cleanest fold-level balance.
t17 is strongly positive-heavy in every held-out fold, while t25 is
negative-heavy in every held-out fold.

## Audio and Transcript Coverage

Audio:

| Metric | Value |
|---|---:|
| wav files under `all-files` | 1,186 |
| labeled metadata rows | 1,051 |
| metadata rows with matching wav | 1,051 |
| missing labeled wav files | 0 |
| total labeled audio duration | 5.54 hours |
| mean duration | 18.99 sec |
| median duration | 20.00 sec |
| Q1 / Q3 duration | 20.00 / 20.00 sec |
| min / max duration | 0.31 / 20.00 sec |
| files > 20 sec | 0 |
| files > 30 sec | 0 |

Transcripts:

| Metric | Value |
|---|---:|
| transcript entries | 1,186 |
| labeled metadata rows with transcript | 1,051 |
| missing labeled transcripts | 0 |
| empty labeled transcripts | 0 |
| mean transcript length | 266 chars / 39 words |
| median transcript length | 273 chars / 40 words |
| min / max transcript length | 5 / 449 chars |

The extra 135 local wav/transcript files are not part of the labeled merged
metadata used for training/evaluation.

## Correlations and Leakage Checks

Subject-level correlations with depression score:

| Variable | Pearson r | Finding |
|---|---:|---|
| anxiety score | +0.510 | Strong comorbidity/overlap signal |
| age | -0.175 | Weak negative association |
| files per subject | -0.065 | No meaningful file-count leakage |
| mean `w2v2_predicted_score` | +0.045 | Audio-derived score barely tracks BDI |

File-level correlation:

| Variable | Pearson r | N |
|---|---:|---:|
| `w2v2_predicted_score` vs BDI | +0.045 | 1,051 |

Group means by threshold:

| Threshold | Group | Depression score | Anxiety score | Age | Files/subject | Mean w2v2 score |
|---:|---|---:|---:|---:|---:|---:|
| 17 | depressed | 25.82 | 27.69 | 41.04 | 8.71 | 21.64 |
| 17 | non-depressed | 11.73 | 16.30 | 47.35 | 8.86 | 21.63 |
| 21 | depressed | 28.27 | 30.60 | 41.35 | 8.73 | 21.65 |
| 21 | non-depressed | 14.21 | 17.31 | 44.72 | 8.79 | 21.62 |
| 25 | depressed | 30.43 | 32.41 | 40.89 | 8.98 | 21.66 |
| 25 | non-depressed | 15.91 | 19.05 | 44.28 | 8.62 | 21.63 |

Findings:

- Anxiety rises strongly with depression score. This is expected clinically, but
  it means the depression label is not independent of anxiety burden.
- Files per subject are almost identical across groups. This reduces concern that
  the model can solve the Turkish task by counting chunks.
- The existing `w2v2_predicted_score` is almost constant with respect to BDI.
  It does not look like a strong acoustic proxy for depression severity in this
  dataset.

## Relation to Current Model Results

The current no-emotion result table uses positive-class F1. Under that metric:

| Threshold | Modality | ACC | Positive F1 | Precision | Recall |
|---:|---|---:|---:|---:|---:|
| 17 | Audio + Text | 0.710 | 0.799 | 0.751 | 0.871 |
| 17 | Audio only | 0.717 | 0.814 | 0.744 | 0.913 |
| 17 | Text only | 0.711 | 0.772 | 0.802 | 0.788 |
| 21 | Audio + Text | 0.700 | 0.710 | 0.704 | 0.741 |
| 21 | Audio only | 0.642 | 0.688 | 0.625 | 0.778 |
| 21 | Text only | 0.707 | 0.694 | 0.749 | 0.694 |

Interpretation:

- At t17, the task is positive-heavy. A recall-heavy model can score high
  positive-F1 even when specificity is weak. This explains why audio-only can
  edge audio+text under positive-F1.
- At t21, the balance is much cleaner. Audio+text has the best positive-F1, while
  text-only has slightly higher accuracy.
- Positive-F1 alone can overstate models that predict many positives, especially
  at t17. Macro-F1 and confusion matrices should be reported alongside it.
- Because the audio-derived score has almost no correlation with BDI, any strong
  audio-only result should be interpreted carefully and checked against confusion
  matrices, fold variance, and possible shortcut behavior.

## Practical Recommendations

1. Use t21 as the primary Turkish binary label: BDI >= 21 means moderate-or-worse
   depression and gives the best class balance.
2. Keep t17 and t25 as sensitivity analyses:
   - t17 asks whether borderline clinical depression is grouped with depression.
   - t25 asks whether only higher-moderate and severe cases are treated as positive.
3. Report all Turkish results with at least:
   - accuracy
   - positive-F1
   - macro-F1
   - precision/recall
   - pooled confusion matrix
4. Avoid choosing the BDI threshold based on model F1. The threshold should be a
   clinical/statistical design choice, not a model-selection knob.
5. Treat the Turkish audio-only gains cautiously until replicated. The dataset
   statistics do not show a strong continuous audio-score signal for BDI.
