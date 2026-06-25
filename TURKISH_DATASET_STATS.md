# Turkish Depression Dataset — Statistical Report

Subject-level analysis, N = 120 (one row per `patient_id`). Generated from
`outputs/splits/turkish_subjects.json` + `outputs/manifests/turkish_manifest.jsonl`.
Pure-numpy stats (no scipy): Mann–Whitney via tie-corrected normal approximation.

---

## 0. Headline takeaways

1. **Do not tune the label threshold to maximize model F1** — that is circular (redefining
   the ground truth to flatter the classifier). Keep the clinical cutoff (25); report
   threshold *sensitivity* only. The model's *decision* threshold may be tuned (Youden-J /
   max-F1) but **on inner-val only**, never on the test holdout.
2. The label is defined by a cut (25) that sits **past the mode** of a near-normal score
   distribution (62nd percentile). A ±3 shift of the cut flips **25% of the cohort's labels** —
   the binary task is intrinsically unstable.
3. **The audio signal barely separates the classes.** A purpose-built acoustic regressor
   (`w2v2_predicted_score`) achieves only **AUROC ≈ 0.57** and r = +0.04 with the score. This
   caps every audio-based model regardless of threshold and explains audio-only F1 = 0.24.
4. The depression label is **strongly entangled with anxiety** (Cohen's d = 1.22, r = 0.51).

---

## 1. Depression score (`depresyon_skoru`) distribution

- n = 120, mean = 21.48, sd = 8.88, min = 3, Q1 = 16, median = 21, Q3 = 27, max = 49
- skewness = +0.35, excess kurtosis = +0.14 → roughly normal, unimodal
- **threshold 25 = 62nd percentile** (74 of 120 below)

```
[ 0, 5):  2
[ 5,10):  7
[10,15): 15
[15,20): 27   <- mode
[20,25): 23   } 50 subjects (42%) straddle the cut
[25,30): 27   }
[30,35): 10
[35,40):  4
[40,45):  4
[45,50):  1
```

The cut bisects the densest region → maximal label ambiguity.

## 2. Class balance & threshold sensitivity

| threshold | depressed N | depressed % | non-dep N | label flips vs t=25 |
|---|---|---|---|---|
| 18 | 77 | 64.2% | 43 | 31 |
| 20 | 69 | 57.5% | 51 | 23 |
| 22 | 57 | 47.5% | 63 | 11 |
| 24 | 47 | 39.2% | 73 | 1 |
| **25** | **46** | **38.3%** | **74** | **0 (reference)** |
| 26 | 41 | 34.2% | 79 | 5 |
| 28 | 27 | 22.5% | 93 | 19 |
| 30 | 19 | 15.8% | 101 | 27 |
| 32 | 14 | 11.7% | 106 | 32 |

- ±3 around the cut (22 ↔ 28) flips **30 subjects = 25% of the cohort**.
- Any single-threshold F1 is fragile; report F1 at {22, 25, 28} as a robustness band.

## 3. Group differences — depressed vs non-depressed (subject level)

| variable | depressed mean | non-dep mean | Cohen's d | point-biserial r | Mann–Whitney p |
|---|---|---|---|---|---|
| anxiety_score | 32.41 | 19.05 | **+1.22** | **+0.51** | <0.001 |
| age | 40.89 | 44.28 | −0.26 | −0.13 | 0.275 |
| education | 1.98 | 1.85 | +0.11 | +0.05 | 0.558 |
| num_files | 8.98 | 8.62 | +0.14 | +0.07 | 0.569 |

Only anxiety separates the groups (large effect). Crucially, **num_files does not encode the
label** (d = 0.14) — so, unlike DAIC, chunk count is not a leakage channel.

## 4. Correlations with the continuous score

- anxiety_score vs depresyon_skoru: Pearson r = **+0.51**
- age vs depresyon_skoru: Pearson r = −0.18

## 5. Acoustic ceiling — `w2v2_predicted_score` vs the label

`w2v2_predicted_score` is a regression model built to predict the depression score from audio.
It is the cleanest estimate of how much depression signal the *audio* carries.

- file-level: corr(w2v2_pred, depresyon_skoru) = **r = +0.04** (n = 1051) — essentially none
- file-level AUROC(w2v2_pred → binary label) = **0.572**
- subject-level AUROC (mean w2v2 per subject) = **0.570**
- Youden-J optimal cut: J = 0.166 (sens 0.65, spec 0.51) — weak
- Max-F1 optimal cut: degenerates to predict-all-positive (sens 1.0, spec 0.0, F1 0.554)

**Interpretation:** even a dedicated acoustic predictor is barely above chance. The audio→
depression mapping in this dataset is weak. This is a *data* ceiling, not a model failure, and
it bounds any audio-based F1 — consistent with audio-only LLM F1 = 0.24 and audio degrading
the multimodal run. (Caveat: r ≈ 0 with the score yet AUROC 0.57 suggests `w2v2_predicted_score`
itself may be a poor/odd predictor — worth verifying its provenance before quoting it as a hard
ceiling.)

## 6. Comorbidity

- comorbid subjects: 42 / 120 (35%)
- depressed rate: comorbid=0 → 0.36, comorbid=1 → 0.43
- φ(comorbid, depressed-label) = +0.07 → comorbidity barely predicts the depression label;
  the merged cohort is safe to train on as a single population.

---

## 7. Recommendations

1. **Keep the label threshold at the clinical value (25).** Do not select it on model F1.
   Add a sensitivity row (F1 at 22 / 25 / 28) — `threshold` is already a config knob.
2. **Tune only the model's decision threshold**, on inner-val, via Youden-J or max-F1, and
   freeze it before scoring the test holdout (respects the eval-determinism rule).
3. **Report AUROC + macro-F1 as the headline**, not positive-F1 alone — positive-F1 is the
   harshest lens on a hard, threshold-unstable, imbalanced task. AUROC is already implemented.
4. **Treat audio as a likely liability here** (freeze the Whisper encoder; lead with text-only)
   given the ~0.57 acoustic ceiling.
5. Consider whether the *clinically meaningful* task is depression-vs-not at all, given the
   anxiety entanglement (d = 1.22) — a depression-vs-anxiety or severity-regression framing may
   be better supported by the signal that actually exists in the data.
