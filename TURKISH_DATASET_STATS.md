# Turkish Depression Dataset — Statistical Report

Subject-level analysis, N = 120 (one row per `patient_id`). Generated from
`outputs/splits/turkish_subjects.json` + `outputs/manifests/turkish_manifest.jsonl`.
Pure-numpy stats (no scipy): Mann–Whitney via tie-corrected normal approximation.

---

## 0. Headline takeaways

1. **Do not tune the label threshold to maximize model F1** — that is circular (redefining
   the ground truth to flatter the classifier). The label threshold was instead set on
   **psychometric grounds** (see §2a): `depresyon_skoru` is BDI, and BDI has published severity
   bands (1–10 normal, 11–16 mild, 17–20 borderline clinical, 21–30 moderate, 31–40 severe,
   >40 extreme). **Primary cut = 21 (depressed iff score ≥ 21, i.e. moderate-or-worse)** —
   this is the original clinical cutoff (25) revised once the scale was identified as BDI,
   not re-tuned to any model's F1. The model's *decision* threshold may still be tuned
   (Youden-J / max-F1) but **on inner-val only**, never on the test holdout.
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
| 17 | 83 | 69.2% | 37 | 37 |
| 18 | 77 | 64.2% | 43 | 31 |
| 20 | 69 | 57.5% | 51 | 23 |
| **21** | **62** | **51.7%** | **58** | **16** |
| 22 | 57 | 47.5% | 63 | 11 |
| 24 | 47 | 39.2% | 73 | 1 |
| 25 (orig.) | 46 | 38.3% | 74 | 0 (reference) |
| 26 | 41 | 34.2% | 79 | 5 |
| 28 | 27 | 22.5% | 93 | 19 |
| 30 | 19 | 15.8% | 101 | 27 |
| 32 | 14 | 11.7% | 106 | 32 |

- ±3 around the original cut (22 ↔ 28) flips **30 subjects = 25% of the cohort** — the task is
  inherently threshold-sensitive regardless of which cut is used.
- 21 is the best-balanced cut available (51.7% positive) and the only one that aligns with a
  BDI band edge (borderline|moderate) rather than bisecting a band. 17 (69.2% positive, thin
  minority) is kept as a sensitivity/robustness point, not a primary candidate.
- Report F1 at {17, 21, 25} as a robustness band when comparing across thresholds.

### 2a. Why 21, not 25 or 17

`depresyon_skoru` is the **Beck Depression Inventory (BDI)** total score. Its published severity
bands are: 1–10 normal, 11–16 mild, 17–20 borderline clinical, 21–30 moderate, 31–40 severe,
>40 extreme.

- **t=25** (original) sits inside the "moderate" band and, per §1, cuts through the score
  distribution's mode — worst case for label stability.
- **t=17** sits at the "mild → borderline" edge; pulls in the borderline band (ambiguous by
  definition) as positive → thinnest, least defensible minority class (37 subjects).
- **t=21** sits exactly at the "borderline → moderate" edge: borderline-clinical subjects
  (17–20, n=21) are excluded from "depressed", moderate-or-worse are included. This is the
  most clinically legible cut and happens to also be the best-balanced (62/58).

Threshold configs coexist with zero code conflicts via isolated `output_dirs` per threshold
(`outputs/manifests[_t17|_t21]`, matching `splits_*`, `run_root` suffixed `_t17`/`_t21`); default
configs (no suffix) now use t=20 as an interim value, t=21 configs (`*_t21.yaml`) are the
intended primary going forward.

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

1. **Use the BDI-band-aligned label threshold (21 = moderate-or-worse).** Chosen on
   psychometric grounds (§2a), not selected to maximize model F1. Report F1 at {17, 21, 25}
   as a robustness band — all three threshold configs already exist (`*_t21.yaml`,
   default = t20, `*_t17.yaml`).
2. **Tune only the model's decision threshold**, on inner-val, via Youden-J or max-F1, and
   freeze it before scoring the test holdout (respects the eval-determinism rule).
3. **Report AUROC + macro-F1 as the headline**, not positive-F1 alone — positive-F1 is the
   harshest lens on a hard, threshold-unstable, imbalanced task. AUROC is already implemented.
4. **Treat audio as a likely liability here** (freeze the Whisper encoder; lead with text-only)
   given the ~0.57 acoustic ceiling.
5. Consider whether the *clinically meaningful* task is depression-vs-not at all, given the
   anxiety entanglement (d = 1.22) — a depression-vs-anxiety or severity-regression framing may
   be better supported by the signal that actually exists in the data.
6. **All results reported in `depression_results_table*.csv` to date were trained under the
   original t=25 cutoff and are stale** relative to this threshold decision — re-run the 5-fold
   sweep against `*_t21.yaml` configs before treating any Turkish row as current.
