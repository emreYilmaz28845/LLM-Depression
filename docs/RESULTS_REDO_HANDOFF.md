# Results Redo — Handoff

Status as of 2026-06-29. Purpose: re-run the depression results on **one
standardized recipe** so every cell of `depression_results_table_no_emo.csv` is
comparable, then hand the finished numbers to a follow-up agent for discussion.

Companions: `TURKISH_DETERMINISM_STATUS.md` (why the Turkish numbers were noisy),
`configs/README.md` (config layout), the daily notes `2026-06-15.md` /
`2026-06-24.md` (the 12-cell selection-metric study and the audio/text-only runs).

---

## 1. The standardized recipe (decided)

Every `configs/main/*.yaml` uses the same recipe:

- **Frozen audio encoder** — the code default (`audio_adapter.enabled:false`,
  `train_projector:false`, guarded by `enforce_audio_encoder_freeze` in
  `src/model/qwen2audio_lora.py`). Only archived `*_nofreeze` configs train it.
- **macro-F1 selection** — `training.selection_metric: inner_val_macro_f1`
  (+ `early_stopping.metric`). Base-rate robust: penalizes majority-class
  collapse in *either* direction, so the same criterion works for
  minority-positive datasets (DAIC/CMDC) and majority-positive ones (Turkish T17).
  Do **not** vary the selection metric per dataset.
- **Teacher-forced eval** — `evaluation.sample_prediction_mode` and
  `headline_mode: original_teacher_forced`.

Consequence: **AUROC is not available** (teacher-forced emits a hard label, no
score to rank). Headline metrics are **macro-F1 + positive-F1 + the all-positive
baseline** (accuracy is a trap on imbalanced sets; see §6).

Why teacher-forced over likelihood: the `likelihood` backend *collapses* on DAIC
(repeated `F1=0.000, ACC=0.657` = predict-all-negative at the 70% base rate).
Teacher-forced is stable across datasets, so it's the common backend.

---

## 2. What needs redoing (10 configs)

| Dataset | Modality | Redo? | Reason |
| --- | --- | --- | --- |
| DAIC | audio+text | no | already recipe-aligned (table value 0.737 stands) |
| DAIC | text-only | no | already aligned (0.692 stands) |
| DAIC | audio-only | **yes** | was reg3 + likelihood |
| CMDC | all 3 | **yes** | were likelihood |
| Turkish t17 | all 3 | **yes** | pre-determinism + likelihood |
| Turkish t21 | all 3 | **yes** | deterministic but likelihood backend |

EDAIC audio-only/text-only also drifted from the recipe but are **not** in the
no-emo table; out of scope unless the follow-up wants them.

---

## 3. CLI commands (run on the BSC login node)

```bash
export PROJECT_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
cd $PROJECT_ROOT
bash scripts/sync_to_cluster.sh    # only if local code changed since last sync
```

### CMDC — 5-fold CV, 3 modalities fanned out, mean over folds (no full-train)
```bash
RUN_NAME_PREFIX=cmdc_tf bash scripts/run_cmdc_cv.sh
```

### DAIC — fixed train/val/test split, 3 modalities (only audio-only required)
```bash
# all three:
RUN_NAME_PREFIX=daic_tf bash scripts/run_daic_fixed.sh
# or just audio-only:
CONFIGS="$PROJECT_ROOT/configs/main/daic_audio_only_selmacrof1_tf.yaml" \
  RUN_NAME_PREFIX=daic_tf bash scripts/run_daic_fixed.sh
```

### Turkish — 5-fold CV, 3 modalities per submission, mean over folds
```bash
# BDI>=21
CONFIGS="$PROJECT_ROOT/configs/main/turkish_t21_audio_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/turkish_t21_text_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/turkish_t21_audio_text_selmacrof1_tf.yaml" \
  RUN_NAME_PREFIX=t21_tf sbatch --export=ALL scripts/run_turkish_5fold.sh
# BDI>=17
CONFIGS="$PROJECT_ROOT/configs/main/turkish_t17_audio_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/turkish_t17_text_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/turkish_t17_audio_text_selmacrof1_tf.yaml" \
  RUN_NAME_PREFIX=t17_tf sbatch --export=ALL scripts/run_turkish_5fold.sh
```

### Determinism replicas (recommended for the headline)
The Turkish saga showed single runs are noisy at ~50/50 balance with a weak
signal. Run each submission a **second** time with a `_rep2` prefix to get a
2-seed mean ± std for the table:
```bash
RUN_NAME_PREFIX=cmdc_tf_rep2 bash scripts/run_cmdc_cv.sh
RUN_NAME_PREFIX=daic_tf_rep2 bash scripts/run_daic_fixed.sh
CONFIGS="...t21 x3" RUN_NAME_PREFIX=t21_tf_rep2 sbatch --export=ALL scripts/run_turkish_5fold.sh
CONFIGS="...t17 x3" RUN_NAME_PREFIX=t17_tf_rep2 sbatch --export=ALL scripts/run_turkish_5fold.sh
```

### Sync results back
```bash
rsync -avhP ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/ \
  $PROJECT_ROOT/output_model/
rsync -avhP ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/ \
  $PROJECT_ROOT/logs/
```

---

## 4. Where the results land & how to read them

`run_name = <PREFIX>_<config_stem>`, e.g. `cmdc_tf_cmdc_audio_only_selmacrof1_tf`.

- **CMDC / Turkish (CV):**
  `output_model/<modality>/<dataset>/<run_name>/final_summary.json`
  → `active_backend_summary_row` → use the `*_mean` fields:
  `accuracy_mean`, `positive_f1_mean`, `precision_mean`, `recall_mean`,
  `macro_f1_mean` (and `*_std` for error bars).
  modality dir = `audio_only|text_only|audio_text`; dataset dir = `cmdc`,
  `turkish_t17`, `turkish_t21`.
- **DAIC (fixed):** no CV summary — read the standalone eval:
  `output_model/<modality>/daic/<run_name>/fold_0/best_model/standalone_eval/metrics_*.json`
  (or grep the eval log `logs/slurm_eval/daic/eval-<JOBID>-*.log` for the
  `Finished evaluation ... ACC=.. F1=.. Precision=.. Recall=..` line; macro_f1 is
  in the `Standalone evaluation complete: {...}` dict).

---

## 5. Filling `depression_results_table_no_emo.csv`

Columns: `Dataset, Method, ACC, F1, Precision, Recall, MacroF1, AUROC, AllPosBaseF1`.

- `F1` = positive-class F1 (`positive_f1[_mean]`). `MacroF1` = `macro_f1[_mean]`.
- `AUROC` = **n/a** for all teacher-forced rows (i.e. everything now).
- `AllPosBaseF1` is a per-dataset constant (base-rate dependent, already in file):
  DAIC **0.459**, CMDC **0.500**, Turkish t17 **0.818**, Turkish t21 **0.681**.
- For 2-seed replicas, enter the mean (and keep std handy for the discussion).

Currently in the file: DAIC rows are filled & source-verified; T21 rows hold the
*old likelihood* deterministic mean (to be overwritten by the teacher-forced
redo); T17 rows are `TODO`; CMDC rows are the old likelihood numbers (to be
overwritten). Paper rows have no macro-F1/AUROC (those papers don't report them).

---

## 6. Context for the follow-up discussion

- **Why CV didn't make Turkish trustworthy:** CV averages over *data splits*, not
  over *training nondeterminism*. The folds are seeded-identical across runs, so
  run-to-run drift was 100% training noise — worst at T21 (~50/50 balance, near-
  chance audio AUROC 0.57). Determinism flags (committed in `2170150`) shrank the
  CV-aggregate spread ~9× (Δ0.015 pos-F1 vs the old Δ0.131). Per-fold still
  wobbles; report mean ± std, not a single fold.
- **Imbalance vs the metric:** positive-F1 is only "punishing" when depressed is
  the minority (DAIC/CMDC, baseline ~0.46–0.50). When depressed is the majority
  (Turkish T17, baseline 0.818), a trivial all-positive predictor already scores
  0.82 — so a high pos-F1 there is near-meaningless. That's why macro-F1 + the
  all-positive baseline column matter; they make rows comparable across inverse
  imbalance.
- **DAIC instability mirror:** DAIC's likelihood backend collapses to all-negative
  (F1=0). Teacher-forced is why we standardized on it.
- **Open items for discussion:** (a) is 2-seed enough or go to 3+? (b) EDAIC in or
  out of the table? (c) DAIC audio-only uses a different `reg` lineage than
  audio+text — confirm the recipe is comparable within DAIC. (d) CMDC AUROC was
  ~1.0 under the old likelihood eval (tiny corpus / possible leakage flagged in
  `2026-06-15.md`) — sanity-check CMDC's near-perfect numbers.

---

## 7. Scripts reference

| Script | Dataset | What it does |
| --- | --- | --- |
| `scripts/run_cmdc_cv.sh` | CMDC | fan out 3 modalities, each a 5-fold CV chain → per-config summary (mean over folds) |
| `scripts/run_daic_fixed.sh` | DAIC | fan out 3 modalities, each a single fixed-split train + test eval |
| `scripts/run_turkish_5fold.sh` | Turkish | (existing) fan out 3 modalities, each a 5-fold CV chain + summary |

All three submit sbatch jobs and return; modality chains run concurrently, folds
within a chain run sequentially via `afterok`.
