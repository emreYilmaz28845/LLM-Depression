# Androids Interview Hidden-State Classifier Report

Run ID: `androids_hidden_20260731T152518Z_c8f3832`  
Source commit: `c8f3832`  
Manifest canonical SHA-256: `01a351f7277e4763a8bb9e4983bba190b265becafafca6d7ee04bdcfc948cbed`  
Official split SHA-256: `f75dd2ba7bb324af26de8c5ae3497d2108e6b50815c0ef6cbcade7de70992518`

## Acceptance

The production acceptance audit passed with 45 fold/head results and nine pooled results. Each pooled result contains 116 unique held-out subjects: 52 controls and 64 patients. The audit recomputed predictions and metrics from the saved compact prediction artifacts.

## Protocol

- Modalities: Audio only, Audio + Text, and Text only; full-turn inputs are excluded.
- Hidden representation: final-layer last-valid-prompt-token vector from the existing five-fold best-model checkpoints.
- Audio aggregation: arithmetic probability mean from window to turn to subject.
- Audio fit weights: `1 / (turns_for_subject * windows_for_turn)`, rescaled to mean one; subject and within-subject turn totals are audited.
- Text fit: one vector per subject with unit weight.
- Decision rule: fixed threshold 0.5; exact ties are invalid and counted wrong in strict headline metrics.
- Fixed heads: the repository `logreg_raw` and `xgb_raw` defaults. Optuna: `standard_d6`, three subject-stratified inner folds, pooled inner OOF Macro-F1, 150 trials, seed 1337.
- No PCA, oversampling, controls, or outer-fold result selection was used.

## Pooled results

| Modality | Head | Accuracy | Positive F1 | Negative F1 | Macro-F1 | AUROC | Confusion |
|---|---|---:|---:|---:|---:|---:|---|
| Audio only | Logistic Regression | 0.844828 | 0.854839 | 0.833333 | 0.844086 | 0.952825 | `[[45,7],[11,53]]` |
| Audio only | XGBoost fixed raw | 0.844828 | 0.854839 | 0.833333 | 0.844086 | 0.942308 | `[[45,7],[11,53]]` |
| Audio only | XGBoost Optuna (150 trials, standard_d6) | 0.836207 | 0.845528 | 0.825688 | 0.835608 | 0.935998 | `[[45,7],[12,52]]` |
| Audio + Text | Logistic Regression | 0.836207 | 0.845528 | 0.825688 | 0.835608 | 0.936298 | `[[45,7],[12,52]]` |
| Audio + Text | XGBoost fixed raw | 0.887931 | 0.894309 | 0.880734 | 0.887521 | 0.953726 | `[[48,4],[9,55]]` |
| Audio + Text | XGBoost Optuna (150 trials, standard_d6) | 0.853448 | 0.866142 | 0.838095 | 0.852118 | 0.947716 | `[[44,8],[9,55]]` |
| Text only | Logistic Regression | 0.844828 | 0.857143 | 0.830189 | 0.843666 | 0.918870 | `[[44,8],[10,54]]` |
| Text only | XGBoost fixed raw | 0.836207 | 0.850394 | 0.819048 | 0.834721 | 0.871244 | `[[43,9],[10,54]]` |
| Text only | XGBoost Optuna (150 trials, standard_d6) | 0.827586 | 0.846154 | 0.803922 | 0.825038 | 0.890024 | `[[41,11],[9,55]]` |

## Fold and job accounting

The audit recorded `45` fold/head results across five outer folds and `9` pooled results.
The synchronized job registry contains `46` rows.
Scheduler accounting captured `45` top-level jobs with states `{"COMPLETED": 45}`; summed recorded elapsed time is `23023.0` seconds and the longest recorded job is `2917.0` seconds.
GPFS accounting at audit time: `604835414016` bytes available on `/gpfs/projects` (74% used).
Each Optuna result was required to contain exactly 150 COMPLETE trials with zero failed trials; inner validation assignments were subject-disjoint and covered each outer-training subject exactly once.

## Retrieval and limitations

The remote acceptance audit was performed against the full hidden vectors and model artifacts on MN5. The local handoff is compact: prediction rows, metrics, configurations, trial summaries, best parameters, audits, extraction metadata, row inventories, registry, and logs. Adapter/checkpoint files, hidden-vector NPZ caches, model binaries, and Optuna SQLite databases remain excluded from the default retrieval.

The hidden heads are a controlled representation-level comparison, not a new end-to-end fine-tuning run. Metrics are outer-fold pooled subject results, and the Audio + Text label is intentionally generic in the workbook.
