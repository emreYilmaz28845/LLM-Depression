# Symmetric merged execution/results

- Run ID: `symmetric_merged_smoke_6fba6e632653`
- Git commit: `9888e814872c37d14535cf3555a814b62ba56b59`
- CV fold rows: 300
- CV pooled rows: 60
- DAIC official-test rows: 12
- CV pooled CSV: `reports/symmetric_merged/symmetric_merged_smoke_6fba6e632653/symmetric_merged_cv_pooled.csv`
- Job registry: `/home/emre/Projects/AudioLLM/LLM-Depression/outputs/symmetric_merged_jobs/symmetric_merged_smoke_6fba6e632653.json`
- Job IDs: 44066877, 44118085, 44118086, 44127637, 44127638, 44127639, 44127641, 44127642, 44127643, 44127644, 44127645, 44127646, 44127647, 44127648, 44127649, 44127650, 44127651, 44127652, 44130957, 44130958, 44130959, 44131790, 44131791, 44131792, 44132714, 44132715, 44132716, 44133190, 44133191, 44133192, 44133394, 44133395, 44133396, 44133419, 44133421, 44133422, 44134500, 44134501, 44134502, 44134552, 44134553, 44134554, 44134751, 44134752, 44134753, 44136283, 44136285, 44136286, 44148582, 44148583, 44148585, 44148586, 44148587, 44148588, 44148589, 44148590, 44154610
- Registry status: `terminal_success`

- Slurm accounting rows: 57; accounted runtime/resource rows: 57
- Runtime/storage metadata: `reports/symmetric_merged/symmetric_merged_smoke_6fba6e632653/symmetric_merged_execution_metadata.json`

## Acceptance audits

- `audio_text`: smoke=passed, cv=passed, final=passed
- `audio_only`: cv=passed, final=passed
- `text_only`: cv=passed, final=passed

## Protocol

Five datasets, three modalities, Qwen + standardized Logistic Regression + fixed XGBoost + 150-trial grouped Optuna XGBoost.

## Limitations

- Best-model checkpoints, hidden feature arrays, classifier joblibs, and Optuna SQLite databases remain on MN5 GPFS.
- Reported CV headline metrics are pooled from non-overlapping subject-level outer-fold predictions by dataset.
