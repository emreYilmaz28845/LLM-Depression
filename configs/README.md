# Configs

```text
configs/
  quarantines.yaml   # subject quarantine list; every config references it
  main/              # active configs
  experiments/       # active non-headline research
  archive/           # superseded recipes retained for reproducibility
```

## Harmonized main recipe

The active harmonized family is:

`harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1`

It covers D3TEC, Turkish BDI≥17 with Qwen3-ASR, Androids, DAIC-WOZ, and CMDC in audio-only, text-only, and audio+text modes.

- One participant-audio window per prompt; never a joint-audio bundle.
- Windows are at most 30 seconds and do not overlap.
- Audio+text repeats the full participant transcript on every window.
- Every training window appears once per epoch; DataLoader shuffling changes only its order.
- `training.class_balance: none`.
- D3TEC, Turkish, Androids, and CMDC use subject → source unit → window loss weighting and response-subject evaluation.
- DAIC uses participant-only speech packed from raw timestamp intervals into consecutive 30-second chunks, subject-normalized loss weighting, and all-chunk subject aggregation.
- Validation checkpoint selection and early stopping use `inner_val_macro_f1`, mode `max`.
- Evaluation uses `original_teacher_forced` and reports strict subject-level metrics.
- The audio encoder remains frozen because `audio_adapter.enabled` and `train_projector` are false.

Naming:

```text
<dataset>[_t<threshold>]_<modality>_harmonized_selmacrof1_tf[_variant].yaml
```

The nine superseded DAIC, CMDC, and Turkish positive-F1 main configs were moved to:

```text
configs/archive/pre_harmonized_posf1_20260809/
```

## E-DAIC exception

E-DAIC was outside the harmonization scope and was not inspected, moved, or rewritten. Its three existing positive-F1 configs remain in `main/` unchanged. They are not members of the harmonized family.

## Current coverage

`main/` contains 18 configs:

- 15 harmonized configs: five datasets × three modalities.
- 3 unchanged E-DAIC configs outside the harmonized family.

See `docs/harmonized_dataset_baseline.md` for the methodology and dataset-specific adapters.

## Harmonized reproduction matrix

The standalone execution matrix is `configs/experiments/harmonized/standalone_matrix.yaml`. It expands to 63 four-GPU training jobs: one DAIC fold and five folds for each other dataset, across three modalities. D3TEC, Androids, and DAIC also receive separate deterministic evaluation jobs. Hidden-state postprocessing runs fixed Logistic Regression and fixed XGBoost; it does not run Optuna.

The matching merged configs are:

- `configs/experiments/merged/symmetric_merged_harmonized_audio_text.yaml`
- `configs/experiments/merged/symmetric_merged_harmonized_audio_only.yaml`
- `configs/experiments/merged/symmetric_merged_harmonized_text_only.yaml`

They use only the 15 harmonized component configs. Each component and merged fit has a maximum of 20 epochs, validation macro-F1 checkpoint selection, patience 3, and no XGBoost Optuna. Merged cross-validation selects by mean dataset macro-F1; the final training epoch is the rounded median selected cross-validation epoch.

MN5 execution order:

1. `scripts/submit_harmonized_preflight.sh` rebuilds all manifests on GPFS and validates paths, files, hashes, splits, and merged protocols without using a GPU.
2. `scripts/submit_harmonized_standalone.sh` submits the standalone reproduction matrix only after that preflight passes.
3. `scripts/submit_harmonized_merged.sh` submits the merged smoke, cross-validation, and final stages separately.

All launchers default to dry-run. Their default throttles reserve seven four-GPU training lanes (28 H100s) and at most four one-GPU auxiliary jobs, for a hard ceiling of 32 allocated H100s.
