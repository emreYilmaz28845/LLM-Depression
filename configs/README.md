# Configs

```
configs/
  quarantines.yaml   # subject quarantine list — referenced by every config via
                     # ${PROJECT_ROOT}/configs/quarantines.yaml; DO NOT move.
  main/              # canonical configs — one per dataset × modality. Run these.
  archive/           # all prior experiments (regN sweeps, K sweeps, seeds,
                     # ablations, nofreeze/selloss/valloss, emotion, qwen3omni,
                     # eatd). Kept for reproducibility; not part of the headline.
```

## The `main/` recipe

Every config in `main/` uses the same standardized recipe:

- **teacher-forced eval** — `evaluation.sample_prediction_mode: original_teacher_forced`
  and `headline_mode: original_teacher_forced`.
- **positive-F1 selection** — `training.selection_metric: inner_val_positive_f1`
  and `early_stopping.metric: inner_val_positive_f1`, with
  `selection_metric_mode: max`.
- **frozen audio encoder** — the default (`audio_adapter.enabled` and
  `train_projector` both default to `false` in `src/model/qwen2audio_lora.py`, and
  `enforce_audio_encoder_freeze` guards it). No config needs an explicit freeze block;
  the archived `*_nofreeze` configs are the only ones that train the encoder.

AUROC is **not** reported under this recipe — teacher-forced decoding emits a hard
label, not a continuous score, so there is no ranking to compute AUROC over.

## Coverage

`main/` holds: DAIC, EDAIC, CMDC (×3 modalities each) and Turkish at BDI≥17 and
BDI≥21 (×3 modalities each). EATD and Turkish BDI≥25 are not in scope (archived /
not created).

Naming: `<dataset>[_t<threshold>]_<modality>_selposf1_tf.yaml`.
