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

## Gemma 4 DAIC family

The Gemma 4 backbone comparison is scoped to DAIC only and uses three configs:

```text
daic_<modality>_harmonized_selmacrof1_tf_gemma4_12b.yaml   (text_only | audio_only | audio_text)
```

- Backend: `model_backend: gemma4` on `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (see
  `docs/GEMMA4_DAIC_IMPLEMENTATION_RUNBOOK.md`).
- They preserve every scientific invariant of their Qwen counterparts
  (dataset, seed, sample mode, packed30 chunking, subject-normalized weights,
  inner-validation macro-F1 selection, teacher-forced evaluation) and change
  only: backbone, model path/revision, isolated output roots
  (`output_model/harmonized_v1_gemma4/`), the LoRA target regex (six modules
  per layer across all 48 decoder layers, exactly 288), and
  `evaluation.evaluation_view: harmonized_all_windows_full_coverage`.
- Manifests and splits are shared with the Qwen DAIC harmonized campaign;
  `build_manifest.py` is backend-agnostic.
- The dedicated MN5 environment (`gemma4_12b_tf5_14_1`) is offline-only:
  installed from a local wheelhouse, model loaded from the GPFS snapshot with
  `local_files_only=True`.

## E-DAIC exception

E-DAIC was outside the harmonization scope and was not inspected, moved, or rewritten. Its three existing positive-F1 configs remain in `main/` unchanged. They are not members of the harmonized family.

## Current coverage

`main/` contains 21 configs:

- 15 harmonized configs: five datasets × three modalities.
- 3 Gemma 4 DAIC configs (see the Gemma 4 DAIC family section above).
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

Both GPU launchers require `GITHUB_ISSUE` and `GITHUB_PR`. For the full harmonized reproduction campaign, use Issue #12 and primary methodology PR #10. The production Git SHA must contain both PR #10 and its PR #11 acceptance-auditor correction. These fields provide scientific context; the full Git SHA and deployed-source hash remain the canonical source identity.

All launchers default to dry-run. Their default throttles reserve seven four-GPU training lanes (28 H100s) and at most four one-GPU auxiliary jobs, for a hard ceiling of 32 allocated H100s.

## Harmonized English-translation family

Issue #20 tracks the English-transcript comparison. The eight canonical English configs in `main/` are named `<dataset>_<modality>_harmonized_selmacrof1_tf[_qwen3asr]_en.yaml` and are derived only from the native harmonized counterparts, never from `configs/experiments/translation_en/` (historical recipe, do not reuse).

- Recipe ID: `harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1`.
- Each config adds a `transcripts:` block: `variant: english`, `cache_path: ${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}/harmonized_en_complete_v1/<dataset>/accepted.jsonl`, `minimum_status: automatic_low`, `require_complete: true`, `include_failed: false`.
- Outputs are English-specific: `outputs/manifests_harmonized_en/`, `outputs/splits_harmonized_en/`, `output_model/harmonized_v1_en/`.
- Only audio+text and text-only exist for D3TEC, Androids, CMDC, and Turkish t17. No English audio-only, DAIC, or E-DAIC configs.
- The fixed English matrix is `configs/experiments/harmonized/english_translation_matrix.yaml`: 8 experiments, 40 training folds, 20 separate evaluation folds (D3TEC, Androids), 40 hidden-extraction/fixed-head folds, exactly 100 jobs, no Optuna, no merged training, no audio-only cells.

MN5 execution order:

1. `scripts/submit_harmonized_en_preflight.sh` rebuilds the four English manifests on GPFS from the repaired `harmonized_en_complete_v1` translation cache, audits translation completeness, native/English input equivalence, and tokenizer/context fit, and records the expected 100-job scope. Requires `GITHUB_ISSUE=20` and the implementation `GITHUB_PR`.
2. `scripts/submit_harmonized_en_standalone.sh` submits the English matrix only after that preflight audit passes with `status: passed` and zero failures. Use the same `GITHUB_ISSUE=20` and `GITHUB_PR`.
3. `scripts/submit_harmonized_standalone_retry.sh` retries failed cells with new attempt identities; it accepts the English roots and prefixes through `PREFLIGHT_COMPONENTS=4`, `PREFLIGHT_MERGED=0`, `SUBMISSIONS_ROOT`, `CONTEXTS_ROOT`, `FEATURES_ROOT`, `CLASSIFIERS_ROOT`, `RUN_PREFIX`, `GROUP_PREFIX`, and `LOGICAL_PREFIX` (native defaults are unchanged).
