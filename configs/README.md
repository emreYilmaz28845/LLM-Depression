# Configs

## Canonical evaluation policy for new experiments

New experiments use the likelihood evaluation. This is the canonical decision rule, not a per-run choice:

- **backend**: `evaluation.sample_prediction_mode: likelihood` with `evaluation.headline_mode: likelihood`; the
  per-subject decision is the argmax of the mean dep/non candidate scores (`argmax(mean dep_score,
  mean non_score)`), and Turkish pooled text uses its locked pair-margin rule
  (`subject_score_aggregation: turkish_pooled_text_pair_mean_margin_strict_v1`);
- **view**: `evaluation.evaluation_view: harmonized_all_windows_full_coverage`;
- **aggregation**: strict subject-level (`evaluation.aggregation_level: subject`);
- **headline metrics**: Macro-F1, Positive-F1 and UAR, read from `headline/binary_strict_*`
  (`binary_strict_uar` is the unweighted average recall, i.e. balanced accuracy); AUROC is not a headline;
- **INVALID counts as wrong**: a subject whose prediction is not a valid label counts as an error under the
  strict rule, and `valid_only_*` is not a headline;
- **no mixed columns**: historical teacher-forced results are never placed in the same comparison column (or
  table block) as likelihood results. Teacher forcing (`original_teacher_forced`) stays a separately labelled
  legacy/diagnostic view: its configs are archived under `configs/archive/pre_likelihood_20260917/` and its
  former canonical workbook values live in the workbook's "Legacy TF" sheet.

Teacher-forced evaluation is not run for new experiments. Historical configs, archived results and existing
`run_config.yaml` files are not rewritten: they remain the record of what ran under the older recipe, and a
report that cites them must say which backend/view each number comes from. Checkpoint selection and early
stopping stay `inner_val_macro_f1`, mode `max`.

## Cross-validation reporting rule

For every model, head, language, and standalone CV dataset, report the
unweighted mean of the per-fold strict subject-level metrics. This includes
D3TEC (Spanish) and Androids (Italian). Report Macro-F1 and Positive-F1 with
the same fold set. Require all five distinct folds; do not average a partial
run. With multiple seeds, first average folds within each seed, then average
the seed means. Do not concatenate outer-fold predictions to compute a
pooled-CV headline. This rule replaces the older dataset-specific convention.

DAIC official-test results stay single-test results. Merged CV retains its
dataset mean within fold followed by the fold mean. A mean does not turn
selected-validation results into held-out test results.

This is a reporting rule, not a model configuration option. Do not change
`evaluation.aggregation_level`, `subject_score_aggregation`, training,
checkpoint selection, or single-fold evaluation to implement it. Turkish
`pooled_t17` combines question conditions, not outer-fold F1, and is unchanged.
Keep historical artifacts intact and regenerate reports from their exact
fold evidence. Old pooled-CV reports are historical, not current headlines.

```text
configs/
  quarantines.yaml   # subject quarantine list; every config references it
  main/              # active canonical configs
  labels/            # single-token A/B likelihood experiment configs (not canonical)
  experiments/       # active non-headline research
  archive/           # superseded recipes retained for reproducibility
```

## Harmonized main recipe

The active harmonized family is:

`harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_v1_promptcontext_v1`

It covers D3TEC, Turkish BDI≥17 with Qwen3-ASR, Androids, DAIC-WOZ, and CMDC in audio-only, text-only, and audio+text modes.

- One participant-audio window per prompt; never a joint-audio bundle.
- Windows are at most 30 seconds and do not overlap.
- Audio+text repeats the full participant transcript on every window.
- Every training window appears once per epoch; DataLoader shuffling changes only its order.
- `training.class_balance: none`.
- D3TEC, Turkish, Androids, and CMDC use subject → source unit → window loss weighting and response-subject evaluation.
- DAIC uses participant-only speech packed from raw timestamp intervals into consecutive 30-second chunks, subject-normalized loss weighting, and all-chunk subject aggregation.
- Validation checkpoint selection and early stopping use `inner_val_macro_f1`, mode `max`.
- Evaluation uses candidate-label **likelihood** (`sample_prediction_mode: likelihood`) as the canonical decision rule; the per-subject decision is the argmax of the mean dep/non candidate scores. Teacher forcing (`original_teacher_forced`) is a labelled legacy view: its configs are archived under `configs/archive/pre_likelihood_20260917/` and its former canonical workbook values live in the workbook's "Legacy TF" sheet.
- The audio encoder remains frozen because `audio_adapter.enabled` and `train_projector` are false.

### Default backbone policy

The 15 unqualified canonical configs use the current production backbones:

- text-only: `model_backend: qwen38` with the pinned Qwen3.8-27B snapshot;
- audio-only and audio+text: `model_backend: qwen3omni` with the
  Qwen3-Omni-30B-A3B Thinker. The Talker is not retained;
- both families use `promptcontext_v1`, FSDP, BF16 inference, CPU activation
  offload, likelihood evaluation and standalone evaluation after training.

The pre-migration Qwen2/Qwen2-Audio versions of those exact 15 files are kept
under `configs/archive/pre_default_backbone_20260923/`. Explicit Gemma, English,
official-development, E-DAIC and secondary Turkish configs keep their named
backends and recipes. Run `python scripts/build_canonical_backend_configs.py
--check` after editing a canonical config.

Naming:

```text
<dataset>[_t<threshold>]_<modality>_harmonized_selmacrof1_likelihood_v1[_variant].yaml
```

The superseded teacher-forced configs were moved to:

```text
configs/archive/pre_likelihood_20260917/
```

The nine earlier superseded DAIC, CMDC, and Turkish positive-F1 main configs live in:

```text
configs/archive/pre_harmonized_posf1_20260809/
```

## Prepared A/B likelihood family

The 15 core Qwen harmonized cells also have configs named
`*_likelihood_ab_v1.yaml` under `configs/labels/`. They cover D3TEC, Turkish
positive-only BDI≥17, Androids, DAIC, and CMDC in audio-only, text-only, and
audio+text modes. They have not been trained as a family. The earlier
teacher-forced configs are archived for their historical runs
(`configs/archive/pre_likelihood_20260917/`). The earlier DAIC
pilot config (`daic_text_only_harmonized_selmacrof1_likelihood_ab.yaml`) sits
beside them and is superseded by the v1 family.

Each new config records `short_internal_ab_labels` with explicit
`A = Depressed` and `B = Non-depressed` mapping. Its prompt prints the same
legend; training and evaluation both use those internal labels. Checkpoint
selection and the headline use `likelihood` with `inner_val_macro_f1` in max
mode. Every config records `evaluation_view: harmonized_all_windows_full_coverage` and writes to an isolated
`output_model/likelihood_ab_v1/<modality>/<dataset>/` root. Manifest and split
paths, dataset settings, windowing, weights, and LoRA settings match the
corresponding canonical likelihood config (and its archived teacher-forced
source). Report Macro-F1, Positive-F1, and UAR
from strict subject-level metrics; UAR is `binary_strict_uar`.

Before using a model, check that A and B are each one token at the rendered
prompt boundary with its actual processor. The weight-free audit for the two
Qwen backbones is:

```bash
python tools/verify_ab_label_tokens.py \
  --text-model-dir /path/to/Qwen2-7B-Instruct \
  --audio-model-dir /path/to/Qwen2-Audio-7B-Instruct \
  --output outputs/tokenizer_probes/ab_family_audit.json
```

The existing launchers and merged-training configs run the canonical
`*_likelihood_v1` family. No training job is implied by adding these YAML files.

## Gemma 4 DAIC family

The Gemma 4 backbone comparison is scoped to DAIC only and uses three configs:

```text
daic_<modality>_harmonized_selmacrof1_likelihood_v1_gemma4_12b.yaml   (text_only | audio_only | audio_text)
```

- Backend: `model_backend: gemma4` on `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (see
  `docs/GEMMA4_DAIC_IMPLEMENTATION_RUNBOOK.md`).
- They preserve every scientific invariant of their Qwen counterparts
  (dataset, seed, sample mode, packed30 chunking, subject-normalized weights,
  inner-validation macro-F1 selection, likelihood evaluation) and change
  only: backbone, model path/revision, isolated output roots
  (`output_model/harmonized_v1_gemma4_likelihood/`), the LoRA target regex (six modules
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

Count the current inventory with `find configs/main -maxdepth 1 -type f -name '*.yaml' | wc -l`; do not copy an old total into plans or reports. The active families include:

- the 15 core harmonized default configs: five datasets × three modalities
  (Qwen3.8 text-only; Qwen3-Omni Thinker audio-only and audio+text);
- 5 isolated Turkish negative-only t17 secondary configs: three native
  modalities plus English audio+text and text-only;
- the 3 Gemma 4 DAIC configs described above;
- 3 unchanged E-DAIC configs outside the harmonized family;
- additional active, explicitly named comparison families. Inspect their YAML instead of inferring semantics from the total count.

## Turkish positive-only canonical family and negative-only secondary family

### Four-source Turkish and geriatri extension

`turkish_all_geriatri_t17_<modality>_harmonized_selmacrof1_likelihood_v1_*` is a
separate, native-Turkish comparison family for audio-only, text-only, and
audio+text. Each config reads four sources: the existing Turkish positive and
negative question sets, plus the geriatri positive and negative question sets.
The source list in each YAML is the input contract. The two existing sets share
their patient IDs and BDO scores. The two geriatri sets also share patients, but
geriatri patient codes are reused independently of the existing cohort. Manifest
IDs therefore use `geriatri:` for that cohort; both question sets stay in the
same patient fold. Config names mirror the pooled baseline cell of the same
modality so the two arms read as one pair.

Labels use `depresyon_skoru >= 17` from the source CSVs. The geriatri workbook
is an audit reference: one BDO entry differs from both geriatri CSVs (14 in
the CSVs, 35 in the workbook). The CSV score is the agreed label source. BAÖ
and the CSV `label` category do not set the depression target. Geriatri WAV
basenames are matched after Unicode normalization; the original filenames and
audio paths are retained in the manifest. Both geriatri sets need their own
Qwen3-ASR transcript JSONL before a combined manifest can be built.

The family is fold-locked: `split.locked_original_folds_path` and
`split.locked_original_folds_sha256` point at the canonical pooled folds file
(sha256 `3262a009db52c6d049e223947a9be6ce119a31e816b5c3072ce84b3ad92ecd58`), the
original 120 participants keep exactly those folds, and the geriatri participants
are placed deterministically and balanced within label strata. The build fails
closed when the declared hash, the original cohort membership, or the fold count
disagrees, and it records the resulting fold-lock audit beside the split. The
manifest is fold-agnostic, so a rebuild that only changes the split keeps the
recorded four-source manifest content hash.

Each config carries the baseline cell's recipe: the same model identity, prompt
text and version, labels, LoRA policy, checkpoint selection, windowing,
evaluation view, aggregation and effective global batch. Two runtime facts
differ and are intentional: only the data contract changes, and the Qwen3-Omni
cells declare two training nodes with accumulation 16 (the shape the baseline
cells were submitted with) instead of one node with accumulation 32.

This family has isolated manifest, split, and model output roots. It does not
replace or rewrite the positive-only and negative-only experiments.

Canonical Turkish is positive-only: the `turkish_pos_only_t17_*` configs cover
the question-set-1 recordings (filenames `*-1-*`). The name `mixed` was wrong
and is retired; see `experiments/definitions/turkish_pos_only_rename_map.yaml`
for the old→new map. Old `turkish_t17_*` files remain as legacy history.

The `turkish_negative_only_t17_*` configs reuse the harmonized Turkish recipe
for the negative-question recordings (filenames `*-2-*`). They are a secondary comparison, not an
independent population: the 120 subjects and their threshold-17 labels are the
same as canonical Turkish. Keep each subject in the same fold across both
variants and never pool the variants without cross-variant patient grouping.

The native family has audio-only, audio+text, and text-only configs. The English
family has audio+text and text-only configs; an English audio-only config would
be input-identical to native audio-only and is intentionally absent. The loader
uses `metadata_schema: minimal_t17`, derives the label from
`depresyon_skoru >= 17`, and does not consume acoustic features, anxiety,
demographic, or comorbidity fields. Native and English manifests, splits,
translation caches, and model roots are isolated from canonical Turkish.
The configs read `whisper_transcripts_qwen3_asr_reviewed.jsonl`, an audited
derivative of the immutable Qwen3-ASR output. Native-speaker corrections are
applied with `scripts/apply_reviewed_transcript_corrections.py`. The English
configs use `harmonized_en_complete_v3/turkish_negative_only_t17`; v3 preserves
the failed first production attempt and cancelled same-output retry, keeps the
1,169 unaffected translations, and records the reviewed transcript/translation
correction in `repair_provenance.json`.

See `docs/harmonized_dataset_baseline.md` for the methodology and dataset-specific adapters.

## Harmonized reproduction matrix

The standalone execution matrix is `configs/experiments/harmonized/standalone_matrix.yaml`. It expands to 63 four-GPU training jobs: one DAIC fold and five folds for each other dataset, across three modalities. D3TEC, Androids, and DAIC also receive separate deterministic evaluation jobs. Hidden-state postprocessing runs fixed Logistic Regression and fixed XGBoost; it does not run Optuna.

The matching merged configs are:

- `configs/experiments/merged/symmetric_merged_harmonized_audio_text_likelihood_v1.yaml`
- `configs/experiments/merged/symmetric_merged_harmonized_audio_only_likelihood_v1.yaml`
- `configs/experiments/merged/symmetric_merged_harmonized_text_only_likelihood_v1.yaml`

They use only the 15 harmonized component configs. Each component and merged fit has a maximum of 20 epochs, validation macro-F1 checkpoint selection, patience 3, and no XGBoost Optuna. Merged cross-validation selects by mean dataset macro-F1; the final training epoch is the rounded median selected cross-validation epoch.

MN5 execution order:

1. `scripts/submit_harmonized_preflight.sh` rebuilds all manifests on GPFS and validates paths, files, hashes, splits, and merged protocols without using a GPU.
2. `scripts/submit_harmonized_standalone.sh` submits the standalone reproduction matrix only after that preflight passes.
3. `scripts/submit_harmonized_merged.sh` submits the merged smoke, cross-validation, and final stages separately.

Both GPU launchers require `GITHUB_ISSUE` and `GITHUB_PR`. For the full harmonized reproduction campaign, use Issue #12 and primary methodology PR #10. The production Git SHA must contain both PR #10 and its PR #11 acceptance-auditor correction. These fields provide scientific context; the full Git SHA and deployed-source hash remain the canonical source identity.

All launchers default to dry-run. Their default lane counts run the whole matrix in parallel: one four-GPU training lane per training task and one one-GPU auxiliary lane per auxiliary job. There is no project-wide GPU cap; the scheduler, account, and QoS limits are the only binding constraints. Use `MAX_CONCURRENT_TRAINS` / `MAX_CONCURRENT_AUX` (or `MAX_CONCURRENT_POSTPROCESS`) to tune lane counts.

## Harmonized English-translation family

Issue #20 tracks the English-transcript comparison. The eight canonical English configs in `main/` are named `<dataset>_<modality>_harmonized_selmacrof1_likelihood_v1[_qwen3asr]_en.yaml` and are derived only from the native harmonized counterparts, never from `configs/experiments/translation_en/` (historical recipe, do not reuse).

- Recipe ID: `harmonized_full_transcript_single30_allwindows_selmacrof1_likelihood_en_v1`.
- Each config adds a `transcripts:` block: `variant: english`, `cache_path: ${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}/harmonized_en_complete_v1/<dataset>/accepted.jsonl`, `minimum_status: automatic_low`, `require_complete: true`, `include_failed: false`.
- Outputs are English-specific: `outputs/manifests_harmonized_en/`, `outputs/splits_harmonized_en/`, `output_model/harmonized_v1_en_likelihood/`.
- Only audio+text and text-only exist for D3TEC, Androids, CMDC, and Turkish t17. No English audio-only, DAIC, or E-DAIC configs.
- The fixed English matrix is `configs/experiments/harmonized/english_translation_matrix.yaml`: 8 experiments, 40 training folds, 20 separate evaluation folds (D3TEC, Androids), 40 hidden-extraction/fixed-head folds, exactly 100 jobs, no Optuna, no merged training, no audio-only cells.

MN5 execution order:

1. `scripts/submit_harmonized_en_preflight.sh` rebuilds the four English manifests on GPFS from the repaired `harmonized_en_complete_v1` translation cache, audits translation completeness, native/English input equivalence, and tokenizer/context fit, and records the expected 100-job scope. Requires `GITHUB_ISSUE=20` and the implementation `GITHUB_PR`.
2. `scripts/submit_harmonized_en_standalone.sh` submits the English matrix only after that preflight audit passes with `status: passed` and zero failures. Use the same `GITHUB_ISSUE=20` and `GITHUB_PR`.
3. `scripts/submit_harmonized_standalone_retry.sh` retries failed cells with new attempt identities; it accepts the English roots and prefixes through `PREFLIGHT_COMPONENTS=4`, `PREFLIGHT_MERGED=0`, `SUBMISSIONS_ROOT`, `CONTEXTS_ROOT`, `FEATURES_ROOT`, `CLASSIFIERS_ROOT`, `RUN_PREFIX`, `GROUP_PREFIX`, and `LOGICAL_PREFIX` (native defaults are unchanged).
