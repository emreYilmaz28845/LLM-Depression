# LLM-Depression

Leakage-safe binary depression classification with parameter-efficient audio-language models. The repository compares audio+text, audio-only, and text-only conditions across D3TEC, Turkish, Androids, DAIC-WoZ, CMDC, and legacy E-DAIC. The main model families are Qwen2-Audio-7B / Qwen2-7B and Gemma 4 12B.

The central research question is whether raw speech adds reliable subject-level information beyond transcripts after controlling subject leakage, unequal recording length, language, dataset, model family, and evaluation procedure. The intended contribution is a harmonized multilingual evaluation protocol and empirical comparison, not a claim that this is the first audio-language model used for depression detection.

This is the repository overview, not the source of truth for individual experiment settings. Read, in order:

1. `docs/DEVICES.md` — host topology, environments, and sync boundaries.
2. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` — full cluster lifecycle: submit → monitor → sync back → validate.
3. `configs/README.md` — canonical config recipe and naming.
4. `docs/SIGNAL_FLOW.md` — how a raw recording becomes a prediction (manifest → examples → collator → model → metrics).

Do not infer a current protocol from an archived config or historical result document.

## Canonical recipe

The active harmonized configurations live in `configs/main/`, named `<dataset>[_t<threshold>]_<modality>_harmonized_selmacrof1_tf[_variant].yaml`. They use:

- teacher-forced label decoding (`original_teacher_forced`) as the current historical headline backend;
- `headline/binary_strict_*` metrics, where invalid decoded labels count as wrong (`valid_only_*` is ignored);
- validation macro-F1 (`inner_val_macro_f1`, mode max) for checkpoint selection and early stopping;
- a frozen audio encoder by default (`DepAdapter` and projector training are opt-in);
- English prompts and external labels `Depressed` / `Non-depressed`; transcripts stay in their original language;
- no AUROC — teacher-forced decoding emits a hard label, so there is no ranking to compute AUROC over.

### Evaluation warning

`original_teacher_forced` is the repository's current historical headline protocol, but it is not a clean deployable classifier decision. It reconstructs label tokens under a gold-conditioned continuation. Some audio paths later aggregate gold-independent candidate-label score margins, while text-only paths generally retain the reconstructed-label decision. Results carrying the same backend name can therefore have different effective decision semantics across modalities.

Do not use the current teacher-forced view as final evidence that one modality outperforms another. Paper-primary comparisons must first use one gold-independent rule for every modality, such as normalized candidate-label likelihood or constrained two-label decoding, with the same subject-level aggregation. Keep teacher-forced results as a clearly labelled legacy or diagnostic view. See `docs/LLM_CLASSIFICATION_INFERENCE_INVESTIGATION.md`.

Current canonical coverage:

| Dataset | Modalities | Notes |
|---|---|---|
| D3TEC | audio+text, audio-only, text-only | All ≤30 s response windows; full participant transcript for audio+text |
| Turkish | audio+text, audio-only, text-only | BDI threshold 17, Qwen3-ASR, all recording windows, five-fold `train_val` CV |
| Androids | audio+text, audio-only, text-only | All participant-turn windows; full participant transcript for audio+text |
| DAIC | audio+text, audio-only, text-only | One participant-only packed30 chunk per prompt; all chunks used |
| CMDC | audio+text, audio-only, text-only | All answer windows, including audio after the first 30 seconds |
| EDAIC | audio+text, audio-only, text-only | Unchanged legacy positive-F1 K4 configs; outside the harmonized family |

`configs/experiments/` holds active non-headline research; `configs/archive/` is history and must not be treated as the current recipe. Turkish BDI≥21, Turkish BDI≥25, and EATD are not current headline configs.

Current research families extend the core modality matrix with:

- Qwen versus Gemma 4 backbone comparisons;
- native-language versus English-translated transcripts for non-English datasets;
- standalone versus symmetric merged multi-dataset training;
- direct generative labels versus Logistic Regression and XGBoost heads over hidden representations;
- fixed-head and Optuna-tuned classifier studies.

These datasets share a binary output format, not one identical clinical target. D3TEC and DAIC derive labels from questionnaire thresholds, while Androids and CMDC use diagnosis-oriented labels. Merged-model results must therefore be reported per dataset and described as cross-corpus learning, not as if every positive label represented the same clinical construct.

## Local environment

The shell starts in conda `base`, which has no PyTorch. Activate the project env first:

```bash
conda activate llmdep4090
python -m pytest tests/     # from the repo root; bare pytest fails to import src/scripts
```

The no-model sanity suite is `./scripts/sanity_tests_no_model.sh` (builds manifests and audits splits; needs the dataset roots). `scripts/sanity_tests_with_model.sh` is for machines with the local base models and GPU capacity.

Config defaults are BSC/GPFS absolute paths; override them for local runs:

```bash
export DAIC_DATASET_ROOT=/path/to/DAIC
export DAIC_UNPROCESSED_ROOT=/path/to/DAIC-WOZ/unprocessed
export DAIC_LABEL_ROOT=/path/to/DAIC-WOZ/minimal_zips
export EDAIC_DATASET_ROOT=/path/to/EDAIC
export CMDC_DATASET_ROOT=/path/to/CMDC
export TURKISH_DATASET_ROOT=/path/to/Turkish
export D3TEC_DATASET_ROOT=/path/to/D3TEC
export ANDROIDS_DATASET_ROOT=/path/to/Androids-Corpus
export MODEL_PATH=/path/to/Qwen2-Audio-7B-Instruct
export TEXT_MODEL_PATH=/path/to/Qwen2-7B-Instruct
```

## Build manifests

Manifests and splits are shared across modalities — build them once per dataset, and include transcripts even for audio-only runs:

```bash
for config in \
  configs/main/d3tec_audio_text_harmonized_selmacrof1_tf.yaml \
  configs/main/turkish_pos_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml \
  configs/main/androids_audio_text_harmonized_selmacrof1_tf.yaml \
  configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml \
  configs/main/cmdc_audio_text_harmonized_selmacrof1_tf.yaml; do
  python src/data/build_manifest.py --config "$config"
done
```

Every config references `configs/quarantines.yaml` via `${PROJECT_ROOT}` — never move it. Config values support `${VAR}` / `${VAR:-default}` interpolation, and any key can be overridden on the CLI with `--set path.to.key=value`.

## Train and evaluate

Real training runs on MN5 through Slurm (see MN5 lifecycle below). The commands below are the application interface:

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml \
  --fold 0 \
  --run_name <unique-run-name>

python src/evaluate.py \
  --config configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/daic/<run-name>/fold_0/best_model
```

`best_model` is the evaluated checkpoint (validation macro-F1 selection for harmonized runs) — never substitute `last_model` silently. Evaluation bypasses `AudioTextDataset` (deterministic, no augmentation).

Local 5-fold reproduction loops: `scripts/run_daic_5fold.sh`, `scripts/run_edaic_5fold.sh`, `scripts/run_cmdc_5fold.sh`, `scripts/run_turkish_5fold.sh`.

Small local smoke on one GPU:

```bash
torchrun --nproc_per_node=1 src/train.py \
  --config configs/main/<config>.yaml \
  --fold 0 \
  --run_name <unique-smoke-name> \
  --set training.num_train_epochs=1 \
  --set split.smoke_subject_limit=6
```

### DAIC harmonized protocol

DAIC audio is rebuilt from the raw interview and timestamped `Participant` intervals. Ellie audio and text are excluded. Participant speech is packed into consecutive non-overlapping chunks of at most 30 seconds. Every prompt contains one chunk, every training chunk appears once per epoch, subject-normalized loss weights equalize subject influence, and evaluation averages all chunk score margins at subject level. Historical K4 results are not interchangeable with this protocol.

### Turkish protocol

The leakage unit is `patient_id`. The canonical BDI≥17 configs use five-fold `train_val` CV: the outer fold both selects the checkpoint and supplies the reported fold score, so it is not an independent held-out test.

## MN5 lifecycle

Two endpoints, different jobs: `transfer1.bsc.es` for rsync and file inspection, the scheduler login (`alogin1`/`alogin2.bsc.es`) for `sbatch`/`squeue`/`sacct`. Training runs only on Slurm compute nodes — never on transfer or login nodes. Read `docs/DEVICES.md` and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` before any cluster action.

Sync the repo with `bash scripts/sync_to_cluster.sh` (captures `.provenance/`, respects `.gitignore`). Canonical single-fold submission:

```bash
CONFIG="$PWD/configs/main/<config>.yaml" \
RUN_NAME=<unique-run-name> \
FOLD=0 \
bash scripts/submit_train_and_eval.sh
```

Submission is not completion. Monitor jobs, rsync the compact evidence back (metrics JSONs, `predictions_subject_level.csv`, `final_summary.json`, `run_config.yaml` — not just checkpoints), validate locally, then report. Cluster mutations require explicit user authorization.

### Harmonized reproduction launchers

The harmonized matrix must pass a CPU-only MN5 preflight before any GPU job is submitted. The preflight rebuilds manifests from GPFS data, rejects local `/media/...` paths, checks every referenced file, and builds the merged protocols. Start every command with `DRY_RUN=1` and inspect its output before using `DRY_RUN=0`:

```bash
RUN_ID=<unique-id> DRY_RUN=1 bash scripts/submit_harmonized_preflight.sh
RUN_ID=<same-id> DRY_RUN=1 bash scripts/submit_harmonized_standalone.sh
RUN_ID=<same-id> STAGE=smoke DRY_RUN=1 bash scripts/submit_harmonized_merged.sh
```

After the real preflight finishes successfully, reuse its `RUN_ID` for the standalone launcher and each merged stage. The launchers default to full-matrix parallelism: every training cell gets its own four-GPU lane and every auxiliary job its own one-GPU lane, so the scheduler, account, and QoS limits are the only binding constraints. There is no project-wide GPU cap; raise or lower the `MAX_CONCURRENT_*` lane counts when a plan needs it. Hidden-state postprocessing runs fixed Logistic Regression and fixed XGBoost only; XGBoost Optuna is disabled. The merged smoke, cross-validation, and final stages remain separate so later stages cannot start before their acceptance checks.

## Experiment tracking and reporting

Every reported result must come with its provenance — a bare number is a bug. Runs carry sidecar files beside the authoritative `run_config.yaml` and are indexed in a rebuildable local SQLite registry:

```bash
python tools/rebuild_experiment_registry.py --scan-root output_model --dry-run
python tools/exp.py list
python tools/exp.py show <attempt-id>
python tools/exp.py provenance <metric-id>
python tools/generate_run_report.py --attempt-id <attempt-id> --fold <n>
python tools/generate_group_report.py --attempts <csv> --metric-name <name> --namespace <ns> --backend <b> --view <v> --aggregation <agg>
python tools/export_run_to_wandb.py --attempt-id <id> --mode dry_run
```

Training/evaluation can be given `--experiment-context <json>` so rank 0 writes the sidecars on the cluster. The canonical results workbook is `depression_results_clean.xlsx`, generated by `scripts/build_clean_workbook.py` — never hand-edit the cells. A headline number must identify run/attempt + fold, config and hashes, checkpoint, backend, view, aggregation, job/resubmission chain, and a locally verified artifact path.

## Specialized workflows

Active non-headline workflows, each with its own doc and configs:

- Hidden-state classifiers and Optuna HPO: `configs/features/*.yaml` matrices, `scripts/run_optuna_slurm.sh`, `docs/OPTUNA_RAW_XGBOOST_FOLLOWUP.md`
- Translation overlays: `configs/experiments/harmonized/english_translation_matrix.yaml` and `configs/experiments/harmonized/gemma4_english_translation_matrix.yaml`
- D3TEC: `docs/D3TEC_IMPLEMENTATION.md`
- Merged training: `docs/SYMMETRIC_MERGED_PROTOCOL_PLAN.md`
- Qwen3-Omni: `docs/QWEN3_OMNI_IMPLEMENTATION.md`

Read the workflow doc and its current configs/scripts before executing.

## Dataset and pipeline comparison

This table describes the current harmonized protocol. It deliberately contains no result values. Read metrics only from locally verified artifacts or the generated workbook with complete run, fold, split, backend, view, aggregation, and checkpoint provenance. See `docs/dataset_pipeline_audit.md` and `docs/DATASET_QUESTION_STRUCTURE_RECOVERY.md` for the detailed evidence and open questions.

| Dataset | Language and target | Natural unit | Current harmonized audio coverage | Evaluation protocol | Main scientific risk |
|---|---|---|---|---|---|
| D3TEC | Spanish; questionnaire-threshold label | Response to one of 27 elicitation prompts | Every non-overlapping response window of at most 30 seconds | Nested five-fold `train_val_test`; subject-level aggregation | Arbitrary cuts in long responses, inferred prompt metadata, and full-transcript repetition beside local audio |
| Turkish | Turkish; BDI threshold | Provider-cut recording associated with a patient | Every available natural-unit window | Five-fold `train_val`; the selected outer fold is not an independent test | Source recording protocol, speaker content, and held-out-test status require careful interpretation |
| Androids | Italian; diagnosis-oriented label | Manually isolated participant turn | Every non-overlapping participant-turn window of at most 30 seconds | Nested five-fold `train_val_test`; subject-level aggregation | Question context is omitted and recording/turn patterns may act as shortcuts |
| DAIC-WoZ | English; PHQ-8 threshold | Participant-only speech packed into consecutive chunks | Every participant-only packed30 chunk, one chunk per prompt | Official train/development/test handling with subject-normalized aggregation | Ellie question context and response latency are omitted; full transcript is repeated beside local audio |
| CMDC | Mandarin; diagnosis-oriented label | Participant answer to one interview topic | Every answer window, including audio beyond the first 30 seconds | Five-fold `train_val`; the selected outer fold is not an independent test | Symptom-heavy questions, institution effects, omitted question text, and non-held-out reporting |
| E-DAIC | English; legacy depression label protocol | Legacy K4 bundle | Legacy fixed bundle rather than harmonized all-window coverage | Legacy positive-F1 checkpoint selection | Not methodologically interchangeable with the harmonized family |

Across harmonized audio+text configs, the complete participant transcript is repeated beside each local audio window. This is a deliberate practical baseline, not true local audio-text alignment. Paper-primary work should compare it with locally aligned transcript windows, transcript-only inference, audio-only inference, and late fusion.

The recovered question structures also differ sharply across datasets: D3TEC uses balanced-valence prompts, Androids uses largely neutral everyday questions, DAIC includes screening and clinical branches, and CMDC uses symptom-heavy topics. Question-aware ablations and dataset-shortcut checks are needed before claiming a corpus-independent depression signal.
