# LLM-Depression

Leakage-safe binary depression classification with Qwen2-Audio/Qwen2 LoRA across audio+text, audio-only, and text-only modalities.

This is the repository overview, not the source of truth for individual experiment settings. Read, in order:

1. `docs/DEVICES.md` for local/MN5 environments and synchronization boundaries.
2. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` for the complete cluster lifecycle.
3. `configs/README.md` and the selected YAML for the current recipe.
4. `docs/SIGNAL_FLOW.md` for the data/model/evaluation path.

Use experiment-specific plans under `docs/` for non-headline work. Do not infer a current protocol from an archived config or historical result document.

## Canonical recipe

Canonical configurations live only in `configs/main/`. They use:

- English prompts and external labels `Depressed` / `Non-depressed`;
- transcripts in their original language;
- teacher-forced label-token decoding (`original_teacher_forced`) as the headline backend;
- `headline/binary_strict_*` metrics, where invalid decoded labels count as wrong;
- validation positive-F1 (`inner_val_positive_f1`, mode `max`) for checkpoint selection;
- a frozen audio encoder by default;
- no AUROC for the canonical teacher-forced hard-label recipe.

`configs/experiments/` contains active non-headline research. `configs/archive/` is historical and must not be treated as the current recipe.

Current canonical coverage is:

| Dataset | Modalities | Notes |
|---|---|---|
| DAIC | audio+text, audio-only, text-only | Audio bundles contain a constant K=4 chunks; canonical audio evaluation uses balanced K4 coverage |
| EDAIC | audio+text, audio-only, text-only | Subject-level audio uses K=4 |
| CMDC | audio+text, audio-only, text-only | Response samples aggregate to the configured headline level |
| Turkish | audio+text, audio-only, text-only | BDI threshold 17, Qwen3-ASR transcripts, five-fold `train_val` CV |

Turkish BDI≥21, Turkish BDI≥25, and EATD are not current headline configs. Consult experimental or archived files only when explicitly reproducing those protocols.

## Local environment and validation

The shell normally starts in Conda `base`, which does not contain PyTorch. Activate the project environment before Python tests:

```bash
conda activate llmdep4090
python -m pytest tests/
```

Run a targeted file while iterating:

```bash
python -m pytest tests/test_experiment_tracking_contracts.py -q
```

Never use bare `pytest`; tests import both `src` and `scripts`. The model-free dataset suite is:

```bash
./scripts/sanity_tests_no_model.sh
```

It requires the relevant dataset roots and writes manifest/audit artifacts. Use `scripts/sanity_tests_with_model.sh` only when the matching local base models and GPU capacity have been verified.

For local data/model resolution, set the relevant variables explicitly:

```bash
export DAIC_DATASET_ROOT=/path/to/DAIC
export EDAIC_DATASET_ROOT=/path/to/EDAIC
export CMDC_DATASET_ROOT=/path/to/CMDC
export TURKISH_DATASET_ROOT=/path/to/Turkish
export MODEL_PATH=/path/to/Qwen2-Audio-7B-Instruct
export TEXT_MODEL_PATH=/path/to/Qwen2-7B-Instruct
```

## Build manifests

Manifests and splits are shared across modalities. Build from one current config per dataset; preprocessing inputs must include transcripts even for audio-only runs.

```bash
python src/data/build_manifest.py --config \
  configs/main/daic_audio_text_selposf1_tf.yaml \
  configs/main/edaic_audio_text_selposf1_tf.yaml \
  configs/main/cmdc_audio_text_selposf1_tf.yaml

python src/data/build_manifest.py --config \
  configs/main/turkish_t17_audio_text_selposf1_tf_qwen3asr.yaml
```

All configs reference `configs/quarantines.yaml` through `${PROJECT_ROOT}`. Never move it.

## Train and evaluate

All real training runs on MN5 through Slurm. The generic commands below show the application interface; use the repository wrappers and the MN5 runbook for real submission.

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/main/daic_audio_text_selposf1_tf.yaml \
  --fold 0 \
  --run_name <unique-run-name>

python src/evaluate.py \
  --config configs/main/daic_audio_text_selposf1_tf.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/daic/<run-name>/fold_0/best_model
```

For a small local smoke, use one GPU and explicitly reduce the workload:

```bash
torchrun --nproc_per_node=1 src/train.py \
  --config configs/main/<config>.yaml \
  --fold 0 \
  --run_name <unique-smoke-name> \
  --set training.num_train_epochs=1 \
  --set split.smoke_subject_limit=6
```

Any YAML key can be overridden with `--set path.to.key=value`. Record every override in experiment provenance.

### DAIC leakage constraint

DAIC chunk count encodes the label. Never create a canonical subject-audio example with a label-dependent or variable number of chunks. Each audio-bearing example must contain K=4 chunks. Training resamples K per epoch; the current canonical evaluation policy constructs balanced K4 bundles to cover the subject's chunks and aggregates them at subject level. State whether a reported DAIC result used balanced full coverage or a single fixed-K4 view.

### Turkish protocol

The leakage unit is `patient_id`. The canonical BDI≥17 configurations use five-fold `train_val` CV: the outer fold selects the checkpoint and supplies the reported fold score, so it is not an independent held-out test. Do not describe it as one. The previous `train_val_test` protocol and other thresholds are non-canonical unless explicitly selected.

## MN5 lifecycle

Read `docs/DEVICES.md` and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` before any SSH, rsync, or cluster action. Use:

- `transfer1.bsc.es` for rsync and file inspection;
- the currently verified scheduler login for `sbatch`, `squeue`, and `sacct`;
- Slurm compute nodes for training/evaluation, never transfer or login nodes.

Canonical single-fold submission from the verified scheduler login:

```bash
CONFIG="$PWD/configs/main/<config>.yaml" \
RUN_NAME=<unique-run-name> \
FOLD=0 \
bash scripts/submit_train_and_eval.sh
```

Submission is not completion. Monitor through terminal accounting, retrieve compact evidence and logs, validate locally, and only then report results. Cluster mutations require explicit user authorization.

## Experiment tracking and reporting

Useful entrypoints:

```bash
python tools/rebuild_experiment_registry.py --scan-root output_model --dry-run
python tools/exp.py list
python tools/exp.py show <attempt-id>
python tools/exp.py provenance <metric-id>
python tools/generate_run_report.py --attempt-id <attempt-id> --fold <n>
```

Every reported metric must identify the run/attempt and fold, config and hashes, checkpoint, backend, view, aggregation, job/resubmission chain, and a locally verified artifact path. Generate `depression_results_clean.xlsx` through `scripts/build_clean_workbook.py`; never hand-edit workbook cells.

## Specialized workflows

- Hidden-state classifiers and Optuna: `configs/features/*.yaml` matrix configs, `scripts/run_optuna_slurm.sh`, and `docs/OPTUNA_RAW_XGBOOST_FOLLOWUP.md`
- Translation overlays: `configs/features/translation_en_matrix.yaml`
- D3TEC: `docs/D3TEC_IMPLEMENTATION.md`
- Merged training: `docs/SYMMETRIC_MERGED_PROTOCOL_PLAN.md`

Read the workflow-specific document and current scripts/configs before executing it.
