# LLM-Depression

Leakage-safe binary depression classification with Qwen2-Audio-7B / Qwen2-7B LoRA fine-tuning, across audio+text, audio-only, and text-only modalities on DAIC-WoZ, E-DAIC, CMDC, and a Turkish corpus.

This is the repository overview, not the source of truth for individual experiment settings. Read, in order:

1. `docs/DEVICES.md` — host topology, environments, and sync boundaries.
2. `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` — full cluster lifecycle: submit → monitor → sync back → validate.
3. `configs/README.md` — canonical config recipe and naming.
4. `docs/SIGNAL_FLOW.md` — how a raw recording becomes a prediction (manifest → examples → collator → model → metrics).

Do not infer a current protocol from an archived config or historical result document.

## Canonical recipe

Canonical configurations live only in `configs/main/` — one per dataset × modality, named `<dataset>[_t<threshold>]_<modality>_selposf1_tf.yaml`. All of them use:

- teacher-forced label decoding (`original_teacher_forced`) as the headline evaluation backend;
- `headline/binary_strict_*` metrics, where invalid decoded labels count as wrong (`valid_only_*` is ignored);
- validation positive-F1 (`inner_val_positive_f1`, mode max) for checkpoint selection;
- a frozen audio encoder by default (`DepAdapter` and projector training are opt-in);
- English prompts and external labels `Depressed` / `Non-depressed`; transcripts stay in their original language;
- no AUROC — teacher-forced decoding emits a hard label, so there is no ranking to compute AUROC over.

Current canonical coverage:

| Dataset | Modalities | Notes |
|---|---|---|
| DAIC | audio+text, audio-only, text-only | Subject-audio uses fixed K=4 chunks; canonical eval is balanced K4 joint bundles covering all chunks |
| EDAIC | audio+text, audio-only, text-only | Subject-audio uses K=4 |
| CMDC | audio+text, audio-only, text-only | Response samples aggregate to the configured headline level |
| Turkish | audio+text, audio-only, text-only | BDI threshold 17, Qwen3-ASR transcripts, five-fold `train_val` CV |

`configs/experiments/` holds active non-headline research; `configs/archive/` is history and must not be treated as the current recipe. Turkish BDI≥21, Turkish BDI≥25, and EATD are not current headline configs.

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
export EDAIC_DATASET_ROOT=/path/to/EDAIC
export CMDC_DATASET_ROOT=/path/to/CMDC
export TURKISH_DATASET_ROOT=/path/to/Turkish
export MODEL_PATH=/path/to/Qwen2-Audio-7B-Instruct
export TEXT_MODEL_PATH=/path/to/Qwen2-7B-Instruct
```

## Build manifests

Manifests and splits are shared across modalities — build them once per dataset, and include transcripts even for audio-only runs:

```bash
python src/data/build_manifest.py --config \
  configs/main/daic_audio_text_selposf1_tf.yaml \
  configs/main/edaic_audio_text_selposf1_tf.yaml \
  configs/main/cmdc_audio_text_selposf1_tf.yaml

python src/data/build_manifest.py --config \
  configs/main/turkish_t17_audio_text_selposf1_tf_qwen3asr.yaml
```

Every config references `configs/quarantines.yaml` via `${PROJECT_ROOT}` — never move it. Config values support `${VAR}` / `${VAR:-default}` interpolation, and any key can be overridden on the CLI with `--set path.to.key=value`.

## Train and evaluate

Real training runs on MN5 through Slurm (see MN5 lifecycle below). The commands below are the application interface:

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

`best_model` is the evaluated checkpoint (val positive-F1 selection) — never substitute `last_model` silently. Evaluation bypasses `AudioTextDataset` (deterministic, no augmentation).

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

### DAIC leakage constraint

Chunk count perfectly encodes the DAIC label, so every canonical subject-audio example uses a fixed K=4 chunks. Training resamples K per epoch; canonical evaluation builds balanced K4 joint bundles that cover all of a subject's chunks and aggregates at subject level. State which eval view a reported DAIC result used (full coverage vs fixed-K4 — same checkpoint can score 0.841 vs 0.755).

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
- Translation overlays: `configs/features/translation_en_matrix.yaml`
- D3TEC: `docs/D3TEC_IMPLEMENTATION.md`
- Merged training: `docs/SYMMETRIC_MERGED_PROTOCOL_PLAN.md`
- Qwen3-Omni: `docs/QWEN3_OMNI_IMPLEMENTATION.md`

Read the workflow doc and its current configs/scripts before executing.
