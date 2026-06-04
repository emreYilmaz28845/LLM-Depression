# LLM-Depression

Leakage-safe Qwen2-Audio reproduction pipeline for binary depression detection with audio + text only.

## Core Rules
- Audio + text only
- English prompts
- Original transcript language
- Fixed labels: `Depressed` and `Non-depressed`
- No SECap
- Subject-level leakage-safe splits with configurable subject/segment evaluation reporting
- Likelihood is the headline evaluation; generation is secondary

## Environment

Target runtime: MareNostrum5 `qwen_mn5_rebuilt`

Capture commands:

```bash
./scripts/capture_environment.sh
```

## Build Manifests

```bash
python src/data/build_manifest.py --config \
  configs/daic_audio_text.yaml \
  configs/cmdc_audio_text.yaml \
  configs/eatd_audio_text.yaml
```

Or:

```bash
./scripts/validate_manifests.sh
```

## Validation / No-Model Checks

```bash
./scripts/sanity_tests_no_model.sh
```

This runs:
- manifest creation
- DAIC join audit generation
- CMDC fold proof output
- EATD SDS consistency and pooled class-count recovery

## DAIC Training / Eval

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/daic_audio_text.yaml \
  --fold 0 \
  --run_name daic_reproduction
```

Use `--set evaluation.aggregation_level=segment` to switch checkpoint selection and held-out headline metrics to segment-level evaluation while still writing subject-level aggregate outputs.

Use `--set lora.last_n_layers=2` to restrict LoRA to the final two language-model decoder layers.

Final evaluation is on the official DAIC dev partition only.

Standalone checkpoint evaluation:

```bash
python src/evaluate.py \
  --config configs/daic_audio_text.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/daic/daic_reproduction/fold_0/best_model
```

Segment-level override:

```bash
python src/evaluate.py \
  --config configs/edaic_audio_text_reg3.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/edaic/edaic_reproduction/fold_0/best_model \
  --set evaluation.aggregation_level=segment
```

Example training override for last-two-layer LoRA:

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/edaic_audio_text_reg3.yaml \
  --fold 0 \
  --run_name edaic_last2_lora \
  --set lora.last_n_layers=2
```

Standalone DAIC test-only evaluation with repeated participant full transcripts per chunk:

```bash
python src/evaluate.py \
  --config configs/daic_audio_text_test_eval_full_transcript.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/daic/daic_reproduction/fold_0/best_model
```

Enable `DepAdapter` from CLI overrides:

```bash
sbatch --export=ALL,CONFIG=$PWD/configs/daic_audio_text_paper_audio_text.yaml,FOLD=0,RUN_NAME=daic_dep_adapter,EXTRA_TRAIN_ARGS="--set audio_adapter.enabled=true --set audio_adapter.adapter_dim=512 --set audio_adapter.dropout=0.1 --set audio_adapter.train_projector=false" scripts/run_train_slurm.sh
```

Enable `DepAdapter` and train `multi_modal_projector` too:

```bash
sbatch --export=ALL,CONFIG=$PWD/configs/daic_audio_text_paper_audio_text.yaml,FOLD=0,RUN_NAME=daic_dep_adapter_projector,EXTRA_TRAIN_ARGS="--set audio_adapter.enabled=true --set audio_adapter.adapter_dim=512 --set audio_adapter.dropout=0.1 --set audio_adapter.train_projector=true" scripts/run_train_slurm.sh
```

## CMDC 5-Fold Training / Eval

```bash
./scripts/run_cmdc_5fold.sh
```

Equivalent single fold:

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/cmdc_audio_text.yaml \
  --fold 0 \
  --run_name cmdc_reproduction
```

## EATD 3-Fold Training / Eval

Default mode is `subject` mode.

```bash
./scripts/run_eatd_3fold.sh
```

Equivalent single fold:

```bash
torchrun --nproc_per_node=4 src/train.py \
  --config configs/eatd_audio_text.yaml \
  --fold 0 \
  --run_name eatd_reproduction
```

## EATD Sample Modes

- `subject`: combine negative/neutral/positive into one subject sample
- `response`: treat negative/neutral/positive as separate samples and aggregate back to subject level

Default: `subject`

Subject-mode length controls are in `configs/eatd_audio_text.yaml`:
- `max_audio_seconds_per_response`
- `max_total_audio_seconds`
- `transcript_max_chars`

## With-Model Smoke Test

Default model path:

```bash
MODEL_PATH=/home/emre/models/Qwen2-Audio-7B-Instruct ./scripts/sanity_tests_with_model.sh
```

This checks:
- processor/tokenizer load
- model + LoRA load
- one collated batch
- label-mask debug
- one forward pass
- one generation
- one likelihood score pass

## Slurm

Generic Slurm entrypoints:

```bash
sbatch scripts/run_train_slurm.sh
sbatch scripts/run_eval_slurm.sh
```

Optuna HPO entrypoint:

```bash
sbatch scripts/run_optuna_slurm.sh
```

Set variables such as:
- `CONFIG`
- `FOLD`
- `RUN_NAME`
- `CHECKPOINT_DIR`
- `MODEL_PATH`

For Optuna studies you will typically set:
- `CONFIG`
- `FOLD`
- `N_TRIALS`
- `MODEL_PATH`
- `STUDY_NAME`
- `EXTRA_HPO_ARGS`

Example:

```bash
sbatch --export=ALL,CONFIG=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/configs/daic_audio_text.yaml,FOLD=0,N_TRIALS=40,STUDY_NAME=daic_fold0_optuna,EXTRA_HPO_ARGS="--run-name-prefix daic_hpo --save_strategy hpo_minimal --trial-train-epochs 10 --lr-min 5e-6 --lr-max 5e-5 --lora-r-choices 2,4,8 --lora-alpha-choices 4,8,16 --lora-dropout-min 0.1 --lora-dropout-max 0.3 --weight-decay-min 0.01 --weight-decay-max 0.1" scripts/run_optuna_slurm.sh
```

By default, Optuna now searches:
- `lr`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `weight_decay`

Default safe HPO profile:
- `40` trials
- `10` epochs per trial
- `lr`: `5e-6` to `5e-5` with log sampling
- `lora_r`: `2, 4, 8`
- `lora_alpha`: `4, 8, 16`
- `lora_dropout`: `0.1` to `0.3`
- `weight_decay`: `0.01` to `0.1`

You can disable the last two with:

```bash
--no-search-lora-dropout --no-search-weight-decay
```

Study artifacts are written under:

```text
outputs/optuna/{dataset}/{study_name}/
```

Key HPO outputs:
- `study_config.json`
- `study_results.json`
- `study_results_table.csv`
- `{study_name}.db`
- `trial_runtime/`
- `materialized_best_trial_summary.json` when `--materialize-best-trial` is enabled

## Outputs

Main outputs are written under:

```text
output_model/audio_text/{dataset}/{run_name}/fold_{k}/
```

Important artifacts:
- `best_model/`
- `last_model/`
- `logs/split_used.json`
- `logs/sample_partition_counts.json`
- `logs/train_truncation.jsonl`
- `logs/val_truncation.jsonl`
- `logs/final_eval_truncation.jsonl`
- `eval/best_checkpoint/`
- `eval/last_checkpoint/`
- `eval/best_vs_last_checkpoint_metrics.json`
