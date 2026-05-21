# LLM-Depression

Leakage-safe Qwen2-Audio reproduction pipeline for binary depression detection with audio + text only.

## Core Rules
- Audio + text only
- English prompts
- Original transcript language
- Fixed labels: `Depressed` and `Non-depressed`
- No SECap
- Subject-level leakage-safe reporting
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

Final evaluation is on the official DAIC dev partition only.

Standalone checkpoint evaluation:

```bash
python src/evaluate.py \
  --config configs/daic_audio_text.yaml \
  --fold 0 \
  --checkpoint_dir output_model/audio_text/daic/daic_reproduction/fold_0/best_model
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

Set variables such as:
- `CONFIG`
- `FOLD`
- `RUN_NAME`
- `CHECKPOINT_DIR`
- `MODEL_PATH`

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
