# DAIC audio-only LP-FT: LoRA trained through a warm-started classification head

Status: design, not implemented. Date: 2026-11-15. Lane: exp-audio-lora-recipe-fix-20261114 (PR #322).

## Question

Can LoRA fine-tuning of Qwen3-Omni improve on frozen Qwen3-Omni features + subject-mean L2 head on DAIC audio-only, when the fine-tuning objective is the head's classification loss instead of candidate-label likelihood?

## Why this arm

Three LoRA recipes trained on candidate-label likelihood all stayed below frozen + head on the DAIC official test (47 subjects, seed 7, subject level, macro-F1):

| Features | Decision | Macro-F1 |
|---|---|---|
| Frozen Qwen3-Omni | subject-mean L2 head | 0.770 |
| Paper LoRA | same head | 0.724 |
| Recipe-fix v1 LoRA | same head | 0.676 |
| Recipe-fix v2 (regularized) LoRA | same head | 0.585 |
| Recipe-fix v2 LoRA | LLM likelihood | 0.593 |

The features got worse for a head with every likelihood-trained recipe. This matches the feature-distortion argument of Kumar et al. (ICLR 2022, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"): fine-tuning from a randomly initialised or mismatched objective moves the representation away from what a linear probe uses, and linear probing first and then fine-tuning (LP-FT) reduces this. Here the "linear probe" already exists: the frozen subject-mean L2 head.

## Design

### Model and head

- Base: Qwen3-Omni-30B-A3B Thinker, audio encoder frozen, LoRA on attention q/k/v/o (as in v2: rank 8, alpha 16, dropout 0.1).
- Feature: final-layer post-norm hidden state at the last valid prompt token, the same vector as `src/features/extract_qwen_hidden.py` (`POOLING_NAME = "last_valid_prompt_token"`, `hidden_layer: final`). The training forward must use the same function (`_forward_final_hidden_states` without `inference_mode`) or a test must prove identical vectors.
- Head: one linear layer in fp32 on the raw 2048-d vector. The frozen head's train standardisation (`fit_scaler` mean and scale) is folded into the weights: `w' = w / scale`, `b' = b - (mean / scale) @ w`.
- Prompt: the harmonized DAIC prompt without a candidate label. One forward per chunk; no likelihood scoring.

### Warm start (the LP step)

- Refit the frozen head deterministically from the existing base-Omni DAIC cache with the paper settings (official train 107 subjects, L2 = 10.0 chosen on official dev, balanced class weights, threshold 0.5). It must reproduce the reference: test macro-F1 0.770, AUROC 0.892.
- Save `head_init.npz` (folded weight, bias, scaler mean/scale, L2, cache sha256s) and record its sha256 in `run_config.yaml`.
- At step 0 the LoRA B matrices are zero, so the model plus head equals frozen + head exactly (up to bf16/FSDP numerics).

### Objective

- Chunk-level BCE on the head logit, with the existing `class_equal_subject_mass` weights over packed30 subject-normalized chunk weights.
- Why not subject-level: the exact two-pass subject MIL in `src/daic_mil.py` requires one process, and Qwen3-Omni training needs FSDP over 4 GPUs (peak 36 GiB per GPU in v2). Subject-level loss under FSDP needs equal forward counts on every rank per subject; out of scope for this pilot.
- Prediction stays subject-level and identical to the frozen head: mean of chunk logits per subject (the head is linear, so this equals the head on the subject-mean feature), threshold 0.5.

### Optimisation

- Two parameter groups: LoRA lr 5e-5, weight decay 0.01; head lr 1e-4, weight decay matching the frozen head's L2 scale (fixed in config, not tuned).
- Effective batch 16 (accumulation 4 on 1 node x 4 GPUs), warmup 3 %, update-counted schedule (PR #286).
- At most 8 epochs. Early stopping on inner-val head BCE (subject level), patience 2.

### Selection and evaluation

- Epoch 0 (before any update) is evaluated on official dev and is a valid checkpoint candidate. If it wins, the result is "fine-tuning does not help".
- Checkpoint selection: inner-val macro-F1 (project invariant), ties broken by lower val BCE, then earlier epoch.
- `best_model` saves the LoRA adapter and `head.pt` (+ sha256). The test evaluation loads both; no `last_model` fallback.
- Metrics written in the existing `headline/binary_strict_*` schema (macro-F1, positive-F1, UAR; AUROC of the head probability as a secondary number). INVALID cannot occur.
- This is a different decision rule from likelihood. It is reported as its own row ("LoRA + head, LP-FT"), never in the likelihood LoRA column.

### Known compromise

Official dev (35 subjects) chooses the frozen head's L2 and also the LP-FT epoch. This is the same protocol as the frozen reference, but with 35 subjects the epoch choice is noisy.

## Gates before the real run

1. Unit tests: scaler fold-in equals standardise-then-head; train-time feature function equals the extraction vector; epoch 0 is a selection candidate; `head.pt` round-trip; objective dispatch refuses missing `head_init`.
2. Parity smoke on MN5 (non-reportable): load base + zero LoRA + head, run the dev and test forward without training. Dev and test subject logits must match the frozen head's logits (max abs difference to be fixed before the run, target < 0.05) and test macro-F1 must equal 0.770. If parity fails, stop: it is an implementation bug, not a result.
3. Short training smoke (20 subjects, 1 epoch) to check FSDP with the extra head parameters, memory and checkpoint saving.

## Pre-registered decision rule (DAIC official test, 47 subjects, seed 7)

- Fine-tuning helps: test macro-F1 >= 0.80 and selected epoch >= 1.
- Fine-tuning does not help: selected epoch 0, or test macro-F1 <= 0.77.
- Otherwise inconclusive (single seed, 47 subjects; one or two subjects move macro-F1 by about 0.03 to 0.05).

## Cost (MN5 quota core-hours; 1 GPU-hour = 20)

Measured v2 jobs (1 node x 4 GPUs): train 49046399 2 h 09 min (6 epochs, about 12 min training + 8 min likelihood dev eval per epoch, 10 min start-up); eval 49046400 6 min; extractions 7-8 min each.

| Job | Estimate | Core-hours |
|---|---|---|
| Parity smoke (no training) | about 25 min x 4 GPUs | about 35 |
| Training smoke (20 subjects, 1 epoch) | about 30 min x 4 GPUs | about 40 |
| LP-FT train (start-up + epoch-0 eval + up to 8 epochs of about 16 min) | 1.3 h if it stops at epoch 3-4, 2.4 h at most | 105-195 |
| Test evaluation | about 15 min x 4 GPUs | about 20 |
| Total | | about 200 expected, about 290 at most |

No hidden-state extraction is needed. One failed-attempt rerun would add up to about 195. Quota gate remainder on 2026-11-14: about 259k.

## Implementation work (estimate about 1-2 days)

1. `scripts/build_lpft_head_init.py`: refit and fold the frozen head, verify 0.770 / 0.892, write `head_init.npz` + identity JSON.
2. `src/model/lpft_head.py`: fp32 linear head; load/save with sha256.
3. `src/train.py`: `training.objective: head_bce` branch (prompt-only collator path, final-hidden capture, weighted BCE, two parameter groups, head inside the FSDP-prepared module), epoch-0 selection eval, head-based dev evaluation, `head.pt` in `best_model`.
4. Test evaluation entry point for `best_model` + `head.pt` writing the standard prediction and metric files.
5. Config `configs/experiments/daic_audio_only_harmonized_lpft_head_v1.yaml` and a new logical run in `experiments/definitions/audio-lora-recipe-fix-20261114.yaml`.
6. Tests listed under gates.
