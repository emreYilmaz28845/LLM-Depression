# DAIC audio-only LP-FT: LoRA trained through a warm-started classification head

Status: implemented (lane exp-audio-lpft-head-20261115). Date: 2026-11-15. Lane: exp-audio-lora-recipe-fix-20261114 (PR #322).

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
- `best_model` saves the LoRA adapter and `lpft_head.pt` (+ sha256). The test evaluation loads both; no `last_model` fallback.
- Metrics written in the existing `headline/binary_strict_*` schema (macro-F1, positive-F1, UAR; AUROC of the head probability as a secondary number). INVALID cannot occur.
- This is a different decision rule from likelihood. It is reported as its own row ("LoRA + head, LP-FT"), never in the likelihood LoRA column.

### Known compromise

Official dev (35 subjects) chooses the frozen head's L2 and also the LP-FT epoch. This is the same protocol as the frozen reference, but with 35 subjects the epoch choice is noisy.

## Gates before the real run

1. Unit tests (`tests/test_lpft_head.py`): scaler fold-in equals standardise-then-head; head forward and L2 term; head state round trip; last-prompt-token indexing; collator emits targets and refuses a prompt that is not a token prefix; selection eval scores subject-mean logits; epoch-0 parity check.
2. Local head-init check (`scripts/build_lpft_head_init.py`): the refit selects L2 10.0 on official dev and reproduces test macro-F1 0.770 / AUROC 0.892; the folded head differs from the standardised head by about 4e-15; `scripts/lpft_head_apply.py` on the frozen test cache with the saved fp32 head gives 0.770 again.
3. In-run gates instead of separate MN5 smokes (implementation decision): `split.smoke_subject_limit` also caps the selection split, which would break the parity check, and the real run reaches every gate early. Epoch 0 runs the dev evaluation through the training-time feature path and stops the job unless dev macro-F1 equals 0.7464 exactly and subject BCE is within 0.01 of 0.4993 (minimum dev subject margin 0.166, so bf16 noise should not flip a subject). Epoch 0 then saves `best_model` with `lpft_head.pt` before any update, and epoch 1 exercises FSDP with the replicated head. This saves the two smoke jobs (about 75 core-hours).

## Pre-registered decision rule (DAIC official test, 47 subjects, seed 7)

- Fine-tuning helps: test macro-F1 >= 0.80 and selected epoch >= 1.
- Fine-tuning does not help: selected epoch 0, or test macro-F1 <= 0.77.
- Otherwise inconclusive (single seed, 47 subjects; one or two subjects move macro-F1 by about 0.03 to 0.05).

## Cost (MN5 quota core-hours; 1 GPU-hour = 20)

Measured v2 jobs (1 node x 4 GPUs): train 49046399 2 h 09 min (6 epochs, about 12 min training + 8 min likelihood dev eval per epoch, 10 min start-up); eval 49046400 6 min; extractions 7-8 min each.

| Job | Estimate | Core-hours |
|---|---|---|
| LP-FT train (start-up + epoch-0 eval + up to 8 epochs of about 16 min) | 1.3 h if it stops at epoch 3-4, 2.4 h at most | 105-195 |
| Paired likelihood eval job (secondary, not the decision rule) | about 10 min x 4 GPUs | about 15 |
| Hidden-state extraction of best_model, test view | about 8 min x 4 GPUs | about 15 |
| Total | | about 140 expected, about 230 at most |

No hidden-state extraction is needed. One failed-attempt rerun would add up to about 195. Quota gate remainder on 2026-11-14: about 259k.

## Implementation work (estimate about 1-2 days)

1. `scripts/build_lpft_head_init.py`: refit and fold the frozen head, verify 0.770 / 0.892, write `head_init.npz` + identity JSON.
2. `src/lpft_head.py`: fp32 linear head; load/save with sha256.
3. `src/train.py`: `training.objective: head_bce` branch (prompt-only collator path, final-hidden capture, weighted BCE, two parameter groups, head inside the FSDP-prepared module), epoch-0 selection eval, head-based dev evaluation, `lpft_head.pt` in `best_model`.
4. `scripts/lpft_head_apply.py`: applies `best_model/lpft_head.pt` to the best_model hidden-state extraction (test view) and writes metrics and subject predictions with the frozen-head metric code.
5. Config `configs/experiments/daic_audio_only_harmonized_lpft_head_v1.yaml` and a new logical run in `experiments/definitions/audio-lora-recipe-fix-20261114.yaml`.
6. Tests listed under gates.
