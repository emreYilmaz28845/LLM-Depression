# Handoff — Subject-level K-chunk audio (+text) for depression detection

This document hands off the **subject-level K-chunk** work on the Qwen2-Audio-7B + LoRA
binary depression-detection pipeline. Read this before continuing; it captures the problem,
the design decisions, the bugs we hit, and what's left.

> **THE central problem is OVERFITTING — see §0.** Everything else (subject K-chunking, the reg
> grid, fixed K) is a response to it. Read §0 first.

## 0. The main problem: overfitting

Tiny corpora, a 7B backbone, and a leaky audio channel. That combination overfits hard.

### Why it happens
- **Tiny labelled sets.** DAIC test N=47, EDAIC N=56, CMDC N=78, EATD N=162 (and only ~14–30
  positives each). A 7B model + LoRA has *far* more capacity than these sets can constrain. It
  memorises subjects instead of learning depression cues.
- **Audio leaks shortcuts.** The chunk-count leak (10 vs 15 chunks → separability ≈ 1.000, see §2)
  is the clearest example: the model learns "count chunks" instead of "hear depression". Audio also
  carries speaker identity, recording/channel conditions, and site artifacts that correlate with
  label in small corpora — the model latches onto those instead of generalizable affect cues.
- **Severe class imbalance.** ~3:1 (DAIC/EDAIC) up to 4.4:1 (EATD). The cheap minimum is to collapse
  to the majority (or, under light reg, over-predict the minority). We see *both* failure modes:
  reg2 collapses to **all-positive** on DAIC audio+text (ACC 0.298); EATD default collapses to
  over-predicting depression (83 FP, below baseline).
- **Long sequences, few examples.** ~750 audio tokens/30s × K chunks = a lot of parameters-to-fit
  per gradient step with very few independent examples — high variance, easy memorisation.

### How we're already fighting it
- **Subject-level fixed K=4** (§2): closes the chunk-count leak and forces one example per subject so
  the model can't exploit per-chunk count or over-weight talkative subjects.
- **Per-epoch stochastic K-chunk sampling** (§3): each epoch the subject is represented by a *different*
  K-subset of its chunks → combinatorial augmentation (C(N,K) views), the model can't memorise a fixed
  audio fingerprint. This is our main **runtime data-augmentation** lever today.
- **A real regularization sweep** (reg1–4, §5): increasing dropout, weight decay, lower lr, fewer
  LoRA modules / last-n-layers, lower max_grad_norm, early stopping on inner-val positive-F1.
  reg3/reg4 (heavy reg) are the stable configs; light reg overfits/collapses.
- **Text is the clean channel.** Transcript length does NOT leak (separability 0.457). Text-only is
  the robust baseline; audio is the part that overfits.
- **Frozen audio encoder (DepressInstruct recipe) — IMPLEMENTED.** LoRA `target_modules`
  (`q/k/v_proj`) also match the Whisper `audio_tower` attention, so LoRA was silently being trained
  *on the audio encoder* — exactly the overfit liability DepressInstruct warns about. The legacy
  `hybrid` configs excluded it; the `subject_audio reg1–4` configs did **not**, so the reported
  subject-level numbers (§6) were produced with a leaking encoder. Fix: `build_lora_config` now
  defaults `exclude_modules=".*audio_tower.*|.*multi_modal_projector.*"` (opt out with
  `lora.tune_audio_encoder: true`), and `enforce_audio_encoder_freeze` (in `qwen2audio_lora.py`)
  verifies/freezes any leaked encoder LoRA each load and logs the count. **Action: re-run the
  DAIC/EDAIC `subject_audio` + `subject_audio_text` reg sweep with the freeze before trusting §6.**

### What we could still do (candidate next steps)
- **Audio data augmentation at runtime** — **WAVEFORM AUG IMPLEMENTED** (see below); SpecAugment
  (mel time/freq masking) and reverb still candidate.
- **More aggressive chunk-subset augmentation**: the K-sweep is DONE (§8 K-ablation) — K=4 is the peak,
  not beaten by K∈{2,3,5,6}; only the precision↑/recall↓-with-K trend is defensible. Multi-view training
  (several K-subsets/subject/epoch) and multi-view eval (average probs over views — the sanctioned §3
  exception) are still open.

#### Waveform acoustic augmentation — IMPLEMENTED (2026-06-15)
`apply_audio_augment` in `src/data/runtime.py`: train-only waveform aug attacking the speaker/channel/site
shortcut. Effects (each fires with prob `prob`, magnitude ~U[lo,hi], range=null disables): **pitch shift**,
**time-stretch**, **gain**, **additive noise (target SNR)**. Routed through the dataset's seeded
`_rng`/`_np_rng` so the per-epoch view is reproducible for a given `seed`.
- **Train-only by construction**: passed to `train_dataset` ONLY (not selection/audit); eval loads audio via
  `src/evaluate.py` backends, NOT `AudioTextDataset`, so the §3 determinism rule holds with no extra guard.
- **VRAM-safe**: output truncated to ≤ original length, so time-stretch can't inflate the audio-token budget.
- **Robust**: pitch/time-stretch need `librosa.effects` (→ numba); if unavailable they no-op with a one-time
  warning while noise/gain (pure NumPy) stay active.
- **Config**: `data.audio_augment` block; turn-key config `daic_subject_audio_text_reg1_selmacrof1_tf_aug.yaml`
  (canonical decided default + aug, K=4). Read the train→inner-val gap (overfit signal), not the noisy N=47 test F1.

#### Cross-corpus eval (EDAIC→DAIC) — DEFERRED (planned, do LATER)
The truest generalization probe and the cleanest place to measure whether waveform aug actually helps. The
augment levers above are best validated here (in-corpus audio ≈ text at the noise floor, §6). Intentionally
deferred for now per user; revisit after the in-corpus aug screen.
- **Class-imbalance handling**: class-balanced / weighted sampling, focal or class-weighted loss,
  threshold calibration on inner-val instead of fixed 0.5. Essential for EATD.
- **Stronger parameter regularization**: even lower rank (4), higher dropout, LoRA only on attention,
  shorter training / lower patience, or freezing the audio encoder entirely and training only the
  text+projector path.
- **Better evaluation discipline**: confirm overfitting with an early-stopping-disabled long run
  (watch train-vs-inner-val gap), and report **paired** audio-vs-text with McNemar significance so we
  stop chasing noise-level deltas.
- **Cross-corpus eval** (train EDAIC → test DAIC): the truest overfitting/generalization probe. If
  audio truly helps, it should help most here; if it's memorising, it collapses.

## 1. The overarching problem

Binary depression detection (depressed / non-depressed) on four corpora:

- **DAIC**, **EDAIC** — English, **fixed** train/val/test splits.
- **CMDC**, **EATD** — Chinese, **cross-validation** (CMDC 5-fold, EATD 3-fold; metrics pooled
  by summing per-fold confusion matrices over held-out folds).

Backbone: `Qwen2-Audio-7B-Instruct` (audio path) or `Qwen2-7B-Instruct` (text-only path) + LoRA (PEFT).
Cluster env: `qwen_mn5_rebuilt`. Hardware: **4× H100**. Local validation env (has soundfile/torch/yaml):
`/home/emre/miniconda3/envs/secap/bin/python` (the base env lacks these).

**Central concern: overfitting**, especially on the audio channel. Text is the strong, stable
channel; audio was the overfit liability. Everything below is in service of making audio
generalize without leaking.

## 2. Why subject-level K-chunking exists (the leak we closed)

The original audio mode was **chunk-level** (`sample_mode: chunk`): one training example per audio
chunk. Problem discovered:

- **Chunk-count leaks the label.** Non-depressed subjects were segmented into ~10 chunks,
  depressed into ~15. Chunk count alone separated the classes with separability ≈ 1.000. Any
  model that even implicitly counts chunks cheats.
- Transcript **length** does NOT leak (separability ≈ 0.457) — so the text channel is clean.

**Fix = `sample_mode: subject_audio`**: exactly **one example per subject**, carrying a **fixed
K audio chunks** (we use **K=4**). Fixing K neutralizes the chunk-count leak — every subject
contributes the same number of chunks regardless of class.

### Regularization view of K
- Distinct K-subsets per subject = C(N, K); epoch-to-epoch overlap ≈ K²/N.
- **Larger K → weaker regularization** (more overlap, closer to using everything).
- So K is a knob: small K = stronger combinatorial augmentation but less audio seen per epoch.
- **K=all is NOT advisable**: (a) re-introduces the chunk-count leak (depressed subjects expose
  more chunks), and (b) 15 chunks × ~750 tokens/30s is far too long to fit in memory.

## 3. The eval-determinism rule (HARD CONSTRAINT — do not violate)

> Training may use stochastic K-chunk sampling per subject per epoch, but **validation/test must
> be deterministic**. No random sampling for reported val/test metrics unless you implement
> repeated-sampling evaluation and average probabilities across views.

Implementation of the asymmetry:
- **Training** randomness lives ONLY in `AudioTextDataset.__getitem__` (via `chunk_sampling="random"`,
  seeded `random.Random`). Each epoch re-samples K of N chunks per subject.
- **Eval** reads the **baked** `example["audio_paths"]` directly (deterministic, evenly-spaced K
  chosen by `_evenly_spaced_indices`). Selection/eval datasets use `chunk_sampling="deterministic"`.

This mode was added as a **new explicit mode**, NOT a replacement of `sample_mode: chunk`. Both coexist.

## 4. Where the code lives (key files & functions)

- **`src/data/runtime.py`** — example builder:
  - `qwen2audio_audio_token_length(mel_frames)`: `(((mel-1)//2+1)-2)//2+1` → 3000 mel frames ≈ 750 audio tokens (30s).
  - `_evenly_spaced_indices(total, count)` — deterministic eval view.
  - `_build_subject_level_audio_examples(...)` — groups chunks by subject, validates single label
    (and single transcript when `use_text`), bakes deterministic `audio_paths` (K), keeps full
    `subject_chunk_paths` pool, records `chunks_per_subject`, `max_audio_seconds_per_chunk`.
    For `use_text` it injects the transcript via
    `render_user_prompt_text(..., is_subject_bundle=True, audio_context_override=...)`.
  - Dispatch in `build_examples`: `if sample_mode == "subject_audio" and dataset_name in {"daic","edaic"}`.
  - `AudioTextDataset` gained `chunk_sampling` (None/"deterministic"/"random") + `chunk_sampling_seed`;
    `_resolve_audio_plan` samples K from the pool when "random".
- **`src/train.py`** — instrumentation + training wiring:
  - `train_dataset` → `chunk_sampling="random"` when subject_audio; `selection_dataset` → `"deterministic"`.
  - Per-epoch, at top of loop after `model.train()`: `restore_model_for_training(unwrap(model), config)`
    re-enables gradient checkpointing + `enable_input_require_grads` + `use_cache=False` (eval flips these).
  - **Checkpoint selection is configurable** via `_resolve_selection_metric` (`training.selection_metric`,
    default `inner_val_positive_f1`; `selection_metric_mode: auto` → `min` for `*_loss`). It chooses which
    epoch is saved as `best_model` and is **independent of `early_stopping.metric`** (which only decides
    *when* to stop). Before this, selection was hardcoded to `positive_f1` — see §8. Available metric keys:
    `inner_val_{positive_f1,macro_f1,accuracy,precision,recall,loss}`. NB: `hpo.py` maximizes `best_metric`,
    so do not pair a `*_loss` selection metric with HPO without flipping its direction.
  - VRAM/audio logging: `_log_peak_gpu_memory` (max_allocated/max_reserved GiB),
    `_audit_audio_budget` (counts `<|AUDIO|>` tokens → `audio_budget_audit_<partition>.json`),
    `_percentile`. `torch.cuda.reset_peak_memory_stats()` before the loop; `peak_gpu_memory.json` at end.
  - After each epoch barrier: `gc.collect(); torch.cuda.empty_cache()`.
- **`src/model/qwen2audio_lora.py`** & **`src/model/text_lora.py`**: `restore_model_for_training(model, config)`
  (symmetric). **`src/model/runtime.py`**: dispatch wrapper picking backend by `resolve_input_modality`.

## 5. Configs (the reg grid)

Naming: `{dataset}_{modality}_reg{N}.yaml` where modality ∈ {audio_only, text_only, subject_audio,
subject_audio_text}. Subject configs set `chunks_per_subject: 4`, `max_audio_seconds_per_chunk: 30.0`,
`audio_budget_audit: true`. DAIC configs keep DAIC's `split.mode: fixed` (select on val),
`likelihood` eval, `manifest_variant: preprocessed_full_transcript_all_splits`; DAIC text uses Qwen2-7B-Instruct.

Reg grid (rank / α / dropout / lr / wd / warmup / max_grad_norm / patience / #target_modules):
- **reg1**: 16 / 32 / 0.05 / 2e-4 / 0    / 0.03 / 1.0 / 3 / 7 modules
- **reg2**: 16 / 32 / 0.10 / 1e-4 / 0.01 / —    / 0.8 / 3 / 7 modules
- **reg3**: 8  / 16 / 0.20 / 5e-5 / 0.05 / —    / 0.5 / 2 / 4 modules (last_n_layers=2)
- **reg4**: 8  / 16 / 0.25 / 3e-5 / 0.08 / —    / 0.4 / 2 / 4 modules

**reg3/reg4 are the safe, stable picks for audio paths. reg1/reg2 (light reg) are unstable and
collapse to all-positive on some datasets — do not trust them blindly.**

## 6. Current results (see `results_subject_level_daic_edaic.md` for full tables)

Best per dataset:
- **DAIC**: audio+text reg1 — ACC 0.851, F1 0.696 (but reg2 collapsed to all-positive here).
- **EDAIC**: audio+text reg2 — ACC 0.768, F1 0.667.
- **CMDC**: audio+text default — ACC 0.987 ⚠️ **SUSPECT** (near-perfect on 78 subjects; likely a
  corpus-wide artifact/leak — must be scrutinized, not reported as a win).
- **EATD**: audio+text default — ACC 0.420 ⚠️ **BELOW the 0.815 majority baseline**; collapsed to
  over-predicting depression (83 FP) on a 4.4:1 imbalanced set.

Key finding (read carefully — the results file is too generous):
- **Audio does NOT measurably beat text-only.** DAIC Δ F1 = +0.004, EDAIC Δ F1 = 0.000 (best-of-sweep
  vs best-of-sweep). On test N=47/56, one subject flipping moves F1 by ~0.03–0.05, so these deltas are
  **below the single-subject noise floor**. The only consistent (but weak) signal is small positive ACC
  bumps (+0.021, +0.036) that never go negative. **Conclusion: audio+text ≈ text-only; audio contributes
  no defensible signal once text is present.** Do not claim an audio advantage without a **paired,
  same-reg, same-seed** comparison + **McNemar's test** on subject-level predictions — we have not run that.
- **What IS clearly true:** subject_audio fixed-K beats chunk-level audio (EDAIC subject_audio reg3
  F1 0.552 vs chunk audio_only reg2 F1 0.437). The leak fix + subject aggregation matter; audio-on-top-
  of-text does not.

## 7. Open / suggested next steps

### IN-FLIGHT — audio-encoder freeze re-run (DepressInstruct, lit-review approach #1)
LoRA `target_modules` (`q/k/v_proj`) also matched the Whisper `audio_tower`, so the
`subject_audio*` runs were silently LoRA-tuning the audio encoder — the overfit liability. **Fixed:**
`build_lora_config` (`src/model/lora_common.py`) now defaults
`exclude_modules=".*audio_tower.*|.*multi_modal_projector.*"` (opt out with `lora.tune_audio_encoder: true`);
`enforce_audio_encoder_freeze` (`src/model/qwen2audio_lora.py`) verifies/freezes any leaked encoder LoRA
at load and logs `encoder_lora_leaked_params` / `encoder_lora_frozen_params`. Verify a run actually froze
by grepping the train log for `Audio-encoder freeze guard`.

- **Re-run driver:** `scripts/run_frozenenc_daic_edaic.sh` — chains all 14 DAIC/EDAIC audio configs
  (`subject_audio*` reg1–4; EDAIC has no reg1) sequentially via `SBATCH_DEPENDENCY=afterany:<prev_train_id>`
  (one 4×H100 node at a time). `--no-chain` for parallel; `RUN_SUFFIX` overrides the `_frozenenc` run-name
  suffix. text_only is excluded (no encoder → freeze is a no-op, its numbers stay valid).
- **DAIC best-config spot-checks submitted** with the freeze: `daic_subject_audio_reg3_frozenenc`,
  `daic_subject_audio_text_reg1_frozenenc`.
- **The §6 / results_subject_level_daic_edaic.md `subject_audio*` + `audio+text` numbers are STALE** (leaking
  encoder) — replace them with the frozen-encoder re-run before drawing conclusions.
- **TODO:** rebuild the results tables from the `_frozenenc` runs; then re-assess the audio-vs-text question
  (§6) on the clean numbers. Consider generating the missing EDAIC reg1 configs for a paired comparison.

### Other open items (confirm with user before starting)
1. **Investigate CMDC 0.987** for corpus artifacts/leakage (recording/site conditions vs label)
   before trusting or reporting it.
2. **Fix EATD**: it only has the single default (light) config. Port the **reg3 recipe + class
   handling** (it's heavily imbalanced 132:30); add class-balanced sampling.
3. **CMDC & EATD lack a reg sweep and modality ablation** — only the single default audio+text config
   exists. Generate reg1–4 and audio_only / text_only configs to put them on equal footing with DAIC/EDAIC.
4. Optional: EDAIC→DAIC cross-corpus eval; a long run with early stopping disabled to confirm the
   overfitting story directly.

## 8. Freeze × selection-metric × decision-rule study (DAIC reg1 audio+text)

Controlled 2×3×2 grid on a single recipe (reg1: rank16/α32/dropout0.05/lr2e-4, 7 LoRA modules),
DAIC fixed split, **test N=47 (33 non-dep / 14 dep)**. Everything held constant except three axes:
**audio-encoder freeze** (Yes/No), **checkpoint-selection metric** (positive-F1 / macro-F1 / val-loss),
and **test decision rule** (`likelihood` / `original_teacher_forced`). `F1` = positive-class F1.
`(eN)` = epoch saved as `best_model`. TF INVALIDs are scored as **wrong** (strict mapping, see below).

All-negative baseline: ACC 0.702, F1 0.000.

### Frozen encoder (DepressInstruct; 524K trainable LoRA params, decoder-only)
| selection | eval | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --------- | ---- | ----- | ----- | --------- | ------ | ------------------ |
| positive-F1 (e4) | likelihood | 0.766 | 0.718 | 0.560 | 1.000 | 22,11 / 0,14 |
| positive-F1 (e4) | teacher_forced | 0.745 | 0.700 | 0.538 | 1.000 | 21,12 / 0,14 |
| macro-F1 (e4) | likelihood | 0.766 | 0.718 | 0.560 | 1.000 | 22,11 / 0,14 |
| **macro-F1 (e4)** | **teacher_forced** | 0.787 | **0.737** | 0.583 | 1.000 | 23,10 / 0,14 |
| val-loss (e3) | likelihood | 0.787 | 0.643 | 0.643 | 0.643 | 28,5 / 5,9 |
| val-loss (e3) | teacher_forced | 0.809 | 0.690 | 0.667 | 0.714 | 28,5 / 4,10 |

### Non-frozen encoder (`lora.tune_audio_encoder: true`; 3.93M trainable LoRA params, ~7.5×)
| selection | eval | ACC | F1 | Precision | Recall | CM [TN,FP / FN,TP] |
| --------- | ---- | ----- | ----- | --------- | ------ | ------------------ |
| positive-F1 (e4) | likelihood | **0.851** | 0.696 | 0.889 | 0.571 | 32,1 / 6,8 |
| positive-F1 (e6) | teacher_forced | 0.787 | 0.706 | 0.600 | 0.857 | 25,8 / 2,12 |
| macro-F1 (e3) | likelihood | 0.809 | 0.667 | 0.692 | 0.643 | 29,4 / 5,9 |
| macro-F1 (e6) | teacher_forced | 0.787 | 0.706 | 0.600 | 0.857 | 25,8 / 2,12 |
| val-loss (e2) ⚠️ | likelihood | 0.787 | 0.444 | 1.000 | 0.286 | 33,0 / 10,4 |
| val-loss (e2) ⚠️ | teacher_forced | 0.830 | 0.600 | 1.000 | 0.429 | 33,0 / 8,6 |

### Findings
- **The famous 0.851 ACC is the NON-frozen (leaky) model, reproduced exactly** ([32,1/6,8]) — it's a
  conservative, high-precision (0.889) / low-recall (0.571) operating point that misses 43% of depressed.
  High accuracy by rarely saying "depressed" + the encoder shortcut, not by hearing depression.
- **Best positive-F1 in the whole grid is a FROZEN cell** (macro-F1 × TF, F1 0.737). Training the audio
  encoder buys **no defensible F1 gain**; freezing removes the leak, matches/beats F1, and shrinks
  trainable params ~7.5×. This is the §0 overfitting thesis confirmed on a controlled grid.
- **val-loss selection is a trap when the encoder is NOT frozen.** The bigger encoder-LoRA capacity
  overfits fast, so val-loss bottoms at **epoch 2** (underfit) and the model collapses toward all-negative
  (F1 0.444). Frozen + val-loss was fine (epoch 3). → do **not** pair `selection_metric: inner_val_loss`
  with the non-frozen encoder.
- **macro-F1 ≈ positive-F1 when frozen** (both pick epoch 4, identical results); they only diverge under
  the noisier non-frozen dynamics.
- **likelihood vs teacher_forced gaps are all 1–3 subjects** — within the single-subject noise floor
  (±0.03–0.05 F1 on N=47). No decision rule is reliably better; TF is just cheaper.
- **Inner-val is noisy (N=35 / 12 pos): val-F1 ≈ 0.42–0.56 while test-F1 ≈ 0.64–0.74.** This is the small
  selection set, not a bug — but it means a single config's single test number has wide error bars; do not
  rank configs by sub-0.05 F1 deltas.

### Recommended default for all future experiments (DECIDED 2026-06-15)
> **Audio encoder frozen + `selection_metric: inner_val_macro_f1` + `original_teacher_forced` eval.**

Rationale from the grid above:
- **Freeze the encoder** — settled best practice. Kills the LoRA-into-Whisper leak, gives the best
  positive-F1 in the whole grid (0.737), and cuts trainable params ~7.5×. No defensible reason to train it.
- **macro-F1 selection** — overall the best-balanced choice. It is not strictly dominant: the
  precision/recall trade-off is real — macro-F1 buys **recall 1.000** (precision 0.583), while val-loss
  selection gives **better precision (0.667) at recall 0.714**. If a deployment needs precision over
  recall, val-loss is a defensible alternative (it's safe *when frozen* — epoch 3, no collapse). Default
  to macro-F1; switch to val-loss only when precision is the priority.
- **teacher_forced eval** — TF and likelihood are within the noise floor (1–3 subjects); TF is cheaper,
  so it's the standard verdict. (`likelihood` remains a fine, slightly-lower-variance fallback.)

The canonical config embodying this is `daic_subject_audio_text_reg1_selmacrof1_tf.yaml`; clone it as the
template for new datasets/regs. Further cells of this grid are **not worth running** — the trade-offs are
understood.

### K-ablation on the decided default (DAIC, single-seed)
Held everything at the decided default (frozen encoder + macro-F1 selection + TF eval, reg1 recipe)
and varied **only** `chunks_per_subject` K ∈ {2,3,4,5,6}. Configs
`daic_subject_audio_text_reg1_selmacrof1_tf{,_k2,_k3,_k5,_k6}.yaml`. Test N=47 (33 non-dep / 14 dep),
all-negative baseline ACC 0.702. `INVALID` = TF wrong-first-token (= wrong class).

| K | ACC | posF1 | macroF1 | Prec | Rec | CM [TN,FP / FN,TP] | INVALID |
| - | ----- | ----- | ----- | ----- | ----- | ------------------ | ------- |
| 2 | 0.702 | 0.650 | 0.695 | 0.500 | 0.929 | 20,13 / 1,13 | 14 |
| 3 | 0.723 | 0.683 | 0.719 | 0.519 | 1.000 | 20,13 / 0,14 | 13 |
| **4** | 0.787 | **0.737** | **0.779** | 0.583 | 1.000 | 23,10 / 0,14 | — |
| 5 | 0.723 | 0.629 | 0.704 | 0.524 | 0.786 | 23,10 / 3,11 | 13 |
| 6 | 0.830 | 0.636 | 0.763 | 0.875 | 0.500 | 32,1 / 7,7 | 8 |

(K=4 row is the §8 decided-default run; macroF1 computed from its CM.)

Findings:
- **K=4 is the peak** on both positive-F1 (0.737) and macro-F1 (0.779) — the sweep validates it, doesn't beat it.
- **Monotonic operating-point shift**: small K over-predicts depressed (K=2/3 recall 0.93–1.0, precision ≈0.5);
  large K turns conservative (K=6 precision 0.875, recall 0.500, misses half the depressed). Consistent with the
  regularization-strength view of K (overlap ≈ K²/N): small K = stronger aug → leans to the minority; large K =
  weaker reg → reverts to the cautious high-precision shape that mimics the leaky model (K=6 CM 32,1/7,7 echoes
  the old non-frozen 0.851-ACC operating point). **K=6's ACC 0.830 is a trap** — high only because it rarely says
  "depressed".
- **Single-seed caveat**: N=47, one subject ≈ 0.03–0.05 F1; K2/K3/K5 differ by 1–3 subjects (within noise). Only
  the monotonic precision↑/recall↓ across all 5 points and "K=4 not beaten" are defensible. Multi-seed replicates
  (seed-variant configs `..._s2024/_s7` + `scripts/run_daic_ksweep.sh`) are built but not yet run.

### Decision-rule mechanics (why `likelihood` is the default; see `src/evaluate.py`, `src/aggregate.py`)
- `likelihood` — compares teacher-forced log-likelihood of "Depressed" vs "Non-depressed"; a forced
  2-way choice that **never** yields INVALID. Lowest variance, the principled default.
- `original_teacher_forced` — feeds `prompt + GOLD label`, takes per-position arg-max. For **multi-token
  labels with distinct first tokens** (`"Dep…"` vs `"Non…"`) a wrong first token + gold continuation
  decodes to a corrupted hybrid (`"Non"+"…ressed"`) → parses INVALID. So **INVALID ⟺ wrong first token
  ⟺ wrong class**; it is a *wrong prediction*, not "model confusion". `_strict_binary_prediction(gold,pred)`
  maps INVALID → `1 - gold`, so the **headline / `binary_strict_*`** metrics already count it as wrong
  (read those; **ignore `valid_only_*`**, which drops INVALIDs and over-reports).
- `generation` — free decode (no gold injection); "what the model says out loud", wrong-is-wrong, can
  INVALID only on genuine rambling. Use this (not TF) if you want a *decode-based* verdict without the
  teacher-forcing artifact.

### Configs / artifacts
- Frozen variants: `daic_subject_audio_text_reg1{,_selloss,_selmacrof1}{,_tf}.yaml`
  (+ early `_valloss{,_tf}` runs where only early-stopping — not selection — used loss, before the
  selection fix; those are equivalent to positive-F1 selection).
- Non-frozen variants: `daic_subject_audio_text_reg1{,_selloss,_selmacrof1}{,_tf}_nofreeze.yaml`
  (and `_tf_nofreeze` for positive-F1 × TF).
- Verify freeze state from the train log: `Audio adaptation state | … tune_audio_encoder=<bool>
  encoder_lora_leaked_params=<n>` (frozen → 0; non-frozen → 3,932,160).

## 9. Persistent memory pointers

Auto-memory at `/home/emre/.claude/projects/-home-emre-Projects-AudioLLM/memory/`:
- `eval-determinism-rule.md` — the §3 hard constraint.
- `edaic-overfitting-investigation.md` — audio-vs-text ablation; audio is the overfit liability.
- `subject-audio-kchunk-mode.md` — how `sample_mode=subject_audio` + per-epoch sampling works.
- `audio-encoder-lora-leak-freeze.md` — the LoRA-into-Whisper-encoder leak, the freeze fix, and that the
  DAIC/EDAIC audio results need re-running.
- `selection-metric-and-freeze-grid.md` — `training.selection_metric` knob + the §8 grid conclusions.
