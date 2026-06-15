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
- **A real regularization sweep** (reg1–4, §6): increasing dropout, weight decay, lower lr, fewer
  LoRA modules / last-n-layers, lower max_grad_norm, early stopping on inner-val positive-F1.
  reg3/reg4 (heavy reg) are the stable configs; light reg overfits/collapses.
- **Text is the clean channel.** Transcript length does NOT leak (separability 0.457). Text-only is
  the robust baseline; audio is the part that overfits.
- **Frozen audio encoder (DepressInstruct recipe) — IMPLEMENTED.** LoRA `target_modules`
  (`q/k/v_proj`) also match the Whisper `audio_tower` attention, so LoRA was silently being trained
  *on the audio encoder* — exactly the overfit liability DepressInstruct warns about. The legacy
  `hybrid` configs excluded it; the `subject_audio reg1–4` configs did **not**, so the reported
  subject-level numbers (§7) were produced with a leaking encoder. Fix: `build_lora_config` now
  defaults `exclude_modules=".*audio_tower.*|.*multi_modal_projector.*"` (opt out with
  `lora.tune_audio_encoder: true`), and `enforce_audio_encoder_freeze` (in `qwen2audio_lora.py`)
  verifies/freezes any leaked encoder LoRA each load and logs the count. **Action: re-run the
  DAIC/EDAIC `subject_audio` + `subject_audio_text` reg sweep with the freeze before trusting §7.**

### What we could still do (NOT yet implemented — candidate next steps)
- **Audio data augmentation at runtime** (cheap, high-value): SpecAugment-style time/frequency masking
  on the mel features; additive noise / SNR jitter; speed/tempo perturbation (±5–10%); random gain;
  light reverb. These attack speaker/channel memorisation directly. Apply **train-only**, keep eval
  deterministic (same rule as §3).
- **More aggressive chunk-subset augmentation**: smaller K (K=2–3) for *stronger* combinatorial
  regularization; or multi-view training (sample several K-subsets per subject per epoch).
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
  more chunks), and (b) OOMs — 15 chunks × ~750 tokens/30s is far too long.

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
- **`src/train.py`** — instrumentation + the OOM fix wiring:
  - `train_dataset` → `chunk_sampling="random"` when subject_audio; `selection_dataset` → `"deterministic"`.
  - Per-epoch, at top of loop after `model.train()`: `restore_model_for_training(unwrap(model), config)` (see §5).
  - VRAM/audio logging: `_log_peak_gpu_memory` (max_allocated/max_reserved GiB),
    `_audit_audio_budget` (counts `<|AUDIO|>` tokens → `audio_budget_audit_<partition>.json`),
    `_percentile`. `torch.cuda.reset_peak_memory_stats()` before the loop; `peak_gpu_memory.json` at end.
  - After each epoch barrier: `gc.collect(); torch.cuda.empty_cache()`.
- **`src/model/qwen2audio_lora.py`** & **`src/model/text_lora.py`**: `restore_model_for_training(model, config)`
  (symmetric). **`src/model/runtime.py`**: dispatch wrapper picking backend by `resolve_input_modality`.

## 5. The OOM bug and its real fix (important — easy to regress)

Symptom: OOM on epoch ≥ 2 (not epoch 1), at long sequence lengths (fp32 logits ~1.8 GiB at seq ~3100).

- **First (wrong) diagnosis:** fragmentation → added `empty_cache`. Did NOT fix it.
- **Real cause:** `prepare_model_for_evaluation` **disables gradient checkpointing** and sets
  `use_cache=True`. `model.train()` does **not** restore checkpointing, so epoch 2+ trains WITHOUT
  gradient checkpointing → activation blowup → OOM.
- **Fix:** `restore_model_for_training(model, config)` (disable-then-enable gradient checkpointing +
  re-apply `enable_input_require_grads` + `use_cache=False`), called every epoch after `model.train()`.
  Confirmed: VRAM now flat at ~27.74 GiB across all epochs (reg2 re-run).

**If you see epoch-2 OOM again, check that `restore_model_for_training` is still being called each epoch.**

## 6. Configs (the reg grid)

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

## 7. Current results (see `results_subject_level_daic_edaic.md` for full tables)

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

## 8. Open / suggested next steps (none in-flight; confirm with user before starting)

1. **Investigate CMDC 0.987** for corpus artifacts/leakage (recording/site conditions vs label)
   before trusting or reporting it.
2. **Fix EATD**: it only has the single default (light) config. Port the **reg3 recipe + class
   handling** (it's heavily imbalanced 132:30); add class-balanced sampling.
3. **CMDC & EATD lack a reg sweep and modality ablation** — only the single default audio+text config
   exists. Generate reg1–4 and audio_only / text_only configs to put them on equal footing with DAIC/EDAIC.
4. Optional: EDAIC→DAIC cross-corpus eval; a long run with early stopping disabled to confirm the
   overfitting story directly.

## 9. Persistent memory pointers

Auto-memory at `/home/emre/.claude/projects/-home-emre-Projects-AudioLLM/memory/`:
- `eval-determinism-rule.md` — the §3 hard constraint.
- `edaic-overfitting-investigation.md` — audio-vs-text ablation; audio is the overfit liability.
- `subject-audio-kchunk-mode.md` — how `sample_mode=subject_audio` + per-epoch sampling works.
