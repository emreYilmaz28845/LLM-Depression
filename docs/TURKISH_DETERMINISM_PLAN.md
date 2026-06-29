# Determinism & Reporting Plan

Follow-up to `TURKISH_RESULTS_MISMATCH.md`. That investigation concluded the
Turkish audio-only rerun gap (esp. T21) is **unseeded training nondeterminism**,
not transcript/manifest content. This document lists what we can actually do
about it, in priority order.

## Why bitwise determinism is not reachable here

Two hard limits:

1. **`bf16` precision** — determinism flags fix kernel selection and reduction
   *order*, not numerical precision. bf16 accumulation still wobbles run-to-run.
2. **Qwen2-Audio encoder ops** — some have no deterministic CUDA kernel, so
   `torch.use_deterministic_algorithms(True)` (strict) would raise mid-run; we
   can only run it with `warn_only=True`, leaving those ops nondeterministic.

So the realistic goal is **reduced variance + honest reporting**, not bitwise
reproducibility. The work splits into tiers; Tier 0 and Tier 2 are the
high-value items, Tier 1 is a nice-to-have, Tier 3 is usually not worth it.

---

## Tier 0 — Provenance (do regardless of everything else)

No run currently records the code it ran, which is half of why the mismatch was
hard to diagnose. For each run, log into the artifact dir:

- `git rev-parse HEAD`
- `git status --porcelain` (dirty-tree marker)
- `pip freeze` (or the existing `scripts/capture_environment.sh` output)

Without this you cannot confirm the "old" vs "new" runs used the same commit
(the `d5b6a51` quarantine and `5297ca9` transcript-fix commits both post-date the
old run).

---

## Tier 1 — Shrink run-to-run noise (cheap, reversible)

Reduces drift; does **not** make the old number reappear and does **not** give
bitwise reproducibility.

1. **Extend `set_seed` (`src/utils.py:396`)**, behind a `deterministic` flag:
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - `torch.use_deterministic_algorithms(True, warn_only=True)`
     (`warn_only` is mandatory — strict mode crashes on Qwen2-Audio ops)
   - `os.environ.setdefault("PYTHONHASHSEED", str(seed))` as a fallback

2. **Export two env vars before `torchrun`** in `scripts/run_train_slurm.sh`
   (and `run_eval_slurm.sh`):
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8` — *required* once deterministic
     algorithms are on, or cuBLAS GEMMs throw. Must be set in the shell, not
     Python: cuBLAS reads it at handle creation, before our code runs.
   - `PYTHONHASHSEED=0` — must precede interpreter start to be fully effective.

3. **Gate on a config flag** (`training.deterministic`, default the team's
   choice) so the ~10–20% slowdown is opt-in/out.

Data pipeline is already deterministic (sampler uses a fixed-seed generator at
`src/train.py:409`, `dataloader_num_workers: 0`, no augmentation), so GPU compute
is the only remaining source — which is exactly what Tier 1 targets.

---

## Tier 2 — Stop trusting a single run (the actual fix)

Because the audio signal is near-chance (AUROC ≈ 0.57, see
`TURKISH_DATASET_STATS.md §5`) and T21 sits on the decision boundary, a
single-run F1 is effectively a coin flip. Fix the *reporting*, not just the seed:

1. **Run each config with N seeds (e.g. 5)** and report **mean ± std**, not a
   point estimate. The "0.688 vs 0.557" T21 gap almost certainly lives inside one
   standard deviation.
2. **Lead with AUROC + macro-F1**; drop positive-F1 as the headline. AUROC is
   base-rate-invariant, so it removes the illusion that T17 (F1 0.81) beats T21
   (F1 0.56) — both collapse toward ~0.57, exposing that T17's score was largely
   the majority-class baseline.
3. **Add an all-positive baseline column** to the results table so every F1 is
   read against "predict everyone depressed":
   - T17: precision 0.692, recall 1.0 → **F1 0.818** (old T17 audio-only = 0.814)
   - T21: precision 0.517, recall 1.0 → **F1 0.681** (old T21 audio-only = 0.688)

   Both old runs ≈ the all-positive baseline, i.e. the model was not really
   discriminating — it was riding the class balance.

This is what makes thresholds comparable and makes "did the rerun regress?"
answerable with a variance bar instead of a single noisy number.

---

## Tier 3 — True bitwise reproducibility (skip unless required)

Switch `bf16 → fp32` and use strict `torch.use_deterministic_algorithms(True)`,
replacing or CPU-offloading the unsupported Qwen2-Audio ops. Large compute/memory
cost for marginal value on a task already capped at AUROC 0.57. Not recommended
here.

---

## Recommended order

1. **Tier 0** (provenance) — always.
2. **Tier 2** (multi-seed + AUROC + baseline column) — resolves the actual
   T21/T17 confusion.
3. **Tier 1** (determinism flags) — optional noise reduction.
4. **Tier 3** — only if a downstream requirement demands bitwise repro.
